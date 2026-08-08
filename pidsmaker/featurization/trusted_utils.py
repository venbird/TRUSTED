import hashlib
import ipaddress
import json
import math
import os
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from gensim.models import Word2Vec

from pidsmaker.featurization.featurization_utils import get_corpus
from pidsmaker.utils.dataset_utils import get_node_map, get_rel2id
from pidsmaker.utils.utils import get_indexid2msg, get_split_to_files, log, log_tqdm, tokenize_label

SEMANTIC_MODEL_FILE = "trusted_semantic_model.pkl"
STRUCTURAL_SCALER_FILE = "trusted_structural_scaler.pkl"
TRUST_PROFILE_FILE = "trusted_trust_profile.pkl"
ALL_SPLIT_STATS_FILE = "trusted_all_split_stats.pkl"
FEATURE_MANIFEST_FILE = "trusted_feature_manifest.json"


def get_trusted_cfg(cfg):
    return cfg.featurization.trusted


def get_artifact_paths(cfg):
    model_dir = cfg.featurization._model_dir
    return {
        "semantic_model": os.path.join(model_dir, SEMANTIC_MODEL_FILE),
        "structural_scaler": os.path.join(model_dir, STRUCTURAL_SCALER_FILE),
        "trust_profile": os.path.join(model_dir, TRUST_PROFILE_FILE),
        "all_split_stats": os.path.join(model_dir, ALL_SPLIT_STATS_FILE),
        "manifest": os.path.join(model_dir, FEATURE_MANIFEST_FILE),
    }


def _save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_trusted_artifacts(cfg):
    paths = get_artifact_paths(cfg)
    return {
        "semantic_model": _load_pickle(paths["semantic_model"]),
        "structural_scaler": _load_pickle(paths["structural_scaler"]),
        "trust_profile": _load_pickle(paths["trust_profile"]),
        "all_split_stats": _load_pickle(paths["all_split_stats"]),
    }


def normalize_node_type(node_type):
    if node_type == "subject":
        return "process"
    return str(node_type or "unknown").lower()


def _split_label(label):
    label = str(label or "missing")
    return [token for token in re.split(r"[^A-Za-z0-9_./:\\-]+", label.lower()) if token]


def _ip_class(token):
    try:
        ip = ipaddress.ip_address(token)
    except ValueError:
        return None
    if ip.is_private:
        return "PRIVATE_IP"
    if ip.is_loopback:
        return "LOOPBACK_IP"
    if ip.is_multicast:
        return "MULTICAST_IP"
    if ip.is_reserved:
        return "RESERVED_IP"
    return "PUBLIC_IP"


def _path_tokens(token):
    path_tokens = []
    if "/" not in token and "\\" not in token:
        return path_tokens

    normalized = token.replace("\\", "/")
    path_tokens.append("PATH")
    if normalized.startswith("/tmp/") or "/tmp/" in normalized:
        path_tokens.append("TMP_DIR")
    if normalized.startswith("/usr/bin/") or normalized.startswith("/bin/"):
        path_tokens.append("SYS_BIN")
    if normalized.startswith("/usr/sbin/") or normalized.startswith("/sbin/"):
        path_tokens.append("SYS_SBIN")
    if normalized.startswith("/home/"):
        path_tokens.append("HOME_DIR")
    if normalized.startswith("/var/"):
        path_tokens.append("VAR_DIR")

    suffix = Path(normalized).suffix.lower().lstrip(".")
    if suffix:
        path_tokens.append(f"EXT_{suffix[:16]}")
        if suffix in {"sh", "bash", "py", "pl", "rb", "js", "ps1"}:
            path_tokens.append("SCRIPT_FILE")
        elif suffix in {"exe", "dll", "so", "dylib"}:
            path_tokens.append("BINARY_FILE")
        elif suffix in {"conf", "cfg", "ini", "yml", "yaml", "json", "xml"}:
            path_tokens.append("CONFIG_FILE")

    basename = Path(normalized).name
    if basename:
        path_tokens.append(f"NAME_{_coarse_token(basename)}")
    return path_tokens


def _coarse_token(token):
    if not token:
        return "EMPTY"
    if len(token) >= 32 and re.fullmatch(r"[0-9a-f]+", token):
        return "HEX_HASH"
    if len(token) >= 24 and re.fullmatch(r"[a-z0-9+/=_-]+", token):
        return "LONG_ENCODED"
    if re.fullmatch(r"\d+", token):
        return "NUMERIC"
    if re.search(r"\d", token) and re.search(r"[a-z]", token):
        return "ALNUM"
    return token[:48]


def normalize_tokens(label, node_type):
    tokens = [f"TYPE_{normalize_node_type(node_type).upper()}"]
    raw_tokens = _split_label(label)
    for raw in raw_tokens:
        ip_token = _ip_class(raw)
        if ip_token:
            tokens.append(ip_token)
            continue

        tokens.extend(_path_tokens(raw))
        coarse = _coarse_token(raw)
        tokens.append(f"TOKEN_{coarse}")
        if raw in {"-enc", "--encodedcommand", "base64"}:
            tokens.append("ENCODED_CMD")
        if len(raw) > 80:
            tokens.append("LONG_TOKEN")

    if not raw_tokens:
        tokens.append("MISSING_LABEL")
    return tokens


def hashing_vector(tokens, dim):
    vec = np.zeros(dim, dtype=np.float32)
    if dim <= 0:
        return vec
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="little", signed=False)
        idx = value % dim
        sign = 1.0 if (value >> 63) == 0 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec




def _word_weights(n, decline_percentage):
    if n <= 0:
        return []
    d = -1 / n * decline_percentage / 100
    first = 1 / n - 0.5 * (n - 1) * d
    return [first + i * d for i in range(n)]


def train_context_word2vec(cfg, semantic_dim):
    tm_cfg = get_trusted_cfg(cfg)
    corpus = get_corpus(cfg, gather_multi_dataset=cfg.featurization.multi_dataset_training)
    if not corpus:
        return {
            "encoder": tm_cfg.semantic_encoder,
            "semantic_dim": semantic_dim,
            "normalization": "empty_context_word2vec",
            "vectors": {},
            "decline_rate": float(cfg.featurization.word2vec.decline_rate),
            "vocab_size": 0,
        }

    model = Word2Vec(
        corpus,
        alpha=cfg.featurization.word2vec.alpha,
        vector_size=semantic_dim,
        window=cfg.featurization.word2vec.window_size,
        min_count=cfg.featurization.word2vec.min_count,
        sg=cfg.featurization.word2vec.use_skip_gram,
        workers=cfg.featurization.word2vec.num_workers,
        epochs=cfg.featurization.epochs,
        compute_loss=cfg.featurization.word2vec.compute_loss,
        negative=cfg.featurization.word2vec.negative,
        seed=cfg.featurization.seed,
    )
    vectors = {
        word: np.asarray(model.wv[word], dtype=np.float32)
        for word in model.wv.index_to_key
    }
    return {
        "encoder": tm_cfg.semantic_encoder,
        "semantic_dim": semantic_dim,
        "normalization": "train_split_context_word2vec",
        "vectors": vectors,
        "decline_rate": float(cfg.featurization.word2vec.decline_rate),
        "vocab_size": len(vectors),
    }


def semantic_vector(label, node_type, semantic_model):
    encoder = str(semantic_model.get("encoder", "hashing")).strip().lower()
    dim = int(semantic_model["semantic_dim"])
    if encoder in {"word2vec_context", "context_word2vec", "word2vec"}:
        tokens = tokenize_label(label, node_type)
        vectors = semantic_model.get("vectors", {})
        zeros = np.zeros(dim, dtype=np.float32)
        word_vectors = [vectors.get(token, zeros) for token in tokens]
        weights = _word_weights(len(tokens), semantic_model.get("decline_rate", 30.0))
        if not word_vectors:
            return zeros
        weighted = [weight * vec for weight, vec in zip(weights, word_vectors)]
        vec = np.mean(weighted, axis=0).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    return hashing_vector(normalize_tokens(label, node_type), dim)

def _edge_types(cfg):
    rel2id = get_rel2id(cfg)
    items = [(key, value) for key, value in rel2id.items() if isinstance(key, str)]
    return [key for key, _ in sorted(items, key=lambda item: item[1])]


def _node_types():
    node_map = get_node_map()
    items = [(key, value) for key, value in node_map.items() if isinstance(key, str)]
    return [key for key, _ in sorted(items, key=lambda item: item[1])]


def _empty_counter_dict():
    return defaultdict(Counter)


def collect_graph_stats(cfg, split_to_files, splits):
    indexid2msg = get_indexid2msg(cfg)
    stats = {
        "node_frequency": Counter(),
        "in_degree": Counter(),
        "out_degree": Counter(),
        "in_edge_types": _empty_counter_dict(),
        "out_edge_types": _empty_counter_dict(),
        "all_edge_types": _empty_counter_dict(),
        "neighbor_types": _empty_counter_dict(),
        "graph_count": 0,
        "edge_count": 0,
    }

    for split in splits:
        for path in log_tqdm(split_to_files.get(split, []), desc=f"TRUSTED stats: {split}"):
            graph = torch.load(path)
            stats["graph_count"] += 1
            for node in graph.nodes:
                stats["node_frequency"][str(node)] += 1

            for u, v, _key, attr in graph.edges(data=True, keys=True):
                u, v = str(u), str(v)
                label = str(attr.get("label", "UNKNOWN"))
                stats["edge_count"] += 1
                stats["out_degree"][u] += 1
                stats["in_degree"][v] += 1
                stats["out_edge_types"][u][label] += 1
                stats["in_edge_types"][v][label] += 1
                stats["all_edge_types"][u][label] += 1
                stats["all_edge_types"][v][label] += 1

                u_type = normalize_node_type(indexid2msg.get(u, ["unknown", ""])[0])
                v_type = normalize_node_type(indexid2msg.get(v, ["unknown", ""])[0])
                stats["neighbor_types"][u][v_type] += 1
                stats["neighbor_types"][v][u_type] += 1

    return stats


def _counter_to_distribution(counter, keys):
    arr = np.array([float(counter.get(key, 0.0)) for key in keys], dtype=np.float32)
    total = float(arr.sum())
    if total > 0:
        arr /= total
    return arr


def js_similarity(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_sum = left.sum()
    right_sum = right.sum()
    if left_sum <= 0 and right_sum <= 0:
        return 1.0
    if left_sum <= 0 or right_sum <= 0:
        return 0.0
    left = left / left_sum
    right = right / right_sum
    mid = 0.5 * (left + right)
    eps = 1e-12
    kl_left = np.sum(np.where(left > 0, left * np.log2((left + eps) / (mid + eps)), 0.0))
    kl_right = np.sum(np.where(right > 0, right * np.log2((right + eps) / (mid + eps)), 0.0))
    distance = math.sqrt(max(0.0, 0.5 * (kl_left + kl_right)))
    return float(max(0.0, min(1.0, 1.0 - distance)))


def build_feature_state(cfg):
    tm_cfg = get_trusted_cfg(cfg)
    split_to_files = get_split_to_files(cfg, cfg.transformation._graphs_dir)
    edge_types = _edge_types(cfg)
    node_types = [normalize_node_type(node_type) for node_type in _node_types()]

    train_stats = collect_graph_stats(cfg, split_to_files, ["train"])
    all_stats = collect_graph_stats(cfg, split_to_files, ["train", "val", "test"])

    max_freq = max(train_stats["node_frequency"].values() or [1])
    max_in_degree = max(train_stats["in_degree"].values() or [1])
    max_out_degree = max(train_stats["out_degree"].values() or [1])
    max_total_degree = max(
        [
            train_stats["in_degree"].get(node, 0) + train_stats["out_degree"].get(node, 0)
            for node in train_stats["node_frequency"]
        ]
        or [1]
    )

    semantic_dim = int(tm_cfg.semantic_dim)
    structural_dim = 3 + len(edge_types) * 2 + len(node_types) + 1
    trust_dim = 1 if tm_cfg.use_trust_score else 0
    total_dim = semantic_dim
    if tm_cfg.use_structural_features:
        total_dim += structural_dim
    total_dim += trust_dim

    semantic_encoder = str(tm_cfg.semantic_encoder).strip().lower()
    if semantic_encoder in {"word2vec_context", "context_word2vec", "word2vec"}:
        log("Training TRUSTED context Word2Vec semantic encoder")
        semantic_model = train_context_word2vec(cfg, semantic_dim)
    else:
        semantic_model = {
            "encoder": tm_cfg.semantic_encoder,
            "semantic_dim": semantic_dim,
            "normalization": "generic_token_hashing",
        }
    structural_scaler = {
        "edge_types": edge_types,
        "node_types": node_types,
        "max_in_degree": float(max_in_degree),
        "max_out_degree": float(max_out_degree),
        "max_total_degree": float(max_total_degree),
        "degree_log_norm": bool(tm_cfg.degree_log_norm),
        "histogram_l1_norm": bool(tm_cfg.histogram_l1_norm),
        "structural_dim": structural_dim,
    }
    trust_profile = {
        "max_freq": float(max_freq),
        "node_frequency": dict(train_stats["node_frequency"]),
        "neighbor_types": {node: dict(counter) for node, counter in train_stats["neighbor_types"].items()},
        "edge_types": {node: dict(counter) for node, counter in train_stats["all_edge_types"].items()},
        "trust_min": float(tm_cfg.trust_min),
        "trust_max": float(tm_cfg.trust_max),
        "unseen_trust": float(tm_cfg.unseen_trust),
        "occ_weight": float(tm_cfg.occ_weight),
        "neighbor_weight": float(tm_cfg.neighbor_weight),
        "behavior_weight": float(tm_cfg.behavior_weight),
    }

    manifest = {
        "method": "trusted",
        "construction_used_method": cfg.construction.used_method,
        "semantic_encoder": tm_cfg.semantic_encoder,
        "semantic_dim": semantic_dim,
        "structural_dim": structural_dim if tm_cfg.use_structural_features else 0,
        "trust_dim": trust_dim,
        "total_dim": total_dim,
        "configured_emb_dim": int(cfg.featurization.emb_dim),
        "use_structural_features": bool(tm_cfg.use_structural_features),
        "use_trust_score": bool(tm_cfg.use_trust_score),
        "train_graph_count": int(train_stats["graph_count"]),
        "all_graph_count": int(all_stats["graph_count"]),
        "train_edge_count": int(train_stats["edge_count"]),
        "all_edge_count": int(all_stats["edge_count"]),
        "train_node_count": len(train_stats["node_frequency"]),
        "all_node_count": len(all_stats["node_frequency"]),
        "edge_types": edge_types,
        "node_types": node_types,
    }

    return semantic_model, structural_scaler, trust_profile, all_stats, manifest


def save_feature_state(cfg, semantic_model, structural_scaler, trust_profile, all_stats, manifest):
    paths = get_artifact_paths(cfg)
    _save_pickle(semantic_model, paths["semantic_model"])
    _save_pickle(structural_scaler, paths["structural_scaler"])
    _save_pickle(trust_profile, paths["trust_profile"])
    _save_pickle(all_stats, paths["all_split_stats"])
    save_json(manifest, paths["manifest"])
    log(f"Saved TRUSTED feature manifest to {paths['manifest']}")


def _scaled_degree(value, max_value, use_log):
    if max_value <= 0:
        return 0.0
    if use_log:
        return float(np.log1p(value) / (np.log1p(max_value) + 1e-12))
    return float(value / max_value)


def structural_vector(node, stats, structural_scaler, train_seen):
    node = str(node)
    use_log = structural_scaler["degree_log_norm"]
    in_degree = stats["in_degree"].get(node, 0)
    out_degree = stats["out_degree"].get(node, 0)
    total_degree = in_degree + out_degree

    degree_features = np.array(
        [
            _scaled_degree(in_degree, structural_scaler["max_in_degree"], use_log),
            _scaled_degree(out_degree, structural_scaler["max_out_degree"], use_log),
            _scaled_degree(total_degree, structural_scaler["max_total_degree"], use_log),
        ],
        dtype=np.float32,
    )
    in_hist = _counter_to_distribution(stats["in_edge_types"].get(node, {}), structural_scaler["edge_types"])
    out_hist = _counter_to_distribution(stats["out_edge_types"].get(node, {}), structural_scaler["edge_types"])
    neigh = _counter_to_distribution(stats["neighbor_types"].get(node, {}), structural_scaler["node_types"])
    seen = np.array([1.0 if train_seen else 0.0], dtype=np.float32)
    return np.concatenate([degree_features, in_hist, out_hist, neigh, seen]).astype(np.float32)


def trust_score(node, current_stats, trust_profile, structural_scaler):
    node = str(node)
    train_freq = trust_profile["node_frequency"].get(node, 0)
    if train_freq <= 0:
        return float(trust_profile["unseen_trust"])

    occ = math.log1p(train_freq) / (math.log1p(trust_profile["max_freq"]) + 1e-12)
    train_neighbor = _counter_to_distribution(
        trust_profile["neighbor_types"].get(node, {}),
        structural_scaler["node_types"],
    )
    current_neighbor_counter = current_stats["neighbor_types"].get(
        node, trust_profile["neighbor_types"].get(node, {})
    )
    current_neighbor = _counter_to_distribution(current_neighbor_counter, structural_scaler["node_types"])

    train_behavior = _counter_to_distribution(
        trust_profile["edge_types"].get(node, {}),
        structural_scaler["edge_types"],
    )
    current_behavior_counter = current_stats["all_edge_types"].get(
        node, trust_profile["edge_types"].get(node, {})
    )
    current_behavior = _counter_to_distribution(current_behavior_counter, structural_scaler["edge_types"])

    nei = js_similarity(current_neighbor, train_neighbor)
    beh = js_similarity(current_behavior, train_behavior)
    score = (
        trust_profile["occ_weight"] * occ
        + trust_profile["neighbor_weight"] * nei
        + trust_profile["behavior_weight"] * beh
    )
    return float(np.clip(score, trust_profile["trust_min"], trust_profile["trust_max"]))


def build_indexid2vec(cfg):
    artifacts = load_trusted_artifacts(cfg)
    semantic_model = artifacts["semantic_model"]
    structural_scaler = artifacts["structural_scaler"]
    trust_profile = artifacts["trust_profile"]
    current_stats = artifacts["all_split_stats"]
    tm_cfg = get_trusted_cfg(cfg)

    indexid2msg = get_indexid2msg(cfg)

    indexid2vec = {}
    for indexid, msg in log_tqdm(indexid2msg.items(), desc="Embedding TRUSTED nodes"):
        node_type, label = msg[0], msg[1]
        parts = [semantic_vector(label, node_type, semantic_model)]

        train_seen = str(indexid) in trust_profile["node_frequency"]
        if tm_cfg.use_structural_features:
            parts.append(structural_vector(indexid, current_stats, structural_scaler, train_seen))
        if tm_cfg.use_trust_score:
            parts.append(np.array([trust_score(indexid, current_stats, trust_profile, structural_scaler)], dtype=np.float32))

        indexid2vec[indexid] = np.concatenate(parts).astype(np.float32)

    return indexid2vec


def ensure_feature_state(cfg):
    paths = get_artifact_paths(cfg)
    missing = [path for path in paths.values() if not os.path.exists(path)]
    if missing:
        log(f"TRUSTED artifacts missing; rebuilding: {missing}")
        state = build_feature_state(cfg)
        save_feature_state(cfg, *state)
