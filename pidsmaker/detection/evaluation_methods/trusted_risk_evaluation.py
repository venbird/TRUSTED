"""Temporal risk propagation node evaluation.

This evaluator reuses ORTHRUS edge-level inference outputs and replaces only
the post-inference node scoring logic.
"""

import os
import pickle
import re
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch

from pidsmaker.detection.evaluation_methods.evaluation_utils import (
    compute_kmeans_labels,
    get_ground_truth_nids,
)
from pidsmaker.detection.evaluation_methods.node_metrics import evaluate_node_results
from pidsmaker.utils.utils import (
    get_all_graphs_for_dates,
    get_node_to_path_and_type,
    listdir_sorted,
    log,
    log_tqdm,
)

REQUIRED_EDGE_COLUMNS = {"loss", "srcnode", "dstnode", "time", "edge_type"}
_TRUST_PROFILE_CACHE = {}


def _topk_mean(values, top_k):
    if not values:
        return 0.0
    top_values = sorted(values, reverse=True)[:top_k]
    return float(np.mean(top_values))


def _update_max_risk(risks, sources, node, score, source):
    if score <= 0:
        return
    if score > risks[node]:
        risks[node] = score
        sources[node] = source


def _get_eval_cfg(cfg):
    return cfg.evaluation.temporal_risk_evaluation


def _compute_threshold(scores, threshold_method, params=None):
    if not scores:
        return 0.0

    threshold_method = threshold_method.strip()
    if threshold_method == "max_val_loss":
        return float(max(scores))
    if threshold_method == "mean_val_loss" or threshold_method == "magic":
        return float(np.mean(scores))
    if threshold_method == "percentile_90" or threshold_method == "nodlink":
        return float(np.percentile(scores, 90))
    if threshold_method == "tail_val_loss":
        if params is None:
            raise ValueError("tail_val_loss requires temporal risk evaluation params")
        tail_score = float(np.percentile(scores, params.tail_percentile))
        max_score = float(max(scores))
        return tail_score + params.tail_blend_alpha * (max_score - tail_score)

    raise ValueError(f"Invalid temporal risk threshold method `{threshold_method}`")


def _load_train_node_set(cfg):
    graph_dir = cfg.transformation._graphs_dir
    train_set_paths = get_all_graphs_for_dates(graph_dir, cfg.dataset.train_dates)

    train_node_set = set()
    for train_path in train_set_paths:
        train_graph = torch.load(train_path)
        train_node_set |= set(train_graph.nodes())
    return train_node_set


def _iter_graph_edges(graph):
    if graph.is_multigraph():
        for src, dst, _, data in graph.edges(keys=True, data=True):
            yield int(src), int(dst), data
    else:
        for src, dst, data in graph.edges(data=True):
            yield int(src), int(dst), data


def _node_type(node, trust_profile):
    node_info = trust_profile["node_to_path_type"].get(int(node), {})
    return str(node_info.get("type", "unknown")).lower()


def _vector_from_counter(counter, keys):
    if not counter:
        return np.zeros(len(keys), dtype=float)
    total = float(sum(counter.values()))
    if total <= 0:
        return np.zeros(len(keys), dtype=float)
    return np.array([counter.get(key, 0.0) / total for key in keys], dtype=float)


def _jensen_shannon_distance(current_counter, history_counter, keys):
    p = _vector_from_counter(current_counter, keys)
    q = _vector_from_counter(history_counter, keys)
    if not p.any() or not q.any():
        return 1.0

    m = 0.5 * (p + q)

    def kl_divergence(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    divergence = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return float(np.sqrt(max(divergence, 0.0)))


def _distribution_similarity(current_counter, history_counter, keys):
    distance = _jensen_shannon_distance(current_counter, history_counter, keys)
    return float(np.clip(1.0 - distance, 0.0, 1.0))


def _build_trust_profile(cfg):
    cache_key = (
        cfg.transformation._graphs_dir,
        tuple(cfg.dataset.train_dates),
    )
    if cache_key in _TRUST_PROFILE_CACHE:
        return _TRUST_PROFILE_CACHE[cache_key]

    node_to_path_type = get_node_to_path_and_type(cfg)
    trusted_profile_path = os.path.join(
        getattr(cfg.featurization, "_model_dir", ""),
        "trusted_trust_profile.pkl",
    )
    if (
        getattr(cfg.featurization, "used_method", "") == "trusted"
        and os.path.exists(trusted_profile_path)
    ):
        with open(trusted_profile_path, "rb") as handle:
            trusted_profile = pickle.load(handle)

        def normalize_neighbor_keys(counter):
            normalized = Counter()
            for key, value in counter.items():
                node_type = "subject" if str(key).lower() == "process" else str(key).lower()
                normalized[node_type] += value
            return normalized

        history_neighbor_type_counts = defaultdict(Counter)
        for node, counter in trusted_profile.get("neighbor_types", {}).items():
            history_neighbor_type_counts[int(node)] = normalize_neighbor_keys(counter)

        history_edge_type_counts = defaultdict(Counter)
        for node, counter in trusted_profile.get("edge_types", {}).items():
            history_edge_type_counts[int(node)] = Counter(counter)

        node_frequency = Counter(
            {int(node): int(count) for node, count in trusted_profile.get("node_frequency", {}).items()}
        )
        train_node_set = set(node_frequency.keys())
        edge_type_keys = sorted(
            {key for counter in history_edge_type_counts.values() for key in counter}
        )
        neighbor_type_keys = sorted(
            {key for counter in history_neighbor_type_counts.values() for key in counter}
        )
        profile = {
            "node_to_path_type": node_to_path_type,
            "train_node_set": train_node_set,
            "node_frequency": node_frequency,
            "max_node_frequency": max(node_frequency.values(), default=1),
            "history_edge_type_counts": history_edge_type_counts,
            "history_neighbor_type_counts": history_neighbor_type_counts,
            "edge_type_keys": edge_type_keys,
            "neighbor_type_keys": neighbor_type_keys,
        }
        _TRUST_PROFILE_CACHE[cache_key] = profile
        return profile

    graph_dir = cfg.transformation._graphs_dir
    train_set_paths = get_all_graphs_for_dates(graph_dir, cfg.dataset.train_dates)

    train_node_set = set()
    node_frequency = Counter()
    history_edge_type_counts = defaultdict(Counter)
    history_neighbor_type_counts = defaultdict(Counter)
    edge_type_keys = set()
    neighbor_type_keys = set()

    base_profile = {"node_to_path_type": node_to_path_type}
    for train_path in train_set_paths:
        train_graph = torch.load(train_path)
        train_node_set |= set(int(node) for node in train_graph.nodes())
        for node in train_graph.nodes():
            node_frequency[int(node)] += 1

        for src, dst, data in _iter_graph_edges(train_graph):
            edge_type = str(data.get("label", "unknown"))
            src_type = _node_type(src, base_profile)
            dst_type = _node_type(dst, base_profile)

            edge_type_keys.add(edge_type)
            neighbor_type_keys.update([src_type, dst_type])

            history_edge_type_counts[src][edge_type] += 1
            history_edge_type_counts[dst][edge_type] += 1
            history_neighbor_type_counts[src][dst_type] += 1
            history_neighbor_type_counts[dst][src_type] += 1

    profile = {
        "node_to_path_type": node_to_path_type,
        "train_node_set": train_node_set,
        "node_frequency": node_frequency,
        "max_node_frequency": max(node_frequency.values(), default=1),
        "history_edge_type_counts": history_edge_type_counts,
        "history_neighbor_type_counts": history_neighbor_type_counts,
        "edge_type_keys": sorted(edge_type_keys),
        "neighbor_type_keys": sorted(neighbor_type_keys),
    }
    _TRUST_PROFILE_CACHE[cache_key] = profile
    return profile


def _get_node_trust(node, trust_profile, params, current_profile=None):
    node = int(node)
    if current_profile is None:
        current_profile = {}
    cache = current_profile.setdefault("node_trust_cache", {})
    if node in cache:
        return cache[node]

    node_type = _node_type(node, trust_profile)
    node_frequency = trust_profile["node_frequency"][node]
    occurrence = np.log1p(node_frequency) / np.log1p(trust_profile["max_node_frequency"])

    neighbor_keys = sorted(
        set(trust_profile["neighbor_type_keys"])
        | set(current_profile.get("neighbor_type_counts", {}).get(node, {}).keys())
    )
    edge_type_keys = sorted(
        set(trust_profile["edge_type_keys"])
        | set(current_profile.get("edge_type_counts", {}).get(node, {}).keys())
    )
    neighbor_stability = _distribution_similarity(
        current_profile.get("neighbor_type_counts", {}).get(node, Counter()),
        trust_profile["history_neighbor_type_counts"].get(node, Counter()),
        neighbor_keys,
    )
    behavior_stability = _distribution_similarity(
        current_profile.get("edge_type_counts", {}).get(node, Counter()),
        trust_profile["history_edge_type_counts"].get(node, Counter()),
        edge_type_keys,
    )

    trust = (
        params.trust_occurrence_weight * occurrence
        + params.trust_neighbor_weight * neighbor_stability
        + params.trust_behavior_weight * behavior_stability
    )
    if node in trust_profile["train_node_set"]:
        trust = max(trust, params.seen_node_trust)

    trust = float(np.clip(trust, params.trust_min, params.trust_max))
    cache[node] = {
        "trust": trust,
        "low_trust": 1.0 - trust,
        "type": node_type,
        "occurrence": float(occurrence),
        "neighbor_stability": neighbor_stability,
        "behavior_stability": behavior_stability,
        "train_frequency": node_frequency,
    }
    return cache[node]


def _parse_node_type_set(node_types):
    return {
        node_type.strip().lower()
        for node_type in str(node_types).split(",")
        if node_type.strip()
    }


def _canonical_burst_key(node, node_info, params):
    node_type = str(node_info.get("type", "unknown")).lower()
    path = str(node_info.get("path", ""))
    cmd = str(node_info.get("cmd", ""))
    mode = str(params.duplicate_burst_key).strip().lower()

    if mode == "basename":
        key = os.path.basename(path) or path
    elif mode == "parent":
        key = os.path.dirname(path) or path
    elif mode == "cmd_or_path":
        key = cmd if node_type == "subject" and cmd and cmd != "None" else path
    elif mode == "browser_cache":
        key = re.sub(r"/(cache2/entries|Cache|cache|tmp)/[^/]+$", r"/\1/*", path)
        key = re.sub(r"/[0-9A-Fa-f]{20,}$", "/*HASH*", key)
    else:
        key = path

    return (node_type, key)


def _apply_duplicate_burst_discount(node_results, trust_profile, params):
    if not params.duplicate_burst_discount_enabled or not node_results:
        return node_results

    node_types = _parse_node_type_set(params.duplicate_burst_node_types)
    node_to_path_type = (
        trust_profile["node_to_path_type"]
        if trust_profile is not None
        else {}
    )

    groups = defaultdict(list)
    for node, result in node_results.items():
        node_info = node_to_path_type.get(int(node), {})
        node_type = str(node_info.get("type", "unknown")).lower()
        if node_types and node_type not in node_types:
            continue
        groups[_canonical_burst_key(node, node_info, params)].append(
            (float(result.get("score", 0.0)), node)
        )

    min_rank = max(1, int(params.duplicate_burst_min_rank))
    min_factor = float(np.clip(params.duplicate_burst_min_factor, 0.0, 1.0))
    weight = max(0.0, float(params.duplicate_burst_weight))

    for grouped_nodes in groups.values():
        if len(grouped_nodes) < min_rank:
            continue
        grouped_nodes.sort(reverse=True)
        for rank, (_, node) in enumerate(grouped_nodes, start=1):
            if rank < min_rank:
                continue
            factor = 1.0 / (1.0 + weight * np.log1p(rank - min_rank + 1.0))
            factor = max(min_factor, float(factor))
            result = node_results[node]
            original_score = float(result.get("score", 0.0))
            result["pre_burst_discount_score"] = original_score
            result["duplicate_burst_rank"] = rank
            result["duplicate_burst_factor"] = factor
            result["score"] = original_score * factor
            result["final_score"] = result["score"]

    return node_results


def _safe_log_degree_norm(degree, min_norm):
    return max(float(min_norm), float(np.log1p(max(degree, 0) + 1.0)))


def _apply_context_direct_decay(context, direct, params):
    if params.context_direct_decay <= 0 or context <= 0:
        return context
    factor = 1.0 / (1.0 + direct / params.context_direct_decay)
    factor = max(float(params.context_direct_min_factor), float(factor))
    return float(context * factor)


def _fuse_score(direct_score, standard_score, params, effective_residual_weight=None):
    mode = str(params.score_fusion_mode).strip().lower()
    if mode == "standard":
        return float(standard_score)
    if mode == "direct_anchor":
        return float(direct_score)
    if mode not in {"direct_anchor_residual", "adaptive_direct_anchor_residual"}:
        raise ValueError(f"Invalid temporal risk score_fusion_mode {params.score_fusion_mode}")

    residual_weight = (
        params.residual_weight
        if effective_residual_weight is None
        else effective_residual_weight
    )
    residual = max(0.0, float(standard_score) - float(direct_score))
    cap_base = max(abs(float(direct_score)), 1e-12)
    residual_cap = max(float(params.residual_min_cap), params.residual_cap_ratio * cap_base)
    return float(direct_score + min(residual_weight * residual, residual_cap))


def _top_node_ids(scores_by_node, tail_size):
    return {
        node
        for node, _ in sorted(
            scores_by_node.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:tail_size]
    }


def _select_adaptive_residual_weight(val_results, params):
    if not val_results:
        return 0.0

    direct_scores = {
        node: float(result.get("max_direct_score", result.get("direct_anchor_score", 0.0)))
        for node, result in val_results.items()
    }
    standard_scores = {
        node: float(result.get("max_standard_score", result.get("standard_score", 0.0)))
        for node, result in val_results.items()
    }

    tail_size = max(
        int(params.adaptive_min_tail_size),
        int(np.ceil(len(direct_scores) * params.adaptive_tail_fraction)),
    )
    tail_size = min(tail_size, len(direct_scores))
    if tail_size <= 0:
        return 0.0

    direct_top = _top_node_ids(direct_scores, tail_size)
    direct_tail_mean = float(
        np.mean(sorted(direct_scores.values(), reverse=True)[:tail_size])
    )
    direct_tail_mean = max(direct_tail_mean, 1e-12)

    candidate_weights = [
        params.residual_weight,
        params.residual_weight * 0.5,
        params.residual_weight * 0.25,
        params.residual_weight * 0.1,
        0.0,
    ]
    for weight in candidate_weights:
        candidate_scores = {
            node: _fuse_score(direct_scores[node], standard_scores[node], params, weight)
            for node in direct_scores
        }
        candidate_top = _top_node_ids(candidate_scores, tail_size)
        overlap = len(direct_top & candidate_top) / tail_size
        candidate_tail = sorted(candidate_scores.values(), reverse=True)[:tail_size]
        candidate_tail_mean = float(np.mean(candidate_tail))
        tail_growth = (candidate_tail_mean - direct_tail_mean) / direct_tail_mean
        residual_share = float(
            np.mean(
                [
                    max(0.0, candidate_scores[node] - direct_scores[node])
                    / max(candidate_scores[node], 1e-12)
                    for node in candidate_top
                ]
            )
        )
        if (
            overlap >= params.adaptive_min_tail_overlap
            and tail_growth <= params.adaptive_max_tail_growth
            and residual_share <= params.adaptive_max_residual_share
        ):
            log(
                "Adaptive residual weight selected: "
                f"{weight:.4f} (tail_overlap={overlap:.3f}, "
                f"tail_growth={tail_growth:.3f}, residual_share={residual_share:.3f})"
            )
            return float(weight)

    log("Adaptive residual guard rejected trust residual; using direct anchor scores.")
    return 0.0


def _get_source_propagation_risks(direct_risk, adjacency, params, trust_profile, current_profile):
    if not direct_risk:
        return {}

    if params.propagation_gate_percentile > 0:
        gate_threshold = float(
            np.percentile(list(direct_risk.values()), params.propagation_gate_percentile)
        )
    else:
        gate_threshold = None

    source_risks = {}
    for node, risk in direct_risk.items():
        gate = 1.0
        if gate_threshold is not None and risk < gate_threshold:
            gate *= params.propagation_gate_floor

        if trust_profile is not None and params.propagation_trust_discount > 0:
            node_trust = _get_node_trust(node, trust_profile, params, current_profile)
            gate *= 1.0 - params.propagation_trust_discount * node_trust["trust"]

        if params.source_degree_normalization:
            gate /= _safe_log_degree_norm(len(adjacency.get(node, ())), params.degree_norm_min)

        source_risks[node] = float(risk * np.clip(gate, 0.0, 1.0))
    return source_risks


def _read_edge_window(csv_file):
    df = pd.read_csv(csv_file, usecols=lambda column: column in REQUIRED_EDGE_COLUMNS)
    missing = REQUIRED_EDGE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {csv_file}")
    return df


def _compute_window_risks(df, params, trust_profile=None):
    incident_losses = defaultdict(list)
    adjacency = defaultdict(set)
    edges = []
    current_profile = {
        "edge_type_counts": defaultdict(Counter),
        "neighbor_type_counts": defaultdict(Counter),
    }

    for row in df.itertuples(index=False):
        src = int(row.srcnode)
        dst = int(row.dstnode)
        loss = float(row.loss)
        edge_type = str(row.edge_type)

        incident_losses[src].append(loss)
        incident_losses[dst].append(loss)
        current_profile["edge_type_counts"][src][edge_type] += 1
        current_profile["edge_type_counts"][dst][edge_type] += 1

        if src != dst:
            adjacency[src].add(dst)
            adjacency[dst].add(src)
            edges.append((src, dst))
            if trust_profile is not None:
                current_profile["neighbor_type_counts"][src][
                    _node_type(dst, trust_profile)
                ] += 1
                current_profile["neighbor_type_counts"][dst][
                    _node_type(src, trust_profile)
                ] += 1

    direct_risk = {
        node: _topk_mean(losses, params.top_k)
        for node, losses in incident_losses.items()
    }
    mode = str(params.score_fusion_mode).strip().lower()
    if (
        mode == "direct_anchor"
        and not params.use_trust_context
        and params.alpha == 0
        and params.delta == 0
    ):
        empty_risk = defaultdict(float)
        score_multiplier = defaultdict(lambda: 1.0)
        return direct_risk, empty_risk, empty_risk, score_multiplier, {}

    source_propagation_risk = _get_source_propagation_risks(
        direct_risk, adjacency, params, trust_profile, current_profile
    )

    one_hop_risk = defaultdict(float)
    one_hop_source = {}
    two_hop_risk = defaultdict(float)
    two_hop_source = {}

    for src, dst in edges:
        _update_max_risk(
            one_hop_risk, one_hop_source, src, source_propagation_risk.get(dst, 0.0), dst
        )
        _update_max_risk(
            one_hop_risk, one_hop_source, dst, source_propagation_risk.get(src, 0.0), src
        )

    for src, dst in edges:
        source = one_hop_source.get(dst)
        if source is not None and source != src and source not in adjacency[src]:
            _update_max_risk(two_hop_risk, two_hop_source, src, one_hop_risk[dst], source)

        source = one_hop_source.get(src)
        if source is not None and source != dst and source not in adjacency[dst]:
            _update_max_risk(two_hop_risk, two_hop_source, dst, one_hop_risk[src], source)

    propagated_risk = {}
    nodes = set(direct_risk) | set(one_hop_risk) | set(two_hop_risk)
    for node in nodes:
        prop = (
            params.beta_1hop * one_hop_risk.get(node, 0.0)
            + params.beta_2hop * two_hop_risk.get(node, 0.0)
        )
        if params.receiver_degree_normalization:
            prop /= _safe_log_degree_norm(len(adjacency.get(node, ())), params.degree_norm_min)
        propagated_risk[node] = min(float(prop), params.max_propagated_risk)

    context_risk = defaultdict(float)
    score_multiplier = defaultdict(lambda: 1.0)
    if not params.use_trust_context:
        return direct_risk, propagated_risk, context_risk, score_multiplier, {}

    risk_source_types = _parse_node_type_set(params.trust_risk_source_types)
    untrusted_source_neighbor_risk = defaultdict(float)
    untrusted_source_twohop_risk = defaultdict(float)

    for src, dst in edges:
        dst_trust = _get_node_trust(dst, trust_profile, params, current_profile)
        if dst_trust["type"].lower() in risk_source_types:
            untrusted_source_neighbor_risk[src] = max(
                untrusted_source_neighbor_risk[src],
                direct_risk.get(dst, 0.0) * dst_trust["low_trust"],
            )

        src_trust = _get_node_trust(src, trust_profile, params, current_profile)
        if src_trust["type"].lower() in risk_source_types:
            untrusted_source_neighbor_risk[dst] = max(
                untrusted_source_neighbor_risk[dst],
                direct_risk.get(src, 0.0) * src_trust["low_trust"],
            )

    for node in nodes:
        for neighbor in adjacency.get(node, ()):
            untrusted_source_twohop_risk[node] = max(
                untrusted_source_twohop_risk[node],
                untrusted_source_neighbor_risk.get(neighbor, 0.0),
            )

    for node in nodes:
        node_trust = _get_node_trust(node, trust_profile, params, current_profile)
        if node_trust["type"].lower() in risk_source_types:
            self_risk_weight = params.untrusted_source_self_weight
        else:
            self_risk_weight = params.untrusted_subject_self_weight

        score_multiplier[node] = (
            1.0 + self_risk_weight * node_trust["low_trust"]
        ) * (1.0 - params.trusted_score_discount * node_trust["trust"])

        context = (
            params.untrusted_source_neighbor_weight
            * untrusted_source_neighbor_risk.get(node, 0.0)
            + params.untrusted_source_twohop_weight
            * untrusted_source_twohop_risk.get(node, 0.0)
        )
        context_risk[node] = min(float(context), params.max_trust_context_risk)

    trust_details = {
        node: _get_node_trust(node, trust_profile, params, current_profile)
        for node in nodes
    }
    return direct_risk, propagated_risk, context_risk, score_multiplier, trust_details


def _compute_temporal_scores(tw_path, cfg, effective_residual_weight=None):
    params = _get_eval_cfg(cfg)
    trust_profile = _build_trust_profile(cfg) if params.use_trust_context else None
    if trust_profile is None and params.duplicate_burst_discount_enabled:
        trust_profile = {"node_to_path_type": get_node_to_path_and_type(cfg)}
    state_risk = defaultdict(float)
    persistence_count = defaultdict(int)
    node_results = defaultdict(dict)

    filelist = listdir_sorted(tw_path)
    started_at = time.time()
    for tw, filename in enumerate(log_tqdm(sorted(filelist), desc="Compute TRP labels")):
        csv_file = os.path.join(tw_path, filename)
        df = _read_edge_window(csv_file)
        (
            direct_risk,
            propagated_risk,
            context_risk,
            score_multiplier,
            trust_details,
        ) = _compute_window_risks(df, params, trust_profile=trust_profile)

        current_nodes = set(direct_risk) | set(propagated_risk) | set(context_risk)
        for node in list(state_risk.keys()):
            if node not in current_nodes:
                state_risk[node] *= params.absent_decay
                persistence_count[node] = 0
                if state_risk[node] < 1e-12:
                    del state_risk[node]
                    persistence_count.pop(node, None)

        state_inputs = {
            node: direct_risk.get(node, 0.0)
            + (propagated_risk.get(node, 0.0) if params.state_include_propagated else 0.0)
            for node in current_nodes
        }
        if state_inputs and params.state_persistence_min_windows > 1:
            persistence_threshold = float(
                np.percentile(
                    list(state_inputs.values()),
                    params.state_persistence_percentile,
                )
            )
        else:
            persistence_threshold = None

        for node in current_nodes:
            direct = direct_risk.get(node, 0.0)
            propagated = propagated_risk.get(node, 0.0)
            state_input = state_inputs.get(node, 0.0)
            if persistence_threshold is not None:
                if state_input >= persistence_threshold:
                    persistence_count[node] += 1
                else:
                    persistence_count[node] = 0
                state_factor = min(
                    1.0,
                    persistence_count[node] / max(params.state_persistence_min_windows, 1),
                )
            else:
                state_factor = 1.0

            state_risk[node] = (
                params.decay * state_risk[node]
                + (1.0 - params.decay) * state_input
            )
            state_risk[node] = min(float(state_risk[node]), params.max_state_risk)
            state = state_risk[node]
            persistent_state = state * state_factor
            base_score = direct + params.alpha * propagated + params.delta * persistent_state
            context = _apply_context_direct_decay(
                context_risk.get(node, 0.0),
                direct,
                params,
            )
            standard_score = (
                base_score * score_multiplier.get(node, 1.0)
                + context
            )
            direct_anchor_score = direct
            final_score = _fuse_score(
                direct_anchor_score,
                standard_score,
                params,
                effective_residual_weight,
            )

            node_results[node]["max_direct_score"] = max(
                direct_anchor_score,
                node_results[node].get("max_direct_score", float("-inf")),
            )
            node_results[node]["max_standard_score"] = max(
                standard_score,
                node_results[node].get("max_standard_score", float("-inf")),
            )

            if final_score > node_results[node].get("score", float("-inf")):
                node_results[node]["score"] = final_score
                node_results[node]["final_score"] = final_score
                node_results[node]["standard_score"] = standard_score
                node_results[node]["direct_anchor_score"] = direct_anchor_score
                node_results[node]["residual_score"] = max(
                    0.0,
                    standard_score - direct_anchor_score,
                )
                node_results[node]["effective_residual_weight"] = (
                    params.residual_weight
                    if effective_residual_weight is None
                    else effective_residual_weight
                )
                node_results[node]["base_score"] = base_score
                node_results[node]["direct_risk"] = direct
                node_results[node]["propagated_risk"] = propagated
                node_results[node]["state_risk"] = state
                node_results[node]["persistent_state_risk"] = persistent_state
                node_results[node]["state_persistence_count"] = persistence_count.get(node, 0)
                node_results[node]["state_persistence_factor"] = state_factor
                node_results[node]["trust_context_risk"] = context
                node_results[node]["raw_trust_context_risk"] = context_risk.get(node, 0.0)
                node_results[node]["score_multiplier"] = score_multiplier.get(node, 1.0)
                if trust_profile is not None:
                    node_trust = trust_details.get(node, {})
                    node_results[node]["trust"] = node_trust.get("trust", 0.0)
                    node_results[node]["trust_occurrence"] = node_trust.get("occurrence", 0.0)
                    node_results[node]["trust_neighbor_stability"] = node_trust.get(
                        "neighbor_stability", 0.0
                    )
                    node_results[node]["trust_behavior_stability"] = node_trust.get(
                        "behavior_stability", 0.0
                    )
                    node_results[node]["train_frequency"] = node_trust.get("train_frequency", 0)
                node_results[node]["tw_with_max_loss"] = tw

        if (tw + 1) % 250 == 0:
            elapsed = time.time() - started_at
            log(f"TRP processed {tw + 1}/{len(filelist)} windows in {elapsed:.1f}s")

    return _apply_duplicate_burst_discount(node_results, trust_profile, params)


def get_node_predictions(val_tw_path, test_tw_path, cfg, **kwargs):
    ground_truth_nids, _ = get_ground_truth_nids(cfg)
    params = _get_eval_cfg(cfg)

    log(f"Loading validation TRP scores from {val_tw_path}...")
    effective_residual_weight = None
    if str(params.score_fusion_mode).strip().lower() == "adaptive_direct_anchor_residual":
        val_probe_results = _compute_temporal_scores(val_tw_path, cfg)
        effective_residual_weight = _select_adaptive_residual_weight(val_probe_results, params)
    val_results = _compute_temporal_scores(
        val_tw_path,
        cfg,
        effective_residual_weight=effective_residual_weight,
    )
    val_scores = [result["score"] for result in val_results.values()]
    thr = _compute_threshold(val_scores, params.threshold_method, params) * params.threshold_multiplier
    log(f"Temporal risk threshold: {thr:.3f}")

    log(f"Loading test TRP scores from {test_tw_path}...")
    results = _compute_temporal_scores(
        test_tw_path,
        cfg,
        effective_residual_weight=effective_residual_weight,
    )

    train_node_set = _load_train_node_set(cfg)
    use_kmeans = params.use_kmeans

    for node_id, result in results.items():
        pred_score = result["score"]
        result["y_true"] = int(node_id in ground_truth_nids)
        result["is_seen"] = int(int(node_id) in train_node_set)
        result["y_hat"] = 0 if use_kmeans else int(pred_score > thr)

    if use_kmeans:
        results = compute_kmeans_labels(results, topk_K=params.kmeans_top_K)

    return results, thr


def main(
    val_tw_path,
    test_tw_path,
    model_epoch_dir,
    cfg,
    tw_to_malicious_nodes,
    **kwargs,
):
    results, thr = get_node_predictions(
        cfg=cfg,
        val_tw_path=val_tw_path,
        test_tw_path=test_tw_path,
    )
    return evaluate_node_results(
        results=results,
        thr=thr,
        model_epoch_dir=model_epoch_dir,
        cfg=cfg,
        tw_to_malicious_nodes=tw_to_malicious_nodes,
    )
