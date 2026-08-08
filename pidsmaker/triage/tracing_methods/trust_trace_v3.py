#!/usr/bin/env python
"""TrustTrace-v3: single-file, evidence-only provenance reconstruction.

This clean implementation does not import legacy implementation code.
Ground Truth is never read by this module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import torch

RELATION_IDS = {
    1: "EVENT_CONNECT", 2: "EVENT_EXECUTE", 3: "EVENT_OPEN",
    4: "EVENT_READ", 5: "EVENT_RECVFROM", 6: "EVENT_RECVMSG",
    7: "EVENT_SENDMSG", 8: "EVENT_SENDTO", 9: "EVENT_WRITE",
    10: "EVENT_CLONE",
}
MISSING = {"", "none", "null", "nan", "na", "n/a", "undefined", "<na>"}
NETWORK_TYPES = {"network", "netflow", "socket", "flow"}
PROCESS_TYPES = {"subject", "process", "proc", "task"}


@dataclass(frozen=True)
class Config:
    incident_gap_windows: int = 8
    context_before_minutes: int = 15
    context_after_minutes: int = 30
    time_bucket_seconds: int = 60
    max_nodes: int = 60
    max_edges: int = 100
    core_max_nodes: int = 24
    core_max_edges: int = 36
    stage_witnesses_per_role: int = 3
    high_risk_sidecar_threshold: float = 0.85
    low_trust_threshold: float = 0.5
    high_trust_threshold: float = 0.8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_time(value) -> datetime:
    return datetime.fromisoformat(str(value)[:26])


def clean(value):
    if value is None or str(value).strip().lower() in MISSING:
        return None
    return str(value).strip()


def node_type(value) -> str:
    value = (clean(value) or "unknown").lower()
    return {"path": "file", "pathname": "file", "regular_file": "file",
            "proc": "process", "task": "process", "endpoint": "network"}.get(value, value)


def metadata_entry(raw_id: str, metadata: dict) -> dict:
    for name in ("node_to_path_type", "node_to_path", "path_type", "node_metadata"):
        container = metadata.get(name)
        if not isinstance(container, dict):
            continue
        value = container.get(raw_id, container.get(str(raw_id)))
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, (list, tuple)) and value:
            return {"path": value[0], "node_type": value[1] if len(value) > 1 else None}
        if value is not None:
            return {"path": value}
    return {}


def resolve_entity(raw_id, attrs: dict, metadata: dict) -> dict:
    raw = str(raw_id)
    meta = metadata_entry(raw, metadata)
    kind = node_type(meta.get("node_type") or meta.get("type") or attrs.get("node_type") or attrs.get("type"))
    label = None
    for key in ("path", "file_path", "canonical_path", "executable_path", "label", "name"):
        label = clean(meta.get(key)) or clean(attrs.get(key))
        if label:
            break
    if kind == "unknown" and label and label.startswith(("/", "./", "../")):
        kind = "file"
    if kind == "file" and label:
        label = re.sub(r"/+", "/", label.replace("\\", "/"))
        return {"canonical_id": f"file:{label}", "label": label, "node_type": "file",
                "resolution_source": "metadata_or_attribute", "identity_status": "resolved"}
    if kind in NETWORK_TYPES:
        ip = clean(meta.get("remote_ip")) or clean(meta.get("ip")) or clean(attrs.get("remote_ip")) or clean(attrs.get("ip"))
        port = clean(meta.get("remote_port")) or clean(meta.get("port")) or clean(attrs.get("remote_port")) or clean(attrs.get("port"))
        endpoint = f"{ip}:{port}" if ip and port else label
        if endpoint:
            endpoint = endpoint.replace(" ", "")
            return {"canonical_id": f"network:{endpoint}", "label": endpoint, "node_type": "network",
                    "resolution_source": "metadata_or_attribute", "identity_status": "resolved"}
    token = re.sub(r"[^A-Za-z0-9_.:-]+", "_", raw).strip("_") or hashlib.sha256(raw.encode()).hexdigest()[:16]
    return {"canonical_id": f"{kind}:raw:{token}", "label": label or f"{kind}#{token}",
            "node_type": kind, "resolution_source": "typed_raw_fallback",
            "identity_status": "raw_id_scoped"}


def is_process(kind: str) -> bool:
    return str(kind).lower() in PROCESS_TYPES


def is_network(kind: str) -> bool:
    return str(kind).lower() in NETWORK_TYPES


def raw_relation(value) -> str:
    try:
        return RELATION_IDS[int(value)]
    except (KeyError, TypeError, ValueError):
        return str(value or "UNKNOWN").upper()


def relation_semantics(relation: str, src_type: str, dst_type: str):
    value = raw_relation(relation)
    source_process, target_process = is_process(src_type), is_process(dst_type)
    source_network, target_network = is_network(src_type), is_network(dst_type)
    if "INJECT" in value:
        return "process_inject", 1.0, True, False
    if any(token in value for token in ("CLONE", "FORK", "SPAWN", "PROCESS_CREATE")):
        return "process_create", 1.0, True, False
    if "EXECUTE" in value or value == "EXEC":
        return "execute", 1.0, True, False
    if "ACCEPT" in value or "RECV" in value:
        return "network_receive", 0.95, True, False
    if "CONNECT" in value:
        return "network_connect", 1.0, True, False
    if any(token in value for token in ("SENDTO", "SENDMSG", "EVENT_SEND")):
        return "network_send", 1.0, True, False
    if "WRITE" in value or "MODIFY" in value or value == "CREATE":
        return "write", 1.0, True, source_process and target_network
    if "READ" in value or "OPEN" in value:
        return "read", 0.35, True, source_process and target_network
    return "unknown", 0.1, False, source_network and target_network


def flow_endpoints(src: str, dst: str, relation: str, src_type: str, dst_type: str):
    canonical, _, known, _ = relation_semantics(relation, src_type, dst_type)
    if not known:
        return src, dst
    if canonical in {"network_send", "network_connect"}:
        return (src, dst) if is_process(src_type) and is_network(dst_type) else (dst, src)
    if canonical == "network_receive":
        return (src, dst) if is_network(src_type) and is_process(dst_type) else (dst, src)
    if canonical in {"execute", "read"}:
        return (src, dst) if is_process(dst_type) else (dst, src)
    if canonical == "write":
        return (src, dst) if is_process(src_type) else (dst, src)
    return src, dst


def graph_edges(graph):
    if graph.is_multigraph():
        yield from graph.edges(keys=True, data=True)
    else:
        for src, dst, attrs in graph.edges(data=True):
            yield src, dst, None, attrs


def load_results(path: Path, edge_paths: list[Path]) -> dict[int, dict[str, dict]]:
    raw = torch.load(path, map_location="cpu")
    if not isinstance(raw, dict):
        raise ValueError("detector result must be a mapping")
    if raw and all(isinstance(value, dict) and "y_hat" in value for value in raw.values()):
        mapped = {}
        for node, value in raw.items():
            window = int(value.get("tw_with_max_loss", -1))
            if 0 <= window < len(edge_paths):
                mapped.setdefault(window, {})[str(node)] = value
        raw = mapped
    result = {}
    for window, values in raw.items():
        try:
            index = int(window)
        except (TypeError, ValueError):
            continue
        result[index] = {}
        for node, value in (values or {}).items():
            if not isinstance(value, dict):
                continue
            item = {}
            for key in ("y_hat", "score", "direct_score", "direct_risk", "final_score", "trust_score", "trust"):
                if key in value:
                    try:
                        item[key] = int(value[key]) if key == "y_hat" else float(value[key])
                    except (TypeError, ValueError):
                        pass
            result[index][str(node)] = item
    return result


def load_losses(path: Path) -> dict:
    result = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                key = (str(row["srcnode"]), str(row["dstnode"]), int(row["time"]), raw_relation(row.get("edge_type")))
                result[key].append(float(row["loss"]))
            except (KeyError, TypeError, ValueError):
                continue
    return dict(result)


def lookup_loss(table: dict, src: str, dst: str, timestamp: int, relation: str):
    values = table.get((str(src), str(dst), int(timestamp), raw_relation(relation)))
    return (max(values), "mapped") if values else (None, "unknown")


def window_bounds(path: Path):
    start, end = path.stem.split("~", 1)
    return parse_time(start), parse_time(end)


def graph_candidates(root: Path, timestamp: datetime):
    day = root / f"graph_{timestamp.date()}"
    if not day.exists():
        return []
    result = []
    for path in sorted(day.iterdir()):
        try:
            start_raw, end_raw = path.name.split("~", 1)
            start, end = parse_time(start_raw), parse_time(end_raw)
        except (TypeError, ValueError):
            continue
        if start <= timestamp <= end:
            result.append((path, start, end))
    return result


def select_graph(root: Path, timestamp: datetime, anchors: set[str]):
    scored = []
    for path, start, end in graph_candidates(root, timestamp):
        graph = torch.load(path, map_location="cpu")
        score = sum(anchor in {str(node) for node in graph.nodes} for anchor in anchors)
        scored.append((score, str(path), path, start, end, graph))
    return max(scored, key=lambda value: (value[0], value[1])) if scored else None


def form_incidents(results: dict[int, dict[str, dict]], cfg: Config):
    incidents = []
    for window in sorted(results):
        anchors = sorted(node for node, value in results[window].items() if int(value.get("y_hat", 0)) == 1)
        if not anchors:
            continue
        if incidents and window - incidents[-1]["windows"][-1] <= cfg.incident_gap_windows:
            incidents[-1]["windows"].append(window)
            incidents[-1]["anchors"] = sorted(set(incidents[-1]["anchors"]) | set(anchors))
        else:
            incidents.append({"incident_id": f"incident_{len(incidents)}", "windows": [window], "anchors": anchors})
    return incidents


def expand_windows(primary: dict, edge_paths: list[Path], cfg: Config):
    bounds = [window_bounds(edge_paths[index]) for index in primary["windows"]]
    start = min(value[0] for value in bounds) - timedelta(minutes=cfg.context_before_minutes)
    end = max(value[1] for value in bounds) + timedelta(minutes=cfg.context_after_minutes)
    trace = []
    for index, path in enumerate(edge_paths):
        begin, finish = window_bounds(path)
        if finish >= start and begin <= end:
            trace.append(index)
    primary_set = set(primary["windows"])
    return trace, sorted(set(trace) - primary_set)


def rank_values(values: dict[str, float | None]) -> dict[str, float | None]:
    finite = {key: float(value) for key, value in values.items() if value is not None and math.isfinite(float(value))}
    if not finite:
        return {key: None for key in values}
    distinct = sorted(set(finite.values()))
    scale = max(1, len(distinct) - 1)
    mapping = {value: index / scale for index, value in enumerate(distinct)}
    return {key: mapping.get(finite.get(key)) for key in values}


def collect(primary: dict, results: dict, graphs: dict, loss_tables: dict):
    nodes, raw_to_canonical, resolutions, events = {}, {}, [], []
    primary_windows = set(primary["windows"])
    context_windows = set(primary["context_windows"])
    for window in sorted(graphs):
        graph, metadata = graphs[window], dict(getattr(graphs[window], "graph", {}) or {})
        for raw, attrs in graph.nodes(data=True):
            raw_id = str(raw)
            resolved = resolve_entity(raw_id, dict(attrs), metadata)
            raw_to_canonical.setdefault(raw_id, resolved["canonical_id"])
            canonical = raw_to_canonical[raw_id]
            info = results.get(window, {}).get(raw_id, {})
            anchor = window in primary_windows and int(info.get("y_hat", 0)) == 1
            direct = next((info[key] for key in ("direct_score", "direct_risk", "score") if key in info), None)
            final = next((info[key] for key in ("final_score", "score", "direct_score") if key in info), direct)
            trust = next((info[key] for key in ("trust_score", "trust") if key in info), None)
            if canonical not in nodes:
                nodes[canonical] = {"node_id": canonical, "label": resolved["label"], "node_type": resolved["node_type"],
                                    "direct_score": direct, "final_score": final, "trust_score": trust,
                                    "is_anchor": anchor, "raw_node_ids": [raw_id],
                                    "resolution_source": resolved["resolution_source"],
                                    "identity_status": resolved["identity_status"]}
            else:
                node = nodes[canonical]
                node["is_anchor"] = node["is_anchor"] or anchor
                if raw_id not in node["raw_node_ids"]:
                    node["raw_node_ids"].append(raw_id)
                for key, value in (("direct_score", direct), ("final_score", final)):
                    if value is not None and (node.get(key) is None or value > node[key]):
                        node[key] = value
                if node.get("trust_score") is None and trust is not None:
                    node["trust_score"] = trust
            resolutions.append({"window_id": window, "raw_id": raw_id, "canonical_id": canonical,
                                "display_label": resolved["label"], "node_type": resolved["node_type"],
                                "resolution_source": resolved["resolution_source"],
                                "identity_status": resolved["identity_status"]})
        for raw_src, raw_dst, _, attrs in graph_edges(graph):
            attrs, src_id, dst_id = dict(attrs or {}), str(raw_src), str(raw_dst)
            event_id = str(attrs.get("event_uuid") or attrs.get("uuid") or "").strip()
            if not event_id or src_id not in raw_to_canonical or dst_id not in raw_to_canonical:
                continue
            relation = raw_relation(attrs.get("edge_type", attrs.get("relation", attrs.get("label"))))
            timestamp = int(attrs.get("time", attrs.get("timestamp_ns", 0)) or 0)
            loss, status = lookup_loss(loss_tables.get(window, {}), src_id, dst_id, timestamp, relation)
            events.append({"event_id": event_id, "raw_src": src_id, "raw_dst": dst_id,
                           "src": raw_to_canonical[src_id], "dst": raw_to_canonical[dst_id],
                           "src_type": nodes[raw_to_canonical[src_id]]["node_type"],
                           "dst_type": nodes[raw_to_canonical[dst_id]]["node_type"],
                           "raw_relation": relation, "timestamp_ns": timestamp,
                           "edge_loss": loss, "edge_loss_status": status,
                           "is_context_window": window in context_windows})
    unique = {}
    for event in events:
        old = unique.get(event["event_id"])
        if old is None or (event["edge_loss"] is not None, -event["timestamp_ns"]) > (old["edge_loss"] is not None, -old["timestamp_ns"]):
            unique[event["event_id"]] = event
    return nodes, list(unique.values()), resolutions


def aggregate(events: list[dict], cfg: Config):
    buckets = defaultdict(list)
    width = max(1, cfg.time_bucket_seconds) * 1_000_000_000
    for event in events:
        relation, score, known, conflict = relation_semantics(event["raw_relation"], event["src_type"], event["dst_type"])
        src, dst = flow_endpoints(event["src"], event["dst"], event["raw_relation"], event["src_type"], event["dst_type"])
        buckets[(src, dst, relation, event["timestamp_ns"] // width)].append((event, score, known, conflict))
    edges = []
    for key, values in sorted(buckets.items(), key=lambda value: value[0]):
        events_here = [value[0] for value in values]
        first = min(events_here, key=lambda value: (value["timestamp_ns"], value["event_id"]))
        event_ids = sorted({value["event_id"] for value in events_here})
        relations = sorted({value["raw_relation"] for value in events_here})
        conflict = any(value[3] for value in values)
        digest = hashlib.sha256(("|".join(key[:3]) + "|" + "|".join(event_ids)).encode()).hexdigest()[:16]
        losses = [value["edge_loss"] for value in events_here if value["edge_loss"] is not None]
        edges.append({"edge_id": f"AE{digest}", "src": key[0], "dst": key[1], "flow_src": key[0], "flow_dst": key[1],
                      "relation": key[2], "raw_relation": first["raw_relation"], "raw_relations": relations,
                      "first_seen_ns": min(value["timestamp_ns"] for value in events_here),
                      "last_seen_ns": max(value["timestamp_ns"] for value in events_here),
                      "raw_event_ids": event_ids, "event_count": len(event_ids),
                      "max_edge_loss": max(losses) if losses else None,
                      "edge_loss_status": "mapped" if len(losses) == len(events_here) else ("partially_mapped" if losses else "unknown"),
                      "is_context_window": any(value["is_context_window"] for value in events_here),
                      "claim_eligible": key[2] not in {"unknown", "read"} and not conflict,
                      "semantic_conflict": conflict})
    return edges


def edge_roles(edge: dict, nodes: dict, ingress_cutoff_ns: int | None = None):
    relation, dst = edge["relation"], nodes[edge["dst"]]
    if relation == "write" and dst["node_type"] == "file":
        return ["write_or_landing"]
    if relation == "execute":
        return ["execution"]
    if relation in {"network_connect", "network_send"} and is_network(dst["node_type"]):
        roles = ["outbound_communication"]
        # CDM direction can encode an ingress loader as a process-side send.
        # Use only temporal evidence available in the trace: a transfer before
        # the first payload write is an ingress witness; later transfers remain
        # outbound communication. No case-specific labels are consulted.
        if ingress_cutoff_ns is not None and edge["first_seen_ns"] <= ingress_cutoff_ns:
            roles.append("network_ingress")
        return roles
    if relation == "network_receive":
        return ["network_ingress"]
    if relation in {"process_create", "process_inject"}:
        return [relation]
    return []


def score_edges(edges: list[dict], nodes: dict):
    direct = rank_values({key: value.get("direct_score") for key, value in nodes.items()})
    final = rank_values({key: value.get("final_score") for key, value in nodes.items()})
    for node_id, node in nodes.items():
        node["direct_risk_rank"], node["final_risk_rank"] = direct.get(node_id), final.get(node_id)
        trust = node.get("trust_score")
        if node["is_anchor"] and trust is not None and trust >= 0.8:
            node["role"] = "trusted_but_directly_anomalous"
        elif node["is_anchor"]:
            node["role"] = "direct_anchor"
        elif trust is not None and trust < 0.5:
            node["role"] = "low_trust_context"
        else:
            node["role"] = "supporting_context"
    write_times = [edge["first_seen_ns"] for edge in edges if "write_or_landing" in edge_roles(edge, nodes)]
    ingress_cutoff_ns = min(write_times) if write_times else None
    for edge in edges:
        source, target = nodes[edge["src"]], nodes[edge["dst"]]
        edge["stage_roles"] = edge_roles(edge, nodes, ingress_cutoff_ns)
        edge["relevance_reasons"] = (["direct_alert"] if source["is_anchor"] or target["is_anchor"] else []) + edge["stage_roles"]
        trust_values = [value for value in (source.get("trust_score"), target.get("trust_score")) if value is not None]
        trust_context = sum(1.0 - value for value in trust_values) / len(trust_values) if trust_values else 0.0
        risk = max(source.get("final_risk_rank") or 0.0, target.get("final_risk_rank") or 0.0)
        edge["priority"] = 4.0 * bool(source["is_anchor"] or target["is_anchor"]) + 3.0 * bool(edge["stage_roles"]) + 1.5 * risk + 0.5 * trust_context
        edge["priority"] += min(1.0, max(0.0, float(edge["max_edge_loss"] or 0.0)) / 10.0)
        edge["priority"] -= 0.25 * bool(edge["is_context_window"])
        edge["priority"] -= 6.0 * bool(edge["semantic_conflict"])


def connector_tree(start: set[str], edges: list[dict]):
    adjacency = defaultdict(list)
    for edge in edges:
        if edge["semantic_conflict"]:
            continue
        adjacency[edge["src"]].append((edge["dst"], edge))
        adjacency[edge["dst"]].append((edge["src"], edge))
    queue, seen, parent = deque(sorted(start)), set(start), {}
    while queue:
        node = queue.popleft()
        for nxt, edge in sorted(adjacency[node], key=lambda item: -item[1]["priority"]):
            if nxt not in seen:
                seen.add(nxt)
                parent[nxt] = (node, edge)
                queue.append(nxt)
    return parent


def tree_path(start: set[str], target: str, parent: dict):
    if target in start:
        return []
    if target not in parent:
        return None
    path, node = [], target
    while node not in start:
        node, edge = parent[node]
        path.append(edge)
    path.reverse()
    return path


def shortest_connector(start: set[str], target: str, edges: list[dict]):
    return tree_path(start, target, connector_tree(start, edges))


def connector_to_edge(start: set[str], target: dict, parent: dict):
    """Return the shortest real-edge path that attaches target to start."""
    if not start:
        return [target]
    if target["src"] in start or target["dst"] in start:
        return [target]
    paths = []
    for endpoint in (target["src"], target["dst"]):
        path = tree_path(start, endpoint, parent)
        if path is not None:
            paths.append(path if any(edge["edge_id"] == target["edge_id"] for edge in path)
                         else path + [target])
    if not paths:
        return None
    return min(paths, key=lambda path: (len(path), -sum(edge["priority"] for edge in path),
                                        tuple(edge["edge_id"] for edge in path)))


def best_phase_chain(edges: list[dict]):
    """Find the strongest observable time-monotonic write-execute-network chain."""
    executions = defaultdict(list)
    networks = defaultdict(list)
    for edge in edges:
        if "execution" in edge["stage_roles"]:
            executions[edge["src"]].append(edge)
        if "outbound_communication" in edge["stage_roles"]:
            networks[edge["src"]].append(edge)
    chains = []
    for write in (edge for edge in edges if "write_or_landing" in edge["stage_roles"]):
        for execute in executions[write["dst"]]:
            if execute["first_seen_ns"] < write["first_seen_ns"]:
                continue
            for network in networks[execute["dst"]]:
                if network["first_seen_ns"] < execute["first_seen_ns"]:
                    continue
                chain = [write, execute, network]
                anchor_hits = len({node for edge in chain for node in (edge["src"], edge["dst"])
                                   if edge.get("relevance_reasons") and "direct_alert" in edge["relevance_reasons"]})
                chains.append((anchor_hits, sum(edge["priority"] for edge in chain), chain))
    if not chains:
        return None
    chains.sort(key=lambda item: (-item[0], -item[1],
                                  tuple(edge["first_seen_ns"] for edge in item[2]),
                                  tuple(edge["edge_id"] for edge in item[2])))
    return chains[0][2]


def select_edges(edges: list[dict], nodes: dict, anchors: set[str], cfg: Config, core: bool):
    edge_limit, node_limit = (cfg.core_max_edges, cfg.core_max_nodes) if core else (cfg.max_edges, cfg.max_nodes)
    ordered = sorted((edge for edge in edges if not edge["semantic_conflict"]),
                     key=lambda edge: (-edge["priority"], edge["first_seen_ns"], edge["edge_id"]))
    selected, selected_ids, selected_nodes, roles_seen = [], set(), set(), set()

    def add(edge):
        new_nodes = {edge["src"], edge["dst"]} - selected_nodes
        if edge["edge_id"] in selected_ids or len(selected) >= edge_limit or len(selected_nodes | new_nodes) > node_limit:
            return False
        selected.append(edge)
        selected_ids.add(edge["edge_id"])
        selected_nodes.update((edge["src"], edge["dst"]))
        roles_seen.update(edge["stage_roles"])
        return True

    def path_delta(path):
        fresh, seen_here = [], set()
        for edge in path:
            if edge["edge_id"] not in selected_ids and edge["edge_id"] not in seen_here:
                fresh.append(edge)
                seen_here.add(edge["edge_id"])
        new_nodes = {node for edge in fresh for node in (edge["src"], edge["dst"])} - selected_nodes
        return fresh, new_nodes

    def add_path(path):
        if path is None:
            return False
        fresh, new_nodes = path_delta(path)
        if len(selected) + len(fresh) > edge_limit or len(selected_nodes | new_nodes) > node_limit:
            return False
        return all(add(edge) for edge in fresh) if fresh else True

    # A time-consistent causal chain is the preferred connected seed. If the
    # source lacks such a chain, begin from the strongest direct-alert edge.
    seed = best_phase_chain(ordered)
    if seed:
        add_path(seed)
    else:
        incident = [edge for edge in ordered if edge["claim_eligible"] and
                    ({edge["src"], edge["dst"]} & anchors)]
        if incident:
            add(incident[0])
        elif ordered:
            add(ordered[0])

    # Add missing stages by joint witness-plus-connector cost. Every accepted
    # path intersects the current graph, so disconnected paper cores cannot form.
    for role in ("network_ingress", "write_or_landing", "execution", "outbound_communication", "process_create", "process_inject"):
        if role in roles_seen:
            continue
        parent = connector_tree(selected_nodes, ordered)
        options = []
        for edge in ordered:
            if role not in edge["stage_roles"] or edge["edge_id"] in selected_ids:
                continue
            path = connector_to_edge(selected_nodes, edge, parent)
            if path is None:
                continue
            fresh, new_nodes = path_delta(path)
            options.append((len(new_nodes), len(fresh), -sum(item["priority"] for item in fresh),
                            edge["first_seen_ns"], edge["edge_id"], path))
        for option in sorted(options):
            if add_path(option[-1]):
                break

    if core:
        # Preserve a bounded number of distinct endpoint pairs per stage. This
        # prevents repeated events on one pair from displacing a second
        # observable attack branch while keeping the paper graph compact.
        for role in ("write_or_landing", "execution", "outbound_communication", "network_ingress"):
            while True:
                seen_pairs = {(edge["src"], edge["dst"]) for edge in selected if role in edge["stage_roles"]}
                if len(seen_pairs) >= cfg.stage_witnesses_per_role:
                    break
                parent = connector_tree(selected_nodes, ordered)
                options = []
                for edge in ordered:
                    pair = (edge["src"], edge["dst"])
                    if role not in edge["stage_roles"] or pair in seen_pairs or edge["edge_id"] in selected_ids:
                        continue
                    path = connector_to_edge(selected_nodes, edge, parent)
                    if path is None:
                        continue
                    fresh, new_nodes = path_delta(path)
                    options.append((0 if "direct_alert" in edge["relevance_reasons"] else 1,
                                    len(new_nodes), len(fresh), -edge["priority"],
                                    edge["first_seen_ns"], edge["edge_id"], path))
                accepted = False
                for option in sorted(options):
                    if add_path(option[-1]):
                        accepted = True
                        break
                if not accepted:
                    break

    # Preserve as many direct-risk anchors as the fixed budget permits. Trust
    # affects only ordering among contextual paths; it cannot remove a seed or
    # an already selected direct-alert witness.
    remaining_anchors = set(anchors) - selected_nodes
    while remaining_anchors:
        parent = connector_tree(selected_nodes, ordered)
        options = []
        for anchor in sorted(remaining_anchors):
            path = tree_path(selected_nodes, anchor, parent)
            if path is None:
                continue
            fresh, new_nodes = path_delta(path)
            risk = nodes.get(anchor, {}).get("final_risk_rank") or 0.0
            options.append((len(new_nodes), len(fresh), -risk,
                            -sum(edge["priority"] for edge in fresh), anchor, path))
        accepted = False
        for option in sorted(options):
            if add_path(option[-1]):
                accepted = True
                break
        remaining_anchors = set(anchors) - selected_nodes
        if not accepted:
            break

    while len(selected) < edge_limit:
        choices = [edge for edge in ordered if edge["edge_id"] not in selected_ids and
                   (edge["src"] in selected_nodes or edge["dst"] in selected_nodes)]
        if not choices:
            break
        if core:
            choices.sort(key=lambda edge: (0 if edge["stage_roles"] else 1,
                                           0 if any(role not in roles_seen for role in edge["stage_roles"]) else 1,
                                           0 if "direct_alert" in edge["relevance_reasons"] else 1,
                                           0 if edge["src"] in selected_nodes and edge["dst"] in selected_nodes else 1,
                                           -edge["priority"], edge["first_seen_ns"], edge["edge_id"]))
        else:
            choices.sort(key=lambda edge: (0 if any(role not in roles_seen for role in edge["stage_roles"]) else 1,
                                           0 if edge["src"] in selected_nodes and edge["dst"] in selected_nodes else 1,
                                           0 if "direct_alert" in edge["relevance_reasons"] else 1,
                                           -edge["priority"], edge["first_seen_ns"], edge["edge_id"]))
        if not add(choices[0]):
            ordered.remove(choices[0])
    return [edge["edge_id"] for edge in selected]


def phase_witnesses(edge_by_id: dict, selected_ids: list[str]):
    writes = [edge_by_id[value] for value in selected_ids if "write_or_landing" in edge_by_id[value]["stage_roles"]]
    executes = [edge_by_id[value] for value in selected_ids if "execution" in edge_by_id[value]["stage_roles"]]
    networks = [edge_by_id[value] for value in selected_ids if "outbound_communication" in edge_by_id[value]["stage_roles"]]
    result = []
    for write in writes:
        for execute in executes:
            if write["dst"] != execute["src"] or execute["first_seen_ns"] < write["first_seen_ns"]:
                continue
            for network in networks:
                if execute["dst"] == network["src"] and network["first_seen_ns"] >= execute["first_seen_ns"]:
                    result.append({"certificate_id": f"P{len(result)+1:03d}",
                                   "edge_ids": [write["edge_id"], execute["edge_id"], network["edge_id"]],
                                   "stages": ["write_or_landing", "execution", "outbound_communication"],
                                   "timestamps_ns": [write["first_seen_ns"], execute["first_seen_ns"], network["first_seen_ns"]],
                                   "time_monotonic": True})
                    if len(result) >= 16:
                        return result
    return result


def components(node_ids: set[str], edges: list[dict]) -> int:
    adjacency = defaultdict(set)
    for edge in edges:
        adjacency[edge["src"]].add(edge["dst"])
        adjacency[edge["dst"]].add(edge["src"])
    unseen, count = set(node_ids), 0
    while unseen:
        count += 1
        stack = [unseen.pop()]
        while stack:
            for nxt in adjacency[stack.pop()]:
                if nxt in unseen:
                    unseen.remove(nxt)
                    stack.append(nxt)
    return count


def write_dot(path: Path, nodes: list[dict], edges: list[dict], name: str):
    lines = [f"digraph {name} {{", "  rankdir=LR;", "  node [shape=box,fontname=\"Helvetica\"];" ]
    for node in nodes:
        label = str(node["label"]).replace('"', "'")
        color = "#d95f02" if node["is_anchor"] else "#d9d9d9"
        lines.append(f'  "{node["node_id"]}" [label="{label}\\n{node["node_type"]}",style=filled,fillcolor="{color}"];')
    for edge in edges:
        lines.append(f'  "{edge["src"]}" -> "{edge["dst"]}" [label="{edge["raw_relation"]}"];')
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export(out: Path, primary: dict, cfg: Config, nodes: dict, edges: list[dict], main_ids: list[str], core_ids: list[str], resolutions: list[dict], source_graphs: list[dict]):
    out.mkdir(parents=False, exist_ok=False)
    edge_by_id = {edge["edge_id"]: edge for edge in edges}
    outputs = {}
    for name, ids in (("connected_summary_graph", main_ids), ("main_graph", main_ids), ("paper_core", core_ids)):
        selected_edges = [edge_by_id[value] for value in ids]
        selected_nodes = sorted({edge["src"] for edge in selected_edges} | {edge["dst"] for edge in selected_edges})
        payload = {"format": "trust_trace_v3", "incident_id": primary["incident_id"],
                   "ground_truth_read": False, "synthetic_edges": 0,
                   "nodes": [nodes[value] for value in selected_nodes], "edges": selected_edges}
        path = out / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        write_dot(out / f"{name}.dot", payload["nodes"], payload["edges"], name)
        outputs[name] = (selected_nodes, selected_edges)
    main_nodes, main_edges = outputs["main_graph"]
    core_nodes, core_edges = outputs["paper_core"]
    sidecar = [{"node_id": key, "label": value["label"], "node_type": value["node_type"],
                "final_risk_rank": value.get("final_risk_rank"),
                "reason": "high_risk_without_real_selected_connection"}
               for key, value in sorted(nodes.items()) if key not in set(main_nodes) and
               value.get("final_risk_rank") is not None and value["final_risk_rank"] >= cfg.high_risk_sidecar_threshold]
    (out / "unconnected_high_risk_nodes.json").write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")
    validation = {"ground_truth_read": False, "synthetic_edge_count": 0,
                  "invalid_event_uuid_count": sum(not edge["raw_event_ids"] for edge in core_edges),
                  "main_node_count": len(main_nodes), "main_edge_count": len(main_edges),
                  "core_node_count": len(core_nodes), "core_edge_count": len(core_edges),
                  "main_weak_components": components(set(main_nodes), main_edges) if main_nodes else 0,
                  "core_weak_components": components(set(core_nodes), core_edges) if core_nodes else 0,
                  "core_stage_coverage": sorted({role for edge in core_edges for role in edge["stage_roles"]}),
                  "phase_witnesses": phase_witnesses(edge_by_id, core_ids)}
    (out / "trace_validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    (out / "entity_resolution_audit.json").write_text(json.dumps(resolutions, indent=2, sort_keys=True), encoding="utf-8")
    with (out / "candidate_edge_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["edge_id", "src", "dst", "relation", "raw_relation", "priority", "claim_eligible", "semantic_conflict", "stage_roles", "selected_main", "selected_core", "raw_event_ids"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for edge in edges:
            row = {key: json.dumps(edge.get(key)) if isinstance(edge.get(key), list) else edge.get(key) for key in fields}
            row["selected_main"], row["selected_core"] = edge["edge_id"] in main_ids, edge["edge_id"] in core_ids
            writer.writerow(row)
    manifest = {"format": "trust_trace_v3", "created_at_utc": datetime.utcnow().isoformat() + "Z",
                "incident": primary, "config": asdict(cfg), "ground_truth_read": False,
                "source_graphs": source_graphs, "sha256": {}}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["sha256"] = {path.name: sha256_file(path) for path in out.iterdir() if path.name != "run_manifest.json"}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def run_case(result_path: Path, graph_root: Path, loss_root: Path, out: Path, incident_id: str, cfg: Config):
    edge_paths = sorted(loss_root.glob("*.csv"))
    if not edge_paths:
        raise SystemExit(f"no_edge_loss_csv:{loss_root}")
    results = load_results(result_path, edge_paths)
    incidents = form_incidents(results, cfg)
    primary = next((value for value in incidents if value["incident_id"] == incident_id), None)
    if primary is None:
        raise SystemExit(f"unknown_incident:{incident_id}")
    trace_windows, context_windows = expand_windows(primary, edge_paths, cfg)
    primary = dict(primary, trace_windows=trace_windows, context_windows=context_windows)
    graphs, losses, source_graphs = {}, {}, []
    anchors = set(primary["anchors"])
    for window in trace_windows:
        begin, _ = window_bounds(edge_paths[window])
        selected = select_graph(graph_root, begin, anchors)
        if selected is None:
            continue
        _, _, graph_path, _, _, graph = selected
        graphs[window], losses[window] = graph, load_losses(edge_paths[window])
        source_graphs.append({"window_id": window, "graph_path": str(graph_path),
                              "graph_sha256": sha256_file(graph_path),
                              "edge_loss_csv": str(edge_paths[window]),
                              "edge_loss_csv_sha256": sha256_file(edge_paths[window]),
                              "is_context_window": window in context_windows})
    nodes, raw_events, resolutions = collect(primary, results, graphs, losses)
    edges = aggregate(raw_events, cfg)
    score_edges(edges, nodes)
    canonical_anchors = {key for key, value in nodes.items() if value["is_anchor"]}
    main_ids = select_edges(edges, nodes, canonical_anchors, cfg, core=False)
    core_ids = select_edges(edges, nodes, canonical_anchors, cfg, core=True)
    export(out, primary, cfg, nodes, edges, main_ids, core_ids, resolutions, source_graphs)


def self_test():
    assert relation_semantics("EVENT_CONNECT", "process", "network")[0] == "network_connect"
    assert relation_semantics("EVENT_SENDTO", "process", "network")[0] == "network_send"
    assert relation_semantics("EVENT_WRITE", "process", "network")[3] is True
    assert flow_endpoints("file:a", "process:b", "EVENT_EXECUTE", "file", "process") == ("file:a", "process:b")
    test_nodes = {value: {"final_risk_rank": 1.0 if value == "p2" else 0.0}
                  for value in ("p1", "f", "p2", "n", "remote")}
    test_edges = [
        {"edge_id": "w", "src": "p1", "dst": "f", "priority": 2.0, "first_seen_ns": 1,
         "stage_roles": ["write_or_landing"], "claim_eligible": True, "semantic_conflict": False,
         "relevance_reasons": []},
        {"edge_id": "x", "src": "f", "dst": "p2", "priority": 3.0, "first_seen_ns": 2,
         "stage_roles": ["execution"], "claim_eligible": True, "semantic_conflict": False,
         "relevance_reasons": ["direct_alert"]},
        {"edge_id": "n", "src": "p2", "dst": "n", "priority": 2.0, "first_seen_ns": 3,
         "stage_roles": ["outbound_communication"], "claim_eligible": True, "semantic_conflict": False,
         "relevance_reasons": ["direct_alert"]},
        {"edge_id": "i", "src": "remote", "dst": "p1", "priority": 1.0, "first_seen_ns": 0,
         "stage_roles": ["network_ingress"], "claim_eligible": True, "semantic_conflict": False,
         "relevance_reasons": []},
    ]
    selected = select_edges(test_edges, test_nodes, {"p2"}, Config(), core=True)
    chosen = [edge for edge in test_edges if edge["edge_id"] in selected]
    assert components({node for edge in chosen for node in (edge["src"], edge["dst"])}, chosen) == 1
    assert phase_witnesses({edge["edge_id"]: edge for edge in test_edges}, selected)
    print("trust_trace_v3_self_test: PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result")
    parser.add_argument("--graphs-dir")
    parser.add_argument("--edge-loss-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--incident-id")
    parser.add_argument("--config-json", default="{}")
    parser.add_argument("--list-incidents", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.result or not args.edge_loss_dir:
        raise SystemExit("--result and --edge-loss-dir are required")
    cfg = Config(**json.loads(args.config_json))
    result_path, loss_root = Path(args.result), Path(args.edge_loss_dir)
    edge_paths = sorted(loss_root.glob("*.csv"))
    if args.list_incidents:
        print(json.dumps(form_incidents(load_results(result_path, edge_paths), cfg), indent=2, sort_keys=True))
        return
    if not args.graphs_dir or not args.output_dir or not args.incident_id:
        raise SystemExit("--graphs-dir, --output-dir and --incident-id are required")
    run_case(result_path, Path(args.graphs_dir), loss_root, Path(args.output_dir), args.incident_id, cfg)
    print(json.dumps({"output_dir": args.output_dir, "incident_id": args.incident_id, "ground_truth_read": False}, sort_keys=True))


if __name__ == "__main__":
    main()
