"""Validated, content-addressed skeleton definitions loaded from YAML."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import yaml


Edge = Tuple[int, int]
SymmetryPair = Tuple[int, int]


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate mapping keys when loading a skeleton YAML document."""
    pass


def _construct_unique_mapping(loader, node, deep=False):
    """Build one YAML mapping while rejecting duplicate keys."""
    loader.flatten_mapping(node)
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"Duplicate YAML key: {key!r}.")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class SkeletonDefinition:
    """Immutable point ordering and topology contract for one skeleton."""

    skeleton_id: str
    point_count: int
    root_index: int
    model_graph_edges: Tuple[Edge, ...]
    bone_vector_edges: Tuple[Edge, ...]
    symmetry_pairs: Tuple[SymmetryPair, ...]
    channels: Tuple[str, ...]
    frame_count: Optional[int]
    config_hash: str
    node_ids: Tuple[str, ...]


def _require_mapping(value, name):
    """Validate that a configuration value is a string-keyed mapping."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping.")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings.")
    return value


def _require_positive_int(value, name):
    """Validate a strictly positive integer while excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _parse_edges(value, name, point_count, *, undirected=False, allow_empty=False):
    """Validate and normalize a list of point-index edges."""
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of index pairs.")
    if not value and not allow_empty:
        raise ValueError(f"{name} must not be empty.")

    edges = []
    seen = set()
    for position, pair in enumerate(value):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"{name}[{position}] must contain two point indices.")
        left, right = pair
        if any(isinstance(index, bool) or not isinstance(index, int) for index in pair):
            raise ValueError(f"{name}[{position}] indices must be integers.")
        if not (0 <= left < point_count and 0 <= right < point_count):
            raise ValueError(f"{name}[{position}]={pair} is outside point_count={point_count}.")
        if left == right:
            raise ValueError(f"{name}[{position}] must not be a self-edge.")
        key = (min(left, right), max(left, right)) if undirected else (left, right)
        if key in seen:
            raise ValueError(f"{name} contains a duplicate edge: {key}.")
        seen.add(key)
        edges.append((left, right))

    if undirected:
        edges = sorted((min(left, right), max(left, right)) for left, right in edges)
    return tuple(edges)


def _require_connected_graph(edges, point_count):
    """Ensure every declared node is reachable in the undirected model graph."""
    if point_count == 1:
        return
    adjacency = [[] for _ in range(point_count)]
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    visited = {0}
    frontier = [0]
    while frontier:
        current = frontier.pop()
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)
    if len(visited) != point_count:
        raise ValueError("model_graph_edges must connect every declared node.")


def _validate_bone_tree(edges, point_count, root_index):
    """Ensure every non-root node has exactly one route to the root."""
    if len(edges) != point_count - 1:
        raise ValueError("bone_vector_edges must contain exactly one edge per non-root node.")
    outputs = [output for output, _ in edges]
    expected = set(range(point_count)) - {root_index}
    if len(outputs) != len(set(outputs)) or set(outputs) != expected:
        raise ValueError("bone_vector_edges must assign each non-root node exactly once.")

    parent = dict(edges)
    for start in expected:
        current = start
        visited = set()
        while current != root_index:
            if current in visited:
                raise ValueError("bone_vector_edges contains a cycle outside the root.")
            visited.add(current)
            current = parent[current]


def _normalize_config(raw):
    """Validate and canonically order a version-1 skeleton mapping."""
    allowed = {
        "schema_version",
        "skeleton_id",
        "point_count",
        "root_index",
        "nodes",
        "model_graph_edges",
        "bone_vector_edges",
        "symmetry_pairs",
        "channels",
        "frame_count",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown skeleton configuration fields: {unknown}.")
    required = allowed - {"symmetry_pairs", "frame_count"}
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"Missing skeleton configuration fields: {missing}.")
    if (
        isinstance(raw["schema_version"], bool)
        or not isinstance(raw["schema_version"], int)
        or raw["schema_version"] != 1
    ):
        raise ValueError("schema_version must be 1.")

    skeleton_id = raw["skeleton_id"]
    if not isinstance(skeleton_id, str) or not skeleton_id.strip():
        raise ValueError("skeleton_id must be a non-empty string.")
    point_count = _require_positive_int(raw["point_count"], "point_count")
    root_index = raw["root_index"]
    if isinstance(root_index, bool) or not isinstance(root_index, int):
        raise ValueError("root_index must be an integer.")
    if not 0 <= root_index < point_count:
        raise ValueError("root_index must be inside point_count.")

    nodes = raw["nodes"]
    if not isinstance(nodes, list) or len(nodes) != point_count:
        raise ValueError("nodes must contain exactly point_count entries.")
    parsed_nodes = []
    for position, node in enumerate(nodes):
        if not isinstance(node, dict) or set(node) != {"id", "index"}:
            raise ValueError(f"nodes[{position}] must contain exactly id and index.")
        node_id = node["id"]
        index = node["index"]
        if not isinstance(node_id, str) or not node_id.strip():
            raise ValueError(f"nodes[{position}].id must be a non-empty string.")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"nodes[{position}].index must be an integer.")
        parsed_nodes.append((index, node_id.strip()))
    if len({node_id for _, node_id in parsed_nodes}) != point_count:
        raise ValueError("node ids must be unique.")
    if sorted(index for index, _ in parsed_nodes) != list(range(point_count)):
        raise ValueError("node indices must be unique and contiguous from 0 to point_count - 1.")
    node_ids = tuple(node_id for _, node_id in sorted(parsed_nodes))

    model_graph_edges = _parse_edges(
        raw["model_graph_edges"],
        "model_graph_edges",
        point_count,
        undirected=True,
        allow_empty=point_count == 1,
    )
    _require_connected_graph(model_graph_edges, point_count)

    bone_vector_edges = _parse_edges(
        raw["bone_vector_edges"],
        "bone_vector_edges",
        point_count,
    )
    _validate_bone_tree(bone_vector_edges, point_count, root_index)
    bone_vector_edges = tuple(sorted(bone_vector_edges, key=lambda edge: edge[0]))

    symmetry_pairs = _parse_edges(
        raw.get("symmetry_pairs", []),
        "symmetry_pairs",
        point_count,
        undirected=True,
        allow_empty=True,
    )
    paired_nodes = [node for pair in symmetry_pairs for node in pair]
    if root_index in paired_nodes:
        raise ValueError("root_index must not participate in symmetry_pairs.")
    if len(paired_nodes) != len(set(paired_nodes)):
        raise ValueError("symmetry_pairs must be disjoint.")

    channels = raw["channels"]
    if (
        not isinstance(channels, list)
        or not channels
        or any(not isinstance(channel, str) or not channel.strip() for channel in channels)
    ):
        raise ValueError("channels must be a non-empty list of channel names.")
    channels = tuple(channel.strip() for channel in channels)
    if len(channels) != len(set(channels)):
        raise ValueError("channels must be unique.")

    frame_count = raw.get("frame_count")
    if frame_count is not None:
        frame_count = _require_positive_int(frame_count, "frame_count")

    normalized = {
        "schema_version": 1,
        "skeleton_id": skeleton_id.strip(),
        "point_count": point_count,
        "root_index": root_index,
        "nodes": [
            {"id": node_id, "index": index}
            for index, node_id in enumerate(node_ids)
        ],
        "model_graph_edges": [list(edge) for edge in model_graph_edges],
        "bone_vector_edges": [list(edge) for edge in bone_vector_edges],
        "symmetry_pairs": [list(pair) for pair in symmetry_pairs],
        "channels": list(channels),
        "frame_count": frame_count,
    }
    return normalized


def skeleton_from_mapping(raw: dict) -> SkeletonDefinition:
    """Create a validated skeleton from an embedded version-1 mapping."""
    raw = _require_mapping(raw, "skeleton configuration")
    normalized = _normalize_config(raw)
    canonical = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    config_hash = hashlib.sha256(canonical).hexdigest()
    return SkeletonDefinition(
        skeleton_id=normalized["skeleton_id"],
        point_count=normalized["point_count"],
        root_index=normalized["root_index"],
        model_graph_edges=tuple(tuple(edge) for edge in normalized["model_graph_edges"]),
        bone_vector_edges=tuple(tuple(edge) for edge in normalized["bone_vector_edges"]),
        symmetry_pairs=tuple(tuple(pair) for pair in normalized["symmetry_pairs"]),
        channels=tuple(normalized["channels"]),
        frame_count=normalized["frame_count"],
        config_hash=config_hash,
        node_ids=tuple(node["id"] for node in normalized["nodes"]),
    )


def load_skeleton(path: str | Path) -> SkeletonDefinition:
    """Load, validate, and hash a version-1 skeleton YAML file."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Skeleton YAML file does not exist: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            raw = yaml.load(stream, Loader=_UniqueKeyLoader)
    except (yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"Invalid skeleton YAML {config_path}: {exc}") from exc
    return skeleton_from_mapping(raw)
