"""Dependency-free JSON and SVG exports for generated agent graphs."""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from dataclasses import asdict, is_dataclass
from enum import Enum
from html import escape
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from gogagent.core.types import OrgGraphSnapshot, SimilarityEdge, TransitionEdge


_NODE_WIDTH = 180
_NODE_HEIGHT = 62
_GRAPH_WIDTH = 220
_GRAPH_HEIGHT = 86
_MARGIN = 42


def export_snapshot(
    snapshot: OrgGraphSnapshot, directory: str | Path
) -> tuple[Path, Path]:
    """Write one executable MAS DAG as ``<graph_id>.json`` and SVG."""

    output_dir = _ensure_directory(directory)
    stem = _safe_filename(snapshot.graph_id)
    json_path = output_dir / f"{stem}.json"
    svg_path = output_dir / f"{stem}.svg"
    _write_json(json_path, snapshot.to_dict())
    svg_path.write_text(_render_snapshot_svg(snapshot), encoding="utf-8")
    return json_path, svg_path


def export_gog(
    snapshots: Sequence[OrgGraphSnapshot],
    transitions: Sequence[TransitionEdge],
    similarities: Sequence[SimilarityEdge],
    directory: str | Path,
) -> tuple[Path, Path]:
    """Write the outer Organization GoG as ``gog.json`` and ``gog.svg``."""

    output_dir = _ensure_directory(directory)
    json_path = output_dir / "gog.json"
    svg_path = output_dir / "gog.svg"
    _write_json(
        json_path,
        {
            "snapshots": [snapshot.to_dict() for snapshot in snapshots],
            "transitions": [transition.to_dict() for transition in transitions],
            "similarities": [similarity.to_dict() for similarity in similarities],
        },
    )
    svg_path.write_text(
        _render_gog_svg(snapshots, transitions, similarities), encoding="utf-8"
    )
    return json_path, svg_path


def _ensure_directory(directory: str | Path) -> Path:
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _safe_filename(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return stem or "graph"


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(
            data,
            default=_json_default,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _topological_levels(snapshot: OrgGraphSnapshot) -> dict[str, int]:
    node_ids = [node.node_id for node in snapshot.nodes]
    known_nodes = set(node_ids)
    indegree = {node_id: 0 for node_id in node_ids}
    outgoing: dict[str, list[str]] = defaultdict(list)
    for edge in snapshot.edges:
        if edge.src not in known_nodes or edge.dst not in known_nodes:
            continue
        outgoing[edge.src].append(edge.dst)
        indegree[edge.dst] += 1

    queue = deque(node_id for node_id in node_ids if indegree[node_id] == 0)
    levels = {node_id: 0 for node_id in queue}
    while queue:
        src = queue.popleft()
        for dst in outgoing[src]:
            levels[dst] = max(levels.get(dst, 0), levels[src] + 1)
            indegree[dst] -= 1
            if indegree[dst] == 0:
                queue.append(dst)

    # ConstraintEngine should guarantee a DAG. The fallback keeps malformed
    # snapshots inspectable, which is useful while debugging graph generation.
    next_level = max(levels.values(), default=-1) + 1
    for node_id in node_ids:
        if node_id not in levels:
            levels[node_id] = next_level
            next_level += 1
    return levels


def _positions_by_level(
    levels: Mapping[str, int],
    *,
    item_width: int,
    item_height: int,
    x_gap: int,
    y_gap: int,
) -> tuple[dict[str, tuple[int, int]], int, int]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for item_id, level in levels.items():
        grouped[level].append(item_id)
    for ids in grouped.values():
        ids.sort()

    max_items = max((len(ids) for ids in grouped.values()), default=1)
    max_level = max(grouped, default=0)
    content_height = max_items * item_height + max(0, max_items - 1) * y_gap
    positions: dict[str, tuple[int, int]] = {}
    for level, ids in grouped.items():
        level_height = len(ids) * item_height + max(0, len(ids) - 1) * y_gap
        y_offset = _MARGIN + (content_height - level_height) // 2
        for row, item_id in enumerate(ids):
            positions[item_id] = (
                _MARGIN + level * (item_width + x_gap),
                y_offset + row * (item_height + y_gap),
            )

    width = 2 * _MARGIN + (max_level + 1) * item_width + max_level * x_gap
    height = 2 * _MARGIN + content_height
    return positions, max(width, 360), max(height, 180)


def _svg_document(width: int, height: int, body: Iterable[str]) -> str:
    content = "\n".join(body)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#49657a"/>
    </marker>
    <style>
      text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #17324d; }}
      .title {{ font-size: 16px; font-weight: 700; }}
      .label {{ font-size: 13px; font-weight: 600; }}
      .detail {{ font-size: 11px; fill: #49657a; }}
      .edge-label {{ font-size: 11px; fill: #334e68; paint-order: stroke; stroke: #ffffff; stroke-width: 4px; }}
    </style>
  </defs>
  <rect width="100%" height="100%" fill="#f8fbfd"/>
{content}
</svg>
"""


def _render_snapshot_svg(snapshot: OrgGraphSnapshot) -> str:
    levels = _topological_levels(snapshot)
    positions, width, layout_height = _positions_by_level(
        levels,
        item_width=_NODE_WIDTH,
        item_height=_NODE_HEIGHT,
        x_gap=126,
        y_gap=36,
    )
    title_height = 56
    height = layout_height + title_height
    shifted = {
        node_id: (x, y + title_height) for node_id, (x, y) in positions.items()
    }
    body = [
        f'  <text x="{_MARGIN}" y="29" class="title">Executable MAS DAG: {escape(snapshot.graph_id)}</text>',
        f'  <text x="{_MARGIN}" y="48" class="detail">domain={escape(snapshot.domain)} | step={snapshot.step}</text>',
    ]

    for edge in snapshot.edges:
        if edge.src not in shifted or edge.dst not in shifted:
            continue
        src_x, src_y = shifted[edge.src]
        dst_x, dst_y = shifted[edge.dst]
        x1, y1 = src_x + _NODE_WIDTH, src_y + _NODE_HEIGHT // 2
        x2, y2 = dst_x, dst_y + _NODE_HEIGHT // 2
        mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2 - 7
        body.append(
            f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#49657a" stroke-width="2" marker-end="url(#arrow)"/>'
        )
        body.append(
            f'  <text x="{mid_x}" y="{mid_y}" text-anchor="middle" class="edge-label">{escape(edge.payload)}</text>'
        )

    for node in snapshot.nodes:
        x, y = shifted[node.node_id]
        body.append(
            f'  <rect x="{x}" y="{y}" width="{_NODE_WIDTH}" height="{_NODE_HEIGHT}" rx="10" fill="#e5f4ff" stroke="#2680b3" stroke-width="2"/>'
        )
        body.append(
            f'  <text x="{x + 12}" y="{y + 25}" class="label">{escape(node.role)}</text>'
        )
        body.append(
            f'  <text x="{x + 12}" y="{y + 45}" class="detail">id={escape(node.node_id)} | runner={escape(node.runner)}</text>'
        )
    return _svg_document(width, height, body)


def _render_gog_svg(
    snapshots: Sequence[OrgGraphSnapshot],
    transitions: Sequence[TransitionEdge],
    similarities: Sequence[SimilarityEdge],
) -> str:
    levels = {snapshot.graph_id: snapshot.step for snapshot in snapshots}
    positions, width, layout_height = _positions_by_level(
        levels,
        item_width=_GRAPH_WIDTH,
        item_height=_GRAPH_HEIGHT,
        x_gap=160,
        y_gap=54,
    )
    title_height = 62
    height = layout_height + title_height
    shifted = {
        graph_id: (x, y + title_height) for graph_id, (x, y) in positions.items()
    }
    body = [
        f'  <text x="{_MARGIN}" y="29" class="title">Organization Graph-of-Graphs</text>',
        f'  <text x="{_MARGIN}" y="49" class="detail">{len(snapshots)} graph snapshots | {len(transitions)} edit transitions | {len(similarities)} similarity edges</text>',
    ]

    for edge in similarities:
        if edge.src_graph_id not in shifted or edge.dst_graph_id not in shifted:
            continue
        x1, y1 = _center(shifted[edge.src_graph_id], _GRAPH_WIDTH, _GRAPH_HEIGHT)
        x2, y2 = _center(shifted[edge.dst_graph_id], _GRAPH_WIDTH, _GRAPH_HEIGHT)
        body.append(
            f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#9f7aea" stroke-width="2" stroke-dasharray="7 6" opacity="0.72"/>'
        )
        body.append(
            f'  <text x="{(x1 + x2) // 2}" y="{(y1 + y2) // 2 - 7}" text-anchor="middle" class="edge-label">similarity={edge.similarity:.2f}</text>'
        )

    for edge in transitions:
        if edge.src_graph_id not in shifted or edge.dst_graph_id not in shifted:
            continue
        src_x, src_y = shifted[edge.src_graph_id]
        dst_x, dst_y = shifted[edge.dst_graph_id]
        x1, y1 = src_x + _GRAPH_WIDTH, src_y + _GRAPH_HEIGHT // 2
        x2, y2 = dst_x, dst_y + _GRAPH_HEIGHT // 2
        body.append(
            f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#49657a" stroke-width="2.5" marker-end="url(#arrow)"/>'
        )
        body.append(
            f'  <text x="{(x1 + x2) // 2}" y="{(y1 + y2) // 2 - 8}" text-anchor="middle" class="edge-label">{escape(edge.action.value)}</text>'
        )

    for snapshot in snapshots:
        x, y = shifted[snapshot.graph_id]
        roles = ", ".join(node.role for node in snapshot.nodes) or "(empty)"
        body.append(
            f'  <rect x="{x}" y="{y}" width="{_GRAPH_WIDTH}" height="{_GRAPH_HEIGHT}" rx="12" fill="#fff7e6" stroke="#d97706" stroke-width="2"/>'
        )
        body.append(
            f'  <text x="{x + 13}" y="{y + 24}" class="label">{escape(snapshot.graph_id)}</text>'
        )
        body.append(
            f'  <text x="{x + 13}" y="{y + 44}" class="detail">domain={escape(snapshot.domain)} | step={snapshot.step}</text>'
        )
        body.append(
            f'  <text x="{x + 13}" y="{y + 64}" class="detail">roles={escape(_truncate(roles, 31))}</text>'
        )
    return _svg_document(width, height, body)


def _center(position: tuple[int, int], width: int, height: int) -> tuple[int, int]:
    return position[0] + width // 2, position[1] + height // 2


def _truncate(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    return value[: max_length - 3] + "..."
