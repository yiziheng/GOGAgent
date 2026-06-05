"""Small graph artifact writers for JSON and SVG output."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import html
import json
from pathlib import Path
from typing import Any, Mapping


def save_graph_json(graph: Any, path: str | Path) -> Path:
    """Save a serializable ``gog.json`` representation."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(graph_to_dict(graph), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def save_graph_svg(graph: Any, path: str | Path) -> Path:
    """Save a simple visible SVG with nodes, edges, and subgraph markers."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph_dict = graph_to_dict(graph)
    nodes = _node_list(graph_dict)
    edges = _edge_list(graph_dict)
    svg = render_graph_svg(nodes, edges)
    output_path.write_text(svg, encoding="utf-8")
    return output_path


def render_graph_svg(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> str:
    """Render a compact left-to-right graph SVG."""

    width = max(360, 180 * max(len(nodes), 1))
    height = 180
    positions: dict[str, tuple[int, int]] = {}
    for index, node in enumerate(nodes):
        node_id = str(node.get("node_id", node.get("id", f"node_{index}")))
        positions[node_id] = (80 + index * 160, 80)

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" '
        'orient="auto" markerUnits="strokeWidth">',
        '<path d="M0,0 L0,6 L9,3 z" fill="#333" />',
        "</marker>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]

    for edge in edges:
        src = str(edge.get("src", edge.get("source", edge.get("from", ""))))
        dst = str(edge.get("dst", edge.get("target", edge.get("to", ""))))
        if src not in positions or dst not in positions:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        lines.append(
            f'<line x1="{x1 + 45}" y1="{y1}" x2="{x2 - 45}" y2="{y2}" '
            'stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>'
        )

    for index, node in enumerate(nodes):
        node_id = str(node.get("node_id", node.get("id", f"node_{index}")))
        x, y = positions[node_id]
        label = _node_label(node, node_id)
        subgraph = _is_subgraph_node(node)
        stroke = "#7c3aed" if subgraph else "#2563eb"
        fill = "#f5f3ff" if subgraph else "#eff6ff"
        dash = ' stroke-dasharray="5 3"' if subgraph else ""
        lines.append(
            f'<rect x="{x - 55}" y="{y - 28}" width="110" height="56" rx="10" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2"{dash}/>'
        )
        lines.append(
            f'<text x="{x}" y="{y - 4}" text-anchor="middle" '
            'font-family="Arial, sans-serif" font-size="12" fill="#111827">'
            f"{html.escape(label)}</text>"
        )
        if subgraph:
            lines.append(
                f'<text x="{x}" y="{y + 16}" text-anchor="middle" '
                'font-family="Arial, sans-serif" font-size="10" fill="#6d28d9">'
                "subgraph</text>"
            )

    lines.append("</svg>")
    return "\n".join(lines)


def graph_to_dict(graph: Any) -> dict[str, Any]:
    """Convert Graph-like objects to plain JSON-compatible dictionaries."""

    if graph is None:
        return {"nodes": [], "edges": []}
    if isinstance(graph, Mapping):
        return _jsonable(dict(graph))
    to_dict = getattr(graph, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return _jsonable(dict(converted))
    if is_dataclass(graph):
        return _jsonable(asdict(graph))

    nodes = getattr(graph, "nodes", {})
    edges = getattr(graph, "edges", [])
    return _jsonable(
        {
            "in_node": getattr(graph, "in_node", None),
            "out_node": getattr(graph, "out_node", None),
            "nodes": nodes,
            "edges": edges,
        }
    )


def _node_list(graph_dict: Mapping[str, Any]) -> list[dict[str, Any]]:
    nodes = graph_dict.get("nodes", [])
    if isinstance(nodes, Mapping):
        return [
            {"node_id": str(node_id), **(node if isinstance(node, Mapping) else {"value": node})}
            for node_id, node in nodes.items()
        ]
    if isinstance(nodes, list):
        return [
            node if isinstance(node, dict) else {"node_id": f"node_{index}", "value": node}
            for index, node in enumerate(nodes)
        ]
    return []


def _edge_list(graph_dict: Mapping[str, Any]) -> list[dict[str, Any]]:
    edges = graph_dict.get("edges", [])
    if isinstance(edges, list):
        return [edge if isinstance(edge, dict) else graph_to_dict(edge) for edge in edges]
    return []


def _node_label(node: Mapping[str, Any], fallback: str) -> str:
    return str(node.get("name") or node.get("role") or node.get("node_id") or fallback)


def _is_subgraph_node(node: Mapping[str, Any]) -> bool:
    if int(node.get("depth", 1) or 1) > 1:
        return True
    if str(node.get("node_kind", "")).lower() == "graph":
        return True
    executor = node.get("executor")
    return isinstance(executor, Mapping) and "nodes" in executor


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if is_dataclass(value):
        return _jsonable(asdict(value))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
