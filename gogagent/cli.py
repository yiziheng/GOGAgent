"""Small CLI for the refactored GOG runtime."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

from gogagent.actions import ACTION_ORDER, get_action_spec
from gogagent.agents import list_agent_specs
from gogagent.artifacts import save_graph_svg
from gogagent.graph import load_graph


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect refactored GOGAgent runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("agents", help="print registered agent specs as JSON")
    subparsers.add_parser("actions", help="print action specs as JSON")

    render = subparsers.add_parser("render", help="render a saved gog.json to SVG")
    render.add_argument("--graph-json", type=Path, required=True)
    render.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "agents":
        print_json(list_agent_specs())
    elif args.command == "actions":
        print_json([get_action_spec(action).__dict__ for action in ACTION_ORDER])
    elif args.command == "render":
        graph = load_graph(args.graph_json)
        output_path = save_graph_svg(graph, args.out)
        print_json({"svg": str(output_path)})
    else:
        raise AssertionError(f"unhandled command {args.command!r}")


def print_json(value: Any) -> None:
    print(json.dumps(to_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True))


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    main()
