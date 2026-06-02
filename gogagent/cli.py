"""Production CLI for training GoG memory and evaluating real benchmarks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from gogagent.evaluation import BenchmarkRunner, EvaluationConfig
from gogagent.llm import OpenAICompatibleLLM


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real GOGAgent workflows")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("eval", help="evaluate a real benchmark split")
    _add_backend_arguments(evaluate)
    evaluate.add_argument("--dataset", choices=("gsm8k", "mmlu", "humaneval"), required=True)
    evaluate.add_argument("--data-path", type=Path, required=True)
    evaluate.add_argument("--split", default="test")
    evaluate.add_argument("--artifact-root", type=Path, default=Path("artifacts/evals"))
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--workers", type=int, default=1)
    evaluate.add_argument("--start-index", type=int, default=0)
    evaluate.add_argument("--limit", type=int, default=None)
    evaluate.add_argument("--resume", action="store_true")
    evaluate.add_argument("--gog-memory", type=Path, default=None)

    train = subparsers.add_parser("train-mmlu", help="build GoG memory from an MMLU split")
    _add_backend_arguments(train)
    train.add_argument("--data-path", type=Path, required=True)
    train.add_argument("--split", default="dev")
    train.add_argument("--artifact-root", type=Path, default=Path("artifacts/training"))
    train.add_argument("--run-id", required=True)
    train.add_argument("--start-index", type=int, default=0)
    train.add_argument("--limit", type=int, default=None)
    train.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = _backend(args)
    print(json.dumps({"backend": backend.describe()}, ensure_ascii=False))
    if args.command == "eval":
        summary = BenchmarkRunner(
            EvaluationConfig(
                dataset=args.dataset,
                data_path=args.data_path,
                artifact_root=args.artifact_root,
                run_id=args.run_id,
                split=args.split,
                workers=args.workers,
                start_index=args.start_index,
                limit=args.limit,
                resume=args.resume,
                gog_memory=args.gog_memory,
            ),
            backend,
        ).run()
    else:
        from gogagent.training.mmlu_runner import MMLUMemoryTrainer, MMLUTrainingConfig

        summary = MMLUMemoryTrainer(
            MMLUTrainingConfig(
                data_path=args.data_path,
                artifact_root=args.artifact_root,
                run_id=args.run_id,
                split=args.split,
                start_index=args.start_index,
                limit=args.limit,
                resume=args.resume,
            ),
            backend,
        ).run()
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _add_backend_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", default=os.environ.get("GOGAGENT_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("GOGAGENT_MODEL", DEFAULT_MODEL))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("GOGAGENT_TIMEOUT_SECONDS", "120")))
    parser.add_argument("--max-retries", type=int, default=int(os.environ.get("GOGAGENT_MAX_RETRIES", "2")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("GOGAGENT_MAX_TOKENS", "1024")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("GOGAGENT_TEMPERATURE", "0")))
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default=None)
    parser.add_argument(
        "--api-key-stdin",
        action="store_true",
        help="read one API key line from stdin without persisting it",
    )


def _backend(args: Any) -> OpenAICompatibleLLM:
    api_key = sys.stdin.readline().rstrip("\r\n") if args.api_key_stdin else None
    if args.api_key_stdin and not api_key:
        raise RuntimeError("--api-key-stdin requires one non-empty line on stdin")
    return OpenAICompatibleLLM(
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        timeout=args.timeout,
        max_retries=args.max_retries,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        thinking=args.thinking,
    )


if __name__ == "__main__":
    main()
