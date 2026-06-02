#!/usr/bin/env bash
set -euo pipefail

ALLOWED_ROOT="/data2/jiangjiaqi/yzh"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/MMLU}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
TRAIN_RUN_ID="${TRAIN_RUN_ID:-deepseek-v4-flash-mmlu-dev}"
EVAL_RUN_ID="${EVAL_RUN_ID:-deepseek-v4-flash-mmlu-test}"
WORKERS="${WORKERS:-16}"

assert_under_allowed_root() {
  local resolved
  resolved="$(readlink -m "$1")"
  case "${resolved}" in
    "${ALLOWED_ROOT}"|"${ALLOWED_ROOT}"/*) ;;
    *)
      echo "refusing path outside ${ALLOWED_ROOT}: ${resolved}" >&2
      exit 1
      ;;
  esac
}

assert_under_allowed_root "${PROJECT_ROOT}"
assert_under_allowed_root "${DATA_ROOT}"
assert_under_allowed_root "${ARTIFACT_ROOT}"
test -d "${DATA_ROOT}/dev"
test -d "${DATA_ROOT}/test"

if [[ -t 0 ]]; then
  read -r -s -p "DeepSeek API key: " API_KEY
  echo
else
  IFS= read -r API_KEY
fi
test -n "${API_KEY}"

run_with_key() {
  printf '%s\n' "${API_KEY}" | python -m gogagent.cli "$@" \
    --api-key-stdin \
    --base-url "https://api.deepseek.com" \
    --model "deepseek-v4-flash" \
    --max-tokens 256
}

train_args=(
  train-mmlu
  --data-path "${DATA_ROOT}/dev"
  --split dev
  --artifact-root "${ARTIFACT_ROOT}/training"
  --run-id "${TRAIN_RUN_ID}"
  --resume
)
if [[ -n "${TRAIN_LIMIT:-}" ]]; then
  train_args+=(--limit "${TRAIN_LIMIT}")
fi
run_with_key "${train_args[@]}"

eval_args=(
  eval
  --dataset mmlu
  --data-path "${DATA_ROOT}/test"
  --split test
  --artifact-root "${ARTIFACT_ROOT}/evals"
  --run-id "${EVAL_RUN_ID}"
  --workers "${WORKERS}"
  --gog-memory "${ARTIFACT_ROOT}/training/${TRAIN_RUN_ID}/memory.json"
  --resume
)
if [[ -n "${EVAL_LIMIT:-}" ]]; then
  eval_args+=(--limit "${EVAL_LIMIT}")
fi
run_with_key "${eval_args[@]}"

unset API_KEY
