#!/usr/bin/env bash
set -euo pipefail

ALLOWED_ROOT="/data2/jiangjiaqi/yzh"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-${PROJECT_ROOT}/data/MMLU_subsets/train_test150}"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-${PROJECT_ROOT}/data/MMLU_subsets/eval_gptswarm_val153}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
TRAIN_RUN_ID="${TRAIN_RUN_ID:-deepseek-v4-flash-mmlu-train-test150}"
EVAL_RUN_ID="${EVAL_RUN_ID:-deepseek-v4-flash-mmlu-eval-gptswarm-val153}"
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-${ARTIFACT_ROOT}/policies/${TRAIN_RUN_ID}.pt}"
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
assert_under_allowed_root "${TRAIN_DATA_ROOT}"
assert_under_allowed_root "${EVAL_DATA_ROOT}"
assert_under_allowed_root "${ARTIFACT_ROOT}"
test -d "${TRAIN_DATA_ROOT}/test"
test -d "${EVAL_DATA_ROOT}/val"

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
    --max-tokens 256 \
    --thinking disabled
}

train_args=(
  train-mmlu
  --data-path "${TRAIN_DATA_ROOT}/test"
  --split test
  --artifact-root "${ARTIFACT_ROOT}/training"
  --run-id "${TRAIN_RUN_ID}"
  --policy-checkpoint-out "${POLICY_CHECKPOINT}"
  --resume
)
if [[ -n "${TRAIN_LIMIT:-}" ]]; then
  train_args+=(--limit "${TRAIN_LIMIT}")
fi
run_with_key "${train_args[@]}"

eval_args=(
  eval
  --dataset mmlu
  --data-path "${EVAL_DATA_ROOT}/val"
  --split val
  --artifact-root "${ARTIFACT_ROOT}/evals"
  --run-id "${EVAL_RUN_ID}"
  --workers "${WORKERS}"
  --policy-checkpoint "${POLICY_CHECKPOINT}"
  --resume
)
if [[ -n "${EVAL_LIMIT:-}" ]]; then
  eval_args+=(--limit "${EVAL_LIMIT}")
fi
run_with_key "${eval_args[@]}"

SUMMARY_PATH="${ARTIFACT_ROOT}/evals/${EVAL_RUN_ID}/summary.json"
if [[ -f "scripts/record_best_run.py" && -f "${SUMMARY_PATH}" ]]; then
  python scripts/record_best_run.py \
    --summary "${SUMMARY_PATH}" \
    --dataset mmlu \
    --eval-data "${EVAL_DATA_ROOT}/val" \
    --eval-split val \
    --eval-run-id "${EVAL_RUN_ID}" \
    --policy-checkpoint "${POLICY_CHECKPOINT}" \
    --train-run-id "${TRAIN_RUN_ID}" \
    --train-data "${TRAIN_DATA_ROOT}/test" \
    --registry-dir "${ARTIFACT_ROOT}/best_runs" \
    --model deepseek-v4-flash \
    --base-url https://api.deepseek.com \
    --max-tokens 256 \
    --thinking disabled
fi

unset API_KEY
