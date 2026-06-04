#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

TRAIN_DATA=""
TRAIN_SPLIT="test"
TRAIN_RUN_ID="mmlu-policy-train"
TRAIN_LIMIT=""
TRAIN_START_INDEX="0"

EVAL_DATA=""
EVAL_DATASET="mmlu"
EVAL_SPLIT="val"
EVAL_RUN_ID="mmlu-policy-eval"
EVAL_LIMIT=""
EVAL_START_INDEX="0"
WORKERS="1"

ARTIFACT_ROOT="${PROJECT_ROOT}/artifacts"
MODEL_CHECKPOINT="${PROJECT_ROOT}/artifacts/policies/gog_gnn_policy.pt"
LOAD_MODEL=""
RESUME="1"

BASE_URL="https://api.deepseek.com"
MODEL="deepseek-v4-flash"
MAX_TOKENS="1024"
TEMPERATURE="0"
THINKING="disabled"

POLICY_LEARNING_RATE="0.01"
POLICY_GAMMA="0.9"
POLICY_EPSILON="0.05"

usage() {
  cat <<'EOF'
Run GOGAgent as one pipeline:
  optional MMLU train -> save torch GNN policy -> eval with saved policy.

Usage:
  bash scripts/run_policy_train_eval.sh --eval-data PATH [options]

Core options:
  --train-data PATH          MMLU training directory containing *_<split>.csv.
                             Empty/omitted/none/skip means skip training.
  --train-split SPLIT        Train split suffix. Default: test.
  --eval-data PATH           Evaluation data path. Required.
  --eval-dataset NAME        mmlu, gsm8k, or humaneval. Default: mmlu.
  --eval-split SPLIT         Eval split suffix for MMLU. Default: val.
  --model-checkpoint PATH    Save trained GNN here, and load it for eval.
                             Default: artifacts/policies/gog_gnn_policy.pt.
  --load-model PATH          Initial GNN checkpoint. If training is skipped,
                             this is the model used for eval.

Run controls:
  --train-run-id ID          Default: mmlu-policy-train.
  --eval-run-id ID           Default: mmlu-policy-eval.
  --artifact-root PATH       Default: artifacts.
  --train-limit N            Optional training item limit.
  --eval-limit N             Optional eval item limit.
  --train-start-index N      Default: 0.
  --eval-start-index N       Default: 0.
  --workers N                Eval workers. Default: 1.
  --no-resume                Do not pass --resume to train/eval.

Backend:
  --base-url URL             Default: https://api.deepseek.com.
  --llm-model NAME           Default: deepseek-v4-flash.
  --max-tokens N             Default: 1024.
  --temperature FLOAT        Default: 0.
  --thinking enabled|disabled
                             Default: disabled.

Policy:
  --policy-learning-rate LR  Default: 0.01.
  --policy-gamma GAMMA       Default: 0.9.
  --policy-epsilon EPS       Default: 0.05 during training.

API key:
  The script reads one API key line from stdin or prompts interactively.
  It does not write the key to env files or project files.

Examples:
  bash scripts/run_policy_train_eval.sh \
    --train-data data/MMLU_subsets/train_test150/test \
    --eval-data data/MMLU_subsets/hard19/val \
    --model-checkpoint artifacts/policies/hard19_policy.pt

  bash scripts/run_policy_train_eval.sh \
    --train-data skip \
    --load-model artifacts/policies/hard19_policy.pt \
    --eval-data data/MMLU_subsets/hard19/val
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-data) TRAIN_DATA="$2"; shift 2 ;;
    --train-split) TRAIN_SPLIT="$2"; shift 2 ;;
    --train-run-id) TRAIN_RUN_ID="$2"; shift 2 ;;
    --train-limit) TRAIN_LIMIT="$2"; shift 2 ;;
    --train-start-index) TRAIN_START_INDEX="$2"; shift 2 ;;
    --eval-data) EVAL_DATA="$2"; shift 2 ;;
    --eval-dataset) EVAL_DATASET="$2"; shift 2 ;;
    --eval-split) EVAL_SPLIT="$2"; shift 2 ;;
    --eval-run-id) EVAL_RUN_ID="$2"; shift 2 ;;
    --eval-limit) EVAL_LIMIT="$2"; shift 2 ;;
    --eval-start-index) EVAL_START_INDEX="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --artifact-root) ARTIFACT_ROOT="$2"; shift 2 ;;
    --model-checkpoint) MODEL_CHECKPOINT="$2"; shift 2 ;;
    --load-model) LOAD_MODEL="$2"; shift 2 ;;
    --no-resume) RESUME="0"; shift ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --llm-model) MODEL="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --thinking) THINKING="$2"; shift 2 ;;
    --policy-learning-rate) POLICY_LEARNING_RATE="$2"; shift 2 ;;
    --policy-gamma) POLICY_GAMMA="$2"; shift 2 ;;
    --policy-epsilon) POLICY_EPSILON="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

cd "${PROJECT_ROOT}"

if [[ -z "${EVAL_DATA}" ]]; then
  echo "--eval-data is required" >&2
  exit 2
fi
if [[ ! -e "${EVAL_DATA}" ]]; then
  echo "eval data not found: ${EVAL_DATA}" >&2
  exit 2
fi

TRAIN_SKIPPED="0"
case "${TRAIN_DATA}" in
  ""|"none"|"NONE"|"skip"|"SKIP"|"null"|"NULL")
    TRAIN_SKIPPED="1"
    ;;
esac

if [[ "${TRAIN_SKIPPED}" == "0" && ! -d "${TRAIN_DATA}" ]]; then
  echo "train data directory not found: ${TRAIN_DATA}" >&2
  exit 2
fi

mkdir -p "$(dirname "${MODEL_CHECKPOINT}")"

if [[ "${TRAIN_SKIPPED}" == "1" ]]; then
  if [[ -n "${LOAD_MODEL}" ]]; then
    EVAL_POLICY_CHECKPOINT="${LOAD_MODEL}"
  else
    EVAL_POLICY_CHECKPOINT="${MODEL_CHECKPOINT}"
  fi
  if [[ ! -f "${EVAL_POLICY_CHECKPOINT}" ]]; then
    echo "training skipped, but model checkpoint is missing: ${EVAL_POLICY_CHECKPOINT}" >&2
    exit 2
  fi
else
  EVAL_POLICY_CHECKPOINT="${MODEL_CHECKPOINT}"
fi

if [[ -t 0 ]]; then
  read -r -s -p "DeepSeek API key: " API_KEY
  echo
else
  IFS= read -r API_KEY
fi
if [[ -z "${API_KEY}" ]]; then
  echo "empty API key" >&2
  exit 2
fi

run_with_key() {
  local args=("$@")
  printf '%s\n' "${API_KEY}" | "${PYTHON_BIN}" -m gogagent.cli "${args[@]}" \
    --api-key-stdin \
    --base-url "${BASE_URL}" \
    --model "${MODEL}" \
    --max-tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --thinking "${THINKING}"
}

resume_arg=()
if [[ "${RESUME}" == "1" ]]; then
  resume_arg=(--resume)
fi

if [[ "${TRAIN_SKIPPED}" == "0" ]]; then
  train_args=(
    train-mmlu
    --data-path "${TRAIN_DATA}"
    --split "${TRAIN_SPLIT}"
    --artifact-root "${ARTIFACT_ROOT}/training"
    --run-id "${TRAIN_RUN_ID}"
    --start-index "${TRAIN_START_INDEX}"
    --policy-checkpoint-out "${MODEL_CHECKPOINT}"
    --policy-learning-rate "${POLICY_LEARNING_RATE}"
    --policy-gamma "${POLICY_GAMMA}"
    --policy-epsilon "${POLICY_EPSILON}"
    "${resume_arg[@]}"
  )
  if [[ -n "${TRAIN_LIMIT}" ]]; then
    train_args+=(--limit "${TRAIN_LIMIT}")
  fi
  if [[ -n "${LOAD_MODEL}" ]]; then
    train_args+=(--policy-checkpoint-in "${LOAD_MODEL}")
  fi

  echo "[train] data=${TRAIN_DATA} split=${TRAIN_SPLIT}"
  echo "[train] saving policy=${MODEL_CHECKPOINT}"
  run_with_key "${train_args[@]}"
else
  echo "[train] skipped; loading policy=${EVAL_POLICY_CHECKPOINT}"
fi

eval_args=(
  eval
  --dataset "${EVAL_DATASET}"
  --data-path "${EVAL_DATA}"
  --split "${EVAL_SPLIT}"
  --artifact-root "${ARTIFACT_ROOT}/evals"
  --run-id "${EVAL_RUN_ID}"
  --workers "${WORKERS}"
  --start-index "${EVAL_START_INDEX}"
  --policy-checkpoint "${EVAL_POLICY_CHECKPOINT}"
  "${resume_arg[@]}"
)
if [[ -n "${EVAL_LIMIT}" ]]; then
  eval_args+=(--limit "${EVAL_LIMIT}")
fi

echo "[eval] dataset=${EVAL_DATASET} data=${EVAL_DATA} split=${EVAL_SPLIT}"
echo "[eval] loading policy=${EVAL_POLICY_CHECKPOINT}"
run_with_key "${eval_args[@]}"

SUMMARY_PATH="${ARTIFACT_ROOT}/evals/${EVAL_RUN_ID}/summary.json"
echo "[done] eval summary: ${SUMMARY_PATH}"
echo "[done] policy checkpoint: ${EVAL_POLICY_CHECKPOINT}"

if [[ -f "scripts/record_best_run.py" && -f "${SUMMARY_PATH}" ]]; then
  "${PYTHON_BIN}" scripts/record_best_run.py \
    --summary "${SUMMARY_PATH}" \
    --dataset "${EVAL_DATASET}" \
    --eval-data "${EVAL_DATA}" \
    --eval-split "${EVAL_SPLIT}" \
    --eval-run-id "${EVAL_RUN_ID}" \
    --policy-checkpoint "${EVAL_POLICY_CHECKPOINT}" \
    --train-run-id "${TRAIN_RUN_ID}" \
    --train-data "${TRAIN_DATA}" \
    --registry-dir "${ARTIFACT_ROOT}/best_runs" \
    --model "${MODEL}" \
    --base-url "${BASE_URL}" \
    --max-tokens "${MAX_TOKENS}" \
    --thinking "${THINKING}"
fi

unset API_KEY
