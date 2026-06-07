#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-python}"
CHECKPOINT="${REPO_ROOT}/artifacts/policies/bc-mmlu-test150-bigmlp512-keep/policy.pt"
DATA_DIR="${REPO_ROOT}/data/MMLU_subsets/gptswarm153/val"
SELECTION_JSONL="${REPO_ROOT}/data/MMLU_subsets/gptswarm153/selection.jsonl"
OUTPUT_DIR="${REPO_ROOT}/artifacts/evals/policy_mmlu"
RUN_ID="gptswarm153-bigmlp512-policy"
ENV_FILE="${REPO_ROOT}/.env"
DEVICE="cuda"
TASK_ENCODER_DEVICE="cuda"
MAX_ACTIONS=6
LIMIT=""
CONSTRUCT_ONLY=0
NO_ITEM_ARTIFACTS=0
OVERWRITE=0
NO_PROGRESS=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_best_policy_gptswarm153.sh [options]

Run the current best graph-construction policy on the GPTSwarm-style MMLU 153 set.

Common options:
  --checkpoint PATH          Policy checkpoint. Default: artifacts/policies/bc-mmlu-test150-bigmlp512-keep/policy.pt
  --run-id ID                Output run id. Default: gptswarm153-bigmlp512-policy
  --device DEVICE            Torch policy device. Default: cuda
  --task-encoder-device D    SentenceTransformer device. Default: cuda
  --limit N                  Evaluate first N examples only
  --construct-only           Only build graphs and action stats, no LLM calls
  --no-item-artifacts        Do not save per-item gog.json/gog.svg
  --no-progress              Disable tqdm progress output
  --overwrite                Replace an existing output run directory
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --task-encoder-device)
      TASK_ENCODER_DEVICE="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --construct-only)
      CONSTRUCT_ONLY=1
      shift
      ;;
    --no-item-artifacts)
      NO_ITEM_ARTIFACTS=1
      shift
      ;;
    --no-progress)
      NO_PROGRESS=1
      shift
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "checkpoint does not exist: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "data directory does not exist: ${DATA_DIR}" >&2
  exit 1
fi
if [[ "${CONSTRUCT_ONLY}" -eq 0 && ! -f "${ENV_FILE}" ]]; then
  echo "missing .env file: ${ENV_FILE}" >&2
  exit 1
fi

CMD=(
  "${PYTHON}" "${REPO_ROOT}/eval/eval_policy.py"
  --checkpoint "${CHECKPOINT}"
  --dataset mmlu
  --data-path "${DATA_DIR}"
  --split val
  --selection-jsonl "${SELECTION_JSONL}"
  --output-dir "${OUTPUT_DIR}"
  --run-id "${RUN_ID}"
  --env "${ENV_FILE}"
  --device "${DEVICE}"
  --task-encoder-device "${TASK_ENCODER_DEVICE}"
  --max-actions "${MAX_ACTIONS}"
)
if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi
if [[ "${CONSTRUCT_ONLY}" -eq 1 ]]; then
  CMD+=(--construct-only)
fi
if [[ "${NO_ITEM_ARTIFACTS}" -eq 1 ]]; then
  CMD+=(--no-item-artifacts)
fi
if [[ "${NO_PROGRESS}" -eq 1 ]]; then
  CMD+=(--no-progress)
fi
if [[ "${OVERWRITE}" -eq 1 ]]; then
  CMD+=(--overwrite)
fi

echo "[policy-eval] checkpoint=${CHECKPOINT}"
echo "[policy-eval] data=${DATA_DIR}"
echo "[policy-eval] run=${OUTPUT_DIR}/${RUN_ID}"
if [[ "${NO_PROGRESS}" -eq 1 ]]; then
  echo "[policy-eval] tqdm progress=disabled"
else
  echo "[policy-eval] tqdm progress=enabled"
fi
PYTHONUNBUFFERED=1 "${CMD[@]}"
