#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-python}"
STEPS=""
RUN_ID=""
OUTPUT_DIR="${REPO_ROOT}/artifacts/policies"
EPOCHS=10
LR=0.0001
DEVICE="cpu"
TASK_ENCODER_DEVICE=""
LIMIT=""
CLASS_WEIGHT="balanced"
CLASS_WEIGHT_ALPHA="0.5"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_bc_policy.sh --steps PATH [options]

Train the GOG graph-construction policy from BC steps.jsonl.

Common options:
  --steps PATH              Required BC steps.jsonl path
  --run-id ID               Checkpoint run id
  --output-dir PATH         Output root. Default: artifacts/policies
  --epochs N                Default: 10
  --lr FLOAT                Default: 0.0001
  --device DEVICE           Torch training device. Default: cpu
  --task-encoder-device D   Optional SentenceTransformer device
  --limit N                 Train on the first N step rows
  --class-weight MODE       none|balanced. Default: balanced
  --class-weight-alpha A    Balanced weight smoothing. Default: 0.5
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --lr)
      LR="$2"
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
    --class-weight)
      CLASS_WEIGHT="$2"
      shift 2
      ;;
    --class-weight-alpha)
      CLASS_WEIGHT_ALPHA="$2"
      shift 2
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

if [[ -z "${STEPS}" ]]; then
  echo "missing required --steps PATH" >&2
  usage >&2
  exit 2
fi
if [[ ! -f "${STEPS}" ]]; then
  echo "steps file does not exist: ${STEPS}" >&2
  exit 1
fi

CMD=(
  "${PYTHON}" -m train.BC.train_bc
  --steps "${STEPS}"
  --output-dir "${OUTPUT_DIR}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --device "${DEVICE}"
  --class-weight "${CLASS_WEIGHT}"
  --class-weight-alpha "${CLASS_WEIGHT_ALPHA}"
)
if [[ -n "${RUN_ID}" ]]; then
  CMD+=(--run-id "${RUN_ID}")
fi
if [[ -n "${TASK_ENCODER_DEVICE}" ]]; then
  CMD+=(--task-encoder-device "${TASK_ENCODER_DEVICE}")
fi
if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

echo "[train-bc] steps=${STEPS}"
echo "[train-bc] output=${OUTPUT_DIR}${RUN_ID:+/${RUN_ID}}"
"${CMD[@]}"
