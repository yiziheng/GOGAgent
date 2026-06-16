#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-python}"
CHECKPOINT=""
DATASET="mmlu"
DATA_PATH=""
SPLIT=""
SELECTION_JSONL=""
ENV_FILE="${REPO_ROOT}/.env"
RUN_ID=""
EPOCHS=1
GROUP_SIZE=4
MAX_ACTIONS=6
MAX_DEPTH=2
MAX_NODES=8
TEMPERATURE=1.0
REWARD_MODE="dense"
KL_BETA=0.01
LR=0.00001
DEVICE="cpu"
TASK_ENCODER_DEVICE=""
LIMIT=""
NO_ITEM_ARTIFACTS=0
NO_PROGRESS=0
OVERWRITE=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_rl_policy.sh --checkpoint PATH [options]

Refine a GOG graph-construction policy with GRPO-style RL.

Common options:
  --checkpoint PATH          Required base BC/RL policy checkpoint
  --dataset NAME             mmlu|mmlu_pro|gsm8k|humaneval. Default: mmlu
  --data-path PATH           Dataset path. Defaults exist for mmlu and mmlu_pro
  --split NAME               Dataset split. Default: test for mmlu, train for mmlu_pro
  --run-id ID                Output run id
  --epochs N                 Default: 1
  --group-size K             Rollouts per problem. Default: 4
  --max-actions N            Default: 6
  --max-depth N              Default: 2
  --max-nodes N              Default: 8
  --lr FLOAT                 Default: 0.00001
  --kl-beta FLOAT            Default: 0.01
  --temperature FLOAT        Sampling temperature. Default: 1.0
  --reward-mode MODE         dense|answer_only. Default: dense
  --device DEVICE            Torch policy device. Default: cpu
  --task-encoder-device D    Optional SentenceTransformer device
  --limit N                  Train on first N examples only
  --no-item-artifacts        Do not save per-rollout gog.json/gog.svg
  --no-progress              Disable tqdm progress output
  --overwrite                Replace existing output run directories
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --data-path)
      DATA_PATH="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --selection-jsonl)
      SELECTION_JSONL="$2"
      shift 2
      ;;
    --env)
      ENV_FILE="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --group-size)
      GROUP_SIZE="$2"
      shift 2
      ;;
    --max-actions)
      MAX_ACTIONS="$2"
      shift 2
      ;;
    --max-depth)
      MAX_DEPTH="$2"
      shift 2
      ;;
    --max-nodes)
      MAX_NODES="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --reward-mode)
      REWARD_MODE="$2"
      shift 2
      ;;
    --kl-beta)
      KL_BETA="$2"
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

if [[ -z "${CHECKPOINT}" ]]; then
  echo "missing required --checkpoint PATH" >&2
  usage >&2
  exit 2
fi
case "${DATASET}" in
  mmlupro)
    DATASET="mmlu_pro"
    ;;
esac
case "${DATASET}" in
  mmlu)
    DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/MMLU_subsets/train_test150/test}"
    SPLIT="${SPLIT:-test}"
    ;;
  mmlu_pro)
    DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/MMLU_Pro_subsets/train300}"
    SPLIT="${SPLIT:-train}"
    ;;
  gsm8k|humaneval)
    if [[ -z "${DATA_PATH}" ]]; then
      echo "missing --data-path for dataset=${DATASET}" >&2
      exit 2
    fi
    SPLIT="${SPLIT:-test}"
    ;;
  *)
    echo "unsupported dataset: ${DATASET}" >&2
    exit 2
    ;;
esac
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "checkpoint does not exist: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -e "${DATA_PATH}" ]]; then
  echo "data path does not exist: ${DATA_PATH}" >&2
  exit 1
fi
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "missing .env file: ${ENV_FILE}" >&2
  exit 1
fi

CMD=(
  "${PYTHON}" -m train.RL.train_grpo
  --checkpoint "${CHECKPOINT}"
  --dataset "${DATASET}"
  --data-path "${DATA_PATH}"
  --split "${SPLIT}"
  --env "${ENV_FILE}"
  --epochs "${EPOCHS}"
  --group-size "${GROUP_SIZE}"
  --max-actions "${MAX_ACTIONS}"
  --max-depth "${MAX_DEPTH}"
  --max-nodes "${MAX_NODES}"
  --temperature "${TEMPERATURE}"
  --reward-mode "${REWARD_MODE}"
  --kl-beta "${KL_BETA}"
  --lr "${LR}"
  --device "${DEVICE}"
)
if [[ -n "${SELECTION_JSONL}" ]]; then
  CMD+=(--selection-jsonl "${SELECTION_JSONL}")
fi
if [[ -n "${RUN_ID}" ]]; then
  CMD+=(--run-id "${RUN_ID}")
fi
if [[ -n "${TASK_ENCODER_DEVICE}" ]]; then
  CMD+=(--task-encoder-device "${TASK_ENCODER_DEVICE}")
fi
if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
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

echo "[train-rl] checkpoint=${CHECKPOINT}"
echo "[train-rl] data=${DATA_PATH}"
echo "[train-rl] dataset=${DATASET} split=${SPLIT}"
echo "[train-rl] group_size=${GROUP_SIZE} epochs=${EPOCHS} reward_mode=${REWARD_MODE}"
echo "[train-rl] max_actions=${MAX_ACTIONS} max_depth=${MAX_DEPTH} max_nodes=${MAX_NODES}"
PYTHONUNBUFFERED=1 "${CMD[@]}"
