#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cpu}"
TASK_ENCODER_DEVICE="${TASK_ENCODER_DEVICE:-cpu}"
MAX_ACTIONS="${MAX_ACTIONS:-6}"
BC_EPOCHS="${BC_EPOCHS:-10}"
BC_LR="${BC_LR:-0.001}"
RL_EPOCHS="${RL_EPOCHS:-1}"
RL_GROUP_SIZE="${RL_GROUP_SIZE:-2}"
RL_LIMIT="${RL_LIMIT:-3}"
RL_LR="${RL_LR:-0.000001}"
RL_REWARD_MODE="${RL_REWARD_MODE:-answer_only}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
QUICK_START_CONSTRUCT_ONLY="${QUICK_START_CONSTRUCT_ONLY:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash quick_start.sh DATASET

Example:
  bash quick_start.sh mmlu

The script first looks for:
  checkpoints/DATASET/policy.pt

If the policy exists, it runs eval_policy directly.
If the policy is missing, it trains BC, runs a small RL refinement, saves the
final policy to checkpoints/DATASET/policy.pt, then runs eval_policy.

Optional environment knobs:
  PYTHON, DEVICE, TASK_ENCODER_DEVICE, EVAL_LIMIT
  BC_EPOCHS, BC_LR
  RL_EPOCHS, RL_GROUP_SIZE, RL_LIMIT, RL_LR, RL_REWARD_MODE
  QUICK_START_CONSTRUCT_ONLY=1  # build graphs only, no LLM calls
EOF
}

die() {
  echo "quick_start: $*" >&2
  exit 1
}

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || die "missing required file: ${path}"
}

require_dir() {
  local path="$1"
  [[ -d "${path}" ]] || die "missing required directory: ${path}"
}

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

DATASET="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
POLICY_PATH="${REPO_ROOT}/checkpoints/${DATASET}/policy.pt"
ENV_FILE="${REPO_ROOT}/.env"

case "${DATASET}" in
  mmlu)
    EVAL_DATA_PATH="${REPO_ROOT}/data/MMLU_subsets/gptswarm153/val"
    EVAL_SPLIT="val"
    EVAL_SELECTION_JSONL="${REPO_ROOT}/data_splits/mmlu/gptswarm153_selection.jsonl"
    EVAL_OUTPUT_DIR="${REPO_ROOT}/artifacts/evals/policy_mmlu"
    EVAL_RUN_ID="quickstart-mmlu-policy"

    if [[ -d "${REPO_ROOT}/data/MMLU_subsets/bc_train_test150/test" ]]; then
      TRAIN_DATA_PATH="${REPO_ROOT}/data/MMLU_subsets/bc_train_test150/test"
      TRAIN_SPLIT="test"
    elif [[ -d "${REPO_ROOT}/data/MMLU/data/dev" ]]; then
      TRAIN_DATA_PATH="${REPO_ROOT}/data/MMLU/data/dev"
      TRAIN_SPLIT="dev"
    else
      die "no MMLU training data found. Expected data/MMLU_subsets/bc_train_test150/test or data/MMLU/data/dev"
    fi
    ;;
  *)
    die "unsupported dataset '${DATASET}'. quick_start currently has a concrete default pipeline for: mmlu"
    ;;
esac

require_dir "${EVAL_DATA_PATH}"
if [[ -n "${EVAL_SELECTION_JSONL}" ]]; then
  require_file "${EVAL_SELECTION_JSONL}"
fi
if [[ "${QUICK_START_CONSTRUCT_ONLY}" != "1" ]]; then
  require_file "${ENV_FILE}"
fi

cd "${REPO_ROOT}"

if [[ -f "${POLICY_PATH}" ]]; then
  echo "[quick-start] found policy: ${POLICY_PATH}"
else
  echo "[quick-start] no policy found at ${POLICY_PATH}"
  echo "[quick-start] training a ${DATASET} policy first"

  require_file "${ENV_FILE}"
  require_dir "${TRAIN_DATA_PATH}"

  BC_TRAJ_RUN_ID="${DATASET}-quickstart-fixed-bc"
  BC_POLICY_RUN_ID="${DATASET}-quickstart-bc"
  RL_RUN_ID="${DATASET}-quickstart-rl"
  BC_TRAJ_DIR="${REPO_ROOT}/artifacts/bc_trajectories/${BC_TRAJ_RUN_ID}"
  BC_STEPS="${BC_TRAJ_DIR}/steps.jsonl"
  BC_CHECKPOINT="${REPO_ROOT}/artifacts/policies/${BC_POLICY_RUN_ID}/policy.pt"
  RL_CHECKPOINT="${REPO_ROOT}/artifacts/policies/${RL_RUN_ID}/policy.pt"

  echo "[quick-start] generating fixed BC trajectories"
  "${PYTHON}" -m train.BC.generate_fixed_trajectories \
    --dataset "${DATASET}" \
    --data-path "${TRAIN_DATA_PATH}" \
    --split "${TRAIN_SPLIT}" \
    --output-dir "${REPO_ROOT}/artifacts/bc_trajectories" \
    --run-id "${BC_TRAJ_RUN_ID}"

  require_file "${BC_STEPS}"

  echo "[quick-start] training BC policy"
  "${PYTHON}" -m train.BC.train_bc \
    --steps "${BC_STEPS}" \
    --run-id "${BC_POLICY_RUN_ID}" \
    --output-dir "${REPO_ROOT}/artifacts/policies" \
    --epochs "${BC_EPOCHS}" \
    --lr "${BC_LR}" \
    --device "${DEVICE}" \
    --task-encoder-device "${TASK_ENCODER_DEVICE}" \
    --class-weight none

  require_file "${BC_CHECKPOINT}"

  echo "[quick-start] running small RL refinement"
  "${PYTHON}" -m train.RL.train_grpo \
    --checkpoint "${BC_CHECKPOINT}" \
    --dataset "${DATASET}" \
    --data-path "${TRAIN_DATA_PATH}" \
    --split "${TRAIN_SPLIT}" \
    --env "${ENV_FILE}" \
    --run-id "${RL_RUN_ID}" \
    --epochs "${RL_EPOCHS}" \
    --group-size "${RL_GROUP_SIZE}" \
    --limit "${RL_LIMIT}" \
    --lr "${RL_LR}" \
    --reward-mode "${RL_REWARD_MODE}" \
    --device "${DEVICE}" \
    --task-encoder-device "${TASK_ENCODER_DEVICE}" \
    --no-item-artifacts \
    --no-progress \
    --overwrite

  require_file "${RL_CHECKPOINT}"
  mkdir -p "$(dirname "${POLICY_PATH}")"
  cp "${RL_CHECKPOINT}" "${POLICY_PATH}"
  echo "[quick-start] saved policy: ${POLICY_PATH}"
fi

echo "[quick-start] running policy eval"
EVAL_CMD=(
  "${PYTHON}" -m eval.eval_policy
  --checkpoint "${POLICY_PATH}"
  --dataset "${DATASET}"
  --data-path "${EVAL_DATA_PATH}"
  --split "${EVAL_SPLIT}"
  --output-dir "${EVAL_OUTPUT_DIR}"
  --run-id "${EVAL_RUN_ID}"
  --device "${DEVICE}"
  --task-encoder-device "${TASK_ENCODER_DEVICE}"
  --max-actions "${MAX_ACTIONS}"
  --overwrite
)
if [[ -n "${EVAL_SELECTION_JSONL}" ]]; then
  EVAL_CMD+=(--selection-jsonl "${EVAL_SELECTION_JSONL}")
fi
if [[ "${QUICK_START_CONSTRUCT_ONLY}" == "1" ]]; then
  EVAL_CMD+=(--construct-only)
else
  EVAL_CMD+=(--env "${ENV_FILE}")
fi
if [[ -n "${EVAL_LIMIT}" ]]; then
  EVAL_CMD+=(--limit "${EVAL_LIMIT}")
fi

"${EVAL_CMD[@]}"

echo
echo "[quick-start] done"
echo "Policy:  ${POLICY_PATH}"
echo "Summary: ${EVAL_OUTPUT_DIR}/${EVAL_RUN_ID}/summary.json"
