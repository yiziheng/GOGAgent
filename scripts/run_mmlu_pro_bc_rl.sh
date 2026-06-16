#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-python}"
TRAIN_DATA="${REPO_ROOT}/data/MMLU_Pro_subsets/train300"
SPLIT="train"
ENV_FILE="${REPO_ROOT}/.env"
RUN_ID=""
BC_EPOCHS=10
BC_LR=0.0001
RL_EPOCHS=1
RL_GROUP_SIZE=2
RL_LR=0.000001
RL_REWARD_MODE="answer_only"
RL_LIMIT=""
MAX_ACTIONS=6
MAX_DEPTH=2
MAX_NODES=8
DEVICE="${DEVICE:-cpu}"
TASK_ENCODER_DEVICE="${TASK_ENCODER_DEVICE:-cpu}"
OUTPUT_CHECKPOINT="${REPO_ROOT}/checkpoints/mmlu_pro/policy.pt"
TRAJECTORY_OUTPUT_DIR="${REPO_ROOT}/artifacts/bc_trajectories"
POLICY_OUTPUT_DIR="${REPO_ROOT}/artifacts/policies"
RL_OUTPUT_DIR="${REPO_ROOT}/artifacts/rl"
NO_ITEM_ARTIFACTS=1
NO_PROGRESS=0
OVERWRITE=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_mmlu_pro_bc_rl.sh [options]

Generate fixed MMLU-Pro BC trajectories, train a BC policy, refine it with
small-step RL, and copy the final checkpoint to checkpoints/mmlu_pro/policy.pt.

Common options:
  --train-data PATH          Default: data/MMLU_Pro_subsets/train300
  --split NAME               Default: train
  --run-id ID                Default: mmlu_pro-YYYYmmddTHHMMSS
  --env PATH                 Default: .env
  --output-checkpoint PATH   Default: checkpoints/mmlu_pro/policy.pt
  --bc-epochs N             Default: 10
  --bc-lr FLOAT             Default: 0.0001
  --rl-epochs N             Default: 1
  --rl-group-size K          Default: 2
  --rl-lr FLOAT              Default: 0.000001
  --rl-reward-mode MODE      dense|answer_only. Default: answer_only
  --rl-limit N               Optional RL example limit
  --max-actions N            Default: 6
  --max-depth N              Default: 2
  --max-nodes N              Default: 8
  --device DEVICE            Torch policy device. Default: cpu
  --task-encoder-device D    SentenceTransformer device. Default: cpu
  --keep-item-artifacts      Save per-rollout gog.json/gog.svg during RL
  --no-progress              Disable tqdm progress output
  --overwrite                Remove existing artifact run dirs for this run id
  --dry-run                  Print commands without running them
EOF
}

die() {
  echo "run_mmlu_pro_bc_rl: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-data)
      TRAIN_DATA="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --env)
      ENV_FILE="$2"
      shift 2
      ;;
    --output-checkpoint)
      OUTPUT_CHECKPOINT="$2"
      shift 2
      ;;
    --bc-epochs)
      BC_EPOCHS="$2"
      shift 2
      ;;
    --bc-lr)
      BC_LR="$2"
      shift 2
      ;;
    --rl-epochs)
      RL_EPOCHS="$2"
      shift 2
      ;;
    --rl-group-size)
      RL_GROUP_SIZE="$2"
      shift 2
      ;;
    --rl-lr)
      RL_LR="$2"
      shift 2
      ;;
    --rl-reward-mode)
      RL_REWARD_MODE="$2"
      shift 2
      ;;
    --rl-limit)
      RL_LIMIT="$2"
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
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --task-encoder-device)
      TASK_ENCODER_DEVICE="$2"
      shift 2
      ;;
    --keep-item-artifacts)
      NO_ITEM_ARTIFACTS=0
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
    --dry-run)
      DRY_RUN=1
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

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="mmlu_pro-$(date -u +%Y%m%dT%H%M%SZ)"
fi

BC_TRAJ_RUN_ID="${RUN_ID}-fixed-bc"
BC_POLICY_RUN_ID="${RUN_ID}-bc"
RL_RUN_ID="${RUN_ID}-rl"
BC_TRAJ_DIR="${TRAJECTORY_OUTPUT_DIR}/${BC_TRAJ_RUN_ID}"
BC_STEPS="${BC_TRAJ_DIR}/steps.jsonl"
BC_CHECKPOINT="${POLICY_OUTPUT_DIR}/${BC_POLICY_RUN_ID}/policy.pt"
RL_CHECKPOINT="${POLICY_OUTPUT_DIR}/${RL_RUN_ID}/policy.pt"

[[ -e "${TRAIN_DATA}" ]] || die "missing train data path: ${TRAIN_DATA}"
if [[ "${DRY_RUN}" -eq 0 ]]; then
  [[ -f "${ENV_FILE}" ]] || die "missing env file for RL: ${ENV_FILE}"
fi

if [[ "${OVERWRITE}" -eq 1 ]]; then
  for path in "${BC_TRAJ_DIR}" "${POLICY_OUTPUT_DIR}/${BC_POLICY_RUN_ID}" \
    "${RL_OUTPUT_DIR}/${RL_RUN_ID}" "${POLICY_OUTPUT_DIR}/${RL_RUN_ID}"; do
    case "${path}" in
      "${REPO_ROOT}"|"${REPO_ROOT}/"|"/"|"")
        die "refusing to remove unsafe path: ${path}"
        ;;
    esac
    rm -rf "${path}"
  done
fi

GEN_TRAJ_CMD=(
  "${PYTHON}" -m train.BC.generate_fixed_trajectories
  --dataset mmlu_pro
  --data-path "${TRAIN_DATA}"
  --split "${SPLIT}"
  --output-dir "${TRAJECTORY_OUTPUT_DIR}"
  --run-id "${BC_TRAJ_RUN_ID}"
  --actions ADD_ADVERSARIAL_JUDGE UP STOP
  --max-depth "${MAX_DEPTH}"
  --max-nodes "${MAX_NODES}"
)

TRAIN_BC_CMD=(
  "${PYTHON}" -m train.BC.train_bc
  --steps "${BC_STEPS}"
  --output-dir "${POLICY_OUTPUT_DIR}"
  --run-id "${BC_POLICY_RUN_ID}"
  --epochs "${BC_EPOCHS}"
  --lr "${BC_LR}"
  --device "${DEVICE}"
  --task-encoder-device "${TASK_ENCODER_DEVICE}"
)

TRAIN_RL_CMD=(
  "${PYTHON}" -m train.RL.train_grpo
  --checkpoint "${BC_CHECKPOINT}"
  --dataset mmlu_pro
  --data-path "${TRAIN_DATA}"
  --split "${SPLIT}"
  --env "${ENV_FILE}"
  --run-id "${RL_RUN_ID}"
  --rl-output-dir "${RL_OUTPUT_DIR}"
  --policy-output-dir "${POLICY_OUTPUT_DIR}"
  --epochs "${RL_EPOCHS}"
  --group-size "${RL_GROUP_SIZE}"
  --lr "${RL_LR}"
  --reward-mode "${RL_REWARD_MODE}"
  --max-actions "${MAX_ACTIONS}"
  --max-depth "${MAX_DEPTH}"
  --max-nodes "${MAX_NODES}"
  --device "${DEVICE}"
  --task-encoder-device "${TASK_ENCODER_DEVICE}"
)

if [[ -n "${RL_LIMIT}" ]]; then
  TRAIN_RL_CMD+=(--limit "${RL_LIMIT}")
fi
if [[ "${NO_ITEM_ARTIFACTS}" -eq 1 ]]; then
  TRAIN_RL_CMD+=(--no-item-artifacts)
fi
if [[ "${NO_PROGRESS}" -eq 1 ]]; then
  TRAIN_RL_CMD+=(--no-progress)
fi
if [[ "${OVERWRITE}" -eq 1 ]]; then
  TRAIN_RL_CMD+=(--overwrite)
fi

print_cmd() {
  printf '  %q' "$@"
  echo
}

echo "[mmlu_pro-bc-rl] run_id=${RUN_ID}"
echo "[mmlu_pro-bc-rl] train_data=${TRAIN_DATA} split=${SPLIT}"
echo "[mmlu_pro-bc-rl] output_checkpoint=${OUTPUT_CHECKPOINT}"
echo
echo "[1/3] Generate fixed BC trajectories"
print_cmd "${GEN_TRAJ_CMD[@]}"
echo
echo "[2/3] Train BC policy"
print_cmd "${TRAIN_BC_CMD[@]}"
echo
echo "[3/3] Refine with RL"
print_cmd "${TRAIN_RL_CMD[@]}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo
  echo "dry-run only: no commands were executed"
  exit 0
fi

PYTHONUNBUFFERED=1 "${GEN_TRAJ_CMD[@]}"
[[ -f "${BC_STEPS}" ]] || die "BC steps were not created: ${BC_STEPS}"

PYTHONUNBUFFERED=1 "${TRAIN_BC_CMD[@]}"
[[ -f "${BC_CHECKPOINT}" ]] || die "BC checkpoint was not created: ${BC_CHECKPOINT}"

PYTHONUNBUFFERED=1 "${TRAIN_RL_CMD[@]}"
[[ -f "${RL_CHECKPOINT}" ]] || die "RL checkpoint was not created: ${RL_CHECKPOINT}"

mkdir -p "$(dirname "${OUTPUT_CHECKPOINT}")"
cp "${RL_CHECKPOINT}" "${OUTPUT_CHECKPOINT}"

echo
echo "Done."
echo "BC trajectories: ${BC_TRAJ_DIR}"
echo "BC checkpoint:   ${BC_CHECKPOINT}"
echo "RL checkpoint:   ${RL_CHECKPOINT}"
echo "Saved policy:    ${OUTPUT_CHECKPOINT}"
