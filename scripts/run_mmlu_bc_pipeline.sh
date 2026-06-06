#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-python}"
SOURCE_DIR="${REPO_ROOT}/data/MMLU/data/test"
SAMPLE_DIR="${REPO_ROOT}/data/MMLU_subsets/bc_train_test150/test"
TRAJECTORY_OUTPUT_DIR="${REPO_ROOT}/artifacts/bc_trajectories"
ENV_FILE="${REPO_ROOT}/.env"
TOTAL=150
SEED=42
SPLIT="test"
RUN_ID=""
MAX_ACTIONS=6
MAX_DEPTH=2
MAX_NODES=8
STYLES=()
OVERWRITE_SAMPLE=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_mmlu_bc_pipeline.sh [options]

Sample a balanced MMLU BC training subset, then generate DeepSeek teacher
action trajectories for behavior cloning.

Common options:
  --source-dir PATH        Original MMLU split directory. Default: data/MMLU/data/test
  --sample-dir PATH        Sampled subset output directory. Default: data/MMLU_subsets/bc_train_test150/test
  --total N                Number of sampled questions. Default: 150
  --seed N                 Sampling seed. Default: 42
  --run-id ID              BC trajectory artifact run id. Default: mmlu-bc-test<TOTAL>-seed<SEED>
  --env PATH               .env file for GOGAGENT_API_KEY etc. Default: .env
  --overwrite-sample       Replace --sample-dir if it already exists
  --dry-run                Show sampling plan and generation command without calling the LLM

Trajectory options:
  --styles STYLE...        Teacher styles. Default: generator default styles
  --max-actions N          Max actions per teacher trajectory. Default: 6
  --max-depth N            Max graph nesting depth. Default: 2
  --max-nodes N            Max total graph nodes. Default: 8

Examples:
  bash scripts/run_mmlu_bc_pipeline.sh --overwrite-sample

  bash scripts/run_mmlu_bc_pipeline.sh \
    --total 150 \
    --seed 42 \
    --run-id bc-test150-v1 \
    --styles accuracy_first cost_aware hard_case_adversarial \
    --overwrite-sample
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-dir)
      SOURCE_DIR="$2"
      shift 2
      ;;
    --sample-dir)
      SAMPLE_DIR="$2"
      shift 2
      ;;
    --total)
      TOTAL="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
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
    --output-dir)
      TRAJECTORY_OUTPUT_DIR="$2"
      shift 2
      ;;
    --styles)
      shift
      STYLES=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        STYLES+=("$1")
        shift
      done
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
    --overwrite-sample)
      OVERWRITE_SAMPLE=1
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
  RUN_ID="mmlu-bc-test${TOTAL}-seed${SEED}"
fi

if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "missing source MMLU directory: ${SOURCE_DIR}" >&2
  exit 1
fi

if [[ "${DRY_RUN}" -eq 0 && ! -f "${ENV_FILE}" ]]; then
  echo "missing .env file: ${ENV_FILE}" >&2
  exit 1
fi

SAMPLE_CMD=(
  "${PYTHON}" "${REPO_ROOT}/scripts/sample_mmlu_bc_train.py"
  --source-dir "${SOURCE_DIR}"
  --output-dir "${SAMPLE_DIR}"
  --split "${SPLIT}"
  --total "${TOTAL}"
  --seed "${SEED}"
)
if [[ "${OVERWRITE_SAMPLE}" -eq 1 ]]; then
  SAMPLE_CMD+=(--overwrite)
fi

GENERATE_CMD=(
  "${PYTHON}" -m train.BC.generate_trajectories
  --dataset mmlu
  --data-path "${SAMPLE_DIR}"
  --split "${SPLIT}"
  --env "${ENV_FILE}"
  --output-dir "${TRAJECTORY_OUTPUT_DIR}"
  --run-id "${RUN_ID}"
  --max-actions "${MAX_ACTIONS}"
  --max-depth "${MAX_DEPTH}"
  --max-nodes "${MAX_NODES}"
)
if [[ "${#STYLES[@]}" -gt 0 ]]; then
  GENERATE_CMD+=(--styles "${STYLES[@]}")
fi

echo "[1/2] Sampling MMLU BC subset"
echo "  source: ${SOURCE_DIR}"
echo "  sample: ${SAMPLE_DIR}"
echo "  total:  ${TOTAL}"
echo "  seed:   ${SEED}"
if [[ "${DRY_RUN}" -eq 1 ]]; then
  "${SAMPLE_CMD[@]}" --dry-run
  echo
  echo "[2/2] BC trajectory generation command"
  printf '  %q' "${GENERATE_CMD[@]}"
  echo
  echo
  echo "dry-run only: no files were written by the sampler and no LLM calls were made"
  exit 0
fi

"${SAMPLE_CMD[@]}"

echo
echo "[2/2] Generating BC teacher trajectories"
echo "  run id: ${RUN_ID}"
echo "  output: ${TRAJECTORY_OUTPUT_DIR}/${RUN_ID}"
"${GENERATE_CMD[@]}"

echo
echo "Done."
echo "Sampled data: ${SAMPLE_DIR}"
echo "Trajectories: ${TRAJECTORY_OUTPUT_DIR}/${RUN_ID}/trajectories.jsonl"
echo "BC steps:     ${TRAJECTORY_OUTPUT_DIR}/${RUN_ID}/steps.jsonl"
echo "Summary:      ${TRAJECTORY_OUTPUT_DIR}/${RUN_ID}/summary.json"
