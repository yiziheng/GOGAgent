#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"

"${PYTHON}" -m compileall -q gogagent scripts tests train
"${PYTHON}" -c "from gogagent import Graph, GraphMessage, OpenAICompatibleClient; print(GraphMessage(role='test', content='ok', answer='A').to_dict())"
"${PYTHON}" -c "from gogagent.llm import OpenAICompatibleClient; print(OpenAICompatibleClient(base_url='https://api.deepseek.com', model='deepseek-v4-flash', api_key='test-key').describe())"
ENV_SMOKE="$(mktemp)"
trap 'rm -f "${ENV_SMOKE}"' EXIT
cat > "${ENV_SMOKE}" <<'EOF'
GOGAGENT_API_KEY=test-key
GOGAGENT_BASE_URL=https://api.deepseek.com
GOGAGENT_MODEL=deepseek-v4-flash
GOGAGENT_THINKING=disabled
EOF
"${PYTHON}" -c "from gogagent.config import llm_client_from_env; print(llm_client_from_env('${ENV_SMOKE}').describe())"
"${PYTHON}" -m gogagent.cli --help
"${PYTHON}" -m gogagent.cli agents >/dev/null
"${PYTHON}" -m gogagent.cli actions >/dev/null
bash -n scripts/run_mmlu_bc_pipeline.sh
bash -n scripts/run_mmlu_pro_bc_rl.sh
bash -n scripts/train_bc_policy.sh
bash -n scripts/train_rl_policy.sh
bash scripts/run_mmlu_bc_pipeline.sh --help >/dev/null
bash scripts/run_mmlu_pro_bc_rl.sh --help >/dev/null
"${PYTHON}" tests/smoke_round1_refactor.py
"${PYTHON}" tests/test_oracle_mmlu.py
"${PYTHON}" tests/test_policy_graph_encoding.py
"${PYTHON}" tests/test_bc_trajectory_generation.py
if grep -R -I -n -E "gogagent\\.(adapters|core|training|evaluation|gog|oracle|llm\\.base|llm\\.openai_compatible)|OpenAICompatibleLLM|LLMBackend|LLMResponse|MacroAction|RolloutEngine|memory_gog|memory\\.json|smoke_contracts|run_policy" \
  --include="*.py" \
  gogagent scripts tests train 2>/dev/null; then
  echo "legacy runtime references remain" >&2
  exit 1
fi
