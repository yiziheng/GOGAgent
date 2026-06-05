#!/usr/bin/env bash
set -euo pipefail

python -m compileall -q gogagent scripts tests
python -c "from gogagent import Graph, GraphMessage, OpenAICompatibleClient; print(GraphMessage(role='test', content='ok', answer='A').to_dict())"
python -c "from gogagent.llm import OpenAICompatibleClient; print(OpenAICompatibleClient(base_url='https://api.deepseek.com', model='deepseek-v4-flash', api_key='test-key').describe())"
ENV_SMOKE="$(mktemp)"
trap 'rm -f "${ENV_SMOKE}"' EXIT
cat > "${ENV_SMOKE}" <<'EOF'
GOGAGENT_API_KEY=test-key
GOGAGENT_BASE_URL=https://api.deepseek.com
GOGAGENT_MODEL=deepseek-v4-flash
GOGAGENT_THINKING=disabled
EOF
python -c "from gogagent.config import llm_client_from_env; print(llm_client_from_env('${ENV_SMOKE}').describe())"
python -m gogagent.cli --help
python -m gogagent.cli agents >/dev/null
python -m gogagent.cli actions >/dev/null
python tests/smoke_round1_refactor.py
if grep -R -I -n -E "gogagent\\.(adapters|core|training|evaluation|policy|gog|oracle|llm\\.base|llm\\.openai_compatible)|OpenAICompatibleLLM|LLMBackend|LLMResponse|MacroAction|RolloutEngine|memory_gog|memory\\.json|smoke_contracts|run_policy" \
  --include="*.py" \
  gogagent scripts tests 2>/dev/null; then
  echo "legacy runtime references remain" >&2
  exit 1
fi
