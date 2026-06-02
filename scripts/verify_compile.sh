#!/usr/bin/env bash
set -euo pipefail

python -m compileall -q gogagent scripts
python -c "from gogagent.adapters.registry import get_adapter; [get_adapter(name) for name in ('gsm8k', 'mmlu', 'humaneval')]"
python -c "from gogagent.llm import OpenAICompatibleLLM; print(OpenAICompatibleLLM(base_url='https://api.deepseek.com', model='deepseek-v4-flash').describe())"
python -m gogagent.cli --help
python -m gogagent.cli eval --help
python -m gogagent.cli train-mmlu --help
if grep -R -I -n -E "MockLLM|mock_llm|gogagent\\.llm\\.mock" \
  --include="*.py" \
  gogagent scripts 2>/dev/null; then
  echo "mock references remain in production code" >&2
  exit 1
fi
