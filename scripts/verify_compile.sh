#!/usr/bin/env bash
set -euo pipefail

python -m compileall -q gogagent scripts tests
python -c "from gogagent.adapters.registry import get_adapter; [get_adapter(name) for name in ('gsm8k', 'mmlu', 'humaneval')]"
python -m scripts.smoke_loaders
python -m scripts.smoke_actions
python -m scripts.smoke_training
python -m scripts.smoke_all
