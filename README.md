# GOGAgent

`GOGAgent` constructs task-specific multi-agent DAGs inside an Organization
Graph-of-Graphs (GoG). The production workflow supports `mmlu`, `gsm8k`, and
`humaneval` with a real OpenAI-compatible LLM backend.

## Architecture

```text
public benchmark task
  -> shared typed macro policy
  -> domain adapter compiler
  -> executable inner agent DAG snapshot
  -> label-blind feedback + fixed Supervisor
  -> outer Organization GoG memory
```

Every non-terminal macro edit creates a complete immutable snapshot and an
`E_edit` edge. Lightweight `E_sim` edges connect structurally similar
snapshots. The `QScorer` consumes neighbor statistics from persisted GoG
memory, so prior training experience can change later graph construction.

Gold labels remain outside inference. Dataset loaders return
`DatasetExample(public_task, gold)`, but `RolloutEngine` receives only
`public_task`. Evaluation or training code scores the result after rollout.

## Server Setup

```bash
git clone git@github.com:yiziheng/GOGAgent.git
cd GOGAgent
conda env create -f environment.yml
conda activate GOGAgent
bash scripts/verify_compile.sh
```

`scripts/verify_compile.sh` performs production compile, import, CLI-help, and
mock-reference checks. It does not consume API quota.

## Backend Configuration

The CLI defaults to:

```text
base URL: https://api.deepseek.com
model:    deepseek-v4-flash
thinking: disabled in the server MMLU runner
```

Supply credentials only through interactive stdin. Do not save API keys in the
repository, command-line arguments, environment variables, shell history,
`.env`, artifacts, or GoG memory.

```bash
bash scripts/run_mmlu_full.sh
```

Optional runtime variables:

```text
GOGAGENT_TIMEOUT_SECONDS
GOGAGENT_MAX_RETRIES
GOGAGENT_MAX_TOKENS
GOGAGENT_TEMPERATURE
```

Backend manifests store only credential-free settings.

## MMLU Training And Evaluation

Place the copied MMLU data and budget subsets inside the repository:

```text
data/MMLU/
  val/
  test/
data/MMLU_subsets/
  train_test150/test/
  eval_gptswarm_val153/val/
```

Run the budgeted resumable workflow. Training uses 150 examples sampled from
`test`, covering all 57 subjects. Evaluation uses the GPTSwarm-compatible
subset: concatenate sorted `val` CSV files, shuffle with
`numpy.default_rng(888)`, then take the first 153 examples.

```bash
bash scripts/run_mmlu_full.sh
```

Equivalent explicit commands:

```bash
python -m gogagent.cli train-mmlu \
  --data-path data/MMLU_subsets/train_test150/test \
  --split test \
  --run-id deepseek-v4-flash-mmlu-train-test150 \
  --api-key-stdin \
  --resume

python -m gogagent.cli eval \
  --dataset mmlu \
  --data-path data/MMLU_subsets/eval_gptswarm_val153/val \
  --split val \
  --run-id deepseek-v4-flash-mmlu-eval-gptswarm-val153 \
  --workers 8 \
  --gog-memory artifacts/training/deepseek-v4-flash-mmlu-train-test150/memory.json \
  --api-key-stdin \
  --resume
```

Results are written incrementally:

```text
artifacts/training/<run-id>/
  memory.json
  train_summary.json

artifacts/evals/<run-id>/
  manifest.json
  events.jsonl
  summary.json
  summary.tsv
  items/<task-id>-<digest>/
    input.json
    status.json
    result.json
    rollout/
      result.json
      trace.jsonl
      gog.json
      gog.svg
      snapshots/*.json
      snapshots/*.svg
```

Restart the same command with `--resume` to skip completed items.

## Other Benchmarks

GSM8K and HumanEval use the same `eval` entrypoint:

```bash
python -m gogagent.cli eval --dataset gsm8k --data-path data/gsm8k.jsonl --run-id gsm8k-run
python -m gogagent.cli eval --dataset humaneval --data-path data/humaneval.jsonl --run-id humaneval-run
```

HumanEval is fail-closed: generated code is never executed by the main
evaluation process. Configure `GOGAGENT_HUMANEVAL_SANDBOX_COMMAND` with a
separately deployed, container-isolated worker before scoring HumanEval.
