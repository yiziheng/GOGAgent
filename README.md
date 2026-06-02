# GOGAgent

`GOGAgent` is a lightweight, network-free MVP for supervisor-guided Organization
Graph-of-Graphs construction. It is intentionally separate from `GDesigner`.

## Core Design

```text
shared typed macro policy
  -> DomainAdapter compiler
  -> executable inner DAG snapshot
  -> label-blind observation + Supervisor summary
  -> outer Organization GoG memory
```

Each outer GoG node is a complete executable MAS DAG snapshot. Every non-terminal
macro edit creates a new snapshot and an `E_edit` transition. Lightweight
`E_sim` edges connect structurally similar snapshots. `QScorer` reads neighbor
statistics so GoG participates in decisions instead of acting as a log.

The train-only `gogagent.training` package pushes terminal oracle quality down
to each selected macro edit with distance decay, visible-feedback shaping, and
local token-cost penalties. It stores only shaped experience summaries in GoG;
gold values are not retained.

`OrganizationGoG.save()` and `OrganizationGoG.load()` persist those label-free
experience summaries. Pass a loaded checkpoint as `RolloutEngine(...,
gog_memory=memory)` to seed a new episode with frozen historical snapshots and
neighbor statistics while keeping the original training memory unchanged.

The MVP supports:

- `gsm8k`
- `mmlu`
- `humaneval`

The runtime never imports train-only reward oracles. MMLU labels therefore
cannot enter prompts, policy state, the Supervisor, action masks, or inference
GoG memory.

## Dataset Boundary

`gogagent.datasets` contains dependency-free loaders for canonical GSM8K JSONL,
MMLU CSV directories, and HumanEval JSONL. Each loader returns a
`DatasetExample(public_task, gold)` split. Pass only `public_task` into
`RolloutEngine.run`; import `gogagent.oracle.registry` only from future
training code.

## Environment

The current MVP runtime deliberately uses only the Python standard library.
The dependency files still install `gogagent` as a package so commands work
outside the repository root.

### Conda Server Setup

```bash
git clone git@github.com:yiziheng/GOGAgent.git
cd GOGAgent
conda env create -f environment.yml
conda activate GOGAgent
bash scripts/verify_compile.sh
```

### Pip Server Setup

```bash
git clone git@github.com:yiziheng/GOGAgent.git
cd GOGAgent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
bash scripts/verify_compile.sh
```

Install the development dependencies when running `pytest`:

```bash
python -m pip install -r requirements-dev.txt
pytest
```

## Compile And Smoke Check

No API key or dataset download is required:

```bash
conda run -n GOGAgent bash scripts/verify_compile.sh
```

Run one mock episode:

```bash
conda run -n GOGAgent python -m gogagent.cli --domain mmlu
```

## Visible Graph Artifacts

Every rollout writes:

```text
artifacts/runs/<timestamp>/<domain>/<episode>/
  result.json
  trace.jsonl
  gog.json
  gog.svg
  snapshots/
    <graph-id>.json
    <graph-id>.svg
```

The JSON files are replayable debugging records. The SVG files can be opened
directly to inspect generated agent DAGs and the outer Graph-of-Graphs.
