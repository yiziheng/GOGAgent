# GOGAgent

Reference code for GOGAgent, a graph-of-graphs multi-agent construction framework with behavior cloning and RL refinement.

## Setup

```bash
conda create -n GOGAgent python=3.10 -y
conda activate GOGAgent
pip install -r requirements.txt
cp .env.example .env
```

Fill `GOGAGENT_API_KEY` in `.env` before running LLM evaluation.
Policy evaluation uses a local SentenceTransformer task encoder. If the model is
not cached on the machine, make sure Hugging Face access or a mirror such as
`HF_ENDPOINT=https://hf-mirror.com` is configured before running `quick_start.sh`.

## Data

This repository does not ship full benchmark datasets. Prepare MMLU locally under:

```text
data/MMLU_subsets/gptswarm153/val
```

The public GPTSwarm-style selection metadata is included at:

```text
data_splits/mmlu/gptswarm153_selection.jsonl
```

## Quick Start

```bash
bash quick_start.sh mmlu
```

The script first loads:

```text
checkpoints/mmlu/policy.pt
```

If the checkpoint is missing, it runs fixed-template BC training, a small RL refinement, saves the resulting policy to `checkpoints/mmlu/policy.pt`, and then evaluates it.

For a no-LLM graph-construction check:

```bash
QUICK_START_CONSTRUCT_ONLY=1 EVAL_LIMIT=2 bash quick_start.sh mmlu
```

## Main Modules

```text
gogagent/   core agents, actions, graph runtime, policy, prompts, rewards
train/      behavior cloning and RL training
eval/       fixed-graph and policy evaluation
scripts/    dataset and training helpers
```
