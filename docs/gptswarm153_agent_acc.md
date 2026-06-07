# GPTSwarm153 Agent Combination Accuracy Log

This document records MMLU GPTSwarm-style 153-sample evaluation results for different Agent combinations, fixed graphs, policies, and single-LLM baselines.

## Dataset

```text
dataset: MMLU GPTSwarm-style validation subset
size: 153
local path: data/MMLU_subsets/gptswarm153/val
selection: data/MMLU_subsets/gptswarm153/selection.jsonl
metric: accuracy = correct / completed
```

## Leaderboard

| Date | Variant | Graph / Actions | Policy / Checkpoint | LLM | Correct | Acc | Result Path | Notes |
|---|---|---|---|---|---:|---:|---|---|
| 2026-06 | Best observed policy | Dynamic policy graph | `artifacts/policies/bc-mmlu-test150-bigmlp512-keep/policy.pt` | `deepseek-v4-flash` | 138/153 | 90.20% | TBD | Historical best from discussion: 15 wrong. Fill exact run path after rerun or artifact lookup. |
| TBD | DeepSeek-only / Solver-only | `STOP` | none | `deepseek-v4-flash` | TBD | TBD | TBD | Single SolverAgent baseline. |
| TBD | Fixed graph: planner | `ADD_PLAN_SKETCH STOP` | none | `deepseek-v4-flash` | TBD | TBD | TBD | Tests whether a lightweight plan helps without dynamic policy. |
| TBD | Fixed graph: task brief + planner | `ADD_TASK_BRIEF ADD_PLAN_SKETCH STOP` | none | `deepseek-v4-flash` | TBD | TBD | TBD | Tests whether question restatement helps. |
| TBD | Fixed graph: adversarial judge | `ADD_ADVERSARIAL_JUDGE STOP` | none | `deepseek-v4-flash` | TBD | TBD | TBD | Debate-style graph; may help hard cases but can hurt easy cases. |
| TBD | Fixed graph: verifier | `ADD_FORMAT_VERIFIER STOP` | none | `deepseek-v4-flash` | TBD | TBD | TBD | Mostly format robustness, not expected to improve reasoning much. |

## Required Fields

Every new row should record:

```text
Date
Variant name
Graph/actions or policy type
Checkpoint path, if any
LLM model and key config: temperature, thinking, max_tokens
Correct / total
Accuracy
Result path containing summary.json, results.jsonl, and results.tsv
Short note: prompt version, code commit, or special observation
```

## Recommended Commands

Fixed graph:

```bash
python eval/eval_fix.py \
  --dataset mmlu \
  --data-path data/MMLU_subsets/gptswarm153/val \
  --split val \
  --selection-jsonl data/MMLU_subsets/gptswarm153/selection.jsonl \
  --actions STOP \
  --run-id gptswarm153-solver-only \
  --output-dir artifacts/evals/fixed \
  --env .env
```

Policy graph:

```bash
bash scripts/run_best_policy_gptswarm153.sh \
  --run-id gptswarm153-best-policy
```

## Notes

- `docs/` is currently ignored by Git. Use `git add -f docs/gptswarm153_agent_acc.md` if this log should be committed.
- For strict comparison, keep `.env` consistent across runs, especially `GOGAGENT_MODEL`, `GOGAGENT_TEMPERATURE`, `GOGAGENT_THINKING`, and `GOGAGENT_MAX_TOKENS`.
- Remote LLM evaluation is not perfectly deterministic. Important claims should be rerun or checked by per-question consistency.
