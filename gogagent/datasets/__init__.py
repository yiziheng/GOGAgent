"""Dataset file loaders with an explicit public-task / train-only-gold split."""

from gogagent.datasets.loaders import (
    DatasetExample,
    SUPPORTED_DATASETS,
    iter_examples,
    load_examples,
    load_gsm8k_jsonl,
    load_humaneval_jsonl,
    load_mmlu_directory,
    load_multiagentbench_jsonl,
    normalize_dataset,
)
from gogagent.datasets.mmlu_fewshot import (
    attach_mmlu_fewshot_examples,
    load_mmlu_fewshot_by_subject,
)
from gogagent.datasets.prompt_specs import (
    DatasetPromptSpec,
    GENERIC_PROMPT_SPEC,
    MMLU_DIRECT_SYSTEM_PROMPT,
    PROMPT_SPECS,
    answer_instruction,
    format_mmlu_direct_task,
    format_mmlu_fewshot_task,
    format_problem,
    get_prompt_spec,
    parse_mmlu_answer_like,
    parse_answer_text,
)
from gogagent.datasets.problems import (
    answer_format_for_dataset,
    enrich_example,
    load_selection,
    make_problem,
)

__all__ = [
    "DatasetPromptSpec",
    "DatasetExample",
    "GENERIC_PROMPT_SPEC",
    "MMLU_DIRECT_SYSTEM_PROMPT",
    "PROMPT_SPECS",
    "SUPPORTED_DATASETS",
    "attach_mmlu_fewshot_examples",
    "answer_format_for_dataset",
    "answer_instruction",
    "enrich_example",
    "format_problem",
    "format_mmlu_direct_task",
    "format_mmlu_fewshot_task",
    "get_prompt_spec",
    "iter_examples",
    "load_examples",
    "load_gsm8k_jsonl",
    "load_humaneval_jsonl",
    "load_mmlu_directory",
    "load_multiagentbench_jsonl",
    "load_mmlu_fewshot_by_subject",
    "load_selection",
    "make_problem",
    "normalize_dataset",
    "parse_answer_text",
    "parse_mmlu_answer_like",
]
