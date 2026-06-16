"""Independent correctness oracle for MMLU-Pro dynamic-choice tasks."""

from __future__ import annotations

from typing import Any, Mapping

from gogagent.datasets.mmlu_pro import extract_mmlu_pro_label
from gogagent.datasets.prompt_specs import mmlu_pro_choice_labels, mmlu_pro_options


def score_mmlu_pro(example: Mapping[str, Any], prediction: Any, gold: Any) -> bool:
    """Return whether ``prediction`` matches ``gold`` for dynamic MMLU-Pro labels."""

    options = mmlu_pro_options(example)
    labels = mmlu_pro_choice_labels(example)
    predicted_label = extract_mmlu_pro_label(
        prediction,
        labels=labels,
        options={str(label): str(value) for label, value in options.items()},
    )
    gold_label = extract_mmlu_pro_label(
        gold,
        labels=labels,
        options={str(label): str(value) for label, value in options.items()},
    )
    return predicted_label == gold_label
