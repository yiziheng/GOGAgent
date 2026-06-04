"""Label-blind MMLU task features for policy conditioning."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Mapping, Sequence

from gogagent.adapters.mmlu_subjects import subject_profile


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*|\d+(?:\.\d+)?")
_PROFILE_ORDER = ("stem", "humanities", "social_sciences", "professional", "general")
_KEYWORDS = {
    "math": {
        "calculate",
        "equation",
        "probability",
        "ratio",
        "percent",
        "number",
        "value",
        "mean",
        "variance",
        "graph",
        "matrix",
    },
    "law": {
        "law",
        "court",
        "legal",
        "contract",
        "tort",
        "liable",
        "statute",
        "constitutional",
    },
    "medicine": {
        "patient",
        "symptom",
        "diagnosis",
        "disease",
        "treatment",
        "clinical",
        "virus",
        "drug",
        "deficiency",
    },
    "history": {
        "century",
        "war",
        "empire",
        "revolution",
        "treaty",
        "dynasty",
        "historical",
    },
    "philosophy": {
        "argument",
        "ethical",
        "moral",
        "philosopher",
        "fallacy",
        "premise",
        "conclusion",
    },
    "economics": {
        "market",
        "demand",
        "supply",
        "inflation",
        "gdp",
        "price",
        "firm",
    },
}
_QUESTION_TYPE_KEYWORDS = {
    "definition": {"defined", "definition", "refers", "meaning", "term"},
    "compare": {"compare", "contrast", "difference", "similar", "most likely"},
    "causal": {"cause", "because", "why", "effect", "result"},
    "diagnosis": {"diagnosis", "symptom", "patient", "finding", "condition"},
    "exception": {"except", "not", "least", "false", "incorrect"},
}
_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "which",
    "what",
    "following",
}


def encode_mmlu_task(task: Mapping[str, Any]) -> dict[str, Any]:
    """Return compact, label-blind task features for policy selection."""

    subject = _public_subject(task)
    profile = subject_profile(subject)
    question = _public_question(task)
    options = _public_options(task)
    question_tokens = _tokens(question)
    option_tokens = [_tokens(option) for option in options]
    option_lengths = [len(tokens) for tokens in option_tokens]
    all_text_tokens = question_tokens + [token for tokens in option_tokens for token in tokens]
    keyword_flags = {
        name: float(any(token in words for token in all_text_tokens))
        for name, words in _KEYWORDS.items()
    }
    type_flags = {
        name: float(any(token in words for token in question_tokens))
        for name, words in _QUESTION_TYPE_KEYWORDS.items()
    }
    option_overlap = _mean_pairwise_jaccard(option_tokens)
    option_length_mean = _mean(option_lengths)
    option_length_variance = _variance(option_lengths)
    subject_hash = _signed_hash(subject)
    vector = [
        *[1.0 if profile == item else 0.0 for item in _PROFILE_ORDER],
        _clip(len(question) / 600.0),
        _clip(len(question_tokens) / 120.0),
        _clip(option_length_mean / 40.0),
        _clip(max(option_lengths or [0]) / 60.0),
        _clip((max(option_lengths or [0]) - min(option_lengths or [0])) / 50.0),
        _clip(option_length_variance / 100.0),
        option_overlap,
        keyword_flags["math"],
        keyword_flags["law"],
        keyword_flags["medicine"],
        keyword_flags["history"],
        keyword_flags["philosophy"],
        keyword_flags["economics"],
        type_flags["definition"],
        type_flags["compare"],
        type_flags["causal"],
        type_flags["diagnosis"],
        type_flags["exception"],
        _clip(_numeric_density(all_text_tokens) * 4.0),
        subject_hash,
    ]
    return {
        "subject": subject,
        "subject_profile": profile,
        "question_length": len(question),
        "question_word_count": len(question_tokens),
        "option_count": len(options),
        "option_avg_word_count": round(option_length_mean, 6),
        "option_length_variance": round(option_length_variance, 6),
        "option_overlap_mean": round(option_overlap, 6),
        "keyword_flags": keyword_flags,
        "question_type_flags": type_flags,
        "task_vector": [round(float(value), 6) for value in vector],
    }


def _public_question(task: Mapping[str, Any]) -> str:
    return str(task.get("question", task.get("prompt", "")))


def _public_subject(task: Mapping[str, Any]) -> str:
    return str(task.get("subject", task.get("category", "unknown")))


def _public_options(task: Mapping[str, Any]) -> tuple[str, ...]:
    raw_options = task.get("options", task.get("choices", ()))
    if isinstance(raw_options, Mapping):
        return tuple(str(raw_options.get(label, "")) for label in ("A", "B", "C", "D"))
    if isinstance(raw_options, Sequence) and not isinstance(raw_options, (str, bytes)):
        return tuple(str(option) for option in raw_options)
    return ()


def _tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in _WORD_RE.findall(text)
        if token.lower() not in _STOPWORDS
    ]


def _mean(values: Sequence[int | float]) -> float:
    return sum(float(value) for value in values) / len(values) if values else 0.0


def _variance(values: Sequence[int | float]) -> float:
    if not values:
        return 0.0
    average = _mean(values)
    return sum((float(value) - average) ** 2 for value in values) / len(values)


def _mean_pairwise_jaccard(options: Sequence[Sequence[str]]) -> float:
    sets = [set(tokens) for tokens in options]
    scores: list[float] = []
    for index, left in enumerate(sets):
        for right in sets[index + 1 :]:
            union = left | right
            scores.append(len(left & right) / len(union) if union else 0.0)
    return _mean(scores)


def _numeric_density(tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    numeric = sum(1 for token in tokens if token[0].isdigit())
    return numeric / len(tokens)


def _clip(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _signed_hash(value: str) -> float:
    total = sum((index + 1) * ord(character) for index, character in enumerate(value))
    return math.sin(float(total))
