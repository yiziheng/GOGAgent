"""Public MMLU subject profile mapping."""

from __future__ import annotations


_SUBJECT_PROFILES = {
    "stem": {
        "abstract_algebra",
        "astronomy",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "electrical_engineering",
        "elementary_mathematics",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_statistics",
        "machine_learning",
    },
    "humanities": {
        "business_ethics",
        "formal_logic",
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "jurisprudence",
        "logical_fallacies",
        "moral_disputes",
        "moral_scenarios",
        "philosophy",
        "prehistory",
        "world_religions",
    },
    "social_sciences": {
        "econometrics",
        "global_facts",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_psychology",
        "human_sexuality",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
    },
    "professional": {
        "anatomy",
        "clinical_knowledge",
        "college_medicine",
        "human_aging",
        "international_law",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "nutrition",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "virology",
    },
}


def subject_profile(subject: str) -> str:
    """Map a public MMLU subject to one coarse, fixed prompt profile."""

    normalized = subject.strip().lower().replace(" ", "_").replace("-", "_")
    for profile, subjects in _SUBJECT_PROFILES.items():
        if normalized in subjects:
            return profile
    return "general"
