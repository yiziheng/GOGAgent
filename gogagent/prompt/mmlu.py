"""MMLU prompt set for GOGAgent."""

from __future__ import annotations

from types import MappingProxyType

from gogagent.prompt.base import AgentPromptSet



MMLU_CONTEXT_INSTRUCTION = (
    "The available context is only advisory and may be wrong. "
    "Use it only if it helps your assigned role."
)
MMLU_SOLVER_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. "
    "Answer the following multiple-choice question. "
    "Return only one letter: A, B, C, or D."
)
MMLU_AGENT_SYSTEM_PROMPTS = MappingProxyType(
    {
        "solver": MMLU_SOLVER_SYSTEM_PROMPT,
        "solver_brief_rationale": (
            "You are a knowledgeable MMLU assistant. "
            "Choose the single best option. "
            "Keep reasoning short and do not reveal step-by-step chain of thought. "
            "Follow the requested Answer, Reason, and Risk output format exactly."
        ),
        "task_brief": (
            "You are a task brief agent. "
            "Your job is to state what the question asks in one short sentence. "
            "Do not answer the question."
        ),
        "plan_sketch": (
            "You are a MMLU planning agent. "
            "Decide whether the question needs extra solving structure. "
            "Default to Repeat count: 1. "
            "Use Repeat count: 2 only for clear multi-step math, logic, symbolic, "
            "or chain reasoning. "
            "Write at most two short hints, not a full solution. "
            "Include exactly one line: Repeat count: 1 or Repeat count: 2. "
            "Do not answer the question."
        ),
        "adversarial_judge": (
            "You are an MMLU option-shuffle self-consistency agent. "
            "Solve the question from the shuffled options independently. "
            "Do not reveal reasoning. "
            "Return exactly one capital letter: A, B, C, or D."
        ),
        "adversarial_judge_brief_rationale": (
            "You are an independent MMLU second-opinion agent. "
            "Solve the question independently from the question and options. "
            "Keep reasoning short and do not reveal step-by-step chain of thought. "
            "Follow the requested Answer, Reason, and Risk output format exactly."
        ),
        "adversarial_arbitrator": (
            "You are a fair MMLU arbitration judge. "
            "Two agents may have given different answers. "
            "Choose the single best answer based only on the question and options. "
            "Do not favor either agent by default. "
            "Do not reveal reasoning. "
            "Return exactly one capital letter: A, B, C, or D."
        ),
        "supervisor": (
            "You are a conservative MMLU supervisor. "
            "Your job is to review the solver's answer, find clear mistakes, "
            "and return the final option. "
            "The solver is a strong baseline, so keep its answer unless the "
            "question, options, or domain rule clearly support another option. "
            "Pay special attention to NOT, EXCEPT, LEAST, MOST, BEST, TRUE, FALSE, "
            "necessary, sufficient, arithmetic, and option-label mismatch. "
            "Do not reveal reasoning. "
            "Return exactly one capital letter: A, B, C, or D."
        ),
        "format_verifier": (
            "You are a format verifier. "
            "Your job is to return the parseable final answer only."
        ),
        "challenger": (
            "You are a MMLU challenge agent. "
            "Find one serious reason the upstream answer may be wrong. "
            "Compare only options A, B, C, and D. "
            "Do not give a final answer."
        ),
        "defender": (
            "You are a MMLU defense agent. "
            "Use the challenge only if it is valid. "
            "Keep the upstream answer unless another option is clearly better. "
            "Return exactly one capital letter: A, B, C, or D."
        ),
        "judge": (
            "You are a MMLU final judge. "
            "Compare the upstream answer, challenge, and defense fairly. "
            "Choose the single best option. "
            "Return exactly one capital letter: A, B, C, or D."
        ),
        "format_checker": (
            "You are a format checker. "
            "Your job is to check whether the upstream answer is parseable. "
            "Be concise."
        ),
        "answer_normalizer": (
            "You are an answer normalizer. "
            "Your job is to convert the best available answer into the required final format."
        ),
        "task_classifier": (
            "You are a task classifier. "
            "Your job is to name the task type in a few words. "
            "Do not answer the question."
        ),
    }
)

MMLU_PROMPTS = AgentPromptSet(
    dataset="mmlu",
    agent_system_prompts=MMLU_AGENT_SYSTEM_PROMPTS,
    default_text_system_template="",
    json_system_template="",
    context_instruction=MMLU_CONTEXT_INSTRUCTION,
)
