"""Agent registry and concrete prompt-wrapper agents."""

from gogagent.agents.adversarial_judge import AdversarialJudgeAgent
from gogagent.agents.answer_normalizer import AnswerNormalizerAgent
from gogagent.agents.base import Agent
from gogagent.agents.challenger import ChallengerAgent
from gogagent.agents.defender import DefenderAgent
from gogagent.agents.format_checker import FormatCheckerAgent
from gogagent.agents.format_verifier import FormatVerifierAgent
from gogagent.agents.judge import JudgeAgent
from gogagent.agents.mmlu_shuffle import MMLUMajorityVoteAgent, ShuffledMMLUSolverAgent
from gogagent.agents.plan_sketch import PlanSketchAgent
from gogagent.agents.registry import (
    agent_from_dict,
    create_agent,
    get_agent,
    get_agent_class,
    is_standalone_agent,
    list_agent_specs,
)
from gogagent.agents.solver import SolverAgent
from gogagent.agents.supervisor import SupervisorAgent
from gogagent.agents.task_brief import TaskBriefAgent
from gogagent.agents.task_classifier import TaskClassifierAgent

__all__ = [
    "AdversarialJudgeAgent",
    "Agent",
    "AnswerNormalizerAgent",
    "ChallengerAgent",
    "DefenderAgent",
    "FormatCheckerAgent",
    "FormatVerifierAgent",
    "JudgeAgent",
    "MMLUMajorityVoteAgent",
    "PlanSketchAgent",
    "ShuffledMMLUSolverAgent",
    "SolverAgent",
    "SupervisorAgent",
    "TaskBriefAgent",
    "TaskClassifierAgent",
    "agent_from_dict",
    "create_agent",
    "get_agent",
    "get_agent_class",
    "is_standalone_agent",
    "list_agent_specs",
]
