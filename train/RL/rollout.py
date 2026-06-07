"""Rollout execution for GRPO-style policy refinement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from gogagent.actions.base import ActionConstraints, ActionName, total_node_count
from gogagent.artifacts import safe_path
from gogagent.actions.registry import apply_action
from gogagent.artifacts import RunRecorder
from gogagent.datasets import DatasetExample, make_problem
from gogagent.graph.executor import execute_graph
from gogagent.graph.factory import make_initial_graph
from gogagent.llm import AgentContext
from gogagent.policy import GraphEncoder, PolicyNetwork
from gogagent.reward import check_output_format
from train.RL.buffer import RolloutStep, TrajectoryRollout
from train.RL.rewards import compute_rl_reward
from train.RL.sampler import sample_action_step


def rollout_group(
    *,
    epoch: int,
    example_index: int,
    group_index: int,
    example: DatasetExample,
    task_embedding: torch.Tensor,
    graph_encoder: GraphEncoder,
    policy_network: PolicyNetwork,
    context: AgentContext,
    client_description: Mapping[str, Any],
    run_dir: Path,
    constraints: ActionConstraints,
    group_size: int,
    max_actions: int,
    temperature: float,
    save_item_artifacts: bool,
    generator: torch.Generator | None = None,
) -> list[TrajectoryRollout]:
    """Sample and execute a group of rollouts for one training problem."""

    return [
        run_one_rollout(
            epoch=epoch,
            example_index=example_index,
            group_index=group_index,
            rollout_index=rollout_index,
            example=example,
            task_embedding=task_embedding,
            graph_encoder=graph_encoder,
            policy_network=policy_network,
            context=context,
            client_description=client_description,
            run_dir=run_dir,
            constraints=constraints,
            max_actions=max_actions,
            temperature=temperature,
            save_item_artifacts=save_item_artifacts,
            generator=generator,
        )
        for rollout_index in range(1, group_size + 1)
    ]


def run_one_rollout(
    *,
    epoch: int,
    example_index: int,
    group_index: int,
    rollout_index: int,
    example: DatasetExample,
    task_embedding: torch.Tensor,
    graph_encoder: GraphEncoder,
    policy_network: PolicyNetwork,
    context: AgentContext,
    client_description: Mapping[str, Any],
    run_dir: Path,
    constraints: ActionConstraints,
    max_actions: int,
    temperature: float,
    save_item_artifacts: bool,
    generator: torch.Generator | None,
) -> TrajectoryRollout:
    """Sample one graph trajectory, execute it, and compute reward."""

    public_task = dict(example.public_task)
    task_id = str(public_task.get("task_id", f"item-{example_index}"))
    item_dir = (
        run_dir
        / f"epoch{epoch:03d}"
        / f"{example_index:03d}-{safe_path(task_id)}"
        / f"rollout{rollout_index:02d}"
    )
    recorder = RunRecorder(item_dir)
    problem = make_problem(example)
    graph = make_initial_graph(graph_id=f"rl_{safe_path(task_id)}_r{rollout_index:02d}")
    graph.metadata.update(
        {
            "construction": "rl_sample",
            "epoch": epoch,
            "group_index": group_index,
            "rollout_index": rollout_index,
        }
    )

    steps: list[RolloutStep] = []
    action_records: list[dict[str, Any]] = []
    action_sequence: list[str] = []
    stopped = False
    for step_index in range(1, max_actions + 1):
        step = sample_action_step(
            graph=graph,
            task_embedding=task_embedding,
            graph_encoder=graph_encoder,
            policy_network=policy_network,
            constraints=constraints,
            temperature=temperature,
            step=step_index,
            generator=generator,
        )
        steps.append(step)
        action_sequence.append(step.action.value)
        before = step.graph_before
        record = {
            "event": "rl_sampled_action",
            "step": step_index,
            "action": step.action.value,
            "legal": step.action in step.legal_actions,
            "legal_actions": [action.value for action in step.legal_actions],
            "top_actions": step.top_actions,
            "before": before,
        }
        if step.action == ActionName.STOP:
            stopped = True
            record["after"] = before
            action_records.append(record)
            recorder.record_trace(record)
            break
        graph = apply_action(graph, step.action)
        record["after"] = graph.to_dict()
        action_records.append(record)
        recorder.record_trace(record)

    graph.metadata.update(
        {
            "rl_action_sequence": list(action_sequence),
            "rl_stopped": stopped,
            "rl_max_actions": max_actions,
            "graph_node_count": total_node_count(graph),
        }
    )
    if save_item_artifacts:
        graph_json, graph_svg = recorder.save_graph(graph)
        artifacts = {
            **recorder.paths(),
            "gog_json": str(graph_json),
            "gog_svg": str(graph_svg),
        }
    else:
        artifacts = recorder.paths()

    (item_dir / "input.json").write_text(
        json.dumps({"problem": problem, "gold": example.gold}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    llm_start = len(context.llm_calls)
    output = None
    status = "ok"
    error_text = None
    try:
        output = execute_graph(graph, problem, context=context)
        llm_calls = list(context.llm_calls[llm_start:])
        format_result = check_output_format(output)
        reward_breakdown = compute_rl_reward(
            dataset=example.dataset,
            public_task=public_task,
            gold=example.gold,
            final_output=output,
            action_records=action_records,
        )
        recorder.record_trace(
            {
                "event": "final_output",
                "output": output.to_dict(),
                "format": format_result.to_dict(),
                "reward": reward_breakdown.to_dict(),
                "llm_call_count": len(llm_calls),
                "llm_calls": llm_calls,
            }
        )
        rollout = TrajectoryRollout(
            epoch=epoch,
            example_index=example_index,
            group_index=group_index,
            rollout_index=rollout_index,
            task_id=task_id,
            dataset=example.dataset,
            subject=_optional_str(public_task.get("subject")),
            question=_optional_str(public_task.get("question", public_task.get("prompt"))),
            gold=example.gold,
            action_sequence=action_sequence,
            steps=steps,
            final_graph=graph.to_dict(),
            status=status,
            reward=float(reward_breakdown.total),
            reward_breakdown=reward_breakdown.to_dict(),
            prediction=output.answer,
            correct=reward_breakdown.oracle_result.correct,
            format_valid=format_result.valid,
            output=output.to_dict(),
            llm_call_count=len(llm_calls),
            llm_calls=llm_calls,
            item_dir=str(item_dir),
            artifacts=artifacts,
        )
    except Exception as exc:  # noqa: BLE001 - RL keeps item-level failures trainable.
        status = "error"
        error_text = f"{type(exc).__name__}: {exc}"
        llm_calls = list(context.llm_calls[llm_start:])
        reward_breakdown = compute_rl_reward(
            dataset=example.dataset,
            public_task=public_task,
            gold=example.gold,
            final_output=output,
            action_records=action_records,
        )
        recorder.record_trace(
            {
                "event": "rollout_error",
                "error": error_text,
                "reward": reward_breakdown.to_dict(),
                "llm_call_count": len(llm_calls),
                "llm_calls": llm_calls,
            }
        )
        rollout = TrajectoryRollout(
            epoch=epoch,
            example_index=example_index,
            group_index=group_index,
            rollout_index=rollout_index,
            task_id=task_id,
            dataset=example.dataset,
            subject=_optional_str(public_task.get("subject")),
            question=_optional_str(public_task.get("question", public_task.get("prompt"))),
            gold=example.gold,
            action_sequence=action_sequence,
            steps=steps,
            final_graph=graph.to_dict(),
            status=status,
            reward=float(reward_breakdown.total),
            reward_breakdown=reward_breakdown.to_dict(),
            error=error_text,
            llm_call_count=len(llm_calls),
            llm_calls=llm_calls,
            item_dir=str(item_dir),
            artifacts=artifacts,
        )

    recorder.save_summary(
        {
            **rollout.to_dict(),
            "backend": dict(client_description),
        }
    )
    return rollout


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
