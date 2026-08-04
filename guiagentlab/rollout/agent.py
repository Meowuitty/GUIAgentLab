"""MobileWorld multi-turn agent loop for verl."""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import ray
import torch
from PIL import Image

from guiagentlab.data.datasets import SYSTEM_PROMPT
from guiagentlab.replay.library import ScoredReplayLibrary
from guiagentlab.replay.milestones import MilestoneReplayLibrary
from guiagentlab.env.client import MobileWorldClient
from guiagentlab.env.errors import FailureKind, InfrastructureError
from guiagentlab.rollout.teacher_intervention import ConservativeTeacherIntervention
from guiagentlab.reward import (
    MilestoneObservation,
    MilestoneTracker,
    ProcessRewardConfig,
    ProcessRewardJudge,
    ProcessRewardResult,
    asymmetric_milestone_rewards,
    compose_admire_rewards,
    curriculum_epoch,
    discounted_returns,
    normalize_intermediate_rewards,
)
from guiagentlab.reward.loop_guard import DeterministicLoopGuard
from guiagentlab.rollout.action import (
    PolicyOutputError,
    action_requires_environment_step,
    is_terminal_action,
    parse_action_or_wait,
)
from guiagentlab.rollout.maiui import (
    build_maiui_history,
    canonical_prompt_prefix,
    gui_state_detail_fingerprint,
    gui_state_fingerprint,
)
from guiagentlab.rollout.manager import (
    ADMIREAgentLoopOutput,
    GiGPOAgentLoopOutput,
    _action_reward,
    _decode_screenshot,
    _is_successful_terminal_step,
    _new_rollout_metrics,
    _stable_int,
)
from guiagentlab.rollout.recording import EpisodeRecorder, json_value, utc_now
from guiagentlab.training.tracking import write_json_atomic
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__name__)


@register("mobileworld_agent")
class MobileWorldAgentLoop(AgentLoopBase):
    """Failure-aware multi-turn VLM loop.

    An infrastructure exception discards the whole attempt and draws another
    container. Exhausting retries raises and stops the batch instead of poisoning
    training rewards.
    """

    def __init__(self, *args, **kwargs):
        self.teacher_server_manager = kwargs.pop("teacher_server_manager", None)
        self.teacher_key = kwargs.pop("teacher_key", None)
        super().__init__(*args, **kwargs)
        settings = self.config.agent_environment
        self.settings = settings
        self.pool = ray.get_actor(str(settings.pool_name))
        self.max_steps = int(settings.max_steps)
        self.max_attempts = int(settings.max_infrastructure_attempts)
        self.response_length = int(self.rollout_config.response_length)
        self.max_action_tokens = int(settings.get("max_action_tokens", self.response_length))
        self.history_length = int(settings.get("history_length", 3))
        self.allow_ask_user = bool(settings.get("allow_ask_user", False))
        self.step_wait_time = float(settings.get("step_wait_time", 2.0))
        self.emit_step_samples = bool(settings.get("emit_step_samples", True))
        self.invalid_action_penalty = float(settings.get("invalid_action_penalty", 0.1))
        self.loop_guard_enabled = bool(settings.get("loop_guard_enabled", True))
        self.loop_guard_repeat_threshold = int(settings.get("loop_guard_repeat_threshold", 3))
        self.loop_guard_max_cycle_length = int(settings.get("loop_guard_max_cycle_length", 8))
        self.loop_guard_coordinate_bucket = int(settings.get("loop_guard_coordinate_bucket", 25))
        self.skip_repeated_teacher_queries = bool(
            settings.get("skip_repeated_teacher_queries", False)
        )
        self.teacher_intervention_enabled = bool(
            settings.get("teacher_intervention_enabled", False)
        )
        self.teacher_max_interventions = int(
            settings.get("teacher_max_interventions", 2)
        )
        self.teacher_cooldown_steps = int(
            settings.get("teacher_cooldown_steps", 2)
        )
        self.teacher_no_change_threshold = int(
            settings.get("teacher_no_change_threshold", 3)
        )
        self.advantage_estimator = str(self.config.algorithm.adv_estimator).lower()
        self.admire_milestone_weight = float(
            self.config.algorithm.get("admire_milestone_weight", 0.3)
        )
        self.admire_milestone_decay = float(
            self.config.algorithm.get("admire_milestone_decay", 0.99)
        )
        self.admire_failed_hit_bonus = float(
            self.config.algorithm.get("admire_failed_hit_bonus", 0.5)
        )
        self.admire_invalid_coefficient = float(
            self.config.algorithm.get("admire_invalid_coefficient", 0.5)
        )
        self.admire_invalid_reward = float(
            self.config.algorithm.get("admire_invalid_reward", -1.0)
        )
        self.admire_loop_reward = float(
            self.config.algorithm.get("admire_loop_reward", -0.25)
        )
        self.admire_successful_terminal_reward = float(
            self.config.algorithm.get(
                "admire_successful_terminal_reward",
                0.25,
            )
        )
        self.process_reward_config: ProcessRewardConfig | None = None
        self.process_reward_pool = None
        if self.advantage_estimator == "gigpo":
            self.process_reward_config = ProcessRewardConfig.from_environment()
            self.process_reward_pool = ray.get_actor(f"{str(settings.pool_name)}_prm")
        if self.max_action_tokens <= 0 or self.max_action_tokens > self.response_length:
            raise ValueError(
                f"agent_environment.max_action_tokens must be in [1, {self.response_length}]"
            )
        if self.allow_ask_user:
            raise ValueError("GUI-only MAI-UI training does not support ask_user episodes")
        if self.advantage_estimator == "admire_grpo":
            if self.admire_milestone_weight < 0:
                raise ValueError("ADMIRE milestone weight must be non-negative")
            if not 0 <= self.admire_milestone_decay <= 1:
                raise ValueError("ADMIRE milestone decay must be in [0, 1]")
            if self.admire_failed_hit_bonus < 0:
                raise ValueError("ADMIRE failed-hit bonus must be non-negative")
        self.recorder = EpisodeRecorder(os.environ.get("GUIAGENTLAB_EPISODE_DIR"))

    async def _tear_down_with_slot(
        self,
        lease: dict[str, str],
        client: MobileWorldClient,
        task_name: str,
    ) -> None:
        await self.pool.begin_teardown.remote(
            lease,
            float(
                self.settings.get(
                    "teardown_timeout",
                    self.settings.acquire_timeout,
                )
            ),
        )
        try:
            await asyncio.to_thread(client.tear_down, task_name)
        finally:
            await self.pool.end_teardown.remote(lease)

    @staticmethod
    def _record_process_reward(
        step: dict[str, Any],
        step_record: dict[str, Any],
        result: ProcessRewardResult,
    ) -> None:
        step["process_reward"] = float(result.reward)
        step["process_reward_reason"] = result.reason
        step["process_reward_source"] = result.source
        step["process_reward_valid"] = bool(result.valid)
        step_record.update(
            {
                "process_reward": float(result.reward),
                "process_reward_reason": result.reason,
                "process_reward_source": result.source,
                "process_reward_valid": bool(result.valid),
            }
        )

    def _submit_teacher_query(
        self,
        *,
        prompt_ids: list[int],
        response_ids: list[int],
        images: list[Image.Image],
        routing_key: object,
    ) -> asyncio.Task[tuple[torch.Tensor, torch.Tensor]]:
        if self.teacher_server_manager is None:
            raise RuntimeError("OPD teacher manager is not available")
        if hasattr(routing_key, "item"):
            routing_key = routing_key.item()
        return asyncio.create_task(
            self.teacher_server_manager.compute_teacher_logprobs_single(
                sequence_ids=prompt_ids + response_ids,
                multi_modal_data={"images": images},
                mm_processor_kwargs=self.mm_processor_kwargs,
                routing_key=routing_key,
            )
        )

    async def _generate_teacher_action(
        self,
        *,
        prompt_ids: list[int],
        images: list[Image.Image],
        routing_key: object,
    ) -> TokenOutput:
        """Generate one deterministic teacher proposal."""
        if self.teacher_server_manager is None:
            raise RuntimeError("OPD teacher manager is not available")
        if hasattr(routing_key, "item"):
            routing_key = routing_key.item()
        return await self.teacher_server_manager.generate_teacher_action_single(
            prompt_ids=prompt_ids,
            max_tokens=self.max_action_tokens,
            multi_modal_data={"images": images},
            mm_processor_kwargs=self.mm_processor_kwargs,
            routing_key=routing_key,
        )

    async def _submit_process_reward(self, **kwargs: Any) -> dict[str, Any]:
        if self.process_reward_pool is None:
            raise RuntimeError("GiGPO process-reward pool is not initialized")
        submitted_at = time.monotonic()
        payload = await self.process_reward_pool.score.remote(**kwargs)
        payload["end_to_end_seconds"] = time.monotonic() - submitted_at
        return payload

    async def _collect_process_rewards(
        self,
        pending: dict[int, asyncio.Task[dict[str, Any]]],
        gigpo_steps: list[dict[str, Any]],
        step_records: list[dict[str, Any]],
    ) -> None:
        if not pending:
            return
        indices = list(pending)
        payloads = await asyncio.gather(
            *(pending[index] for index in indices),
            return_exceptions=True,
        )
        for step_index, payload in zip(indices, payloads, strict=True):
            if isinstance(payload, BaseException):
                self._record_process_reward(
                    gigpo_steps[step_index],
                    step_records[step_index],
                    ProcessRewardJudge.unscored(
                        f"process reward pool unavailable after "
                        f"{type(payload).__name__}: {payload}",
                        "unscored_pool_error",
                    ),
                )
                continue
            self._record_process_reward(
                gigpo_steps[step_index],
                step_records[step_index],
                ProcessRewardResult(
                    reward=float(payload["reward"]),
                    reason=str(payload["reason"]),
                    source=str(payload["source"]),
                    valid=bool(payload["valid"]),
                ),
            )
            step_records[step_index].update(
                {
                    "process_reward_queue_seconds": float(payload["queue_seconds"]),
                    "process_reward_latency_seconds": float(payload["request_seconds"]),
                    "process_reward_end_to_end_seconds": float(payload["end_to_end_seconds"]),
                }
            )

    @staticmethod
    def _infrastructure_error(exc: BaseException) -> dict[str, Any]:
        if isinstance(exc, InfrastructureError):
            return {
                "type": type(exc).__name__,
                "kind": exc.kind.value,
                "message": exc.message,
                "endpoint": exc.endpoint,
                "retryable": exc.retryable,
                "details": json_value(exc.details),
            }
        return {
            "type": type(exc).__name__,
            "kind": "no_capacity" if isinstance(exc, TimeoutError) else "internal",
            "message": str(exc),
        }

    async def run(
        self, sampling_params: dict[str, Any], priority: int = 0, **kwargs
    ) -> AgentLoopOutput | list[AgentLoopOutput]:
        extra_info = kwargs.get("extra_info", {}) or {}
        task_name = str(extra_info.get("task_name") or kwargs.get("task_name") or "")
        if not task_name:
            raise ValueError("dataset record has no extra_info.task_name")
        goal_hint = extra_info.get("goal")
        evaluation_mode = os.environ.get("GUIAGENTLAB_EVAL_MODE")
        sample_count = (
            int(self.config.actor_rollout_ref.rollout.val_kwargs.get("n", 1))
            if evaluation_mode
            else int(self.rollout_config.get("n", 1))
        )
        sample_index = int(priority) % max(sample_count, 1)
        episode_id = self.recorder.new_episode_id()
        episode_started_at = utc_now()
        episode_started = time.monotonic()
        infrastructure_attempts: list[str] = []
        last_error: BaseException | None = None
        for attempt_index in range(1, self.max_attempts + 1):
            lease: dict[str, str] | None = None
            client: MobileWorldClient | None = None
            initialized = False
            stage = "acquire"
            attempt_record: dict[str, Any] = {
                "episode_id": episode_id,
                "attempt": attempt_index,
                "task_name": task_name,
                "task_index": extra_info.get("index"),
                "started_at": utc_now(),
                "stage": stage,
                "steps": [],
            }
            try:
                lease = await self.pool.acquire.remote(float(self.settings.acquire_timeout))
                attempt_record["endpoint"] = lease["endpoint"]
                attempt_record["container_role"] = lease["role"]
                client = MobileWorldClient(
                    lease["endpoint"],
                    device=str(self.settings.device),
                    timeout=float(self.settings.request_timeout),
                    step_wait_time=self.step_wait_time,
                )
                stage = attempt_record["stage"] = "health"
                await asyncio.to_thread(client.health)
                stage = attempt_record["stage"] = "controller_initialization"
                await asyncio.to_thread(client.initialize_controller)
                stage = attempt_record["stage"] = "goal"
                goal = str(goal_hint or await asyncio.to_thread(client.goal, task_name))
                attempt_record["goal"] = goal
                stage = attempt_record["stage"] = "initialization_queue"
                await self.pool.begin_initialization.remote(
                    lease,
                    float(
                        self.settings.get(
                            "initialization_timeout",
                            self.settings.acquire_timeout,
                        )
                    ),
                )
                try:
                    stage = attempt_record["stage"] = "task_initialization"
                    await asyncio.to_thread(client.initialize_task, task_name)
                    initialized = True
                    stage = attempt_record["stage"] = "initial_screenshot"
                    screenshot = await asyncio.to_thread(client.screenshot_png)
                finally:
                    await self.pool.end_initialization.remote(lease)
                output = await self._run_valid_attempt(
                    client,
                    task_name,
                    goal,
                    screenshot,
                    sampling_params,
                    priority=int(priority),
                    raw_prompt=kwargs.get("raw_prompt"),
                    episode_group=kwargs.get("uid", task_name),
                    teacher_routing_key=(
                        kwargs.get(self.teacher_key)
                        if self.teacher_server_manager is not None and self.teacher_key is not None
                        else None
                    ),
                    episode_id=episode_id,
                    attempt_index=attempt_index,
                    attempt_record=attempt_record,
                    training_step=int(kwargs.get("training_step", -1)),
                )
                stage = attempt_record["stage"] = "teardown"
                cleanup_error = None
                container_disposition = "released"
                try:
                    await self._tear_down_with_slot(lease, client, task_name)
                    initialized = False
                    stage = attempt_record["stage"] = "release"
                    await self.pool.release.remote(lease)
                except (InfrastructureError, TimeoutError) as cleanup_exc:
                    # Evaluation already produced an authoritative reward.  A
                    # post-episode cleanup failure invalidates the container,
                    # not the completed trajectory or its GRPO prompt group.
                    cleanup_error = self._infrastructure_error(cleanup_exc)
                    initialized = False
                    await self.pool.quarantine.remote(lease, str(cleanup_exc))
                    container_disposition = "quarantined_recovery_started"
                attempt_record["stage"] = "complete"
                attempt_record["finished_at"] = utc_now()
                for item in output if isinstance(output, list) else [output]:
                    item.extra_fields.update(
                        {
                            "episode_id": episode_id,
                            "endpoint": lease["endpoint"],
                            "infrastructure_attempts_discarded": len(infrastructure_attempts),
                            "post_episode_cleanup_failed": cleanup_error is not None,
                        }
                    )
                episode_payload = {
                    "episode_id": episode_id,
                    "experiment_name": str(self.config.trainer.experiment_name),
                    "model_path": str(self.config.actor_rollout_ref.model.path),
                    "evaluation_mode": evaluation_mode,
                    "task_name": task_name,
                    "task_index": extra_info.get("index"),
                    "sample_index": sample_index,
                    "uid": kwargs.get("uid"),
                    "goal": goal,
                    "raw_prompt": kwargs.get("raw_prompt"),
                    "sampling_params": sampling_params,
                    "process_reward": (
                        self.process_reward_config.public_metadata()
                        if self.process_reward_config is not None
                        else None
                    ),
                    "started_at": episode_started_at,
                    "finished_at": utc_now(),
                    "duration_seconds": time.monotonic() - episode_started,
                    "endpoint": lease["endpoint"],
                    "container_role": lease["role"],
                    "successful_attempt": attempt_index,
                    "infrastructure_attempts": infrastructure_attempts,
                    "cleanup_error": cleanup_error,
                    "container_disposition": container_disposition,
                    "valid": True,
                    "step_count": len(attempt_record["steps"]),
                    "termination": attempt_record["termination"],
                    "final_score": float(attempt_record["final_score"]),
                    "evaluation_reason": attempt_record.get("evaluation_reason"),
                    "model_errors": attempt_record.get("model_errors", []),
                    "initial_screenshot": attempt_record.get("initial_screenshot"),
                    "steps": attempt_record["steps"],
                }
                await asyncio.to_thread(self.recorder.write_episode, episode_id, episode_payload)
                return output
            except (InfrastructureError, TimeoutError) as exc:
                last_error = exc
                teardown_error = None
                if initialized and client is not None and stage != "teardown":
                    try:
                        assert lease is not None
                        await self._tear_down_with_slot(lease, client, task_name)
                        initialized = False
                    except InfrastructureError as cleanup_exc:
                        teardown_error = self._infrastructure_error(cleanup_exc)
                disposition = "none"
                if lease is not None:
                    await self.pool.quarantine.remote(lease, str(exc))
                    disposition = "quarantined_recovery_started"
                attempt_record.update(
                    {
                        "stage": attempt_record.get("stage", stage),
                        "finished_at": utc_now(),
                        "valid": False,
                        "termination": "infrastructure_failure",
                        "error": self._infrastructure_error(exc),
                        "cleanup_error": teardown_error,
                        "container_disposition": disposition,
                    }
                )
                relative = f"infrastructure/{episode_id}-attempt-{attempt_index:02d}.json"
                infrastructure_attempts.append(relative)
                await asyncio.to_thread(
                    self.recorder.write_infrastructure_attempt,
                    episode_id,
                    attempt_index,
                    attempt_record,
                )
                logger.warning(
                    "discarding infrastructure-invalid rollout task=%s attempt=%d endpoint=%s: %s",
                    task_name,
                    attempt_index,
                    lease["endpoint"] if lease else "none",
                    exc,
                )
            except BaseException as exc:
                if lease is not None:
                    try:
                        await self.pool.quarantine.remote(lease, str(exc))
                    except Exception:
                        pass
                attempt_record.update(
                    {
                        "stage": attempt_record.get("stage", stage),
                        "finished_at": utc_now(),
                        "valid": False,
                        "termination": "internal_error",
                        "error": self._infrastructure_error(exc),
                    }
                )
                await asyncio.to_thread(
                    self.recorder.write_internal_error,
                    episode_id,
                    attempt_index,
                    attempt_record,
                )
                raise
        raise RuntimeError(
            f"MobileWorld rollout exhausted {self.max_attempts} infrastructure attempts"
        ) from last_error

    async def _run_valid_attempt(
        self,
        client: MobileWorldClient,
        task_name: str,
        goal: str,
        screenshot_png: bytes,
        sampling_params: dict[str, Any],
        *,
        priority: int,
        raw_prompt: Any,
        episode_group: object,
        teacher_routing_key: object,
        episode_id: str,
        attempt_index: int,
        attempt_record: dict[str, Any],
        training_step: int,
    ) -> AgentLoopOutput | list[AgentLoopOutput]:
        pending_process_rewards: dict[int, asyncio.Task[dict[str, Any]]] = {}
        pending_teacher_queries: dict[
            int, asyncio.Task[tuple[torch.Tensor, torch.Tensor]]
        ] = {}
        valid_attempt = False
        try:
            output = await self._run_valid_attempt_impl(
                client,
                task_name,
                goal,
                screenshot_png,
                sampling_params,
                priority=priority,
                raw_prompt=raw_prompt,
                episode_group=episode_group,
                teacher_routing_key=teacher_routing_key,
                episode_id=episode_id,
                attempt_index=attempt_index,
                attempt_record=attempt_record,
                training_step=training_step,
                pending_process_rewards=pending_process_rewards,
                pending_teacher_queries=pending_teacher_queries,
            )
            valid_attempt = True
            return output
        finally:
            unfinished = [task for task in pending_process_rewards.values() if not task.done()]
            for task in unfinished:
                task.cancel()
            if unfinished:
                await asyncio.gather(*unfinished, return_exceptions=True)
            if not valid_attempt:
                teacher_tasks = list(pending_teacher_queries.values())
                for task in teacher_tasks:
                    if not task.done():
                        task.cancel()
                if teacher_tasks:
                    await asyncio.gather(*teacher_tasks, return_exceptions=True)

    async def materialize_scored_replay(
        self,
        library: ScoredReplayLibrary,
        *,
        task_name: str,
        source_kwargs: dict[str, Any],
        replaced_trajectory_uid: str,
    ) -> list[GiGPOAgentLoopOutput]:
        """Render an offline success path with the current MAI-UI processor."""
        episode, cached_scores = await asyncio.to_thread(library.load, task_name)
        screenshot_payloads = await asyncio.to_thread(library.image_bytes, episode)
        screenshots = [
            _decode_screenshot(payload, f"offline-replay:{task_name}")
            for payload in screenshot_payloads
        ]
        extra_info = source_kwargs.get("extra_info", {}) or {}
        source_task = str(extra_info.get("task_name") or source_kwargs.get("task_name") or "")
        if source_task != task_name:
            raise ValueError(
                f"Success replay source task differs from replay task: {source_task} != {task_name}"
            )
        source_goal = str(extra_info.get("goal") or "")
        replay_goal = str(episode["goal"])
        if source_goal.strip() != replay_goal.strip():
            raise ValueError(f"Success replay task goal differs for {task_name}")

        raw_prompt = source_kwargs.get("raw_prompt")
        if isinstance(raw_prompt, np.ndarray):
            raw_prompt = raw_prompt.tolist()
        prefix = canonical_prompt_prefix(
            raw_prompt,
            replay_goal,
            SYSTEM_PROMPT,
        )
        current_screenshot_index = int(episode["initial_screenshot"])
        history_images = [screenshots[current_screenshot_index]]
        assistant_responses: list[str] = []
        replay_steps: list[dict[str, Any]] = []
        for index, (step, cached) in enumerate(
            zip(episode["steps"], cached_scores, strict=True)
        ):
            before_index = int(step["screenshot_before"])
            after_index = int(step["screenshot_after"])
            if current_screenshot_index != before_index:
                raise ValueError(
                    f"Success replay screenshot chain breaks at {task_name} step {index}: "
                    f"{current_screenshot_index} != {before_index}"
                )
            messages, images = build_maiui_history(
                prefix,
                history_images,
                assistant_responses,
                self.history_length,
            )
            prompt_ids = await self.apply_chat_template(messages, images=images)
            model_text = str(step["model_text"])
            response_ids = self.tokenizer.encode(
                model_text,
                add_special_tokens=False,
            )
            if not response_ids or len(response_ids) > self.max_action_tokens:
                raise ValueError(
                    f"Success replay response length is invalid at {task_name} step {index}: "
                    f"{len(response_ids)}"
                )
            before_image = screenshots[before_index]
            after_image = screenshots[after_index]
            replay_steps.append(
                {
                    "prompt_ids": list(prompt_ids),
                    "response_ids": list(response_ids),
                    "images": list(images),
                    "anchor_uid": gui_state_fingerprint(before_image),
                    "anchor_detail_uid": gui_state_detail_fingerprint(before_image),
                    "process_reward": float(cached["reward"]),
                    "process_reward_reason": str(cached.get("reason", "")),
                    "process_reward_source": "replay",
                    "process_reward_valid": True,
                    "state_changed": bool(
                        gui_state_detail_fingerprint(before_image)
                        != gui_state_detail_fingerprint(after_image)
                    ),
                }
            )
            assistant_responses.append(model_text)
            history_images.append(after_image)
            current_screenshot_index = after_index

        process_rewards = [step["process_reward"] for step in replay_steps]
        normalized = normalize_intermediate_rewards(process_rewards)
        returns = discounted_returns(
            normalized,
            float(self.config.algorithm.gamma),
        )
        replay_episode_id = (
            f"success_replay-{episode['episode_id']}-{uuid4().hex[:12]}"
        )
        episode_uid = _stable_int(source_kwargs["uid"])
        trajectory_uid = f"{replaced_trajectory_uid}:success_replay:{episode['episode_id']}"
        common_extra = {
            "environment_valid": True,
            "task_name": task_name,
            "termination": "model_terminal",
            "evaluation_reason": "successful replay trajectory",
            "episode_reward": 1.0,
            "trajectory_uid": trajectory_uid,
            "teacher_queries": 0,
            "teacher_queries_skipped": 0,
            "episode_id": replay_episode_id,
            "endpoint": "offline_replay",
            "infrastructure_attempts_discarded": 0,
            "post_episode_cleanup_failed": False,
            "success_replay": True,
            "success_replay_source_episode_id": str(episode["episode_id"]),
            "success_replay_replaced_trajectory_uid": replaced_trajectory_uid,
        }
        output = []
        for index, step in enumerate(replay_steps):
            successful_terminal = index == len(replay_steps) - 1
            output.append(
                GiGPOAgentLoopOutput(
                    prompt_ids=step["prompt_ids"],
                    response_ids=step["response_ids"],
                    response_mask=[1] * len(step["response_ids"]),
                    response_logprobs=None,
                    multi_modal_data={"images": step["images"]},
                    reward_score=1.0,
                    num_turns=index + 2,
                    metrics=_new_rollout_metrics(),
                    extra_fields={
                        **common_extra,
                        "step_index": index,
                        "episode_uid": episode_uid,
                        "anchor_uid": step["anchor_uid"],
                        "anchor_detail_uid": step["anchor_detail_uid"],
                        "process_reward": step["process_reward"],
                        "normalized_process_reward": normalized[index],
                        "process_reward_reason": step["process_reward_reason"],
                        "process_reward_source": step["process_reward_source"],
                        "process_reward_valid": step["process_reward_valid"],
                        "step_returns": returns[index],
                        "action_valid": True,
                        "state_changed": step["state_changed"],
                        "loop_detected": False,
                        "successful_terminal": successful_terminal,
                        "action_reward_adjustment": 0.0,
                    },
                    episode_uid=episode_uid,
                    anchor_uid=step["anchor_uid"],
                    anchor_detail_uid=step["anchor_detail_uid"],
                    step_returns=returns[index],
                )
            )

        replay_dir = Path(
            os.environ.get("GUIAGENTLAB_EPISODE_DIR", ".")
        ) / "replay"
        await asyncio.to_thread(
            write_json_atomic,
            replay_dir / f"{replay_episode_id}.json",
            {
                "schema_version": 1,
                "episode_id": replay_episode_id,
                "task_name": task_name,
                "uid": str(source_kwargs["uid"]),
                "replaced_trajectory_uid": replaced_trajectory_uid,
                "source_episode_id": str(episode["episode_id"]),
                "step_count": len(output),
                "final_score": 1.0,
            },
        )
        return output

    async def materialize_admire_replay(
        self,
        library: MilestoneReplayLibrary,
        *,
        task_name: str,
        source_kwargs: dict[str, Any],
        replaced_trajectory_uid: str,
    ) -> list[ADMIREAgentLoopOutput]:
        """Render a cached successful path with evaluator-derived ADMIRE rewards."""
        episode, cached_milestones = await asyncio.to_thread(
            library.load,
            task_name,
        )
        screenshot_payloads = await asyncio.to_thread(library.image_bytes, episode)
        screenshots = [
            _decode_screenshot(payload, f"offline-replay:{task_name}")
            for payload in screenshot_payloads
        ]
        extra_info = source_kwargs.get("extra_info", {}) or {}
        source_task = str(
            extra_info.get("task_name") or source_kwargs.get("task_name") or ""
        )
        if source_task != task_name:
            raise ValueError(
                f"Success replay source task differs from replay task: {source_task} != {task_name}"
            )
        source_goal = str(extra_info.get("goal") or "")
        replay_goal = str(episode["goal"])
        if source_goal.strip() != replay_goal.strip():
            raise ValueError(f"Success replay task goal differs for {task_name}")

        raw_prompt = source_kwargs.get("raw_prompt")
        if isinstance(raw_prompt, np.ndarray):
            raw_prompt = raw_prompt.tolist()
        prefix = canonical_prompt_prefix(
            raw_prompt,
            replay_goal,
            SYSTEM_PROMPT,
        )
        current_screenshot_index = int(episode["initial_screenshot"])
        history_images = [screenshots[current_screenshot_index]]
        assistant_responses: list[str] = []
        replay_steps: list[dict[str, Any]] = []
        for index, (step, milestone) in enumerate(
            zip(episode["steps"], cached_milestones, strict=True)
        ):
            before_index = int(step["screenshot_before"])
            after_index = int(step["screenshot_after"])
            if current_screenshot_index != before_index:
                raise ValueError(
                    f"Success replay screenshot chain breaks at {task_name} step {index}: "
                    f"{current_screenshot_index} != {before_index}"
                )
            messages, images = build_maiui_history(
                prefix,
                history_images,
                assistant_responses,
                self.history_length,
            )
            prompt_ids = await self.apply_chat_template(messages, images=images)
            model_text = str(step["model_text"])
            response_ids = self.tokenizer.encode(
                model_text,
                add_special_tokens=False,
            )
            if not response_ids or len(response_ids) > self.max_action_tokens:
                raise ValueError(
                    f"Success replay response length is invalid at {task_name} step {index}: "
                    f"{len(response_ids)}"
                )
            before_image = screenshots[before_index]
            after_image = screenshots[after_index]
            _, action_valid, warning = parse_action_or_wait(
                model_text,
                before_image.size,
            )
            if not action_valid:
                raise ValueError(
                    f"Success replay source action is invalid at {task_name} step {index}: "
                    f"{warning}"
                )
            replay_steps.append(
                {
                    "prompt_ids": list(prompt_ids),
                    "response_ids": list(response_ids),
                    "images": list(images),
                    "state_changed": bool(
                        gui_state_detail_fingerprint(before_image)
                        != gui_state_detail_fingerprint(after_image)
                    ),
                    "active_terminal": bool(is_terminal_action(step["action"])),
                    "milestone_completed": list(milestone["completed"]),
                    "milestone_newly_completed": list(
                        milestone["newly_completed"]
                    ),
                    "milestone_progress": float(milestone["progress"]),
                    "milestone_hit": float(milestone["hit"]),
                }
            )
            assistant_responses.append(model_text)
            history_images.append(after_image)
            current_screenshot_index = after_index

        training_step = int(source_kwargs.get("training_step", 1))
        if training_step < 1:
            training_step = 1
        admire_epoch = curriculum_epoch(
            training_step,
            total_training_steps=int(
                self.config.actor_rollout_ref.actor.optim.total_training_steps
            ),
            total_epochs=int(self.config.trainer.total_epochs),
        )
        observations = [
            MilestoneObservation(
                registered=True,
                completed=tuple(step["milestone_completed"]),
                newly_completed=tuple(step["milestone_newly_completed"]),
                progress=float(step["milestone_progress"]),
                hit=float(step["milestone_hit"]),
            )
            for step in replay_steps
        ]
        milestone_rewards = asymmetric_milestone_rewards(
            observations,
            successful=True,
            failed_hit_bonus=self.admire_failed_hit_bonus,
        )
        successful_terminal = [
            _is_successful_terminal_step(
                active_terminal=step["active_terminal"],
                step_index=index,
                step_count=len(replay_steps),
                termination="model_terminal",
                episode_reward=1.0,
            )
            for index, step in enumerate(replay_steps)
        ]
        total_rewards, milestone_coefficient = compose_admire_rewards(
            outcome=1.0,
            milestone_rewards=milestone_rewards,
            action_valid=[True] * len(replay_steps),
            loop_detected=[False] * len(replay_steps),
            successful_terminal=successful_terminal,
            epoch=admire_epoch,
            milestone_weight=self.admire_milestone_weight,
            milestone_decay=self.admire_milestone_decay,
            invalid_coefficient=self.admire_invalid_coefficient,
            invalid_reward=self.admire_invalid_reward,
            loop_reward=self.admire_loop_reward,
            successful_terminal_reward=self.admire_successful_terminal_reward,
        )

        replay_episode_id = f"success_replay-{episode['episode_id']}-{uuid4().hex[:12]}"
        trajectory_uid = f"{replaced_trajectory_uid}:success_replay:{episode['episode_id']}"
        common_extra = {
            "environment_valid": True,
            "task_name": task_name,
            "termination": "model_terminal",
            "evaluation_reason": "successful replay trajectory",
            "episode_reward": 1.0,
            "trajectory_uid": trajectory_uid,
            "teacher_queries": 0,
            "teacher_queries_skipped": 0,
            "teacher_intervention_calls": 0,
            "teacher_intervention_applications": 0,
            "episode_id": replay_episode_id,
            "endpoint": "offline_replay",
            "infrastructure_attempts_discarded": 0,
            "post_episode_cleanup_failed": False,
            "success_replay": True,
            "success_replay_source_episode_id": str(episode["episode_id"]),
            "success_replay_replaced_trajectory_uid": replaced_trajectory_uid,
            "milestone_registered": True,
            "milestone_ids": list(cached_milestones[0]["states"]),
            "admire_training_step": training_step,
            "admire_epoch": admire_epoch,
            "admire_milestone_coefficient": milestone_coefficient,
        }
        output = []
        for index, step in enumerate(replay_steps):
            output.append(
                ADMIREAgentLoopOutput(
                    prompt_ids=step["prompt_ids"],
                    response_ids=step["response_ids"],
                    response_mask=[1] * len(step["response_ids"]),
                    response_logprobs=None,
                    multi_modal_data={"images": step["images"]},
                    reward_score=total_rewards[index],
                    num_turns=index + 2,
                    metrics=_new_rollout_metrics(),
                    extra_fields={
                        **common_extra,
                        "step_index": index,
                        "action_valid": True,
                        "state_changed": step["state_changed"],
                        "loop_detected": False,
                        "successful_terminal": successful_terminal[index],
                        "milestone_completed": step["milestone_completed"],
                        "milestone_newly_completed": step[
                            "milestone_newly_completed"
                        ],
                        "milestone_progress": step["milestone_progress"],
                        "milestone_hit": step["milestone_hit"],
                        "milestone_reward": milestone_rewards[index],
                        "admire_total_reward": total_rewards[index],
                        "action_reward_adjustment": total_rewards[index] - 1.0,
                        "teacher_intervention_called": False,
                        "teacher_intervention_applied": False,
                        "teacher_intervention_reason": None,
                    },
                    admire_total_reward=total_rewards[index],
                )
            )

        replay_dir = Path(
            os.environ.get("GUIAGENTLAB_EPISODE_DIR", ".")
        ) / "replay"
        await asyncio.to_thread(
            write_json_atomic,
            replay_dir / f"{replay_episode_id}.json",
            {
                "schema_version": 1,
                "episode_id": replay_episode_id,
                "task_name": task_name,
                "uid": str(source_kwargs["uid"]),
                "replaced_trajectory_uid": replaced_trajectory_uid,
                "source_episode_id": str(episode["episode_id"]),
                "step_count": len(output),
                "milestone_hit_steps": [
                    index
                    for index, value in enumerate(milestone_rewards)
                    if value > 0
                ],
                "final_score": 1.0,
            },
        )
        return output

    async def _run_valid_attempt_impl(
        self,
        client: MobileWorldClient,
        task_name: str,
        goal: str,
        screenshot_png: bytes,
        sampling_params: dict[str, Any],
        *,
        priority: int,
        raw_prompt: Any,
        episode_group: object,
        teacher_routing_key: object,
        episode_id: str,
        attempt_index: int,
        attempt_record: dict[str, Any],
        training_step: int,
        pending_process_rewards: dict[int, asyncio.Task[dict[str, Any]]],
        pending_teacher_queries: dict[
            int, asyncio.Task[tuple[torch.Tensor, torch.Tensor]]
        ],
    ) -> AgentLoopOutput | list[AgentLoopOutput]:
        image = _decode_screenshot(screenshot_png, client.endpoint)
        milestone_tracker: MilestoneTracker | None = None
        current_milestone_snapshot = None
        admire_epoch = 0
        if self.advantage_estimator == "admire_grpo":
            current_milestone_snapshot = await asyncio.to_thread(
                client.milestones,
                task_name,
            )
            milestone_tracker = MilestoneTracker(current_milestone_snapshot)
            admire_epoch = curriculum_epoch(
                training_step,
                total_training_steps=int(
                    self.config.actor_rollout_ref.actor.optim.total_training_steps
                ),
                total_epochs=int(self.config.trainer.total_epochs),
            )
            attempt_record["milestones"] = {
                "registered": milestone_tracker.registered,
                "ids": list(milestone_tracker.milestone_ids),
                "initial_completed": list(milestone_tracker.initial_completed),
                "training_step": training_step,
                "epoch": admire_epoch,
            }

        async def observe_milestones_after_action() -> MilestoneObservation | None:
            nonlocal current_milestone_snapshot
            if milestone_tracker is None:
                return None
            if milestone_tracker.registered:
                current_milestone_snapshot = await asyncio.to_thread(
                    client.milestones,
                    task_name,
                )
            assert current_milestone_snapshot is not None
            try:
                return milestone_tracker.observe(current_milestone_snapshot)
            except ValueError as exc:
                raise InfrastructureError(
                    FailureKind.PROTOCOL,
                    f"milestone evaluator contract changed: {exc}",
                    client.endpoint,
                ) from exc

        def record_milestone_observation(
            step: dict[str, Any],
            step_record: dict[str, Any],
            observation: MilestoneObservation | None,
        ) -> None:
            if observation is None:
                return
            values = {
                "milestone_registered": observation.registered,
                "milestone_completed": list(observation.completed),
                "milestone_newly_completed": list(observation.newly_completed),
                "milestone_progress": float(observation.progress),
                "milestone_hit": float(observation.hit),
            }
            step.update(values)
            step_record.update(values)

        initial_screenshot = await asyncio.to_thread(
            self.recorder.write_screenshot,
            episode_id,
            attempt_index,
            0,
            screenshot_png,
        )
        attempt_record["initial_screenshot"] = initial_screenshot
        prefix = canonical_prompt_prefix(
            [] if raw_prompt is None else list(raw_prompt),
            goal,
            SYSTEM_PROMPT,
        )
        history_images = [image]
        assistant_responses: list[str] = []
        parsed_actions: list[dict[str, Any]] = []
        request_id = uuid4().hex
        metrics: dict[str, Any] = _new_rollout_metrics()
        score = 0.0
        reason: str | None = None
        termination = "max_steps"
        turns = 0
        gigpo_steps: list[dict[str, Any]] = []
        teacher_enabled = self.teacher_server_manager is not None
        teacher_intervention = ConservativeTeacherIntervention(
            enabled=self.teacher_intervention_enabled and teacher_enabled,
            max_calls=self.teacher_max_interventions,
            cooldown_steps=self.teacher_cooldown_steps,
            no_change_threshold=self.teacher_no_change_threshold,
        )
        no_change_streak = 0
        current_anchor = gui_state_fingerprint(image)
        current_anchor_detail = gui_state_detail_fingerprint(image)
        loop_guard = (
            DeterministicLoopGuard(
                repeat_threshold=self.loop_guard_repeat_threshold,
                max_cycle_length=self.loop_guard_max_cycle_length,
                coordinate_bucket=self.loop_guard_coordinate_bucket,
            )
            if (
                self.loop_guard_enabled
                and (
                    self.advantage_estimator in ("gigpo", "admire_grpo")
                    or (teacher_enabled and self.skip_repeated_teacher_queries)
                    or teacher_intervention.enabled
                )
            )
            else None
        )

        current_screenshot = initial_screenshot
        current_png = screenshot_png
        for step_index in range(self.max_steps):
            attempt_record["stage"] = f"generation_step_{step_index}"
            messages, images = build_maiui_history(
                prefix,
                history_images,
                assistant_responses,
                self.history_length,
            )
            prompt_ids = await self.apply_chat_template(messages, images=images)
            step_prompt_ids = list(prompt_ids)
            step_images = list(images)
            turn_sampling_params = dict(sampling_params)
            # ``data.max_response_length`` is the original MAI-UI *per action*
            # generation limit. Each environment step is a fresh generation
            # over the rolling 3-image context; screenshot tokens must never
            # consume the next action's 512-token allowance.
            turn_sampling_params["max_tokens"] = self.max_action_tokens
            generation_started = time.monotonic()
            output: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=turn_sampling_params,
                image_data=images,
                priority=priority,
            )
            metrics["generate_sequences"] += time.monotonic() - generation_started
            turns += 1
            in_teacher_cooldown = teacher_intervention.start_step()
            student_generated = list(output.token_ids)
            gigpo_steps.append(
                {
                    "prompt_ids": step_prompt_ids,
                    "response_ids": student_generated,
                    # Teacher intervention can replace selected responses.
                    # Keep this optional field uniform across the batch and let
                    # the trainer recompute student old_log_probs for every row.
                    "response_logprobs": (
                        None
                        if teacher_intervention.enabled
                        else (list(output.log_probs) if output.log_probs else None)
                    ),
                    "images": step_images,
                    "anchor_uid": current_anchor,
                    "anchor_detail_uid": current_anchor_detail,
                    "process_reward": None,
                    "milestone_registered": False,
                    "milestone_completed": [],
                    "milestone_newly_completed": [],
                    "milestone_progress": 0.0,
                    "milestone_hit": 0.0,
                    "loop_detected": False,
                    "loop_reason": None,
                    "teacher_query_skipped": False,
                    "teacher_future": None,
                    "active_terminal": False,
                    "state_changed": None,
                    "teacher_intervention_called": False,
                    "teacher_intervention_applied": False,
                    "teacher_intervention_reason": None,
                }
            )
            student_text = self.tokenizer.decode(
                student_generated,
                skip_special_tokens=True,
            )
            step_record: dict[str, Any] = {
                "index": step_index,
                "screenshot_before": current_screenshot,
                "model_text": student_text,
                "prompt_token_count": len(step_prompt_ids),
                "generated_token_count": len(student_generated),
                "teacher_in_cooldown": in_teacher_cooldown,
            }
            attempt_record["steps"].append(step_record)
            action, model_output_valid, parse_warning = parse_action_or_wait(
                student_text, history_images[-1].size
            )
            step_record["model_output_valid"] = model_output_valid
            if parse_warning:
                step_record["parse_warning"] = parse_warning
            if not model_output_valid:
                model_error = {
                    "type": PolicyOutputError.__name__,
                    "message": parse_warning,
                    "fallback_action": "wait",
                }
                step_record["error"] = model_error
                attempt_record.setdefault("model_errors", []).append(model_error)

            if action["action_type"] == "ask_user" and not self.allow_ask_user:
                step_record["rejected_action"] = action
                action = {"action_type": "wait"}
                step_record["model_output_valid"] = False
                model_error = {
                    "type": PolicyOutputError.__name__,
                    "message": "ask_user is outside the configured GUI-only action space",
                    "fallback_action": "wait",
                }
                step_record["error"] = model_error
                attempt_record.setdefault("model_errors", []).append(model_error)

            student_action = dict(action)
            student_action_valid = bool(step_record["model_output_valid"])
            student_guard = copy.deepcopy(loop_guard)
            student_loop_reason = (
                student_guard.observe(
                    anchor_uid=gigpo_steps[-1]["anchor_uid"],
                    anchor_detail_uid=gigpo_steps[-1]["anchor_detail_uid"],
                    action=student_action,
                )
                if student_guard is not None
                else None
            )
            trigger_reasons = teacher_intervention.trigger_reasons(
                action_valid=student_action_valid,
                loop_reason=student_loop_reason,
                no_change_streak=no_change_streak,
                in_cooldown=in_teacher_cooldown,
            )

            generated = student_generated
            text = student_text
            applied_teacher_action = False
            loop_reason = student_loop_reason
            if trigger_reasons:
                # Consuming the budget before awaiting is deliberate: rejected
                # and failed teacher calls are interventions too.
                teacher_intervention.record_call()
                gigpo_steps[-1]["teacher_intervention_called"] = True
                gigpo_steps[-1]["teacher_intervention_reason"] = "+".join(trigger_reasons)
                step_record["teacher_intervention_called"] = True
                step_record["teacher_intervention_reason"] = list(trigger_reasons)
                step_record["student_model_text"] = student_text
                step_record["student_action"] = student_action
                step_record["student_model_output_valid"] = student_action_valid
                teacher_started = time.monotonic()
                try:
                    teacher_output = await self._generate_teacher_action(
                        prompt_ids=step_prompt_ids,
                        images=step_images,
                        routing_key=teacher_routing_key,
                    )
                    step_record["teacher_seconds"] = (
                        time.monotonic() - teacher_started
                    )
                    teacher_generated = list(teacher_output.token_ids)
                    teacher_text = self.tokenizer.decode(
                        teacher_generated,
                        skip_special_tokens=True,
                    )
                    teacher_action, teacher_valid, teacher_warning = (
                        parse_action_or_wait(
                            teacher_text,
                            history_images[-1].size,
                        )
                    )
                    if (
                        teacher_action["action_type"] == "ask_user"
                        and not self.allow_ask_user
                    ):
                        teacher_action = {"action_type": "wait"}
                        teacher_valid = False
                        teacher_warning = (
                            "teacher proposed ask_user outside the GUI-only action space"
                        )
                    teacher_guard = copy.deepcopy(loop_guard)
                    teacher_loop_reason = (
                        teacher_guard.observe(
                            anchor_uid=gigpo_steps[-1]["anchor_uid"],
                            anchor_detail_uid=gigpo_steps[-1]["anchor_detail_uid"],
                            action=teacher_action,
                        )
                        if teacher_guard is not None
                        else None
                    )
                    step_record.update(
                        {
                            "teacher_model_text": teacher_text,
                            "teacher_generated_token_count": len(
                                teacher_generated
                            ),
                            "teacher_action": teacher_action,
                            "teacher_action_valid": bool(teacher_valid),
                            "teacher_parse_warning": teacher_warning,
                            "teacher_loop_reason": teacher_loop_reason,
                        }
                    )
                    if teacher_valid and teacher_loop_reason is None:
                        applied_teacher_action = True
                        teacher_intervention.record_application()
                        generated = teacher_generated
                        text = teacher_text
                        action = teacher_action
                        model_output_valid = True
                        parse_warning = teacher_warning
                        loop_guard = teacher_guard
                        loop_reason = None
                        gigpo_steps[-1]["response_logprobs"] = None
                        if "parse_warning" in step_record:
                            step_record["student_parse_warning"] = step_record.pop(
                                "parse_warning"
                            )
                        step_record.pop("error", None)
                    else:
                        rejection_reasons = []
                        if not teacher_valid:
                            rejection_reasons.append("invalid_teacher_action")
                        if teacher_loop_reason is not None:
                            rejection_reasons.append("teacher_action_loops")
                        step_record["teacher_rejection_reason"] = "+".join(
                            rejection_reasons
                        )
                except Exception as exc:
                    step_record["teacher_seconds"] = (
                        time.monotonic() - teacher_started
                    )
                    step_record["teacher_action_valid"] = False
                    step_record["teacher_rejection_reason"] = (
                        f"teacher_call_failed:{type(exc).__name__}"
                    )
                    logger.warning(
                        "OPD teacher rescue call failed for %s step %d: %s",
                        episode_id,
                        step_index,
                        exc,
                    )

            if not applied_teacher_action:
                action = student_action
                model_output_valid = student_action_valid
                loop_guard = student_guard
                loop_reason = student_loop_reason

            gigpo_steps[-1]["response_ids"] = generated
            gigpo_steps[-1]["active_terminal"] = bool(is_terminal_action(action))
            gigpo_steps[-1]["action_valid"] = bool(model_output_valid)
            gigpo_steps[-1]["teacher_intervention_applied"] = applied_teacher_action
            gigpo_steps[-1]["loop_reason"] = loop_reason
            if (
                loop_reason is not None
                and self.advantage_estimator in ("gigpo", "admire_grpo")
            ):
                gigpo_steps[-1]["loop_detected"] = True
                step_record["loop_detected"] = True
            step_record["model_text"] = text
            step_record["generated_token_count"] = len(generated)
            step_record["model_output_valid"] = bool(model_output_valid)
            step_record["action"] = action
            step_record["teacher_intervention_applied"] = applied_teacher_action
            if parse_warning:
                step_record["parse_warning"] = parse_warning
            assistant_responses.append(text)
            parsed_actions.append(dict(action))

            if (
                loop_reason is not None
                and teacher_enabled
                and self.skip_repeated_teacher_queries
            ):
                repeated_indices = loop_guard.last_repeated_indices
                for repeated_index in repeated_indices:
                    repeated_step = gigpo_steps[repeated_index]
                    repeated_record = attempt_record["steps"][repeated_index]
                    repeated_step["teacher_query_skipped"] = True
                    repeated_record["teacher_query_skipped"] = True
                    repeated_record["teacher_skip_reason"] = loop_reason
                    pending = pending_teacher_queries.pop(repeated_index, None)
                    if pending is not None:
                        pending.cancel()
                        await asyncio.gather(pending, return_exceptions=True)
                        repeated_step["teacher_future"] = None

            if teacher_enabled and not gigpo_steps[-1]["teacher_query_skipped"]:
                teacher_future = self._submit_teacher_query(
                    prompt_ids=step_prompt_ids,
                    response_ids=generated,
                    images=step_images,
                    routing_key=teacher_routing_key,
                )
                pending_teacher_queries[step_index] = teacher_future
                gigpo_steps[-1]["teacher_future"] = teacher_future
                step_record["teacher_query_submitted"] = True

            if not action_requires_environment_step(action):
                record_milestone_observation(
                    gigpo_steps[-1],
                    step_record,
                    await observe_milestones_after_action(),
                )
                termination = "model_terminal"
                attempt_record["stage"] = "evaluation"
                evaluation_started = time.monotonic()
                evaluation = await asyncio.to_thread(client.evaluate, task_name)
                metrics["compute_score"] += time.monotonic() - evaluation_started
                score, reason = evaluation.score, evaluation.reason
                self._record_process_reward(
                    gigpo_steps[-1],
                    step_record,
                    ProcessRewardResult(
                        reward=float(score),
                        reason=reason or "terminal environment evaluation",
                        source="environment_final",
                    ),
                )
                break

            attempt_record["stage"] = f"environment_step_{step_index}"
            tool_started = time.monotonic()
            result = await asyncio.to_thread(client.step, action)

            # MAI-UI treats the simulated user's answer as the next user turn.
            # It does not attach a fresh screenshot to that turn because the
            # phone state did not change while the user answered.
            if action["action_type"] == "ask_user":
                user_answer = result.get("result")
                if not isinstance(user_answer, str) or not user_answer:
                    raise InfrastructureError(
                        FailureKind.PROTOCOL,
                        "ask_user returned no textual user response",
                        client.endpoint,
                        details={"response": json_value(result)},
                    )
                messages.append({"role": "user", "content": user_answer})
                step_record["user_response"] = user_answer
                step_record["screenshot_after"] = current_screenshot
                gigpo_steps[-1]["state_changed"] = False
                step_record["state_changed"] = False
                record_milestone_observation(
                    gigpo_steps[-1],
                    step_record,
                    await observe_milestones_after_action(),
                )
                metrics["tool_calls"] += time.monotonic() - tool_started
                continue

            attempt_record["stage"] = f"screenshot_step_{step_index}"
            next_png = await asyncio.to_thread(client.screenshot_png)
            next_image = _decode_screenshot(next_png, client.endpoint)
            record_milestone_observation(
                gigpo_steps[-1],
                step_record,
                await observe_milestones_after_action(),
            )
            next_screenshot = await asyncio.to_thread(
                self.recorder.write_screenshot,
                episode_id,
                attempt_index,
                step_index + 1,
                next_png,
            )
            step_record["screenshot_after"] = next_screenshot
            current_screenshot = next_screenshot
            observation = {
                "role": "user",
                "content": [{"type": "image"}],
            }
            messages.append(observation)
            history_images.append(next_image)
            metrics["tool_calls"] += time.monotonic() - tool_started
            current_anchor = gui_state_fingerprint(next_image)
            current_anchor_detail = gui_state_detail_fingerprint(next_image)
            gigpo_steps[-1]["state_changed"] = bool(
                current_anchor_detail != gigpo_steps[-1]["anchor_detail_uid"]
            )
            step_record["state_changed"] = gigpo_steps[-1]["state_changed"]
            if gigpo_steps[-1]["state_changed"]:
                no_change_streak = 0
            else:
                no_change_streak += 1
            step_record["no_change_streak"] = no_change_streak
            current_png_before_action = current_png
            current_png = next_png
            final_step = is_terminal_action(action) or step_index + 1 >= self.max_steps
            if final_step:
                if is_terminal_action(action):
                    termination = "model_terminal"
                attempt_record["stage"] = "evaluation"
                evaluation_started = time.monotonic()
                evaluation = await asyncio.to_thread(client.evaluate, task_name)
                metrics["compute_score"] += time.monotonic() - evaluation_started
                score, reason = evaluation.score, evaluation.reason
                self._record_process_reward(
                    gigpo_steps[-1],
                    step_record,
                    ProcessRewardResult(
                        reward=float(score),
                        reason=reason or "terminal environment evaluation",
                        source="environment_final",
                    ),
                )
                break
            if self.advantage_estimator != "gigpo":
                process_reward = ProcessRewardResult(
                    reward=0.0,
                    reason="process rewards are not used by this algorithm",
                    source="not_applicable",
                )
            else:
                if loop_reason is not None:
                    process_reward = ProcessRewardResult(
                        reward=0.0,
                        reason=loop_reason,
                        source="rule_loop_guard",
                    )
                else:
                    pending_process_rewards[step_index] = asyncio.create_task(
                        self._submit_process_reward(
                            task_goal=goal,
                            before_png=current_png_before_action,
                            after_png=next_png,
                            parsed_action=dict(action),
                            previous_actions=[
                                dict(previous_action)
                                for previous_action in parsed_actions[:-1]
                            ],
                        )
                    )
                    process_reward = None
            if process_reward is not None:
                self._record_process_reward(
                    gigpo_steps[-1],
                    step_record,
                    process_reward,
                )
        else:
            attempt_record["stage"] = "evaluation"
            evaluation_started = time.monotonic()
            evaluation = await asyncio.to_thread(client.evaluate, task_name)
            metrics["compute_score"] += time.monotonic() - evaluation_started
            score, reason = evaluation.score, evaluation.reason
            if gigpo_steps:
                self._record_process_reward(
                    gigpo_steps[-1],
                    attempt_record["steps"][-1],
                    ProcessRewardResult(
                        reward=float(score),
                        reason=reason or "terminal environment evaluation",
                        source="environment_final",
                    ),
                )

        attempt_record["stage"] = "process_reward_collection"
        await self._collect_process_rewards(
            pending_process_rewards,
            gigpo_steps,
            attempt_record["steps"],
        )

        attempt_record["termination"] = termination
        attempt_record["final_score"] = float(score)
        attempt_record["evaluation_reason"] = reason
        attempt_record["teacher_intervention_calls"] = teacher_intervention.calls
        attempt_record["teacher_intervention_applications"] = (
            teacher_intervention.applications
        )

        common_extra = {
            "environment_valid": True,
            "task_name": task_name,
            "termination": termination,
            "evaluation_reason": reason,
            "episode_reward": float(score),
            "trajectory_uid": episode_id,
            "teacher_queries": sum(
                step["teacher_future"] is not None for step in gigpo_steps
            ),
            "teacher_queries_skipped": sum(
                bool(step["teacher_query_skipped"]) for step in gigpo_steps
            ),
            "teacher_intervention_calls": teacher_intervention.calls,
            "teacher_intervention_applications": teacher_intervention.applications,
            "success_replay": False,
            "success_replay_source_episode_id": None,
            "success_replay_replaced_trajectory_uid": None,
        }
        if self.advantage_estimator == "admire_grpo":
            if not gigpo_steps:
                raise RuntimeError("ADMIRE rollout produced no model step")
            assert milestone_tracker is not None
            observations = [
                MilestoneObservation(
                    registered=bool(step["milestone_registered"]),
                    completed=tuple(step["milestone_completed"]),
                    newly_completed=tuple(step["milestone_newly_completed"]),
                    progress=float(step["milestone_progress"]),
                    hit=float(step["milestone_hit"]),
                )
                for step in gigpo_steps
            ]
            milestone_rewards = asymmetric_milestone_rewards(
                observations,
                successful=float(score) > 0,
                failed_hit_bonus=self.admire_failed_hit_bonus,
            )
            successful_terminal = [
                _is_successful_terminal_step(
                    active_terminal=step["active_terminal"],
                    step_index=index,
                    step_count=len(gigpo_steps),
                    termination=termination,
                    episode_reward=score,
                )
                for index, step in enumerate(gigpo_steps)
            ]
            total_rewards, milestone_coefficient = compose_admire_rewards(
                outcome=float(score),
                milestone_rewards=milestone_rewards,
                action_valid=[bool(step["action_valid"]) for step in gigpo_steps],
                loop_detected=[bool(step["loop_detected"]) for step in gigpo_steps],
                successful_terminal=successful_terminal,
                epoch=admire_epoch,
                milestone_weight=self.admire_milestone_weight,
                milestone_decay=self.admire_milestone_decay,
                invalid_coefficient=self.admire_invalid_coefficient,
                invalid_reward=self.admire_invalid_reward,
                loop_reward=self.admire_loop_reward,
                successful_terminal_reward=self.admire_successful_terminal_reward,
            )
            for index, step in enumerate(gigpo_steps):
                step.update(
                    {
                        "milestone_reward": milestone_rewards[index],
                        "successful_terminal": successful_terminal[index],
                        "admire_total_reward": total_rewards[index],
                    }
                )
                attempt_record["steps"][index].update(
                    {
                        "milestone_reward": milestone_rewards[index],
                        "successful_terminal": successful_terminal[index],
                        "admire_total_reward": total_rewards[index],
                        "admire_milestone_coefficient": milestone_coefficient,
                    }
                )
            common_extra.update(
                {
                    "milestone_registered": milestone_tracker.registered,
                    "milestone_ids": list(milestone_tracker.milestone_ids),
                    "admire_training_step": training_step,
                    "admire_epoch": admire_epoch,
                    "admire_milestone_coefficient": milestone_coefficient,
                }
            )
            return [
                ADMIREAgentLoopOutput(
                    prompt_ids=step["prompt_ids"],
                    response_ids=step["response_ids"],
                    response_mask=[1] * len(step["response_ids"]),
                    response_logprobs=step["response_logprobs"],
                    multi_modal_data={"images": step["images"]},
                    reward_score=step["admire_total_reward"],
                    num_turns=index + 2,
                    metrics=metrics,
                    extra_fields={
                        **common_extra,
                        "step_index": index,
                        "action_valid": step["action_valid"],
                        "state_changed": step["state_changed"],
                        "loop_detected": step["loop_detected"],
                        "successful_terminal": step["successful_terminal"],
                        "milestone_completed": step["milestone_completed"],
                        "milestone_newly_completed": step[
                            "milestone_newly_completed"
                        ],
                        "milestone_progress": step["milestone_progress"],
                        "milestone_hit": step["milestone_hit"],
                        "milestone_reward": step["milestone_reward"],
                        "admire_total_reward": step["admire_total_reward"],
                        "action_reward_adjustment": (
                            step["admire_total_reward"] - float(score)
                        ),
                        "teacher_intervention_called": step["teacher_intervention_called"],
                        "teacher_intervention_applied": step["teacher_intervention_applied"],
                        "teacher_intervention_reason": step["teacher_intervention_reason"],
                        **(
                            {"_guiagentlab_teacher_future": step["teacher_future"]}
                            if step["teacher_future"] is not None
                            else {}
                        ),
                    },
                    admire_total_reward=step["admire_total_reward"],
                )
                for index, step in enumerate(gigpo_steps)
                if not (teacher_enabled and step["teacher_query_skipped"])
            ]
        if self.advantage_estimator == "gigpo":
            gamma = float(self.config.algorithm.gamma)
            episode_uid = _stable_int(episode_group)
            missing = [
                index for index, step in enumerate(gigpo_steps) if step["process_reward"] is None
            ]
            if missing:
                raise RuntimeError(
                    f"GiGPO steps are missing process rewards at indices {missing[:5]}"
                )
            process_rewards = [float(step["process_reward"]) for step in gigpo_steps]
            normalized_process_rewards = normalize_intermediate_rewards(process_rewards)
            step_returns = discounted_returns(
                normalized_process_rewards,
                gamma,
            )
            for index, normalized_reward in enumerate(normalized_process_rewards):
                gigpo_steps[index]["normalized_process_reward"] = normalized_reward
                attempt_record["steps"][index]["normalized_process_reward"] = normalized_reward
                gigpo_steps[index]["successful_terminal"] = _is_successful_terminal_step(
                    active_terminal=gigpo_steps[index]["active_terminal"],
                    step_index=index,
                    step_count=len(gigpo_steps),
                    termination=termination,
                    episode_reward=score,
                )
                attempt_record["steps"][index]["loop_detected"] = gigpo_steps[index][
                    "loop_detected"
                ]
                attempt_record["steps"][index]["successful_terminal"] = gigpo_steps[index][
                    "successful_terminal"
                ]
            return [
                GiGPOAgentLoopOutput(
                    prompt_ids=step["prompt_ids"],
                    response_ids=step["response_ids"],
                    response_mask=[1] * len(step["response_ids"]),
                    response_logprobs=step["response_logprobs"],
                    multi_modal_data={"images": step["images"]},
                    reward_score=_action_reward(
                        score, step["action_valid"], self.invalid_action_penalty
                    ),
                    num_turns=index + 2,
                    metrics=metrics,
                    extra_fields={
                        **common_extra,
                        "step_index": index,
                        "episode_uid": episode_uid,
                        "anchor_uid": step["anchor_uid"],
                        "anchor_detail_uid": step["anchor_detail_uid"],
                        "process_reward": step["process_reward"],
                        "normalized_process_reward": step["normalized_process_reward"],
                        "process_reward_reason": step["process_reward_reason"],
                        "process_reward_source": step["process_reward_source"],
                        "process_reward_valid": step["process_reward_valid"],
                        "step_returns": step_returns[index],
                        "action_valid": step["action_valid"],
                        "state_changed": step["state_changed"],
                        "teacher_intervention_called": step["teacher_intervention_called"],
                        "teacher_intervention_applied": step["teacher_intervention_applied"],
                        "teacher_intervention_reason": step["teacher_intervention_reason"],
                        "loop_detected": step["loop_detected"],
                        "successful_terminal": step["successful_terminal"],
                        "action_reward_adjustment": _action_reward(
                            score,
                            step["action_valid"],
                            self.invalid_action_penalty,
                        )
                        - float(score),
                        **(
                            {"_guiagentlab_teacher_future": step["teacher_future"]}
                            if step["teacher_future"] is not None
                            else {}
                        ),
                    },
                    episode_uid=episode_uid,
                    anchor_uid=step["anchor_uid"],
                    anchor_detail_uid=step["anchor_detail_uid"],
                    step_returns=step_returns[index],
                )
                for index, step in enumerate(gigpo_steps)
                if not (teacher_enabled and step["teacher_query_skipped"])
            ]

        if not gigpo_steps:
            raise RuntimeError("MobileWorld rollout produced no model step")
        samples = [
            AgentLoopOutput(
                prompt_ids=step["prompt_ids"],
                response_ids=step["response_ids"],
                response_mask=[1] * len(step["response_ids"]),
                response_logprobs=step["response_logprobs"],
                multi_modal_data={"images": step["images"]},
                reward_score=_action_reward(
                    score, step["action_valid"], self.invalid_action_penalty
                ),
                num_turns=index + 2,
                metrics=metrics,
                extra_fields={
                    **common_extra,
                    "step_index": index,
                    "action_valid": step["action_valid"],
                    "state_changed": step["state_changed"],
                    "teacher_intervention_called": step["teacher_intervention_called"],
                    "teacher_intervention_applied": step["teacher_intervention_applied"],
                    "teacher_intervention_reason": step["teacher_intervention_reason"],
                    "action_reward_adjustment": _action_reward(
                        score,
                        step["action_valid"],
                        self.invalid_action_penalty,
                    )
                    - float(score),
                    **(
                        {"_guiagentlab_teacher_future": step["teacher_future"]}
                        if step["teacher_future"] is not None
                        else {}
                    ),
                },
            )
            for index, step in enumerate(gigpo_steps)
            if not (teacher_enabled and step["teacher_query_skipped"])
        ]
        # Evaluation consumes one episode result; training mirrors the original
        # pipeline and updates on every active action in that episode.
        return samples if self.emit_step_samples else samples[-1]
