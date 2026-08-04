"""verl rollout workers and central MobileWorld pool integration."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from PIL import Image

from guiagentlab.replay.library import ScoredReplayLibrary
from guiagentlab.replay.milestones import MilestoneReplayLibrary
from guiagentlab.replay.success_injection import select_all_failure_replacements
from guiagentlab.env.errors import FailureKind, InfrastructureError
from guiagentlab.env.ray_pool import AsyncContainerPool
from guiagentlab.env.recovery import RecoveryConfig
from guiagentlab.reward import (
    AsyncProcessRewardJudge,
    ProcessRewardConfig,
)
from guiagentlab.rollout.maiui import (
    decode_screenshot,
)
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorker,
    DictConfigWrap,
    ToolListWrap,
)
from verl.utils.ray_utils import auto_await
from verl.utils.rollout_trace import rollout_trace_attr

logger = logging.getLogger(__name__)


def _new_rollout_metrics() -> dict[str, float]:
    """Return every timing bucket consumed by the agent-loop output contract."""
    return {
        "generate_sequences": 0.0,
        "tool_calls": 0.0,
        "compute_score": 0.0,
    }


def _stable_int(value: object) -> int:
    digest = hashlib.sha256(repr(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _action_reward(episode_reward: float, action_valid: bool, penalty: float) -> float:
    """Apply an invalid-action penalty to that action row only."""
    return float(episode_reward) - (0.0 if action_valid else float(penalty))


def _is_successful_terminal_step(
    *,
    active_terminal: bool,
    step_index: int,
    step_count: int,
    termination: str,
    episode_reward: float,
) -> bool:
    """Identify the successful model action that actively ended a trajectory."""
    return bool(
        active_terminal
        and step_index == step_count - 1
        and termination == "model_terminal"
        and float(episode_reward) > 0
    )


def _flatten_step_outputs(
    inputs: list[Any], input_non_tensor_batch: dict[str, Any] | None
) -> tuple[list[Any], dict[str, Any] | None]:
    counts = [len(item) if isinstance(item, list) else 1 for item in inputs]
    flattened = [step for item in inputs for step in (item if isinstance(item, list) else [item])]
    if input_non_tensor_batch is None or all(count == 1 for count in counts):
        return flattened, input_non_tensor_batch

    expanded = {}
    for key, values in input_non_tensor_batch.items():
        array = np.asarray(values)
        if array.ndim == 0 or len(array) != len(counts):
            raise ValueError(f"cannot align non-tensor rollout field {key!r} with step outputs")
        expanded[key] = np.repeat(array, counts, axis=0)
    return flattened, expanded


def _restore_expanded_input_fields(
    result: Any,
    expanded: dict[str, Any] | None,
) -> None:
    """Restore source fields that verl omits when its reward loop is active."""
    if expanded is None:
        return
    output_size = len(result)
    for key, values in expanded.items():
        array = np.asarray(values)
        if array.ndim == 0 or len(array) != output_size:
            raise ValueError(
                f"expanded non-tensor rollout field {key!r} has "
                f"{0 if array.ndim == 0 else len(array)} rows, expected {output_size}"
            )
        # Input fields are authoritative for source identity. In particular,
        # priority maps every action row back to the repeated prompt batch.
        result.non_tensor_batch[key] = array


def _offload_multi_modal_inputs(result: Any) -> None:
    """Keep large vision tensors in Ray's object store instead of the trainer."""
    multi_modal_inputs = result.non_tensor_batch.get("multi_modal_inputs")
    if multi_modal_inputs is None:
        return

    refs = np.empty(len(multi_modal_inputs), dtype=object)
    image_seqlens = np.empty(len(multi_modal_inputs), dtype=object)
    for index, inputs in enumerate(multi_modal_inputs):
        if inputs is None:
            refs[index] = None
            image_seqlens[index] = None
            continue
        lengths = inputs.get("images_seqlens")
        image_seqlens[index] = (
            lengths.detach().cpu().numpy() if isinstance(lengths, torch.Tensor) else lengths
        )
        refs[index] = ray.put(inputs)

    result.non_tensor_batch["multi_modal_inputs"] = refs
    result.non_tensor_batch["multi_modal_images_seqlens"] = image_seqlens


class GiGPOAgentLoopOutput(AgentLoopOutput):
    """AgentLoopOutput fields consumed by the explicit verl GiGPO patch."""

    episode_uid: int
    anchor_uid: int
    anchor_detail_uid: int
    step_returns: float

    def as_dict(self) -> dict[str, Any]:
        import torch

        value = super().as_dict()
        value["episode_uid"] = torch.tensor(self.episode_uid, dtype=torch.int64)
        value["anchor_uid"] = torch.tensor(self.anchor_uid, dtype=torch.int64)
        value["anchor_detail_uid"] = torch.tensor(self.anchor_detail_uid, dtype=torch.int64)
        value["step_returns"] = torch.tensor(self.step_returns, dtype=torch.float32)
        return value


class ADMIREAgentLoopOutput(AgentLoopOutput):
    """AgentLoopOutput carrying the composed ADMIRE step reward."""

    admire_total_reward: float

    def as_dict(self) -> dict[str, Any]:
        value = super().as_dict()
        value["admire_total_reward"] = torch.tensor(
            self.admire_total_reward,
            dtype=torch.float32,
        )
        return value


class GUIAgentLoopWorker(AgentLoopWorker):
    """Flatten per-action samples while retaining verl's standard postprocessing."""

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ):
        if agent_name != "mobileworld_agent":
            return await super()._run_agent_loop(
                sampling_params,
                trajectory,
                agent_name=agent_name,
                trace=trace,
                **kwargs,
            )

        # The upstream worker owns the teacher client, while the concrete agent
        # loop owns environment stepping. Instantiate the MobileWorld loop here
        # so it can submit teacher work immediately after each student action,
        # instead of waiting for the completed episode to reach postprocessing.
        from guiagentlab.rollout.agent import MobileWorldAgentLoop

        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
            trace=trace,
        ):
            agent_loop = MobileWorldAgentLoop(
                trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.llm_client,
                tokenizer=self.tokenizer,
                processor=self.processor,
                dataset_cls=self.dataset_cls,
                data_config=DictConfigWrap(self.config.data),
                tools=ToolListWrap(self.tools),
                teacher_server_manager=(
                    self.teacher_server_manager
                    if self.distillation_enabled and not trajectory["validate"]
                    else None
                ),
                teacher_key=(self.teacher_key if self.distillation_enabled else None),
            )
            output = await agent_loop.run(
                sampling_params,
                training_step=int(trajectory["step"]),
                **kwargs,
            )
            return await self._agent_loop_postprocess(
                output,
                trajectory["validate"],
                **kwargs,
            )

    async def _compute_teacher_logprobs(
        self,
        output,
        prompt_ids,
        response_ids,
        validate,
        sample_kwargs=None,
    ) -> None:
        teacher_future = output.extra_fields.pop("_guiagentlab_teacher_future", None)
        if teacher_future is not None:
            if validate:
                teacher_future.cancel()
                await asyncio.gather(teacher_future, return_exceptions=True)
                return
            teacher_ids, teacher_logprobs = await teacher_future
            output.extra_fields["teacher_ids"] = teacher_ids
            output.extra_fields["teacher_logprobs"] = teacher_logprobs
            return
        await super()._compute_teacher_logprobs(
            output,
            prompt_ids,
            response_ids,
            validate,
            sample_kwargs=sample_kwargs,
        )

    async def _agent_loop_postprocess(self, output, validate, **kwargs):
        if isinstance(output, list):
            return [
                await super()._agent_loop_postprocess(item, validate, **kwargs) for item in output
            ]
        return await super()._agent_loop_postprocess(output, validate, **kwargs)

    def _postprocess(
        self,
        inputs,
        input_non_tensor_batch: dict | None = None,
        validate: bool = False,
    ):
        flattened, expanded = _flatten_step_outputs(inputs, input_non_tensor_batch)
        result = super()._postprocess(flattened, expanded, validate)
        _restore_expanded_input_fields(result, expanded)
        _offload_multi_modal_inputs(result)
        step_returns = result.non_tensor_batch.pop("step_returns", None)
        if step_returns is not None:
            if any(value is None for value in step_returns):
                raise ValueError("step_returns must be present for every emitted GiGPO step")
            result.batch["step_returns"] = torch.as_tensor(
                np.asarray(step_returns, dtype=np.float32)
            )
        admire_total_rewards = result.non_tensor_batch.pop(
            "admire_total_reward",
            None,
        )
        if admire_total_rewards is not None:
            if any(value is None for value in admire_total_rewards):
                raise ValueError(
                    "admire_total_reward must be present for every emitted ADMIRE step"
                )
            result.batch["admire_total_reward"] = torch.as_tensor(
                np.asarray(admire_total_rewards, dtype=np.float32)
            )
        return result

    async def materialize_scored_replay(
        self,
        *,
        library_path: str,
        scores_path: str,
        task_name: str,
        source_kwargs: dict[str, Any],
        replaced_trajectory_uid: str,
    ):
        """Build current-tokenizer action rows without querying the policy or PRM."""
        cache_key = (library_path, scores_path)
        if getattr(self, "_guiagentlab_replay_cache_key", None) != cache_key:
            self._guiagentlab_replay_library = ScoredReplayLibrary(
                library_path,
                scores_path,
            )
            self._guiagentlab_replay_cache_key = cache_key

        from guiagentlab.rollout.agent import MobileWorldAgentLoop

        agent_loop = MobileWorldAgentLoop(
            trainer_config=DictConfigWrap(config=self.config),
            server_manager=self.llm_client,
            tokenizer=self.tokenizer,
            processor=self.processor,
            dataset_cls=self.dataset_cls,
            data_config=DictConfigWrap(self.config.data),
            tools=ToolListWrap(self.tools),
            teacher_server_manager=None,
            teacher_key=None,
        )
        outputs = await agent_loop.materialize_scored_replay(
            self._guiagentlab_replay_library,
            task_name=task_name,
            source_kwargs=source_kwargs,
            replaced_trajectory_uid=replaced_trajectory_uid,
        )
        processed = [
            await self._agent_loop_postprocess(
                output,
                False,
                **source_kwargs,
            )
            for output in outputs
        ]
        expanded: dict[str, np.ndarray] = {}
        for key, value in source_kwargs.items():
            values = np.empty(len(processed), dtype=object)
            values[:] = [value] * len(processed)
            expanded[key] = values
        return self._postprocess(processed, expanded, False)

    async def materialize_admire_replay(
        self,
        *,
        library_path: str,
        milestones_path: str,
        task_name: str,
        source_kwargs: dict[str, Any],
        replaced_trajectory_uid: str,
    ):
        """Build current-tokenizer ADMIRE rows from cached evaluator traces."""
        cache_key = (library_path, milestones_path)
        if getattr(self, "_guiagentlab_admire_replay_cache_key", None) != cache_key:
            self._guiagentlab_admire_replay_library = MilestoneReplayLibrary(
                library_path,
                milestones_path,
            )
            self._guiagentlab_admire_replay_cache_key = cache_key

        from guiagentlab.rollout.agent import MobileWorldAgentLoop

        agent_loop = MobileWorldAgentLoop(
            trainer_config=DictConfigWrap(config=self.config),
            server_manager=self.llm_client,
            tokenizer=self.tokenizer,
            processor=self.processor,
            dataset_cls=self.dataset_cls,
            data_config=DictConfigWrap(self.config.data),
            tools=ToolListWrap(self.tools),
            teacher_server_manager=None,
            teacher_key=None,
        )
        outputs = await agent_loop.materialize_admire_replay(
            self._guiagentlab_admire_replay_library,
            task_name=task_name,
            source_kwargs=source_kwargs,
            replaced_trajectory_uid=replaced_trajectory_uid,
        )
        processed = [
            await self._agent_loop_postprocess(
                output,
                False,
                **source_kwargs,
            )
            for output in outputs
        ]
        expanded: dict[str, np.ndarray] = {}
        for key, value in source_kwargs.items():
            values = np.empty(len(processed), dtype=object)
            values[:] = [value] * len(processed)
            expanded[key] = values
        return self._postprocess(processed, expanded, False)


class AsyncProcessRewardPool:
    """One cluster-wide, bounded async client for external process rewards."""

    def __init__(self, config: ProcessRewardConfig, max_concurrency: int = 100) -> None:
        if max_concurrency <= 0 or max_concurrency > 100:
            raise ValueError("PRM max_concurrency must be between 1 and 100")
        self.max_concurrency = int(max_concurrency)
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self._judge = AsyncProcessRewardJudge(config)

    async def score(self, **kwargs: Any) -> dict[str, Any]:
        queued_at = time.monotonic()
        async with self._semaphore:
            started_at = time.monotonic()
            result = await self._judge.score(**kwargs)
            finished_at = time.monotonic()
        return {
            "reward": float(result.reward),
            "reason": result.reason,
            "source": result.source,
            "valid": bool(result.valid),
            "queue_seconds": started_at - queued_at,
            "request_seconds": finished_at - started_at,
        }

    async def close(self) -> None:
        await self._judge.close()


def _read_endpoints(path: str) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def _decode_screenshot(png: bytes, endpoint: str) -> Image.Image:
    """Decode a screenshot without allowing one corrupt response to kill Ray."""
    try:
        return decode_screenshot(png)
    except (OSError, ValueError) as exc:
        raise InfrastructureError(
            FailureKind.PROTOCOL,
            "screenshot payload is not a decodable PNG",
            endpoint,
            details={"payload_bytes": len(png)},
        ) from exc


class GUIAgentLoopManager(AgentLoopManager):
    """Creates exactly one named leasing actor before rollout workers start."""

    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = ray.remote(GUIAgentLoopWorker)
        super().__init__(*args, **kwargs)
        settings = self.config.agent_environment
        self.success_replay_enabled = bool(settings.get("success_replay_enabled", False))
        self.success_replay_library_path = str(settings.get("success_replay_library_path", ""))
        self.success_replay_scores_path = str(
            settings.get("success_replay_scores_path", "")
        )
        self.success_replay_milestones_path = str(
            settings.get("success_replay_milestones_path", "")
        )
        self.success_replay_method = str(
            self.config.algorithm.adv_estimator
        ).lower()
        self._success_replay_worker_cursor = 0

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        instance = cls(*args, **kwargs)
        settings = instance.config.agent_environment
        pool_name = str(settings.pool_name)
        try:
            pool_actor = ray.get_actor(pool_name)
        except ValueError:
            endpoints = _read_endpoints(str(settings.servers_file))
            recovery_config = RecoveryConfig.from_environment()
            actor_class = ray.remote(max_concurrency=2048)(AsyncContainerPool)
            pool_actor = actor_class.options(name=pool_name).remote(
                endpoints,
                int(settings.active),
                int(settings.spares),
                int(settings.get("initialization_concurrency", 4)),
                int(settings.get("teardown_concurrency", 1)),
                int(settings.get("recovery_concurrency", 2)),
                (recovery_config.serializable() if recovery_config is not None else None),
            )
            await pool_actor.start.remote()
        # A named actor without a retained handle follows its owner lifetime and
        # may be collected before agent-loop workers resolve it by name.
        instance._guiagentlab_pool_actor = pool_actor
        if str(instance.config.algorithm.adv_estimator).lower() == "gigpo":
            prm_pool_name = f"{pool_name}_prm"
            prm_max_concurrency = int(settings.get("prm_max_concurrency", 100))
            try:
                prm_pool_actor = ray.get_actor(prm_pool_name)
            except ValueError:
                prm_actor_class = ray.remote(max_concurrency=100)(AsyncProcessRewardPool)
                prm_pool_actor = prm_actor_class.options(name=prm_pool_name).remote(
                    ProcessRewardConfig.from_environment(),
                    prm_max_concurrency,
                )
            instance._guiagentlab_prm_pool_actor = prm_pool_actor
        if instance.success_replay_enabled:
            advantage_estimator = str(
                instance.config.algorithm.adv_estimator
            ).lower()
            if advantage_estimator not in {"gigpo", "admire_grpo"}:
                raise ValueError(
                    "Success replay is supported only with GiGPO or ADMIRE-GRPO"
                )
            rollout_correction = instance.config.algorithm.get(
                "rollout_correction"
            )
            correction_enabled = bool(
                rollout_correction
                and (
                    rollout_correction.get("bypass_mode", False)
                    or rollout_correction.get("rollout_is") is not None
                    or rollout_correction.get("rollout_rs") is not None
                )
            )
            if correction_enabled:
                raise ValueError(
                    "Success replay cannot use rollout correction without source-policy "
                    "log probabilities"
                )
            if not instance.success_replay_library_path:
                raise ValueError("Success replay requires a replay-library path")
            if advantage_estimator == "gigpo":
                if not instance.success_replay_scores_path:
                    raise ValueError("GiGPO success replay requires process rewards")
                instance._guiagentlab_replay_library = await asyncio.to_thread(
                    ScoredReplayLibrary,
                    instance.success_replay_library_path,
                    instance.success_replay_scores_path,
                )
            else:
                if not instance.success_replay_milestones_path:
                    raise ValueError("ADMIRE success replay requires milestones")
                instance._guiagentlab_replay_library = await asyncio.to_thread(
                    MilestoneReplayLibrary,
                    instance.success_replay_library_path,
                    instance.success_replay_milestones_path,
                )
        await instance._init_agent_loop_workers()
        return instance

    @auto_await
    async def generate_sequences(self, prompts):
        output = await super().generate_sequences(prompts)
        if not self.success_replay_enabled or bool(prompts.meta_info.get("validate", False)):
            return output
        return await self._replace_all_failure_groups(output, prompts)

    async def _replace_all_failure_groups(self, output, prompts):
        selection = select_all_failure_replacements(
            output.non_tensor_batch,
            group_size=int(self.rollout_config.n),
            eligible_tasks=self._guiagentlab_replay_library.tasks,
        )
        prompt_priorities = np.asarray(
            prompts.non_tensor_batch["priority"], dtype=np.int64
        )
        prompt_by_priority = {
            int(priority): index for index, priority in enumerate(prompt_priorities)
        }

        requests: list[dict[str, Any]] = []
        target_rows: list[np.ndarray] = []
        for replacement in selection.replacements:
            source_index = prompt_by_priority[replacement.priority]
            source_kwargs = {
                key: values[source_index]
                for key, values in prompts.non_tensor_batch.items()
            }
            worker = self.agent_loop_workers[
                self._success_replay_worker_cursor % len(self.agent_loop_workers)
            ]
            self._success_replay_worker_cursor += 1
            requests.append(
                {
                    "worker": worker,
                    "library_path": self.success_replay_library_path,
                    "scores_path": self.success_replay_scores_path,
                    "milestones_path": getattr(
                        self,
                        "success_replay_milestones_path",
                        "",
                    ),
                    "task_name": replacement.task_name,
                    "source_kwargs": source_kwargs,
                    "replaced_trajectory_uid": replacement.trajectory_uid,
                    "method": getattr(
                        self,
                        "success_replay_method",
                        "gigpo",
                    ),
                }
            )
            target_rows.append(replacement.rows)

        if not requests:
            output.meta_info.setdefault("timing", {}).update(
                {
                    "success_replay/all_failure_groups": selection.all_failure_groups,
                    "success_replay/eligible_all_failure_groups": 0,
                    "success_replay/ineligible_source_groups": (
                        selection.ineligible_source_groups
                    ),
                    "success_replay/eligible_library_tasks": len(
                        self._guiagentlab_replay_library.tasks
                    ),
                    "success_replay/replaced_trajectories": 0,
                    "success_replay/replay_action_rows": 0,
                }
            )
            return output

        replay_calls = []
        for request in requests:
            common = {
                "library_path": request["library_path"],
                "task_name": request["task_name"],
                "source_kwargs": request["source_kwargs"],
                "replaced_trajectory_uid": request["replaced_trajectory_uid"],
            }
            if request["method"] == "admire_grpo":
                replay_calls.append(
                    request["worker"].materialize_admire_replay.remote(
                        milestones_path=request["milestones_path"],
                        **common,
                    )
                )
            else:
                replay_calls.append(
                    request["worker"].materialize_scored_replay.remote(
                        scores_path=request["scores_path"],
                        **common,
                    )
                )
        replay_batches = await asyncio.gather(*replay_calls)
        remove = np.concatenate(target_rows)
        keep = np.ones(len(output), dtype=bool)
        keep[remove] = False
        kept = output.select_idxs(keep)

        # Offline actions were not sampled by the rollout engine, so they have
        # no honest rollout-policy log probability. The trainer recomputes
        # ``old_log_probs`` for every row before PPO; omit this optional debug
        # tensor for the whole mixed batch instead of fabricating zeros or
        # reusing probabilities from the source policy.
        rollout_log_probs_omitted = False
        if "rollout_log_probs" in kept.batch and all(
            "rollout_log_probs" not in replay.batch for replay in replay_batches
        ):
            kept.batch.pop("rollout_log_probs")
            rollout_log_probs_omitted = True

        expected_tensor_keys = set(kept.batch.keys())
        expected_non_tensor_keys = set(kept.non_tensor_batch)
        for replay in replay_batches:
            if set(replay.batch.keys()) != expected_tensor_keys:
                raise ValueError(
                    "Success replay tensor fields differ from online rollout: "
                    f"replay_only={sorted(set(replay.batch.keys()) - expected_tensor_keys)}, "
                    f"online_only={sorted(expected_tensor_keys - set(replay.batch.keys()))}"
                )
            missing_non_tensor = expected_non_tensor_keys.difference(
                replay.non_tensor_batch
            )
            replay_only = set(replay.non_tensor_batch).difference(
                expected_non_tensor_keys
            )
            if missing_non_tensor or replay_only:
                raise ValueError(
                    "Success replay metadata differs from online rollout: "
                    f"missing={sorted(missing_non_tensor)}, replay_only={sorted(replay_only)}"
                )

        original_meta = output.meta_info
        replaced = type(output).concat([kept, *replay_batches])
        replaced.meta_info = original_meta
        replaced.meta_info.setdefault("timing", {}).update(
            {
                "success_replay/all_failure_groups": selection.all_failure_groups,
                "success_replay/eligible_all_failure_groups": len(requests),
                "success_replay/ineligible_source_groups": selection.ineligible_source_groups,
                "success_replay/eligible_library_tasks": len(
                    self._guiagentlab_replay_library.tasks
                ),
                "success_replay/replaced_trajectories": len(requests),
                "success_replay/replay_action_rows": sum(len(batch) for batch in replay_batches),
                "success_replay/rollout_log_probs_omitted": int(
                    rollout_log_probs_omitted
                ),
            }
        )
        logger.info(
            "Success replay replaced %d all-failure trajectories with %d cached action rows",
            len(requests),
            sum(len(batch) for batch in replay_batches),
        )
        return replaced
