"""Dynamic MobileWorld evaluation through OpenAI-compatible model servers."""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from PIL import Image

from guiagentlab.data.datasets import SYSTEM_PROMPT
from guiagentlab.env.client import MobileWorldClient
from guiagentlab.env.errors import FailureKind, InfrastructureError
from guiagentlab.env.pool import ContainerPool, Lease
from guiagentlab.env.recovery import EndpointRecovery, RecoveryConfig
from guiagentlab.rollout.action import (
    action_requires_environment_step,
    is_terminal_action,
    parse_action_or_wait,
)
from guiagentlab.rollout.maiui import (
    build_maiui_history,
    canonical_prompt_prefix,
    decode_screenshot,
    to_openai_messages,
)
from guiagentlab.rollout.recording import EpisodeRecorder, json_value, utc_now

logger = logging.getLogger(__name__)


class ModelBackendError(RuntimeError):
    """The model endpoint failed; the trajectory must not enter the denominator."""


class ModelOutputError(ValueError):
    """The endpoint responded, but the response was not a valid MAI-UI action."""


@dataclass(frozen=True, slots=True)
class EvaluationJob:
    task_name: str
    goal: str
    task_index: int
    sample_index: int


def _agent_class():
    """Build the MAI-UI evaluator with the same GUI prompt used in training."""
    try:
        from loguru import logger as mobileworld_logger

        from mobile_world.agents.implementations.mai_ui_agent import (
            MAIUINaivigationAgent,
        )
        from mobile_world.runtime.utils.models import JSONAction
    except ImportError as exc:  # pragma: no cover - exercised by shell preflight
        raise RuntimeError(
            "MobileWorld sources are unavailable; add mobile_world/src to PYTHONPATH"
        ) from exc

    mobileworld_logger.remove()
    mobileworld_logger.add(
        sys.stderr,
        level=os.environ.get("GUIAGENTLAB_MOBILEWORLD_LOG_LEVEL", "WARNING"),
    )

    class GUIOnlyMAIUIAgent(MAIUINaivigationAgent):
        @property
        def system_prompt(self) -> str:
            return SYSTEM_PROMPT

        def _build_messages(
            self, obs_image: Any, tool_call: Any, ask_user_response: Any
        ) -> list[dict[str, Any]]:
            del obs_image
            if tool_call is not None or ask_user_response is not None:
                raise ValueError("GUI-only evaluation does not accept tool/user observations")
            screenshots = [entry[0] for entry in self.history_images]
            responses = [str(item.get("content", "")) for item in self.history_responses]
            prefix = canonical_prompt_prefix(None, self.instruction, SYSTEM_PROMPT)
            messages, images = build_maiui_history(
                prefix, screenshots, responses, self.history_n
            )
            return to_openai_messages(messages, images)

        def openai_chat_completions_create(self, *args: Any, **kwargs: Any) -> str | None:
            # The historical Greedy path did not send top_k; it is irrelevant
            # when temperature=0.  Sampling must really use the requested
            # training value instead of silently inheriting generation_config.
            if float(self.temperature) > 0:
                extra_body = dict(kwargs.pop("extra_body", {}) or {})
                extra_body["top_k"] = int(self.top_k)
                kwargs["extra_body"] = extra_body
            return super().openai_chat_completions_create(*args, **kwargs)

        def predict_canonical(
            self, observation: dict[str, Any]
        ) -> tuple[str, Any, bool, str | None]:
            """Generate text, then use the same projector as the VERL loop."""
            obs_image = observation["screenshot"]
            tool_call = observation.get("tool_call")
            ask_user_response = observation.get("ask_user_response")
            self.history_images.append((obs_image, tool_call, ask_user_response))
            if len(self.history_images) != len(self.history_responses) + 1:
                raise RuntimeError("MAI-UI screenshot/action history is inconsistent")
            messages = self._build_messages(obs_image, tool_call, ask_user_response)
            prediction = self.openai_chat_completions_create(
                model=self.model_name,
                messages=messages,
                retry_times=3,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            if prediction is None:
                raise ValueError("Planner LLM failed")
            # Preserve every raw action in history, including an invalid one
            # that executes as WAIT. This exactly matches the VERL loop.
            self.history_responses.append({"role": "assistant", "content": prediction})
            action, valid, warning = parse_action_or_wait(prediction, obs_image.size)
            return prediction, JSONAction(**action), valid, warning

    return GUIOnlyMAIUIAgent


def _decode_png(png: bytes, endpoint: str) -> Image.Image:
    try:
        return decode_screenshot(png)
    except (OSError, ValueError) as exc:
        raise InfrastructureError(
            FailureKind.PROTOCOL,
            "screenshot payload is not a decodable PNG",
            endpoint,
            details={"payload_bytes": len(png)},
        ) from exc


def _action_dict(action: Any) -> dict[str, Any]:
    value = action.model_dump(exclude_none=True)
    if not isinstance(value, dict) or not isinstance(value.get("action_type"), str):
        raise ModelOutputError("MAI-UI returned no executable action")
    return value


def _error_payload(exc: BaseException, endpoint: str | None = None) -> dict[str, Any]:
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
        "kind": "model_backend" if isinstance(exc, ModelBackendError) else "internal",
        "message": str(exc),
        "endpoint": endpoint,
    }


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_value(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_model_endpoints(path: str | Path) -> list[str]:
    endpoints = [
        line.strip().rstrip("/")
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not endpoints or len(set(endpoints)) != len(endpoints):
        raise ValueError("model endpoint list must be nonempty and unique")
    return endpoints


def _check_model_endpoint(endpoint: str, model_name: str) -> None:
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.get(f"{endpoint}/models", timeout=10)
        response.raise_for_status()
        ids = {str(item["id"]) for item in response.json().get("data", [])}
    except (requests.RequestException, ValueError, KeyError) as exc:
        raise ModelBackendError(f"model endpoint is not ready: {endpoint}: {exc}") from exc
    if model_name not in ids:
        raise ModelBackendError(
            f"model endpoint {endpoint} does not serve {model_name!r}; available={sorted(ids)}"
        )


class OpenAIEvaluator:
    def __init__(
        self,
        *,
        environment_servers: str | Path,
        model_servers: str | Path,
        output_dir: str | Path,
        model_name: str,
        model_path: str,
        active: int,
        spares: int,
        workers: int,
        max_steps: int,
        max_attempts: int,
        exhausted_task_policy: str,
        request_timeout: float,
        acquire_timeout: float,
        step_wait_time: float,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        history_n: int,
        mode: str,
        experiment_name: str,
    ) -> None:
        recovery_config = RecoveryConfig.from_environment()
        self.pool = ContainerPool.from_file(
            environment_servers,
            active=active,
            spares=spares,
            recovery=(
                EndpointRecovery(recovery_config)
                if recovery_config is not None
                else None
            ),
        )
        self.model_endpoints = _read_model_endpoints(model_servers)
        self.output_dir = Path(output_dir).resolve()
        self.recorder = EpisodeRecorder(self.output_dir)
        self.model_name = model_name
        self.model_path = model_path
        self.workers = workers
        self.max_steps = max_steps
        self.max_attempts = max_attempts
        if exhausted_task_policy not in {"abort", "record_failure"}:
            raise ValueError(
                "exhausted_task_policy must be 'abort' or 'record_failure'"
            )
        self.exhausted_task_policy = exhausted_task_policy
        self.request_timeout = request_timeout
        self.acquire_timeout = acquire_timeout
        self.step_wait_time = step_wait_time
        self.runtime_conf = {
            "history_n": history_n,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        self.mode = mode
        self.experiment_name = experiment_name
        self.progress_path = self.output_dir.parent / "progress.json"
        self._progress_lock = threading.Lock()
        self._abort_event = threading.Event()
        self._fatal_errors: list[BaseException] = []
        self._agent_type = _agent_class()

    def preflight(self) -> None:
        for endpoint in self.model_endpoints:
            _check_model_endpoint(endpoint, self.model_name)

    def run(self, jobs: list[EvaluationJob]) -> dict[str, int]:
        completed = self._completed_jobs()
        pending = [job for job in jobs if (job.task_name, job.sample_index) not in completed]
        work: queue.Queue[EvaluationJob] = queue.Queue()
        for job in pending:
            work.put(job)
        counters = {
            "total": len(jobs),
            "completed": len(jobs) - len(pending),
            "successful": sum(completed.values()),
            "failed": len(completed) - sum(completed.values()),
            "errors": 0,
            "running": 0,
        }
        started = time.monotonic()
        self._write_progress(counters, started)

        def worker(slot: int) -> None:
            while not self._abort_event.is_set():
                try:
                    job = work.get_nowait()
                except queue.Empty:
                    return
                with self._progress_lock:
                    counters["running"] += 1
                self._write_progress(counters, started)
                try:
                    score = self._run_job(job, slot)
                except BaseException as exc:
                    self._abort_event.set()
                    with self._progress_lock:
                        self._fatal_errors.append(exc)
                    logger.exception(
                        "evaluation job failed task=%s sample=%d",
                        job.task_name,
                        job.sample_index,
                    )
                    with self._progress_lock:
                        counters["errors"] += 1
                else:
                    with self._progress_lock:
                        counters["successful" if score > 0 else "failed"] += 1
                finally:
                    with self._progress_lock:
                        counters["running"] -= 1
                        counters["completed"] += 1
                    self._write_progress(counters, started)
                    work.task_done()

        thread_count = min(self.workers, len(pending))
        threads = [
            threading.Thread(target=worker, args=(slot,), name=f"eval-{slot:02d}")
            for slot in range(thread_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self._write_progress(counters, started)
        if self._fatal_errors:
            raise RuntimeError(
                "evaluation aborted after a fatal model/infrastructure error; "
                "partial results must not be summarized"
            ) from self._fatal_errors[0]
        return counters

    def _run_job(self, job: EvaluationJob, slot: int) -> float:
        episode_id = self.recorder.new_episode_id()
        episode_started_at = utc_now()
        episode_started = time.monotonic()
        infrastructure_attempts: list[str] = []
        last_error: BaseException | None = None
        last_attempt_record: dict[str, Any] = {}
        last_model_endpoint: str | None = None
        goal = job.goal

        for attempt in range(1, self.max_attempts + 1):
            lease: Lease | None = None
            client: MobileWorldClient | None = None
            initialized = False
            cleanup_succeeded = True
            model_endpoint = self.model_endpoints[(slot + attempt - 1) % len(self.model_endpoints)]
            last_model_endpoint = model_endpoint
            attempt_record: dict[str, Any] = {
                "episode_id": episode_id,
                "attempt": attempt,
                "task_name": job.task_name,
                "sample_index": job.sample_index,
                "model_endpoint": model_endpoint,
                "started_at": utc_now(),
                "steps": [],
                "stage": "acquire",
            }
            last_attempt_record = attempt_record
            try:
                lease = self.pool.acquire(timeout=self.acquire_timeout)
                attempt_record["environment_endpoint"] = lease.endpoint
                attempt_record["container_role"] = lease.role.value
                client = MobileWorldClient(
                    lease.endpoint,
                    timeout=self.request_timeout,
                    step_wait_time=self.step_wait_time,
                )
                attempt_record["stage"] = "health"
                client.health()
                attempt_record["stage"] = "controller_initialization"
                client.initialize_controller()
                goal = goal or client.goal(job.task_name)
                attempt_record["goal"] = goal
                attempt_record["stage"] = "task_initialization"
                client.initialize_task(job.task_name)
                initialized = True
                attempt_record["stage"] = "initial_screenshot"
                screenshot = client.screenshot_png()
                initial_ref = self.recorder.write_screenshot(
                    episode_id, attempt, 0, screenshot
                )
                attempt_record["initial_screenshot"] = initial_ref
                score, reason, termination = self._run_attempt(
                    client,
                    model_endpoint,
                    job,
                    goal,
                    screenshot,
                    episode_id,
                    attempt,
                    attempt_record,
                )
                attempt_record["stage"] = "teardown"
                cleanup_error = None
                container_disposition = "released"
                try:
                    client.tear_down(job.task_name)
                    initialized = False
                    self.pool.release(lease)
                except (InfrastructureError, TimeoutError) as cleanup_exc:
                    # The score was already computed.  Preserve the valid
                    # episode and replace the dirty environment asynchronously.
                    cleanup_error = _error_payload(cleanup_exc)
                    initialized = False
                    self.pool.quarantine(lease, cleanup_exc)
                    container_disposition = "quarantined_recovery_started"
                lease = None
                payload = {
                    "episode_id": episode_id,
                    "experiment_name": self.experiment_name,
                    "model_path": self.model_path,
                    "model_name": self.model_name,
                    "model_endpoint": model_endpoint,
                    "evaluation_mode": self.mode,
                    "task_name": job.task_name,
                    "task_index": job.task_index,
                    "sample_index": job.sample_index,
                    "goal": goal,
                    "sampling_params": self.runtime_conf,
                    "inference_transport": "openai_chat_completions",
                    "started_at": episode_started_at,
                    "finished_at": utc_now(),
                    "duration_seconds": time.monotonic() - episode_started,
                    "endpoint": attempt_record["environment_endpoint"],
                    "container_role": attempt_record.get("container_role"),
                    "successful_attempt": attempt,
                    "infrastructure_attempts": infrastructure_attempts,
                    "cleanup_error": cleanup_error,
                    "container_disposition": container_disposition,
                    "valid": True,
                    "step_count": len(attempt_record["steps"]),
                    "termination": termination,
                    "final_score": score,
                    "evaluation_reason": reason,
                    "model_errors": attempt_record.get("model_errors", []),
                    "initial_screenshot": initial_ref,
                    "steps": attempt_record["steps"],
                }
                self.recorder.write_episode(episode_id, payload)
                return score
            except (InfrastructureError, ModelBackendError, TimeoutError) as exc:
                last_error = exc
                cleanup_error = None
                if (
                    initialized
                    and client is not None
                    and attempt_record.get("stage") != "teardown"
                ):
                    try:
                        client.tear_down(job.task_name)
                        initialized = False
                    except InfrastructureError as cleanup_exc:
                        cleanup_succeeded = False
                        cleanup_error = _error_payload(cleanup_exc)
                if lease is not None:
                    if isinstance(exc, ModelBackendError) and cleanup_succeeded:
                        self.pool.release(lease)
                    else:
                        self.pool.quarantine(lease, exc)
                attempt_record.update(
                    {
                        "finished_at": utc_now(),
                        "valid": False,
                        "termination": "infrastructure_failure",
                        "error": _error_payload(exc, model_endpoint),
                        "cleanup_error": cleanup_error,
                    }
                )
                relative = f"infrastructure/{episode_id}-attempt-{attempt:02d}.json"
                infrastructure_attempts.append(relative)
                self.recorder.write_infrastructure_attempt(
                    episode_id, attempt, attempt_record
                )
                logger.warning(
                    "discarded attempt task=%s sample=%d attempt=%d: %s",
                    job.task_name,
                    job.sample_index,
                    attempt,
                    exc,
                )
            except BaseException as exc:
                if lease is not None:
                    self.pool.quarantine(lease, exc)
                attempt_record.update(
                    {
                        "finished_at": utc_now(),
                        "valid": False,
                        "termination": "internal_error",
                        "error": _error_payload(exc, model_endpoint),
                    }
                )
                self.recorder.write_internal_error(episode_id, attempt, attempt_record)
                raise
        if self.exhausted_task_policy == "record_failure":
            assert last_error is not None
            failure = _error_payload(last_error, last_model_endpoint)
            reason = (
                f"skipped after {self.max_attempts} failed attempts: "
                f"{failure['kind']}: {failure['message']}"
            )
            payload = {
                "episode_id": episode_id,
                "experiment_name": self.experiment_name,
                "model_path": self.model_path,
                "model_name": self.model_name,
                "model_endpoint": last_model_endpoint,
                "evaluation_mode": self.mode,
                "task_name": job.task_name,
                "task_index": job.task_index,
                "sample_index": job.sample_index,
                "goal": goal,
                "sampling_params": self.runtime_conf,
                "inference_transport": "openai_chat_completions",
                "started_at": episode_started_at,
                "finished_at": utc_now(),
                "duration_seconds": time.monotonic() - episode_started,
                "endpoint": last_attempt_record.get("environment_endpoint"),
                "container_role": last_attempt_record.get("container_role"),
                "successful_attempt": None,
                "attempts_exhausted": self.max_attempts,
                "infrastructure_attempts": infrastructure_attempts,
                "cleanup_error": last_attempt_record.get("cleanup_error"),
                "container_disposition": "skipped_after_retry",
                "valid": True,
                "result_status": "skipped_after_retry",
                "skipped_after_retry": True,
                "skip_reason": failure,
                "step_count": 0,
                "termination": "skipped_after_retry",
                "final_score": 0.0,
                "evaluation_reason": reason,
                "model_errors": [
                    {
                        "type": "AttemptsExhausted",
                        "message": reason,
                        "cause": failure,
                    }
                ],
                "steps": [],
            }
            self.recorder.write_episode(episode_id, payload)
            logger.error(
                "recorded failed task after retry exhaustion task=%s sample=%d: %s",
                job.task_name,
                job.sample_index,
                reason,
            )
            return 0.0
        raise RuntimeError(
            f"evaluation exhausted {self.max_attempts} non-model attempts for "
            f"{job.task_name} sample {job.sample_index}"
        ) from last_error

    def _run_attempt(
        self,
        client: MobileWorldClient,
        model_endpoint: str,
        job: EvaluationJob,
        goal: str,
        screenshot: bytes,
        episode_id: str,
        attempt: int,
        attempt_record: dict[str, Any],
    ) -> tuple[float, str | None, str]:
        agent = self._agent_type(
            llm_base_url=model_endpoint,
            model_name=self.model_name,
            api_key="empty",
            runtime_conf=self.runtime_conf,
            tools=[],
        )
        agent.initialize(goal)
        observation: dict[str, Any] = {
            "screenshot": _decode_png(screenshot, client.endpoint),
            "tool_call": None,
            "ask_user_response": None,
        }
        current_ref = attempt_record["initial_screenshot"]
        termination = "max_steps"
        try:
            for index in range(self.max_steps):
                attempt_record["stage"] = f"generation_step_{index}"
                before_usage = agent.get_total_token_usage().copy()
                try:
                    (
                        prediction,
                        action_model,
                        model_output_valid,
                        parse_warning,
                    ) = agent.predict_canonical(observation)
                except ValueError as exc:
                    if str(exc) == "Planner LLM failed":
                        raise ModelBackendError(
                            f"OpenAI endpoint failed after retries: {model_endpoint}"
                        ) from exc
                    model_error = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                    attempt_record.setdefault("model_errors", []).append(model_error)
                    attempt_record["steps"].append(
                        {
                            "index": index,
                            "screenshot_before": current_ref,
                            "model_text": None,
                            "action": {"action_type": "unknown", "text": str(exc)},
                            "model_output_valid": False,
                            "error": model_error,
                        }
                    )
                    termination = "invalid_model_output"
                    break
                action = _action_dict(action_model)
                after_usage = agent.get_total_token_usage()
                step_record: dict[str, Any] = {
                    "index": index,
                    "screenshot_before": current_ref,
                    "model_text": prediction,
                    "action": action,
                    "model_output_valid": model_output_valid,
                    "prompt_token_count": max(
                        0,
                        int(after_usage["prompt_tokens"])
                        - int(before_usage["prompt_tokens"]),
                    ),
                    "generated_token_count": max(
                        0,
                        int(after_usage["completion_tokens"])
                        - int(before_usage["completion_tokens"]),
                    ),
                }
                attempt_record["steps"].append(step_record)
                if parse_warning:
                    step_record["parse_warning"] = parse_warning
                if not model_output_valid:
                    model_error = {
                        "type": ModelOutputError.__name__,
                        "message": parse_warning,
                        "fallback_action": "wait",
                    }
                    step_record["error"] = model_error
                    attempt_record.setdefault("model_errors", []).append(model_error)

                action_type = action["action_type"]
                if action_type == "ask_user":
                    step_record["rejected_action"] = action
                    action = {"action_type": "wait"}
                    action_type = "wait"
                    step_record["action"] = action
                    step_record["model_output_valid"] = False
                    model_error = {
                        "type": ModelOutputError.__name__,
                        "message": "ask_user is outside the configured GUI-only action space",
                        "fallback_action": "wait",
                    }
                    step_record["error"] = model_error
                    attempt_record.setdefault("model_errors", []).append(model_error)

                if not action_requires_environment_step(action):
                    termination = "model_terminal"
                    break

                attempt_record["stage"] = f"environment_step_{index}"
                client.step(action)

                # The baseline client captured a settled frame after every
                # executed action, including ANSWER, before evaluation.
                attempt_record["stage"] = f"screenshot_step_{index}"
                next_png = client.screenshot_png()
                next_ref = self.recorder.write_screenshot(
                    episode_id, attempt, index + 1, next_png
                )
                step_record["screenshot_after"] = next_ref
                current_ref = next_ref
                if is_terminal_action(action):
                    termination = "model_terminal"
                    break

                observation = {
                    "screenshot": _decode_png(next_png, client.endpoint),
                    "tool_call": None,
                    "ask_user_response": None,
                }

            attempt_record["stage"] = "evaluation"
            evaluation = client.evaluate(job.task_name)
            attempt_record["final_score"] = evaluation.score
            attempt_record["evaluation_reason"] = evaluation.reason
            attempt_record["termination"] = termination
            return evaluation.score, evaluation.reason, termination
        finally:
            agent.done()

    def _completed_jobs(self) -> dict[tuple[str, int], bool]:
        if self.recorder.root is None:
            return {}
        completed: dict[tuple[str, int], bool] = {}
        for path in sorted((self.recorder.root / "episodes").glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("valid") is not True:
                continue
            key = (str(payload["task_name"]), int(payload.get("sample_index", 0)))
            if key in completed:
                raise ValueError(f"duplicate completed trajectory: {key}")
            completed[key] = float(payload.get("final_score", 0)) > 0
        return completed

    def _write_progress(self, counters: dict[str, int], started: float) -> None:
        with self._progress_lock:
            elapsed = time.monotonic() - started
            completed = counters["completed"]
            rate = completed / elapsed if elapsed > 0 else 0.0
            remaining = counters["total"] - completed
            _atomic_json(
                self.progress_path,
                {
                    **counters,
                    "pending": max(0, remaining - counters["running"]),
                    "elapsed_seconds": elapsed,
                    "jobs_per_hour": rate * 3600,
                    "eta_seconds": remaining / rate if rate else None,
                    "updated_at": utc_now(),
                },
            )


def _jobs(dataset: str | Path, samples: int) -> list[EvaluationJob]:
    frame = pd.read_parquet(dataset)
    jobs = []
    seen: set[str] = set()
    for row_index, row in frame.iterrows():
        extra = row["extra_info"]
        task_name = str(extra["task_name"])
        if task_name in seen:
            raise ValueError(f"dataset contains duplicate task: {task_name}")
        seen.add(task_name)
        goal = str(extra.get("goal") or "")
        for sample_index in range(samples):
            jobs.append(
                EvaluationJob(
                    task_name=task_name,
                    goal=goal,
                    task_index=int(extra.get("index", row_index)),
                    sample_index=sample_index,
                )
            )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--environment-servers", required=True)
    parser.add_argument("--model-servers", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--mode", choices=("greedy", "sample"), required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--active", type=int, default=56)
    parser.add_argument("--spares", type=int, default=8)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument(
        "--exhausted-task-policy",
        choices=("abort", "record_failure"),
        default="abort",
    )
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--acquire-timeout", type=float, default=7200)
    parser.add_argument("--step-wait-time", type=float, default=3)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, default=1)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--history-n", type=int, default=3)
    args = parser.parse_args()
    if args.samples <= 0 or args.workers <= 0 or args.max_attempts <= 0:
        parser.error("samples, workers, and max-attempts must be positive")
    jobs = _jobs(args.dataset, args.samples)
    evaluator = OpenAIEvaluator(
        environment_servers=args.environment_servers,
        model_servers=args.model_servers,
        output_dir=args.output_dir,
        model_name=args.model_name,
        model_path=args.model_path,
        active=args.active,
        spares=args.spares,
        workers=args.workers,
        max_steps=args.max_steps,
        max_attempts=args.max_attempts,
        exhausted_task_policy=args.exhausted_task_policy,
        request_timeout=args.request_timeout,
        acquire_timeout=args.acquire_timeout,
        step_wait_time=args.step_wait_time,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        history_n=args.history_n,
        mode=args.mode,
        experiment_name=args.experiment_name,
    )
    evaluator.preflight()
    counters = evaluator.run(jobs)
    print(json.dumps(counters, sort_keys=True))
    if counters["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
