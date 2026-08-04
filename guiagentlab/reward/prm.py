"""OpenAI-compatible process-reward scoring for GUI actions."""

from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from PIL import Image, ImageDraw

_COORDINATE_ACTIONS = {"click", "double_tap", "long_press", "drag", "scroll"}
_MAX_REASON_LENGTH = 1000

PROCESS_REWARD_PROMPT = """Evaluate whether the current mobile GUI action makes
positive progress toward the task.

Task:
{task_goal}

Previous actions, newest first:
{action_history}

Current action:
{current_action}

The first image is the screen before the action. The second image is the screen after the action.
For coordinate actions, the first image marks the interaction location.

Return only this JSON object:
{{"reason": "short explanation", "reward": 0 or 1}}

Use reward 1 only when the action advances a necessary task step. Use reward 0 for ineffective,
repeated, irrelevant, incorrect, or regressive actions.""".strip()


@dataclass(frozen=True, slots=True)
class ProcessRewardConfig:
    """Connection settings for an external process-reward judge."""

    base_url: str = ""
    model: str = ""
    api_key: str = field(default="", repr=False)
    timeout: float = 120.0
    max_retries: int = 1

    @classmethod
    def from_environment(cls) -> ProcessRewardConfig:
        base_url = os.environ.get("GUIAGENTLAB_PRM_BASE_URL", "").strip().rstrip("/")
        model = os.environ.get("GUIAGENTLAB_PRM_MODEL", "").strip()
        api_key = os.environ.get("GUIAGENTLAB_PRM_API_KEY", "").strip()
        config = cls(
            base_url=base_url,
            model=model,
            api_key=api_key,
        )
        config.validate()
        return config

    def validate(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("GUIAGENTLAB_PRM_BASE_URL must be an absolute HTTP(S) URL")
        if not self.model:
            raise ValueError("GUIAGENTLAB_PRM_MODEL must be set")
        if not self.api_key:
            raise ValueError("GUIAGENTLAB_PRM_API_KEY must be set")
        if self.timeout <= 0:
            raise ValueError("process-reward timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("process-reward max_retries must be non-negative")

    def public_metadata(self) -> dict[str, Any]:
        """Return reproducibility metadata without exposing credentials."""
        return {
            "base_url": self.base_url,
            "model": self.model,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }


@dataclass(frozen=True, slots=True)
class ProcessRewardResult:
    reward: float
    reason: str
    source: str
    valid: bool = True


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    """Compute one backward discounted return for every action reward."""
    if not 0 <= float(gamma) <= 1:
        raise ValueError("gamma must be between zero and one")
    output = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + float(gamma) * running
        output[index] = running
    return output


def normalize_intermediate_rewards(rewards: list[float]) -> list[float]:
    """Average intermediate rewards while keeping the terminal reward unchanged."""
    if len(rewards) <= 1:
        return [float(reward) for reward in rewards]
    intermediate_count = len(rewards) - 1
    return [
        *[float(reward) / intermediate_count for reward in rewards[:-1]],
        float(rewards[-1]),
    ]


def _image_data_url(png: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(png).decode('ascii')}"


def _annotate_action(png: bytes, action: dict[str, Any], radius: int = 15) -> bytes:
    action_type = str(action.get("action_type", ""))
    if action_type not in _COORDINATE_ACTIONS:
        return png
    try:
        image = Image.open(io.BytesIO(png))
        image.load()
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)

        def dot(x: Any, y: Any, color: str) -> None:
            x_value, y_value = int(x), int(y)
            draw.ellipse(
                (
                    x_value - radius,
                    y_value - radius,
                    x_value + radius,
                    y_value + radius,
                ),
                fill=color,
                outline=color,
            )

        if action_type in {"click", "double_tap", "long_press"}:
            dot(action["x"], action["y"], "red")
        elif action_type == "drag":
            start = (int(action["start_x"]), int(action["start_y"]))
            end = (int(action["end_x"]), int(action["end_y"]))
            dot(*start, "red")
            dot(*end, "blue")
            draw.line((start, end), fill="red", width=3)
        elif action_type == "scroll" and "x" in action and "y" in action:
            dot(action["x"], action["y"], "red")

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    except (KeyError, OSError, TypeError, ValueError):
        return png


def _canonical_action(action: dict[str, Any]) -> str:
    """Serialize only the parsed GUI action, without policy reasoning."""
    return json.dumps(
        action,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _action_history(previous_actions: list[dict[str, Any]]) -> str:
    selected = previous_actions[-5:]
    if not selected:
        return "No previous actions."
    return "\n".join(
        f"Step {index}: {_canonical_action(action)}"
        for index, action in enumerate(reversed(selected), start=1)
    )


def _parse_process_reward(content: str) -> tuple[float, str]:
    candidates = [content.strip()]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.I | re.S)
    embedded = re.search(r"\{[^{}]*\"reward\"\s*:\s*[^{}]+\}", content, re.S)
    if fenced:
        candidates.append(fenced.group(1))
    if embedded:
        candidates.append(embedded.group(0))

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        try:
            reward = float(payload["reward"])
        except (KeyError, TypeError, ValueError):
            continue
        if reward not in {0.0, 1.0}:
            continue
        reason = str(payload.get("reason", "")).strip()[:_MAX_REASON_LENGTH]
        return reward, reason
    raise ValueError("process-reward response does not contain valid binary JSON")


class ProcessRewardJudge:
    """Score one GUI transition with an external vision-language model."""

    def __init__(self, config: ProcessRewardConfig) -> None:
        config.validate()
        self.config = config

    @staticmethod
    def unscored(reason: str, source: str) -> ProcessRewardResult:
        """Represent an unavailable score without inventing a positive reward."""
        return ProcessRewardResult(
            reward=0.0,
            reason=reason[:_MAX_REASON_LENGTH],
            source=source,
            valid=False,
        )

    def score(
        self,
        *,
        task_goal: str,
        before_png: bytes,
        after_png: bytes,
        parsed_action: dict[str, Any],
        previous_actions: list[dict[str, Any]],
    ) -> ProcessRewardResult:
        messages = _process_reward_messages(
            task_goal=task_goal,
            before_png=before_png,
            after_png=after_png,
            parsed_action=parsed_action,
            previous_actions=previous_actions,
        )
        try:
            content = self._request(messages)
            reward, reason = _parse_process_reward(content)
            return ProcessRewardResult(
                reward=reward,
                reason=reason,
                source="external_model",
            )
        except Exception as exc:
            return self.unscored(
                f"process reward unavailable after {type(exc).__name__}: {exc}",
                "unscored_prm_error",
            )

    def _request(self, messages: list[dict[str, Any]]) -> str:
        from openai import OpenAI

        client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )
        completion = client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=0.1,
        )
        content = completion.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("process-reward response is empty")
        return content.strip()


def _process_reward_messages(
    *,
    task_goal: str,
    before_png: bytes,
    after_png: bytes,
    parsed_action: dict[str, Any],
    previous_actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    annotated_before = _annotate_action(before_png, parsed_action)
    prompt = PROCESS_REWARD_PROMPT.format(
        task_goal=task_goal,
        action_history=_action_history(previous_actions),
        current_action=_canonical_action(parsed_action),
    )
    return [
        {
            "role": "system",
            "content": "You evaluate progress made by mobile GUI actions.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(annotated_before)},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(after_png)},
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]


class AsyncProcessRewardJudge:
    """Reuse one asynchronous client for concurrent GUI-transition scoring."""

    def __init__(self, config: ProcessRewardConfig) -> None:
        from openai import AsyncOpenAI

        config.validate()
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    async def score(
        self,
        *,
        task_goal: str,
        before_png: bytes,
        after_png: bytes,
        parsed_action: dict[str, Any],
        previous_actions: list[dict[str, Any]],
    ) -> ProcessRewardResult:
        messages = _process_reward_messages(
            task_goal=task_goal,
            before_png=before_png,
            after_png=after_png,
            parsed_action=parsed_action,
            previous_actions=previous_actions,
        )
        try:
            completion = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=0.1,
            )
            content = completion.choices[0].message.content
            if not isinstance(content, str) or not content.strip():
                raise ValueError("process-reward response is empty")
            reward, reason = _parse_process_reward(content.strip())
            return ProcessRewardResult(
                reward=reward,
                reason=reason,
                source="external_model",
            )
        except Exception as exc:
            return ProcessRewardJudge.unscored(
                f"process reward unavailable after {type(exc).__name__}: {exc}",
                "unscored_prm_error",
            )

    async def close(self) -> None:
        await self.client.close()
