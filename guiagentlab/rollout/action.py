"""Strict MAI-UI-to-MobileWorld action projection."""

from __future__ import annotations

import json
import re
from numbers import Real
from typing import Any


class PolicyOutputError(ValueError):
    """The model returned text that is not a valid GUI policy action."""

_COORDINATE_SCALE = 999
DEFAULT_SCREEN_SIZE = (1080, 2400)
_AVAILABLE_APPS = {
    "桌面",
    "contacts",
    "settings",
    "设置",
    "clock",
    "maps",
    "chrome",
    "calendar",
    "files",
    "gallery",
    "淘店",
    "taodian",
    "mattermost",
    "mastodon",
    "mail",
    "sms",
    "camera",
}
_APP_NAME_ALIASES = {
    # Android and the model commonly call the same app "Messages" while the
    # MobileWorld action contract names it "SMS".
    "message": "SMS",
    "messages": "SMS",
}
_DIRECT_ACTIONS = {
    "app_switch",
    "click",
    "double_tap",
    "long_press",
    "swipe",
    "drag",
    "input_text",
    "keyboard_enter",
    "navigate_back",
    "navigate_home",
    "open_app",
    "scroll",
    "wait",
    "answer",
    "ask_user",
    "status",
}


def _decode_action_object(text: str) -> dict[str, Any]:
    normalized = (
        text.strip()
        .replace("［", "[")
        .replace("］", "]")
        .replace("｛", "{")
        .replace("｝", "}")
        .replace("，", ",")
        .replace("：", ":")
    )
    tagged = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", normalized, re.I | re.S)
    candidate = tagged.group(1) if tagged else normalized
    candidate = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", candidate, flags=re.I)
    start = candidate.find("{")
    if start < 0:
        raise PolicyOutputError("model output contains no JSON action")
    try:
        value, _ = json.JSONDecoder().raw_decode(candidate[start:])
    except json.JSONDecodeError as exc:
        # Curly quotes are valid characters *inside* a JSON string (for
        # example an answer containing “Amazon”). Only normalize them as a
        # fallback for model outputs that used them as JSON delimiters.
        curly_normalized = candidate.replace("“", '"').replace("”", '"')
        try:
            value, _ = json.JSONDecoder().raw_decode(curly_normalized[start:])
        except json.JSONDecodeError:
            raise PolicyOutputError(f"malformed JSON action: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise PolicyOutputError("action must be a JSON object")

    if "arguments" in value:
        if value.get("name", "mobile_use") != "mobile_use":
            raise PolicyOutputError(f"unsupported tool: {value.get('name')!r}")
        value = value["arguments"]
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise PolicyOutputError("tool arguments are malformed JSON") from exc
        if not isinstance(value, dict):
            raise PolicyOutputError("tool arguments must be a JSON object")
    return dict(value)


def _normalize_model_text(text: str) -> str:
    """Normalize punctuation variants handled by the original MAI-UI projector."""
    replacements = {
        "［": "[",
        "］": "]",
        "【": "[",
        "】": "]",
        "｛": "{",
        "｝": "}",
        "（": "(",
        "）": ")",
        "，": ",",
        "：": ":",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _numeric(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise PolicyOutputError(f"{field} must be numeric")
    return float(value)


def _point(value: Any, field: str) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) not in {2, 4}:
        raise PolicyOutputError(f"{field} must contain two coordinates or a four-value box")
    numbers = [_numeric(item, field) for item in value]
    if len(numbers) == 4:
        return (numbers[0] + numbers[2]) / 2, (numbers[1] + numbers[3]) / 2
    return numbers[0], numbers[1]


def _validate_screen_size(screen_size: tuple[int, int]) -> tuple[int, int]:
    width, height = screen_size
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise ValueError(f"screen_size must contain positive integer dimensions: {screen_size!r}")
    return width, height


def _pixels(x: Any, y: Any, screen_size: tuple[int, int]) -> tuple[int, int]:
    x_value = _numeric(x, "x")
    y_value = _numeric(y, "y")
    width, height = _validate_screen_size(screen_size)
    if 0 <= x_value <= _COORDINATE_SCALE and 0 <= y_value <= _COORDINATE_SCALE:
        return (
            int(x_value / _COORDINATE_SCALE * width),
            int(y_value / _COORDINATE_SCALE * height),
        )
    return int(x_value), int(y_value)


def _coordinate(
    action: dict[str, Any], screen_size: tuple[int, int], *aliases: str
) -> tuple[int, int]:
    for field in aliases:
        if field in action:
            return _pixels(*_point(action[field], field), screen_size)
    if "x" in action and "y" in action:
        return _pixels(action["x"], action["y"], screen_size)
    raise PolicyOutputError("coordinate action is missing a valid coordinate")


def _reverse_swipe_direction(direction: str) -> str:
    return {
        "up": "down",
        "down": "up",
        "left": "left",
        "right": "right",
    }[direction]


def parse_action(
    text: str, screen_size: tuple[int, int] = DEFAULT_SCREEN_SIZE
) -> dict[str, Any]:
    """Parse one MAI-UI tool call into the canonical MobileWorld server action.

    MAI-UI emits coordinates in a 0..999 space and describes finger-swipe
    direction. MobileWorld's standard agent converts that to actual screenshot
    pixels and to its inverse vertical ``scroll`` convention before calling the
    server. Both VERL and independent OpenAI evaluation use this function.
    """
    value = _decode_action_object(text)
    raw_type = value.get("action_type", value.get("action"))
    if not isinstance(raw_type, str):
        raise PolicyOutputError("action has no string action/action_type field")
    action_type = raw_type.lower()

    aliases = {
        "double_click": "double_tap",
        "type": "input_text",
        "open": "open_app",
        "terminate": "status",
        "finished": "status",
    }
    action_type = aliases.get(action_type, action_type)

    if action_type == "system_button":
        button = str(value.get("button", "")).lower()
        try:
            action_type = {
                "back": "navigate_back",
                "home": "navigate_home",
                "menu": "app_switch",
                "enter": "keyboard_enter",
            }[button]
        except KeyError as exc:
            raise PolicyOutputError(f"unsupported system button: {button!r}") from exc

    if action_type not in _DIRECT_ACTIONS:
        raise PolicyOutputError(f"unsupported action_type: {action_type!r}")

    if action_type in {"click", "double_tap", "long_press"}:
        x, y = _coordinate(
            value, screen_size, "coordinate", raw_type, "click", "target"
        )
        return {"action_type": action_type, "x": x, "y": y}

    if action_type == "drag":
        if "start_coordinate" in value and "end_coordinate" in value:
            start_x, start_y = _pixels(
                *_point(value["start_coordinate"], "start_coordinate"), screen_size
            )
            end_x, end_y = _pixels(
                *_point(value["end_coordinate"], "end_coordinate"), screen_size
            )
        else:
            start_x, start_y = _pixels(
                value.get("start_x"), value.get("start_y"), screen_size
            )
            end_x, end_y = _pixels(
                value.get("end_x"), value.get("end_y"), screen_size
            )
        return {
            "action_type": "drag",
            "start_x": start_x,
            "start_y": start_y,
            "end_x": end_x,
            "end_y": end_y,
        }

    if action_type in {"swipe", "scroll"}:
        direction = str(value.get("direction", "up")).lower()
        if direction not in {"up", "down", "left", "right"}:
            raise PolicyOutputError(f"unsupported swipe direction: {direction!r}")
        # ``swipe`` is the model-facing gesture. MobileWorld's historical
        # MAI-UI projector sends it as the inverse vertical ``scroll`` action.
        if action_type == "swipe":
            direction = _reverse_swipe_direction(direction)
        action: dict[str, Any] = {"action_type": "scroll", "direction": direction}
        if any(field in value for field in ("coordinate", "swipe", "x", "y")):
            action["x"], action["y"] = _coordinate(
                value, screen_size, "coordinate", "swipe"
            )
        return action

    if action_type in {"input_text", "answer", "ask_user"}:
        return {"action_type": action_type, "text": str(value.get("text", ""))}
    if action_type == "open_app":
        app_name = value.get("app_name", value.get("text"))
        if not isinstance(app_name, str) or not app_name:
            raise PolicyOutputError("open action is missing an app name")
        normalized_app = app_name.casefold()
        if normalized_app == "home":
            return {"action_type": "navigate_home"}
        app_name = _APP_NAME_ALIASES.get(normalized_app, app_name)
        if app_name.casefold() not in _AVAILABLE_APPS:
            raise PolicyOutputError(f"unsupported app name: {app_name!r}")
        return {"action_type": "open_app", "app_name": app_name}
    if action_type == "status":
        status = str(value.get("goal_status", value.get("status", "success"))).lower()
        if status == "failure":
            status = "fail"
        if status not in {"success", "fail", "failed"}:
            raise PolicyOutputError(f"unsupported terminal status: {status!r}")
        return {"action_type": "status", "goal_status": status}
    return {"action_type": action_type}


def _regex_fallback(
    text: str, screen_size: tuple[int, int]
) -> dict[str, Any] | None:
    """Recover the same common malformed outputs accepted by the old projector."""
    normalized = _normalize_model_text(text)
    lowered = normalized.lower()

    for raw_type in ("click", "long_press", "double_tap", "double_click"):
        patterns = (
            rf'[\'\"]action[\'\"]\s*:\s*[\'\"]{raw_type}[\'\"].*?'
            rf'[\'\"]coordinate[\'\"]\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]',
            rf"\b{raw_type}\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)",
            rf"\b{raw_type}\s+(?:at\s+)?(?:\(\s*)?(\d+)\s*,\s*(\d+)(?:\s*\))?",
        )
        for pattern in patterns:
            match = re.search(pattern, lowered, re.S)
            if match:
                x, y = _pixels(
                    int(match.group(1)), int(match.group(2)), screen_size
                )
                action_type = (
                    "double_tap" if raw_type in {"double_tap", "double_click"} else raw_type
                )
                return {"action_type": action_type, "x": x, "y": y}

    match = re.search(
        r'[\'\"]action[\'\"]\s*:\s*[\'\"]swipe[\'\"].*?'
        r'[\'\"]direction[\'\"]\s*:\s*[\'\"](up|down|left|right)[\'\"]',
        lowered,
        re.S,
    ) or re.search(r"\bswipe\s+(up|down|left|right)\b", lowered)
    if match:
        action: dict[str, Any] = {
            "action_type": "scroll",
            "direction": _reverse_swipe_direction(match.group(1)),
        }
        coordinate = re.search(
            r'[\'\"]coordinate[\'\"]\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]',
            lowered,
        )
        if coordinate:
            action["x"], action["y"] = _pixels(
                int(coordinate.group(1)), int(coordinate.group(2)), screen_size
            )
        return action

    text_actions = {
        "type": "input_text",
        "open": "open_app",
        "answer": "answer",
    }
    for raw_type, action_type in text_actions.items():
        match = re.search(
            rf'[\'\"]action[\'\"]\s*:\s*[\'\"]{raw_type}[\'\"].*?'
            r'[\'\"]text[\'\"]\s*:\s*[\'\"]([^\'\"]*)[\'\"]',
            normalized,
            re.S | re.I,
        )
        if not match:
            continue
        value = match.group(1)
        if action_type == "open_app":
            if not value or value.casefold() not in _AVAILABLE_APPS:
                return None
            return {"action_type": action_type, "app_name": value}
        return {"action_type": action_type, "text": value}

    match = re.search(
        r'[\'\"]action[\'\"]\s*:\s*[\'\"]drag[\'\"].*?'
        r'[\'\"]start_coordinate[\'\"]\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\].*?'
        r'[\'\"]end_coordinate[\'\"]\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]',
        lowered,
        re.S,
    )
    if match:
        start_x, start_y = _pixels(
            int(match.group(1)), int(match.group(2)), screen_size
        )
        end_x, end_y = _pixels(
            int(match.group(3)), int(match.group(4)), screen_size
        )
        return {
            "action_type": "drag",
            "start_x": start_x,
            "start_y": start_y,
            "end_x": end_x,
            "end_y": end_y,
        }

    match = re.search(
        r'[\'\"]action[\'\"]\s*:\s*[\'\"]system_button[\'\"].*?'
        r'[\'\"]button[\'\"]\s*:\s*[\'\"](back|home|menu|enter)[\'\"]',
        lowered,
        re.S,
    )
    if match:
        return {
            "action_type": {
                "back": "navigate_back",
                "home": "navigate_home",
                "menu": "app_switch",
                "enter": "keyboard_enter",
            }[match.group(1)]
        }

    if re.search(r'[\'\"]action[\'\"]\s*:\s*[\'\"]wait[\'\"]', lowered):
        return {"action_type": "wait"}

    match = re.search(
        r'[\'\"]action[\'\"]\s*:\s*[\'\"]terminate[\'\"].*?'
        r'[\'\"]status[\'\"]\s*:\s*[\'\"](success|fail|failure)[\'\"]',
        lowered,
        re.S,
    )
    if match:
        status = "fail" if match.group(1) == "failure" else match.group(1)
        return {"action_type": "status", "goal_status": status}
    return None


def parse_action_or_wait(
    text: str, screen_size: tuple[int, int] = DEFAULT_SCREEN_SIZE
) -> tuple[dict[str, Any], bool, str | None]:
    """Project one model response, preserving the original wait-on-error behavior."""
    try:
        return parse_action(text, screen_size), True, None
    except PolicyOutputError as exc:
        recovered = _regex_fallback(text, screen_size)
        if recovered is not None:
            warning = f"strict parser failed; recovered by compatibility parser: {exc}"
            return recovered, True, warning
        return {"action_type": "wait"}, False, str(exc)


def is_terminal_action(action: dict[str, Any]) -> bool:
    return action.get("action_type") in {"status", "answer"}


def action_requires_environment_step(action: dict[str, Any]) -> bool:
    """Return whether the canonical action should be sent to ``/step``.

    ``status`` is a model-side terminal signal. ``answer`` remains an
    environment action because MobileWorld records the answer before scoring.
    """
    return action.get("action_type") != "status"
