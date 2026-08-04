"""Agent 5: Android settings and media milestone evaluators."""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Mapping
from typing import Any

from mobile_world.runtime.controller import AndroidController
from mobile_world.runtime.utils.helpers import execute_adb
from mobile_world.tasks.milestones.types import MilestoneEvaluator

OWNED_TASKS = (
    "AdjustBrightnessMinimumTask",
    "AdjustFontIconMinimumTask",
    "ChangeWallpaperTask",
    "TakeSelfieTask",
    "SharePhotosTask",
)


_SYSTEM_WALLPAPER = "/data/system/users/0/wallpaper"
_ORIGINAL_WALLPAPER = "/data/system/users/0/wallpaper_orig"
_SENT_EMAIL_PATH = "/sdcard/Android/data/com.gmailclone/files/sentEmail.json"
_MISSING_FILE = "__GUIAGENTLAB_MISSING_FILE__"
_MISSING_DIRECTORY = "__GUIAGENTLAB_MISSING_DIRECTORY__"
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".dng")


def _adb_output(command: str) -> str:
    """Run one read-only ADB command, propagating backend failures."""
    result = execute_adb(command, output=False)
    if not result.success:
        detail = result.error.strip() if result.error else "unknown ADB error"
        raise RuntimeError(f"ADB backend read failed: {detail}")
    return result.output.strip()


def _device_command(controller: AndroidController, shell_command: str) -> str:
    device = shlex.quote(controller.device)
    return f"adb -s {device} shell {shell_command}"


def _read_system_setting(
    controller: AndroidController,
    namespace: str,
    key: str,
    *,
    default: str,
) -> str:
    output = _adb_output(
        _device_command(controller, f"settings get {shlex.quote(namespace)} {shlex.quote(key)}")
    )
    if not output or output == "null":
        # Android omits settings that are still at platform defaults. This is
        # valid backend state, not an infrastructure failure.
        return default
    return output


def _read_display_density(controller: AndroidController) -> int:
    output = _adb_output(_device_command(controller, "wm density"))
    override = re.search(r"Override density:\s*(\d+)", output)
    if override:
        return int(override.group(1))
    physical = re.search(r"Physical density:\s*(\d+)", output)
    if physical:
        return int(physical.group(1))
    raise ValueError(f"Unrecognized wm density output: {output!r}")


def _read_mtime(controller: AndroidController, path: str) -> str:
    output = _adb_output(_device_command(controller, f"stat -c %Y {shlex.quote(path)}"))
    if not output.isdecimal():
        raise ValueError(f"Unrecognized mtime for {path}: {output!r}")
    return output


def _read_current_wallpaper_state(
    controller: AndroidController,
    initial: Mapping[str, Any],
) -> dict[str, str]:
    """Read only the authoritative state fields captured during task initialization."""
    state = {}
    for path in (_ORIGINAL_WALLPAPER, _SYSTEM_WALLPAPER):
        if path in initial:
            state[path] = _read_mtime(controller, path)

    if "wallpaper_id" in initial or "dimensions" in initial:
        output = _adb_output(_device_command(controller, "dumpsys wallpaper"))
        if "wallpaper_id" in initial:
            match = re.search(r"id=(\d+)", output)
            if match is None:
                raise ValueError("dumpsys wallpaper has no wallpaper id")
            state["wallpaper_id"] = match.group(1)
        if "dimensions" in initial:
            width = re.search(r"mWidth=(\d+)", output)
            height = re.search(r"mHeight=(\d+)", output)
            if width is None or height is None:
                raise ValueError("dumpsys wallpaper has no dimensions")
            state["dimensions"] = f"{width.group(1)}x{height.group(1)}"

    if not state:
        raise ValueError("Initial wallpaper state has no readable backend fields")
    return state


def _read_optional_json(
    controller: AndroidController,
    path: str,
) -> Mapping[str, Any] | None:
    quoted_path = shlex.quote(path)
    script = (
        f"if [ -f {quoted_path} ]; then cat {quoted_path}; "
        f"else printf %s {shlex.quote(_MISSING_FILE)}; fi"
    )
    output = _adb_output(_device_command(controller, shlex.quote(script)))
    if output == _MISSING_FILE:
        return None
    payload = json.loads(output)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _row_field(row: str, field: str) -> str | None:
    match = re.search(rf"(?:^|[\s,]){re.escape(field)}=([^,]*)", row)
    if not match:
        return None
    value = match.group(1).strip()
    return None if value in {"", "NULL"} else value


def _required_int_field(row: str, field: str) -> int:
    value = _row_field(row, field)
    if value is None:
        raise ValueError(f"MediaStore row has no {field}: {row!r}")
    return int(value)


def _recent_image_rows(
    controller: AndroidController,
    start_timestamp: float,
) -> list[str]:
    projection = "_id:_display_name:_data:date_added:_size:mime_type:is_pending"
    output = _adb_output(
        _device_command(
            controller,
            f"content query --uri content://media/external/images/media --projection {projection}",
        )
    )
    recent_rows = []
    threshold = int(start_timestamp)
    for line in output.splitlines():
        row = line.strip()
        if not row.startswith("Row:"):
            continue
        date_added = _required_int_field(row, "date_added")
        name = (_row_field(row, "_display_name") or "").lower()
        mime_type = (_row_field(row, "mime_type") or "").lower()
        is_image = mime_type.startswith("image/") or name.endswith(_IMAGE_SUFFIXES)
        if date_added > threshold and is_image:
            recent_rows.append(row)
    return recent_rows


def _read_picture_image_count(controller: AndroidController) -> int:
    directory = "/sdcard/Pictures"
    quoted_directory = shlex.quote(directory)
    script = (
        f"if [ -d {quoted_directory} ]; then ls -1A {quoted_directory}; "
        f"else printf %s {shlex.quote(_MISSING_DIRECTORY)}; fi"
    )
    output = _adb_output(_device_command(controller, shlex.quote(script)))
    if output == _MISSING_DIRECTORY or not output:
        return 0
    return sum(line.strip().lower().endswith(_IMAGE_SUFFIXES) for line in output.splitlines())


def _attachment_names(email: Mapping[str, Any]) -> list[str]:
    attachments = email.get("attachments", [])
    if not isinstance(attachments, list):
        raise TypeError("Sent-email attachments must be a list")
    names = []
    for attachment in attachments:
        if isinstance(attachment, str):
            names.append(attachment)
            continue
        if not isinstance(attachment, dict) or not isinstance(attachment.get("name"), str):
            raise TypeError("Every sent-email attachment must have a string name")
        names.append(attachment["name"])
    return names


def evaluate_adjust_brightness_minimum(
    task: Any,
    controller: AndroidController,
) -> dict[str, bool]:
    brightness = int(
        _read_system_setting(
            controller,
            "system",
            "screen_brightness",
            default="128",
        )
    )
    minimum = int(task.min_brightness)
    return {
        "brightness_in_lowest_quarter": brightness <= max(minimum, 255 // 4),
        "brightness_at_minimum": brightness <= minimum,
    }


def evaluate_adjust_font_icon_minimum(
    task: Any,
    controller: AndroidController,
) -> dict[str, bool]:
    font_scale = float(
        _read_system_setting(
            controller,
            "system",
            "font_scale",
            default="1.0",
        )
    )
    density = _read_display_density(controller)
    return {
        "font_size_at_minimum": font_scale == float(task.target_font_scale),
        "display_size_at_minimum": density == int(task.target_density),
    }


def evaluate_change_wallpaper(
    task: Any,
    controller: AndroidController,
) -> dict[str, bool]:
    initial = task._initial_wallpaper_state
    if not isinstance(initial, Mapping):
        raise TypeError("Initial wallpaper state must be a mapping")

    current = _read_current_wallpaper_state(controller, initial)
    return {
        # Without image matching, the exact backend fact available to this task
        # is that Android persisted a wallpaper state different from baseline.
        "wallpaper_state_changed": current
        != {key: str(value) for key, value in initial.items()},
    }


def evaluate_take_selfie(
    task: Any,
    controller: AndroidController,
) -> dict[str, bool]:
    rows = _recent_image_rows(controller, float(task._start_timestamp))
    published_rows = [row for row in rows if _row_field(row, "is_pending") != "1"]
    current_photo_count = _read_picture_image_count(controller)
    initial_photo_count = int(task._initial_photo_count)
    new_photo_detected = bool(
        published_rows or current_photo_count > initial_photo_count
    )
    return {
        # Mirrors both success paths in the original evaluator: a recent
        # MediaStore row or an increased Pictures image count.
        "new_photo_detected": new_photo_detected,
    }


def evaluate_share_photos(
    task: Any,
    controller: AndroidController,
) -> dict[str, bool]:
    email = _read_optional_json(controller, _SENT_EMAIL_PATH)
    if email is None:
        return {
            "email_addressed_to_kevin": False,
            "flower_message_in_email": False,
            "all_flower_photos_attached": False,
        }

    recipient = email.get("to")
    body = email.get("body")
    attachment_names = _attachment_names(email)
    required_names = set(task.REQUIRED_IMAGES)
    return {
        "email_addressed_to_kevin": recipient == task.EMAIL_ADDRESS,
        "flower_message_in_email": (
            isinstance(body, str) and task.EMAIL_TEXT.casefold() in body.casefold()
        ),
        "all_flower_photos_attached": (
            len(attachment_names) == len(required_names) and set(attachment_names) == required_names
        ),
    }


# Agent 5 owns this mapping and the matching test file exclusively.
TASK_EVALUATORS: dict[str, MilestoneEvaluator] = {
    "AdjustBrightnessMinimumTask": evaluate_adjust_brightness_minimum,
    "AdjustFontIconMinimumTask": evaluate_adjust_font_icon_minimum,
    "ChangeWallpaperTask": evaluate_change_wallpaper,
    "TakeSelfieTask": evaluate_take_selfie,
    "SharePhotosTask": evaluate_share_photos,
}
