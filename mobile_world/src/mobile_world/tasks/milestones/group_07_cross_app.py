"""Agent 7: Cross-application and remaining milestone evaluators."""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mobile_world.runtime.app_helpers.fossify_calendar import get_calendar_events
from mobile_world.runtime.controller import AndroidController
from mobile_world.runtime.utils.helpers import execute_adb
from mobile_world.tasks.milestones.types import MilestoneEvaluator

OWNED_TASKS = (
    "MastodonCreateMemoTask",
    "MastodonChangeHeaderTask",
    "MattermostReplyToMessageTask",
    "SendInterviewEmailTask",
    "SetAlarmTask",
)

_SENT_EMAIL_PATH = "/sdcard/Android/data/com.gmailclone/files/sentEmail.json"
_NO_SENT_EMAIL = "__GUIAGENTLAB_NO_SENT_EMAIL__"
_ALARM_DB_PATH = "/data/user_de/0/com.google.android.deskclock/databases/alarms.db"


def _contains_casefold(value: str, expected: str) -> bool:
    return expected.casefold() in value.casefold()


def _evaluate_mastodon_create_memo(
    task: Any,
    controller: AndroidController,
) -> dict[str, bool]:
    del controller
    events = get_calendar_events()
    scheduled_events = [
        event
        for event in events
        if event["start_ts"] == task.EXPECTED_EVENT_START_TIME
        and event["end_ts"] == task.EXPECTED_EVENT_END_TIME
    ]

    return {
        "event_time_matches": bool(scheduled_events),
        "event_title_matches": any(
            _contains_casefold(event["title"], task.EXPECTED_TITLE) for event in scheduled_events
        ),
        "event_location_matches": any(
            _contains_casefold(event["title"], task.EXPECTED_TITLE)
            and _contains_casefold(event["location"], task.EXPECTED_LOCATION)
            for event in scheduled_events
        ),
        "event_reminder_is_one_day": any(
            _contains_casefold(event["title"], task.EXPECTED_TITLE)
            and _contains_casefold(event["location"], task.EXPECTED_LOCATION)
            and event["reminder_1_minutes"] == task.EXPECTED_REMINDER_TIME
            for event in scheduled_events
        ),
    }


def _get_mastodon_account_info(username: str) -> dict[str, Any] | None:
    from mobile_world.runtime.app_helpers import mastodon

    return mastodon.get_user_account_info(username)


def _get_mastodon_header_path(account_id: int, header_file_name: str) -> str:
    from mobile_world.runtime.app_helpers import mastodon

    return mastodon.get_header_path(account_id, header_file_name)


def _evaluate_mastodon_change_header(
    task: Any,
    controller: AndroidController,
) -> dict[str, bool]:
    del controller
    account = _get_mastodon_account_info(task.EXPECTED_USERNAME)
    if account is None:
        raise RuntimeError(f"Mastodon account unavailable: {task.EXPECTED_USERNAME}")

    initial = task._initial_header_state
    if not isinstance(initial, Mapping):
        raise TypeError("Initial Mastodon header state must be a mapping")

    account_id = account.get("account_id")
    header_file_name = account.get("header_file_name")
    if not account_id or not header_file_name:
        return {
            "header_resource_is_registered": False,
            "header_resource_changed": False,
        }

    header_path = Path(_get_mastodon_header_path(account_id, header_file_name))
    try:
        header_stat = header_path.stat()
    except FileNotFoundError:
        return {
            "header_resource_is_registered": False,
            "header_resource_changed": False,
        }

    resource_is_registered = (
        header_path.is_file()
        and header_stat.st_size > 0
        and account.get("header_file_size") == header_stat.st_size
    )
    current = {
        "header_file_name": header_file_name,
        "header_file_size": account.get("header_file_size"),
        "header_updated_at": account.get("header_updated_at"),
    }
    return {
        "header_resource_is_registered": resource_is_registered,
        # The backend has no structured source-image identity. Track the exact
        # persisted resource replacement without falling back to image matching.
        "header_resource_changed": bool(
            resource_is_registered and current != dict(initial)
        ),
    }


def _evaluate_mattermost_reply(
    task: Any,
    controller: AndroidController,
) -> dict[str, bool]:
    del controller
    messages = _get_mattermost_messages()
    if messages is None:
        raise RuntimeError("Mattermost messages backend is unavailable")

    target_messages = [message for message in messages if message[0] == task.EARLIER_MSG_ID]
    if not target_messages:
        raise RuntimeError(f"Mattermost target post unavailable: {task.EARLIER_MSG_ID}")
    target_message = target_messages[0]
    replies = [
        message
        for message in messages
        if message[6] == task.EARLIER_MSG_ID
        and message[4] == target_message[4]
        and message[5] == target_message[5]
    ]
    return {
        "target_thread_has_own_reply": bool(replies),
        "reply_contains_osworld_result": any("35.5" in message[8] for message in replies),
    }


def _get_mattermost_messages() -> list[tuple[Any, ...]] | None:
    from mobile_world.runtime.app_helpers import mattermost

    return mattermost.get_latest_messages()


def _read_sent_email(controller: AndroidController) -> dict[str, Any] | None:
    device = shlex.quote(controller.device)
    path = shlex.quote(_SENT_EMAIL_PATH)
    shell_command = (
        f"if [ -f {path} ]; then cat {path}; else printf {shlex.quote(_NO_SENT_EMAIL)}; fi"
    )
    result = execute_adb(
        f"adb -s {device} shell {shlex.quote(shell_command)}",
        output=False,
    )
    if not result.success:
        raise RuntimeError(f"Failed to read Gmail sent record: {result.error}")
    if result.output == _NO_SENT_EMAIL:
        return None

    email = json.loads(result.output)
    if not isinstance(email, dict):
        raise TypeError("Gmail sent record must be a JSON object")
    return email


def _evaluate_send_interview_email(
    task: Any,
    controller: AndroidController,
) -> dict[str, bool]:
    email = _read_sent_email(controller)
    sent_recorded = email is not None
    recipient = email.get("to") if email is not None else None
    body = email.get("body") if email is not None else None
    return {
        "email_sent_recorded": sent_recorded,
        "email_recipient_is_kevin": sent_recorded
        and isinstance(recipient, str)
        and recipient.strip().casefold() == task.correct_recipient.casefold(),
        "email_body_has_interview_time": sent_recorded
        and isinstance(body, str)
        and _contains_casefold(body, task.expected_message_partial),
    }


def _query_alarm_rows(
    controller: AndroidController,
    *,
    hour: int,
    minute: int,
) -> list[dict[str, Any]]:
    sql = (
        "SELECT hour, minutes, enabled, daysofweek, vibrate, ringtone, label, blackout_end "
        f"FROM alarm_templates WHERE hour={hour} AND minutes={minute} ORDER BY _id ASC;"
    )
    sqlite_command = f"sqlite3 -json {shlex.quote(_ALARM_DB_PATH)} {shlex.quote(sql)}"
    result = execute_adb(
        f"adb -s {shlex.quote(controller.device)} shell {shlex.quote(sqlite_command)}",
        output=False,
        root_required=True,
    )
    if not result.success:
        raise RuntimeError(f"Failed to query Android Clock alarms: {result.error}")
    if not result.output:
        return []

    rows = json.loads(result.output)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise TypeError("Android Clock alarm query must return a JSON array of objects")
    return rows


def _evaluate_set_alarm(
    task: Any,
    controller: AndroidController,
) -> dict[str, bool]:
    del task
    matching_alarms = [
        alarm
        for alarm in _query_alarm_rows(controller, hour=8, minute=25)
        if alarm["hour"] == 8 and alarm["minutes"] == 25
    ]
    # The task evaluator parses the first row returned by the same SQLite
    # query. Bind every predicate to that one row so multiple 08:25 alarms
    # cannot be combined into a false full match.
    alarm = matching_alarms[0] if matching_alarms else None
    enabled = alarm is not None and alarm["enabled"] == 1
    weekend = enabled and alarm["daysofweek"] == 96
    non_vibrating = weekend and alarm["vibrate"] == 0
    beebeep = (
        non_vibrating
        and isinstance(alarm["ringtone"], str)
        and "beebeep" in alarm["ringtone"].casefold()
    )

    return {
        "alarm_time_is_0825": alarm is not None,
        "alarm_is_enabled": bool(enabled),
        "alarm_repeats_on_weekends": bool(weekend),
        "alarm_vibration_is_off": bool(non_vibrating),
        "alarm_ringtone_is_beebeep": bool(beebeep),
    }


# Agent 7 owns this mapping and the matching test file exclusively.
TASK_EVALUATORS: dict[str, MilestoneEvaluator] = {
    "MastodonCreateMemoTask": _evaluate_mastodon_create_memo,
    "MastodonChangeHeaderTask": _evaluate_mastodon_change_header,
    "MattermostReplyToMessageTask": _evaluate_mattermost_reply,
    "SendInterviewEmailTask": _evaluate_send_interview_email,
    "SetAlarmTask": _evaluate_set_alarm,
}
