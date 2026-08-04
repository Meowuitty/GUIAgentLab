"""Agent 3: Calendar, SMS, and alarm milestone evaluators."""

from __future__ import annotations

import datetime
import json
import re
from typing import Any

from mobile_world.runtime.controller import AndroidController
from mobile_world.runtime.utils.helpers import execute_adb
from mobile_world.tasks.milestones.types import MilestoneEvaluator

OWNED_TASKS = (
    "ScheduleCoffeeTimeViaSmsTask",
    "ScheduleLunchViaSmsTask",
    "CheckConferenceAndSendSmsTask1",
    "CheckConferenceAndSendSmsTask2",
    "CheckEventTimeTask",
)

_CALENDAR_DB = "/data/user/0/org.fossify.calendar/databases/events.db"
_ALARMS_DB = "/data/user_de/0/com.google.android.deskclock/databases/alarms.db"
_SMS_ROW = re.compile(
    r"Row:\s+\d+\s+address=(?P<address>.*?),\s+body=(?P<body>.*)",
    flags=re.DOTALL,
)


def _run_read_query(
    controller: AndroidController,
    remote_command: str,
    *,
    source: str,
) -> str:
    """Run one read-only ADB query, raising when its authoritative source fails."""
    result = execute_adb(
        f'adb -s {controller.device} shell "{remote_command}"',
        output=False,
    )
    if not result.success:
        raise RuntimeError(f"Failed to query {source}: {result.error}")
    return result.output.strip()


def _get_sent_sms(controller: AndroidController) -> list[dict[str, str]]:
    """Read durable sent-message address/body pairs from Android's SMS provider."""
    output = _run_read_query(
        controller,
        "su root content query --uri content://sms/sent --projection address:body",
        source="content://sms/sent",
    )
    if not output or output == "No result found.":
        return []

    rows = []
    for raw_row in re.split(r"\n(?=Row:\s+\d+\s+address=)", output):
        match = _SMS_ROW.fullmatch(raw_row)
        if match is None:
            raise RuntimeError(f"Unexpected content://sms/sent row: {raw_row!r}")
        rows.append(
            {
                "address": match.group("address"),
                "body": match.group("body"),
            }
        )
    return rows


def _get_calendar_event_times(controller: AndroidController) -> list[dict[str, int]]:
    """Read event boundaries from Fossify Calendar's authoritative events table."""
    output = _run_read_query(
        controller,
        (
            f"su root sqlite3 -json {_CALENDAR_DB} "
            "'SELECT start_ts,end_ts FROM events;'"
        ),
        source="Fossify Calendar events table",
    )
    if not output:
        return []
    try:
        rows = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Invalid JSON from Fossify Calendar events table") from exc
    if not isinstance(rows, list) or any(
        not isinstance(row, dict)
        or type(row.get("start_ts")) is not int
        or type(row.get("end_ts")) is not int
        for row in rows
    ):
        raise RuntimeError("Unexpected rows from Fossify Calendar events table")
    return rows


def _get_alarm_states(
    controller: AndroidController,
    *,
    hour: int,
    minute: int,
) -> list[dict[str, int | bool]]:
    """Read matching alarm time/enabled fields from DeskClock's alarm_templates table."""
    output = _run_read_query(
        controller,
        (
            f"su root sqlite3 {_ALARMS_DB} "
            f"'SELECT hour,minutes,enabled FROM alarm_templates "
            f"WHERE hour={hour} AND minutes={minute} ORDER BY _id ASC;'"
        ),
        source="DeskClock alarm_templates table",
    )
    if not output:
        return []

    rows = []
    for raw_row in output.splitlines():
        fields = raw_row.split("|")
        if len(fields) != 3:
            raise RuntimeError(f"Unexpected DeskClock alarm row: {raw_row!r}")
        try:
            row_hour, row_minute, enabled = (int(field) for field in fields)
        except ValueError as exc:
            raise RuntimeError(f"Unexpected DeskClock alarm row: {raw_row!r}") from exc
        if enabled not in (0, 1):
            raise RuntimeError(f"Unexpected DeskClock enabled value: {enabled!r}")
        rows.append(
            {
                "hour": row_hour,
                "minutes": row_minute,
                "enabled": bool(enabled),
            }
        )
    return rows


def _rows_to_phone(
    rows: list[dict[str, str]],
    phone_number: str,
) -> list[dict[str, str]]:
    return [row for row in rows if row["address"] == phone_number]


def _body_contains(body: str, expected: str | list[str]) -> bool:
    expected_parts = expected if isinstance(expected, list) else [expected]
    lowered_body = body.lower()
    return all(str(part).lower() in lowered_body for part in expected_parts)


def _schedule_coffee_time_via_sms(
    task: Any,
    controller: AndroidController,
) -> dict[str, bool]:
    """Evaluate the recipient and reply body in durable content://sms/sent rows."""
    recipient_rows = _rows_to_phone(_get_sent_sms(controller), task.sender_phone)
    return {
        # content://sms/sent.address equals Marry's task-defined phone number.
        "reply_sent_to_marry": bool(recipient_rows),
        # The same recipient has a sent row containing the task evaluator's reply text.
        "unavailability_reply_sent_to_marry": any(
            _body_contains(row["body"], task.expected_reply) for row in recipient_rows
        ),
    }


def _schedule_lunch_via_sms(
    task: Any,
    controller: AndroidController,
) -> dict[str, bool]:
    """Evaluate the sent reply and exact Fossify Calendar event interval."""
    recipient_rows = _rows_to_phone(_get_sent_sms(controller), task.sender_phone)
    events = _get_calendar_event_times(controller)

    start_time = datetime.datetime(
        task.expected_date.year,
        task.expected_date.month,
        task.expected_date.day,
        task.expected_start_hour,
        task.expected_start_minute,
        tzinfo=datetime.UTC,
    )
    start_ts = int(start_time.timestamp())
    end_ts = int(
        (start_time + datetime.timedelta(hours=task.expected_duration_hours)).timestamp()
    )
    return {
        # content://sms/sent.address equals Marry's task-defined phone number.
        "reply_sent_to_marry": bool(recipient_rows),
        # The same recipient has a sent row containing the task evaluator's OK reply.
        "ok_reply_sent_to_marry": any(
            _body_contains(row["body"], task.expected_reply) for row in recipient_rows
        ),
        # Fossify events.start_ts/end_ts exactly cover 2025-10-17 11:00-12:00 UTC.
        "lunch_event_scheduled": any(
            event["start_ts"] == start_ts and event["end_ts"] == end_ts
            for event in events
        ),
    }


def _conference_sms(
    task: Any,
    controller: AndroidController,
    *,
    dates_milestone_id: str,
) -> dict[str, bool]:
    """Evaluate Mia's recipient field and the task-defined date content."""
    recipient_rows = _rows_to_phone(_get_sent_sms(controller), task.correct_phone_number)
    return {
        # content://sms/sent.address equals Mia Scott's task-defined phone number.
        "sms_sent_to_mia": bool(recipient_rows),
        # A sent row to Mia contains the date text accepted by task.is_successful().
        dates_milestone_id: any(
            _body_contains(row["body"], task.expected_message_content)
            for row in recipient_rows
        ),
    }


def _check_conference_and_send_sms_1(
    task: Any,
    controller: AndroidController,
) -> dict[str, bool]:
    return _conference_sms(
        task,
        controller,
        dates_milestone_id="paris_dates_sent_to_mia",
    )


def _check_conference_and_send_sms_2(
    task: Any,
    controller: AndroidController,
) -> dict[str, bool]:
    return _conference_sms(
        task,
        controller,
        dates_milestone_id="tokyo_dates_sent_to_mia",
    )


def _check_event_time(
    task: Any,  # noqa: ARG001
    controller: AndroidController,
) -> dict[str, bool]:
    """Evaluate the matching time and enabled bit in DeskClock's durable alarm row."""
    alarms = _get_alarm_states(controller, hour=18, minute=0)
    # check_alarm_via_adb(), used by the task evaluator, parses the first row
    # returned for this time. Bind both predicates to the same deterministic
    # first row so a later enabled duplicate cannot complete the milestone
    # while the task evaluator rejects the earlier disabled row.
    alarm = alarms[0] if alarms else None
    return {
        # alarm_templates has a row whose hour/minutes fields are 18 and 0.
        "alarm_created_for_1800": alarm is not None,
        # The evaluator-visible first row has enabled=1.
        "alarm_enabled_for_1800": alarm is not None and bool(alarm["enabled"]),
    }


# Agent 3 owns this mapping and the matching test file exclusively.
TASK_EVALUATORS: dict[str, MilestoneEvaluator] = {
    "ScheduleCoffeeTimeViaSmsTask": _schedule_coffee_time_via_sms,
    "ScheduleLunchViaSmsTask": _schedule_lunch_via_sms,
    "CheckConferenceAndSendSmsTask1": _check_conference_and_send_sms_1,
    "CheckConferenceAndSendSmsTask2": _check_conference_and_send_sms_2,
    "CheckEventTimeTask": _check_event_time,
}
