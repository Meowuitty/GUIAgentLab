"""Agent 4: Mail-driven workflow milestone evaluators."""

from __future__ import annotations

import json
import re
import shlex
from typing import Any

from mobile_world.runtime.controller import AndroidController
from mobile_world.runtime.utils.helpers import execute_adb
from mobile_world.tasks.milestones.types import MilestoneEvaluator

OWNED_TASKS = (
    "RequestCarpoolingTask",
    "AcceptMeetingTask",
    "CheckRegistrationTask",
    "CheckSetMeetTimeTask",
    "CheckDepartTimeTask",
)

_SENT_EMAIL_PATH = "/sdcard/Android/data/com.gmailclone/files/sentEmail.json"
_MISSING_EMAIL_SENTINEL = "__GUIAGENTLAB_SENT_EMAIL_MISSING__"
_CALENDAR_DB_PATH = "/data/user/0/org.fossify.calendar/databases/events.db"

_CARPOOL_PHONE = "3522228876"
_CARPOOL_MESSAGE = "Hey, could you help send Bob to the competition tomorrow? Thanks."
_DANIEL_EMAIL = "dan123@gmail.com"
_MEETING_REPLY_SUBJECT = "RE: Meeting Thursday"
_MEETING_REPLY_BODY = "I'll be there at 10:00 AM on Thursday"
_KATHY_EMAIL = "kathy@gmail.com"
_REGISTRATION_SUBJECT = "Putnam Registration Confirmation"
_BOARD_MEETING_TITLE = "Board Meeting"
_BOARD_MEETING_START_TS = 1763218800  # 2025-11-15 15:00:00 UTC
_BOARD_MEETING_END_TS = _BOARD_MEETING_START_TS + 3600
_DEPART_PHONE = "34567843456"
_DEPART_MESSAGE = "Do you know what time we're leaving tomorrow?"


def _device(controller: AndroidController) -> str:
    device = controller.device
    if not isinstance(device, str) or not device:
        raise ValueError("Android controller has no device serial")
    return shlex.quote(device)


def _normalized_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def _contains_normalized(value: str, expected: str) -> bool:
    return _normalized_text(expected) in _normalized_text(value)


def _normalized_phone(value: str) -> str:
    return "".join(character for character in value if character.isdigit())


def _read_sent_sms(controller: AndroidController) -> list[tuple[str, str]]:
    command = (
        f"adb -s {_device(controller)} shell content query "
        "--uri content://sms/sent --projection address:body"
    )
    result = execute_adb(command, output=False, root_required=True)
    if not result.success:
        raise RuntimeError(f"Failed to query sent SMS backend: {result.error}")
    if not result.output or result.output.strip() == "No result found.":
        return []

    messages: list[tuple[str, str]] = []
    for line in result.output.splitlines():
        address_match = re.search(r"\baddress=([^,]*)", line)
        if address_match is None or "body=" not in line:
            raise ValueError(f"Malformed sent SMS row: {line}")
        body = line.split("body=", 1)[1].strip()
        messages.append((address_match.group(1).strip(), body))
    return messages


def _read_sent_email(controller: AndroidController) -> dict[str, Any] | None:
    command = (
        f'adb -s {_device(controller)} shell "if [ -f {_SENT_EMAIL_PATH} ]; '
        f"then cat {_SENT_EMAIL_PATH}; else echo {_MISSING_EMAIL_SENTINEL}; fi\""
    )
    result = execute_adb(command, output=False)
    if not result.success:
        raise RuntimeError(f"Failed to read sent email backend: {result.error}")
    if result.output == _MISSING_EMAIL_SENTINEL:
        return None
    if not result.output:
        raise ValueError("Sent email backend returned an empty file")

    email = json.loads(result.output)
    if not isinstance(email, dict):
        raise ValueError("Sent email backend must contain a JSON object")
    return email


def _email_field(email: dict[str, Any], field: str) -> str:
    value = email.get(field)
    if not isinstance(value, str):
        raise ValueError(f"Sent email field {field!r} must be a string")
    return value


def _read_calendar_events(controller: AndroidController) -> list[dict[str, Any]]:
    sql = "select title, start_ts, end_ts from events"
    command = (
        f'adb -s {_device(controller)} shell "sqlite3 -json {_CALENDAR_DB_PATH} '
        f'\\"{sql}\\""'
    )
    result = execute_adb(command, output=False, root_required=True)
    if not result.success:
        raise RuntimeError(f"Failed to query calendar backend: {result.error}")
    if not result.output:
        return []

    events = json.loads(result.output)
    if not isinstance(events, list):
        raise ValueError("Calendar backend must return a JSON array")
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Calendar backend returned a non-object event")
        if not isinstance(event.get("title"), str):
            raise ValueError("Calendar event title must be a string")
        for field in ("start_ts", "end_ts"):
            value = event.get(field)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"Calendar event {field} must be an integer")
    return events


def _request_carpooling_milestones(
    _task: Any,
    controller: AndroidController,
) -> dict[str, bool]:
    messages = _read_sent_sms(controller)
    correct_recipient = [
        body
        for address, body in messages
        if _normalized_phone(address) == _CARPOOL_PHONE
    ]
    return {
        "sms_sent_to_daniel": bool(correct_recipient),
        "carpool_request_message_sent": any(
            _contains_normalized(body, _CARPOOL_MESSAGE) for body in correct_recipient
        ),
    }


def _accept_meeting_milestones(
    _task: Any,
    controller: AndroidController,
) -> dict[str, bool]:
    email = _read_sent_email(controller)
    if email is None:
        return {
            "reply_sent_to_daniel": False,
            "reply_subject_preserves_thread": False,
            "meeting_commitment_sent": False,
        }

    recipient_matches = _normalized_text(_email_field(email, "to")) == _DANIEL_EMAIL
    subject_matches = (
        _normalized_text(_email_field(email, "subject"))
        == _normalized_text(_MEETING_REPLY_SUBJECT)
    )
    body_matches = _contains_normalized(
        _email_field(email, "body"),
        _MEETING_REPLY_BODY,
    )
    return {
        "reply_sent_to_daniel": recipient_matches,
        "reply_subject_preserves_thread": recipient_matches and subject_matches,
        "meeting_commitment_sent": recipient_matches and subject_matches and body_matches,
    }


def _check_registration_milestones(
    _task: Any,
    controller: AndroidController,
) -> dict[str, bool]:
    email = _read_sent_email(controller)
    if email is None:
        return {
            "inquiry_sent_to_kathy": False,
            "registration_subject_set": False,
            "inquiry_body_present": False,
        }

    recipient_matches = _normalized_text(_email_field(email, "to")) == _KATHY_EMAIL
    subject_matches = (
        _normalized_text(_email_field(email, "subject"))
        == _normalized_text(_REGISTRATION_SUBJECT)
    )
    body_present = bool(_email_field(email, "body").strip())
    return {
        "inquiry_sent_to_kathy": recipient_matches,
        "registration_subject_set": recipient_matches and subject_matches,
        "inquiry_body_present": recipient_matches and subject_matches and body_present,
    }


def _check_set_meet_time_milestones(
    _task: Any,
    controller: AndroidController,
) -> dict[str, bool]:
    events = _read_calendar_events(controller)
    titled_events = [
        event
        for event in events
        if _normalized_text(event["title"]) == _normalized_text(_BOARD_MEETING_TITLE)
    ]
    correctly_started_events = [
        event
        for event in titled_events
        if event["start_ts"] == _BOARD_MEETING_START_TS
    ]
    return {
        "board_meeting_event_saved": bool(titled_events),
        "meeting_starts_at_email_time": bool(correctly_started_events),
        "meeting_is_one_hour": any(
            event["end_ts"] == _BOARD_MEETING_END_TS for event in correctly_started_events
        ),
    }


def _check_depart_time_milestones(
    _task: Any,
    controller: AndroidController,
) -> dict[str, bool]:
    messages = _read_sent_sms(controller)
    correct_recipient = [
        body
        for address, body in messages
        if _normalized_phone(address) == _DEPART_PHONE
    ]
    return {
        "sms_sent_to_carl": bool(correct_recipient),
        "departure_time_question_sent": any(
            _contains_normalized(body, _DEPART_MESSAGE) for body in correct_recipient
        ),
    }


# Agent 4 owns this mapping and the matching test file exclusively.
TASK_EVALUATORS: dict[str, MilestoneEvaluator] = {
    "RequestCarpoolingTask": _request_carpooling_milestones,
    "AcceptMeetingTask": _accept_meeting_milestones,
    "CheckRegistrationTask": _check_registration_milestones,
    "CheckSetMeetTimeTask": _check_set_meet_time_milestones,
    "CheckDepartTimeTask": _check_depart_time_milestones,
}
