"""Conflict-free routing from task names to agent-owned milestone modules."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from mobile_world.runtime.controller import AndroidController

TASK_GROUP_MODULES = {
    "MastodonAddBookmarkTask": "group_01_mastodon_content",
    "MastodonRemoveBookmarkTask": "group_01_mastodon_content",
    "MastodonConditionalFavoTask": "group_01_mastodon_content",
    "MastodonPinTootsTask": "group_01_mastodon_content",
    "MastodonReportTask": "group_01_mastodon_content",
    "MastodonCreateListTask": "group_02_mastodon_social",
    "MastodonReplyTask": "group_02_mastodon_social",
    "MastodonRevisePollTask": "group_02_mastodon_social",
    "MastodonFollowTask": "group_02_mastodon_social",
    "MastodonUnfollowTask": "group_02_mastodon_social",
    "ScheduleCoffeeTimeViaSmsTask": "group_03_calendar_sms",
    "ScheduleLunchViaSmsTask": "group_03_calendar_sms",
    "CheckConferenceAndSendSmsTask1": "group_03_calendar_sms",
    "CheckConferenceAndSendSmsTask2": "group_03_calendar_sms",
    "CheckEventTimeTask": "group_03_calendar_sms",
    "RequestCarpoolingTask": "group_04_mail_workflows",
    "AcceptMeetingTask": "group_04_mail_workflows",
    "CheckRegistrationTask": "group_04_mail_workflows",
    "CheckSetMeetTimeTask": "group_04_mail_workflows",
    "CheckDepartTimeTask": "group_04_mail_workflows",
    "AdjustBrightnessMinimumTask": "group_05_device_media",
    "AdjustFontIconMinimumTask": "group_05_device_media",
    "ChangeWallpaperTask": "group_05_device_media",
    "TakeSelfieTask": "group_05_device_media",
    "SharePhotosTask": "group_05_device_media",
    "InvoiceReceiptCopyTask": "group_06_files_mall",
    "CheckPuchasedItem": "group_06_files_mall",
    "SearchItemAndCheckoutTask": "group_06_files_mall",
    "CartManagementTask": "group_06_files_mall",
    "ReadQwen3PaperTask4": "group_06_files_mall",
    "MastodonCreateMemoTask": "group_07_cross_app",
    "MastodonChangeHeaderTask": "group_07_cross_app",
    "MattermostReplyToMessageTask": "group_07_cross_app",
    "SendInterviewEmailTask": "group_07_cross_app",
    "SetAlarmTask": "group_07_cross_app",
}


def _group_module(task_name: str):
    module_name = TASK_GROUP_MODULES.get(task_name)
    if module_name is None:
        return None
    module = import_module(f"mobile_world.tasks.milestones.{module_name}")
    if task_name not in module.OWNED_TASKS:
        raise RuntimeError(f"milestone ownership mismatch for {task_name}: {module_name}")
    return module


def evaluate_task_milestones(
    task: Any,
    controller: AndroidController,
) -> dict[str, bool] | None:
    """Evaluate one registered task using its authoritative backend state."""
    module = _group_module(task.name)
    if module is None:
        return None
    evaluator = module.TASK_EVALUATORS.get(task.name)
    if evaluator is None:
        return None
    states = evaluator(task, controller)
    if not isinstance(states, dict) or not states:
        raise ValueError(f"{task.name} milestone evaluator returned no states")
    if any(not isinstance(key, str) or not key for key in states):
        raise ValueError(f"{task.name} milestone ids must be non-empty strings")
    if any(type(value) is not bool for value in states.values()):
        raise ValueError(f"{task.name} milestone values must be booleans")
    return states


def registered_task_names() -> frozenset[str]:
    """Return tasks whose agent-owned modules contain completed evaluators."""
    registered = set()
    for task_name in TASK_GROUP_MODULES:
        module = _group_module(task_name)
        if task_name in module.TASK_EVALUATORS:
            registered.add(task_name)
    return frozenset(registered)
