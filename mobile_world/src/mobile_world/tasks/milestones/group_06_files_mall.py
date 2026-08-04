"""Agent 6: Files, document-reader, and Taodian milestone evaluators."""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from typing import Any

from mobile_world.runtime.controller import AndroidController
from mobile_world.runtime.utils.constants import ARTIFACTS_ROOT
from mobile_world.runtime.utils.helpers import execute_adb
from mobile_world.tasks.milestones.types import MilestoneEvaluator

OWNED_TASKS = (
    "InvoiceReceiptCopyTask",
    "CheckPuchasedItem",
    "SearchItemAndCheckoutTask",
    "CartManagementTask",
    "ReadQwen3PaperTask4",
)

_MISSING_DIRECTORY = "__MILESTONE_DIRECTORY_MISSING__"
_DIRECTORY_PRESENT = "__MILESTONE_DIRECTORY_PRESENT__"


def _read_directory_files(
    controller: AndroidController,
    directory: str,
) -> frozenset[str] | None:
    """Read one Android directory without changing it.

    ``None`` means the query succeeded and the directory does not exist. An ADB
    failure is an infrastructure error and therefore raises.
    """

    quoted_directory = shlex.quote(directory)
    script = (
        f"if [ -d {quoted_directory} ]; then "
        f"printf '{_DIRECTORY_PRESENT}\\n'; "
        f"ls -1A {quoted_directory}; "
        f"else printf '{_MISSING_DIRECTORY}\\n'; fi"
    )
    command = (
        f"adb -s {shlex.quote(controller.device)} shell {shlex.quote(script)}"
    )
    result = execute_adb(command, output=False)
    if not result.success:
        raise RuntimeError(
            f"failed to read Android directory {directory}: {result.error}"
        )

    lines = result.output.splitlines()
    if lines == [_MISSING_DIRECTORY]:
        return None
    if not lines or lines[0] != _DIRECTORY_PRESENT:
        raise RuntimeError(
            f"unexpected directory query response for {directory}: {result.output!r}"
        )
    return frozenset(lines[1:])


def _read_latest_task_callback(
    task_name: str,
    controller: AndroidController,
) -> Mapping[str, Any] | None:
    """Read the newest callback persisted for this task and device."""

    callback_dir = ARTIFACTS_ROOT / controller.device / "task_callbacks"
    if not callback_dir.exists():
        return None

    callback_files = list(callback_dir.glob(f"{task_name}_callback_*.json"))
    if not callback_files:
        return None
    callback_files.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)

    with callback_files[0].open(encoding="utf-8") as callback_file:
        callback = json.load(callback_file)
    if not isinstance(callback, dict):
        raise TypeError(f"{task_name} callback must be a JSON object")
    return callback


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"callback field {field!r} must be an object")
    return value


def _require_item_list(value: Any, field: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise TypeError(f"callback field {field!r} must be a list")
    items = []
    for index, item in enumerate(value):
        items.append(_require_mapping(item, f"{field}[{index}]"))
    return items


def _product_ids(items: list[Mapping[str, Any]], field: str) -> set[str]:
    product_ids = set()
    for index, item in enumerate(items):
        product_id = item.get("prodId")
        if not isinstance(product_id, str | int):
            raise TypeError(f"callback field {field}[{index}].prodId is invalid")
        product_ids.add(str(product_id))
    return product_ids


def evaluate_invoice_receipt_copy(
    task: Any,
    controller: AndroidController,
) -> dict[str, bool]:
    """Read Finance/invoice contents from the Android filesystem."""

    expected_names = {
        str(item["filename"])
        for item in task.invoice_files_created
        if item["is_this_month"]
    }
    if not expected_names:
        raise RuntimeError("InvoiceReceiptCopyTask has no initialized expected files")

    files = _read_directory_files(controller, "/sdcard/Finance/invoice")
    folder_exists = files is not None
    current_names = set() if files is None else set(files)
    return {
        "invoice_folder_created": bool(folder_exists),
        "expected_invoice_copied": bool(
            folder_exists and expected_names <= current_names
        ),
        # An empty directory must not receive correctness credit merely because
        # it vacuously contains no unexpected files.
        "invoice_folder_matches_expected": bool(
            folder_exists and current_names == expected_names
        ),
    }


def evaluate_check_puchased_item(
    task: Any,
    controller: AndroidController,
) -> dict[str, bool]:
    """Read the environment-owned terminal answer using the task's exact parser."""

    try:
        answer = int(str(controller.interaction_cache).strip())
    except (TypeError, ValueError):
        answer = None
    return {
        "correct_shoe_size_answer_submitted": answer == int(task.CORRECT_ANSWER),
    }


def evaluate_search_item_and_checkout(
    task: Any,
    controller: AndroidController,
) -> dict[str, bool]:
    """Read Taodian's persisted checkout callback."""

    callback = _read_latest_task_callback(type(task).__name__, controller)
    states = {
        "checkout_recorded": False,
        "target_tattoo_selected": False,
    }
    if callback is None:
        return states

    task_name = callback.get("task_name")
    if not isinstance(task_name, str):
        raise TypeError("Taodian callback task_name must be a string")
    if task_name != "提交订单":
        return states

    products = _require_item_list(callback.get("product_info"), "product_info")
    if not products:
        raise ValueError("Taodian checkout callback has no products")
    first_name = products[0].get("prodName")
    if not isinstance(first_name, str):
        raise TypeError("Taodian checkout product name must be a string")

    states["checkout_recorded"] = True
    states["target_tattoo_selected"] = "万圣节" in first_name and "临时纹身" in first_name
    return states


def evaluate_cart_management(
    task: Any,
    controller: AndroidController,
) -> dict[str, bool]:
    """Read Taodian's persisted cart-deletion callback."""

    callback = _read_latest_task_callback(type(task).__name__, controller)
    states = {
        "cart_deletion_recorded": False,
        "only_short_sleeve_items_selected": False,
        "remaining_cart_matches_expected": False,
    }
    if callback is None:
        return states

    task_name = callback.get("task_name")
    if not isinstance(task_name, str):
        raise TypeError("Taodian callback task_name must be a string")
    if task_name != "购物车删除选中":
        return states

    current_items = _require_item_list(
        callback.get("current_cart_items"), "current_cart_items"
    )
    deleted_items = _require_item_list(
        callback.get("items_to_delete"), "items_to_delete"
    )
    current_ids = _product_ids(current_items, "current_cart_items")
    deleted_ids = _product_ids(deleted_items, "items_to_delete")
    expected_left = {str(product_id) for product_id in task.items_left_prod_ids}

    states["cart_deletion_recorded"] = True
    states["only_short_sleeve_items_selected"] = bool(
        deleted_ids and deleted_ids.isdisjoint(expected_left)
    )
    states["remaining_cart_matches_expected"] = (
        current_ids - deleted_ids == expected_left
    )
    return states


def evaluate_read_qwen3_paper(
    task: Any,
    controller: AndroidController,
) -> dict[str, bool]:
    """Read the environment-owned terminal answer using the task's exact parser."""

    try:
        answer = float(controller.interaction_cache)
    except (TypeError, ValueError):
        answer = None
    return {
        "correct_vision_encoder_size_submitted": answer in task.CORRECT_ANSWERS,
    }


TASK_EVALUATORS: dict[str, MilestoneEvaluator] = {
    "InvoiceReceiptCopyTask": evaluate_invoice_receipt_copy,
    "CheckPuchasedItem": evaluate_check_puchased_item,
    "SearchItemAndCheckoutTask": evaluate_search_item_and_checkout,
    "CartManagementTask": evaluate_cart_management,
    "ReadQwen3PaperTask4": evaluate_read_qwen3_paper,
}
