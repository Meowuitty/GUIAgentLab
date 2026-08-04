"""Agent 1: read-only Mastodon content-state milestone evaluators."""

from typing import Any

from mobile_world.runtime.controller import AndroidController
from mobile_world.tasks.milestones.types import MilestoneEvaluator

OWNED_TASKS = (
    "MastodonAddBookmarkTask",
    "MastodonRemoveBookmarkTask",
    "MastodonConditionalFavoTask",
    "MastodonPinTootsTask",
    "MastodonReportTask",
)


def _connect_to_postgres() -> tuple[Any, Any]:
    """Load the optional Mastodon runtime only when an evaluator is called."""
    from mobile_world.runtime.app_helpers import mastodon

    return mastodon.connect_to_postgres()


def _read_rows(query: str, params: tuple[object, ...]) -> list[tuple[Any, ...]]:
    """Execute one authoritative PostgreSQL SELECT without hiding backend errors."""
    connection, cursor = _connect_to_postgres()
    if connection is None or cursor is None:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()
        raise RuntimeError("Mastodon PostgreSQL connection is unavailable")

    try:
        cursor.execute(query, params)
        return list(cursor.fetchall())
    finally:
        cursor.close()
        connection.close()


def _status_ids_for_user(table: str, username: str) -> set[int]:
    """Read status IDs from one allow-listed Mastodon relationship table."""
    table_specs = {
        "bookmarks": ("bookmarks", "b"),
        "favourites": ("favourites", "f"),
        "status_pins": ("status_pins", "p"),
    }
    table_name, alias = table_specs[table]
    rows = _read_rows(
        f"""
        SELECT {alias}.status_id
        FROM {table_name} {alias}
        JOIN accounts a ON a.id = {alias}.account_id
        WHERE a.username = %s
          AND a.domain IS NULL
        """,
        (username,),
    )
    return {int(row[0]) for row in rows}


def evaluate_add_bookmark(task: Any, _controller: AndroidController) -> dict[str, bool]:
    """Read ``bookmarks.status_id`` rows owned by the local ``test`` account."""
    bookmarked_ids = _status_ids_for_user("bookmarks", task.EXPECTED_USERNAME)
    return {
        "kitty_cats_toot_115342692663348018_bookmarked": (
            115342692663348018 in bookmarked_ids
        ),
        "kitty_cats_toot_115359670141158913_bookmarked": (
            115359670141158913 in bookmarked_ids
        ),
    }


def evaluate_remove_bookmark(task: Any, _controller: AndroidController) -> dict[str, bool]:
    """Read target absence while preserving the non-target bookmark collection."""
    bookmarked_ids = _status_ids_for_user("bookmarks", task.EXPECTED_USERNAME)
    target_ids = set(task.EXPECTED_STATUS_ID)
    return {
        "pets_toot_115410818912936581_not_bookmarked": (
            115410818912936581 not in bookmarked_ids
        ),
        "pets_toot_115410836820181445_not_bookmarked": (
            115410836820181445 not in bookmarked_ids
        ),
        # The task's final evaluator rejects an empty bookmark collection. This
        # prevents deleting every bookmark from earning full milestone progress.
        "other_bookmarks_preserved": bool(bookmarked_ids.difference(target_ids)),
    }


def evaluate_conditional_favorite(
    task: Any,
    _controller: AndroidController,
) -> dict[str, bool]:
    """Read the required ``favourites.status_id`` rows for the local account."""
    favorite_ids = _status_ids_for_user("favourites", task.EXPECTED_USERNAME)
    return {
        "dogs_toot_115410810887077411_favorited": 115410810887077411 in favorite_ids,
        "dogs_toot_115410813905484454_favorited": 115410813905484454 in favorite_ids,
    }


def evaluate_pin_toots(task: Any, _controller: AndroidController) -> dict[str, bool]:
    """Read the target post's durable ``status_pins`` relation."""
    pinned_ids = _status_ids_for_user("status_pins", task.EXPECTED_USERNAME)
    expected_id = next(iter(task.EXPECTED_PINNED_TOOTS))
    return {
        "first_published_toot_pinned": expected_id in pinned_ids,
    }


def evaluate_report(task: Any, _controller: AndroidController) -> dict[str, bool]:
    """Read the latest target report, target status text, and account block rows."""
    report_rows = _read_rows(
        """
        SELECT
            r.comment,
            r.category,
            reporter.username,
            target.username
        FROM reports r
        JOIN accounts reporter ON reporter.id = r.account_id
        JOIN accounts target ON target.id = r.target_account_id
        WHERE %s = ANY(r.status_ids)
        ORDER BY r.created_at DESC
        LIMIT 1
        """,
        (task.EXPECTED_TOOT_ID,),
    )
    report = report_rows[0] if report_rows else None

    toot_rows = _read_rows(
        """
        SELECT s.text
        FROM statuses s
        WHERE s.id = %s
        """,
        (task.EXPECTED_TOOT_ID,),
    )
    toot_content = toot_rows[0][0] if toot_rows else None

    blocked_rows = _read_rows(
        """
        SELECT blocked.username
        FROM blocks b
        JOIN accounts blocker ON blocker.id = b.account_id
        JOIN accounts blocked ON blocked.id = b.target_account_id
        WHERE blocker.username = %s
          AND blocker.domain IS NULL
        """,
        (task.EXPECTED_REPORTER_USERNAME,),
    )
    blocked_usernames = {str(row[0]) for row in blocked_rows}

    return {
        "gas_leak_post_reported_by_test": (
            report is not None
            and report[2] == task.EXPECTED_REPORTER_USERNAME
            and report[3] == task.EXPECTED_BLOCKED_USERNAME
        ),
        "report_category_is_spam": (
            report is not None and report[1] == task.EXPECTED_CATEGORY
        ),
        "report_reason_matches_post_content": (
            report is not None
            and toot_content is not None
            and report[0] == toot_content
        ),
        "frank_blocked_by_test": (
            task.EXPECTED_BLOCKED_USERNAME in blocked_usernames
        ),
    }


# Agent 1 owns this mapping and the matching test file exclusively.
TASK_EVALUATORS: dict[str, MilestoneEvaluator] = {
    "MastodonAddBookmarkTask": evaluate_add_bookmark,
    "MastodonRemoveBookmarkTask": evaluate_remove_bookmark,
    "MastodonConditionalFavoTask": evaluate_conditional_favorite,
    "MastodonPinTootsTask": evaluate_pin_toots,
    "MastodonReportTask": evaluate_report,
}
