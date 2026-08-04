"""Agent 2: Mastodon social/profile milestone evaluators."""

from typing import Any

from mobile_world.runtime.controller import AndroidController
from mobile_world.tasks.milestones.types import MilestoneEvaluator

_FAMILY_LIST_QUERY = """
    WITH target_list AS (
        SELECT l.id, l.title, l.replies_policy
        FROM lists l
        JOIN accounts owner ON owner.id = l.account_id
        WHERE owner.username = %s
          AND owner.domain IS NULL
          AND l.title = %s
        ORDER BY l.created_at DESC
        LIMIT 1
    )
    SELECT target_list.title, target_list.replies_policy, member.username
    FROM target_list
    LEFT JOIN list_accounts la ON la.list_id = target_list.id
    LEFT JOIN accounts member ON member.id = la.account_id
    ORDER BY la.id
"""

_LATEST_TOOT_QUERY = """
    SELECT s.in_reply_to_id, s.text
    FROM statuses s
    JOIN accounts author ON author.id = s.account_id
    WHERE author.username = %s
    ORDER BY s.created_at DESC
    LIMIT 1
"""

_POLL_OPTIONS_QUERY = """
    SELECT p.options
    FROM statuses s
    JOIN polls p ON p.id = s.poll_id
    WHERE s.id = %s
"""

_FOLLOWING_USERNAMES_QUERY = """
    SELECT target.username
    FROM follows f
    JOIN accounts owner ON owner.id = f.account_id
    JOIN accounts target ON target.id = f.target_account_id
    WHERE owner.username = %s
      AND owner.domain IS NULL
    ORDER BY f.created_at DESC
"""

OWNED_TASKS = (
    "MastodonCreateListTask",
    "MastodonReplyTask",
    "MastodonRevisePollTask",
    "MastodonFollowTask",
    "MastodonUnfollowTask",
)


def _mastodon_backend() -> Any:
    """Load the optional Mastodon runtime dependencies only when evaluated."""
    from mobile_world.runtime.app_helpers import mastodon

    return mastodon


def _require_healthy_backend(backend: Any) -> None:
    """Raise instead of treating an unavailable Mastodon backend as task state."""
    if backend.is_mastodon_healthy() is not True:
        raise RuntimeError("Mastodon backend is not healthy")


def _query_rows(
    backend: Any,
    query: str,
    params: tuple[Any, ...],
) -> list[tuple[Any, ...]]:
    """Run one read-only query, preserving connection and SQL failures."""
    connection, cursor = backend.connect_to_postgres()
    if connection is None or cursor is None:
        raise ConnectionError("Could not connect to the Mastodon database")

    try:
        cursor.execute(query, params)
        return list(cursor.fetchall())
    finally:
        cursor.close()
        connection.close()


def evaluate_create_list(
    task: Any,
    _controller: AndroidController,
) -> dict[str, bool]:
    """Read ``lists``/``list_accounts`` using the task helper's SQL fields.

    The title and replies policy come from the selected ``lists`` row. Member
    usernames come from the joined ``list_accounts`` and ``accounts`` rows.
    """
    backend = _mastodon_backend()
    _require_healthy_backend(backend)
    rows = _query_rows(
        backend,
        _FAMILY_LIST_QUERY,
        (task.EXPECTED_USERNAME, task.EXPECTED_LIST_TITLE),
    )
    if not rows:
        return {
            "family_list_created": False,
            "reply_policy_followed_only": False,
            "all_family_members_added": False,
            "no_extra_family_members": False,
        }

    _, replies_policy, _ = rows[0]
    member_usernames = {
        member_username
        for _, _, member_username in rows
        if isinstance(member_username, str)
    }
    expected_members = set(task.EXPECTED_LIST_MEMBERS)
    return {
        "family_list_created": True,
        "reply_policy_followed_only": bool(
            replies_policy == task.EXPECTED_REPLIES_POLICY
        ),
        "all_family_members_added": bool(expected_members.issubset(member_usernames)),
        "no_extra_family_members": bool(member_usernames.issubset(expected_members)),
    }


def evaluate_reply(
    task: Any,
    _controller: AndroidController,
) -> dict[str, bool]:
    """Read the newest ``statuses`` row using the task helper's SQL fields.

    ``in_reply_to_id`` identifies the parent toot and ``text`` stores the
    persisted reply body, matching the task's own success evaluator.
    """
    backend = _mastodon_backend()
    _require_healthy_backend(backend)
    rows = _query_rows(
        backend,
        _LATEST_TOOT_QUERY,
        (task.EXPECTED_USERNAME,),
    )
    if not rows:
        return {
            "reply_targets_moussaka_toot": False,
            "reply_contains_expected_content": False,
        }

    in_reply_to_id, text = rows[0]
    return {
        "reply_targets_moussaka_toot": bool(
            in_reply_to_id == task.EXPECTED_BEING_REPLIED_TOOT_ID
        ),
        "reply_contains_expected_content": bool(
            isinstance(text, str)
            and task.EXPECTED_REPLY_CONTENT.lower() in text.lower()
        ),
    }


def evaluate_revise_poll(
    task: Any,
    _controller: AndroidController,
) -> dict[str, bool]:
    """Read ``statuses.poll_id`` and the referenced ``polls.options`` array.

    These are the same Mastodon database fields queried by the task's success
    evaluator. The predicates describe the requested removals, replacement,
    preserved options, and final option count without depending on edit order.
    """
    backend = _mastodon_backend()
    _require_healthy_backend(backend)
    rows = _query_rows(
        backend,
        _POLL_OPTIONS_QUERY,
        (task.EXPECTED_TOOT_ID,),
    )
    if not rows:
        return {
            "usa_option_removed": False,
            "brazil_replaced_with_canada": False,
            "russia_and_china_preserved": False,
            "three_poll_options_remain": False,
        }

    (options,) = rows[0]
    normalized_options = {
        option.lower().strip() for option in (options or [])
    }

    return {
        "usa_option_removed": bool("usa" not in normalized_options),
        "brazil_replaced_with_canada": bool(
            "brazil" not in normalized_options and "canada" in normalized_options
        ),
        "russia_and_china_preserved": bool(
            {"russia", "china"}.issubset(normalized_options)
        ),
        "three_poll_options_remain": bool(
            len(options or []) == task.EXPECTED_POLL_OPTIONS_COUNT
        ),
    }


def evaluate_follow(
    task: Any,
    _controller: AndroidController,
) -> dict[str, bool]:
    """Read the requested ``follows`` relationship to Robert's account."""
    backend = _mastodon_backend()
    _require_healthy_backend(backend)
    rows = _query_rows(
        backend,
        _FOLLOWING_USERNAMES_QUERY,
        (task.EXPECTED_USERNAME,),
    )
    return {
        # Following is one atomic, uniqueness-constrained backend mutation.
        # Counting the same relationship twice would inflate failed-trajectory
        # progress without representing another subgoal.
        "robert_followed": any(
            target_username == task.EXPECTED_TARGET_USERNAME
            for (target_username,) in rows
        ),
    }


def evaluate_unfollow(
    task: Any,
    _controller: AndroidController,
) -> dict[str, bool]:
    """Read current ``follows`` rows using ``get_following_users`` SQL fields.

    One predicate protects the three requested relationships; the other
    requires every relationship outside that exact keep-set to be absent.
    """
    backend = _mastodon_backend()
    _require_healthy_backend(backend)
    rows = _query_rows(
        backend,
        _FOLLOWING_USERNAMES_QUERY,
        (task.EXPECTED_USERNAME,),
    )
    actual_usernames = {target_username for (target_username,) in rows}
    expected_usernames = set(task.EXPECTED_KEEP_FOLLOWING_USERS)
    return {
        "latest_three_users_preserved": bool(
            expected_usernames.issubset(actual_usernames)
        ),
        "all_other_users_unfollowed": bool(
            actual_usernames.issubset(expected_usernames)
        ),
    }


# Agent 2 owns this mapping and the matching test file exclusively.
TASK_EVALUATORS: dict[str, MilestoneEvaluator] = {
    "MastodonCreateListTask": evaluate_create_list,
    "MastodonReplyTask": evaluate_reply,
    "MastodonRevisePollTask": evaluate_revise_poll,
    "MastodonFollowTask": evaluate_follow,
    "MastodonUnfollowTask": evaluate_unfollow,
}
