"""Exact backend milestone evaluators used by ADMIRE-GRPO."""

from mobile_world.tasks.milestones.registry import (
    TASK_GROUP_MODULES,
    evaluate_task_milestones,
    registered_task_names,
)

__all__ = [
    "TASK_GROUP_MODULES",
    "evaluate_task_milestones",
    "registered_task_names",
]
