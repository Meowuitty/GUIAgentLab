"""Shared types for task milestone evaluator groups."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mobile_world.runtime.controller import AndroidController

MilestoneEvaluator = Callable[[Any, AndroidController], dict[str, bool]]
