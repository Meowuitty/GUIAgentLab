"""Small, generic registration boundary for agentic-training extensions."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from typing import Any

_EXTENSIONS: dict[str, Callable[..., Any]] = {}
_LOADED_FACTORIES: set[str] = set()


def _load_extension_factories() -> None:
    """Load opt-in project registrations inside the current worker process."""
    for spec in os.environ.get("VERL_AGENT_EXTENSION_FACTORIES", "").split(","):
        spec = spec.strip()
        if not spec or spec in _LOADED_FACTORIES:
            continue
        module_name, separator, factory_name = spec.partition(":")
        if not separator or not module_name or not factory_name:
            raise ValueError(
                "VERL_AGENT_EXTENSION_FACTORIES entries must use module:function"
            )
        factory = getattr(importlib.import_module(module_name), factory_name)
        if not callable(factory):
            raise TypeError(f"agent extension factory {spec!r} is not callable")
        factory()
        _LOADED_FACTORIES.add(spec)


def register_agent_extension(
    name: str,
    callback: Callable[..., Any],
) -> Callable[..., Any]:
    """Register one process-local extension callback.

    Registration is intentionally explicit and happens before verl's Hydra
    entrypoint starts. Re-registering the same callable is idempotent; binding
    a different implementation to the same name is rejected.
    """
    normalized = name.strip()
    if not normalized:
        raise ValueError("agent extension name must not be empty")
    if not callable(callback):
        raise TypeError(f"agent extension {normalized!r} must be callable")
    existing = _EXTENSIONS.get(normalized)
    if existing is not None and existing is not callback:
        raise ValueError(f"agent extension {normalized!r} is already registered")
    _EXTENSIONS[normalized] = callback
    return callback


def get_agent_extension(name: str) -> Callable[..., Any]:
    """Return a required extension callback with a useful startup error."""
    if name not in _EXTENSIONS:
        _load_extension_factories()
    try:
        return _EXTENSIONS[name]
    except KeyError as error:
        raise RuntimeError(
            f"required agent extension {name!r} is not registered; "
            "start verl through the project adapter or configure "
            "VERL_AGENT_EXTENSION_FACTORIES"
        ) from error


def registered_agent_extensions() -> tuple[str, ...]:
    """Expose stable names for diagnostics without leaking callback objects."""
    return tuple(sorted(_EXTENSIONS))
