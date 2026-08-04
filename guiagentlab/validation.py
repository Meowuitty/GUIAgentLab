"""Read-only environment preflight and source-consistency checks."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from guiagentlab.data.datasets import normalize_goal, read_task_specs
from guiagentlab.env.client import MobileWorldClient
from guiagentlab.env.recovery import EndpointRecovery, RecoveryConfig


def read_endpoints(path: str | Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def environment_status(
    servers_file: str | Path,
    *,
    timeout: float = 5,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    endpoints = read_endpoints(servers_file)
    if limit is not None:
        endpoints = endpoints[:limit]

    def inspect(endpoint: str) -> dict[str, Any]:
        try:
            health = MobileWorldClient(endpoint, timeout=timeout).health()
            return {"endpoint": endpoint, "healthy": True, "health": health}
        except BaseException as exc:
            return {"endpoint": endpoint, "healthy": False, "error": str(exc)}

    results = []
    with ThreadPoolExecutor(max_workers=min(64, len(endpoints) or 1)) as executor:
        futures = {executor.submit(inspect, endpoint): endpoint for endpoint in endpoints}
        for future in as_completed(futures):
            results.append(future.result())
    order = {endpoint: index for index, endpoint in enumerate(endpoints)}
    return sorted(results, key=lambda item: order[item["endpoint"]])


def warm_environment_endpoints(
    endpoints: list[str],
    *,
    concurrency: int = 8,
    timeout: float = 180,
    probe_task: str = "AdjustBrightnessMaximumTask",
) -> list[dict[str, Any]]:
    """Load and tear down one snapshot on every endpoint before parallel rollout."""
    if concurrency <= 0:
        raise ValueError("warm-up concurrency must be positive")

    def warm(endpoint: str) -> dict[str, Any]:
        client = MobileWorldClient(endpoint, timeout=timeout, step_wait_time=0)
        initialized = False
        try:
            client.health()
            client.initialize_controller()
            client.initialize_task(probe_task)
            initialized = True
            client.screenshot_png(wait_to_stabilize=False)
            client.tear_down(probe_task)
            initialized = False
            return {"endpoint": endpoint, "warmed": True}
        except BaseException as exc:
            if initialized:
                try:
                    client.tear_down(probe_task)
                except BaseException:
                    pass
            return {"endpoint": endpoint, "warmed": False, "error": str(exc)}

    results = []
    with ThreadPoolExecutor(max_workers=min(concurrency, len(endpoints) or 1)) as executor:
        futures = {executor.submit(warm, endpoint): endpoint for endpoint in endpoints}
        for future in as_completed(futures):
            results.append(future.result())
    order = {endpoint: index for index, endpoint in enumerate(endpoints)}
    return sorted(results, key=lambda item: order[item["endpoint"]])


def _recover_endpoints(
    recovery: EndpointRecovery,
    endpoints: list[str],
    *,
    concurrency: int,
) -> tuple[list[str], dict[str, str]]:
    if concurrency <= 0:
        raise ValueError("GUIAGENTLAB_RECOVERY_CONCURRENCY must be positive")
    recovered: list[str] = []
    errors: dict[str, str] = {}
    if not endpoints:
        return recovered, errors
    with ThreadPoolExecutor(max_workers=min(concurrency, len(endpoints))) as executor:
        futures = {
            executor.submit(recovery.recover_endpoint, endpoint): endpoint
            for endpoint in endpoints
        }
        for future in as_completed(futures):
            endpoint = futures[future]
            try:
                future.result()
            except BaseException as exc:
                errors[endpoint] = str(exc)
            else:
                recovered.append(endpoint)
    return recovered, errors


def preflight(
    servers_file: str | Path,
    dataset_path: str | Path,
    *,
    active: int = 32,
    spares: int = 8,
    recover: bool = False,
    warm: bool = False,
    warm_concurrency: int = 8,
) -> dict[str, Any]:
    endpoints = read_endpoints(servers_file)
    expected = active + spares
    if len(endpoints) < expected:
        raise ValueError(
            f"expected at least {expected} endpoints ({active}+{spares}), got {len(endpoints)}"
        )
    endpoints = endpoints[:expected]
    status = environment_status(servers_file, limit=expected)
    failed = [item for item in status if not item["healthy"]]
    recovered_endpoints: list[str] = []
    recovery: EndpointRecovery | None = None
    recovery_concurrency = int(
        os.environ.get("GUIAGENTLAB_RECOVERY_CONCURRENCY", "1")
    )
    if recover:
        recovery_config = RecoveryConfig.from_environment()
        if recovery_config is None:
            raise RuntimeError(
                "endpoint recovery requires GUIAGENTLAB_STATE_DIR or "
                "GUIAGENTLAB_ENDPOINT_STATE"
            )
        recovery = EndpointRecovery(recovery_config)
        persisted = recovery.statuses(endpoints)
        recovery_targets = sorted(
            {str(item["endpoint"]) for item in failed}.union(persisted)
        )
        recovered, recovery_errors = _recover_endpoints(
            recovery,
            recovery_targets,
            concurrency=recovery_concurrency,
        )
        recovered_endpoints.extend(recovered)
        if recovery_targets:
            status = environment_status(servers_file, limit=expected)
            failed = [item for item in status if not item["healthy"]]
        if recovery_errors:
            raise RuntimeError(
                f"{len(recovery_errors)} MobileWorld endpoints failed outer recovery: "
                f"{list(recovery_errors.items())[:3]}"
            )
    if failed:
        raise RuntimeError(f"{len(failed)} MobileWorld endpoints are unhealthy: {failed[:3]}")

    warmed_endpoints = 0
    if warm:
        warm_status = warm_environment_endpoints(
            endpoints,
            concurrency=warm_concurrency,
        )
        warm_failed = [item for item in warm_status if not item["warmed"]]
        warmed_endpoints = len(warm_status) - len(warm_failed)
        if warm_failed and recovery is not None:
            warm_targets = [str(item["endpoint"]) for item in warm_failed]
            for item in warm_failed:
                recovery.mark(
                    str(item["endpoint"]),
                    "quarantined",
                    f"snapshot warm-up failed: {item.get('error', 'unknown error')}",
                )
            recovered, recovery_errors = _recover_endpoints(
                recovery,
                warm_targets,
                concurrency=recovery_concurrency,
            )
            recovered_endpoints.extend(recovered)
            if recovery_errors:
                raise RuntimeError(
                    f"{len(recovery_errors)} MobileWorld endpoints failed outer recovery "
                    f"after snapshot warm-up: {list(recovery_errors.items())[:3]}"
                )
            retry_status = warm_environment_endpoints(
                warm_targets,
                concurrency=min(warm_concurrency, len(warm_targets)),
            )
            warm_failed = [item for item in retry_status if not item["warmed"]]
            warmed_endpoints = len(endpoints) - len(warm_failed)
        if warm_failed:
            raise RuntimeError(
                f"{len(warm_failed)} MobileWorld endpoints failed snapshot warm-up: "
                f"{warm_failed[:3]}"
            )

    task_specs = read_task_specs(dataset_path)
    client = MobileWorldClient(endpoints[0], timeout=30)
    server_tasks = {task.name: task for task in client.tasks()}
    missing = sorted(set(task_specs.task_name).difference(server_tasks))
    if missing:
        raise RuntimeError(f"server is missing {len(missing)} dataset tasks: {missing[:10]}")

    # Goal equality detects a mismatched MobileWorld runtime.
    goal_mismatches = []
    for row in task_specs.itertuples(index=False):
        actual = client.goal(row.task_name)
        if normalize_goal(actual) != normalize_goal(row.goal):
            goal_mismatches.append(
                {"task_name": row.task_name, "expected": row.goal, "actual": actual}
            )
    if goal_mismatches:
        raise RuntimeError(
            f"server/source mismatch for {len(goal_mismatches)} tasks: {goal_mismatches[:3]}"
        )
    external = task_specs.requires_external_network.astype(bool)
    return {
        "endpoints": len(endpoints),
        "active": active,
        "spares": spares,
        "server_tasks": len(server_tasks),
        "benchmark_tasks": len(task_specs),
        "local_tasks": int((~external).sum()),
        "external_network_tasks": int(external.sum()),
        "goal_mismatches": 0,
        "recovered_endpoints": sorted(set(recovered_endpoints)),
        "warmed_endpoints": warmed_endpoints,
    }
