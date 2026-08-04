"""Register GUIAgentLab methods before entering verl's Hydra application."""

from __future__ import annotations

import os
import sys

from guiagentlab.config import PROJECT_ROOT

_EXTENSION_FACTORY = (
    "guiagentlab.training.verl_adapter:install_verl_extensions"
)


def install_verl_extensions() -> None:
    """Install every GUIAgentLab callback at the single engine boundary."""
    verl_root = str(PROJECT_ROOT / "verl")
    if verl_root not in sys.path:
        sys.path.insert(0, verl_root)

    import guiagentlab.rollout.agent  # noqa: F401
    from guiagentlab.methods.admire import verl_admire_grpo_advantage
    from guiagentlab.methods.advantages import (
        hierarchical_action_weights,
        verl_trajectory_grpo_advantage,
    )
    from guiagentlab.methods.gigpo import verl_gigpo_advantage
    from guiagentlab.rollout.samples import validate_rollout_group_semantics
    from verl.experimental.agent_extensions import register_agent_extension
    from verl.trainer.ppo.core_algos import register_adv_est

    register_adv_est("admire_grpo")(verl_admire_grpo_advantage)
    register_adv_est("gigpo")(verl_gigpo_advantage)
    register_agent_extension(
        "hierarchical_action_weights",
        hierarchical_action_weights,
    )
    register_agent_extension(
        "trajectory_grpo_advantage",
        verl_trajectory_grpo_advantage,
    )
    register_agent_extension(
        "validate_rollout_batch",
        validate_rollout_group_semantics,
    )


def main() -> None:
    factories = [
        item
        for item in os.environ.get(
            "VERL_AGENT_EXTENSION_FACTORIES", ""
        ).split(",")
        if item
    ]
    if _EXTENSION_FACTORY not in factories:
        factories.append(_EXTENSION_FACTORY)
    os.environ["VERL_AGENT_EXTENSION_FACTORIES"] = ",".join(factories)
    install_verl_extensions()
    from verl.trainer.main_ppo import main as verl_main

    verl_main()


if __name__ == "__main__":
    main()
