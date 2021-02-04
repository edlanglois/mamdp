#!/usr/bin/env python
"""Summarize the final policies from a collection of training results
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import shutil
import sys
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from mamdp.rl.envs import ENVIRONMENTS


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: A list of argument strings to use instead of sys.argv.

    Returns:
        An `argparse.Namespace` object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "files",
        type=pathlib.Path,
        nargs="*",
        metavar="FILE",
        help="input policy.json files",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.99,
        help=(
            "count fraction of policies that are at least this "
            "certain about all actions"
        ),
    )
    parser.add_argument(
        "--argmax",
        action="store_true",
        help="Use the argmax action instead of thresholding. Overrides --threshold.",
    )
    parser.add_argument(
        "--state", type=int, help="Only summarize the policy at this state."
    )
    parser.add_argument(
        "--actions",
        type=int,
        nargs="+",
        help="Ignore all policy actions apart from these.",
    )
    parser.add_argument(
        "--env-policy-str",
        action="store_true",
        help="Environment-specific policy formatting.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run script.

    Args:
        argv: A list of argument strings to use instead of sys.argv.
    """
    args = parse_args(argv)
    if args.threshold < 0.5:
        raise ValueError(
            "A threshold < 0.5 can be satisfied by multiple actions simultaneously."
        )

    summary: Dict[
        str, Dict[str, Dict[Optional[Tuple[int, ...]], int]]
    ] = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(int))
    )

    state_restriction = args.state
    action_restriction = args.actions
    for file in args.files:
        with open(file, "r") as f:
            data = json.load(f)

        metadata = data["metadata"]
        env_name = metadata["env"]
        agent_name = metadata["agent"]
        policy = np.asarray(data["policies"][-1]["policy"])
        env = ENVIRONMENTS[env_name]()

        if action_restriction:
            for a in range(env.num_actions):
                if a not in action_restriction:
                    policy[:, a] = 0

        deterministic_policy: Optional[Tuple[int, ...]] = tuple(policy.argmax(axis=-1))
        # Check that the policy fully satisfies the threshold
        if not args.argmax and not ((policy > args.threshold).sum(axis=-1) == 1).all():
            deterministic_policy = None

        if deterministic_policy is not None and state_restriction is not None:
            deterministic_policy = (deterministic_policy[state_restriction],)

        summary[env_name][agent_name][deterministic_policy] += 1

    for env_name, env_summary in summary.items():
        for agent_name, agent_summary in env_summary.items():
            print()
            print(env_name, agent_name)
            total_policies = sum(agent_summary.values())
            for policy, count in agent_summary.items():
                if policy is None:
                    policy_str = "None"
                elif args.env_policy_str:
                    policy_ = np.zeros((env.num_states, env.num_actions))
                    if state_restriction is not None:
                        (action,) = policy
                        policy_[state_restriction, action] = 1
                    else:
                        policy_[np.arange(env.num_states), policy] = 1
                    policy_str = "\n" + env.policy_string(policy_)
                else:
                    if state_restriction is not None:
                        (action,) = policy
                        policy_state_actions: Iterable[Tuple[int, int]] = [
                            (state_restriction, action)
                        ]
                    else:
                        policy_state_actions = enumerate(policy)
                    policy_str = "".join(
                        [
                            "[",
                            "; ".join(
                                f"{env.state_string(s)} -> {env.action_string(a)}"
                                for (s, a) in policy_state_actions
                            ),
                            "]",
                        ]
                    )
                print(f"{count} / {total_policies} : {policy_str}")


if __name__ == "__main__":
    try:
        _np = sys.modules["numpy"]
    except KeyError:
        pass
    else:
        _np.set_printoptions(  # type: ignore
            linewidth=shutil.get_terminal_size().columns
        )
    main()
