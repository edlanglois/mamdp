#!/usr/bin/env python
"""Evaluate some baseline policy-environment pairs"""
from __future__ import annotations

import argparse
import shutil
import sys
from typing import Optional, Sequence

import numpy as np

from mamdp.cli import utils
from mamdp.rl import run
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
    return parser.parse_args(argv)


def evaluate(
    env_name: str,
    policy_name: str,
    policy: np.ndarray,
    num_steps: int = 10_000,
    max_episode_steps: int = 100,
    seed: int = 0,
):
    print(env_name, policy_name)
    env = ENVIRONMENTS[env_name]()
    print(env.policy_string(policy))
    stats = run.evaluate_policy(
        env=env,
        policy=policy,
        num_steps=num_steps,
        max_episode_steps=max_episode_steps,
        seed=seed,
        progressbar=True,
    )
    utils.print_key_values(stats)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run script.

    Args:
        argv: A list of argument strings to use instead of sys.argv.
    """
    args = parse_args(argv)
    del args

    UP, LEFT, DOWN, RIGHT = range(4)

    whisky_gold = ENVIRONMENTS["whisky-gold"]()
    # Ignore the whisky: Go right unless on final column then up
    move_right = np.zeros((whisky_gold.num_states, whisky_gold.num_actions))
    move_right[:, RIGHT] = 1
    move_right[[5, 11, 17, 23, 29, 35], RIGHT] = 0
    move_right[[5, 11, 17, 23, 29, 35], UP] = 1
    evaluate(
        "whisky-gold",
        "move-right",
        move_right,
    )

    print()
    # Drink the whisky: Go up unless on top row then go right
    drunk_policy = np.zeros((whisky_gold.num_states, whisky_gold.num_actions))
    drunk_policy[0:6, RIGHT] = 1
    drunk_policy[6:18, UP] = 1
    drunk_policy[18:24, RIGHT] = 1
    drunk_policy[24:36, UP] = 1
    evaluate(
        "whisky-gold",
        "drunk",
        drunk_policy,
    )

    print()
    # Avoid the whisky: Move onto 2nd row then move right until last column then up
    sober_policy = np.zeros((whisky_gold.num_states, whisky_gold.num_actions))
    sober_policy[0:6, DOWN] = 1
    sober_policy[6:11, RIGHT] = 1
    sober_policy[11, UP] = 1
    sober_policy[12:18, UP] = 1
    sober_policy[18:24, DOWN] = 1
    sober_policy[24:29, RIGHT] = 1
    sober_policy[29, UP] = 1
    sober_policy[30:36, UP] = 1
    evaluate(
        "whisky-gold",
        "sober",
        sober_policy,
    )


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
