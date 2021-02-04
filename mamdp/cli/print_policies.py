#!/usr/bin/env python
"""Print policies from a policies.json file"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
from typing import Optional, Sequence

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
        "file",
        type=pathlib.Path,
        metavar="FILE",
        help="input policy.json file",
    )
    parser.add_argument(
        "--final", action="store_true", help="only print the final policy"
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run script.

    Args:
        argv: A list of argument strings to use instead of sys.argv.
    """
    args = parse_args(argv)

    with open(args.file, "r") as f:
        data = json.load(f)

    metadata = data["metadata"]
    env_name = metadata["env"]
    agent_name = metadata["agent"]
    env = ENVIRONMENTS[env_name]()
    print(env_name, agent_name)
    policies = data["policies"]
    if args.final:
        policies = [policies[-1]]
    for policy_info in policies:
        policy = np.asarray(policy_info["policy"])
        print()
        print(policy_info["step_index"])
        print(env.policy_string(policy))


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
