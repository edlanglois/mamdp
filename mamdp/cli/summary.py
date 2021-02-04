#!/usr/bin/env python
"""Summarize training results"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
from typing import Optional, Sequence

import mamdp
from mamdp.cli import utils
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
        help="Input result files.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run script.

    Args:
        argv: A list of argument strings to use instead of sys.argv.
    """
    args = parse_args(argv)
    for file in args.files:
        with open(file, "r") as f:
            data, _, policy = mamdp.serialization.load_results(f)

        print()
        print("=" * 30)

        utils.print_key_values(data)

        env = ENVIRONMENTS[data["metadata"]["env"]]()
        print()
        print("Policy")
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
