#!/usr/bin/env python
"""Plot training rewards (output of train_eval)"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import mamdp


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
        help="input result files",
    )
    parser.add_argument(
        "--smoothing",
        type=int,
        metavar="WINDOW_SIZE",
        default=1000,
        help="size of rolling window smoothing applied to plot",
    )
    parser.add_argument(
        "--samples",
        type=int,
        metavar="N",
        default=1000,
        help="number of sample points to plot for each trajectory",
    )
    parser.add_argument(
        "--logx", action="store_true", help="plot x-axis on a log scale"
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run script.

    Args:
        argv: A list of argument strings to use instead of sys.argv.
    """
    args = parse_args(argv)
    step_dataframes = []
    keys = []
    for file in args.files:
        with open(file, "r") as f:
            data, steps, _ = mamdp.serialization.load_results(f)
            steps.index.name = "Step Index"

        metadata = data["metadata"]
        keys.append(
            (
                metadata["env"],
                metadata["agent"],
                metadata["training_parameters"]["seed"],
            )
        )
        step_dataframes.append(steps)

    steps = pd.concat(step_dataframes, keys=keys, names=("Env", "Agent", "Seed"))

    rewards = steps["reward"]
    if args.smoothing:
        rewards = rewards.rolling(args.smoothing).mean()

    if args.samples:
        rewards = rewards[:: len(rewards) // args.samples]

    sns.set()
    ax = sns.lineplot(
        x=rewards.index.get_level_values(3),
        y=rewards,
        hue=rewards.index.get_level_values(1),
    )
    if args.logx:
        ax.set(xscale="log")
    plt.show()


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
