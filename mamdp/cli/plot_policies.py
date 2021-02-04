#!/usr/bin/env python
"""Plot policy statistics"""
from __future__ import annotations

import argparse
import json
import operator
import pathlib
import shutil
import sys
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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
    parser.add_argument("--state", required=True, type=int, help="state index")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="save figure here; extension determinies filetype",
    )
    parser.add_argument(
        "--size",
        nargs=2,
        type=float,
        metavar=("W", "H"),
        help="figure dimensions in inches.",
    )
    return parser.parse_args(argv)


def agent_title(name: str) -> str:
    if name == "es":
        return "UH-CMA-ES"
    if name == "q-learning":
        return "Q-Learning"
    return name.replace("-", " ").title()


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run script.

    Args:
        argv: A list of argument strings to use instead of sys.argv.
    """
    args = parse_args(argv)
    results: List[Dict[str, Any]] = []
    for file in args.files:
        with open(file, "r") as f:
            data = json.load(f)

        metadata = data["metadata"]
        file_data = {
            "env": metadata["env"],
            "agent": metadata["agent"],
            **metadata["training_parameters"],
        }
        for eval_info in data["evaluations"]:
            results.append(
                {
                    **file_data,
                    "step": eval_info["step_index"],
                    **eval_info["evaluation"],
                }
            )

    data = pd.DataFrame.from_records(results)
    data["target_state_visit_probability"] = data["state_visit_probability"].map(
        operator.itemgetter(args.state)
    )
    del results

    sns.set_theme(style="whitegrid")
    if args.size:
        plt.figure(figsize=args.size)
    ax = sns.lineplot(
        x="step",
        y="target_state_visit_probability",
        style="agent",
        hue="agent",
        data=data,
        linewidth=3,
    )
    ax.set_xlim([1, data["step"].max() + 1])
    ax.set(xscale="log", xlabel="Training Step", ylabel="Probability")
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(
        handles,
        [agent_title(name) for name in labels],
        loc="upper center",
        ncol=2,
        fancybox=True,
        bbox_to_anchor=(0.5, 1.23),
        # shadow=True,
        # prop={"size": 8},
    )

    if args.output:
        if str(args.output).endswith(".pgf"):
            plt.rcParams.update(
                {
                    "font.family": "serif",
                    "font.serif": [],  # use latex default serif font
                }
            )
        plt.savefig(args.output, bbox_inches="tight")
    else:
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
