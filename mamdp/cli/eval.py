#!/usr/bin/env python
"""Evaluate policies"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

try:
    import tqdm
except ImportError:
    tqdm = None

import mamdp
from mamdp.cli import utils
from mamdp.rl import run


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
        "input", type=pathlib.Path, help="policy.json files to evaluate"
    )
    parser.add_argument(
        "output", type=pathlib.Path, help="write evaluations to this JSON File"
    )
    parser.add_argument(
        "--num-steps", default=10_000, type=int, help="number of evaluation steps"
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        metavar="M",
        help="maximum number of steps per episode during training",
    )
    parser.add_argument(
        "--policy-dtype",
        default="float",
        type=str,
        help="policy floating-point type; lower precision can save work with caching",
    )
    parser.add_argument("--seed", default=0, type=int, help="evaluation seed")
    parser.add_argument("-v", "--version", action="version", version=mamdp.__version__)
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="suppress standard output"
    )
    return parser.parse_args(argv)


def evaluated_policies(
    policies: Iterable[Dict[str, Any]],
    policy_dtype: str,
    env: str,
    eval_params: Dict[str, Any],
    quiet: bool,
) -> Iterable[Dict[str, Any]]:
    progressbar = not quiet
    if progressbar and tqdm:
        policies = tqdm.tqdm(policies)

    # Dict mapping policy => results
    results_cache: Dict[Tuple[float, ...], Dict[str, Any]] = {}

    for policy_info in policies:
        step = policy_info["step_index"]
        policy = np.asarray(policy_info["policy"], dtype=policy_dtype)
        policy_key = tuple(policy.flat)

        try:
            results = results_cache[policy_key]
        except KeyError:
            results = run.evaluate_policy(
                env, policy, progressbar=progressbar, **eval_params
            )
            results_cache[policy_key] = results

        yield {
            "step_index": step,
            "policy_info": policy.tolist(),
            "evaluation": results,
        }


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run script.

    Args:
        argv: A list of argument strings to use instead of sys.argv.
    """
    args = parse_args(argv)
    eval_params = {
        "num_steps": args.num_steps,
        "max_episode_steps": args.max_episode_steps,
        "seed": args.seed,
    }

    with open(args.input, "r") as f:
        data = json.load(f)

    with open(args.output, "w") as f, utils.IncrementalJSONEncoder(f) as encoder:
        metadata = data["metadata"]
        metadata["evaluation_parameters"] = eval_params
        if not args.quiet:
            utils.print_key_values(metadata)
        encoder.write("metadata", metadata)

        encoder.write_iterator(
            "evaluations",
            evaluated_policies(
                data["policies"],
                policy_dtype=args.policy_dtype,
                env=metadata["env"],
                eval_params=eval_params,
                quiet=args.quiet,
            ),
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
