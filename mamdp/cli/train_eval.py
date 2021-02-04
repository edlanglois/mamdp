#!/usr/bin/env python
"""Train and evaluate an agent on an environment."""
from __future__ import annotations

import argparse
import copy
import pathlib
import shutil
import sys
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

try:
    import tqdm
except ImportError:
    tqdm = None

import mamdp
from mamdp.cli import utils
from mamdp.rl import agents, envs, run

if TYPE_CHECKING:
    import numpy as np

    from mamdp.rl.types import Step


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

    env_parser = parser.add_argument_group("Environment")
    env_parser.add_argument(
        "-e",
        "--environment",
        choices=envs.ENVIRONMENTS,
        default="chain-mdp",
        metavar="ENV",
        help="the environment to train on",
    )

    agent_parser = parser.add_argument_group("Agent")
    agent_parser.add_argument(
        "-a",
        "--agent",
        choices=agents.AGENTS,
        default="random",
        metavar="AGENT",
        help="the agent to train",
    )
    agent_parser.add_argument(
        "--discount-factor",
        type=float,
        help="rate of discounting future rewards",
    )
    agent_parser.add_argument(
        "--learning-rate",
        type=float,
        help="size of updates for some agents",
    )
    agent_parser.add_argument(
        "--exploration-rate",
        type=float,
        help="how often some agents take exploratory actions",
    )
    agent_parser.add_argument(
        "--policy-parametrization",
        type=str,
        help="how some agents parametrize their policy",
    )
    agent_parser.add_argument(
        "--num-sample-evaluations",
        type=int,
        help="initial number of evalautions per sample for UH-ES",
    )
    agent_parser.add_argument(
        "--scale-sample-evaluations",
        type=float,
        help="UH-ES scale factor on changes to num evaluations per sample",
    )

    training_parser = parser.add_argument_group("Training")
    training_parser.add_argument(
        "--steps",
        type=int,
        default=100_000,
        metavar="N",
        help="number of training steps",
    )
    training_parser.add_argument(
        "--max-episode-steps",
        type=int,
        metavar="M",
        help="maximum number of steps per episode during training",
    )
    training_parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="seed for the training run; derives agent & env seeds",
    )

    eval_parser = parser.add_argument_group("Evaluation")
    eval_parser.add_argument(
        "--eval-steps",
        type=int,
        default=10_000,
        metavar="N",
        help="number of evaluation steps",
    )
    eval_parser.add_argument(
        "--eval-seed",
        type=int,
        default=2,
        help="seed for the evaluation run; derives agent & env seeds",
    )

    parser.add_argument(
        "-o", "--output", type=pathlib.Path, help="Write results to this file as JSON."
    )
    parser.add_argument("-v", "--version", action="version", version=mamdp.__version__)
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="suppress standard output"
    )
    return parser.parse_args(argv)


def train_eval(
    env_name: str,
    agent_name: str,
    agent_kwargs: Dict[str, Any],
    num_training_steps: int,
    training_seed: int,
    max_episode_steps: Optional[int] = None,
    num_eval_steps: Optional[int] = None,
    eval_seed: int = 2,
    quiet: bool = False,
) -> Tuple[Dict[str, Any], List[Step], np.ndarray]:
    info: Dict[str, Any] = {}
    info["metadata"] = {
        "env": env_name,
        "agent": agent_name,
        "agent_kwargs": copy.deepcopy(agent_kwargs),
        "timestamp": time.time(),
        "training_parameters": {
            "num_steps": num_training_steps,
            "seed": training_seed,
            "max_episode_steps": max_episode_steps,
        },
        "evaluation_parameters": {
            "num_steps": num_eval_steps,
            "seed": eval_seed,
            "max_episode_steps": None,
        },
    }

    env, agent = run.prepare_env_agent(
        env=env_name,
        agent=agent_name,
        agent_kwargs=agent_kwargs,
        seed=training_seed,
    )
    training_steps_iter = env.run(
        agent,
        learn=True,
        num_steps=num_training_steps,
        max_episode_steps=max_episode_steps,
    )

    if not quiet:
        utils.print_key_values(info)
        print()
        print("Training...")
        if tqdm is not None:
            training_steps_iter = tqdm.tqdm(
                training_steps_iter, total=num_training_steps
            )

    training_steps = list(training_steps_iter)

    policy = agent.policy_matrix(greedy=True)
    if not quiet:
        print()
        print("Trained Policy")
        print(env.policy_string(policy))
        try:
            action_values = agent.q  # type: ignore
        except AttributeError:
            pass
        else:
            print()
            print("State-Action Values")
            print(env.action_values_string(action_values))

    if num_eval_steps:
        if not quiet:
            print()
            print("Evaluating...")

        evaluation_statistics = run.evaluate_policy(
            env=env_name,
            policy=policy,
            num_steps=num_eval_steps,
            max_episode_steps=None,
            seed=eval_seed,
            progressbar=not quiet,
        )
        info["evaluation_statistics"] = evaluation_statistics

        if not quiet:
            print()
            utils.print_key_values(evaluation_statistics)
    return info, training_steps, policy


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run script.

    Args:
        argv: A list of argument strings to use instead of sys.argv.
    """
    args = parse_args(argv)
    # If writing to a file then try opening it first to make sure we can before we spend
    # time running the experiment.
    output_path = args.output
    if output_path:
        with open(output_path, "a") as f:
            pass

    agent_kwargs = {
        "discount_factor": args.discount_factor,
        "learning_rate": args.learning_rate,
        "exploration_rate": args.exploration_rate,
        "policy": args.policy_parametrization,
        "num_sample_evaluations": args.num_sample_evaluations,
        "scale_sample_evaluations": args.scale_sample_evaluations,
    }
    agent_kwargs = {k: v for k, v in agent_kwargs.items() if v is not None}
    info, training_steps, policy = train_eval(
        env_name=args.environment,
        agent_name=args.agent,
        agent_kwargs=agent_kwargs,
        num_training_steps=args.steps,
        max_episode_steps=args.max_episode_steps,
        training_seed=args.seed,
        num_eval_steps=args.eval_steps,
        eval_seed=args.eval_seed,
        quiet=args.quiet,
    )
    if output_path:
        with open(output_path, "w") as f:
            if not args.quiet:
                print("Writing results to:", pathlib.Path(f.name).resolve(), sep="\n")
            mamdp.serialization.dump_results(info, training_steps, policy, f)


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
