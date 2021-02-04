#!/usr/bin/env python
"""Train an agent on an environment."""
from __future__ import annotations

import argparse
import copy
import math
import pathlib
import shutil
import sys
from typing import IO, TYPE_CHECKING, Any, Dict, Iterable, Optional, Sequence

try:
    import tqdm
except ImportError:
    tqdm = None

import mamdp
from mamdp.cli import utils
from mamdp.rl import agents, envs, run

if TYPE_CHECKING:
    import numpy as np

    from mamdp.rl.agents import Agent
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
        "--initial-step-size", type=str, help="initial step size for UH-CMA-ES"
    )
    agent_parser.add_argument(
        "--num-sample-evaluations",
        type=int,
        help="initial number of evalautions per sample for UH-CMA-ES",
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

    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        required=True,
        help="write policies to this JSON file ",
    )
    parser.add_argument(
        "--save-policy-log-steps",
        default=0.1,
        type=float,
        help="how frequently to record the policy, in log10 steps",
    )
    parser.add_argument("-v", "--version", action="version", version=mamdp.__version__)
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="suppress standard output"
    )
    return parser.parse_args(argv)


def saved_policies(
    steps: Iterable[Step], agent: Agent, save_policy_log_steps: float
) -> Iterable[Dict[str, Any]]:
    """Yields dictionaries of policy information to save."""
    last_recorded = None
    log_last_recorded = float("-inf")
    for i, step in enumerate(steps):
        log_step = math.log10(i + 1)
        if log_step - log_last_recorded < save_policy_log_steps:
            continue

        log_last_recorded = log_step
        last_recorded = i
        yield {
            "step_index": i,
            "policy": agent.policy_matrix(greedy=True).tolist(),
        }
    if last_recorded != i:
        yield {"step_index": i, "policy": agent.policy_matrix(greedy=True).tolist()}


def train(
    env_name: str,
    agent_name: str,
    agent_kwargs: Dict[str, Any],
    num_steps: int,
    seed: int,
    output: IO[str],
    max_episode_steps: Optional[int] = None,
    save_policy_log_steps: float = 0.1,
    quiet: bool = False,
) -> np.ndarray:
    metadata = {
        "env": env_name,
        "agent": agent_name,
        "agent_kwargs": copy.deepcopy(agent_kwargs),
        "training_parameters": {
            "num_steps": num_steps,
            "seed": seed,
            "max_episode_steps": max_episode_steps,
        },
    }

    with utils.IncrementalJSONEncoder(output) as encoder:
        encoder.write("metadata", metadata)

        env, agent = run.prepare_env_agent(
            env=env_name,
            agent=agent_name,
            agent_kwargs=agent_kwargs,
            seed=seed,
        )
        training_steps = env.run(
            agent,
            learn=True,
            num_steps=num_steps,
            max_episode_steps=max_episode_steps,
        )

        if not quiet:
            utils.print_key_values(metadata)
            print()
            print("Training...")
            if tqdm is not None:
                training_steps = tqdm.tqdm(training_steps, total=num_steps)

        encoder.write_iterator(
            "policies",
            saved_policies(
                training_steps, agent, save_policy_log_steps=save_policy_log_steps
            ),
        )

    policy = agent.policy_matrix(greedy=True)
    if not quiet:
        print()
        print("Trained Policy")
        print(env.policy_string(policy))

    return policy


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run script.

    Args:
        argv: A list of argument strings to use instead of sys.argv.
    """
    args = parse_args(argv)
    agent_kwargs = {
        "discount_factor": args.discount_factor,
        "learning_rate": args.learning_rate,
        "exploration_rate": args.exploration_rate,
        "policy": args.policy_parametrization,
        "initial_step_size": args.initial_step_size,
        "initial_num_sample_evaluations": args.num_sample_evaluations,
        "scale_sample_evaluations": args.scale_sample_evaluations,
    }
    agent_kwargs = {k: v for k, v in agent_kwargs.items() if v is not None}
    with open(args.output, "w") as f:
        train(
            env_name=args.environment,
            agent_name=args.agent,
            agent_kwargs=agent_kwargs,
            num_steps=args.steps,
            seed=args.seed,
            save_policy_log_steps=args.save_policy_log_steps,
            max_episode_steps=args.max_episode_steps,
            quiet=args.quiet,
            output=f,
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
