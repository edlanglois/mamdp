"""Functions to deal with runs."""
from __future__ import annotations

import random
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Tuple,
    Type,
    Union,
)

import numpy as np

from .agents import AGENTS, Agent, FixedPolicyAgent
from .envs import ENVIRONMENTS, Environment

try:
    import tqdm
except ImportError:
    tqdm = None

if TYPE_CHECKING:
    from .types import Step


def prepare_env_agent(
    env: Union[str, Callable[[], Environment], Environment],
    agent: Union[str, Type[Agent]],
    agent_kwargs: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> Tuple[Environment, Agent]:
    """Prepare an environment and an agent for a run.

    Args:
        env: Environment (instance, class, or name)
        agent: Agent (class or name)
        agent_kwargs: Additional keyword arguments available to be used when
            initializing the agent.
        seed: If set, seed the environment and agent with derived seeds.

    Returns:
        env: The environment instance.
        agent: The agent instance.
    """
    env_seed, agent_seed = _split_seed(seed)

    if isinstance(env, str):
        env = ENVIRONMENTS[env]
    if callable(env):
        env = env()  # type: ignore
    env.seed(env_seed)

    if isinstance(agent, str):
        agent = AGENTS[agent]
    if agent_kwargs is None:
        agent_kwargs = {}
    agent_instance = agent.init(
        num_states=env.num_states,
        num_actions=env.num_actions,
        seed=agent_seed,
        **agent_kwargs,
    )

    return env, agent_instance


def evaluate_policy(
    env: Union[str, Callable[[], Environment], Environment],
    policy: np.ndarray,
    num_steps: int,
    max_episode_steps: Optional[int] = None,
    seed: Optional[int] = None,
    progressbar: bool = False,
) -> Dict[str, float]:
    """Evaluate a policy matrix on an environment."""
    env_seed, agent_seed = _split_seed(seed)

    if isinstance(env, str):
        env = ENVIRONMENTS[env]
    if callable(env):
        env = env()  # type: ignore
    env.seed(env_seed)
    agent = FixedPolicyAgent(policy, seed=agent_seed)

    steps = env.run(
        agent, learn=False, num_steps=num_steps, max_episode_steps=max_episode_steps
    )
    if progressbar and tqdm is not None:
        steps = tqdm.tqdm(steps, total=num_steps)

    total_num_steps = 0
    total_reward: float = 0
    total_num_episodes = 0
    total_episode_reward: float = 0
    total_episode_lengths = 0
    total_visited_states: np.ndarray = np.zeros(env.num_states, dtype=int)
    episode_num_steps = 0
    episode_reward: float = 0
    episode_visited_states: np.ndarray = np.zeros(env.num_states, dtype=bool)
    for step in steps:
        total_num_steps += 1
        total_reward += step.reward
        episode_num_steps += 1
        episode_reward += step.reward
        episode_visited_states[step.state] = True
        episode_visited_states[step.next_state] = True
        if step.done:
            total_num_episodes += 1
            total_episode_reward += episode_reward
            total_episode_lengths += episode_num_steps
            total_visited_states += episode_visited_states
            episode_num_steps = 0
            episode_reward = 0
            episode_visited_states[:] = False

    stats = {}
    stats["mean_step_reward"] = _nandiv(total_reward, total_num_steps)
    stats["mean_episode_reward"] = _nandiv(total_episode_reward, total_num_episodes)
    stats["mean_episode_length"] = _nandiv(total_episode_lengths, total_num_episodes)
    # Probability that each state is visited in an episode
    stats["state_visit_probability"] = (
        total_visited_states / total_num_episodes
    ).tolist()
    return stats


def _nandiv(a: float, b: float) -> float:
    """Division that returns NaN if b == 0"""
    try:
        return a / b
    except ZeroDivisionError:
        return float("NaN")


def _split_seed(seed: Optional[int]) -> Tuple[Optional[int], Optional[int]]:
    if seed is None:
        return None, None
    rand = random.Random(seed)
    a = rand.randint(0, 2 ** 32 - 1)
    b = rand.randint(0, 2 ** 32 - 1)
    return a, b


def episodes(steps: Iterable[Step]) -> Iterable[Iterable[Step]]:
    """Group steps into episodes."""
    while True:
        yield _take_episode(steps)


def _take_episode(steps: Iterable[Step]) -> Iterable[Step]:
    """Yields steps until the end of the episode."""
    for step in steps:
        yield step
        if step.done:
            break
