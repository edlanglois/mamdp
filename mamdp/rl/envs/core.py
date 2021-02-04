"""Core Environments"""
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Generator, Iterable, Optional, Set, Tuple

import numpy as np
import tabulate

from ..types import Action, PolicyMatrix, State, Step

if TYPE_CHECKING:
    from ..agents import Agent


__all__ = [
    "Environment",
    "MarkovDecisionProcess",
    "ModifiedActionMarkovDecisionProcess",
    "reward_range",
    "TabularMarkovDecisionProcess",
]


class Environment:
    """Base class of a discrete environment."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        reward_range: Tuple[float, float] = (float("-inf"), float("inf")),
        seed: Optional[int] = None,
    ):
        """Initialize an Environment

        Args:
            num_states: Number of environment states.
            num_actions: Number of environment actions.
                The same number of actions are available in all states.
            reward_range: The minimum and maximum possible reward values.
            seed: Optional random seed for deterministic randomness.
        """
        self.num_states = num_states
        self.num_actions = num_actions

        min_reward, max_reward = reward_range
        if min_reward > max_reward:
            raise ValueError(f"Reward range {reward_range} is empty.")
        self.reward_range = reward_range
        self.seed(seed)

    def initial_state(self) -> State:
        """Sample an initial state."""
        raise NotImplementedError

    def agent_step(self, state: State, agent: Agent) -> Step:
        """Perform one step with the given agent."""
        raise NotImplementedError

    def run(
        self,
        agent: Agent,
        state: Optional[State] = None,
        learn: bool = True,
        num_steps: Optional[int] = None,
        max_episode_steps: Optional[int] = None,
    ) -> Generator[Step, None, None]:
        """Run the environment with the given agent.

        Args:
            agent: An agent to interact with the environment.
            state: Optional initial state. Uses initial_state() by default.
            learn: Whether to update the agent while running.
            num_steps: Number of steps to run. Runs forever if None.
            max_episode_steps: Optional maximum number of steps per episode.
                Force `step.done = True` when the limit is reached.

        Yields:
            The curren step. This happens after any optional agent update.
        """
        state = self.initial_state()
        if num_steps is None:
            step_counter: Iterable[int] = itertools.count()
        else:
            step_counter = range(num_steps)
        previous_episode_end = -1
        for i in step_counter:
            step = self.agent_step(state, agent)
            if (
                max_episode_steps is not None
                and i - previous_episode_end >= max_episode_steps
            ):
                step = step.replace(done=True)
            if learn:
                agent.update(step)
            yield step
            if step.done:
                state = self.initial_state()
                previous_episode_end = i
            else:
                state = step.next_state

    def action_string(self, action: Action) -> str:
        """Represent the action as an (ideally short) human-readable string."""
        return str(action)

    def state_string(self, state: State) -> str:
        """Represent the state as an (ideally short) human-readable string."""
        return str(state)

    def policy_string(self, policy: PolicyMatrix) -> str:
        """Represent the policy as a human-readable string."""
        action_strings = [self.action_string(a) for a in range(self.num_actions)]
        state_strings = [self.state_string(s) for s in range(self.num_states)]
        return tabulate.tabulate(
            policy, headers=action_strings, showindex=state_strings
        )

    def action_values_string(self, action_values: np.ndarray) -> str:
        """Represent an array of action values as a string."""
        action_strings = [self.action_string(a) for a in range(self.num_actions)]
        state_strings = [self.state_string(s) for s in range(self.num_states)]
        return tabulate.tabulate(
            action_values, headers=action_strings, showindex=state_strings
        )

    def seed(self, seed: Optional[int] = None) -> None:
        """Seed any pseudorandom number generators used by the environment."""
        self._rand = np.random.RandomState(seed)


class ModifiedActionMarkovDecisionProcess(Environment):
    """Markov Decision Process with a Modified Action."""

    def agent_step(self, state: State, agent: Agent) -> Step:
        return self.policy_step(state, agent.policy_matrix())

    def policy_step(self, state: State, policy: PolicyMatrix) -> Step:
        """Perform one step with the given policy matrix."""
        action = self.sample_action(state, policy)
        return self.step(state, action)

    def sample_action(self, state: State, policy: PolicyMatrix) -> Action:
        """Sample an action given a state and a policy."""
        raise NotImplementedError

    def _sample_policy_action(self, state: State, policy: PolicyMatrix) -> Action:
        """Sample an action according to the policy."""
        return self._rand.choice(self.num_actions, p=policy[state])

    def step(self, state: State, action: Action) -> Step:
        """Sample a transition for the given state and action."""
        raise NotImplementedError


class MarkovDecisionProcess(ModifiedActionMarkovDecisionProcess):
    """A Markov Decision Process"""

    def agent_step(self, state: State, agent: Agent) -> Step:
        return self.step(state, agent.act(state))

    def sample_action(self, state: State, policy: PolicyMatrix) -> Action:
        return self._sample_policy_action(state, policy)


class TabularMarkovDecisionProcess(MarkovDecisionProcess):
    """An MDP defined by parameter tables."""

    def __init__(
        self,
        transitions: np.ndarray,
        rewards: np.ndarray,
        initial_states: Iterable[int],
        terminal_states: Set[int],
        seed: Optional[int] = None,
    ):
        """Initialize a TabularMarkovDecisionProcess

        Args
            transitions: An array of shape num_states x num_actions x num_states
                where transitions[s, a, s'] = Pr(s' | s, a)

            rewards: An array of transition rewards. For transition (s, a, s'), either
                r = rewards[s'] or
                r = rewards[s, a] or
                r = rewards[s, a, s']
                depending on the number of dimensions of `rewards`.

            initial_states: A set of initial states.
            terminal_states: A set of terminal states.
            seed: Optional random seed.
        """
        num_states, num_actions, _num_states = transitions.shape
        if _num_states != num_states:
            raise ValueError(
                "Transitions must have shape num_states x num_actions x num_states"
            )
        if not np.allclose(np.sum(transitions, axis=-1), 1):
            raise ValueError("Require sum(transitions[s, a, :]) = 1 for every s, a")

        super().__init__(
            num_states=num_states,
            num_actions=num_actions,
            reward_range=reward_range(rewards),
            seed=seed,
        )

        self.transitions = transitions
        self.rewards = rewards
        self.initial_states = list(initial_states)
        for state in self.initial_states:
            if state >= num_states or state < 0:
                raise IndexError(
                    f"Initial state {state} is not in {{0, ..., {num_states - 1}}}"
                )
        self.terminal_states = terminal_states
        for state in self.terminal_states:
            if state >= num_states or state < 0:
                raise IndexError(
                    f"Terminal state {state} is not in {{0, ..., {num_states - 1}}}"
                )

    def initial_state(self) -> State:
        return self._rand.choice(self.initial_states)

    def step(self, state: State, action: Action) -> Step:
        next_state = self._rand.choice(
            self.num_states, p=self.transitions[state, action]
        )
        if self.rewards.ndim == 1:
            reward = self.rewards[next_state]
        elif self.rewards.ndim == 2:
            reward = self.rewards[state, action]
        elif self.rewards.ndim == 3:
            reward = self.rewards[state, action, next_state]
        else:
            raise ValueError("rewards must have 1, 2 or 3 dimensions.")

        done = next_state in self.terminal_states
        return Step(
            state=state, action=action, next_state=next_state, reward=reward, done=done
        )


def reward_range(rewards: Iterable[float]) -> Tuple[float, float]:
    """Calculate a reward range from a set of rewards"""
    rewards_iter = iter(rewards)
    try:
        reward = next(rewards_iter)
    except StopIteration:
        return (float("-inf"), float("inf"))

    min_reward = max_reward = reward
    for reward in rewards_iter:
        min_reward = min(min_reward, reward)
        max_reward = max(max_reward, reward)
    return min_reward, max_reward
