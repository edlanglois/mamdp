"""Backwards Bandit Environment"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..types import Action, PolicyMatrix, State, Step
from .core import ModifiedActionMarkovDecisionProcess

__all__ = [
    "ExpInvertingDeterministicBandit",
    "LinearInvertingDeterministicBandit",
]


class LinearInvertingDeterministicBandit(ModifiedActionMarkovDecisionProcess):
    """A deterministic bandit environment with actions sampled prop. to 1 - pi(a)

    Rewards are (in order) [1, 0, -1]
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize an InvertingDeterministicBandit"""
        super().__init__(num_states=1, num_actions=3, reward_range=(-1, 1), seed=seed)

    def initial_state(self) -> State:
        return 0

    def step(self, state: State, action: Action) -> Step:
        assert state == 0
        reward = [1, 0, -1][action]
        return Step(state=state, action=action, next_state=0, reward=reward, done=True)

    def sample_action(self, state: State, policy: PolicyMatrix) -> Action:
        empirical_action_distribution = 1 - policy[state]
        empirical_action_distribution /= empirical_action_distribution.sum()
        return self._rand.choice(self.num_actions, p=empirical_action_distribution)


class ExpInvertingDeterministicBandit(ModifiedActionMarkovDecisionProcess):
    """A deterministic bandit environment with actions sampled prop. to exp(-3 pi(a|s))

    Rewards are (in order) [1, 0, -1]
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize an InvertingDeterministicBandit"""
        super().__init__(num_states=1, num_actions=3, reward_range=(-1, 1), seed=seed)

    def initial_state(self) -> State:
        return 0

    def step(self, state: State, action: Action) -> Step:
        assert state == 0
        reward = [1, 0, -1][action]
        return Step(state=state, action=action, next_state=0, reward=reward, done=True)

    def sample_action(self, state: State, policy: PolicyMatrix) -> Action:
        empirical_action_distribution = np.exp(-3 * policy[state])
        empirical_action_distribution /= empirical_action_distribution.sum()
        return self._rand.choice(self.num_actions, p=empirical_action_distribution)
