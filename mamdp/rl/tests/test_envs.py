"""Unit tests for envs.py"""
from __future__ import annotations

import itertools
import numbers
from typing import Optional

import numpy as np
import pytest

from mamdp.rl import envs
from mamdp.rl.agents import Agent
from mamdp.rl.types import Action, PolicyMatrix, State, Step


@pytest.fixture(params=envs.ENVIRONMENTS.values(), ids=envs.ENVIRONMENTS.keys())
def env_cls(request):
    return request.param


@pytest.fixture()
def env(env_cls):
    return env_cls(seed=1)


class _RandomAgent(Agent):
    """An agent that always acts randomly.

    Used to help test envs.
    """

    def act(self, state: State, greedy: bool = False) -> Action:
        return self._rand.choice(self.num_actions)

    def policy_matrix(self, greedy: bool = False) -> PolicyMatrix:
        return np.full(
            (self.num_states, self.num_actions), fill_value=1 / self.num_actions
        )

    def update(self, step: Step) -> None:
        pass


@pytest.fixture()
def env_agent(env):
    return _RandomAgent.init(
        num_states=env.num_states,
        num_actions=env.num_actions,
        seed=2,
    )


MDPS = {
    name: env
    for name, env in envs.ENVIRONMENTS.items()
    if isinstance(env(), envs.MarkovDecisionProcess)
}


@pytest.fixture(params=MDPS.values(), ids=MDPS.keys())
def mdp_cls(request):
    return request.param


@pytest.fixture()
def mdp(mdp_cls):
    return mdp_cls(seed=2)


def test_environment_num_states_ispositiveint(env):
    assert isinstance(env.num_states, numbers.Integral)
    assert env.num_states > 0


def test_environment_num_actions_ispositiveint(env):
    assert isinstance(env.num_actions, numbers.Integral)
    assert env.num_actions > 0


def test_environment_reward_range_isfloatrange(env):
    low, high = env.reward_range
    assert isinstance(low, numbers.Real)
    assert isinstance(high, numbers.Real)
    assert low <= high


def test_environment_initial_state_valid(env):
    s0 = env.initial_state()
    assert isinstance(s0, numbers.Integral)
    assert 0 <= s0 < env.num_states


def test_environment_initial_state_deterministic(env_cls):
    states = set()
    for _ in range(10):
        env = env_cls(seed=1)
        states.add(env.initial_state())

    assert len(states) == 1


def test_environment_seed_initial_state_deterministic(env_cls):
    env = env_cls()
    states = set()
    for _ in range(10):
        env.seed(1)
        states.add(env.initial_state())

    assert len(states) == 1


def _assert_valid_step(
    step: Step,
    env: envs.Environment,
    state: State,
    action: Optional[Action] = None,
):
    assert step.state == state
    assert 0 <= step.action < env.num_actions
    if action is not None:
        assert step.action == action
    assert 0 <= step.next_state < env.num_states
    min_reward, max_reward = env.reward_range
    assert min_reward <= step.reward <= max_reward
    assert isinstance(step.done, bool)


def test_environment_step_valid(env, env_agent):
    step = env.agent_step(0, env_agent)
    _assert_valid_step(step, env, state=0)


def test_environment_step_all_valid(env, env_agent):
    for state in range(min(100, env.num_states)):
        step = env.agent_step(state, env_agent)
        _assert_valid_step(step, env, state=state)


def test_environment_run_nocrash(env, env_agent):
    for step in itertools.islice(env.run(env_agent), 10):
        assert isinstance(step, Step)


def test_mdp_step_valid(mdp):
    step = mdp.step(0, 0)
    _assert_valid_step(step, mdp, state=0, action=0)


def test_mdp_step_all_valid(mdp):
    for state in range(min(10, mdp.num_states)):
        for action in range(min(10, mdp.num_actions)):
            step = mdp.step(state, action)
            _assert_valid_step(step, mdp, state=state, action=action)
