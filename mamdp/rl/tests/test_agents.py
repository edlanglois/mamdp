"""Unit tests for agents.py"""
from __future__ import annotations

import numpy as np
import pytest

from mamdp.rl import agents
from mamdp.rl.types import Step


@pytest.fixture(params=agents.AGENTS.values(), ids=agents.AGENTS.keys())
def agent_cls(request):
    return request.param


@pytest.fixture(params=[False, True], ids=("nongreedy", "greedy"))
def greedy(request):
    return request.param


@pytest.fixture()
def agent23(agent_cls):
    return agent_cls.init(num_states=2, num_actions=3, seed=1)


def test_agent_init_act_update_nocrash(agent_cls):
    agent = agent_cls.init(num_states=2, num_actions=3, seed=1)
    s0 = 0
    a0 = agent.act(s0)
    s1 = 1
    step = Step(state=s0, action=a0, next_state=s1, reward=0.5, done=True)
    agent.update(step)


def test_agent_act(agent23, greedy):
    assert 0 <= agent23.act(0) < 3
    assert 0 <= agent23.act(1) < 3


def test_agent_policy_matrix_shape(agent23, greedy):
    assert agent23.policy_matrix(greedy).shape == (2, 3)


def test_agent_policy_matrix_sum(agent23, greedy):
    assert np.allclose(np.sum(agent23.policy_matrix(greedy), axis=-1), 1)


def test_agent_deterministic(agent_cls):
    actions = set()
    seed = 1
    for _ in range(10):
        agent = agent_cls.init(num_states=1, num_actions=10, seed=seed)
        actions.add(agent.act(0))
    assert len(actions) == 1
