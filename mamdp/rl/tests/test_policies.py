"""Test policies.py"""
import functools

import numpy as np
import pytest
import scipy.optimize

from mamdp.rl import policies


@pytest.fixture(params=[0, 1])
def rng(request):
    return np.random.default_rng(seed=request.param)


@pytest.fixture(
    params=[
        functools.partial(policies.LogitsMatrixPolicy, num_states=3, num_actions=2),
    ]
)
def policy(request):
    return request.param()


def test_basic_calls_no_errors(policy: policies.DiscretePolicy):
    num_params = policy.num_params()
    x = np.zeros(num_params)
    policy.policy_matrix(x)
    policy.penalty(x)
    assert policy.penalty_grad(x).shape == x.shape


def test_policy_log_policy_grad_vs_empirical(
    policy: policies.DiscretePolicy, rng: np.random.Generator
):
    x0 = rng.uniform(-1, 1, policy.num_params())
    n = 10
    actions = rng.integers(policy.num_actions, size=n)
    states = rng.integers(policy.num_states, size=n)
    weights = rng.uniform(0, 1, n)

    def f(x):
        policy_matrix = policy.policy_matrix(x)
        return np.sum(weights * np.log(policy_matrix[states, actions]))

    def f_grad(x):
        return policy.log_policy_grad(
            actions=actions, states=states, weights=weights, x=x
        )

    assert scipy.optimize.check_grad(f, f_grad, x0) < 1e-6


def test_policy_penalty_grad_vs_empirical(
    policy: policies.DiscretePolicy, rng: np.random.Generator
):
    x0 = rng.uniform(-1, 1, policy.num_params())

    assert scipy.optimize.check_grad(policy.penalty, policy.penalty_grad, x0) < 1e-6
