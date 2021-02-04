"""Policy parametrizations."""

from __future__ import annotations

from typing import Optional

import numpy as np


class DiscretePolicy:
    """Discrete policy interface."""

    def __init__(self, num_states: int, num_actions: int):
        self.num_states = num_states
        self.num_actions = num_actions

    def num_params(self) -> int:
        """Parameter vector size."""
        raise NotImplementedError

    def policy_matrix(self, x: np.array, out: Optional[np.array] = None) -> np.array:
        """The policy matrix for a given parameter vector.

        Args:
            x: Policy parameter vector. A float array of shape [NUM_PARAMS]
            out: Optional array in which to store the outputs.

        Returns:
            The policy matrix. A matrix of shape [NUM_STATES, NUM_ACTIONS]
        """
        raise NotImplementedError

    def log_policy_grad(
        self, actions: np.array, states: np.array, weights: np.array, x: np.array
    ) -> np.array:
        """The gradient vector d/dx ∑ᵢ wᵢ log Pr(actionᵢ | stateᵢ, x)

        Args:
            actions: Action indices. An integer array of shape [NUM_SAMPLES]
            states: State indices. An integer array of shape [NUM_SAMPLES]
            x: Policy parameter vector. A float array of shape [NUM_PARAMS]
            weights: Optional weights for the terms in the sum.
                A float array of shape [NUM_SAMPLES]

        Returns:
            Batched gradient vectors. A float array of shape [NUM_SAMPLES, NUM_PARAMS]
        """
        raise NotImplementedError

    def penalty(self, x: np.array) -> float:
        """A penalty term to encourage well-behaved or canonical representations.

        Should only penalize redundant parameterizations, should not limit the
        representational power of the policy.

        Args:
            x: Policy parameter vector. A float array of shape [NUM_PARAMS]

        Returns:
            The penalty value (larger is worse).
        """
        raise NotImplementedError

    def penalty_grad(self, x: np.array) -> np.array:
        """The gradient of the penalty term with respect to policy parameters.

        Args:
            x: Policy parameter vector. A float array of shape [NUM_PARAMS]

        Returns:
            The penalty gradient. A float array with the same shape as x.
        """
        raise NotImplementedError


def softmax(a: np.array, axis: Optional[int] = None, out: Optional[np.array] = None):
    """Softmax of elements over a given axis.

    Args:
        a: Input array.
        axis: The axis or axes over which the softmax sum is performed.
            If axis = None (default), sums over the entire array.
        out: Optional array in which to store the outputs.

    Returns:
        An array with the same shape as a.
    """
    offsets = np.max(a, axis=axis, keepdims=True)
    out = np.exp(a - offsets, out=out)
    sums = np.sum(out, axis=axis, keepdims=True)
    out /= sums
    return out


class LogitsMatrixPolicy(DiscretePolicy):
    """A policy parametrized as a matrix of logits."""

    def num_params(self):
        return self.num_states * self.num_actions

    def policy_matrix(self, x: np.array, out: Optional[np.array] = None) -> np.array:
        return softmax(x.reshape(self.num_states, self.num_actions), axis=-1, out=out)

    def log_policy_grad(
        self, actions: np.array, states: np.array, weights: np.array, x: np.array
    ) -> np.array:
        if actions.shape != states.shape:
            raise ValueError("actions and states have different shapes")
        if actions.shape != weights.shape:
            raise ValueError("actions and weights have different shapes")
        (num_samples,) = actions.shape

        x_mat = x.reshape(self.num_states, self.num_actions)
        softmax_x_mat = softmax(x_mat, axis=-1)
        out = np.zeros_like(x_mat)
        for state, action, weight in zip(states, actions, weights):
            out[state, action] += weight
            out[state] -= weight * softmax_x_mat[state]
        return out.reshape(-1)

    def penalty(self, x: np.array) -> float:
        # Penalize any rows where the maximum action logit is too far from 0
        # Uses hinge loss with hinge at 1
        x_mat = x.reshape(self.num_states, self.num_actions)
        return np.mean(np.maximum(np.abs(np.max(x_mat, axis=-1)) - 1, 0))

    def penalty_grad(self, x: np.array) -> float:
        x_mat = x.reshape(self.num_states, self.num_actions)
        x_max = np.max(x_mat, axis=-1)
        # Penalty gradient relative to maximum elements
        penalty_grad_max = np.sign(x_max) * (np.abs(x_max) > 1)
        penalty_grad_max /= self.num_states

        x_mat_grad = np.zeros_like(x_mat)
        x_mat_grad[range(self.num_states), np.argmax(x_mat, axis=-1)] = penalty_grad_max
        return x_mat_grad.reshape(-1)


class ProbsMatrixPolicy(DiscretePolicy):
    """A policy parametrized as a matrix of shifted probabilities.

    Each value x is transformed into a pre-normalization probability as
        p' = max(x + 1, eps)
    This encourages parameterizations centered around 0 in the interval [-1, 1]

    eps is a small positive value to ensure numerical stability when normalizing.
    """

    def __init__(self, num_states: int, num_actions: int, eps=1e-8, hinge=1.5):
        super().__init__(num_states=num_states, num_actions=num_actions)
        self.eps = eps
        self.hinge = hinge

    def num_params(self):
        return self.num_states * self.num_actions

    def policy_matrix(self, x: np.array, out: Optional[np.array] = None) -> np.array:
        x_mat = x.reshape(-1, self.num_actions)
        out = np.maximum(x_mat + 1, self.eps, out=out)
        out /= np.sum(out, axis=-1, keepdims=True)
        return out

    def penalty(self, x: np.array) -> float:
        # Average hinge loss on absolute value
        return np.mean(np.maximum(np.abs(x) - self.hinge, 0))
