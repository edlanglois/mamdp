"""Agents"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Type, TypeVar, Union

import numpy as np

from ..optimization import es
from . import policies

if TYPE_CHECKING:
    from .types import Action, PolicyMatrix, State, Step

__all__ = [
    "Agent",
    "AGENTS",
    "EmpiricalSarsaAgent",
    "FixedPolicyAgent",
    "NaivePolicyGradientAgent",
    "QLearningAgent",
    "RandomAgent",
    "VirtualSarsaAgent",
]

_AgentT = TypeVar("_AgentT", bound="Agent")


class Agent:
    """An agent for an environment with descrete state and action spaces.

    States and actions are identified by integers starting at 0.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        seed: Optional[int] = None,
    ):
        """Initialize an agent.

        Args:
            num_states: Number of environment states.
            num_actions: Number of environment actions.
            seed: An optional random seed for action samples.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self._rand = np.random.RandomState(seed)

    @classmethod
    def init(
        cls: Type[_AgentT],
        *,
        num_states: int,
        num_actions: int,
        seed: Optional[int] = None,
        **kwargs
    ) -> _AgentT:
        """Initialize an agent given a set of parameters."""
        del kwargs  # Ignore extra arguments
        return cls(num_states=num_states, num_actions=num_actions, seed=seed)

    def act(self, state: State, greedy: bool = False) -> Action:
        """Sample an action for the given state.

        Args:
            state: The state in which to act.
            greedy: If True, the agent should act to maximize reward under the
                assumption that no further update()'s will take place.
        """
        return self._rand.choice(
            self.num_actions, p=self.policy_matrix(greedy=greedy)[state]
        )

    def policy_matrix(self, greedy: bool = False) -> PolicyMatrix:
        """A num_states x num_actions matrix where policy_matrix()[s, a] = Pr(a|s).

        Args:
            greedy: If True, return a greedy policy that does not explicitly attempt to
                explore.
        """
        raise NotImplementedError

    def update(self, step: Step) -> None:
        """Update the agent given one environment step."""
        raise NotImplementedError


class RandomAgent(Agent):
    """An agent that always acts randomly."""

    def act(self, state: State, greedy: bool = False) -> Action:
        del greedy  # Only one kind of behaviour
        del state  # Ignores the state
        return self._rand.choice(self.num_actions)

    def policy_matrix(self, greedy: bool = False) -> PolicyMatrix:
        del greedy  # Only one kind of behaviour
        return np.full(
            (self.num_states, self.num_actions), fill_value=1 / self.num_actions
        )

    def update(self, step: Step) -> None:
        pass


_FixedPolicyAgentT = TypeVar("_FixedPolicyAgentT", bound="FixedPolicyAgent")


class FixedPolicyAgent(Agent):
    """An agent that executes a fixed policy."""

    def __init__(self, policy: np.ndarray, seed: Optional[int] = None):
        policy = np.asarray(policy)
        num_states, num_actions = policy.shape
        super().__init__(num_states=num_states, num_actions=num_actions, seed=seed)
        self.policy = policy

    @classmethod
    def init(
        cls: Type[_FixedPolicyAgentT],
        *,
        num_states: int,
        num_actions: int,
        seed: Optional[int] = None,
        **kwargs
    ) -> _FixedPolicyAgentT:
        raise RuntimeError("Not supported; use __init__")

    def policy_matrix(self, greedy: bool = False) -> PolicyMatrix:
        return self.policy


_TabularQAgentT = TypeVar("_TabularQAgentT", bound="_TabularQAgent")


class _TabularQAgent(Agent):
    """Agent that maintains a Q table."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        discount_factor: float,
        exploration_rate: float,
        learning_rate: Optional[float],
        seed: Optional[int] = None,
    ):
        """Initialize the agent.

        Args:
            num_states: Number of environment states.
            num_actions: Number of environment actions.
            discount_factor: Environment discount factor.
            exploration_rate: Probability of choosing an action uniformly at random.
            learning_rate: Optional learning rate for Q table updates.
                By default, the Q table uses the mean of all past observations.
            seed: An optional random seed for action samples.
        """
        super().__init__(
            num_states=num_states,
            num_actions=num_actions,
            seed=seed,
        )
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.q = np.zeros([num_states, num_actions])
        self.counts = np.zeros([num_states, num_actions], dtype=int)

    @classmethod
    def init(
        cls: Type[_TabularQAgentT],
        *,
        num_states: int,
        num_actions: int,
        discount_factor: float = 1,
        exploration_rate: float = 0.1,
        learning_rate: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> _TabularQAgent:
        del kwargs  # Ignore extra arguments
        return cls(
            num_states=num_states,
            num_actions=num_actions,
            discount_factor=discount_factor,
            learning_rate=learning_rate,
            exploration_rate=exploration_rate,
            seed=seed,
        )

    def act(self, state: State, greedy: bool = False) -> Action:
        if not greedy and self._rand.uniform() < self.exploration_rate:
            return self._rand.randint(self.num_actions)
        else:
            return np.argmax(self.q[state])

    def policy_matrix(self, greedy: bool = False) -> PolicyMatrix:
        matrix = np.identity(self.num_actions)[np.argmax(self.q, axis=-1)]
        if not greedy:
            matrix *= 1 - self.exploration_rate
            matrix += self.exploration_rate / self.num_actions
        return matrix

    def _update(
        self, state: State, action: Action, reward: float, next_state_value: float
    ) -> None:
        self.counts[state, action] += 1
        if self.learning_rate is not None:
            learning_rate = self.learning_rate
        else:
            learning_rate = 1 / self.counts[state, action]
        self.q[state, action] += learning_rate * (
            reward + self.discount_factor * next_state_value - self.q[state, action]
        )

    def estimate_value(self, state: State, action: Optional[Action] = None) -> float:
        """Estimate the value of the state (and optionally action)."""
        if action is None:
            return np.max(self.q[state])
        else:
            return self.q[state, action]


class QLearningAgent(_TabularQAgent):
    """Tabular Q Learning Agent.

    Maintains a Q table and updates using the next state value max(Q[s', :])
    """

    def update(self, step: Step) -> None:
        if step.done:
            next_state_value = 0
        else:
            next_state_value = np.max(self.q[step.next_state])
        self._update(step.state, step.action, step.reward, next_state_value)


class VirtualSarsaAgent(_TabularQAgent):
    """Tabular Virtual Sarsa Agent.

    Maintains a Q table and updates using the next state value Q[s', a']
    where a' = act(s').

    This is virtual in the sense that a' is simulated and does not necessarily equal the
    true a' taken in the environment.
    """

    def update(self, step: Step) -> None:
        next_state = step.next_state
        next_action = self.act(next_state)
        if step.done:
            next_state_value = 0
        else:
            next_state_value = self.q[next_state, next_action]
        self._update(step.state, step.action, step.reward, next_state_value)


class EmpiricalSarsaAgent(_TabularQAgent):
    """Tabular Empirical Sarsa Agent.

    Maintains a Q table and updates use the next state value Q[s', a'] where a' is the
    action taken in state s'.

    Because a' is not known immediately, need to save the previous step.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_step = None

    def update(self, step: Step) -> None:
        if self._last_step is not None and step.state == self._last_step.next_state:
            # Do the update for the previous state
            self._update(
                self._last_step.state,
                self._last_step.action,
                self._last_step.reward,
                self.q[step.state, step.action],
            )
        if step.done:
            # This is a terminal state.
            # There won't be another step, and it would have value 0 if it happened.
            self._update(step.state, step.action, step.reward, 0)
            self._last_step = None
        else:
            self._last_step = step


def discounted_suffix_cumsum(a: np.array, discount_factor: float):
    """Cumulative sum of suffixes with discounting."""
    a = np.asarray(a)
    try:
        (n,) = a.shape
    except ValueError:
        raise ValueError("a must be a 1-dimensional array")
    out = np.empty_like(a, dtype=np.result_type(a, discount_factor))
    value: float = 0
    for i, ai in enumerate(a[::-1]):
        value *= discount_factor
        value += ai
        out[n - 1 - i] = value
    return out


_NaivePolicyGradientAgentT = TypeVar(
    "_NaivePolicyGradientAgentT", bound="NaivePolicyGradientAgent"
)


class NaivePolicyGradientAgent(Agent):
    """Agent implementing the MDP policy gradient.

    This is Naive in the sense that the MDP policy gradient equations are used even
    though these equations to not correspond to the true policy gradient for MAMDPs.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        discount_factor: float,
        learning_rate: float,
        policy: Union[str, Type[policies.DiscretePolicy]] = "logits",
        seed: Optional[int] = None,
        **kwargs
    ):
        del kwargs
        super().__init__(
            num_states=num_states,
            num_actions=num_actions,
            seed=seed,
        )
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        if isinstance(policy, str):
            policy_cls = {
                "logits": policies.LogitsMatrixPolicy,
                "probs": policies.ProbsMatrixPolicy,
            }[policy]
        else:
            policy_cls = policy
        self.policy = policy_cls(num_states, num_actions)
        self.policy_params = np.zeros([self.policy.num_params()])
        self._policy_matrix = self.policy.policy_matrix(self.policy_params)
        self._episode_states: List[State] = []
        self._episode_actions: List[Action] = []
        self._episode_rewards: List[float] = []

    @classmethod
    def init(
        cls: Type[_NaivePolicyGradientAgentT],
        num_states: int,
        num_actions: int,
        discount_factor: float = 1,
        learning_rate: float = 0.1,
        policy: Union[str, Type[policies.DiscretePolicy]] = "logits",
        seed: Optional[int] = None,
        **kwargs
    ) -> _NaivePolicyGradientAgentT:
        del kwargs  # Ignore extra arguments
        return cls(
            num_states=num_states,
            num_actions=num_actions,
            discount_factor=discount_factor,
            learning_rate=learning_rate,
            policy=policy,
            seed=seed,
        )

    def policy_matrix(self, greedy: bool = False) -> PolicyMatrix:
        return self._policy_matrix

    def update(self, step: Step) -> None:
        self._episode_states.append(step.state)
        self._episode_actions.append(step.action)
        self._episode_rewards.append(step.reward)
        if step.done:
            actions = np.asarray(self._episode_actions)
            states = np.asarray(self._episode_states)
            rewards_to_go = discounted_suffix_cumsum(
                self._episode_rewards, self.discount_factor
            )
            gradient = (
                self.policy.log_policy_grad(
                    actions=actions,
                    states=states,
                    x=self.policy_params,
                    weights=rewards_to_go,
                )
                / len(actions)
            )
            gradient -= self.policy.penalty_grad(self.policy_params)
            self.policy_params += self.learning_rate * gradient
            self._policy_matrix = self.policy.policy_matrix(self.policy_params)
            self._episode_states = []
            self._episode_actions = []
            self._episode_rewards = []


_EvolutionStrategiesAgentT = TypeVar(
    "_EvolutionStrategiesAgentT", bound="EvolutionStrategiesAgent"
)


class EvolutionStrategiesAgent(Agent):
    """Direct policy optimization agent using UH-CMA-ES.

    Reference:
    Uncertainty Handling CMA-ES for Reinforcement Learning
    by Verena Heidrich-Meisner and Christian Igel (2009)
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        discount_factor: float,
        initial_step_size: float,
        initial_num_sample_evaluations: int,
        scale_sample_evaluations: float,
        policy: Union[str, Type[policies.DiscretePolicy]] = "logits",
        seed: Optional[int] = None,
    ):
        super().__init__(
            num_states=num_states,
            num_actions=num_actions,
            seed=seed,
        )
        self.discount_factor = discount_factor
        if isinstance(policy, str):
            policy_cls = {
                "logits": policies.LogitsMatrixPolicy,
                "probs": policies.ProbsMatrixPolicy,
            }[policy]
        else:
            policy_cls = policy
        self.policy = policy_cls(num_states, num_actions)
        # If num_sample_evaluations starts too low then ES can get stuck in a local
        # optimum
        self.minimizer = es.interactive_uh_cma_es(
            x0=np.zeros([self.policy.num_params()]),
            step_size=initial_step_size,
            num_sample_evaluations=initial_num_sample_evaluations,
            scale_sample_evaluations=scale_sample_evaluations,
            rand=self._rand,
        )
        test_policy_vector, self.search_index, greedy_policy_vector = next(
            self.minimizer
        )

        self.test_policy = self.policy.policy_matrix(test_policy_vector)
        self.test_policy_penalty = self.policy.penalty(test_policy_vector)
        self.greedy_policy = self.policy.policy_matrix(greedy_policy_vector)
        self._episode_return: float = 0
        self._episode_discount: float = 1

    @classmethod
    def init(
        cls: Type[_EvolutionStrategiesAgentT],
        num_states: int,
        num_actions: int,
        discount_factor: float = 1,
        initial_step_size: float = 0.1,
        initial_num_sample_evaluations: int = 100,
        scale_sample_evaluations: float = 1.5,
        policy: Union[str, Type[policies.DiscretePolicy]] = "logits",
        seed: Optional[int] = None,
        **kwargs
    ) -> _EvolutionStrategiesAgentT:
        del kwargs  # Ignore extra arguments
        return cls(
            num_states=num_states,
            num_actions=num_actions,
            discount_factor=discount_factor,
            initial_step_size=initial_step_size,
            initial_num_sample_evaluations=initial_num_sample_evaluations,
            scale_sample_evaluations=scale_sample_evaluations,
            policy=policy,
            seed=seed,
        )

    def policy_matrix(self, greedy: bool = False) -> PolicyMatrix:
        return self.greedy_policy if greedy else self.test_policy

    def update(self, step: Step) -> None:
        self._episode_return += self._episode_discount * step.reward
        self._episode_discount *= self.discount_factor
        if step.done:
            (
                test_policy_vector,
                self.search_index,
                greedy_policy_vector,
            ) = self.minimizer.send(-self._episode_return + self.test_policy_penalty)

            self.test_policy = self.policy.policy_matrix(test_policy_vector)
            self.test_policy_penalty = self.policy.penalty(test_policy_vector)
            self.greedy_policy = self.policy.policy_matrix(greedy_policy_vector)
            self._episode_return = 0
            self._episode_discount = 1


AGENTS = {
    "empirical-sarsa": EmpiricalSarsaAgent,
    "q-learning": QLearningAgent,
    "random": RandomAgent,
    "virtual-sarsa": VirtualSarsaAgent,
    "policy-gradient": NaivePolicyGradientAgent,
    "es": EvolutionStrategiesAgent,
}
