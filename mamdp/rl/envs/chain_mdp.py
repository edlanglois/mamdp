"""Chain MDP Environment"""
from __future__ import annotations

from typing import Optional

from ..types import Action, State, Step
from .core import MarkovDecisionProcess

__all__ = [
    "ChainMDP",
]


class ChainMDP(MarkovDecisionProcess):
    """A simple 1-dimensional n-state chain MDP.

    At each state the agent can either return to the start (state 0) or advance one
    state towards the goal (state n-1).
    There is a reward of -1 on each step and the episode ends upon reaching the goal.
    """

    def __init__(self, num_states: int = 4, seed: Optional[int] = None):
        super().__init__(
            num_states=num_states,
            num_actions=2,
            reward_range=(float("-inf"), 0),
            seed=seed,
        )

    def initial_state(self) -> State:
        return 0

    def step(self, state: State, action: Action) -> Step:
        next_state = 0 if action == 0 else min(state + 1, self.num_states - 1)
        return Step(
            state=state,
            action=action,
            next_state=next_state,
            reward=-1,
            done=(next_state == self.num_states - 1),
        )

    def action_string(self, action: Action) -> str:
        return ["Start", "Next"][action]
