"""Basic types for reinforcement learning"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import NewType, Tuple

__all__ = [
    "State",
    "Action",
    "PolicyMatrix",
    "RewardRange",
    "Step",
]

State = int
Action = int
# See https://github.com/python/mypy/issues/6701
PolicyMatrix = NewType("PolicyMatrix", "np.ndarray")  # type: ignore
RewardRange = Tuple[float, float]


@dataclass(frozen=True)
class Step:
    """One step in an RL environment.

    Attributes:
        state: The initial state.
        action: The selected action.
        next_state: The successor state given `state` and `action`.
        reward: The reward for this transition.
        done: Whether `next_state` is terminal.
            All trajectories from terminal states have 0 return.
    """

    state: State
    action: Action
    next_state: State
    reward: float
    done: bool

    def asdict(self):
        return asdict(self)

    def replace(self, **changes):
        return replace(self, **changes)
