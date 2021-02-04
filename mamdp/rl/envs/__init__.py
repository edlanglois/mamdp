"""RL Environments"""
import functools
from typing import Callable, Dict

from .chain_mdp import ChainMDP
from .core import (
    Environment,
    MarkovDecisionProcess,
    ModifiedActionMarkovDecisionProcess,
    TabularMarkovDecisionProcess,
)
from .gridworlds import OffSwitch, WhiskyGold
from .inverting_bandit import (
    ExpInvertingDeterministicBandit,
    LinearInvertingDeterministicBandit,
)
from .simulation_oversight import SimulationOversight

__all__ = [
    "ChainMDP",
    "Environment",
    "ENVIRONMENTS",
    "ExpInvertingDeterministicBandit",
    "InvertingDeterministicBandit",
    "MarkovDecisionProcess",
    "ModifiedActionMarkovDecisionProcess",
    "OffSwitch",
    "SimulationOversight",
    "TabularMarkovDecisionProcess",
    "WhiskyGold",
]

ENVIRONMENTS: Dict[str, Callable[[], Environment]] = {
    "chain-mdp": ChainMDP,
    "exp-inverting-bandit": ExpInvertingDeterministicBandit,
    "linear-inverting-bandit": LinearInvertingDeterministicBandit,
    "off-switch": OffSwitch,
    "simulation-oversight": SimulationOversight,
    "whisky-gold": WhiskyGold,
    "whisky-gold-small": functools.partial(
        WhiskyGold, height=2, width=4, whisky_reward=1, drunk_bias=True
    ),
}
