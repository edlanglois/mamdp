"""Encoding and decoding of run results"""
from __future__ import annotations

import json
from typing import IO, Any, Dict, Iterable, List, Tuple, TypeVar

import numpy as np
import pandas as pd

from .rl.types import Step

__all__ = ["dump_results"]


K = TypeVar("K")
V = TypeVar("V")


def _unzip_dicts(dicts: Iterable[Dict[K, V]]) -> Dict[K, List[V]]:
    """Unzip a list of dicts to a dict of lists."""
    dicts = iter(dicts)
    try:
        first = next(dicts)
    except StopIteration:
        return {}
    out = {k: [v] for k, v in first.items()}
    for d in dicts:
        for k, vs in out.items():
            vs.append(d[k])
    return out


def dump_results(
    info: Dict[str, Any], steps: Iterable[Step], policy: np.ndarray, fp: IO[str]
) -> None:
    data = {
        **info,
        "steps": _unzip_dicts(s.asdict() for s in steps),
        "policy": policy.tolist(),
    }
    json.dump(data, fp)


def load_results(
    fp: IO[str],
) -> Tuple[Dict[str, Any], pd.DataFrame, np.ndarray]:
    data = json.load(fp)
    steps = pd.DataFrame(data.pop("steps"))
    policy = np.asarray(data.pop("policy"))
    return data, steps, policy
