"""Command-line utilities"""
from __future__ import annotations

import json
from typing import IO, Any, Dict, Iterable

try:
    import tabulate
except ImportError:
    tabulate = None  # type: ignore

__all__ = [
    "IncrementalJSONEncoder",
    "print_key_values",
]


def print_key_values(d: Dict[str, Any], depth=0):
    """Pretty-print a nested key-value dictionary."""
    indent = "  " * depth
    for k, v in d.items():
        if hasattr(v, "items"):
            print(f"{indent}{k.replace('_', ' ').title()}:")
            print_key_values(v, depth=depth + 1)
        else:
            print(f"{indent}{k.replace('_', ' ').title()}: {v}")


class IncrementalJSONEncoder:
    """Context manager to incrementally encode a JSON dictionary key by key."""

    def __init__(self, output: IO[str]):
        # Store in _output when outside of the context,
        # copy to output when in the context.
        # This prevents writes from happening outside the context manger.
        self.output = output

    def __enter__(self):
        self.output.write("{")
        return _IncrementalJSONEncoderContext(self.output)

    def __exit__(self, exc_type, exc_value, traceback):
        self.output.write("}")


class _IncrementalJSONEncoderContext:
    def __init__(self, output: IO[str]):
        self.output = output
        self._first_key = True

    def write(self, key: str, value: Any) -> None:
        self._write_key_separator()
        self.output.write('"{key}": {value}'.format(key=key, value=json.dumps(value)))

    def write_iterator(self, key: str, values: Iterable[Any]) -> None:
        self._write_key_separator()
        self.output.write('"{key}": ['.format(key=key))
        first = True
        for value in values:
            if not first:
                self.output.write(", ")
            else:
                first = False
            self.output.write(json.dumps(value))
        self.output.write("]")

    def _write_key_separator(self):
        if not self._first_key:
            self.output.write(", ")
        else:
            self._first_key = False
