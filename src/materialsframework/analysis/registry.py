"""Analyzer registry backed by the materialsframework.analyzers entry-point group."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from materialsframework.analysis.base import BaseAnalyzer

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


def list_analyzers() -> list[str]:
    """Return sorted names of all registered analyzers.

    Returns:
        Sorted list of registered analyzer names.
    """
    return sorted(ep.name for ep in entry_points(group="materialsframework.analyzers"))


def get_analyzer(name: str, **kwargs) -> BaseAnalyzer:
    """Instantiate an analyzer by its registered name.

    Args:
        name: Registered analyzer name (e.g. "eos", "bain", "phonopy").
        **kwargs: Forwarded to the analyzer's ``__init__``.

    Returns:
        An initialized analyzer instance.

    Raises:
        ValueError: If no analyzer is registered under the given name.
    """
    eps = {ep.name: ep for ep in entry_points(group="materialsframework.analyzers")}
    if name not in eps:
        raise ValueError(f"Unknown analyzer {name!r}. Available: {', '.join(sorted(eps))}")
    return eps[name].load()(**kwargs)
