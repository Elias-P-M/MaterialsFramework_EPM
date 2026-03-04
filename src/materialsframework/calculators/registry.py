"""Calculator registry backed by the materialsframework.calculators entry-point group."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from materialsframework.tools.calculator import BaseCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


def list_calculators() -> list[str]:
    """Return sorted names of all registered calculators.

    Returns:
        Sorted list of registered calculator names.
    """
    return sorted(ep.name for ep in entry_points(group="materialsframework.calculators"))


def get_calculator(name: str, **kwargs) -> BaseCalculator:
    """Instantiate a calculator by its registered name.

    Args:
        name: Registered calculator name (e.g. "mace", "chgnet").
        **kwargs: Forwarded to the calculator's ``__init__``.

    Returns:
        An initialized calculator instance.

    Raises:
        ValueError: If no calculator is registered under the given name.
    """
    eps = {ep.name: ep for ep in entry_points(group="materialsframework.calculators")}
    if name not in eps:
        raise ValueError(f"Unknown calculator {name!r}. Available: {', '.join(sorted(eps))}")
    return eps[name].load()(**kwargs)
