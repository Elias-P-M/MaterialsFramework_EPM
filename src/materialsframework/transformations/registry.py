"""Transformation registry backed by the materialsframework.transformations entry-point group."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from materialsframework.transformations.base import BaseTransformation

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


def list_transformations() -> list[str]:
    """Return sorted names of all registered transformations.

    Returns:
        Sorted list of registered transformation names.
    """
    return sorted(ep.name for ep in entry_points(group="materialsframework.transformations"))


def get_transformation(name: str, **kwargs) -> BaseTransformation:
    """Instantiate a transformation by its registered name.

    Args:
        name: Registered transformation name (e.g. "eos", "bain", "phonopy").
        **kwargs: Forwarded to the transformation's ``__init__``.

    Returns:
        An initialized transformation instance.

    Raises:
        ValueError: If no transformation is registered under the given name.
    """
    eps = {ep.name: ep for ep in entry_points(group="materialsframework.transformations")}
    if name not in eps:
        raise ValueError(f"Unknown transformation {name!r}. Available: {', '.join(sorted(eps))}")
    return eps[name].load()(**kwargs)
