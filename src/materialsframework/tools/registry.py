"""Tool registry backed by the materialsframework.tools entry-point group."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


def list_tools() -> list[str]:
    """Return sorted names of all registered tools.

    Returns:
        Sorted list of registered tool names.
    """
    return sorted(ep.name for ep in entry_points(group="materialsframework.tools"))


def get_tool(name: str, **kwargs) -> Any:
    """Instantiate a tool by its registered name.

    Args:
        name: Registered tool name (e.g. "cluster_expansion", "stability_map").
        **kwargs: Forwarded to the tool's ``__init__``.

    Returns:
        An initialized tool instance.

    Raises:
        ValueError: If no tool is registered under the given name.
    """
    eps = {ep.name: ep for ep in entry_points(group="materialsframework.tools")}
    if name not in eps:
        raise ValueError(f"Unknown tool {name!r}. Available: {', '.join(sorted(eps))}")
    return eps[name].load()(**kwargs)
