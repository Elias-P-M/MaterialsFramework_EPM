"""A calculator that returns random energies for every call."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from materialsframework.tools.calculator import BaseCalculator

if TYPE_CHECKING:
    from ase import Atoms
    from pymatgen.core import Structure


class RandomCalculator(BaseCalculator):
    """A tiny calculator that returns random energies and forces for every call."""

    AVAILABLE_PROPERTIES = ["energy", "forces"]

    def calculate(self, structure=None, **kwargs) -> dict[str, Any]:
        """Pretend to calculate: return a random energy and random forces.

        Args:
            structure: The structure to calculate (used to determine number of atoms).
            **kwargs: Additional keyword arguments (ignored).
        """
        energy = float(random.uniform(-1000.0, 0.0))
        forces = [[random.uniform(-10, 10) for _ in range(3)] for _ in range(len(structure))]
        return {"energy": energy, "forces": forces}

    def relax(self, structure: Atoms | Structure, **kwargs) -> dict[str, Any]:
        """Pretend to relax: return the same structure, a random energy, and random forces.

        Args:
            structure: The structure to relax (used to determine number of atoms).
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            A dictionary containing the given structure, a random energy, and random forces.
        """
        energy = float(random.uniform(-1000.0, 0.0))
        forces = [[random.uniform(-10, 10) for _ in range(3)] for _ in range(len(structure))]
        return {"final_structure": structure, "energy": energy, "forces": forces}
