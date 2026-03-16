"""This module provides a class to generate structures and tasks for coefficient of thermal expansion (CTE) workflows.

The `CTETransformation` class prepares per-temperature structure/task inputs for MD-based
CTE analysis.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor

if TYPE_CHECKING:
    from pymatgen.core import Structure

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class CTETransformation:
    """Build per-temperature structure/task inputs for CTE analysis."""

    def __init__(
        self,
        ensemble: str = "npt_berendsen",
        pressure: float = 1.0,
    ) -> None:
        """Initialize the transformation object.

        Args:
            ensemble: Requested MD ensemble label.
            pressure: Target pressure value in atm.
        """
        self.ensemble = ensemble
        self.pressure = pressure
        self.ase_adaptor = AseAtomsAdaptor()

        self.structures: dict[float, Structure] = {}
        self.tasks: list[dict[str, Any]] = []

    def apply_transformation(
        self,
        structure: Structure | Atoms,
        temperatures: Sequence[float],
        steps: int = 10000,
    ) -> None:
        """Prepare structures and task metadata for each target temperature.

        Args:
            structure: Input structure for MD sampling.
            temperatures: Target temperatures in Kelvin.
            steps: Number of MD steps per temperature.

        Raises:
            ValueError: If temperatures are invalid or steps is non-positive.
        """
        validated_temperatures = self._validate_temperatures(temperatures)
        if steps <= 0:
            raise ValueError("steps must be a positive integer.")

        if isinstance(structure, Atoms):
            structure = self.ase_adaptor.get_structure(structure)

        self.structures = {}
        self.tasks = []

        for temperature in validated_temperatures:
            temperature_value = float(temperature)
            self.structures[temperature_value] = structure.copy()
            self.tasks.append(
                {
                    "temperature": temperature_value,
                    "steps": int(steps),
                    "ensemble": self.ensemble,
                    "pressure": float(self.pressure),
                }
            )

    @staticmethod
    def _validate_temperatures(temperatures: Sequence[float]) -> list[float]:
        """Validate and return temperatures as floats.

        Args:
            temperatures: Candidate temperatures in Kelvin.

        Raises:
            ValueError: If temperatures are empty, non-numeric, non-finite, or non-positive.

        Returns:
            Validated temperatures.
        """
        if not isinstance(temperatures, Sequence) or isinstance(temperatures, str):
            raise ValueError("temperatures must be provided as a non-empty sequence of positive values in Kelvin.")
        if len(temperatures) == 0:
            raise ValueError("temperatures must contain at least one value.")

        validated_temperatures: list[float] = []
        for temperature in temperatures:
            if not isinstance(temperature, int | float):
                raise ValueError("All temperatures must be numeric values in Kelvin.")
            temperature_value = float(temperature)
            if not np.isfinite(temperature_value):
                raise ValueError("All temperatures must be finite values in Kelvin.")
            if temperature_value <= 0:
                raise ValueError("All temperatures must be greater than 0 K.")
            validated_temperatures.append(temperature_value)
        return validated_temperatures
