"""This module provides a class to perform a coefficient of thermal expansion (CTE) analysis on a given structure.

The `CTEAnalyzer` class estimates the volumetric coefficient of thermal expansion by running
temperature-dependent MD sampling through `CTETransformation` and fitting the resulting
volume-temperature data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor

from materialsframework.transformations.cte import CTETransformation

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatgen.core import Structure

    from materialsframework.tools.calculator import BaseCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class CTEAnalyzer:
    """A class used to estimate volumetric CTE from NPT-MD volume-temperature data."""

    _MIN_DISTINCT_TEMPERATURES = 2

    def __init__(
        self,
        calculator: BaseCalculator | None = None,
        cte_transformation: CTETransformation | None = None,
    ) -> None:
        """Initializes the `CTEAnalyzer` object.

        Args:
            calculator (BaseCalculator | None, optional): MD-capable calculator.
                Defaults to lazy `M3GNetCalculator`.
            cte_transformation (CTETransformation | None, optional): Transformation object
                for running the temperature sweep workflow.
        """
        self.ase_adaptor = AseAtomsAdaptor()
        self._calculator = calculator
        self._cte_transformation = cte_transformation

    def calculate(
        self,
        structure: Structure | Atoms,
        temperatures: Sequence[float],
        steps: int = 10000,
    ) -> dict[str, list | dict]:
        """Calculates temperature-dependent volumes and volumetric CTE.

        Args:
            structure (Structure | Atoms): Input structure.
            temperatures (Sequence[float]): Temperatures in Kelvin.
            steps (int, optional): Number of MD steps per temperature. Defaults to ``200``.

        Raises:
            ValueError: If fewer than two distinct temperatures are provided, the
                calculator is not MD-capable, or MD output is malformed.

        Returns:
            dict[str, list | dict]: Dictionary with keys:
                - ``temperatures``: Input temperatures in Kelvin.
                - ``volumes``: Final volume for each temperature in Å³.
                - ``per_temperature``: Per-temperature metrics from MD.
                - ``cte``: Volumetric CTE summary in K⁻¹ and ppm/K.
        """
        if isinstance(structure, Atoms):
            structure = self.ase_adaptor.get_structure(structure)

        validated_temperatures = self.cte_transformation._validate_temperatures(temperatures)
        unique_temperatures = {float(t) for t in validated_temperatures}
        if len(unique_temperatures) < self._MIN_DISTINCT_TEMPERATURES:
            raise ValueError("At least two distinct temperatures are required to compute CTE.")

        self.cte_transformation.apply_transformation(
            structure=structure,
            temperatures=validated_temperatures,
            steps=steps,
        )

        if not hasattr(self.calculator, "run") or not callable(self.calculator.run):
            raise ValueError("The calculator object must have a callable 'run' method for MD.")

        per_temperature: list[dict[str, float]] = []
        for task in self.cte_transformation.tasks:
            if hasattr(self.calculator, "ensemble"):
                self.calculator.ensemble = task["ensemble"]
            if hasattr(self.calculator, "pressure"):
                self.calculator.pressure = task["pressure"]
            if hasattr(self.calculator, "temperature"):
                self.calculator.temperature = task["temperature"]

            md_result = self.calculator.run(
                structure=self.cte_transformation.structures[task["temperature"]],
                steps=task["steps"],
            )

            final_structure = md_result.get("final_structure")
            if final_structure is None:
                raise ValueError("MD calculator 'run' output must include a 'final_structure'.")
            if isinstance(final_structure, Atoms):
                final_structure = self.ase_adaptor.get_structure(final_structure)

            sampled_temperatures = md_result.get("temperature", [])
            sampled_potential_energies = md_result.get("potential_energy", [])
            per_temperature.append(
                {
                    "temperature": float(task["temperature"]),
                    "sampled_temperature_mean": (
                        float(np.mean(sampled_temperatures)) if len(sampled_temperatures) > 0 else float(task["temperature"])
                    ),
                    "volume": float(final_structure.volume),
                    "potential_energy_mean": (
                        float(np.mean(sampled_potential_energies)) if len(sampled_potential_energies) > 0 else float("nan")
                    ),
                }
            )

        temperatures_array = np.array([entry["temperature"] for entry in per_temperature], dtype=float)
        volumes_array = np.array([entry["volume"] for entry in per_temperature], dtype=float)

        slope, intercept = np.polyfit(temperatures_array, volumes_array, 1)
        reference_index = int(np.argmin(temperatures_array))
        reference_temperature = float(temperatures_array[reference_index])
        reference_volume = float(volumes_array[reference_index])

        alpha_volumetric = float(slope / reference_volume)
        alpha_volumetric_ppm = alpha_volumetric * 1e6

        return {
            "temperatures": temperatures_array.tolist(),
            "volumes": volumes_array.tolist(),
            "per_temperature": per_temperature,
            "cte": {
                "volumetric_per_k": alpha_volumetric,
                "volumetric_ppm_per_k": alpha_volumetric_ppm,
                "slope_ang3_per_k": float(slope),
                "intercept_ang3": float(intercept),
                "reference_temperature": reference_temperature,
                "reference_volume": reference_volume,
            },
        }

    @property
    def calculator(self) -> BaseCalculator:
        """Return calculator instance used for the CTE workflow.

        Returns:
            Calculator instance.
        """
        if self._calculator is None:
            from materialsframework.calculators.m3gnet import M3GNetCalculator

            self._calculator = M3GNetCalculator()
        return self._calculator

    @property
    def cte_transformation(self) -> CTETransformation:
        """Return CTE transformation object used by this analyzer.

        Returns:
            CTE transformation instance.
        """
        if self._cte_transformation is None:
            self._cte_transformation = CTETransformation()
        return self._cte_transformation
