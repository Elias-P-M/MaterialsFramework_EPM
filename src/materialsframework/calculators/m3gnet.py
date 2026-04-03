"""This module provides a class for performing calculations using the M3GNet potential.

The `M3GNetCalculator` class is designed to calculate properties such as potential energy,
forces, and stresses, and to perform structure relaxation using a specified M3GNet model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from torch import Tensor

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class M3GNetCalculator(BaseCalculator, BaseMDCalculator):
    """A calculator class for performing material property calculations and structure relaxation using the M3GNet potential.

    The `M3GNetCalculator` class supports the calculation of properties such as potential energy,
    forces, and stresses. It also allows for the relaxation of structures using a specified M3GNet model.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute,
                                          including "potential_energy", "forces", and "stress".

    References:
        - M3GNet: https://doi.org/10.1038/s43588-022-00349-3
    """

    AVAILABLE_PROPERTIES = ["energy", "forces", "stress"]

    def __init__(
        self,
        model: str = "M3GNet-MP-2021.2.8-PES",
        state_attr: Tensor | None = None,
        stress_unit: Literal["eV/A3", "GPa"] = "GPa",
        stress_weight: float = 1.0,
        use_voigt: bool = False,
        use_warp: bool = False,
        **kwargs,
    ) -> None:
        """Initializes the M3GNetCalculator with the specified model and calculation settings.

        This method sets up the calculator with a predefined M3GNet model, which will be used
        to calculate properties and perform structure relaxation. Additional parameters
        for the relaxation process can be passed via `basecalculator_kwargs`.

        Args:
            model (str, optional): The M3GNet model to use. Defaults to "M3GNet-MP-2021.2.8-PES".
            state_attr (Tensor | None, optional): State attributes to include in the potential energy calculation.
                                                  This allows for additional model customization. Defaults to None.
            stress_unit (Literal["eV/A3", "GPa"], optional): The unit for stress calculations. If set to "GPa", stress will be calculated in GPa.
            stress_weight (float, optional): Conversion factor from GPa to eV/ang^3. If set to 1.0, stress is calculated in GPa.
                                             Defaults to 1.0.
            use_voigt (bool, optional): Whether to use Voigt notation for stress. Defaults to False.
            use_warp (bool, optional): Whether to use WARP for neighbor list construction. Defaults to False.
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.

        Examples:
            >>> m3gnet_calculator = M3GNetCalculator(model="M3GNet-MP-2021.2.8-PES")

        Note:
            The remaining parameters for the M3GNet potential are set to their default values.
        """
        basecalculator_kwargs = {key: kwargs.pop(key) for key in BaseCalculator.__init__.__annotations__ if key in kwargs}
        basemd_kwargs = {key: kwargs.pop(key) for key in BaseMDCalculator.__init__.__annotations__ if key in kwargs}

        # BaseCalculator and BaseMDCalculator specific attributes
        BaseCalculator.__init__(self, **basecalculator_kwargs)
        BaseMDCalculator.__init__(self, **basemd_kwargs)

        # M3GNet specific attributes
        self.model = model
        self.state_attr = state_attr
        self.stress_unit = stress_unit
        self.stress_weight = stress_weight
        self.use_voigt = use_voigt
        self.use_warp = use_warp

        self._calculator = None
        self._potential = None

    @property
    def potential(self):
        """Loads and returns the M3GNet potential associated with this calculator instance.

        This property lazily loads the M3GNet model specified during initialization if it
        has not already been loaded. The loaded potential is then used for all subsequent
        calculations.

        Returns:
            CHGNet: The loaded M3GNet model instance used for calculations.
        """
        if self._potential is None:
            import matgl

            self._potential = matgl.load_model(path=self.model)
        return self._potential

    @property
    def calculator(self) -> Calculator:
        """Creates and returns the ASE Calculator object associated with this calculator instance.

        This property initializes the Calculator object using the M3GNet potential and other
        relevant attributes such as `state_attr` and `stress_weight`. If the Calculator object
        has already been created, it will return the existing instance.

        Returns:
            Calculator: The ASE Calculator object configured with the M3GNet potential.
        """
        if self._calculator is None:
            from matgl.ext.ase import PESCalculator

            self._calculator = PESCalculator(
                potential=self.potential,
                state_attr=self.state_attr,
                stress_unit=self.stress_unit,
                stress_weight=self.stress_weight,
                use_voigt=self.use_voigt,
                use_warp=self.use_warp,
            )
        return self._calculator
