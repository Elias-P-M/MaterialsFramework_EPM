"""This module provides a class for performing calculations and structure relaxation using the Eqnorm potential.

The `EqnormCalculator` class is designed to calculate properties such as potential energy, forces,
stresses, and to perform structure relaxation using a specified Eqnorm model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class EqnormCalculator(BaseCalculator, BaseMDCalculator):
    """A calculator class for performing material property calculations and structure relaxation using the Eqnorm potential.

    The `EqnormCalculator` class supports the calculation of properties such as potential energy,
    forces, and stresses. It also allows for the relaxation of structures using a specified Eqnorm model.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute,
                                          including "energy", "forces", and "stresses".

    """

    AVAILABLE_PROPERTIES = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model: Literal["eqnorm-mptrj", "eqnorm-omat", "eqnorm-max-mptrj"] = "eqnorm-mptrj",
        model_name: str = "eqnorm",
        device: str = "cpu",
        compile_model: bool = False,
        **kwargs,
    ) -> None:
        """Initializes the EqnormCalculator with the specified model and calculation settings.

        This method sets up the calculator with a predefined Eqnorm model, which will be used
        to calculate properties and perform structure relaxation. Additional parameters
        for the relaxation process can be passed via `basecalculator_kwargs`.

        Args:
            model (Literal["eqnorm-mptrj", "eqnorm-omat", "eqnorm-max-mptrj"]): The Eqnorm model variant. Defaults to "eqnorm-mptrj".
            model_name (str): The name of the Eqnorm model to use for calculations.
            device (str, optional): The device to use for calculations. Defaults to "cpu".
            compile_model (bool, optional): Whether to compile the model with torch.compile. Defaults to False.
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        basecalculator_kwargs = {key: kwargs.pop(key) for key in BaseCalculator.__init__.__annotations__ if key in kwargs}
        basemd_kwargs = {key: kwargs.pop(key) for key in BaseMDCalculator.__init__.__annotations__ if key in kwargs}

        # BaseCalculator and BaseMDCalculator specific attributes
        BaseCalculator.__init__(self, **basecalculator_kwargs)
        BaseMDCalculator.__init__(self, **basemd_kwargs)

        # Eqnorm specific attributes
        self.model = model
        self.model_name = model_name
        self.device = device
        self.compile = compile_model

        self._calculator = None

    @property
    def calculator(self) -> Calculator:
        """Creates and returns the ASE Calculator object associated with this calculator instance.

        This property initializes the Calculator object using the Eqnorm potential and other settings
        specified during the initialization of this calculator. The Calculator object is then returned
        to the caller. If the Calculator object has already been created, it is returned directly.

        Returns:
            Calculator: The ASE Calculator object configured with the Eqnorm potential.
        """
        if self._calculator is None:
            from eqnorm.calculator import EqnormCalculator

            self._calculator = EqnormCalculator(
                model_variant=self.model,
                model_name=self.model_name,
                device=self.device,
                compile=self.compile,
            )
        return self._calculator
