"""This module provides a class for performing calculations and structure relaxation using the eSEN potential.

The `eSENCalculator` class is designed to calculate properties such as potential energy, forces,
stresses, and to perform structure relaxation using a specified eSEN model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class eSENCalculator(BaseCalculator, BaseMDCalculator):
    """A calculator class for performing material property calculations and structure relaxation using the eSEN potential.

    The `eSENCalculator` class supports the calculation of properties such as potential energy,
    forces, and stresses. It also allows for the relaxation of structures using a specified eSEN model.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute,
                                          including "energy", "forces", and "stress".

    References:
        - eSEN: https://doi.org/10.48550/arXiv.2502.12147
    """

    AVAILABLE_PROPERTIES = ["energy", "forces", "stress"]

    def __init__(
        self,
        model: str = "esen-sm-conserving-all-omol",
        checkpoint_path: str | None = None,
        device: Literal["cpu", "cuda"] = "cpu",
        **kwargs,
    ) -> None:
        """Initializes the eSENCalculator with the specified model and calculation settings.

        This method sets up the calculator with a predefined eSEN model, which will be used
        to calculate properties and perform structure relaxation. Additional parameters
        for the relaxation process can be passed via `basecalculator_kwargs`.

        Args:
            model (str): The name of the eSEN model to use for calculations. Must be one of the
                models available in fairchem-core (e.g. ``esen-sm-conserving-all-omol``,
                ``esen-md-direct-all-omol``, ``esen-sm-conserving-all-oc25``). Ignored if
                ``checkpoint_path`` is provided. Defaults to ``"esen-sm-conserving-all-omol"``.
            checkpoint_path (str, optional): Path to a local eSEN checkpoint file. When provided,
                the model registry name is ignored.
            device (Literal["cpu", "cuda"]): The device to use for the calculations. Defaults to "cpu".
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        basecalculator_kwargs = {key: kwargs.pop(key) for key in BaseCalculator.__init__.__annotations__ if key in kwargs}
        basemd_kwargs = {key: kwargs.pop(key) for key in BaseMDCalculator.__init__.__annotations__ if key in kwargs}

        # BaseCalculator and BaseMDCalculator specific attributes
        BaseCalculator.__init__(self, **basecalculator_kwargs)
        BaseMDCalculator.__init__(self, **basemd_kwargs)

        # eSEN specific attributes
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.device = device

        self._calculator = None

    @property
    def calculator(self) -> Calculator:
        """Creates and returns the ASE Calculator object associated with this calculator instance.

        Returns:
            Calculator: The ASE Calculator object configured with the eSEN potential.
        """
        if self._calculator is None:
            from fairchem.core import FAIRChemCalculator, pretrained_mlip
            from fairchem.core.units.mlip_unit import load_predict_unit

            if self.checkpoint_path is not None:
                predictor = load_predict_unit(path=self.checkpoint_path, device=self.device)
            else:
                predictor = pretrained_mlip.get_predict_unit(model_name=self.model, device=self.device)
            self._calculator = FAIRChemCalculator(predict_unit=predictor)
        return self._calculator
