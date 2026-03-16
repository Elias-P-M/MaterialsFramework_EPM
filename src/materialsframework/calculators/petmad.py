"""This module provides a class for calculations and relaxations with PET-MAD models via UPET."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class PetMadCalculator(BaseCalculator, BaseMDCalculator):
    """A calculator class for material property calculations using PET-MAD models served by UPET.

    The `PetMadCalculator` class supports the calculation of properties such as potential energy,
    forces, and stresses. It also allows for the relaxation of structures.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute,
                                          including "energy", "forces", and "stresses".

    References:
        - UPET: https://github.com/lab-cosmo/upet
    """

    AVAILABLE_PROPERTIES = ["energy", "forces", "stress"]

    def __init__(
        self,
        model: str = "pet-mad-s",
        version: str = "latest",
        checkpoint_path: str | None = None,
        device: Literal["cuda", "cpu", "mps"] = "cpu",
        **kwargs,
    ) -> None:
        """Initializes the PetMadCalculator with the specified model and calculation settings.

        This method sets up the calculator with a PET-MAD model family entry served by UPET.
        Additional parameters for the relaxation process can be passed via
        `basecalculator_kwargs`.

        Args:
            model (str): PET-MLIP model to use. Default is "pet-mad-s". Ignored if `checkpoint_path` is provided.
            version (str): Version of the model to use. Default is "latest". Ignored if `checkpoint_path` is provided.
            checkpoint_path (str, optional): Path to the model checkpoint file. If not provided,
                                                the model will be downloaded using the "version" parameter.
            device (str): The device to use for calculations. Options are "cuda", "cpu", or "mps".
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        basecalculator_kwargs = {key: kwargs.pop(key) for key in BaseCalculator.__init__.__annotations__ if key in kwargs}
        basemd_kwargs = {key: kwargs.pop(key) for key in BaseMDCalculator.__init__.__annotations__ if key in kwargs}

        # BaseCalculator and BaseMDCalculator specific attributes
        BaseCalculator.__init__(self, **basecalculator_kwargs)
        BaseMDCalculator.__init__(self, **basemd_kwargs)

        # PET-MAD via UPET specific attributes
        self.model = model
        self.version = version
        self.checkpoint_path = checkpoint_path
        self.device = device

        self._calculator = None

    @property
    def calculator(self) -> Calculator:
        """Creates and returns the ASE Calculator object associated with this calculator instance.

        This property initializes the Calculator object using UPET and PET-MAD model settings
        specified during the initialization of this calculator. The Calculator object is then returned
        to the caller. If the Calculator object has already been created, it is returned directly.

        Returns:
            Calculator: The ASE Calculator object configured via UPET.
        """
        if self._calculator is None:
            from upet.calculator import UPETCalculator

            self._calculator = UPETCalculator(
                model=self.model,
                version=self.version,
                checkpoint_path=self.checkpoint_path,
                device=self.device,
            )
        return self._calculator
