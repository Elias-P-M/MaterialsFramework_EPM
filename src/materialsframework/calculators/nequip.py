"""This module provides a class for performing calculations and structure relaxation using the NequIP potential.

The `NequIPCalculator` class is designed to calculate properties such as potential energy, forces,
stresses, and to perform structure relaxation using a specified NequIP model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class NequIPCalculator(BaseCalculator, BaseMDCalculator):
    """A calculator class for performing material property calculations and structure relaxation using the NequixNequIP potential.

    The `NequIPCalculator` class supports the calculation of properties such as potential energy,
    forces, and stresses. It also allows for the relaxation of structures using a specified NequixNequIP model.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute,
                                          including "energy", "forces", and "stresses".

    References:
        - NequIP: https://arxiv.org/abs/2504.16068
    """

    AVAILABLE_PROPERTIES = ["energy", "energies", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model: str = "",
        device: Literal["cpu", "cuda"] = "cpu",
        chemical_species_to_atom_type_map: dict[str, str] | bool | None = True,
        auto_diff: bool = False,
        compile_mode: Literal["auto", "eager", "compile", "compiled"] = "auto",
        **kwargs,
    ) -> None:
        """Initializes the NequIPCalculator with the specified model and calculation settings.

        This method sets up the calculator with a predefined NequIP model, which will be used
        to calculate properties and perform structure relaxation. Additional parameters
        for the relaxation process can be passed via `basecalculator_kwargs`.

        Args:
            model (str): The NequIP model to use.
            device (Literal["cuda", "cpu"]): The device to use for calculations. Defaults to "cpu".
            chemical_species_to_atom_type_map (Optional[Union[Dict[str, str], bool]], optional): A mapping from chemical species to atom types expected by the NequIP model. Defaults to True, which means that the mapping will be automatically inferred from the model.
            auto_diff (bool, optional): If True, return an auto-differentiable ASE calculator
                that preserves the PyTorch computation graph on per-atom energies and exposes
                a ``heat_flux_forward(atoms)`` method for use with
                :class:`materialsframework.tools.heat_flux.HeatFluxObserver` in the Green-Kubo
                workflow. An eager-mode or TorchScript NequIP model is required; AOT Inductor
                ``.nequip.pt2`` should support autograd. Defaults to False.
            compile_mode (Literal["auto", "eager", "compile", "compiled"], optional): Controls which
                NequIP loader is used:
                - ``"compiled"`` — ``from_compiled_model`` (AOT Inductor ``.nequip.pt2``); does
                  not support autograd.
                - ``"eager"`` / ``"compile"`` — ``_from_saved_model`` (``.zip`` / ``.nequip.pth``);
                  supports autograd.
                - ``"auto"`` (default) — picks ``"eager"`` when ``auto_diff=True`` or when the
                  model extension is ``.zip`` / ``.nequip.pth``, otherwise ``"compiled"``.
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        basecalculator_kwargs = {key: kwargs.pop(key) for key in BaseCalculator.__init__.__annotations__ if key in kwargs}
        basemd_kwargs = {key: kwargs.pop(key) for key in BaseMDCalculator.__init__.__annotations__ if key in kwargs}

        # BaseCalculator and BaseMDCalculator specific attributes
        BaseCalculator.__init__(self, **basecalculator_kwargs)
        BaseMDCalculator.__init__(self, **basemd_kwargs)

        # NequIP specific attributes
        self.model = model
        self.device = device
        self.chemical_species_to_atom_type_map = chemical_species_to_atom_type_map
        self.auto_diff = auto_diff
        self.compile_mode = compile_mode

        self._calculator = None

    def _resolve_compile_mode(self) -> Literal["eager", "compile", "compiled"]:
        """Resolve ``"auto"`` compile mode from ``auto_diff`` and the model extension."""
        if self.compile_mode != "auto":
            return self.compile_mode
        model_lower = self.model.lower()
        if self.auto_diff or model_lower.endswith((".zip", ".nequip.pth")):
            return "eager"
        return "compiled"

    @property
    def calculator(self) -> Calculator:
        """Creates and returns the ASE Calculator object associated with this calculator instance.

        This property initializes the Calculator object using the NequIP potential and other settings
        specified during the initialization of this calculator. The Calculator object is then returned
        to the caller. If the Calculator object has already been created, it is returned directly.

        When ``self.auto_diff`` is True, an auto-differentiable subclass is returned instead; its
        ``calculate()`` preserves the PyTorch computation graph and ``heat_flux_forward(atoms)``
        runs a dedicated forward pass for the Green-Kubo heat-flux workflow.

        Returns:
            Calculator: The ASE Calculator object configured with the NequIP potential.
        """
        if self._calculator is None:
            from nequip.integrations.ase import NequIPCalculator as _NequIPASECalculator

            cls = _make_autodiff_nequip_class(_NequIPASECalculator) if self.auto_diff else _NequIPASECalculator

            mode = self._resolve_compile_mode()
            if mode == "compiled":
                self._calculator = cls.from_compiled_model(
                    compile_path=self.model,
                    device=self.device,
                    chemical_species_to_atom_type_map=self.chemical_species_to_atom_type_map,
                )
            else:
                self._calculator = cls._from_saved_model(
                    model_path=self.model,
                    compile_mode=mode,
                    device=self.device,
                    chemical_species_to_atom_type_map=self.chemical_species_to_atom_type_map,
                )

        return self._calculator


def _make_autodiff_nequip_class(base_cls):
    """Build an auto-differentiable subclass of the NequIP ASE calculator.

    The subclass overrides ``calculate()`` to keep ``position`` tensors on the
    computation graph and adds ``heat_flux_forward(atoms)`` that runs a fresh
    forward pass returning ``(per_atom_energies_eV, positions_A)`` for use by
    :class:`materialsframework.tools.heat_flux.HeatFluxObserver`.

    Implemented as a factory so that the NequIP imports (``nequip``, ``torch``)
    stay inside the lazy-import path and do not execute at module load time.
    """
    import torch
    from ase.calculators.calculator import Calculator as _ASECalculator
    from ase.calculators.calculator import all_changes
    from ase.stress import full_3x3_to_voigt_6_stress
    from nequip.data import AtomicDataDict

    class AutoDiffNequIPCalculator(base_cls):
        """NequIP ASE calculator with autograd-friendly forward passes.

        After each :meth:`calculate` call the following tensors are available
        for downstream autograd:

        - ``_live_positions`` (N, 3) with ``requires_grad_(True)``.
        - ``_live_per_atom_energies`` (N,) still attached to the graph.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._live_positions: torch.Tensor | None = None
            self._live_per_atom_energies: torch.Tensor | None = None

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            """Forward pass with position-gradient tracking enabled."""
            _ASECalculator.calculate(self, atoms)

            data = self.atoms_to_data(atoms)
            data[AtomicDataDict.POSITIONS_KEY].requires_grad_(True)

            out = self.call_model(data)

            self._live_positions = data[AtomicDataDict.POSITIONS_KEY]
            pae = out.get(AtomicDataDict.PER_ATOM_ENERGY_KEY)
            self._live_per_atom_energies = pae.squeeze(-1) if pae is not None else None

            self.results = {}
            if AtomicDataDict.TOTAL_ENERGY_KEY in out:
                self.results["energy"] = self.energy_units_to_eV * (
                    out[AtomicDataDict.TOTAL_ENERGY_KEY].detach().cpu().numpy().reshape(())
                )
                self.results["free_energy"] = self.results["energy"]
            if AtomicDataDict.PER_ATOM_ENERGY_KEY in out:
                self.results["energies"] = self.energy_units_to_eV * (
                    out[AtomicDataDict.PER_ATOM_ENERGY_KEY].detach().squeeze(-1).cpu().numpy()
                )
            if AtomicDataDict.FORCE_KEY in out:
                self.results["forces"] = (self.energy_units_to_eV / self.length_units_to_A) * (
                    out[AtomicDataDict.FORCE_KEY].detach().cpu().numpy()
                )
            if AtomicDataDict.STRESS_KEY in out:
                stress = out[AtomicDataDict.STRESS_KEY].detach().cpu().numpy()
                stress = stress.reshape(3, 3) * (self.energy_units_to_eV / self.length_units_to_A**3)
                self.results["stress"] = full_3x3_to_voigt_6_stress(stress)

            self.save_extra_outputs(out)

        def heat_flux_forward(self, atoms) -> tuple[torch.Tensor, torch.Tensor]:
            """Run a fresh forward pass for heat-flux computation.

            The pass is independent of the most recent :meth:`calculate` call
            to avoid any graph-invalidation surprises across multiple backward
            passes (the Green-Kubo O(N) path does three, the O(N^2) path
            does N).

            Args:
                atoms: ASE atoms object representing the current MD frame.

            Returns:
                tuple[torch.Tensor, torch.Tensor]:
                    - ``per_atom_energies`` of shape ``(N,)`` in eV.
                    - ``positions`` of shape ``(N, 3)`` in angstrom with
                      ``requires_grad_(True)``.

            Raises:
                RuntimeError: If the underlying NequIP model does not emit
                    ``PER_ATOM_ENERGY_KEY`` (required for Eq. 8).
            """
            _orig_grad = torch.autograd.grad

            def _retain_grad(*args, **kwargs):
                return _orig_grad(*args, **{**kwargs, "retain_graph": True})

            with torch.enable_grad():
                data = self.atoms_to_data(atoms)
                pos_t = (
                    data[AtomicDataDict.POSITIONS_KEY].detach().clone().requires_grad_(True)
                )
                data[AtomicDataDict.POSITIONS_KEY] = pos_t

                torch.autograd.grad = _retain_grad
                try:
                    out = self.call_model(data)
                finally:
                    torch.autograd.grad = _orig_grad

                pae = out.get(AtomicDataDict.PER_ATOM_ENERGY_KEY)
                if pae is None:
                    raise RuntimeError(
                        "NequIP model does not expose per-atom energies; "
                        "heat-flux forward requires PER_ATOM_ENERGY_KEY in the model output."
                    )
                # These multiplications must stay inside enable_grad: ASE wraps
                # calculator calls in torch.no_grad(), so any operation on pae/pos_t
                # that happens outside the context creates a detached tensor with no
                # grad_fn, breaking the backward pass in HeatFluxObserver._heat_flux.
                e_t_ev = pae.squeeze(-1) * self.energy_units_to_eV
                pos_t_angstrom = pos_t * self.length_units_to_A
            return e_t_ev, pos_t_angstrom

    return AutoDiffNequIPCalculator
