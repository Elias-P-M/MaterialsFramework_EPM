r"""Green-Kubo thermal conductivity analyzer.

The :class:`GreenKuboAnalyzer` computes the lattice thermal conductivity
:math:`\kappa` of a material from equilibrium molecular dynamics using the
Green-Kubo relation. The heat flux follows the auto-differentiation
formulation of Langer et al., *Phys. Rev. B* **108**, L100302 (2023),
equation 8 (the "unfolded" semilocal heat flux), together with the
convective term of Eq. 3. See
:mod:`materialsframework.tools.heat_flux` for the microscopic flux
implementation and :mod:`materialsframework.tools.autocorrelation` for the
FFT autocorrelation and GK integration utilities.

The analyzer expects a calculator that (a) exposes per-atom potential
energies (``"energies"`` in
:attr:`~materialsframework.tools.calculator.BaseCalculator.AVAILABLE_PROPERTIES`),
(b) inherits from
:class:`~materialsframework.tools.md.BaseMDCalculator` so that the MD
integrator setup is delegated to the calculator itself, and (c) wraps an
auto-differentiable ASE calculator exposing ``heat_flux_forward(atoms)``.
The MVP implementation supports
:class:`materialsframework.calculators.nequip.NequIPCalculator` with
``auto_diff=True``; other per-atom-energy MLIPs
(``SevenNetCalculator``, ``HIENetCalculator``, MACE ``node_energies``) can be
added by implementing the same protocol.

Protocol
========
1. (optional) Relax the input structure.
2. Prepare ASE atoms (seed MB velocities, zero CoM/rotation) via
   ``calculator.prepare_atoms``.
3. Equilibration phase in the user-selected ensemble
   (``nvt_nose_hoover`` or ``npt_berendsen``/``npt_nose_hoover``). For NPT
   ensembles the cell is time-averaged over the second half of the run and
   the final configuration is rescaled to that mean cell
   (``scale_atoms=True``) — positions and velocities are the *instantaneous*
   snapshot from the last step (this is the correct NPT→NVE handoff:
   averaging positions would destroy the thermal state by placing atoms at
   lattice minima while their velocities correspond to finite-T).
4. (opt-in) Short NVT Nose-Hoover re-equilibration at the mean cell to
   tightly pin :math:`T` before production.
5. Unconditional velocity rescaling so :math:`T_{\text{kin}}` equals the
   target temperature at the NVE entry point.
6. NVE production with :class:`HeatFluxObserver` and
   :class:`TrajectoryObserver` attached.
7. Integrate the heat-flux autocorrelation (with empirical-mean
   subtraction; see :mod:`materialsframework.tools.autocorrelation`)
   per Langer et al. to produce :math:`\kappa`.

The returned dictionary contains :math:`\kappa` decomposed into
potential (:math:`\mathbf{J}_{\text{pot}}`, the sum-over-pairs term of
Langer Eq. 8) and convective (:math:`\mathbf{J}_{\text{conv}} =
\sum_I (KE_I + U_I)\,\dot{\mathbf{R}}_I/V`, Eq. 3) contributions,
alongside the total.

Known limitations (tracked separately):
    - Heat flux costs O(N) backward passes per sample (one per atom for
      the per-atom-energy Jacobian).  A single-backward "fast virials"
      path through the model's pairwise distances would be much faster
      but requires the calculator to expose internal pairwise tensors.
    - No cepstral-analysis noise reduction on the HFACF
      (Knoop et al., PRB 107, 224304, 2023).
    - Does not work with DFT, eventually we aim to include DFT and
    semi-DFT compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from pymatgen.core import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from materialsframework.tools.autocorrelation import green_kubo_tensor
from materialsframework.tools.heat_flux import HeatFluxObserver
from materialsframework.tools.md import BaseMDCalculator
from materialsframework.tools.trajectory import TrajectoryObserver

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms

    from materialsframework.tools.calculator import BaseCalculator

__author__ = "Elias Martin"
__email__ = "epm1337@tamu.edu"


class GreenKuboAnalyzer:
    r"""Thermal conductivity from equilibrium MD via the Green-Kubo relation.

    The analyzer delegates MD parameters and integrator setup to the
    supplied :class:`~materialsframework.tools.md.BaseMDCalculator`; the
    keyword arguments on :meth:`__init__` that shadow MD parameters
    (``timestep``, ``temperature``, ``pressure``, ``ttime``, ``taut``,
    ``taup``, ``compressibility``) act as *overrides*: if provided, they
    are pushed onto the calculator before each :meth:`calculate` call, so
    the user can tweak a single run without constructing a new calculator.
    """

    _MD_OVERRIDE_FIELDS: tuple[str, ...] = (
        "timestep",
        "temperature",
        "pressure",
        "ttime",
        "taut",
        "taup",
        "compressibility",
    )

    def __init__(
        self,
        calculator: BaseCalculator | None = None,
        equilibration_ensemble: Literal[
            "nvt_nose_hoover", "npt_berendsen", "npt_nose_hoover"
        ] = "npt_berendsen",
        equilibration_steps: int = 5000,
        nvt_reequilibration_steps: int = 0,
        production_steps: int = 50000,
        sampling_interval: int = 1,
        average_cell_second_half: bool = True,
        unbiased_acf: bool = True,
        subtract_mean_acf: bool = True,
        t_cutoff_fs: float | None = None,
        timestep: float | None = None,
        temperature: float | None = None,
        pressure: float | None = None,
        ttime: float | None = None,
        taut: float | None = None,
        taup: float | None = None,
        compressibility: float | None = None,
    ) -> None:
        r"""Initializes the GreenKuboAnalyzer with the specified MD parameters.

        Args:
            calculator (BaseCalculator | None, optional): Calculator used for
                relaxation and MD. Must (a) expose ``"energies"`` in
                ``AVAILABLE_PROPERTIES``, (b) inherit from
                :class:`~materialsframework.tools.md.BaseMDCalculator` so the
                MD integrator setup is delegated to the calculator, and (c)
                expose ``heat_flux_forward(atoms)`` on its ASE calculator.
                Defaults to
                :class:`~materialsframework.calculators.nequip.NequIPCalculator`
                with ``auto_diff=True`` when not supplied.
            equilibration_ensemble (Literal["nvt_nose_hoover", "npt_berendsen", "npt_nose_hoover"], optional):
                Ensemble for the first equilibration phase. Defaults to
                ``"npt_berendsen"``.
            equilibration_steps (int, optional): Number of MD steps in the
                first equilibration phase. Defaults to 5000.
            nvt_reequilibration_steps (int, optional): Number of NVT
                Nose-Hoover re-equilibration steps run *after* the first
                equilibration phase (at the mean NPT cell if applicable).
                Opt-in feature recommended when the first phase uses a
                loose Berendsen coupling: tightens :math:`T` before NVE.
                Defaults to 0 (no re-equilibration).
            production_steps (int, optional): Number of NVE production MD
                steps during which the heat flux is sampled. Defaults to
                50000.
            sampling_interval (int, optional): The heat flux (and trajectory
                snapshot) is recorded every ``sampling_interval`` MD steps
                during production. Defaults to 1 (every step).
            average_cell_second_half (bool, optional): If True and the first
                equilibration ensemble is NPT, record the cell during the
                second half of equilibration and apply the time-averaged
                cell (with ``scale_atoms=True``) before the next phase. Has
                no effect in NVT. Defaults to True.
            unbiased_acf (bool, optional): If True (default) the heat-flux
                autocorrelation uses the unbiased estimator (each lag ``t``
                normalised by ``T - t``); if False the biased estimator
                (every lag normalised by ``T``) is used, which damps the
                high-lag noise to zero and usually gives a much better
                behaved running :math:`\kappa(\tau)` for finite
                trajectories. Defaults to True.
            subtract_mean_acf (bool, optional): If True (default) subtract
                the empirical mean of :math:`\mathbf{J}(t)` before
                autocorrelating.  Removes the finite-sample DC bias that
                would otherwise add a linear ramp to the running
                :math:`\kappa(\tau)`.  Set to False only to reproduce the
                naive non-subtracted formulation for diagnostics.
            t_cutoff_fs (float | None, optional): If provided, the running
                Green-Kubo integral (and its tensor counterpart) is
                truncated at lag ``t_cutoff_fs`` (in fs); useful to stop
                the random-walk noise of the unbiased estimator from
                polluting the result. Defaults to None (full-range
                integration).
            timestep (float | None, optional): If provided, overrides the
                calculator's MD timestep (fs) for this analyzer. Defaults
                to None (use ``calculator.timestep``).
            temperature (float | None, optional): If provided, overrides the
                calculator's target temperature (K). Defaults to None.
            pressure (float | None, optional): If provided, overrides the
                calculator's target pressure (atm — matches
                :class:`BaseMDCalculator` convention). Defaults to None.
            ttime (float | None, optional): If provided, overrides the
                Nose-Hoover thermostat coupling time (fs). Defaults to None.
            taut (float | None, optional): If provided, overrides the
                Berendsen thermostat coupling time (fs). Defaults to None.
            taup (float | None, optional): If provided, overrides the
                Berendsen barostat coupling time (fs). Defaults to None.
            compressibility (float | None, optional): If provided, overrides
                the isothermal compressibility (bar⁻¹). Defaults to None.

        Note:
            The MD-parameter overrides are **mutating pushes** onto
            ``self.calculator``: calling :meth:`calculate` writes each
            non-None override onto the calculator before the run. Subsequent
            uses of the same calculator instance (including outside this
            analyzer) will see the overridden values.

        Raises:
            ValueError: If ``equilibration_ensemble`` is not supported.
        """
        if equilibration_ensemble not in (
            "nvt_nose_hoover",
            "npt_berendsen",
            "npt_nose_hoover",
        ):
            raise ValueError(
                f"equilibration_ensemble must be one of 'nvt_nose_hoover', "
                f"'npt_berendsen', 'npt_nose_hoover'; got {equilibration_ensemble!r}."
            )

        self._calculator = calculator
        self.equilibration_ensemble = equilibration_ensemble
        self.equilibration_steps = equilibration_steps
        self.nvt_reequilibration_steps = nvt_reequilibration_steps
        self.production_steps = production_steps
        self.sampling_interval = sampling_interval
        self.average_cell_second_half = average_cell_second_half
        self.unbiased_acf = unbiased_acf
        self.subtract_mean_acf = subtract_mean_acf
        self.t_cutoff_fs = t_cutoff_fs

        self._overrides: dict[str, Any] = {
            field: value
            for field, value in (
                ("timestep", timestep),
                ("temperature", temperature),
                ("pressure", pressure),
                ("ttime", ttime),
                ("taut", taut),
                ("taup", taup),
                ("compressibility", compressibility),
            )
            if value is not None
        }

        self.ase_adaptor = AseAtomsAdaptor()

    @property
    def calculator(self) -> BaseCalculator:
        """Returns the calculator used for relaxation and MD.

        If no calculator was supplied at construction time, a default
        :class:`~materialsframework.calculators.nequip.NequIPCalculator` with
        ``auto_diff=True`` is created on first access.

        Returns:
            BaseCalculator: The calculator instance.
        """
        if self._calculator is None:
            from materialsframework.calculators.nequip import NequIPCalculator

            self._calculator = NequIPCalculator(auto_diff=True)
        return self._calculator

    @property
    def _md(self) -> BaseMDCalculator:
        """Return :attr:`calculator` narrowed to :class:`BaseMDCalculator`.

        Runtime-checks the inheritance and carries the narrowed type to the
        static analyser so callers can touch MD-specific attributes
        (``timestep``, ``temperature``, ``build_dyn``, ``prepare_atoms``, …)
        without repeated casts.
        """
        calculator = self.calculator
        if not isinstance(calculator, BaseMDCalculator):
            raise TypeError(
                f"GreenKuboAnalyzer requires a BaseMDCalculator subclass for MD setup; "
                f"{type(calculator).__name__} is not one."
            )
        return calculator

    def _apply_overrides(self) -> None:
        """Push any non-None ``__init__`` MD-parameter overrides onto the calculator."""
        for field, value in self._overrides.items():
            setattr(self.calculator, field, value)

    def calculate(
        self,
        structure: Structure | Atoms | Molecule,
        is_relaxed: bool = False,
    ) -> dict[str, Any]:
        r"""Runs the Green-Kubo workflow on the given structure.

        Args:
            structure (Structure | Atoms | Molecule): Input structure.
                Accepts a pymatgen ``Structure`` / ``Molecule`` or an ASE
                ``Atoms`` object.
            is_relaxed (bool, optional): If True, skip the initial relaxation
                and run MD directly on the input structure. Defaults to
                False.

        Returns:
            dict[str, Any]: Results dictionary with keys:

                - ``kappa`` (float): Total isotropic thermal conductivity in
                  W/(m*K), equal to ``(1/3) * trace(kappa_tensor)``.
                - ``kappa_pot`` (float): Isotropic potential-only contribution
                  (the sum-over-pairs term of Langer Eq. 8).
                - ``kappa_conv`` (float): Isotropic convective-only
                  contribution (from Langer Eq. 3).
                - ``kappa_tensor`` (np.ndarray): Full thermal-conductivity
                  tensor, shape ``(3, 3)``, in W/(m*K). Equal to
                  ``kappa_tensor_pot + kappa_tensor_conv + cross-term``.
                - ``kappa_tensor_pot`` (np.ndarray): Potential-only tensor,
                  shape ``(3, 3)``.
                - ``kappa_tensor_conv`` (np.ndarray): Convective-only tensor,
                  shape ``(3, 3)``.
                - ``kappa_cumulative`` (np.ndarray): Running :math:`\kappa(\tau)`
                  from the total heat flux, shape ``(T,)``, for plateau checks.
                - ``kappa_tensor_cumulative`` (np.ndarray): Running tensor
                  :math:`\kappa_{\alpha\beta}(\tau)`, shape ``(T, 3, 3)``.
                - ``J_pot`` (np.ndarray): Potential heat flux, shape ``(T, 3)``,
                  in ASE internal units ``eV * sqrt(eV/amu) / A^3``.
                - ``J_conv`` (np.ndarray): Convective heat flux, shape ``(T, 3)``.
                - ``dt_sample_fs`` (float): Time between heat-flux samples (fs).
                - ``temperature`` (float): Target temperature (K).
                - ``volume`` (float): Production-run cell volume (A^3).
                - ``trajectory`` (TrajectoryObserver): The MD trajectory
                  recorded during production (temperatures, velocities,
                  per-atom potential energies, forces, stresses).
                - ``final_structure`` (Structure): The atomic configuration
                  at the end of the production run.

        Raises:
            ValueError: If ``self.calculator`` does not expose ``"energies"``
                in ``AVAILABLE_PROPERTIES``, is not a ``BaseMDCalculator``
                subclass, or if its ASE calculator does not implement
                ``heat_flux_forward(atoms)``.
        """
        if "energies" not in self.calculator.AVAILABLE_PROPERTIES:
            raise ValueError(
                f"GreenKuboAnalyzer requires a calculator exposing per-atom 'energies' in "
                f"AVAILABLE_PROPERTIES; {type(self.calculator).__name__} does not."
            )

        md = self._md  # runtime check + narrowing for build_dyn / prepare_atoms / params

        self._apply_overrides()

        if isinstance(structure, Molecule):
            structure = self.ase_adaptor.get_structure(self.ase_adaptor.get_atoms(structure))

        if not is_relaxed:
            structure = self.calculator.relax(structure)["final_structure"]

        atoms = cast(
            "Atoms",
            self.ase_adaptor.get_atoms(structure) if isinstance(structure, Structure) else structure.copy(),
        )

        md.prepare_atoms(atoms)

        ase_calc = md.calculator
        if not hasattr(ase_calc, "heat_flux_forward"):
            raise ValueError(
                f"The ASE calculator returned by {type(md).__name__}.calculator does not implement "
                "heat_flux_forward(atoms). For NequIPCalculator, pass auto_diff=True at construction time."
            )

        if self.equilibration_steps > 0:
            self._equilibrate(atoms)

        if self.nvt_reequilibration_steps > 0:
            self._thermalize(atoms)

        self._rescale_velocities_to_target(atoms)

        trajectory, hf_observer = self._run_production(atoms)

        dt_sample_fs = md.timestep * self.sampling_interval
        volume = atoms.get_volume()

        j_pot = np.asarray(hf_observer.J_pot)
        j_conv = np.asarray(hf_observer.J_conv)
        j_total = j_pot + j_conv

        target_temperature = float(md.temperature)

        kappa_tensor, kappa_tensor_cum = green_kubo_tensor(
            j_total, dt_sample_fs, target_temperature, volume,
            unbiased=self.unbiased_acf, t_cutoff_fs=self.t_cutoff_fs,
            subtract_mean=self.subtract_mean_acf,
        )
        kappa_tensor_pot, _ = green_kubo_tensor(
            j_pot, dt_sample_fs, target_temperature, volume,
            unbiased=self.unbiased_acf, t_cutoff_fs=self.t_cutoff_fs,
            subtract_mean=self.subtract_mean_acf,
        )
        kappa_tensor_conv, _ = green_kubo_tensor(
            j_conv, dt_sample_fs, target_temperature, volume,
            unbiased=self.unbiased_acf, t_cutoff_fs=self.t_cutoff_fs,
            subtract_mean=self.subtract_mean_acf,
        )

        kappa = float(np.trace(kappa_tensor)) / 3.0
        kappa_pot = float(np.trace(kappa_tensor_pot)) / 3.0
        kappa_conv = float(np.trace(kappa_tensor_conv)) / 3.0
        kappa_cum = np.trace(kappa_tensor_cum, axis1=1, axis2=2) / 3.0

        return {
            "kappa": kappa,
            "kappa_pot": kappa_pot,
            "kappa_conv": kappa_conv,
            "kappa_tensor": kappa_tensor,
            "kappa_tensor_pot": kappa_tensor_pot,
            "kappa_tensor_conv": kappa_tensor_conv,
            "kappa_cumulative": kappa_cum,
            "kappa_tensor_cumulative": kappa_tensor_cum,
            "J_pot": j_pot,
            "J_conv": j_conv,
            "dt_sample_fs": dt_sample_fs,
            "temperature": target_temperature,
            "volume": volume,
            "trajectory": trajectory,
            "final_structure": self.ase_adaptor.get_structure(cast("Any", atoms)),
        }

    def _equilibrate(self, atoms: Atoms) -> None:
        """Run the first equilibration phase.

        Builds the integrator via ``self.calculator.build_dyn`` in the
        requested ensemble. When the ensemble is NPT and
        ``average_cell_second_half`` is True, the cell is time-averaged over
        the second half of the run and applied with ``scale_atoms=True`` at
        the end — positions and velocities are the instantaneous snapshot
        from the last MD step; only the cell is averaged.
        """
        dyn = self._md.build_dyn(atoms, ensemble=self.equilibration_ensemble)

        is_npt = self.equilibration_ensemble.startswith("npt")
        if is_npt and self.average_cell_second_half:
            cell_samples: list[np.ndarray] = []
            step_counter = {"n": 0}
            half = self.equilibration_steps // 2

            def _record_cell() -> None:
                step_counter["n"] += 1
                if step_counter["n"] > half:
                    cell_samples.append(np.array(atoms.get_cell()))

            dyn.attach(_record_cell, interval=1)
            dyn.run(self.equilibration_steps)

            if cell_samples:
                atoms.set_cell(np.mean(cell_samples, axis=0), scale_atoms=True)
            return

        dyn.run(self.equilibration_steps)

    def _thermalize(self, atoms: Atoms) -> None:
        """Run a short NVT Nose-Hoover re-equilibration at the current cell."""
        dyn = self._md.build_dyn(atoms, ensemble="nvt_nose_hoover")
        dyn.run(self.nvt_reequilibration_steps)

    def _rescale_velocities_to_target(self, atoms: Atoms) -> None:
        r"""Rescale velocities so the instantaneous kinetic temperature equals the target.

        Cheap safety net at the transition into NVE: after equilibration,
        any residual mismatch between the instantaneous :math:`T_{\text{kin}}`
        and the target is removed by a uniform velocity rescale. The NVE run
        that follows therefore starts at exactly the target :math:`T`; its
        mean :math:`T` over the trajectory is then determined by equipartition
        on the total energy of this snapshot.
        """
        target_temperature = float(self._md.temperature)
        current_temperature = float(atoms.get_temperature())
        if current_temperature > 0.0:
            scale = np.sqrt(target_temperature / current_temperature)
            atoms.set_velocities(atoms.get_velocities() * scale)

    def _run_production(self, atoms: Atoms) -> tuple[TrajectoryObserver, HeatFluxObserver]:
        """Run the NVE production phase with the heat-flux and trajectory observers."""
        md = self._md
        dyn = md.build_dyn(atoms, ensemble="nve")

        trajectory = TrajectoryObserver(
            atoms,
            include_temperature=True,
            include_velocities=True,
            include_atomic_potentials=True,
        )
        hf_observer = HeatFluxObserver(atoms, md.calculator)

        dyn.attach(trajectory, interval=self.sampling_interval)
        dyn.attach(hf_observer, interval=self.sampling_interval)
        dyn.run(self.production_steps)

        return trajectory, hf_observer
