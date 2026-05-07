r"""Heat-flux observer for the Green-Kubo workflow.

This module provides :class:`HeatFluxObserver`, an ASE dynamics observer that
computes the microscopic heat flux at each MD step using automatic
differentiation of per-atom energies through atomic positions. The observer is
back-end agnostic: it delegates the forward pass to a user-supplied ASE
calculator that satisfies the *auto-differentiable* protocol described below.

Reference
=========
The heat-flux formulation follows
Langer, Knoop, Carbogno, Scheffler and Rupp,
*Phys. Rev. B* **108**, L100302 (2023), equation 8 (the "unfolded" semilocal
heat flux), together with the convective term of Eq. 3:

.. math::

    V\,\mathbf{J}^\alpha &= \sum_I \!\left(\tfrac{1}{2} m_I v_I^2 + U_I\right) v_I^\alpha
        + \sum_{I,J} (R_I^\alpha - R_J^\alpha)\,
            \frac{\partial U_I}{\partial \mathbf{R}_J}\!\cdot\!\mathbf{v}_J.

Equivalent and used here: the Hardy form of a per-atom energy virial
:math:`\Pi_J^{\alpha\beta} = \sum_I (R_I^\alpha - R_J^\alpha)
\partial U_I/\partial R_J^\beta` contracted with velocities,

.. math::

    V\,\mathbf{J}_{\text{pot}}^\alpha &=
        \sum_{J,\beta}\Pi_J^{\alpha\beta} v_J^\beta,
    \qquad
    V\,\mathbf{J}_{\text{conv}}^\alpha = \sum_I (KE_I + U_I)\,v_I^\alpha.

Implementation
==============
The observer materialises the per-atom-energy Jacobian
:math:`\partial U_I/\partial \mathbf{R}_J` (one backward pass per atom) and
contracts it with **minimum-image** displacement vectors (via
:func:`ase.geometry.find_mic`).  This is the same construction as the
"hardy virials" path in the GKnet reference implementation (Langer et al.,
their `compute_hardy_virials`) and is exact for any simulation cell whose
side length exceeds twice the model's effective cutoff (the standard
half-box criterion for MIC neighbour lists).

A naive "barycenter" form
:math:`V\mathbf{J} = \sum_J \mathbf{R}_J (\mathbf{F}_J\!\cdot\!\mathbf{v}_J)
+ \nabla\!\sum_I U_I R_I^\alpha\!\cdot\!\mathbf{v}` (three backward passes
instead of N) is **not used**: it is only correct when all atom positions
remain in a single, unwrapped image, which is never the case in a periodic
MD simulation — every neighbour pair across the cell boundary contributes
a spurious lattice-vector to :math:`(R_I - R_J)` that the autograd
Jacobian cannot see.  Earlier revisions of this module exposed that path
behind a ``full_jacobian=False`` flag; it has been removed because it is
silently wrong for any periodic cell, regardless of size.

Auto-differentiable calculator protocol
=======================================
The ASE calculator passed to :class:`HeatFluxObserver` must provide:

- ``heat_flux_forward(atoms)`` — runs a forward pass with position gradients
  tracked and returns a tuple ``(per_atom_energy_tensor, position_tensor)``
  where ``per_atom_energy_tensor`` has shape ``(N,)`` in eV and
  ``position_tensor`` has shape ``(N, 3)`` in angstrom with
  ``requires_grad_(True)``.

The only ``auto_diff=True`` implementation shipped in this MVP is
:class:`materialsframework.calculators.nequip.NequIPCalculator`.

Known limitation:
    The cost is O(N) backward passes per heat-flux sample; for large cells
    or long trajectories this dominates wall-clock time.  A faster but
    still correct path would take a single backward pass through the
    model's pairwise-distance head (gknet's "fast virials"), but that
    requires the model to expose its internal pairwise tensors and is
    not implemented here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class HeatFluxObserver:
    r"""ASE MD observer that records the potential and convective heat flux.

    At each call the observer runs a fresh forward pass on the supplied
    auto-differentiable calculator (via its ``heat_flux_forward(atoms)``
    method), materialises the per-atom-energy Jacobian
    :math:`\partial U_I/\partial \mathbf{R}_J` (N backward passes), and
    contracts it with minimum-image displacements to evaluate Langer Eq. 8.
    The result is split into the potential contribution
    :math:`\mathbf{J}_{\text{pot}}` (the sum-over-pairs term) and the
    convective contribution
    :math:`\mathbf{J}_{\text{conv}} = \sum_I (KE_I + U_I)\,\mathbf{v}_I/V`
    of Eq. 3. Both are stored in ASE internal flux units
    ``eV * sqrt(eV/amu) / A^3``.

    The observer is attached to an ASE dynamics object via
    ``dyn.attach(observer, interval=N)`` so that it fires every ``N`` steps.

    Attributes:
        atoms (Atoms): ASE atoms object driven by the dynamics.
        calc (Calculator): Auto-differentiable ASE calculator.
        J_pot (list[np.ndarray]): Potential heat flux samples, one
            ``(3,)`` vector per call.
        J_conv (list[np.ndarray]): Convective heat flux samples, one
            ``(3,)`` vector per call.
    """

    def __init__(
        self,
        atoms: Atoms,
        ase_calculator: Calculator,
    ) -> None:
        """Initialise the observer.

        Args:
            atoms (Atoms): The ASE atoms object integrated by the dynamics.
            ase_calculator (Calculator): An auto-differentiable ASE calculator
                implementing the ``heat_flux_forward`` protocol described in
                the module docstring.

        Raises:
            TypeError: If ``ase_calculator`` does not expose a
                ``heat_flux_forward`` method.
        """
        if not hasattr(ase_calculator, "heat_flux_forward"):
            raise TypeError(
                f"{type(ase_calculator).__name__} does not expose a "
                "'heat_flux_forward(atoms)' method. The HeatFluxObserver "
                "requires an auto-differentiable calculator; for NequIP, "
                "instantiate NequIPCalculator(auto_diff=True)."
            )
        self.atoms = atoms
        self.calc = ase_calculator
        self.J_pot: list[np.ndarray] = []
        self.J_conv: list[np.ndarray] = []

    def __call__(self) -> None:
        """Compute and store one heat-flux sample (J_pot, J_conv)."""
        e_t, pos_t = self.calc.heat_flux_forward(self.atoms)
        j_pot, j_conv = self._heat_flux(e_t, pos_t)
        self.J_pot.append(j_pot)
        self.J_conv.append(j_conv)

    def _heat_flux(self, e_t, pos_t) -> tuple[np.ndarray, np.ndarray]:
        """Per-atom-energy Jacobian heat flux with MIC pairwise displacements.

        Builds ``jac[I, J, b] = dU_I/dR_J[b]`` with N backward passes and
        contracts it with minimum-image displacement vectors,
        :math:`R_I - R_J`, to evaluate Langer Eq. 8 directly.
        """
        import torch
        from ase.geometry import find_mic

        n_atoms = len(e_t)
        jac = torch.zeros(n_atoms, n_atoms, 3, device=pos_t.device, dtype=pos_t.dtype)
        for i in range(n_atoms):
            jac[i] = torch.autograd.grad(
                e_t[i],
                pos_t,
                retain_graph=(i < n_atoms - 1),
                create_graph=False,
            )[0]
        jac_np = jac.detach().cpu().numpy()  # eV/A

        velocities = self.atoms.get_velocities()  # sqrt(eV/amu) (Å / ASE_time)
        positions = self.atoms.get_positions()  # Å (may be unwrapped)
        u_i = e_t.detach().cpu().numpy()  # eV
        masses = self.atoms.get_masses()  # amu
        volume = self.atoms.get_volume()  # Å^3

        ke_i = 0.5 * masses * np.sum(velocities**2, axis=1)  # eV
        e_i = u_i + ke_i  # eV

        # Convective flux J_conv = sum_I (KE_I + U_I) v_I / V (Langer Eq. 3).
        j_conv = np.einsum("i,ia->a", e_i, velocities) / volume

        # Pairwise displacement r_ij = R_i - R_j.  MIC-correct for any
        # neighbour pair when the cell is at least 2 * cutoff on every
        # axis (the same condition under which the model's neighbour
        # list is well-defined).  For non-interacting pairs (outside
        # cutoff) the Jacobian column is zero so their contribution is
        # zero regardless.
        r_ij = positions[:, None, :] - positions[None, :, :]
        if np.any(self.atoms.get_pbc()):
            flat_ij = r_ij.reshape(-1, 3)
            mic_ij, _ = find_mic(flat_ij, self.atoms.get_cell(), self.atoms.get_pbc())
            r_ij = mic_ij.reshape(n_atoms, n_atoms, 3)

        # power_flow[i, j] = (dU_i/dR_j) . v_j
        # J_pot[a] = (1/V) sum_{i,j} r_ij[i,j,a] * power_flow[i,j]
        #         = (1/V) sum_{i,j} (R_i - R_j)^a (dU_i/dR_j) . v_j
        power_flow = np.einsum("ijb,jb->ij", jac_np, velocities)
        j_pot = np.einsum("ija,ij->a", r_ij, power_flow) / volume

        return j_pot, j_conv
