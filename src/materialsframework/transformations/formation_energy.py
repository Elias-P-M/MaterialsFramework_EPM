"""This module provides a class to generate structures for formation energy calculations.

The `FormationEnergyTransformation` class facilitates the generation of structures
required for the calculation of formation energies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from ase.build import bulk
from pymatgen.core import Element
from pymatgen.io.ase import AseAtomsAdaptor

if TYPE_CHECKING:
    from pymatgen.core import Structure

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"

_FALLBACK_RADIUS = 1.4  # Å — used for elements without a tabulated atomic radius


class FormationEnergyTransformation:
    """A class used to generate structures for formation energy calculations.

    The `FormationEnergyTransformation` class provides methods to generate structures
    for the calculation of formation energies.

    For each element in the compound, three candidate crystal structures (FCC, BCC, HCP)
    are generated using the element's atomic radius to estimate the lattice parameters.
    The analyzer is responsible for relaxing all candidates with the same MLIP and
    selecting the lowest-energy phase as the elemental reference.

    Attributes:
        pure_structures (list): A list of ``(candidates, n_atoms)`` tuples, where
            ``candidates`` is a list of FCC/BCC/HCP pymatgen ``Structure`` objects and
            ``n_atoms`` is the count of that element in the compound.
    """

    def __init__(self):
        """Initializes the `FormationEnergyTransformation` object."""
        self.ase_adaptor = AseAtomsAdaptor()
        self.pure_structures: list[tuple[list[Structure], int]] = []

    def apply_transformation(self, structure: Atoms | Structure) -> None:
        """Apply the transformation to generate elemental reference structures.

        For each element present in ``structure``, three candidate crystal structures
        (FCC, BCC, HCP) are generated using the element's empirical atomic radius to
        estimate lattice parameters. The ``pure_structures`` list is reset on every call
        so that repeated invocations on the same instance remain correct.

        Args:
            structure (Atoms | Structure): The compound structure whose composition
                determines which elemental references are generated.
        """
        self.pure_structures = []

        if isinstance(structure, Atoms):
            structure = self.ase_adaptor.get_structure(structure)

        for element, num in structure.composition.get_el_amt_dict().items():
            r = float(Element(element).atomic_radius or _FALLBACK_RADIUS)

            a_fcc = 2 * r * np.sqrt(2)
            a_bcc = 4 * r / np.sqrt(3)
            a_hcp = 2 * r
            c_hcp = a_hcp * np.sqrt(8 / 3)

            candidates = [
                self.ase_adaptor.get_structure(bulk(element, "fcc", a=a_fcc)),
                self.ase_adaptor.get_structure(bulk(element, "bcc", a=a_bcc)),
                self.ase_adaptor.get_structure(bulk(element, "hcp", a=a_hcp, c=c_hcp)),
            ]

            self.pure_structures.append((candidates, int(num)))
