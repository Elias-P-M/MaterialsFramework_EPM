"""This module provides a class to generate BCC interstitial structures for hydrogen solubility calculations.

The `HSolubilityTransformation` class generates host and H-inserted structures at
octahedral/tetrahedral sites for composition-driven or structure-driven H-solubility
workflows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from pymatgen.core import Composition
from pymatgen.io.ase import AseAtomsAdaptor

from materialsframework.tools.sqsgen import SqsGenerator

if TYPE_CHECKING:
    from pymatgen.core import Structure

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"

_VALID_SITE_TYPES = {"octahedral", "tetrahedral"}
_SITE_FRACTIONS = {
    "octahedral": [
        (0.5, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, 0.5),
        (0.5, 0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
    ],
    "tetrahedral": [
        (0.25, 0.5, 0.0),
        (0.75, 0.5, 0.0),
        (0.5, 0.25, 0.0),
        (0.5, 0.75, 0.0),
        (0.25, 0.0, 0.5),
        (0.75, 0.0, 0.5),
        (0.5, 0.0, 0.25),
        (0.5, 0.0, 0.75),
        (0.0, 0.25, 0.5),
        (0.0, 0.75, 0.5),
        (0.0, 0.5, 0.25),
        (0.0, 0.5, 0.75),
    ],
}


class HSolubilityTransformation:
    """A transformation that generates H-interstitial BCC structures.

    The transformation can start from an explicit host structure or generate a BCC
    host through ``SqsGenerator`` for composition-driven workflows.
    """

    def __init__(self, sqs_gen: SqsGenerator | None = None) -> None:
        """Initializes the transformation.

        Args:
            sqs_gen (SqsGenerator | None): Optional SQS generator for composition-driven host generation.
        """
        self.ase_adaptor = AseAtomsAdaptor()
        self._sqs_gen = sqs_gen

        self.host_structure: Structure | None = None
        self.structures: dict[str, list[Structure]] = {}

    def apply_transformation(
        self,
        structure: Atoms | Structure | None = None,
        composition: Composition | str | None = None,
        supercell_size: tuple[int, int, int] = (1, 1, 1),
        site_types: tuple[str, ...] = ("octahedral", "tetrahedral"),
        max_sites_per_type: int | None = 1,
    ) -> None:
        """Generates structures with H inserted into BCC octahedral/tetrahedral sites.

        Args:
            structure (Atoms | Structure | None): Host structure.
            composition (Composition | str | None): Composition used with ``SqsGenerator`` when ``structure`` is omitted.
            supercell_size (tuple[int, int, int], optional): Supercell replication for the host.
            site_types (tuple[str, ...], optional): Site families to generate.
            max_sites_per_type (int | None, optional): Maximum generated structures per site type.
                Use ``None`` to keep all generated candidates.

        Raises:
            ValueError: If core inputs are invalid.
        """
        self._validate_site_types(site_types)

        if structure is None and composition is None:
            raise ValueError("Either `structure` or `composition` must be provided.")

        if max_sites_per_type is not None and max_sites_per_type < 1:
            raise ValueError("`max_sites_per_type` must be >= 1 when provided.")

        host_structure = self._build_host_structure(structure, composition, supercell_size)
        self.host_structure = host_structure
        self.structures = {}

        for site_type in site_types:
            generated = self._insert_hydrogen(host_structure, site_type, max_sites_per_type)
            if not generated:
                raise ValueError(f"No valid `{site_type}` interstitial sites were generated.")
            self.structures[site_type] = generated

    def _build_host_structure(
        self,
        structure: Atoms | Structure | None,
        composition: Composition | str | None,
        supercell_size: tuple[int, int, int],
    ) -> Structure:
        """Create host structure from direct input or SQS generation.

        Args:
            structure: Direct host structure, if provided.
            composition: Composition used to build SQS host if structure is absent.
            supercell_size: Supercell replication tuple.

        Raises:
            ValueError: If neither structure nor composition can produce a host.

        Returns:
            Host structure for interstitial insertion.
        """
        if structure is not None:
            if isinstance(structure, Atoms):
                structure = self.ase_adaptor.get_structure(structure)

            host = structure.copy()
            if supercell_size != (1, 1, 1):
                host.make_supercell(supercell_size)
            return host

        if isinstance(composition, str):
            composition = Composition(composition)

        if composition is None:
            raise ValueError("`composition` cannot be None when `structure` is not provided.")

        sqs_result = self.sqs_gen.generate(
            composition=composition,
            crystal_structure="bcc",
            supercell_size=supercell_size,
        )
        return sqs_result["structure"]

    @staticmethod
    def _validate_site_types(site_types: tuple[str, ...]) -> None:
        """Validate requested site type labels.

        Args:
            site_types: Requested site families.

        Raises:
            ValueError: If any site type is not supported.
        """
        invalid = [site_type for site_type in site_types if site_type not in _VALID_SITE_TYPES]
        if invalid:
            raise ValueError(f"Invalid site type(s): {invalid}. Allowed values are: {sorted(_VALID_SITE_TYPES)}.")

    def _insert_hydrogen(
        self,
        host_structure: Structure,
        site_type: str,
        max_sites_per_type: int | None,
        overlap_tol: float = 0.3,
    ) -> list[Structure]:
        """Insert hydrogen into candidate interstitial sites.

        Args:
            host_structure: Host structure receiving interstitial hydrogen.
            site_type: Interstitial family (`octahedral` or `tetrahedral`).
            max_sites_per_type: Maximum generated structures for this family.
            overlap_tol: Proximity check cutoff in Angstrom.

        Returns:
            Generated host-plus-hydrogen structures.
        """
        generated: list[Structure] = []
        seen: set[tuple[float, float, float]] = set()

        for frac_coords in _SITE_FRACTIONS[site_type]:
            frac = np.mod(np.array(frac_coords, dtype=float), 1.0)
            frac_key = tuple(np.round(frac, 8))
            if frac_key in seen:
                continue

            cart = host_structure.lattice.get_cartesian_coords(frac)
            if host_structure.get_sites_in_sphere(cart, overlap_tol):
                continue

            defect_structure = host_structure.copy()
            defect_structure.append("H", frac, coords_are_cartesian=False, validate_proximity=False)
            generated.append(defect_structure)
            seen.add(frac_key)

            if max_sites_per_type is not None and len(generated) >= max_sites_per_type:
                break

        return generated

    @property
    def sqs_gen(self) -> SqsGenerator:
        """Return the SQS generator used in composition-driven workflows.

        Returns:
            SQS generator instance.
        """
        if self._sqs_gen is None:
            self._sqs_gen = SqsGenerator()
        return self._sqs_gen
