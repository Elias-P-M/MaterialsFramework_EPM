"""Tests for HSolubilityTransformation."""

from __future__ import annotations

import pytest
from pymatgen.io.ase import AseAtomsAdaptor

from materialsframework.transformations.h_solubility import HSolubilityTransformation


def test_init() -> None:
    """Transformation initializes expected default state."""
    t = HSolubilityTransformation()
    assert t._sqs_gen is None
    assert t.host_structure is None
    assert t.structures == {}


def test_apply_transformation_generates_octahedral_and_tetrahedral(bcc_fe) -> None:
    """apply_transformation creates one structure per default site type."""
    t = HSolubilityTransformation()
    t.apply_transformation(bcc_fe)

    assert set(t.structures) == {"octahedral", "tetrahedral"}
    assert len(t.structures["octahedral"]) == 1
    assert len(t.structures["tetrahedral"]) == 1

    for generated in t.structures["octahedral"] + t.structures["tetrahedral"]:
        assert generated.num_sites == bcc_fe.num_sites + 1
        assert generated.composition["H"] == 1


def test_apply_transformation_accepts_ase_atoms(bcc_fe) -> None:
    """apply_transformation accepts ASE Atoms inputs."""
    ase_bcc_fe = AseAtomsAdaptor.get_atoms(bcc_fe)
    t = HSolubilityTransformation()
    t.apply_transformation(ase_bcc_fe, site_types=("octahedral",))

    assert len(t.structures["octahedral"]) == 1


def test_apply_transformation_validates_site_types(bcc_fe) -> None:
    """Unknown site labels raise ValueError."""
    t = HSolubilityTransformation()

    with pytest.raises(ValueError, match="Invalid site type"):
        t.apply_transformation(bcc_fe, site_types=("bridge",))
