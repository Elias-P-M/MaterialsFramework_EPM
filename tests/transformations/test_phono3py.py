"""Tests for Phono3pyDisplacementTransformation."""

from __future__ import annotations

import pytest
from pymatgen.core import Structure

pytest.importorskip("phono3py")

from materialsframework.transformations.phono3py import (
    Phono3pyDisplacementTransformation,
)


def test_init() -> None:
    """Transformation initialises with all state attributes as None."""
    t = Phono3pyDisplacementTransformation()
    assert t.phonon is None
    assert t.phonon_displacements is None
    assert t.phonon_supercells_with_displacements is None
    assert t.supercell_displacements is None
    assert t.supercells_with_displacements is None


@pytest.mark.integration
def test_apply_transformation_populates_phonon(bcc_fe) -> None:
    """apply_transformation creates a Phono3py object."""
    t = Phono3pyDisplacementTransformation()
    t.apply_transformation(bcc_fe)
    assert t.phonon is not None


@pytest.mark.integration
def test_apply_transformation_populates_supercells(bcc_fe) -> None:
    """apply_transformation produces non-empty supercell lists."""
    t = Phono3pyDisplacementTransformation()
    t.apply_transformation(bcc_fe)
    assert t.supercells_with_displacements is not None
    assert len(t.supercells_with_displacements) > 0
    for s in t.supercells_with_displacements:
        assert isinstance(s, Structure)


@pytest.mark.integration
def test_apply_transformation_populates_phonon_supercells(bcc_fe) -> None:
    """apply_transformation populates phonon supercells for 2nd-order force constants."""
    t = Phono3pyDisplacementTransformation()
    t.apply_transformation(bcc_fe)
    assert t.phonon_supercells_with_displacements is not None
    assert len(t.phonon_supercells_with_displacements) > 0
