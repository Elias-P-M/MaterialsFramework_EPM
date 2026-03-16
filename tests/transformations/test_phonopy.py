"""Tests for PhonopyDisplacementTransformation."""

from __future__ import annotations

import pytest
from pymatgen.core import Structure

pytest.importorskip("phonopy")

from materialsframework.transformations.phonopy import PhonopyDisplacementTransformation


def test_init() -> None:
    """Transformation initialises with all state attributes as None."""
    t = PhonopyDisplacementTransformation()
    assert t.phonon is None
    assert t.displaced_structures is None
    assert t.displacements is None


@pytest.mark.integration
def test_apply_transformation_populates_phonon(bcc_fe) -> None:
    """apply_transformation creates a Phonopy object."""
    t = PhonopyDisplacementTransformation()
    t.apply_transformation(bcc_fe)
    assert t.phonon is not None


@pytest.mark.integration
def test_apply_transformation_populates_displaced_structures(bcc_fe) -> None:
    """apply_transformation produces a non-empty list of displaced structures."""
    t = PhonopyDisplacementTransformation()
    t.apply_transformation(bcc_fe)
    assert t.displaced_structures is not None
    assert len(t.displaced_structures) > 0
    for s in t.displaced_structures:
        assert isinstance(s, Structure)


@pytest.mark.integration
def test_apply_transformation_stores_displacements(bcc_fe) -> None:
    """apply_transformation stores the displacement vectors."""
    t = PhonopyDisplacementTransformation()
    t.apply_transformation(bcc_fe)
    assert t.displacements is not None
