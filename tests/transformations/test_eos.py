"""Tests for EOSTransformation."""

from __future__ import annotations

import pytest
from pymatgen.core import Structure

from materialsframework.transformations.eos import EOSTransformation


def test_default_params() -> None:
    """EOSTransformation stores the correct default start, stop, num, and structures."""
    t = EOSTransformation()
    assert t._strains[0] == pytest.approx(-0.1)
    assert t._strains[-1] == pytest.approx(0.1)
    assert len(t._strains) == 11
    assert t.structures == []


def test_custom_params() -> None:
    """Custom start/stop/num produces the correct number of strains."""
    t = EOSTransformation(start=-0.05, stop=0.05, num=5)
    assert len(t._strains) == 5


def test_apply_transformation_populates_structures(bcc_fe) -> None:
    """apply_transformation() populates structures with num entries."""
    t = EOSTransformation(start=-0.02, stop=0.02, num=4)
    t.apply_transformation(bcc_fe)
    assert len(t.structures) == 4


def test_apply_transformation_structures_are_pymatgen(bcc_fe) -> None:
    """Each entry in structures is a pymatgen Structure."""
    t = EOSTransformation(start=-0.02, stop=0.02, num=3)
    t.apply_transformation(bcc_fe)
    for s in t.structures:
        assert isinstance(s, Structure)


def test_apply_transformation_preserves_site_count(bcc_fe) -> None:
    """Deformed structures keep the same number of sites as the original."""
    t = EOSTransformation(start=-0.02, stop=0.02, num=3)
    t.apply_transformation(bcc_fe)
    for s in t.structures:
        assert len(s) == len(bcc_fe)


def test_apply_transformation_accumulates_on_repeat_call(bcc_fe) -> None:
    """Calling apply_transformation() twice appends structures (documented behavior)."""
    t = EOSTransformation(start=-0.02, stop=0.02, num=3)
    t.apply_transformation(bcc_fe)
    t.apply_transformation(bcc_fe)
    assert len(t.structures) == 6
