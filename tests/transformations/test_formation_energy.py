"""Tests for FormationEnergyTransformation."""

from __future__ import annotations

from pymatgen.core import Structure

from materialsframework.transformations.formation_energy import (
    FormationEnergyTransformation,
)


def test_init() -> None:
    """FormationEnergyTransformation initialises with an empty pure_structures list."""
    t = FormationEnergyTransformation()
    assert t.pure_structures == []


def test_apply_transformation_populates_pure_structures(l10_feni) -> None:
    """apply_transformation generates one pure-element entry per composition element."""
    t = FormationEnergyTransformation()
    t.apply_transformation(l10_feni)
    assert len(t.pure_structures) == 2  # Fe and Ni


def test_pure_structures_are_candidate_lists(l10_feni) -> None:
    """Each entry in pure_structures is a (list[Structure], int) tuple with FCC/BCC/HCP candidates."""
    t = FormationEnergyTransformation()
    t.apply_transformation(l10_feni)
    for candidates, num in t.pure_structures:
        assert isinstance(candidates, list)
        assert len(candidates) == 3  # FCC, BCC, HCP
        for struct in candidates:
            assert isinstance(struct, Structure)
        assert num > 0


def test_apply_transformation_resets_on_repeated_calls(l10_feni) -> None:
    """Calling apply_transformation twice resets pure_structures, preventing accumulation."""
    t = FormationEnergyTransformation()
    t.apply_transformation(l10_feni)
    count_first = len(t.pure_structures)
    t.apply_transformation(l10_feni)
    assert len(t.pure_structures) == count_first  # reset, not doubled
