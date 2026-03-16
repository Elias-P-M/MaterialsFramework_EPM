"""Tests for CubicElasticConstantsDeformationTransformation."""

from __future__ import annotations

import pytest
from pymatgen.core import Structure

from materialsframework.transformations.cubic_elastic_constants import (
    CubicElasticConstantsDeformationTransformation,
)


def test_default_params() -> None:
    """Default delta_max and step_size are stored and deltas array is correct."""
    t = CubicElasticConstantsDeformationTransformation()
    assert t.delta_max == pytest.approx(0.05)
    assert t.step_size == pytest.approx(0.01)
    assert len(t.deltas) == 11


def test_custom_params() -> None:
    """Custom delta_max and step_size produce the expected delta count."""
    t = CubicElasticConstantsDeformationTransformation(delta_max=0.02, step_size=0.01)
    assert len(t.deltas) == 5


def test_apply_transformation_uniform(bcc_fe) -> None:
    """Uniform distortions are generated for every delta value."""
    t = CubicElasticConstantsDeformationTransformation(delta_max=0.02, step_size=0.01)
    t.apply_transformation(bcc_fe)
    assert len(t.uniform_distorted_structures) == 5


def test_apply_transformation_orthorhombic(bcc_fe) -> None:
    """Orthorhombic distortions are generated only for non-negative deltas."""
    t = CubicElasticConstantsDeformationTransformation(delta_max=0.02, step_size=0.01)
    t.apply_transformation(bcc_fe)
    assert len(t.orthorhombic_distorted_structures) == 3


def test_apply_transformation_monoclinic(bcc_fe) -> None:
    """Monoclinic distortions are generated only for non-negative deltas."""
    t = CubicElasticConstantsDeformationTransformation(delta_max=0.02, step_size=0.01)
    t.apply_transformation(bcc_fe)
    assert len(t.monoclinic_distorted_structures) == 3


def test_distorted_structures_are_structures(bcc_fe) -> None:
    """All generated structures are valid pymatgen Structure instances."""
    t = CubicElasticConstantsDeformationTransformation(delta_max=0.02, step_size=0.01)
    t.apply_transformation(bcc_fe)
    for struct in t.uniform_distorted_structures.values():
        assert isinstance(struct, Structure)
