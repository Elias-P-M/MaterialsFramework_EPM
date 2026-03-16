"""Tests for StabilityMap static helpers (no pycalphad Database required)."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("pycalphad")
pytest.importorskip("pandarallel")

pytestmark = pytest.mark.integration

from materialsframework.tools.stability_map import StabilityMap


def test_generate_compositions_sums_to_one() -> None:
    """All generated compositions sum to 1.0."""
    comps = StabilityMap._generate_compositions(["A", "B", "C"], step=0.5)
    assert not comps.empty
    for _, row in comps.iterrows():
        assert abs(row.sum() - 1.0) < 1e-6


def test_generate_compositions_excludes_pure_components() -> None:
    """No row has a component value of exactly 1.0."""
    comps = StabilityMap._generate_compositions(["A", "B"], step=0.1)
    for _, row in comps.iterrows():
        assert not any(val == 1.0 for val in row)


def test_generate_compositions_columns() -> None:
    """DataFrame columns match the provided element list."""
    elements = ["Co", "Cr", "Fe", "Ni"]
    comps = StabilityMap._generate_compositions(elements, step=0.5)
    assert list(comps.columns) == elements


def test_determine_section_no_negatives() -> None:
    """_determine_section returns 0 when all eigenvalues are positive."""
    row = pd.Series({"eigenvalue_1": 1.0, "eigenvalue_2": 2.0, "other": 0.5})
    assert StabilityMap._determine_section(row) == 0


def test_determine_section_one_negative() -> None:
    """_determine_section returns 1 when exactly one eigenvalue is negative."""
    row = pd.Series({"eigenvalue_1": -1.0, "eigenvalue_2": 2.0})
    assert StabilityMap._determine_section(row) == 1


def test_determine_section_all_negative() -> None:
    """_determine_section counts all negative eigenvalues."""
    row = pd.Series({"eigenvalue_1": -1.0, "eigenvalue_2": -0.5, "eigenvalue_3": -0.1})
    assert StabilityMap._determine_section(row) == 3


def test_orthogonalization_matrix_shape() -> None:
    """ORTHOGONALIZATION class attribute has the expected (9, 9) shape."""
    mat = StabilityMap.ORTHOGONALIZATION
    assert mat.shape == (9, 9)
