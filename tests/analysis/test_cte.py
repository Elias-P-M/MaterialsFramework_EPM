"""Tests for CTEAnalyzer."""

from __future__ import annotations

import pytest

from materialsframework.analysis.cte import CTEAnalyzer
from materialsframework.transformations.cte import CTETransformation

_TWO_POINTS = 2


def test_cte_transformation_lazy_property() -> None:
    """Accessing cte_transformation creates a CTETransformation instance."""
    analyzer = CTEAnalyzer()
    assert isinstance(analyzer.cte_transformation, CTETransformation)


@pytest.mark.integration
def test_calculate_returns_structured_cte_output(calc, bcc_fe) -> None:
    """calculate() returns per-temperature metrics and CTE summary fields."""
    analyzer = CTEAnalyzer(calculator=calc)
    result = analyzer.calculate(bcc_fe, temperatures=[300.0, 350.0], steps=2)

    assert {"temperatures", "volumes", "per_temperature", "cte"} <= result.keys()
    assert len(result["per_temperature"]) == _TWO_POINTS
    assert "volumetric_per_k" in result["cte"]
    assert "volumetric_ppm_per_k" in result["cte"]


@pytest.mark.integration
def test_calculate_accepts_ase_atoms(calc, ase_bcc_fe) -> None:
    """calculate() accepts ase.Atoms input."""
    analyzer = CTEAnalyzer(calculator=calc)
    result = analyzer.calculate(ase_bcc_fe, temperatures=[300.0, 350.0], steps=2)
    assert len(result["volumes"]) == _TWO_POINTS


def test_calculate_rejects_non_distinct_temperatures(bcc_fe) -> None:
    """calculate() requires at least two distinct temperatures."""
    analyzer = CTEAnalyzer()
    with pytest.raises(ValueError, match="two distinct temperatures"):
        analyzer.calculate(bcc_fe, temperatures=[300.0, 300.0], steps=2)


def test_calculate_rejects_invalid_temperatures(bcc_fe) -> None:
    """calculate() propagates explicit invalid-temperature validation."""
    analyzer = CTEAnalyzer()
    with pytest.raises(ValueError, match="greater than 0 K"):
        analyzer.calculate(bcc_fe, temperatures=[300.0, -5.0], steps=2)
