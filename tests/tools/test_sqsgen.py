"""Tests for SqsGenerator."""

from __future__ import annotations

import pytest
from pymatgen.core import Structure

pytest.importorskip("sqsgenerator")

from materialsframework.tools.sqsgen import SqsGenerator


def test_sqs_property_raises_before_generate() -> None:
    """Accessing .sqs before generate() raises ValueError."""
    t = SqsGenerator()
    with pytest.raises(ValueError, match="SQS has not been generated yet"):
        _ = t.sqs


def test_objective_property_raises_before_generate() -> None:
    """Accessing .objective before generate() raises ValueError."""
    t = SqsGenerator()
    with pytest.raises(ValueError, match="SQS has not been generated yet"):
        _ = t.objective


def test_default_params() -> None:
    """Default iteration count and mode are stored correctly."""
    t = SqsGenerator()
    assert t._iterations == 1000
    assert t._mode == "random"
    assert t._structure_format == "pymatgen"


@pytest.mark.integration
def test_generate_returns_structure_and_objective() -> None:
    """generate() returns a dict with 'structure' and 'objective' keys."""
    t = SqsGenerator(iterations=50)
    result = t.generate("Fe0.5Co0.5", crystal_structure="bcc", supercell_size=(2, 2, 2))
    assert "structure" in result
    assert "objective" in result
    assert isinstance(result["structure"], Structure)
    assert isinstance(result["objective"], float)


@pytest.mark.integration
def test_generate_populates_properties() -> None:
    """After generate(), .sqs and .objective properties are accessible."""
    t = SqsGenerator(iterations=50)
    t.generate("Fe0.5Co0.5", crystal_structure="bcc", supercell_size=(2, 2, 2))
    assert isinstance(t.sqs, Structure)
    assert isinstance(t.objective, float)


@pytest.mark.integration
def test_generate_composition_matches(bcc_fe) -> None:
    """Generated structure contains the expected elements."""
    t = SqsGenerator(iterations=50)
    result = t.generate("Fe0.5Co0.5", crystal_structure="bcc", supercell_size=(2, 2, 2))
    elements = {str(el) for el in result["structure"].elements}
    assert "Fe" in elements
    assert "Co" in elements
