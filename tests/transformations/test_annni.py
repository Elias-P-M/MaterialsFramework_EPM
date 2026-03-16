"""Tests for ANNNIStackingFaultTransformation."""

from __future__ import annotations

import pytest
from pymatgen.core import Structure

pytest.importorskip("sqsgenerator")

from materialsframework.transformations.annni import ANNNIStackingFaultTransformation


def test_init() -> None:
    """Transformation initialises with empty structures dict and no SQS transformation."""
    t = ANNNIStackingFaultTransformation()
    assert t.structures == {}
    assert t._sqs_gen is None


def test_sqs_gen_lazy_property() -> None:
    """Accessing .sqs_gen before apply_transformation creates a default instance."""
    from materialsframework.tools.sqsgen import SqsGenerator

    t = ANNNIStackingFaultTransformation()
    sqs = t.sqs_gen
    assert isinstance(sqs, SqsGenerator)


@pytest.mark.integration
def test_apply_transformation_populates_all_phases() -> None:
    """apply_transformation fills structures dict with fcc, hcp, and dhcp entries."""
    t = ANNNIStackingFaultTransformation()
    t.apply_transformation(
        "Fe0.5Co0.5",
        fcc_supercell_size=(2, 1, 1),
        hcp_supercell_size=(1, 1, 1),
        dhcp_supercell_size=(1, 1, 1),
    )
    for key in ("fcc", "hcp", "dhcp"):
        assert key in t.structures
        assert isinstance(t.structures[key], Structure)


@pytest.mark.integration
def test_apply_transformation_accepts_string_composition() -> None:
    """apply_transformation accepts a plain string as composition."""
    t = ANNNIStackingFaultTransformation()
    t.apply_transformation(
        "Fe0.5Co0.5",
        fcc_supercell_size=(2, 1, 1),
        hcp_supercell_size=(1, 1, 1),
        dhcp_supercell_size=(1, 1, 1),
    )
    assert len(t.structures) == 3
