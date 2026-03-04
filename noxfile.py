"""Nox sessions for MaterialsFramework integration tests.

Each session installs a specific bundle extra and runs the corresponding
calculator tests against real MLIP implementations.

Usage:
    uv run nox -s test-base           # no ML extras required
    uv run nox -s test-bundle-main    # bundle-main conflict group
    uv run nox -s test-bundle-mace    # bundle-mace conflict group
    uv run nox -s test-bundle-grace   # bundle-grace conflict group
    uv run nox                        # all sessions
"""

from __future__ import annotations

import nox

nox.options.default_venv_backend = "uv"
PYTHON = "3.11"

BASE_TESTS = [
    "tests/test_registry.py",
    "tests/test_base_calculator.py",
    "tests/test_trajectory.py",
]

BUNDLE_MAIN_TESTS = [
    "tests/calculators/test_alignn.py",
    "tests/calculators/test_alphanet.py",
    "tests/calculators/test_chgnet.py",
    "tests/calculators/test_deepmd.py",
    "tests/calculators/test_eqnorm.py",
    "tests/calculators/test_eqv2.py",
    "tests/calculators/test_esen.py",
    "tests/calculators/test_gptff.py",
    "tests/calculators/test_hienet.py",
    "tests/calculators/test_m3gnet.py",
    "tests/calculators/test_mattersim.py",
    "tests/calculators/test_megnet.py",
    "tests/calculators/test_nequip.py",
    "tests/calculators/test_nequix.py",
    "tests/calculators/test_newtonnet.py",
    "tests/calculators/test_orb.py",
    "tests/calculators/test_petmad.py",
    "tests/calculators/test_sevennet.py",
    "tests/calculators/test_uma.py",
]

BUNDLE_MACE_TESTS = [
    "tests/calculators/test_alignn.py",
    "tests/calculators/test_alphanet.py",
    "tests/calculators/test_chgnet.py",
    "tests/calculators/test_deepmd.py",
    "tests/calculators/test_gptff.py",
    "tests/calculators/test_hienet.py",
    "tests/calculators/test_mace.py",
    "tests/calculators/test_megnet.py",
    "tests/calculators/test_m3gnet.py",
    "tests/calculators/test_nequip.py",
    "tests/calculators/test_nequix.py",
    "tests/calculators/test_newtonnet.py",
    "tests/calculators/test_orb.py",
    "tests/calculators/test_petmad.py",
    "tests/calculators/test_sevennet.py",
]

BUNDLE_GRACE_TESTS = [
    "tests/calculators/test_alignn.py",
    "tests/calculators/test_alphanet.py",
    "tests/calculators/test_chgnet.py",
    "tests/calculators/test_deepmd.py",
    "tests/calculators/test_eqnorm.py",
    "tests/calculators/test_eqv2.py",
    "tests/calculators/test_esen.py",
    "tests/calculators/test_gptff.py",
    "tests/calculators/test_grace.py",
    "tests/calculators/test_hienet.py",
    "tests/calculators/test_m3gnet.py",
    "tests/calculators/test_mattersim.py",
    "tests/calculators/test_megnet.py",
    "tests/calculators/test_nequip.py",
    "tests/calculators/test_newtonnet.py",
    "tests/calculators/test_orb.py",
    "tests/calculators/test_petmad.py",
    "tests/calculators/test_sevennet.py",
    "tests/calculators/test_uma.py",
]


@nox.session(python=PYTHON, name="test-base")
def test_base(session: nox.Session) -> None:
    """Run tests that require no ML extras."""
    session.install("-e", ".")
    session.install("pytest")
    session.run("pytest", *BASE_TESTS, "-v")


@nox.session(python=PYTHON, name="test-bundle-main")
def test_bundle_main(session: nox.Session) -> None:
    """Run tests for bundle-main calculators (fairchem, mattersim, nequix, eqnorm)."""
    session.install("-e", ".[bundle-main]")
    session.install("pytest")
    session.run("pytest", *BUNDLE_MAIN_TESTS, "-v")


@nox.session(python=PYTHON, name="test-bundle-mace")
def test_bundle_mace(session: nox.Session) -> None:
    """Run tests for bundle-mace calculators (MACE)."""
    session.install("-e", ".[bundle-mace]")
    session.install("pytest")
    session.run("pytest", *BUNDLE_MACE_TESTS, "-v")


@nox.session(python=PYTHON, name="test-bundle-grace")
def test_bundle_grace(session: nox.Session) -> None:
    """Run tests for bundle-grace calculators (GRACE)."""
    session.install("-e", ".[bundle-grace]")
    session.install("pytest")
    session.run("pytest", *BUNDLE_GRACE_TESTS, "-v")
