=========
Changelog
=========

Version 0.13.0
==============
- Added ``CTEAnalyzer`` and ``CTETransformation`` — coefficient of thermal expansion from NPT-MD volume trends
- Added ``HSolubilityAnalyzer`` and ``HSolubilityTransformation`` — hydrogen insertion and solution energies
- Added ``USFEAnalyzer`` and ``USFETransformation`` — generalized stacking fault energy curves and unstable SFE
- Added ``RandomCalculator`` — dependency-free testing stub returning random energies and forces
- Added ``BondLatticeParameter`` tool — bond-length-based lattice parameter prediction for FCC, BCC, and HCP alloys
- Added ``SqsGenerator`` tool (moved from transformations, renamed from ``SpecialQuasirandomStructures``)
- Added entry-point-backed registries for all four domains (``get_calculator``, ``get_analyzer``, ``get_transformation``, ``get_tool``)
- Added comprehensive test suite covering all calculators, analyzers, transformations, and tools
- Added MkDocs documentation site with API reference, user guide, and getting-started pages
- Added GitHub Actions CI/CD workflows for testing, linting, documentation, and PyPI release
- Migrated ``PetMadCalculator`` backend from deprecated ``pet-mad`` to ``upet`` while preserving the public API
- Migrated from ``setup.py`` / conda environments to ``pyproject.toml`` with ``uv`` and ``hatchling``
- Replaced eager imports with lazy ``__getattr__``-based loading across all subpackages
- ``calculator`` property on ``BaseCalculator`` and ``BaseMDCalculator`` is now formally abstract (``@abstractmethod``)

Version 0.12.0
==============
- Added NPT Berendsen ensemble to MD
- Added Inhomogeneous NPT Berendsen ensemble to MD
- Added ``ClusterExpansion`` and ``FormationEnergyAnalyzer`` implementations
- Added ``Eqnorm`` implementation
- Implemented ``elastic`` module for calculating elastic tensor components

Version 0.11.0
==============
- Added ``PET-MAD`` implementation

Version 0.10.0
==============
- Added ``PosEGNN``, ``GPTFF``, ``NewtonNet``, and ``eSEN`` implementations

Version 0.9.0
=============
- Added ``AlphaNet``, ``DeePMD``, ``DiveNet``, and ``eqV2`` implementations
- Added ``Cahn Hilliard``, ``Sqs2tdb``, and ``Stability Map`` implementations

Version 0.8.0
=============
- Added ``GRACE`` implementation

Version 0.7.0
=============
- Added ``MatterSim`` implementation

Version 0.6.0
=============
- Added ``ORB`` and ``SevenNet`` implementations
- Refactored relaxation and calculator classes
- Generalized the ``BaseMDCalculator`` class

Version 0.5.0
=============
- Added custom matgl relaxer implementation

Version 0.4.0
=============
- Added ``M3GNet-MD`` implementation

Version 0.3.0
=============
- Added ``ANNNI-SFE`` and ``EOS`` implementations

Version 0.2.0
=============
- Added ``Calculator`` and ``Relaxer`` abstract base classes
- Added ``MACE``, ``CHGNet`` and ``MEGNet`` implementations

Version 0.1.0
=============
- Initial release
