# MaterialsFramework

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15731044.svg)](https://doi.org/10.5281/zenodo.15731044)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://opensource.org/license/gpl-3-0)
[![Python](https://img.shields.io/badge/python-3.12-brightgreen.svg)](https://www.python.org/)

A modular and extensible framework for deploying, benchmarking, and experimenting with state-of-the-art machine learning interatomic potentials (MLIPs) in materials science.

---

## Key Features

- **24 calculator implementations** behind a single interface — 22 ML-backed calculators plus `RandomCalculator` (testing) and `VASPCalculator` (external backend)
- **ASE + pymatgen** interoperability — accepts both `Atoms` and `Structure` objects throughout
- **Geometry optimization** via `relax()` using BFGS, FIRE, LBFGS, and cell-shape relaxation with `FrechetCellFilter`
- **Molecular dynamics** (NVE, NVT, NPT, inhomogeneous NPT) via `run()`
- **Property analysis** workflows — EOS fitting, elastic constants (cubic and general), phonons (Phonopy/Phono3py), formation energies, Bain path, ANNNI stacking faults, USFE, CTE, and H-solubility
- **Phase stability tools** — cluster expansion, Cahn–Hilliard phase field model, stability maps, PhaseForge

## Quick Example

```python
from ase.build import bulk
from materialsframework.calculators import MACECalculator

struct = bulk(name="Cu", crystalstructure="fcc", a=3.6, cubic=True)

calc = MACECalculator()
res = calc.relax(struct)

print(res["final_structure"])
print(res["forces"])
print(res["stress"])
```

## Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/dogusariturk/MaterialsFramework.git
cd MaterialsFramework
uv sync
```

Install one MLIP extra (or a compatible group), for example:

```bash
uv sync --extra chgnet
uv sync --extra chgnet --extra matgl --extra sevennet
```

See:

- [Installation guide](getting-started/installation.md) for setup and command patterns
- [MLIP conflicts and compatibility matrix](getting-started/mlip-conflicts.md) for conflict logic and pairwise compatibility

## Citing

If you use MaterialsFramework in your research, please cite:

> Sarıtürk, D., & Arroyave, R. (2025). *MaterialsFramework*. Zenodo. <https://doi.org/10.5281/zenodo.15731044>

```bibtex
@software{sariturk_2025_15731044,
  author    = {Sarıtürk, Doğuhan and Arroyave, Raymundo},
  title     = {MaterialsFramework},
  month     = jun,
  year      = 2025,
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.15731044},
  url       = {https://doi.org/10.5281/zenodo.15731044},
}
```

## License

Distributed under the [GPLv3 License](https://opensource.org/license/gpl-3-0).
