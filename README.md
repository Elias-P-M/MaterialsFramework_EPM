<div align="center">

# MaterialsFramework

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15731044.svg)](https://doi.org/10.5281/zenodo.15731044)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://opensource.org/license/gpl-3-0)
[![Python](https://img.shields.io/badge/python-3.12-brightgreen.svg)](https://www.python.org/)

**A modular and extensible framework for deploying, benchmarking, and experimenting with state-of-the-art machine learning interatomic potentials in materials science.**

[Report a Bug](https://github.com/dogusariturk/MaterialsFramework/issues/new?labels=bug) · [Request a Feature](https://github.com/dogusariturk/MaterialsFramework/issues/new?labels=enhancement)

</div>

---

## Overview

MaterialsFramework provides a single, uniform API for 20+ ML interatomic potentials alongside a suite of materials-science analysis and tooling — all in one package.

- **20+ calculators** — 22 ML-backed + `RandomCalculator` + `VASPCalculator`
- **11 property analyzers** — EOS, phonons, elastic constants, stacking faults, and more
- **7 tools** — phase-field, cluster expansion, SQS generation, stability maps, and more
- **Dual input support** — accepts both `ase.Atoms` and `pymatgen.Structure` natively

---

## Supported MLIPs

| MLIP      | Package           | API                                      | Repository                                                         | Paper                                                       |
|-----------|-------------------|------------------------------------------|--------------------------------------------------------------------|-------------------------------------------------------------|
| ALIGNN    | `alignn`          | [API](docs/api/calculators/alignn.md)    | [Repo](https://github.com/usnistgov/alignn)                        | [Paper](https://arxiv.org/abs/2106.01829)                   |
| AlphaNet  | `alphanet`        | [API](docs/api/calculators/alphanet.md)  | [Repo](https://github.com/zmyybc/AlphaNet)                         | [Paper](https://arxiv.org/abs/2501.07155)                   |
| CHGNet    | `chgnet`          | [API](docs/api/calculators/chgnet.md)    | [Repo](https://github.com/CederGroupHub/chgnet)                    | [Paper](https://arxiv.org/abs/2302.14231)                   |
| DeePMD    | `deepmd-kit`      | [API](docs/api/calculators/deepmd.md)    | [Repo](https://github.com/deepmodeling/deepmd-kit)                 | [Paper](https://arxiv.org/abs/2506.01686)                   |
| EqNorm    | `eqnorm`          | [API](docs/api/calculators/eqnorm.md)    | [Repo](https://github.com/yzchen08/eqnorm)                         | N/A                                                         |
| EqV2      | `fairchem-core`   | [API](docs/api/calculators/eqv2.md)      | [Repo](https://github.com/FAIR-Chem/fairchem)                      | [Paper](https://arxiv.org/abs/2410.12771)                   |
| eSEN      | `fairchem-core`   | [API](docs/api/calculators/esen.md)      | [Repo](https://github.com/FAIR-Chem/fairchem)                      | [Paper](https://arxiv.org/abs/2502.12147)                   |
| GPTFF     | `gptff`           | [API](docs/api/calculators/gptff.md)     | [Repo](https://github.com/atomly-materials-research-lab/GPTFF)     | [Paper](https://doi.org/10.1016/j.scib.2024.08.039)         |
| GRACE     | `tensorpotential` | [API](docs/api/calculators/grace.md)     | [Repo](https://github.com/ICAMS/grace-tensorpotential)             | [Paper](https://arxiv.org/abs/2508.17936)                   |
| HIENet    | `hienet`          | [API](docs/api/calculators/hienet.md)    | [Repo](https://github.com/divelab/AIRS/tree/main/OpenMat/HIENet)   | [Paper](https://arxiv.org/abs/2503.05771)                   |
| M3GNet    | `matgl`           | [API](docs/api/calculators/m3gnet.md)    | [Repo](https://github.com/materialsvirtuallab/m3gnet)              | [Paper](https://arxiv.org/abs/2202.02450)                   |
| MACE      | `mace-torch`      | [API](docs/api/calculators/mace.md)      | [Repo](https://github.com/ACEsuit/mace)                            | [Paper](https://arxiv.org/abs/2401.00096)                   |
| MatterSim | `mattersim`       | [API](docs/api/calculators/mattersim.md) | [Repo](https://github.com/microsoft/mattersim)                     | [Paper](https://arxiv.org/abs/2405.04967)                   |
| MEGNet    | `matgl`           | [API](docs/api/calculators/megnet.md)    | [Repo](https://github.com/materialsvirtuallab/megnet)              | [Paper](https://arxiv.org/abs/1812.05055)                   |
| NequIP    | `nequip`          | [API](docs/api/calculators/nequip.md)    | [Repo](https://github.com/mir-group/nequip)                        | [Paper](https://arxiv.org/abs/2504.16068)                   |
| Nequix    | `nequix`          | [API](docs/api/calculators/nequix.md)    | [Repo](https://github.com/atomicarchitects/nequix)                 | [Paper](https://arxiv.org/abs/2508.16067)                   |
| NewtonNet | `newtonnet`       | [API](docs/api/calculators/newtonnet.md) | [Repo](https://github.com/THGLab/NewtonNet)                        | [Paper](https://doi.org/10.1039/D2DD00008C)                 |
| ORB       | `orb-models`      | [API](docs/api/calculators/orb.md)       | [Repo](https://github.com/orbital-materials/orb-models)            | [Paper](https://arxiv.org/abs/2504.06231)                   |
| PetMad    | `upet`            | [API](docs/api/calculators/petmad.md)    | [Repo](https://github.com/lab-cosmo/upet)                          | [Paper](https://www.nature.com/articles/s41467-025-65662-7) |
| PosEGNN   | N/A               | [API](docs/api/calculators/posegnn.md)   | [Repo](https://github.com/IBM/materials/tree/main/models/pos_egnn) | N/A                                                         |
| SevenNet  | `sevenn`          | [API](docs/api/calculators/sevennet.md)  | [Repo](https://github.com/MDIL-SNU/SevenNet)                       | [Paper](https://arxiv.org/abs/2510.11241)                   |
| UMA       | `fairchem-core`   | [API](docs/api/calculators/uma.md)       | [Repo](https://github.com/FAIR-Chem/fairchem)                      | N/A                                                         |

Non-MLIP calculators: `RandomCalculator` (dependency-free testing stub) and `VASPCalculator` (external licensed VASP backend).

> [!WARNING]
> **PosEGNN** has no installable package on any public index. To use it, clone the repository and add the module directory to your `PYTHONPATH` manually:
> ```bash
> git clone --depth 1 https://github.com/IBM/materials.git
> export PYTHONPATH="$PWD/materials/models/pos_egnn:$PYTHONPATH"
> ```

---

## Property Analyzers

| Analyzer                        | Description                                                 |
|---------------------------------|-------------------------------------------------------------|
| `ANNNIStackingFaultAnalyzer`    | ANNNI-based intrinsic and extrinsic stacking fault energies |
| `BainPathAnalyzer`              | Energy along the FCC<->BCC Bain transformation path         |
| `CTEAnalyzer`                   | Coefficient of thermal expansion from NPT-MD volume trends  |
| `CubicElasticConstantsAnalyzer` | Cubic elastic constants and derived moduli (B, G, E, ν)     |
| `ElasticConstantsAnalyzer`      | Full elastic tensor and Voigt–Reuss–Hill averages           |
| `EOSAnalyzer`                   | Equation-of-state curve fitting from E–V data               |
| `FormationEnergyAnalyzer`       | Formation energy per atom                                   |
| `HSolubilityAnalyzer`           | Hydrogen insertion and solution energies                    |
| `Phono3pyAnalyzer`              | Anharmonic force constants and lattice thermal conductivity |
| `PhonopyAnalyzer`               | Phonon DOS, band structure, and thermal properties          |
| `USFEAnalyzer`                  | Generalized stacking fault energy curves and unstable SFE   |

---

## Tools

| Tool                   | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `BondLatticeParameter` | Lattice parameter estimation from bond lengths for FCC/BCC/HCP alloys       |
| `ClusterExpansion`     | Cluster expansion model construction and fitting                            |
| `PhaseFieldModel`      | Cahn–Hilliard phase-field simulations (includes grid and parameter helpers) |
| `Sqs2tdb`              | Converts SQS output files to TDB format for CALPHAD workflows (PhaseForge)  |
| `SqsGenerator`         | Special quasirandom structure generation                                    |
| `StabilityMap`         | Composition–temperature stability map generation                            |
| `TrajectoryObserver`   | Records energies, forces, stresses, and trajectory frames during MD         |

---

## Getting Started

### Installation

MaterialsFramework uses [uv](https://docs.astral.sh/uv/) for dependency management. Install it if not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone the repository and install the core dependencies:

```bash
git clone https://github.com/dogusariturk/MaterialsFramework.git
cd MaterialsFramework
uv sync
```

Install one or more compatible MLIP extras:

```bash
# Single MLIP
uv sync --extra chgnet

# Compatible multi-MLIP stack
uv sync --extra chgnet --extra matgl --extra sevennet
```

See the [installation guide](docs/getting-started/installation.md) for full setup instructions and the [MLIP compatibility matrix](docs/getting-started/mlip-conflicts.md) for conflict details.

### Quick Example

```python
from ase.build import bulk
from materialsframework.calculators import MACECalculator

structure = bulk("Cu", crystalstructure="fcc", a=3.6, cubic=True)
calc = MACECalculator()

result = calc.relax(structure)
print(result["final_structure"])
print(result["energy"])
```

---

## Citing

If you use MaterialsFramework in your research, please cite:

> Sarıtürk, D., & Arroyave, R. (2025). MaterialsFramework. Zenodo. https://doi.org/10.5281/zenodo.15731044

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

---

## License

Distributed under the GPL-3.0-or-later License. See [GPL-3.0](https://opensource.org/license/gpl-3-0) for details.

## Contact

Doguhan Sariturk — [doguhan.sariturk@gmail.com](mailto:doguhan.sariturk@gmail.com)
