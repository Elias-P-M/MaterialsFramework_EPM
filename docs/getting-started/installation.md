# Installation

## Prerequisites

MaterialsFramework uses [uv](https://docs.astral.sh/uv/) for dependency and environment management. Install it once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone the repository:

```bash
git clone https://github.com/dogusariturk/MaterialsFramework.git
cd MaterialsFramework
```

## Supported Platforms

Managed `uv` environments are currently configured for:

- Linux `x86_64`

This is controlled by `[tool.uv].environments` in `pyproject.toml`:

```toml
"sys_platform == 'linux' and platform_machine == 'x86_64' and implementation_name == 'cpython'"
```

If you are on a different platform, installation may still be possible with manual dependency management, but it is not currently a supported `uv` target in this repository.

## Core Installation (no ML extras)

```bash
uv sync
```

This installs the framework core (pymatgen, ASE, numpy, scipy) plus `RandomCalculator` and `VASPCalculator`, which have no heavy ML dependencies.

## Install MLIP Extras

Each MLIP has its own extra (for example `mace`, `chgnet`, `eqv2`, `petmad`).
Install either one MLIP extra or a compatibility-safe group.

```bash
# Core + one MLIP extra
uv sync --extra chgnet

# Core + compatible multi-MLIP stack (example)
uv sync --extra chgnet --extra matgl --extra sevennet
```

If `uv` reports an extra conflict, choose a different combination using the dedicated compatibility page:

- [MLIP conflicts and compatibility matrix](mlip-conflicts.md)

## MLIP Extras Overview (from `pyproject.toml`)

| Calculator | Extra | Dependency spec |
|---|---|---|
| ALIGNN | `alignn` | `alignn>=2025.4.1` |
| AlphaNet | `alphanet` | `alphanet @ git+https://github.com/zmyybc/AlphaNet.git@v0.1.2` |
| CHGNet | `chgnet` | `chgnet>=0.4.2` |
| DeePMD | `deepmd` | `deepmd-kit[torch]>=3.1.2` |
| EqNorm | `eqnorm` | `eqnorm @ git+https://github.com/yzchen08/eqnorm.git@57527db...` |
| EquiformerV2 | `eqv2` | `fairchem-core>=1.10.0,<2.0` |
| eSEN | `esen` | `fairchem-core>=1.10.0,<2.0` |
| GPTFF | `gptff` | `gptff @ git+https://github.com/atomly-materials-research-lab/GPTFF.git@4cef93a...` |
| GRACE | `grace` | `tensorpotential>=0.5.7` |
| HIENet | `hienet` | `hienet @ git+https://github.com/divelab/AIRS.git@579c133...#subdirectory=OpenMat/HIENet` |
| M3GNet / MEGNet | `matgl` | `matgl>=2.0.0` |
| MACE | `mace` | `mace-torch>=0.3.15` |
| MatterSim | `mattersim` | `mattersim>=1.2.1` |
| NequIP | `nequip` | `nequip>=0.17.0` |
| Nequix | `nequix` | `nequix>=0.4.3` |
| NewtonNet | `newtonnet` | `newtonnet>=2.0.0` |
| ORB | `orb` | `orb-models>=0.5.5` |
| PET-MAD | `petmad` | `upet>=0.2.1` |
| PosEGNN | `posegnn` | No resolvable package on public indexes |
| SevenNet | `sevennet` | `sevenn>=0.12.0` |
| UMA | `uma` | `fairchem-core>=2.0.0` |

`PosEGNNCalculator` is intentionally not installable through managed extras. `VASPCalculator` is external and requires a separately installed licensed VASP binary.

## Development Setup

```bash
# Core + dev tools (ruff, ty, nox, pytest, pre-commit)
uv sync --group dev

# Core + dev tools + selected MLIP extras
uv sync --group dev --extra chgnet --extra matgl --extra sevennet

```

## Running Tests

```bash
# Unit tests
uv run pytest -m "not integration" -v

# Integration tests
uv run pytest -m integration -v
```

Integration tests (`@pytest.mark.integration`) instantiate real ML calculators, download model weights on first run, and execute short relaxations on small structures. They are slow on first run but fast on subsequent runs once weights are cached.

## Documentation

```bash
uv sync --extra docs
uv run mkdocs serve
```

Then open <http://localhost:8000>.
