# Quick Start

This guide shows the minimal code needed to relax a crystal structure using MaterialsFramework.

## Geometry Optimization

```python
from ase.build import bulk
from materialsframework.calculators import MACECalculator

# Build an FCC copper structure
struct = bulk(name="Cu", crystalstructure="fcc", a=3.6, cubic=True)

# Instantiate the calculator (model weights are downloaded on first use)
calc = MACECalculator()

# Run geometry optimization (cell shape + atomic positions)
res = calc.relax(struct)

# Inspect results
print(res["final_structure"])   # pymatgen Structure
print(res["forces"])            # numpy array, eV/Å
print(res["stress"])            # numpy array, GPa
```

The `relax()` method returns a dict with at minimum:

| Key | Description |
|-----|-------------|
| `final_structure` | Relaxed structure as a pymatgen `Structure` |
| `trajectory` | List of intermediate structures |
| `energy` | Final total energy (eV) |
| `forces` | Forces on each atom (eV/Å) |
| `stress` | Stress tensor (GPa) |

## Swapping Calculators

Any supported calculator implementation can be used as a drop-in replacement:

```python
from materialsframework.calculators import MACECalculator

calc = MACECalculator()
res = calc.relax(struct)
```

See the [Calculators API reference](../api/calculators/index.md) for all available calculators and their parameters.

## Next Steps

- [Geometry Optimization](../user-guide/relaxation.md) — optimizer choices, convergence criteria
- [Molecular Dynamics](../user-guide/md.md) — NVT and NPT ensembles
- [Property Analysis](../user-guide/analysis.md) — EOS, elastic constants, phonons
