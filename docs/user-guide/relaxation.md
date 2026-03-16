# Geometry Optimization

The `relax()` method on any calculator performs structure optimization using ASE optimizers combined with `FrechetCellFilter` for simultaneous cell-shape and position relaxation.

## Basic Usage

```python
from ase.build import bulk
from materialsframework.calculators import M3GNetCalculator

struct = bulk("Fe", "bcc", a=2.87, cubic=True)
calc = M3GNetCalculator()
res = calc.relax(struct)

print(res["final_structure"])
print(res["energy"])
print(res["forces"])
print(res["stress"])
```

## Optimizer Options

The `relax()` method accepts an `optimizer` keyword argument. Supported ASE optimizers include:

| Optimizer | Description |
|-----------|-------------|
| `"BFGS"` | Broyden–Fletcher–Goldfarb–Shanno (default) |
| `"FIRE"` | Fast Inertial Relaxation Engine |
| `"LBFGS"` | Limited-memory BFGS |
| `"MDMin"` | Velocity-Verlet with damping |

```python
res = calc.relax(struct, optimizer="FIRE")
```

## Convergence Criteria

Control convergence via `fmax` (force convergence, eV/Å) and `steps` (maximum steps):

```python
res = calc.relax(struct, fmax=0.01, steps=500)
```

## Cell Relaxation

By default, `relax()` optimizes both atomic positions and cell shape. Pass `relax_cell=False` to fix the cell:

```python
res = calc.relax(struct, relax_cell=False)
```

## Trajectory

The returned dict includes a `trajectory` key with all intermediate structures:

```python
for frame in res["trajectory"]:
    print(frame)  # pymatgen Structure at each optimization step
```

## Input Formats

Both pymatgen `Structure` and ASE `Atoms` objects are accepted:

```python
from pymatgen.core import Structure

# From pymatgen Structure
pmg_struct = Structure(...)
res = calc.relax(pmg_struct)

# From ASE Atoms
from ase.build import bulk
ase_atoms = bulk("Cu", "fcc", a=3.6)
res = calc.relax(ase_atoms)
```
