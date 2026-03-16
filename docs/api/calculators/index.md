# Calculators

MaterialsFramework exposes 24 calculator implementations behind a unified interface (22 ML-backed calculators plus `RandomCalculator` and `VASPCalculator`). Every calculator inherits from both `BaseCalculator` and `BaseMDCalculator`, providing `relax()` for geometry optimization and `run()` for molecular dynamics.

## Available Calculators

| Calculator   | Class                 | Backend                   |
|--------------|-----------------------|---------------------------|
| ALIGNN       | `AlignnCalculator`    | alignn                    |
| AlphaNet     | `AlphaNetCalculator`  | alphanet                  |
| CHGNet       | `CHGNetCalculator`    | chgnet                    |
| DeePMD       | `DeePMDCalculator`    | deepmd-kit                |
| EqNorm       | `EqnormCalculator`    | eqnorm                    |
| EquiformerV2 | `EqV2Calculator`      | fairchem-core             |
| eSEN         | `eSENCalculator`      | fairchem-core             |
| GPTFF        | `GPTFFCalculator`     | gptff                     |
| GRACE        | `GraceCalculator`     | tensorpotential           |
| HIENet       | `HIENetCalculator`    | hienet                    |
| M3GNet       | `M3GNetCalculator`    | matgl                     |
| MACE         | `MACECalculator`      | mace-torch                |
| MatterSim    | `MatterSimCalculator` | mattersim                 |
| MEGNet       | `MEGNetCalculator`    | matgl                     |
| NequIP       | `NequIPCalculator`    | nequip                    |
| Nequix       | `NequixCalculator`    | nequix                    |
| NewtonNet    | `NewtonNetCalculator` | newtonnet                 |
| ORB          | `ORBCalculator`       | orb-models                |
| PET-MAD      | `PetMadCalculator`    | upet (`pet-mad-s`)        |
| PosEGNN      | `PosEGNNCalculator`   | N/A                       |
| Random       | `RandomCalculator`    | (built-in, no ML backend) |
| SevenNet     | `SevenNetCalculator`  | sevenn                    |
| UMA          | `UMACalculator`       | fairchem-core             |
| VASP         | `VASPCalculator`      | VASP (external)           |

## Common Interface

All calculators expose the same two methods:

```python
# Geometry optimization
res = calc.relax(structure, fmax=0.05, steps=500, optimizer="BFGS", relax_cell=True)

# Molecular dynamics
res = calc.run(structure=structure, steps=1000)
```

See [Base Classes](base.md) for the full API of `BaseCalculator` and `BaseMDCalculator`.
