# Property Analysis

MaterialsFramework provides analyzer classes for common materials property calculations. Each analyzer follows the same pattern: instantiate with a calculator, then call `calculate(structure)`.

## Equation of State

Fits energy-volume data to extract bulk modulus and equilibrium volume.

```python
from ase.build import bulk
from materialsframework.calculators import GraceCalculator
from materialsframework.analysis import EOSAnalyzer

struct = bulk("Cu", "fcc", a=3.6, cubic=True)
calc = GraceCalculator()

eos = EOSAnalyzer(calculator=calc)
res = eos.calculate(struct)

print(res["e0"])       # equilibrium energy (eV)
print(res["v0"])       # equilibrium volume (Å³)
print(res["b0"])       # bulk modulus (eV/Å³)
print(res["b0_GPa"])   # bulk modulus (GPa)
print(res["b1"])       # pressure derivative of bulk modulus
```

## Elastic Constants (Cubic)

Calculates C11, C12, C44 and derived moduli for cubic/orthogonal cells.

!!! caution
    `CubicElasticConstantsAnalyzer` only works with cubic or orthogonal cells.

```python
from materialsframework.analysis import CubicElasticConstantsAnalyzer

cubic_ec = CubicElasticConstantsAnalyzer(calculator=calc)
res = cubic_ec.calculate(struct)

print(res["c11"])
print(res["c12"])
print(res["c44"])
print(res["youngs_modulus"])
print(res["voigt_reuss_hill_bulk_modulus"])
print(res["voigt_reuss_hill_shear_modulus"])
print(res["poisson_ratio"])
print(res["pugh_ratio"])
```

## Elastic Constants (General)

For non-cubic cells — computes the full elastic tensor and Voigt-Reuss-Hill averages.

```python
from materialsframework.analysis import ElasticConstantsAnalyzer

ec = ElasticConstantsAnalyzer(calculator=calc)
res = ec.calculate(struct)
```

## Phonons (Phonopy)

Computes phonon density of states, band structure, and thermal properties.

```python
from materialsframework.analysis import PhonopyAnalyzer

phonopy = PhonopyAnalyzer(calculator=calc)
res = phonopy.calculate(struct)

print(res["total_dos"])
print(res["projected_dos"])
print(res["thermal_properties"])
```

## Lattice Thermal Conductivity (Phono3py)

Requires `uv sync --extra phono3py`.

```python
from materialsframework.analysis import Phono3pyAnalyzer

phono3py = Phono3pyAnalyzer(calculator=calc)
res = phono3py.calculate(struct)
```

## Bain Path

Analyzes the energy landscape along the Bain transformation path (FCC ↔ BCC).

```python
from materialsframework.analysis import BainPathAnalyzer

bain = BainPathAnalyzer(calculator=calc)
res = bain.calculate(struct)
```

## ANNNI Stacking Fault Energy

Computes stacking fault energies using the ANNNI model.

```python
from materialsframework.analysis import ANNNIStackingFaultAnalyzer

annni = ANNNIStackingFaultAnalyzer(calculator=calc)
res = annni.calculate(struct)
```

## Formation Energy

```python
from materialsframework.analysis import FormationEnergyAnalyzer

fe = FormationEnergyAnalyzer(calculator=calc)
res = fe.calculate(struct)
print(res["formation_energy"])  # eV/atom
```

## USFE (Generalized Stacking Faults)

Computes GSFE curves and unstable stacking fault energy (USFE) for supported BCC
slip systems.

```python
from materialsframework.analysis import USFEAnalyzer

usfe = USFEAnalyzer(calculator=calc, slip_plane="110", num_steps=11)
res = usfe.calculate(struct)
print(res["usfe_mJ_m2"])
```

## Thermal Expansion (CTE)

Runs NPT MD at multiple temperatures and estimates volumetric CTE from the
volume-temperature trend.

```python
from materialsframework.analysis import CTEAnalyzer

cte = CTEAnalyzer(calculator=calc)
res = cte.calculate(struct, temperatures=[300, 600, 900], steps=10000)
print(res["cte"]["volumetric_ppm_per_k"])
```

## H Solubility

Evaluates H interstitial insertion energies for octahedral and tetrahedral sites
and reports the minimum solution energy.

```python
from materialsframework.analysis import HSolubilityAnalyzer

h_sol = HSolubilityAnalyzer(calculator=calc)
res = h_sol.calculate(struct, site_types=("octahedral", "tetrahedral"))
print(res["solution_energy"])
```
