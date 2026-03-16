# Transformations

The `transformations` package implements the **Transformation** half of MaterialsFramework's analyzer-transformation pattern. Each transformation takes a reference structure and generates a set of deformed or displaced structures for evaluation by a calculator.

Transformations are generally used indirectly through their paired Analyzer class, but can be used directly for custom workflows.

## Available Transformations

| Class                                            | Module                    | Description                                                     |
|--------------------------------------------------|---------------------------|-----------------------------------------------------------------|
| `ANNNIStackingFaultTransformation`               | `annni`                   | Generate ANNNI stacking fault structures                        |
| `BainDisplacementTransformation`                 | `bain`                    | Generate Bain path structures                                   |
| `CTETransformation`                              | `cte`                     | Prepare per-temperature MD tasks for CTE analysis               |
| `CubicElasticConstantsDeformationTransformation` | `cubic_elastic_constants` | Deformations for cubic elastic constants                        |
| `ElasticConstantsDeformationTransformation`      | `elastic_constants`       | Deformations for general elastic tensor                         |
| `EOSTransformation`                              | `eos`                     | Volume-scaled structures for EOS fitting                        |
| `FormationEnergyTransformation`                  | `formation_energy`        | Generate pure-element reference structures for formation energy |
| `HSolubilityTransformation`                      | `h_solubility`            | Generate BCC octahedral/tetrahedral H interstitial structures   |
| `Phono3pyDisplacementTransformation`             | `phono3py`                | Displaced supercells for Phono3py                               |
| `PhonopyDisplacementTransformation`              | `phonopy`                 | Displaced supercells for Phonopy                                |
| `USFETransformation`                             | `usfe`                    | Generate rigidly displaced structures along BCC slip systems    |
