# MLIP Conflicts and Compatibility Matrix

This page summarizes MLIP compatibility based on the current `pyproject.toml`:

- `[project.optional-dependencies]` for available extras
- `[tool.uv].conflicts` for incompatibility rules

If your `uv sync --extra ...` command fails due to conflicting extras, use the tables below to choose a compatible combination.

## Scope and Notes

- `posegnn` is listed as an extra but has no resolvable package on public indexes.
- `vasp` is not managed by extras and requires a separately installed licensed VASP binary.
- Compatibility here reflects declared `uv` conflicts only.

## Compact MLIP Compatibility Table

| Extra | Dependency spec | Conflicts with | Can co-install with |
|---|---|---|---|
| `alignn` | `alignn>=2025.4.1` | `chgnet`, `deepmd`, `eqv2`, `esen`, `mace`, `mattersim`, `nequip`, `nequix`, `orb`, `petmad`, `sevennet`, `uma` | `alphanet`, `eqnorm`, `gptff`, `grace`, `hienet`, `matgl`, `newtonnet` |
| `alphanet` | `alphanet @ git+https://github.com/zmyybc/AlphaNet.git@v0.1.2` | none | `alignn`, `chgnet`, `deepmd`, `eqnorm`, `eqv2`, `esen`, `gptff`, `grace`, `hienet`, `mace`, `matgl`, `mattersim`, `nequip`, `nequix`, `newtonnet`, `orb`, `petmad`, `sevennet`, `uma` |
| `chgnet` | `chgnet>=0.4.2` | `alignn`, `mattersim`, `newtonnet` | `alphanet`, `deepmd`, `eqnorm`, `eqv2`, `esen`, `gptff`, `grace`, `hienet`, `mace`, `matgl`, `nequip`, `nequix`, `orb`, `petmad`, `sevennet`, `uma` |
| `deepmd` | `deepmd-kit[torch]>=3.1.2` | `alignn`, `eqv2`, `esen`, `mace`, `mattersim`, `newtonnet`, `uma` | `alphanet`, `chgnet`, `eqnorm`, `gptff`, `grace`, `hienet`, `matgl`, `nequip`, `nequix`, `orb`, `petmad`, `sevennet` |
| `eqnorm` | `eqnorm @ git+https://github.com/yzchen08/eqnorm.git@57527db44fee9fab3fb5775dd860ca4bbe014f44` | `mace`, `newtonnet` | `alignn`, `alphanet`, `chgnet`, `deepmd`, `eqv2`, `esen`, `gptff`, `grace`, `hienet`, `matgl`, `mattersim`, `nequip`, `nequix`, `orb`, `petmad`, `sevennet`, `uma` |
| `eqv2` | `fairchem-core>=1.10.0,<2.0` | `alignn`, `deepmd`, `mace`, `mattersim`, `newtonnet`, `orb`, `petmad`, `uma` | `alphanet`, `chgnet`, `eqnorm`, `esen`, `gptff`, `grace`, `hienet`, `matgl`, `nequip`, `nequix`, `sevennet` |
| `esen` | `fairchem-core>=1.10.0,<2.0` | `alignn`, `deepmd`, `mace`, `mattersim`, `newtonnet`, `orb`, `petmad`, `uma` | `alphanet`, `chgnet`, `eqnorm`, `eqv2`, `gptff`, `grace`, `hienet`, `matgl`, `nequip`, `nequix`, `sevennet` |
| `gptff` | `gptff @ git+https://github.com/atomly-materials-research-lab/GPTFF.git@4cef93a9aa573c86c6bb9100d3d48978ad616429` | none | `alignn`, `alphanet`, `chgnet`, `deepmd`, `eqnorm`, `eqv2`, `esen`, `grace`, `hienet`, `mace`, `matgl`, `mattersim`, `nequip`, `nequix`, `newtonnet`, `orb`, `petmad`, `sevennet`, `uma` |
| `grace` | `tensorpotential>=0.5.7` | none | `alignn`, `alphanet`, `chgnet`, `deepmd`, `eqnorm`, `eqv2`, `esen`, `gptff`, `hienet`, `mace`, `matgl`, `mattersim`, `nequip`, `nequix`, `newtonnet`, `orb`, `petmad`, `sevennet`, `uma` |
| `hienet` | `hienet @ git+https://github.com/divelab/AIRS.git@579c13340e3f2699e73bc67444d563dabd015d6c#subdirectory=OpenMat/HIENet` | none | `alignn`, `alphanet`, `chgnet`, `deepmd`, `eqnorm`, `eqv2`, `esen`, `gptff`, `grace`, `mace`, `matgl`, `mattersim`, `nequip`, `nequix`, `newtonnet`, `orb`, `petmad`, `sevennet`, `uma` |
| `mace` | `mace-torch>=0.3.15` | `alignn`, `deepmd`, `eqnorm`, `eqv2`, `esen`, `mattersim`, `nequip`, `nequix`, `newtonnet`, `sevennet`, `uma` | `alphanet`, `chgnet`, `gptff`, `grace`, `hienet`, `matgl`, `orb`, `petmad` |
| `matgl` | `matgl>=2.0.0` | none | `alignn`, `alphanet`, `chgnet`, `deepmd`, `eqnorm`, `eqv2`, `esen`, `gptff`, `grace`, `hienet`, `mace`, `mattersim`, `nequip`, `nequix`, `newtonnet`, `orb`, `petmad`, `sevennet`, `uma` |
| `mattersim` | `mattersim>=1.2.1` | `alignn`, `chgnet`, `deepmd`, `eqv2`, `esen`, `mace`, `nequip`, `newtonnet`, `sevennet`, `uma` | `alphanet`, `eqnorm`, `gptff`, `grace`, `hienet`, `matgl`, `nequix`, `orb`, `petmad` |
| `nequip` | `nequip>=0.17.0` | `alignn`, `mace`, `mattersim`, `newtonnet` | `alphanet`, `chgnet`, `deepmd`, `eqnorm`, `eqv2`, `esen`, `gptff`, `grace`, `hienet`, `matgl`, `nequix`, `orb`, `petmad`, `sevennet`, `uma` |
| `nequix` | `nequix>=0.4.3` | `alignn`, `mace` | `alphanet`, `chgnet`, `deepmd`, `eqnorm`, `eqv2`, `esen`, `gptff`, `grace`, `hienet`, `matgl`, `mattersim`, `nequip`, `newtonnet`, `orb`, `petmad`, `sevennet`, `uma` |
| `newtonnet` | `newtonnet>=2.0.0` | `chgnet`, `deepmd`, `eqnorm`, `eqv2`, `esen`, `mace`, `mattersim`, `nequip`, `orb`, `sevennet`, `uma` | `alignn`, `alphanet`, `gptff`, `grace`, `hienet`, `matgl`, `nequix`, `petmad` |
| `orb` | `orb-models>=0.5.5` | `alignn`, `eqv2`, `esen`, `newtonnet`, `uma` | `alphanet`, `chgnet`, `deepmd`, `eqnorm`, `gptff`, `grace`, `hienet`, `mace`, `matgl`, `mattersim`, `nequip`, `nequix`, `petmad`, `sevennet` |
| `petmad` | `upet>=0.2.1` | `alignn`, `eqv2`, `esen` | `alphanet`, `chgnet`, `deepmd`, `eqnorm`, `gptff`, `grace`, `hienet`, `mace`, `matgl`, `mattersim`, `nequip`, `nequix`, `newtonnet`, `orb`, `sevennet`, `uma` |
| `sevennet` | `sevenn>=0.12.0` | `alignn`, `mace`, `mattersim`, `newtonnet` | `alphanet`, `chgnet`, `deepmd`, `eqnorm`, `eqv2`, `esen`, `gptff`, `grace`, `hienet`, `matgl`, `nequip`, `nequix`, `orb`, `petmad`, `uma` |
| `uma` | `fairchem-core>=2.0.0` | `alignn`, `deepmd`, `eqv2`, `esen`, `mace`, `mattersim`, `newtonnet`, `orb` | `alphanet`, `chgnet`, `eqnorm`, `gptff`, `grace`, `hienet`, `matgl`, `nequip`, `nequix`, `petmad`, `sevennet` |

## Pairwise Compatibility Grid

Legend:

- `Y` = can be installed together
- `N` = declared conflict in `uv`
- `-` = same extra

| MLIP | alignn | alphanet | chgnet | deepmd | eqnorm | eqv2 | esen | gptff | grace | hienet | mace | matgl | mattersim | nequip | nequix | newtonnet | orb | petmad | sevennet | uma |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| alignn | - | Y | N | N | Y | N | N | Y | Y | Y | N | Y | N | N | N | Y | N | N | N | N |
| alphanet | Y | - | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| chgnet | N | Y | - | Y | Y | Y | Y | Y | Y | Y | Y | Y | N | Y | Y | N | Y | Y | Y | Y |
| deepmd | N | Y | Y | - | Y | N | N | Y | Y | Y | N | Y | N | Y | Y | N | Y | Y | Y | N |
| eqnorm | Y | Y | Y | Y | - | Y | Y | Y | Y | Y | N | Y | Y | Y | Y | N | Y | Y | Y | Y |
| eqv2 | N | Y | Y | N | Y | - | Y | Y | Y | Y | N | Y | N | Y | Y | N | N | N | Y | N |
| esen | N | Y | Y | N | Y | Y | - | Y | Y | Y | N | Y | N | Y | Y | N | N | N | Y | N |
| gptff | Y | Y | Y | Y | Y | Y | Y | - | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| grace | Y | Y | Y | Y | Y | Y | Y | Y | - | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| hienet | Y | Y | Y | Y | Y | Y | Y | Y | Y | - | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| mace | N | Y | Y | N | N | N | N | Y | Y | Y | - | Y | N | N | N | N | Y | Y | N | N |
| matgl | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | - | Y | Y | Y | Y | Y | Y | Y | Y |
| mattersim | N | Y | N | N | Y | N | N | Y | Y | Y | N | Y | - | N | Y | N | Y | Y | N | N |
| nequip | N | Y | Y | Y | Y | Y | Y | Y | Y | Y | N | Y | N | - | Y | N | Y | Y | Y | Y |
| nequix | N | Y | Y | Y | Y | Y | Y | Y | Y | Y | N | Y | Y | Y | - | Y | Y | Y | Y | Y |
| newtonnet | Y | Y | N | N | N | N | N | Y | Y | Y | N | Y | N | N | Y | - | N | Y | N | N |
| orb | N | Y | Y | Y | Y | N | N | Y | Y | Y | Y | Y | Y | Y | Y | N | - | Y | Y | N |
| petmad | N | Y | Y | Y | Y | N | N | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | - | Y | Y |
| sevennet | N | Y | Y | Y | Y | Y | Y | Y | Y | Y | N | Y | N | Y | Y | N | Y | Y | - | Y |
| uma | N | Y | Y | N | Y | N | N | Y | Y | Y | N | Y | N | Y | Y | N | N | Y | Y | - |
