# 🚧 g-xTB — Development Version

This is a preliminary version of g-xTB, a general-purpose semiempirical quantum mechanical method approximating ωB97M-V/def2-TZVPPD properties.

## 📄 Preprint

See the preprint at ChemRxiv: https://chemrxiv.org/engage/chemrxiv/article-details/685434533ba0887c335fc974

## 📦 Installation

> [!WARNING]
> `gxtb` currently works only on Linux-based machines.

Place the `gxtb` binary in a directory belonging to your `$PATH` variable (e.g., `$USER/bin/`).

Place the following parameter and basis files into a dedicated directory, which you export in the `$GXTBHOME` variable: 
- `.gxtb` — parameter file (`-p`)
- `.eeq` — electronegativity equilibration parameters (`-e`)
- `.basisq` — atom-in-molecule AO basis (`-b`)
If `$GXTBHOME` is not defined, the `gxtb` binary searches first in your home directory `$HOME` and then in the current directory (`./`). You can overwrite the location of the parameter files with the resepctive command-line flags (`-p`, `-e`, and `-b`). 

## Usage

By default, `gxtb` expects a coordinate file in TURBOMOLE format (`coord`) using atomic units (i.e. Bohr). 

### Run examples

```
gxtb                       # default: coord file = TURBOMOLE format
gxtb -c <coord_file_name>  # explicit coordinate file (TURBOMOLE or XYZ)
gxtb -c <xyz_file_name>    # XYZ file supported
```

Place the following optional control files in your working directory:
- `.CHRG` # Integer charge of the system (default: neutral)
- `.UHF` # Integer number of open shells (e.g., 2 for triplet, 0 for singlet UKS)

If `.CHRG` or `.UHF` are not present: 
- Even electrons: neutral singlet (RKS)
- Odd electrons: neutral doublet (UKS)

## ⚙️ Additional Features

### Numerical Gradient

```
gxtb -grad
```
Or if a file named `.GRAD` is present, a numerical gradient is computed (expensive!).
Molecular symmetry is exploited to speed up calculations.

### Geometry Optimization with `xtb`

To optimize geometries using xtb with gxtb as a driver:
```
xtb struc.xyz --driver "gxtb -grad -c xtbdriver.xyz" --opt
```
Or with a `coord` file in TURBOMOLE format:
```
xtb coord --driver "gxtb -grad -c xtbdriver.coord" --opt
```

💡 You may use `--opt loose` for faster convergence, as there is currently no analytical nuclear gradient — gradients are evaluated numerically and can be noisy.

### Numerical Hessian

```
gxtb -hess
```
Computes a numerical Hessian (very expensive).

## Current Coverage

- Reasonably parameterized for elements Z = 1–58, 71–89, and 92
- A revised dispersion model (`revD4`) is in progress and may slightly affect final results

## 📊 Output and Analysis

- All computed properties aim to approximate ωB97M-V/def2-TZVPPD
- EEQ_BC charges mimic Hirshfeld charges from that reference
- Use the `-molden` flag to write a `.molden` file with orbitals and basis info:
```
gxtb -molden
```
Useful for visualization and post-processing.
