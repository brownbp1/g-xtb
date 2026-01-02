# 🚧 g-xTB — Development Version

This is a preliminary version of g-xTB, a general-purpose semiempirical quantum mechanical method approximating ωB97M-V/def2-TZVPPD properties.

## 📄 Preprint

See the preprint at ChemRxiv: [g-xTB: A General-Purpose Extended Tight-Binding Electronic Structure Method For the Elements H to Lr (Z=1–103)](https://chemrxiv.org/engage/chemrxiv/article-details/685434533ba0887c335fc974)

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

## 🧪 Batch conformer optimization + g-xTB energies (script)

If you have a multi-conformer SDF (e.g. 20k conformers of the same molecule) and want
**optimized geometries** plus **final g-xTB energies**, use:

`scripts/gxtb_optimize_conformers.py`

### Recommended workflow for large conformer sets

For very large batches, the recommended approach is:

- **Geometry optimization**: `xtb` using **GFN2-xTB** (fast, analytical gradients)
- **Final energy**: `gxtb` using **g-xTB** (this repository) as a **single-point** on the optimized geometry

This corresponds to `--mode xtb_opt_gxtb_sp`.

### What method produced which energies?

Per conformer directory `conf_XXXXX/`:

- **`xtb_opt.out`**: output of `xtb input.xyz --opt ...`
  - **Method**: GFN2-xTB (xtb)
  - **Energy units**: Hartree (Eh), shown in the `TOTAL ENERGY ... Eh` line
- **`xtbopt.xyz`**: optimized geometry from xtb
  - **Coordinate units**: Å (XYZ convention)
- **`gxtb_sp.out`**: output of `gxtb -c xtbopt.xyz ...`
  - **Method**: g-xTB (gxtb)
  - **Energy units**: Hartree (Eh), shown in the `total ...` line
- **`energy`**: energy file written by gxtb
  - Contains an `$energy` block; the **second column** is the total energy in **Eh**

The top-level summary file:

- **`energies.csv`**: columns `index,energy_Eh,status`
  - **`energy_Eh`** is the **g-xTB total energy** (from gxtb), in **Hartree (Eh)**

### Why are gxtb absolute energies so different from xtb energies?

This is expected. **Do not compare absolute energies between `xtb` (GFN2-xTB) and `gxtb` (g-xTB).**

- `xtb` (GFN2-xTB) uses its own semiempirical energy definition (valence-only framework), so its
  printed **TOTAL ENERGY** has a method-specific absolute reference.
- `gxtb` (g-xTB) prints a **DFT-like total energy scale** and includes large **per-atom core energy increments**
  (see the `atomic core increments` line in `gxtb_sp.out`). This shifts the absolute energy by a large
  (mostly constant-for-stoichiometry) amount.

For conformer ranking you should use **relative energies within the same method**:

- Use `energies.csv` (g-xTB) and compute \(\Delta E\) relative to the minimum.
- Optionally also compute \(\Delta E\) from `xtb_opt.out` (GFN2-xTB), but those \(\Delta E\) values are from a different method.

### Unit conversions (from Hartree)

- 1 Eh = 2625.49962 kJ/mol
- 1 Eh = 627.509474 kcal/mol
- 1 Eh = 27.211386 eV

### Charge and spin

The script writes `.CHRG` per conformer (by default from the SDF formal charge; or use `--charge` to force),
and optionally `.UHF` (use `--uhf`). This matches the control file conventions described above.

### Modes

- **`--mode xtb_opt_gxtb_sp` (recommended)**: optimize with xtb (GFN2-xTB), then gxtb single-point
- **`--mode gxtb_opt`**: optimize with xtb using gxtb as a numerical-gradient driver (slow/fragile; useful for refining a small subset)
- **`--mode gxtb_sp`**: gxtb single-point on the input geometry (no optimization)

### Example

```bash
export GXTBHOME=/path/to/g-xtb/parameters

python scripts/gxtb_optimize_conformers.py conformers.sdf \
  --mode xtb_opt_gxtb_sp \
  --xtb-opt-level loose \
  --workers 32 \
  --outdir gxtb_runs
```

### Conformational free energies for a ligand

Given `gxtb_runs/energies.csv` from the previous step, you can compute Boltzmann
probabilities and conformational free energies with:

```bash
# basic: free energies + Boltzmann probabilities
python scripts/gxtb_conformer_thermo.py gxtb_runs \
  --temperature 298.15 \
  --ref-index 1
```

This will write `gxtb_runs/conformer_free_energies.csv` with columns:

- `index` – conformer index (matching `energies.csv`)
- `E_Eh` – g-xTB total energy in Hartree
- `dE_rel_min_Eh`, `dE_rel_min_kcal` – relative to the minimum-energy conformer
- `prob` – Boltzmann probability at the given temperature
- `dG_vs_ref_kcal` – conformational free energy of conformer i relative to the reference

Additionally, a one-line summary of the reference conformer (probability and
ΔG_conf in kcal/mol) is printed to stdout.

### Optional: PNear metric (energy–structure funnel)

If you also provide per-conformer RMSDs to a chosen reference conformer, the script
can compute a PNear-like metric:

```bash
python scripts/gxtb_conformer_thermo.py gxtb_runs \
  --temperature 298.15 \
  --ref-index 1 \
  --rmsd-csv rmsd_vs_ref.csv \
  --rmsd-column rmsd_to_ref \
  --pn-lambda 1.0
```

Here `rmsd_vs_ref.csv` must contain at least:

- `index` – conformer indices matching `energies.csv`
- `rmsd_to_ref` – RMSD (Å) to the reference conformer

The script then evaluates

\\[
P_\\text{Near} = \\frac{\\sum_i e^{-(\\mathrm{RMSD}_i/\\lambda)^2} w_i}{\\sum_i w_i}
\\]

with \\(w_i\\) the Boltzmann weights from the g-xTB energies and \\(\\lambda\\) a
user-chosen RMSD scale (Å). A larger PNear means that low-energy conformers are
clustered near the reference.

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
