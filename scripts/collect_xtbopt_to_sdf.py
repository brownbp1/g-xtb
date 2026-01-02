#!/usr/bin/env python3
"""
Collect xtb-optimized geometries (xtbopt.xyz) into a single multi-conformer SDF.

Intended workflow:
  1. Run `gxtb_optimize_conformers.py` with --mode xtb_opt_gxtb_sp, which creates:
       run_dir/conf_00001/xtbopt.xyz
       run_dir/conf_00002/xtbopt.xyz
       ...
  2. Use the original input SDF as a topology template (bonds, atom order).
  3. Build a single RDKit Mol with one conformer per xtbopt.xyz, and write it as SDF.

This makes it easy to:
  - work in a single SDF (for measuring torsions, RMSDs, etc.)
  - keep atom ordering and connectivity consistent with the original SDF.

Assumptions:
  - All conformers in the input SDF share the same connectivity.
  - Atom order in xtbopt.xyz matches that of the original SDF (true if generated
    via `gxtb_optimize_conformers.py` and xtb did not reorder atoms).
"""

import argparse
from pathlib import Path
from typing import List, Tuple

from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Geometry import Point3D


def read_xyz_coords(xyz_path: Path) -> List[Tuple[str, float, float, float]]:
    """Read a simple XYZ file and return (symbol, x, y, z) tuples."""
    lines = xyz_path.read_text().strip().splitlines()
    if len(lines) < 3:
        raise ValueError(f"{xyz_path} does not look like a valid XYZ file")
    # First line: atom count (ignored here, but we can sanity-check)
    try:
        nat = int(lines[0].strip().split()[0])
    except Exception:
        raise ValueError(f"First line of {xyz_path} must contain atom count")
    coords: List[Tuple[str, float, float, float]] = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        sym = parts[0]
        x, y, z = map(float, parts[1:4])
        coords.append((sym, x, y, z))
    if len(coords) != nat:
        # Not fatal, but warn via exception so the user can inspect.
        raise ValueError(
            f"{xyz_path}: atom count mismatch (header {nat}, coords {len(coords)})"
        )
    return coords


def build_multi_conformer_mol(
    template_mol: rdchem.Mol,
    xtbopt_files: List[Path],
) -> rdchem.Mol:
    """
    Create a copy of template_mol and attach one Conformer per xtbopt.xyz file.
    """
    n_atoms = template_mol.GetNumAtoms()
    mol = Chem.Mol(template_mol)
    # Remove any existing conformers.
    for cid in list(range(mol.GetNumConformers())):
        mol.RemoveConformer(cid)

    for i, xyz_path in enumerate(xtbopt_files, start=1):
        coords = read_xyz_coords(xyz_path)
        if len(coords) != n_atoms:
            raise ValueError(
                f"{xyz_path}: expected {n_atoms} atoms, found {len(coords)}"
            )
        conf = rdchem.Conformer(n_atoms)
        for idx, (_, x, y, z) in enumerate(coords):
            conf.SetAtomPosition(idx, Point3D(x, y, z))
        conf.SetId(i - 1)
        mol.AddConformer(conf, assignId=True)

    return mol


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect xtbopt.xyz geometries into a multi-conformer SDF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "run_dir",
        help="Directory containing conf_XXXXX subdirectories from gxtb_optimize_conformers.py.",
    )
    parser.add_argument(
        "--input-sdf",
        required=True,
        help="Original multi-conformer SDF used as input (provides topology/atom order).",
    )
    parser.add_argument(
        "--output-sdf",
        default=None,
        help="Output SDF path. Defaults to <run_dir>/xtbopt_conformers.sdf.",
    )

    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"ERROR: run_dir {run_dir} is not a directory")

    input_sdf = Path(args.input_sdf).resolve()
    if not input_sdf.is_file():
        raise SystemExit(f"ERROR: input SDF not found: {input_sdf}")

    if args.output_sdf:
        output_sdf = Path(args.output_sdf).resolve()
    else:
        output_sdf = run_dir / "xtbopt_conformers.sdf"

    # Discover conf_XXXXX dirs and xtbopt.xyz files.
    conf_dirs = sorted(
        d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("conf_")
    )
    if not conf_dirs:
        raise SystemExit(f"ERROR: no conf_XXXXX directories found under {run_dir}")

    xtbopt_files: List[Path] = []
    for d in conf_dirs:
        xyz_path = d / "xtbopt.xyz"
        if not xyz_path.is_file():
            raise SystemExit(f"ERROR: {xyz_path} not found")
        xtbopt_files.append(xyz_path)

    # Use first molecule from input SDF as topology template.
    suppl = Chem.SDMolSupplier(str(input_sdf), removeHs=False)
    template = None
    for mol in suppl:
        if mol is not None:
            template = mol
            break
    if template is None:
        raise SystemExit(f"ERROR: no valid molecules found in {input_sdf}")

    if template.GetNumAtoms() != len(read_xyz_coords(xtbopt_files[0])):
        raise SystemExit(
            "ERROR: atom count mismatch between template SDF and xtbopt.xyz. "
            "Make sure you passed the correct input SDF."
        )

    mol = build_multi_conformer_mol(template, xtbopt_files)

    w = Chem.SDWriter(str(output_sdf))
    # Write one record per conformer.
    for conf in mol.GetConformers():
        w.write(mol, confId=conf.GetId())
    w.close()

    print(
        f"Wrote {mol.GetNumConformers()} conformers with xtb-optimized geometries to {output_sdf}"
    )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


