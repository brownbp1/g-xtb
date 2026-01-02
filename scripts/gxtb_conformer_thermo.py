#!/usr/bin/env python3
"""
Post-process g-xTB conformer energies to conformational free energies (and optionally PNear).

This script takes an `energies.csv` (as written by `gxtb_optimize_conformers.py`)
and computes:

- ΔE_i relative to the lowest-energy conformer (in Eh and kcal/mol)
- Boltzmann probabilities at a given temperature
- Conformational free energy penalty for a chosen reference conformer

Typical use case:
- You have many conformers of a ligand.
- You know which conformer is the “reference” (e.g. bound pose).
- You want to know how favorable that conformer is relative to the full ensemble
  at temperature T.

Definitions:
- All input energies are assumed to be **g-xTB total energies** in **Hartree (Eh)**.
- Relative enthalpies:
    ΔE_i = E_i - E_min
- Boltzmann weights at temperature T:
    w_i = exp( -ΔE_i / (k_B T) ), k_B in kcal/mol/K
- Probabilities:
    p_i = w_i / Σ_j w_j
- Conformational free energy of reference conformer (index ref):
    ΔG_conf(ref) = -RT ln(p_ref)   [kcal/mol]
  which is the free energy penalty of constraining the ensemble to the reference.

Additionally we report for every conformer:
- dG_vs_ref_kcal = -RT ln( p_i / p_ref )  (G_i - G_ref)

Optional PNear metric:
If you provide RMSDs to a reference conformer (e.g. bound pose), the script can
also compute a PNear-like metric:

  PNear = sum_i [ exp( - (RMSD_i / λ)^2 ) * w_i ] / sum_i [ w_i ]

where λ is a user-chosen RMSD scale (in Å), and w_i are the Boltzmann weights
based on g-xTB energies. High PNear means that low-energy conformers are
clustered near the reference.

Units:
- Energies in `energies.csv` are Hartree (Eh).
- Output free energies are in kcal/mol.
"""

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from rdkit import Chem
from rdkit.Chem import rdMolAlign, rdMolTransforms

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

K_B_KCAL_PER_MOL_K = 0.00198720425864083  # kcal/mol/K
HARTREE_TO_KCAL = 627.509474


def read_energies(csv_path: Path) -> List[Tuple[int, float]]:
    """
    Read index, energy_Eh from energies.csv, skipping rows without valid energies
    or with non-ok status (if status column present).
    """
    entries: List[Tuple[int, float]] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx_str = row.get("index")
            e_str = row.get("energy_Eh", "")
            status = row.get("status", "ok")
            if not idx_str:
                continue
            try:
                idx = int(idx_str)
            except ValueError:
                continue
            if not e_str:
                continue
            if status and status.strip().lower() != "ok":
                continue
            try:
                e = float(e_str)
            except ValueError:
                continue
            entries.append((idx, e))
    if not entries:
        raise ValueError(f"No valid energies found in {csv_path}")
    return entries


def read_rmsd(
    csv_path: Path,
    index_column: str = "index",
    rmsd_column: str = "rmsd",
) -> Dict[int, float]:
    """
    Read per-conformer RMSD values from a CSV file.

    The CSV must contain at least two columns:
      - `index` (or a custom index_column): conformer index (int)
      - `rmsd` (or a custom rmsd_column): RMSD to the reference (Å)

    Returns:
      dict[index] = rmsd
    """
    rmsd_map: Dict[int, float] = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx_str = row.get(index_column)
            r_str = row.get(rmsd_column)
            if not idx_str or not r_str:
                continue
            try:
                idx = int(idx_str)
                r = float(r_str)
            except ValueError:
                continue
            rmsd_map[idx] = r
    if not rmsd_map:
        raise ValueError(f"No valid RMSD entries found in {csv_path}")
    return rmsd_map


def compute_conformer_free_energies(
    entries: List[Tuple[int, float]],
    temperature: float,
    ref_index: Optional[int],
    rmsd_map: Optional[Dict[int, float]] = None,
    pn_lambda: Optional[float] = None,
) -> Tuple[List[dict], Optional[dict], Optional[float]]:
    """
    Compute ΔE, Boltzmann probabilities, and conformational free energy.

    Returns:
      - per-conformer list of dicts
      - summary dict for reference conformer (or None if no reference)
      - PNear value (or None if no RMSD provided)
    """
    # Sort by index to keep a stable order.
    entries = sorted(entries, key=lambda x: x[0])
    energies_eh = [e for _, e in entries]
    idxs = [i for i, _ in entries]

    e_min = min(energies_eh)
    dE_eh = [e - e_min for e in energies_eh]
    dE_kcal = [de * HARTREE_TO_KCAL for de in dE_eh]

    kT = K_B_KCAL_PER_MOL_K * temperature
    # Boltzmann weights (relative to min).
    weights = [math.exp(-de_kcal / kT) for de_kcal in dE_kcal]
    Z = sum(weights)
    probs = [w / Z for w in weights]

    ref_summary: Optional[dict] = None
    ref_prob = None
    if ref_index is not None:
        try:
            ref_pos = idxs.index(ref_index)
        except ValueError:
            raise ValueError(f"Reference index {ref_index} not found in energies.csv")
        ref_prob = probs[ref_pos]
        dG_conf_ref = -kT * math.log(ref_prob)
        ref_summary = {
            "ref_index": ref_index,
            "E_ref_Eh": energies_eh[ref_pos],
            "E_ref_rel_min_kcal": dE_kcal[ref_pos],
            "P_ref": ref_prob,
            "dG_conf_ref_kcal": dG_conf_ref,
        }

    per_conf: List[dict] = []
    rmsds: List[Optional[float]] = []
    for i, (idx, e_eh) in enumerate(entries):
        p_i = probs[i]
        row = {
            "index": idx,
            "E_Eh": e_eh,
            "dE_rel_min_Eh": dE_eh[i],
            "dE_rel_min_kcal": dE_kcal[i],
            "prob": p_i,
        }
        r: Optional[float] = None
        if rmsd_map is not None:
            r = rmsd_map.get(idx)
            if r is not None:
                row["rmsd"] = r
        rmsds.append(r)
        if ref_prob is not None:
            # ΔG_i - ΔG_ref = -RT ln(p_i/p_ref)
            dG_vs_ref = -kT * math.log(p_i / ref_prob)
            row["dG_vs_ref_kcal"] = dG_vs_ref
        per_conf.append(row)

    # Optional PNear calculation if we have RMSDs and a lambda.
    pn_val: Optional[float] = None
    if rmsd_map is not None and pn_lambda is not None and pn_lambda > 0.0:
        num = 0.0
        denom = 0.0
        for p_i, r in zip(probs, rmsds):
            if r is None:
                continue
            weight = math.exp(- (r / pn_lambda) ** 2)
            num += weight * p_i
            denom += p_i
        if denom > 0.0:
            pn_val = num / denom

    return per_conf, ref_summary, pn_val


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute conformer free energies from g-xTB energies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Notes:\n"
            "  - Input energies must be g-xTB total energies (Hartree) from energies.csv.\n"
            "  - This script treats all conformers as microstates of the same molecule.\n"
            "  - ΔG_conf(ref) = -RT ln(P_ref), reported in kcal/mol.\n"
        ),
    )
    parser.add_argument(
        "path",
        help="Path to energies.csv or to a directory containing energies.csv.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=298.15,
        help="Temperature in K for Boltzmann weights. [default: %(default)s]",
    )
    parser.add_argument(
        "--ref-index",
        type=int,
        default=None,
        help="Reference conformer index (1-based), matching the 'index' column in energies.csv.",
    )
    parser.add_argument(
        "--rmsd-csv",
        default=None,
        help="Optional CSV file containing RMSD values vs. a reference conformer. "
             "Must contain columns 'index' and 'rmsd' (or use --rmsd-column/index-column).",
    )
    parser.add_argument(
        "--rmsd-column",
        default="rmsd",
        help="RMSD column name in --rmsd-csv. [default: %(default)s]",
    )
    parser.add_argument(
        "--index-column",
        default="index",
        help="Index column name in --rmsd-csv. [default: %(default)s]",
    )
    parser.add_argument(
        "--pn-lambda",
        type=float,
        default=1.0,
        help="RMSD length scale λ (Å) used in the PNear Gaussian weight. [default: %(default)s]",
    )
    parser.add_argument(
        "--sdf",
        default=None,
        help="Optional multi-conformer SDF with geometries (e.g. xtbopt_conformers.sdf). "
             "If provided and --rmsd-csv is not, RMSDs to a reference conformer are "
             "computed automatically.",
    )
    parser.add_argument(
        "--ref-sdf",
        default=None,
        help="Optional SDF containing a single reference conformer (e.g. bound pose). "
             "Used for RMSD and torsion references. Topology must match --sdf "
             "(same atom count; same connectivity; for torsions, same atom order).",
    )
    parser.add_argument(
        "--rmsd-ref-index",
        type=int,
        default=None,
        help="Reference conformer index (1-based) for RMSD when using --sdf. "
             "Defaults to --ref-index or 1.",
    )
    parser.add_argument(
        "--torsion",
        action="append",
        default=[],
        help="Glycosidic torsion definition as 'i,j,k,l' (1-based atom indices). "
             "Can be given multiple times.",
    )
    parser.add_argument(
        "--torsion-ref-index",
        type=int,
        default=None,
        help="Reference conformer index (1-based) for torsion differences. "
             "Defaults to --ref-index or 1.",
    )
    parser.add_argument(
        "--torsion-lambda",
        type=float,
        default=30.0,
        help="Angle scale λ (deg) used when aggregating torsion deviations into "
             "a single torsion_dist measure. [default: %(default)s]",
    )
    parser.add_argument(
        "--cv-column",
        default=None,
        help="Column name to use as the collective variable (CV) for 1D free energy "
             "landscapes. Defaults to 'torsion_dist' if torsions are given, else "
             "'rmsd' if available, else 'dE_rel_min_kcal'.",
    )
    parser.add_argument(
        "--cv-column2",
        default=None,
        help="Optional second column for 2D free energy surfaces. If provided, "
             "generates 2D FES(cv1, cv2) plots.",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=None,
        help="Bin width for 1D free energy profile along the CV. Units follow the "
             "chosen --cv-column. If omitted, a heuristic default is chosen.",
    )
    parser.add_argument(
        "--bin-width2",
        type=float,
        default=None,
        help="Bin width for second dimension in 2D plots. Defaults to --bin-width if omitted.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting of the 1D free energy profile.",
    )
    parser.add_argument(
        "--fes-max-kcal",
        type=float,
        default=None,
        help=(
            "Optional explicit maximum free energy (kcal/mol) for the color scale in 2D FES plots. "
            "If not set, a data-driven value based on mean(F) + sigma_cutoff * std(F) is used."
        ),
    )
    parser.add_argument(
        "--fes-sigma-cutoff",
        type=float,
        default=2.5,
        help=(
            "When --fes-max-kcal is not set, the 2D FES color scale upper bound is "
            "min(max(F), mean(F) + fes_sigma_cutoff * std(F)). [default: %(default)s]"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for output CSV (default: conformer_free_energies.csv next to energies.csv).",
    )

    args = parser.parse_args(argv)

    p = Path(args.path).resolve()
    if p.is_dir():
        csv_path = p / "energies.csv"
    else:
        csv_path = p
    if not csv_path.is_file():
        raise SystemExit(f"ERROR: energies.csv not found at {csv_path}")

    entries = read_energies(csv_path)

    # Optionally load conformer SDF once, for RMSD and torsions.
    conf_mols: Optional[List[Chem.Mol]] = None
    if args.sdf:
        sdf_path = Path(args.sdf).resolve()
        if not sdf_path.is_file():
            raise SystemExit(f"ERROR: SDF file not found: {sdf_path}")
        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        conf_mols = [m for m in suppl if m is not None]
        if not conf_mols:
            raise SystemExit(f"ERROR: no valid molecules in {sdf_path}")
        if len(conf_mols) != len(entries):
            raise SystemExit(
                f"ERROR: SDF has {len(conf_mols)} molecules but energies.csv has {len(entries)} entries. "
                "They must correspond one-to-one (same ordering)."
            )

    # Optional separate reference SDF (e.g. bound ligand pose).
    ref_mol_global: Optional[Chem.Mol] = None
    if args.ref_sdf:
        ref_sdf_path = Path(args.ref_sdf).resolve()
        if not ref_sdf_path.is_file():
            raise SystemExit(f"ERROR: ref SDF not found: {ref_sdf_path}")
        ref_suppl = Chem.SDMolSupplier(str(ref_sdf_path), removeHs=False)
        ref_mols = [m for m in ref_suppl if m is not None]
        if len(ref_mols) != 1:
            raise SystemExit(f"ERROR: ref SDF {ref_sdf_path} must contain exactly one molecule")
        ref_mol_global = ref_mols[0]
        if conf_mols is not None:
            # Topology check: same atom count and same connectivity (via canonical SMILES).
            if ref_mol_global.GetNumAtoms() != conf_mols[0].GetNumAtoms():
                raise SystemExit(
                    "ERROR: reference SDF topology mismatch: atom counts differ between --ref-sdf and --sdf."
                )
            smi_conf = Chem.MolToSmiles(conf_mols[0], isomericSmiles=False, canonical=True)
            smi_ref = Chem.MolToSmiles(ref_mol_global, isomericSmiles=False, canonical=True)
            if smi_conf != smi_ref:
                raise SystemExit(
                    "ERROR: reference SDF topology mismatch: connectivity differs between --ref-sdf and --sdf."
                )

    rmsd_map: Optional[Dict[int, float]] = None
    # Strategy for RMSD:
    #   1) If --rmsd-csv given, use that.
    #   2) Else if --sdf given, compute RMSDs from SDF vs a reference conformer
    #      possibly from --ref-sdf.
    if args.rmsd_csv:
        rmsd_map = read_rmsd(
            Path(args.rmsd_csv).resolve(),
            index_column=args.index_column,
            rmsd_column=args.rmsd_column,
        )
    elif conf_mols is not None:
        if ref_mol_global is not None:
            ref_mol_for_rmsd = ref_mol_global
        else:
            # Determine reference conformer index for RMSD from conformer SDF.
            if args.rmsd_ref_index is not None:
                rmsd_ref_idx = args.rmsd_ref_index
            elif args.ref_index is not None:
                rmsd_ref_idx = args.ref_index
            else:
                rmsd_ref_idx = 1
            if rmsd_ref_idx < 1 or rmsd_ref_idx > len(conf_mols):
                raise SystemExit(f"ERROR: rmsd_ref_index {rmsd_ref_idx} out of range 1..{len(conf_mols)}")
            ref_mol_for_rmsd = conf_mols[rmsd_ref_idx - 1]
        rmsd_map = {}
        for i, mol in enumerate(conf_mols, start=1):
            rms = rdMolAlign.GetBestRMS(ref_mol_for_rmsd, mol)
            rmsd_map[i] = rms

    pn_lambda = args.pn_lambda if rmsd_map is not None else None

    per_conf, ref_summary, pn_val = compute_conformer_free_energies(
        entries,
        temperature=args.temperature,
        ref_index=args.ref_index,
        rmsd_map=rmsd_map,
        pn_lambda=pn_lambda,
    )

    # Optional torsion analysis based on SDF and user-provided atom indices.
    torsion_defs: List[Tuple[int, int, int, int]] = []
    if args.torsion:
        if conf_mols is None:
            raise SystemExit("--torsion requires --sdf to provide geometries.")
        mols = conf_mols
        # Parse torsion definitions, expect 1-based indices i,j,k,l.
        for spec in args.torsion:
            parts = spec.replace(" ", "").split(",")
            if len(parts) != 4:
                raise SystemExit(f"Invalid --torsion '{spec}', expected 'i,j,k,l'")
            try:
                a1, a2, a3, a4 = [int(x) - 1 for x in parts]
            except ValueError:
                raise SystemExit(f"Invalid --torsion '{spec}', indices must be integers")
            torsion_defs.append((a1, a2, a3, a4))
        # Choose torsion reference conformer index (1-based) or use separate ref_sdf.
        if ref_mol_global is not None:
            # For torsions we also require identical atom ordering (so that the
            # user-specified indices refer to the same atoms).
            for idx in range(ref_mol_global.GetNumAtoms()):
                if ref_mol_global.GetAtomWithIdx(idx).GetAtomicNum() != mols[0].GetAtomWithIdx(idx).GetAtomicNum():
                    raise SystemExit(
                        "ERROR: atom ordering differs between --ref-sdf and --sdf; cannot safely use "
                        "atom indices for torsions. Please reorder the reference or omit --ref-sdf."
                    )
            ref_conf = ref_mol_global.GetConformer()
        else:
            if args.torsion_ref_index is not None:
                t_ref_idx = args.torsion_ref_index
            elif args.ref_index is not None:
                t_ref_idx = args.ref_index
            else:
                t_ref_idx = 1
            if t_ref_idx < 1 or t_ref_idx > len(mols):
                raise SystemExit(f"ERROR: torsion_ref_index {t_ref_idx} out of range 1..{len(mols)}")
            ref_conf = mols[t_ref_idx - 1].GetConformer()

        # Precompute reference torsion angles.
        ref_torsions_deg: List[float] = []
        for (a1, a2, a3, a4) in torsion_defs:
            ang = rdMolTransforms.GetDihedralDeg(ref_conf, a1, a2, a3, a4)
            ref_torsions_deg.append(ang)

        def wrap_deg(x: float) -> float:
            """Wrap angle difference into (-180, 180] deg."""
            while x > 180.0:
                x -= 360.0
            while x <= -180.0:
                x += 360.0
            return x

        # Compute torsions per conformer and attach to per_conf rows.
        # We assume entries and mols share the same order and length.
        for i, (entry, mol, row) in enumerate(zip(entries, mols, per_conf)):
            conf = mol.GetConformer()
            torsion_vals: List[float] = []
            torsion_deltas: List[float] = []
            for (a1, a2, a3, a4), ref_ang in zip(torsion_defs, ref_torsions_deg):
                ang = rdMolTransforms.GetDihedralDeg(conf, a1, a2, a3, a4)
                d = wrap_deg(ang - ref_ang)
                torsion_vals.append(ang)
                torsion_deltas.append(d)
            # Aggregate distance in torsion space (using lambda in degrees).
            lam = max(args.torsion_lambda, 1e-6)
            torsion_dist = math.sqrt(sum((d / lam) ** 2 for d in torsion_deltas))
            # Attach to row with systematic column names.
            for j, (ang, d) in enumerate(zip(torsion_vals, torsion_deltas), start=1):
                row[f"torsion{j}_deg"] = ang
                row[f"d_torsion{j}_deg"] = d
            row["torsion_dist"] = torsion_dist

    # Decide which CV columns to generate profiles for:
    # - if --cv-column is given: just that one
    # - else: automatically scan per_conf rows for distance-like and CV-like columns.
    if args.cv_column:
        cv_columns = [args.cv_column]
    else:
        # Start with a stable order: energy distance, rmsd, torsion distance.
        cv_columns = []
        if "dE_rel_min_kcal" in per_conf[0]:
            cv_columns.append("dE_rel_min_kcal")
        if "rmsd" in per_conf[0]:
            cv_columns.append("rmsd")
        if "torsion_dist" in per_conf[0]:
            cv_columns.append("torsion_dist")
        # Add individual torsion distances and raw torsions if present.
        for key in per_conf[0].keys():
            if key.startswith("d_torsion") and key.endswith("_deg"):
                if key not in cv_columns:
                    cv_columns.append(key)
        for key in per_conf[0].keys():
            if key.startswith("torsion") and key.endswith("_deg") and not key.startswith("d_torsion"):
                if key not in cv_columns:
                    cv_columns.append(key)

    # Bin width heuristics per column, can be overridden globally with --bin-width.
    def default_bin_width(col: str) -> float:
        if args.bin_width is not None:
            return args.bin_width
        if col.endswith("_deg"):
            return 10.0
        if col in ("torsion_dist", "rmsd"):
            return 0.1
        if col == "dE_rel_min_kcal":
            return 0.5
        return 0.5

    # Write output CSV.
    fieldnames = [
        "index",
        "E_Eh",
        "dE_rel_min_Eh",
        "dE_rel_min_kcal",
        "prob",
    ]
    if rmsd_map is not None:
        fieldnames.append("rmsd")
    # Torsion-related fields if requested.
    if torsion_defs:
        for j in range(1, len(torsion_defs) + 1):
            fieldnames.append(f"torsion{j}_deg")
            fieldnames.append(f"d_torsion{j}_deg")
        fieldnames.append("torsion_dist")
    if args.ref_index is not None:
        fieldnames.append("dG_vs_ref_kcal")

    if args.output:
        out_path = Path(args.output).resolve()
    else:
        out_path = csv_path.parent / "conformer_free_energies.csv"

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_conf:
            writer.writerow(row)

    print(f"Wrote conformer free energies to {out_path}")

    # Build 1D free energy profiles for each selected CV column.
    for cv_col in cv_columns:
        # Extract (CV, prob) pairs.
        cv_vals: List[float] = []
        cv_wts: List[float] = []
        for row in per_conf:
            v = row.get(cv_col)
            if v is None:
                continue
            try:
                v_f = float(v)
            except (TypeError, ValueError):
                continue
            cv_vals.append(v_f)
            cv_wts.append(float(row.get("prob", 0.0)))

        if not cv_vals or sum(cv_wts) <= 0.0:
            continue

        bw = default_bin_width(cv_col)
        vmin = min(cv_vals)
        vmax = max(cv_vals)
        if vmax == vmin:
            continue
        # Ensure at least one bin.
        n_bins = max(1, int(math.ceil((vmax - vmin) / bw)))
        edges = [vmin + i * bw for i in range(n_bins + 1)]
        centers = [0.5 * (edges[i] + edges[i + 1]) for i in range(n_bins)]
        bin_counts = [0.0 for _ in range(n_bins)]
        for v, w in zip(cv_vals, cv_wts):
            # Assign to bin index.
            idx_bin = int((v - vmin) / bw)
            if idx_bin < 0:
                idx_bin = 0
            elif idx_bin >= n_bins:
                idx_bin = n_bins - 1
            bin_counts[idx_bin] += w
        # Normalize to probabilities.
        total_w = sum(bin_counts)
        if total_w <= 0.0:
            continue
        bin_probs = [c / total_w for c in bin_counts]
        # Convert to free energy in kcal/mol (up to a constant).
        kT = K_B_KCAL_PER_MOL_K * args.temperature
        freeE = [(-kT * math.log(p) if p > 0.0 else float("inf")) for p in bin_probs]
        finite_F = [f for f in freeE if math.isfinite(f)]
        if finite_F:
            Fmin = min(finite_F)
            freeE = [f - Fmin if math.isfinite(f) else f for f in freeE]

        profile_path = out_path.parent / (out_path.stem + f"_{cv_col}_profile.csv")
        with profile_path.open("w", newline="") as pf:
            pw = csv.writer(pf)
            pw.writerow(
                ["bin_center", "bin_left", "bin_right", "prob", "F_kcal_per_mol"]
            )
            for c_center, left, right, p, fE in zip(
                centers, edges[:-1], edges[1:], bin_probs, freeE
            ):
                pw.writerow([c_center, left, right, p, fE])

        print(
            f"Wrote 1D free energy profile for CV='{cv_col}' to {profile_path} "
            f"(bin_width={bw})"
        )

        # Optional plotting (1D).
        if not args.no_plot:
            try:
                import matplotlib.pyplot as plt
                import matplotlib as mpl
                import numpy as np
            except ImportError:
                print(
                    "matplotlib not available; skipping free energy plot.",
                    file=sys.stderr,
                )
            else:
                # Publication-quality style settings
                mpl.rcParams['font.family'] = 'sans-serif'
                mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
                mpl.rcParams['font.size'] = 12
                mpl.rcParams['axes.linewidth'] = 1.5
                mpl.rcParams['lines.linewidth'] = 2.0
                mpl.rcParams['xtick.major.width'] = 1.5
                mpl.rcParams['ytick.major.width'] = 1.5
                mpl.rcParams['xtick.direction'] = 'out'
                mpl.rcParams['ytick.direction'] = 'out'
                mpl.rcParams['figure.dpi'] = 300
                mpl.rcParams['savefig.bbox'] = 'tight'

                # Prepare smoothed free energy curve via spline (if SciPy available)
                centers_arr = np.array(centers, dtype=float)
                freeE_arr = np.array(freeE, dtype=float)
                mask_finite = np.isfinite(freeE_arr)
                x_base = centers_arr[mask_finite]
                y_base = freeE_arr[mask_finite]
                if x_base.size >= 2:
                    x_dense = np.linspace(x_base.min(), x_base.max(), 400)
                    try:
                        from scipy.interpolate import UnivariateSpline
                    except ImportError:
                        # Fallback: linear interpolation
                        y_smooth = np.interp(x_dense, x_base, y_base)
                    else:
                        # Smoothing parameter heuristic scaled by number of points
                        s = max(len(x_base) * (0.1 ** 2), 1e-8)
                        spline = UnivariateSpline(x_base, y_base, s=s)
                        y_smooth = spline(x_dense)
                else:
                    x_dense = centers_arr
                    y_smooth = freeE_arr

                fig, axF = plt.subplots(figsize=(5, 3.5))

                # Histogram on secondary axis (probabilities)
                axP = axF.twinx()
                bar_width = bw * 0.9
                axP.bar(
                    centers_arr,
                    bin_probs,
                    width=bar_width,
                    color="0.85",
                    edgecolor="none",
                    alpha=0.8,
                    align="center",
                )
                axP.set_ylabel("Probability", fontsize=10)
                axP.tick_params(axis="y", labelsize=9)
                if max(bin_probs) > 0.0:
                    axP.set_ylim(0.0, max(bin_probs) * 1.1)

                # Smoothed free energy curve
                axF.plot(x_dense, y_smooth, "-k", lw=2)

                # Mark reference value on the CV axis, if available.
                ref_x: Optional[float] = None
                # Structural references:
                if cv_col in ("torsion_dist", "rmsd") or (
                    cv_col.startswith("d_torsion") and cv_col.endswith("_deg")
                ):
                    ref_x = 0.0
                elif torsion_defs and "ref_torsions_deg" in locals():
                    # Columns like torsion1_deg, torsion2_deg, ...
                    if cv_col.startswith("torsion") and cv_col.endswith("_deg") and not cv_col.startswith("d_torsion"):
                        num_str = cv_col[len("torsion") : -len("_deg")]
                        try:
                            j = int(num_str)
                            if 1 <= j <= len(ref_torsions_deg):
                                ref_x = ref_torsions_deg[j - 1]
                        except ValueError:
                            ref_x = None
                # If a thermodynamic reference conformer was provided, also mark its CV value.
                ref_cv_from_index: Optional[float] = None
                if args.ref_index is not None:
                    for row in per_conf:
                        if row.get("index") == args.ref_index and cv_col in row:
                            try:
                                ref_cv_from_index = float(row[cv_col])
                            except (TypeError, ValueError):
                                ref_cv_from_index = None
                            break

                if ref_x is not None:
                    axF.axvline(ref_x, color="red", linestyle="--", linewidth=1.5, label="structural ref")
                if ref_cv_from_index is not None and (ref_x is None or abs(ref_cv_from_index - ref_x) > 1e-6):
                    axF.axvline(ref_cv_from_index, color="blue", linestyle=":", linewidth=1.5, label="ref_index")
                if (ref_x is not None) or (ref_cv_from_index is not None):
                    axF.legend(frameon=False, fontsize=9, loc="best")

                axF.set_xlabel(cv_col, fontsize=12, fontweight="bold")
                axF.set_ylabel("Free energy (kcal/mol)", fontsize=12, fontweight="bold")
                axF.set_title(f"1D FES along {cv_col}\n(T = {args.temperature:.1f} K)", fontsize=13)

                # Improve tick labels
                axF.tick_params(axis="both", which="major", labelsize=10)

                # Add grid but make it subtle
                axF.grid(True, linestyle=":", alpha=0.6)

                fig.tight_layout()
                plot_path = profile_path.with_suffix(".png")
                fig.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close(fig)

    # 2D Free Energy Surface (if requested)
    if args.cv_column and args.cv_column2:
        cv1 = args.cv_column
        cv2 = args.cv_column2
        
        cv1_vals = []
        cv2_vals = []
        wts = []
        
        for row in per_conf:
            v1 = row.get(cv1)
            v2 = row.get(cv2)
            if v1 is None or v2 is None:
                continue
            try:
                v1_f = float(v1)
                v2_f = float(v2)
            except (TypeError, ValueError):
                continue
            cv1_vals.append(v1_f)
            cv2_vals.append(v2_f)
            wts.append(float(row.get("prob", 0.0)))

        if cv1_vals and sum(wts) > 0.0:
            bw1 = default_bin_width(cv1)
            bw2 = args.bin_width2 if args.bin_width2 is not None else default_bin_width(cv2)
            
            min1, max1 = min(cv1_vals), max(cv1_vals)
            min2, max2 = min(cv2_vals), max(cv2_vals)
            
            nbins1 = max(1, int(math.ceil((max1 - min1) / bw1)))
            nbins2 = max(1, int(math.ceil((max2 - min2) / bw2)))
            
            # Create 2D histogram
            # bin_counts[i][j] where i is index for cv1, j for cv2
            bin_counts_2d = [[0.0 for _ in range(nbins2)] for _ in range(nbins1)]
            
            for v1, v2, w in zip(cv1_vals, cv2_vals, wts):
                i = int((v1 - min1) / bw1)
                j = int((v2 - min2) / bw2)
                
                # Clamp indices
                i = max(0, min(i, nbins1 - 1))
                j = max(0, min(j, nbins2 - 1))
                
                bin_counts_2d[i][j] += w
                
            total_w = sum(sum(row) for row in bin_counts_2d)
            
            if total_w > 0.0:
                # Prepare grid for plotting/saving
                x_centers = [min1 + (i + 0.5) * bw1 for i in range(nbins1)]
                y_centers = [min2 + (j + 0.5) * bw2 for j in range(nbins2)]
                
                # Compute FES grid
                fes_grid = []
                min_f = float('inf')
                
                kT = K_B_KCAL_PER_MOL_K * args.temperature
                
                for i in range(nbins1):
                    row_f = []
                    for j in range(nbins2):
                        p = bin_counts_2d[i][j] / total_w
                        if p > 0:
                            f = -kT * math.log(p)
                            if f < min_f:
                                min_f = f
                            row_f.append(f)
                        else:
                            row_f.append(float('inf'))
                    fes_grid.append(row_f)
                
                # Shift to zero
                if math.isfinite(min_f):
                    for i in range(nbins1):
                        for j in range(nbins2):
                            if math.isfinite(fes_grid[i][j]):
                                fes_grid[i][j] -= min_f
                                
                # Save 2D data
                fes2d_path = out_path.parent / (out_path.stem + f"_{cv1}_{cv2}_fes.csv")
                with fes2d_path.open("w", newline="") as pf:
                    pw = csv.writer(pf)
                    pw.writerow([f"{cv1}_center", f"{cv2}_center", "prob", "F_kcal_per_mol"])
                    for i in range(nbins1):
                        for j in range(nbins2):
                            p = bin_counts_2d[i][j] / total_w
                            pw.writerow([x_centers[i], y_centers[j], p, fes_grid[i][j]])
                            
                print(f"Wrote 2D FES for {cv1} vs {cv2} to {fes2d_path}")

                # Plot 2D FES
                if not args.no_plot:
                    try:
                        import matplotlib.pyplot as plt
                        import matplotlib as mpl
                        import numpy as np
                    except ImportError:
                        pass
                    else:
                        # Publication-quality styling similar to 1D plots
                        mpl.rcParams['font.family'] = 'sans-serif'
                        mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
                        mpl.rcParams['font.size'] = 12
                        mpl.rcParams['axes.linewidth'] = 1.5
                        mpl.rcParams['xtick.major.width'] = 1.5
                        mpl.rcParams['ytick.major.width'] = 1.5
                        mpl.rcParams['xtick.direction'] = 'out'
                        mpl.rcParams['ytick.direction'] = 'out'
                        mpl.rcParams['figure.dpi'] = 300
                        mpl.rcParams['savefig.bbox'] = 'tight'

                        fig, ax = plt.subplots(figsize=(5.5, 4.5))

                        # Prepare data for contourf
                        X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')
                        Z = np.array(fes_grid)
                        # Mask only invalid values; high energies will be shown at top of color scale.
                        Z_masked = np.ma.masked_invalid(Z)

                        finite_Z = Z[np.isfinite(Z)]
                        if finite_Z.size > 0:
                            if args.fes_max_kcal is not None:
                                z_max = float(args.fes_max_kcal)
                            else:
                                mean_F = float(np.mean(finite_Z))
                                std_F = float(np.std(finite_Z))
                                if std_F <= 0.0:
                                    z_max = float(np.max(finite_Z))
                                else:
                                    z_max = float(
                                        min(
                                            np.max(finite_Z),
                                            mean_F + args.fes_sigma_cutoff * std_F,
                                        )
                                    )
                                if z_max <= 0.0:
                                    z_max = float(np.max(finite_Z))
                        else:
                            z_max = 1.0

                        # Filled contours + line contours for better visual structure
                        levels = np.linspace(0.0, z_max, 21)
                        cf = ax.contourf(X, Y, Z_masked, levels=levels, cmap='viridis')
                        ax.contour(X, Y, Z_masked, levels=levels[::4], colors='k', linewidths=0.4)

                        cbar = fig.colorbar(cf, ax=ax)
                        cbar.set_label('Free energy (kcal/mol)', fontsize=11)
                        cbar.ax.tick_params(labelsize=9)

                        ax.set_xlabel(cv1, fontsize=12, fontweight='bold')
                        ax.set_ylabel(cv2, fontsize=12, fontweight='bold')
                        ax.set_title(f"2D FES: {cv1} vs {cv2}\n(T = {args.temperature:.1f} K)", fontsize=13)

                        # Mark structural reference point if available.
                        ref_x_2d: Optional[float] = None
                        ref_y_2d: Optional[float] = None

                        # Helper to decode torsion index from column name.
                        def _torsion_index_from_col(col: str) -> Optional[int]:
                            if col.startswith("torsion") and col.endswith("_deg") and not col.startswith("d_torsion"):
                                num_str = col[len("torsion") : -len("_deg")]
                                try:
                                    return int(num_str)
                                except ValueError:
                                    return None
                            return None

                        if torsion_defs and "ref_torsions_deg" in locals():
                            idx1 = _torsion_index_from_col(cv1)
                            idx2 = _torsion_index_from_col(cv2)
                            if idx1 is not None and 1 <= idx1 <= len(ref_torsions_deg):
                                ref_x_2d = ref_torsions_deg[idx1 - 1]
                            if idx2 is not None and 1 <= idx2 <= len(ref_torsions_deg):
                                ref_y_2d = ref_torsions_deg[idx2 - 1]

                        # Distance-like CVs use 0 as the reference.
                        if ref_x_2d is None and (
                            cv1 in ("torsion_dist", "rmsd") or (cv1.startswith("d_torsion") and cv1.endswith("_deg"))
                        ):
                            ref_x_2d = 0.0
                        if ref_y_2d is None and (
                            cv2 in ("torsion_dist", "rmsd") or (cv2.startswith("d_torsion") and cv2.endswith("_deg"))
                        ):
                            ref_y_2d = 0.0

                        if ref_x_2d is not None and ref_y_2d is not None:
                            ax.plot(
                                ref_x_2d,
                                ref_y_2d,
                                marker="*",
                                color="red",
                                markersize=10,
                                markeredgecolor="k",
                                zorder=5,
                                label="structural ref",
                            )
                            ax.legend(frameon=False, fontsize=9, loc="upper right")

                        # Subtle grid, but keep background clean
                        ax.grid(True, linestyle=':', alpha=0.4)
                        ax.tick_params(axis='both', which='major', labelsize=10)

                        ax.set_xlim(min1, max1)
                        ax.set_ylim(min2, max2)

                        fig.tight_layout()
                        plot_path = fes2d_path.with_suffix(".png")
                        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
                        plt.close(fig)

    if ref_summary is not None:
        print(
            f"Reference conformer index: {ref_summary['ref_index']}\n"
            f"  E_ref (Eh)          : {ref_summary['E_ref_Eh']:.12f}\n"
            f"  E_ref - E_min (kcal): {ref_summary['E_ref_rel_min_kcal']:.3f}\n"
            f"  P_ref (Boltzmann)   : {ref_summary['P_ref']:.4e}\n"
            f"  ΔG_conf(ref) (kcal) : {ref_summary['dG_conf_ref_kcal']:.3f}"
        )
    else:
        print("No reference index specified; per-conformer probabilities written only.")

    if pn_val is not None:
        print(f"PNear (λ={args.pn_lambda:.2f} Å): {pn_val:.4f}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
