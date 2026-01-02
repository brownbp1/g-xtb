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
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

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

    rmsd_map: Optional[Dict[int, float]] = None
    if args.rmsd_csv:
        rmsd_map = read_rmsd(
            Path(args.rmsd_csv).resolve(),
            index_column=args.index_column,
            rmsd_column=args.rmsd_column,
        )

    pn_lambda = args.pn_lambda if rmsd_map is not None else None

    per_conf, ref_summary, pn_val = compute_conformer_free_energies(
        entries,
        temperature=args.temperature,
        ref_index=args.ref_index,
        rmsd_map=rmsd_map,
        pn_lambda=pn_lambda,
    )

    if args.output:
        out_path = Path(args.output).resolve()
    else:
        out_path = csv_path.parent / "conformer_free_energies.csv"

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
    if args.ref_index is not None:
        fieldnames.append("dG_vs_ref_kcal")

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_conf:
            writer.writerow(row)

    print(f"Wrote conformer free energies to {out_path}")

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


