#!/usr/bin/env python3
"""
Brute-force scan of 1D/2D torsional landscapes with MMFF94s minimization.

High-level idea (mining-minima style):

- Take a single reference conformer (SDF with one molecule).
- Specify one or two torsions (1-based atom indices i,j,k,l).
- Enumerate a grid of target torsion values (e.g. every 30 degrees).
- For each grid point:
    - Copy the reference coordinates.
    - Set the specified torsion(s) to the target angle(s).
    - Minimize with MMFF94s.
- Write all minimized conformers into a multi-record SDF, annotating:
    - torsion1_target_deg, torsion1_final_deg, ...
    - grid indices, etc.

The resulting SDF can then be fed into the usual
`gxtb_optimize_conformers.py --mode xtb_opt_gxtb_sp` workflow, followed by
`gxtb_conformer_thermo.py` for a mining-minima style analysis.
"""

import argparse
from dataclasses import dataclass
import math
import random
from pathlib import Path
from typing import List, Tuple, Optional

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Geometry import Point3D


@dataclass
class TorsionDef:
    """One torsion definition and its angle grid."""

    a1: int
    a2: int
    a3: int
    a4: int
    angles: List[float]  # degrees


def parse_torsion_spec(spec: str) -> Tuple[int, int, int, int]:
    """
    Parse a torsion spec 'i,j,k,l' with 1-based indices.
    Returns 0-based indices (a1, a2, a3, a4).
    """
    parts = spec.replace(" ", "").split(",")
    if len(parts) != 4:
        raise ValueError(f"Invalid torsion spec '{spec}', expected 'i,j,k,l'")
    try:
        a1, a2, a3, a4 = [int(x) - 1 for x in parts]
    except ValueError:
        raise ValueError(f"Invalid torsion spec '{spec}', indices must be integers")
    return a1, a2, a3, a4


def parse_grid_spec(grid: str, default_step: float) -> List[float]:
    """
    Parse a grid specification.

    Accepted forms:
      - "step"                -> 0..360 (exclusive) in increments of step
      - "start:end:step"      -> start..end (exclusive) in increments of step

    Angles are in degrees.
    """
    text = grid.strip()
    if ":" not in text:
        try:
            step = float(text)
        except ValueError:
            raise ValueError(f"Invalid grid spec '{grid}', expected float or 'start:end:step'")
        return list(_frange(0.0, 360.0, step))

    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid grid spec '{grid}', expected 'start:end:step'")
    try:
        start = float(parts[0])
        end = float(parts[1])
        step = float(parts[2])
    except ValueError:
        raise ValueError(f"Invalid grid spec '{grid}', expected numeric start:end:step")
    if step <= 0.0:
        raise ValueError(f"Invalid grid spec '{grid}', step must be > 0")
    return list(_frange(start, end, step))


def _frange(start: float, end: float, step: float):
    """Range for floats with inclusive start and exclusive end."""
    x = start
    # Avoid infinite loops due to rounding.
    n_max = int(math.ceil((end - start) / step)) + 1
    for _ in range(n_max):
        if x >= end:
            break
        yield x
        x += step


def generate_torsion_grid(tdefs: List[TorsionDef]) -> List[Tuple[float, ...]]:
    """
    Generate the cartesian product of torsion angle grids.
    Returns list of tuples (ang1, ang2, ...) in degrees.
    """
    if not tdefs:
        return []
    # Simple recursive cartesian product.
    grids: List[List[float]] = [td.angles for td in tdefs]

    def _rec(idx: int, prefix: Tuple[float, ...], out: List[Tuple[float, ...]]):
        if idx == len(grids):
            out.append(prefix)
            return
        for ang in grids[idx]:
            _rec(idx + 1, prefix + (ang,), out)

    result: List[Tuple[float, ...]] = []
    _rec(0, tuple(), result)
    return result


def set_torsions_and_minimize(
    mol: Chem.Mol,
    torsions: List[TorsionDef],
    angles: Tuple[float, ...],
    mmff_max_iter: int,
    coord_jitter: float = 0.0,
    torsion_tol_deg: float = 1.0,
    torsion_force: float = 100.0,
    conf_id: int = 0,
    allow_ring_torsions: bool = False,
) -> Tuple[Chem.Mol, List[float]]:
    """
    Take a molecule with a single conformer, set the torsions to target angles,
    minimize with MMFF94, and return:
      - a copy of the molecule with minimized coordinates,
      - the final torsion angles (deg) after minimization.
    """
    if len(torsions) != len(angles):
        raise ValueError("Number of torsion defs and angle sets do not match")

    # Work in-place on the requested conformer to avoid large memory blow-ups
    # when scanning many grid points and generating ensembles per point.
    m = mol
    conf = m.GetConformer(conf_id)

    # Optionally randomize coordinates slightly to generate an ensemble
    # of distinct starting structures around each grid point.
    if coord_jitter > 0.0:
        for idx in range(conf.GetNumAtoms()):
            p = conf.GetAtomPosition(idx)
            dx = random.gauss(0.0, coord_jitter)
            dy = random.gauss(0.0, coord_jitter)
            dz = random.gauss(0.0, coord_jitter)
            conf.SetAtomPosition(idx, Point3D(p.x + dx, p.y + dy, p.z + dz))

    for td, ang in zip(torsions, angles):
        # RDKit cannot "set" a dihedral by rotating a bond that is in a ring.
        # We can still *restrain* that dihedral during minimization, but we
        # must skip the explicit SetDihedralDeg step.
        bond = m.GetBondBetweenAtoms(td.a2, td.a3)
        if bond is not None and bond.IsInRing():
            if not allow_ring_torsions:
                raise ValueError(
                    f"Central bond (j,k)=({td.a2+1},{td.a3+1}) is in a ring; RDKit cannot set "
                    f"dihedral for torsion ({td.a1+1},{td.a2+1},{td.a3+1},{td.a4+1}). "
                    "Pick a torsion whose central bond is not in a ring, or pass --allow-ring-torsions "
                    "to rely on forcefield restraints only."
                )
            # Skip explicit set; minimizer restraints (added below) may still drive it.
            continue
        rdMolTransforms.SetDihedralDeg(conf, td.a1, td.a2, td.a3, td.a4, ang)

    # Minimize with MMFF94s.
    props = AllChem.MMFFGetMoleculeProperties(m, mmffVariant="MMFF94s")
    if props is not None:
        ff = AllChem.MMFFGetMoleculeForceField(m, props, confId=conf.GetId())
        if ff is not None:
            # Constrain torsions near the target angles during minimization if supported.
            minmax = []
            for ang in angles:
                minmax.append((ang - torsion_tol_deg, ang + torsion_tol_deg))

            for td, (amin, amax) in zip(torsions, minmax):
                _add_torsion_constraint(
                    ff,
                    td.a1,
                    td.a2,
                    td.a3,
                    td.a4,
                    amin,
                    amax,
                    torsion_force,
                )
            ff.Initialize()
            ff.Minimize(maxIts=mmff_max_iter)

    # Measure final torsions.
    final_torsions: List[float] = []
    for td in torsions:
        ang = rdMolTransforms.GetDihedralDeg(conf, td.a1, td.a2, td.a3, td.a4)
        final_torsions.append(ang)

    return m, final_torsions


def _add_torsion_constraint(
    ff,
    a1: int,
    a2: int,
    a3: int,
    a4: int,
    amin_deg: float,
    amax_deg: float,
    force_const: float,
) -> None:
    """
    Add a torsion constraint to an RDKit ForceField object if possible.

    Tries, in order:
      - AllChem.MMFFAddTorsionConstraint
      - AllChem.UFFAddTorsionConstraint

    If none are available, does nothing (minimization will not strictly enforce the torsion).
    """
    for helper_name in ("MMFFAddTorsionConstraint", "UFFAddTorsionConstraint"):
        helper = getattr(AllChem, helper_name, None)
        if helper is None:
            continue
        # Most common signature:
        #   helper(ff, i, j, k, l, relative, minDeg, maxDeg, forceConstant)
        try:
            helper(ff, a1, a2, a3, a4, False, float(amin_deg), float(amax_deg), float(force_const))
            return
        except TypeError:
            pass
        # Alternate signature without 'relative'
        try:
            helper(ff, a1, a2, a3, a4, float(amin_deg), float(amax_deg), float(force_const))
            return
        except TypeError:
            pass
    # No supported constraint helper found; leave unconstrained.


def _embed_conformers_etkdg(
    mol: Chem.Mol,
    n_confs: int,
    seed: int,
    prune_rms: float,
    num_threads: int,
) -> List[int]:
    """Embed multiple conformers using (E)TKDG, returning conformer ids."""
    # Prefer newer ETKDG versions if available.
    params = None
    for name in ("ETKDGv3", "ETKDGv2", "ETKDG"):
        fn = getattr(AllChem, name, None)
        if fn is not None:
            try:
                params = fn()
                break
            except Exception:
                params = None
    if params is not None:
        try:
            params.randomSeed = int(seed)
        except Exception:
            pass
        try:
            params.pruneRmsThresh = float(prune_rms)
        except Exception:
            pass
        try:
            params.numThreads = int(num_threads)
        except Exception:
            pass
        cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=int(n_confs), params=params))
        return [int(x) for x in cids]

    # Fallback: older signature
    cids = list(
        AllChem.EmbedMultipleConfs(
            mol,
            numConfs=int(n_confs),
            randomSeed=int(seed),
            pruneRmsThresh=float(prune_rms),
            numThreads=int(num_threads),
        )
    )
    return [int(x) for x in cids]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Brute-force scan 1D/2D torsional landscapes by setting torsions on a reference "
            "geometry and minimizing with MMFF94s."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # 1D scan of torsion 19,20,27,28 every 30 degrees\n"
            "  scan_torsional_landscape.py ref.sdf \\\n"
            "    --torsion 19,20,27,28 --grid 0:360:30 --output 1d_scan.sdf\n\n"
            "  # 2D scan of torsions (19,20,27,28) and (19,20,27,29)\n"
            "  scan_torsional_landscape.py ref.sdf \\\n"
            "    --torsion 19,20,27,28 --grid 0:360:30 \\\n"
            "    --torsion 19,20,27,29 --grid 0:360:30 \\\n"
            "    --output 2d_scan.sdf\n"
        ),
    )
    parser.add_argument(
        "sdf",
        help="Input SDF containing a single reference conformer (heavy atoms and hydrogens).",
    )
    parser.add_argument(
        "--torsion",
        action="append",
        default=[],
        help="Torsion definition as 'i,j,k,l' (1-based atom indices). "
             "Can be given once (1D) or twice (2D).",
    )
    parser.add_argument(
        "--grid",
        action="append",
        default=[],
        help=(
            "Angle grid specification per torsion. Must be given the same number of times "
            "as --torsion. Forms: 'step' (0..360 step) or 'start:end:step', all in degrees."
        ),
    )
    parser.add_argument(
        "--mmff-max-iter",
        type=int,
        default=200,
        help="Maximum MMFF94s iterations per minimized conformer. [default: %(default)s]",
    )
    parser.add_argument(
        "--confs-per-grid",
        type=int,
        default=1,
        help="Number of MMFF-minimized conformers to generate per torsion grid point "
             "(i.e., ensemble size per fixed torsion value). [default: %(default)s]",
    )
    parser.add_argument(
        "--ensemble-method",
        choices=["etkdg", "jitter"],
        default="etkdg",
        help="How to generate conformers within each grid point before MMFF94s minimization. "
             "'etkdg' uses RDKit distance-geometry embedding; 'jitter' perturbs the reference "
             "coordinates. [default: %(default)s]",
    )
    parser.add_argument(
        "--embed-prune-rms",
        type=float,
        default=0.25,
        help="Prune threshold (Å) used by ETKDG embedding to enforce diversity. [default: %(default)s]",
    )
    parser.add_argument(
        "--embed-seed",
        type=int,
        default=0,
        help="Random seed for ETKDG embedding. Use 0 for nondeterministic. [default: %(default)s]",
    )
    parser.add_argument(
        "--embed-threads",
        type=int,
        default=0,
        help="Number of threads for ETKDG embedding (0 lets RDKit choose). [default: %(default)s]",
    )
    parser.add_argument(
        "--coord-jitter",
        type=float,
        default=0.2,
        help="Standard deviation (Å) of per-atom Gaussian coordinate jitter applied "
             "before minimization, used to decorrelate conformers within each grid point "
             "when --confs-per-grid > 1. Set to 0.0 to disable. [default: %(default)s]",
    )
    parser.add_argument(
        "--torsion-tol-deg",
        type=float,
        default=1.0,
        help="Half-width (deg) of the torsion constraint window during MMFF94s minimization. "
             "Smaller enforces a tighter 'fixed torsion'. [default: %(default)s]",
    )
    parser.add_argument(
        "--torsion-force",
        type=float,
        default=100.0,
        help="Force constant used for torsion constraints during MMFF94s minimization. [default: %(default)s]",
    )
    parser.add_argument(
        "--allow-ring-torsions",
        action="store_true",
        help="Allow torsion definitions whose central bond (j-k) is in a ring. "
             "RDKit cannot explicitly set such dihedrals; the script will instead "
             "rely on torsion restraints during MMFF94s minimization.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output multi-record SDF with all minimized torsion scan conformers.",
    )
    parser.add_argument(
        "--max-total-confs",
        type=int,
        default=200000,
        help="Safety limit: maximum total conformers to write (grid_points * confs_per_grid). "
             "[default: %(default)s]",
    )

    args = parser.parse_args(argv)

    sdf_path = Path(args.sdf).resolve()
    if not sdf_path.is_file():
        raise SystemExit(f"ERROR: input SDF not found: {sdf_path}")

    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mols = [m for m in suppl if m is not None]
    if not mols:
        raise SystemExit(f"ERROR: no valid molecules in {sdf_path}")
    if len(mols) != 1:
        raise SystemExit(f"ERROR: expected a single-molecule SDF, found {len(mols)} molecules")

    ref_mol = mols[0]
    if ref_mol.GetNumConformers() == 0:
        raise SystemExit("ERROR: reference molecule has no 3D conformer.")

    if not args.torsion:
        raise SystemExit("ERROR: at least one --torsion must be provided.")
    if len(args.torsion) > 2:
        raise SystemExit("ERROR: currently only 1D or 2D torsion scans are supported (max 2 torsions).")

    if args.grid and len(args.grid) != len(args.torsion):
        raise SystemExit("ERROR: number of --grid specs must match number of --torsion entries.")

    # Build torsion defs.
    tdefs: List[TorsionDef] = []
    for i, tors in enumerate(args.torsion):
        a1, a2, a3, a4 = parse_torsion_spec(tors)
        grid_spec = args.grid[i] if i < len(args.grid) else "30.0"
        angles = parse_grid_spec(grid_spec, default_step=30.0)
        if not angles:
            raise SystemExit(f"ERROR: empty grid for torsion '{tors}'")
        tdefs.append(TorsionDef(a1=a1, a2=a2, a3=a3, a4=a4, angles=angles))

    grid_points = generate_torsion_grid(tdefs)
    if not grid_points:
        raise SystemExit("ERROR: no grid points generated; check --grid specifications.")
    total_to_write = len(grid_points) * int(args.confs_per_grid)
    if total_to_write > int(args.max_total_confs):
        raise SystemExit(
            f"ERROR: scan would generate {total_to_write} conformers "
            f"({len(grid_points)} grid points * {args.confs_per_grid} confs/grid), "
            f"exceeding --max-total-confs={args.max_total_confs}. "
            "Increase --max-total-confs or coarsen the grid / reduce confs per grid."
        )

    out_path = Path(args.output).resolve()
    writer = Chem.SDWriter(str(out_path))

    print(f"Reference SDF   : {sdf_path}")
    print(f"Output SDF      : {out_path}")
    print(f"Torsions        : {len(tdefs)}")
    for idx, td in enumerate(tdefs, start=1):
        print(
            f"  torsion{idx}: (a1={td.a1+1}, a2={td.a2+1}, a3={td.a3+1}, a4={td.a4+1}), "
            f"{len(td.angles)} angles"
        )
    print(f"Total grid points: {len(grid_points)}")

    conf_counter = 0
    for grid_idx, angles in enumerate(grid_points, start=1):
        # Create a working molecule with multiple conformers for this grid point.
        work = Chem.Mol(ref_mol)
        # Remove any existing conformers (EmbedMultipleConfs appends).
        for cid in list(range(work.GetNumConformers())):
            work.RemoveConformer(cid)

        if args.confs_per_grid == 1:
            # Just copy the reference conformer into work.
            work.AddConformer(Chem.Conformer(ref_mol.GetConformer()), assignId=True)
            conf_ids = [0]
        else:
            if args.ensemble_method == "etkdg":
                conf_ids = _embed_conformers_etkdg(
                    work,
                    n_confs=args.confs_per_grid,
                    seed=args.embed_seed,
                    prune_rms=args.embed_prune_rms,
                    num_threads=args.embed_threads,
                )
                if not conf_ids:
                    raise SystemExit(
                        f"ERROR: ETKDG embedding failed at grid_index={grid_idx}. "
                        "Try increasing --embed-seed, reducing --embed-prune-rms, or using --ensemble-method jitter."
                    )
            else:
                # Jitter method: create conformers by copying the reference and jittering.
                conf_ids = []
                for rep in range(args.confs_per_grid):
                    c = Chem.Conformer(ref_mol.GetConformer())
                    work.AddConformer(c, assignId=True)
                    conf_ids.append(work.GetNumConformers() - 1)

        # For each conformer, set torsions + minimize with constraints.
        for rep, cid in enumerate(conf_ids, start=1):
            coord_jitter = 0.0
            if args.confs_per_grid > 1 and args.ensemble_method == "jitter":
                coord_jitter = args.coord_jitter
            m_min, final_tors = set_torsions_and_minimize(
                work,
                torsions=tdefs,
                angles=angles,
                mmff_max_iter=args.mmff_max_iter,
                coord_jitter=coord_jitter,
                torsion_tol_deg=args.torsion_tol_deg,
                torsion_force=args.torsion_force,
                conf_id=cid,
                allow_ring_torsions=args.allow_ring_torsions,
            )
            # Annotate SD properties on the molecule, then write the specific conformer.
            for i, (target, final) in enumerate(zip(angles, final_tors), start=1):
                m_min.SetProp(f"torsion{i}_target_deg", f"{target:.3f}")
                m_min.SetProp(f"torsion{i}_final_deg", f"{final:.3f}")
            m_min.SetProp("grid_index", str(grid_idx))
            m_min.SetProp("rep_index", str(rep))
            writer.write(m_min, confId=cid)
            conf_counter += 1

    writer.close()
    print(f"Wrote {conf_counter} minimized conformers to {out_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


