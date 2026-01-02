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
import time
from pathlib import Path
from typing import List, Tuple, Optional

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdMolTransforms
from rdkit.Geometry import Point3D


class _Timers:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.totals = {}  # name -> seconds

    def add(self, name: str, dt: float) -> None:
        if not self.enabled:
            return
        self.totals[name] = self.totals.get(name, 0.0) + float(dt)

    def report(self, prefix: str = "") -> None:
        if not self.enabled:
            return
        if not self.totals:
            return
        items = sorted(self.totals.items(), key=lambda kv: kv[1], reverse=True)
        total = sum(v for _, v in items)
        print(prefix + "Timing summary:")
        for k, v in items:
            frac = (v / total * 100.0) if total > 0 else 0.0
            print(prefix + f"  - {k:22s}: {v:8.2f} s  ({frac:5.1f}%)")
        print(prefix + f"  - {'TOTAL':22s}: {total:8.2f} s")


class _Timer:
    def __init__(self, timers: _Timers, name: str):
        self.timers = timers
        self.name = name
        self.t0 = None

    def __enter__(self):
        if self.timers.enabled:
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.timers.enabled and self.t0 is not None:
            dt = time.perf_counter() - self.t0
            self.timers.add(self.name, dt)
        return False


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
) -> Tuple[Chem.Mol, List[float], Optional[float]]:
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
    mmff_e: Optional[float] = None
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
            if mmff_max_iter > 0:
                ff.Minimize(maxIts=mmff_max_iter)
            try:
                mmff_e = float(ff.CalcEnergy())
            except Exception:
                mmff_e = None

    # Measure final torsions.
    final_torsions: List[float] = []
    for td in torsions:
        ang = rdMolTransforms.GetDihedralDeg(conf, td.a1, td.a2, td.a3, td.a4)
        final_torsions.append(ang)

    return m, final_torsions, mmff_e


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


def _embed_until_n(
    mol: Chem.Mol,
    n_target: int,
    seed: int,
    prune_rms: float,
    num_threads: int,
    max_attempts: int = 5,
) -> List[int]:
    """
    Try to embed until we have at least n_target conformers.

    Notes:
    - RDKit's pruning (pruneRmsThresh) can aggressively collapse ensembles for large molecules.
      If we get too few conformers, we progressively relax pruning down to 0.0.
    """
    conf_ids: List[int] = []
    remaining = int(n_target)
    attempt = 0
    while remaining > 0 and attempt < max_attempts:
        attempt += 1
        # Relax pruning on subsequent attempts.
        pr = float(prune_rms) if attempt == 1 else 0.0
        # Vary seed across attempts (0 means nondeterministic, keep it 0).
        this_seed = int(seed)
        if this_seed != 0:
            this_seed = this_seed + attempt
        new_ids = _embed_conformers_etkdg(
            mol,
            n_confs=remaining,
            seed=this_seed,
            prune_rms=pr,
            num_threads=num_threads,
        )
        conf_ids.extend(new_ids)
        remaining = n_target - len(conf_ids)
    return conf_ids


def _get_heavy_atom_ids(mol: Chem.Mol) -> List[int]:
    """Return atom indices for heavy atoms (atomic number > 1)."""
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]


def _rmsd_noalign(mol: Chem.Mol, conf_id1: int, conf_id2: int, atom_ids: List[int]) -> float:
    """
    RMSD computed directly from coordinates, assuming both conformers are already
    in the same reference frame (pre-aligned).
    """
    c1 = mol.GetConformer(conf_id1)
    c2 = mol.GetConformer(conf_id2)
    s = 0.0
    for i in atom_ids:
        p1 = c1.GetAtomPosition(i)
        p2 = c2.GetAtomPosition(i)
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        dz = p1.z - p2.z
        s += dx * dx + dy * dy + dz * dz
    return math.sqrt(s / float(len(atom_ids)))


def _align_to_reference(
    mol: Chem.Mol,
    conf_ids: List[int],
    ref_cid: int,
    atom_ids: List[int],
) -> None:
    """Align each conformer in conf_ids to ref_cid (in-place)."""
    amap = [(i, i) for i in atom_ids]
    for cid in conf_ids:
        if cid == ref_cid:
            continue
        rdMolAlign.AlignMol(mol, mol, prbCid=cid, refCid=ref_cid, atomMap=amap)


def _select_diverse_subset(
    mol: Chem.Mol,
    conf_ids: List[int],
    energies: List[Optional[float]],
    k: int,
    atom_ids: Optional[List[int]] = None,
) -> List[int]:
    """
    Select k diverse conformers from conf_ids using greedy farthest-point sampling.

    Seed: lowest MMFF energy (if available), otherwise the first conformer.
    Criterion: maximize the minimum RMSD to the selected set (heavy-atom RMSD by default).
    """
    if k <= 0:
        return []
    if len(conf_ids) <= k:
        return list(conf_ids)
    if atom_ids is None:
        atom_ids = _get_heavy_atom_ids(mol)

    # Seed selection with lowest-energy conformer (ignore None energies).
    best_idx = 0
    best_e = None
    for i, e in enumerate(energies):
        if e is None:
            continue
        if best_e is None or e < best_e:
            best_e = e
            best_idx = i
    selected = [conf_ids[best_idx]]
    remaining = [cid for cid in conf_ids if cid != selected[0]]

    # Greedy farthest-point: repeatedly add the conformer that maximizes its distance
    # to the current selected set (distance = min RMSD to selected).
    #
    # NOTE: we assume conformers have been aligned to a common reference frame already,
    # so we use a fast no-alignment RMSD.
    while len(selected) < k and remaining:
        best_cid = None
        best_min_dist = -1.0
        for cid in remaining:
            min_d = float("inf")
            for s_cid in selected:
                d = _rmsd_noalign(mol, s_cid, cid, atom_ids=atom_ids)
                if d < min_d:
                    min_d = d
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_cid = cid
        if best_cid is None:
            break
        selected.append(best_cid)
        remaining.remove(best_cid)

    return selected


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
        default=0.0,
        help=(
            "Prune threshold (Å) used by ETKDG embedding to enforce diversity. "
            "Set to 0.0 to disable pruning (recommended if you want exactly "
            "--confs-per-grid conformers). [default: %(default)s]"
        ),
    )
    parser.add_argument(
        "--embed-seed",
        type=int,
        default=0,
        help="Random seed for ETKDG embedding. Use 0 for nondeterministic. [default: %(default)s]",
    )
    parser.add_argument(
        "--embed-oversample",
        type=int,
        default=1,
        help="Oversampling factor for ETKDG candidate conformers per grid point. "
             "The script will embed (confs_per_grid * embed_oversample) candidates, "
             "minimize them with torsion restraints, then select a diverse subset of "
             "size confs_per_grid. [default: %(default)s]",
    )
    parser.add_argument(
        "--embed-max-candidates",
        type=int,
        default=200,
        help="Safety limit: maximum number of ETKDG candidate conformers per grid point "
             "(after applying oversampling). [default: %(default)s]",
    )
    parser.add_argument(
        "--embed-threads",
        type=int,
        default=0,
        help="Number of threads for ETKDG embedding (0 lets RDKit choose). [default: %(default)s]",
    )
    parser.add_argument(
        "--candidate-mmff-iter",
        type=int,
        default=0,
        help="Number of MMFF94s minimization steps to apply to *candidates* before diversity selection. "
             "0 means do not minimize candidates (fastest). Only the selected conformers "
             "are always fully minimized with --mmff-max-iter. [default: %(default)s]",
    )
    parser.add_argument(
        "--selection-seed-lowE",
        type=int,
        default=2,
        help="Number of lowest-energy candidates to always include before filling the remainder "
             "by RMSD-based diversity selection. Energies come from MMFF94s after "
             "--candidate-mmff-iter steps (or just an energy evaluation if 0). [default: %(default)s]",
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
    parser.add_argument(
        "--no-timing",
        action="store_true",
        help="Disable printing timing breakdowns.",
    )

    args = parser.parse_args(argv)

    timers = _Timers(enabled=(not args.no_timing))

    with _Timer(timers, "setup+read_input"):
        sdf_path = Path(args.sdf).resolve()
    if not sdf_path.is_file():
        raise SystemExit(f"ERROR: input SDF not found: {sdf_path}")

    with _Timer(timers, "read_sdf"):
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

    with _Timer(timers, "parse_inputs"):
        # Build torsion defs.
        tdefs: List[TorsionDef] = []
        for i, tors in enumerate(args.torsion):
            a1, a2, a3, a4 = parse_torsion_spec(tors)
            grid_spec = args.grid[i] if i < len(args.grid) else "30.0"
            angles = parse_grid_spec(grid_spec, default_step=30.0)
            if not angles:
                raise SystemExit(f"ERROR: empty grid for torsion '{tors}'")
            tdefs.append(TorsionDef(a1=a1, a2=a2, a3=a3, a4=a4, angles=angles))

    with _Timer(timers, "build_grid"):
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
        grid_timers = _Timers(enabled=(not args.no_timing))
        # Create a working molecule with multiple conformers for this grid point.
        with _Timer(timers, "per_grid:setup_work"):
            work = Chem.Mol(ref_mol)
            # Remove any existing conformers (EmbedMultipleConfs appends).
            for cid in list(range(work.GetNumConformers())):
                work.RemoveConformer(cid)

        if args.confs_per_grid == 1:
            # Just copy the reference conformer into work.
            with _Timer(timers, "per_grid:seed_conformer"):
                work.AddConformer(Chem.Conformer(ref_mol.GetConformer()), assignId=True)
                conf_ids = [0]
        else:
            if args.ensemble_method == "etkdg":
                n_target = int(args.confs_per_grid)
                oversample = max(1, int(args.embed_oversample))
                n_cand = n_target * oversample
                if n_cand > int(args.embed_max_candidates):
                    raise SystemExit(
                        f"ERROR: requested {n_cand} ETKDG candidates per grid point "
                        f"({n_target} * oversample {oversample}) exceeds --embed-max-candidates={args.embed_max_candidates}. "
                        "Reduce --embed-oversample or increase --embed-max-candidates."
                    )

                with _Timer(timers, "per_grid:embed_etkdg"):
                    conf_ids = _embed_until_n(
                        work,
                        n_target=n_cand,
                        seed=int(args.embed_seed),
                        prune_rms=float(args.embed_prune_rms),
                        num_threads=int(args.embed_threads),
                    )
                if not conf_ids:
                    raise SystemExit(
                        f"ERROR: ETKDG embedding failed at grid_index={grid_idx}. "
                        "Try increasing --embed-seed, reducing --embed-prune-rms, or using --ensemble-method jitter."
                    )
                if len(conf_ids) < n_cand:
                    print(
                        f"WARNING: only embedded {len(conf_ids)}/{n_cand} candidates at grid_index={grid_idx}. "
                        "Consider --embed-prune-rms 0.0, increasing --embed-seed, or using --ensemble-method jitter.",
                    )
            else:
                # Jitter method: create conformers by copying the reference and jittering.
                with _Timer(timers, "per_grid:seed_jitter"):
                    conf_ids = []
                    for rep in range(args.confs_per_grid):
                        c = Chem.Conformer(ref_mol.GetConformer())
                        work.AddConformer(c, assignId=True)
                        conf_ids.append(work.GetNumConformers() - 1)

        # Candidate stage: score quickly (optional short minimization), then select a diverse subset.
        cand_ids: List[int] = list(conf_ids)
        cand_energies: List[Optional[float]] = []
        with _Timer(timers, "per_grid:candidate_score"):
            for cid in cand_ids:
                coord_jitter = 0.0
                if args.confs_per_grid > 1 and args.ensemble_method == "jitter":
                    coord_jitter = args.coord_jitter
                _, _, mmff_e = set_torsions_and_minimize(
                    work,
                    torsions=tdefs,
                    angles=angles,
                    mmff_max_iter=int(args.candidate_mmff_iter),
                    coord_jitter=coord_jitter,
                    torsion_tol_deg=args.torsion_tol_deg,
                    torsion_force=args.torsion_force,
                    conf_id=cid,
                    allow_ring_torsions=args.allow_ring_torsions,
                )
                cand_energies.append(mmff_e)

        # Select conformers to keep:
        # - Always include N_lowE lowest-energy candidates (if energies available)
        # - Fill the rest by RMSD-diversity from the remaining pool
        n_keep = int(args.confs_per_grid)
        if len(cand_ids) <= n_keep:
            selected = list(cand_ids)
        else:
            # Pick low-energy seeds
            idxs = list(range(len(cand_ids)))
            idxs.sort(key=lambda i: (cand_energies[i] is None, cand_energies[i] if cand_energies[i] is not None else 0.0))
            n_seed = max(0, min(int(args.selection_seed_lowE), n_keep))
            seed_ids = [cand_ids[i] for i in idxs[:n_seed]]

            # Align all candidates to the best available seed (or first candidate).
            ref_cid = seed_ids[0] if seed_ids else cand_ids[0]
            atom_ids = _get_heavy_atom_ids(work)
            with _Timer(timers, "per_grid:align_candidates"):
                _align_to_reference(work, cand_ids, ref_cid, atom_ids)

            # Diversity-fill
            remaining_ids = [cid for cid in cand_ids if cid not in seed_ids]
            remaining_es = [cand_energies[cand_ids.index(cid)] for cid in remaining_ids]
            # Run farthest-point selection on remaining, but starting set is seed_ids.
            # We do this by temporarily selecting k' from remaining, using the first as seed
            # and then merging; to respect the existing seed set, we greedily add points
            # based on min-distance to (seed_ids + already picked).
            selected = list(seed_ids)
            with _Timer(timers, "per_grid:diversity_select"):
                while len(selected) < n_keep and remaining_ids:
                best_cid = None
                best_min_dist = -1.0
                for cid in remaining_ids:
                    min_d = float("inf")
                    for s_cid in selected:
                        d = _rmsd_noalign(work, s_cid, cid, atom_ids=atom_ids)
                        if d < min_d:
                            min_d = d
                    if min_d > best_min_dist:
                        best_min_dist = min_d
                        best_cid = cid
                    if best_cid is None:
                        break
                    selected.append(best_cid)
                    remaining_ids.remove(best_cid)

        # Final stage: fully minimize only the selected conformers, then write.
        with _Timer(timers, "per_grid:final_minimize+write"):
            for rep, cid in enumerate(selected, start=1):
                m_min, final_tors, mmff_e = set_torsions_and_minimize(
                    work,
                    torsions=tdefs,
                    angles=angles,
                    mmff_max_iter=int(args.mmff_max_iter),
                    coord_jitter=0.0,
                    torsion_tol_deg=args.torsion_tol_deg,
                    torsion_force=args.torsion_force,
                    conf_id=cid,
                    allow_ring_torsions=args.allow_ring_torsions,
                )
                for i, (target, final) in enumerate(zip(angles, final_tors), start=1):
                    work.SetProp(f"torsion{i}_target_deg", f"{target:.3f}")
                    work.SetProp(f"torsion{i}_final_deg", f"{final:.3f}")
                if mmff_e is not None:
                    work.SetProp("mmff94s_energy", f"{mmff_e:.8f}")
                work.SetProp("grid_index", str(grid_idx))
                work.SetProp("rep_index", str(rep))
                writer.write(work, confId=cid)
                conf_counter += 1

    writer.close()
    print(f"Wrote {conf_counter} minimized conformers to {out_path}")
    timers.report(prefix="")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


