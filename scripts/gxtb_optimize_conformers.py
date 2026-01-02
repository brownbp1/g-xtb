#!/usr/bin/env python3
"""
Batch conformer workflow for many conformers in a multi-conformer SDF.

Workflow:
  1. Read all records from an SDF (typically conformers of the same molecule).
  2. For each record, write an XYZ file and create a working directory.
  3. Run one of the supported modes (see below).
  4. Parse final energies and write a summary CSV.

Modes (select with --mode):

  - xtb_opt_gxtb_sp (recommended for large batches):
      * Optimize geometry with xtb (GFN2-xTB, analytic gradients)
      * Run gxtb single-point energy (g-xTB) on the optimized geometry

  - gxtb_opt:
      * Geometry optimization using xtb with gxtb as a numerical-gradient driver:
          xtb input.xyz --driver "gxtb -grad -c xtbdriver.xyz" --opt
      * This can be very slow/fragile for large batches.

  - gxtb_sp:
      * gxtb single-point energy on the input geometry (no optimization)

Energy provenance + units:
  - energies.csv: energy_Eh is the gxtb (g-xTB) TOTAL ENERGY in Hartree (Eh)
  - xtb_opt.out: xtb TOTAL ENERGY is GFN2-xTB, in Hartree (Eh)
  - gxtb_sp.out + energy: gxtb energy is g-xTB, in Hartree (Eh)
  - IMPORTANT: absolute energies between xtb (GFN2-xTB) and gxtb (g-xTB) are on
    different absolute scales (different energy zero; gxtb includes large core
    increments). Compare *relative* energies within the same method.

Requirements:
  - Python 3.8+
  - RDKit (for SDF reading and XYZ writing)
  - xtb in $PATH (or specify via --xtb)
  - gxtb available (from this repo or in $PATH)
  - GXTBHOME set to the g-xTB parameter directory, or let this script
    auto-set it to "<repo_root>/parameters".

Example (single node, 8 parallel workers):
  python scripts/gxtb_optimize_conformers.py \\
      input_conformers.sdf \\
      --outdir gxtb_runs \\
      --workers 8

This will create:
  gxtb_runs/
    conf_00001/
      input.xyz
      xtb_opt.out        # (GFN2-xTB) optimization log
      xtbopt.xyz         # optimized geometry (Å)
      gxtb_sp.out        # (g-xTB) single-point log
      energy             # (g-xTB) TURBOMOLE-like $energy file (Eh)
    conf_00002/
      ...
  gxtb_runs/energies.csv
"""

import argparse
import csv
import os
import signal
import shutil
import sys
import textwrap
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple, List

import subprocess

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - runtime environment dependent
    sys.stderr.write(
        "ERROR: RDKit is required to run this script but could not be imported.\n"
        "Install via e.g. conda:\n"
        "  conda install -c conda-forge rdkit\n"
    )
    raise


def write_xyz(mol: "Chem.Mol", path: Path) -> None:
    """Write an RDKit Mol (with a single conformer) to an XYZ file."""
    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()
    coords = conf.GetPositions()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]

    with path.open("w") as f:
        f.write(f"{n_atoms}\n")
        title = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        f.write(f"{title}\n")
        for sym, (x, y, z) in zip(symbols, coords):
            f.write(f"{sym:2s} {x:16.8f} {y:16.8f} {z:16.8f}\n")


def parse_final_energy(workdir: Path, stdout: str) -> Optional[float]:
    """
    Extract the final total energy in Eh.

    Strategy:
      - Prefer the 'energy' file written by xtb if present.
      - Otherwise, take the last float on the last line containing 'TOTAL ENERGY'
        from the captured stdout.
    """
    energy_file = workdir / "energy"
    if energy_file.is_file():
        try:
            lines = energy_file.read_text().strip().splitlines()
            # gxtb writes a TURBOMOLE-like block:
            #   $energy
            #     1   -1.2345 ...
            #   $end
            for line in lines:
                s = line.strip()
                if not s or s.startswith("$"):
                    continue
                parts = s.replace("D", "E").split()
                if len(parts) >= 2:
                    # first token is typically an integer "cycle"
                    try:
                        int(parts[0])
                        return float(parts[1])
                    except Exception:
                        pass
                # fallback: first parseable float
                for tok in parts:
                    try:
                        return float(tok)
                    except ValueError:
                        continue
        except Exception:
            pass

    last_val: Optional[float] = None
    for line in stdout.splitlines():
        up = line.upper()
        if "TOTAL ENERGY" in up or up.strip().startswith("TOTAL"):
            parts = line.replace("D", "E").split()
            for token in parts:
                try:
                    val = float(token)
                    last_val = val
                except ValueError:
                    continue
    return last_val


def preflight_check(
    gxtb_bin: str,
    gxthome: Optional[str],
    xtb_bin: str,
) -> None:
    """
    Fail fast if the runtime environment is not sane.

    In practice, gxtb failures with:
      "forrtl: severe (24): end-of-file during read ... file .../.gxtb"
    almost always mean the parameter file is missing/truncated/corrupt or the
    wrong file is being used for the running binary.
    """
    env = os.environ.copy()
    if gxthome is not None:
        env["GXTBHOME"] = gxthome

    # Basic existence checks for parameter files (if GXTBHOME is known).
    if gxthome is not None:
        p = Path(gxthome)
        required = [p / ".gxtb", p / ".eeq", p / ".basisq"]
        missing = [str(x) for x in required if not x.is_file()]
        if missing:
            raise RuntimeError(
                "GXTBHOME is set but required parameter files are missing:\n"
                + "\n".join(f"  - {m}" for m in missing)
                + "\nMake sure GXTBHOME points to a directory containing .gxtb, .eeq, .basisq."
            )

        # Heuristic size sanity checks (helps detect partial copies).
        too_small = []
        for x in required:
            try:
                if x.stat().st_size < 4096:
                    too_small.append(f"{x} ({x.stat().st_size} bytes)")
            except FileNotFoundError:
                pass
        if too_small:
            raise RuntimeError(
                "GXTBHOME parameter files look suspiciously small (possibly truncated or wrong files):\n"
                + "\n".join(f"  - {t}" for t in too_small)
            )

        # Catch a common footgun: an empty ".gxt" file in GXTBHOME can be
        # mistakenly picked up, leading to "EOF during read".
        #
        # Some gxtb builds will also auto-create an empty ".gxt" if they end up
        # with a truncated parameter filename. Since it's empty, it's safe to delete.
        gxt = p / ".gxt"
        if gxt.is_file() and gxt.stat().st_size == 0:
            try:
                gxt.unlink()
            except Exception as exc:
                raise RuntimeError(
                    f"Found an empty parameter file {gxt} but could not delete it ({exc}).\n"
                    f"Delete it manually: rm -f {gxt}\n"
                )

    # Ensure gxtb at least starts with a trivial input.
    tmp = Path.cwd() / ".gxtb_preflight_tmp"
    tmp.mkdir(exist_ok=True)
    try:
        h2 = tmp / "h2.xyz"
        h2.write_text("2\nH2\nH 0 0 0\nH 0 0 0.74\n")
        # IMPORTANT:
        # Do NOT rely on gxtb's implicit parameter file selection. Some builds
        # default to ".gxt" instead of ".gxtb". Also avoid long absolute paths.
        cmd = [gxtb_bin, "-c", "h2.xyz"]
        if gxthome is not None:
            # Create a short symlink inside the temp dir so the parameter paths
            # stay short and never get truncated.
            params_link = tmp / "params"
            if params_link.is_symlink() or params_link.exists():
                try:
                    if params_link.resolve() != Path(gxthome).resolve():
                        params_link.unlink()
                except Exception:
                    # If we can't resolve/unlink, continue and hope it points correctly.
                    pass
            if not params_link.exists():
                params_link.symlink_to(Path(gxthome))
            cmd += [
                "-p",
                "params/.gxtb",
                "-e",
                "params/.eeq",
                "-b",
                "params/.basisq",
            ]
        res = subprocess.run(
            cmd,
            cwd=str(tmp),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            msg = res.stdout.splitlines()[-40:]
            raise RuntimeError(
                "Preflight gxtb single-point failed. This usually means a broken/mismatched parameter setup.\n"
                f"gxtb={gxtb_bin}\n"
                f"GXTBHOME={env.get('GXTBHOME','(unset)')}\n"
                f"exit_code={res.returncode}\n"
                "Last output lines:\n" + "\n".join(msg)
            )
    finally:
        # Keep the directory if the user wants to inspect it.
        pass


def run_xtb_on_xyz(
    idx: int,
    workdir: Path,
    xtb_bin: str,
    gxtb_bin: str,
    gxthome: Optional[str],
    use_driver_wrapper: bool,
    mode: str,
    xtb_opt_level: str,
    charge: Optional[int],
    uhf: Optional[int],
) -> Tuple[int, Optional[float], str]:
    """
    Run xtb+gxtb optimization in a given working directory.

    Returns (index, energy_Eh_or_None, status_string).
    """
    cwd = workdir
    input_xyz = cwd / "input.xyz"
    if not input_xyz.is_file():
        return idx, None, f"missing input.xyz in {cwd}"

    env = os.environ.copy()

    # Set GXTBHOME if requested / auto-detected.
    if gxthome is not None:
        env["GXTBHOME"] = gxthome

    # Avoid over-subscribing cores when we use multiple workers.
    env.setdefault("OMP_NUM_THREADS", "1")

    # If we know GXTBHOME, create a short per-run symlink to avoid any risk of
    # truncated parameter filenames in Fortran code.
    param_args: List[str] = []
    if gxthome is not None:
        params_link = cwd / "params"
        try:
            if params_link.is_symlink() or params_link.exists():
                # leave as is
                pass
            else:
                params_link.symlink_to(Path(gxthome))
        except Exception:
            # If symlink creation fails (e.g., on some filesystems), fall back to a short global symlink in /tmp.
            tmp_link = Path("/tmp") / f"gxtb_params_{abs(hash(gxthome)) % 10**9}"
            try:
                if not tmp_link.exists():
                    tmp_link.symlink_to(Path(gxthome))
                param_args = ["-p", str(tmp_link / ".gxtb"), "-e", str(tmp_link / ".eeq"), "-b", str(tmp_link / ".basisq")]
            except Exception:
                # Last resort: absolute paths (may be fragile on some builds).
                param_args = ["-p", str(Path(gxthome) / ".gxtb"), "-e", str(Path(gxthome) / ".eeq"), "-b", str(Path(gxthome) / ".basisq")]
        else:
            # Prefer short relative paths; gxtb resolves them against cwd.
            param_args = ["-p", "params/.gxtb", "-e", "params/.eeq", "-b", "params/.basisq"]

    # Optional control files (used by gxtb, and can also affect xtb).
    if charge is not None:
        (cwd / ".CHRG").write_text(f"{int(charge)}\n")
    if uhf is not None:
        (cwd / ".UHF").write_text(f"{int(uhf)}\n")

    if mode == "gxtb_opt":
        if use_driver_wrapper:
            # xtb's --driver interface can be fragile when passing a command string
            # containing spaces/args. A wrapper script avoids quoting/tokenization
            # issues and gives us a separate log for the driver itself.
            driver_sh = cwd / "driver.sh"
            driver_log = cwd / "gxtb_driver.out"
            driver_code = cwd / "gxtb_driver.exitcode"
            gxthome_line = f'export GXTBHOME="{gxthome}"\n' if gxthome is not None else ""

            driver_sh.write_text(
                "#!/usr/bin/env bash\n"
                "set -u\n"
                f"{gxthome_line}"
                'export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"\n'
                f'"{gxtb_bin}" -grad -c xtbdriver.xyz {" ".join(param_args)} > "{driver_log.name}" 2>&1\n'
                "code=$?\n"
                f'echo \"$code\" > "{driver_code.name}"\n'
                "exit $code\n"
            )
            driver_sh.chmod(0o755)

            cmd = [
                xtb_bin,
                str(input_xyz),
                "--driver",
                "./driver.sh",
                "--opt",
            ]
        else:
            cmd = [
                xtb_bin,
                str(input_xyz),
                "--driver",
                f"{gxtb_bin} -grad -c xtbdriver.xyz",
                "--opt",
            ]
    elif mode == "xtb_opt_gxtb_sp":
        # 1) Optimize with xtb (GFN2-xTB analytic gradients).
        # 2) Single-point energy with gxtb on the optimized structure.
        opt_cmd = [xtb_bin, str(input_xyz), "--opt"]
        if xtb_opt_level:
            opt_cmd.append(xtb_opt_level)
        try:
            opt_res = subprocess.run(
                opt_cmd,
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return idx, None, f"xtb binary not found: {xtb_bin}"
        except Exception as exc:  # pragma: no cover
            return idx, None, f"exception running xtb: {exc}"

        (cwd / "xtb_opt.out").write_text(opt_res.stdout)
        if opt_res.returncode != 0:
            if opt_res.returncode < 0:
                sig = -opt_res.returncode
                sig_name = next((s.name for s in signal.Signals if s.value == sig), f"SIG{sig}")
                return idx, None, f"xtb(opt) crashed: {sig_name} ({opt_res.returncode})"
            return idx, None, f"xtb(opt) exited with code {opt_res.returncode}"

        opt_xyz = cwd / "xtbopt.xyz"
        if not opt_xyz.is_file():
            return idx, None, "xtb(opt) finished but xtbopt.xyz not found"

        # gxtb will try to read "gxtbrestart" from the current directory if present.
        # That file can be incompatible across different systems/runs and cause hard failures.
        # It's safest to delete it for single-point batches.
        try:
            (cwd / "gxtbrestart").unlink()
        except FileNotFoundError:
            pass

        # Compatibility hack: some builds appear to truncate 'xtbopt.xyz' to 'xtbop'
        # when parsing file arguments. Provide a symlink so even a truncated open()
        # still sees valid coordinates.
        xtbop = cwd / "xtbop"
        try:
            if xtbop.is_file() and xtbop.stat().st_size == 0:
                xtbop.unlink()
            if not xtbop.exists():
                xtbop.symlink_to(opt_xyz.name)
        except Exception:
            # Non-fatal; continue with the intended filename.
            pass

        # gxtb single-point
        # IMPORTANT: avoid passing absolute coordinate paths to gxtb; some builds
        # have path parsing bugs and may truncate the filename.
        sp_cmd = [gxtb_bin, "-c", opt_xyz.name] + param_args
        try:
            sp_res = subprocess.run(
                sp_cmd,
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return idx, None, f"gxtb binary not found: {gxtb_bin}"
        except Exception as exc:  # pragma: no cover
            return idx, None, f"exception running gxtb: {exc}"

        (cwd / "gxtb_sp.out").write_text(sp_res.stdout)
        if sp_res.returncode != 0:
            if sp_res.returncode < 0:
                sig = -sp_res.returncode
                sig_name = next((s.name for s in signal.Signals if s.value == sig), f"SIG{sig}")
                return idx, None, f"gxtb(sp) crashed: {sig_name} ({sp_res.returncode})"
            return idx, None, f"gxtb(sp) exited with code {sp_res.returncode}"

        energy = parse_final_energy(cwd, sp_res.stdout)
        if energy is None:
            return idx, None, "could not parse gxtb(sp) energy"
        return idx, energy, "ok"
    elif mode == "gxtb_sp":
        # Single point only (no optimization)
        try:
            (cwd / "gxtbrestart").unlink()
        except FileNotFoundError:
            pass
        sp_cmd = [gxtb_bin, "-c", input_xyz.name] + param_args
        try:
            sp_res = subprocess.run(
                sp_cmd,
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return idx, None, f"gxtb binary not found: {gxtb_bin}"
        except Exception as exc:  # pragma: no cover
            return idx, None, f"exception running gxtb: {exc}"

        (cwd / "gxtb_sp.out").write_text(sp_res.stdout)
        if sp_res.returncode != 0:
            if sp_res.returncode < 0:
                sig = -sp_res.returncode
                sig_name = next((s.name for s in signal.Signals if s.value == sig), f"SIG{sig}")
                return idx, None, f"gxtb(sp) crashed: {sig_name} ({sp_res.returncode})"
            return idx, None, f"gxtb(sp) exited with code {sp_res.returncode}"

        energy = parse_final_energy(cwd, sp_res.stdout)
        if energy is None:
            return idx, None, "could not parse gxtb(sp) energy"
        return idx, energy, "ok"
    else:
        return idx, None, f"unknown mode: {mode}"

    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return idx, None, f"xtb binary not found: {xtb_bin}"
    except Exception as exc:  # pragma: no cover - defensive
        return idx, None, f"exception running xtb: {exc}"

    # Save combined output to a log file for later inspection.
    (cwd / "xtb.out").write_text(result.stdout)

    if result.returncode != 0:
        # Negative return codes mean "terminated by signal" on POSIX shells.
        if result.returncode < 0:
            sig = -result.returncode
            sig_name = next((s.name for s in signal.Signals if s.value == sig), f"SIG{sig}")
            return idx, None, f"xtb crashed: {sig_name} ({result.returncode})"
        # If the driver wrapper is enabled, point the user to the driver log.
        if use_driver_wrapper and (cwd / "gxtb_driver.out").is_file():
            return idx, None, f"xtb exited with code {result.returncode} (see gxtb_driver.out)"
        return idx, None, f"xtb exited with code {result.returncode}"

    energy = parse_final_energy(cwd, result.stdout)
    if energy is None:
        return idx, None, "could not parse final energy"

    return idx, energy, "ok"


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run xtb+gxtb conformer workflows on all conformers in an SDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Notes:
              - Each SDF record is treated as a separate conformer.
              - Results are written under --outdir as conf_XXXXX directories.
              - A summary CSV (energies.csv) is written with columns:
                    index, energy_Eh, status

              - Units:
                  * energy_Eh is Hartree (Eh)
                  * 1 Eh = 2625.49962 kJ/mol = 627.509474 kcal/mol = 27.211386 eV

              - Energy provenance (important!):
                  * xtb_opt.out: xtb optimization energy (GFN2-xTB)
                  * gxtb_sp.out + energy file + energies.csv: gxtb single-point energy (g-xTB)

            Parallel usage examples:
              # Use 16 workers on a single node
              python scripts/gxtb_optimize_conformers.py confs.sdf --workers 16

              # Under SLURM, one big job on 32 cores:
              srun -n 1 -c 32 python scripts/gxtb_optimize_conformers.py confs.sdf --workers 32
            """
        ),
    )
    parser.add_argument(
        "sdf",
        help="Input SDF file containing all conformers.",
    )
    parser.add_argument(
        "--outdir",
        default="gxtb_conformers",
        help="Output directory (will be created if missing). [default: %(default)s]",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help="Number of parallel workers (xtb processes). [default: half the cores]",
    )
    parser.add_argument(
        "--xtb",
        dest="xtb_bin",
        default="xtb",
        help="Path to xtb binary. [default: %(default)s]",
    )
    parser.add_argument(
        "--gxtb",
        dest="gxtb_bin",
        default=None,
        help="Path to gxtb binary. [default: auto-detect from PATH or this repo]",
    )
    parser.add_argument(
        "--gxthome",
        dest="gxthome",
        default=None,
        help=(
            "Path to GXTBHOME (parameter directory). "
            "If not given and $GXTBHOME is unset, defaults to '<repo_root>/parameters' when available."
        ),
    )
    parser.add_argument(
        "--max-confs",
        type=int,
        default=0,
        help="Optional maximum number of conformers to process (0 = all).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="If set, delete existing files under each conf_XXXXX directory before running.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip environment/parameter sanity checks (not recommended).",
    )
    parser.add_argument(
        "--no-driver-wrapper",
        action="store_true",
        help="Disable wrapper script for xtb --driver (not recommended).",
    )
    parser.add_argument(
        "--mode",
        default="gxtb_opt",
        choices=["gxtb_opt", "xtb_opt_gxtb_sp", "gxtb_sp"],
        help=(
            "Run mode. "
            "gxtb_opt: optimize with xtb using gxtb numerical gradients (slow/fragile). "
            "xtb_opt_gxtb_sp: optimize with xtb (GFN2) then gxtb single-point. "
            "gxtb_sp: gxtb single-point only."
        ),
    )
    parser.add_argument(
        "--xtb-opt-level",
        default="",
        choices=["", "loose", "tight", "verytight"],
        help="xtb optimization level for xtb_opt_gxtb_sp mode. [default: standard]",
    )
    parser.add_argument(
        "--charge",
        type=int,
        default=None,
        help="Total charge for all conformers; writes .CHRG in each run directory.",
    )
    parser.add_argument(
        "--uhf",
        type=int,
        default=None,
        help="Number of unpaired electrons (UHF); writes .UHF in each run directory.",
    )

    args = parser.parse_args(argv)

    sdf_path = Path(args.sdf).resolve()
    if not sdf_path.is_file():
        sys.stderr.write(f"ERROR: SDF file not found: {sdf_path}\n")
        return 1

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Auto-detect gxtb binary if not given.
    if args.gxtb_bin is None:
        repo_root = Path(__file__).resolve().parents[1]
        candidate = repo_root / "binary" / "gxtb"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            args.gxtb_bin = str(candidate)
        else:
            args.gxtb_bin = "gxtb"

    # Auto-detect GXTBHOME if not given and env not set.
    if args.gxthome is None and "GXTBHOME" not in os.environ:
        repo_root = Path(__file__).resolve().parents[1]
        candidate_dir = repo_root / "parameters"
        if candidate_dir.is_dir():
            args.gxthome = str(candidate_dir)

    gxthome_effective = args.gxthome if args.gxthome is not None else os.environ.get("GXTBHOME")

    print(f"Input SDF      : {sdf_path}")
    print(f"Output dir     : {outdir}")
    print(f"xtb binary     : {args.xtb_bin}")
    print(f"gxtb binary    : {args.gxtb_bin}")
    print(f"GXTBHOME       : {gxthome_effective or '(unset)'}")
    print(f"Workers        : {args.workers}")
    print(f"Mode           : {args.mode}")
    if args.mode == "xtb_opt_gxtb_sp":
        print(f"xtb opt level  : {args.xtb_opt_level or '(standard)'}")
    if args.charge is not None:
        print(f"Charge         : {args.charge} (forced)")
    else:
        print("Charge         : from SDF formal charge (per conformer)")
    if args.uhf is not None:
        print(f"UHF            : {args.uhf}")

    if not args.skip_preflight:
        try:
            preflight_check(args.gxtb_bin, gxthome_effective, args.xtb_bin)
        except Exception as exc:
            sys.stderr.write(f"\nERROR: preflight check failed:\n{exc}\n\n")
            sys.stderr.write(
                "If you're sure your environment is correct, you can bypass this with --skip-preflight.\n"
            )
            return 1

    # Step 1: split SDF into per-conformer XYZ files.
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    jobs: List[Tuple[int, Path, int]] = []
    n_total = 0

    for idx, mol in enumerate(suppl, start=1):
        if mol is None:
            continue
        n_total += 1
        if args.max_confs and n_total > args.max_confs:
            break

        conf_dir = outdir / f"conf_{idx:05d}"
        conf_dir.mkdir(parents=True, exist_ok=True)
        if args.clean:
            for p in conf_dir.iterdir():
                try:
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink()
                except FileNotFoundError:
                    pass
        xyz_path = conf_dir / "input.xyz"
        write_xyz(mol, xyz_path)
        mol_charge = int(Chem.GetFormalCharge(mol))
        jobs.append((idx, conf_dir, mol_charge))

    if not jobs:
        sys.stderr.write("ERROR: no valid molecules found in SDF.\n")
        return 1

    print(f"Prepared {len(jobs)} conformers.")

    # Step 2: run xtb+gxtb in parallel.
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_map = {
            pool.submit(
                run_xtb_on_xyz,
                idx,
                conf_dir,
                args.xtb_bin,
                args.gxtb_bin,
                gxthome_effective,
                (not args.no_driver_wrapper),
                args.mode,
                args.xtb_opt_level,
                (args.charge if args.charge is not None else mol_charge),
                args.uhf,
            ): idx
            for idx, conf_dir, mol_charge in jobs
        }

        for fut in as_completed(future_map):
            idx = future_map[fut]
            try:
                conf_idx, energy, status = fut.result()
            except Exception as exc:  # pragma: no cover - defensive
                conf_idx, energy, status = idx, None, f"exception in worker: {exc}"
            results.append((conf_idx, energy, status))
            if len(results) % 100 == 0:
                print(f"... finished {len(results)} / {len(jobs)} conformers", flush=True)

    # Step 3: write summary CSV.
    results.sort(key=lambda x: x[0])
    csv_path = outdir / "energies.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "energy_Eh", "status"])
        for idx, energy, status in results:
            writer.writerow([idx, f"{energy:.12f}" if energy is not None else "", status])

    n_ok = sum(1 for _, e, s in results if e is not None and s == "ok")
    print(f"Done. Successful: {n_ok}/{len(results)}. Summary: {csv_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


