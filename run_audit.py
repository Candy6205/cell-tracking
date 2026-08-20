"""
run_audit.py -- CPU kernel entry point for stranded_audit.

Answers, by counting rather than estimating, whether the targets stranded below
the 0.5 edge gate are real cells or spurious detections -- and exactly how many
of the recorded 450 false negatives that gate is responsible for.

No GPU, no ILP. Node matching and one pass over the GT edges.

The path probing below is duplicated from run_sweep.py. Two kernel entry points
each needing to find four mounts is not worth a shared module that would have
to be re-uploaded to the code dataset every time either changes.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

SCRIPT_VERSION = "2026-08-20a"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)


def _first_existing(*paths):
    for p in paths:
        q = pathlib.Path(p)
        if q.exists():
            return q
    return None


def _find_npz_dir(root="/kaggle/input", maxdepth=4):
    """Rule 9: prune. The competition mount holds millions of zarr chunks."""
    skip = ("competitions", "biohub-cell-tracking-during-development")
    stack = [(pathlib.Path(root), 0)]
    while stack:
        d, depth = stack.pop()
        try:
            entries = sorted(os.scandir(d), key=lambda e: e.name)
        except OSError:
            continue
        for e in entries:
            if e.is_file() and e.name.endswith(".npz"):
                return pathlib.Path(e.path).parent
            if not e.is_dir():
                continue
            if e.name in skip or e.name.endswith((".zarr", ".geff", ".n5", ".ome")):
                continue
            if depth + 1 <= maxdepth:
                stack.append((pathlib.Path(e.path), depth + 1))
    return None


PACK = _first_existing(
    "/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1",
    "/kaggle/input/biohub-tracking-support-pack-50ep-v1")
CODE = _first_existing(
    "/kaggle/input/datasets/faustine649/cell-tracking/cell tracking",
    "/kaggle/input/datasets/faustine649/cell-tracking",
    "/kaggle/input/cell-tracking/cell tracking",
    "/kaggle/input/cell-tracking")
CACHE = _find_npz_dir()
COMP = _first_existing(
    "/kaggle/input/competitions/biohub-cell-tracking-during-development",
    "/kaggle/input/biohub-cell-tracking-during-development")

for label, p in (("code", CODE), ("cache", CACHE), ("comp", COMP)):
    if p is None:
        top = sorted(e.name for e in os.scandir("/kaggle/input")) \
            if pathlib.Path("/kaggle/input").exists() else []
        raise SystemExit(f"{label} mount not found. /kaggle/input holds: {top}")
print(f"code  {CODE}\ncache {CACHE}\ncomp  {COMP}", flush=True)

# viz_cells has zarr-free readers, so the wheels are not strictly required
# here. Install them anyway when the pack is present: a silent fallback to a
# different reader is exactly the class of thing rule 1 exists to surface.
if PACK is not None:
    KEEP_OUT = {"numpy", "scipy", "torch", "pandas", "pyarrow", "numba",
                "llvmlite", "matplotlib", "pillow", "networkx",
                "scikit-learn", "scikit-image", "setuptools", "wheel", "pip"}
    WH = pathlib.Path("/kaggle/working/wheelhouse")
    shutil.rmtree(WH, ignore_errors=True)
    WH.mkdir(parents=True)
    for w in sorted(PACK.rglob("*.whl")):
        if w.name.split("-")[0].replace("_", "-").lower() not in KEEP_OUT:
            shutil.copy(w, WH / w.name)
    staged = sorted(WH.glob("*.whl"))
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-index",
         "--no-deps"] + [str(w) for w in staged],
        capture_output=True, text=True)
    print(f"pip exit {r.returncode} ({len(staged)} wheels)", flush=True)
    shutil.rmtree(WH, ignore_errors=True)
else:
    print("[degraded] support pack not attached -- falling back to the "
          "zarr-free readers in viz_cells. Coordinates and scales still come "
          "from the same metadata, but if anything reads oddly, this is why.",
          flush=True)

sys.path.insert(0, str(CODE))
if PACK is not None:
    # division_ceiling now runs the real ILP through link_replay, which imports
    # the official predict module and tracksdata. Those live in the repo inside
    # the support pack, not in the code dataset.
    for p in (PACK / "repo" / "src", PACK / "repo" / "scripts"):
        if p.exists():
            sys.path.insert(0, str(p))
        else:
            print(f"[degraded] {p} missing -- link_replay cannot import the "
                  f"official build_graph/ILPSolver and the division funnel "
                  f"will fail rather than silently use something else",
                  flush=True)

import stranded_audit as sa                                   # noqa: E402
import division_ceiling as dc                                 # noqa: E402

print("stranded_audit", sa.SCRIPT_VERSION,
      " division_ceiling", dc.SCRIPT_VERSION, flush=True)

print("\n" + "=" * 62)
print("1. what the 0.5 edge gate costs")
print("=" * 62, flush=True)
res_a = sa.run(CACHE, COMP, out="/kaggle/working/stranded_audit.csv")

print("\n" + "=" * 62)
print("2. can orphan forks recover anything on THIS pipeline")
print("=" * 62, flush=True)
res_d = dc.run(CACHE, COMP, out="/kaggle/working/division_ceiling.csv")

print("\nAUDIT DONE", res_a, flush=True)
print("CEILING DONE", res_d, flush=True)
