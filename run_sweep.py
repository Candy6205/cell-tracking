"""
run_sweep.py -- CPU kernel entry point for the cached-candidate replay.

Stage 1 deliberately does ONE configuration, not a sweep: the submitted
0.876 setting (threshold 0.50, disappearance 1.5). It has to reproduce the
recorded local 0.9269 before any other row in any sweep means anything, and it
also tells us what one full pass costs in wall time. Only then is a real sweep
worth launching.

No GPU. `build_graph` and `ILPSolver.solve` are CPU, so this costs no quota.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import time

SCRIPT_VERSION = "2026-08-20a"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)


def _first_existing(*paths):
    for p in paths:
        q = pathlib.Path(p)
        if q.exists():
            return q
    return None


PACK = _first_existing(
    "/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1",
    "/kaggle/input/biohub-tracking-support-pack-50ep-v1")
assert PACK is not None, "support pack not attached"

# The code dataset lost its "cell tracking/" prefix when it was re-uploaded as
# a zip on 2026-08-20, so probe both layouts rather than assuming either.
CODE = _first_existing(
    "/kaggle/input/datasets/faustine649/cell-tracking/cell tracking",
    "/kaggle/input/datasets/faustine649/cell-tracking",
    "/kaggle/input/cell-tracking/cell tracking",
    "/kaggle/input/cell-tracking")
assert CODE is not None, "code dataset not attached"

def _find_npz_dir(root="/kaggle/input", maxdepth=4):
    """
    Locate the cached .npz, wherever the kernel output happened to mount.

    Rule 9: never rglob /kaggle/input. The competition mount holds millions of
    one-chunk-per-frame files and a blind walk appears to hang for minutes.
    Prune the competition tree and every zarr-like directory, and cap depth.
    """
    import os
    skip = ("competitions", "biohub-cell-tracking-during-development")
    stack = [(pathlib.Path(root), 0)]
    seen_top = []
    while stack:
        d, depth = stack.pop()
        try:
            entries = sorted(os.scandir(d), key=lambda e: e.name)
        except OSError:
            continue
        if depth == 1:
            seen_top.append(d.name)
        for e in entries:
            if e.is_file() and e.name.endswith(".npz"):
                return pathlib.Path(e.path).parent, seen_top
            if not e.is_dir():
                continue
            if e.name in skip or e.name.endswith((".zarr", ".geff", ".n5", ".ome")):
                continue
            if depth + 1 <= maxdepth:
                stack.append((pathlib.Path(e.path), depth + 1))
    return None, seen_top


CACHE, _tops = _find_npz_dir()
if CACHE is None:
    import os
    top = sorted(e.name for e in os.scandir("/kaggle/input")) \
        if pathlib.Path("/kaggle/input").exists() else []
    raise SystemExit(
        f"no .npz found under /kaggle/input. Top level holds: {top}. "
        f"The cache kernel output is attached via kernel_sources in "
        f"kernel-metadata.json; if the slug is missing from that list the "
        f"mount will not exist."
    )

COMP = _first_existing(
    "/kaggle/input/competitions/biohub-cell-tracking-during-development",
    "/kaggle/input/biohub-cell-tracking-during-development")
assert COMP is not None, "competition data not attached"

print(f"pack  {PACK}\ncode  {CODE}\ncache {CACHE}\ncomp  {COMP}", flush=True)

# ---- offline wheels: tracksdata, pyscipopt, rustworkx. Same exclusion list as
# ---- cache_candidates.py -- installing numpy/scipy/torch clobbers the image.
KEEP_OUT = {"numpy", "scipy", "torch", "pandas", "pyarrow", "numba", "llvmlite",
            "matplotlib", "pillow", "networkx", "scikit-learn", "scikit-image",
            "setuptools", "wheel", "pip"}
WH = pathlib.Path("/kaggle/working/wheelhouse")
shutil.rmtree(WH, ignore_errors=True)
WH.mkdir(parents=True)
for w in sorted(PACK.rglob("*.whl")):
    if w.name.split("-")[0].replace("_", "-").lower() not in KEEP_OUT:
        shutil.copy(w, WH / w.name)
staged = sorted(WH.glob("*.whl"))
r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps"]
    + [str(w) for w in staged], capture_output=True, text=True)
print(f"pip exit {r.returncode} ({len(staged)} wheels)", flush=True)
if r.returncode != 0:
    print(r.stderr[-1500:], flush=True)
shutil.rmtree(WH, ignore_errors=True)

for p in (PACK / "repo" / "src", PACK / "repo" / "scripts", CODE):
    sys.path.insert(0, str(p))

try:
    import pyscipopt                                          # noqa: F401
except Exception as exc:                                      # noqa: BLE001
    raise SystemExit(f"[degraded] pyscipopt unavailable ({exc}) -- the ILP "
                     f"cannot run and a greedy fallback would answer a "
                     f"different question. Aborting.")

import link_replay as lr                                      # noqa: E402

print("link_replay", lr.SCRIPT_VERSION, flush=True)

# ---- 1. inventory: how much wider the candidate set is than the submission saw
print("\n=== inventory ===", flush=True)
lr.inventory(CACHE)

# ---- 2. the anchor. This MUST land on the recorded 0.9269.
print("\n=== anchor: the submitted configuration ===", flush=True)
t0 = time.time()
summary, rows = lr.score_config(
    lr.load_cache(CACHE), COMP,
    threshold=lr.BASE["threshold"],
    appearance=lr.BASE["appearance"],
    disappearance=lr.BASE["disappearance"],
    division=lr.BASE["division"])
dt = time.time() - t0

print(f"\ndatasets         {summary['n']}")
print(f"edge TP/FP/FN    {summary['edge_tp']} / {summary['edge_fp']} / "
      f"{summary['edge_fn']}")
print(f"edge_jaccard     {summary['edge_jaccard']:.4f}")
print(f"adj_edge_jaccard {summary['adj_edge_jaccard']:.4f}   "
      f"<- recorded 0.9269")
print(f"division_jaccard {summary['division_jaccard']:.4f}")
print(f"score_full       {summary['score_full']:.4f}")
print(f"detect_ceiling   {summary['detect_ceiling']:.4f}")
print(f"link_rate        {summary['link_rate']:.4f}")
print(f"\none full pass over 12 datasets: {dt:.0f}s "
      f"({dt / 60:.1f} min) -- a 12-cell sweep would cost "
      f"{12 * dt / 3600:.1f} h", flush=True)

delta = summary["adj_edge_jaccard"] - 0.9269
if abs(delta) > 0.01:
    print(f"\n[degraded] the anchor is off by {delta:+.4f}. The cached "
          f"candidates do NOT reproduce the submitted pipeline, so no sweep "
          f"built on them can be trusted. Find this before sweeping.",
          flush=True)
else:
    print(f"\nanchor reproduces to {delta:+.4f} -- the cache is faithful and "
          f"a sweep is worth launching.", flush=True)

print("ANCHOR DONE", flush=True)
