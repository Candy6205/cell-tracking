"""
run_sweep.py -- CPU kernel entry point for the cached-candidate replay.

Stage 1 deliberately does ONE configuration, not a sweep: the submitted
0.876 setting (threshold 0.50, disappearance 1.5). It has to reproduce the
recorded local 0.9269 before any other row in any sweep means anything, and it
also tells us what one full pass costs in wall time. Only then is a real sweep
worth launching.

No GPU. `build_graph` and `ILPSolver.solve` are CPU, so this costs no quota.

Two stages now share this kernel, because they share the mount and the 54-wheel
install and every finished run emails the account owner (toolchain trap 7):

    RUN_LINK_SWEEP   the threshold x disappearance replay
    RUN_FORK_SWEEP   fork_pass, on top of whatever linking optimum came out

Flip either to False to run the other alone.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import time

SCRIPT_VERSION = "2026-08-21a"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

# Which stages this run does. Both share one mount and one wheel install.
RUN_LINK_SWEEP = True
RUN_FORK_SWEEP = True


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

# ---- 2. the sweep.
#
# The anchor run on 2026-08-20 reproduced the submitted configuration exactly
# (TP/FP/FN 9863/318/450, adj_J 0.9269 against a recorded 0.9269) at 154 s per
# pass, so the cache is faithful and a real sweep is worth its wall time.
#
# The loop is threshold-major and every row prints as it completes, so the
# informative rows land first and a timeout still leaves a readable table.
# The 0.50 / 1.5 row is the submitted configuration and is the anchor: if it
# drifts from 0.9269, stop reading the rest.
print("\n=== sweep ===", flush=True)

# Round 1 (2026-08-20) swept threshold x disappearance and found
# 0.35 / 2.5 -> adj_J 0.9337, +0.0068 over the submitted 0.50 / 1.5. Its
# optimum sat on the disappearance edge of the grid, and 0.25 and 0.15 were
# WORSE than 0.35 -- what running out of discrimination looks like.
#
# Round 2 result: velocity is a SUBSTITUTE for disappearance, not a complement.
# It helped at low disappearance (0.35/1.5: 0.9296 -> 0.9338) and hurt at high
# (0.50/3.5: 0.9283 -> 0.9215) -- both prune unreliable links, and doing both
# over-prunes (nodes 168675 -> 143028). Best overall stayed at velocity 0, so
# round 3 drops the dimension; it doubled the cost of every cell for nothing.
#
# What round 2 did find: disappearance was on the grid edge AGAIN.
#     disapp   1.5      2.5      3.5
#     adj_J   0.9296   0.9337   0.9382     at threshold 0.35, still climbing
# progress.md recorded disappearance 3.0 going bad, but that was measured at
# threshold 0.5. Opening the gate replaces what the pruning removes, so the two
# are coupled and neither could be read alone before the cache existed.
#
# Round 3 pushes disappearance to 5.5 and keeps 0.50/1.5 in the grid purely as
# the anchor -- without it there is nothing to confirm this run scores the same
# way the last one did.
#
# The old round-2 note follows.
# Round 2 added the velocity term. `window_size = 2` means the edge model
# structurally cannot see that a cell was already moving; the public tier
# compensates in post-processing with a velocity weight of 0.5 and we never
# have. If that is the missing discriminator, the lower thresholds should stop
# being worse.
#
# Divisions stay off throughout: forks are impossible at division_weight 1.0,
# the term is worth 0.1, and 12 datasets hold ~12 division events. Edges carry
# 90% of the score and 128,883 annotations against 151.
if RUN_LINK_SWEEP:
    res = lr.sweep(CACHE, root=COMP,
                   thresholds=(0.50, 0.40, 0.35, 0.30),
                   disappearances=(1.5, 3.5, 4.5, 5.5),
                   velocities=(0.0,),
                   appearance=0.0, division=1.0,
                   out="/kaggle/working/replay_sweep.csv")

    print(f"\nbest threshold        {res['best_threshold']}")
    print(f"best disappearance    {res['best_disappearance']}")
    print(f"best velocity_weight  {res['best_velocity_weight']}")
    print(f"best adj_edge_jaccard {res['best_adj_edge_jaccard']:.4f}")
    print(f"delta vs submitted    {res['delta_vs_baseline']:+.4f}")
    print("SWEEP DONE", flush=True)
    BEST_THR = res["best_threshold"]
    BEST_DIS = res["best_disappearance"]
else:
    # The converged optimum from round 3, so the fork stage can run alone.
    BEST_THR, BEST_DIS = 0.40, 3.5
    print(f"\n[skipped] link sweep off; forks run at the recorded optimum "
          f"threshold {BEST_THR} / disappearance {BEST_DIS}", flush=True)

# ---- 3. forks the SOLVER refused, added back after the solve.
#
# division_ceiling.py established the premise: 9 of the 12 GT divisions have
# BOTH daughters carrying a candidate above 0.5. The gate is not what hides
# them -- `division_weight 1.0` with `appearance_weight 0.0` makes a fork need
# edge_prob > 1.0, so it is forbidden and the second daughter ends the solve
# with in-degree 0. fork_pass adds exactly those edges back.
#
# READ THE EDGE COLUMNS. This cache holds ~9 division events, so its
# division_jaccard is noise. What 12 datasets CAN answer is whether the pass
# is free: if adj_edge_jaccard does not fall, it is, and the division count
# then has to be measured on a wider cache.
#
# The floor list starts at 1.01, which no probability reaches, so that row is
# the no-fork control and must reproduce the linking sweep's cell at the same
# (threshold, disappearance).
if RUN_FORK_SWEEP:
    print("\n=== forks ===", flush=True)
    import fork_pass as fp                                       # noqa: E402

    print("fork_pass", fp.SCRIPT_VERSION, flush=True)
    fp.selftest(verbose=True)
    fres = fp.sweep(CACHE, root=COMP,
                    floors=(1.01, 0.90, 0.70, 0.50, 0.30),
                    cap=fp.DEFAULT_CAP,
                    threshold=BEST_THR, disappearance=BEST_DIS,
                    appearance=0.0,
                    out="/kaggle/working/fork_sweep.csv")
    print(f"\nfork rows {fres['n_rows']}")
    print("FORKS DONE", flush=True)
