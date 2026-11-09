"""edgeprob_rank2.py -- v3 of edgeprob_rank.py: the SECOND-SEED head joins,
and the heads are also tried WITHOUT a ranker.

SCRIPT_VERSION 2026-09-03b-edgeprob-rank2

v2 (biohub-edgeprob-rank v2, 2026-09-03) settled that +BIDIR -- the 18
columns plus the official head's p_fwd / p_rev / h_fused, D175 pool --
scores F4-6bba 0.9261 / F4-44b6 0.8515 adj_J against the shipped D187
ranker's 0.9216 / 0.8353, and it shipped as ref 55974245 = LB 0.932.
This kernel asks two further questions on the same harness:

  1. Does a SECOND, independently trained copy of the same head
     (pilkwang/biohub-temporal-unet3d-seed314159-v1, the "secondary"
     model the public 0.948 notebook fuses; cached over our primaries by
     biohub-edgeprob2 exactly as biohub-edgeprob cached the first) add
     anything on top of the first? Arms +BIDIR2 (second head alone),
     +BOTH (both heads' three columns each), +AVG (the two heads
     averaged: p_avg_fwd, p_avg_rev, h_avg), all against +BIDIR.
  2. The user's question (2026-09-03): why use the head only as a
     FEATURE? Arms HEAD1 (h_fused of the first head fed straight to the
     ILP as the edge probability, no ranker) and HEAD12 (h_avg of both
     heads, likewise). Same ILP, same floor 0.10, same post-link patches.
     The 2026-08-22 chimera lost 0.011 with the official head as the
     linker on xl detections, single direction, single seed; this is
     the bidirectional, two-seed version of that test.

Reference arm for every paired delta: +BIDIR (what ships). BASE18 is
not retrained here (v2 recorded it). ANCHOR_D187 stays as the hard
anchor of the harness (F4-6bba 0.9216); +BIDIR's v2 numbers are soft
anchors of the retraining (0.9261 / 0.8515, div_J 0.1515 / 0.0000).

Joined layout: 23 harness columns, then head 1's p_fwd p_rev l_fwd l_rev
h_fused (23..27), head 2's the same (28..32), then p_avg_fwd p_avg_rev
h_avg (33..35): 36 columns. Both caches must agree on n_per_frame and on
the exact (t, i, j) key set, asserted per dataset.

Attach, in addition to v2's sources: kernel faustine649/biohub-edgeprob2.
Expected ~3.5 h CPU: features ~35 min, stage B 7 arms x 24 datasets.

--- v2's docstring follows ---
edgeprob_rank.py -- does the OFFICIAL edge head's probability add anything
to OUR edge ranker? v2: D175 pool, evaluated on the F4 arm (what ships).

SCRIPT_VERSION 2026-09-03a-edgeprob-rank

THE QUESTION. The pipeline links xl detections with an 18-column geometric
ranker (HistGradientBoosting) and never sees the end-to-end-trained edge
head (progress.md 2026-08-24, "the edge ranker never sees edge_prob").
`edgeprob_cache.py` (kernel faustine649/biohub-edgeprob, GPU) re-ran the
official head over OUR primaries and dumped, per dataset, its softmax
probability for every candidate pair the CPU ranker considers, forward
(t -> t+1) and on the reversed movie (t+1 -> t). This CPU kernel joins
those numbers onto the ranker's rows and asks, with the pipeline's own
verdict, whether they help.

WHAT v1 SETTLED (biohub-edgeprob-rank v1, 2026-09-02, D32 pool, BASE
primaries, 116.5 min): +BIDIR (18 + p_fwd + p_rev + h_fused) was the best
of four arms, +0.0046 6bba / +0.0106 44b6 adj_J over BASE18, same sign in
both clean groups; +PFWD moved less and +ALL added nothing over +BIDIR.
Those two arms are dropped here. v2 asks the question that decides a
submission: the same gain, from a ranker trained on the FULL pool, on the
F4 primaries that ship.

ARMS (edge ranker columns; everything else identical):

  BASE18    the 18 shipped columns, retrained on D175
  +BIDIR    18 + p_fwd + p_rev + h_fused, trained on D175

  h_fused = harmonic mean of p_fwd and p_rev (the public 0.948 notebook's
  forward/reverse fusion); p_fwd alone where p_rev is NaN.

  ANCHOR_D187  the shipped D187 pkl, loaded not trained -- proves the
               harness (F4-6bba adj_J must read 0.9216). Not a judged arm.

TRAINING POOL = D175: all 199 train movies minus HOLDOUT_6BBA (12) minus
the seeded 44b6 draw (12) -- dctrain's split (biohub-dctrain-175-weights,
split_manifest.json; cross-checked when that dataset happens to be
mounted, not required). The pool size is asserted to be 175. Both
evaluation groups are therefore OUTSIDE the pool of every trained arm.
⚠ ANCHOR_D187 is the exception: the D187 ranker trained on all 71 44b6
movies, so its 44b6 row is in-sample and its 44b6 anchor is soft; the
two trained arms are clean there.

GT labelling exactly as poolsize.py labels edges: a primary is matched to
its nearest GT node within MATCH_UM 5 um; a candidate pair (i, j) is kept
only when i matched a GT node THAT HAS CHILDREN in the GT (sparse
annotation: a matched parent with no recorded child says nothing, and is
dropped rather than counted as a negative); label 1 when j matched one of
those children, else 0. Same hyperparameters as the shipped ranker
(poolsize.py:935): HistGradientBoostingClassifier(max_iter=300,
random_state=0), a 10% seeded validation split for the edge AUC, then a
refit on 100% of the pool. Pool features come from the BASE (detcache)
primaries joined with biohub-edgeprob's BASE edgeprob_<name>.npz, which
is what the head was cached on for all 199.

EVALUATION on the F4 ARM (ep05 DeepCenter as a fourth equal-weight
heatmap, ref 55889500, what ships at 0.925): primaries from
fdet_F4_<name>.npz (biohub-fusecache), joined with edgeprob_F4_<name>.npz
(biohub-edgeprob v3, the head over those same F4 primaries). Two groups,
12 datasets each, full harness (solve_one -> patch_final -> the official
scorer), one line per dataset:

  6bba   the 12-dataset holdout every recorded number comes from;
  44b6   12 drawn by random.Random(20260827).sample from the 44b6 OUTSIDE
         the D32 pool (dctrain.py:158-163 verbatim, the user's standing
         rule) -- dctrain's excluded 12.

SCORE_BASE_ARM (default False) additionally scores the same three rankers
on the BASE primaries (detcache + edgeprob_<name>.npz), which doubles
stage B; its anchors are v1's (ANCHOR_D187 BASE-6bba 0.9293 hard).

No zebrahub arm: biohub-edgeprob scored competition movies only.

ANCHORS (hard, asserted after everything has printed so a miss cannot
waste the run's tables -- poolsize lost a summary to a late KeyError):

  ANCHOR_D187 F4-6bba adj_J 0.9216 / div_J 0.0606  (f4diag 2026-09-01,
              rankwhy, edgewhy: the D187 pkl on the F4 caches with this
              exact machinery). adj_J is HARD (tolerance 0.0005); div_J
              is printed and warned.
  Soft (printed, warned, not asserted): ANCHOR_D187 F4-44b6 0.8353 /
  div_J 0.0000 -- in-sample for D187, and it depends on the 44b6 draw
  matching dctrain's manifest.
  With SCORE_BASE_ARM: ANCHOR_D187 BASE-6bba 0.9293 hard, BASE-44b6
  0.8185 soft (v1 reproduced both).
  BASE18 has no anchor in v2: no recorded number exists for a D175
  ranker on this harness.

OUTPUT. edge_ranker_ep175_bidir.pkl (and _base18) with keys `model`,
`cols`, `pool`, `extra_cols`, `feature_names`, `n_features`, `h_fused`,
`version`, so a submission kernel can load it; the exact column layout
the model expects is printed under "RANKER LAYOUT".

ASSEMBLY. The harness is rankwhy.py's (2026-09-02b, itself dcpool.py's
verbatim: loader, 18-column pair_features, pass1_tracks and the
23-column pair_features that overrides it, _gap_close, _short_filter,
_linefit, _add_safe_divisions, score_dir, solve_edges) with the shipped
post-link constants as its defaults (SHORT_MIN 4, GAP1_CAP_FRAC 0.02,
GAP_NEED_CAND False = cap20, ref 55817630). Cut by line content:
preamble (support pack wheels, repo clone, imports), constants, the
detcache walk, `_logit` .. `solve_edges`, and the acceleration block
that holds `warn_once` / `pass1_tracks` / the 23-column pair_features.
That slice is byte-identical to v1's. rankwhy's `solve_one` is re-typed
below with the model and its columns as parameters and the features
precomputed, so the arms share one feature pass per dataset;
`patch_final` is rankwhy's verbatim; `load_dets(name, arm)` is rankwhy's
(BASE reads detcache_, F4 reads fdet_F4_).

THE JOIN. edgeprob_<name>.npz / edgeprob_F4_<name>.npz rows are (t, i, j)
into seeds_of order of the matching primaries. Before any index is
trusted, `n_per_frame` must equal this harness's own primaries per frame
for that det arm; then every harness pair must find its row (a miss is a
hard error, never a silent 0) and the row counts must agree, so the two
pair sets are proven identical.

MEMORY. 199 pool datasets are reduced to labelled rows as they are
computed; full features stay in memory only for the 24 F4 evaluation
sets (and the 24 BASE ones when SCORE_BASE_ARM is on).

Kaggle: CPU (enable_gpu false), enable_internet true (the scorer is
cloned), one line per dataset, budget guard at BUDGET_S. Expected ~95
min: features ~28 min (v1: 7.0 min for 56 sets), stage B ~66 min (v1:
109 min for 5 arms x 24; here 3 x 24). Prints `EDGEPROB-RANK OK` only
when every arm scored every dataset and the hard anchors held.

Attach (kernel-metadata.json): dataset_sources
pilkwang/biohub-tracking-support-pack-50ep-v1 and faustine649/cell-tracking
(optionally faustine649/biohub-dctrain-175-weights for the manifest
cross-check); kernel_sources faustine649/biohub-edgeprob (v3: 199 BASE +
24 F4 caches), faustine649/biohub-edge-ranker, faustine649/biohub-cache-train
(the 199 detcaches), faustine649/biohub-poolsize (the two pkls),
faustine649/biohub-fusecache (fdet_F4_ caches);
competition_sources biohub-cell-tracking-during-development (GT + scorer).
Push with `py kag.py kernels push -p .` (the connector drops sources).
"""

from __future__ import annotations

import gc
import json
import os
import pathlib
import pickle
import random
import subprocess
import sys
import time
from collections import defaultdict

SCRIPT_VERSION = "2026-09-28a-ncsweep"  # forkprop was "2026-09-17a-forkprop"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

HOLDOUT_6BBA = [
    "6bba_268e1230", "6bba_2312ac41", "6bba_283bf9f1", "6bba_3a1849c2",
    "6bba_3abfe10a", "6bba_5c824876", "6bba_61dd1e0d", "6bba_7af54fde",
    "6bba_7b5d3b2c", "6bba_afb141ff", "6bba_aeee7805", "6bba_c27cba08",
]
HOLDOUT = HOLDOUT_6BBA
ILP_DISAPP = 1.5
ILP_FLOOR = 0.10                            # candidates fed to the ILP
GATE_UM = 10.0
MATCH_UM = 5.0
DIV_W = 0.55

SEED_44B6 = 20260827    # dctrain.py:158 -- random, seeded, outside D32
N_44B6 = 12
N_POOL32 = 32
N_POOL175 = 175         # dctrain's pool: 199 - 12 (6bba) - 12 (44b6)

# Score the BASE primaries too (v1's arm)? Doubles stage B (~+66 min).
SCORE_BASE_ARM = False

# WHICH detection cache the "F4" path reads. "F4" is divbench9 verbatim and
# carries the hard anchor; "F4HP" swaps in hparacache's hpara placement of the
# SAME detections and is expected to move the anchors -- that movement is the
# measurement. See the module docstring of make_hparabench.py.
DET_SRC = "F4HP"
assert DET_SRC in ("F4", "F4HP"), DET_SRC
print(f"DET_SRC = {DET_SRC}", flush=True)

# Recorded anchors, see the docstring. Keys: (arm, det_arm, group).
ANCHOR_HARD = {("ANCHOR_D187", "F4", "6bba"): 0.9216}
ANCHOR_SOFT = {("ANCHOR_D187", "F4", "44b6"): 0.8353,
               ("+BIDIR", "F4", "6bba"): 0.9261,     # v2, retrained here
               ("+BIDIR", "F4", "44b6"): 0.8515}
ANCHOR_DIV = {("ANCHOR_D187", "F4", "6bba"): 0.0606,     # printed, warned
              ("ANCHOR_D187", "F4", "44b6"): 0.0000,
              ("+BIDIR", "F4", "6bba"): 0.1515,
              ("+BIDIR", "F4", "44b6"): 0.0000}
if SCORE_BASE_ARM:
    ANCHOR_HARD[("ANCHOR_D187", "BASE", "6bba")] = 0.9293
    ANCHOR_SOFT[("ANCHOR_D187", "BASE", "44b6")] = 0.8185
if DET_SRC == "F4HP":
    # The whole point of the run is that these numbers move. Asserting them
    # would abort the measurement; dropping them silently would hide a
    # harness change, so they are demoted to SOFT and still printed.
    ANCHOR_SOFT.update(ANCHOR_HARD)
    ANCHOR_HARD = {}
    print("  DET_SRC=F4HP: the F4 anchors are demoted to soft. A large move "
          "in ANCHOR_D187-6bba (0.9216) is the measurement; a move of more "
          "than a couple of points is more likely a broken join -- check "
          "the JOIN counters before believing it.", flush=True)
ANCHOR_TOL = 0.0005

BUDGET_S = 10.0 * 3600  # Kaggle kills a CPU kernel at 12 h; stop cleanly first


# ---- [assembled from rankwhy.py: A preamble] ----
# chimera-style environment: pack wheels (pyscipopt for the ILP) + the
# real tracking_cellmot from a fresh clone (scoring), --no-deps so pip
# cannot half-upgrade numpy (toolchain trap).
r = subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/royerlab/"
                    "kaggle-cell-tracking-competition.git",
                    "/kaggle/working/repo"], capture_output=True, text=True)
print(f"git clone exit {r.returncode}", flush=True)

import shutil                                                  # noqa: E402
PACK = pathlib.Path("/kaggle/input/datasets/pilkwang/"
                    "biohub-tracking-support-pack-50ep-v1")
if not PACK.exists():
    PACK = pathlib.Path("/kaggle/input/biohub-tracking-support-pack-50ep-v1")
assert PACK.exists(), "support pack not attached"
KEEP_OUT = {"numpy", "scipy", "torch", "pandas", "pyarrow", "numba",
            "llvmlite", "matplotlib", "pillow", "networkx", "scikit-learn",
            "scikit-image", "setuptools", "wheel", "pip"}
WH = pathlib.Path("/kaggle/working/wheelhouse")
shutil.rmtree(WH, ignore_errors=True)
WH.mkdir(parents=True)
for w in sorted(PACK.rglob("*.whl")):
    if w.name.split("-")[0].replace("_", "-").lower() not in KEEP_OUT:
        shutil.copy(w, WH / w.name)
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "--no-index", "--no-deps"]
                   + [str(w) for w in sorted(WH.glob("*.whl"))],
                   capture_output=True, text=True)
print(f"pip wheels exit {r.returncode}", flush=True)
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "--no-deps", "-e", "/kaggle/working/repo"],
                   capture_output=True, text=True)
print(f"pip -e repo exit {r.returncode}", flush=True)
for p in (PACK / "repo" / "src", PACK / "repo" / "scripts"):
    sys.path.insert(0, str(p))
sys.path.insert(0, "/kaggle/working/repo/src")

import numpy as np                                             # noqa: E402
from scipy.optimize import linear_sum_assignment               # noqa: E402
from scipy.spatial import cKDTree                              # noqa: E402
import tracksdata as td                                        # noqa: E402
import predict_unet_transformer as PIPE                         # noqa: E402
from tracking_cellmot.io import open_dataset                   # noqa: E402
from tracking_cellmot.metrics import (                         # noqa: E402
    evaluate as compute_metric, node_recall, per_sample_metrics, summarise)
from sklearn.ensemble import HistGradientBoostingClassifier    # noqa: E402
from sklearn.metrics import roc_auc_score                      # noqa: E402

CODE_ROOTS = [
    "/kaggle/input/datasets/faustine649/cell-tracking/cell tracking",
    "/kaggle/input/datasets/faustine649/cell-tracking",
    "/kaggle/input/cell-tracking/cell tracking",
    "/kaggle/input/cell-tracking",
]
code = next((c for c in CODE_ROOTS
             if pathlib.Path(c, "division_recovery.py").exists()), None)
assert code is not None
sys.path.insert(0, code)
import division_recovery as dr                                 # noqa: E402
from viz_cells import (find_root, geff_path, read_geff,        # noqa: E402
                       list_datasets)

# ---- [assembled from rankwhy.py: B constants] ----
SCALE = np.array([1.625, 0.40625, 0.40625])
UNET_THRESH = 0.15
CAND_THR = 0.05
SHORT_MIN = 4            # SHIPPED value (short-sweep, submitted); no sweep here
LINEFIT_WEIGHT = 0.8     # REBOUND per configuration
LINEFIT_WINDOW = 2       # REBOUND per configuration
MAX_LINK_UM = 10.0
TIGHT_UM = 6.0
LINK_SIM_W = 2.0
GAP1_GATE_UM = 9.0       # REBOUND per configuration
GAP1_SNAP_UM = 3.2       # REBOUND per configuration
GAP1_MIN_CAND_SCORE = 0.10   # REBOUND per configuration
GAP1_CAP_FRAC = 0.02     # SHIPPED (cap20, ref 55817630)
# Shipped behaviour: bridge ONE missing frame, and only onto a real
# candidate from the repair band. Both are swept below; BASE must
# reproduce the recorded anchor with these values, which is what
# proves the rewritten _gap_close did not change behaviour.
GAP_SPAN = 1
GAP_NEED_CAND = False    # SHIPPED (nocand, ref 55817630)
# ---- the gates. Each is OFF at its neutral value, so BASE is unchanged.
# GAP_VEL_COS  minimum cosine between the incoming velocity at the gap's
#              start and the outgoing velocity at its end. -1.0 accepts
#              anything, which is the shipped behaviour.
# GAP_MIN_LEN  minimum node count of BOTH fragments being joined. 0 is off.
# GAP_CLEAR_UM a bridge is refused if a PRIMARY node already sits this
#              close to the interpolated position. 0.0 is off.
# GAP_LEN_MODE decides WHICH bridges GAP_MIN_LEN is allowed to judge.
#   "always"     every bridge. gap3's len4/len8 arms, and the reason
#                they delete bridges a real detected peak was backing.
#   "if_nocand"  only bridges with no candidate behind them. The user's
#                design: the candidate IS the evidence, the length
#                requirement is only the fallback for when there is none.
GAP_LEN_MODE = "always"
# ---- the second detector ------------------------------------------------
# None is OFF, and BASE is then byte-identical to cap20.
DC_TAU = None
DC_MERGE_UM = 4.0          # same separation our own NMS enforces
DC_ADD_SCORE = 0.12        # repair band only: >= MIN_CAND 0.10, < THRESH 0.15
_DC_STATS = [0, 0, 0]      # frames, xl pool points, DC points ADDED to pool
from collections import defaultdict as _dd
_SNAP_STATS = _dd(lambda: [0, 0, 0])   # DC_TAU -> [midpoint, xl snap, DC snap]
_CAND_IS_DC = None         # per-frame bool arrays, set by the arm loop
GAP_VEL_COS = -1.0
GAP_MIN_LEN = 0
GAP_CLEAR_UM = 0.0    # REBOUND per configuration
DIV_PARENT_UM = 12.0
DIV_SISTER_UM = 15.0
DIV_CHILD_UM = 10.0
DIV_DIVERGE_UM = 2.25
DIV_W_SISTER = 0.15
DIV_FRAME_CAP = 0.0076
DIV_GLOBAL_CAP = 0.00375

# ---- [assembled from rankwhy.py: C detcaches] ----
# ---- detection caches from the v2 kernel output -------------------------
BASE = pathlib.Path("/kaggle/input")
CACHES = {}
stack = [BASE]
while stack:
    d = stack.pop()
    if d.name.endswith((".zarr", ".geff")) \
            or "cell-tracking-during-development" in d.name:
        continue
    for f in d.glob("detcache_*.npz"):
        CACHES[f.stem[len("detcache_"):]] = f
    if len(d.parts) - len(BASE.parts) < 4:
        stack += [p for p in d.iterdir() if p.is_dir()]
print(f"detection caches found: {len(CACHES)}", flush=True)
assert len(CACHES) >= 40, "attach biohub-edge-ranker as kernel source"

# ---- [assembled from rankwhy.py: D plumbing] ----
def _logit(s):
    s = np.clip(s, 1e-4, 1 - 1e-4)
    return np.log(s / (1 - s))


def seeds_of(dets_t):
    coords, scores = dets_t
    coords = np.asarray(coords, float).reshape(-1, 3)
    scores = np.asarray(scores, float).reshape(-1)
    m = scores >= UNET_THRESH
    return coords[m], scores[m]


def _link_sim(prev_xyz, curr_xyz, prev_vel, prev_sc, curr_sc):
    if len(prev_xyz) == 0 or len(curr_xyz) == 0:
        return []
    P = prev_xyz * SCALE[None, :]
    C = curr_xyz * SCALE[None, :]
    pred = P + (0.5 * prev_vel if prev_vel is not None else 0.0)
    N, M = len(P), len(C)
    BIG = 1e9
    sim = LINK_SIM_W * np.abs(_logit(prev_sc)[:, None] - _logit(curr_sc)[None])

    def _hun(pi, ci, gate):
        if len(pi) == 0 or len(ci) == 0:
            return []
        Draw = np.sqrt(((P[pi][:, None] - C[ci][None]) ** 2).sum(2))
        D = np.sqrt(((pred[pi][:, None] - C[ci][None]) ** 2).sum(2))
        cost = np.where(Draw > gate, BIG, D) + \
            np.where(Draw > gate, 0.0, sim[np.ix_(pi, ci)])
        ri, rc = linear_sum_assignment(cost)
        return [(int(pi[r]), int(ci[c]))
                for r, c in zip(ri, rc) if cost[r, c] < BIG]

    links = _hun(np.arange(N), np.arange(M), min(TIGHT_UM, MAX_LINK_UM))
    up = {p for p, _ in links}
    uc = {c for _, c in links}
    fp = np.array([i for i in range(N) if i not in up], int)
    fc = np.array([j for j in range(M) if j not in uc], int)
    return links + _hun(fp, fc, MAX_LINK_UM)


def pass1_velocities(dets):
    T = len(dets)
    vels = [np.zeros((len(seeds_of(dets[t])[0]), 3)) for t in range(T)]
    for t in range(T - 1):
        P, ps = seeds_of(dets[t])
        C, cs = seeds_of(dets[t + 1])
        if len(P) == 0 or len(C) == 0:
            continue
        for pi, ci in _link_sim(P, C, vels[t] if len(vels[t]) else None,
                                ps, cs):
            vels[t + 1][ci] = (C[ci] - P[pi]) * SCALE
    return vels


def pair_features(dets, vels, T):
    out = []
    for t in range(T - 1):
        P, ps = seeds_of(dets[t])
        C, cs = seeds_of(dets[t + 1])
        if len(P) == 0 or len(C) == 0:
            out.append(([], np.zeros((0, 18), np.float32), P, C))
            continue
        Pu = P * SCALE[None, :]
        Cu = C * SCALE[None, :]
        tree_p = cKDTree(Pu)
        tree_c = cKDTree(Cu)
        # gate query only: shrink z by GATE_Z/GATE_XY so a ball of
        # radius GATE_XY in this space IS the ellipsoid we want. The
        # distance FEATURE below stays physical and unchanged.
        _an = GATE_Z / GATE_XY
        _Sq = np.array([SCALE[0] / _an, SCALE[1], SCALE[2]])
        Pq = P * _Sq[None, :]
        Cq = C * _Sq[None, :]
        tree_q = cKDTree(Cq)
        dens_p = np.array([len(tree_p.query_ball_point(x, 7.0)) - 1
                           for x in Pu])
        dens_c = np.array([len(tree_c.query_ball_point(x, 7.0)) - 1
                           for x in Cu])
        pred = Pu + vels[t]
        pairs = []
        rows = []
        cand_of_i = defaultdict(list)
        cand_of_j = defaultdict(list)
        raw = {}
        for i, x in enumerate(Pu):
            for j in tree_q.query_ball_point(Pq[i], GATE_XY):
                d = float(np.linalg.norm(Cu[j] - x))
                raw[(i, j)] = d
                cand_of_i[i].append((d, j))
                cand_of_j[j].append((d, i))
        for i in cand_of_i:
            cand_of_i[i].sort()
        for (i, j), d in raw.items():
            delta = Cu[j] - Pu[i]
            md = float(np.linalg.norm(Cu[j] - pred[i]))
            rank = next(k for k, (_, jj) in enumerate(cand_of_i[i])
                        if jj == j)
            rows.append([d, delta[0], abs(delta[0]),
                         float(np.linalg.norm(delta[1:])), md, d - md,
                         ps[i], cs[j],
                         abs(float(_logit(np.array([ps[i]]))[0]
                                   - _logit(np.array([cs[j]]))[0])),
                         dens_p[i], dens_c[j], len(P), len(C),
                         rank, len(cand_of_i[i]), len(cand_of_j[j]),
                         float(np.linalg.norm(vels[t][i])), t / max(1, T)])
            pairs.append((i, j))
        out.append((pairs, np.asarray(rows, np.float32), P, C))
    return out


root = find_root(None)

GATE_XY = 10.0
GATE_Z = 10.0
# ---- graph plumbing (xl pipeline tail) ---------------------------------
def _dist_um(a, b):
    d = (np.asarray(a, float) - np.asarray(b, float)) * SCALE
    return float(np.sqrt((d * d).sum()))


def _short_filter(nodes, edges):
    parent = {nid: nid for nid in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    out_count = defaultdict(int)
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
        out_count[a] += 1
    comps = defaultdict(list)
    for nid in nodes:
        comps[find(nid)].append(nid)
    keep = set()
    for members in comps.values():
        if len(members) >= SHORT_MIN or \
                any(out_count[n] >= 2 for n in members):
            keep.update(members)
    nodes2 = {nid: n for nid, n in nodes.items() if nid in keep}
    edges2 = [(a, b) for a, b in edges if a in nodes2 and b in nodes2]
    return nodes2, edges2


def _linefit(nodes, edges):
    pred = defaultdict(list)
    succ = defaultdict(list)
    for a, b in edges:
        if int(nodes[b]["t"]) == int(nodes[a]["t"]) + 1:
            succ[a].append(b)
            pred[b].append(a)
    orig = {k: v["xyz"].copy() for k, v in nodes.items()}
    updates = {}
    for nid in nodes:
        neigh = [(0, nid)]
        cur = nid
        for step in range(1, LINEFIT_WINDOW + 1):
            ps = pred.get(cur, [])
            if len(ps) != 1:
                break
            cur = ps[0]
            neigh.append((-step, cur))
        cur = nid
        for step in range(1, LINEFIT_WINDOW + 1):
            ss = succ.get(cur, [])
            if len(ss) != 1:
                break
            cur = ss[0]
            neigh.append((step, cur))
        if len(neigh) < 3:
            continue
        dt = np.array([a for a, _ in neigh], float)
        xyz = np.stack([orig[n] for _, n in neigh])
        fit = np.array([np.polyval(np.polyfit(dt, xyz[:, ax], 1), 0.0)
                        for ax in range(3)])
        if np.isfinite(fit).all():
            updates[nid] = (1 - LINEFIT_WEIGHT) * orig[nid] + \
                LINEFIT_WEIGHT * fit
    for nid, xyz in updates.items():
        nodes[nid]["xyz"] = xyz


_PRIM_TREES = {}

# made / allowed / demand-seen / times the cap bound, keyed by the arm's
# own constants so no tag has to be plumbed through the sweep loop.
# gap3 printed none of this, which is why it could not tell "len8 picks
# better bridges" apart from "len8 shrinks demand below the budget".
_GAP_LOG = defaultdict(lambda: [0, 0, 0, 0])


def _prim_tree(t, by_t, nodes):
    """KD-tree of the PRIMARY nodes at frame t, in micrometres. Cached
    per call of _gap_close -- the caller clears it."""
    if t in _PRIM_TREES:
        return _PRIM_TREES[t]
    ids = by_t.get(t, [])
    tr = None
    if ids:
        tr = cKDTree(np.asarray([nodes[g]["xyz"] for g in ids], float)
                     * SCALE)
    _PRIM_TREES[t] = tr
    return tr


def _frag_len(nid, link, back, cap=12):
    """How many nodes the fragment containing `nid` runs for, walking
    backward through `pred` or forward through `succ`. Capped so a very
    long track costs nothing to measure."""
    n = 1
    cur = nid
    while n < cap:
        nxt = link.get(cur)
        if nxt is None:
            break
        cur = nxt
        n += 1
    return n


def _br_node_score(g):
    so = BR_CTX.get("seed_of_node")
    if so is None:
        return -1.0
    s = so.get(g)
    if s is None:
        return -1.0          # an inserted node has no detection score
    t_, i_ = s
    return float(BR_CTX["seeds"][t_][1][i_])


def _br_frame_trees(t, by_t, nodes, outdeg):
    """(primary tree, single-child-node tree) at frame t, cached per call."""
    key = ("br", t)
    if key in _PRIM_TREES:
        return _PRIM_TREES[key]
    ids = by_t.get(t, [])
    tr = cKDTree(np.asarray([nodes[g]["xyz"] for g in ids], float) * SCALE) if ids else None
    m_ids = [g for g in ids if outdeg.get(g, 0) == 1]
    mt = cKDTree(np.asarray([nodes[g]["xyz"] for g in m_ids], float) * SCALE) if m_ids else None
    _PRIM_TREES[key] = (tr, mt)
    return _PRIM_TREES[key]


def _br_feats(t, step, ends, starts, P, C, D, nodes, succ, pred, by_t, outdeg,
              cand, cand_sc, cand_trees, T):
    """Feature rows for every (end, start) pair with D[r, c] <= GAP1_GATE_UM
    (D is already divided by span). Returns (pairs [(r, c)], rows)."""
    span = step - 1
    pairs, rows = [], []
    n_starts_near = (D <= GAP1_GATE_UM).sum(1)
    n_ends_near = (D <= GAP1_GATE_UM).sum(0)
    order = np.argsort(D, axis=1)
    for r_, ge in enumerate(ends):
        row_d = D[r_]
        ok = np.where(row_d <= GAP1_GATE_UM)[0]
        if len(ok) == 0:
            continue
        a = P[r_]
        pa = pred.get(ge)
        v_in = a - np.asarray(nodes[pa]["xyz"], float) * SCALE if pa is not None else np.zeros(3)
        nv_in = float(np.linalg.norm(v_in))
        len_end = _frag_len(ge, pred, back=True)
        end_is_daughter = 1.0 if (pa is not None and outdeg.get(pa, 0) >= 2) else 0.0
        tr_t, _ = _br_frame_trees(t, by_t, nodes, outdeg)
        dens = float(len(tr_t.query_ball_point(a, 7.0)) - 1) if tr_t is not None else 0.0
        s_end = _br_node_score(ge)
        srt = order[r_]
        d_second = float(row_d[srt[1]] * span) if len(srt) > 1 else 99.0
        for c_ in ok:
            gs = starts[c_]
            b = C[c_]
            delta = b - a
            d = float(np.linalg.norm(delta))
            sb = succ.get(gs)
            v_out = np.asarray(nodes[sb]["xyz"], float) * SCALE - b if sb is not None else np.zeros(3)
            nv_out = float(np.linalg.norm(v_out))
            cos_io = float(v_in @ v_out) / (nv_in * nv_out) if nv_in > 1e-6 and nv_out > 1e-6 else 0.0
            cos_id = float(v_in @ delta) / (nv_in * d) if nv_in > 1e-6 and d > 1e-6 else 0.0
            err_in = float(np.linalg.norm(delta - v_in * step))
            err_out = float(np.linalg.norm(delta - v_out * step))
            err_mean = float(np.linalg.norm(delta - 0.5 * (v_in + v_out) * step))
            len_start = _frag_len(gs, succ, back=False)
            rank = int(np.where(srt == c_)[0][0])
            clear_min, n_hits, sc_max = 99.0, 0, 0.0
            for k in range(1, step):
                frac = k / float(step)
                m_ = (1.0 - frac) * a + frac * b
                tt = t + k
                trk, _ = _br_frame_trees(tt, by_t, nodes, outdeg)
                if trk is not None:
                    clear_min = min(clear_min, float(trk.query(m_)[0]))
                tree = cand_trees[tt]
                if tree is not None:
                    dist, idx = tree.query(m_)
                    if dist <= GAP1_SNAP_UM and cand_sc[tt][idx] >= GAP1_MIN_CAND_SCORE:
                        n_hits += 1
                        sc_max = max(sc_max, float(cand_sc[tt][idx]))
            _, mt = _br_frame_trees(t + step - 1, by_t, nodes, outdeg)
            n_moth, d_moth = 0, 99.0
            if mt is not None:
                near = mt.query_ball_point(b, DIV_PARENT_UM)
                n_moth = len(near)
                if n_moth:
                    d_moth = float(mt.query(b)[0])
            rows.append([span, d, d / span, abs(float(delta[0])), float(np.linalg.norm(delta[1:])),
                         nv_in, nv_out, cos_io, cos_id, err_in, err_out, err_mean,
                         len_end, len_start, float(n_starts_near[r_]), float(n_ends_near[c_]),
                         rank, d_second, clear_min, n_hits, sc_max, n_moth, d_moth,
                         end_is_daughter, s_end, _br_node_score(gs), dens, t / max(1, T)])
            pairs.append((r_, c_))
    return pairs, np.asarray(rows, np.float32).reshape(-1, len(BR_NAMES))


def _br_label(t, step, ge, gs, nodes):
    """1 if the start's GT node is a descendant of the end's GT node at
    exactly `step` generations, 0 if both match GT and it is not, -1 if
    either endpoint has no GT node within MATCH_UM."""
    gt = BR_CTX.get("gt")
    if gt is None:
        return -1
    trees, kids = gt
    if t not in trees or (t + step) not in trees:
        return -1
    rows_a, tr_a = trees[t]
    rows_b, tr_b = trees[t + step]
    da, ia = tr_a.query(np.asarray(nodes[ge]["xyz"], float) * SCALE)
    db, ib = tr_b.query(np.asarray(nodes[gs]["xyz"], float) * SCALE)
    if da > MATCH_UM or db > MATCH_UM:
        return -1
    gen = [int(rows_a[int(ia)])]
    for _ in range(step):
        gen = [c for p_ in gen for c in kids.get(p_, [])]
        if not gen:
            return 0
    return 1 if int(rows_b[int(ib)]) in gen else 0


def _br_outdeg(edges):
    od = defaultdict(int)
    for a_, _b in edges:
        od[a_] += 1
    return od


def _br_collect(T, nodes, succ, pred, by_t, edges, cand, cand_sc, cand_trees):
    """Pass 1: after the shipped span-1 pass, record every span-2 and span-3
    pair with its features and GT label."""
    outdeg = _br_outdeg(edges)
    for span in (2, 3):
        step = span + 1
        for t in range(T - step):
            ends = [g for g in by_t.get(t, []) if g not in succ and g in pred]
            starts = [g for g in by_t.get(t + step, []) if g not in pred and g in succ]
            if not ends or not starts:
                continue
            P = np.asarray([nodes[g]["xyz"] for g in ends], float) * SCALE
            C = np.asarray([nodes[g]["xyz"] for g in starts], float) * SCALE
            D = np.sqrt(((P[:, None] - C[None]) ** 2).sum(2)) / float(span)
            pairs, rows = _br_feats(t, step, ends, starts, P, C, D, nodes, succ, pred,
                                    by_t, outdeg, cand, cand_sc, cand_trees, T)
            for (r_, c_), row in zip(pairs, rows):
                lab = _br_label(t, step, ends[r_], starts[c_], nodes)
                BR_STAT[f"collect_span{span}_pairs"] += 1
                if lab >= 0:
                    BR_STAT[f"collect_span{span}_labelled"] += 1
                    BR_STAT[f"collect_span{span}_pos"] += lab
                BR_ROWS.append((BR_CTX["name"], BR_CTX["fold"], span, lab, row))


def _br_cost(t, step, ends, starts, P, C, D, cost, nodes, succ, pred, by_t, edges,
             cand, cand_sc, cand_trees, T, BIG):
    """Pass 2, learned arms: replace the distance cost for span >= 2 by
    1 - p over pairs with p >= BR_TAU; everything else is BIG."""
    span = step - 1
    outdeg = _br_outdeg(edges)
    pairs, rows = _br_feats(t, step, ends, starts, P, C, D, nodes, succ, pred, by_t,
                            outdeg, cand, cand_sc, cand_trees, T)
    new = np.full(cost.shape, BIG)
    if not pairs:
        return new
    p = BR_MODEL.predict_proba(rows)[:, 1]
    BR_STAT[f"span{span}_proposals"] += len(pairs)
    for (r_, c_), p_ in zip(pairs, p):
        if p_ >= BR_TAU:
            new[r_, c_] = 1.0 - float(p_)
            BR_STAT[f"span{span}_over_tau"] += 1
    return new


def _gap_close(nodes, edges, cand, cand_sc, cand_trees, next_id):
    """Bridge a track break by inserting up to GAP_SPAN nodes.

    Rewritten from the version that only ever bridged ONE missing frame
    (`ends` at t joined to `starts` at t+2). Two behaviours are now
    parameters, because flicker.py showed both matter and neither had
    ever been tested:

      GAP_SPAN        how many consecutive missing frames to bridge.
                      1 reproduces the shipped behaviour exactly.
      GAP_NEED_CAND   whether every inserted node must land on a real
                      candidate from the repair band. The shipped code
                      required it. At the missed frames 79.8% have NO
                      candidate within 5 um, so requiring it caps the
                      repair at about a fifth of dropouts, and a
                      two-frame bridge needing one at both frames would
                      fire on roughly 4%.

    With GAP_SPAN=1 and GAP_NEED_CAND=True this must be byte-identical
    in behaviour to the original -- that is the BASE arm's job to prove.
    """
    T = len(cand)
    _PRIM_TREES.clear()
    succ = {}
    pred = {}
    for a, b in edges:
        succ[a] = b
        pred[b] = a
    by_t = defaultdict(list)
    for nid, n in nodes.items():
        by_t[n["t"]].append(nid)
    cand_used = [np.zeros(len(c), bool) for c in cand]
    max_gaps = max(1, int(GAP1_CAP_FRAC * len(nodes)))
    n_added = 0
    n_demand = 0
    BIG = 1e9
    # Shortest gaps first. A cell missing one frame should be repaired by
    # the confident one-frame rule before a longer, weaker bridge is
    # allowed to consume the same endpoint.
    for span in range(1, GAP_SPAN + 1):
        step = span + 1
        for t in range(T - step):
            if n_added >= max_gaps:
                break
            ends = [g for g in by_t.get(t, [])
                    if g not in succ and g in pred]
            starts = [g for g in by_t.get(t + step, [])
                      if g not in pred and g in succ]
            if not ends or not starts:
                continue
            P = np.asarray([nodes[g]["xyz"] for g in ends], float) * SCALE
            C = np.asarray([nodes[g]["xyz"] for g in starts],
                           float) * SCALE
            D = np.sqrt(((P[:, None] - C[None]) ** 2).sum(2))
            # A longer bridge should be allowed proportionally more total
            # travel, otherwise a 2-frame bridge held to a 1-frame budget
            # would essentially never fire and the run would be testing
            # the gate rather than the span.
            #
            # ⚠ v1 WROTE `D = D / float(step)` AND THAT BROKE THE BASE
            # ARM. `step` is `span + 1`, so at span 1 -- the shipped
            # configuration -- it divided by 2 and silently doubled
            # GAP1_GATE_UM from 9 to 18 um. BASE then reproduced
            # biohub-poolsize's D187 row at adj_J 0.9253 / div_J 0.1282
            # against the recorded 0.9254 / 0.1316, which is exactly the
            # kind of small drift a must-reproduce anchor exists to
            # catch. The SHORT_MIN deltas survived it because the same
            # loosened gate sat under both halves and cancelled in the
            # difference; the GAP arms did not, because they compare
            # different spans and therefore different gates.
            #
            # Normalise by SPAN, not step: span 1 divides by 1 and is
            # byte-identical to the shipped code.
            D = D / float(span)
            cost = np.where(D > GAP1_GATE_UM, BIG, D)
            if span >= 2 and BR_MODEL is not None:
                cost = _br_cost(t, step, ends, starts, P, C, D, cost, nodes, succ,
                                pred, by_t, edges, cand, cand_sc, cand_trees, T, BIG)
            ri, ci = linear_sum_assignment(cost)
            for r_, c_ in zip(ri, ci):
                if n_added >= max_gaps or cost[r_, c_] >= BIG:
                    continue
                ge, gs = ends[r_], starts[c_]
                if ge in succ or gs in pred:
                    continue
                a_xyz = nodes[ge]["xyz"]
                b_xyz = nodes[gs]["xyz"]
                n_demand += 1

                # ---- the gates ---------------------------------------
                # Each is skipped at its neutral value so BASE is
                # byte-identical to the shipped behaviour.
                if GAP_VEL_COS > -1.0:
                    pa, sb = pred.get(ge), succ.get(gs)
                    if pa is None or sb is None:
                        continue
                    v_in = (a_xyz - nodes[pa]["xyz"]) * SCALE
                    v_out = (nodes[sb]["xyz"] - b_xyz) * SCALE
                    na, nb = np.linalg.norm(v_in), np.linalg.norm(v_out)
                    if na < 1e-6 or nb < 1e-6:
                        # a stationary endpoint carries no direction, so
                        # this gate cannot judge it. Let it through
                        # rather than silently rejecting every still cell.
                        pass
                    elif float(v_in @ v_out) / (na * nb) < GAP_VEL_COS:
                        continue
                if GAP_CLEAR_UM > 0.0:
                    blocked = False
                    for k in range(1, step):
                        frac = k / float(step)
                        m_ = ((1.0 - frac) * a_xyz + frac * b_xyz) * SCALE
                        tr_ = _prim_tree(t + k, by_t, nodes)
                        if tr_ is not None and \
                                tr_.query(m_)[0] <= GAP_CLEAR_UM:
                            blocked = True
                            break
                    if blocked:
                        continue

                # resolve every inserted position BEFORE mutating, so a
                # bridge that fails halfway leaves nothing behind
                picks = []
                for k in range(1, step):
                    frac = k / float(step)
                    mid = (1.0 - frac) * a_xyz + frac * b_xyz
                    tt = t + k
                    tree = cand_trees[tt]
                    hit = None
                    if tree is not None:
                        dist, idx = tree.query(mid * SCALE)
                        if dist <= GAP1_SNAP_UM \
                                and not cand_used[tt][idx] \
                                and cand_sc[tt][idx] >= GAP1_MIN_CAND_SCORE:
                            hit = idx
                    picks.append((tt, hit, mid))
                n_missing = sum(1 for _, _h, _ in picks if _h is None)

                # ---- the candidate requirement -----------------------
                if GAP_NEED_CAND and n_missing:
                    continue

                # ---- the length gate, moved to AFTER the lookup ------
                # gap3 ran this BEFORE resolving the candidate, which
                # cannot express "only when there is no candidate". The
                # lookup has no side effects (cand_used is marked in the
                # insert loop below, not above), so a bridge rejected
                # here leaves nothing behind and "always" reproduces
                # gap3 exactly. The three anchor arms test that.
                if GAP_MIN_LEN > 0 and (GAP_LEN_MODE == "always"
                                        or n_missing):
                    if _frag_len(ge, pred, back=True) < GAP_MIN_LEN or \
                            _frag_len(gs, succ, back=False) < GAP_MIN_LEN:
                        continue
                prev = ge
                for tt, hit, mid in picks:
                    _sn = _SNAP_STATS[DC_TAU]
                    if hit is None:
                        use = np.asarray(mid, float)
                        _sn[0] += 1
                    else:
                        use = np.asarray(cand[tt][hit], float)
                        cand_used[tt][hit] = True
                        if _CAND_IS_DC is not None \
                                and bool(_CAND_IS_DC[tt][hit]):
                            _sn[2] += 1
                        else:
                            _sn[1] += 1
                    ng = next_id
                    next_id += 1
                    nodes[ng] = {"t": tt, "xyz": use}
                    edges.append((prev, ng))
                    succ[prev] = ng
                    pred[ng] = prev
                    prev = ng
                edges.append((prev, gs))
                succ[prev] = gs
                pred[gs] = prev
                n_added += 1
                BR_STAT[f"span{span}_bridged"] += 1
    if BR_COLLECT:
        _br_collect(T, nodes, succ, pred, by_t, edges, cand, cand_sc, cand_trees)
    _r = _GAP_LOG[(GAP_NEED_CAND, GAP_SPAN, round(GAP1_GATE_UM, 2),
                   round(GAP1_CAP_FRAC, 5))]
    _r[0] += n_added
    _r[1] += max_gaps
    _r[2] += n_demand
    # ⚠ n_demand stops accumulating once the cap binds, because the
    # frame loop breaks there. So demand-seen UNDERSTATES true demand
    # for any capped arm -- read the cap20 arms for the real figure.
    _r[3] += 1 if n_added >= max_gaps else 0
    return next_id


STEAL_LOG = {}          # (p, q) -> dict, one per stolen daughter
# a defaultdict, not a dict: stage B clears it per arm so the counters can be
# attributed, and a cleared plain dict makes the next += a KeyError
PATCH_CAPS = defaultdict(int)


def _add_safe_divisions(nodes, edges, pfun=None, steal=None):
    """The orphan-fork patch, VERBATIM, plus an optional second pass that
    takes a daughter the ILP already gave to somebody else.

    `steal` is None (REF: exactly divbench3) or a float margin: the mother's
    ranker probability for the edge she would make must beat the current
    parent's by more than that. -inf runs it unconditionally.

    The orphan pass runs first and to completion, so REF behaviour is
    untouched by construction; the steal pass only sees mothers and
    daughters the orphan pass did not use, shares the per-frame cap, and has
    its own global cap of the same size.
    """
    succ = defaultdict(list)
    pred = defaultdict(list)
    for a, b in edges:
        succ[a].append(b)
        pred[b].append(a)
    by_t = defaultdict(list)
    for nid, n in nodes.items():
        by_t[n["t"]].append(nid)
    orphan_tree = {}
    orphan_ids = {}
    for t, ids in by_t.items():
        orph = [g for g in ids if not pred.get(g)]
        orphan_ids[t] = orph
        if orph:
            orphan_tree[t] = cKDTree(
                np.asarray([nodes[g]["xyz"] for g in orph], float) * SCALE)
    proposals = []
    for p, n in nodes.items():
        ch = succ.get(p, [])
        if len(ch) != 1:
            continue
        c1 = ch[0]
        t = n["t"]
        if nodes[c1]["t"] != t + 1 or len(pred.get(p, [])) != 1:
            continue
        if _dist_um(n["xyz"], nodes[c1]["xyz"]) > DIV_CHILD_UM:
            continue
        tree = orphan_tree.get(t + 1)
        if tree is None:
            continue
        for idx in tree.query_ball_point(n["xyz"] * SCALE, r=DIV_PARENT_UM):
            q = orphan_ids[t + 1][idx]
            if q == c1:
                continue
            d_pq = _dist_um(n["xyz"], nodes[q]["xyz"])
            d_s = _dist_um(nodes[c1]["xyz"], nodes[q]["xyz"])
            if d_s > DIV_SISTER_UM:
                continue
            d_q, q_idx = tree.query(nodes[c1]["xyz"] * SCALE)
            if orphan_ids[t + 1][q_idx] != q or d_q > DIV_SISTER_UM:
                continue
            c1n, qn = succ.get(c1, []), succ.get(q, [])
            if len(c1n) != 1 or len(qn) != 1:
                continue
            dv_ = _dist_um(nodes[c1n[0]]["xyz"], nodes[qn[0]]["xyz"]) - d_s
            if dv_ < DIV_DIVERGE_UM:
                continue
            if FP_RANK == "contrast":
                key_ = (-(fp_contrast(nodes, p) + fp_contrast(nodes, c1) + fp_contrast(nodes, q)), d_pq + DIV_W_SISTER * d_s)
            else:
                key_ = (d_pq + DIV_W_SISTER * d_s,)
            proposals.append((key_, t, p, q, c1, d_pq, d_s, dv_))
    proposals.sort()
    used_p = set()
    used_q = set()
    per_frame = defaultdict(int)
    n_frame = {t: len(ids) for t, ids in by_t.items()}
    max_global = max(1, int(DIV_GLOBAL_CAP * FP_CAP * len(nodes)))
    added = 0
    for score, t, p, q, c1, d_pq, d_s, dv_ in proposals:
        if added >= max_global:
            PATCH_CAPS["orphan_global"] += 1
            break
        if p in used_p or q in used_q:
            continue
        if per_frame[t] >= max(1, int(DIV_FRAME_CAP * FP_CAP * n_frame[t])):
            PATCH_CAPS["orphan_frame"] += 1
            continue
        FORK_LOG[(p, q)] = dict(c1=c1, d_pq=d_pq, d_s=d_s, dv=dv_,
                                n_orph=len(orphan_ids.get(t + 1, [])), steal=0)
        edges.append((p, q))
        succ[p].append(q)
        pred[q].append(p)
        used_p.add(p)
        used_q.add(q)
        per_frame[t] += 1
        added += 1
    PATCH_CAPS["orphan_added"] += added
    # ---- divadmit: second daughters from the primaries the graph dropped ----
    if DA_ADMIT and DA_CTX.get("seeds") is not None:
        seeds_da = DA_CTX["seeds"]
        son = DA_CTX["seed_of_node"]
        present = {son[g] for g in nodes if son.get(g) is not None}
        node_tree = {}
        for t, ids in by_t.items():
            node_tree[t] = cKDTree(np.asarray([nodes[g]["xyz"] for g in ids], float) * SCALE)
        drop = {}          # t -> (indices into seeds[t][0], coords in um, tree)
        for t in range(len(seeds_da)):
            c_ = np.asarray(seeds_da[t][0], float).reshape(-1, 3)
            if not len(c_):
                continue
            keep = np.array([i_ for i_ in range(len(c_)) if (t, i_) not in present], int)
            if not len(keep):
                continue
            cu = c_[keep] * SCALE
            drop[t] = (keep, cu, cKDTree(cu))
            PATCH_CAPS["admit_candidates"] += len(keep)
        aprop = []
        for p, n in nodes.items():
            ch = succ.get(p, [])
            if len(ch) != 1 or p in used_p:
                continue
            c1 = ch[0]
            t = n["t"]
            if nodes[c1]["t"] != t + 1 or len(pred.get(p, [])) != 1:
                continue
            if _dist_um(n["xyz"], nodes[c1]["xyz"]) > DIV_CHILD_UM:
                continue
            dr = drop.get(t + 1)
            if dr is None:
                continue
            keep, cu, tr = dr
            pu = np.asarray(n["xyz"], float) * SCALE
            c1u = np.asarray(nodes[c1]["xyz"], float) * SCALE
            hits = tr.query_ball_point(pu, r=DIV_PARENT_UM)
            if not hits:
                continue
            d_near, i_near = tr.query(c1u)
            if d_near > DIV_SISTER_UM or int(i_near) not in hits:
                continue
            hi = int(i_near)
            d_pq = float(np.linalg.norm(cu[hi] - pu))
            d_s = float(np.linalg.norm(cu[hi] - c1u))
            if d_s < DA_MIN_SEP_UM:
                PATCH_CAPS["admit_dup"] += 1
                continue
            d_any, _i = node_tree[t + 1].query(cu[hi])
            if d_any < DA_MIN_SEP_UM:
                PATCH_CAPS["admit_dup"] += 1
                continue
            aprop.append((d_pq + DIV_W_SISTER * d_s, t, p, (t + 1, int(keep[hi])), c1, d_pq, d_s))
        aprop.sort()
        PATCH_CAPS["admit_proposals"] += len(aprop)
        max_admit = max(1, int(DIV_GLOBAL_CAP * len(nodes)))
        nxt = (max(nodes) + 1) if nodes else 1
        used_seed = set()
        n_adm = 0
        for score, t, p, sq, c1, d_pq, d_s in aprop:
            if n_adm >= max_admit:
                PATCH_CAPS["admit_global"] += 1
                break
            if p in used_p or sq in used_seed:
                continue
            if per_frame[t] >= max(1, int(DIV_FRAME_CAP * n_frame[t])):
                PATCH_CAPS["admit_frame"] += 1
                continue
            xyz = np.round(np.asarray(seeds_da[sq[0]][0], float).reshape(-1, 3)[sq[1]]).astype(float)
            q = nxt
            nxt += 1
            nodes[q] = {"t": t + 1, "xyz": xyz}
            son[q] = sq
            DA_CTX["added"].append(q)
            by_t[t + 1].append(q)
            FORK_LOG[(p, q)] = dict(c1=c1, d_pq=d_pq, d_s=d_s, dv=float("nan"),
                                    n_orph=len(drop[t + 1][0]), steal=0, admit=1)
            edges.append((p, q))
            succ[p].append(q)
            pred[q].append(p)
            used_p.add(p)
            used_q.add(q)
            used_seed.add(sq)
            per_frame[t] += 1
            n_adm += 1
        PATCH_CAPS["admit_added"] += n_adm
    if steal is None:
        return added, 0

    # ---- second pass: the daughter the ILP already placed -----------------
    steal_ids, steal_tree, parent_of = {}, {}, {}
    for t, ids in by_t.items():
        cand = []
        for g in ids:
            pr = pred.get(g, [])
            if len(pr) != 1 or g in used_q:
                continue
            r = pr[0]
            if len(succ.get(r, [])) != 1:      # r already forks: leave it alone
                continue
            cand.append(g)
            parent_of[g] = r
        steal_ids[t] = cand
        if cand:
            steal_tree[t] = cKDTree(
                np.asarray([nodes[g]["xyz"] for g in cand], float) * SCALE)
    sprop = []
    for p, n in nodes.items():
        ch = succ.get(p, [])
        if len(ch) != 1 or p in used_p:
            continue
        c1 = ch[0]
        t = n["t"]
        if nodes[c1]["t"] != t + 1 or len(pred.get(p, [])) != 1:
            continue
        if _dist_um(n["xyz"], nodes[c1]["xyz"]) > DIV_CHILD_UM:
            continue
        tree = steal_tree.get(t + 1)
        if tree is None:
            continue
        for idx in tree.query_ball_point(n["xyz"] * SCALE, r=DIV_PARENT_UM):
            q = steal_ids[t + 1][idx]
            if q == c1 or parent_of.get(q) == p:
                continue
            d_pq = _dist_um(n["xyz"], nodes[q]["xyz"])
            d_s = _dist_um(nodes[c1]["xyz"], nodes[q]["xyz"])
            if d_s > DIV_SISTER_UM:
                continue
            # c1 is itself stealable, and it is its own nearest neighbour:
            # ask for the nearest stealable node that is not c1
            kk = min(2, len(steal_ids[t + 1]))
            dd, ii = tree.query(nodes[c1]["xyz"] * SCALE, k=kk)
            near, d_q = None, np.inf
            for d_, i_ in zip(np.atleast_1d(dd), np.atleast_1d(ii)):
                g_ = steal_ids[t + 1][int(i_)]
                if g_ != c1:
                    near, d_q = g_, float(d_)
                    break
            if near != q or d_q > DIV_SISTER_UM:
                continue
            c1n, qn = succ.get(c1, []), succ.get(q, [])
            if len(c1n) != 1 or len(qn) != 1:
                continue
            dv_ = _dist_um(nodes[c1n[0]]["xyz"], nodes[qn[0]]["xyz"]) - d_s
            if dv_ < DIV_DIVERGE_UM:
                continue
            sprop.append((d_pq + DIV_W_SISTER * d_s, t, p, q, c1, d_pq, d_s, dv_))
    sprop.sort()
    PATCH_CAPS["steal_proposals"] += len(sprop)
    stolen = 0
    max_steal = max(1, int(DIV_GLOBAL_CAP * len(nodes)))
    for score, t, p, q, c1, d_pq, d_s, dv_ in sprop:
        if stolen >= max_steal:
            PATCH_CAPS["steal_global"] += 1
            break
        if p in used_p or q in used_q:
            continue
        r = parent_of.get(q)
        if r is None or len(succ.get(r, [])) != 1 or succ[r][0] != q:
            continue
        p_pq = pfun(p, q) if pfun else float("nan")
        p_rq = pfun(r, q) if pfun else float("nan")
        have = bool(np.isfinite(p_pq) and np.isfinite(p_rq))
        if not have:
            PATCH_CAPS["steal_no_prob"] += 1
        if steal > float("-inf"):
            # a margin rule needs both numbers; the pair gate is GATE_UM 10 um
            # and the patch reaches 12, so some proposals simply have none
            if not have:
                continue
            if p_pq - p_rq <= steal:
                PATCH_CAPS["steal_margin"] += 1
                continue
        if per_frame[t] >= max(1, int(DIV_FRAME_CAP * n_frame[t])):
            PATCH_CAPS["steal_frame"] += 1
            continue
        FORK_LOG[(p, q)] = dict(c1=c1, d_pq=d_pq, d_s=d_s, dv=dv_,
                                n_orph=len(steal_ids.get(t + 1, [])), steal=1)
        STEAL_LOG[(p, q)] = dict(r=r, p_pq=float(p_pq), p_rq=float(p_rq),
                                 d_rq=_dist_um(nodes[r]["xyz"], nodes[q]["xyz"]))
        edges.remove((r, q))
        edges.append((p, q))
        succ[r].remove(q)
        pred[q] = [p]
        succ[p].append(q)
        used_p.add(p)
        used_q.add(q)
        per_frame[t] += 1
        stolen += 1
    PATCH_CAPS["steal_added"] += stolen
    return added, stolen
def _adj_of(row):
    """adj_edge_jaccard for ONE dataset row.

    `summarise` builds that column; nothing on record says
    `per_sample_metrics` also puts it on the row. Use the key when it is
    there and reconstruct it from the definition when it is not, rather
    than letting a KeyError kill a print block that runs after four
    hours of compute.
    """
    v = row.get("adj_edge_jaccard")
    if v is not None and v == v:
        return float(v)
    j = float(row.get("edge_jaccard", float("nan")))
    r = float(row.get("total_node_ratio", 0.0) or 0.0)
    return max(0.0, j * (1.0 - 0.1 * r))


def _with_ratio(rows):
    """summarise(), plus the node-count column it does not aggregate.

    adj_J = edge_J * (1 - 0.1 * total_node_ratio), and `summarise` drops
    the middle term, so a sweep reading only adj_J cannot say whether a
    change improved the EDGES or merely the NODE COUNT. Weighted by
    tp+fp+fn, the same weights adj_edge_jaccard uses.

    Purely ADDITIVE: summarise's own keys are untouched. Lifted from
    thr5, where it was unit-tested.
    """
    out = dict(summarise(rows))
    good = [r for r in rows
            if r.get("total_node_ratio") == r.get("total_node_ratio")
            and r["edge_tp"] == r["edge_tp"]]
    w = [r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in good]
    tw = sum(w)
    out["total_node_ratio"] = (
        sum(r["total_node_ratio"] * wi for r, wi in zip(good, w)) / tw
        if tw else float("nan"))
    return out


def score_dir(pred_dir):
    rows = []
    per = {}
    for gp in sorted(pathlib.Path(pred_dir).glob("*.geff")):
        name = gp.stem
# load_image=False: the metric reads dset.tracks and dset.scale only.
# Loading the image is what demanded a GPU (io.py:199 pins it); see
# cpu-probe. Skipping it is what lets this whole line run on CPU.
        dset = open_dataset(root / "train" / name, require_tracks=True,
                              load_image=False)
        res_ = td.graph.IndexedRXGraph.from_geff(str(gp))
        g = res_[0] if isinstance(res_, tuple) else res_
        er = compute_metric(g, dset.tracks, scale=dset.scale)
        rec = (node_recall(g, dset.tracks)
               if g.num_edges() > 0 and g.num_nodes() > 0 else 0.0)
        gt = read_geff(geff_path(root, "train", name))
        est = gt["extra"].get("estimated_number_of_nodes")
        row = per_sample_metrics(er, float(est) if est else
                                 float("nan"), rec)
        rows.append(row)
        per[name] = row
    # per-dataset rows come back too: a weighted mean can hide one
    # dataset carrying an entire delta, which is exactly what this run
    # exists to check.
    return _with_ratio(rows), per


def _bucket_by_source(coords, cand_edges):
    tcol = coords[:, 0].astype(int)
    by_t = defaultdict(list)
    for e in cand_edges:
        by_t[int(tcol[e[0]])].append(e)
    return by_t


def solve_edges(coords, cand_edges, T, div_w, window_size):
    """Solve the ILP in overlapping time windows. Returns global edge pairs.

    Window k covers frames [a, b] and k+1 covers [b, c]: they SHARE ONE
    FRAME but never share an EDGE, since k owns a->a+1 .. (b-1)->b and
    k+1 owns b->b+1 .. So the merged edge set cannot conflict and needs no
    reconciliation. Same scheme as ultrack's window_size/overlap_size.

    Measured on the 12-dataset holdout: window 25 scores 0.9382 against
    the global solve's 0.9368, at 2.08x the linking speed.
    """
    tcol = coords[:, 0].astype(int)
    by_t = _bucket_by_source(coords, cand_edges)
    if window_size is None:
        bounds = [(0, T - 1)]
    else:
        bounds, a = [], 0
        while a < T - 1:
            b = min(a + window_size, T - 1)
            bounds.append((a, b))
            a = b
    out = []
    dropped = 0
    for (a, b) in bounds:
        keep_n = np.where((tcol >= a) & (tcol <= b))[0]
        if len(keep_n) == 0:
            continue
        local_of_g = {int(g): i for i, g in enumerate(keep_n)}
        sub_coords = coords[keep_n]
        sub_edges = []
        for t in range(a, b):
            for (s, tg, p, d) in by_t.get(t, ()):
                ls, lt = local_of_g.get(int(s)), local_of_g.get(int(tg))
                if ls is not None and lt is not None:
                    sub_edges.append((ls, lt, p, d))
        if not sub_edges:
            continue
        graph = PIPE.build_graph(sub_coords, sub_edges)
        if graph.num_edges() == 0:
            continue
        solver = td.solvers.ILPSolver(
            edge_weight=-1.0 * td.EdgeAttr("edge_prob"),
            appearance_weight=0.0,
            disappearance_weight=ILP_DISAPP,
            division_weight=div_w)
        with PIPE.suppress_output():
            graph = solver.solve(graph)
        na = graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
        # Map solved nodes back BY COORDINATE, never by node_id: node_id is
        # not a stable handle across the solve (progress.md 02f7c8b).
        key_of_g = {}
        for g, c in zip(keep_n, sub_coords):
            key_of_g[(int(c[0]), int(c[1]), int(c[2]), int(c[3]))] = int(g)
        cols = {k: na[k].to_list() for k in ("node_id", "t", "z", "y", "x")}
        g_of_nid = {}
        for k in range(len(cols["node_id"])):
            key = (int(cols["t"][k]), int(round(float(cols["z"][k]))),
                   int(round(float(cols["y"][k]))),
                   int(round(float(cols["x"][k]))))
            gg = key_of_g.get(key)
            if gg is not None:
                g_of_nid[int(cols["node_id"][k])] = gg
        if graph.num_edges() == 0:
            continue
        ea = graph.edge_attrs(attr_keys=["source_id", "target_id"])
        for s_, t_ in zip(ea["source_id"].to_list(),
                          ea["target_id"].to_list()):
            gs, gt = g_of_nid.get(int(s_)), g_of_nid.get(int(t_))
            if gs is None or gt is None:
                dropped += 1
                continue
            out.append((gs, gt))
    if dropped:
        print(f"    WARN {dropped} solved edges unmatched by coordinate "
              f"-- this movie is a LOWER BOUND", flush=True)
    return out

# ---- [assembled from rankwhy.py: E accel/pair_features] ----
# ---- acceleration lookahead --------------------------------------------
# Togashi's stack item 4 (public-solutions-recon.md:313): "Acceleration
# lookahead in motion prediction (bonus for consistent acceleration), not
# just velocity." Never transplanted -- the recon lists items 1 and 5 as
# taken and leaves 2, 3, 4, 6.
#
# It is the same shape of gap div_kinematics just proved real on the
# division side: the model gets ONE temporal number. `pair_features` has
# `md`, the residual against a CONSTANT-VELOCITY extrapolation, where the
# velocity itself comes from a single previous transition. A cell that is
# turning or speeding up breaks that model, and nothing in the 18 columns
# can say so.
#
# Acceleration needs THREE positions, so it cannot exist in the two frames
# the official edge head sees, nor in a one-step velocity. That is the
# whole point.

NFEAT_BASE = 18
N_ACC = 5
NFEAT_ACC = NFEAT_BASE + N_ACC
EPS = 1e-9

_WARN_COUNT = defaultdict(int)


def warn_once(key, msg):
    """Rule 1: a fallback that does not announce itself produces a
    plausible wrong number."""
    if _WARN_COUNT[key] == 0:
        print(f"  WARN {msg}", flush=True)
    _WARN_COUNT[key] += 1


def warning_summary():
    if not _WARN_COUNT:
        print("  no degradations fired", flush=True)
        return
    print("  degradation summary:", flush=True)
    for k_, n_ in sorted(_WARN_COUNT.items()):
        print(f"    {k_}: {n_}", flush=True)


def pass1_tracks(dets):
    """Velocities AND the backward link map from one greedy pass.
    pass1_velocities discards the links; acceleration needs them."""
    T = len(dets)
    seeds = [seeds_of(dets[t]) for t in range(T)]
    vels = [np.zeros((len(seeds[t][0]), 3)) for t in range(T)]
    prev_of = [np.full(len(seeds[t][0]), -1, np.int32) for t in range(T)]
    for t in range(T - 1):
        P, ps = seeds[t]
        C, cs = seeds[t + 1]
        if len(P) == 0 or len(C) == 0:
            continue
        for pi, ci in _link_sim(P, C, vels[t] if len(vels[t]) else None,
                                ps, cs):
            vels[t + 1][ci] = (C[ci] - P[pi]) * SCALE
            prev_of[t + 1][ci] = pi
    return seeds, vels, prev_of


def accel_of(seeds, prev_of, t, i):
    """(acceleration vector um/frame^2, steps available).

    Needs THREE positions: t, t-1, t-2. With fewer, acceleration is
    undefined -- return zero and say how many steps there were, so the
    model can split on the count instead of reading 0 as 'not
    accelerating'."""
    if t < 2:
        return np.zeros(3), 0
    i1 = int(prev_of[t][i]) if i < len(prev_of[t]) else -1
    if i1 < 0:
        warn_once("accel_no_parent",
                  "a cell had no backward link, so acceleration is "
                  "undefined -> all 5 columns are 0 and acc_hist says 0; "
                  "that is NOT a measurement of constant velocity")
        return np.zeros(3), 0
    i2 = int(prev_of[t - 1][i1]) if i1 < len(prev_of[t - 1]) else -1
    if i2 < 0:
        warn_once("accel_one_step",
                  "a cell had only ONE backward step, so acceleration "
                  "needs a third position it does not have -> 0 with "
                  "acc_hist 1")
        return np.zeros(3), 1
    P0, _ = seeds[t]
    P1, _ = seeds[t - 1]
    P2, _ = seeds[t - 2]
    if i >= len(P0) or i1 >= len(P1) or i2 >= len(P2):
        return np.zeros(3), 0
    p0 = P0[i] * SCALE
    p1 = P1[i1] * SCALE
    p2 = P2[i2] * SCALE
    return (p0 - p1) - (p1 - p2), 2


def pair_features(dets, vels, T):
    """The 18 base columns EXACTLY as the transplant computes them, then 5
    acceleration columns. `vels` is accepted and ignored -- this rebuilds
    the pass so it can keep the link map the acceleration needs."""
    seeds, vels_own, prev_of = pass1_tracks(dets)
    out = []
    for t in range(T - 1):
        P, ps = seeds[t]
        C, cs = seeds[t + 1]
        if len(P) == 0 or len(C) == 0:
            out.append(([], np.zeros((0, NFEAT_ACC), np.float32), P, C))
            continue
        Pu = P * SCALE[None, :]
        Cu = C * SCALE[None, :]
        tree_p = cKDTree(Pu)
        tree_c = cKDTree(Cu)
        dens_p = np.array([len(tree_p.query_ball_point(x, 7.0)) - 1
                           for x in Pu])
        dens_c = np.array([len(tree_c.query_ball_point(x, 7.0)) - 1
                           for x in Cu])
        pred = Pu + vels_own[t]
        acc = np.zeros((len(P), 3))
        acc_n = np.zeros(len(P))
        for i in range(len(P)):
            a, n = accel_of(seeds, prev_of, t, i)
            acc[i] = a
            acc_n[i] = n
        # constant velocity plus half the acceleration -- the standard
        # second-order step, and what "lookahead" means here
        pred_acc = pred + 0.5 * acc
        pairs, rows = [], []
        cand_of_i = defaultdict(list)
        cand_of_j = defaultdict(list)
        raw = {}
        for i, x in enumerate(Pu):
            for j in tree_c.query_ball_point(x, GATE_UM):
                d = float(np.linalg.norm(Cu[j] - x))
                raw[(i, j)] = d
                cand_of_i[i].append((d, j))
                cand_of_j[j].append((d, i))
        for i in cand_of_i:
            cand_of_i[i].sort()
        for (i, j), d in raw.items():
            delta = Cu[j] - Pu[i]
            md = float(np.linalg.norm(Cu[j] - pred[i]))
            md_a = float(np.linalg.norm(Cu[j] - pred_acc[i]))
            rank = next(k for k, (_, jj) in enumerate(cand_of_i[i])
                        if jj == j)
            v = vels_own[t][i]
            na = float(np.linalg.norm(acc[i]))
            nv = float(np.linalg.norm(v))
            acc_cos = (float(np.dot(v, acc[i])) / (nv * na)
                       if nv > EPS and na > EPS else 0.0)
            rows.append([d, delta[0], abs(delta[0]),
                         float(np.linalg.norm(delta[1:])), md, d - md,
                         ps[i], cs[j],
                         abs(float(_logit(np.array([ps[i]]))[0]
                                   - _logit(np.array([cs[j]]))[0])),
                         dens_p[i], dens_c[j], len(P), len(C),
                         rank, len(cand_of_i[i]), len(cand_of_j[j]),
                         nv, t / max(1, T),
                         # ---- acceleration lookahead ----
                         na, acc_cos, md_a, md - md_a, acc_n[i]])
            pairs.append((i, j))
        out.append((pairs, np.asarray(rows, np.float32), P, C))
    return out



# ======================================================================
# edgeprob_rank: everything below is NEW. Everything above is rankwhy.py's
# harness cut by line content (docstring, ASSEMBLY) -- byte-identical to v1.
# ======================================================================

from sklearn.ensemble import HistGradientBoostingClassifier    # noqa: E402
from sklearn.metrics import roc_auc_score                      # noqa: E402


def _walk_files(match):
    """edgeprob_cache.py's bounded search of /kaggle/input: prunes the
    competition tree (millions of zarr chunks; a blind walk appears to
    hang -- toolchain trap 5) and every .zarr / .geff directory."""
    hits = {}
    for root_, dirs, fnames in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs
                   if "cell-tracking-during-development" not in d
                   and not d.endswith(".zarr") and not d.endswith(".geff")]
        for fn in fnames:
            key = match(fn)
            if key is not None:
                hits.setdefault(key, pathlib.Path(root_) / fn)
    return hits


def _npz_prefix(prefix):
    return lambda fn: (fn[len(prefix):-len(".npz")]
                       if fn.startswith(prefix) and fn.endswith(".npz")
                       else None)


# ---- the official head's probabilities: biohub-edgeprob v3's output -----
# edgeprob_<name>.npz        the head over the BASE (detcache) primaries,
#                            all 199 train movies;
# edgeprob_F4_<name>.npz     the head over the F4 (fdet_F4_) primaries,
#                            the 24 evaluation sets.
# Dataset names begin with their domain tag (6bba_, 44b6_, ...), never
# "F4_", so the prefix split below is unambiguous.
_EP_ALL = _walk_files(_npz_prefix("edgeprob_"))
EDGEPROB = {k: v for k, v in _EP_ALL.items()
            if not k.startswith("F4_") and not k.startswith("F4hp_")}
EDGEPROB_F4HP = {k[len("F4hp_"):]: v for k, v in _EP_ALL.items()
                 if k.startswith("F4hp_")}
EDGEPROB_F4 = {k[len("F4_"):]: v for k, v in _EP_ALL.items()
               if k.startswith("F4_")}
del _EP_ALL
assert EDGEPROB, ("no edgeprob_*.npz under /kaggle/input -- attach "
                  "faustine649/biohub-edgeprob (v3) as a kernel source")
assert EDGEPROB_F4, ("no edgeprob_F4_*.npz under /kaggle/input -- "
                     "biohub-edgeprob v3 (the F4 pass) is not attached, or "
                     "an older version is")
_cj = _walk_files(lambda fn: fn if fn == "edgeprob_counts.json" else None)
assert _cj, ("edgeprob_counts.json not found -- biohub-edgeprob's output is "
             "incomplete (the json is written last) or not attached")
COUNTS = json.loads(next(iter(_cj.values())).read_text())
print(f"edgeprob caches mounted: {len(EDGEPROB)} BASE + {len(EDGEPROB_F4)} "
      f"F4 (biohub-edgeprob script_version {COUNTS.get('script_version')})",
      flush=True)
assert float(COUNTS["unet_thresh"]) == UNET_THRESH, (
    f"biohub-edgeprob used UNET_THRESH {COUNTS['unet_thresh']}, this "
    f"harness {UNET_THRESH}: (t, i, j) would index different primaries")
assert float(COUNTS["gate_um"]) == GATE_UM, (
    f"biohub-edgeprob gated at {COUNTS['gate_um']} um, this harness at "
    f"{GATE_UM}: the pair sets would differ")
assert np.allclose(np.asarray(COUNTS["scale"], float), SCALE), (
    f"biohub-edgeprob SCALE {COUNTS['scale']} != {SCALE.tolist()}")
if COUNTS.get("failed"):
    raise SystemExit(f"[degraded] biohub-edgeprob reported "
                     f"{len(COUNTS['failed'])} failed dataset(s): "
                     f"{COUNTS['failed']} -- rerun it before ranking")
if COUNTS.get("skipped_time"):
    raise SystemExit(f"[degraded] biohub-edgeprob stopped on budget with "
                     f"{COUNTS['skipped_time']} datasets not attempted -- "
                     f"rerun it before ranking")
if not COUNTS.get("reverse", False):
    warn_once("no_reverse",
              "biohub-edgeprob ran with REVERSE False -> p_rev and l_rev "
              "are NaN, h_fused falls back to p_fwd, and the +BIDIR arm "
              "measures only p_fwd")

# ---- the SECOND head's probabilities: biohub-edgeprob2 v1's output ------
_EP2_ALL = _walk_files(_npz_prefix("edgeprob2_"))
EDGEPROB2 = {k: v for k, v in _EP2_ALL.items()
             if not k.startswith("F4_") and not k.startswith("F4hp_")}
EDGEPROB2_F4HP = {k[len("F4hp_"):]: v for k, v in _EP2_ALL.items()
                  if k.startswith("F4hp_")}
EDGEPROB2_F4 = {k[len("F4_"):]: v for k, v in _EP2_ALL.items()
                if k.startswith("F4_")}
del _EP2_ALL
assert EDGEPROB2, ("no edgeprob2_*.npz under /kaggle/input -- attach "
                   "faustine649/biohub-edgeprob2 as a kernel source")
assert EDGEPROB2_F4, "no edgeprob2_F4_*.npz -- biohub-edgeprob2 is incomplete"
_cj2 = _walk_files(lambda fn: fn if fn == "edgeprob2_counts.json" else None)
assert _cj2, "edgeprob2_counts.json not found -- biohub-edgeprob2 incomplete"
COUNTS2 = json.loads(next(iter(_cj2.values())).read_text())
print(f"edgeprob2 caches mounted: {len(EDGEPROB2)} BASE + {len(EDGEPROB2_F4)} "
      f"F4 (head {COUNTS2.get('head')}, sha256 "
      f"{str(COUNTS2.get('weights_sha256'))[:12]}, script_version "
      f"{COUNTS2.get('script_version')})", flush=True)
for _k in ("unet_thresh", "gate_um"):
    assert float(COUNTS2[_k]) == float(COUNTS[_k]), (
        f"biohub-edgeprob2 {_k} {COUNTS2[_k]} != biohub-edgeprob {COUNTS[_k]}")
assert np.allclose(np.asarray(COUNTS2["scale"], float), SCALE)
if COUNTS2.get("failed") or COUNTS2.get("skipped_time"):
    raise SystemExit(f"[degraded] biohub-edgeprob2 incomplete: failed "
                     f"{COUNTS2.get('failed')}, skipped "
                     f"{COUNTS2.get('skipped_time')} -- rerun it")
assert COUNTS2.get("reverse", False), "biohub-edgeprob2 ran without REVERSE"

# ---- the F4 primaries: biohub-fusecache's output ------------------------
FDET_F4 = _walk_files(_npz_prefix("fdet_F4_"))
print(f"fdet_F4 caches: {len(FDET_F4)}", flush=True)
assert len(FDET_F4) >= 24, "attach biohub-fusecache for the F4 detections"
FDET_F4HP = _walk_files(_npz_prefix("fdet_F4hp_"))
print(f"fdet_F4hp caches: {len(FDET_F4HP)}", flush=True)
if DET_SRC == "F4HP":
    assert len(FDET_F4HP) >= 140, (
        f"DET_SRC is F4HP but only {len(FDET_F4HP)} fdet_F4hp_*.npz are "
        f"attached -- hparacache v3 caches 140 (128 6bba + 12 44b6). Round 1"
        f" ran on 24 and could only move 12 of the 128 scored movies.")
    # OVERRIDE, not replace. hparacache v3 covers 140 movies (all 128 6bba
    # + the 12 44b6 of the eval draw); the rest of the 187-movie pool stays
    # on refine. v1 assigned FDET_F4 = FDET_F4HP and died on 116 missing
    # 6bba pool caches.
    #
    # ROUND 1 SCORED 128 MOVIES AND ONLY 12 OF THEM CHANGED, which diluted
    # every aggregate by about 10x. The check that matters is therefore
    # COVERAGE OF THE SCORED SET, not a count: every 6bba movie the bench
    # scores must be on hpara coordinates, or the arm is a blend of the two
    # placements reported as one.
    _both = sorted(set(FDET_F4) & set(FDET_F4HP))
    _hp6 = sorted(n for n in FDET_F4HP if n.startswith("6bba_"))
    FDET_F4 = {**FDET_F4, **FDET_F4HP}
    print(f"  hpara caches override {len(_both)} of {len(FDET_F4)} F4 "
          f"entries ({len(_hp6)} of them 6bba); the rest of the pool stays "
          f"on refine", flush=True)
    assert len(_hp6) >= 128, (
        f"only {len(_hp6)} 6bba movies carry an fdet_F4hp cache -- the bench "
        f"scores 128, so {128 - len(_hp6)} of them would silently run on "
        f"refine coordinates inside the arm called hpara")


# ---- the two shipped rankers, loaded not trained ------------------------
def _load_pkl(fname, lo, hi):
    hit = _walk_files(lambda fn: fn if fn == fname else None)
    assert hit, (f"{fname} not found -- attach faustine649/biohub-poolsize "
                 f"as a kernel source")
    pk = next(iter(hit.values()))
    with open(pk, "rb") as f:
        blob = pickle.load(f)
    pool = blob.get("pool") if isinstance(blob, dict) else None
    assert pool is not None and lo <= len(pool) <= hi, (
        f"{pk}: pool {0 if pool is None else len(pool)} outside [{lo}, {hi}]")
    print(f"  ranker {fname}: pool {len(pool)} ({pk})", flush=True)
    return blob["model"], list(blob["cols"]), sorted(set(pool))


# D187: the anchor arm. D32: loaded ONLY for its pool, which defines the
# 44b6 draw (dctrain drew from the 44b6 outside D32's pool); it is not
# scored in v2.
MODEL_D187, MCOLS_D187, POOL187 = _load_pkl("edge_ranker_d187.pkl", 100, 199)
_MODEL_D32, MCOLS_D32, POOL32 = _load_pkl("edge_ranker_d32.pkl", 1, 99)
del _MODEL_D32
assert len(POOL32) == N_POOL32, (
    f"edge_ranker_d32.pkl's pool has {len(POOL32)} datasets, expected "
    f"{N_POOL32}")
assert MCOLS_D32 == list(range(NFEAT_BASE)), (
    f"edge_ranker_d32.pkl cols {MCOLS_D32}: expected the 18 base columns")
assert not set(POOL32) & set(HOLDOUT_6BBA), "D32 pool overlaps the holdout"
assert not set(POOL187) & set(HOLDOUT_6BBA), "D187 pool overlaps the holdout"

# ---- the split: 175 pool + 12 + 12 -------------------------------------
ALL_TRAIN = sorted(p.name[:-len(".zarr")]
                   for p in (root / "train").glob("*.zarr"))
assert len(ALL_TRAIN) == 199, f"expected 199 train movies, found {len(ALL_TRAIN)}"
assert all(n in ALL_TRAIN for n in HOLDOUT_6BBA + POOL32), \
    "a holdout / pool name is not in train/"
# dctrain.py:158-163 verbatim, so this draw equals dctrain's excluded 12.
eligible_44b6 = sorted(n for n in ALL_TRAIN
                       if n.startswith("44b6_") and n not in set(POOL32))
assert len(eligible_44b6) >= N_44B6, \
    f"only {len(eligible_44b6)} 44b6 datasets outside D32's pool"
SET_44B6 = sorted(random.Random(SEED_44B6).sample(eligible_44b6, N_44B6))
EXCLUDED = set(HOLDOUT_6BBA) | set(SET_44B6)
POOL175 = sorted(n for n in ALL_TRAIN if n not in EXCLUDED)
assert len(POOL175) == N_POOL175, (
    f"pool is {len(POOL175)}, expected {N_POOL175} (dctrain.py's split)")
assert not (set(POOL175) & EXCLUDED), "pool overlaps an evaluation group"

# The edgeprob kernel recorded the sets it scored; when it did, they must
# be THESE (v3 may or may not carry the keys; the per-dataset presence
# checks below are what actually guard the join).
for _key, _want, _what in (("set_44b6", SET_44B6, "44b6 draw"),
                           ("pool32", POOL32, "D32 pool"),
                           ("holdout_6bba", sorted(HOLDOUT_6BBA),
                            "6bba holdout")):
    if _key in COUNTS:
        assert sorted(COUNTS[_key]) == _want, (
            f"biohub-edgeprob's {_what} differs from this kernel's -- its "
            f"seed or pool changed; do not join across different draws")
    else:
        print(f"  edgeprob_counts.json has no '{_key}' -> {_what} not "
              f"cross-checked against biohub-edgeprob", flush=True)

# Optional cross-check against dctrain's split_manifest.json when its
# weights happen to be mounted. NOT required -- do not attach a source for
# it. Its "all" is dctrain's train+val = the 175 pool.
_man = _walk_files(lambda fn: fn if fn == "split_manifest.json" else None)
if _man:
    _seen = set(json.loads(next(iter(_man.values())).read_text())["all"])
    assert _seen == set(POOL175), (
        f"D175 pool differs from split_manifest.json's 'all' "
        f"({len(_seen)} names): only-here {sorted(set(POOL175) - _seen)[:6]} "
        f"only-there {sorted(_seen - set(POOL175))[:6]} -- the draw no "
        f"longer reproduces dctrain's; the 44b6 soft anchors do not apply")
    print("  D175 pool cross-checked against split_manifest.json: identical "
          "(the 44b6 soft anchors apply)", flush=True)
else:
    print("  split_manifest.json not mounted -> D175 pool NOT cross-checked "
          "against dctrain's (compare the printed names by hand before "
          "reading the 44b6 soft anchors)", flush=True)

NAMES_EVAL = HOLDOUT_6BBA + SET_44B6
assert len(NAMES_EVAL) == len(set(NAMES_EVAL)) == 24, "groups overlap"
assert len(set(NAMES_EVAL) | set(POOL175)) == 199
print(f"6bba holdout ({len(HOLDOUT_6BBA)}): {' '.join(HOLDOUT_6BBA)}", flush=True)
print(f"44b6 seed={SEED_44B6} from {len(eligible_44b6)} outside-D32 "
      f"({len(SET_44B6)}): {' '.join(SET_44B6)}", flush=True)
print(f"D175 pool ({len(POOL175)}): {' '.join(POOL175)}", flush=True)
_ov = sorted(set(POOL187) & set(NAMES_EVAL))
print(f"  ANCHOR_D187's pool overlaps the evaluation sets on {len(_ov)} "
      f"datasets ({sum(n.startswith('44b6_') for n in _ov)} of them 44b6): "
      f"its 44b6 row is IN-SAMPLE; the four D175 arms are clean on "
      f"both groups", flush=True)

# Rule 1: a missing dataset is a crash here, never a shrunken comparison.
_missing = [n for n in ALL_TRAIN if n not in CACHES]
assert not _missing, (f"no detcache for: {_missing} -- attach "
                      f"biohub-edge-ranker and biohub-cache-train")
_missing = [n for n in ALL_TRAIN if n not in EDGEPROB]
assert not _missing, (f"no BASE edgeprob cache for: {_missing} -- "
                      f"biohub-edgeprob v3 must have scored all 199")
_missing = [n for n in NAMES_EVAL if n not in EDGEPROB_F4]
assert not _missing, (f"no edgeprob_F4 cache for: {_missing} -- "
                      f"biohub-edgeprob v3 must have scored the 24 F4 sets")
_missing = [n for n in ALL_TRAIN if n not in EDGEPROB2]
assert not _missing, f"no BASE edgeprob2 cache for: {_missing}"
_missing = [n for n in NAMES_EVAL if n not in EDGEPROB2_F4]
assert not _missing, f"no edgeprob2_F4 cache for: {_missing}"
_missing = [n for n in NAMES_EVAL if n not in FDET_F4]
assert not _missing, f"no fdet_F4 cache for: {_missing} -- attach biohub-fusecache"
_missing = [n for n in ALL_TRAIN if not geff_path(root, "train", n).exists()]
assert not _missing, f"no GT geff for: {_missing}"

# ---- columns and arms -------------------------------------------------
# The harness's pair_features (the 23-column override) yields the 18 base
# columns followed by 5 acceleration columns nobody uses (accel FAILED,
# poolsize header). The head's five columns are appended after those.
NCOL_HARNESS = NFEAT_ACC                      # 23
C_PFWD, C_PREV, C_LFWD, C_LREV, C_HFUSED = (NCOL_HARNESS + k for k in range(5))
C2_PFWD, C2_PREV, C2_LFWD, C2_LREV, C2_HFUSED = (NCOL_HARNESS + 5 + k
                                                 for k in range(5))
C_AVG_FWD, C_AVG_REV, C_HAVG = (NCOL_HARNESS + 10 + k for k in range(3))
NCOL_ALL = NCOL_HARNESS + 13                  # 36
NEW_NAMES = {C_PFWD: "p_fwd", C_PREV: "p_rev", C_LFWD: "l_fwd",
             C_LREV: "l_rev", C_HFUSED: "h_fused",
             C2_PFWD: "p2_fwd", C2_PREV: "p2_rev", C2_LFWD: "l2_fwd",
             C2_LREV: "l2_rev", C2_HFUSED: "h2_fused",
             C_AVG_FWD: "p_avg_fwd", C_AVG_REV: "p_avg_rev", C_HAVG: "h_avg"}
BASE_NAMES = ["d_um", "dz_um", "abs_dz_um", "dxy_um", "md_um", "d_minus_md",
              "score_i", "score_j", "abs_logit_diff", "dens_i", "dens_j",
              "n_prev", "n_curr", "rank_of_j_for_i", "n_cand_i", "n_cand_j",
              "speed_i", "t_frac"]           # pair_features row order
assert len(BASE_NAMES) == NFEAT_BASE
BASE_COLS = list(range(NFEAT_BASE))
ARMS = {
    "+BIDIR": BASE_COLS + [C_PFWD, C_PREV, C_HFUSED],
    "+BIDIR2": BASE_COLS + [C2_PFWD, C2_PREV, C2_HFUSED],
    "+BOTH": BASE_COLS + [C_PFWD, C_PREV, C_HFUSED, C2_PFWD, C2_PREV, C2_HFUSED],
    "+AVG": BASE_COLS + [C_AVG_FWD, C_AVG_REV, C_HAVG],
}
ARM_ORDER = ["+BIDIR", "+BIDIR2", "+BOTH", "+AVG"]      # trained
# head-only arms: no ranker; the named column IS the edge probability
HEAD_ONLY = {"HEAD1": C_HFUSED, "HEAD12": C_HAVG}
EVAL_ARMS = ["ANCHOR_D187"] + ARM_ORDER + list(HEAD_ONLY)
REF = "+BIDIR"                                          # paired deltas vs this
ARM_TAG = {"ANCHOR_D187": "d187", "+BIDIR": "bidir", "+BIDIR2": "bidir2",
           "+BOTH": "both", "+AVG": "avg", "HEAD1": "head1", "HEAD12": "head12"}
PKL_NAME = {"+BIDIR": "edge_ranker_ep175v3_bidir.pkl",
            "+BIDIR2": "edge_ranker_ep175v3_bidir2.pkl",
            "+BOTH": "edge_ranker_ep175v3_both.pkl",
            "+AVG": "edge_ranker_ep175v3_avg.pkl"}
GROUPS = (("6bba", HOLDOUT_6BBA), ("44b6", SET_44B6))
DET_ARMS = ["F4"] + (["BASE"] if SCORE_BASE_ARM else [])
DOMAINS = [f"{a}-{g}" for a in DET_ARMS for g, _ in GROUPS]
if DET_SRC == "F4HP":
    for _nm, _base, _hp in (("edgeprob", EDGEPROB_F4, EDGEPROB_F4HP),
                            ("edgeprob2", EDGEPROB2_F4, EDGEPROB2_F4HP)):
        # THE OVERRIDE REPLACES, IT NEVER ADDS -- and that is not a
        # detail. `edgeprob2` is the OPTIONAL second head: divbench9 reads
        # it with `EP2_OF["F4"].get(name)`, and only 24 movies have an
        # `edgeprob2_F4_` cache at all, so 116 of the 128 scored movies are
        # solved without it in BOTH arms. ep2hpcache cached 140. Merging
        # those in would hand the hpara arm a second head on 116 movies
        # where the control has none -- the arms would then differ in the
        # PIPELINE, not in the placement, and the delta would be
        # unreadable. So the hpara dict is first cut down to the keys the
        # refine arm already has.
        _add = sorted(set(_hp) - set(_base))
        _hp = {k: v for k, v in _hp.items() if k in _base}
        _miss6 = sorted(n for n in _base
                        if n.startswith("6bba_") and n not in _hp)
        assert not _miss6, (
            f"{_nm}: {len(_miss6)} scored 6bba movies have a REFINE cache "
            f"with no hpara counterpart ({_miss6[:3]}...) -- those would be "
            f"scored with a head that read the refine coordinates against "
            f"hpara points, and the (t, i, j) join fails outright.")
        _ov = sorted(_hp)
        _ov6 = [n for n in _ov if n.startswith("6bba_")]
        if _nm == "edgeprob":
            # The MANDATORY head. Every scored movie must have one.
            assert len(_ov6) >= 128, (
                f"edgeprob: only {len(_ov6)} of the 128 scored 6bba movies "
                f"have an F4hp cache -- attach biohub-ephpcache.")
        _base.update(_hp)
        print(f"  {_nm}: {len(_ov)} entries switched to the hpara head "
              f"({len(_ov6)} of them 6bba); {len(_add)} hpara caches IGNORED "
              f"because the refine arm has no cache for those movies",
              flush=True)

EP_OF = {"BASE": EDGEPROB, "F4": EDGEPROB_F4}
EP2_OF = {"BASE": EDGEPROB2, "F4": EDGEPROB2_F4}


def load_dets(name, arm="BASE"):
    """BASE = the xl three-branch average that shipped at 0.922.
    F4 = the same with ep05 DeepCenter as a fourth equal-weight map,
    which is what ships at 0.925 (ref 55889500). rankwhy's loader."""
    z = np.load(CACHES[name] if arm == "BASE" else FDET_F4[name])
    T = max(int(k[1:]) for k in z.files if k.startswith("c")) + 1
    return [(z[f"c{t}"], z[f"s{t}"]) for t in range(T)]


JOIN = dict(datasets=0, pairs=0, nan_p_rev=0, empty_frames=0)


def _load_head(name, ep_path, T, ours):
    """One head's cache -> (sorted key, 5 columns in key order), after the
    per-frame primary counts and the index ranges have been checked."""
    z = np.load(ep_path)
    npf = np.asarray(z["n_per_frame"], np.int64)
    assert len(npf) == T, (
        f"{name}: {ep_path.name} n_per_frame has {len(npf)} frames, the det cache {T}")
    assert np.array_equal(npf, ours), (
        f"{name}: primaries per frame differ between {ep_path.name} and "
        f"this harness at frames {np.where(npf != ours)[0][:8].tolist()} "
        f"-- (t, i, j) cannot be trusted (BASE/F4 mismatch?)")
    et = np.asarray(z["t"], np.int64)
    ei = np.asarray(z["i"], np.int64)
    ej = np.asarray(z["j"], np.int64)
    K = int(npf.max()) + 1
    if len(et):
        assert et.min() >= 0 and et.max() <= T - 2, (
            f"{name}: {ep_path.name} t outside [0, {T - 2}]")
    assert np.all(ei < npf[et]) and np.all(ej < npf[et + 1]), (
        f"{name}: an {ep_path.name} (i, j) index exceeds its frame's primary count")
    key = (et * K + ei) * K + ej
    order = np.argsort(key, kind="stable")
    skey = key[order]
    assert len(skey) == 0 or np.all(np.diff(skey) > 0), (
        f"{name}: duplicate (t, i, j) rows in {ep_path.name}")
    p_fwd = np.asarray(z["p_fwd"], np.float32)[order]
    p_rev = np.asarray(z["p_rev"], np.float32)[order]
    l_fwd = np.asarray(z["l_fwd"], np.float32)[order]
    l_rev = np.asarray(z["l_rev"], np.float32)[order]
    assert np.isfinite(p_fwd).all() and np.isfinite(l_fwd).all(), (
        f"{name}: non-finite forward probability or logit in {ep_path.name}")
    assert np.all((p_fwd >= 0) & (p_fwd <= 1)), f"{name}: p_fwd out of [0,1]"
    nan_rev = ~np.isfinite(p_rev)
    if nan_rev.any():
        JOIN["nan_p_rev"] += int(nan_rev.sum())
        warn_once("p_rev_nan",
                  f"{name}: {int(nan_rev.sum())} of {len(p_rev)} pairs have "
                  f"NaN p_rev in {ep_path.name} -> h_fused uses p_fwd alone "
                  f"there; the HistGradientBoosting arms see the NaN natively")
    pr = np.where(nan_rev, 0.0, p_rev).astype(np.float32)
    h_fused = np.where(nan_rev, p_fwd,
                       2.0 * p_fwd * pr / np.maximum(p_fwd + pr, 1e-12)
                       ).astype(np.float32)
    return K, skey, np.stack([p_fwd, p_rev, l_fwd, l_rev, h_fused], axis=1)


def features_ext(name, dets, T, ep_path, ep2_path):
    """The harness's own pass (pass1_tracks + pair_features, exactly what
    solve_one computes) with the head's five columns joined by (t, i, j)
    from the edgeprob cache at `ep_path` (BASE or F4, matching `dets`).

    Returns (seeds, feats) where feats[t] = (pairs, rows[:, NCOL_ALL], P, C)
    in the harness's own layout, so solve_one below consumes it unchanged.
    """
    seeds, vels, prev_of = pass1_tracks(dets)
    feats = pair_features(dets, None, T)
    ours = np.array([len(seeds[t][0]) for t in range(T)], np.int64)
    K, skey, ex1 = _load_head(name, ep_path, T, ours)
    if ep2_path is None:                    # head 2 not cached for this movie
        K2, skey2, ex2 = K, skey, np.full(ex1.shape, np.nan, np.float32)
    else:
        K2, skey2, ex2 = _load_head(name, ep2_path, T, ours)
    assert K2 == K and np.array_equal(skey, skey2), (
        f"{name}: the two heads' caches hold different (t, i, j) sets "
        f"({len(skey)} vs {len(skey2)} rows) -- were they cached on the "
        f"same primaries?")
    # the two heads averaged: probabilities in [0, 1] both, reverse NaN in
    # one head falls back to the other's value
    p_avg_fwd = 0.5 * (ex1[:, 0] + ex2[:, 0])
    _r1, _r2 = ex1[:, 1], ex2[:, 1]
    p_avg_rev = np.where(np.isfinite(_r1) & np.isfinite(_r2), 0.5 * (_r1 + _r2),
                         np.where(np.isfinite(_r1), _r1, _r2)).astype(np.float32)
    h_avg = (0.5 * (ex1[:, 4] + ex2[:, 4])).astype(np.float32)
    extra = np.concatenate(
        [ex1, ex2, np.stack([p_avg_fwd, p_avg_rev, h_avg], axis=1)],
        axis=1).astype(np.float32)
    assert extra.shape[1] == NCOL_ALL - NCOL_HARNESS, extra.shape
    out = []
    n_rows = 0
    for t in range(T - 1):
        pairs, rows, P, C = feats[t]
        if not pairs:
            JOIN["empty_frames"] += 1
            out.append((pairs, np.zeros((0, NCOL_ALL), np.float32), P, C))
            continue
        assert rows.shape[1] == NCOL_HARNESS, rows.shape
        pij = np.asarray(pairs, np.int64)
        k = (t * K + pij[:, 0]) * K + pij[:, 1]
        pos = np.searchsorted(skey, k)
        pos_c = np.minimum(pos, max(len(skey) - 1, 0))
        found = (pos < len(skey)) & (skey[pos_c] == k)
        assert found.all(), (
            f"{name} t={t}: {int((~found).sum())} of {len(k)} harness pairs "
            f"have no edgeprob row -- the gate or the primaries differ; a "
            f"silent 0 here would be a plausible wrong number")
        out.append((pairs, np.concatenate([rows, extra[pos]], axis=1)
                    .astype(np.float32), P, C))
        n_rows += len(k)
    assert n_rows == len(skey), (
        f"{name}: harness produced {n_rows} pairs, the edgeprob cache holds "
        f"{len(skey)} -- the pair sets are not identical")
    JOIN["datasets"] += 1
    JOIN["pairs"] += n_rows
    return seeds, out


def label_rows(name, feats):
    """poolsize.py's GT labelling (its training loop, lines 862-905), the
    logic verbatim. A row is kept only when the source primary matched a
    GT node WITH children; label 1 iff the target primary matched one of
    those children. Returns (X, y)."""
    gt = read_geff(geff_path(root, "train", name))
    g_t = np.asarray(gt["t"], int)
    g_pos = np.stack([np.asarray(gt[c], float)
                      for c in ("z", "y", "x")], axis=1) * SCALE
    row_of = {int(v): i for i, v in enumerate(gt["node_ids"])}
    child_of = {}
    for a, b in gt["edges"]:
        ra, rb = row_of.get(int(a)), row_of.get(int(b))
        if ra is not None and rb is not None:
            child_of.setdefault(ra, []).append(rb)
    gt_by_t = defaultdict(list)
    for i in range(len(g_t)):
        gt_by_t[int(g_t[i])].append(i)
    X_list, y_list = [], []
    for t in range(len(feats)):
        pairs, rows, P, C = feats[t]
        if not pairs:
            continue
        Pu, Cu = P * SCALE[None, :], C * SCALE[None, :]

        def match(pos_u, idx_list):
            if not idx_list:
                return {}
            gtree = cKDTree(g_pos[idx_list])
            m = {}
            for di, x in enumerate(pos_u):
                dd, gi = gtree.query(x)
                if dd <= MATCH_UM:
                    m[di] = idx_list[int(gi)]
            return m

        m_p = match(Pu, gt_by_t.get(t, []))
        m_c = match(Cu, gt_by_t.get(t + 1, []))
        det_of_gt = {g: d for d, g in m_c.items()}
        for (i, j), row in zip(pairs, rows):
            if i not in m_p:
                continue
            kids = child_of.get(m_p[i], [])
            if not kids:
                continue
            j_true = {det_of_gt.get(b) for b in kids}
            j_true.discard(None)
            X_list.append(row)
            y_list.append(1 if j in j_true else 0)
    if not X_list:
        return np.zeros((0, NCOL_ALL), np.float32), np.zeros(0, int)
    return np.stack(X_list).astype(np.float32), np.asarray(y_list)


# ---- FEATURES + JOIN: all 199 train movies on BASE -----------------------
t_all = time.time()
SIX = sorted(n for n in ALL_TRAIN if n.startswith("6bba_"))
FOUR = sorted(n for n in ALL_TRAIN if n.startswith("44b6_"))
assert len(SIX) == 128 and len(FOUR) == 71, (len(SIX), len(FOUR))
FOLD_OF = {n: i % 2 for i, n in enumerate(SIX)}      # alternate by sorted name
_missing = [n for n in SIX if n not in FDET_F4]
assert not _missing, f"no fdet_F4 cache for {len(_missing)} 6bba movies: {_missing[:5]}"
_missing = [n for n in SIX if n not in EDGEPROB_F4]
assert not _missing, f"no edgeprob F4 cache for {len(_missing)} 6bba movies: {_missing[:5]}"
MODEL_BIDIR, MCOLS_BIDIR, POOL_BIDIR = _load_pkl("edge_ranker_ep175v3_bidir.pkl", 170, 180)
assert not (set(POOL_BIDIR) & set(SET_44B6)), "the production ranker saw the 44b6 draw"
PROD_FLOOR = 0.48                                     # ref 56030644, LB 0.933
STEAL_ARMS = []       # divbench6 sweeps the SOLVER, not the patch
# (arm name, DIV_W). The shipped value is 0.55; ranker-fine's 12-event curve
# read forks 111 / 190 / 394 / 1024 at 0.55 / 0.45 / 0.35 / 0.20, so these
# three span an order of magnitude in fork production. 0.65 and above is not
# swept: that direction makes FEWER forks and the gate cannot help there.
# bridgerank: no re-solve arms. The four arms below are post-link replays on
# the SAME solved graph; DIVW_ARMS carries their names so the report and the
# cross iterate over them unchanged (the dw value is the shipped one).
DA_ARMS = [("CR_NOCAP_DV150", dict(diverge=1.5, admit=False, rank="contrast", cap=100.0, src="REF")),
           ("ZCR_NOCAP_DV150", dict(diverge=1.5, admit=False, rank="contrast", cap=100.0, src="Z")),
           ("Z50CR_NOCAP_DV150", dict(diverge=1.5, admit=False, rank="contrast", cap=100.0, src="Z50")),
           ("Z65CR_NOCAP_DV150", dict(diverge=1.5, admit=False, rank="contrast", cap=100.0, src="Z65"))]
NC_ARMS = (("Z", 0.35), ("Z50", 0.50), ("Z65", 0.65))        # ncsweep: the node cost per z solve
FP_RANK = "dist"          # shipped: proposals by d_pq + 0.15 d_s; "contrast": by the image contrast, distance as tie-break
FP_CAP = 1.0              # multiplier on DIV_FRAME_CAP and DIV_GLOBAL_CAP in the orphan pass
FP_MEMO = {}              # (t, z, y, x) -> contrast, per movie
FP_STAT = defaultdict(int)
BR_ARMS = DA_ARMS         # the replay loop below iterates BR_ARMS_RUN
DA_ADMIT = False          # shipped patch; set per arm in pass 2
DA_MIN_SEP_UM = 3.0       # an admitted daughter this close to an existing node is a duplicate detection
DA_CTX = {}               # seeds, seed_of_node, added ids -- set per arm
DIVW_ARMS = [(a_, DIV_W) for a_, _c in BR_ARMS]
BR_COLLECT = False        # pass 1: record (end, start) pairs + labels after span 1
BR_MODEL = None           # pass 2: the fold model for the movie being patched
BR_TAU = 0.0
BR_ROWS = []              # (movie, fold, span, label, feats)
BR_CTX = {}               # per movie: seeds, seed_of_node, gt
BR_STAT = defaultdict(int)
BR_NAMES = ["span", "d_um", "d_per_frame", "dz_um", "dxy_um", "v_in", "v_out",
            "cos_in_out", "cos_in_delta", "err_in", "err_out", "err_mean",
            "len_end", "len_start", "n_starts_near_end", "n_ends_near_start",
            "rank_by_d", "d_second", "clear_min", "n_cand_hits", "cand_score_max",
            "n_mother_cand", "d_mother_min", "end_is_daughter", "s_end", "s_start",
            "dens_end", "t_frac"]
BR_DIR = pathlib.Path("/tmp/br")
BR_DIR.mkdir(parents=True, exist_ok=True)
_jobs = [(n, "44b6") for n in FOUR] + [(n, "6bba") for n in SIX]
print(f"\n=== FEATURES + JOIN: {len(_jobs)} passes, all 199 train movies on "
      f"BASE (xl) primaries + BASE edgeprob; every 6bba movie keeps its full "
      f"features for the solve ===", flush=True)
FEATS = {}                 # name -> (seeds, feats), 6bba only
LAB4 = {}                  # name -> (X, y) on F4/F4HP features, 6bba only
LAB = {}                   # name -> (X, y) labelled rows, all 199
for k_, (name, role) in enumerate(_jobs):
    t0 = time.time()
    dets = load_dets(name, "BASE")
    T = len(dets)
    seeds, feats = features_ext(name, dets, T, EP_OF["BASE"][name],
                                EP2_OF["BASE"][name])
    X_, y_ = label_rows(name, feats)
    LAB[name] = (X_, y_)
    n_pairs = int(sum(len(p) for p, _r, _P, _C in feats))
    n4 = ""
    if role == "6bba" or name in SET_44B6:
        dets4 = load_dets(name, "F4")
        T4 = len(dets4)
        seeds4, feats4 = features_ext(name, dets4, T4, EP_OF["F4"][name],
                                      EP2_OF["F4"].get(name))
        FEATS[name] = (seeds4, feats4)
        if role == "6bba":
            LAB4[name] = label_rows(name, feats4)
        n4 = f" F4 pairs={int(sum(len(p) for p, _r, _P, _C in feats4))}"
        del dets4
    print(f"[{k_ + 1}/{len(_jobs)}] {name} {role}"
          f"{' fold' + str(FOLD_OF[name]) if role == '6bba' else ''}: T={T} "
          f"pairs={n_pairs} labelled={len(y_)} pos={int(y_.sum())}{n4} "
          f"{time.time() - t0:.0f}s", flush=True)
    del dets, X_, y_, seeds, feats
    gc.collect()
print(f"  join: {JOIN['datasets']} passes, {JOIN['pairs']} pairs, 100% "
      f"matched; NaN p_rev {JOIN['nan_p_rev']}; empty frames "
      f"{JOIN['empty_frames']}", flush=True)
print(f"  features in {(time.time() - t_all) / 60:.1f} min", flush=True)


def _auc(y, p):
    y = np.asarray(y)
    if len(y) == 0 or y.min() == y.max():
        warn_once("auc_one_class",
                  "an AUC was asked of rows with a single class -> NaN, "
                  "not a number")
        return float("nan")
    return float(roc_auc_score(y, p))


# ---- STAGE A: two out-of-fold +BIDIR rankers ------------------------------
print("\n=== STAGE A: two out-of-fold +BIDIR rankers (18 + p_fwd p_rev h_fused; "
      "poolsize hyperparameters), each trained on the 71 44b6 movies + the "
      "other 64 6bba movies ===", flush=True)
COLS = ARMS["+BIDIR"]
MODELS_FOLD = {}
for k in (0, 1):
    t0 = time.time()
    train_names = FOUR + [n for n in SIX if FOLD_OF[n] != k]
    held = [n for n in SIX if FOLD_OF[n] == k]
    Xa = np.concatenate([LAB[n][0] for n in train_names]).astype(np.float32)[:, COLS]
    ya = np.concatenate([LAB[n][1] for n in train_names])
    Xh = np.concatenate([LAB[n][0] for n in held]).astype(np.float32)[:, COLS]
    yh = np.concatenate([LAB[n][1] for n in held])
    perm = np.random.default_rng(0).permutation(len(ya))
    n_val = max(1000, len(ya) // 10)
    vi, ti = perm[:n_val], perm[n_val:]
    m = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    m.fit(Xa[ti], ya[ti])
    val_auc = _auc(ya[vi], m.predict_proba(Xa[vi])[:, 1])
    m.fit(Xa, ya)
    held_auc = _auc(yh, m.predict_proba(Xh)[:, 1])
    MODELS_FOLD[k] = m
    print(f"  fold{k}: trained on {len(train_names)} movies / {len(ya)} rows "
          f"({ya.mean():.3f} pos), val AUC {val_auc:.4f}; held-out fold "
          f"({len(held)} 6bba movies, {len(yh)} rows) AUC {held_auc:.4f}  "
          f"{time.time() - t0:.0f}s", flush=True)
    del Xa, ya, Xh, yh
    gc.collect()
del LAB
gc.collect()

print(f"\n=== RANKER AUC ON {DET_SRC} COORDINATES (out of fold, the "
      f"{len(LAB4)} movies whose coordinates differ between the arms) ===",
      flush=True)
print("  Same pairs, same GT labels, same fitted model as stage B; the ONLY "
      "difference from the arm this is compared against is where the points "
      "were placed. A drop here says the RANKER mis-reads hpara; a drop in "
      "score with this flat says the damage is downstream.", flush=True)
_p4, _y4 = [], []
for _k in (0, 1):
    _ns = [n for n in sorted(LAB4) if FOLD_OF[n] == _k]
    if not _ns:
        continue
    _X = np.concatenate([LAB4[n][0] for n in _ns]).astype(np.float32)[:, COLS]
    _y = np.concatenate([LAB4[n][1] for n in _ns])
    _p = MODELS_FOLD[_k].predict_proba(_X)[:, 1]
    _p4.append(_p)
    _y4.append(_y)
    print(f"  fold{_k}: {len(_ns)} movies, {len(_y)} rows ({_y.mean():.3f} "
          f"pos), AUC {_auc(_y, _p):.5f}", flush=True)
if _p4:
    _yy = np.concatenate(_y4)
    print(f"  POOLED {DET_SRC}: {len(_yy)} rows, AUC "
          f"{_auc(_yy, np.concatenate(_p4)):.5f}   <- compare arm to arm",
          flush=True)
del LAB4
gc.collect()
MODELS = {}                # arm_model() is kept but unused here


# ---- STAGE B: the full pipeline, the only verdict that counts ----------
def solve_one(dets, T, seeds, feats, model, cols, disapp=None, floor=None,
              divw=None):
    """rankwhy.solve_one with the model, its columns and the (joined)
    features as parameters. Everything up to and including the ILP, plus
    the sub-threshold repair pool. The post-link stages run in patch_final."""
    if model is None:                      # head-only: the column IS p
        probs = [np.nan_to_num(rows[:, cols], nan=0.0).astype(float)
                 if len(pairs) else np.zeros(0)
                 for pairs, rows, P, C in feats]
    else:
        probs = [model.predict_proba(rows[:, cols])[:, 1]
                 if len(pairs) else np.zeros(0)
                 for pairs, rows, P, C in feats]
    coords_rows, offset, n_glob = [], {}, 0
    for t in range(T):
        c, s_ = seeds[t]
        offset[t] = n_glob
        n_glob += len(c)
        for xyz in c:
            coords_rows.append([t, xyz[0], xyz[1], xyz[2]])
    coords = np.asarray(np.round(coords_rows), np.int16)
    cand_edges = []
    for t in range(T - 1):
        pairs, rows, P, C = feats[t]
        for (i_, j_), pr in zip(pairs, probs[t]):
            if pr >= (ILP_FLOOR if floor is None else floor):
                cand_edges.append((offset[t] + i_, offset[t + 1] + j_,
                                   float(pr), 0.0))
    graph = PIPE.build_graph(coords, cand_edges)
    if graph.num_edges() > 0:
        solver = td.solvers.ILPSolver(
            edge_weight=-1.0 * td.EdgeAttr("edge_prob"),
            appearance_weight=0.0,
            disappearance_weight=(ILP_DISAPP if disapp is None else disapp),
            division_weight=(DIV_W if divw is None else divw))
        with PIPE.suppress_output():
            graph = solver.solve(graph)
    na = graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
    nid_l = [int(v) for v in na["node_id"].to_list()]
    remap = {v: i for i, v in enumerate(nid_l)}
    nodes_arr = np.stack([np.asarray(na[c_].to_list(), float)
                          for c_ in ("t", "z", "y", "x")], axis=1)
    rows_e, dropped = [], 0
    if graph.num_edges() > 0:
        ea = graph.edge_attrs(attr_keys=["source_id", "target_id"])
        for a_, b_ in zip(ea["source_id"].to_list(),
                          ea["target_id"].to_list()):
            ra, rb = remap.get(int(a_)), remap.get(int(b_))
            if ra is None or rb is None:
                dropped += 1
                continue
            rows_e.append([ra, rb])
    if dropped:
        warn_once("ilp_endpoint_missing",
                  f"{dropped} solved edges had an endpoint absent from "
                  f"node_attrs and were dropped -> a LOWER BOUND")
    e2 = np.array(rows_e, int) if rows_e else np.zeros((0, 2), int)
    nodes0 = {i_ + 1: {"t": int(nodes_arr[i_, 0]),
                       "xyz": nodes_arr[i_, 1:].astype(float)}
              for i_ in range(len(nodes_arr))}
    edges0 = [(int(a_) + 1, int(b_) + 1) for a_, b_ in e2]
    cand, cand_sc, cand_trees = [], [], []
    for t in range(T):
        c_all = np.asarray(dets[t][0], float)
        s_all = np.asarray(dets[t][1], float)
        cm = (s_all >= CAND_THR) & (s_all < UNET_THRESH)
        cand.append(c_all[cm])
        cand_sc.append(s_all[cm])
        cand_trees.append(cKDTree(cand[-1] * SCALE) if len(cand[-1])
                          else None)
    pmap = {}
    for t in range(T - 1):
        pairs, rows, P, C = feats[t]
        for (i_, j_), pr, row in zip(pairs, probs[t], rows):
            pmap[(t, int(i_), int(j_))] = (float(pr), float(row[C_PFWD]),
                                            float(row[C_PREV]))
    cd = {}
    for g_, (t_, z_, y_, x_) in enumerate(coords.tolist()):
        cd.setdefault((int(t_), int(z_), int(y_), int(x_)), (int(t_), g_ - offset[int(t_)]))
    seed_of_node = {}
    for i_ in range(len(nodes_arr)):
        key = tuple(int(round(v)) for v in nodes_arr[i_])
        seed_of_node[i_ + 1] = cd.get(key)
    del graph, probs
    return nodes0, edges0, (cand, cand_sc, cand_trees), (pmap, seed_of_node)


def patch_final(nodes0, edges0, pool, out_path, divs=True, pfun=None, steal=None):
    """rankwhy.patch_final verbatim: the four post-link stages on a FRESH
    COPY, write the geff, return the final graph."""
    cand, cand_sc, cand_trees = pool
    nodes = {g: {"t": v["t"], "xyz": v["xyz"].copy()}
             for g, v in nodes0.items()}
    edges = list(edges0)
    nid2 = len(nodes) + 1
    nid2 = _gap_close(nodes, edges, cand, cand_sc, cand_trees, nid2)
    nodes, edges = _short_filter(nodes, edges)
    _linefit(nodes, edges)
    PRE_STATE["nodes"] = nodes
    PRE_STATE["edges"] = list(edges)
    FORK_LOG.clear()
    STEAL_LOG.clear()
    if divs:
        _add_safe_divisions(nodes, edges, pfun=pfun, steal=steal)
    ids = sorted(nodes)
    PRE_STATE["ids"] = ids
    PRE_STATE["final_edges"] = list(edges)
    rm = {g: i for i, g in enumerate(ids)}
    arr = np.array([[nodes[g]["t"], *nodes[g]["xyz"]] for g in ids], float)
    e3 = np.array([[rm[a_], rm[b_]] for a_, b_ in edges], int)
    dr._save(str(out_path), arr, e3)
    return nodes, edges


def score_one(gp, name):
    """score_dir's per-dataset body: the official scorer on one geff."""
    dset = open_dataset(root / "train" / name, require_tracks=True,
                        load_image=False)
    res_ = td.graph.IndexedRXGraph.from_geff(str(gp))
    g = res_[0] if isinstance(res_, tuple) else res_
    er = compute_metric(g, dset.tracks, scale=dset.scale)
    rec = (node_recall(g, dset.tracks)
           if g.num_edges() > 0 and g.num_nodes() > 0 else 0.0)
    gt = read_geff(geff_path(root, "train", name))
    est = gt["extra"].get("estimated_number_of_nodes")
    return per_sample_metrics(er, float(est) if est else float("nan"), rec)


def pdir(arm, domain):
    d = pathlib.Path(f"/kaggle/working/ep_{domain}_{ARM_TAG[arm]}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def arm_model(arm):
    if arm == "ANCHOR_D187":
        return MODEL_D187, MCOLS_D187
    if arm in HEAD_ONLY:
        return None, HEAD_ONLY[arm]
    return MODELS[arm], ARMS[arm]


from tracking_cellmot.division_metrics import score_divisions   # noqa: E402

FORK_LOG = {}
PRE_STATE = {}
DOMAIN = "BASE-6bba-all"
ARM_TAG = {"REF": "ref"}
FEAT20 = ["src", "d_pc1", "d_pq", "d_s", "dv", "p_c1", "p_q", "pf_c1", "pf_q",
          "pr_q", "sc_p", "sc_c1", "sc_q", "len_back", "len_c1", "len_q",
          "dens10", "n_orph", "t_frac", "rank_gap"]
FEAT_NEW = ["dv2", "ang", "sc_q_r", "sc_p_r", "dens5", "p_q_marg", "q_ends", "steal"]
FEAT_IMG = ["i_p", "c_p", "r_p", "br_p", "c_c1", "c_q", "r_c1", "r_q", "sym_i", "sym_r", "dau_ratio",
            "r_hm_p", "r_hm_c1", "r_hm_q", "rhm_max", "rhm_sum"]   # size variant: raw-image half-max radius
FEAT_NAMES = FEAT20 + FEAT_NEW + FEAT_IMG      # the first 20 are divbench3's, in order; the last 11 read the image

# ---------------------------------------------------------------------------
# The 67 COUNTED forks from 44b6's 71 movies (61 FP + 6 TP), lifted verbatim
# out of divbench5's forks_44b6.csv -- the same literal divbench8 used, which
# is what makes its +0.0037 and this run's joint column the same measurement.
# ---------------------------------------------------------------------------
FEAT44 = ['src', 'd_pc1', 'd_pq', 'd_s', 'dv', 'p_c1', 'p_q', 'pf_c1', 'pf_q', 'pr_q', 'sc_p', 'sc_c1', 'sc_q', 'len_back', 'len_c1', 'len_q', 'dens10', 'n_orph', 't_frac', 'rank_gap', 'dv2', 'ang', 'sc_q_r', 'sc_p_r', 'dens5', 'p_q_marg', 'q_ends', 'steal']
ROWS44 = [
    (1,[1,4.1151,8.3363,10.876,2.4125,0.99095,0.0029257,0.95523,0.13841,0.0022598,0.43706,0.45343,0.32243,3,12,12,4,11,0.29293,0.98803,5.0454,117.67,0.11531,0.43278,0,np.nan,0,0]),
    (1,[0,2.8752,1.4988,3.8679,1.0136,0.58258,0.56017,0.86076,0.82046,0.38619,0.42556,0.39483,0.39797,12,12,5,1,np.nan,0.58586,0.022409,2.142,121.07,0.78285,0.86616,0,0.55335,1,0]),
    (1,[0,0.44644,4.3286,4.2824,1.722,0.63998,0.61329,0.8744,0.83979,0.38223,0.37555,0.35302,0.34672,12,12,12,1,np.nan,0.26263,0.026696,3.8387,81.114,0.51954,0.66519,0,0.61003,0,0]),
    (1,[1,2.0472,4.8683,6.0338,2.5764,0.94912,0.075724,0.94316,0.24875,0.083781,0.32239,0.33415,0.24072,12,12,12,2,7,0.36364,0.8734,4.3478,115.29,0.11956,0.40108,0,-0.11295,1,0]),
    (1,[0,5.0614,2.6081,6.2966,2.5735,0.70451,0.60631,0.88246,0.75596,0.46187,0.42987,0.3757,0.4204,12,12,12,0,np.nan,0.55556,0.098208,5.4618,105.89,0.85981,0.89484,0,np.nan,0,0]),
    (1,[1,1.7802,4.4273,5.3591,4.6201,0.78471,0.36679,0.73471,0.37619,0.27553,0.34479,0.30911,0.32095,7,1,12,1,12,0.57576,0.41792,np.nan,112.18,0.60686,0.7107,0,0.034576,1,0]),
    (1,[1,0.83494,5.6107,6.1589,2.6777,0.63522,0.32427,0.96123,0.57836,0.21563,0.38835,0.4222,0.29644,12,12,3,1,12,0.79798,0.31095,3.8422,127.9,0.49452,0.87784,0,np.nan,1,0]),
    (1,[0,6.8734,3.1157,7.0175,2.2035,0.64817,0.61144,0.82395,0.5601,0.49129,0.25603,0.25481,0.22833,2,12,11,1,np.nan,0.57576,0.036729,4.7375,79.635,0.53394,0.69759,0,0.56171,1,0]),
    (2,[0,5.774,3.4477,8.8651,2.4341,0.86959,0.83053,0.9087,0.84767,0.4976,0.41727,0.45395,0.44849,4,9,12,1,np.nan,0.040404,0.039063,5.2184,146.93,0.7964,0.5485,0,0.77149,0,0]),
    (1,[0,2.1069,2.2592,4.0284,1.758,0.61807,0.55498,0.91761,0.67338,0.38248,0.36352,0.381,0.32478,12,9,12,0,np.nan,0.53535,0.063092,3.1077,134.61,0.19295,0.29914,0,0.53743,1,0]),
    (1,[0,3.3267,2.8554,4.9005,1.5094,0.64905,0.55562,0.69359,0.69979,0.42173,0.41723,0.36377,0.41129,12,4,4,0,np.nan,0.94949,0.093428,3.357,104.62,0.4812,0.50778,0,0.52999,0,0]),
    (1,[1,0.76358,6.4869,7.0711,2.4276,0.95781,0.018841,0.95714,0.066003,0.034449,0.36233,0.3688,0.2648,3,12,12,1,11,0.030303,0.93896,4.1345,137.79,0.61372,0.85294,0,np.nan,1,0]),
    (1,[0,1.3047,3.9649,4.5898,1.253,0.67912,0.57957,0.9451,0.65386,0.47714,0.29502,0.31511,0.28503,12,12,3,1,np.nan,0.31313,0.099556,2.4257,110.62,0.67593,0.70475,0,0.55898,1,0]),
    (1,[1,1.2118,5.9153,6.7042,2.5355,0.7374,0.47273,0.58676,0.50292,0.37011,0.36418,0.29519,0.33933,12,8,12,2,16,0.84848,0.26467,3.6326,126.3,0.80749,0.85652,0,0.45321,0,0]),
    (1,[0,4.3671,1.6011,4.7223,1.3234,0.62968,0.62641,0.87636,0.87603,0.55184,0.40434,0.36243,0.35398,5,12,12,0,np.nan,0.050505,0.0032743,2.786,92.725,0.90712,0.98355,0,0.62566,1,0]),
    (1,[1,1.3009,6.6108,7.8695,3.0903,0.96964,0.012602,0.91646,0.043552,0.008155,0.39657,0.35427,0.33567,2,12,12,2,14,0.020202,0.95704,6.6383,164.01,0.6341,0.84059,0,-0.20508,1,0]),
    (1,[0,3.2795,1.5823,4.7245,1.5098,0.59823,0.57144,0.68177,0.88495,0.55473,0.33491,0.35021,0.28454,8,3,11,1,np.nan,0.080808,0.026794,2.3008,150.83,0.89477,0.95714,0,np.nan,1,0]),
    (1,[0,2.9125,2.6155,4.8592,1.1532,0.74183,0.57313,0.89042,0.51084,0.50059,0.35048,0.35573,0.36624,12,12,12,1,np.nan,0.84848,0.1687,1.8635,122.95,0.97943,0.96837,1,0.21,0,0]),
    (2,[0,2.1734,6.5029,7.7295,3.7615,0.94098,0.63259,0.84914,0.73345,0.22059,0.44743,0.45148,0.43839,2,12,12,0,np.nan,0.18182,0.30839,6.0286,116.78,0.65907,0.73126,0,0.62636,1,0]),
    (1,[1,2.4867,8.2449,10.581,2.7842,0.99717,0.0022142,0.98775,0.040033,0.0058841,0.45991,0.44848,0.37155,12,12,5,2,11,0.63636,0.99496,4.8943,157.17,0.30119,0.83158,0,-0.51499,1,0]),
    (1,[1,1.9983,9.2557,10.864,2.4154,0.99703,0.0022693,0.93512,0.060776,4.2638e-05,0.44145,0.44082,0.1855,12,6,5,2,7,0.92929,0.99476,4.3443,140.08,0.019713,0.68384,0,-0.0012568,1,0]),
    (2,[1,1.6895,4.8564,6.0418,3.4421,0.66305,0.36494,0.58743,0.33651,0.40216,0.34988,0.25056,0.22991,12,12,5,0,9,0.35354,0.29811,5.3347,127.83,0.48663,0.95314,0,0.3553,1,0]),
    (1,[1,0.77525,5.6381,6.1161,2.4098,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,0.19646,0.19641,12,5,12,1,13,0.15152,np.nan,3.8952,125.02,0.28449,np.nan,-1,np.nan,0,0]),
    (1,[1,1.0091,6.5117,7.1609,2.6292,0.72858,0.3656,0.77819,0.57951,0.32442,0.26046,0.30719,0.28313,12,9,12,1,27,0.48485,0.36298,3.2625,126.73,0.88339,0.80647,0,0.10769,1,0]),
    (1,[0,3.4101,1.3176,4.5085,0.8806,0.60011,0.59498,0.64621,0.4845,0.36461,0.29077,0.28529,0.32988,12,12,6,2,np.nan,0.71717,0.0051278,2.3276,140.78,0.95674,0.89441,0,0.53891,1,0]),
    (1,[1,1.9156,4.4023,6.0262,2.3137,0.64758,0.3955,0.95207,0.45655,0.41841,0.32115,0.38011,0.30446,8,5,11,0,7,0.080808,0.25207,4.0547,141.86,0.61599,0.67264,0,np.nan,1,0]),
    (1,[1,2.574,5.4107,7.8548,2.7025,0.80209,0.38596,0.87328,0.34741,0.34578,0.34181,0.30462,0.32537,12,8,11,0,6,0.87879,0.41613,3.5966,157.84,0.68732,0.73989,0,np.nan,0,0]),
    (1,[0,1.6823,3.0786,4.336,1.9823,0.62922,0.58208,0.82486,0.83812,0.42372,0.34368,0.3296,0.29767,12,12,12,2,np.nan,0.27273,0.047143,3.1489,128.82,0.77219,0.88113,0,0.57185,1,0]),
    (1,[0,1.6263,3.302,4.5351,2.077,0.60451,0.58508,0.83254,0.80283,0.42848,0.27404,0.23265,0.25639,12,2,5,3,np.nan,0.93939,0.019426,3.0435,130.82,0.1631,0.21148,0,0.50975,0,0]),
    (1,[0,2.7992,2.6331,3.9463,2.0892,0.63036,0.56883,0.78653,0.75328,0.48443,0.32888,0.29541,0.28352,1,4,5,1,np.nan,0.010101,0.061537,2.5751,93.13,0.74596,0.85939,0,np.nan,1,0]),
    (1,[1,1.2503,5.9642,7.1354,2.5673,0.76293,0.35465,0.70178,0.2981,0.37237,0.36053,0.37868,0.30455,12,12,12,1,12,0.43434,0.40828,3.5983,157.5,0.80848,0.91203,0,np.nan,1,0]),
    (1,[0,1.9652,1.8661,3.2817,1.0737,0.60607,0.55983,0.81643,0.79169,0.41523,0.36528,0.29618,0.31106,12,11,12,1,np.nan,0.74747,0.046241,2.2935,117.84,0.55789,0.81608,0,0.5374,0,0]),
    (1,[1,1.258,5.5688,5.0733,3.0675,0.7288,0.44398,0.77377,0.57633,0.35056,0.37697,0.37111,0.37178,4,12,12,1,20,0.24242,0.28482,3.9955,60.706,0.84722,0.86601,0,0.43628,1,0]),
    (1,[0,2.6903,2.4733,5.1594,2.0591,0.67905,0.63926,0.91997,0.93177,0.45644,0.34732,0.33248,0.27692,9,5,6,1,np.nan,0.92929,0.03979,4.7371,175.43,0.41709,0.75206,0,0.63818,0,0]),
    (2,[0,3.9746,3.4159,7.1314,2.3882,0.82831,0.67897,0.8615,0.87748,0.45621,0.44163,0.3996,0.42588,12,12,12,0,np.nan,0.19192,0.14934,5.0056,149.48,0.92955,0.96637,0,np.nan,0,0]),
    (1,[1,1.7205,9.9023,11.312,2.2795,0.99628,0.022955,0.98924,0.23157,0.056075,0.37597,0.36304,0.16437,7,9,9,0,3,0.89899,0.97332,3.5255,142.27,0.021124,0.71339,0,np.nan,0,0]),
    (1,[1,0.5924,4.6163,5.0655,2.7065,0.7462,0.4588,0.18536,0.4194,0.44159,0.35691,0.33319,0.30507,12,9,9,3,20,0.89899,0.2874,3.8324,136.96,0.80311,0.91292,0,0.039221,0,0]),
    (2,[0,3.9542,4.0785,7.0564,3.2483,0.87255,0.59606,0.63334,0.53744,0.45551,0.46453,0.47506,0.41014,12,12,12,0,np.nan,0.52525,0.27649,5.8805,122.9,0.36162,0.74219,0,0.59386,1,0]),
    (1,[1,2.3,6.6586,8.8597,4.3604,0.90562,0.19475,0.89403,0.44565,0.14967,0.22167,0.18379,0.19644,3,12,7,4,32,0.030303,0.71087,6.656,160.47,0.39512,0.59517,0,-0.033039,1,0]),
    (1,[0,1.7475,3.8124,4.4129,2.3904,0.60608,0.5806,0.73627,0.79809,0.41216,0.27004,0.29544,0.21923,6,12,4,1,np.nan,0.060606,0.025488,4.1123,98.134,0.57699,0.8344,0,0.52682,1,0]),
    (1,[1,1.5252,5.5553,7.0175,2.6751,0.60145,0.32913,0.84147,0.35676,0.26645,0.29689,0.294,0.26753,2,12,4,0,30,0.19192,0.27231,4.425,161.36,0.82501,0.91092,0,0.30525,1,0]),
    (1,[1,2.2229,5.5549,7.6486,4.1216,0.50953,0.19723,0.48415,0.52542,0.15909,0.30502,0.25476,0.25812,12,9,12,1,25,0.33333,0.31231,7.6064,156.83,0.64027,0.83873,0,0.070326,1,0]),
    (1,[1,2.1292,5.3973,6.7911,4.3097,0.69937,0.46488,0.89042,0.7081,0.31417,0.24538,0.23839,0.2942,12,2,4,2,32,0.63636,0.23449,5.9294,122.81,0.80166,0.57125,0,0.24529,1,0]),
    (2,[1,1.862,6.7069,8.2038,2.8946,0.81168,0.36894,0.82122,0.4302,0.28295,0.35862,0.28418,0.32629,12,12,12,1,28,0.65657,0.44275,4.5906,139.01,0.90058,0.96063,0,0.33388,1,0]),
    (1,[0,1.7649,3.2146,4.5342,2.2966,0.70845,0.57014,0.9068,0.93971,0.34653,0.33056,0.29259,0.37585,12,6,12,5,np.nan,0.52525,0.13831,3.9676,128.8,0.75752,0.57686,0,0.56944,1,0]),
    (1,[0,2.579,1.8099,4.1409,1.1599,0.63783,0.55588,0.6785,0.90715,0.21906,0.32117,0.36429,0.38886,4,12,12,5,np.nan,0.58586,0.081941,2.5414,140.67,0.80584,0.54,0,0.55397,1,0]),
    (1,[1,1.6067,10.156,11.509,3.3107,0.62668,0.01455,0.70669,0.096276,0.036577,0.27491,0.30616,0.24245,2,10,12,4,24,0.85859,0.61213,4.521,144.97,0.25106,0.36334,0,-0.089783,0,0]),
    (1,[1,2.4287,7.5696,8.9242,2.6745,0.60788,0.51065,0.8042,0.73529,0.45098,0.31246,0.36518,0.18796,1,11,6,4,25,0.87879,0.097233,4.5874,116.57,0.087507,0.50556,0,np.nan,1,0]),
    (1,[1,2.9573,5.4401,6.1442,2.3459,0.55974,0.35012,0.51931,0.34111,0.35572,0.41755,0.45069,0.4199,4,12,12,0,17,0.060606,0.20962,5.0676,88.952,0.74032,0.72985,0,np.nan,1,0]),
    (1,[0,5.5975,2.7858,7.7487,2.5709,0.89594,0.5653,0.83914,0.70983,0.34196,0.51071,0.43892,0.41062,12,12,12,0,np.nan,0.70707,0.33064,5.3094,132.2,0.70039,0.99917,0,0.5563,0,0]),
    (1,[0,3.5682,2.4296,5.0904,1.8244,0.77458,0.67228,0.73546,0.87333,0.50207,0.41613,0.46567,0.44803,12,12,12,2,np.nan,0.27273,0.1023,3.6785,114.82,0.47868,0.33013,0,0.66878,1,0]),
    (1,[1,2.0359,6.3106,8.1549,np.nan,0.9341,0.20802,0.92848,0.64646,0.15307,0.34328,0.28482,0.28239,12,0,12,1,10,0.79798,0.72608,np.nan,151.28,0.77305,0.90908,0,0.18543,1,0]),
    (1,[1,1.252,6.9832,7.9474,2.2777,0.99073,0.026301,0.96065,0.19303,0.040031,0.44871,0.43712,0.21517,2,12,4,0,15,0.020202,0.96443,5.0093,137.2,0.31101,0.99953,0,-0.03352,1,0]),
    (1,[1,0.35639,5.7771,5.966,2.3329,0.74788,0.20334,0.90307,0.36605,0.24193,0.21271,0.2579,0.21336,5,5,3,1,13,0.12121,0.54455,5.4103,120.52,0.30021,0.29634,0,0.19674,1,0]),
    (1,[1,0.76926,7.5988,8.0764,2.3997,0.84606,0.10354,0.75676,0.33311,0.12921,0.21642,0.23987,0.20331,12,12,3,1,15,0.22222,0.74252,4.1277,126.14,0.24052,0.31765,0,np.nan,1,0]),
    (1,[1,1.4665,5.3516,6.7607,2.3252,0.92969,0.036197,0.95562,0.087872,0.039461,0.22409,0.19539,0.18064,12,12,12,2,14,0.54545,0.89349,3.3597,161.88,0.11682,0.36651,0,-0.35029,1,0]),
    (1,[1,1.2621,7.0979,7.6668,2.5941,0.94827,0.063484,0.96144,0.13212,0.079767,0.31833,0.33325,0.21218,12,12,12,3,15,0.59596,0.88479,5.178,112.33,0.29328,0.82363,0,-0.32498,1,0]),
    (1,[1,2.1617,6.3849,7.7897,2.3413,0.50183,0.17819,0.70337,0.37938,0.23799,0.17962,0.15552,0.21431,12,1,8,3,24,0.62626,0.32364,np.nan,123.51,0.30566,0.11159,0,0.16024,1,0]),
    (1,[1,2.0447,4.3836,5.7961,2.9803,0.87121,0.17916,0.86841,0.29135,0.16016,0.16716,0.18336,0.17862,5,11,11,2,27,0.87879,0.69205,4.1633,124.67,0.10696,0.059393,0,0.044559,0,0]),
    (1,[0,2.6454,2.9778,3.7391,1.5095,0.66244,0.60244,0.82838,0.86765,0.39293,0.29309,0.25295,0.21658,12,8,5,1,np.nan,0.88889,0.059996,2.9648,83.129,0.31861,0.73307,0,0.59958,1,0]),
    (1,[1,2.5171,6.6315,6.3796,3.2765,0.81391,0.2008,0.92333,0.26727,0.1763,0.40384,0.40985,0.26374,1,4,12,1,13,0.010101,0.61311,5.1784,73.264,0.15429,0.70612,0,-0.2025,1,0]),
    (1,[0,4.0667,2.4257,5.3746,0.99341,0.79324,0.67594,0.85704,0.87886,0.41882,0.34246,0.32609,0.32127,12,12,4,0,np.nan,0.67677,0.1173,2.5508,109.13,0.31459,0.39793,0,np.nan,1,0]),
    (1,[1,2.4507,4.2093,5.4524,2.6586,0.57816,0.52454,0.92337,0.82684,0.45283,0.39552,0.35834,0.32944,7,12,12,0,7,0.35354,0.053623,3.8462,106.92,0.37475,0.6762,0,np.nan,1,0]),
    (1,[1,0.60475,11.712,11.517,2.6209,0.97187,np.nan,0.94563,np.nan,np.nan,0.32084,0.42056,0.3718,12,3,12,1,3,0.44444,np.nan,3.5691,69.844,0.56327,0.33688,0,np.nan,1,0]),
    (1,[0,1.1981,5.449,6.2552,2.9248,0.6249,0.56864,0.68435,0.8808,0.36822,0.42938,0.34683,0.36032,12,12,12,1,np.nan,0.48485,0.056262,5.8746,127.79,0.51158,0.84176,0,0.56197,1,0]),
    (1,[1,1.5721,8.6297,9.744,2.3739,0.62659,0.038368,0.5213,0.17958,0.043508,0.36507,0.38005,0.23887,7,12,8,3,18,0.63636,0.58822,3.9333,131.56,0.74681,0.9592,0,-0.36648,1,0]),
    (1,[1,1.3635,6.5871,6.6912,3.0074,0.68413,0.47365,0.77768,0.55409,0.38562,0.24719,0.23466,0.23136,10,12,4,1,9,0.54545,0.21048,6.5008,88.475,0.36884,0.4526,0,0.43496,1,0]),
]
assert FEAT44 == FEAT20 + FEAT_NEW, "the 44b6 fork table's columns are not this run's"   # forkprop: image columns padded below
X44 = np.array([r_[1] for r_ in ROWS44], float)
Y44 = np.array([1 if r_[0] == 2 else 0 for r_ in ROWS44], int)
assert X44.shape == (67, len(FEAT20) + len(FEAT_NEW)), X44.shape
X44 = np.hstack([X44, np.full((len(X44), len(FEAT_IMG)), np.nan)])   # the joint anchor: no image columns
_f44 = [p_ for p_ in pathlib.Path("/kaggle/input").glob("**/forks_44b6img.csv")]
assert _f44, "attach the dataset faustine649/biohub-forktables (forks_44b6img.csv)"
import pandas as _pd                                              # noqa: E402
_t44 = _pd.read_csv(_f44[0])
for _c in FEAT_NAMES:
    if _c not in _t44.columns:
        _t44[_c] = np.nan                                       # size variant: the dataset's table predates the size columns
assert all(_c in _t44.columns for _c in FEAT_NAMES), list(_t44.columns[5:])
_t44 = _t44[_t44.label >= 1]
X44IMG = _t44[FEAT_NAMES].to_numpy(float)
Y44IMG = (_t44.label == 2).astype(int).to_numpy()
print(f"  jointimg: {len(X44IMG)} counted 44b6 forks with image columns from {_f44[0]} (TP {int(Y44IMG.sum())})", flush=True)
assert int(Y44.sum()) == 6, int(Y44.sum())
WALK_ORDER = ["mother_absent", "daughter_absent", "recovered", "mother_deleted_post",
              "1a_mother_no_child", "1b_mother_forked_elsewhere",
              "0_sister_deleted_post", "2_mother_no_single_pred",
              "3_child_wrong_frame", "4_mother_child_over_10um",
              "5a_sister_not_orphan", "5b_no_orphans_in_frame",
              "5c_sister_over_12um_of_mother", "6_sister_gap_over_15um",
              "7_not_the_nearest_orphan", "8a_daughter_not_one_child",
              "8b_diverge_under_2.25um", "9_passed_all_gates_then_capped_or_taken"]


def _gt_struct(name):
    gt = read_geff(geff_path(root, "train", name))
    g_t = np.asarray(gt["t"], int)
    g_pos = np.stack([np.asarray(gt[c], float) for c in ("z", "y", "x")], axis=1)
    ids = [int(v) for v in gt["node_ids"]]
    row_of = {v: i for i, v in enumerate(ids)}
    kids = defaultdict(list)
    for a, b in gt["edges"]:
        ra, rb = row_of.get(int(a)), row_of.get(int(b))
        if ra is not None and rb is not None:
            kids[ra].append(rb)
    mothers = {ids[r]: (r, [k for k in ks]) for r, ks in kids.items() if len(ks) >= 2}
    return g_t, g_pos, ids, mothers


_GTE_CACHE = {}


def _gt_edge_rows(name):
    """(set of GT edges as (row, row), {t: (rows, KDTree)}) for the true-other
    test in the 5a walk. Cached: the geff read is not free."""
    if name in _GTE_CACHE:
        return _GTE_CACHE[name]
    gt = read_geff(geff_path(root, "train", name))
    g_t = np.asarray(gt["t"], int)
    g_pos = np.stack([np.asarray(gt[c], float) for c in ("z", "y", "x")], axis=1)
    ids = [int(v) for v in gt["node_ids"]]
    row_of = {v: i for i, v in enumerate(ids)}
    es = set()
    for a, b in gt["edges"]:
        ra, rb = row_of.get(int(a)), row_of.get(int(b))
        if ra is not None and rb is not None:
            es.add((ra, rb))
    by_t = {}
    for t_ in np.unique(g_t):
        rows_ = np.where(g_t == t_)[0]
        by_t[int(t_)] = (rows_, cKDTree(g_pos[rows_] * SCALE))
    _GTE_CACHE[name] = (es, by_t)
    return _GTE_CACHE[name]


def _chain_len(start, step_map, cap=12):
    n, cur = 0, start
    while n < cap:
        nxt = step_map.get(cur, [])
        if len(nxt) != 1:
            break
        cur = nxt[0]
        n += 1
    return n


def _aux_extra(seeds, pmap):
    """Per-movie lookups the new fork features need, built once per solve.

    scsort: every detection score in the movie, sorted, so a raw score can be
            read as a per-movie quantile (sc_q / sc_p are the two strongest
            features and both are raw scores whose scale moves with the movie
            and with the imaging batch).
    best2 : for each (t, j) the two best ranker probabilities over all parents
            i in frame t, with their i, so "is this mother the best available
            explanation for this node" is a lookup.
    """
    allsc = np.concatenate([np.asarray(s_, float) for _c, s_ in seeds]) if seeds \
        else np.zeros(0)
    scsort = np.sort(allsc)
    best2 = {}
    for (t_, i_, j_), v in pmap.items():
        pr = v[0]
        if not np.isfinite(pr):
            continue
        cur = best2.get((t_, j_))
        if cur is None:
            best2[(t_, j_)] = [(pr, i_), (-1.0, -1)]
        elif pr > cur[0][0]:
            cur[1] = cur[0]
            cur[0] = (pr, i_)
        elif pr > cur[1][0]:
            cur[1] = (pr, i_)
    return scsort, best2


# ---- forkimg block ----
from scipy import ndimage as _ndi                                # noqa: E402
from viz_cells import open_image as _open_image                  # noqa: E402
FI_HALF_UM = 8.0
FI_PEAK_UM = 2.0
FI_SIZE_UM = 3.5
FI_SIGMA_VOX = (0.6, 1.5, 1.5)
IMG_CTX = {}                       # name, img (set around REF's _judge), frame cache
FI_STAT = defaultdict(int)


def _fi_crop(img, t, xyz_vox):
    """smoothed crop around a node, its offset; None when the frame is out of range."""
    if t < 0 or t >= img.shape[0]:
        return None, None
    hz = int(np.ceil(FI_HALF_UM / SCALE[0]))
    hy = int(np.ceil(FI_HALF_UM / SCALE[1]))
    z, y, x = (int(round(float(v))) for v in xyz_vox)
    z0, y0, x0 = max(0, z - hz), max(0, y - hy), max(0, x - hy)
    z1, y1, x1 = z + hz + 1, y + hy + 1, x + hy + 1
    try:
        crop = np.asarray(img[t, z0:z1, y0:y1, x0:x1])
    except Exception:                                            # noqa: BLE001
        fr = IMG_CTX.get("frames", {})
        if t not in fr:
            fr.clear()
            fr[t] = np.asarray(img[t])
            IMG_CTX["frames"] = fr
        crop = fr[t][z0:z1, y0:y1, x0:x1]
    if crop.size == 0:
        return None, None
    sm = _ndi.gaussian_filter(crop.astype(np.float32), FI_SIGMA_VOX)
    return sm, np.array([z0, y0, x0], float)


def _fi_node(img, t, xyz_vox):
    """(peak, contrast, radius of gyration um) of one node; NaNs when unreadable."""
    sm, off = _fi_crop(img, t, xyz_vox)
    if sm is None:
        return np.nan, np.nan, np.nan
    c = np.asarray(xyz_vox, float) - off
    idx = np.indices(sm.shape).reshape(3, -1).T.astype(float)
    d = np.linalg.norm((idx - c) * SCALE, axis=1)
    v = sm.reshape(-1)
    near = d <= FI_PEAK_UM
    if not near.any():
        return np.nan, np.nan, np.nan
    bg = float(np.median(v))
    j = int(np.argmax(np.where(near, v, -np.inf)))
    peak = float(v[j])
    con = (peak - bg) / max(bg, 1e-6)
    dp = np.linalg.norm((idx - idx[j]) * SCALE, axis=1)
    m = dp <= FI_SIZE_UM
    w = np.clip(v[m] - bg, 0, None)
    rg = float(np.sqrt((w * dp[m] ** 2).sum() / w.sum())) if w.sum() > 0 else np.nan
    return peak, float(con), rg


def _fi_hm(img, t, xyz_vox):
    """raw-image half-maximum radius (um) of one node: on the smoothed crop, the connected set at or above
    bg + 0.5 (peak - bg) that contains the peak (peak = max within FI_PEAK_UM of the node, bg = crop median),
    reported as the radius of the sphere of equal volume. NaN when unreadable."""
    sm, off = _fi_crop(img, t, xyz_vox)
    if sm is None:
        return np.nan
    c = np.asarray(xyz_vox, float) - off
    zz, yy, xx = np.indices(sm.shape)
    d = np.sqrt(((zz - c[0]) * SCALE[0]) ** 2 + ((yy - c[1]) * SCALE[1]) ** 2 + ((xx - c[2]) * SCALE[2]) ** 2)
    near = d <= FI_PEAK_UM
    if not near.any():
        return np.nan
    bg = float(np.median(sm))
    j = np.unravel_index(int(np.argmax(np.where(near, sm, -np.inf))), sm.shape)
    peak = float(sm[j])
    if peak <= bg:
        return np.nan
    lab_, _n = _ndi.label(sm >= bg + 0.5 * (peak - bg))
    vol = int((lab_ == lab_[j]).sum()) * float(np.prod(SCALE))
    return float((3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0))


def fork_img_feats(name, nodes, pred, p, c1, q):
    out = {k_: np.nan for k_ in FEAT_IMG}
    img = IMG_CTX.get("img")
    if img is None or IMG_CTX.get("name") != name:
        FI_STAT["no_img"] += 1
        return out
    t = int(nodes[p]["t"])
    i_p, c_p, r_p = _fi_node(img, t, nodes[p]["xyz"])
    pr = pred.get(p, [])
    if len(pr) == 1:
        i_pm, _c, _r = _fi_node(img, t - 1, nodes[pr[0]]["xyz"])
    else:
        i_pm = np.nan
    i_1, c_1, r_1 = _fi_node(img, t + 1, nodes[c1]["xyz"])
    i_q, c_q, r_q = _fi_node(img, t + 1, nodes[q]["xyz"])
    out.update(i_p=i_p, c_p=c_p, r_p=r_p,
               br_p=(i_p / i_pm) if (np.isfinite(i_pm) and i_pm > 0) else np.nan,
               c_c1=c_1, c_q=c_q, r_c1=r_1, r_q=r_q,
               sym_i=(abs(i_1 - i_q) / (i_1 + i_q)) if (np.isfinite(i_1) and np.isfinite(i_q) and (i_1 + i_q) > 0) else np.nan,
               sym_r=(abs(r_1 - r_q) / (r_1 + r_q)) if (np.isfinite(r_1) and np.isfinite(r_q) and (r_1 + r_q) > 0) else np.nan,
               dau_ratio=((i_1 + i_q) / (2 * i_p)) if (np.isfinite(i_1) and np.isfinite(i_q) and np.isfinite(i_p) and i_p > 0) else np.nan)
    r_hp, r_h1, r_hq = _fi_hm(img, t, nodes[p]["xyz"]), _fi_hm(img, t + 1, nodes[c1]["xyz"]), _fi_hm(img, t + 1, nodes[q]["xyz"])
    out.update(r_hm_p=r_hp, r_hm_c1=r_h1, r_hm_q=r_hq,
               rhm_max=(max(r_h1, r_hq) / r_hp) if (np.isfinite(r_hp) and np.isfinite(r_h1) and np.isfinite(r_hq) and r_hp > 0) else np.nan,
               rhm_sum=((r_h1 + r_hq) / r_hp) if (np.isfinite(r_hp) and np.isfinite(r_h1) and np.isfinite(r_hq) and r_hp > 0) else np.nan)
    FI_STAT["forks"] += 1
    FI_STAT["nan_rows"] += int(not np.isfinite(i_p))
    return out


def fp_contrast(nodes, g):
    """the image contrast of one node (memoised per movie); -1 when no image."""
    img = IMG_CTX.get("img")
    if img is None:
        FP_STAT["no_img"] += 1
        return -1.0
    n_ = nodes[g]
    key = (int(n_["t"]),) + tuple(int(round(float(v))) for v in n_["xyz"])
    if key not in FP_MEMO:
        _pk, con, _rg = _fi_node(img, int(n_["t"]), n_["xyz"])
        FP_MEMO[key] = float(con) if np.isfinite(con) else -1.0
        FP_STAT["crops"] += 1
    return FP_MEMO[key]
# ---- end forkimg block ----


def fork_rows(name, k, seeds, aux, tp_forks, fp_forks):
    """Every fork in the final graph, its features, its verdict."""
    pmap, seed_of_node, scsort, best2 = aux
    nodes, edges, ids = PRE_STATE["nodes"], PRE_STATE["final_edges"], PRE_STATE["ids"]
    succ, pred = defaultdict(list), defaultdict(list)
    for a, b in edges:
        succ[a].append(b)
        pred[b].append(a)
    T = max(v["t"] for v in nodes.values()) + 1
    trees = {}
    for t in range(T):
        c, _s = seeds[t]
        trees[t] = cKDTree(np.asarray(c, float) * SCALE) if len(c) else None
    pos_of = {ids[i]: i for i in range(len(ids))}
    tp_n = {ids[i] for i in tp_forks if i < len(ids)}
    fp_n = {ids[i] for i in fp_forks if i < len(ids)}

    def _rank(v):
        if not np.isfinite(v) or not len(scsort):
            return np.nan
        return float(np.searchsorted(scsort, v) / len(scsort))

    def _step(g):
        n_ = succ.get(g, [])
        return n_[0] if len(n_) == 1 else None

    rows = []
    for p, ch in succ.items():
        if len(ch) < 2:
            continue
        t = nodes[p]["t"]
        pq = [(q, FORK_LOG.get((p, q))) for q in ch]
        patched = [q for q, lg in pq if lg is not None]
        if patched:
            q = patched[0]
            src = 1
            c1 = FORK_LOG[(p, q)]["c1"]
            if c1 not in ch:
                c1 = [x for x in ch if x != q][0]
        else:
            src = 0
            # the lower-ranked child is "q"
            sp = seed_of_node.get(p)
            pr = {}
            for x in ch:
                sx = seed_of_node.get(x)
                pr[x] = pmap.get((sp[0], sp[1], sx[1]), (np.nan,))[0] if sp and sx and sx[0] == sp[0] + 1 else np.nan
            ch_sorted = sorted(ch, key=lambda x: (np.nan_to_num(pr[x], nan=-1.0)))
            q, c1 = ch_sorted[0], ch_sorted[-1]

        def _p(a, b):
            sa, sb = seed_of_node.get(a), seed_of_node.get(b)
            if sa is None or sb is None or sb[0] != sa[0] + 1:
                return (np.nan, np.nan, np.nan)
            return pmap.get((sa[0], sa[1], sb[1]), (np.nan, np.nan, np.nan))

        def _sc(a):
            sa = seed_of_node.get(a)
            return float(seeds[sa[0]][1][sa[1]]) if sa is not None else np.nan
        p_c1, pf_c1, _ = _p(p, c1)
        p_q, pf_q, pr_q = _p(p, q)
        d_pc1 = _dist_um(nodes[p]["xyz"], nodes[c1]["xyz"])
        d_pq = _dist_um(nodes[p]["xyz"], nodes[q]["xyz"])
        d_s = _dist_um(nodes[c1]["xyz"], nodes[q]["xyz"])
        c1n, qn = succ.get(c1, []), succ.get(q, [])
        dv = (_dist_um(nodes[c1n[0]]["xyz"], nodes[qn[0]]["xyz"]) - d_s
              if len(c1n) == 1 and len(qn) == 1 else np.nan)
        # --- the eight new ones ------------------------------------------
        c1nn = _step(c1n[0]) if len(c1n) == 1 else None
        qnn = _step(qn[0]) if len(qn) == 1 else None
        dv2 = (_dist_um(nodes[c1nn]["xyz"], nodes[qnn]["xyz"]) - d_s
               if c1nn is not None and qnn is not None else np.nan)
        v1 = (np.asarray(nodes[c1]["xyz"], float) - np.asarray(nodes[p]["xyz"], float)) * SCALE
        v2 = (np.asarray(nodes[q]["xyz"], float) - np.asarray(nodes[p]["xyz"], float)) * SCALE
        n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        ang = (float(np.degrees(np.arccos(np.clip(float(v1 @ v2) / (n1 * n2), -1.0, 1.0))))
               if n1 > 1e-9 and n2 > 1e-9 else np.nan)
        sc_p_v, sc_q_v = _sc(p), _sc(q)
        tr = trees.get(t)
        pxyz = np.asarray(nodes[p]["xyz"], float) * SCALE
        dens10 = (len(tr.query_ball_point(pxyz, r=10.0)) - 1) if tr is not None else np.nan
        dens5 = (len(tr.query_ball_point(pxyz, r=5.0)) - 1) if tr is not None else np.nan
        sq, sp_ = seed_of_node.get(q), seed_of_node.get(p)
        p_q_marg = np.nan
        if sq is not None and sp_ is not None and np.isfinite(p_q):
            b = best2.get((sq[0] - 1, sq[1]))
            if b is not None:
                other = b[0][0] if b[0][1] != sp_[1] else b[1][0]
                if other >= 0:
                    p_q_marg = float(p_q - other)
        cur, steps = q, 0
        while steps < 200:
            nx = _step(cur)
            if nx is None:
                break
            cur, steps = nx, steps + 1
        q_ends = 1.0 if (len(succ.get(cur, [])) == 0 and nodes[cur]["t"] < T - 1) else 0.0
        lg = FORK_LOG.get((p, q), {})
        rank_gap = (p_c1 - p_q) if (p_c1 == p_c1 and p_q == p_q) else np.nan
        row = dict(src=src, d_pc1=d_pc1, d_pq=d_pq, d_s=d_s, dv=dv, p_c1=p_c1, p_q=p_q,
                   pf_c1=pf_c1, pf_q=pf_q, pr_q=pr_q, sc_p=sc_p_v, sc_c1=_sc(c1), sc_q=sc_q_v,
                   len_back=_chain_len(p, pred), len_c1=_chain_len(c1, succ),
                   len_q=_chain_len(q, succ), dens10=dens10,
                   n_orph=lg.get("n_orph", np.nan), t_frac=t / max(1, T - 1),
                   rank_gap=rank_gap,
                   dv2=dv2, ang=ang, sc_q_r=_rank(sc_q_v), sc_p_r=_rank(sc_p_v),
                   dens5=dens5, p_q_marg=p_q_marg, q_ends=q_ends,
                   steal=float(lg.get("steal", 0)))
        row.update(fork_img_feats(name, nodes, pred, p, c1, q))
        label = 2 if p in tp_n else (1 if p in fp_n else 0)     # 2 TP, 1 FP, 0 uncounted
        rows.append((name, k, p, q, label, [row[f] for f in FEAT_NAMES]))
    return rows
def walk_lost(name, scores, pfun=None, steal_rows=None):
    """divgate2's walk on the pre-patch graph for every GT division the
    scorer did not pair. Returns {condition: count} and a list of
    (mother_gt_id, condition).

    When `steal_rows` is given, every case that stops at 5a is written into
    it in full: the ranker's p for the edge the patch WOULD make against the
    p for the edge the ILP DID make, both distances, whether the competing
    edge is a true GT edge, and whether the competitor already forks (which
    the steal pass refuses). divsteal measured this on Zebrahub with n=54 and
    on ONE case in each competition domain; this is the competition
    population, on production primaries."""
    g_t, g_pos, gids, mothers = _gt_struct(name)
    nodes, edges = PRE_STATE["nodes"], PRE_STATE["edges"]
    gt_edge_rows, gt_by_t = _gt_edge_rows(name)

    def gt_row_of(g):
        """graph node -> GT row within MATCH_UM, or None."""
        t_ = int(nodes[g]["t"])
        b = gt_by_t.get(t_)
        if b is None:
            return None
        rows_, tr_ = b
        d_, i_ = tr_.query(np.asarray(nodes[g]["xyz"], float) * SCALE)
        return rows_[int(i_)] if d_ <= MATCH_UM else None
    succ, pred = defaultdict(list), defaultdict(list)
    pre_set = set()
    for a, b in edges:
        succ[a].append(b)
        pred[b].append(a)
        pre_set.add((a, b))
    by_t = defaultdict(list)
    for g, v in nodes.items():
        by_t[int(v["t"])].append(g)
    trees = {}
    for t, ids_ in by_t.items():
        trees[t] = (ids_, cKDTree(np.asarray([nodes[g]["xyz"] for g in ids_], float) * SCALE))
    orph_ids, orph_tree = {}, {}
    for t, ids_ in by_t.items():
        o = [g for g in ids_ if not pred.get(g)]
        orph_ids[t] = o
        if o:
            orph_tree[t] = cKDTree(np.asarray([nodes[g]["xyz"] for g in o], float) * SCALE)

    def match(r):
        t = int(g_t[r])
        if t not in trees:
            return None
        ids_, tr = trees[t]
        d, i = tr.query(g_pos[r] * SCALE)
        return ids_[int(i)] if d <= MATCH_UM else None

    out = defaultdict(int)
    det = []
    for gid, (rm, kid_rows) in mothers.items():
        if scores.get(gid, 0) == 1:
            out["recovered"] += 1
            det.append((gid, "recovered"))
            continue
        ga = match(rm)
        if ga is None:
            out["mother_absent"] += 1
            det.append((gid, "mother_absent"))
            continue
        d1, d2 = match(kid_rows[0]), match(kid_rows[1])
        if d1 is None or d2 is None:
            out["daughter_absent"] += 1
            det.append((gid, "daughter_absent"))
            continue
        kept = [g for g in (d1, d2) if (ga, g) in pre_set]
        ch = succ.get(ga, [])
        cond = None
        if len(ch) == 0:
            cond = "1a_mother_no_child"
        elif len(ch) >= 2:
            cond = "1b_mother_forked_elsewhere"
        if cond is None:
            c1 = ch[0]
            t_ = int(nodes[ga]["t"])
            q = d2 if (kept and kept[0] == d1) else d1
            if len(pred.get(ga, [])) != 1:
                cond = "2_mother_no_single_pred"
            elif int(nodes[c1]["t"]) != t_ + 1:
                cond = "3_child_wrong_frame"
            elif _dist_um(nodes[ga]["xyz"], nodes[c1]["xyz"]) > DIV_CHILD_UM:
                cond = "4_mother_child_over_10um"
            elif pred.get(q):
                cond = "5a_sister_not_orphan"
                if steal_rows is not None:
                    r = pred[q][0]
                    p_pq = pfun(ga, q) if pfun else float("nan")
                    p_rq = pfun(r, q) if pfun else float("nan")
                    gr = gt_row_of(r)
                    gq = gt_row_of(q)
                    true_other = (1 if (gr is not None and gq is not None
                                        and (gr, gq) in gt_edge_rows) else
                                  (0 if (gr is not None and gq is not None) else -1))
                    c1n_, qn_ = succ.get(c1, []), succ.get(q, [])
                    dv_ = (_dist_um(nodes[c1n_[0]]["xyz"], nodes[qn_[0]]["xyz"])
                           - _dist_um(nodes[c1]["xyz"], nodes[q]["xyz"])
                           if len(c1n_) == 1 and len(qn_) == 1 else float("nan"))
                    steal_rows.append(dict(
                        movie=name, gt_mother=int(gid), p_pq=float(p_pq),
                        p_rq=float(p_rq),
                        d_pq=_dist_um(nodes[ga]["xyz"], nodes[q]["xyz"]),
                        d_rq=_dist_um(nodes[r]["xyz"], nodes[q]["xyz"]),
                        d_s=_dist_um(nodes[c1]["xyz"], nodes[q]["xyz"]),
                        dv=float(dv_), true_other=true_other,
                        r_forks=1 if len(succ.get(r, [])) > 1 else 0,
                        q_multi_pred=1 if len(pred.get(q, [])) > 1 else 0))
            elif orph_tree.get(t_ + 1) is None:
                cond = "5b_no_orphans_in_frame"
            elif _dist_um(nodes[ga]["xyz"], nodes[q]["xyz"]) > DIV_PARENT_UM:
                cond = "5c_sister_over_12um_of_mother"
            else:
                d_s = _dist_um(nodes[c1]["xyz"], nodes[q]["xyz"])
                if d_s > DIV_SISTER_UM:
                    cond = "6_sister_gap_over_15um"
                else:
                    d_q, q_idx = orph_tree[t_ + 1].query(np.asarray(nodes[c1]["xyz"], float) * SCALE)
                    if orph_ids[t_ + 1][q_idx] != q or d_q > DIV_SISTER_UM:
                        cond = "7_not_the_nearest_orphan"
                    else:
                        c1n, qn = succ.get(c1, []), succ.get(q, [])
                        if len(c1n) != 1 or len(qn) != 1:
                            cond = "8a_daughter_not_one_child"
                        else:
                            dv = _dist_um(nodes[c1n[0]]["xyz"], nodes[qn[0]]["xyz"]) - d_s
                            cond = ("8b_diverge_under_2.25um" if dv < DIV_DIVERGE_UM
                                    else "9_passed_all_gates_then_capped_or_taken")
        out[cond] += 1
        det.append((gid, cond))
    return out, det


print(f"\n=== STAGE B: {len(SIX)} 6bba movies on F4 primaries, one production solve "
      f"each at DIV_W {DIV_W} (out-of-fold +BIDIR, disappearance {ILP_DISAPP}, "
      f"floor {PROD_FLOOR}) plus {len(DIVW_ARMS)} arms that RE-SOLVE at "
      f"{', '.join(str(d) for _a, d in DIVW_ARMS)}; every fork logged and judged ===",
      flush=True)
print("  ANCHOR: D187 on the 12-movie holdout at floor 0.10 must read adj_J 0.9216 "
      "(F4-6bba hard, div_J 0.0606). Soft: the production +BIDIR pkl at floor 0.48 "
      "on the same 12 = ilpsweep's F48 row 0.9261. REF over the 128 must read "
      "50/249/75, div_J 0.1337, adj_J 0.9133, score 0.9266 (divbench3).", flush=True)

TMP = "/tmp/db4"            # geffs are intermediates: ~900 of them under
pathlib.Path(TMP).mkdir(parents=True, exist_ok=True)   # /kaggle/working make
CAPS_BY_ARM = {}           # the output slow to list and slow to fetch
ROWS = []
PER = {}
ANCHOR_ROWS = []
SOFT_ROWS = []
INCOMPLETE = []
FORKS = []             # REF forks: (name, fold, p, q, label, feats)
ARM_FORKS = {a: [] for a, _ in DIVW_ARMS}
ARM_ROWS = {a: [] for a, _ in DIVW_ARMS}
ARM_PER = {a: {} for a, _ in DIVW_ARMS}
ARM_MOVIE = {a: {} for a, _ in DIVW_ARMS}
STEAL5A = []           # one dict per GT division that stops at 5a
MOVIE = {}             # name -> dict(arr, ids, edges, rm)
WALK = defaultdict(int)
WALK_FOLD = {0: defaultdict(int), 1: defaultdict(int)}
tb = time.time()


def _snapshot():
    """A COMPACT final graph: coordinates as float32, edges already remapped
    to row indices as int32, and (p, q) -> edge row for the forks the pruner
    will want to remove. divbench3 kept python lists and dicts here and one
    arm; this run keeps four, so the same layout would be several GB.
    """
    nodes, edges, ids = PRE_STATE["nodes"], PRE_STATE["final_edges"], PRE_STATE["ids"]
    arr = np.array([[nodes[g]["t"], *nodes[g]["xyz"]] for g in ids], np.float32)
    rm = {g: i for i, g in enumerate(ids)}
    e = np.array([[rm[a_], rm[b_]] for a_, b_ in edges], np.int32) if edges \
        else np.zeros((0, 2), np.int32)
    out = defaultdict(int)
    for a_, _b in edges:
        out[a_] += 1
    key = {(a_, b_): i for i, (a_, b_) in enumerate(edges) if out[a_] > 1}
    return dict(arr=arr, e=e, key=key)


def _rescore(name, snap, drop_rows, tag):
    """Write the final graph minus those edge ROWS, score it."""
    if drop_rows:
        keep = np.ones(len(snap["e"]), bool)
        keep[np.fromiter(drop_rows, int, len(drop_rows))] = False
        e3 = snap["e"][keep]
    else:
        e3 = snap["e"]
    outp = pathlib.Path(f"{TMP}/prune_{tag}") / f"{name}.geff"
    outp.parent.mkdir(parents=True, exist_ok=True)
    dr._save(str(outp), np.asarray(snap["arr"], float), np.asarray(e3, int))
    return score_one(outp, name)


def _judge(name, k, outp, seeds, aux):
    """official scorer + score_divisions + the fork table for one graph."""
    row = score_one(outp, name)
    dset = open_dataset(root / "train" / name, require_tracks=True, load_image=False)
    res_ = td.graph.IndexedRXGraph.from_geff(str(outp))
    g = res_[0] if isinstance(res_, tuple) else res_
    ds = score_divisions(g, dset.tracks, scale=dset.scale)
    fr = fork_rows(name, k, seeds, aux, ds.tp_forks, ds.fp_forks)
    del g, res_, dset
    return row, ds, fr



# ---- forkprop_z: the division-variable ILP (hypz.py's machinery, ported verbatim) --------------------------------
from pyscipopt import Model, quicksum            # noqa: E402  (support-pack wheel)
DUP_STAT = defaultdict(int)
HYP_SEL = []
Z_SEL = []            # per solve: (t, p, a, b, w, s, label) of the selected z triples
Z_STAT = {a_: defaultdict(int) for a_ in ('Z', 'Z50', 'Z65')}
Z_S0 = -9.0                    # logit of scorer B at which a triple costs exactly DIV_W (hypz's Z10)
Z_CFG = dict(cost=0.35, lam=0.10, wmin=0.10)
HYP_ARMS = [('Z10', dict(lam=0.10, wmin=0.10))]   # load_triples prints the discount counts per entry
def z_cost(s, cfg):
    """the division cost of one triple under an arm: DIV_W minus lam per logit above Z_S0, clipped to [wmin, wmax]"""
    w = DIV_W - cfg["lam"] * (float(s) - Z_S0)
    return float(min(cfg.get("wmax", cfg.get("vcost", DIV_W)), max(cfg["wmin"], w)))


# ---- scorer B, refitted out of fold from divtrip's candidate table -----------------------------------------
import pandas as _pd                                            # noqa: E402
TRIP_GEO = ["d_pa", "d_pb", "d_ab", "asym", "mid_off", "ang", "dz_pa", "dz_pb", "dz_ab", "t_frac", "has_m", "speed_m",
            "cont_a", "cont_b", "cont_min", "cont_mid", "has_a2", "has_b2", "sep_grow", "sc_p", "sc_a", "sc_b", "sc_min",
            "dens_p10", "n_t1_12", "orph_a", "orph_b", "orph_min", "orph_max"]
TRIP_IMG = ["i_p", "c_p", "r_p", "hm_p", "c_a", "c_b", "hm_a", "hm_b", "hm_max_ratio", "hm_sum_ratio", "i_ratio", "br_p"]


def _trip_hgb():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, min_samples_leaf=20, early_stopping=False,
                                          random_state=0)


def _trip_w(y):
    n1 = max(1, int(np.sum(y == 1)))
    return np.where(y == 1, (len(y) - n1) / n1, 1.0)


def load_triples():
    """{movie: [(t, p, a, b, s, label)]} from divtrip_candidates.csv, s = logit of scorer B out of fold by the table's folds"""
    hit = _walk_files(lambda fn: fn if fn == "divtrip_candidates.csv" else None)
    assert hit, "attach the kernel faustine649/biohub-divtrip (divtrip_candidates.csv)"
    tb = _pd.read_csv(hit["divtrip_candidates.csv"])
    for c_ in ["movie", "fold", "t", "p", "a", "b", "label"] + TRIP_GEO + TRIP_IMG:
        assert c_ in tb.columns, c_
    X = tb[TRIP_GEO + TRIP_IMG].to_numpy(float)
    y = tb["label"].to_numpy(int)
    f = tb["fold"].to_numpy(int)
    p = np.full(len(y), np.nan)
    per = []
    for k_ in sorted(set(f.tolist())):
        tr, te = f != k_, f == k_
        m_ = _trip_hgb().fit(X[tr], y[tr], sample_weight=_trip_w(y[tr]))
        p[te] = m_.predict_proba(X[te])[:, 1]
        if len(set(y[te].tolist())) == 2:
            per.append((k_, int(y[te].sum()), roc_auc_score(y[te], p[te])))
    assert np.isfinite(p).all()
    s = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    print(f"  scorer B refit on {len(tb):,} pooled triples ({int(y.sum())} positives, {tb.movie.nunique()} movies): "
          f"per-fold AUC " + ", ".join(f"fold {k_} ({n_} pos) {a_:.3f}" for k_, n_, a_ in per)
          + f"; pooled {roc_auc_score(y, p):.4f}; logit p50 / p90 / p99 {np.quantile(s, .5):.2f} / {np.quantile(s, .9):.2f} / "
          f"{np.quantile(s, .99):.2f}, positives p25 / p50 / p75 {np.quantile(s[y == 1], .25):.2f} / "
          f"{np.quantile(s[y == 1], .5):.2f} / {np.quantile(s[y == 1], .75):.2f}", flush=True)
    for a_, cfg_ in HYP_ARMS:
        if cfg_.get("lam") is None:
            continue
        w_ = np.array([z_cost(v_, cfg_) for v_ in s])
        print(f"    {a_}: cost below DIV_W for {int((w_ < DIV_W - 1e-9).sum()):,} triples ({int((w_ < DIV_W - 1e-9)[y == 1].sum())} "
              f"positives); at wmin {int((w_ <= cfg_['wmin'] + 1e-9).sum()):,} ({int((w_ <= cfg_['wmin'] + 1e-9)[y == 1].sum())} positives)",
              flush=True)
    out = defaultdict(list)
    for mv, t_, p_, a_, b_, s_, l_ in zip(tb["movie"].tolist(), tb["t"].tolist(), tb["p"].tolist(), tb["a"].tolist(),
                                          tb["b"].tolist(), s.tolist(), y.tolist()):
        out[mv].append((int(t_), int(p_), int(a_), int(b_), float(s_), int(l_)))
    del tb, X
    return out


TRIP_OF = load_triples()
print(f"  triples loaded for {len(TRIP_OF)} movies; {sum(1 for n_ in SIX if n_ in TRIP_OF)} of the {len(SIX)} 6bba "
      f"movies covered", flush=True)
assert sum(1 for n_ in SIX if n_ in TRIP_OF) >= len(SIX) - 2, 'the triple table does not cover the 6bba movies'


def _hyp_exempt(edges0, seed_of_node, final_edges):
    """hypx: REF's fork daughters as (t, seed) pairs. ex_d = the two children of every fork of the ILP solution
    (edges0) and of the patched graph (final_edges: the orphan patch's forks included); ex_t = ex_d plus each
    child's downstream chain in the patched graph up to the next fork or end. Nodes without a seed (gap-close
    insertions) are walked through but not returned. Returns (ex_d, ex_t, n_daughter_nodes, n_track_nodes)."""
    def _succ(edges):
        s = defaultdict(list)
        for a_, b_ in edges:
            s[a_].append(b_)
        return s
    s0, s1 = _succ(edges0), _succ(final_edges)
    d = set()
    for s in (s0, s1):
        for a_, bs in s.items():
            if len(bs) >= 2:
                d.update(bs)
    tn = set(d)
    for g in list(d):
        cur = g
        while len(s1.get(cur, [])) == 1:
            cur = s1[cur][0]
            if cur in tn:
                break
            tn.add(cur)
    ex_d = {seed_of_node[g] for g in d if seed_of_node.get(g) is not None}
    ex_t = {seed_of_node[g] for g in tn if seed_of_node.get(g) is not None}
    return ex_d, ex_t, len(d), len(tn)


def solve_hyp(dets, T, seeds, feats, model, cols, parents, node_cost,
              disapp=None, floor=None, divw=None, exempt=None, trip=None):
    """The shipped ILP re-stated in pyscipopt, plus a cost per selected node and,
    when `parents` is given (hypcache's p{t}), the exclusivity constraint
    child + every parent holding it <= 1. parents None + node_cost 0 = REF.

    Returns what solve_one returns: nodes0, edges0, (cand, cand_sc, cand_trees),
    (pmap, seed_of_node). Every child detection is a node of the output, selected
    or not, EXCEPT the children of a selected parent; unselected parents are not
    output. HYP_SEL and DUP_STAT carry the bookkeeping of this solve."""
    t_solve = time.time()
    fl = PROD_FLOOR if floor is None else floor
    dw = DIV_W if divw is None else divw
    dis = ILP_DISAPP if disapp is None else disapp
    probs = [model.predict_proba(rows[:, cols])[:, 1] if len(pairs) else np.zeros(0)
             for pairs, rows, P, C in feats]
    offset, coords_rows, is_par, par_children = {}, [], [], {}
    n_glob = 0
    for t in range(T):
        c, _s = seeds[t]
        offset[t] = n_glob
        n = len(c)
        npar = 0 if parents is None else len(parents[t])
        nch = n - npar
        assert nch >= 0, (t, n, npar)
        for q, xyz in enumerate(c):
            coords_rows.append([t, xyz[0], xyz[1], xyz[2]])
            is_par.append(q >= nch)
        if npar:
            pt = np.asarray(parents[t], int).reshape(-1, 2)
            assert pt.size == 0 or (pt.min() >= 0 and pt.max() < nch), (
                f"t={t}: a parent's child index is outside the children ({nch} children)")
            for q in range(npar):
                par_children[offset[t] + nch + q] = (offset[t] + int(pt[q, 0]),
                                                     offset[t] + int(pt[q, 1]))
        n_glob += n
    coords = np.asarray(np.round(coords_rows), np.int16) if coords_rows else np.zeros((0, 4), np.int16)
    N = n_glob
    cand = []
    for t in range(T - 1):
        pairs, rows, P, C = feats[t]
        for (i_, j_), pr in zip(pairs, probs[t]):
            if pr >= fl:
                cand.append((offset[t] + int(i_), offset[t + 1] + int(j_), float(pr)))
    # ---- hypz: one z per scored triple; its two edges as candidates (division-only when below the floor)
    zt = []                                  # (gp, ga, gb, w, s, label, k_a, k_b)
    z_of = defaultdict(list)                 # mother global id -> z indices
    div_only = {}                            # cand index -> z indices allowed to use it
    Z_SEL.clear()
    if trip:
        kidx = {(gs, gt_): k for k, (gs, gt_, _p) in enumerate(cand)}
        pm_t = {}
        for t in range(T - 1):
            pairs, rows, P, C = feats[t]
            for (i_, j_), pr in zip(pairs, probs[t]):
                pm_t[(t, int(i_), int(j_))] = float(pr)
        for (t_, ip, ia, ib, w_, s_, l_) in trip:
            DUP_STAT['z_offered'] += 1
            if t_ >= T - 1 or ip >= len(seeds[t_][0]) or max(ia, ib) >= len(seeds[t_ + 1][0]) or ia == ib:
                DUP_STAT['z_bad_index'] += 1
                continue
            pa, pb = pm_t.get((t_, ip, ia)), pm_t.get((t_, ip, ib))
            if pa is None or pb is None:
                DUP_STAT['z_no_pair'] += 1
                DUP_STAT['z_no_pair_tp'] += int(l_)
                continue
            gp, ga, gb = offset[t_] + ip, offset[t_ + 1] + ia, offset[t_ + 1] + ib
            ks = []
            for gc_, pc_ in ((ga, pa), (gb, pb)):
                k_ = kidx.get((gp, gc_))
                if k_ is None:
                    k_ = len(cand)
                    cand.append((gp, gc_, pc_))
                    kidx[(gp, gc_)] = k_
                    div_only[k_] = []
                    DUP_STAT['z_edges_added'] += 1
                ks.append(k_)
            j_ = len(zt)
            for k_ in ks:
                if k_ in div_only:
                    div_only[k_].append(j_)
            zt.append((gp, ga, gb, float(w_), float(s_), int(l_), ks[0], ks[1]))
            z_of[gp].append(j_)
        DUP_STAT['z_vars'] += len(zt)
        DUP_STAT['z_vars_tp'] += sum(1 for r_ in zt if r_[5] == 1)
    m = Model("hyp")
    m.hideOutput()
    x = [m.addVar(vtype="B", name=f"x{g}") for g in range(N)]
    a = [m.addVar(vtype="B", name=f"a{g}") for g in range(N)]
    d = [m.addVar(vtype="B", name=f"d{g}") for g in range(N)]
    v = [m.addVar(vtype="B", name=f"v{g}") for g in range(N)]
    y = [m.addVar(vtype="B", name=f"y{k}") for k in range(len(cand))]
    z = [m.addVar(vtype="B", name=f"z{j}") for j in range(len(zt))]
    in_e, out_e = defaultdict(list), defaultdict(list)
    for k, (gs, gt_, _p) in enumerate(cand):
        out_e[gs].append(k)
        in_e[gt_].append(k)
    for g in range(N):
        m.addCons(quicksum(y[k] for k in in_e[g]) + a[g] == x[g])
        m.addCons(quicksum(y[k] for k in out_e[g]) + d[g] == x[g] + v[g] + quicksum(z[j] for j in z_of.get(g, ())))
        m.addCons(v[g] + quicksum(z[j] for j in z_of.get(g, ())) <= x[g])
    for j, (_gp, _ga, _gb, _w, _s, _l, ka, kb) in enumerate(zt):
        m.addCons(z[j] <= y[ka])
        m.addCons(z[j] <= y[kb])
    for k, js in div_only.items():
        m.addCons(y[k] <= quicksum(z[j] for j in js))
    holders = defaultdict(list)
    for gp, (ga, gb) in par_children.items():
        holders[ga].append(gp)
        holders[gb].append(gp)
    for gch, gps in holders.items():
        m.addCons(x[gch] + quicksum(x[gp] for gp in gps) <= 1)
    obj = quicksum(-p * y[k] for k, (_gs, _gt, p) in enumerate(cand)) \
        + dis * quicksum(d) + dw * quicksum(v) \
        + quicksum(zt[j][3] * z[j] for j in range(len(zt)))
    ex = set()
    if exempt:
        for t_, q_ in exempt:
            t_, q_ = int(t_), int(q_)
            if t_ in offset and 0 <= q_ < len(seeds[t_][0]):
                ex.add(offset[t_] + q_)
    DUP_STAT["exempt"] += len(ex)
    if node_cost:
        obj = obj + node_cost * quicksum(x[g] for g in range(N) if g not in ex)
    m.setObjective(obj, "minimize")
    m.optimize()
    status = m.getStatus()
    assert status == "optimal", f"SCIP status {status}"
    xs = np.array([m.getVal(x[g]) > 0.5 for g in range(N)], bool) if N else np.zeros(0, bool)
    ys = [k for k in range(len(cand)) if m.getVal(y[k]) > 0.5]
    n_div = sum(1 for g in range(N) if m.getVal(v[g]) > 0.5)
    for j, (gp, ga, gb, w_, s_, l_, _ka, _kb) in enumerate(zt):
        if m.getVal(z[j]) > 0.5:
            t_ = int(coords[gp, 0])
            Z_SEL.append((t_, int(gp - offset[t_]), int(ga - offset[t_ + 1]), int(gb - offset[t_ + 1]), w_, s_, l_))
    DUP_STAT['z_selected'] += len(Z_SEL)
    DUP_STAT['z_sel_tp'] += sum(1 for r_ in Z_SEL if r_[6] == 1)
    n_div += len(Z_SEL)
    del m, x, a, d, v, y, z
    DUP_STAT["nodes_offered"] += N
    DUP_STAT["nodes_selected"] += int(xs.sum())
    DUP_STAT["edges_selected"] += len(ys)
    DUP_STAT["ilp_divisions"] += n_div
    DUP_STAT["solve_s"] += int(time.time() - t_solve)
    # ---- output: children (all), selected parents, minus the children of a selected parent
    HYP_SEL.clear()
    drop = set()
    for gp, (ga, gb) in par_children.items():
        DUP_STAT["hyp_par_off"] += 1
        if xs[gp]:
            drop.update((ga, gb))
            DUP_STAT["hyp_par_sel"] += 1
            HYP_SEL.append((int(coords[gp, 0]), coords[gp, 1:].astype(float),
                            coords[ga, 1:].astype(float), coords[gb, 1:].astype(float)))
    nodes0, gid_of, seed_of_node = {}, {}, {}
    nid = 1
    for g in range(N):
        if is_par[g] and not xs[g]:
            continue
        if g in drop:
            DUP_STAT["children_dropped"] += 1
            continue
        t_ = int(coords[g, 0])
        nodes0[nid] = {"t": t_, "xyz": coords[g, 1:].astype(float)}
        gid_of[g] = nid
        seed_of_node[nid] = (t_, int(g - offset[t_]))
        nid += 1
    edges0 = []
    for k in ys:
        gs, gt_, _p = cand[k]
        assert gs in gid_of and gt_ in gid_of, "a selected edge touches a dropped node"
        edges0.append((gid_of[gs], gid_of[gt_]))
    cand_p, cand_sc, cand_trees = [], [], []
    for t in range(T):
        c_all = np.asarray(dets[t][0], float)
        s_all = np.asarray(dets[t][1], float)
        cm = (s_all >= CAND_THR) & (s_all < UNET_THRESH)
        cand_p.append(c_all[cm])
        cand_sc.append(s_all[cm])
        cand_trees.append(cKDTree(cand_p[-1] * SCALE) if len(cand_p[-1]) else None)
    pmap = {}
    for t in range(T - 1):
        pairs, rows, P, C = feats[t]
        for (i_, j_), pr, row in zip(pairs, probs[t], rows):
            pmap[(t, int(i_), int(j_))] = (float(pr), float(row[C_PFWD]), float(row[C_PREV]))
    del probs
    return nodes0, edges0, (cand_p, cand_sc, cand_trees), (pmap, seed_of_node)


def z_solves(name, dets, T, seeds, feats, model, edges0, seed_of_node, final_edges):
    """the two z compositions of one movie: Z (node cost 0.35, REF's exemption, Z10's triples) and ZX (a first solve
    at cost 0, its z-fork daughters + chains added to REF's exemption, then the costed solve). Returns {tag: (nodes,
    edges, pool, pmap, seed_of_node)} in solve_one's shapes; Z_STAT[tag] carries the counts."""
    ex_t = _hyp_exempt(edges0, seed_of_node, final_edges)[1]
    trip = [(t_, p_, a_, b_, z_cost(s_, Z_CFG), s_, l_) for (t_, p_, a_, b_, s_, l_) in TRIP_OF.get(name, [])]
    out = {}
    assert Z_CFG['cost'] == NC_ARMS[0][1], (Z_CFG, NC_ARMS)
    for tag, cost_ in NC_ARMS:
        DUP_STAT.clear()
        _ts = time.time()
        n_, e_, p_, aux_ = solve_hyp(dets, T, seeds, feats, model, COLS, None, cost_, disapp=ILP_DISAPP,
                                     floor=PROD_FLOOR, exempt=ex_t, trip=trip)
        Z_STAT[tag]['wall_s'] += int(time.time() - _ts)
        for k_, v_ in DUP_STAT.items():
            Z_STAT[tag][k_] += v_
        Z_STAT[tag]['movies'] += 1
        out[tag] = (n_, e_, p_, aux_[0], aux_[1])
    return out


for name in SIX:
    if time.time() - t_all > BUDGET_S:
        INCOMPLETE.append(name)
        continue
    t0 = time.time()
    dets = load_dets(name, "F4")
    T = len(dets)
    seeds, feats = FEATS.pop(name)
    k = FOLD_OF[name]
    nodes0, edges0, pool, aux0 = solve_one(dets, T, seeds, feats, MODELS_FOLD[k], COLS,
                                           disapp=ILP_DISAPP, floor=PROD_FLOOR)
    pmap, seed_of_node = aux0
    scsort, best2 = _aux_extra(seeds, pmap)
    aux = (pmap, seed_of_node, scsort, best2)

    def pfun(a, b, _pm=pmap, _so=seed_of_node):
        sa, sb = _so.get(a), _so.get(b)
        if sa is None or sb is None or sb[0] != sa[0] + 1:
            return float("nan")
        return _pm.get((sa[0], sa[1], sb[1]), (float("nan"),))[0]

    outp = pathlib.Path(f"{TMP}/ref") / f"{name}.geff"
    outp.parent.mkdir(parents=True, exist_ok=True)
    PATCH_CAPS.clear()
    # bridgerank pass 1: GT for the bridge labels, then collect inside the
    # shipped patch (GAP_SPAN is 1 here, so nothing longer is bridged and the
    # REF graph is round 2's).
    _gt = read_geff(geff_path(root, "train", name))
    _g_t = np.asarray(_gt["t"], int)
    _g_pos = np.stack([np.asarray(_gt[c] , float) for c in ("z", "y", "x")], axis=1) * SCALE
    _row_of = {int(v): i for i, v in enumerate(_gt["node_ids"])}
    _kids = defaultdict(list)
    for _a, _b in _gt["edges"]:
        _ra, _rb = _row_of.get(int(_a)), _row_of.get(int(_b))
        if _ra is not None and _rb is not None:
            _kids[_ra].append(_rb)
    _trees = {}
    for _tt in np.unique(_g_t):
        _rows = np.where(_g_t == _tt)[0]
        _trees[int(_tt)] = (_rows, cKDTree(_g_pos[_rows]))
    BR_CTX.clear()
    BR_CTX.update(name=name, fold=k, seeds=seeds, seed_of_node=seed_of_node,
                  gt=(_trees, _kids))
    BR_STAT.clear()
    assert GAP_SPAN == 1 and BR_MODEL is None
    patch_final(nodes0, edges0, pool, outp)
    _brs = dict(BR_STAT)
    BR_CTX["gt"] = None
    del _gt, _g_t, _g_pos, _row_of, _kids, _trees
    for k_, v_ in PATCH_CAPS.items():
        CAPS_BY_ARM.setdefault("REF", defaultdict(int))[k_] += v_
    _tfi = time.time()
    try:
        IMG_CTX.clear()
        IMG_CTX.update(name=name, img=_open_image(pathlib.Path(root) / "train" / f"{name}.zarr"))
    except Exception as _e:                                       # noqa: BLE001
        warn_once("forkimg_open", f"{name}: image not opened ({_e}); image features NaN")
        IMG_CTX.clear()
    row, ds, fr = _judge(name, k, outp, seeds, aux)
    IMG_CTX.clear()
    FI_STAT["seconds"] += int(time.time() - _tfi)
    if (len(ds.tp_forks), len(ds.fp_forks)) != (row["division_tp"], row["division_fp"]):
        warn_once("fork_count_mismatch",
                  f"{name}: score_divisions tp/fp {len(ds.tp_forks)}/{len(ds.fp_forks)} "
                  f"vs per-sample {row['division_tp']}/{row['division_fp']}")
    ROWS.append(row)
    PER[name] = row
    FORKS.extend(fr)
    MOVIE[name] = _snapshot()
    scores = {int(kk): int(vv) for kk, vv in ds.scores.items()}
    n5a_before = len(STEAL5A)
    w, det = walk_lost(name, scores, pfun=pfun, steal_rows=STEAL5A)
    for c_, n_ in w.items():
        WALK[c_] += n_
        WALK_FOLD[k][c_] += n_
    del ds

    cells = [f"br pairs s2 {_brs.get('collect_span2_pairs', 0)} "
             f"(lab {_brs.get('collect_span2_labelled', 0)}, pos {_brs.get('collect_span2_pos', 0)}) "
             f"s3 {_brs.get('collect_span3_pairs', 0)} "
             f"(lab {_brs.get('collect_span3_labelled', 0)}, pos {_brs.get('collect_span3_pos', 0)})"]
    # the arms are replayed in pass 2 on this exact solve
    with open(BR_DIR / f"{name}.pkl", "wb") as _fh:
        pickle.dump((nodes0, edges0, pool[0], pool[1], seeds, pmap, seed_of_node), _fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    _tz = time.time()
    _zs = z_solves(name, dets, T, seeds, feats, MODELS_FOLD[k], edges0, seed_of_node, PRE_STATE["final_edges"])
    for _tag, (_nz, _ez, _pz, _pmz, _soz) in _zs.items():
        with open(BR_DIR / f"{name}_{_tag}.pkl", "wb") as _fh:
            pickle.dump((_nz, _ez, _pz[0], _pz[1], seeds, _pmz, _soz), _fh, protocol=pickle.HIGHEST_PROTOCOL)
    cells.append(f"z solves {time.time() - _tz:.0f}s: " + " ".join(f"{_t} nodes {len(_zs[_t][0])} edges {len(_zs[_t][1])}" for _t, _c in NC_ARMS)
                 + f" (REF nodes {len(nodes0)} edges {len(edges0)})")
    del _zs

    extra = ""
    if name in HOLDOUT_6BBA:
        n_a, e_a, p_a, _ = solve_one(dets, T, seeds, feats, MODEL_D187, MCOLS_D187,
                                     disapp=1.5, floor=0.10)
        outa = pathlib.Path(f"{TMP}/anchor_d187") / f"{name}.geff"
        outa.parent.mkdir(parents=True, exist_ok=True)
        patch_final(n_a, e_a, p_a, outa)
        ANCHOR_ROWS.append(score_one(outa, name))
        n_s, e_s, p_s, _ = solve_one(dets, T, seeds, feats, MODEL_BIDIR, MCOLS_BIDIR,
                                     disapp=ILP_DISAPP, floor=PROD_FLOOR)
        outs = pathlib.Path(f"{TMP}/soft_bidir") / f"{name}.geff"
        outs.parent.mkdir(parents=True, exist_ok=True)
        patch_final(n_s, e_s, p_s, outs)
        SOFT_ROWS.append(score_one(outs, name))
        extra = f"  anchor {_adj_of(ANCHOR_ROWS[-1]):.4f} soft {_adj_of(SOFT_ROWS[-1]):.4f}"
        del n_a, e_a, p_a, n_s, e_s, p_s
    n_patch = sum(1 for r_ in fr if r_[5][0] == 1)
    print(f"  {name} fold{k} gt_div={row['division_tp'] + row['division_fn']:>2} "
          f"div {row['division_tp']}/{row['division_fp']}/{row['division_fn']:<3} "
          f"forks {len(fr):>4} ({n_patch} patch) 5a={len(STEAL5A) - n5a_before} "
          f"lost: {', '.join(f'{c_} {n_}' for c_, n_ in sorted(w.items()) if c_ != 'recovered') or '-'}"
          f"   {_adj_of(row):.4f} | " + " | ".join(cells)
          + f"{extra}  {time.time() - t0:.0f}s", flush=True)
    del dets, seeds, feats, nodes0, edges0, pool, aux, aux0, pmap, seed_of_node, best2
    gc.collect()
print(f"  stage B: {len(SIX) - len(INCOMPLETE)} of {len(SIX)} movies "
      f"({(time.time() - tb) / 60:.1f} min)", flush=True)
if INCOMPLETE:
    print(f"INCOMPLETE: {len(INCOMPLETE)} movies not attempted: {INCOMPLETE}", flush=True)

BR_FOLD_MODEL = {}
BR_MODEL_OK = False

# ---- BRIDGERANK PASS 2b: the four arms, replayed on the pickled solves -----
BR_ARMS_RUN = list(BR_ARMS)
DIVW_ARMS = [(a_, DIV_W) for a_, _c in BR_ARMS_RUN]
BR_STAT_ARM = {a_: defaultdict(int) for a_, _c in BR_ARMS_RUN}
print(f"\n=== PASS 2: {len(BR_ARMS_RUN)} post-link arms replayed on the pickled solve of "
      f"each movie: " + ", ".join(f"{a_} (rank {c_['rank']}, caps x{c_['cap']:g}, DIV_DIVERGE_UM {c_['diverge']})"
                                  for a_, c_ in BR_ARMS_RUN) + " ===", flush=True)
tb2 = time.time()
INCOMPLETE2 = []
for name in [n for n in SIX if n in PER]:
    if time.time() - t_all > BUDGET_S:
        INCOMPLETE2.append(name)
        continue
    t0 = time.time()
    SOLVES = {}
    for _src in ("REF", "Z", "Z50", "Z65"):
        with open(BR_DIR / (f"{name}.pkl" if _src == "REF" else f"{name}_{_src}.pkl"), "rb") as _fh:
            _n0, _e0, _c, _cs, seeds, _pm, _so = pickle.load(_fh)
        _ct = [cKDTree(c_ * SCALE) if len(c_) else None for c_ in _c]
        _ss, _b2 = _aux_extra(seeds, _pm)
        SOLVES[_src] = (_n0, _e0, (_c, _cs, _ct), _pm, _so, (_pm, _so, _ss, _b2))
    nodes0, edges0, pool, pmap, seed_of_node, aux = SOLVES["REF"]
    cand_, cand_sc_, cand_trees_ = pool
    k = FOLD_OF[name]
    BR_CTX.clear()
    BR_CTX.update(name=name, fold=k, seeds=seeds, seed_of_node=seed_of_node, gt=None)
    try:
        IMG_CTX.clear()
        IMG_CTX.update(name=name, img=_open_image(pathlib.Path(root) / "train" / f"{name}.zarr"))
    except Exception as _e:                                       # noqa: BLE001
        warn_once("forkimg_open2", f"{name}: image not opened in pass 2 ({_e}); contrast -1, image features NaN")
        IMG_CTX.clear()
    FP_MEMO.clear()
    cells = []
    for arm, cfg in BR_ARMS_RUN:
        DIV_DIVERGE_UM = cfg["diverge"]
        DA_ADMIT = cfg["admit"]
        FP_RANK = cfg["rank"]
        FP_CAP = cfg["cap"]
        nodes0, edges0, pool, pmap, seed_of_node, aux = SOLVES[cfg.get("src", "REF")]
        DA_CTX.clear()
        DA_CTX.update(seeds=seeds, seed_of_node=seed_of_node, added=[])
        BR_STAT.clear()
        outa = pathlib.Path(f"{TMP}/arm_{arm}") / f"{name}.geff"
        outa.parent.mkdir(parents=True, exist_ok=True)
        PATCH_CAPS.clear()
        patch_final(nodes0, edges0, pool, outa)
        for k_, v_ in PATCH_CAPS.items():
            CAPS_BY_ARM.setdefault(arm, defaultdict(int))[k_] += v_
        for k_, v_ in BR_STAT.items():
            BR_STAT_ARM[arm][k_] += v_
        row_a, ds_a, fr_a = _judge(name, k, outa, seeds, aux)
        ARM_ROWS[arm].append(row_a)
        ARM_PER[arm][name] = row_a
        ARM_FORKS[arm].extend(fr_a)
        ARM_MOVIE[arm][name] = _snapshot()
        for _g in DA_CTX.get("added", []):
            seed_of_node.pop(_g, None)
        cells.append(f"{arm} adm {PATCH_CAPS.get('admit_added', 0)} orph {PATCH_CAPS.get('orphan_added', 0)} "
                     f"{row_a['division_tp']}/{row_a['division_fp']}/{row_a['division_fn']} "
                     f"{_adj_of(row_a):.4f}")
        del ds_a
    DIV_DIVERGE_UM = 2.25
    DA_ADMIT = False
    FP_RANK = "dist"
    FP_CAP = 1.0
    DA_CTX.clear()
    IMG_CTX.clear()
    print(f"  {name} fold{k} REF {_adj_of(PER[name]):.4f} | " + " | ".join(cells)
          + f"  {time.time() - t0:.0f}s", flush=True)
    del nodes0, edges0, pool, cand_, cand_sc_, cand_trees_, seeds, pmap, seed_of_node, aux, SOLVES
    gc.collect()
print(f"  pass 2: {len(SIX) - len(INCOMPLETE) - len(INCOMPLETE2)} movies "
      f"({(time.time() - tb2) / 60:.1f} min)", flush=True)
if INCOMPLETE2:
    print(f"INCOMPLETE2: {len(INCOMPLETE2)} movies not replayed: {INCOMPLETE2}", flush=True)
    # the cross indexes every arm by SIX_OK; drop the movies no arm has
    for _n in INCOMPLETE2:
        PER.pop(_n, None)
    ROWS = [PER[n] for n in SIX if n in PER]
print("\n=== THE z SOLVES, summed over the movies (the same triples and exemption; the node cost is the only difference). "
      "TP by divtrip's 5 um label ===", flush=True)
for _tag, _cost in NC_ARMS:
    d_ = Z_STAT[_tag]
    print(f"  {_tag:<4} node cost {_cost:.2f} | movies {d_['movies']} | z variables {d_['z_vars']:,} (TP {d_['z_vars_tp']}), z selected "
          f"{d_['z_selected']:,} (TP {d_['z_sel_tp']}) | nodes selected {d_['nodes_selected']:,} of {d_['nodes_offered']:,} "
          f"({d_['nodes_selected'] / max(d_['nodes_offered'], 1) * 100:.2f}%), edges {d_['edges_selected']:,}, ILP divisions "
          f"{d_['ilp_divisions']:,}, exempt {d_['exempt']:,}, solve {d_['solve_s']:,}s (wall {d_['wall_s']:,}s)", flush=True)
print("\n=== THE PATCH PER ARM (orphan pass + admit pass), summed over the movies replayed ===", flush=True)
_bk = ["orphan_added", "orphan_frame", "orphan_global", "admit_candidates", "admit_proposals",
       "admit_dup", "admit_added", "admit_frame", "admit_global"]
print(f"  {'arm':<10}" + "".join(f"{k_:>17}" for k_ in _bk), flush=True)
for a_ in ["REF"] + [x[0] for x in BR_ARMS_RUN]:
    d_ = CAPS_BY_ARM.get(a_, {})
    print(f"  {a_:<10}" + "".join(f"{d_.get(k_, 0):>17,}" for k_ in _bk), flush=True)

# ---- report ----------------------------------------------------------------
print("\n=== ANCHORS / official scorer ===", flush=True)
ANCHOR_FAIL = []
if len(ANCHOR_ROWS) == len(HOLDOUT_6BBA):
    ra = _with_ratio(ANCHOR_ROWS)
    got = float(ra["adj_edge_jaccard"])
    ok = abs(got - 0.9216) <= ANCHOR_TOL
    print(f"  ANCHOR_D187  F4-6bba(12)  adj_J {got:.4f}  div_J {ra['division_jaccard']:.4f}"
          f"  want 0.9216 HARD (div_J 0.0606) -> {'OK' if ok else 'MISS'} ({got - 0.9216:+.4f})",
          flush=True)
    if not ok:
        ANCHOR_FAIL.append(("ANCHOR_D187", "F4-6bba", got, 0.9216))
    rs = _with_ratio(SOFT_ROWS)
    print(f"  +BIDIR pkl   F4-6bba(12)  adj_J {rs['adj_edge_jaccard']:.4f}  div_J "
          f"{rs['division_jaccard']:.4f}  want 0.9261 soft (ilpsweep F48) -> "
          f"{'OK' if abs(rs['adj_edge_jaccard'] - 0.9261) <= ANCHOR_TOL else 'MISS'}",
          flush=True)
else:
    ANCHOR_FAIL.append(("ANCHOR_D187", "F4-6bba", float("nan"), 0.9216))

SIX_OK = [n for n in SIX if n in PER]
RES = _with_ratio(ROWS)
TP = sum(r["division_tp"] for r in ROWS)
FP = sum(r["division_fp"] for r in ROWS)
FN = sum(r["division_fn"] for r in ROWS)
N_GT = TP + FN
base_J = TP / max(1, TP + FP + FN)
ref_ok = (TP, FP, FN) == (50, 249, 75)
print(f"  REF   F4-6bba(128) div {TP}/{FP}/{FN} div_J {base_J:.4f} adj_J "
      f"{RES['adj_edge_jaccard']:.4f} score {RES['score']:.4f}  want 50/249/75 "
      f"0.1337 0.9133 0.9266 (divbench3) -> {'OK' if ref_ok else 'MISS'}", flush=True)
if not ref_ok:
    ANCHOR_FAIL.append(("REF_FORKS", "F4-6bba", TP * 1000 + FP, 50 * 1000 + 249))
print("  hard anchors reproduced: the harness is the recorded one." if not ANCHOR_FAIL
      else "  HARD ANCHOR MISS", flush=True)
for k in (0, 1):
    rows_k = [PER[n] for n in SIX if FOLD_OF[n] == k and n in PER]
    rk = _with_ratio(rows_k)
    print(f"  fold{k}: div {sum(r['division_tp'] for r in rows_k)}/"
          f"{sum(r['division_fp'] for r in rows_k)}/{sum(r['division_fn'] for r in rows_k)} "
          f"div_J {rk['division_jaccard']:.4f} adj_J {rk['adj_edge_jaccard']:.4f} "
          f"score {rk['score']:.4f}", flush=True)

print("\n=== WHERE THE LOST DIVISIONS GO (F4; divbench3 in brackets) ===", flush=True)
BASE_WALK = {"mother_absent": 9, "daughter_absent": 22, "5a_sister_not_orphan": 23,
             "8b_diverge_under_2.25um": 11,
             "9_passed_all_gates_then_capped_or_taken": 3, "recovered": 50}
print(f"  {'condition':<42} {'all':>5} {'fold0':>6} {'fold1':>6} {'db3':>6}", flush=True)
for c_ in WALK_ORDER:
    if WALK.get(c_, 0) or BASE_WALK.get(c_, 0):
        print(f"  {c_:<42} {WALK.get(c_, 0):>5} {WALK_FOLD[0].get(c_, 0):>6} "
              f"{WALK_FOLD[1].get(c_, 0):>6} {BASE_WALK.get(c_, 0):>6}", flush=True)

print("\n=== THE PATCH'S OWN BUDGET, per arm, summed over the 128 movies "
      "(rule: instrument the stage before sweeping it -- a proposal refused by "
      "a CAP is not a proposal refused on QUALITY) ===", flush=True)
_ck = ["orphan_added", "orphan_global", "orphan_frame", "steal_proposals",
       "steal_added", "steal_no_prob", "steal_margin", "steal_global", "steal_frame"]
print(f"  {'arm':<12}" + "".join(f"{k_:>17}" for k_ in _ck), flush=True)
for a_ in ["REF"] + [x[0] for x in STEAL_ARMS]:
    d_ = CAPS_BY_ARM.get(a_, {})
    print(f"  {a_:<12}" + "".join(f"{d_.get(k_, 0):>17,}" for k_ in _ck), flush=True)

# ---- the 5a cases, in full --------------------------------------------------
print(f"\n=== 5a: EVERY GT DIVISION WHOSE SECOND DAUGHTER THE ILP HAD ALREADY GIVEN "
      f"AWAY ({len(STEAL5A)} cases, F4, 128 6bba movies) ===", flush=True)
with open("/kaggle/working/steal5a.csv", "w") as fh:
    keys = ["movie", "gt_mother", "p_pq", "p_rq", "d_pq", "d_rq", "d_s", "dv",
            "true_other", "r_forks", "q_multi_pred"]
    fh.write(",".join(keys) + "\n")
    for r_ in STEAL5A:
        fh.write(",".join(str(r_[k_]) for k_ in keys) + "\n")
if STEAL5A:
    def _q(vals, f):
        v = np.array([x for x in vals if np.isfinite(x)], float)
        return float(np.quantile(v, f)) if len(v) else float("nan")
    pm = [r_["p_pq"] for r_ in STEAL5A]
    po = [r_["p_rq"] for r_ in STEAL5A]
    both = [(a, b) for a, b in zip(pm, po) if np.isfinite(a) and np.isfinite(b)]
    print(f"  {'quantity':<44} {'value':>28}", flush=True)
    print(f"  {'competing edge annotated at both ends':<44} "
          f"{sum(1 for r_ in STEAL5A if r_['true_other'] >= 0):>28}", flush=True)
    print(f"  {'... of those, a TRUE GT edge':<44} "
          f"{sum(1 for r_ in STEAL5A if r_['true_other'] == 1):>28}", flush=True)
    print(f"  {'competitor already forks (steal refuses)':<44} "
          f"{sum(r_['r_forks'] for r_ in STEAL5A):>28}", flush=True)
    print(f"  {'p_mother median (p25 / p75)':<44} "
          f"{_q(pm, .5):.3f} ({_q(pm, .25):.3f} / {_q(pm, .75):.3f})".rjust(28), flush=True)
    print(f"  {'p_other  median (p25 / p75)':<44} "
          f"{_q(po, .5):.3f} ({_q(po, .25):.3f} / {_q(po, .75):.3f})".rjust(28), flush=True)
    print(f"  {'both probabilities available':<44} {len(both):>28}", flush=True)
    for m_ in (0.0, 0.1, 0.2):
        n_ = sum(1 for a, b in both if a - b > m_)
        print(f"  {'p_mother beats p_other by more than ' + str(m_):<44} {n_:>28}", flush=True)
    print(f"  {'d_mother median':<44} {_q([r_['d_pq'] for r_ in STEAL5A], .5):>27.2f}u",
          flush=True)
    print(f"  {'d_other  median':<44} {_q([r_['d_rq'] for r_ in STEAL5A], .5):>27.2f}u",
          flush=True)
    print("  steal5a.csv written", flush=True)

# ---- the forks ---------------------------------------------------------------
X_f = np.array([r_[5] for r_ in FORKS], float) if FORKS else np.zeros((0, len(FEAT_NAMES)))
lab = np.array([r_[4] for r_ in FORKS], int)
fold_f = np.array([r_[1] for r_ in FORKS], int)
src_f = X_f[:, 0].astype(int) if len(X_f) else np.zeros(0, int)
counted = lab >= 1
with open("/kaggle/working/forknc_dw55.csv", "w") as fh:
    fh.write("movie,fold,p,q,label," + ",".join(FEAT_NAMES) + "\n")
    for r_ in FORKS:
        fh.write(f"{r_[0]},{r_[1]},{r_[2]},{r_[3]},{r_[4]}," + ",".join(f"{v:.5g}" for v in r_[5]) + "\n")
for arm, _m in DIVW_ARMS:
    with open(f"/kaggle/working/forknc_{arm}.csv", "w") as fh:
        fh.write("movie,fold,p,q,label," + ",".join(FEAT_NAMES) + "\n")
        for r_ in ARM_FORKS[arm]:
            fh.write(f"{r_[0]},{r_[1]},{r_[2]},{r_[3]},{r_[4]}," + ",".join(f"{v:.5g}" for v in r_[5]) + "\n")
print(f"\n=== EVERY FORK THE 128 REF GRAPHS CARRY: {len(FORKS)} (divbench3: 7,004); "
      f"verdict x source; forks_f4_v4.csv + one csv per steal arm written ===", flush=True)
print(f"  {'source':<8} {'TP':>6} {'FP':>6} {'uncounted':>10} {'precision(counted)':>19}", flush=True)
for s_, nm in ((1, "patch"), (0, "ilp")):
    m_ = src_f == s_
    tp_, fp_, un_ = int((lab[m_] == 2).sum()), int((lab[m_] == 1).sum()), int((lab[m_] == 0).sum())
    print(f"  {nm:<8} {tp_:>6} {fp_:>6} {un_:>10} {tp_ / max(1, tp_ + fp_):>19.3f}", flush=True)

print("\n=== FORK FEATURES by verdict (median TP / FP / uncounted; AUC TP-vs-FP on counted). "
      "The eight new ones are the last eight rows ===", flush=True)
print(f"  {'feature':<9} {'TP med':>8} {'FP med':>8} {'UNC med':>8} {'AUC':>7} {'n_TP':>5} {'n_FP':>5}",
      flush=True)
for j_, fn_ in enumerate(FEAT_NAMES):
    if fn_ == "src":
        continue
    col = X_f[:, j_]
    meds = []
    for L_ in (2, 1, 0):
        v = col[(lab == L_) & np.isfinite(col)]
        meds.append(np.median(v) if len(v) else np.nan)
    mm = counted & np.isfinite(col)
    auc = (_auc((lab[mm] == 2).astype(int), col[mm])
           if mm.sum() and (lab[mm] == 2).any() and (lab[mm] == 1).any() else float("nan"))
    print(f"  {fn_:<9} {meds[0]:>8.3f} {meds[1]:>8.3f} {meds[2]:>8.3f} {auc:>7.3f} "
          f"{int((mm & (lab == 2)).sum()):>5} {int((mm & (lab == 1)).sum()):>5}", flush=True)


def _curve_line(keep, label, ngt=None, lab_=None, fold_=None, wf=None):
    lb = lab if lab_ is None else lab_
    fo = fold_f if fold_ is None else fold_
    ng = N_GT if ngt is None else ngt
    tp_k = int((lb[keep] == 2).sum())
    fp_k = int((lb[keep] == 1).sum())
    fn_k = ng - tp_k
    J = tp_k / max(1, tp_k + fp_k + fn_k)
    per = []
    for k in (0, 1):
        mk = keep & (fo == k)
        tpk = int((lb[mk] == 2).sum())
        fpk = int((lb[mk] == 1).sum())
        ngt_k = sum((wf or WALK_FOLD)[k].values())
        per.append(tpk / max(1, tpk + fpk + (ngt_k - tpk)))
    print(f"  {label:<38} {tp_k:>4} {fp_k:>5} {fn_k:>4} {J:>7.4f} {J - base_J:>+8.4f} "
          f"{0.1 * (J - base_J):>+9.4f} {per[0]:>8.4f} {per[1]:>8.4f}", flush=True)
    return J


HDR = (f"  {'cut':<38} {'TP':>4} {'FP':>5} {'FN':>4} {'div_J':>7} {'d div_J':>8} "
       f"{'~d score':>9} {'fold0 J':>8} {'fold1 J':>8}")


def _gate_oof(X_use, folds, n_splits, lab_, fold_):
    """Out-of-fold gate p over every row. n_splits 2 = divbench3's alternating
    movie split (the anchor); 3 = GroupKFold over movies."""
    p = np.full(len(lab_), np.nan)
    cnt = lab_ >= 1
    aucs = []
    for tr_m, te_m in folds:
        tr = tr_m & cnt
        if tr.sum() < 20 or (lab_[tr] == 2).sum() < 5:
            warn_once("gate_fold_small", f"a gate fold had {int(tr.sum())} counted forks")
            continue
        mg = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05,
                                            min_samples_leaf=10, random_state=0)
        mg.fit(X_use[tr], (lab_[tr] == 2).astype(int))
        p[te_m] = mg.predict_proba(X_use[te_m])[:, 1]
        tc = te_m & cnt
        if tc.sum() and (lab_[tc] == 2).any() and (lab_[tc] == 1).any():
            aucs.append((int(tc.sum()), _auc((lab_[tc] == 2).astype(int), p[tc])))
    pooled = (_auc((lab_[cnt] == 2).astype(int), p[cnt])
              if np.isfinite(p[cnt]).all() else float("nan"))
    return p, aucs, pooled


FOLDS2 = [(fold_f != k, fold_f == k) for k in (0, 1)]
_mv = np.array([r_[0] for r_ in FORKS])
_um = sorted(set(_mv))
_g3 = {m: i % 3 for i, m in enumerate(_um)}
g3 = np.array([_g3[m] for m in _mv])
FOLDS3 = [(g3 != k, g3 == k) for k in (0, 1, 2)]

print("\n=== THE FORK GATE (HistGB 100 it, lr 0.05, msl 10; trained on the OTHER fold's "
      "counted forks, TP=1 FP=0). 20 features / 2 folds is divbench3's ANCHOR "
      "(AUC 0.9283 / 0.8326, p>0.03 -> 42/47/83, div_J 0.2442) ===", flush=True)
J20 = list(range(len(FEAT20)))
JIMG = [FEAT_NAMES.index(f_) for f_ in FEAT_IMG]
print(f"\n  forkprop: image features on {FI_STAT['forks']:,} forks ({FI_STAT['nan_rows']:,} unreadable, {FI_STAT['no_img']:,} without an image), "
      f"proposal contrast crops {FP_STAT['crops']:,} ({FP_STAT['no_img']:,} without an image)", flush=True)
GATE_P = {}
for tag, cols, folds in (("20 feats, 2 folds (ANCHOR)", J20, FOLDS2),
                         ("20 feats, 3 folds", J20, FOLDS3),
                         ("28 feats, 2 folds", list(range(len(FEAT20) + len(FEAT_NEW))), FOLDS2),
                         ("28 feats, 3 folds", list(range(len(FEAT20) + len(FEAT_NEW))), FOLDS3),
                         ("20 feats + 16 image, 2 folds", J20 + JIMG, FOLDS2)):
    p_g, aucs, pooled = _gate_oof(X_f[:, cols], folds, len(folds), lab, fold_f)
    GATE_P[tag] = p_g
    print(f"\n  {tag}: folds " + " ".join(f"{n}:{a:.4f}" for n, a in aucs)
          + f"   pooled out-of-fold AUC {pooled:.4f}", flush=True)
    print(HDR, flush=True)
    ok_g = np.isfinite(p_g)
    _curve_line(ok_g, "none")
    for thr in (0.01, 0.02, 0.03, 0.05, 0.1):
        _curve_line(ok_g & (p_g > thr), f"fixed p > {thr}")
    for frac in (0.7, 0.8, 0.9):
        t_ = np.quantile(p_g[ok_g], frac)
        _curve_line(ok_g & (p_g > t_), f"quantile {int(frac * 100)}% (p > {t_:.4f})")
print(f"\n  oracle (every counted FP fork gone, every TP kept): div_J "
      f"{TP / max(1, TP + FN):.4f} ({0.1 * (TP / max(1, TP + FN) - base_J):+.4f} of score)",
      flush=True)

# the production gate: every counted fork, 20 features (what ships today) and 28
for tag, cols, names_ in (("dw55_20", J20, FEAT20),):
    mg_full = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05,
                                             min_samples_leaf=10, random_state=0)
    mg_full.fit(X_f[counted][:, cols], (lab[counted] == 2).astype(int))
    p_full = mg_full.predict_proba(X_f[:, cols])[:, 1]
    Q_FULL = {f: float(np.quantile(p_full, f)) for f in (0.5, 0.6, 0.7, 0.8, 0.9)}
    with open(f"/kaggle/working/fork_gate_{tag}.pkl", "wb") as fh:
        pickle.dump({"model": mg_full, "feature_names": names_, "version": SCRIPT_VERSION,
                     "trained_on": "counted F4 forks, 128 6bba train movies, TP=1 FP=0",
                     "p_quantiles_all_forks": Q_FULL, "detector": "F4",
                     "ilp": {"disapp": ILP_DISAPP, "floor": PROD_FLOOR}}, fh)
    print(f"  fork_gate_{tag}.pkl written; p quantiles over all {len(FORKS)} forks: "
          + " ".join(f"{int(f * 100)}%={v:.4f}" for f, v in Q_FULL.items()), flush=True)

# ---- the steal arms ---------------------------------------------------------
print(f"\n=== THE DIV_W ARMS, UNGATED (each is its own ILP solve; official scorer, "
      f"all {len(SIX_OK)} movies) ===", flush=True)
print(f"  {'arm':<16} {'forks':>7} {'ilp':>7} {'patch':>7} {'div tp/fp/fn':>14} {'div_J':>7} "
      f"{'adj_J':>7} {'edge_J':>7} {'node_r':>8} {'score':>7} {'d adj_J':>8} {'d div_J':>8} "
      f"{'d score':>8} {'f0 dsc':>7} {'f1 dsc':>7}", flush=True)
b_ref = RES
f0_ref = _with_ratio([PER[n] for n in SIX_OK if FOLD_OF[n] == 0])
f1_ref = _with_ratio([PER[n] for n in SIX_OK if FOLD_OF[n] == 1])


def _armline(tag, rows_a, forks_a, per_a, dtp=None):
    rc = _with_ratio(rows_a)
    f0 = _with_ratio([per_a[n] for n in SIX_OK if FOLD_OF[n] == 0 and n in per_a])
    f1 = _with_ratio([per_a[n] for n in SIX_OK if FOLD_OF[n] == 1 and n in per_a])
    nf = len(forks_a) if forks_a is not None else -1
    ni = sum(1 for r_ in forks_a if r_[5][0] == 0) if forks_a else 0
    print(f"  {tag:<16} {nf:>7,} {ni:>7,} {nf - ni:>7,} "
          f"{sum(r['division_tp'] for r in rows_a):>4}/"
          f"{sum(r['division_fp'] for r in rows_a):>4}/"
          f"{sum(r['division_fn'] for r in rows_a):<4} "
          f"{rc['division_jaccard']:>7.4f} {rc['adj_edge_jaccard']:>7.4f} "
          f"{rc.get('edge_jaccard', float('nan')):>7.4f} "
          f"{rc.get('total_node_ratio', float('nan')):>+8.4f} {rc['score']:>7.4f} "
          f"{rc['adj_edge_jaccard'] - b_ref['adj_edge_jaccard']:>+8.4f} "
          f"{rc['division_jaccard'] - base_J:>+8.4f} "
          f"{rc['score'] - b_ref['score']:>+8.4f} "
          f"{f0['score'] - f0_ref['score']:>+7.4f} {f1['score'] - f1_ref['score']:>+7.4f}",
          flush=True)
    return rc


_armline(f"REF (dw {DIV_W})", ROWS, FORKS, PER)
for arm, dw in DIVW_ARMS:
    _armline(f"{arm} (dw {dw})", ARM_ROWS[arm], ARM_FORKS[arm], ARM_PER[arm])

print(f"\n=== THE CROSS: {len(DIVW_ARMS) + 1} DIV_W arms x 2 gates, real pruning at "
      f"p > 0.03, official scorer. Each gate is trained OUT OF FOLD on that arm's own "
      f"counted forks -- the arm changes the fork population -- and the joint column "
      f"adds divbench5's 67 counted 44b6 forks to both training folds ===", flush=True)
print(f"  {'cell':<20} {'div tp/fp/fn':>14} {'div_J':>7} {'adj_J':>7} {'score':>7} "
      f"{'vs REF+gate6':>13} {'f0':>8} {'f1':>8}   AUC / anchor", flush=True)


def _gate_and_prune(tag, forks_a, movies_a, per_a, thr, add=None, cols=None):
    cols = J20 if cols is None else cols
    """Fit a 2-fold gate on this arm's counted forks, prune, rescore."""
    Xa = np.array([r_[5] for r_ in forks_a], float) if forks_a else np.zeros((0, len(FEAT_NAMES)))
    la = np.array([r_[4] for r_ in forks_a], int)
    fa = np.array([r_[1] for r_ in forks_a], int)
    pa = np.full(len(forks_a), np.nan)
    cnt = la >= 1
    aucs = []
    for kk in (0, 1):
        tr = (fa != kk) & cnt
        if tr.sum() < 20 or (la[tr] == 2).sum() < 5:
            warn_once("arm_gate_small", f"{tag}: fold{kk} had {int(tr.sum())} counted forks")
            continue
        mg = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05,
                                            min_samples_leaf=10, random_state=0)
        Xtr_, ytr_ = Xa[tr][:, cols], (la[tr] == 2).astype(int)
        if add is not None and len(add[0]):
            Xtr_ = np.vstack([Xtr_, np.asarray(add[0], float)[:, cols]])
            ytr_ = np.concatenate([ytr_, np.asarray(add[1], int)])
        mg.fit(Xtr_, ytr_)
        te = fa == kk
        pa[te] = mg.predict_proba(Xa[te][:, cols])[:, 1]
        tc = te & cnt
        if tc.sum() and (la[tc] == 2).any() and (la[tc] == 1).any():
            aucs.append((int(tc.sum()), _auc((la[tc] == 2).astype(int), pa[tc])))
    by_movie = defaultdict(list)
    for i_, r_ in enumerate(forks_a):
        by_movie[r_[0]].append((i_, r_[2], r_[3]))
    rows_c, pf = [], {0: [], 1: []}
    for name in SIX_OK:
        key = movies_a[name]["key"]
        want = [(p_, q_) for i_, p_, q_ in by_movie[name]
                if np.isfinite(pa[i_]) and pa[i_] <= thr]
        miss = [e_ for e_ in want if e_ not in key]
        if miss:
            warn_once("prune_edge_missing",
                      f"{name}: {len(miss)} fork edges to prune are not in the "
                      f"snapshot's fork map -> NOT removed, the cut is a LOWER bound")
        drop = {key[e_] for e_ in want if e_ in key}
        row = _rescore(name, movies_a[name], drop, f"{tag}_p{thr}") if drop else per_a[name]
        rows_c.append(row)
        pf[FOLD_OF[name]].append(row)
    return _with_ratio(rows_c), rows_c, pf, aucs


THR = 0.03
CELLS = []          # (label, arm, gate) in the order they are printed
for _arm, _fa, _ma, _pa in ([("REF", FORKS, MOVIE, PER)]
                            + [(a_, ARM_FORKS[a_], ARM_MOVIE[a_], ARM_PER[a_])
                               for a_, _d in DIVW_ARMS]):
    _gates = ((("gate6", None), ("joint", (X44, Y44)), ("gate6img", None), ("jointimg", (X44IMG, Y44IMG))) if _arm == "REF"
              else (("joint", (X44, Y44)), ("jointimg", (X44IMG, Y44IMG))))
    for _gn, _add in _gates:
        CELLS.append((f"{_arm} + {_gn}", _arm, _gn, _fa, _ma, _pa, _add))
COLS_OF = {c_[0]: (J20 + JIMG if c_[2].endswith("img") else J20) for c_ in CELLS}
WANT = {"REF + gate6": "want 45/53/80 0.2528 0.9155 0.9408 (hparabench round 2)",
        "REF + joint": "want 49/51/76 0.2784 0.9155 0.9433 (round 2, what ships)"}
XRES = {}
for lab, _arm, _gn, _fa, _ma, _pa, _add in CELLS:
    rc, rows2, pf2, aucs = _gate_and_prune(lab.replace(" ", ""), _fa, _ma, _pa, THR,
                                           add=_add, cols=COLS_OF[lab])
    XRES[lab] = dict(rc=rc, rows=rows2, pf=pf2)
    base = XRES.get("REF + gate6")
    f0, f1 = _with_ratio(pf2[0]), _with_ratio(pf2[1])
    d = rc["score"] - base["rc"]["score"]
    df0 = f0["score"] - _with_ratio(base["pf"][0])["score"]
    df1 = f1["score"] - _with_ratio(base["pf"][1])["score"]
    print(f"  {lab:<20} "
          f"{sum(r['division_tp'] for r in rows2):>4}/"
          f"{sum(r['division_fp'] for r in rows2):>4}/"
          f"{sum(r['division_fn'] for r in rows2):<4} {rc['division_jaccard']:>7.4f} "
          f"{rc['adj_edge_jaccard']:>7.4f} {rc['score']:>7.4f} {d:>+13.4f} "
          f"{df0:>+8.4f} {df1:>+8.4f}   "
          + " ".join(f"{n}:{a:.4f}" for n, a in aucs)
          + "   " + WANT.get(lab, ""), flush=True)

print(f"\n  ADDITIVITY. If DIV_W and the joint gate fix different forks, the bottom "
      f"right equals the sum of the two single moves; if they fix the same ones, it "
      f"equals the larger of them.", flush=True)
_s = {k: XRES[k]["rc"]["score"] for k in XRES}
if "dw35 + joint" in _s:
    _dw = _s["dw35 + gate6"] - _s["REF + gate6"]
    _jt = _s["REF + joint"] - _s["REF + gate6"]
    _both = _s["dw35 + joint"] - _s["REF + gate6"]
    print(f"    DIV_W alone {_dw:+.4f}   joint alone {_jt:+.4f}   "
          f"sum if independent {_dw + _jt:+.4f}   larger alone {max(_dw, _jt):+.4f}",
          flush=True)
    print(f"    MEASURED TOGETHER {_both:+.4f}   -> "
          f"{100.0 * _both / (_dw + _jt) if (_dw + _jt) else float('nan'):.0f}% of the "
          f"independent sum", flush=True)

print(f"\n  PAIRED BOOTSTRAP over the {len(SIX_OK)} movies, 2,000 draws, on the OFFICIAL "
      f"SCORE of the pruned graphs -- the same resample for every cell, so the movie "
      f"draw cancels. divbench6's +0.0015 had no interval and this is the check on it "
      f"===", flush=True)
_rb = np.random.default_rng(9090)
_draws = [_rb.choice(len(SIX_OK), size=len(SIX_OK), replace=True) for _ in range(2000)]
BS = {k: np.array([_with_ratio([XRES[k]["rows"][i] for i in d])["score"] for d in _draws])
      for k in XRES}
print(f"    {'cell':<20} {'mean d vs REF+gate6':>20} {'90% interval':>24}  "
      f"distinguishable", flush=True)
for lab, _a, _g, _f, _m, _p, _x in CELLS:
    if lab == "REF + gate6":
        continue
    d = BS[lab] - BS["REF + gate6"]
    lo, hi = float(np.quantile(d, .05)), float(np.quantile(d, .95))
    print(f"    {lab:<20} {float(np.mean(d)):>+20.4f}   [{lo:>+8.4f}, {hi:>+8.4f}]  "
          f"{'YES' if (lo > 0 or hi < 0) else 'no'}", flush=True)
if "dw35 + joint" in BS:
    d = BS["dw35 + joint"] - BS["REF + joint"]
    lo, hi = float(np.quantile(d, .05)), float(np.quantile(d, .95))
    print(f"    {'dw35+joint vs REF+joint':<20} {float(np.mean(d)):>+13.4f}   "
          f"[{lo:>+8.4f}, {hi:>+8.4f}]  {'YES' if (lo > 0 or hi < 0) else 'no'}"
          f"   <- does DIV_W still buy anything once the gate is the joint one?",
          flush=True)

print(f"\n  PAIRED BOOTSTRAP of every arm + joint AGAINST REF + joint (what ships), "
      f"score and adj_J separately ===", flush=True)
_BSA = {k: np.array([_with_ratio([XRES[k]["rows"][i] for i in d])["adj_edge_jaccard"] for d in _draws])
        for k in XRES}
for a_, _c in BR_ARMS_RUN:
    lab = f"{a_} + joint"
    if lab not in BS:
        continue
    for nm_, bank in (("score", BS), ("adj_J", _BSA)):
        d = bank[lab] - bank["REF + joint"]
        lo, hi = float(np.quantile(d, .05)), float(np.quantile(d, .95))
        print(f"    {lab:<16} {nm_:<6} {float(np.mean(d)):>+9.4f}   [{lo:>+8.4f}, {hi:>+8.4f}]  "
              f"{'YES' if (lo > 0 or hi < 0) else 'no'}", flush=True)
print(f"\n  PAIRED BOOTSTRAP of every arm + jointimg AGAINST REF + jointimg (the image gate under submission), "
      f"score and adj_J ===", flush=True)
for lab_ in ["REF + jointimg"] + [f"{a_} + jointimg" for a_, _c in BR_ARMS_RUN]:
    if lab_ not in BS or "REF + joint" not in BS:
        continue
    for nm_, bank in (("score", BS), ("adj_J", _BSA)):
        d = bank[lab_] - bank["REF + jointimg"] if lab_ != "REF + jointimg" else bank[lab_] - bank["REF + joint"]
        lo, hi = float(np.quantile(d, .05)), float(np.quantile(d, .95))
        ref_ = "REF + joint" if lab_ == "REF + jointimg" else "REF + jointimg"
        print(f"    {lab_:<26} vs {ref_:<15} {nm_:<6} {float(np.mean(d)):>+9.4f}   [{lo:>+8.4f}, {hi:>+8.4f}]  "
              f"{'YES' if (lo > 0 or hi < 0) else 'no'}", flush=True)
print("\n  THE NODE COST AGAINST 0.35: Z50CR / Z65CR vs ZCR (the same triples, patch and gate recipe; the node cost is the only "
      "difference), paired bootstrap over the movies, gated and ungated ===", flush=True)
for _pair in (("Z50CR_NOCAP_DV150 + jointimg", "ZCR_NOCAP_DV150 + jointimg"), ("Z50CR_NOCAP_DV150 + joint", "ZCR_NOCAP_DV150 + joint"),
              ("Z65CR_NOCAP_DV150 + jointimg", "ZCR_NOCAP_DV150 + jointimg"), ("Z65CR_NOCAP_DV150 + joint", "ZCR_NOCAP_DV150 + joint"),
              ("Z65CR_NOCAP_DV150 + jointimg", "Z50CR_NOCAP_DV150 + jointimg")):
    if _pair[0] not in BS or _pair[1] not in BS:
        continue
    for nm_, bank in (("score", BS), ("adj_J", _BSA)):
        d = bank[_pair[0]] - bank[_pair[1]]
        lo, hi = float(np.quantile(d, .05)), float(np.quantile(d, .95))
        print(f"    {_pair[0]:<29} vs {_pair[1]:<29} {nm_:<6} {float(np.mean(d)):>+9.4f}   [{lo:>+8.4f}, {hi:>+8.4f}]  "
              f"{'YES' if (lo > 0 or hi < 0) else 'no'}", flush=True)
print("    reading: a node cost above 0.35 is worth a submission only if its arm beats ZCR on the score with the interval off zero; "
      "the hidden scorer has so far paid more for fewer nodes than the local one (three submissions on one line).", flush=True)
print("    reading: an arm counts only if its jointimg cell beats REF + jointimg with the interval off zero; the "
      "DIST_CAP4_DV150 control says whether the ranking or the cap did it.", flush=True)

for a_, _c in BR_ARMS_RUN:
    _csv_a = pathlib.Path("/kaggle/working") / f"forkprop_arm_{a_}.csv"
    _keys_a = sorted({k for r in ARM_PER[a_].values() for k in r})
    with open(_csv_a, "w", encoding="utf-8", newline="\n") as _fh:
        _fh.write("dataset,fold," + ",".join(_keys_a) + "\n")
        for _n in sorted(ARM_PER[a_]):
            _r = ARM_PER[a_][_n]
            _fh.write(f"{_n},{FOLD_OF.get(_n, -1)}," + ",".join(repr(_r.get(_k)) for _k in _keys_a) + "\n")
    print(f"  {_csv_a.name} written (ungated per-movie rows of arm {a_})", flush=True)
warning_summary()
print(f"\nDIVBENCH9 DONE in {(time.time() - t_all) / 60:.1f} min (stage B "
      f"{(time.time() - tb) / 60:.1f} min), {len(ROWS)} of {len(SIX)} movies scored", flush=True)
_csv = pathlib.Path("/kaggle/working") / f"forkprop_REF_{DET_SRC}.csv"
_keys = sorted({k for r in PER.values() for k in r})
with open(_csv, "w", encoding="utf-8", newline="\n") as _fh:
    _fh.write("dataset,fold," + ",".join(_keys) + "\n")
    for _n in sorted(PER):
        _r = PER[_n]
        _fh.write(f"{_n},{FOLD_OF.get(_n, -1)},"
                  + ",".join(repr(_r.get(_k)) for _k in _keys) + "\n")
print(f"\n  {_csv.name} written ({len(PER)} movies, {len(_keys)} columns). "
      f"Pair it with the other DET_SRC run -- same movies, same folds, same "
      f"order -- and bootstrap the per-movie score difference; a single "
      f"aggregate cannot show whether one movie carries the whole delta, "
      f"which is exactly the failure this comparison is exposed to.",
      flush=True)

if DET_SRC == "F4HP":
    _moved = [f for f in ANCHOR_FAIL if f[0] in ("ANCHOR_D187", "REF_FORKS")]
    ANCHOR_FAIL = [f for f in ANCHOR_FAIL if f not in _moved]
    for _f in _moved:
        print(f"  [expected] {_f[0]} {_f[1]} moved to {_f[2]} from {_f[3]} -- "
              f"this anchor sits on the 12 movies whose coordinates this arm "
              f"changes, so it CANNOT hold; it is the measurement", flush=True)
assert not ANCHOR_FAIL, f"hard anchor miss: {ANCHOR_FAIL}"
_r2 = (TP, FP, FN) == (55, 237, 70) and abs(RES["adj_edge_jaccard"] - 0.9146) <= ANCHOR_TOL
print(f"  ROUND-2 ANCHOR: REF {TP}/{FP}/{FN} adj_J {RES['adj_edge_jaccard']:.4f} want 55/237/70 "
      f"0.9146 -> {'OK' if _r2 else 'MISS'}", flush=True)
_r2c = XRES.get("REF + joint", {}).get("rc", {})
print(f"  ROUND-2 ANCHOR: REF + joint score {_r2c.get('score', float('nan')):.4f} want 0.9433 -> "
      f"{'OK' if abs(_r2c.get('score', 0) - 0.9433) <= ANCHOR_TOL else 'MISS'}", flush=True)
if len(ROWS) == len(SIX) and not INCOMPLETE and not INCOMPLETE2 and _r2:
    print("NCSWEEP OK", flush=True)
else:
    print(f"[degraded] FORKPROPZ INCOMPLETE or anchor miss: {len(ROWS)}/{len(SIX)}, "
          f"round-2 anchor {'OK' if _r2 else 'MISS'}", flush=True)
