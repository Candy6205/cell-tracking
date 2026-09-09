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

SCRIPT_VERSION = "2026-09-10a-rkprod"
# Which primaries the fold rankers LEARN on. Stage B always solves
# DET_SRC (F4), so "BASE" is the shipped mismatch and "F4" removes it.
RANK_TRAIN = "F4"        # rkprod: NOT an arm selector any more -- it is
                         # what makes the loop build LAB4 for all 199.
                         # Both pkls are trained, from LAB and from LAB4.
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
DET_SRC = "F4"
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
            proposals.append((d_pq + DIV_W_SISTER * d_s, t, p, q, c1, d_pq, d_s, dv_))
    proposals.sort()
    used_p = set()
    used_q = set()
    per_frame = defaultdict(int)
    n_frame = {t: len(ids) for t, ids in by_t.items()}
    max_global = max(1, int(DIV_GLOBAL_CAP * len(nodes)))
    added = 0
    for score, t, p, q, c1, d_pq, d_s, dv_ in proposals:
        if added >= max_global:
            PATCH_CAPS["orphan_global"] += 1
            break
        if p in used_p or q in used_q:
            continue
        if per_frame[t] >= max(1, int(DIV_FRAME_CAP * n_frame[t])):
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
# fuse44b6 + ep44b6 (2026-09-07, 59 of 59 each) completed the F4 side of the
# pool. Without them RANK_TRAIN="F4" would silently train on 140 movies
# against the BASE arm's 175, and the delta would be pool size, not primaries.
if RANK_TRAIN == "F4":
    _need = [n for n in ALL_TRAIN if n not in FDET_F4]
    assert not _need, (
        f"RANK_TRAIN=F4 but {len(_need)} movies have no fdet_F4_ cache "
        f"({_need[:4]}) -- attach faustine649/biohub-fuse44b6. A smaller pool "
        f"than the BASE arm's would make the delta pool size, not primaries.")
    _need = [n for n in ALL_TRAIN if n not in EDGEPROB_F4]
    assert not _need, (
        f"RANK_TRAIN=F4 but {len(_need)} movies have no edgeprob_F4_ cache "
        f"({_need[:4]}) -- attach faustine649/biohub-ep44b6.")
    print(f"  RANK_TRAIN=F4: all {len(ALL_TRAIN)} movies carry F4 primaries "
          f"and F4 head probabilities; the F4 pool IS the BASE pool.",
          flush=True)
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
DIVW_ARMS = [("dw35", 0.35)]   # divbench6's winner: +0.0015, both folds up
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
    _solved = role == "6bba" or name in SET_44B6
    if _solved or RANK_TRAIN == "F4":
        dets4 = load_dets(name, "F4")
        T4 = len(dets4)
        seeds4, feats4 = features_ext(name, dets4, T4, EP_OF["F4"][name],
                                      EP2_OF["F4"].get(name))
        # rkprod solves nothing, so the full features are NEVER kept -- only
        # the labelled rows. That is what lets all 199 movies be labelled on
        # BOTH primary sets inside one kernel's memory.
        LAB4[name] = label_rows(name, feats4)
        n4 = f" F4 pairs={int(sum(len(p) for p, _r, _P, _C in feats4))}"
        del dets4, seeds4, feats4
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


# ---- THE TWO PRODUCTION RANKERS -------------------------------------------
PROD_FLOOR_CHK = 0.48        # ref 56030644 / LB 0.933: the ILP candidate floor
COLS = ARMS["+BIDIR"]
assert COLS[:NFEAT_BASE] == list(range(NFEAT_BASE)), COLS
EXTRA = [NEW_NAMES[c] for c in COLS[NFEAT_BASE:]]
assert EXTRA == ["p_fwd", "p_rev", "h_fused"], EXTRA
assert len(COLS) == 21, len(COLS)

# The pool is the shipped one, asserted rather than assumed: a ranker that saw
# an evaluation movie would make every local number on this project unreadable.
PROD_POOL = list(POOL175)
assert len(PROD_POOL) == 175, len(PROD_POOL)
assert not (set(PROD_POOL) & set(HOLDOUT_6BBA)), "pool overlaps the 6bba holdout"
assert not (set(PROD_POOL) & set(SET_44B6)), "pool overlaps the 44b6 draw"
assert set(PROD_POOL) == set(POOL_BIDIR), (
    f"this pool is not the shipped ranker's: only-here "
    f"{sorted(set(PROD_POOL) - set(POOL_BIDIR))[:4]} only-there "
    f"{sorted(set(POOL_BIDIR) - set(PROD_POOL))[:4]}")
for _t, _tab in (("LAB", LAB), ("LAB4", LAB4)):
    _need = [n for n in PROD_POOL if n not in _tab]
    assert not _need, f"{_t} is missing {len(_need)} pool movies: {_need[:4]}"

print(f"\n=== THE TWO PRODUCTION RANKERS, D175 pool ({len(PROD_POOL)} movies) ===",
      flush=True)
print("  Estimator, columns, seed, split and refit = edgeprob_rank2.py:1707-1731\n"
      "  exactly (HistGB max_iter 300, random_state 0, seeded 10% val split,\n"
      "  then a refit on the whole pool). The ONLY difference between the two\n"
      "  rows below is which table the rows come from.", flush=True)

MODELS_PROD, ROWS_PROD = {}, {}
for tag, table, fname in (("BASE", LAB, "edge_ranker_prod_base_bidir.pkl"),
                          ("F4", LAB4, "edge_ranker_prod_f4_bidir.pkl")):
    t0 = time.time()
    Xa = np.concatenate([table[n][0] for n in PROD_POOL]).astype(np.float32)[:, COLS]
    ya = np.concatenate([table[n][1] for n in PROD_POOL])
    perm = np.random.default_rng(0).permutation(len(ya))
    n_val = max(1000, len(ya) // 10)
    vi, ti = perm[:n_val], perm[n_val:]
    m = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    m.fit(Xa[ti], ya[ti])
    val_auc = _auc(ya[vi], m.predict_proba(Xa[vi])[:, 1])
    m.fit(Xa, ya)                          # refit on the whole pool
    MODELS_PROD[tag] = m
    ROWS_PROD[tag] = (len(ya), int(ya.sum()), val_auc)
    with open(f"/kaggle/working/{fname}", "wb") as f:
        pickle.dump({"model": m,
                     "cols": list(COLS),
                     "pool": list(PROD_POOL),
                     "extra_cols": EXTRA,
                     "feature_names": BASE_NAMES + EXTRA,
                     "n_features": len(COLS),
                     "h_fused": "2*p_fwd*p_rev/(p_fwd+p_rev); p_fwd where "
                                "p_rev is NaN",
                     "version": SCRIPT_VERSION,
                     "rank_train": tag,
                     "detector": "F4",
                     "n_train_rows": int(len(ya)),
                     "n_train_pos": int(ya.sum()),
                     "val_auc": float(val_auc)}, f)
    print(f"  {tag:<4} {fname}  {len(ya)} rows ({ya.mean():.3f} pos)  "
          f"val AUC {val_auc:.4f}  {time.time() - t0:.0f}s", flush=True)
    del Xa, ya
    gc.collect()

# ---- THE ANCHOR -----------------------------------------------------------
# Both pkls are new. The leaderboard control for the F4 one is ref 56086066,
# which loaded a pkl this kernel did not make, so the question that decides
# whether ONE submission is enough is: does a BASE-trained ranker from THIS
# code behave like the shipped one? Read on the 12 HOLDOUT_6BBA movies' BASE
# rows -- in neither pool, so held out for both models.
print(f"\n=== THE ANCHOR: this code's BASE ranker against the SHIPPED one ===",
      flush=True)
print(f"  Read on the {len(HOLDOUT_6BBA)} HOLDOUT_6BBA movies' BASE rows, "
      f"held out of both pools.", flush=True)
_hx = np.concatenate([LAB[n][0] for n in HOLDOUT_6BBA]).astype(np.float32)
_hy = np.concatenate([LAB[n][1] for n in HOLDOUT_6BBA])
p_ship = MODEL_BIDIR.predict_proba(_hx[:, MCOLS_BIDIR])[:, 1]
p_ctl = MODELS_PROD["BASE"].predict_proba(_hx[:, COLS])[:, 1]
p_f4 = MODELS_PROD["F4"].predict_proba(_hx[:, COLS])[:, 1]
print(f"  {len(_hy)} rows, {int(_hy.sum())} positive", flush=True)
print(f"  {'pair':<22}{'AUC':>9}{'max|dp|':>10}{'mean|dp|':>10}"
      f"{'floor 0.48 side differs':>26}", flush=True)


def _cmp(nm, pa, pb):
    d = np.abs(pa - pb)
    side = float(((pa > PROD_FLOOR_CHK) != (pb > PROD_FLOOR_CHK)).mean())
    print(f"  {nm:<22}{_auc(_hy, pb):>9.4f}{d.max():>10.5f}{d.mean():>10.6f}"
          f"{side * 100:>25.3f}%", flush=True)
    return side


print(f"  {'SHIPPED ep175_bidir':<22}{_auc(_hy, p_ship):>9.4f}"
      f"{0.0:>10.5f}{0.0:>10.6f}{'--':>26}", flush=True)
_s_ctl = _cmp("this code, BASE", p_ship, p_ctl)
_s_f4 = _cmp("this code, F4", p_ship, p_f4)

print("\n  READING, fixed here before the numbers are looked at:", flush=True)
print("    the BASE row's floor-side disagreement is the control's own\n"
      "    contribution. If it is SMALL against the F4 row's, the F4 pkl is a\n"
      "    single-variable swap and is submitted alone, read against LB 0.944.\n"
      "    If the two are comparable, the BASE pkl is itself a change and BOTH\n"
      "    pkls get submitted, the F4 one read against the BASE one.", flush=True)
print(f"    BASE {_s_ctl * 100:.3f}%   F4 {_s_f4 * 100:.3f}%   ratio "
      f"{(_s_f4 / _s_ctl if _s_ctl else float('inf')):.1f}x", flush=True)

print(f"\nRKPROD DONE in {(time.time() - t_all) / 60:.1f} min", flush=True)
print("RKPROD OK", flush=True)
