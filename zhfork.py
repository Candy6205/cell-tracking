"""The seven post-link constants nobody has ever touched, in THREE domains.

SCRIPT_VERSION 2026-09-09a-zhpara

WHAT IS BEING SWEPT. Reading `submit_ranker.py`, the 0.921 pipeline's
post-link tail carries seven constants inherited verbatim from the public
solution and never once varied:

    _gap_close    GAP1_GATE_UM 9.0, GAP1_SNAP_UM 3.2,
                  GAP1_MIN_CAND_SCORE 0.10, GAP1_CAP_FRAC 0.003
    _short_filter SHORT_MIN 6
    _linefit      LINEFIT_WEIGHT 0.8, LINEFIT_WINDOW 2

THE PROBLEM THE USER NAMED, and it governs the whole design. Sweeping
knobs against the same 12 holdout datasets is US overfitting, not a model
overfitting -- researcher degrees of freedom. With adj_J noise around
0.002-0.003 on that holdout, the best of ~25 comparisons is expected to
sit 2 to 2.5 SD above the truth by chance, i.e. **a fake +0.005 to
+0.007, exactly the size of everything being chased.** And all 12 are
`6bba`, one imaging batch, so even a real effect could be batch-specific.
The local holdout has already inverted against the leaderboard once:
windowing measured 0.9382 locally against global's 0.9363 and then
scored 0.920 against 0.921 on the LB.

THE FIX IS THREE DOMAINS, and the user's suggestion of adding Zebrahub is
what makes the third one possible:

    6bba      poolsize.py's 12 holdout, verbatim
    44b6      12 datasets from the OTHER imaging batch. Very different
              data -- 327 cells/frame against 97, displacement p99
              4.90 um against 7.81, duplicate-frame rate 0% against 7.47%
    zebrahub  a different instrument entirely, scored with the OFFICIAL
              metric against its own dense GT (the machinery is
              `zh_pipeline.py`'s, carried over)

**PRE-REGISTERED DECISION RULE, fixed before the run: a change counts
only if its delta has the SAME SIGN in all three domains.** A
same-sign-in-three result is a property of the method; a win in one
domain is a property of that domain. No exceptions after the fact, and
no reading of the largest delta as "the winner" if the signs disagree.

WHY THE RANKER'S LEAK DOES NOT BREAK THIS. D187 trained on 187 of the 199
competition datasets, so its edge probabilities on the 44b6 arm are
optimistic. That inflates the ABSOLUTE adj_J there -- but every
configuration in this sweep shares the same ranker, the same detections
and the same ILP solution, so the bias is identical in both arms of every
comparison and cancels in the DELTA. This is precisely why the sweep is
restricted to post-link constants: touching NMS or the detector would
change which edges the ranker scores, and then the leak would not cancel.

COST. The ILP solve dominates, and it does not depend on any constant
being swept. So each dataset is solved ONCE and every configuration is
applied to a fresh copy of that solved graph. Fifteen configurations
therefore cost little more than one solve plus fifteen cheap patch-and-
score passes.

ANCHOR. The BASE configuration is the shipped one and its 6bba row must
reproduce `biohub-poolsize`'s D187: score 0.9386, adj_J 0.9254, div_J
0.1316, TP/FP/FN 5/26/7.
"""
from __future__ import annotations

import gc
import pathlib
import pickle
import random
import subprocess
import sys
import time
from collections import defaultdict

SCRIPT_VERSION = "2026-09-25a-zhfork"  # zhpara4 was "2026-09-09c-zhpara4"
# v4, after the daughter-stealing diagnosis (user's): the ranker's edge
# gain (+0.0055 adj_J) was cancelled by divisions because 1:1 Hungarian
# steals second daughters into wrong continuations, draining the orphan
# pool the division patch mines. Reconciliation: the ranker's LABELS
# already treat both GT children as positives, so a capacity-2 global
# solve (tracksdata ILP fed RANKER probs, division surcharge priced)
# can take both daughters natively. B line tests that; A line continues
# the Hungarian p_min sweep upward.
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

HOLDOUT = [
    "6bba_268e1230", "6bba_2312ac41", "6bba_283bf9f1", "6bba_3a1849c2",
    "6bba_3abfe10a", "6bba_5c824876", "6bba_61dd1e0d", "6bba_7af54fde",
    "6bba_7b5d3b2c", "6bba_afb141ff", "6bba_aeee7805", "6bba_c27cba08",
]
P_MIN_SWEEP = ()        # A line: measured twice, identical. Skipped.
ILP_DIV_SWEEP = (0.20, 0.50, 0.75, 1.50)   # B line, widened
ILP_DISAPP = 1.5
ILP_FLOOR = 0.10                            # candidates fed to the ILP
GATE_UM = 10.0
MATCH_UM = 5.0

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

SCALE = np.array([1.625, 0.40625, 0.40625])
UNET_THRESH = 0.15
CAND_THR = 0.05
SHORT_MIN = 6            # REBOUND per configuration
LINEFIT_WEIGHT = 0.8     # REBOUND per configuration
LINEFIT_WINDOW = 2       # REBOUND per configuration
MAX_LINK_UM = 10.0
TIGHT_UM = 6.0
LINK_SIM_W = 2.0
GAP1_GATE_UM = 9.0       # REBOUND per configuration
GAP1_SNAP_UM = 3.2       # REBOUND per configuration
GAP1_MIN_CAND_SCORE = 0.10   # REBOUND per configuration
GAP1_CAP_FRAC = 0.003
# Shipped behaviour: bridge ONE missing frame, and only onto a real
# candidate from the repair band. Both are swept below; BASE must
# reproduce the recorded anchor with these values, which is what
# proves the rewritten _gap_close did not change behaviour.
GAP_SPAN = 1
GAP_NEED_CAND = True
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
DC_ADD_SCORE = 0.15        # == UNET_THRESH; see the header
_DC_STATS = [0, 0, 0]      # frames, our points, points ADDED
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
print("  (zhfork: the 6bba detection caches are not needed; the competition domains are not run)", flush=True)


DCDET = {f.stem[len("dcdet_"):]: f
         for f in pathlib.Path("/kaggle/input").glob("*/**/dcdet_*.npz")}


def _dc_merge(dets, name):
    """ADD DeepCenter's peaks to ours, never remove or reorder ours.

    A DeepCenter peak is kept only when it is further than DC_MERGE_UM
    from EVERY one of our peaks at that frame. No joint NMS: that would
    let a higher-scoring DeepCenter peak suppress one of ours, and the
    whole safety argument for this change is that our detections are
    untouched.
    """
    if DC_TAU is None or name not in DCDET:
        return dets
    z = np.load(DCDET[name])
    out = []
    for t, (c, sc) in enumerate(dets):
        c = np.asarray(c, float).reshape(-1, 3)
        sc = np.asarray(sc, float).reshape(-1)
        kc = f"c{t}"
        if kc not in z.files:
            out.append((c, sc))
            continue
        dc = np.asarray(z[kc], float).reshape(-1, 3)
        ds = np.asarray(z[f"s{t}"], float).reshape(-1)
        dc = dc[ds >= DC_TAU]
        mine = c[sc >= UNET_THRESH]
        _DC_STATS[0] += 1
        _DC_STATS[1] += len(mine)
        if len(dc) == 0:
            out.append((c, sc))
            continue
        if len(mine):
            d, _ = cKDTree(mine * SCALE[None, :]).query(dc * SCALE[None, :])
            dc = dc[d > DC_MERGE_UM]
        _DC_STATS[2] += len(dc)
        if len(dc) == 0:
            out.append((c, sc))
            continue
        out.append((np.concatenate([c, dc], 0),
                    np.concatenate([sc, np.full(len(dc), DC_ADD_SCORE)], 0)))
    return out


def load_dets(name):
    z = np.load(CACHES[name])
    T = max(int(k[1:]) for k in z.files if k.startswith("c")) + 1
    return _dc_merge([(z[f"c{t}"], z[f"s{t}"]) for t in range(T)], name)


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
                    if hit is None:
                        use = np.asarray(mid, float)
                    else:
                        use = np.asarray(cand[tt][hit], float)
                        cand_used[tt][hit] = True
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


def _add_safe_divisions(nodes, edges):
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
            if _dist_um(nodes[c1n[0]]["xyz"], nodes[qn[0]]["xyz"]) - d_s \
                    < DIV_DIVERGE_UM:
                continue
            proposals.append((d_pq + DIV_W_SISTER * d_s, t, p, q))
    proposals.sort()
    used_p = set()
    used_q = set()
    per_frame = defaultdict(int)
    n_frame = {t: len(ids) for t, ids in by_t.items()}
    max_global = max(1, int(DIV_GLOBAL_CAP * len(nodes)))
    added = 0
    for score, t, p, q in proposals:
        if added >= max_global:
            break
        if p in used_p or q in used_q:
            continue
        if per_frame[t] >= max(1, int(DIV_FRAME_CAP * n_frame[t])):
            continue
        edges.append((p, q))
        succ[p].append(q)
        pred[q].append(p)
        used_p.add(p)
        used_q.add(q)
        per_frame[t] += 1
        added += 1
    return added


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


# ---------------- the division classifier: build and train ---------------
# The unit is a TRIPLE (parent, candidate a, candidate b), not an edge.
# That is the whole point: the edge ranker scores each edge on its own and
# therefore cannot express "this long edge is a second daughter BECAUSE the
# other daughter is over there" -- the hoct paper (arXiv 2607.11754, from
# the host's own lab) makes the same argument, that edges must attend to
# one another because a division entangles two lineage paths.
#
# Features: everything, xyz included. The user's call after a long
# measurement pass, and the right one -- the in-domain validation set
# decides what survives, not a prior argument about which axis transfers.
# Both raw micrometres and spacing-normalised versions go in, so the model
# can pick whichever generalises.

MIN_SCORE_A = 0.05          # a candidate must at least be a detection
TOPK = 6                    # candidates considered per parent
LOOKAHEAD = 2               # frames a daughter must persist to count


def spacing_at(pts_um, x_um, k=5):
    if len(pts_um) <= k:
        return np.nan
    d, _ = cKDTree(pts_um).query(x_um, k=min(k + 1, len(pts_um)))
    return float(np.median(np.asarray(d)[1:]))


def triple_feats(P, C, N, i, ja, jb, ps, cs, vel, sp, t, T):
    """One (parent i at t, candidates ja/jb at t+1) triple -> feature row."""
    p = P[i] * SCALE
    a, b = C[ja] * SCALE, C[jb] * SCALE
    va, vb = a - p, b - p
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    sep = float(np.linalg.norm(a - b))
    cosab = float(va @ vb) / max(na * nb, 1e-9)
    ang = float(np.degrees(np.arccos(np.clip(cosab, -1, 1))))
    sp = sp if np.isfinite(sp) and sp > 0 else np.nan
    far, near = max(na, nb), min(na, nb)
    vfar = va if na >= nb else vb
    return [
        na, nb, far, near, near / max(far, 1e-9),           # 0-4 distances
        sep, sep / max(sp, 1e-9), far / max(sp, 1e-9),      # 5-7 normalised
        ang, cosab,                                          # 8-9 opposition
        abs(float(va[0])), abs(float(vb[0])),                # 10-11 z parts
        abs(float(vfar[0])) / max(far, 1e-9),                # 12 z share
        float(np.linalg.norm(va[1:])),                       # 13 xy part a
        float(np.linalg.norm(vb[1:])),                       # 14 xy part b
        float(ps[i]), float(cs[ja]), float(cs[jb]),          # 15-17 scores
        min(cs[ja], cs[jb]) / max(max(cs[ja], cs[jb]), 1e-9),  # 18 sym
        float(np.linalg.norm(vel)) if vel is not None else 0.0,  # 19 speed
        sp if np.isfinite(sp) else 0.0,                      # 20 spacing
        float(len(P)), float(len(C)), float(N),              # 21-23 counts
        t / max(1, T),                                       # 24 phase
    ]


NFEAT = 25
KEEP = 0.01        # negative keep rate; positives are never dropped


NEG_PER_POS = 400        # keep every positive, thin the negatives




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




from sklearn.ensemble import HistGradientBoostingClassifier    # noqa: E402
from sklearn.metrics import roc_auc_score                      # noqa: E402

HOLDOUT = [
    "6bba_268e1230", "6bba_2312ac41", "6bba_283bf9f1", "6bba_3a1849c2",
    "6bba_3abfe10a", "6bba_5c824876", "6bba_61dd1e0d", "6bba_7af54fde",
    "6bba_7b5d3b2c", "6bba_afb141ff", "6bba_aeee7805", "6bba_c27cba08",
]
COLS = {"D32": list(range(NFEAT_BASE)), "D187": list(range(NFEAT_BASE))}
ARMS = ("D32", "D187")
ACC_NAMES = ["acc_len", "acc_cos", "md_acc", "md-md_acc", "acc_hist"]
DIV_W = 0.55
MATCH_UM = 5.0


# ---- the D187 ranker, loaded not retrained -----------------------------
MIN_POOL = 100
_pk = None
for _p in pathlib.Path("/kaggle/input").glob("*/**/edge_ranker_d187.pkl"):
    _pk = _p
    break
assert _pk is not None, ("edge_ranker_d187.pkl not found -- attach "
                         "faustine649/biohub-poolsize as a kernel source.")
with open(_pk, "rb") as _f:
    _blob = pickle.load(_f)
_pool = _blob.get("pool") if isinstance(_blob, dict) else None
assert _pool is not None and len(_pool) >= MIN_POOL, (
    f"the pkl at {_pk} was trained on {0 if _pool is None else len(_pool)} "
    f"datasets; this run requires >= {MIN_POOL}, i.e. the D187 model.")
MODEL = _blob["model"]
MCOLS = _blob["cols"]
print(f"  ranker loaded from {_pk}, pool {len(_pool)}", flush=True)



# ---- zhfork: the shipped composition's constants that the patch and the solve read ---------------------------------
ILP_FLOOR = 0.48                 # zhattach / zhbridge: the production floor
SHORT_MIN = 4
GAP_NEED_CAND = False
GAP1_CAP_FRAC = 0.02
DIV_DIVERGE_UM = 1.5             # forkprop's CR_NOCAP_DV150 arm
FP_RANK = "contrast"
FP_CAP = 100.0
FP_MEMO = {}
FP_STAT = defaultdict(int)
DA_ADMIT = False
DA_MIN_SEP_UM = 3.0
DA_CTX = {}                      # read only when DA_ADMIT (never here)
PATCH_CAPS = defaultdict(int)    # the patch's cap counters
FORK_LOG = {}
STEAL_LOG = {}
PRE_STATE = {}
IMG_CTX = {}
FI_STAT = defaultdict(int)
FI_HALF_UM, FI_PEAK_UM, FI_SIZE_UM = 8.0, 2.0, 3.5
FI_SIGMA_VOX = (0.6, 1.5, 1.5)
BUDGET_S = 10.0 * 3600
MATCH_UM = 5.0
FEAT20 = ["src", "d_pc1", "d_pq", "d_s", "dv", "p_c1", "p_q", "pf_c1", "pf_q",
          "pr_q", "sc_p", "sc_c1", "sc_q", "len_back", "len_c1", "len_q",
          "dens10", "n_orph", "t_frac", "rank_gap"]
FEAT_NEW = ["dv2", "ang", "sc_q_r", "sc_p_r", "dens5", "p_q_marg", "q_ends", "steal"]
SZ2 = ["hm10", "c30", "c70", "prof", "mom"]          # sz2: five more size measures (see _fi_sizes2)
FEAT_IMG = ["i_p", "c_p", "r_p", "br_p", "c_c1", "c_q", "r_c1", "r_q", "sym_i", "sym_r", "dau_ratio",
            "r_hm_p", "r_hm_c1", "r_hm_q", "rhm_max", "rhm_sum"] + [f"{m_}_{s_}" for m_ in SZ2 for s_ in ("p", "c1", "q", "max", "sum")]
FEAT_NAMES = FEAT20 + FEAT_NEW + FEAT_IMG      # the first 20 are divbench3's, in order; the last 11 read the image
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


def _fi_sizes2(img, t, xyz_vox):
    """five more size measures (um) of one node from the same smoothed crop as _fi_hm; NaN when unreadable:
    hm10  half-max radius with the background = the crop's 10th percentile instead of its median
    c30   the connected set at or above bg + 0.3 (peak - bg) holding the peak (bg = median), as an equal-volume sphere
    c70   the same at 0.7
    prof  radial-profile radius: the shell-averaged intensity around the peak (0.5 um shells) falls below
          far + 0.5 (peak - far), far = the mean of the 6-8 um shells; no connectivity, a touching neighbour cannot merge
    mom   the background-subtracted, intensity-weighted radius of gyration inside the half-max component (no fixed window)"""
    out = {k_: np.nan for k_ in SZ2}
    sm, off = _fi_crop(img, t, xyz_vox)
    if sm is None:
        return out
    c = np.asarray(xyz_vox, float) - off
    zz, yy, xx = np.indices(sm.shape)
    d = np.sqrt(((zz - c[0]) * SCALE[0]) ** 2 + ((yy - c[1]) * SCALE[1]) ** 2 + ((xx - c[2]) * SCALE[2]) ** 2)
    near = d <= FI_PEAK_UM
    if not near.any():
        return out
    j = np.unravel_index(int(np.argmax(np.where(near, sm, -np.inf))), sm.shape)
    peak = float(sm[j])
    bg = float(np.median(sm))
    bg10 = float(np.percentile(sm, 10))
    vox = float(np.prod(SCALE))
    dp = np.sqrt(((zz - j[0]) * SCALE[0]) ** 2 + ((yy - j[1]) * SCALE[1]) ** 2 + ((xx - j[2]) * SCALE[2]) ** 2)

    def _rad(mask):
        if not mask[j]:
            return np.nan
        lab_, _n = _ndi.label(mask)
        vol = int((lab_ == lab_[j]).sum()) * vox
        return float((3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0))

    if peak > bg10:
        out["hm10"] = _rad(sm >= bg10 + 0.5 * (peak - bg10))
    if peak > bg:
        out["c30"] = _rad(sm >= bg + 0.3 * (peak - bg))
        out["c70"] = _rad(sm >= bg + 0.7 * (peak - bg))
        lab_, _n = _ndi.label(sm >= bg + 0.5 * (peak - bg))
        comp = lab_ == lab_[j]
        w = np.clip(sm[comp] - bg, 0, None)
        out["mom"] = float(np.sqrt((w * dp[comp] ** 2).sum() / w.sum())) if w.sum() > 0 else np.nan
    shells = np.arange(0.0, 8.5, 0.5)
    prof = np.full(len(shells) - 1, np.nan)
    for i_, (a_, b_) in enumerate(zip(shells[:-1], shells[1:])):
        m_ = (dp >= a_) & (dp < b_)
        if m_.any():
            prof[i_] = float(sm[m_].mean())
    far_ = prof[12:16]
    far = float(np.nanmean(far_)) if np.isfinite(far_).any() else np.nan
    if np.isfinite(far) and peak > far:
        half = far + 0.5 * (peak - far)
        r_ = np.nan
        for i_ in range(len(prof)):
            if np.isfinite(prof[i_]) and prof[i_] < half:
                if i_ == 0:
                    r_ = 0.25
                else:
                    p0, p1 = prof[i_ - 1], prof[i_]
                    x0, x1 = shells[i_ - 1] + 0.25, shells[i_] + 0.25
                    r_ = (x0 + (p0 - half) / (p0 - p1) * (x1 - x0)) if (np.isfinite(p0) and p0 > p1) else x1
                break
        out["prof"] = float(r_)
    return out


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
    s_p, s_1, s_q = _fi_sizes2(img, t, nodes[p]["xyz"]), _fi_sizes2(img, t + 1, nodes[c1]["xyz"]), _fi_sizes2(img, t + 1, nodes[q]["xyz"])
    for m_ in SZ2:
        a_, b_, c_ = s_p[m_], s_1[m_], s_q[m_]
        ok_ = np.isfinite(a_) and np.isfinite(b_) and np.isfinite(c_) and a_ > 0
        out.update({f"{m_}_p": a_, f"{m_}_c1": b_, f"{m_}_q": c_,
                    f"{m_}_max": (max(b_, c_) / a_) if ok_ else np.nan, f"{m_}_sum": ((b_ + c_) / a_) if ok_ else np.nan})
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
        rows.append((name, k, p, q, label, [row[f] for f in FEAT_NAMES],
                     (float(t),) + tuple(float(v) for v in nodes[p]["xyz"]) + tuple(float(v) for v in nodes[c1]["xyz"])
                     + tuple(float(v) for v in nodes[q]["xyz"])))
    return rows
def zh_load(name):
    """(dets, T, gt_nodes, gt_edges, n_cross) for one Zebrahub crop.
    Carried over from zh_pipeline.py unchanged."""
    z = np.load(ZH[name])       # ZH is rebound per arm; see the driver
    T = max(int(k[1:]) for k in z.files if k.startswith("c")) + 1
    dets = _dc_merge([(z[f"c{t}"], z[f"s{t}"]) for t in range(T)], name)
    nodes = np.asarray(z["nodes"], float)
    tid = np.asarray(z["track_id"], np.int64)
    pid = np.asarray(z["parent_track_id"], np.int64)
    keep = (nodes[:, 0] >= 0) & (nodes[:, 0] < T)
    nodes, tid, pid = nodes[keep], tid[keep], pid[keep]
    order = np.lexsort((nodes[:, 0], tid))
    nodes, tid, pid = nodes[order], tid[order], pid[order]
    row_of = {(int(tid[i]), int(nodes[i, 0])): i for i in range(len(nodes))}
    edges = []
    for i in range(1, len(nodes)):
        if tid[i] == tid[i - 1] and nodes[i, 0] == nodes[i - 1, 0] + 1:
            edges.append((i - 1, i))
    first, par_of = {}, {}
    for i in range(len(nodes)):
        k = int(tid[i])
        if k not in first or nodes[i, 0] < nodes[first[k], 0]:
            first[k] = i
        if pid[i] > 0:
            par_of[k] = int(pid[i])
    n_cross = 0
    for k, i in first.items():
        p = par_of.get(k)
        if p is None:
            continue
        j = row_of.get((p, int(nodes[i, 0]) - 1))
        if j is not None:
            edges.append((j, i))
            n_cross += 1
    return dets, T, nodes, np.asarray(edges, int), n_cross



# ---- the crops ----------------------------------------------------------------------------------------------------
ZH_HP = {f.stem[len("zhdet_hp_"):]: f for f in pathlib.Path("/kaggle/input").glob("*/**/zhdet_hp_*.npz")}
assert len(ZH_HP) >= 30, f"attach faustine649/biohub-zhhpcache: {len(ZH_HP)} hpara caches found"
ZH = ZH_HP
CROPS = None
for _c in ["/kaggle/input/biohub-zebrahub-extract/crops", "/kaggle/input/notebooks/faustine649/biohub-zebrahub-extract/crops"]:
    if pathlib.Path(_c).exists():
        CROPS = pathlib.Path(_c)
        break
if CROPS is None:
    _hits = sorted(p_ for p_ in pathlib.Path("/kaggle/input").glob("*/**/crops") if p_.is_dir() and any(p_.glob("*.npz")))
    CROPS = _hits[0] if _hits else None
assert CROPS is not None, "attach faustine649/biohub-zebrahub-extract (crops)"
CROP_OF = {p_.stem: p_ for p_ in CROPS.glob("*.npz")}
NAMES = sorted(n_ for n_ in ZH_HP if n_ in CROP_OF)
print(f"crops with an hpara cache and an image: {len(NAMES)} of {len(ZH_HP)} caches / {len(CROP_OF)} crops", flush=True)
assert len(NAMES) >= 30, NAMES


def solve_zh(dets, T):
    """zhpara4's solve_one with the two lookups forkprop's fork rows need: pmap (t, i, j) -> (p, NaN, NaN) (no edge
    heads here) and seed_of_node (node id -> (t, seed index))."""
    seeds, vels, prev_of = pass1_tracks(dets)
    feats = pair_features(dets, None, T)
    probs = [MODEL.predict_proba(rows[:, MCOLS])[:, 1] if len(pairs) else np.zeros(0) for pairs, rows, P, C in feats]
    coords_rows, offset, n_glob, cd = [], {}, 0, {}
    for t in range(T):
        c, s_ = seeds[t]
        offset[t] = n_glob
        n_glob += len(c)
        for i_, xyz in enumerate(c):
            coords_rows.append([t, xyz[0], xyz[1], xyz[2]])
            cd[(t,) + tuple(int(round(float(v))) for v in xyz)] = (t, i_)
    coords = np.asarray(np.round(coords_rows), np.int16)
    pmap, cand_edges = {}, []
    for t in range(T - 1):
        pairs, rows, P, C = feats[t]
        for (i_, j_), pr in zip(pairs, probs[t]):
            pmap[(t, int(i_), int(j_))] = (float(pr), float("nan"), float("nan"))
            if pr >= ILP_FLOOR:
                cand_edges.append((offset[t] + i_, offset[t + 1] + j_, float(pr), 0.0))
    graph = PIPE.build_graph(coords, cand_edges)
    if graph.num_edges() > 0:
        solver = td.solvers.ILPSolver(edge_weight=-1.0 * td.EdgeAttr("edge_prob"), appearance_weight=0.0,
                                      disappearance_weight=ILP_DISAPP, division_weight=DIV_W)
        with PIPE.suppress_output():
            graph = solver.solve(graph)
    na = graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
    nid_l = [int(v) for v in na["node_id"].to_list()]
    remap = {v: i for i, v in enumerate(nid_l)}
    nodes_arr = np.stack([np.asarray(na[c_].to_list(), float) for c_ in ("t", "z", "y", "x")], axis=1)
    rows_e = []
    if graph.num_edges() > 0:
        ea = graph.edge_attrs(attr_keys=["source_id", "target_id"])
        for a_, b_ in zip(ea["source_id"].to_list(), ea["target_id"].to_list()):
            ra, rb = remap.get(int(a_)), remap.get(int(b_))
            if ra is not None and rb is not None:
                rows_e.append([ra, rb])
    e2 = np.array(rows_e, int) if rows_e else np.zeros((0, 2), int)
    nodes0 = {i_ + 1: {"t": int(nodes_arr[i_, 0]), "xyz": nodes_arr[i_, 1:].astype(float)} for i_ in range(len(nodes_arr))}
    edges0 = [(int(a_) + 1, int(b_) + 1) for a_, b_ in e2]
    seed_of_node = {i_ + 1: cd.get((int(nodes_arr[i_, 0]),) + tuple(int(round(v)) for v in nodes_arr[i_, 1:])) for i_ in range(len(nodes_arr))}
    cand, cand_sc, cand_trees = [], [], []
    for t in range(T):
        c_all = np.asarray(dets[t][0], float)
        s_all = np.asarray(dets[t][1], float)
        cm = (s_all >= CAND_THR) & (s_all < UNET_THRESH)
        cand.append(c_all[cm])
        cand_sc.append(s_all[cm])
        cand_trees.append(cKDTree(cand[-1] * SCALE) if len(cand[-1]) else None)
    del graph, feats, probs
    return nodes0, edges0, (cand, cand_sc, cand_trees), pmap, seed_of_node, seeds


def patch_zh(nodes0, edges0, pool):
    """forkprop's patch_final without the geff: gap-close, short filter, linefit, the orphan patch; PRE_STATE filled."""
    cand, cand_sc, cand_trees = pool
    nodes = {g: {"t": v["t"], "xyz": v["xyz"].copy()} for g, v in nodes0.items()}
    edges = list(edges0)
    nid2 = len(nodes) + 1
    nid2 = _gap_close(nodes, edges, cand, cand_sc, cand_trees, nid2)
    nodes, edges = _short_filter(nodes, edges)
    _linefit(nodes, edges)
    PRE_STATE["nodes"] = nodes
    PRE_STATE["edges"] = list(edges)
    FORK_LOG.clear()
    STEAL_LOG.clear()
    n0 = len(edges)
    _add_safe_divisions(nodes, edges)
    PRE_STATE["ids"] = sorted(nodes)
    PRE_STATE["final_edges"] = list(edges)
    return nodes, edges, len(edges) - n0


def judge_forks(nodes, edges, gt_nodes, gt_edges):
    """every fork mother of the final graph against the dense GT: 2 = the mother matches (MATCH_UM) a GT node with two
    children and the fork's children cover both GT daughters; 1 = the mother matches a GT node but not that division;
    0 = no GT node within MATCH_UM of the mother. Returns {mother id: label} and the GT division count."""
    succ = defaultdict(list)
    for a_, b_ in edges:
        succ[a_].append(b_)
    gsucc = defaultdict(list)
    for a_, b_ in gt_edges:
        gsucc[int(a_)].append(int(b_))
    gt_by_t = defaultdict(list)
    for i_, r_ in enumerate(gt_nodes):
        gt_by_t[int(r_[0])].append(i_)
    trees = {t_: (cKDTree(gt_nodes[idx_][:, 1:] * SCALE), np.asarray(idx_)) for t_, idx_ in gt_by_t.items()}
    n_gt_div = sum(1 for g_, ch in gsucc.items() if len(ch) >= 2)
    out = {}
    for p_, ch in succ.items():
        if len(ch) < 2:
            continue
        t_ = int(nodes[p_]["t"])
        tr_ = trees.get(t_)
        if tr_ is None:
            out[p_] = 0
            continue
        d_, j_ = tr_[0].query(np.asarray(nodes[p_]["xyz"], float) * SCALE)
        if d_ > MATCH_UM:
            out[p_] = 0
            continue
        g_ = int(tr_[1][j_])
        gch = gsucc.get(g_, [])
        if len(gch) < 2:
            out[p_] = 1
            continue
        covered = 0
        for gc_ in gch[:2]:
            gx = gt_nodes[gc_][1:] * SCALE
            if any(np.linalg.norm(np.asarray(nodes[c_]["xyz"], float) * SCALE - gx) <= MATCH_UM for c_ in ch):
                covered += 1
        out[p_] = 2 if covered == 2 else 1
    return out, n_gt_div


# ---- the run ------------------------------------------------------------------------------------------------------
t_all = time.time()
OUT = pathlib.Path("/kaggle/working")
COORD_HDR = ",t,z_p,y_p,x_p,z_c1,y_c1,x_c1,z_q,y_q,x_q"
fh = open(OUT / "zhfork_forks.csv", "w", encoding="utf-8", newline="\n")
fh.write("movie,fold,p,q,label," + ",".join(FEAT_NAMES) + COORD_HDR + "\n")
ALL = []
TOT = defaultdict(int)
for k_, name in enumerate(NAMES):
    if time.time() - t_all > BUDGET_S:
        print(f"[budget] stopped before {name}: {k_} crops done", flush=True)
        break
    t0 = time.time()
    dets, T, gt_nodes, gt_edges, n_cross = zh_load(name)
    nodes0, edges0, pool, pmap, seed_of_node, seeds = solve_zh(dets, T)
    z_ = np.load(CROP_OF[name])
    IMG_CTX.clear()
    IMG_CTX.update(name=name, img=np.asarray(z_["image"]))
    FP_MEMO.clear()
    FP_STAT.clear()
    nodes, edges, n_added = patch_zh(nodes0, edges0, pool)
    verdict, n_gt_div = judge_forks(nodes, edges, gt_nodes, gt_edges)
    ids = PRE_STATE["ids"]
    pos = {g: i for i, g in enumerate(ids)}
    tp_forks = [pos[g] for g, v in verdict.items() if v == 2]
    fp_forks = [pos[g] for g, v in verdict.items() if v == 1]
    scsort, best2 = _aux_extra(seeds, pmap)
    rows = fork_rows(name, k_ % 5, seeds, (pmap, seed_of_node, scsort, best2), tp_forks, fp_forks)
    for r_ in rows:
        fh.write(f"{r_[0]},{r_[1]},{r_[2]},{r_[3]},{r_[4]}," + ",".join(f"{v:.5g}" for v in r_[5])
                 + ("," + ",".join(f"{v:.6g}" for v in r_[6]) if len(r_) > 6 else "") + "\n")
    fh.flush()
    ALL.extend(rows)
    n2, n1, n0 = sum(1 for r_ in rows if r_[4] == 2), sum(1 for r_ in rows if r_[4] == 1), sum(1 for r_ in rows if r_[4] == 0)
    TOT["gt_div"] += n_gt_div
    TOT["forks"] += len(rows)
    TOT["tp"] += n2
    TOT["fp"] += n1
    TOT["unc"] += n0
    TOT["patch_added"] += n_added
    del z_, dets, nodes0, edges0, pool, pmap, nodes, edges
    IMG_CTX.clear()
    gc.collect()
    print(f"  [{k_ + 1}/{len(NAMES)}] {name}: T={T} GT divisions {n_gt_div} | forks {len(rows)} (patch added {n_added}): "
          f"TP {n2} FP {n1} uncounted {n0} | image {FI_STAT['forks']} forks, contrast crops {FP_STAT['crops']} | {time.time() - t0:.0f}s "
          f"(total {(time.time() - t_all) / 60:.1f} min)", flush=True)
fh.close()
print(f"\n=== ZEBRAHUB FORK POPULATION: {TOT['forks']:,} forks in {len(NAMES)} crops, judged TP {TOT['tp']:,} / FP {TOT['fp']:,} / "
      f"uncounted {TOT['unc']:,} (6bba: 725 counted of 18,457); GT divisions {TOT['gt_div']:,}, the pipeline forked on "
      f"{TOT['tp']:,} of them ===", flush=True)

# ---- the gate on a judged population: single columns, then the column sets, 5 folds by crop ------------------------
X_f = np.array([r_[5] for r_ in ALL], float) if ALL else np.zeros((0, len(FEAT_NAMES)))
lab = np.array([r_[4] for r_ in ALL], int)
fold_f = np.array([r_[1] for r_ in ALL], int)
counted = lab >= 1
if counted.sum() and (lab == 2).sum() >= 20 and (lab == 1).sum() >= 20:
    y_all = (lab == 2).astype(int)
    print("\n  single columns on the counted forks: AUC TP vs FP, TP / FP medians (the size columns; NaN skipped)", flush=True)
    for fn_ in FEAT_IMG:
        col = X_f[:, FEAT_NAMES.index(fn_)]
        m_ = counted & np.isfinite(col)
        if m_.sum() and (lab[m_] == 2).any() and (lab[m_] == 1).any():
            print(f"    {fn_:<11} AUC {roc_auc_score(y_all[m_], col[m_]):.3f}  TP {np.median(col[m_ & (lab == 2)]):.3g}  FP {np.median(col[m_ & (lab == 1)]):.3g}", flush=True)
    J20 = list(range(len(FEAT20)))
    JIMG11 = [FEAT_NAMES.index(f_) for f_ in FEAT_IMG[:11]]
    JSZ = {"hm": [FEAT_NAMES.index(f_) for f_ in ("r_hm_p", "r_hm_c1", "r_hm_q", "rhm_max", "rhm_sum")]}
    for m_ in SZ2:
        JSZ[m_] = [FEAT_NAMES.index(f"{m_}_{s_}") for s_ in ("p", "c1", "q", "max", "sum")]
    JRAT = JSZ["hm"][3:] + [FEAT_NAMES.index(f"{m_}_{s_}") for m_ in SZ2 for s_ in ("max", "sum")]
    JIMG = [FEAT_NAMES.index(f_) for f_ in FEAT_IMG]
    SETS = {"img31": J20 + JIMG11, "img36 (size gate)": J20 + JIMG11 + JSZ["hm"], "img33 (ratios)": J20 + JIMG11 + JSZ["hm"][3:],
            **{f"img_{m_}": J20 + JIMG11 + JSZ[m_] for m_ in SZ2}, "img_ratios": J20 + JIMG11 + JRAT, "imgall (61)": J20 + JIMG,
            "gate20": J20}
    print(f"\n=== THE GATE ON A JUDGED POPULATION: {int(counted.sum()):,} counted forks ({int((lab == 2).sum()):,} TP), HistGB "
          f"(100 it, lr 0.05, msl 10) out of fold by crop (5 folds), cut p > 0.03 and at the cut that keeps 90% of the TP ===", flush=True)
    print(f"    {'columns':<20} {'AUC':>6} {'TP kept/TP':>12} {'FP kept/FP':>14} {'FP kept at 90% TP':>18}", flush=True)
    for nm_, cols in SETS.items():
        p_ = np.full(len(lab), np.nan)
        for f_ in sorted(set(fold_f.tolist())):
            tr_ = counted & (fold_f != f_)
            te_ = fold_f == f_
            if tr_.sum() < 50 or (lab[tr_] == 2).sum() < 5:
                continue
            mg = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, min_samples_leaf=10, random_state=0)
            mg.fit(X_f[tr_][:, cols], (lab[tr_] == 2).astype(int))
            p_[te_] = mg.predict_proba(X_f[te_][:, cols])[:, 1]
        ok_ = counted & np.isfinite(p_)
        if not ok_.sum() or not (lab[ok_] == 2).any() or not (lab[ok_] == 1).any():
            continue
        auc_ = roc_auc_score(y_all[ok_], p_[ok_])
        keep = ok_ & (p_ > 0.03)
        tp_all, fp_all = int((ok_ & (lab == 2)).sum()), int((ok_ & (lab == 1)).sum())
        thr90 = float(np.quantile(p_[ok_ & (lab == 2)], 0.10))
        fp90 = int((ok_ & (lab == 1) & (p_ > thr90)).sum())
        print(f"    {nm_:<20} {auc_:>6.3f} {int((keep & (lab == 2)).sum()):>5}/{tp_all:<6} {int((keep & (lab == 1)).sum()):>6}/{fp_all:<7} {fp90:>10} (p > {thr90:.4f})", flush=True)
    print("    reading: a column set earns a place only if it cuts distinctly more false forks at the same true-fork retention; "
          "the domain shift (zebrafish nuclei, the detector out of sample) makes the absolute size values suspect, the "
          "comparison between column sets on this one population is the controlled read.", flush=True)
else:
    print("  too few judged forks for a gate reading", flush=True)
warning_summary()
print(f"ZHFORK DONE in {(time.time() - t_all) / 60:.1f} min", flush=True)
print("ZHFORK OK", flush=True)
