"""The seven post-link constants nobody has ever touched, in THREE domains.

SCRIPT_VERSION 2026-08-28j-accdiag

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

SCRIPT_VERSION = "2026-09-08a-acchp"
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
# ---- the LEARNED bridge gate -------------------------------------------
# Refuse a bridge whose two-frame pair probability is below this. 0.0
# accepts everything, so cap20 stays byte-identical to span4's.
GAP_RANK_TAU = 0.0
BRIDGE_STRIDE = 2
# The SAME 18 columns the shipped D187 ranker uses: poolsize trains with
# COLS = list(range(NFEAT_BASE)) and NFEAT_BASE is 18, so the five
# acceleration columns are computed and discarded there too. Bound here
# as a literal because NFEAT_BASE is defined ~800 lines below; the assert
# next to that definition is what keeps the two in step.
BRIDGE_COLS = list(range(18))
# Tolerance for matching a seed to its graph node. NOT 1e-6: the graph
# stores np.round(...).astype(np.int16) of the seed position. See
# _seed_to_node.
SEED_MATCH_VOX = 1.0
BRIDGE_CLF = None          # set by STAGE A
BRIDGE_P = {}              # (ge, gs) -> probability, set per dataset
_BRIDGE_MISS = [0, 0]      # pairs looked up, pairs with no probability
_BRIDGE_MAP = [0, 0]       # seeds seen, seeds mapped to a graph node
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
ARM_CACHES = {"F4": {}, "F4HP": {}}
stack = [BASE]
while stack:
    d = stack.pop()
    if d.name.endswith((".zarr", ".geff")) \
            or "cell-tracking-during-development" in d.name:
        continue
    for _arm, _pfx in (("F4HP", "fdet_F4hp_"), ("F4", "fdet_F4_")):
        for f in d.glob(_pfx + "*.npz"):
            ARM_CACHES[_arm][f.stem[len(_pfx):]] = f
    if len(d.parts) - len(BASE.parts) < 4:
        stack += [p for p in d.iterdir() if p.is_dir()]
# "fdet_F4_" needs an underscore where "fdet_F4hp_" has an 'h', so the two
# globs cannot overlap -- asserted rather than trusted, because a leak would
# put hpara rows silently into the control arm.
assert not set(ARM_CACHES["F4"]) & set("hp_" + k for k in ARM_CACHES["F4HP"])
for _arm in ("F4", "F4HP"):
    print(f"{_arm} detection caches: {len(ARM_CACHES[_arm])}", flush=True)
CACHES = ARM_CACHES["F4"]
assert len(ARM_CACHES["F4"]) >= 12 and len(ARM_CACHES["F4HP"]) >= 12, (
    "attach biohub-hparacache for fdet_F4hp_ and biohub-fuse6bba / "
    "biohub-fusecache for fdet_F4_")


def load_dets(name, arm="F4"):
    z = np.load(ARM_CACHES[arm][name])
    T = max(int(k[1:]) for k in z.files if k.startswith("c")) + 1
    return [(z[f"c{t}"], z[f"s{t}"]) for t in range(T)]


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

                # ---- the LEARNED gate ---------------------------------
                # gap3 tried four hand-made versions of this question --
                # candidate required, velocity cosine, fragment length,
                # clearance radius -- and gap4 showed all four were
                # rationing a too-small budget rather than selecting good
                # bridges. This asks the same question of a model trained
                # on ground truth. Neutral at 0.0.
                if GAP_RANK_TAU > 0.0:
                    _BRIDGE_MISS[0] += 1
                    _p = BRIDGE_P.get((ge, gs))
                    if _p is None:
                        # No probability for this pair: a seed failed to
                        # map to a graph node. COUNTED, because a high
                        # miss rate would make the arm meaningless rather
                        # than merely strict.
                        _BRIDGE_MISS[1] += 1
                        continue
                    if _p < GAP_RANK_TAU:
                        continue

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

ACC_DEMEAN = False        # set per arm below
NFEAT_BASE = 18
assert BRIDGE_COLS == list(range(NFEAT_BASE)), (
    "BRIDGE_COLS is bound as a literal far above this line; if NFEAT_BASE "
    "ever changes, the two-frame ranker would silently train on a "
    "different column set than the shipped one uses")
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
        # ---- the global-motion correction under test ----------------
        # Nothing in the shipped pipeline subtracts stage motion. A
        # CHANGING drift shows up as the same spurious acceleration on
        # every cell in the frame at once, so the per-frame MEDIAN of the
        # acceleration vectors -- over the cells that actually have three
        # positions -- is an estimate of it. Subtracting it leaves the
        # per-cell residual.
        if ACC_DEMEAN:
            _ok = acc_n >= 2
            if _ok.sum() >= 5:
                acc = acc - np.median(acc[_ok], axis=0)[None, :]
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


# [acchp] STAGE A/B, the D187 ranker and the 44b6 sample are cut here:
# none of them feeds the diagnostic and all need caches this run
# does not mount. See the maker's docstring.

# ======================= the diagnostic ================================
MATCH_UM = 5.0            # poolsize's label matcher, unchanged
ACC_NAMES = ["acc_len", "acc_cos", "md_acc", "md-md_acc", "acc_hist"]

print(f"{chr(10)}=== IS ACCELERATION MEASURING CELLS OR THE STAGE? ===",
      flush=True)
print("  Two errors in progress.md's account of biohub-accel, both found "
      "by the user's question. (1) The zeros ARE labelled -- acc_hist is "
      "the fifth column and the ACC arm used all 23, so the model had "
      "the indicator and still lost. (2) accel_no_parent counts SEEDS "
      "per FRAME and was compared against a count of candidate PAIRS, so "
      "'most candidates cannot produce three' was never measured.",
      flush=True)
print("  Nothing subtracts global stage motion. Constant drift is "
      "absorbed by the velocity column; CHANGING drift appears as the "
      "same spurious acceleration on every cell at once.", flush=True)
print(f"  12 6bba holdout datasets, labels exactly as poolsize builds "
      f"them (MATCH_UM {MATCH_UM}, positive when j matches a GT CHILD of "
      f"i's GT node).", flush=True)

_names = [n for n in HOLDOUT
          if n in ARM_CACHES["F4"] and n in ARM_CACHES["F4HP"]]
print(f"  datasets in BOTH arms: {len(_names)}", flush=True)
assert len(_names) == len(HOLDOUT), (
    f"only {len(_names)} of {len(HOLDOUT)} holdout movies have both a "
    f"fdet_F4_ and a fdet_F4hp_ cache -- the arms would not be paired")
RESULTS = {}
for ARM in ("F4", "F4HP"):
  print(f"{chr(10)}########## ARM {ARM} ##########", flush=True)

  _hist = [0, 0, 0]
  _glob, _cell = [], []
  X_raw, X_dem, Y = [], [], []
  _t0 = time.time()

  for _k, _name in enumerate(_names):
      _dets = load_dets(_name, ARM)
      _T = len(_dets)
      # ---- 1 & 2: how often is it computable, and how big is the stage? --
      _seeds, _v, _prev = pass1_tracks(_dets)
      for _t in range(_T):
          _P, _ = _seeds[_t]
          if len(_P) == 0:
              continue
          _a = np.zeros((len(_P), 3))
          _n = np.zeros(len(_P))
          for _i in range(len(_P)):
              _a[_i], _n[_i] = accel_of(_seeds, _prev, _t, _i)
          for _c in (0, 1, 2):
              _hist[_c] += int((_n == _c).sum())
          _ok = _n >= 2
          if _ok.sum() >= 5:
              _med = np.median(_a[_ok], axis=0)
              _glob.append(float(np.linalg.norm(_med)))
              _cell.append(float(np.median(
                  np.linalg.norm(_a[_ok] - _med[None, :], axis=1))))

      # ---- 3: labelled rows, raw and de-meaned --------------------------
      _gt = read_geff(geff_path(root, "train", _name))
      _gpos = np.stack([np.asarray(_gt[c], float)
                        for c in ("z", "y", "x")], axis=1) * SCALE
      _gtt = np.asarray(_gt["t"], int)
      _row_of = {int(v): i_ for i_, v in enumerate(_gt["node_ids"])}
      _child = {}
      for _a_, _b_ in _gt["edges"]:
          _ra, _rb = _row_of.get(int(_a_)), _row_of.get(int(_b_))
          if _ra is not None and _rb is not None:
              _child.setdefault(_ra, []).append(_rb)
      _by_t = defaultdict(list)
      for _i in range(len(_gtt)):
          _by_t[int(_gtt[_i])].append(_i)

      def _match(pos_u, idx_list):
          if not idx_list:
              return {}
          _tr = cKDTree(_gpos[idx_list])
          _m = {}
          for _di, _x in enumerate(pos_u):
              _dd, _gi = _tr.query(_x)
              if _dd <= MATCH_UM:
                  _m[_di] = idx_list[int(_gi)]
          return _m

      globals()["ACC_DEMEAN"] = False
      _fr = pair_features(_dets, None, _T)
      globals()["ACC_DEMEAN"] = True
      _fd = pair_features(_dets, None, _T)
      globals()["ACC_DEMEAN"] = False

      for _t in range(_T - 1):
          _pairs, _rr, _P, _C = _fr[_t]
          _pairs2, _rd, _, _ = _fd[_t]
          if not _pairs:
              continue
          assert _pairs == _pairs2, "the two passes disagree on the pairs"
          _mp = _match(_P * SCALE[None, :], _by_t.get(_t, []))
          _mc = _match(_C * SCALE[None, :], _by_t.get(_t + 1, []))
          _det_of = {_g: _d for _d, _g in _mc.items()}
          for _idx, (_i, _j) in enumerate(_pairs):
              if _i not in _mp:
                  continue
              _kids = _child.get(_mp[_i], [])
              if not _kids:
                  continue
              _jt = {_det_of.get(_b) for _b in _kids}
              _jt.discard(None)
              X_raw.append(_rr[_idx])
              X_dem.append(_rd[_idx])
              Y.append(1 if _j in _jt else 0)
      del _dets, _fr, _fd, _seeds
      gc.collect()
      print(f"  {_k + 1}/{len(_names)} {_name} ({time.time() - _t0:.0f}s, "
            f"{len(Y):,} rows)", flush=True)

  X_raw = np.stack(X_raw).astype(np.float32)
  X_dem = np.stack(X_dem).astype(np.float32)
  Y = np.asarray(Y)

  print(f"{chr(10)}=== 1. HOW OFTEN IS ACCELERATION COMPUTABLE? ===",
        flush=True)
  _tot = max(1, sum(_hist))
  print(f"  seeds {_tot:,}   acc_hist 0 (no backward link) "
        f"{_hist[0]:,} ({100.0 * _hist[0] / _tot:.1f}%)   "
        f"1 (one step) {_hist[1]:,} ({100.0 * _hist[1] / _tot:.1f}%)   "
        f"2 (FULL) {_hist[2]:,} ({100.0 * _hist[2] / _tot:.1f}%)", flush=True)
  print("  This is the number progress.md never had: it compared a "
        "per-seed counter against a per-pair total.", flush=True)

  print(f"{chr(10)}=== 2. STAGE MOTION AGAINST CELL MOTION ===", flush=True)
  if _glob:
      _g = np.asarray(_glob)
      _c = np.asarray(_cell)
      print(f"  frames with >=5 usable seeds: {len(_g):,}", flush=True)
      print(f"  |median acceleration| per frame  (THE STAGE): "
            f"median {np.median(_g):.4f}  p90 {np.percentile(_g, 90):.4f} "
            f"um/frame^2", flush=True)
      print(f"  median per-cell |acc - median|   (THE CELLS): "
            f"median {np.median(_c):.4f}  p90 {np.percentile(_c, 90):.4f}",
            flush=True)
      print(f"  RATIO stage/cell: {np.median(_g) / max(1e-9, np.median(_c)):.3f}"
            f"   -- near or above 1 means this column is mostly the stage",
            flush=True)
  else:
      print("  no frame had five usable seeds", flush=True)

  print(f"{chr(10)}=== 3. SINGLE-COLUMN AUC, RAW vs DE-MEANED ===", flush=True)
  print(f"  {len(Y):,} labelled pairs, {Y.mean():.4f} positive. Below 0.5 "
        f"means informative with the sign flipped; 0.5 means nothing.",
        flush=True)
  print(f"  {'col':<12} {'raw':>8} {'de-meaned':>11} {'|delta|':>9}",
        flush=True)
  for _o, _nm in enumerate(ACC_NAMES):
      _c = NFEAT_BASE + _o
      try:
          _ar = roc_auc_score(Y, X_raw[:, _c])
          _ad = roc_auc_score(Y, X_dem[:, _c])
      except Exception:                                          # noqa: BLE001
          _ar = _ad = float("nan")
      print(f"  {_nm:<12} {_ar:>8.4f} {_ad:>11.4f} "
            f"{abs(_ad - 0.5) - abs(_ar - 0.5):>+9.4f}", flush=True)
  print(f"  for scale, the strongest BASE columns: "
        f"d {roc_auc_score(Y, X_raw[:, 0]):.4f}   "
        f"md {roc_auc_score(Y, X_raw[:, 4]):.4f}   "
        f"score_c {roc_auc_score(Y, X_raw[:, 7]):.4f}", flush=True)
  RESULTS[ARM] = dict(
      n=len(Y), pos=float(Y.mean()),
      d=roc_auc_score(Y, X_raw[:, 0]), md=roc_auc_score(Y, X_raw[:, 4]),
      md_acc=roc_auc_score(Y, X_raw[:, NFEAT_BASE + 2]),
      score_c=roc_auc_score(Y, X_raw[:, 7]),
      stage=float(np.median(_glob)) if _glob else float("nan"),
      cell=float(np.median(_cell)) if _cell else float("nan"))
  del X_raw, X_dem, Y
  gc.collect()

print(f"{chr(10)}=== F4 (refine) vs F4HP (hpara), SAME movies, SAME "
      f"labels, ONLY the placement differs ===", flush=True)
_r, _h = RESULTS.get("F4"), RESULTS.get("F4HP")
if _r and _h:
    print(f"  labelled rows: F4 {_r['n']:,} ({_r['pos']:.4f} pos), "
          f"F4HP {_h['n']:,} ({_h['pos']:.4f} pos)", flush=True)
    _drift = abs(_h["n"] - _r["n"]) / max(1, _r["n"])
    print(f"  row-count drift {100 * _drift:.3f}%  "
          + ("-- hpara adds no detections, so this should be a few tenths "
             "of a percent at most; more means the LABEL MATCHER moved and "
             "the columns below are not comparable"
             if _drift > 0.01 else "-- as expected"), flush=True)
    print(f"{chr(10)}  {'column':<10} {'F4 refine':>10} {'F4HP hpara':>11} "
          f"{'delta':>9}   note", flush=True)
    for _c, _note in (("d", "raw distance to the candidate"),
                      ("md", "distance from the CONSTANT-VELOCITY prediction"),
                      ("md_acc", "distance from the ACCELERATION-corrected one"),
                      ("score_c", "detection score of the candidate")):
        print(f"  {_c:<10} {_r[_c]:>10.4f} {_h[_c]:>11.4f} "
              f"{_h[_c] - _r[_c]:>+9.4f}   {_note}", flush=True)
    print(f"{chr(10)}  PREDICTED BEFORE THE RUN: md_acc stays below md in "
          f"BOTH arms. The variance ratio 10.50/5.00 does not contain "
          f"sigma, so a smoother position shrinks both predictors equally "
          f"and cannot reorder them. A reversal falsifies that argument.",
          flush=True)
    print(f"  THE COLUMN THAT MATTERS MORE: `d` and `md`. Flat there means "
          f"the placement gain never reaches the features at all, which "
          f"would explain hparabench's null without involving the ranker.",
          flush=True)
    print(f"  stage/cell acceleration ratio: F4 "
          f"{_r['stage'] / max(1e-9, _r['cell']):.3f}  F4HP "
          f"{_h['stage'] / max(1e-9, _h['cell']):.3f}", flush=True)
else:
    print(f"  [degraded] only {sorted(RESULTS)} completed -- no comparison",
          flush=True)

print(f"{chr(10)}=== reading it ===", flush=True)
print("  A large ratio in (2) with an improvement in (3) means the "
      "column was measuring the stage and there is a one-line fix. A "
      "flat (3) either way closes the line on its own evidence rather "
      "than on the wrong note in progress.md.", flush=True)
print("  ⚠ biohub-accel measured permutation importance +0.0000 to "
      "+0.0017 across all five. Even a clean feature starts from "
      "approximately nothing.", flush=True)
warning_summary()
print("ACCHP OK" if len(RESULTS) == 2 else f"[degraded] ACCHP INCOMPLETE: {sorted(RESULTS)}", flush=True)
