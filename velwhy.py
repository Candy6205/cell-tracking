"""velwhy -- is the motion signal under-weighted, or is the velocity missing?

SCRIPT_VERSION 2026-08-29a-edgediag

NO SWEEP. One configuration -- the shipped cap20 + SHORT_MIN 4 (ref
55817630, LB 0.922) -- run through the full harness in three domains,
with every GT edge charged to the FIRST pipeline stage that loses it
(detection sub-threshold / detection cold / outside the 10 um gate /
below the ranker floor / rejected by the ILP / deleted post-link) and
every FINAL edge classified against GT (wrong link / over-detection
half-edge / invisible). The official scorer runs on the same geffs as
the anchor and cross-check.

WHY. The last attribution of this kind (error_analysis.py, mid-August,
classical pipeline) opened 10,313 GT edges, found 60% DROPPED, and
bought the ILP -- the single largest gain this project has had. The
pipeline has since been rebuilt end to end and nobody has re-asked the
question. The 2026-08-24 budget (detection ~1.75%, wrong links ~1.6%,
false edges ~3.3%) predates the ranker, cap20 and the 7.0 um ruler
correction.

Assembly: dcpool.py's harness verbatim (loader, ranker, solve, patches,
zh machinery, manifest-derived 44b6 set); the sweep replaced by the
attribution pass. Anchor: 6bba adj_J 0.9293 via the official scorer.
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

SCRIPT_VERSION = "2026-09-02c-velwhy"
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


# dccache3's epoch-5 detections. The bounded walk prunes the competition
# and zarr/geff trees (trap 20).
def _walk_npz(prefix):
    import os
    hits = {}
    for root_, dirs, fnames in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs
                   if "cell-tracking-during-development" not in d
                   and not d.endswith(".zarr") and not d.endswith(".geff")]
        for fn in fnames:
            if fn.startswith(prefix) and fn.endswith(".npz"):
                hits.setdefault(fn[len(prefix):-len(".npz")],
                                pathlib.Path(root_) / fn)
    return hits


DCDET = _walk_npz("dcdet_ep05_")


def _dc_pool_aug(pool, dets, name):
    """The REPAIR POOL with DeepCenter's epoch-5 peaks added, primaries
    untouched. Read by `_gap_close` only, so DC_TAU is a POST-LINK
    constant here (a constant belongs to the stage that reads it).

    A DC peak enters the pool when its own score is >= DC_TAU and it is
    further than DC_MERGE_UM from every PRIMARY node and every existing
    pool candidate at that frame -- add-only, no reordering, no joint
    NMS. Added points carry DC_ADD_SCORE (0.12): snap-eligible, never
    primary. Returns (cand, cand_sc, cand_trees, is_dc)."""
    cand, cand_sc, _trees = pool
    if DC_TAU is None or name not in DCDET:
        trees = [cKDTree(c * SCALE) if len(c) else None for c in cand]
        return (cand, cand_sc, trees,
                [np.zeros(len(c), bool) for c in cand])
    z = np.load(DCDET[name])
    c2, s2, is_dc = [], [], []
    for t in range(len(cand)):
        c = np.asarray(cand[t], float).reshape(-1, 3)
        sc = np.asarray(cand_sc[t], float).reshape(-1)
        kc = f"c{t}"
        prim, _ps = seeds_of(dets[t])
        _DC_STATS[0] += 1
        _DC_STATS[1] += len(c)
        dc = np.zeros((0, 3))
        if kc in z.files:
            dc = np.asarray(z[kc], float).reshape(-1, 3)
            ds = np.asarray(z[f"s{t}"], float).reshape(-1)
            dc = dc[ds >= DC_TAU]
        if len(dc) and len(prim):
            d, _ = cKDTree(prim * SCALE[None, :]).query(dc * SCALE[None, :])
            dc = dc[d > DC_MERGE_UM]
        if len(dc) and len(c):
            d, _ = cKDTree(c * SCALE[None, :]).query(dc * SCALE[None, :])
            dc = dc[d > DC_MERGE_UM]
        _DC_STATS[2] += len(dc)
        if len(dc):
            c2.append(np.concatenate([c, dc], 0))
            s2.append(np.concatenate([sc, np.full(len(dc), DC_ADD_SCORE)], 0))
            is_dc.append(np.concatenate([np.zeros(len(c), bool),
                                         np.ones(len(dc), bool)]))
        else:
            c2.append(c)
            s2.append(sc)
            is_dc.append(np.zeros(len(c), bool))
    trees = [cKDTree(c * SCALE) if len(c) else None for c in c2]
    return c2, s2, trees, is_dc


def _walk_npz(prefix):
    """mergescore's bounded walk (trap 20: prune the competition tree and
    the zarr/geff trees or this takes minutes)."""
    import os
    hits = {}
    for root_, dirs, fnames in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs
                   if "cell-tracking-during-development" not in d
                   and not d.endswith(".zarr") and not d.endswith(".geff")]
        for fn in fnames:
            if fn.startswith(prefix) and fn.endswith(".npz"):
                hits.setdefault(fn[len(prefix):-len(".npz")],
                                pathlib.Path(root_) / fn)
    return hits


FDET_F4 = _walk_npz("fdet_F4_")
print(f"fdet_F4 caches: {len(FDET_F4)}", flush=True)
assert len(FDET_F4) >= 24, "attach biohub-fusecache for the F4 detections"


def load_dets(name, arm="BASE"):
    """BASE = the xl three-branch average that shipped at 0.922.
    F4 = the same with ep05 DeepCenter as a fourth equal-weight map,
    which is what ships at 0.925 (ref 55889500)."""
    z = np.load(CACHES[name] if arm == "BASE" else FDET_F4[name])
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


# ---- the three evaluation domains --------------------------------------
# 6bba is poolsize's holdout verbatim. 44b6 is chosen AT RUNTIME from
# whatever caches are mounted, sorted, so no dataset name is invented
# here and a missing mount fails loudly instead of silently shrinking the
# comparison.
#
# ⚠ 2026-08-27, THE USER'S STANDING INSTRUCTION: "所有测试集的 44b6 都要
# 随机抽". Every sweep before this one used `sorted(...)[:12]` -- the
# twelve LOWEST hex ids out of 71, identical in every run, and nobody
# ever checked whether those ids encode acquisition order, embryo or
# session. An arbitrary-looking sort key is not a random sample, and a
# fixed prefix of it is the worst case: if the ids carry any structure,
# every three-domain verdict this project has issued shares one skew.
#
# Random, but SEEDED, so the sample is reproducible and a recorded
# anchor still means something. The chosen names are PRINTED every run
# -- a sample nobody can see is a sample nobody can check.
#
# To test whether a result depends on the sample, change SEED_44B6 and
# re-run; that is a different question from changing N_44B6, which
# changes the precision.
# The 44b6 arm is dctrain's EXCLUDED 12, read from the training run's
# own split_manifest.json rather than recomputed -- if the training
# split ever changes, this fails loudly instead of silently evaluating
# on movies the model saw. These 12 were drawn seeded (20260827) from
# the 44b6 outside D32's pool, satisfying both the user's randomness
# rule and the ranker-leak rule. The epoch-5 model never saw them, so
# **this is the first 44b6 arm in the DeepCenter line that means
# anything, and the first time this line can face the three-domain
# rule.**
import json as _json                                           # noqa: E402
_MAN = None
import os as _os                                               # noqa: E402
for _root, _dirs, _fns in _os.walk("/kaggle/input"):
    _dirs[:] = [d for d in _dirs
                if "cell-tracking-during-development" not in d
                and not d.endswith(".zarr") and not d.endswith(".geff")]
    if "split_manifest.json" in _fns:
        _MAN = pathlib.Path(_root) / "split_manifest.json"
        break
assert _MAN is not None, ("split_manifest.json not found -- attach "
                          "faustine649/biohub-dctrain-175-weights")
_seen = set(_json.loads(_MAN.read_text())["all"])
SET_6BBA = [n for n in HOLDOUT if n in CACHES]
SET_44B6 = sorted(n for n in CACHES
                  if n.startswith("44b6_") and n not in _seen)
print(f"  44b6: {len(SET_44B6)} = dctrain's excluded set, from {_MAN}",
      flush=True)
print(f"    {' '.join(SET_44B6)}", flush=True)
assert len(SET_6BBA) == len(HOLDOUT), (
    f"only {len(SET_6BBA)} of {len(HOLDOUT)} holdout caches are mounted; "
    f"the 6bba row would not be comparable to the recorded anchor")
assert len(SET_44B6) == 12, (
    f"expected 12 excluded 44b6, found {len(SET_44B6)} -- the manifest "
    f"and the caches disagree")
assert all(n not in _seen for n in SET_6BBA), (
    "a 6bba holdout dataset appears in dctrain's training pool")
print(f"  domains: 6bba {len(SET_6BBA)}, 44b6 {len(SET_44B6)}", flush=True)

ZH = _walk_npz("zhdet_")
N_ZH = 8
SET_ZH = sorted(ZH)[:N_ZH]
print("  zebrahub is BACK. wide44 dropped it because its question was entirely about the 44b6 sample; this run asks whether a wider bridge pays, which is a candidate change and therefore a three-domain question.", flush=True)
# The inherited WARN here said "biohub-zh-detect is not attached". That
# would be a FALSE diagnosis in this run -- the mount is fine and the
# domain is dropped by choice. Report the reason that is true.
if not SET_ZH:
    print("  WARN biohub-zh-detect is not attached -> the zebrahub domain "
          "is ABSENT and the three-domain rule degrades to two. Say so in "
          "the write-up rather than reporting a two-domain result as if "
          "it satisfied the rule.", flush=True)
else:
    print(f"  domains: zebrahub {len(SET_ZH)}", flush=True)


def zh_load(name):
    """(dets, T, gt_nodes, gt_edges, n_cross) for one Zebrahub crop.
    Carried over from zh_pipeline.py unchanged."""
    z = np.load(ZH[name])
    T = max(int(k[1:]) for k in z.files if k.startswith("c")) + 1
    dets = [(z[f"c{t}"], z[f"s{t}"]) for t in range(T)]
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


# ---- the sweep -----------------------------------------------------------
# One value changed at a time from the shipped configuration. Written down
# before the run; nothing is added afterwards.
print("\n=== EDGE ERROR ATTRIBUTION, shipped configuration, 3 domains "
      "===", flush=True)
print("  No sweep. One config: cap20 + SHORT_MIN 4, ref 55817630's. Every "
      "GT edge is walked down the pipeline and charged to the FIRST stage "
      "that loses it; every final edge is classified against GT. The last "
      "attribution of this kind (error_analysis.py, August, classical "
      "pipeline) found 60% DROPPED and bought the ILP. Nothing has asked "
      "since where the ~7 points of edge_J now go.", flush=True)
print("  ANCHOR: 6bba BASE adj_J must reproduce 0.9293 via the official "
      "scorer (dcpool measured it with this exact machinery).", flush=True)
print("  ⚠ 44b6 caveat: the D187 ranker trained on all 71 44b6, so its "
      "probs there are in-sample-optimistic -> BELOW_FLOOR is "
      "UNDERCOUNTED in that column. 6bba holdout and zh are outside its "
      "pool and clean.", flush=True)

res = {}
PER = {}   # (cfg, domain) -> {dataset: row}
BASE_VALUES = {k: globals()[k] for k in
               ("GAP1_GATE_UM", "GAP1_SNAP_UM", "GAP1_MIN_CAND_SCORE",
                "GAP1_CAP_FRAC", "SHORT_MIN", "LINEFIT_WEIGHT",
                "LINEFIT_WINDOW", "GAP_SPAN", "GAP_NEED_CAND",
                "GAP_VEL_COS", "GAP_MIN_LEN", "GAP_CLEAR_UM",
                "GAP_LEN_MODE", "DC_TAU")}


# ⚠ In dcmerge DC_TAU sat in this set because `_dc_merge` ran inside
# `load_dets`. In THIS file the DC augmentation moved to the pool stage
# (`_dc_pool_aug`, called inside the arm loop), so DC_TAU is a POST-LINK
# constant here and is deliberately NOT in the set. The rule that
# decides membership: find the line that READS the constant and check
# whether it runs inside or outside the arm loop.
DETECTION_TIME_CONSTANTS = {"UNET_THRESH", "CAND_THR", "NMS_UM"}


def apply_config(key, val):
    """Rebind swept constants, or restore all of them for BASE.

    A CONSTANT BELONGS TO THE STAGE THAT READS IT. Everything this
    sweep varies (DC_TAU included -- see the note above) is read by the
    post-link stages, which run inside the arm loop; `load_dets` and
    `solve_one` read none of it, so the solve is shared across arms.

    The patch functions read these as module globals, so this is what
    "changing one thing" means here.

    `key` may be a str (one constant) or a DICT (several at once).
    """
    for k, v in BASE_VALUES.items():
        globals()[k] = v
    if isinstance(key, dict):
        for k, v in key.items():
            assert k in BASE_VALUES, f"{k} is not a swept constant"
            globals()[k] = v
    elif key is not None:
        assert key in BASE_VALUES, f"{key} is not a swept constant"
        globals()[key] = val


def solve_one(dets, T):
    """Everything up to and including the ILP: features, ranker, graph,
    solve, and the sub-threshold repair pool. NONE of it depends on a
    constant being swept, so it runs ONCE per dataset and every
    configuration is applied to a copy of its output."""
    seeds, vels, prev_of = pass1_tracks(dets)
    feats = pair_features(dets, None, T)
    probs = [MODEL.predict_proba(rows[:, MCOLS])[:, 1]
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
            if pr >= ILP_FLOOR:
                cand_edges.append((offset[t] + i_, offset[t + 1] + j_,
                                   float(pr), 0.0))
    graph = PIPE.build_graph(coords, cand_edges)
    if graph.num_edges() > 0:
        solver = td.solvers.ILPSolver(
            edge_weight=-1.0 * td.EdgeAttr("edge_prob"),
            appearance_weight=0.0,
            disappearance_weight=ILP_DISAPP,
            division_weight=DIV_W)
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
    del graph, feats, probs
    return nodes0, edges0, (cand, cand_sc, cand_trees)


def patch_and_save(nodes0, edges0, pool, out_path):
    """The four post-link stages on a FRESH COPY, then write the geff."""
    cand, cand_sc, cand_trees = pool
    nodes = {g: {"t": v["t"], "xyz": v["xyz"].copy()}
             for g, v in nodes0.items()}
    edges = list(edges0)
    nid2 = len(nodes) + 1
    nid2 = _gap_close(nodes, edges, cand, cand_sc, cand_trees, nid2)
    nodes, edges = _short_filter(nodes, edges)
    _linefit(nodes, edges)
    _add_safe_divisions(nodes, edges)
    ids = sorted(nodes)
    rm = {g: i for i, g in enumerate(ids)}
    arr = np.array([[nodes[g]["t"], *nodes[g]["xyz"]] for g in ids], float)
    e3 = np.array([[rm[a_], rm[b_]] for a_, b_ in edges], int)
    dr._save(str(out_path), arr, e3)


def pdir(cfg, domain):
    d = pathlib.Path(f"/kaggle/working/ps_{domain}_{cfg}")
    d.mkdir(parents=True, exist_ok=True)
    return d


res = {}
PER2 = {}
t_all = time.time()


def patch_final(nodes0, edges0, pool, out_path):
    """patch_and_save, with the final graph RETURNED for attribution."""
    cand, cand_sc, cand_trees = pool
    nodes = {g: {"t": v["t"], "xyz": v["xyz"].copy()}
             for g, v in nodes0.items()}
    edges = list(edges0)
    nid2 = len(nodes) + 1
    nid2 = _gap_close(nodes, edges, cand, cand_sc, cand_trees, nid2)
    nodes, edges = _short_filter(nodes, edges)
    _linefit(nodes, edges)
    _add_safe_divisions(nodes, edges)
    ids = sorted(nodes)
    rm = {g: i for i, g in enumerate(ids)}
    arr = np.array([[nodes[g]["t"], *nodes[g]["xyz"]] for g in ids], float)
    e3 = np.array([[rm[a_], rm[b_]] for a_, b_ in edges], int)
    dr._save(str(out_path), arr, e3)
    return nodes, edges


MATCH_R = 7.0   # the scorer's constant, NOT the 5.0 the old diagnostics used


def _greedy_match(gt_pts_um, det_pts_um):
    """1:1, nearest pair first, within MATCH_R. {gt_local_idx: det_idx}."""
    out = {}
    if len(gt_pts_um) == 0 or len(det_pts_um) == 0:
        return out
    tree = cKDTree(det_pts_um)
    cand_pairs = []
    for gi, x in enumerate(gt_pts_um):
        for dj in tree.query_ball_point(x, MATCH_R):
            cand_pairs.append((float(np.linalg.norm(det_pts_um[dj] - x)),
                               gi, dj))
    cand_pairs.sort()
    used_g, used_d = set(), set()
    for _d, gi, dj in cand_pairs:
        if gi in used_g or dj in used_d:
            continue
        out[gi] = dj
        used_g.add(gi)
        used_d.add(dj)
    return out


# FN side: the FIRST stage that loses each GT edge.
CATS = ["det_miss_subthresh", "det_miss_cold", "no_candidate",
        "below_floor", "node_absent", "ilp_rejected", "killed_post",
        "recovered"]
NAGG = {}          # domain -> node-side counters
EROWS = {}         # domain -> [array of per-GT-edge covariate rows]
RROWS = {}         # domain -> one row per GT edge whose correct pair
                   # was NOT the ranker's first choice
NROWS = {}         # domain -> one row per node_absent source
VROWS = {}         # domain -> one row per rank >= 1 case, the orderings
VCOLS = ["nv", "rank_by_d", "rank_by_md", "rank_by_prob", "d_true",
         "md_true", "d_top", "md_top", "n_cand"]
RCOLS = ["rank", "p_true", "p_top", "top_is_gt", "top_is_gt_target",
         "d_src_top", "d_src_true", "d_top_true", "sc_top", "sc_true",
         "bidders_for_top", "survived"]
NCOLS = ["best_out", "best_in", "n_cand", "bidders_for_self",
         "sc_self", "t_frac"]
FROWS = {}         # domain -> [array of per-final-edge covariate rows]
ECOLS = ["bucket", "disp_um", "dens10", "gap2_um", "z_um", "t_frac",
         "sc_src", "sc_tgt", "p_true", "rank_true", "p_top", "is_div"]
FCOLS = ["class", "len_um", "p", "prov", "dens10", "err_um"]
# FP side: every edge in the final graph, classified against GT.
FPC = ["tp_like", "fp_confusion", "fp_half", "fp_invisible"]
AGG = {}
FAGG = {}


def diag_one(domain, name, dets, T, gt_t, gt_pos, gt_edges, out_path):
    nodes0, edges0, pool = solve_one(dets, T)
    nodes_f, edges_f = patch_final(nodes0, edges0, pool, out_path)
    seeds = [seeds_of(dets[t]) for t in range(T)]
    feats = pair_features(dets, None, T)
    prob_of = {}
    # column 0 is the plain distance, column 4 the distance from the
    # velocity-PREDICTED position, column 16 the velocity magnitude --
    # the same three numbers the ranker was given.
    d_of = {}
    md_of = {}
    nv_of = {}
    for t in range(T - 1):
        pairs, rows_, _P, _C = feats[t]
        if len(pairs):
            pv = MODEL.predict_proba(rows_[:, MCOLS])[:, 1]
            for k_, ((i, j), p) in enumerate(zip(pairs, pv)):
                prob_of[(t, i, j)] = float(p)
                d_of[(t, i, j)] = float(rows_[k_, 0])
                md_of[(t, i, j)] = float(rows_[k_, 4])
                nv_of[(t, i)] = float(rows_[k_, 16])
    # ---- covariate helpers (edgewhy) ----------------------------------
    # Everything here is derived from the same seeds/probs the buckets
    # already use, so no new pass over the data and no new assumption.
    prim_tree = [cKDTree(seeds[t_][0] * SCALE[None, :])
                 if len(seeds[t_][0]) else None for t_ in range(T)]
    cand_by_src = defaultdict(list)
    for (t_, i_, j_), p_ in prob_of.items():
        cand_by_src[(t_, i_)].append((p_, j_))
    for k_ in cand_by_src:
        cand_by_src[k_].sort(reverse=True)

    def _dens(t_, pt_um, r=10.0):
        """primaries within r um -- crowding as the linker sees it."""
        tr = prim_tree[t_] if 0 <= t_ < T else None
        return float(len(tr.query_ball_point(pt_um, r))) if tr else 0.0

    def _gap2(t_, pt_um):
        """distance to the SECOND nearest primary: the nearest distractor
        once the cell's own detection is accounted for."""
        tr = prim_tree[t_] if 0 <= t_ < T else None
        if tr is None or tr.n < 2:
            return float("nan")
        d_, _ = tr.query(pt_um, k=2)
        return float(d_[1])

    # seed (t, i) -> solved-graph id, matched by rounded coordinates:
    # solve_one feeds np.round(...).astype(int16) into the graph, so the
    # solved node coords ARE the rounded seed coords.
    id_of = {}
    for g, v in nodes0.items():
        id_of[(int(v["t"]), tuple(int(round(float(x)))
                                  for x in v["xyz"]))] = g
    seed_id = {}
    for t in range(T):
        P, _s = seeds[t]
        for i in range(len(P)):
            g = id_of.get((t, tuple(int(round(float(x)))
                                    for x in P[i])))
            if g is not None:
                seed_id[(t, i)] = g
    e0set = set(edges0)
    efset = set(edges_f)
    gt_by_t = defaultdict(list)
    for r in range(len(gt_t)):
        gt_by_t[int(gt_t[r])].append(r)
    m_seed = {}
    near_pool = {}
    cand, cand_sc, cand_trees = pool
    for t, rows_ in gt_by_t.items():
        if t >= T:
            continue
        g_um = gt_pos[rows_] * SCALE[None, :]
        P, _s = seeds[t]
        mm = _greedy_match(g_um, P * SCALE[None, :]) if len(P) else {}
        tr = cand_trees[t] if t < len(cand_trees) else None
        for k, r in enumerate(rows_):
            if k in mm:
                m_seed[r] = (t, mm[k])
            near_pool[r] = bool(tr is not None
                                and tr.query(g_um[k])[0] <= MATCH_R)
    # ---- rankwhy: primaries backwards to GT, and GT targets as a set --
    seed_gt = {ti_: r_ for r_, ti_ in m_seed.items()}
    gt_is_target = set()
    for u_, v_ in gt_edges:
        if int(gt_t[int(v_)]) == int(gt_t[int(u_)]) + 1:
            gt_is_target.add(int(v_))
    # how many sources at t list a given target j at t+1 as a candidate
    bidders = defaultdict(int)
    best_in_of = {}
    for (t_, i_, j_), p_ in prob_of.items():
        bidders[(t_ + 1, j_)] += 1
        k_ = (t_ + 1, j_)
        if k_ not in best_in_of or p_ > best_in_of[k_]:
            best_in_of[k_] = float(p_)

    a = AGG.setdefault(domain, {c: 0 for c in CATS})
    a.setdefault("gt_edges", 0)
    a.setdefault("nonadjacent", 0)
    gt_out = defaultdict(list)
    for u_, v_ in gt_edges:
        if int(gt_t[int(v_)]) == int(gt_t[int(u_)]) + 1:
            gt_out[int(u_)].append(int(v_))
    erows = []
    for u, v in gt_edges:
        u, v = int(u), int(v)
        tu, tv = int(gt_t[u]), int(gt_t[v])
        a["gt_edges"] += 1
        if tv != tu + 1:
            a["nonadjacent"] += 1        # the metric discards these
            continue
        su, sv = m_seed.get(u), m_seed.get(v)
        if su is None or sv is None:
            miss_sub = (su is None and near_pool.get(u)) or \
                       (sv is None and near_pool.get(v))
            bucket = "det_miss_subthresh" if miss_sub else "det_miss_cold"
        else:
            p = prob_of.get((tu, su[1], sv[1]))
            if p is None:
                bucket = "no_candidate"
            elif p < ILP_FLOOR:
                bucket = "below_floor"
            else:
                ga, gb = seed_id.get(su), seed_id.get(sv)
                if ga is None or gb is None:
                    bucket = "node_absent"
                elif (ga, gb) not in e0set:
                    bucket = "ilp_rejected"
                elif (ga, gb) not in efset:
                    bucket = "killed_post"
                else:
                    bucket = "recovered"
        a[bucket] += 1
        pu = gt_pos[u] * SCALE
        pv = gt_pos[v] * SCALE
        p_true = np.nan
        rank_t = np.nan
        p_top = np.nan
        if su is not None:
            lst = cand_by_src.get((tu, su[1]))
            if lst:
                p_top = float(lst[0][0])
                if sv is not None:
                    for k_ in range(len(lst)):
                        if lst[k_][1] == sv[1]:
                            rank_t = float(k_)
                            p_true = float(lst[k_][0])
                            break
        # ---- what did the ranker put first instead? -------------------
        if su is not None and sv is not None and np.isfinite(rank_t) \
                and rank_t >= 1.0:
            lst2 = cand_by_src.get((tu, su[1]))
            j_top = lst2[0][1]
            tgt_gt = seed_gt.get((tv, j_top))
            P_ = seeds[tu][0]
            C_ = seeds[tv][0]
            src_um = np.asarray(P_[su[1]], float) * SCALE
            top_um = np.asarray(C_[j_top], float) * SCALE
            true_um = np.asarray(C_[sv[1]], float) * SCALE
            # where would the true target sit under two orderings the
            # ranker already has as columns?
            cl = [jj for _pp, jj in lst2]
            dd_ = sorted(cl, key=lambda jj: d_of.get((tu, su[1], jj),
                                                     1e9))
            mm_ = sorted(cl, key=lambda jj: md_of.get((tu, su[1], jj),
                                                      1e9))
            rank_d = float(dd_.index(sv[1])) if sv[1] in dd_ else np.nan
            rank_md = float(mm_.index(sv[1])) if sv[1] in mm_ else np.nan
            VROWS.setdefault(domain, []).append((
                float(nv_of.get((tu, su[1]), np.nan)),
                rank_d, rank_md, float(rank_t),
                float(d_of.get((tu, su[1], sv[1]), np.nan)),
                float(md_of.get((tu, su[1], sv[1]), np.nan)),
                float(d_of.get((tu, su[1], j_top), np.nan)),
                float(md_of.get((tu, su[1], j_top), np.nan)),
                float(len(cl))))
            RROWS.setdefault(domain, []).append((
                float(rank_t),
                float(p_true) if np.isfinite(p_true) else np.nan,
                float(p_top) if np.isfinite(p_top) else np.nan,
                1.0 if tgt_gt is not None else 0.0,
                1.0 if (tgt_gt is not None
                        and tgt_gt in gt_is_target) else 0.0,
                float(np.linalg.norm(top_um - src_um)),
                float(np.linalg.norm(true_um - src_um)),
                float(np.linalg.norm(top_um - true_um)),
                float(seeds[tv][1][j_top]),
                float(seeds[tv][1][sv[1]]),
                float(bidders.get((tv, j_top), 0)),
                1.0 if bucket == "recovered" else 0.0))
        # ---- a node the ILP dropped despite a top-ranked correct pair --
        if bucket == "node_absent" and su is not None:
            lst3 = cand_by_src.get((tu, su[1]), ())
            best_out = float(lst3[0][0]) if lst3 else np.nan
            best_in = best_in_of.get((tu, su[1]), np.nan)
            NROWS.setdefault(domain, []).append((
                best_out, best_in,
                float(len(lst3)),
                float(bidders.get((tu, su[1]), 0)),
                float(seeds[tu][1][su[1]]),
                float(tu) / max(1, T)))
        erows.append((float(CATS.index(bucket)),
                      float(np.linalg.norm(pv - pu)),
                      _dens(tu, pu),
                      _gap2(tv, pv),
                      float(pu[0]),
                      float(tu) / max(1, T),
                      (float(seeds[tu][1][su[1]]) if su is not None
                       else np.nan),
                      (float(seeds[tv][1][sv[1]]) if sv is not None
                       else np.nan),
                      p_true, rank_t, p_top,
                      1.0 if len(gt_out.get(u, ())) >= 2 else 0.0))
    if erows:
        EROWS.setdefault(domain, []).append(np.asarray(erows, np.float32))
    # ---- node-side ledger ---------------------------------------------
    # Almost everything here was already computed for the edge pass and
    # simply never reported.
    nn = NAGG.setdefault(domain, defaultdict(int))
    for t in range(T):
        s_all = np.asarray(dets[t][1], float).reshape(-1)
        nn["dets_total"] += len(s_all)
        nn["repair_band"] += int(((s_all >= CAND_THR)
                                  & (s_all < UNET_THRESH)).sum())
        nn["primaries"] += len(seeds[t][0])
    nn["in_graph"] += len(seed_id)
    _lk = set()
    for a_, b_ in edges0:
        _lk.add(a_)
        _lk.add(b_)
    _sid = set(seed_id.values())
    nn["ilp_linked"] += len(_lk & _sid)
    nn["nodes_after_ilp"] += len(nodes0)
    nn["final_nodes"] += len(nodes_f)
    nn["final_from_seed"] += len(_sid & set(nodes_f))
    nn["final_repair"] += len([g for g in nodes_f if g not in _sid])
    nn["killed_post"] += len(_sid - set(nodes_f))
    _dg = defaultdict(int)
    for a_, b_ in edges_f:
        _dg[a_] += 1
        _dg[b_] += 1
    for g in nodes_f:
        nn[f"deg{min(_dg.get(g, 0), 3)}"] += 1
    # GT-node ledger: matched by a primary, reachable only through the
    # sub-threshold repair band, or nothing there at all.
    for r in range(len(gt_t)):
        if int(gt_t[r]) >= T:
            continue
        nn["gt_nodes"] += 1
        if r in m_seed:
            nn["gt_matched_primary"] += 1
        elif near_pool.get(r):
            nn["gt_subthresh_only"] += 1
        else:
            nn["gt_cold"] += 1

    # ---- final-edge (FP-side) classification --------------------------
    f = FAGG.setdefault(domain, {c: 0 for c in FPC})
    f.setdefault("final_edges", 0)
    fin_by_t = defaultdict(list)
    for g, v in nodes_f.items():
        fin_by_t[int(v["t"])].append(g)
    gt_edge_set = {(int(u), int(v)) for u, v in gt_edges}
    node2gt = {}
    for t, ids in fin_by_t.items():
        rows_ = gt_by_t.get(t, [])
        if not rows_:
            continue
        g_um = gt_pos[rows_] * SCALE[None, :]
        f_um = np.array([nodes_f[g]["xyz"] for g in ids], float) \
            * SCALE[None, :]
        mm = _greedy_match(g_um, f_um)
        inv = {dj: rows_[gi] for gi, dj in mm.items()}
        for k, g in enumerate(ids):
            if k in inv:
                node2gt[g] = inv[k]
    seed_gids = set(seed_id.values())
    gid2ti = {g_: ti_ for ti_, g_ in seed_id.items()}
    frows = []
    for aa, bb in edges_f:
        f["final_edges"] += 1
        ua, ub = node2gt.get(aa), node2gt.get(bb)
        if ua is not None and ub is not None:
            cls = "tp_like" if (ua, ub) in gt_edge_set else "fp_confusion"
        elif ua is not None or ub is not None:
            cls = "fp_half"
        else:
            cls = "fp_invisible"
        f[cls] += 1
        pa_ = np.asarray(nodes_f[aa]["xyz"], float) * SCALE
        pb_ = np.asarray(nodes_f[bb]["xyz"], float) * SCALE
        # PROVENANCE. Three stages can put an edge in the final graph and
        # they are not the same kind of claim: the ILP chose it from
        # ranked candidates, _gap_close invented it to bridge a dropout,
        # _add_safe_divisions added it as a second daughter.
        if (aa, bb) in e0set:
            prov = 0.0                                   # ILP
        elif aa not in seed_gids or bb not in seed_gids:
            prov = 1.0                                   # gap-close bridge
        else:
            prov = 2.0                                   # division patch
        pp_ = np.nan
        ti_a, ti_b = gid2ti.get(aa), gid2ti.get(bb)
        if ti_a is not None and ti_b is not None:
            _x = prob_of.get((ti_a[0], ti_a[1], ti_b[1]))
            if _x is not None:
                pp_ = float(_x)
        err = np.nan
        if cls == "fp_confusion":
            tg = gt_out.get(ua, [])
            if tg:
                err = float(min(np.linalg.norm(gt_pos[k_] * SCALE - pb_)
                                for k_ in tg))
        frows.append((float(FPC.index(cls)),
                      float(np.linalg.norm(pb_ - pa_)),
                      pp_, prov,
                      _dens(int(nodes_f[aa]["t"]), pa_),
                      err))
    if frows:
        FROWS.setdefault(domain, []).append(np.asarray(frows, np.float32))


# ---- two arms x the two competition domains -----------------------------
for _arm in ("F4",):
  for _dm, names in (("6bba", SET_6BBA), ("44b6", SET_44B6)):
    domain = f"{_arm}-{_dm}"
    td0 = time.time()
    outd = pdir("DIAG", domain)
    for name in names:
        dets = load_dets(name, _arm)
        T = len(dets)
        gt = read_geff(geff_path(root, "train", name))
        gt_t = np.asarray(gt["t"], int)
        gt_pos = np.stack([np.asarray(gt[c], float)
                           for c in ("z", "y", "x")], axis=1)
        row_of = {int(v): i for i, v in enumerate(gt["node_ids"])}
        gt_e = [(row_of[int(a_)], row_of[int(b_)])
                for a_, b_ in gt["edges"]
                if int(a_) in row_of and int(b_) in row_of]
        diag_one(domain, name, dets, T, gt_t, gt_pos,
                 np.asarray(gt_e, int), outd / f"{name}.geff")
        gc.collect()
    res[domain], PER2[domain] = score_dir(outd)
    print(f"  {domain}: {len(names)} datasets ({time.time() - td0:.0f}s)",
          flush=True)

# ---- zebrahub: IN. The only domain whose GT covers every cell, so the
# only one where a wrong link can be distinguished from a link into a
# cell nobody labelled. ~2.5% of its cells are unlabelled (zhcomplete),
# so its FP rates are slightly overstated and its FN rates are not.
if SET_ZH:
    tz = time.time()
    GT_DIR = pathlib.Path("/kaggle/working/ps_zh_gt")
    GT_DIR.mkdir(parents=True, exist_ok=True)
    zrows = []
    outd = pdir("DIAG", "zh")
    for name in SET_ZH:
        _db, T, gt_arr, gt_e, n_cross = zh_load(name)
        _z4 = np.load(FDET_F4[name])
        dets = [(_z4[f"c{t}"], _z4[f"s{t}"]) for t in range(T)]
        del _db
        diag_one("F4-zh", name, dets, T, gt_arr[:, 0].astype(int),
                 gt_arr[:, 1:], np.asarray(gt_e, int),
                 outd / f"{name}.geff")
        gp = GT_DIR / f"{name}.geff"
        dr._save(str(gp), gt_arr, gt_e)
        r_ = td.graph.IndexedRXGraph.from_geff(str(gp))
        gtg = r_[0] if isinstance(r_, tuple) else r_
        rr = td.graph.IndexedRXGraph.from_geff(str(outd / f"{name}.geff"))
        g = rr[0] if isinstance(rr, tuple) else rr
        er = compute_metric(g, gtg, scale=tuple(SCALE))
        rec = (node_recall(g, gtg)
               if g.num_edges() > 0 and g.num_nodes() > 0 else 0.0)
        zrows.append(per_sample_metrics(er, float(len(gt_arr)), rec))
        del dets
        gc.collect()
    res["F4-zh"] = summarise(zrows)
    PER2["F4-zh"] = None
    _ZROWS = zrows
    print(f"  zebrahub: {len(SET_ZH)} crops ({time.time() - tz:.0f}s)",
          flush=True)
else:
    _ZROWS = []

# ---- report -------------------------------------------------------------
DOMAINS = [d for d in ("F4-6bba", "F4-44b6", "F4-zh") if d in res]
print("\n=== ANCHOR / official scorer ===", flush=True)
print("  F4-6bba must read adj_J 0.9216 and F4-44b6 0.8353 (f4diag, "
      "2026-09-01, same machinery and the same caches); F4-zh must read "
      "0.8115 (mergescore/fatescore). All three on D187, which is what "
      "ref 55889500 ships. If any anchor misses, the covariate tables "
      "below describe a pipeline nobody has scored.", flush=True)
for d in DOMAINS:
    r = res[d]
    print(f"  {d:<6} score {r['score']:.4f}  adj_J "
          f"{r['adj_edge_jaccard']:.4f}  div_J "
          f"{r['division_jaccard']:.4f}", flush=True)


def _official(d):
    """(edge_tp, edge_fp, edge_fn) or None, from the official scorer."""
    if PER2.get(d):
        rows_ = list(PER2[d].values())
    elif d.endswith("zh") and _ZROWS:
        rows_ = _ZROWS
    else:
        return None
    try:
        return (int(sum(r["edge_tp"] for r in rows_)),
                int(sum(r["edge_fp"] for r in rows_)),
                int(sum(r["edge_fn"] for r in rows_)))
    except Exception:
        return None


def _p(a, b):
    return f"{100.0 * a / b:.1f}%" if b else "  -  "


print("\n=== NODES, STAGE BY STAGE ===", flush=True)
print("  detections split into PRIMARIES (score >= 0.15, the only ones "
      "that can become nodes) and the REPAIR BAND [0.05, 0.15) which "
      "only _gap_close may draw on. Percentages are of primaries unless "
      "stated. `final_repair` are nodes _gap_close created FROM that "
      "band, so they are not primaries at all.", flush=True)
print(f"  {'arm-domain':<12} {'dets':>10} {'primary':>10} {'band':>9} "
      f"{'in_graph':>9} {'ILP':>8} {'killed':>8} {'final':>10} "
      f"{'+repair':>8}", flush=True)
for d in DOMAINS:
    n = NAGG.get(d)
    if not n:
        continue
    pr = n["primaries"]
    print(f"  {d:<12} {n['dets_total']:>10,} {pr:>10,} "
          f"{n['repair_band']:>9,} {_p(n['in_graph'], pr):>9} "
          f"{_p(n['ilp_linked'], pr):>8} {_p(n['killed_post'], pr):>8} "
          f"{n['final_nodes']:>10,} {n['final_repair']:>8,}", flush=True)

print("\n=== FINAL-OUTPUT DEGREE (3 = 3 or more) ===", flush=True)
for d in DOMAINS:
    n = NAGG.get(d)
    if not n:
        continue
    tot = n["final_nodes"]
    print(f"  {d:<12} n={tot:>9,}  " + "  ".join(
        f"deg{k}={_p(n.get(f'deg{k}', 0), tot)}" for k in (0, 1, 2, 3)),
        flush=True)

print("\n=== THE GT-NODE LEDGER: what the detector did to real cells ===",
      flush=True)
print("  matched = a PRIMARY sits within the match radius. subthresh = "
      "no primary, but a 0.05-0.15 candidate is there (the detector saw "
      "it and the gate rejected it). cold = nothing there at all. Only "
      "the annotated cells are counted, and 6bba annotates 9.71% of its "
      "cells against 44b6's 0.77%.", flush=True)
print(f"  {'arm-domain':<12} {'GT nodes':>10} {'matched':>10} "
      f"{'subthresh':>11} {'cold':>10}", flush=True)
for d in DOMAINS:
    n = NAGG.get(d)
    if not n:
        continue
    g = n["gt_nodes"]
    print(f"  {d:<12} {g:>10,} "
          f"{n['gt_matched_primary']:>6,} {_p(n['gt_matched_primary'], g):>3} "
          f"{n['gt_subthresh_only']:>6,} {_p(n['gt_subthresh_only'], g):>4} "
          f"{n['gt_cold']:>6,} {_p(n['gt_cold'], g):>3}", flush=True)

print("\n=== WHERE EVERY GT EDGE GOES (charged to the FIRST losing "
      "stage) ===", flush=True)
print("  det_miss_subthresh = an endpoint missed but a 0.05-0.15 pool "
      "candidate sits within 7 um (the flicker band); det_miss_cold = "
      "nothing there at all. no_candidate = both endpoints detected, "
      "pair outside the 10 um gate. below_floor = ranker prob < 0.10. "
      "ilp_rejected = the solver took a competing edge. killed_post = "
      "the short filter deleted it after linking.", flush=True)
for d in DOMAINS:
    a = AGG.get(d)
    if not a:
        continue
    adj = a["gt_edges"] - a["nonadjacent"]
    print(f"\n  -- {d}: {a['gt_edges']:,} GT edges, {a['nonadjacent']:,} "
          f"non-adjacent (discarded by the metric), adjacent {adj:,} --",
          flush=True)
    for c in CATS:
        print(f"  {c:<20} {a[c]:>8,}  {100.0 * a[c] / max(1, adj):>6.2f}%",
              flush=True)
    off = _official(d)
    fn_est = adj - a["recovered"]
    if off:
        tp_o, fp_o, fn_o = off
        print(f"  cross-check: our lost-edge total {fn_est:,} vs official "
              f"edge_fn {fn_o:,} (residual {fn_o - fn_est:+,} = recovered "
              f"edges the scorer still counts FN via its own matching, "
              f"minus double-counts)", flush=True)

print("\n=== WHAT THE FINAL EDGES ARE (FP side) ===", flush=True)
print("  tp_like = both endpoints matched GT and that GT edge exists. "
      "fp_confusion = both matched, no such GT edge (a WRONG LINK). "
      "fp_half = one endpoint matched (the over-detection channel: an "
      "edge into a spurious node next to a real cell). fp_invisible = "
      "neither endpoint annotated (unpenalised under the lenient rule "
      "on sparse GT; REAL FPs on zh where GT is dense).", flush=True)
for d in DOMAINS:
    f = FAGG.get(d)
    if not f:
        continue
    tot = f["final_edges"]
    print(f"\n  -- {d}: {tot:,} final edges --", flush=True)
    for c in FPC:
        print(f"  {c:<15} {f[c]:>9,}  {100.0 * f[c] / max(1, tot):>6.2f}%",
              flush=True)
    off = _official(d)
    if off:
        print(f"  cross-check: official edge_fp {off[1]:,}, edge_tp "
              f"{off[0]:,}", flush=True)

print("\n=== reading it ===", flush=True)
print("  The FN table prices the repair lines: detection recall work "
      "attacks det_miss_*, a better second detector attacks "
      "det_miss_cold, a wider gate attacks no_candidate, a better "
      "ranker attacks below_floor + ilp_rejected, the short filter "
      "attacks killed_post. The FP table prices the other half: "
      "fp_confusion is the linker linking wrong, fp_half is "
      "over-detection being charged through the lenient edge rule. "
      "Attack the biggest bucket that has a mechanism, not the one "
      "with the nicest tool.", flush=True)
print("  ⚠ zh is the only domain whose GT covers every cell -- its "
      "columns are the honest rates; 6bba/44b6 rates describe annotated "
      "cells only (visibility-biased). 44b6 below_floor is undercounted "
      "(D187 in-sample).", flush=True)

# ------------------------------------------------------------------------
# THE CONDITIONS. Everything above is counts; everything below is when.
# ------------------------------------------------------------------------
def _cat(store, d):
    v = store.get(d)
    return np.concatenate(v, axis=0) if v else None


def _med(arr, col):
    v = arr[:, col]
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else float("nan")


def _pct(arr, col, q):
    v = arr[:, col]
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if len(v) else float("nan")


def _fmt(x, w=7, p=2):
    return f"{x:>{w}.{p}f}" if np.isfinite(x) else f"{'-':>{w}}"


DISP_BINS = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 1e9]
DENS_BINS = [0.0, 2.0, 5.0, 10.0, 20.0, 1e9]


def _bin_label(b, i):
    lo, hi = b[i], b[i + 1]
    return f"{lo:g}-{hi:g}" if hi < 1e8 else f"{lo:g}+"


print("\n\n=== MISSED EDGES: THE CONDITIONS ===", flush=True)
print("  One row per bucket. disp = how far the cell actually moved "
      "between the two frames. dens10 = primaries within 10 um of the "
      "source, i.e. how many competitors the linker had. gap2 = distance "
      "from the true target to the NEAREST OTHER primary; small means a "
      "distractor sits on top of the right answer. p_true = the ranker's "
      "probability for the CORRECT pair, rank = where that pair sat in "
      "the source's candidate list (0 = the ranker's own first choice), "
      "p_top = the probability of whatever the ranker liked best. "
      "`recovered` is the control: read every other row against it.",
      flush=True)
for d in DOMAINS:
    A = _cat(EROWS, d)
    if A is None:
        continue
    print(f"\n  -- {d}: {len(A):,} adjacent GT edges --", flush=True)
    print(f"  {'bucket':<20} {'n':>7} {'disp':>7} {'p90disp':>8} "
          f"{'dens10':>7} {'gap2':>7} {'z_um':>7} {'sc_src':>7} "
          f"{'p_true':>7} {'rank':>6} {'p_top':>7} {'%div':>6}",
          flush=True)
    for ci, c in enumerate(CATS):
        S = A[A[:, 0] == ci]
        if not len(S):
            continue
        dv = S[:, 11]
        print(f"  {c:<20} {len(S):>7,} {_fmt(_med(S, 1))} "
              f"{_fmt(_pct(S, 1, 90), 8)} {_fmt(_med(S, 2))} "
              f"{_fmt(_med(S, 3))} {_fmt(_med(S, 4))} "
              f"{_fmt(_med(S, 6))} {_fmt(_med(S, 8))} "
              f"{_fmt(_med(S, 9), 6, 1)} {_fmt(_med(S, 10))} "
              f"{100.0 * float(dv.sum()) / len(S):>5.1f}%", flush=True)

print("\n=== LOSS RATE BY DISPLACEMENT (how far the cell moved) ===",
      flush=True)
print("  `lost` = every bucket except recovered. The columns after it "
      "are that bin's share of the loss by cause, so a row reads: at "
      "this speed, this fraction of GT edges is lost, and this is what "
      "loses them.", flush=True)
for d in DOMAINS:
    A = _cat(EROWS, d)
    if A is None:
        continue
    rec = float(CATS.index("recovered"))
    print(f"\n  -- {d} --", flush=True)
    print(f"  {'disp um':<10} {'n':>8} {'lost':>7}   " + " ".join(
        f"{c[:11]:>11}" for c in CATS if c != "recovered"), flush=True)
    for i in range(len(DISP_BINS) - 1):
        S = A[(A[:, 1] >= DISP_BINS[i]) & (A[:, 1] < DISP_BINS[i + 1])]
        if not len(S):
            continue
        lost = float((S[:, 0] != rec).sum())
        parts = []
        for c in CATS:
            if c == "recovered":
                continue
            k = float((S[:, 0] == CATS.index(c)).sum())
            parts.append(f"{100.0 * k / len(S):>10.2f}%")
        print(f"  {_bin_label(DISP_BINS, i):<10} {len(S):>8,} "
              f"{100.0 * lost / len(S):>6.2f}%   " + " ".join(parts),
              flush=True)

print("\n=== LOSS RATE BY CROWDING (primaries within 10 um) ===",
      flush=True)
for d in DOMAINS:
    A = _cat(EROWS, d)
    if A is None:
        continue
    rec = float(CATS.index("recovered"))
    print(f"\n  -- {d} --", flush=True)
    print(f"  {'dens10':<10} {'n':>8} {'lost':>7}   " + " ".join(
        f"{c[:11]:>11}" for c in CATS if c != "recovered"), flush=True)
    for i in range(len(DENS_BINS) - 1):
        S = A[(A[:, 2] >= DENS_BINS[i]) & (A[:, 2] < DENS_BINS[i + 1])]
        if not len(S):
            continue
        lost = float((S[:, 0] != rec).sum())
        parts = []
        for c in CATS:
            if c == "recovered":
                continue
            k = float((S[:, 0] == CATS.index(c)).sum())
            parts.append(f"{100.0 * k / len(S):>10.2f}%")
        print(f"  {_bin_label(DENS_BINS, i):<10} {len(S):>8,} "
              f"{100.0 * lost / len(S):>6.2f}%   " + " ".join(parts),
              flush=True)

print("\n=== WAS THE CORRECT PAIR EVEN RANKED FIRST? ===", flush=True)
print("  Over GT edges whose BOTH endpoints were detected and whose "
      "correct pair was a candidate. rank 0 means the ranker's own top "
      "choice was right, so anything lost from rank 0 was lost by the "
      "ILP's global solve rather than by the score.", flush=True)
for d in DOMAINS:
    A = _cat(EROWS, d)
    if A is None:
        continue
    S = A[np.isfinite(A[:, 9])]
    if not len(S):
        continue
    rec = float(CATS.index("recovered"))
    print(f"\n  -- {d}: {len(S):,} GT edges with a scored correct pair --",
          flush=True)
    print(f"  {'rank':<8} {'n':>8} {'share':>7} {'lost':>7} "
          f"{'med p_true':>11} {'med p_top':>10}", flush=True)
    for lo, hi, lbl in ((0, 1, "0"), (1, 2, "1"), (2, 3, "2"),
                        (3, 5, "3-4"), (5, 1e9, "5+")):
        R = S[(S[:, 9] >= lo) & (S[:, 9] < hi)]
        if not len(R):
            continue
        lost = float((R[:, 0] != rec).sum())
        print(f"  {lbl:<8} {len(R):>8,} {100.0 * len(R) / len(S):>6.2f}% "
              f"{100.0 * lost / len(R):>6.2f}% {_fmt(_med(R, 8), 11, 3)} "
              f"{_fmt(_med(R, 10), 10, 3)}", flush=True)

print("\n\n=== WRONG EDGES: THE CONDITIONS ===", flush=True)
print("  prov = which stage created the edge: ILP chose it from ranked "
      "candidates, BRIDGE is a _gap_close insert across a one-frame "
      "dropout, DIVPATCH is _add_safe_divisions' second daughter. "
      "err_um (fp_confusion only) = how far the chosen target sits from "
      "the correct one, so a small err is a near-miss between two "
      "adjacent cells and a large one is a different cell entirely.",
      flush=True)
PROV = ["ILP", "BRIDGE", "DIVPATCH"]
for d in DOMAINS:
    F = _cat(FROWS, d)
    if F is None:
        continue
    print(f"\n  -- {d}: {len(F):,} final edges --", flush=True)
    print(f"  {'class':<15} {'n':>9} {'len_um':>7} {'p90len':>8} "
          f"{'p':>7} {'dens10':>7} {'err_um':>7}   " + " ".join(
              f"{p:>9}" for p in PROV), flush=True)
    for ci, c in enumerate(FPC):
        S = F[F[:, 0] == ci]
        if not len(S):
            continue
        pv = []
        for k in range(3):
            n_ = float((S[:, 3] == k).sum())
            pv.append(f"{100.0 * n_ / len(S):>8.2f}%")
        print(f"  {c:<15} {len(S):>9,} {_fmt(_med(S, 1))} "
              f"{_fmt(_pct(S, 1, 90), 8)} {_fmt(_med(S, 2), 7, 3)} "
              f"{_fmt(_med(S, 4))} {_fmt(_med(S, 5))}   " + " ".join(pv),
              flush=True)

print("\n=== WRONG-LINK ERROR DISTANCE (fp_confusion only) ===", flush=True)
print("  How far the wrong target sits from the right one. Under ~5 um "
      "is a swap between neighbouring cells; far out is a link across "
      "the field.", flush=True)
for d in DOMAINS:
    F = _cat(FROWS, d)
    if F is None:
        continue
    S = F[(F[:, 0] == float(FPC.index("fp_confusion")))
          & np.isfinite(F[:, 5])]
    if not len(S):
        continue
    print(f"\n  -- {d}: {len(S):,} wrong links with a known correct "
          f"target --", flush=True)
    for lo, hi in ((0, 3), (3, 5), (5, 8), (8, 12), (12, 1e9)):
        k = float(((S[:, 5] >= lo) & (S[:, 5] < hi)).sum())
        lbl = f"{lo:g}-{hi:g} um" if hi < 1e8 else f"{lo:g}+ um"
        print(f"  {lbl:<12} {int(k):>7,} {100.0 * k / len(S):>6.2f}%",
              flush=True)

print("\n=== reading the conditions ===", flush=True)
print("  A bucket whose disp is far above `recovered` is a SPEED "
      "failure and wants a wider gate or a motion model. A bucket whose "
      "dens10 is far above it, or whose gap2 is far below it, is a "
      "CROWDING failure and wants a better score, not a wider gate -- a "
      "wider gate adds more of the distractors that caused it. A bucket "
      "whose p_true is high while rank is 0 was not lost by the ranker "
      "at all and only the ILP or the post-link filters can be "
      "responsible.", flush=True)
print("  ⚠ zh is the only domain whose GT covers every cell, so it is "
      "the only place fp_confusion and fp_invisible mean what they say; "
      "on 6bba and 44b6 almost every edge has no annotated endpoint. "
      "44b6 below_floor stays undercounted (D187 in-sample).", flush=True)

print("\n\n=== WHAT THE RANKER PREFERRED INSTEAD ===", flush=True)
print("  One row per GT edge whose correct pair was NOT the ranker's "
      "first choice. `top is a cell` = the preferred target matched an "
      "annotated GT node; on 6bba and 44b6 most cells are unannotated so "
      "this is a LOWER BOUND there and only zebrahub measures it. `top "
      "is another cell's successor` = that GT node is the target of some "
      "GT edge, i.e. the ranker tried to take a successor that belongs "
      "to a different track.", flush=True)
for d in DOMAINS:
    R = RROWS.get(d)
    if not R:
        continue
    A = np.asarray(R, np.float64)
    n = len(A)
    print(f"\n  -- {d}: {n:,} GT edges with the correct pair at rank >= 1 --",
          flush=True)
    print(f"  preferred target IS an annotated cell:      "
          f"{int(A[:, 3].sum()):>7,}  {100.0 * A[:, 3].mean():>5.1f}%",
          flush=True)
    print(f"  ... and is another cell's GT successor:     "
          f"{int(A[:, 4].sum()):>7,}  {100.0 * A[:, 4].mean():>5.1f}%",
          flush=True)
    print(f"  the lost edge survived anyway:              "
          f"{int(A[:, 11].sum()):>7,}  {100.0 * A[:, 11].mean():>5.1f}%",
          flush=True)
    print(f"  distances um   src->preferred {np.median(A[:, 5]):>6.2f}   "
          f"src->true {np.median(A[:, 6]):>6.2f}   "
          f"preferred->true {np.median(A[:, 7]):>6.2f}", flush=True)
    print(f"  detection score   preferred {np.median(A[:, 8]):>6.3f}   "
          f"true {np.median(A[:, 9]):>6.3f}", flush=True)
    print(f"  sources bidding for the preferred target: median "
          f"{np.median(A[:, 10]):>4.1f}", flush=True)
    closer = float((A[:, 5] < A[:, 6]).sum())
    print(f"  the preferred target is CLOSER than the true one: "
          f"{int(closer):,} ({100.0 * closer / n:.1f}%)", flush=True)
    for lo, hi in ((0, 2), (2, 4), (4, 6), (6, 1e9)):
        k = float(((A[:, 7] >= lo) & (A[:, 7] < hi)).sum())
        lbl = f"{lo:g}-{hi:g} um" if hi < 1e8 else f"{lo:g}+ um"
        print(f"    preferred sits {lbl:<9} from the true target: "
              f"{int(k):>7,} ({100.0 * k / n:>5.1f}%)", flush=True)

print("\n=== THE NODES THE SOLVER DROPPED (node_absent) ===", flush=True)
print("  Both endpoints detected, the correct pair the ranker's own top "
      "choice, and the source never reached the solved graph. best_out "
      "is its best outgoing probability, best_in its best incoming one, "
      "bidders_for_self how many sources in the previous frame listed it "
      "as a candidate.", flush=True)
for d in DOMAINS:
    N = NROWS.get(d)
    if not N:
        continue
    A = np.asarray(N, np.float64)
    def _m(c):
        v = A[:, c]
        v = v[np.isfinite(v)]
        return float(np.median(v)) if len(v) else float("nan")
    print(f"\n  -- {d}: {len(A):,} dropped sources --", flush=True)
    print(f"  best_out  median {_m(0):.3f}   best_in median {_m(1):.3f}   "
          f"n_cand median {_m(2):.1f}   bidders median {_m(3):.1f}   "
          f"det score median {_m(4):.3f}", flush=True)
    noin = float((~np.isfinite(A[:, 1])).sum())
    print(f"  had NO incoming candidate at all: {int(noin):,} "
          f"({100.0 * noin / len(A):.1f}%)", flush=True)
    for lo, hi in ((0.0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.01)):
        k = float(((A[:, 0] >= lo) & (A[:, 0] < hi)).sum())
        print(f"    best_out in {lo:.1f}-{hi:.1f}: {int(k):>6,} "
              f"({100.0 * k / len(A):>5.1f}%)", flush=True)

print("\n=== reading it ===", flush=True)
print("  If the preferred target is usually NOT a cell, the distractor "
      "is a spurious detection and detection precision is the lever. If "
      "it usually IS a cell, and specifically another track's successor, "
      "the ranker is being asked to break a tie it cannot see and only "
      "the global solve could fix it -- which it did not, since these "
      "edges are lost 85% of the time. The distance columns separate "
      "those two: a spurious peak tends to sit ON the true target, a "
      "rival cell sits a cell-width away.", flush=True)
print("  ⚠ On 6bba 9.71% of cells are annotated and on 44b6 0.77%, so "
      "`top is an annotated cell` is a floor in those columns, not a "
      "rate. zebrahub's ~97.5% annotation is the only one that reads as "
      "a rate.", flush=True)

print("\n\n=== WAS THE VELOCITY EVEN THERE? ===", flush=True)
print("  On exactly the rank >= 1 population above. nv is the source's "
      "velocity magnitude as the ranker received it; at nv = 0 the "
      "predicted position IS the source position and `md` collapses to "
      "plain distance, so the nearest candidate wins by construction. "
      "rank_by_d and rank_by_md are where the TRUE target sits when the "
      "same candidate list is ordered by those two columns instead of "
      "by the model's probability.", flush=True)
for d in DOMAINS:
    V = VROWS.get(d)
    if not V:
        continue
    A = np.asarray(V, np.float64)
    n = len(A)
    nv = A[:, 0]
    zero = float((nv <= 1e-9).sum())
    print(f"\n  -- {d}: {n:,} cases --", flush=True)
    print(f"  velocity is EXACTLY ZERO: {int(zero):,} "
          f"({100.0 * zero / n:.1f}%)", flush=True)
    fin = nv[np.isfinite(nv) & (nv > 1e-9)]
    if len(fin):
        print(f"  where non-zero: median {np.median(fin):.2f} um  "
              f"p90 {np.percentile(fin, 90):.2f} um", flush=True)
    for lbl, col in (("by raw distance", 1), ("by predicted distance", 2)):
        c = A[:, col]
        c = c[np.isfinite(c)]
        if not len(c):
            continue
        first = float((c == 0).sum())
        print(f"  true target ranked FIRST {lbl:<22}: "
              f"{int(first):>6,} ({100.0 * first / len(c):>5.1f}%)",
              flush=True)
    both = A[np.isfinite(A[:, 1]) & np.isfinite(A[:, 2])]
    if len(both):
        fixed = float((both[:, 2] == 0).sum())
        print(f"  the predicted-distance ordering puts it first while "
              f"the model did not: {int(fixed):,} "
              f"({100.0 * fixed / len(both):.1f}% of {len(both):,})",
              flush=True)
        better = float((both[:, 2] < both[:, 1]).sum())
        worse = float((both[:, 2] > both[:, 1]).sum())
        print(f"  predicted-distance beats raw distance in "
              f"{int(better):,} ({100.0 * better / len(both):.1f}%), "
              f"loses in {int(worse):,} "
              f"({100.0 * worse / len(both):.1f}%)", flush=True)
    print(f"  medians   d_true {np.nanmedian(A[:, 4]):>6.2f}  "
          f"md_true {np.nanmedian(A[:, 5]):>6.2f}  "
          f"d_top {np.nanmedian(A[:, 6]):>6.2f}  "
          f"md_top {np.nanmedian(A[:, 7]):>6.2f}  "
          f"n_cand {np.nanmedian(A[:, 8]):>4.1f}", flush=True)

print("\n=== reading it ===", flush=True)
print("  A high zero-velocity share means the ranker was handed no "
      "motion at all for exactly the cells whose motion decides the "
      "answer, and the repair is upstream of the model: build the "
      "velocities from the ILP's solved tracks rather than from "
      "pass1_tracks' greedy chain, which biohub-accel already recorded "
      "as untried. If instead the velocity is present and the "
      "predicted-distance ordering still does not recover the true "
      "target, the motion signal does not separate these cases and "
      "neither route helps.", flush=True)
print(f"\nVELWHY DONE in {(time.time() - t_all) / 60:.1f} min",
      flush=True)
