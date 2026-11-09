"""nopred -- WHY does a primary have no predecessor in the greedy pre-pass?

SCRIPT_VERSION 2026-09-03a-nopred

THE QUESTION (the user, 2026-09-03). `pass1_tracks` -- the per-frame
two-stage Hungarian (`_link_sim`) that runs before the ILP and is the
only source of the ranker's velocity columns -- leaves 12.7% / 8.4% /
4.2% of the primaries at t >= 1 (F4-6bba / 44b6 / zh) with velocity
EXACTLY ZERO, i.e. no predecessor. twopass (2026-09-03) showed the
ILP's own tracks give those cells no predecessor either, and flowvel
showed that a neighbourhood-flow velocity for them changes nothing.
So: what ARE they? This kernel buckets every zero-velocity primary at
t >= 1 into mutually exclusive causes. Diagnostic only; no sweep; one
ILP solve per dataset (needed for the recovery question below).

DECISION ORDER (first match wins; every primary lands in exactly one):

  1. unmatched        no GT node within MATCH_R 7.0 um at frame t
                      (`_greedy_match`, edgewhy's m_seed join, 1:1
                      nearest-first). zh's GT covers ~97.5% of its
                      cells, so there this is approximately a spurious
                      detection; 6bba / 44b6 annotate 9.7% / 0.77%, so
                      there it is UNVERIFIABLE. Both labels are printed
                      and a competition-domain unmatched primary is
                      never called spurious.
     -- the rest are GT-matched; use the GT node's own parent in the
        GT graph (adjacent edges only, t-1 -> t, as the metric does) --
  2. gt_start         the GT node has no GT parent at t-1: a true track
                      start or a cell entering the field. Split by the
                      distance to the nearest face of the field of view
                      (< BORDER_UM 10 um vs interior), and flagged when
                      the GT node has a NON-adjacent GT parent (a gap
                      edge the metric discards).
  3. division_daughter the GT parent has two GT children. Sub-split:
                      did the pre-pass give the parent's primary to the
                      sister (the sister's prev IS the parent's primary),
                      to some other primary, to nobody, or was the
                      parent not detected at all.
  4. parent_undetected the GT parent exists but no primary sits within
                      MATCH_R of it at t-1. Sub-split `subthresh` (a
                      0.05-0.15 repair-band candidate within MATCH_R,
                      edgewhy's `near_pool`) vs `cold`.
  5. parent_taken     the parent's primary exists and pass-1 assigned
                      it to a DIFFERENT primary at t (`prev_of`). The
                      competitor is labelled unmatched / matched to
                      another GT node; distances competitor -> GT child
                      vs this cell -> GT child, and both -> the parent
                      primary.
  6. parent_out_of_gate the parent's primary exists, is unassigned, and
                      its raw distance to this cell exceeds MAX_LINK_UM.
  7. parent_free_but_unlinked parent primary exists, unassigned, within
                      the gate, and still not linked. Sub-split by
                      which stage could have taken it (raw distance
                      <= min(TIGHT_UM, MAX_LINK_UM) = stage 1, else
                      stage 2). EXPECTED EMPTY: the Hungarian assigns
                      every feasible pair before leaving a row and a
                      column both free, so a non-zero count here is a
                      bug in this diagnostic, not a finding.
  8. other            anything left (e.g. |v| == 0 with a predecessor:
                      two primaries at identical coordinates), counted
                      and sub-labelled.

THE GATES, as `_link_sim` applies them. Stage 1: Hungarian over ALL
prev x curr primaries, a pair is feasible when the RAW distance is
<= min(TIGHT_UM 6.0, MAX_LINK_UM 10.0); the cost of a feasible pair is
the distance from the damped prediction P + 0.5 * v_prev plus
LINK_SIM_W * |logit(score_prev) - logit(score_curr)|. Stage 2: the same
over the primaries BOTH stages left free, gate MAX_LINK_UM. A pair is
"out of gate" only when the raw distance exceeds MAX_LINK_UM; the
prediction never gates, it only prices.

THE BORDER. Competition datasets: the zarr's own (T, Z, Y, X) shape is
read through viz_cells.open_image (lazy; zarr.json only) and the field
extent is shape[-3:] * SCALE. reference.md records the shape as
(100, 64, 256, 256) on every train dataset; a different shape is
warned, a read failure falls back to that shape with a warning.
Zebrahub crops: COMP_SHAPE (64, 256, 256) from zebrahub_extract.py:83,
the crops are cut to the competition shape by construction and the
zhdet_ caches carry no shape key. The distance is to the nearest of the
six faces, z faces included (the field is a 104 x 104 x 104 um cube);
the share of border cases whose nearest face is a z face is printed.

RECOVERY. For every zero-velocity primary that is GT-matched with a GT
parent (buckets 3-7), whether the FINAL graph (ANCHOR: F4 + D187,
`solve_one` + `patch_final`) has an edge INTO this primary's node from
a node at t-1 within MATCH_R of the GT parent -- by position, so a
_gap_close repair node counts. Also, for ALL zero-velocity primaries:
in the final graph at all, and there with a parent vs as a track start.

CONTROL. Per dataset, a seeded random sample of NON-zero-velocity
primaries at t >= 1 (same size as the zero group) goes through the same
buckets 1-4; where the zero group has 5-7 the control has
`linked_to_parent` (pass-1's predecessor is the GT parent's primary)
vs `linked_elsewhere`. That is what makes the zero group's unmatched
and gt_start rates readable.

WHAT IS PRINTED. One line per dataset (primaries at t >= 1, zeros,
bucket counts, adj_J, seconds). Per domain: bucket counts and shares
(of zero-velocity primaries and of all primaries at t >= 1), the
sub-splits with counts and medians of the distances above, recovery
per bucket, the final-graph fate of the zero group, the control
group's buckets, the anchors, `NOPRED OK` when all 32 datasets ran
and the anchors held.

ANCHORS (F4, D187, hard within ANCHOR_TOL 0.0005): F4-6bba adj_J 0.9216
/ div_J 0.0606, F4-44b6 0.8353 / 0.0000, F4-zh 0.8118 / 0.0215 --
rankwhy / velwhy / twopass / flowvel.

ASSEMBLY. twopass.py (2026-09-03a) by line slices: the preamble,
constants, caches, `load_dets`, the plumbing through `patch_final`,
the acceleration block (`pass1_tracks` with `prev_of`), the D187
loader, the three domains and `zh_load`, `solve_one`, `patch_final`,
`score_one`, `score_zh_one`, `seed_node_maps`, `pdir`. twopass's arms
are dropped. Everything under "nopred" is new.

RUNTIME. One solve per dataset, no attribution: rankwhy's ~68 min.
BUDGET_S 10 h guard before each dataset; unattempted datasets are
listed as INCOMPLETE.

Kaggle: CPU, enable_internet true. Sources = twopass's
(kernel-metadata.json, id faustine649/biohub-nopred): dataset_sources
pilkwang/biohub-tracking-support-pack-50ep-v1, faustine649/cell-tracking,
faustine649/biohub-dctrain-175-weights; kernel_sources
faustine649/biohub-edge-ranker, faustine649/biohub-cache-train,
faustine649/biohub-zh-detect, faustine649/biohub-poolsize,
faustine649/biohub-fusecache; competition_sources
biohub-cell-tracking-during-development. Push with
`py kag.py kernels push -p <folder>`.
"""

# ---- [assembled from rankwhy.py: A preamble] ----
from __future__ import annotations

import gc
import pathlib
import pickle
import random
import subprocess
import sys
import time
from collections import defaultdict


SCRIPT_VERSION = "2026-09-03a-nopred"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

# ---- nopred switches -----------------------------------------------------
# Recorded anchors (rankwhy / velwhy / twopass / flowvel, F4, D187).
# Recorded anchors for the ANCHOR arm (rankwhy / velwhy, F4, D187).
ANCHOR_ADJ = {"F4-6bba": 0.9216, "F4-44b6": 0.8353, "F4-zh": 0.8118}
ANCHOR_DIV = {"F4-6bba": 0.0606, "F4-44b6": 0.0000, "F4-zh": 0.0215}
ANCHOR_TOL = 0.0005
BUDGET_S = 10.0 * 3600  # Kaggle kills a CPU kernel at 12 h; stop cleanly first

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

# ---- [assembled from rankwhy.py: B constants + C detcaches] ----
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


# ---- [assembled from rankwhy.py: fdet_F4 walk + load_dets] ----
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


# ---- [assembled from rankwhy.py: D plumbing, _logit .. solve_edges] ----
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


# ---- [assembled from rankwhy.py: E accel block / 23-column pair_features] ----
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


# ---- [assembled from rankwhy.py: F ranker constants + D187 loader] ----
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


# ---- [assembled from rankwhy.py: G the three domains + zh_load] ----
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
print("  zebrahub: the third domain; the only domain whose GT covers every cell, so the only one where 'unmatched' can be read as spurious.", flush=True)
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


# ---- [assembled from edgeprob_rank.py: solve_one re-typed with model/cols/feats] ----
def solve_one(dets, T, seeds, feats, model, cols):
    """rankwhy.solve_one with the model, its columns and the (joined)
    features as parameters. Everything up to and including the ILP, plus
    the sub-threshold repair pool. The post-link stages run in patch_final."""
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
    del graph, probs
    return nodes0, edges0, (cand, cand_sc, cand_trees)


# ---- [assembled from rankwhy.py: patch_final verbatim] ----
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


# ---- [assembled from edgeprob_rank.py: score_one] ----
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


# ==========================================================================
# ---- [assembled from twopass.py: seed_node_maps, the seed <-> node join] ----
# ==========================================================================
def seed_node_maps(seeds, nodes0, T):
    """seed (t, i) <-> solved node id, joined by ROUNDED coordinates.

    solve_one feeds np.round(seed coords).astype(int16) into the graph and
    reads the solved nodes back from node_attrs, so a solved node's
    (t, z, y, x) are the rounded coordinates of one primary (rankwhy's
    diag_one makes the same join). Returns (node_of_seed, seed_of_node,
    n_dup_seed, n_node_unmapped)."""
    key_of = {}
    n_dup = 0
    for t in range(T):
        P, _s = seeds[t]
        for i in range(len(P)):
            k = (t, tuple(int(round(float(x))) for x in P[i]))
            if k in key_of:
                n_dup += 1
            key_of[k] = (t, i)
    seed_of_node = {}
    n_unmapped = 0
    for g, v in nodes0.items():
        k = (int(v["t"]), tuple(int(round(float(x))) for x in v["xyz"]))
        s = key_of.get(k)
        if s is None:
            n_unmapped += 1
        else:
            seed_of_node[g] = s
    node_of_seed = {s: g for g, s in seed_of_node.items()}
    return node_of_seed, seed_of_node, n_dup, n_unmapped


def score_zh_one(gp, gtg, n_gt):
    """rankwhy's zebrahub scoring body: the official metric against the
    crop's own GT graph."""
    rr = td.graph.IndexedRXGraph.from_geff(str(gp))
    g = rr[0] if isinstance(rr, tuple) else rr
    er = compute_metric(g, gtg, scale=tuple(SCALE))
    rec = (node_recall(g, gtg)
           if g.num_edges() > 0 and g.num_nodes() > 0 else 0.0)
    return per_sample_metrics(er, float(n_gt), rec)


def pdir(arm, domain):
    d = pathlib.Path(f"/kaggle/working/tp_{domain}_{arm}")
    d.mkdir(parents=True, exist_ok=True)
    return d




# ---- [assembled from edgewhy.py: MATCH_R + _greedy_match verbatim] ----
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




# ==========================================================================
# ---- nopred: bucket every zero-velocity primary --------------------------
# ==========================================================================
BORDER_UM = 10.0
# zebrahub_extract.py:83 COMP_SHAPE -- the crops are cut to the competition
# shape; reference.md: every train dataset is (100, 64, 256, 256).
COMP_SHAPE_ZYX = (64, 256, 256)
CTRL_SEED = 20260903

BUCKETS = ["unmatched", "gt_start", "division_daughter", "parent_undetected",
           "parent_taken", "parent_out_of_gate", "parent_free_but_unlinked",
           "linked_to_parent", "linked_elsewhere", "other"]
ZERO_BUCKETS = BUCKETS[:7] + ["other"]
CTRL_BUCKETS = BUCKETS[:4] + ["linked_to_parent", "linked_elsewhere", "other"]
SUBS = ["-", "border<10", "interior",
        "sister_took_parent", "parent_to_other", "parent_unassigned",
        "parent_undetected",
        "subthresh", "cold",
        "comp_unmatched", "comp_other_gt",
        "beyond_max_link", "stage1_eligible", "stage2_eligible",
        "linked_to_parent", "linked_elsewhere",
        "linked_zero_disp", "parent_wrong_frame"]
# record columns
RC = ["bucket", "sub", "d_a", "d_b", "d_c", "d_d", "has_gt_parent",
      "in_final", "has_parent", "recovered"]
# what d_a .. d_d mean, per bucket (for the tables)
DIST_LABELS = {
    "unmatched": ["nearest GT at t"],
    "gt_start": ["to nearest face", "face axis (0 z / 1 y / 2 x)",
                 "has non-adjacent GT parent"],
    "division_daughter": ["this -> GT parent", "sister prim -> GT parent"],
    "parent_undetected": ["this -> GT parent"],
    "parent_taken": ["competitor -> GT child", "this -> GT child",
                     "competitor -> parent prim", "this -> parent prim"],
    "parent_out_of_gate": ["raw this -> parent prim",
                           "from damped prediction"],
    "parent_free_but_unlinked": ["raw this -> parent prim",
                                 "from damped prediction"],
    "linked_to_parent": ["this -> parent prim", "this -> its prev"],
    "linked_elsewhere": ["this -> parent prim", "this -> its prev"],
    "other": [],
}


def _norm(a):
    return float(np.linalg.norm(np.asarray(a, float)))


def field_extent_um(group, name):
    """(extent_um (3,), how it was read). The border is at extent."""
    if group == "zh":
        return (np.array(COMP_SHAPE_ZYX, float) * SCALE,
                "COMP_SHAPE (zebrahub_extract.py:83)")
    try:
        from viz_cells import open_image
        img = open_image(root / "train" / f"{name}.zarr")
        shp = tuple(int(v) for v in img.shape)
        zyx = shp[-3:]
        if zyx != COMP_SHAPE_ZYX:
            warn_once("field_shape_unexpected",
                      f"{name}: zarr shape {shp}, expected z/y/x "
                      f"{COMP_SHAPE_ZYX}; border distances use the zarr's "
                      f"own shape")
        return np.array(zyx, float) * SCALE, "zarr shape via open_image"
    except Exception as e:                       # noqa: BLE001
        warn_once("field_shape_fallback",
                  f"could not read the zarr shape for {name} ({e!r}) -> "
                  f"COMP_SHAPE {COMP_SHAPE_ZYX} assumed; every border "
                  f"distance below relies on it")
        return np.array(COMP_SHAPE_ZYX, float) * SCALE, "fallback COMP_SHAPE"


def border_dist(pos_um, extent_um):
    """(distance to the nearest of the six faces, axis of that face)."""
    lo = np.asarray(pos_um, float)
    hi = np.asarray(extent_um, float) - lo
    both = np.minimum(lo, hi)
    ax = int(np.argmin(both))
    return float(both[ax]), ax


class GTContext:
    """Everything bucket_one reads, built once per dataset."""

    def __init__(self, gt_t, gt_pos, gt_edges, T, seeds, vels, prev_of,
                 cand_trees, extent_um):
        self.T = T
        self.seeds = seeds
        self.vels = vels
        self.prev_of = prev_of
        self.extent_um = np.asarray(extent_um, float)
        gt_t = np.asarray(gt_t, int)
        self.gt_t = gt_t
        self.gt_um = np.asarray(gt_pos, float) * SCALE[None, :]
        # adjacent GT edges only (the metric discards the others); a node
        # whose only GT parent edge skips a frame is flagged, not parented
        self.parent_of = {}
        self.children_of = defaultdict(list)
        self.nonadj_child = set()
        self.n_nonadjacent = 0
        for u_, v_ in np.asarray(gt_edges, int).reshape(-1, 2):
            u_, v_ = int(u_), int(v_)
            if int(gt_t[v_]) == int(gt_t[u_]) + 1:
                self.parent_of[v_] = u_
                self.children_of[u_].append(v_)
            else:
                self.nonadj_child.add(v_)
                self.n_nonadjacent += 1
        self.gt_by_t = defaultdict(list)
        for r in range(len(gt_t)):
            self.gt_by_t[int(gt_t[r])].append(r)
        # edgewhy's join: 1:1 nearest-first within MATCH_R, per frame
        self.m_seed = {}
        self.gt_of_seed = {}
        self.near_pool = {}
        self.gt_tree = {}
        for t, rows_ in self.gt_by_t.items():
            if t >= T or t < 0:
                continue
            g_um = self.gt_um[rows_]
            self.gt_tree[t] = cKDTree(g_um)
            P, _s = seeds[t]
            mm = _greedy_match(g_um, P * SCALE[None, :]) if len(P) else {}
            tr = cand_trees[t] if t < len(cand_trees) else None
            for k, r in enumerate(rows_):
                if k in mm:
                    self.m_seed[r] = (t, int(mm[k]))
                    self.gt_of_seed[(t, int(mm[k]))] = r
                self.near_pool[r] = bool(
                    tr is not None and tr.query(g_um[k])[0] <= MATCH_R)


def bucket_one(t, i, ctx, zero):
    """One primary (t >= 1, index i in seeds[t]) -> (bucket, sub, d_a, d_b,
    d_c, d_d, gt_parent_row or -1). `zero` selects the zero-velocity
    decision order (buckets 5-7) or the control's (linked_to_parent /
    linked_elsewhere)."""
    nan = float("nan")
    P, _ps = ctx.seeds[t - 1]
    C, _cs = ctx.seeds[t]
    this_um = np.asarray(C[i], float) * SCALE
    prev_i = int(ctx.prev_of[t][i])
    if zero and prev_i >= 0:
        # |v| == 0 with a predecessor: identical coordinates two frames
        # running. Counted, never folded into "no predecessor".
        return ("other", "linked_zero_disp", nan, nan, nan, nan, -1)
    r = ctx.gt_of_seed.get((t, i))
    if r is None:
        tr = ctx.gt_tree.get(t)
        d = float(tr.query(this_um)[0]) if tr is not None else nan
        return ("unmatched", "-", d, nan, nan, nan, -1)
    p = ctx.parent_of.get(r)
    if p is None:
        db, ax = border_dist(this_um, ctx.extent_um)
        sub = "border<10" if db < BORDER_UM else "interior"
        return ("gt_start", sub, db, float(ax),
                1.0 if r in ctx.nonadj_child else 0.0, nan, -1)
    par_um = ctx.gt_um[p]
    pp = ctx.m_seed.get(p)
    if pp is not None and pp[0] != t - 1:
        return ("other", "parent_wrong_frame", nan, nan, nan, nan, p)
    ip = int(pp[1]) if pp is not None else -1
    kids = ctx.children_of.get(p, [])
    if len(kids) >= 2:
        d_this = _norm(this_um - par_um)
        d_sis = nan
        if ip < 0:
            sub = "parent_undetected"
        else:
            took = False
            for s in kids:
                if s == r:
                    continue
                sp = ctx.m_seed.get(s)
                if sp is not None and sp[0] == t \
                        and int(ctx.prev_of[t][sp[1]]) == ip:
                    took = True
                    d_sis = _norm(np.asarray(C[sp[1]], float) * SCALE
                                  - par_um)
            if zero:
                if took:
                    sub = "sister_took_parent"
                elif bool((np.asarray(ctx.prev_of[t]) == ip).any()):
                    sub = "parent_to_other"
                else:
                    sub = "parent_unassigned"
            else:
                sub = ("linked_to_parent" if prev_i == ip
                       else "linked_elsewhere")
        return ("division_daughter", sub, d_this, d_sis, nan, nan, p)
    if ip < 0:
        sub = "subthresh" if ctx.near_pool.get(p) else "cold"
        return ("parent_undetected", sub, _norm(this_um - par_um),
                nan, nan, nan, p)
    ppos = np.asarray(P[ip], float) * SCALE
    d_raw = _norm(this_um - ppos)
    pred = ppos + 0.5 * np.asarray(ctx.vels[t - 1][ip], float)
    d_pred = _norm(this_um - pred)
    if not zero:
        d_prev = _norm(this_um - np.asarray(P[prev_i], float) * SCALE)
        b = "linked_to_parent" if prev_i == ip else "linked_elsewhere"
        return (b, "-", d_raw, d_prev, nan, nan, p)
    comp = np.where(np.asarray(ctx.prev_of[t]) == ip)[0]
    if len(comp):
        j = int(comp[0])
        c_um = np.asarray(C[j], float) * SCALE
        gtc = ctx.gt_um[r]
        rj = ctx.gt_of_seed.get((t, j))
        sub = "comp_unmatched" if rj is None else "comp_other_gt"
        return ("parent_taken", sub, _norm(c_um - gtc), _norm(this_um - gtc),
                _norm(c_um - ppos), d_raw, p)
    if d_raw > MAX_LINK_UM:
        return ("parent_out_of_gate", "beyond_max_link", d_raw, d_pred,
                nan, nan, p)
    sub = ("stage1_eligible" if d_raw <= min(TIGHT_UM, MAX_LINK_UM)
           else "stage2_eligible")
    return ("parent_free_but_unlinked", sub, d_raw, d_pred, nan, nan, p)


def final_fate(t, i, p, node_of_seed, nodes0, nodes_f, in_src, gt_um):
    """(in_final, has_parent, recovered). recovered = an edge into this
    primary's final node from a node at t-1 within MATCH_R of the GT
    parent (by position, so a repair node counts); nan without a GT
    parent. Positions of ILP nodes come from nodes0 (unshifted); repair
    nodes only exist in nodes_f."""
    nan = float("nan")
    g = node_of_seed.get((t, i))
    if g is None or g not in nodes_f:
        return (0.0, 0.0, nan)
    srcs = in_src.get(g, [])
    has_parent = 1.0 if srcs else 0.0
    if p is None or p < 0:
        return (1.0, has_parent, nan)
    rec = 0.0
    for s in srcs:
        n = nodes0.get(s) or nodes_f.get(s)
        if n is None or int(n["t"]) != t - 1:
            continue
        if _norm(np.asarray(n["xyz"], float) * SCALE - gt_um[p]) <= MATCH_R:
            rec = 1.0
    return (1.0, has_parent, rec)


def bucket_dataset(ctx, node_of_seed, nodes0, nodes_f, edges_f, rng):
    """Records for the zero group and the control sample of one dataset.
    Returns (zero_rec, ctrl_rec, n_prim_t1, n_zero)."""
    T = ctx.T
    in_src = defaultdict(list)
    for a_, b_ in edges_f:
        in_src[b_].append(a_)
    zero_idx, nz_idx = [], []
    for t in range(1, T):
        v = np.asarray(ctx.vels[t], float).reshape(-1, 3)
        mag = np.linalg.norm(v, axis=1) if len(v) else np.zeros(0)
        for i in range(len(mag)):
            (zero_idx if mag[i] == 0.0 else nz_idx).append((t, i))
    n_ctrl = min(len(zero_idx), len(nz_idx))
    pick = (rng.choice(len(nz_idx), size=n_ctrl, replace=False)
            if n_ctrl else np.zeros(0, int))
    ctrl_idx = [nz_idx[k] for k in pick]

    def _rows(idx, zero):
        out = []
        for (t, i) in idx:
            b, s, da, db, dc, dd, p = bucket_one(t, i, ctx, zero)
            inf, hp, rec = final_fate(t, i, p, node_of_seed, nodes0,
                                      nodes_f, in_src, ctx.gt_um)
            out.append((float(BUCKETS.index(b)), float(SUBS.index(s)),
                        da, db, dc, dd, 1.0 if p >= 0 else 0.0,
                        inf, hp, rec))
        return (np.asarray(out, np.float32) if out
                else np.zeros((0, len(RC)), np.float32))

    return (_rows(zero_idx, True), _rows(ctrl_idx, False),
            len(zero_idx) + len(nz_idx), len(zero_idx))


GROUPS = [("6bba", SET_6BBA), ("44b6", SET_44B6)] + \
    ([("zh", SET_ZH)] if SET_ZH else [])
DOMAINS = [f"F4-{g}" for g, _ in GROUPS]

print(f"\n=== NOPRED: {len(DOMAINS)} domains, "
      f"{sum(len(n) for _, n in GROUPS)} datasets, one solve each ===",
      flush=True)
print("  ANCHOR must reproduce F4-6bba adj_J 0.9216 / div_J 0.0606, F4-44b6 "
      "0.8353 / 0.0000, F4-zh 0.8118 / 0.0215 -- asserted at the very "
      "end. Per-dataset columns: primaries at t >= 1, zero-velocity count "
      "and share, then the zero group's buckets in decision order "
      "(unm gts div pund ptak gate free oth), adj_J, seconds.", flush=True)

ROWS = defaultdict(list)      # domain -> [official row]
PER = defaultdict(dict)       # domain -> {name: row}
RES = {}
ZREC = defaultdict(list)      # domain -> [zero-group records]
CREC = defaultdict(list)      # domain -> [control records]
NPRIM = defaultdict(int)      # domain -> primaries at t >= 1
NZERO = defaultdict(int)
NONADJ = defaultdict(int)
EXTENT_SRC = {}
INCOMPLETE = []
t_all = time.time()
GT_DIR = pathlib.Path("/kaggle/working/np_zh_gt")
GT_DIR.mkdir(parents=True, exist_ok=True)

for group, names in GROUPS:
    domain = f"F4-{group}"
    td0 = time.time()
    print(f"  -- {domain} --", flush=True)
    for k_ds, name in enumerate(names):
        if time.time() - t_all > BUDGET_S:
            INCOMPLETE.append((domain, name))
            continue
        t0 = time.time()
        gtg, n_gt = None, None
        if group == "zh":
            _db, T, gt_arr, gt_e, _nc = zh_load(name)
            _z4 = np.load(FDET_F4[name])
            dets = [(_z4[f"c{t}"], _z4[f"s{t}"]) for t in range(T)]
            del _db
            gp_gt = GT_DIR / f"{name}.geff"
            dr._save(str(gp_gt), gt_arr, gt_e)
            r_ = td.graph.IndexedRXGraph.from_geff(str(gp_gt))
            gtg = r_[0] if isinstance(r_, tuple) else r_
            n_gt = len(gt_arr)
            gt_t = gt_arr[:, 0].astype(int)
            gt_pos = gt_arr[:, 1:]
            gt_edges = np.asarray(gt_e, int)
        else:
            dets = load_dets(name, "F4")
            T = len(dets)
            gt = read_geff(geff_path(root, "train", name))
            gt_t = np.asarray(gt["t"], int)
            gt_pos = np.stack([np.asarray(gt[c], float)
                               for c in ("z", "y", "x")], axis=1)
            row_of = {int(v): i for i, v in enumerate(gt["node_ids"])}
            gt_edges = np.asarray(
                [(row_of[int(a_)], row_of[int(b_)])
                 for a_, b_ in gt["edges"]
                 if int(a_) in row_of and int(b_) in row_of], int)
        extent_um, ext_src = field_extent_um(group, name)
        EXTENT_SRC[domain] = ext_src

        seeds, vels_p1, prev_of = pass1_tracks(dets)
        feats = pair_features(dets, None, T)
        nodes0, edges0, pool = solve_one(dets, T, seeds, feats, MODEL, MCOLS)
        outp = pdir("ANCHOR", domain) / f"{name}.geff"
        nodes_f, edges_f = patch_final(nodes0, edges0, pool, outp)
        row = (score_zh_one(outp, gtg, n_gt) if group == "zh"
               else score_one(outp, name))
        ROWS[domain].append(row)
        PER[domain][name] = row

        ctx = GTContext(gt_t, gt_pos, gt_edges, T, seeds, vels_p1, prev_of,
                        pool[2], extent_um)
        node_of_seed, _son, n_dup, n_unm = seed_node_maps(seeds, nodes0, T)
        if n_dup:
            warn_once("seed_dup_voxel",
                      "two primaries rounded to the same voxel in one frame; "
                      "only one joins its solved node, the other reads as "
                      "'not in final' -> counted under n_dup")
        if n_unm:
            warn_once("node_unmapped",
                      "a solved node had no primary at its rounded "
                      "coordinates -> it cannot be a zero-velocity primary's "
                      "final node (counted under n_unmapped)")
        rng = np.random.default_rng(CTRL_SEED + 1000 * len(DOMAINS)
                                    + 100 * DOMAINS.index(domain) + k_ds)
        zr, cr, n_t1, n_zero = bucket_dataset(ctx, node_of_seed, nodes0,
                                              nodes_f, edges_f, rng)
        ZREC[domain].append(zr)
        CREC[domain].append(cr)
        NPRIM[domain] += n_t1
        NZERO[domain] += n_zero
        NONADJ[domain] += ctx.n_nonadjacent
        cnt = [int((zr[:, 0] == BUCKETS.index(b)).sum()) if len(zr) else 0
               for b in ZERO_BUCKETS]
        print(f"  {name:<14} prim {n_t1:>7,} zero {n_zero:>6,} "
              f"({100.0 * n_zero / max(1, n_t1):4.1f}%)  "
              + " ".join(f"{c:>5}" for c in cnt)
              + f"  adj_J {_adj_of(row):.4f}  {time.time() - t0:.0f}s",
              flush=True)
        del dets, seeds, feats, nodes0, edges0, pool, nodes_f, edges_f, ctx
        gc.collect()
    if ROWS[domain]:
        RES[domain] = _with_ratio(ROWS[domain])
    n_done = len(names) - sum(d == domain for d, _ in INCOMPLETE)
    print(f"  {domain}: {n_done} of {len(names)} datasets "
          f"({(time.time() - td0) / 60:.1f} min); field extent from "
          f"{EXTENT_SRC.get(domain, 'n/a')}", flush=True)
if INCOMPLETE:
    print(f"INCOMPLETE: {len(INCOMPLETE)} dataset-domains not attempted "
          f"(budget {BUDGET_S / 3600:.1f} h): {INCOMPLETE}", flush=True)

# ---- report -------------------------------------------------------------
N_OF = {f"F4-{g}": len(n) for g, n in GROUPS}
print("\n=== ANCHORS / official scorer ===", flush=True)
ANCHOR_FAIL = []
for d in DOMAINS:
    want, dv_want = ANCHOR_ADJ[d], ANCHOR_DIV[d]
    r = RES.get(d)
    if r is None or len(PER[d]) != N_OF[d]:
        print(f"  {d:<8} adj_J n/a (domain not fully scored)  want {want:.4f} "
              f"HARD", flush=True)
        ANCHOR_FAIL.append((d, float("nan"), want))
        continue
    got, dv_got = float(r["adj_edge_jaccard"]), float(r["division_jaccard"])
    ok = abs(got - want) <= ANCHOR_TOL
    dv_ok = abs(dv_got - dv_want) <= ANCHOR_TOL
    print(f"  {d:<8} score {r['score']:.4f}  adj_J {got:.4f} (want {want:.4f} "
          f"{'OK' if ok else 'MISS'} {got - want:+.4f})  div_J {dv_got:.4f} "
          f"(want {dv_want:.4f} {'OK' if dv_ok else 'MISS'})  HARD",
          flush=True)
    if not ok:
        ANCHOR_FAIL.append((d, got, want))
    if not dv_ok:
        ANCHOR_FAIL.append((d + " div_J", dv_got, dv_want))
if ANCHOR_FAIL:
    print("  ⚠ HARD ANCHOR MISS -- the tables below describe a pipeline "
          "nobody has scored before; the run ends in error after printing "
          "them.", flush=True)
else:
    print("  hard anchors reproduced: the harness is rankwhy's / twopass's.",
          flush=True)


def _cat(store, d):
    arrs = [a for a in store.get(d, []) if len(a)]
    return (np.concatenate(arrs, 0) if arrs
            else np.zeros((0, len(RC)), np.float32))


def _pct(a, b):
    return f"{100.0 * a / b:5.1f}%" if b else "   -  "


def _med(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return f"{np.median(x):6.2f}" if len(x) else "   nan"


def _rate(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return f"{100.0 * x.mean():5.1f}%" if len(x) else "   -  "


UNMATCHED_LABEL = {
    "F4-6bba": "UNVERIFIABLE (9.7% of cells annotated)",
    "F4-44b6": "UNVERIFIABLE (0.77% of cells annotated)",
    "F4-zh": "approx. SPURIOUS DETECTION (dense GT, ~97.5% annotated)",
}

print("\n=== ZERO-VELOCITY PRIMARIES AT t >= 1: THE BUCKETS ===", flush=True)
print("  Decision order as in the docstring; each primary lands in one "
      "bucket. `% zero` = share of the zero-velocity group, `% all` = share "
      "of every primary at t >= 1. `recovered` = the final graph has an "
      "edge into this primary from a node at t-1 within 7 um of the GT "
      "parent (buckets with a GT parent only). `in final` = the primary's "
      "node survived to the final graph.", flush=True)
for d in DOMAINS:
    Z = _cat(ZREC, d)
    n_all, n_zero = NPRIM[d], len(Z)
    print(f"\n  -- {d}: {n_all:,} primaries at t >= 1, {n_zero:,} with "
          f"velocity exactly zero ({_pct(n_zero, n_all).strip()}); "
          f"unmatched here = {UNMATCHED_LABEL.get(d, '?')}; "
          f"{NONADJ[d]:,} non-adjacent GT edges ignored --", flush=True)
    print(f"  {'bucket':<26} {'n':>8} {'% zero':>7} {'% all':>7} "
          f"{'in final':>9} {'recovered':>10}", flush=True)
    for b in ZERO_BUCKETS:
        m = Z[:, 0] == BUCKETS.index(b)
        n = int(m.sum())
        print(f"  {b:<26} {n:>8,} {_pct(n, n_zero):>7} {_pct(n, n_all):>7} "
              f"{_rate(Z[m, 7]):>9} {_rate(Z[m, 9]):>10}", flush=True)

print("\n=== SUB-SPLITS AND DISTANCES (zero-velocity group) ===", flush=True)
print("  Per bucket: the sub-split with counts and share of the bucket, "
      "then the medians (um) of the distances the bucket records, then "
      "recovery per sub-split. gt_start's z-face share = border cases "
      "whose nearest face is a z face (the field is a 104 um cube; z "
      "faces are where the light sheet's depth ends).", flush=True)
for d in DOMAINS:
    Z = _cat(ZREC, d)
    print(f"\n  -- {d} --", flush=True)
    for b in ZERO_BUCKETS:
        m = Z[:, 0] == BUCKETS.index(b)
        n = int(m.sum())
        if n == 0:
            print(f"  {b:<26} (empty)", flush=True)
            continue
        subs = np.unique(Z[m, 1]).astype(int)
        parts = []
        for s in subs:
            ms = m & (Z[:, 1] == s)
            parts.append(f"{SUBS[s]} {int(ms.sum()):,} "
                         f"({_pct(int(ms.sum()), n).strip()}, "
                         f"rec {_rate(Z[ms, 9]).strip()})")
        print(f"  {b:<26} n {n:>7,}  " + "; ".join(parts), flush=True)
        labs = DIST_LABELS.get(b, [])
        if labs:
            print("      medians: " + "; ".join(
                f"{lab} {_med(Z[m, 2 + k]).strip()}"
                for k, lab in enumerate(labs)), flush=True)
        if b == "gt_start":
            mb = m & (Z[:, 1] == SUBS.index("border<10"))
            zf = int(((Z[:, 3] == 0) & mb).sum())
            print(f"      border<10: z-face share {_pct(zf, int(mb.sum()))}; "
                  f"p10/p90 of border distance over the bucket "
                  f"{np.percentile(Z[m, 2], 10):.1f}/"
                  f"{np.percentile(Z[m, 2], 90):.1f} um; "
                  f"non-adjacent GT parent {int(Z[m, 4].sum()):,}",
                  flush=True)
        if b == "parent_taken":
            closer = int((Z[m, 2] < Z[m, 3]).sum())
            print(f"      competitor closer to the GT child than this cell: "
                  f"{closer:,} of {n:,} ({_pct(closer, n).strip()}); "
                  f"competitor closer to the parent primary: "
                  f"{int((Z[m, 4] < Z[m, 5]).sum()):,}", flush=True)
        if b == "parent_free_but_unlinked":
            print("      ⚠ expected 0 by construction of the Hungarian; a "
                  "non-zero count is a bug in this diagnostic", flush=True)

print("\n=== FINAL-GRAPH FATE of the zero-velocity primaries ===", flush=True)
print("  Does 'no predecessor in pass-1' even reach the score? in final = "
      "the node survived; with parent = it has an in-edge there (ILP edge, "
      "gap bridge or division patch); start = no in-edge. Same three "
      "numbers for the control sample.", flush=True)
print(f"  {'group':<8} {'domain':<8} {'n':>8} {'in final':>9} "
      f"{'with parent':>12} {'as start':>9} {'not in final':>13}",
      flush=True)
for d in DOMAINS:
    for label, store in (("zero", ZREC), ("control", CREC)):
        A = _cat(store, d)
        n = len(A)
        inf = A[:, 7] == 1
        print(f"  {label:<8} {d:<8} {n:>8,} {_pct(int(inf.sum()), n):>9} "
              f"{_pct(int((inf & (A[:, 8] == 1)).sum()), n):>12} "
              f"{_pct(int((inf & (A[:, 8] == 0)).sum()), n):>9} "
              f"{_pct(int((~inf).sum()), n):>13}", flush=True)

print("\n=== CONTROL: a random sample of NON-zero-velocity primaries ===",
      flush=True)
print("  Same size as the zero group per dataset, seeded. Buckets 1-4 as "
      "above; then whether pass-1's predecessor IS the GT parent's primary "
      "(linked_to_parent) or another primary (linked_elsewhere). What "
      "differs from the zero group is what is specific to having no "
      "predecessor.", flush=True)
for d in DOMAINS:
    Cc = _cat(CREC, d)
    n = len(Cc)
    print(f"\n  -- {d}: {n:,} control primaries --", flush=True)
    print(f"  {'bucket':<26} {'n':>8} {'% ctrl':>7} {'in final':>9} "
          f"{'recovered':>10}  sub-split / medians", flush=True)
    for b in CTRL_BUCKETS:
        m = Cc[:, 0] == BUCKETS.index(b)
        nb = int(m.sum())
        extra = ""
        if nb:
            subs = np.unique(Cc[m, 1]).astype(int)
            parts = [f"{SUBS[s]} {int((m & (Cc[:, 1] == s)).sum()):,}"
                     for s in subs if SUBS[s] != "-"]
            labs = DIST_LABELS.get(b, [])
            meds = [f"{lab} {_med(Cc[m, 2 + k]).strip()}"
                    for k, lab in enumerate(labs)]
            extra = "; ".join(parts + meds)
        print(f"  {b:<26} {nb:>8,} {_pct(nb, n):>7} {_rate(Cc[m, 7]):>9} "
              f"{_rate(Cc[m, 9]):>10}  {extra}", flush=True)

print("\n=== SIDE BY SIDE: zero group vs control, the rates that matter ===",
      flush=True)
print(f"  {'domain':<8} {'group':<8} {'unmatched':>10} {'gt_start':>9} "
      f"{'border<10':>10} {'division':>9} {'par undet':>10} "
      f"{'recovered':>10} {'in final':>9}", flush=True)
for d in DOMAINS:
    for label, store in (("zero", ZREC), ("control", CREC)):
        A = _cat(store, d)
        n = len(A)
        if not n:
            continue
        gs = A[:, 0] == BUCKETS.index("gt_start")
        print(f"  {d:<8} {label:<8} "
              f"{_pct(int((A[:, 0] == BUCKETS.index('unmatched')).sum()), n):>10} "
              f"{_pct(int(gs.sum()), n):>9} "
              f"{_pct(int((gs & (A[:, 1] == SUBS.index('border<10'))).sum()), n):>10} "
              f"{_pct(int((A[:, 0] == BUCKETS.index('division_daughter')).sum()), n):>9} "
              f"{_pct(int((A[:, 0] == BUCKETS.index('parent_undetected')).sum()), n):>10} "
              f"{_rate(A[:, 9]):>10} {_rate(A[:, 7]):>9}", flush=True)

print("\n=== reading it ===", flush=True)
print("  Buckets 1 and 2 (unmatched, gt_start) are cells that HAVE no "
      "predecessor in the data as annotated -- no linker can give them "
      "one, and a zero velocity there is correct. Buckets 3-7 are cells "
      "whose GT parent exists at t-1; there the zero is a pre-pass "
      "failure, and the recovery column says whether the ILP and the "
      "patches repaired it anyway. Bucket 5 (parent_taken) is the "
      "greedy assignment giving the parent to a competitor: the "
      "competitor's distances say whether that competitor was the "
      "better-looking choice (closer) or a spurious node (unmatched, "
      "only readable on zh). Bucket 4 (parent_undetected) is the "
      "detector, not the linker.", flush=True)
print("  ⚠ On 6bba / 44b6 'unmatched' cannot be read as spurious: the GT "
      "annotates a small fraction of the cells. zh is the only domain "
      "where unmatched ~ spurious.", flush=True)

warning_summary()

n_scored = sum(len(PER[d]) for d in DOMAINS)
n_expected = sum(len(n) for _, n in GROUPS)
print(f"\nNOPRED DONE in {(time.time() - t_all) / 60:.1f} min, {n_scored} of "
      f"{n_expected} datasets scored", flush=True)
assert not ANCHOR_FAIL, (
    f"hard anchor miss: {[(d, round(g, 4), w) for d, g, w in ANCHOR_FAIL]}"
    f" -- tolerance {ANCHOR_TOL}; do NOT read the buckets as a measurement")
assert len(DOMAINS) == 3, (
    f"only {len(DOMAINS)} domains present ({DOMAINS}); attach "
    f"biohub-zh-detect")
Zfree = sum(int((_cat(ZREC, d)[:, 0]
                 == BUCKETS.index("parent_free_but_unlinked")).sum())
            for d in DOMAINS)
if Zfree:
    print(f"  ⚠ parent_free_but_unlinked is {Zfree}, expected 0 -- the "
          f"gate reading in this diagnostic disagrees with _link_sim",
          flush=True)
if n_scored == n_expected and not INCOMPLETE:
    print("NOPRED OK", flush=True)
else:
    print(f"[degraded] NOPRED INCOMPLETE: {n_scored}/{n_expected} datasets "
          f"scored", flush=True)
