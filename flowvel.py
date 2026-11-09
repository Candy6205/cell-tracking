"""flowvel -- a velocity source OUTSIDE the linker: the motion of the
cell's neighbours. The residue twopass (2026-09-03) named and did not
test.

SCRIPT_VERSION 2026-09-03a-flowvel

THE QUESTION. The edge ranker's motion signal is a velocity per source
cell: columns 4 (distance from the predicted position P + v), 5
(raw distance minus that) and 16 (|v|). Every velocity comes from
`pass1_tracks`, the greedy per-frame Hungarian pre-pass, and a primary
the pre-pass could not give a predecessor has velocity EXACTLY ZERO --
12.7% / 8.4% / 4.2% of the primaries at t >= 1 in F4-6bba / 44b6 / zh
(twopass's velocity table). velwhy found 24.3% of 6bba's rank >= 1
misses sit on such cells. twopass tried the ILP's own tracks as the
source and found they are the SAME edges: the zero rate went UP (14.1
/ 11.1 / 5.3%) and adj_J moved by < 0.001 everywhere. So no linker
inside the pipeline can supply the missing velocities. The tissue can:
early embryos move coherently, and a cell whose own link failed still
sits among neighbours whose links did not.

NEIGHBOURHOOD FLOW, as implemented (`neighbourhood_flow`). For every
primary i at frame t: take the primaries of the SAME frame t (physical
distance, `coords * SCALE`, cKDTree) whose pass-1 velocity is non-zero,
excluding i itself; keep the FLOW_K = 8 nearest of them within
FLOW_R_UM = 25.0 um; if fewer than FLOW_K_MIN = 3 remain the flow is
UNDEFINED (kept zero, and flagged so no arm reads that zero as motion);
otherwise the flow is the COMPONENT-WISE median (z, y, x separately)
of those neighbours' pass-1 velocity vectors, in um/frame -- the same
units and the same `vels[t]` layout `pass1_tracks` produces, so
`refeature` consumes it unchanged. All three constants are module-level
so a later sweep can change them. Frame 0 has no velocities in any
source, so nothing is defined there.

ARMS (one ILP solve per dataset each, as in twopass):

  ANCHOR     today's pipeline: pass-1 velocities, D187 ranker, one
             solve, `patch_final`. Must reproduce the anchors.
  FLOWFILL   velocity = pass-1 where non-zero; neighbourhood flow where
             pass-1 is zero and the flow is defined; zero otherwise.
             Only columns 4 / 5 / 16 change, and only on the rows whose
             SOURCE is a filled cell (`check_flow_rows` asserts exactly
             that on every dataset). The direct test of the residue.
  FLOWALL    velocity = neighbourhood flow wherever defined, pass-1
             otherwise. Asks whether the crowd's motion predicts the
             next position better than the cell's own greedy link even
             where that link exists.
  FLOWBLEND  (flag, default False) 0.5 * pass-1 + 0.5 * flow where both
             exist; pass-1 alone where only it exists; flow alone where
             only it does (i.e. FLOWFILL on those cells).

Same D187 ranker (loaded, not retrained), same ILP constants (ILP_FLOOR
0.10, DIV_W 0.55, ILP_DISAPP 1.5, global solve), same `patch_final`,
same official scorer, same report blocks as twopass.

WHICH COLUMNS MOVE. twopass's `refeature` and its runtime proof
(`check_refeature`: rebuilding columns 4 / 5 / 16 from the pass-1
velocities must reproduce `pair_features`' own rows, asserted on every
dataset; 0.00e+00 in all three domains in biohub-twopass v1) are
carried verbatim. `solved_velocities` is carried too and run on
ANCHOR's solved graph (no extra solve) so the velocity table repeats
twopass's solved-track row as a harness cross-check.

⚠ DISTRIBUTION SHIFT, stated as twopass states it. D187 was trained on
pass-1 velocities and is applied unchanged to filled or replaced ones.
A negative result closes "the shipped ranker profits from
neighbourhood flow without retraining", not "neighbourhood flow is
useless"; the follow-up would be a ranker retrained with the flow
substituted before the label join. Two things soften the caveat for
FLOWFILL: it changes the input only on cells whose velocity was zero,
where the ranker's motion signal was absent rather than merely
different; and twopass measured the ranker under a changed velocity
input (solved-track velocities, same ranker) and adj_J moved by
< 0.001 in every domain -- though twopass also found those inputs
were nearly the same numbers, so that is weak evidence of tolerance,
not proof. FLOWALL is a larger shift: medians are smoother and
smaller than single-link velocities, so column 16 moves for almost
every row. Read FLOWALL with that in mind.

WHAT IS PRINTED. One line per dataset (adj_J per arm, solved node and
edge counts per arm). Per arm and domain: score / edge_J / node ratio
/ adj_J / div_J and the delta vs ANCHOR. Paired per-dataset adj_J
delta vs ANCHOR with up / down / same counts and the three-domain sign
verdict. The velocity table (zero rate, median |v|) for pass-1,
solved-track, and each arm's velocities. The FLOW STATISTICS table per
domain: primaries at t >= 1, how many got a defined flow, how many had
pass-1 velocity zero, how many FLOWFILL filled (zero AND defined), how
many FLOWALL replaced (= defined), median |flow| over the defined, and
the median cosine between the flow and the cell's own pass-1 velocity
where both are non-zero -- the sanity number: coherent tissue gives a
value well above 0, and a flow that does not agree with the cells
that DO have a link cannot be trusted on the cells that do not.
Solved and final node / edge counts per arm.

DOMAINS, ANCHORS, ASSEMBLY, KAGGLE: exactly twopass's (rankwhy's
harness, F4 primaries, D187). Anchors hard within ANCHOR_TOL 0.0005:
F4-6bba adj_J 0.9216 / div_J 0.0606, F4-44b6 0.8353 / 0.0000, F4-zh
0.8118 / 0.0215, asserted AFTER every table has printed. 44b6 is
in-sample for D187. Attach twopass's sources (kernel-metadata.json:
dataset_sources pilkwang/biohub-tracking-support-pack-50ep-v1,
faustine649/cell-tracking, faustine649/biohub-dctrain-175-weights;
kernel_sources faustine649/biohub-edge-ranker,
faustine649/biohub-cache-train, faustine649/biohub-zh-detect,
faustine649/biohub-poolsize, faustine649/biohub-fusecache;
competition_sources biohub-cell-tracking-during-development). CPU,
internet on. Push with `py kag.py kernels push -p <folder>`.

RUNTIME. twopass: 199.5 min for 3 solves x 32 datasets. Here the same
3 solves (4 with FLOWBLEND) plus one cKDTree query per frame: expect
~200 min. BUDGET_S 10 h guard, checked before each dataset; datasets
not attempted are listed as INCOMPLETE. Prints `FLOWVEL OK` only when
every arm scored every dataset and the hard anchors held.
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


SCRIPT_VERSION = "2026-09-03a-flowvel"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

# ---- flowvel switches -----------------------------------------------------
FLOW_K = 8                 # nearest non-zero-velocity neighbours used
FLOW_R_UM = 25.0           # ... within this physical radius (um)
FLOW_K_MIN = 3             # fewer than this -> the flow is undefined
FLOW_BLEND_W = 0.5         # FLOWBLEND: w * pass-1 + (1 - w) * flow
FLOWBLEND = False          # the fourth arm (one more solve per dataset)
VEL_COLS = (4, 5, 16)      # the 18-column ranker's velocity-dependent columns
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
print("  zebrahub: the third domain; neighbourhood flow is a candidate change and therefore a three-domain question.", flush=True)
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
# ---- twopass: velocities from the SOLVED tracks ---------------------------
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


def solved_velocities(seeds, nodes0, edges0, T):
    """Velocity INTO every primary at t >= 1 from the solved graph:
    position(t) - position(solved predecessor at t-1), in um -- exactly
    pass1_tracks' formula (`(C[ci] - P[pi]) * SCALE`, unrounded seed
    coordinates) with the ILP's edge in place of the greedy link. A
    primary with no solved predecessor keeps velocity ZERO, as an
    unlinked seed does in pass 1. Returns (vels, stats)."""
    node_of_seed, seed_of_node, n_dup, n_unmapped = \
        seed_node_maps(seeds, nodes0, T)
    if n_dup:
        warn_once("seed_dup_voxel",
                  "two primaries rounded to the same voxel in one frame; "
                  "only one of them can be joined to its solved node, the "
                  "other keeps velocity 0 -> counted under n_dup")
    if n_unmapped:
        warn_once("node_unmapped",
                  "a solved node had no primary at its rounded coordinates "
                  "-> it contributes no velocity (counted under n_unmapped)")
    vels = [np.zeros((len(seeds[t][0]), 3)) for t in range(T)]
    pred_seed = {}
    n_bad = 0
    for a, b in edges0:
        sa, sb = seed_of_node.get(a), seed_of_node.get(b)
        if sa is None or sb is None:
            continue
        if sb[0] != sa[0] + 1 or sb in pred_seed:
            # never under the harness (candidates are t -> t+1 only and
            # the ILP gives a node at most one parent); counted, not hidden
            n_bad += 1
            continue
        pred_seed[sb] = sa
    if n_bad:
        warn_once("solved_edge_shape",
                  f"{n_bad} solved edges were not a single t -> t+1 parent "
                  f"link and were ignored for the velocity")
    for (t1, j), (t0, i) in pred_seed.items():
        vels[t1][j] = (seeds[t1][0][j] - seeds[t0][0][i]) * SCALE
    n_t1 = int(sum(len(seeds[t][0]) for t in range(1, T)))
    n_in_graph = int(sum(1 for (t, _i) in node_of_seed if t >= 1))
    stats = {"n_t1": n_t1, "n_in_graph": n_in_graph,
             "n_with_pred": int(len(pred_seed)),
             "n_dup": int(n_dup), "n_unmapped": int(n_unmapped)}
    return vels, stats


def refeature(feats, vels, T, damp):
    """The first pass's feature rows with the three velocity columns
    rebuilt from `vels`: 4 md = |C_j - (P_i + damp * v_i)|, 5 d - md,
    16 |v_i|. Column 0 (d) is recomputed in float64 from the same
    coordinates so that d - md matches pair_features' arithmetic; every
    other column is left untouched. Pairs and their order are unchanged,
    which is what lets solve_one consume the result as-is."""
    out = []
    for t in range(T - 1):
        pairs, rows, P, C = feats[t]
        if not len(pairs):
            out.append((pairs, rows, P, C))
            continue
        rows2 = rows.copy()
        Pu = np.asarray(P, float) * SCALE[None, :]
        Cu = np.asarray(C, float) * SCALE[None, :]
        pred = Pu + damp * vels[t]
        ii = np.fromiter((i for i, _ in pairs), np.int64, len(pairs))
        jj = np.fromiter((j for _, j in pairs), np.int64, len(pairs))
        d = np.linalg.norm(Cu[jj] - Pu[ii], axis=1)
        md = np.linalg.norm(Cu[jj] - pred[ii], axis=1)
        rows2[:, 4] = md
        rows2[:, 5] = d - md
        rows2[:, 16] = np.linalg.norm(vels[t][ii], axis=1)
        out.append((pairs, rows2, P, C))
    return out


def check_refeature(feats, vels_p1, T, name):
    """refeature with the PASS-1 velocities at damp 1.0 must reproduce
    pair_features' own columns 4/5/16 -- the proof that those are the
    only columns the velocity reaches, on every dataset."""
    chk = refeature(feats, vels_p1, T, 1.0)
    worst = 0.0
    for t in range(T - 1):
        rows, rows2 = feats[t][1], chk[t][1]
        if not len(rows):
            continue
        worst = max(worst, float(np.abs(rows2 - rows).max()))
        same = [c for c in range(NFEAT_BASE) if c not in VEL_COLS]
        assert np.array_equal(rows2[:, same], rows[:, same]), \
            f"{name} t{t}: refeature touched a non-velocity column"
    assert worst <= 2e-3, (
        f"{name}: refeature with pass-1 velocities differs from "
        f"pair_features by {worst:.3g} in columns {VEL_COLS} -- the column "
        f"identification is wrong, do not read the arms")
    return worst


def vel_mags(vels, T):
    """|v| of every primary at t >= 1 (frame 0 is zero by construction in
    both passes and is left out of the rates)."""
    if T < 2:
        return np.zeros(0)
    V = np.concatenate([np.asarray(vels[t], float).reshape(-1, 3)
                        for t in range(1, T)], 0)
    return np.linalg.norm(V, axis=1) if len(V) else np.zeros(0)


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
    d = pathlib.Path(f"/kaggle/working/fv_{domain}_{arm}")
    d.mkdir(parents=True, exist_ok=True)
    return d


# ==========================================================================
# ---- flowvel: velocity from the NEIGHBOURHOOD, not from a linker ---------
# ==========================================================================
def neighbourhood_flow(seeds, vels, T):
    """Per frame t and primary i: the component-wise median of the pass-1
    velocity vectors of the FLOW_K nearest OTHER primaries of frame t
    (physical distance) that have a non-zero velocity, within
    FLOW_R_UM; undefined (zero, defined[t][i] False) with fewer than
    FLOW_K_MIN such neighbours. Returns (flow, defined) in `vels`'
    layout: flow[t] is (n_t, 3) um/frame, defined[t] is (n_t,) bool."""
    flow = [np.zeros((len(seeds[t][0]), 3)) for t in range(T)]
    defined = [np.zeros(len(seeds[t][0]), bool) for t in range(T)]
    for t in range(T):
        P, _s = seeds[t]
        n = len(P)
        if n == 0:
            continue
        V = np.asarray(vels[t], float).reshape(n, 3)
        idx_nz = np.flatnonzero(np.any(V != 0.0, axis=1))
        if len(idx_nz) < FLOW_K_MIN:
            continue
        Pu = np.asarray(P, float) * SCALE[None, :]
        tree = cKDTree(Pu[idx_nz])
        # +1 because the query may return the cell itself (a non-zero
        # cell is in the tree at distance 0); it is dropped below
        kq = min(FLOW_K + 1, len(idx_nz))
        dist, ind = tree.query(Pu, k=kq, distance_upper_bound=FLOW_R_UM)
        dist = np.asarray(dist, float).reshape(n, kq)
        ind = np.asarray(ind, np.int64).reshape(n, kq)
        valid = np.isfinite(dist)          # beyond R: dist inf, ind == len
        nb = idx_nz[np.minimum(ind, len(idx_nz) - 1)]
        valid &= nb != np.arange(n)[:, None]        # not its own neighbour
        # query results are sorted by distance: keep the nearest FLOW_K
        valid &= np.cumsum(valid, axis=1) <= FLOW_K
        ok = valid.sum(axis=1) >= FLOW_K_MIN
        if not ok.any():
            continue
        Vn = np.where(valid[:, :, None], V[nb], np.nan)
        flow[t][ok] = np.nanmedian(Vn[ok], axis=1)
        defined[t] = ok
    return flow, defined


def compose_velocity(vels_p1, flow, defined, T, mode):
    """The arm's velocity field from pass-1 and the flow. Returns (vels,
    n_changed): n_changed counts the primaries (all frames) whose
    velocity differs from pass-1 by construction of the rule -- filled
    (FLOWFILL), defined (FLOWALL), defined (FLOWBLEND)."""
    out = []
    n_changed = 0
    for t in range(T):
        V = np.asarray(vels_p1[t], float).reshape(-1, 3)
        F = np.asarray(flow[t], float).reshape(-1, 3)
        D = np.asarray(defined[t], bool)
        nz = np.any(V != 0.0, axis=1)
        Vo = V.copy()
        if mode == "FLOWFILL":
            m = D & ~nz
            Vo[m] = F[m]
        elif mode == "FLOWALL":
            m = D
            Vo[m] = F[m]
        elif mode == "FLOWBLEND":
            both = D & nz
            Vo[both] = FLOW_BLEND_W * V[both] + (1.0 - FLOW_BLEND_W) * F[both]
            fill = D & ~nz
            Vo[fill] = F[fill]
            m = D
        else:
            raise ValueError(mode)
        n_changed += int(m.sum())
        out.append(Vo)
    return out, n_changed


def flow_stats(seeds, vels_p1, flow, defined, T):
    """Counts over the primaries at t >= 1 (frame 0 is zero in every
    source, as in the velocity table) plus the |flow| values over the
    defined ones and the cosine(flow, pass-1 v) where both are non-zero."""
    st = defaultdict(int)
    mags, coss = [], []
    for t in range(1, T):
        V = np.asarray(vels_p1[t], float).reshape(-1, 3)
        F = np.asarray(flow[t], float).reshape(-1, 3)
        D = np.asarray(defined[t], bool)
        nz = np.any(V != 0.0, axis=1)
        st["n_t1"] += len(V)
        st["n_defined"] += int(D.sum())
        st["n_p1_zero"] += int((~nz).sum())
        st["n_fill"] += int((D & ~nz).sum())
        if D.any():
            mags.append(np.linalg.norm(F[D], axis=1))
        both = D & nz
        if both.any():
            fn = np.linalg.norm(F[both], axis=1)
            vn = np.linalg.norm(V[both], axis=1)
            good = fn > EPS
            if good.any():
                coss.append((F[both][good] * V[both][good]).sum(1)
                            / (fn[good] * vn[good]))
    return st, (np.concatenate(mags) if mags else np.zeros(0)), \
        (np.concatenate(coss) if coss else np.zeros(0))


def check_flow_rows(feats, feats2, vels_p1, vels_arm, T, name, arm):
    """An arm's refeatured rows must differ from the first pass's rows
    ONLY in VEL_COLS and ONLY where the source's velocity was changed;
    rows whose source kept its pass-1 velocity must be reproduced
    (within check_refeature's 2e-3, float32 round-off of a float64
    recompute). Returns (rows changed, rows total)."""
    n_ch, n_all = 0, 0
    same = [c for c in range(NFEAT_BASE) if c not in VEL_COLS]
    for t in range(T - 1):
        pairs, rows, P, C = feats[t]
        rows2 = feats2[t][1]
        if not len(pairs):
            continue
        ii = np.fromiter((i for i, _ in pairs), np.int64, len(pairs))
        moved = np.any(np.asarray(vels_arm[t])[ii]
                       != np.asarray(vels_p1[t])[ii], axis=1)
        assert np.array_equal(rows2[:, same], rows[:, same]), \
            f"{name} {arm} t{t}: a non-velocity column changed"
        assert rows2.shape[1] == rows.shape[1] and \
            np.array_equal(rows2[:, NFEAT_BASE:], rows[:, NFEAT_BASE:]), \
            f"{name} {arm} t{t}: an acceleration column changed"
        if (~moved).any():
            worst = float(np.abs(rows2[~moved][:, list(VEL_COLS)]
                                 - rows[~moved][:, list(VEL_COLS)]).max())
            assert worst <= 2e-3, (
                f"{name} {arm} t{t}: a row whose source kept its pass-1 "
                f"velocity changed by {worst:.3g} in {VEL_COLS}")
        n_ch += int(moved.sum())
        n_all += len(pairs)
    return n_ch, n_all


ARM_LIST = ["ANCHOR", "FLOWFILL", "FLOWALL"] + \
    (["FLOWBLEND"] if FLOWBLEND else [])
GROUPS = [("6bba", SET_6BBA), ("44b6", SET_44B6)] + \
    ([("zh", SET_ZH)] if SET_ZH else [])
DOMAINS = [f"F4-{g}" for g, _ in GROUPS]
VEL_SRC = ["pass1", "solved"] + [a.lower() for a in ARM_LIST[1:]]

print(f"\n=== FLOWVEL: {len(ARM_LIST)} arms x {len(DOMAINS)} domains, "
      f"{sum(len(n) for _, n in GROUPS)} datasets, {len(ARM_LIST)} solves "
      f"per dataset ===", flush=True)
print(f"  flow: median pass-1 velocity of the {FLOW_K} nearest non-zero "
      f"neighbours within {FLOW_R_UM} um, undefined below {FLOW_K_MIN}; "
      f"FLOWBLEND {'on' if FLOWBLEND else 'off'} (w {FLOW_BLEND_W})",
      flush=True)
print("  ANCHOR must reproduce F4-6bba adj_J 0.9216 / div_J 0.0606, F4-44b6 "
      "0.8353 / 0.0000, F4-zh 0.8118 / 0.0215 (rankwhy, velwhy, twopass) -- "
      "asserted at the very end. FLOW* re-score the SAME D187 ranker on "
      "velocities partly or wholly taken from the neighbourhood (a "
      "distribution shift for the ranker, see the docstring) and "
      "re-solve. Reading: paired adj_J delta vs ANCHOR, same sign in all "
      "three domains.", flush=True)
print("  per-dataset columns: adj_J per arm, then solved nodes and solved "
      "edges per arm (" + "/".join(ARM_LIST) + "), then FLOWFILL's filled "
      "primaries (all frames)", flush=True)

ROWS = defaultdict(list)     # (arm, domain) -> [row]
PER = defaultdict(dict)      # (arm, domain) -> {name: row}
RES = {}
COUNTS = defaultdict(lambda: [0, 0, 0, 0])   # (arm, domain) -> solved n/e, final n/e
VMAG = defaultdict(list)     # (src, domain) -> [|v| arrays]
VSTAT = defaultdict(lambda: defaultdict(int))   # (src, domain) -> counters
FSTAT = defaultdict(lambda: defaultdict(int))   # domain -> flow counters
FMAG = defaultdict(list)     # domain -> [|flow| arrays]
FCOS = defaultdict(list)     # domain -> [cos arrays]
NCHG = defaultdict(int)      # (arm, domain) -> primaries changed (all frames)
RCHG = defaultdict(lambda: [0, 0])   # (arm, domain) -> rows changed / total
REFEAT_WORST = defaultdict(float)
INCOMPLETE = []
t_all = time.time()
GT_DIR = pathlib.Path("/kaggle/working/fv_zh_gt")
GT_DIR.mkdir(parents=True, exist_ok=True)

for group, names in GROUPS:
    domain = f"F4-{group}"
    td0 = time.time()
    print(f"  -- {domain} --  " + " ".join(f"{a:>13}" for a in ARM_LIST),
          flush=True)
    for name in names:
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
        else:
            dets = load_dets(name, "F4")
            T = len(dets)

        def _score(gp):
            return (score_zh_one(gp, gtg, n_gt) if group == "zh"
                    else score_one(gp, name))

        seeds, vels_p1, _prev = pass1_tracks(dets)
        feats = pair_features(dets, None, T)
        REFEAT_WORST[domain] = max(REFEAT_WORST[domain],
                                   check_refeature(feats, vels_p1, T, name))
        VMAG[("pass1", domain)].append(vel_mags(vels_p1, T))
        VSTAT[("pass1", domain)]["n_t1"] += int(
            sum(len(seeds[t][0]) for t in range(1, T)))

        # ---- the neighbourhood flow, once per dataset ------------------
        flow, fdef = neighbourhood_flow(seeds, vels_p1, T)
        fst, fmag, fcos = flow_stats(seeds, vels_p1, flow, fdef, T)
        for k_, v_ in fst.items():
            FSTAT[domain][k_] += v_
        FMAG[domain].append(fmag)
        FCOS[domain].append(fcos)
        varm = {}
        for arm in ARM_LIST[1:]:
            varm[arm], n_ch = compose_velocity(vels_p1, flow, fdef, T, arm)
            NCHG[(arm, domain)] += n_ch
            VMAG[(arm.lower(), domain)].append(vel_mags(varm[arm], T))
        n_fill_ds = sum(int((np.asarray(fdef[t])
                             & ~np.any(np.asarray(vels_p1[t]) != 0.0, axis=1)
                             ).sum()) for t in range(T))

        cells, ncell, ecell = [], [], []
        for arm in ARM_LIST:
            if arm == "ANCHOR":
                f_use = feats
            else:
                f_use = refeature(feats, varm[arm], T, 1.0)
                n_ch, n_all = check_flow_rows(feats, f_use, vels_p1,
                                              varm[arm], T, name, arm)
                RCHG[(arm, domain)][0] += n_ch
                RCHG[(arm, domain)][1] += n_all
            nodes0, edges0, pool = solve_one(dets, T, seeds, f_use,
                                             MODEL, MCOLS)
            if arm == "ANCHOR":
                # twopass's solved-track velocities from the SAME solve:
                # no extra ILP, and the velocity table gets twopass's row
                # back as a cross-check of the harness
                v_s, st = solved_velocities(seeds, nodes0, edges0, T)
                VMAG[("solved", domain)].append(vel_mags(v_s, T))
                for k_, v_ in st.items():
                    VSTAT[("solved", domain)][k_] += v_
                del v_s
            outp = pdir(arm, domain) / f"{name}.geff"
            nodes_f, edges_f = patch_final(nodes0, edges0, pool, outp)
            row = _score(outp)
            ROWS[(arm, domain)].append(row)
            PER[(arm, domain)][name] = row
            c_ = COUNTS[(arm, domain)]
            c_[0] += len(nodes0)
            c_[1] += len(edges0)
            c_[2] += len(nodes_f)
            c_[3] += len(edges_f)
            cells.append(f"{_adj_of(row):>13.4f}")
            ncell.append(str(len(nodes0)))
            ecell.append(str(len(edges0)))
            del pool, nodes_f, edges_f, nodes0, edges0, f_use
        print(f"  {name:<14} " + " ".join(cells)
              + f"  n {'/'.join(ncell)}  e {'/'.join(ecell)}"
              + f"  fill {n_fill_ds}  {time.time() - t0:.0f}s", flush=True)
        del dets, seeds, feats, vels_p1, flow, fdef, varm
        gc.collect()
    for arm in ARM_LIST:
        if ROWS[(arm, domain)]:
            RES[(arm, domain)] = _with_ratio(ROWS[(arm, domain)])
    n_done = len(names) - sum(d == domain for d, _ in INCOMPLETE)
    print(f"  {domain}: {n_done} of {len(names)} datasets x {len(ARM_LIST)} "
          f"arms ({(time.time() - td0) / 60:.1f} min)", flush=True)
if INCOMPLETE:
    print(f"INCOMPLETE: {len(INCOMPLETE)} dataset-domains not attempted "
          f"(budget {BUDGET_S / 3600:.1f} h): {INCOMPLETE}", flush=True)

# ---- report -------------------------------------------------------------
N_OF = {f"F4-{g}": len(n) for g, n in GROUPS}
print("\n=== ANCHORS / official scorer (ANCHOR arm) ===", flush=True)
ANCHOR_FAIL = []
for d in DOMAINS:
    want, dv_want = ANCHOR_ADJ[d], ANCHOR_DIV[d]
    r = RES.get(("ANCHOR", d))
    if r is None or len(PER[("ANCHOR", d)]) != N_OF[d]:
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

print("\n=== PER ARM, per domain (official scorer, weighted as adj_J is) ===",
      flush=True)
print(f"  {'arm':<13} {'domain':<8} {'score':>7} {'edge_J':>7} {'node_r':>8} "
      f"{'adj_J':>7} {'div_J':>7} {'d adj_J':>8} {'d div_J':>8}", flush=True)
for arm in ARM_LIST:
    for d in DOMAINS:
        r = RES.get((arm, d))
        b = RES.get(("ANCHOR", d))
        if r is None:
            continue
        da = (f"{r['adj_edge_jaccard'] - b['adj_edge_jaccard']:+8.4f}"
              if b else f"{'n/a':>8}")
        dv = (f"{r['division_jaccard'] - b['division_jaccard']:+8.4f}"
              if b else f"{'n/a':>8}")
        print(f"  {arm:<13} {d:<8} {r['score']:7.4f} "
              f"{r.get('edge_jaccard', float('nan')):7.4f} "
              f"{r.get('total_node_ratio', float('nan')):+8.4f} "
              f"{r['adj_edge_jaccard']:7.4f} {r['division_jaccard']:7.4f} "
              f"{da} {dv}", flush=True)

print("\n=== PAIRED per-dataset adj_J delta vs ANCHOR ===", flush=True)
print("  mean / min / max over datasets, and how many went up / down / "
      "unchanged. A domain mean carried by one dataset is noise, not a "
      "mechanism. The verdict is the sign in ALL THREE domains; 44b6 is "
      "in-sample for D187.", flush=True)
VERDICT = {}
for arm in ARM_LIST[1:]:
    signs = []
    for d in DOMAINS:
        base, mine = PER[("ANCHOR", d)], PER[(arm, d)]
        common = [n for n in base if n in mine]
        if not common:
            continue
        dl = np.array([_adj_of(mine[n]) - _adj_of(base[n]) for n in common])
        up, dn = int((dl > 1e-9).sum()), int((dl < -1e-9).sum())
        zero = len(dl) - up - dn
        gd = (RES[(arm, d)]["adj_edge_jaccard"]
              - RES[("ANCHOR", d)]["adj_edge_jaccard"])
        signs.append(np.sign(round(gd, 4)))
        print(f"  {arm:<13} {d:<8} domain {gd:+.4f}  per-dataset mean "
              f"{dl.mean():+.4f} min {dl.min():+.4f} max {dl.max():+.4f}  "
              f"up {up} down {dn} same {zero} of {len(dl)}", flush=True)
    if len(signs) == len(DOMAINS) and len(DOMAINS) == 3:
        if all(s > 0 for s in signs):
            v = "all three domains POSITIVE"
        elif all(s < 0 for s in signs):
            v = "all three domains NEGATIVE"
        elif all(s == 0 for s in signs):
            v = "no change in any domain"
        else:
            v = "signs DISAGREE across domains"
    elif len(signs) == len(DOMAINS):
        v = f"only {len(DOMAINS)} domains present -- the rule needs three"
    else:
        v = "not fully scored"
    VERDICT[arm] = v
    print(f"  {arm:<13} verdict: {v}", flush=True)

print("\n=== VELOCITIES: pass-1 (greedy Hungarian), solved tracks, and the "
      "flow arms ===", flush=True)
print("  Over every primary at t >= 1 (frame 0 is zero in every source). "
      "`zero` = velocity EXACTLY zero. `solved` is twopass's row rebuilt "
      "from ANCHOR's solve (no extra ILP; a harness cross-check -- twopass "
      "v1 read 14.1 / 11.1 / 5.3 % zero). For the solved tracks the "
      "no-predecessor primaries split into `not in graph` and `in graph, "
      "no parent`. median |v| in um/frame over all primaries and over the "
      "non-zero ones.", flush=True)
print(f"  {'source':<9} {'domain':<8} {'primaries':>10} {'zero':>9} "
      f"{'zero%':>6} {'not in graph':>13} {'no parent':>10} "
      f"{'med|v| all':>11} {'med|v| nz':>10} {'dup/unmap':>10}", flush=True)
for src in VEL_SRC:
    for d in DOMAINS:
        arrs = VMAG.get((src, d))
        if not arrs:
            continue
        m = np.concatenate(arrs) if arrs else np.zeros(0)
        n = len(m)
        nz = m > 0
        st = VSTAT[(src, d)]
        if src != "solved":
            notin, nopar, dup = "-", "-", "-"
        else:
            notin = f"{st['n_t1'] - st['n_in_graph']:,}"
            nopar = f"{st['n_in_graph'] - st['n_with_pred']:,}"
            dup = f"{st['n_dup']}/{st['n_unmapped']}"
        print(f"  {src:<9} {d:<8} {n:>10,} {int((~nz).sum()):>9,} "
              f"{100.0 * (~nz).mean() if n else float('nan'):>6.1f} "
              f"{notin:>13} {nopar:>10} "
              f"{np.median(m) if n else float('nan'):>11.2f} "
              f"{np.median(m[nz]) if nz.any() else float('nan'):>10.2f} "
              f"{dup:>10}", flush=True)
print("  refeature self-check (max |diff| vs pair_features on pass-1 "
      "velocities, columns 4/5/16): "
      + "  ".join(f"{d} {REFEAT_WORST[d]:.2e}" for d in DOMAINS
                  if d in REFEAT_WORST), flush=True)

print(f"\n=== FLOW STATISTICS (K {FLOW_K}, R {FLOW_R_UM} um, K_MIN "
      f"{FLOW_K_MIN}) ===", flush=True)
print("  Over the primaries at t >= 1. `defined` = at least K_MIN non-zero "
      "neighbours in R; `p1 zero` = pass-1 velocity exactly zero; `filled` "
      "= zero AND defined, the cells FLOWFILL changes; FLOWALL changes "
      "every defined one. med|flow| over the defined; med cos = median "
      "cosine between the flow and the cell's own pass-1 velocity where "
      "both are non-zero (coherent tissue: well above 0). `rows changed` = "
      "ranker rows whose source velocity the arm changed.", flush=True)
print(f"  {'domain':<8} {'primaries':>10} {'defined':>9} {'def%':>6} "
      f"{'p1 zero':>9} {'filled':>8} {'fill%zero':>9} {'med|flow|':>10} "
      f"{'med cos':>8} {'n cos':>8}", flush=True)
for d in DOMAINS:
    st = FSTAT.get(d)
    if not st:
        continue
    fm = np.concatenate(FMAG[d]) if FMAG[d] else np.zeros(0)
    fc = np.concatenate(FCOS[d]) if FCOS[d] else np.zeros(0)
    n = st["n_t1"]
    print(f"  {d:<8} {n:>10,} {st['n_defined']:>9,} "
          f"{100.0 * st['n_defined'] / n if n else float('nan'):>6.1f} "
          f"{st['n_p1_zero']:>9,} {st['n_fill']:>8,} "
          f"{100.0 * st['n_fill'] / st['n_p1_zero'] if st['n_p1_zero'] else float('nan'):>9.1f} "
          f"{np.median(fm) if len(fm) else float('nan'):>10.2f} "
          f"{np.median(fc) if len(fc) else float('nan'):>8.3f} "
          f"{len(fc):>8,}", flush=True)
print(f"  {'arm':<13} {'domain':<8} {'primaries changed':>18} "
      f"{'rows changed':>13} {'rows total':>11} {'rows%':>6}", flush=True)
for arm in ARM_LIST[1:]:
    for d in DOMAINS:
        if (arm, d) not in NCHG:
            continue
        rc = RCHG[(arm, d)]
        print(f"  {arm:<13} {d:<8} {NCHG[(arm, d)]:>18,} {rc[0]:>13,} "
              f"{rc[1]:>11,} "
              f"{100.0 * rc[0] / rc[1] if rc[1] else float('nan'):>6.1f}",
              flush=True)
print("  (primaries changed counts ALL frames, the table above t >= 1; "
      "rows = ranker candidate pairs, sources at t <= T-2)", flush=True)

print("\n=== NODES AND EDGES per arm (summed over the domain) ===", flush=True)
print("  solved = solve_one's output (before the patches); final = after "
      "_gap_close / _short_filter / _linefit / _add_safe_divisions, what "
      "the scorer read. The ILP may keep a different node set on the "
      "re-solve, and adj_J charges node count through total_node_ratio.",
      flush=True)
print(f"  {'arm':<13} {'domain':<8} {'solved n':>10} {'solved e':>10} "
      f"{'final n':>10} {'final e':>10} {'d solved n':>11} {'d solved e':>11}",
      flush=True)
for arm in ARM_LIST:
    for d in DOMAINS:
        c_ = COUNTS.get((arm, d))
        if c_ is None:
            continue
        b_ = COUNTS.get(("ANCHOR", d), c_)
        print(f"  {arm:<13} {d:<8} {c_[0]:>10,} {c_[1]:>10,} {c_[2]:>10,} "
              f"{c_[3]:>10,} {c_[0] - b_[0]:>+11,} {c_[1] - b_[1]:>+11,}",
              flush=True)

print("\n=== reading it ===", flush=True)
print("  The claim under test: the shipped D187 ranker, given a velocity "
      "from the cell's neighbours where the greedy pre-pass gave none "
      "(FLOWFILL), ranks the correct successor first more often, and the "
      "re-solve turns that into adj_J. FLOWALL asks the stronger question "
      "-- is the crowd's motion a better predictor than the cell's own "
      "link everywhere. The verdict is the paired delta vs ANCHOR with "
      "the same sign in all three domains.", flush=True)
print("  ⚠ The ranker was trained on pass-1 velocities and is applied to "
      "filled / replaced ones unchanged. A NEGATIVE result closes only "
      "'the shipped ranker profits from neighbourhood flow without "
      "retraining'. twopass measured the same ranker under solved-track "
      "velocities and adj_J moved < 0.001 in every domain -- but those "
      "inputs were nearly the same numbers, so that is weak evidence the "
      "ranker tolerates a shift, not proof. The flow statistics say how "
      "many cells each arm touched and whether the flow agrees with the "
      "links that exist (med cos); if med cos is near 0 the flow is not "
      "coherent motion and the arms tested noise.", flush=True)

warning_summary()
_gl = dict(_GAP_LOG)
if _gl:
    print(f"  gap bridges (made/allowed/demand/capped): "
          f"{[(k, v) for k, v in _gl.items()][:4]}", flush=True)

n_scored = sum(len(PER[(a, d)]) for a in ARM_LIST for d in DOMAINS)
n_expected = len(ARM_LIST) * sum(len(n) for _, n in GROUPS)
print(f"\nFLOWVEL DONE in {(time.time() - t_all) / 60:.1f} min, {n_scored} of "
      f"{n_expected} arm-datasets scored", flush=True)
assert not ANCHOR_FAIL, (
    f"hard anchor miss: {[(d, round(g, 4), w) for d, g, w in ANCHOR_FAIL]}"
    f" -- tolerance {ANCHOR_TOL}; do NOT read the deltas as a measurement")
assert len(DOMAINS) == 3, (
    f"only {len(DOMAINS)} domains present ({DOMAINS}); the three-domain "
    f"rule was not met -- attach biohub-zh-detect")
if n_scored == n_expected and not INCOMPLETE:
    print("FLOWVEL OK", flush=True)
else:
    print(f"[degraded] FLOWVEL INCOMPLETE: {n_scored}/{n_expected} "
          f"arm-datasets scored", flush=True)
