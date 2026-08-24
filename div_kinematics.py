"""div_kinematics.py -- can the mother trajectory tell a real
division from a coincidence?

The user point, 2026-08-24: a division is a physical event that perturbs
motion, and motion is a derivative -- it cannot be seen inside one frame.
triple_feats has exactly ONE temporal number, the mother speed magnitude
from a single previous transition, direction discarded.

Recorded diagnosis this extends (progress.md, gate round 3): "the model
has no way to recognise the rare configuration where a long edge IS a
second daughter, because it scores each edge independently and a division
is a property of the triple". The triple model was the first half. Time is
the second.

Three column views over ONE feature build:

  BASE  the 25 columns div_train used, verbatim
  HIST  + 10 columns of the mother OWN past: speed now vs recent mean
        (mitotic slow-down), speed trend, path straightness, curvature,
        heading stability, detection-score trend
  FULL  + 4 columns coupling that history to the candidate pair, chiefly
        whether the daughters CENTROID carries the mother momentum
        forward

Trained on Zebrahub only (2001 positives; the competition split holds 18
and trains to 0.43, below chance), validated on held-out 6bba exactly as
div_train v3 did, so BASE is comparable to its recorded 0.8083.

Reported: ROC-AUC, average precision, and FALSE DIVISIONS PER TRUE ONE at
fixed recall -- the metric charges for false divisions and the budget is
about 7 per TP, so AP and FP/TP decide this, not AUC. Then a permutation
pass naming which physical signal carries it, and a model-free comparison
of the medians at true divisions against everything else.

No detection re-runs. Nothing here touches a submission.
"""
from __future__ import annotations

import gc
import pathlib
import pickle
import subprocess
import sys
import time
from collections import defaultdict

SCRIPT_VERSION = "2026-08-24b-div-kinematics"
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
SHORT_MIN = 6
LINEFIT_WEIGHT = 0.8
LINEFIT_WINDOW = 2
MAX_LINK_UM = 10.0
TIGHT_UM = 6.0
LINK_SIM_W = 2.0
GAP1_GATE_UM = 9.0
GAP1_SNAP_UM = 3.2
GAP1_MIN_CAND_SCORE = 0.10
GAP1_CAP_FRAC = 0.003
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


def load_dets(name):
    z = np.load(CACHES[name])
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


from sklearn.metrics import average_precision_score    # noqa: E402
root = find_root(None)
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




# ---- the trajectory, BEFORE and AFTER the candidate division -----------
# User's point, 2026-08-24: a division is a physical event that perturbs
# motion, and motion is a DERIVATIVE -- it does not exist inside one frame.
# triple_feats carries exactly ONE temporal number, the mother's speed
# MAGNITUDE (col 19), from a single previous transition, direction thrown
# away by np.linalg.norm.
#
# Two directions in time, and the user was right to push back on doing
# only the cheap one:
#   BACKWARD  the mother's own past. Free and unambiguous -- before t she
#             is ONE cell. But mitotic rounding is brief, and if the frame
#             interval is coarse relative to it the slow-down may not be
#             sampled at all.
#   FORWARD   what the two daughters then do. Costs a forward chain and is
#             noisier, but the divergence after a split lasts several
#             frames, so it is far more likely to be sampled.
# Not a leak: this is offline tracking over a finished movie and the
# solver already optimises globally over all frames.

HIST_K = 5          # frames of mother history to walk back
FWD_K = 5           # frames of daughter future to walk forward
N_HIST = 10
N_FWD = 10
N_COUPLE = 4
NFEAT_BASE = 25     # triple_feats, unchanged
OFF_HIST = NFEAT_BASE
OFF_FWD = OFF_HIST + N_HIST
OFF_COUPLE = OFF_FWD + N_FWD
NFEAT_FULL = OFF_COUPLE + N_COUPLE      # 49
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
    """Velocities AND both link maps, from one greedy pass.

    prev_of[t][i] = index in frame t-1 linked to cell i at t, or -1.
    next_of[t][i] = index in frame t+1 linked to cell i at t, or -1.
    pass1_velocities discards both; the temporal features need them."""
    T = len(dets)
    seeds = [seeds_of(dets[t]) for t in range(T)]
    vels = [np.zeros((len(seeds[t][0]), 3)) for t in range(T)]
    prev_of = [np.full(len(seeds[t][0]), -1, np.int32) for t in range(T)]
    next_of = [np.full(len(seeds[t][0]), -1, np.int32) for t in range(T)]
    for t in range(T - 1):
        P, ps = seeds[t]
        C, cs = seeds[t + 1]
        if len(P) == 0 or len(C) == 0:
            continue
        for pi, ci in _link_sim(P, C, vels[t] if len(vels[t]) else None,
                                ps, cs):
            vels[t + 1][ci] = (C[ci] - P[pi]) * SCALE
            prev_of[t + 1][ci] = pi
            next_of[t][pi] = ci
    return seeds, vels, prev_of, next_of


def _walk(seeds, link, t0, i0, k, step):
    """Follow a cell through `link` for up to k frames. step is -1 to go
    back, +1 to go forward. Returns (pts_um, scores) with element 0 at
    t0 itself."""
    pts, scs = [], []
    tt, ii = t0, int(i0)
    T = len(seeds)
    while 0 <= tt < T and ii >= 0 and len(pts) <= k:
        P, ps = seeds[tt]
        if ii >= len(P):
            break
        pts.append(P[ii] * SCALE)
        scs.append(float(ps[ii]))
        nt = tt + step
        if not (0 <= nt < T):
            break
        ii = int(link[tt][ii])
        tt = nt
    return np.asarray(pts, float), np.asarray(scs, float)


def _angle_deg(u, v):
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu < EPS or nv < EPS:
        return 0.0
    c = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _steps(pts):
    """Consecutive displacement vectors along pts, in list order."""
    return [pts[m + 1] - pts[m] for m in range(len(pts) - 1)]


def hist_feats(pts, scs):
    """N_HIST columns from the mother's own past. pts is NEWEST FIRST.

    Column 0 is the number of steps actually available; when it is 0 the
    rest are zeros, so the model can split on column 0 and treat 'no
    history' as its own regime rather than as 'a stationary cell'."""
    n = len(pts)
    if n < 2:
        warn_once("hist_absent",
                  "a mother had no backward link (frame 0, or the greedy "
                  "pass left it unlinked) -> all 10 history columns are 0 "
                  "and hist_len says so; they are NOT measurements of a "
                  "stationary cell")
        return [0.0] * N_HIST
    steps = [pts[m] - pts[m + 1] for m in range(n - 1)]   # newest first
    norms = np.array([float(np.linalg.norm(s)) for s in steps])
    v_last = float(norms[0])
    v_mean = float(norms.mean())
    path = float(norms.sum())
    net = float(np.linalg.norm(pts[0] - pts[-1]))
    if len(norms) >= 2:
        # positive slope = faster further in the past = slowing toward t
        v_slope = float(np.polyfit(np.arange(len(norms)), norms, 1)[0])
        curv_last = _angle_deg(steps[0], steps[1])
        curv_mean = float(np.mean([_angle_deg(steps[m], steps[m + 1])
                                   for m in range(len(steps) - 1)]))
    else:
        warn_once("hist_one_step",
                  "a mother had only ONE backward step, so slope and "
                  "curvature are undefined -> substituting 0.0 for those "
                  "three columns; hist_len distinguishes this case")
        v_slope = curv_last = curv_mean = 0.0
    units = np.array([s / max(float(np.linalg.norm(s)), EPS) for s in steps])
    dir_stab = float(np.linalg.norm(units.mean(axis=0)))
    sc_hist = float(np.mean(scs[1:])) if len(scs) > 1 else float(scs[0])
    return [float(len(steps)), v_last, v_mean, v_last / max(v_mean, EPS),
            v_slope, float(net / max(path, EPS)), curv_last, curv_mean,
            dir_stab, float(scs[0]) / max(sc_hist, EPS)]


def fwd_feats(pa, sa, pb, sb, sep0):
    """N_FWD columns from what the two daughters DO next.

    The claim being tested: a real pair both persists and keeps drifting
    apart, symmetrically; a spurious pair has one member vanish, or the
    two converge again because they were simply passing each other.

    Column 0 is how many steps BOTH survived; 0 means the rest are
    zeros."""
    na, nb = len(pa), len(pb)
    surv = min(na, nb) - 1
    if surv < 1:
        warn_once("fwd_absent",
                  "a candidate daughter had no forward link (last frame, "
                  "or the greedy pass dropped it) -> the 8 trajectory "
                  "columns are 0 and surv_both says so; they are NOT "
                  "measurements of a stable pair")
        # surv_gap survives the early return on purpose: ONE member
        # vanishing while the other persists is the strongest spurious
        # signature there is, and zeroing it would throw that away.
        return [0.0, float(abs(na - nb))] + [0.0] * (N_FWD - 2)
    a, b = pa[:surv + 1], pb[:surv + 1]
    seps = np.array([float(np.linalg.norm(a[m] - b[m]))
                     for m in range(len(a))])
    s0 = max(float(sep0), EPS)
    if len(seps) >= 2:
        sep_slope = float(np.polyfit(np.arange(len(seps)), seps, 1)[0])
    else:
        sep_slope = 0.0
    da = a[-1] - a[0]
    db = b[-1] - b[0]
    nda, ndb = float(np.linalg.norm(da)), float(np.linalg.norm(db))
    cos_dirs = (float(np.dot(da, db)) / (nda * ndb)
                if nda > EPS and ndb > EPS else 0.0)
    cen = 0.5 * (a + b)
    csteps = _steps(cen)
    cpath = float(sum(float(np.linalg.norm(s)) for s in csteps))
    cnet = float(np.linalg.norm(cen[-1] - cen[0]))
    ma = float(np.mean(sa[:surv + 1]))
    mb = float(np.mean(sb[:surv + 1]))
    return [float(surv), float(abs(na - nb)),
            float(seps[-1] / s0), float(seps.min() / s0), sep_slope,
            cos_dirs, min(ma, mb), min(ma, mb) / max(max(ma, mb), EPS),
            cpath / max(len(csteps), 1),
            cnet / max(cpath, EPS)]


def couple_feats(pts, a_um, b_um, p_um):
    """N_COUPLE columns tying the mother's history to the candidate pair.

    The physical claim: at a real division the CENTROID of the two
    daughters carries the mother's momentum forward; a spurious pair's
    centroid is dragged off it."""
    mid = 0.5 * (a_um + b_um)
    sep = float(np.linalg.norm(a_um - b_um))
    if len(pts) < 2:
        return [float(np.linalg.norm(mid - p_um)), 0.0, 0.0, 0.0]
    steps = [pts[m] - pts[m + 1] for m in range(len(pts) - 1)]
    norms = np.array([float(np.linalg.norm(s)) for s in steps])
    v_mean = float(norms.mean())
    resid = float(np.linalg.norm(mid - (p_um + steps[0])))
    heading = np.mean([s / max(float(np.linalg.norm(s)), EPS)
                       for s in steps], axis=0)
    axis = a_um - b_um
    na, nh = float(np.linalg.norm(axis)), float(np.linalg.norm(heading))
    axis_vs_motion = (abs(float(np.dot(axis, heading)) / (na * nh))
                      if na > EPS and nh > EPS else 0.0)
    return [resid, resid / max(v_mean, EPS), axis_vs_motion,
            sep / max(v_mean, EPS)]


def triple_feats_ext(P, C, N, i, ja, jb, ps, cs, vel, sp, t, T,
                     seeds, prev_of, next_of):
    """25 base columns EXACTLY as triple_feats computes them, then 10
    backward, 10 forward, 4 coupling. One build, four column views -- the
    arms can differ ONLY in what the model is allowed to see."""
    base = triple_feats(P, C, N, i, ja, jb, ps, cs, vel, sp, t, T)
    hp, hs = _walk(seeds, prev_of, t, i, HIST_K, -1)
    fa, sa = _walk(seeds, next_of, t + 1, ja, FWD_K, +1)
    fb, sb = _walk(seeds, next_of, t + 1, jb, FWD_K, +1)
    a_um, b_um = C[ja] * SCALE, C[jb] * SCALE
    sep0 = float(np.linalg.norm(a_um - b_um))
    return (base + hist_feats(hp, hs) + fwd_feats(fa, sa, fb, sb, sep0)
            + couple_feats(hp, a_um, b_um, P[i] * SCALE))


def build(dets, T, div_set, tag, rng=None):
    """Same gating, same labels, same negative thinning as div_train's
    build -- only the feature row is longer.

    div_set: {(t, i_parent_det, j1_det, j2_det)} known true divisions."""
    X, y, meta = [], [], []
    seeds, vels, prev_of, next_of = pass1_tracks(dets)
    for t in range(T - 1):
        P, ps = seeds[t]
        C, cs = seeds[t + 1]
        if len(P) < 2 or len(C) < 2:
            continue
        Pu, Cu = P * SCALE[None, :], C * SCALE[None, :]
        tree = cKDTree(Cu)
        for i in range(len(P)):
            idx = tree.query_ball_point(Pu[i], GATE_UM)
            if len(idx) < 2:
                continue
            idx = sorted(idx, key=lambda j: np.linalg.norm(Cu[j] - Pu[i]))
            idx = [j for j in idx if cs[j] >= MIN_SCORE_A][:TOPK]
            if len(idx) < 2:
                continue
            sp = spacing_at(Pu, Pu[i])
            truth = {tuple(sorted(v[2:])) for v in div_set if v[0] == t
                     and v[1] == i}
            for m in range(len(idx)):
                for n in range(m + 1, len(idx)):
                    ja, jb = idx[m], idx[n]
                    lab = 1 if tuple(sorted((ja, jb))) in truth else 0
                    if lab == 0 and rng is not None and rng.random() > KEEP:
                        continue
                    X.append(triple_feats_ext(
                        P, C, len(P), i, ja, jb, ps, cs,
                        vels[t][i] if len(vels[t]) > i else None,
                        sp, t, T, seeds, prev_of, next_of))
                    y.append(lab)
                    meta.append(tag)
    return X, y, meta


def comp_divisions(name, dets, T):
    """GT divisions of one competition dataset, mapped onto detections."""
    gp = geff_path(root, "train", name)
    if not gp.exists():
        return set()
    gt = read_geff(gp)
    g_t = np.asarray(gt["t"], int)
    g_pos = np.stack([np.asarray(gt[c], float)
                      for c in ("z", "y", "x")], axis=1)
    row_of = {int(v): i for i, v in enumerate(gt["node_ids"])}
    kids = defaultdict(list)
    for a, b in gt["edges"]:
        ra, rb = row_of.get(int(a)), row_of.get(int(b))
        if ra is not None and rb is not None:
            kids[ra].append(rb)
    out = set()
    for p, ch in kids.items():
        if len(ch) < 2:
            continue
        t = int(g_t[p])
        if t + 1 >= T or g_t[ch[0]] != t + 1 or g_t[ch[1]] != t + 1:
            continue
        P, _ = seeds_of(dets[t])
        C, _ = seeds_of(dets[t + 1])
        if len(P) == 0 or len(C) == 0:
            continue

        def near(pool, x):
            d = np.sqrt((((pool - x) * SCALE) ** 2).sum(1))
            k = int(np.argmin(d))
            return (k, float(d[k])) if d[k] <= MATCH_UM else (None, d[k])

        ip, _ = near(P, g_pos[p])
        if ip is None:
            continue
        j1, _ = near(C, g_pos[ch[0]])
        j2, _ = near(C, g_pos[ch[1]])
        if j1 is None or j2 is None or j1 == j2:
            continue
        out.add((t, ip, j1, j2))
    return out


print("\n=== BUILDING THE TRIPLE DATASET ===", flush=True)
print(f"  competition caches: {len(CACHES)}", flush=True)
comp_names = sorted(CACHES)
Xc, yc, gc_ = [], [], []
n_div_found = 0
t0 = time.time()
for k, name in enumerate(comp_names):
    dets = load_dets(name)
    T = len(dets)
    ds = comp_divisions(name, dets, T)
    n_div_found += len(ds)
    if not ds:
        continue                      # only datasets WITH a division
    X, y, _m = build(dets, T, ds, "comp",
                     np.random.default_rng(hash(name) % 2**31))
    Xc += X
    yc += y
    gc_ += [name.split("_")[0]] * len(y)
print(f"  competition: {n_div_found} divisions land on detections, "
      f"{len(yc)} triples, {sum(yc)} positive ({time.time()-t0:.0f}s)",
      flush=True)

ZH = {}
for f in pathlib.Path("/kaggle/input").glob("*/**/zhdet_*.npz"):
    ZH[f.stem[len("zhdet_"):]] = f
print(f"  zebrahub crops: {len(ZH)}", flush=True)
PER_CROP = 400                 # cap per crop; the mixture is a free knob
Xz, yz = [], []
for k, (name, f) in enumerate(sorted(ZH.items())):
    z = np.load(f)
    T = max(int(kk[1:]) for kk in z.files if kk.startswith("c")) + 1
    dets = [(z[f"c{t}"], z[f"s{t}"]) for t in range(T)]
    nodes = np.asarray(z["nodes"], float)
    tid = np.asarray(z["track_id"], np.int64)
    pid = np.asarray(z["parent_track_id"], np.int64)
    first, byk = {}, defaultdict(dict)
    for r in range(len(nodes)):
        tt = int(nodes[r, 0])
        byk[int(tid[r])][tt] = r
        kk = int(tid[r])
        # BUG FIXED 2026-08-24: this compared tt (a TIME) against
        # first[kk] (a ROW INDEX), so "each track's earliest node" picked
        # essentially arbitrary rows and parent->child pairing collapsed.
        # 26 of 41 crops then reported no division at all, and Zebrahub
        # yielded 68 positives where hundreds of thousands were expected.
        if kk not in first or tt < int(nodes[first[kk], 0]):
            first[kk] = r
    par = {int(tid[r]): int(pid[r]) for r in range(len(nodes)) if pid[r] > 0}
    ds = set()
    for kk, r in first.items():
        p = par.get(kk)
        if p is None:
            continue
        t1 = int(nodes[r, 0])
        if t1 - 1 < 0 or t1 >= T:
            continue
        rp = byk.get(p, {}).get(t1 - 1)
        sib = [q for q, rr in first.items()
               if par.get(q) == p and q != kk
               and int(nodes[rr, 0]) == t1]
        if rp is None or not sib:
            continue
        rs = first[sib[0]]
        P, _ = seeds_of(dets[t1 - 1])
        C, _ = seeds_of(dets[t1])
        if len(P) == 0 or len(C) == 0:
            continue

        def near(pool, x):
            d = np.sqrt((((pool - x) * SCALE) ** 2).sum(1))
            j = int(np.argmin(d))
            return (j, float(d[j])) if d[j] <= MATCH_UM else (None, d[j])

        ip, _ = near(P, nodes[rp, 1:])
        j1, _ = near(C, nodes[r, 1:])
        j2, _ = near(C, nodes[rs, 1:])
        if ip is None or j1 is None or j2 is None or j1 == j2:
            continue
        ds.add((t1 - 1, ip, j1, j2))
        if len(ds) >= PER_CROP:
            break
    if not ds:
        print(f"  WARN {name}: no division landed on detections", flush=True)
        continue
    X, y, _m = build(dets, T, ds, "zh",
                     np.random.default_rng(hash(name) % 2**31))
    Xz += X
    yz += y
    del z, dets
    gc.collect()
print(f"  zebrahub: {len(yz)} triples, {sum(yz)} positive", flush=True)


Xc = np.asarray(Xc, np.float32); yc = np.asarray(yc, np.int8)
Xz = np.asarray(Xz, np.float32); yz = np.asarray(yz, np.int8)
gc_ = np.asarray(gc_)
assert len(yc) and len(yz), "one domain produced nothing"
assert Xz.shape[1] == NFEAT_FULL, \
    f"expected {NFEAT_FULL} columns, got {Xz.shape[1]}"
warning_summary()

# in-domain validation: hold out the 6bba embryo entirely, exactly as
# div_train v3 did, so BASE is directly comparable to its 0.8083.
va = gc_ == "6bba"
print(f"\n  VALIDATE on 6bba: {va.sum()} triples, {yc[va].sum()} pos",
      flush=True)

HIST_NAMES = ["hist_len", "v_last", "v_mean", "decel", "v_slope",
              "straight", "curv_last", "curv_mean", "dir_stab",
              "score_ratio"]
FWD_NAMES = ["surv_both", "surv_gap", "sep_end/0", "sep_min/0",
             "sep_slope", "cos_dirs", "score_min", "score_sym",
             "cent_speed", "cent_straight"]
COUPLE_NAMES = ["cent_resid", "cent_resid/v", "axis_vs_motion", "sep/v"]
NEW_NAMES = HIST_NAMES + FWD_NAMES + COUPLE_NAMES

COLS = {
    # div_train's 25, unchanged -- the anchor
    "BASE": list(range(NFEAT_BASE)),
    # + the mother's own past (backward; free, but mitosis may be brief)
    "HIST": list(range(OFF_FWD)),
    # + what the daughters DO next (forward; the user's push-back)
    "FWD": list(range(NFEAT_BASE)) + list(range(OFF_FWD, OFF_COUPLE)),
    # everything, including the centroid-momentum coupling
    "FULL": list(range(NFEAT_FULL)),
}
ARMS = ("BASE", "HIST", "FWD", "FULL")


def fp_per_tp(y, p, recalls=(0.10, 0.25, 0.50)):
    """How many false divisions must be accepted to collect a given share
    of the true ones. The metric charges for false divisions and the
    budget is ~7 FP per TP at the margin, so THIS is what decides whether
    a division model is usable. ROC-AUC at 0.25% positives does not."""
    order = np.argsort(-np.asarray(p))
    ys = np.asarray(y)[order]
    cum = np.cumsum(ys)
    tot = int(ys.sum())
    out = []
    for r in recalls:
        need = max(1, int(round(r * tot)))
        idx = int(np.searchsorted(cum, need))
        fp = (idx + 1) - need
        out.append((r, need, fp, fp / max(need, 1)))
    return out


print("\n=== TRAIN ON ZEBRAHUB ONLY, VALIDATE ON HELD-OUT 6bba ===",
      flush=True)
print("  (div_train v3 recorded AUC 0.8083 for the BASE columns under "
      "exactly this protocol)", flush=True)
print("  BACKWARD = the mother's past. FORWARD = what the daughters then "
      "do. Not a leak: this is offline tracking over a finished movie and "
      "the solver already optimises globally over all frames.", flush=True)

models = {}
Xv_all, yv = Xc[va], yc[va]
for arm in ARMS:
    m = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    m.fit(Xz[:, COLS[arm]], yz)
    pv = m.predict_proba(Xv_all[:, COLS[arm]])[:, 1]
    auc = roc_auc_score(yv, pv)
    ap = average_precision_score(yv, pv)
    models[arm] = (m, pv)
    print(f"  {arm:<5} {len(COLS[arm]):>2} cols   AUC {auc:.4f}   "
          f"AP {ap:.4f}", flush=True)
    for r, need, fp, ratio in fp_per_tp(yv, pv):
        print(f"        recall {r:.0%} ({need:>2} of {int(yv.sum())} true) "
              f"costs {fp:>5} false divisions = {ratio:>7.1f} FP per TP",
              flush=True)
    with open(f"/kaggle/working/div_model_{arm.lower()}.pkl", "wb") as fh:
        pickle.dump({"model": m, "cols": COLS[arm],
                     "version": SCRIPT_VERSION}, fh)


# ---- which physical signal is actually carrying it? --------------------
print("\n=== WHICH NEW COLUMN CARRIES IT (permutation on 6bba, FULL) ===",
      flush=True)
m_full = models["FULL"][0]
base_auc = roc_auc_score(yv, models["FULL"][1])
rng = np.random.default_rng(0)
drops = []
for off, nm in enumerate(NEW_NAMES):
    col = NFEAT_BASE + off
    Xp = Xv_all.copy()
    rng.shuffle(Xp[:, col])
    a = roc_auc_score(yv, m_full.predict_proba(Xp)[:, 1])
    drops.append((base_auc - a, nm))
    del Xp
for d, nm in sorted(drops, reverse=True):
    fam = ("back" if nm in HIST_NAMES else
           "fwd " if nm in FWD_NAMES else "coup")
    bar = "#" * int(max(0.0, d) * 2000)
    print(f"  [{fam}] {nm:<16} AUC drop {d:+.4f}  {bar}", flush=True)
print("  (a drop near zero means the column does nothing; negative means "
      "shuffling it HELPED, i.e. it was noise)", flush=True)


# ---- does the physics actually exist in this data? ----------------------
# Checked directly, with no model in the way: the medians at TRUE division
# triples against everything else.
print("\n=== THE PHYSICS, CHECKED WITHOUT A MODEL (6bba) ===", flush=True)
pos = yv == 1
neg = yv == 0
print(f"  {int(pos.sum())} true division triples vs {int(neg.sum())} "
      f"negatives", flush=True)
print(f"  {'':<6} {'column':<16} {'true div':>10} {'other':>10} "
      f"{'ratio':>8}", flush=True)
for off, nm in enumerate(NEW_NAMES):
    col = NFEAT_BASE + off
    fam = ("back" if nm in HIST_NAMES else
           "fwd " if nm in FWD_NAMES else "coup")
    a = float(np.nanmedian(Xv_all[pos, col]))
    b = float(np.nanmedian(Xv_all[neg, col]))
    r = a / b if abs(b) > 1e-6 else float("nan")
    print(f"  [{fam}] {nm:<16} {a:>10.3f} {b:>10.3f} {r:>8.2f}", flush=True)

warning_summary()
print("\nDIV KINEMATICS DONE", flush=True)
