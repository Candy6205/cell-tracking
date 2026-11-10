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

SCRIPT_VERSION = "2026-09-18a-lnplace2"
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



def ln_load(name):
    """The rescue detection list and, per frame, how many new peaks sit at the end of the primaries."""
    z = np.load(LN_FDET[name])
    T = max(int(k[1:]) for k in z.files if k.startswith("c")) + 1
    dets = [(np.asarray(z[f"c{t}"], np.float32).reshape(-1, 3), np.asarray(z[f"s{t}"], np.float32).reshape(-1)) for t in range(T)]
    nnew = [int(z[f"nnew{t}"]) if f"nnew{t}" in z.files else 0 for t in range(T)]
    return dets, nnew



root = find_root(None)
print(f"root {root}", flush=True)
import pandas as pd                                                          # noqa: E402
from scipy import ndimage as _ndi                                            # noqa: E402
from viz_cells import open_image as _open_image                              # noqa: E402

LN_FDET = _walk_files(_npz_prefix("fdet_F4ln_"))
print(f"rescue caches (fdet_F4ln_): {len(LN_FDET)}", flush=True)
assert len(LN_FDET) >= 100, "attach biohub-lnrescue"
_csv = [p_ for p_ in pathlib.Path("/kaggle/input").glob("**/lnrank_edges.csv")]
assert _csv, "attach biohub-lnrank (lnrank_edges.csv)"
df = pd.read_csv(_csv[0])
df = df[df.arm == "LN20"].reset_index(drop=True)
print(f"lnrank_edges.csv: {len(df)} LN20 rows over {df.dataset.nunique()} movies", flush=True)

UNET_THRESH = 0.15                # seeds_of: the harness keeps primaries at score >= this; q_i indexes THAT list
HALF_UM = 8.0
SIG = (0.6, 1.5, 1.5)             # forkimg's contrast crop
DET_UM = 3.0                      # a shipped primary this close to a GT node = the cell is detected
CTRL_UM = 2.0                     # control cells: a shipped primary within this of the GT node
LIN_UM = 0.3                      # a GT node this close to the midpoint of its two neighbours looks interpolated


def seeds_of(dets_t):
    coords, scores = dets_t
    coords = np.asarray(coords, float).reshape(-1, 3)
    scores = np.asarray(scores, float).reshape(-1)
    m = scores >= UNET_THRESH
    return coords[m], scores[m]


def _crop(img, t, xyz_vox, sigma):
    if t < 0 or t >= img.shape[0]:
        return None, None
    hz = int(np.ceil(HALF_UM / SCALE[0]))
    hy = int(np.ceil(HALF_UM / SCALE[1]))
    z, y, x = (int(round(float(v))) for v in xyz_vox)
    z0, y0, x0 = max(0, z - hz), max(0, y - hy), max(0, x - hy)
    try:
        crop = np.asarray(img[t, z0:z + hz + 1, y0:y + hy + 1, x0:x + hy + 1])
    except Exception:                                                        # noqa: BLE001
        crop = np.asarray(img[t])[z0:z + hz + 1, y0:y + hy + 1, x0:x + hy + 1]
    if crop.size == 0:
        return None, None
    return _ndi.gaussian_filter(crop.astype(np.float32), sigma), np.array([z0, y0, x0], float)


def con_at(img, t, xyz_um):
    """contrast over the crop median at a position (and the maximum within 2 um of it); NaN when unreadable."""
    xyz_vox = np.asarray(xyz_um, float) / SCALE
    sm, off = _crop(img, t, xyz_vox, SIG)
    if sm is None:
        return np.nan, np.nan
    idx = np.indices(sm.shape).reshape(3, -1).T.astype(float)
    pos_um = (idx + off) * SCALE
    v = sm.reshape(-1)
    bg = float(np.median(v))
    d = np.linalg.norm(pos_um - np.asarray(xyz_um, float), axis=1)
    j0 = int(np.argmin(d))
    near = d <= 2.0
    mx = float(v[near].max()) if near.any() else float(v[j0])
    return (float(v[j0]) - bg) / max(bg, 1e-6), (mx - bg) / max(bg, 1e-6)


def peak_class(cats):
    cats = set(cats)
    if "G_tp" in cats:
        return "linked"
    if "C_below_floor" in cats:
        return "ranker_refused"
    if cats == {"A_below_tau"}:
        return "below_tau"
    return "other"


peaks = {}
for (name, t, qi), g_ in df.groupby(["dataset", "t", "q_i"]):
    peaks[(name, int(t), int(qi))] = dict(cls=peak_class(g_.cat.tolist()), q_score=float(g_.q_score.iloc[0]),
                                          d_gt_csv=float(g_.d_gt.iloc[0]), admitted=int(g_.admitted.iloc[0]))
print(f"peaks: {len(peaks)}; per class " + str(pd.Series([v['cls'] for v in peaks.values()]).value_counts().to_dict()),
      flush=True)


def cell_row(img, g, t, g_t, g_pos, nbr, ship):
    """everything about one GT node g at frame t: detection at t, the track around it, the raw contrast along it."""
    row = {}
    st = ship(t)
    if st is not None:
        d_, i_ = st[0].query(g_pos[g])
        row["d_gt_ship"], row["s_score"] = float(d_), float(st[1][i_])
    else:
        row["d_gt_ship"], row["s_score"] = np.nan, np.nan
    row["con_gt"], row["con_gt_max2"] = con_at(img, t, g_pos[g])
    prev = [gn for gn in nbr[g] if int(g_t[gn]) == t - 1]
    nxt = [gn for gn in nbr[g] if int(g_t[gn]) == t + 1]
    row["deg"] = len(nbr[g])
    row["has_prev"], row["has_next"] = int(bool(prev)), int(bool(nxt))
    for side, lst, tt in (("prev", prev, t - 1), ("next", nxt, t + 1)):
        if lst:
            gn = lst[0]
            st2 = ship(tt)
            row[f"d_{side}_ship"] = float(st2[0].query(g_pos[gn])[0]) if st2 is not None else np.nan
            row[f"con_{side}"], row[f"con_{side}_max2"] = con_at(img, tt, g_pos[gn])
            row[f"step_{side}"] = float(np.linalg.norm(g_pos[gn] - g_pos[g]))
        else:
            row[f"d_{side}_ship"] = row[f"con_{side}"] = row[f"con_{side}_max2"] = row[f"step_{side}"] = np.nan
    if prev and nxt:
        mid = 0.5 * (g_pos[prev[0]] + g_pos[nxt[0]])
        row["lin_dev"] = float(np.linalg.norm(g_pos[g] - mid))
    else:
        row["lin_dev"] = np.nan
    return row


ROWS, CTRL = [], []
t_all = time.time()
rng = np.random.default_rng(0)
by_movie = defaultdict(list)
for k_, v_ in peaks.items():
    by_movie[k_[0]].append((k_[1], k_[2], v_))
n_skip = 0
for mi, name in enumerate(sorted(by_movie)):
    t0 = time.time()
    if name not in LN_FDET:
        warn_once("no_fdet", f"{name}: no fdet_F4ln_ cache")
        continue
    gt = read_geff(geff_path(root, "train", name))
    g_t = np.asarray(gt["t"], int)
    g_pos = np.stack([np.asarray(gt[c_], float) for c_ in ("z", "y", "x")], axis=1) * SCALE
    ids = [int(v) for v in gt["node_ids"]]
    row_of = {v: i for i, v in enumerate(ids)}
    nbr = defaultdict(list)
    for a_, b_ in gt["edges"]:
        ra, rb = row_of.get(int(a_)), row_of.get(int(b_))
        if ra is not None and rb is not None:
            nbr[ra].append(rb)
            nbr[rb].append(ra)
    gt_rows = {int(tt): np.where(g_t == tt)[0] for tt in np.unique(g_t)}
    gt_tree = {tt: cKDTree(g_pos[ix]) for tt, ix in gt_rows.items()}
    dets, nnew = ln_load(name)
    seeds = [seeds_of(dets[t]) for t in range(len(dets))]
    try:
        img = _open_image(pathlib.Path(root) / "train" / f"{name}.zarr")
    except Exception as _e:                                                  # noqa: BLE001
        warn_once("no_img", f"{name}: image not opened ({_e})")
        continue
    ship_cache = {}

    def _ship(t):
        if t < 0 or t >= len(seeds):
            return None
        if t not in ship_cache:
            c_, s_ = seeds[t]
            n_ = len(c_) - nnew[t]
            ship_cache[t] = (cKDTree(c_[:n_] * SCALE), s_[:n_]) if n_ > 0 else None
        return ship_cache[t]

    used_g = set()
    for t, qi, meta in by_movie[name]:
        c_all = seeds[t][0]
        if qi >= len(c_all) or t not in gt_tree:
            n_skip += 1
            continue
        c_um = c_all[qi] * SCALE
        d0, j_ = gt_tree[t].query(c_um)
        g = int(gt_rows[t][int(j_)])
        used_g.add((t, g))
        row = cell_row(img, g, t, g_t, g_pos, nbr, _ship)
        st = _ship(t)
        row.update(dataset=name, t=t, q_i=qi, **meta, d_gt0=float(d0),
                   d_peak_ship=float(st[0].query(c_um)[0]) if st is not None else np.nan,
                   con_peak=con_at(img, t, c_um)[0])
        ROWS.append(row)
    # control: GT nodes of the same frames held by a shipped primary within CTRL_UM, up to 30 per movie
    frames = sorted({t for t, _q, _m in by_movie[name]})
    cands = []
    for t in frames:
        st = _ship(t)
        if st is None or t not in gt_rows:
            continue
        ix = gt_rows[t]
        dd, jj = st[0].query(g_pos[ix])
        for k_ in np.where(dd <= CTRL_UM)[0]:
            g = int(ix[k_])
            if (t, g) not in used_g:
                cands.append((t, g))
    if cands:
        for ci in rng.choice(len(cands), size=min(30, len(cands)), replace=False):
            t, g = cands[ci]
            row = cell_row(img, g, t, g_t, g_pos, nbr, _ship)
            row.update(dataset=name, t=t, g=g)
            CTRL.append(row)
    if mi % 16 == 0 or mi == len(by_movie) - 1:
        print(f"  [{mi + 1}/{len(by_movie)}] {name}: {len(by_movie[name])} peaks, {time.time() - t0:.1f}s "
              f"(total {(time.time() - t_all) / 60:.1f} min)", flush=True)

P = pd.DataFrame(ROWS)
C = pd.DataFrame(CTRL)
P.to_csv("/kaggle/working/lnplace2_peaks.csv", index=False)
C.to_csv("/kaggle/working/lnplace2_control.csv", index=False)
print(f"\n{len(P)} peaks ({n_skip} skipped), {len(C)} control cells; recomputed d_gt matches lnrank's csv within 0.01 um for "
      f"{int((np.abs(P.d_gt0 - P.d_gt_csv) < 0.01).sum())} of {len(P)}", flush=True)


def _q(s, f=0.5):
    s = np.asarray(s, float)
    s = s[np.isfinite(s)]
    return float(np.quantile(s, f)) if len(s) else float("nan")


def _frac(m):
    m = np.asarray(m, float)
    m = m[np.isfinite(m)]
    return 100.0 * float(np.mean(m)) if len(m) else float("nan")


CLASSES = ("ranker_refused", "linked", "below_tau", "other", "ALL", "control")


def _sel(cls):
    return C if cls == "control" else (P if cls == "ALL" else P[P.cls == cls])


print("\n=== IS THE ANNOTATED CELL BEHIND THE PEAK DETECTED AT FRAME t? (the GT node matched to the peak; medians in um) ===", flush=True)
print(f"  {'class':<15} {'n':>4} {'d_gt0':>6} | {'GT->ship':>8} {'<=3':>5} {'s_score':>7} | {'peak->ship':>10} {'<=3':>5} | "
      f"{'prev det':>8} {'next det':>8} | {'con t-1':>7} {'con t':>6} {'con t+1':>7} | {'max2 t':>6} | {'lin_dev':>7} {'<0.3':>5} {'step':>5}",
      flush=True)
for cls in CLASSES:
    s = _sel(cls)
    if not len(s):
        continue
    d_gt0 = f"{_q(s.d_gt0):>6.2f}" if "d_gt0" in s else f"{'--':>6}"
    p_ship = f"{_q(s.d_peak_ship):>10.2f} {_frac(s.d_peak_ship <= DET_UM):>4.0f}%" if "d_peak_ship" in s else f"{'--':>10} {'--':>5}"
    print(f"  {cls:<15} {len(s):>4} {d_gt0} | {_q(s.d_gt_ship):>8.2f} {_frac(s.d_gt_ship <= DET_UM):>4.0f}% {_q(s.s_score):>7.2f} | "
          f"{p_ship} | {_frac(s.d_prev_ship <= DET_UM):>7.0f}% {_frac(s.d_next_ship <= DET_UM):>7.0f}% | "
          f"{_q(s.con_prev):>7.2f} {_q(s.con_gt):>6.2f} {_q(s.con_next):>7.2f} | {_q(s.con_gt_max2):>6.2f} | "
          f"{_q(s.lin_dev):>7.2f} {_frac(s.lin_dev < LIN_UM):>4.0f}% {_q(pd.concat([s.step_prev, s.step_next])):>5.2f}", flush=True)
print("  columns: d_gt0 = peak to its GT node; GT->ship = the GT node to the nearest shipped primary at t, '<=3' = the cell is "
      "already detected, s_score = that primary's score; peak->ship = the peak to the nearest shipped primary; prev/next det = "
      "the GT neighbour at t-1 / t+1 has a shipped primary within 3 um; con = raw contrast over the crop median at the GT "
      "node's position at t-1 / t / t+1, max2 = the maximum within 2 um at t; lin_dev = the node's distance from the midpoint "
      "of its two neighbours (interpolated annotations sit on the line), step = the track's step length per frame.", flush=True)

print("\n=== the refused peaks split by whether the annotated cell already has a shipped primary at t ===", flush=True)
s = P[P.cls == "ranker_refused"]
for lab, m in (("cell DETECTED at t (GT->ship <= 3)", s.d_gt_ship <= DET_UM), ("cell UNDETECTED at t (GT->ship > 3)", s.d_gt_ship > DET_UM)):
    r = s[m]
    if len(r):
        print(f"  {lab:<38} n={len(r):>3}  d_gt0 {_q(r.d_gt0):.2f}  peak->ship {_q(r.d_peak_ship):.2f}  con t-1/t/t+1 "
              f"{_q(r.con_prev):.2f}/{_q(r.con_gt):.2f}/{_q(r.con_next):.2f}  max2 {_q(r.con_gt_max2):.2f}  lin_dev {_q(r.lin_dev):.2f} "
              f"(<0.3: {_frac(r.lin_dev < LIN_UM):.0f}%)  q_score {_q(r.q_score):.2f}", flush=True)
print("\n=== the same split for every class: share of peaks whose annotated cell is already detected at t ===", flush=True)
for cls in ("ranker_refused", "linked", "below_tau", "other", "ALL"):
    r = _sel(cls)
    if len(r):
        print(f"  {cls:<15} n={len(r):>3}  detected {_frac(r.d_gt_ship <= DET_UM):>3.0f}%  undetected {_frac(r.d_gt_ship > DET_UM):>3.0f}%  "
              f"(GT->ship median {_q(r.d_gt_ship):.2f} um)", flush=True)
print("  reading: 'detected' peaks are second points next to a cell a shipped primary already holds -- the ranker is right to "
      "refuse them and they are not missed cells; for the 'undetected' ones, compare con t against con t-1 / t+1 (does the cell "
      "go dark at t only?) and lin_dev against the control (is the annotation at t an interpolation?).", flush=True)
warning_summary()
print(f"\nLNPLACE2 DONE in {(time.time() - t_all) / 60:.1f} min", flush=True)
print("LNPLACE2 OK", flush=True)
