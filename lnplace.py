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

SCRIPT_VERSION = "2026-09-18b-lnplace"
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
from scipy.ndimage import maximum_filter                                     # noqa: E402
from viz_cells import open_image as _open_image                              # noqa: E402

LN_FDET = _walk_files(_npz_prefix("fdet_F4ln_"))
print(f"rescue caches (fdet_F4ln_): {len(LN_FDET)}", flush=True)
assert len(LN_FDET) >= 100, "attach biohub-lnrescue"
_csv = [p_ for p_ in pathlib.Path("/kaggle/input").glob("**/lnrank_edges.csv")]
assert _csv, "attach biohub-lnrank (lnrank_edges.csv)"
df = pd.read_csv(_csv[0])
df = df[df.arm == "LN20"].reset_index(drop=True)
print(f"lnrank_edges.csv: {len(df)} LN20 rows over {df.dataset.nunique()} movies; categories "
      f"{df.cat.value_counts().to_dict()}", flush=True)

R_LIST = (3.0, 5.0, 7.0)
LM_UM = 3.0                       # a raw local maximum this close to the GT centroid = the cell is there in the raw image
LM_THR = ((0.0, "d_lm0"), (0.25, "d_lm25"), (0.5, "d_lm50"), (1.0, "d_lm100"))   # the maximum's contrast floor
DUP_UM = 3.0                      # a re-placed peak this close to a shipped primary would be a duplicate
UNET_THRESH = 0.15                # seeds_of: the harness keeps primaries at score >= this; q_i indexes THAT list
HALF_UM = 8.0
SIGMAS = (("f", (0.6, 1.5, 1.5)), ("c", (1.2, 4.0, 4.0)))    # fine = forkimg's contrast crop (~1 um); coarse ~ a nuclear radius


def seeds_of(dets_t):
    coords, scores = dets_t
    coords = np.asarray(coords, float).reshape(-1, 3)
    scores = np.asarray(scores, float).reshape(-1)
    m = scores >= UNET_THRESH
    return coords[m], scores[m]


def _crop(img, t, xyz_vox, sigma):
    """smoothed crop (half HALF_UM) around a voxel position and its offset; (None, None) out of range."""
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
                                          d_gt_csv=float(g_.d_gt.iloc[0]), admitted=int(g_.admitted.iloc[0]),
                                          n_edges=len(g_), cats="|".join(sorted(set(g_.cat))))
print(f"peaks: {len(peaks)}; per class " + str(pd.Series([v['cls'] for v in peaks.values()]).value_counts().to_dict()),
      flush=True)


def _grid_um(sm, off):
    idx = np.indices(sm.shape).reshape(3, -1).T.astype(float)
    return (idx + off) * SCALE, idx


def replace_on_raw(img, t, c_vox, sigma):
    """the smoothed crop around c: its maximum within R um of c for each R, the intensity centroid (3.5 um)
    around the 5-um maximum, the contrast at c; None when the crop is unreadable."""
    out = {}
    sm, off = _crop(img, t, c_vox, sigma)
    if sm is None:
        return None
    pos_um, idx = _grid_um(sm, off)
    v = sm.reshape(-1)
    c_um = np.asarray(c_vox, float) * SCALE
    d = np.linalg.norm(pos_um - c_um, axis=1)
    bg = float(np.median(v))
    j0 = int(np.argmin(d))
    out["con_peak"] = (float(v[j0]) - bg) / max(bg, 1e-6)
    for R in R_LIST:
        near = d <= R
        j = int(np.argmax(np.where(near, v, -np.inf)))
        out[f"pos_r{int(R)}"] = pos_um[j]
        out[f"con_r{int(R)}"] = (float(v[j]) - bg) / max(bg, 1e-6)
    j5 = int(np.argmax(np.where(d <= 5.0, v, -np.inf)))
    dp = np.linalg.norm(pos_um - pos_um[j5], axis=1)
    m = dp <= 3.5
    w = np.clip(v[m] - bg, 0, None)
    out["pos_cen"] = (w[:, None] * pos_um[m]).sum(0) / w.sum() if w.sum() > 0 else pos_um[j5]
    return out


def raw_maximum_near(img, t, g_um, sigma):
    """crop centred on the GT centroid: the nearest raw local maximum (26-neighbourhood) within 7 um at each
    contrast floor over the crop median, the contrast at the GT voxel; None when unreadable."""
    g_vox = np.asarray(g_um, float) / SCALE
    sm, off = _crop(img, t, g_vox, sigma)
    if sm is None:
        return None
    pos_um, idx = _grid_um(sm, off)
    v = sm.reshape(-1)
    bg = float(np.median(v))
    lm = (sm >= maximum_filter(sm, size=3, mode="nearest")).reshape(-1) & (v > bg)
    d = np.linalg.norm(pos_um - np.asarray(g_um, float), axis=1)
    con = (v - bg) / max(bg, 1e-6)
    j0 = int(np.argmin(d))
    out = dict(con_gt=float(con[j0]))
    # smoothed noise has a local maximum every 1-2 um, so a maximum only counts above a contrast floor
    for thr_, key_ in LM_THR:
        cand = np.where(lm & (d <= 7.0) & (con >= thr_))[0]
        out[key_] = float(d[cand].min()) if len(cand) else np.nan
        if key_ == "d_lm50":
            out["con_lm"] = float(con[cand[np.argmin(d[cand])]]) if len(cand) else np.nan
    out["d_gt_lm"] = out["d_lm50"]
    return out


def measure(img, t, c_vox, g_um, ship_tree):
    """every column of one point (a rescue peak or a control detection) at both scales; None when unreadable."""
    c_um = np.asarray(c_vox, float) * SCALE
    row = {}
    for tag, sig in SIGMAS:
        rp = replace_on_raw(img, t, c_vox, sig)
        rm = raw_maximum_near(img, t, g_um, sig)
        if rp is None or rm is None:
            return None
        suf = "" if tag == "f" else "_c"
        row[f"con_peak{suf}"] = rp["con_peak"]
        for k_, v_ in rm.items():
            row[f"{k_}{suf}"] = v_
        for R in R_LIST:
            p_ = rp[f"pos_r{int(R)}"]
            row[f"d_gt_r{int(R)}{suf}"] = float(np.linalg.norm(p_ - g_um))
            row[f"moved_r{int(R)}{suf}"] = float(np.linalg.norm(p_ - c_um))
            row[f"con_r{int(R)}{suf}"] = rp[f"con_r{int(R)}"]
            row[f"d_ship_r{int(R)}{suf}"] = float(ship_tree.query(p_)[0]) if ship_tree is not None else np.nan
        row[f"d_gt_cen{suf}"] = float(np.linalg.norm(rp["pos_cen"] - g_um))
        row[f"moved_cen{suf}"] = float(np.linalg.norm(rp["pos_cen"] - c_um))
        row[f"d_ship_cen{suf}"] = float(ship_tree.query(rp["pos_cen"])[0]) if ship_tree is not None else np.nan
    row["d_ship0"] = float(ship_tree.query(c_um)[0]) if ship_tree is not None else np.nan
    return row


ROWS, CTRL = [], []
t_all = time.time()
rng = np.random.default_rng(0)
by_movie = defaultdict(list)
for k_, v_ in peaks.items():
    by_movie[k_[0]].append((k_[1], k_[2], v_))
n_skip = n_bad_idx = 0
for mi, name in enumerate(sorted(by_movie)):
    t0 = time.time()
    if name not in LN_FDET:
        warn_once("no_fdet", f"{name}: no fdet_F4ln_ cache")
        continue
    gt = read_geff(geff_path(root, "train", name))
    g_t = np.asarray(gt["t"], int)
    g_pos = np.stack([np.asarray(gt[c_], float) for c_ in ("z", "y", "x")], axis=1) * SCALE
    gt_rows = {int(tt): np.where(g_t == tt)[0] for tt in np.unique(g_t)}
    gt_tree = {tt: cKDTree(g_pos[ix]) for tt, ix in gt_rows.items()}
    dets, nnew = ln_load(name)
    seeds = [seeds_of(dets[t]) for t in range(len(dets))]
    try:
        img = _open_image(pathlib.Path(root) / "train" / f"{name}.zarr")
    except Exception as _e:                                                  # noqa: BLE001
        warn_once("no_img", f"{name}: image not opened ({_e})")
        continue
    ship_tree = {}

    def _ship(t):
        if t not in ship_tree:
            c_ = seeds[t][0]
            n_ = len(c_) - nnew[t]
            ship_tree[t] = cKDTree(c_[:n_] * SCALE) if n_ > 0 else None
        return ship_tree[t]

    for t, qi, meta in by_movie[name]:
        c_all = seeds[t][0]
        if qi >= len(c_all) or t not in gt_tree:
            n_bad_idx += 1
            continue
        c_vox = c_all[qi]
        c_um = c_vox * SCALE
        d0, j_ = gt_tree[t].query(c_um)
        g_um = g_pos[gt_rows[t][int(j_)]]
        row = measure(img, t, c_vox, g_um, _ship(t))
        if row is None:
            n_skip += 1
            continue
        row.update(dataset=name, t=t, q_i=qi, **meta, d_gt0=float(d0), d_gt_csv_check=float(abs(d0 - meta["d_gt_csv"])))
        ROWS.append(row)
    # the control: shipped primaries of the same frames matched to GT within 5 um, up to 30 per movie
    frames = sorted({t for t, _q, _m in by_movie[name]})
    cands = []
    for t in frames:
        if t not in gt_tree:
            continue
        c_all = seeds[t][0]
        n0 = len(c_all) - nnew[t]
        if n0 <= 0:
            continue
        dd, jj = gt_tree[t].query(c_all[:n0] * SCALE)
        for i_ in np.where(dd <= 5.0)[0]:
            cands.append((t, int(i_), float(dd[i_]), g_pos[gt_rows[t][int(jj[i_])]]))
    if cands:
        for ci in rng.choice(len(cands), size=min(30, len(cands)), replace=False):
            t, i_, d0, g_um = cands[ci]
            crow = measure(img, t, seeds[t][0][i_], g_um, _ship(t))
            if crow is None:
                continue
            crow.update(dataset=name, t=t, i=i_, d_gt0=d0)
            CTRL.append(crow)
    if mi % 16 == 0 or mi == len(by_movie) - 1:
        print(f"  [{mi + 1}/{len(by_movie)}] {name}: {len(by_movie[name])} peaks, {time.time() - t0:.1f}s "
              f"(total {(time.time() - t_all) / 60:.1f} min)", flush=True)

P = pd.DataFrame(ROWS)
C = pd.DataFrame(CTRL)
P.to_csv("/kaggle/working/lnplace_peaks.csv", index=False)
C.to_csv("/kaggle/working/lnplace_control.csv", index=False)
n_agree = int((P.d_gt_csv_check < 0.01).sum())
print(f"\n{len(P)} peaks measured ({n_skip} unreadable, {n_bad_idx} with an index beyond the seed list), {len(C)} control "
      f"detections; the recomputed d_gt agrees with lnrank's csv within 0.01 um for {n_agree} of {len(P)} peaks", flush=True)
assert n_agree >= 0.95 * len(P), "the peaks are not the csv's peaks: indexing is wrong (v2's bug)"


def _q(s, f=0.5):
    s = np.asarray(s, float)
    s = s[np.isfinite(s)]
    return float(np.quantile(s, f)) if len(s) else float("nan")


def _frac(m):
    m = np.asarray(m)
    return 100.0 * float(np.mean(m)) if len(m) else float("nan")


CLASSES = ("ranker_refused", "linked", "below_tau", "other", "ALL", "control")


def _sel(cls):
    return C if cls == "control" else (P if cls == "ALL" else P[P.cls == cls])


for tag, sig in SIGMAS:
    suf = "" if tag == "f" else "_c"
    print(f"\n=== RE-PLACEMENT ON THE RAW IMAGE, smoothing sigma {sig} vox ({'fine, forkimg' if tag == 'f' else 'coarse, ~nuclear radius'}); "
          f"per class (medians in um; '<=3' = share within 3 um of the GT centroid) ===", flush=True)
    hdr = (f"  {'class':<15} {'n':>4} {'d_gt0':>6} {'<=3':>5} | {'r3':>6} {'<=3':>5} | {'r5':>6} {'<=3':>5} {'moved':>6} {'dup':>5} | "
           f"{'r7':>6} {'<=3':>5} | {'cen':>6} {'<=3':>5} {'dup':>5} | {'lm<=3':>6} {'d_lm':>6} | {'c_peak':>6} {'c_gt':>6} {'c_r5':>6}")
    print(hdr, flush=True)
    print("  " + "-" * (len(hdr) - 2), flush=True)
    for cls in CLASSES:
        s = _sel(cls)
        if not len(s):
            continue
        print(f"  {cls:<15} {len(s):>4} {_q(s.d_gt0):>6.2f} {_frac(s.d_gt0 <= 3):>4.0f}% | "
              f"{_q(s[f'd_gt_r3{suf}']):>6.2f} {_frac(s[f'd_gt_r3{suf}'] <= 3):>4.0f}% | "
              f"{_q(s[f'd_gt_r5{suf}']):>6.2f} {_frac(s[f'd_gt_r5{suf}'] <= 3):>4.0f}% {_q(s[f'moved_r5{suf}']):>6.2f} "
              f"{_frac(s[f'd_ship_r5{suf}'] < DUP_UM):>4.0f}% | "
              f"{_q(s[f'd_gt_r7{suf}']):>6.2f} {_frac(s[f'd_gt_r7{suf}'] <= 3):>4.0f}% | "
              f"{_q(s[f'd_gt_cen{suf}']):>6.2f} {_frac(s[f'd_gt_cen{suf}'] <= 3):>4.0f}% {_frac(s[f'd_ship_cen{suf}'] < DUP_UM):>4.0f}% | "
              f"{_frac(s[f'd_lm50{suf}'] <= LM_UM):>5.0f}% {_q(s[f'd_lm50{suf}']):>6.2f} | "
              f"{_q(s[f'con_peak{suf}']):>6.2f} {_q(s[f'con_gt{suf}']):>6.2f} {_q(s[f'con_r5{suf}']):>6.2f}", flush=True)
    print("  a raw local maximum within 3 um of the GT centroid, by the maximum's contrast floor over the crop median (share):", flush=True)
    for cls in CLASSES:
        s = _sel(cls)
        if not len(s):
            continue
        print(f"    {cls:<15} n={len(s):>4}  " + "  ".join(f"con>={thr_:<4g} {_frac(s[f'{key_}{suf}'] <= LM_UM):>4.0f}%" for thr_, key_ in LM_THR)
              + f"   contrast at the point {_q(s[f'con_peak{suf}']):.2f}, at the GT voxel {_q(s[f'con_gt{suf}']):.2f}", flush=True)
print("  columns: d_gt0 = the point as placed (lnrescue's peak, or the shipped detection for the control); r3/r5/r7 = the raw "
      "maximum within that radius of the point; cen = the intensity centroid (3.5 um) around the 5-um maximum; moved = how far "
      "r5 moved the point; dup = the re-placed point lands within 3 um of a shipped primary (for the control: of ANOTHER shipped "
      "primary or itself, so read it only for the peaks); lm<=3 = a raw local maximum with contrast >= 0.5 within 3 um of the "
      "GT centroid exists; d_lm = the nearest such maximum within 7 um; c_* = contrast over the crop median at the point / at "
      "the GT voxel / at r5.", flush=True)

print("\n=== the refused peaks against the linked ones (the placement question), both scales ===", flush=True)
for tag, sig in SIGMAS:
    suf = "" if tag == "f" else "_c"
    for cls in ("ranker_refused", "linked", "control"):
        s = _sel(cls)
        if len(s):
            print(f"  [{tag}] {cls:<15} n={len(s):>4}  d_gt 0: {_q(s.d_gt0):.2f} -> r5: {_q(s[f'd_gt_r5{suf}']):.2f} -> cen: "
                  f"{_q(s[f'd_gt_cen{suf}']):.2f}; within 3 um: {_frac(s.d_gt0 <= 3):.0f}% -> {_frac(s[f'd_gt_r5{suf}'] <= 3):.0f}% -> "
                  f"{_frac(s[f'd_gt_cen{suf}'] <= 3):.0f}%; a raw maximum (con >= 0.5) within 3 um of the GT: "
                  f"{_frac(s[f'd_lm50{suf}'] <= LM_UM):.0f}%; r5 within 3 um of a shipped primary: {_frac(s[f'd_ship_r5{suf}'] < DUP_UM):.0f}%",
                  flush=True)
print("  reading: if the refused peaks have a raw maximum at the GT and r5/cen bring them to ~3 um without landing on shipped "
      "primaries (and the control is not made worse), a re-placement step before the ranker is the next bench; if the raw image "
      "has no maximum at the GT for them, no detector can place them there and the line closes on evidence.", flush=True)
warning_summary()
print(f"\nLNPLACE DONE in {(time.time() - t_all) / 60:.1f} min", flush=True)
print("LNPLACE OK", flush=True)
