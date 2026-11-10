"""det_ablate2.py -- the branch ablation, on the CLEAN ruler.

v1 was invalid twice and both mistakes are worth not repeating.

WRONG THRESHOLD: it peaked at `DETECT_THRESH` = min(UNET_THRESH,
CAND_THR) = 0.05, because REPAIR is on -- that is the CANDIDATE pool, not
the seeds the pipeline turns into nodes at UNET_THRESH = 0.15. N/N_true
came out 17-18 against the pipeline's booked 1.01-1.30, which is the tell.

WRONG TEST SET, and this one matters more: it scored on the 12-dataset
6bba holdout, which progress.md already records as "duplicate-frame-
inflated AND possibly inside the public weights' training split".
xiaoleilian's three checkpoints have an UNKNOWN training split. Grading a
detector on data it may have memorised measures nothing at all -- and the
project already built the clean ruler for precisely this question and
wrote "All training reads there".

So: the Zebrahub crops, dense GT, a domain no competition-weights author
could have trained on. Recorded reference points on this same ruler are
xl at recall 0.766 / precision 0.736 and the public detector at 0.650 /
0.637.

Nine configurations from ONE inference pass -- all seven non-empty branch
subsets at four flip views, plus all-three at one and at two views --
because per branch the logits accumulate and the accumulator is snapshot
after 1, 2 and 4 views.

The question: branches 1 and 2 (`traintophat`, `v2_tophat_b32`) receive
PIXEL-IDENTICAL top-hat input, so their errors should be correlated and
they should cancel each other's mistakes poorly, while branch 0
(`bright`) is the only genuinely different view of the data and carries
the worst solo val_recall -- which is not what earns a place in an
ensemble. Nobody has ever measured this.

Detection only. It does not become an adj_J until the full pipeline says
so; that failed twice today already.
"""
import gc
import glob
import json
import pathlib
import pickle
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

SCRIPT_VERSION = "2026-08-24g-det-ablate2"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

DEVICE = "cpu"
if torch.cuda.is_available():
    try:
        _p = nn.Conv3d(1, 1, 3).to("cuda")
        _ = _p(torch.zeros(1, 1, 4, 4, 4, device="cuda")).cpu()
        DEVICE = "cuda"
        del _p
    except Exception as e:                                     # noqa: BLE001
        print("GPU present but conv3d unusable -> CPU:", str(e)[:80])

# ---- detect-side constants (xl-repro verbatim) ------------------------
SCALE = np.array([1.625, 0.40625, 0.40625])
POOL = 4
REPAIR = True
CAND_THR = 0.05
UNET_THRESH = 0.15
NMS_UM = 4.0
DETECT_THRESH = min(UNET_THRESH, CAND_THR) if REPAIR else UNET_THRESH

# ---- pipeline constants (ranker-diag verbatim: the 0.9345 config) -----
GATE_UM = 10.0
MAX_LINK_UM = 10.0
TIGHT_UM = 6.0
LINK_SIM_W = 2.0
SHORT_MIN = 6
LINEFIT_WEIGHT = 0.8
LINEFIT_WINDOW = 2
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
ILP_FLOOR = 0.10                            # candidates fed to the ILP
ILP_DISAPP = 1.5
ILP_DIV_W = 0.55   # fine sweep argmax
ILP_WINDOW = 25    # wincal: 0.9382 vs global 0.9368, 2.08x faster

CANDIROOT = ["/kaggle/input/biohub-cell-tracking-during-development",
             "/kaggle/input/competitions/biohub-cell-tracking-during-development"]
ROOT = next((p for p in CANDIROOT if Path(p, "train").exists()), None)
assert ROOT is not None, "competition not attached"
TEST_DIR = Path(ROOT) / "test"

# ---- xl detect stack (xl-repro verbatim) ------------------------------

def _find_weight(name):
    cands = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert cands, f"{name} not found"
    return cands[0]

def _block(ci, co):
    return nn.Sequential(nn.Conv3d(ci, co, 3, padding=1), nn.BatchNorm3d(co),
                         nn.ReLU(inplace=True),
                         nn.Conv3d(co, co, 3, padding=1), nn.BatchNorm3d(co),
                         nn.ReLU(inplace=True))

class UNet3D(nn.Module):
    def __init__(self, base=24):
        super().__init__()
        self.e1 = _block(1, base)
        self.e2 = _block(base, base * 2)
        self.e3 = _block(base * 2, base * 4)
        self.pool = nn.MaxPool3d(2)
        self.bott = _block(base * 4, base * 8)
        self.u3 = nn.ConvTranspose3d(base * 8, base * 4, 2, stride=2)
        self.d3 = _block(base * 8, base * 4)
        self.u2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.d2 = _block(base * 4, base * 2)
        self.u1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.d1 = _block(base * 2, base)
        self.out = nn.Conv3d(base, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.bott(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


def _load_model(name):
    ck = torch.load(_find_weight(name), map_location=DEVICE)
    m = UNet3D(base=ck.get("base", 24)).to(DEVICE)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    print("loaded", name, "| val_recall", ck.get("val_recall"), flush=True)
    return m

MODELS = [_load_model("unet3d_bright.pt"), _load_model("unet3d_traintophat.pt")]

def read_array_meta(zp):
    with open(Path(zp) / "0" / "zarr.json") as f:
        m = json.load(f)
    return dict(shape=tuple(m["shape"]), dtype=np.dtype(m["data_type"]))

_ZC = {}

def load_volume(zp, t, meta=None):
    try:
        import zarr
        k = str(zp)
        if k not in _ZC:
            _ZC[k] = zarr.open(k, mode="r")["0"]
        return np.asarray(_ZC[k][t])
    except Exception:
        import blosc2
        if meta is None:
            meta = read_array_meta(zp)
        buf = blosc2.decompress(
            open(Path(zp) / "0" / "c" / str(t) / "0" / "0" / "0", "rb").read())
        return np.frombuffer(buf, dtype=meta["dtype"]).reshape(meta["shape"][1:])

def pool_xy(vol, f=POOL):
    Z, Y, X = vol.shape
    Y2, X2 = (Y // f) * f, (X // f) * f
    v = vol[:, :Y2, :X2].astype(np.float32, copy=False)
    return v.reshape(Z, Y2 // f, f, X2 // f, f).mean(axis=(2, 4))

def pool_norm(vol, preproc=""):
    p = pool_xy(vol)
    if preproc == "tophat":
        from scipy.ndimage import grey_opening
        p = np.clip(p - grey_opening(p, size=(1, 7, 7)), 0.0, None)
    lo = float(np.percentile(p, 50))
    hi = float(np.percentile(p, 99.5))
    return np.clip((p - lo) / (hi - lo + 1e-6), -0.5, 6.0).astype(np.float32)

def _refine(vol, zyx, rz=2, ryx=5):
    Z, Y, X = vol.shape
    z, y, x = (int(round(v)) for v in zyx)
    z0, z1 = max(0, z - rz), min(Z, z + rz + 1)
    y0, y1 = max(0, y - ryx), min(Y, y + ryx + 1)
    x0, x1 = max(0, x - ryx), min(X, x + ryx + 1)
    crop = vol[z0:z1, y0:y1, x0:x1].astype(np.float32)
    bg = float(crop.min())
    w = np.clip(crop - bg, 0, None)
    s = float(w.sum())
    if s <= 0:
        return np.array([z, y, x], float), 0.0
    zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1]
    return (np.array([(zz * w).sum(), (yy * w).sum(), (xx * w).sum()]) / s,
            float(crop.max() - bg))

def _physical_nms(coords, scores, radius_um, scale=SCALE):
    if len(coords) <= 1:
        return coords, scores
    pts = coords * scale[None, :]
    order = np.argsort(-scores)
    tree = cKDTree(pts)
    killed = np.zeros(len(coords), bool)
    keep = []
    for i in order:
        if killed[i]:
            continue
        keep.append(int(i))
        killed[tree.query_ball_point(pts[i], r=radius_um)] = True
    keep = np.array(keep)
    return coords[keep], scores[keep]

_BRANCHES = [([MODELS[0]], [1.0], ""), ([MODELS[1]], [1.0], "tophat"),
             ([_load_model("unet3d_v2_tophat_b32.pt")], [1.0], "tophat")]
_FLIP_VIEWS = (0, 1, 2, 3)

def _flip(x, i):
    if i & 1:
        x = np.flip(x, -2)
    if i & 2:
        x = np.flip(x, -1)
    return x

def detect(vol):
    hs = []
    for models, ws, pp in _BRANCHES:
        x = pool_norm(vol, pp)
        acc = None
        for i in _FLIP_VIEWS:
            xv = np.ascontiguousarray(_flip(x, i))
            with torch.no_grad():
                lg = None
                for m, w in zip(models, ws):
                    l_ = m(torch.from_numpy(xv)[None, None].to(DEVICE)
                           )[0, 0].float().cpu().numpy()
                    lg = w * l_ if lg is None else lg + w * l_
            lg = _flip(lg, i)
            acc = lg if acc is None else acc + lg
        hs.append(1.0 / (1.0 + np.exp(-(acc / len(_FLIP_VIEWS)))))
    h = hs[0] if len(hs) == 1 else np.mean(hs, axis=0)
    pk = peak_local_max(h, min_distance=1, threshold_abs=DETECT_THRESH,
                        exclude_border=False)
    if len(pk) == 0:
        return np.zeros((0, 3)), np.zeros(0)
    sc = h[pk[:, 0], pk[:, 1], pk[:, 2]].astype(float)
    coords = pk.astype(float)
    coords[:, 1] = coords[:, 1] * POOL + (POOL - 1) / 2
    coords[:, 2] = coords[:, 2] * POOL + (POOL - 1) / 2
    ref = np.array([_refine(vol, c)[0] for c in coords])
    return _physical_nms(ref, sc, NMS_UM)


# ------------------------------------------------------------- graph ops

def avail_T(zp):
    meta = read_array_meta(zp)
    T = meta["shape"][0]
    present = [t for t in range(T)
               if (Path(zp) / "0" / "c" / str(t) / "0" / "0" / "0").exists()]
    return max(present) + 1 if present else 0

# ---------------- cache xl detections for the rest of train --------------
# The census settled it: 151 GT divisions across all 199 train datasets, and
# only 31 of them sit in the 44 datasets we have detection caches for. This
# caches the other 155 with the SAME xl detector (same three branches, same
# flip views, same thresholds), so the pools can simply be merged.
#
# Two payoffs, both independent of the division work:
#   - ~127 usable in-domain divisions instead of 26 (the anchor set that the
#     Zebrahub / synthetic question needs in order to be answerable at all)
#   - the edge ranker's training pool goes from 32 datasets to 187
#
# Output format matches biohub-edge-ranker exactly: one detcache_<name>.npz
# per dataset with keys c{t} (coords Nx3) and s{t} (scores N), so the
# existing `CACHES` glob picks both up with no code change.

TRAIN_DIR = Path(ROOT) / "train"


# ---- the ablation, v2: right ruler, right threshold --------------------
# v1 (biohub-det-ablate) was invalid twice over and both are recorded so
# nobody repeats them:
#
#  1. WRONG THRESHOLD. It peaked at `DETECT_THRESH`, which is
#     min(UNET_THRESH, CAND_THR) = 0.05 because REPAIR is on -- the
#     CANDIDATE pool, not the seeds. The pipeline's nodes are what
#     `seeds_of` keeps at UNET_THRESH = 0.15. The tell was N/N_true
#     running 17-18 where the pipeline books 1.01-1.30.
#  2. WRONG TEST SET. It scored on the 12-dataset 6bba holdout, which
#     progress.md already records as "duplicate-frame-inflated AND
#     possibly inside the public weights' training split". xiaoleilian's
#     three checkpoints have an UNKNOWN split. Grading a detector on data
#     it may have memorised measures nothing.
#
# The project built a clean ruler for exactly this question and wrote
# "All training reads there": the Zebrahub crops, dense GT, a domain no
# competition-weights author could have trained on. Use it.
#
# Peaking directly at UNET_THRESH is equivalent to the shipped path:
# peak_local_max returns maxima >= the threshold, and physical NMS keeps
# highest-first, so a sub-0.15 peak can never suppress a surviving one.

import itertools                                                # noqa: E402

CODE_ROOTS = [
    "/kaggle/input/datasets/faustine649/cell-tracking/cell tracking",
    "/kaggle/input/datasets/faustine649/cell-tracking",
    "/kaggle/input/cell-tracking/cell tracking",
    "/kaggle/input/cell-tracking",
]
_code = next((c for c in CODE_ROOTS if Path(c, "viz_cells.py").exists()),
             None)
if _code:
    sys.path.insert(0, _code)

BR_NAMES = ["bright", "traintophat", "v2_tophat_b32"]
VIEW_SNAPSHOTS = (1, 2, 4)
MATCH_UM = 5.0
SEED_THRESH = UNET_THRESH          # 0.15 -- the pipeline's actual nodes
BUDGET_S = 9.5 * 3600              # leave room under the 12 h cap

CONFIGS = []
for r in (1, 2, 3):
    for sub in itertools.combinations(range(3), r):
        CONFIGS.append(("+".join(BR_NAMES[i] for i in sub), sub, 4))
CONFIGS.append(("ALL @1 view", (0, 1, 2), 1))
CONFIGS.append(("ALL @2 views", (0, 1, 2), 2))

CROPS = None
for c in ["/kaggle/input/biohub-zebrahub-extract/crops",
          "/kaggle/input/notebooks/faustine649/biohub-zebrahub-extract/"
          "crops"]:
    if Path(c).exists():
        CROPS = Path(c)
        break
if CROPS is None:
    hits = list(Path("/kaggle/input").glob("*/**/crops"))
    CROPS = hits[0] if hits else None
assert CROPS is not None, "zebrahub crops not mounted"
files = sorted(CROPS.glob("*.npz"))
assert files, f"{CROPS} is empty"
print(f"\n=== {len(CONFIGS)} CONFIGURATIONS on {len(files)} ZEBRAHUB "
      f"CROPS, seed threshold {SEED_THRESH} ===", flush=True)
print(f"  crops: {CROPS}", flush=True)


def branch_heatmaps(vol):
    """hm[b][v]: branch b's heatmap from the first v flip views. Logits
    accumulate -- that is where the flip TTA averages -- and the sigmoid
    is applied at each snapshot."""
    out = []
    for models, ws, pp in _BRANCHES:
        x = pool_norm(vol, pp)
        acc, snaps = None, {}
        for i, view in enumerate(_FLIP_VIEWS):
            xv = np.ascontiguousarray(_flip(x, view))
            with torch.no_grad():
                lg = None
                for m, w in zip(models, ws):
                    l_ = m(torch.from_numpy(xv)[None, None].to(DEVICE)
                           )[0, 0].float().cpu().numpy()
                    lg = w * l_ if lg is None else lg + w * l_
            lg = _flip(lg, view)
            acc = lg if acc is None else acc + lg
            n = i + 1
            if n in VIEW_SNAPSHOTS:
                snaps[n] = 1.0 / (1.0 + np.exp(-(acc / n)))
        out.append(snaps)
    return out


def peaks_from(h, vol):
    """The rest of the xl path unchanged: peak, un-pool, full-resolution
    centroid refinement, physical NMS."""
    pk = peak_local_max(h, min_distance=1, threshold_abs=SEED_THRESH,
                        exclude_border=False)
    if len(pk) == 0:
        return np.zeros((0, 3))
    sc = h[pk[:, 0], pk[:, 1], pk[:, 2]].astype(float)
    coords = pk.astype(float)
    coords[:, 1] = coords[:, 1] * POOL + (POOL - 1) / 2
    coords[:, 2] = coords[:, 2] * POOL + (POOL - 1) / 2
    ref = np.array([_refine(vol, c)[0] for c in coords])
    return _physical_nms(ref, sc, NMS_UM)[0]


tally = {lab: {"tp": 0, "fp": 0, "fn": 0, "n": 0} for lab, _, _ in CONFIGS}
n_true_total = 0
n_crops = 0
t0 = time.time()

for f in files:
    if time.time() - t0 > BUDGET_S:
        print(f"\n  WARN wall-clock budget reached: {len(files) - n_crops} "
              f"crops NOT processed, so every row below is measured on "
              f"{n_crops} crops, not {len(files)}", flush=True)
        break
    z = np.load(f)
    if "image" not in z.files:
        print(f"  WARN {f.stem}: no 'image' -> skipped", flush=True)
        continue
    img = z["image"]
    if img.ndim != 4:
        print(f"  WARN {f.stem}: image ndim {img.ndim} != 4 -> skipped",
              flush=True)
        continue
    if "nodes" not in z.files:
        print(f"  WARN {f.stem}: no 'nodes' ground truth -> skipped, so "
              f"this crop contributes to no row", flush=True)
        continue
    nodes = np.asarray(z["nodes"], float)
    gt_by_t = defaultdict(list)
    for r in range(len(nodes)):
        gt_by_t[int(nodes[r, 0])].append(r)

    T = img.shape[0]
    for t in range(T):
        vol = np.asarray(img[t])
        hm = branch_heatmaps(vol)
        idx = gt_by_t.get(t, [])
        gpts = nodes[idx, 1:] * SCALE[None, :] if idx else np.zeros((0, 3))
        n_true_total += len(idx)
        gtree = cKDTree(gpts) if len(gpts) else None
        for lab, sub, nv in CONFIGS:
            h = hm[sub[0]][nv] if len(sub) == 1 else \
                np.mean([hm[b][nv] for b in sub], axis=0)
            coords = peaks_from(h, vol)
            d = tally[lab]
            d["n"] += len(coords)
            if gtree is None or len(coords) == 0:
                d["fp"] += len(coords)
                d["fn"] += len(gpts)
                continue
            dd, gi = gtree.query(coords * SCALE[None, :])
            claimed, tp = set(), 0
            for k in np.argsort(dd):
                if dd[k] <= MATCH_UM and int(gi[k]) not in claimed:
                    claimed.add(int(gi[k]))
                    tp += 1
            d["tp"] += tp
            d["fp"] += len(coords) - tp
            d["fn"] += len(gpts) - tp
        del hm
        if t % 25 == 0:
            gc.collect()
    n_crops += 1
    print(f"  {n_crops}/{len(files)} {f.stem} T={T} "
          f"({time.time() - t0:.0f}s)", flush=True)
    del z, img
    gc.collect()

print(f"\n=== ZEBRAHUB CLEAN RULER: {n_crops} crops, {n_true_total} GT "
      f"nodes, seed threshold {SEED_THRESH} ===", flush=True)
print("  Reference: the recorded xl number on this ruler is recall 0.766 "
      "/ prec 0.736, and the public detector 0.650 / 0.637.", flush=True)
print(f"  {'configuration':<34} {'recall':>7} {'prec':>7} {'F1':>7} "
      f"{'N/N_true':>9}", flush=True)
rows = []
for lab, _sub, _nv in CONFIGS:
    d = tally[lab]
    rec = d["tp"] / max(d["tp"] + d["fn"], 1)
    pre = d["tp"] / max(d["tp"] + d["fp"], 1)
    f1 = 2 * rec * pre / max(rec + pre, 1e-9)
    ratio = d["n"] / max(n_true_total, 1)
    rows.append((f1, rec, pre, ratio, lab))
    print(f"  {lab:<34} {rec:>7.4f} {pre:>7.4f} {f1:>7.4f} "
          f"{ratio:>9.3f}", flush=True)

print("\n  ranked by F1:", flush=True)
for f1, rec, pre, ratio, lab in sorted(rows, reverse=True):
    print(f"    {f1:.4f}  rec {rec:.4f}  prec {pre:.4f}  {lab}", flush=True)

full = next(r for r in rows if r[4] == "+".join(BR_NAMES))
print(f"\n  shipped configuration (all three, 4 views): F1 {full[0]:.4f}, "
      f"recall {full[1]:.4f}, precision {full[2]:.4f}", flush=True)
better = [r for r in rows if r[0] > full[0]]
if better:
    b = max(better)
    print(f"  {len(better)} configuration(s) beat it; best {b[4]} at F1 "
          f"{b[0]:.4f} ({b[0] - full[0]:+.4f}), recall {b[1]:.4f}, "
          f"precision {b[2]:.4f}", flush=True)
    print("  This is DETECTION on an out-of-domain ruler. It does not "
          "become an adj_J until the full pipeline says so -- twice today "
          "an offline win did not survive that.", flush=True)
else:
    print("  nothing beat it: all 12 forward passes earn their place.",
          flush=True)

print("\nDET ABLATION v2 DONE", flush=True)
