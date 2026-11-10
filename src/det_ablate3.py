"""Which of the xl detector's 12 forward passes earn their place?

SCRIPT_VERSION 2026-08-25a-det-ablate3

THE QUESTION IS THE USER'S, from 2026-08-24: three models that look
similar ensemble far better than they should, and how many runs would it
take to try every combination? It has never been answered. `det_ablate`
v1 died on a missing import; when its design was re-read it was invalid
anyway (it scored at `DETECT_THRESH` 0.05, the candidate pool, giving
N/N_true of 17-18); `det_ablate2` v1 died on an inherited assert; and
then the GPU slot was needed elsewhere. **No ablation result exists.**

The two things that made v1 invalid are both fixed by work done
2026-08-25:

  * WRONG TEST SET. v1 scored on the 6bba holdout, and the user's
    objection stands -- the public weights may have been trained on it.
    `biohub-node-bright` v2 built and validated a clean alternative:
    Zebrahub dense GT, greedy 1:1 within 5 um, at the real detection
    gate. Zebrahub is a different instrument the public weights never
    saw.
  * WRONG THRESHOLD. v1 scored the candidate pool. This scores at
    `UNET_THRESH` 0.15, which is what the pipeline actually uses.

WHAT MAKES THIS CHEAP. The 12 passes are 3 branches x 4 flip views. Every
configuration is a SUBSET of the same 12 logit volumes, so the volumes
are computed ONCE per frame and every configuration is then formed on
CPU by averaging a subset. GPU cost equals a single detection pass, not
one pass per configuration.

Note the order of operations, which must be preserved exactly:
`detect()` averages the LOGITS over flip views, applies the sigmoid PER
BRANCH, and only then averages the branches. Averaging in the wrong
order would give a different detector and a meaningless ablation.

WHAT TO EXPECT, so the result can be read against a prediction rather
than rationalised afterwards:

  * branches 1 and 2 receive PIXEL-IDENTICAL input -- both are `tophat`
    (zh_masks.py:228). Their errors should correlate, so {1,2} should
    add little over either alone.
  * branch 0 is the only different view of the data (no top-hat) and has
    the WORST solo val_recall of the three: 0.5731 against 0.6589 and
    0.5789. If the ensemble works because of DIVERSITY rather than
    individual quality, {0,1} and {0,2} should beat {1,2} despite branch
    0 being the weakest member. That is the user's original question,
    stated as a falsifiable prediction.

ANCHOR. The full configuration -- all 3 branches, all 4 flips -- must
land near `biohub-node-bright` v2's reading of the same detector on the
same 41 crops with the same protocol: **recall 0.7659, precision
0.7362**. That run used every 5th frame and this one uses every 10th, so
the two are not identical samples and small differences are expected;
a large one means this reimplementation is not the pipeline's detector.
(v2 landed 0.7738 / 0.7350. It holds.)

v3: FIXED A REAL FLAW THE USER CAUGHT. v2 scored every configuration at
the SAME threshold, 0.15 -- but 0.15 is calibrated for the THREE-BRANCH
AVERAGE, not for a single branch. Branch 2 fired only 75,986 times
against branch 1's 125,419, and v2 read that as "B2 is nearly useless"
(solo F1 0.6239). That reading is not supported: **fewer detections at a
fixed threshold is the signature of a systematically lower probability
scale, not of a worse model.** B2 also had the HIGHEST precision of the
three singles (0.7291 against B0's 0.7235) -- it is conservative, not
bad, and a conservative member is exactly what an ensemble can use.

Same class of error as several others this session: comparing
quantities that live on different scales. So v3 sweeps the gate PER
CONFIGURATION and reports each one at its own best operating point.

The sweep is free. The greedy 1:1 match runs in descending score order,
so the assignment of the detections above any threshold t does not
depend on anything below t. **One greedy pass therefore yields TP and
detection counts at every threshold at once**: TP(t) is the number of
MATCHED detections scoring >= t.

v3 also adds `B0_f1` -- one branch, one view, **a single forward pass**
-- because v2 found the four flip views buy nothing (FULL_f1, 3 passes,
F1 0.7546 against FULL's 0.7539 at 12) and B0 alone nearly matches FULL.
If one pass ties twelve, that is a 12x cheaper detector.
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

SCRIPT_VERSION = "2026-08-25a-det-ablate3"
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


# ---- watershed masks for the Zebrahub crops ----------------------------
# Divisions cannot be learned from competition data: 151 GT divisions in
# the whole 199-dataset train split, 18 positives in the 44b6 training
# half, which trains to AUC 0.43 -- below chance. Zebrahub carries 2001
# and div-kinematics showed it alone reaches 0.808. So the division model
# lives on Zebrahub, and anything it is to learn from morphology needs
# masks HERE.
#
# What morphology buys that the 49 existing features cannot:
#
#   MASS CONSERVATION. At a true division V_a + V_b ~= V_mother, so the
#   ratio is ~1.0; for two arbitrary neighbours each about mother-sized it
#   is ~2.0. A physical constraint a spurious pair has no reason to meet.
#   Each daughter should also sit at (1/2)^(1/3) = 0.794 of the mother's
#   diameter, and the two should match each other.
#
# This is NOT sep_slope's situation. sep_slope failed because the filter
# stack already encodes it -- DIV_DIVERGE_UM >= 2.25 um is a hand-set
# one-frame ancestor, and the 14,159 -> 769 cut was already made with it
# before the model was consulted. The filter stack checks sister distance,
# mutual-nearest, both-children-continue and divergence. NOT ONE OF THEM
# TOUCHES SIZE, and neither does any of the division model's 49 columns:
# 25 geometric, 10 mother history, 10 daughter future, 4 coupling, every
# one of them positional.
#
# Masks are built exactly as biohub-hoct-gate built them and verified
# against: peaks at UNET_THRESH, watershed on the pooled heatmap seeded
# there. The pooled grid is ISOTROPIC by construction -- POOL=4 on Y/X
# turns (1.625, 0.40625, 0.40625) into (1.625, 1.625, 1.625) -- so no
# resampling is needed and regionprops reads real geometry.
#
# One GPU pass, saved per crop. Everything downstream is then free CPU:
# biohub-morph-probe proved skimage.regionprops on these masks reproduces
# HOCT's create_graph at 0.000e+00.

from skimage.segmentation import watershed                      # noqa: E402

MASK_FLOOR = 0.05
OUT = pathlib.Path("/kaggle/working")
BUDGET_S = 9.0 * 3600

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
POOLED = np.array([SCALE[0], SCALE[1] * POOL, SCALE[2] * POOL])
# =======================================================================
# THE ABLATION
# =======================================================================
FRAME_STRIDE = 10
MATCH_UM = 5.0          # node_bright v2's radius, unchanged
SCORE_THR = UNET_THRESH  # 0.15, the gate the pipeline actually uses

# (name, branch indices, flip indices)
CONFIGS = [
    ("B0",        (0,),      (0, 1, 2, 3)),
    ("B1",        (1,),      (0, 1, 2, 3)),
    ("B2",        (2,),      (0, 1, 2, 3)),
    ("B01",       (0, 1),    (0, 1, 2, 3)),
    ("B02",       (0, 2),    (0, 1, 2, 3)),
    ("B12",       (1, 2),    (0, 1, 2, 3)),
    ("FULL",      (0, 1, 2), (0, 1, 2, 3)),
    ("FULL_f1",   (0, 1, 2), (0,)),
    ("FULL_f2",   (0, 1, 2), (0, 1)),
    ("B0_f1",     (0,),      (0,)),          # ONE forward pass
    ("B1_f1",     (1,),      (0,)),
    ("B2_f1",     (2,),      (0,)),
]
# every gate the sweep reports; DETECT_THRESH is the peak-finding floor
GATES = (0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60)

CROPS = None
for _c in ("/kaggle/input/notebooks/faustine649/biohub-zebrahub-extract/"
           "crops",):
    if Path(_c).exists():
        CROPS = Path(_c)
if CROPS is None:
    hits = sorted(Path("/kaggle/input").glob("*/**/crops"))
    CROPS = hits[0] if hits else None
assert CROPS is not None, "zebrahub crops not mounted"
files = sorted(CROPS.glob("*.npz"))
assert files, f"{CROPS} is empty"

print(f"\n=== ABLATION over {len(CONFIGS)} configurations, "
      f"{len(files)} crops, every {FRAME_STRIDE}th frame ===", flush=True)
print(f"  peaks at DETECT_THRESH {DETECT_THRESH}, NMS, then swept "
      f"over gates {GATES}; greedy 1:1 within {MATCH_UM} um, "
      f"Zebrahub dense GT", flush=True)
print("  anchor: biohub-node-bright v2 read this detector at recall "
      "0.7659 / precision 0.7362", flush=True)


def all_logits(vol):
    """The 12 logit volumes, flips already undone. Computed once; every
    configuration below is a subset average of these."""
    out = {}
    for b, (models, ws, pp) in enumerate(_BRANCHES):
        x = pool_norm(vol, pp)
        for i in _FLIP_VIEWS:
            xv = np.ascontiguousarray(_flip(x, i))
            with torch.no_grad():
                lg = None
                for m, w in zip(models, ws):
                    l_ = m(torch.from_numpy(xv)[None, None].to(DEVICE)
                           )[0, 0].float().cpu().numpy()
                    lg = w * l_ if lg is None else lg + w * l_
            out[(b, i)] = _flip(lg, i).astype(np.float32)
    return out


def detect_from(logits, vol, branches, flips):
    """detect()'s tail, on a chosen subset. ORDER MATTERS: mean the
    LOGITS over flips, sigmoid PER BRANCH, then mean the branches --
    exactly what detect() does at zh_masks.py:241-255."""
    hs = []
    for b in branches:
        acc = None
        for i in flips:
            lg = logits[(b, i)]
            acc = lg if acc is None else acc + lg
        hs.append(1.0 / (1.0 + np.exp(-(acc / len(flips)))))
    h = hs[0] if len(hs) == 1 else np.mean(hs, axis=0)
    # PEAK AT DETECT_THRESH, NMS, then the CALLER gates -- the order the
    # caches were built in. Peak-finding directly at 0.15 is NOT the same
    # detector: _physical_nms is order dependent, so a candidate pool
    # containing the low-scoring peaks suppresses differently. Getting
    # this wrong would break the node_bright anchor for a reason that is
    # not an error, which is worse than no anchor.
    pk = peak_local_max(h, min_distance=1, threshold_abs=DETECT_THRESH,
                        exclude_border=False)
    if len(pk) == 0:
        return np.zeros((0, 3)), np.zeros(0)
    sc = h[pk[:, 0], pk[:, 1], pk[:, 2]].astype(float)
    coords = pk.astype(float)
    coords[:, 1] = coords[:, 1] * POOL + (POOL - 1) / 2
    coords[:, 2] = coords[:, 2] * POOL + (POOL - 1) / 2
    ref = np.array([_refine(vol, c)[0] for c in coords])
    # v3 returns the FULL post-NMS list; the caller gates it. NMS runs
    # before the gate in the cache's order, so gating afterwards at any
    # level is exactly what the pipeline would have produced there.
    return _physical_nms(ref, sc, NMS_UM)


def match(det, gt_um):
    """Greedy 1:1, highest score first -- node_bright v2's protocol.

    Returns TP and detection counts AT EVERY GATE from one pass. That is
    exact, not an approximation: the greedy loop consumes detections in
    descending score order, so whether a detection scoring >= t is
    matched cannot depend on any detection scoring < t. Running the
    match once and counting by score therefore gives the same answer as
    re-running it after each gate."""
    coords, sc = det
    tp = np.zeros(len(GATES), np.int64)
    nd = np.zeros(len(GATES), np.int64)
    if len(coords) == 0 or len(gt_um) == 0:
        return tp, nd
    det_um = coords * SCALE[None, :]
    tree = cKDTree(gt_um)
    claimed = set()
    matched = np.zeros(len(coords), bool)
    for i in np.argsort(-sc):
        near = tree.query_ball_point(det_um[i], MATCH_UM)
        if not near:
            continue
        near.sort(key=lambda g: float(np.linalg.norm(gt_um[g] - det_um[i])))
        free = [g for g in near if g not in claimed]
        if free:
            claimed.add(free[0])
            matched[i] = True
    for k, g in enumerate(GATES):
        m = sc >= g
        nd[k] = int(m.sum())
        tp[k] = int((m & matched).sum())
    return tp, nd


TP = {c[0]: np.zeros(len(GATES), np.int64) for c in CONFIGS}
ND = {c[0]: np.zeros(len(GATES), np.int64) for c in CONFIGS}
NG = 0
t0 = time.time()
done = 0
for f in files:
    if time.time() - t0 > BUDGET_S:
        print(f"\n  WARN wall-clock budget reached after {done} crops; "
              f"the counts below cover those crops only", flush=True)
        break
    z = np.load(f)
    if "image" not in z.files or "nodes" not in z.files:
        print(f"  WARN {f.stem}: missing image or nodes -> skipped",
              flush=True)
        continue
    img = z["image"]
    nodes = np.asarray(z["nodes"], float)
    if img.ndim != 4:
        print(f"  WARN {f.stem}: image ndim {img.ndim} != 4 -> skipped",
              flush=True)
        continue
    T = img.shape[0]
    gbt = defaultdict(list)
    for r in range(len(nodes)):
        tt = int(nodes[r, 0])
        if 0 <= tt < T:
            gbt[tt].append(r)
    for t in range(0, T, FRAME_STRIDE):
        gi = gbt.get(t, [])
        if not gi:
            continue
        vol = np.asarray(img[t])
        gt_um = nodes[gi, 1:] * SCALE[None, :]
        NG += len(gi)
        lg = all_logits(vol)
        for nm, bs, fs in CONFIGS:
            tp_, nd_ = match(detect_from(lg, vol, bs, fs), gt_um)
            TP[nm] += tp_
            ND[nm] += nd_
        del lg, vol
    done += 1
    print(f"  {done}/{len(files)} {f.stem}  ({time.time() - t0:.0f}s)",
          flush=True)
    del z, img
    gc.collect()

print(f"\n=== RESULTS: {NG} GT points over {done} crops ===", flush=True)
print("  Each configuration at ITS OWN best gate. Comparing all of them "
      "at 0.15 -- what v2 did -- is comparing probability scales that "
      "are not the same scale, which is what made B2 look useless.",
      flush=True)
print(f"\n  {'config':<10} {'passes':>6} {'gate*':>6} {'dets':>9} "
      f"{'recall':>8} {'prec':>8} {'F1*':>8} {'F1@.15':>8}", flush=True)
best = {}
at15 = {}
i15 = GATES.index(0.15)
for nm, bs, fs in CONFIGS:
    rec = TP[nm] / max(1, NG)
    pre = TP[nm] / np.maximum(1, ND[nm])
    f1 = 2 * rec * pre / np.maximum(1e-9, rec + pre)
    k = int(np.argmax(f1))
    best[nm] = (GATES[k], float(rec[k]), float(pre[k]), float(f1[k]))
    at15[nm] = float(f1[i15])
    print(f"  {nm:<10} {len(bs) * len(fs):>6} {GATES[k]:>6.2f} "
          f"{ND[nm][k]:>9} {rec[k]:>8.4f} {pre[k]:>8.4f} {f1[k]:>8.4f} "
          f"{f1[i15]:>8.4f}", flush=True)

print("\n=== the full F1-vs-gate curve, which is what settles B2 ===",
      flush=True)
print(f"  {'config':<10}" + "".join(f"{g:>8.2f}" for g in GATES),
      flush=True)
for nm, bs, fs in CONFIGS:
    rec = TP[nm] / max(1, NG)
    pre = TP[nm] / np.maximum(1, ND[nm])
    f1 = 2 * rec * pre / np.maximum(1e-9, rec + pre)
    print(f"  {nm:<10}" + "".join(f"{v:>8.4f}" for v in f1), flush=True)

print("\n=== reading it ===", flush=True)
print("  B2 AT ITS OWN GATE is the number that decides whether it is a "
      "weak model or merely a differently calibrated one. If its best F1 "
      "sits near B0's and B1's, v2's 'nearly useless' was an artefact of "
      "scoring everything at 0.15 and must be struck.", flush=True)
print("  Branches 1 and 2 get PIXEL-IDENTICAL input (both tophat), so "
      "B12 adding little over B1 or B2 alone would say the ensemble has "
      "two independent views, not three.", flush=True)
for a, b in (("B01", "B12"), ("B02", "B12"), ("FULL", "B01"),
             ("FULL", "B12"), ("FULL", "FULL_f1"), ("FULL", "B0_f1"),
             ("B0", "B0_f1"), ("B1", "B1_f1"), ("B2", "B2_f1")):
    if a in best and b in best:
        print(f"  {a} - {b}: F1* {best[a][3] - best[b][3]:+.4f}   "
              f"(gates {best[a][0]:.2f} vs {best[b][0]:.2f})", flush=True)

cheap = sorted(((len(bs) * len(fs), nm) for nm, bs, fs in CONFIGS
                if best[nm][3] >= best["FULL"][3] - 0.003))
print(f"\n  Configurations within 0.003 F1 of FULL, cheapest first: "
      f"{[f'{n}({p})' for p, n in cheap]}", flush=True)
print("  FULL runs 12 forward passes. The cheapest tie is the saving, "
      "and it is a saving on every future GPU run as well as on the "
      "submission's runtime budget.", flush=True)
print("\n  WARNING This is DETECTION quality, not adj_J. Detection "
      "recall was costed at ~1.75% of the score and precision is not in "
      "that decomposition at all, so a better F1 here is a reason to "
      "test a configuration in the pipeline, not evidence it scores "
      "higher. The gate that maximises F1 is also not necessarily the "
      "gate that maximises adj_J -- biohub-node-filter measured that "
      "removing detections costs edges faster than it removes noise.",
      flush=True)
print(f"\n  {time.time() - t0:.0f}s total", flush=True)
print("\nDET ABLATE3 DONE", flush=True)