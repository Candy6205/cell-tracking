"""fuserows -- the JUDGE: four models give evidence at every candidate, GT decides.

SCRIPT_VERSION 2026-08-30a-fuserows

detfuse2 promoted F4 / W1101 / DCheavy (zh dense-GT F1 0.7903 / 0.7913
/ 0.7929 against the shipped detector's 0.7774, with dets/frame at or
below it). This kernel turns those blends into detcache-layout files
(fdet_<arm>_<name>.npz, c{t}/s{t}, floor 0.05) for the 32 eval datasets
-- 12 6bba holdout + 12 manifest-excluded 44b6 + 8 zh crops -- so the
dcpool-style solve harness can score them in three domains with no code
change. One shared 4-heatmap pass per frame feeds all three arms.
Model stack and blending are detfuse2 verbatim (header below).

WHY. edgediag charged the pipeline's loss to detection COLD misses --
GT cells with no response >= 0.05 anywhere within 7 um of them in the
three-branch mean -- at 14.49% of zh edges, ~8x the entire linking-side
loss. dcmerge and dcpool both tried to use DeepCenter AFTER peak
extraction (as points) and closed. Never tried: DeepCenter as a FOURTH
HEATMAP inside the weighted average, BEFORE peak finding -- a dim cell
with no xl response but a DC response can surface as a peak in the
blend, while averaging suppresses DC's own false responses (the same
mechanism that made averaging beat unioning, 0.661, in the public
recipe's own measurement).

THE MODEL is dctrain's epoch-5 checkpoint: clean eval everywhere
(dccache3), covers 48.7% / 67.6% of xl's misses on 6bba / 44b6.

MEASURED IN TWO PLACES:
  * Zebrahub, dense GT (the honest denominator): full recall/precision/
    F1 per weight vector at its own gate, 5.0 um (anchor radius) and
    7.0 um (scorer radius). ANCHORS: FULL3 (1,1,1,0) must reproduce
    detfuse's 0.7774 F1* at 7 um / 0.7539 at 5 um; W110h its 0.7816.
  * The 24 competition eval datasets (12 6bba holdout + 12 excluded
    44b6), sparse GT: recall over ANNOTATED cells per arm, detections
    per frame (the surplus proxy), and the cold-miss rescue count --
    of the annotated cells the three-branch mean misses at floor 0.05,
    how many have a DC window response.

This is DETECTION evidence. A winning arm buys a pipeline test
(re-cache + ranker retrain + three-domain adj_J), not a submission.

Attach: xiaoleilian/biohub-unet3d-weights, ...-v2models,
faustine649/biohub-dctrain-175-weights, faustine649/cell-tracking,
kernel output faustine649/biohub-zebrahub-extract, competition. T4.
"""

import gc
import glob
import json
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

SCRIPT_VERSION = "2026-08-30a-fuserows"
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
print("device:", DEVICE, flush=True)

SCALE = np.array([1.625, 0.40625, 0.40625])
POOL = 4
REPAIR = True
CAND_THR = 0.05
UNET_THRESH = 0.15
NMS_UM = 4.0
DETECT_THRESH = min(UNET_THRESH, CAND_THR) if REPAIR else UNET_THRESH

# ---- our code (zarr-free competition IO) ------------------------------
CODE_ROOTS = [
    "/kaggle/input/datasets/faustine649/cell-tracking/cell tracking",
    "/kaggle/input/datasets/faustine649/cell-tracking",
    "/kaggle/input/cell-tracking/cell tracking",
    "/kaggle/input/cell-tracking",
]
code = next((c for c in CODE_ROOTS
             if pathlib.Path(c, "viz_cells.py").exists()), None)
assert code is not None, "attach faustine649/cell-tracking"
sys.path.insert(0, code)
from viz_cells import find_root, geff_path, read_geff, open_image  # noqa: E402

ROOT = find_root(None)
TRAIN_DIR = pathlib.Path(ROOT) / "train"

# ---- xl detect stack (det_ablate3 / detfuse verbatim) -----------------

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
        return np.array([z, y, x], float)
    zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1]
    return np.array([(zz * w).sum(), (yy * w).sum(), (xx * w).sum()]) / s

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


# ---- the fourth branch: dctrain epoch-5 DeepCenter --------------------
class _DCConvBlock3d(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        groups = min(8, co)
        self.block = nn.Sequential(
            nn.Conv3d(ci, co, 3, padding=1, bias=False),
            nn.GroupNorm(groups, co), nn.SiLU(inplace=True),
            nn.Conv3d(co, co, 3, padding=1, bias=False),
            nn.GroupNorm(groups, co), nn.SiLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class _DCDeepCenterUNet3D(nn.Module):
    def __init__(self, in_channels=1, base_channels=24):
        super().__init__()
        c = int(base_channels)
        self.enc1 = _DCConvBlock3d(in_channels, c)
        self.down1 = nn.MaxPool3d(2, 2)
        self.enc2 = _DCConvBlock3d(c, c * 2)
        self.down2 = nn.MaxPool3d(2, 2)
        self.enc3 = _DCConvBlock3d(c * 2, c * 4)
        self.down3 = nn.MaxPool3d(2, 2)
        self.bottleneck = _DCConvBlock3d(c * 4, c * 8)
        self.up3 = nn.ConvTranspose3d(c * 8, c * 4, 2, 2)
        self.dec3 = _DCConvBlock3d(c * 8, c * 4)
        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, 2, 2)
        self.dec2 = _DCConvBlock3d(c * 4, c * 2)
        self.up1 = nn.ConvTranspose3d(c * 2, c, 2, 2)
        self.dec1 = _DCConvBlock3d(c * 2, c)
        self.head = nn.Conv3d(c, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


_ep05 = sorted(pathlib.Path("/kaggle/input").glob(
    "datasets/*/biohub-dctrain-175-weights/best_ep05.pt")) or sorted(
    pathlib.Path("/kaggle/input").glob(
        "*/biohub-dctrain-175-weights/best_ep05.pt"))
assert _ep05, "best_ep05.pt not found -- attach biohub-dctrain-175-weights"
_ck = torch.load(_ep05[0], map_location=DEVICE, weights_only=False)
assert int(_ck.get("epoch", -1)) == 5, f"expected epoch 5, got {_ck.get('epoch')}"
_cfg = dict(_ck.get("config", {}))
DC = _DCDeepCenterUNet3D(base_channels=int(_cfg.get("base_channels", 24)))
DC.load_state_dict(_ck["model_state"])
DC = DC.to(DEVICE).eval()
DC_NORM = (float(_cfg.get("norm_lo_pct", 50.0)),
           float(_cfg.get("norm_hi_pct", 99.5)),
           float(_cfg.get("norm_clip_lo", -0.5)),
           float(_cfg.get("norm_clip_hi", 6.0)))
print(f"ep05 DeepCenter loaded from {_ep05[0]} (epoch "
      f"{_ck.get('epoch')}, norm {DC_NORM})", flush=True)
del _ck

# the training split, to assert the eval sets are outside it
_man = _ep05[0].parent / "split_manifest.json"
assert _man.exists(), "split_manifest.json missing beside best_ep05.pt"
_SEEN = set(json.loads(_man.read_text())["all"])


def _dc_normalize(volume):
    lo_pct, hi_pct, clip_lo, clip_hi = DC_NORM
    vol = np.asarray(volume, dtype=np.float32)
    lo = float(np.percentile(vol, lo_pct))
    hi = float(np.percentile(vol, hi_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    return np.clip((vol - lo) / (hi - lo), clip_lo, clip_hi).astype(np.float32)


def branch_maps4(vol):
    """Four probability maps: the three xl branches (detfuse verbatim)
    plus ep05 DeepCenter, each the sigmoid of its 4-flip logit mean."""
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
    x = _dc_normalize(pool_xy(vol))
    acc = None
    for i in _FLIP_VIEWS:
        xv = np.ascontiguousarray(_flip(x, i))
        with torch.no_grad():
            lg = DC(torch.from_numpy(xv)[None, None].to(DEVICE)
                    )[0, 0].float().cpu().numpy()
        lg = _flip(lg, i)
        acc = lg if acc is None else acc + lg
    hs.append(1.0 / (1.0 + np.exp(-np.clip(acc / len(_FLIP_VIEWS),
                                           -60.0, 60.0))))
    return hs


def detect_weighted(hs, vol, weights):
    w = np.asarray(weights, np.float64)
    h = sum(wi * hi for wi, hi in zip(w, hs)) / w.sum()
    pk = peak_local_max(h, min_distance=1, threshold_abs=DETECT_THRESH,
                        exclude_border=False)
    if len(pk) == 0:
        return np.zeros((0, 3)), np.zeros(0)
    sc = h[pk[:, 0], pk[:, 1], pk[:, 2]].astype(float)
    coords = pk.astype(float)
    coords[:, 1] = coords[:, 1] * POOL + (POOL - 1) / 2
    coords[:, 2] = coords[:, 2] * POOL + (POOL - 1) / 2
    ref = np.array([_refine(vol, c) for c in coords])
    return _physical_nms(ref, sc, NMS_UM)


# =====================================================================
# THE JUDGE (user's design, 2026-08-30): at every candidate location,
# ask all four models for their evidence and learn which patterns are
# cells. Positives = competition GT (in-domain); negatives = zh
# unmatched candidates (the only dense-label source), in TWO variants
# because zh completeness is under audit (zhcomplete):
#   judge_all       every unmatched zh candidate is a negative
#   judge_filtered  only unmatched AND dim AND ep05-silent candidates
# Brightness is used ONLY to filter negatives, never as a model
# feature -- a brightness feature would be the node-filter mistake
# (instrument-dependent absolute values) all over again.
# =====================================================================
from sklearn.ensemble import HistGradientBoostingClassifier    # noqa: E402
from sklearn.metrics import roc_auc_score                      # noqa: E402
import pickle                                                  # noqa: E402
import random                                                  # noqa: E402

CROPS = None
for _c in ("/kaggle/input/notebooks/faustine649/biohub-zebrahub-extract/"
           "crops", "/kaggle/input/biohub-zebrahub-extract/crops"):
    if pathlib.Path(_c).exists():
        CROPS = pathlib.Path(_c)
        break
if CROPS is None:
    import os as _os
    for _root, _dirs, _fn in _os.walk("/kaggle/input"):
        _dirs[:] = [d for d in _dirs
                    if "cell-tracking-during-development" not in d
                    and not d.endswith(".zarr") and not d.endswith(".geff")]
        if "crops" in _dirs:
            CROPS = pathlib.Path(_root) / "crops"
            break
assert CROPS is not None, "zebrahub crops not mounted"

POOLED_UM = np.array([SCALE[0], SCALE[1] * POOL, SCALE[2] * POOL])
DEDUP_UM = 2.0
N_PER_BATCH = 18
SEED = 20260830
COMP_STRIDE = 3
ZH_STRIDE = 5
FEATS = ["h0", "h1", "h2", "hd", "mean3", "n05", "n15"]


def union_candidates(hs):
    """Union of the four maps' peaks (floor 0.05), deduped at 2 um on
    the pooled grid. Returns pooled-voxel coords (int) ranked kept."""
    pks, vals = [], []
    for h in hs:
        pk = peak_local_max(h, min_distance=1, threshold_abs=0.05,
                            exclude_border=False)
        if len(pk):
            pks.append(pk)
            vals.append(h[pk[:, 0], pk[:, 1], pk[:, 2]])
    if not pks:
        return np.zeros((0, 3), int)
    allpk = np.concatenate(pks, 0)
    allv = np.concatenate(vals, 0)
    um = allpk * POOLED_UM[None, :]
    order = np.argsort(-allv)
    tree = cKDTree(um)
    killed = np.zeros(len(allpk), bool)
    keep = []
    for i in order:
        if killed[i]:
            continue
        keep.append(int(i))
        killed[tree.query_ball_point(um[i], r=DEDUP_UM)] = True
    return allpk[np.array(keep)]


def evidence(hs, pk):
    """Feature rows at pooled voxels pk: per-map window max (+-1 vox),
    3-branch mean, and branch-support counts."""
    Zp, Yp, Xp = hs[0].shape
    rows = np.zeros((len(pk), len(FEATS)), np.float32)
    for k, (zc, yc, xc) in enumerate(pk):
        sl = (slice(max(0, zc - 1), min(Zp, zc + 2)),
              slice(max(0, yc - 1), min(Yp, yc + 2)),
              slice(max(0, xc - 1), min(Xp, xc + 2)))
        w = [float(h[sl].max()) for h in hs]
        rows[k, 0:4] = w
        rows[k, 4] = (w[0] + w[1] + w[2]) / 3.0
        rows[k, 5] = sum(1 for v in w if v >= 0.05)
        rows[k, 6] = sum(1 for v in w if v >= 0.15)
    return rows


def to_fullres_um(pk):
    c = pk.astype(float)
    c[:, 1] = c[:, 1] * POOL + (POOL - 1) / 2
    c[:, 2] = c[:, 2] * POOL + (POOL - 1) / 2
    return c * SCALE[None, :]


def match_mask(cand_um, gt_um, rank_val):
    claimed = set()
    matched = np.zeros(len(cand_um), bool)
    if len(cand_um) and len(gt_um):
        tree = cKDTree(gt_um)
        for i in np.argsort(-rank_val):
            near = tree.query_ball_point(cand_um[i], 7.0)
            near.sort(key=lambda g: float(np.linalg.norm(gt_um[g]
                                                         - cand_um[i])))
            free = [g for g in near if g not in claimed]
            if free:
                claimed.add(free[0])
                matched[i] = True
    return matched


# ---- training sets ----------------------------------------------------
names_all = sorted(p.name[:-len(".zarr")] for p in TRAIN_DIR.glob("*.zarr"))
pool_names = sorted(_SEEN)     # the 175 the detector saw; eval 24 excluded
rng = random.Random(SEED)
TRAIN_6BBA = sorted(rng.sample(
    [n for n in pool_names if n.startswith("6bba_")], N_PER_BATCH))
TRAIN_44B6 = sorted(rng.sample(
    [n for n in pool_names if n.startswith("44b6_")], N_PER_BATCH))
CROPS_ALL = sorted(CROPS.glob("*.npz"))
ZH_EVAL = {f.stem for f in CROPS_ALL[:8]}       # the 8 eval crops, HELD OUT
ZH_TRAIN = [f for f in CROPS_ALL if f.stem not in ZH_EVAL]
print(f"\njudge training sets (seed {SEED}): {len(TRAIN_6BBA)} 6bba + "
      f"{len(TRAIN_44B6)} 44b6 from the 175 pool, {len(ZH_TRAIN)} zh "
      f"crops (the 8 eval crops held out)", flush=True)
print("  6bba: " + " ".join(TRAIN_6BBA), flush=True)
print("  44b6: " + " ".join(TRAIN_44B6), flush=True)

X_rows, y_rows, dom_rows, grp_rows = [], [], [], []
neg_bright, neg_full3 = [], []          # zh negatives' filter columns
base_counts = [0, 0, 0]                 # frames, FULL3>=0.15, FULL3>=0.05
judge_pos_scores = []                   # comp positives, for calibration
t0 = time.time()

for k, name in enumerate(TRAIN_6BBA + TRAIN_44B6):
    img = open_image(TRAIN_DIR / f"{name}.zarr")
    gt = read_geff(geff_path(ROOT, "train", name))
    gbt = defaultdict(list)
    for tt, z_, y_, x_ in zip(gt["t"], gt["z"], gt["y"], gt["x"]):
        gbt[int(tt)].append([float(z_), float(y_), float(x_)])
    T = img.shape[0]
    for t in range(0, T, COMP_STRIDE):
        gi = gbt.get(t)
        if not gi:
            continue
        vol = np.asarray(img[t])
        hs = branch_maps4(vol)
        pk = union_candidates(hs)
        if len(pk) == 0:
            continue
        rows_ = evidence(hs, pk)
        cand_um = to_fullres_um(pk)
        gt_um = np.asarray(gi, float) * SCALE[None, :]
        matched = match_mask(cand_um, gt_um, rows_[:, 4])
        # comp: positives only; unmatched are UNLABELED (sparse GT)
        X_rows.append(rows_[matched])
        y_rows.append(np.ones(int(matched.sum()), np.int8))
        dom_rows += ["comp"] * int(matched.sum())
        grp_rows += [name] * int(matched.sum())
        h3 = (hs[0] + hs[1] + hs[2]) / 3.0
        base_counts[0] += 1
        base_counts[1] += int((h3[pk[:, 0], pk[:, 1], pk[:, 2]]
                               >= 0.15).sum())
        base_counts[2] += int((h3[pk[:, 0], pk[:, 1], pk[:, 2]]
                               >= 0.05).sum())
        del vol, hs
    print(f"  comp {k + 1}/{len(TRAIN_6BBA) + len(TRAIN_44B6)} {name} "
          f"({time.time() - t0:.0f}s)", flush=True)
    gc.collect()

for k, f in enumerate(ZH_TRAIN):
    z = np.load(f)
    img = z["image"]
    nodes = np.asarray(z["nodes"], float)
    T = img.shape[0]
    gbt = defaultdict(list)
    for r_ in range(len(nodes)):
        tt = int(nodes[r_, 0])
        if 0 <= tt < T:
            gbt[tt].append(nodes[r_, 1:])
    for t in range(0, T, ZH_STRIDE):
        gi = gbt.get(t, [])
        if not gi:
            continue
        vol = np.asarray(img[t])
        hs = branch_maps4(vol)
        pk = union_candidates(hs)
        if len(pk) == 0:
            continue
        rows_ = evidence(hs, pk)
        cand_um = to_fullres_um(pk)
        gt_um = np.asarray(gi, float) * SCALE[None, :]
        matched = match_mask(cand_um, gt_um, rows_[:, 4])
        X_rows.append(rows_)
        y_rows.append(matched.astype(np.int8))
        dom_rows += ["zh"] * len(rows_)
        grp_rows += [f.stem] * len(rows_)
        # filter columns for the unmatched (negative candidates):
        # raw image brightness at the peak, per-crop calibrated later
        br = np.zeros(len(pk), np.float32)
        for j, (zc, yc, xc) in enumerate(pk):
            y0, x0 = yc * POOL, xc * POOL
            sl = (slice(max(0, zc - 1), min(vol.shape[0], zc + 2)),
                  slice(max(0, y0 - 4), min(vol.shape[1], y0 + 5)),
                  slice(max(0, x0 - 4), min(vol.shape[2], x0 + 5)))
            br[j] = float(vol[sl].max()) if vol[sl].size else 0.0
        neg_bright.append(br)
        neg_full3.append(rows_[:, 4])
        del vol, hs
    print(f"  zh {k + 1}/{len(ZH_TRAIN)} {f.stem} "
          f"({time.time() - t0:.0f}s)", flush=True)
    del z, img
    gc.collect()

X = np.concatenate(X_rows, 0)
y = np.concatenate(y_rows, 0)
dom = np.asarray(dom_rows)
grp = np.asarray(grp_rows)
BR = np.concatenate(neg_bright, 0) if neg_bright else np.zeros(0)
print(f"\nrows: {len(X):,} total; comp positives "
      f"{int((dom == 'comp').sum()):,}; zh rows "
      f"{int((dom == 'zh').sum()):,} of which positives "
      f"{int(y[dom == 'zh'].sum()):,}", flush=True)

# per-crop brightness threshold: q25 of MATCHED zh candidates' brightness
zh_mask = dom == "zh"
crop_thr = {}
zi = np.where(zh_mask)[0]
BR_full = np.zeros(len(X), np.float32)
BR_full[zi] = BR
for c in sorted(set(grp[zh_mask])):
    m = zh_mask & (grp == c) & (y == 1)
    crop_thr[c] = float(np.percentile(BR_full[m], 25)) if m.sum() else 0.0
dim = np.zeros(len(X), bool)
for c, thr in crop_thr.items():
    dim |= zh_mask & (grp == c) & (BR_full < thr)
ep05_silent = X[:, 3] < 0.05

# ---- the two variants -------------------------------------------------
def train_variant(tag, keep_neg_mask):
    train_mask = (y == 1) | keep_neg_mask
    Xt, yt, gt_ = X[train_mask], y[train_mask], grp[train_mask]
    crops_sorted = sorted(set(gt_[(dom[train_mask] == 'zh')]))
    hold = set(crops_sorted[::2])
    tr = np.asarray([g not in hold for g in gt_])
    m = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    m.fit(Xt[tr], yt[tr])
    te = ~tr
    auc = float("nan")
    if te.sum() and len(set(yt[te])) > 1:
        auc = roc_auc_score(yt[te], m.predict_proba(Xt[te])[:, 1])
    n_neg = int((yt == 0).sum())
    print(f"  {tag}: {int(yt.sum()):,} pos / {n_neg:,} neg; held-out-"
          f"crop AUC {auc:.4f}", flush=True)
    mf = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    mf.fit(Xt, yt)
    return mf, auc, n_neg


print("\n=== TRAINING THE TWO VARIANTS ===", flush=True)
neg_all = (y == 0) & zh_mask
neg_filt = neg_all & dim & ep05_silent
judge_all, auc_all, n_all = train_variant("judge_all", neg_all)
judge_filt, auc_filt, n_filt = train_variant("judge_filtered", neg_filt)
print(f"  filtered negatives kept {n_filt:,} of {n_all:,} "
      f"({100.0 * n_filt / max(1, n_all):.1f}%) -- the discarded rest "
      f"are bright and/or ep05-seconded, i.e. possibly unlabeled cells",
      flush=True)

# ---- calibration: judge-score thresholds matching FULL3's output ------
print("\n=== CALIBRATION (no-surplus discipline) ===", flush=True)
per_fr_15 = base_counts[1] / max(1, base_counts[0])
per_fr_05 = base_counts[2] / max(1, base_counts[0])
comp_mask = dom == "comp"
cal = {}
for tag, mdl in (("judge_all", judge_all), ("judge_filtered", judge_filt)):
    s_pos = mdl.predict_proba(X[comp_mask])[:, 1]
    cal[tag] = dict(pos_q10=float(np.percentile(s_pos, 10)),
                    pos_q50=float(np.percentile(s_pos, 50)))
    print(f"  {tag}: comp-positive judge scores q10/q50 "
          f"{cal[tag]['pos_q10']:.3f}/{cal[tag]['pos_q50']:.3f}",
          flush=True)
print(f"  FULL3 reference on the same frames: {per_fr_15:.1f} "
      f"cands/frame >= 0.15, {per_fr_05:.1f} >= 0.05. fusecache2 must "
      f"pick its operating threshold to MATCH these counts per frame "
      f"on the eval data -- calibration by output rate, not by score.",
      flush=True)

OUT = pathlib.Path("/kaggle/working")
for tag, mdl, auc_ in (("all", judge_all, auc_all),
                       ("filtered", judge_filt, auc_filt)):
    with open(OUT / f"fuse_judge_{tag}.pkl", "wb") as fh:
        pickle.dump({"model": mdl, "feats": FEATS, "auc_heldout": auc_,
                     "base_per_fr_15": per_fr_15,
                     "base_per_fr_05": per_fr_05,
                     "train_6bba": TRAIN_6BBA, "train_44b6": TRAIN_44B6,
                     "seed": SEED}, fh)
np.savez_compressed(OUT / "judge_rows.npz", X=X, y=y, dom=dom, grp=grp,
                    bright=BR_full, dim=dim.astype(np.int8),
                    ep05_silent=ep05_silent.astype(np.int8))
print("\nsaved fuse_judge_all.pkl, fuse_judge_filtered.pkl, "
      "judge_rows.npz", flush=True)
print(f"\nFUSEROWS DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
