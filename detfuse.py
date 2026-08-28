"""detfuse.py -- is there anything to gain over the equal-weight branch mean?

SCRIPT_VERSION 2026-08-29a-detfuse

THE QUESTION (user's, 2026-08-28): replace the three-branch equal-weight
average with a LEARNED combiner of the three probability maps. Whether
that is worth building depends on whether the branches have exploitable,
complementary strengths. det_ablate3 established the precondition:

  * an 11% detection-set churn between views that tie on F1;
  * every pairing containing B0 beats the pairing without it;
  * B01 (weights 1,1,0) beats FULL (1,1,1) by +0.0046 F1 -- the equal
    weight is already known not to be optimal.

THIS RUN measures, cheaply, how much a combiner could buy:

  1. WEIGHT SWEEP. Eight fixed weight vectors over the branch sigmoids,
     each scored at its own best gate -- the scalar-fusion headroom.
     FULL and B01 are must-reproduce anchors against det_ablate3.
  2. DISAGREEMENT vs SCORE. det_ablate3 left open whether the churn
     lives among low-scoring peaks. Every FULL peak's per-branch values
     (h0,h1,h2) are recorded with its matched/unmatched flag.
  3. POOR-MAN'S FUSION. A logistic regression on the per-branch logits,
     fit on even-index crops and scored on odd, against the mean score's
     AUC on the same rows. If the 3-vector cannot out-rank its own mean
     here, a voxel-wise learned fusion has nothing to work with.
  4. MISS HEADROOM. For GT cells with NO matched peak: does some single
     branch see them (window max >= gate) where the mean does not? That
     bounds the recall a fusion could recover.

Venue: Zebrahub dense GT only -- the one denominator that is the true
cell population (working-agreement: the denominator rule). This is
DETECTION evidence, not adj_J; any winner still has to be scored through
the pipeline in three domains before it is believed.

Matching is reported at BOTH 5.0 um (det_ablate3's radius, for the
anchors) and 7.0 um (the official scorer's constant, per the 2026-08-26
ruler correction). Rows/miss tables use 7.0.

Assembly: det_ablate3 verbatim for the model stack, all_logits, the
detect tail and the one-pass gate counting; the competition mount is
DROPPED (this script never reads it; det_ablate3 attached it out of an
inherited assert and paid the mount for nothing).

Attach: xiaoleilian/biohub-unet3d-weights, ...-v2models,
kernel output faustine649/biohub-zebrahub-extract. T4 pinned.
"""

import gc
import glob
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

SCRIPT_VERSION = "2026-08-29a-detfuse"
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

# ---- detect-side constants (xl-repro verbatim) ------------------------
SCALE = np.array([1.625, 0.40625, 0.40625])
POOL = 4
REPAIR = True
CAND_THR = 0.05
UNET_THRESH = 0.15
NMS_UM = 4.0
DETECT_THRESH = min(UNET_THRESH, CAND_THR) if REPAIR else UNET_THRESH

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

def _nms_keep(coords, scores, radius_um, scale=SCALE):
    """_physical_nms, returning the KEEP INDICES so a caller can carry
    per-peak side data (the per-branch values) through the suppression."""
    if len(coords) <= 1:
        return np.arange(len(coords))
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
    return np.array(keep)

_BRANCHES = [([MODELS[0]], [1.0], ""), ([MODELS[1]], [1.0], "tophat"),
             ([_load_model("unet3d_v2_tophat_b32.pt")], [1.0], "tophat")]
_FLIP_VIEWS = (0, 1, 2, 3)

def _flip(x, i):
    if i & 1:
        x = np.flip(x, -2)
    if i & 2:
        x = np.flip(x, -1)
    return x


def branch_maps(vol):
    """The three per-branch probability maps, each the sigmoid of its
    4-flip logit mean -- exactly detect()'s hs list, kept separate."""
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
    return hs


def detect_weighted(hs, vol, weights):
    """detect()'s tail on a weighted branch mean. Weights (1,1,1)
    reproduce detect() exactly (the mean divides by the weight sum).
    Returns refined coords, scores, and the per-branch values of every
    surviving peak (side data for the fusion diagnostics)."""
    w = np.asarray(weights, np.float64)
    h = sum(wi * hi for wi, hi in zip(w, hs)) / w.sum()
    pk = peak_local_max(h, min_distance=1, threshold_abs=DETECT_THRESH,
                        exclude_border=False)
    if len(pk) == 0:
        return np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3))
    sc = h[pk[:, 0], pk[:, 1], pk[:, 2]].astype(float)
    bv = np.stack([hb[pk[:, 0], pk[:, 1], pk[:, 2]] for hb in hs],
                  axis=1).astype(np.float32)
    coords = pk.astype(float)
    coords[:, 1] = coords[:, 1] * POOL + (POOL - 1) / 2
    coords[:, 2] = coords[:, 2] * POOL + (POOL - 1) / 2
    ref = np.array([_refine(vol, c)[0] for c in coords])
    keep = _nms_keep(ref, sc, NMS_UM)
    return ref[keep], sc[keep], bv[keep]


GATES = (0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60)
RADII = (5.0, 7.0)      # det_ablate3's anchor radius, and the scorer's


def match_flags(coords, sc, gt_um, radius):
    """Greedy 1:1 within `radius`, highest score first (node_bright v2's
    protocol). Returns the per-detection matched mask and the set of
    claimed GT indices; per-gate TP/ND are counted from the mask, which
    is exact because a detection's match cannot depend on lower scores."""
    matched = np.zeros(len(coords), bool)
    claimed = set()
    if len(coords) and len(gt_um):
        det_um = coords * SCALE[None, :]
        tree = cKDTree(gt_um)
        for i in np.argsort(-sc):
            near = tree.query_ball_point(det_um[i], radius)
            if not near:
                continue
            near.sort(key=lambda g: float(np.linalg.norm(gt_um[g] - det_um[i])))
            free = [g for g in near if g not in claimed]
            if free:
                claimed.add(free[0])
                matched[i] = True
    tp = np.zeros(len(GATES), np.int64)
    nd = np.zeros(len(GATES), np.int64)
    for k, g in enumerate(GATES):
        m = sc >= g
        nd[k] = int(m.sum())
        tp[k] = int((m & matched).sum())
    return matched, claimed, tp, nd


# (name, branch weights) -- all at 4 flips. FULL and B01 are anchors.
ARMS = [
    ("FULL", (1.0, 1.0, 1.0)),   # must reproduce det_ablate3 FULL @5um:
    ("B01",  (1.0, 1.0, 0.0)),   #   0.7738/0.7350 F1*0.7539 gate 0.15
    ("W211", (2.0, 1.0, 1.0)),   # B01: F1* 0.7585 gate 0.20
    ("W311", (3.0, 1.0, 1.0)),
    ("W121", (1.0, 2.0, 1.0)),
    ("W112", (1.0, 1.0, 2.0)),   # wrong-direction control
    ("W110h", (1.0, 1.0, 0.5)),
    ("W210h", (2.0, 1.0, 0.5)),
]

FRAME_STRIDE = 10
BUDGET_S = 3.0 * 3600
OUT = Path("/kaggle/working")

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

print(f"\n=== WEIGHT SWEEP over {len(ARMS)} arms, {len(files)} crops, "
      f"every {FRAME_STRIDE}th frame ===", flush=True)
print(f"  peaks at floor {DETECT_THRESH}, NMS {NMS_UM} um, gates {GATES}; "
      f"greedy 1:1 at both {RADII} um; Zebrahub dense GT", flush=True)

TP = {(nm, r): np.zeros(len(GATES), np.int64) for nm, _ in ARMS for r in RADII}
ND = {(nm, r): np.zeros(len(GATES), np.int64) for nm, _ in ARMS for r in RADII}
NG = 0
rows = []          # FULL-arm peaks: h0,h1,h2, score, matched@7, crop idx
miss_rows = []     # unmatched GT: per-branch window max, mean window max, crop
t0 = time.time()
done = 0
for ci, f in enumerate(files):
    if time.time() - t0 > BUDGET_S:
        print(f"\n  WARN wall-clock budget reached after {done} crops; "
              f"counts below cover those crops only", flush=True)
        break
    z = np.load(f)
    if "image" not in z.files or "nodes" not in z.files:
        print(f"  WARN {f.stem}: missing image or nodes -> skipped", flush=True)
        continue
    img = z["image"]
    nodes = np.asarray(z["nodes"], float)
    if img.ndim != 4:
        print(f"  WARN {f.stem}: image ndim {img.ndim} != 4 -> skipped",
              flush=True)
        continue
    T = img.shape[0]
    gbt = defaultdict(list)
    for r_ in range(len(nodes)):
        tt = int(nodes[r_, 0])
        if 0 <= tt < T:
            gbt[tt].append(r_)
    for t in range(0, T, FRAME_STRIDE):
        gi = gbt.get(t, [])
        if not gi:
            continue
        vol = np.asarray(img[t])
        gt_vox = nodes[gi, 1:]
        gt_um = gt_vox * SCALE[None, :]
        NG += len(gi)
        hs = branch_maps(vol)
        per_arm = {}
        for nm, wts in ARMS:
            coords, sc, bv = detect_weighted(hs, vol, wts)
            per_arm[nm] = (coords, sc, bv)
            for r in RADII:
                _, _, tp_, nd_ = match_flags(coords, sc, gt_um, r)
                TP[(nm, r)] += tp_
                ND[(nm, r)] += nd_
        # --- fusion diagnostics on the FULL arm, scorer radius 7 ------
        coords, sc, bv = per_arm["FULL"]
        matched, claimed, _, _ = match_flags(coords, sc, gt_um, 7.0)
        for i in range(len(sc)):
            rows.append((bv[i, 0], bv[i, 1], bv[i, 2], sc[i],
                         float(matched[i]), ci))
        h_mean = (hs[0] + hs[1] + hs[2]) / 3.0
        Zp, Yp, Xp = hs[0].shape
        for j, g in enumerate(gi):
            if j in claimed:
                continue
            zc = int(round(gt_vox[j, 0]))
            yc = int(round(gt_vox[j, 1] / POOL))
            xc = int(round(gt_vox[j, 2] / POOL))
            sl = (slice(max(0, zc - 2), min(Zp, zc + 3)),
                  slice(max(0, yc - 2), min(Yp, yc + 3)),
                  slice(max(0, xc - 2), min(Xp, xc + 3)))
            if sl[0].start >= sl[0].stop or sl[1].start >= sl[1].stop \
                    or sl[2].start >= sl[2].stop:
                continue
            miss_rows.append((float(hs[0][sl].max()), float(hs[1][sl].max()),
                              float(hs[2][sl].max()), float(h_mean[sl].max()),
                              ci))
        del hs, per_arm, vol
    done += 1
    print(f"  {done}/{len(files)} {f.stem}  ({time.time() - t0:.0f}s)",
          flush=True)
    del z, img
    gc.collect()

# ======================================================================
print(f"\n=== RESULTS: {NG} GT points over {done} crops ===", flush=True)
for r in RADII:
    tag = "ANCHOR radius (det_ablate3)" if r == 5.0 else "SCORER radius"
    print(f"\n--- match radius {r} um ({tag}) ---", flush=True)
    print(f"  {'arm':<7} {'gate*':>6} {'dets':>9} {'recall':>8} {'prec':>8} "
          f"{'F1*':>8} {'F1@.15':>8}", flush=True)
    i15 = GATES.index(0.15)
    for nm, _ in ARMS:
        rec = TP[(nm, r)] / max(1, NG)
        pre = TP[(nm, r)] / np.maximum(1, ND[(nm, r)])
        f1 = 2 * rec * pre / np.maximum(1e-9, rec + pre)
        k = int(np.argmax(f1))
        print(f"  {nm:<7} {GATES[k]:>6.2f} {ND[(nm, r)][k]:>9} "
              f"{rec[k]:>8.4f} {pre[k]:>8.4f} {f1[k]:>8.4f} "
              f"{f1[i15]:>8.4f}", flush=True)
print("\n  anchor check: FULL@5um must read recall 0.7738 prec 0.7350 "
      "F1* 0.7539 at gate 0.15 (det_ablate3, same crops, same stride). "
      "If it does not, nothing else in this log is comparable.", flush=True)

R = np.asarray(rows, np.float32)
np.savez_compressed(OUT / "fuse_rows.npz",
                    h=R[:, :3] if len(R) else np.zeros((0, 3), np.float32),
                    score=R[:, 3] if len(R) else np.zeros(0, np.float32),
                    matched=R[:, 4] if len(R) else np.zeros(0, np.float32),
                    crop=R[:, 5] if len(R) else np.zeros(0, np.float32))
M = np.asarray(miss_rows, np.float32)
np.savez_compressed(OUT / "fuse_miss.npz",
                    m=M if len(M) else np.zeros((0, 5), np.float32))
print(f"\nsaved fuse_rows.npz ({len(R)} peaks) and fuse_miss.npz "
      f"({len(M)} unmatched GT) for offline fusion training", flush=True)

# --- 2. disagreement vs score -----------------------------------------
if len(R):
    d = R[:, :3].max(1) - R[:, :3].min(1)
    sc = R[:, 3]
    hi = d >= 0.30
    print(f"\n=== disagreement (max-min over branches) vs score, "
          f"{len(R)} FULL peaks ===", flush=True)
    print(f"  disagreement quartiles: "
          f"{np.percentile(d, [25, 50, 75, 90]).round(3).tolist()}", flush=True)
    print(f"  peaks with disagreement >= 0.30: {hi.sum()} "
          f"({100.0 * hi.mean():.1f}%)", flush=True)
    for name, mask in (("all peaks", np.ones(len(R), bool)),
                       ("disagreement >= 0.30", hi)):
        if mask.sum():
            q = np.percentile(sc[mask], [25, 50, 75]).round(3).tolist()
            in_band = ((sc[mask] >= 0.05) & (sc[mask] < 0.15)).mean()
            print(f"  {name:<22} n={int(mask.sum()):>7} score q25/50/75 {q} "
                  f" sub-threshold share {100.0 * in_band:.1f}%", flush=True)

# --- 3. poor-man's fusion: logistic vs mean score ---------------------
def _auc(y, s):
    order = np.argsort(s)
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    # midranks for ties
    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    pos = y > 0.5
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))

if len(R):
    print("\n=== poor-man's fusion: logistic on branch logits, "
          "fit even crops / score odd crops ===", flush=True)
    crop_idx = R[:, 5].astype(int)
    tr = (crop_idx % 2 == 0)
    te = ~tr
    y = R[:, 4]
    lg = np.log(np.clip(R[:, :3], 1e-4, 1 - 1e-4)
                / (1 - np.clip(R[:, :3], 1e-4, 1 - 1e-4)))
    if tr.sum() and te.sum() and len(set(y[te])) > 1:
        try:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=1000)
            clf.fit(lg[tr], y[tr])
            p = clf.decision_function(lg[te])
            auc_fuse = _auc(y[te], p)
            auc_mean = _auc(y[te], R[te, 3])
            print(f"  test rows {int(te.sum())} (pos {int(y[te].sum())}), "
                  f"train rows {int(tr.sum())}", flush=True)
            print(f"  AUC mean-score  {auc_mean:.4f}", flush=True)
            print(f"  AUC logistic    {auc_fuse:.4f}   delta "
                  f"{auc_fuse - auc_mean:+.4f}", flush=True)
            print(f"  learned weights on (l0,l1,l2): "
                  f"{clf.coef_[0].round(3).tolist()}  bias "
                  f"{float(clf.intercept_[0]):.3f}", flush=True)
            print("  reading: if the delta is ~0, a learned voxel fusion "
                  "has no per-peak ranking signal beyond the mean, and the "
                  "line stops here. A positive delta prices the ceiling, "
                  "not the pipeline gain.", flush=True)
        except Exception as e:                                 # noqa: BLE001
            print(f"  WARN logistic skipped: {str(e)[:120]} -- the saved "
                  f"fuse_rows.npz answers this offline", flush=True)
    else:
        print("  WARN degenerate split -- see fuse_rows.npz", flush=True)

# --- 4. miss headroom -------------------------------------------------
if len(M):
    print(f"\n=== miss headroom: {len(M)} GT cells unmatched by FULL@7um "
          f"(floor {DETECT_THRESH}) ===", flush=True)
    bmax = M[:, :3].max(1)
    for g in (0.15, 0.05):
        nb = int((bmax >= g).sum())
        nm_ = int((M[:, 3] >= g).sum())
        print(f"  window-max >= {g:.2f}: some branch {nb} "
              f"({100.0 * nb / len(M):.1f}%)  vs  branch mean {nm_} "
              f"({100.0 * nm_ / len(M):.1f}%)", flush=True)
    print("  reading: the gap between those columns is the recall a "
          "per-voxel fusion could recover that the mean suppresses. "
      "(window = +-2 pooled voxels ~ 3.25 um)", flush=True)

print(f"\nDETFUSE DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
