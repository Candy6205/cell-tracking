"""hrefine: a placement head over the shipped F4HP detections -- a small CNN on the raw full-resolution
crop around each detection, trained to predict the offset to the annotated centroid, out of fold by movie.

WHY (2026-09-23). The record's detector diagnosis ends at placement, not finding: lntrace / lnbench showed
recovered cells at 5.7 um from the annotation cannot be linked while cells within 2.8 um can; hplace put the
four heatmaps and their mean at 2.7-3.0 um from the annotated centroid at the 1,643 swap sites (1.3-1.6 at
ordinary cells) with the maps agreeing within 0.14 um, so choosing among maps is not a placement lever; every
raw-image rule tried (mean shift, intensity centroid, dip) moved ordinary detections off their annotation as
much as it helped the swaps. The forum (hengck23, topic 740145) reports the same offset between a detector's
centre and the Kaggle point and a learned per-peak refine head that took his error from 1.91 to 1.57 um. This
kernel trains that head on OUR detections and OUR labels: for every F4HP primary (score >= 0.15) within 7 um
of an annotated node (greedy 1:1 per frame), target = annotated centroid - detection, in um; input = the raw
frame's 7 planes x 41 x 41 voxels (11.4 x 16.7 x 16.7 um) around the detection's voxel, normalised per crop,
plus the sub-voxel residual and the score. v2 (2026-09-23b): BOTH embryos (128 6bba + 71 44b6 caches), five folds
by movie with both embryos in every fold (the JOINT head, out of fold everywhere), plus two CROSS-EMBRYO heads (6bba
only -> applied to 44b6, 44b6 only -> applied to 6bba): v1 trained on 6bba alone and helped 122 of 128 6bba
movies (1.27 -> 1.08 um) while making 7 of the 12 44b6 movies worse -- the hidden test is a different embryo, so
the cross-embryo line is the one that decides.

Pass 1: crops + labels for the annotated primaries and a per-movie sample of unannotated ones; training; the
out-of-fold report (distance to the annotation before / after, by crowding, score, fold; 44b6 apart; the shift
magnitude on unannotated detections; the intensity-centroid placement as a second baseline).
Pass 2 (WRITE_CACHES): every primary of every frame re-placed by its fold model (4-rotation average), written as
fdet_F4hr_<name>.npz in the detcache layout (c{t} / s{t}; sub-threshold peaks untouched, order kept), with the
count of primary pairs within 2 um before / after (duplicates a shift can create).
Attach: the competition data, faustine649/cell-tracking (viz_cells.py), kernel output biohub-hparacache
(fdet_F4hp_). T4.
"""
import os
import pathlib
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

SCRIPT_VERSION = "2026-09-23c-hrefine"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

INPUT_ROOT = "/kaggle/input"
OUT = pathlib.Path("/kaggle/working")
CODE_ROOTS = [
    f"{INPUT_ROOT}/datasets/faustine649/cell-tracking/cell tracking",
    f"{INPUT_ROOT}/datasets/faustine649/cell-tracking",
    f"{INPUT_ROOT}/cell-tracking/cell tracking",
    f"{INPUT_ROOT}/cell-tracking",
]
code = next((c for c in CODE_ROOTS if pathlib.Path(c, "viz_cells.py").exists()), None)
assert code is not None, "attach faustine649/cell-tracking"
sys.path.insert(0, code)
from viz_cells import find_root, geff_path, open_image, read_geff, zarr_path   # noqa: E402

ROOT = find_root(None)
SCALE = np.array([1.625, 0.40625, 0.40625])
UNET_THRESH = 0.15
R_MATCH = 7.0                      # the scorer's match radius
R_DUP = 2.0                        # "two detections on one point" after a shift
NZ, HALF = 3, 20                   # planes each side; crop half-width in voxels
NCH = 2 * NZ + 1
W = 2 * HALF + 1                   # odd width: the centre voxel maps to itself under rot90 / flip
N_FOLDS, EPOCHS, BATCH, LR = 5, 12, 256, 1e-3
N_UNANN_PER_MOVIE, SEED = 300, 0
WRITE_CACHES = False               # v3: training only
CENTER_TARGETS = True              # v3: train on targets centred per movie (the movie's mean offset removed)
BUDGET_S = 10.5 * 3600
U8_SCALE, U8_OFF = 51.0, 1.0       # crop value v in [-1, 4] -> uint8 (v + 1) * 51


def _walk_files(match):
    hits = {}
    for root_, dirs, fnames in os.walk(INPUT_ROOT):
        dirs[:] = [d for d in dirs if "cell-tracking-during-development" not in d
                   and not d.endswith(".zarr") and not d.endswith(".geff")]
        for fn in fnames:
            key = match(fn)
            if key is not None:
                hits.setdefault(key, pathlib.Path(root_) / fn)
    return hits


def _npz_prefix(prefix):
    return lambda fn: fn[len(prefix):-len(".npz")] if fn.startswith(prefix) and fn.endswith(".npz") else None


FDET_HP = _walk_files(_npz_prefix("fdet_F4hp_"))
NAMES6 = sorted(n for n in FDET_HP if n.startswith("6bba_"))
NAMES44 = sorted(n for n in FDET_HP if n.startswith("44b6_"))
print(f"caches: F4hp {len(FDET_HP)} ({len(NAMES6)} 6bba, {len(NAMES44)} 44b6)", flush=True)
assert len(NAMES6) == 128, "attach biohub-hparacache (128 6bba fdet_F4hp_ caches)"
assert len(NAMES44) == 71, "attach biohub-hparacache + biohub-hpara44 (12 + 59 44b6 fdet_F4hp_ caches)"
FOLD_OF = {m: k % N_FOLDS for k, m in enumerate(NAMES6)}
FOLD_OF.update({m: k % N_FOLDS for k, m in enumerate(NAMES44)})     # both embryos in every fold


# ---------------------------------------------------------------- pure functions (toy-tested)
def crops_of(vol, C):
    """vol (Z, Y, X); C (N, 3) full-res float coords. Returns crops (N, NCH, 2H, 2H) float32 in [-1, 4]
    (per-crop p20 -> 0, p99.5 -> 1) and the residual (N, 3) = (C - crop centre voxel) in um."""
    Z, Y, X = vol.shape
    ctr = np.rint(np.asarray(C, float)).astype(int)
    ctr[:, 0] = np.clip(ctr[:, 0], 0, Z - 1)
    ctr[:, 1] = np.clip(ctr[:, 1], 0, Y - 1)
    ctr[:, 2] = np.clip(ctr[:, 2], 0, X - 1)
    pad = np.pad(np.asarray(vol), ((NZ, NZ), (0, 0), (0, 0)), mode="edge")
    pad = np.pad(pad, ((0, 0), (HALF, HALF), (HALF, HALF)), mode="constant")
    zz = ctr[:, 0, None] + np.arange(NCH)[None, :]
    yy = ctr[:, 1, None] + np.arange(W)[None, :]
    xx = ctr[:, 2, None] + np.arange(W)[None, :]
    crop = pad[zz[:, :, None, None], yy[:, None, :, None], xx[:, None, None, :]].astype(np.float32)
    flat = crop.reshape(len(ctr), -1)
    lo = np.percentile(flat, 20.0, axis=1)
    hi = np.percentile(flat, 99.5, axis=1)
    crop = np.clip((crop - lo[:, None, None, None]) / np.maximum(hi - lo, 1.0)[:, None, None, None], -1.0, 4.0)
    res = ((np.asarray(C, float) - ctr) * SCALE).astype(np.float32)
    return crop, res


def to_u8(crop):
    return np.clip(np.rint((crop + U8_OFF) * U8_SCALE), 0, 255).astype(np.uint8)


def from_u8(u8):
    return u8.astype(np.float32) / U8_SCALE - U8_OFF


def match_gt(C, G, R=R_MATCH):
    """Greedy 1:1 matching, nearest first, within R um. C, G: (N, 3) / (M, 3) full-res voxel coords.
    Returns (idx_prim, idx_gt, d_um, crowd) where crowd = number of primaries within R of the matched GT node."""
    C, G = np.asarray(C, float).reshape(-1, 3), np.asarray(G, float).reshape(-1, 3)
    e = (np.zeros(0, int), np.zeros(0, int), np.zeros(0), np.zeros(0, int))
    if len(C) == 0 or len(G) == 0:
        return e
    tree = cKDTree(C * SCALE)
    balls = tree.query_ball_point(G * SCALE, r=R)
    cand = []
    for gi, lst in enumerate(balls):
        for pi in lst:
            cand.append((float(np.linalg.norm((C[pi] - G[gi]) * SCALE)), gi, pi))
    if not cand:
        return e
    cand.sort()
    used_p, used_g, out = set(), set(), []
    for d, gi, pi in cand:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        out.append((pi, gi, d, len(balls[gi])))
    if not out:
        return e
    return (np.array([o[0] for o in out], int), np.array([o[1] for o in out], int),
            np.array([o[2] for o in out], float), np.array([o[3] for o in out], int))


def rot_vec(v, k, flip):
    """Offsets (dz, dy, dx) after np.rot90(crop, k, axes=(y, x)) then a flip along x. (y, x) -> (n-1-x, y) per
    rotation, so (dy, dx) -> (-dx, dy); the flip sends dx -> -dx."""
    v = np.array(v, float, copy=True).reshape(-1, 3)
    for _ in range(k % 4):
        v[:, 1], v[:, 2] = -v[:, 2].copy(), v[:, 1].copy()
    if flip:
        v[:, 2] = -v[:, 2]
    return v


def inv_vec(v, k, flip):
    v = np.array(v, float, copy=True).reshape(-1, 3)
    if flip:
        v[:, 2] = -v[:, 2]
    for _ in range(k % 4):
        v[:, 1], v[:, 2] = v[:, 2].copy(), -v[:, 1].copy()
    return v


def augment(x, res, off, k, flip):
    """x (N, NCH, H, W); res / off (N, 3) um. Rotate the crops k times in (y, x) and flip along x, and
    transform both vectors the same way."""
    x = np.rot90(x, k, axes=(2, 3))
    if flip:
        x = x[:, :, :, ::-1]
    return np.ascontiguousarray(x), rot_vec(res, k, flip).astype(np.float32), rot_vec(off, k, flip).astype(np.float32)


def refine_centroid(vol, zyx, rz=2, ryx=5):
    """hparacache._refine verbatim: the intensity centroid in a (2rz+1) x (2ryx+1)^2 box, min subtracted."""
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


def dup_pairs(C, R=R_DUP):
    C = np.asarray(C, float).reshape(-1, 3)
    if len(C) < 2:
        return 0
    return len(cKDTree(C * SCALE).query_pairs(r=R))


def primaries(z, t):
    c = np.asarray(z[f"c{t}"], np.float32).reshape(-1, 3)
    s = np.asarray(z[f"s{t}"], np.float32).reshape(-1)
    m = s >= UNET_THRESH
    return c, s, m


if __name__ == "__main__":
    t_all = time.time()
    rng = np.random.default_rng(SEED)

    # ------------------------------------------------------------ pass 1: labels + crops
    ROWS, X_U8, RES, OFF = [], [], [], []
    n_prim_total = 0
    for k_, name in enumerate(NAMES6 + NAMES44):
        t0 = time.time()
        img = open_image(zarr_path(ROOT, "train", name))
        gt = read_geff(geff_path(ROOT, "train", name))
        gt_t = np.asarray(gt["t"]).astype(int)
        gzyx = np.stack([np.asarray(gt["z"], float), np.asarray(gt["y"], float), np.asarray(gt["x"], float)], 1)
        z = np.load(FDET_HP[name])
        T = int(img.shape[0])
        fold = FOLD_OF.get(name, -1)
        n_ann = n_pr = 0
        un_left = N_UNANN_PER_MOVIE
        for t in range(T):
            C_all, S_all, m = primaries(z, t)
            C, S = C_all[m], S_all[m]
            n_pr += len(C)
            G = gzyx[gt_t == t]
            pi, gi, d, crowd = match_gt(C, G)
            un_pool = np.setdiff1d(np.arange(len(C)), pi)
            n_un = min(len(un_pool), max(0, int(np.ceil(un_left / max(T - t, 1)))))
            un = rng.choice(un_pool, size=n_un, replace=False) if n_un else np.zeros(0, int)
            un_left -= n_un
            sel = np.concatenate([pi, un]).astype(int)
            if len(sel) == 0:
                continue
            vol = np.asarray(img[t])
            X, res = crops_of(vol, C[sel])
            off = np.full((len(sel), 3), np.nan, np.float32)
            off[:len(pi)] = (G[gi] - C[pi]) * SCALE
            X_U8.append(to_u8(X))
            RES.append(res)
            OFF.append(off)
            for q in range(len(sel)):
                ann = q < len(pi)
                r = dict(dataset=name, t=t, i=int(sel[q]), fold=fold, s=float(S[sel[q]]), ann=int(ann),
                         z=float(C[sel[q], 0]), y=float(C[sel[q], 1]), x=float(C[sel[q], 2]))
                if ann:
                    r.update(gt_i=int(gi[q]), crowd=int(crowd[q]), d_before=float(d[q]),
                             d_refine=float(np.linalg.norm((refine_centroid(vol, C[sel[q]]) - G[gi[q]]) * SCALE)))
                else:
                    r.update(gt_i=-1, crowd=0, d_before=np.nan, d_refine=np.nan)
                ROWS.append(r)
            n_ann += len(pi)
        n_prim_total += n_pr
        del img
        print(f"  [{k_ + 1}/{len(NAMES6) + len(NAMES44)}] {name}: fold {fold}, primaries {n_pr:,}, annotated matched "
              f"{n_ann:,}, unannotated sampled {N_UNANN_PER_MOVIE - un_left}, {time.time() - t0:.0f}s", flush=True)
        if time.time() - t_all > BUDGET_S * 0.4:
            print("  pass-1 budget reached; stopping the crop stage", flush=True)
            break

    P = pd.DataFrame(ROWS)
    X_U8 = np.concatenate(X_U8)
    RES = np.concatenate(RES)
    OFF = np.concatenate(OFF)
    ann = P.ann.values == 1
    is6 = P.dataset.str.startswith("6bba_").values
    # v3: the training target is the offset with the movie's mean offset (over its annotated matched primaries) removed,
    # so the head cannot spend itself on the per-acquisition constant; at deployment the prediction is applied as is
    # (no constant added back), and the report shows both that and the residual-only view.
    _dfo = pd.DataFrame(OFF, columns=["oz", "oy", "ox"])
    _dfo["dataset"] = P.dataset.values
    MOVIE_MEAN = _dfo[ann].groupby("dataset")[["oz", "oy", "ox"]].mean().reindex(P.dataset.values).values.astype(np.float32)
    OFF_T = (OFF - MOVIE_MEAN) if CENTER_TARGETS else OFF
    print(f"  CENTER_TARGETS {CENTER_TARGETS}: per-movie mean offset removed from the targets; |movie mean| p50 "
          f"{np.nanmedian(np.linalg.norm(MOVIE_MEAN[ann], axis=1)):.3f} um", flush=True)
    print(f"\npass 1: {len(P):,} rows ({int(ann.sum()):,} annotated, {int((~ann).sum()):,} unannotated) over "
          f"{P.dataset.nunique()} movies; primaries seen {n_prim_total:,}; crops {X_U8.shape} {X_U8.nbytes / 1e9:.2f} GB; "
          f"{(time.time() - t_all) / 60:.1f} min", flush=True)
    d0 = P.d_before.values
    print(f"  annotated 6bba: d_before p50 {np.nanmedian(d0[ann & is6]):.3f} p90 {np.nanpercentile(d0[ann & is6], 90):.3f} um; "
          f"crowd >= 2 share {np.mean(P.crowd.values[ann & is6] >= 2):.3f}", flush=True)
    P.to_csv(OUT / "hrefine_rows.csv", index=False)

    # ------------------------------------------------------------ training, out of fold by movie
    PRED = np.full((len(P), 3), np.nan, np.float32)        # the joint head, out of fold
    PRED_X = np.full((len(P), 3), np.nan, np.float32)      # the cross-embryo head (trained on the other embryo only)
    MODELS = {}
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("torch", torch.__version__, "device", dev, flush=True)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()

                def blk(a, b):
                    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                                         nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))
                self.b1, self.b2, self.b3 = blk(NCH, 32), blk(32, 64), blk(64, 128)
                self.b4 = blk(128, 128)
                self.head = nn.Sequential(nn.Linear(128 + 4, 64), nn.ReLU(inplace=True), nn.Linear(64, 3))

            def forward(self, x, g):
                x = F.max_pool2d(self.b1(x), 2)
                x = F.max_pool2d(self.b2(x), 2)
                x = F.max_pool2d(self.b3(x), 2)
                x = self.b4(x).mean((2, 3))
                return self.head(torch.cat([x, g], 1))

        def scalars(res, s):
            return np.concatenate([res, s[:, None]], 1).astype(np.float32)

        def predict(model, idx):
            """4-rotation average with the inverse transform on each prediction."""
            model.eval()
            out = np.zeros((len(idx), 3), np.float32)
            s_ = P.s.values.astype(np.float32)
            with torch.no_grad():
                for a in range(0, len(idx), 512):
                    ii = idx[a:a + 512]
                    x0, r0 = from_u8(X_U8[ii]), RES[ii]
                    acc = np.zeros((len(ii), 3), np.float32)
                    for k in range(4):
                        xk, rk, _ = augment(x0, r0, r0, k, False)
                        pk = model(torch.from_numpy(xk).to(dev), torch.from_numpy(scalars(rk, s_[ii])).to(dev)).cpu().numpy()
                        acc += inv_vec(pk, k, False)
                    out[a:a + len(ii)] = acc / 4
            return out

        def train_on(tr, seed, tag):
            t0 = time.time()
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = Net().to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
            steps = EPOCHS * int(np.ceil(len(tr) / BATCH))
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps)
            s_ = P.s.values.astype(np.float32)
            for ep in range(EPOCHS):
                model.train()
                perm = np.random.permutation(tr)
                tot = 0.0
                for a in range(0, len(perm), BATCH):
                    ii = perm[a:a + BATCH]
                    kr, fl = int(np.random.randint(4)), bool(np.random.rand() < 0.5)
                    xb, rb, ob = augment(from_u8(X_U8[ii]), RES[ii], OFF_T[ii], kr, fl)
                    pred = model(torch.from_numpy(xb).to(dev), torch.from_numpy(scalars(rb, s_[ii])).to(dev))
                    loss = F.smooth_l1_loss(pred, torch.from_numpy(ob).to(dev), beta=1.0)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    sched.step()
                    tot += float(loss.detach()) * len(ii)
                if ep % 4 == 3 or ep == EPOCHS - 1:
                    print(f"    {tag} epoch {ep + 1}: loss {tot / len(tr):.4f}", flush=True)
            print(f"  {tag}: train {len(tr):,} ({int(is6[tr].sum()):,} 6bba, {int((~is6[tr]).sum()):,} 44b6), "
                  f"{time.time() - t0:.0f}s", flush=True)
            return model

        # the joint head: five folds by movie, both embryos in every fold, out of fold everywhere
        for k in range(N_FOLDS):
            tr = np.where(ann & (P.fold.values != k))[0]
            te = np.where(P.fold.values == k)[0]
            model = train_on(tr, SEED + k, f"fold {k}")
            PRED[te] = predict(model, te)
            MODELS[k] = model
            torch.save(model.state_dict(), OUT / f"hrefine_fold{k}.pt")
        # the cross-embryo arms: a head that never saw the embryo it is applied to (the hidden test's condition)
        m6 = train_on(np.where(ann & is6)[0], SEED + 10, "X6 (6bba only)")
        PRED_X[np.where(~is6)[0]] = predict(m6, np.where(~is6)[0])
        torch.save(m6.state_dict(), OUT / "hrefine_x6.pt")
        m44 = train_on(np.where(ann & ~is6)[0], SEED + 11, "X44 (44b6 only)")
        PRED_X[np.where(is6)[0]] = predict(m44, np.where(is6)[0])
        torch.save(m44.state_dict(), OUT / "hrefine_x44.pt")
        del m6, m44
    except Exception:                                                   # noqa: BLE001
        traceback.print_exc()
        print("training stage failed", flush=True)

    P[["oz", "oy", "ox"]] = OFF                       # the target offsets, so a damped shift can be priced offline
    P[["pz", "py", "px"]] = PRED
    P[["xz", "xy", "xx"]] = PRED_X
    P["d_after"] = np.where(ann, np.linalg.norm(OFF - PRED, axis=1), np.nan)          # deployment view: prediction applied as is
    P["d_after_x"] = np.where(ann, np.linalg.norm(OFF - PRED_X, axis=1), np.nan)
    P["d_res0"] = np.where(ann, np.linalg.norm(OFF_T, axis=1), np.nan)                # residual-only view: before
    P["d_res"] = np.where(ann, np.linalg.norm(OFF_T - PRED, axis=1), np.nan)          # ... after the joint head
    P["d_res_x"] = np.where(ann, np.linalg.norm(OFF_T - PRED_X, axis=1), np.nan)      # ... after the cross head
    P["shift"] = np.linalg.norm(PRED, axis=1)
    P["shift_x"] = np.linalg.norm(PRED_X, axis=1)
    P.to_csv(OUT / "hrefine_rows.csv", index=False)

    # ------------------------------------------------------------ report
    def q(v, p):
        v = v[np.isfinite(v)]
        return float(np.percentile(v, p)) if len(v) else float("nan")

    def line(lbl, m, col="d_after"):
        a_all = P[col].values
        m = m & np.isfinite(a_all)
        n = int(m.sum())
        if n == 0:
            print(f"  {lbl:<30} n 0", flush=True)
            return
        b, r, a = P.d_before.values[m], P.d_refine.values[m], a_all[m]
        print(f"  {lbl:<30} n {n:>7,}  before {q(b, 50):.3f} / {q(b, 90):.3f}  centroid {q(r, 50):.3f} / {q(r, 90):.3f}  "
              f"after {q(a, 50):.3f} / {q(a, 90):.3f}   <=2um {np.mean(b <= 2):.3f} -> {np.mean(a <= 2):.3f}   "
              f"<=3um {np.mean(b <= 3):.3f} -> {np.mean(a <= 3):.3f}   worse {np.mean(a > b):.3f}", flush=True)

    sh, shx = P["shift"].values, P["shift_x"].values          # not P.shift: DataFrame.shift is a method
    for emb, me in (("6bba", is6), ("44b6", ~is6)):
        ok = np.isfinite(P.d_after.values)
        print(f"\n=== HREFINE out of fold, {emb}: distance to the annotated centroid, um, p50 / p90 (annotated primaries, "
              f"{int((ann & me & ok).sum()):,}); the JOINT head (both embryos in training, other movies held out) ===", flush=True)
        line(f"all {emb}", ann & me)
        line("isolated (crowd 1)", ann & me & (P.crowd.values == 1))
        line("crowded (crowd >= 2)", ann & me & (P.crowd.values >= 2))
        line("crowded (crowd >= 3)", ann & me & (P.crowd.values >= 3))
        for lo, hi in ((0.15, 0.30), (0.30, 0.50), (0.50, 9.0)):
            line(f"score [{lo}, {hi})", ann & me & (P.s.values >= lo) & (P.s.values < hi))
        for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 7.01)):
            line(f"d_before [{lo}, {hi})", ann & me & (P.d_before.values >= lo) & (P.d_before.values < hi))
        for k in range(N_FOLDS):
            line(f"fold {k}", ann & me & (P.fold.values == k))
        other = "44b6" if emb == "6bba" else "6bba"
        print(f"  --- the CROSS-EMBRYO head ({other}-only training, never saw {emb}: the hidden test's condition) ---", flush=True)
        line(f"all {emb} by the {other} head", ann & me, "d_after_x")
        line("isolated (crowd 1)", ann & me & (P.crowd.values == 1), "d_after_x")
        line("crowded (crowd >= 2)", ann & me & (P.crowd.values >= 2), "d_after_x")
        for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 7.01)):
            line(f"d_before [{lo}, {hi})", ann & me & (P.d_before.values >= lo) & (P.d_before.values < hi), "d_after_x")
        print(f"  --- RESIDUAL view ({emb}): distance to the target with the movie's mean offset removed, p50 / p90: before "
              f"{q(P.d_res0.values[ann & me], 50):.3f} / {q(P.d_res0.values[ann & me], 90):.3f}, joint "
              f"{q(P.d_res.values[ann & me], 50):.3f} / {q(P.d_res.values[ann & me], 90):.3f} (worse "
              f"{np.mean(P.d_res.values[ann & me] > P.d_res0.values[ann & me]):.3f}), cross "
              f"{q(P.d_res_x.values[ann & me], 50):.3f} / {q(P.d_res_x.values[ann & me], 90):.3f} (worse "
              f"{np.mean(P.d_res_x.values[ann & me] > P.d_res0.values[ann & me]):.3f}); mean predicted shift joint "
              f"{PRED[ann & me].mean(0).round(3)}, cross {PRED_X[ann & me].mean(0).round(3)}", flush=True)
        print(f"  shift |pred| p50 / p90, joint head: annotated {q(sh[ann & me], 50):.3f} / {q(sh[ann & me], 90):.3f}, "
              f"unannotated {q(sh[~ann & me], 50):.3f} / {q(sh[~ann & me], 90):.3f}; cross head: annotated "
              f"{q(shx[ann & me], 50):.3f} / {q(shx[ann & me], 90):.3f}, unannotated {q(shx[~ann & me], 50):.3f} / "
              f"{q(shx[~ann & me], 90):.3f}", flush=True)
        # per-movie: how many movies gain under each head
        for col, nm in (("d_after", "joint"), ("d_after_x", "cross")):
            g = P[ann & me & np.isfinite(P[col].values)].groupby("dataset")
            gain = g.apply(lambda d: d.d_before.median() - d[col].median())
            print(f"  per movie ({nm} head): {int((gain > 0).sum())} of {len(gain)} movies gain; gain p50 {gain.median():.3f} "
                  f"p10 {gain.quantile(0.1):.3f} p90 {gain.quantile(0.9):.3f} um", flush=True)

    # ------------------------------------------------------------ pass 2: re-placed caches
    if WRITE_CACHES and MODELS:
        print("\n=== pass 2: writing fdet_F4hr_<name>.npz (every primary re-placed by its fold model) ===", flush=True)
        DUP = dict(before=0, after=0, frames=0)
        for k_, name in enumerate(NAMES6 + NAMES44):
            if time.time() - t_all > BUDGET_S:
                print("  budget reached; caches INCOMPLETE from here", flush=True)
                break
            t0 = time.time()
            models = [MODELS[FOLD_OF[name]]]                # the joint head that never saw this movie
            img = open_image(zarr_path(ROOT, "train", name))
            z = np.load(FDET_HP[name])
            T = int(img.shape[0])
            out = {}
            shifts = []
            for t in range(T):
                C_all, S_all, m = primaries(z, t)
                C_new = C_all.copy()
                if m.any():
                    vol = np.asarray(img[t])
                    X, res = crops_of(vol, C_all[m])
                    sc = S_all[m].astype(np.float32)
                    acc = np.zeros((int(m.sum()), 3), np.float32)
                    with torch.no_grad():
                        for model in models:
                            model.eval()
                            for k in range(4):
                                xk, rk, _ = augment(X, res, res, k, False)
                                for a in range(0, len(xk), 1024):
                                    pk = model(torch.from_numpy(xk[a:a + 1024]).to(dev),
                                               torch.from_numpy(scalars(rk[a:a + 1024], sc[a:a + 1024])).to(dev)).cpu().numpy()
                                    acc[a:a + 1024] += inv_vec(pk, k, False)
                    acc /= 4 * len(models)
                    moved = C_all[m] + acc / SCALE
                    Z, Y, X_ = vol.shape
                    moved[:, 0] = np.clip(moved[:, 0], 0, Z - 1)
                    moved[:, 1] = np.clip(moved[:, 1], 0, Y - 1)
                    moved[:, 2] = np.clip(moved[:, 2], 0, X_ - 1)
                    DUP["before"] += dup_pairs(C_all[m])
                    DUP["after"] += dup_pairs(moved)
                    DUP["frames"] += 1
                    shifts.append(np.linalg.norm(acc, axis=1))
                    C_new[m] = moved.astype(np.float32)
                out[f"c{t}"] = C_new
                out[f"s{t}"] = S_all
            np.savez_compressed(OUT / f"fdet_F4hr_{name}.npz", **out)
            sh = np.concatenate(shifts) if shifts else np.zeros(1)
            print(f"  [{k_ + 1}/{len(NAMES6) + len(NAMES44)}] {name}: T {T}, shift p50 {np.median(sh):.3f} p90 "
                  f"{np.percentile(sh, 90):.3f} max {sh.max():.3f} um, {time.time() - t0:.0f}s", flush=True)
            del img
        print(f"  primary pairs within {R_DUP} um over {DUP['frames']:,} frames: before {DUP['before']:,} -> after "
              f"{DUP['after']:,}", flush=True)
    print(f"\nHREFINE DONE in {(time.time() - t_all) / 60:.1f} min", flush=True)
    print("HREFINE OK" if MODELS else "HREFINE TRAINING FAILED", flush=True)
