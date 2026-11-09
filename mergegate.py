"""mergegate: a learned gate over hypcache parent pairs -- raw-image crops through time plus REF's graph
features -- trained to keep the pairs a merge fixes and reject the pairs a merge breaks.

WHY (2026-09-23). mergejudge typed REF's hypcache parent pairs (saddle-linked detection pairs within 8.5 um)
on the final graph: a merge FIXES ~811 pairs (one-cell pairs whose annotated end is alive on a wrong track
177, one-end pairs whose mate carries the annotated end's GT neighbour 363, parallel duplicate tracks 271;
oracle ceiling about +0.009 edge_J) and BREAKS the one-end pairs whose mate carries a persistent track of
its own without carrying A's neighbour (case 2, 4,626). The "both full tracks" veto keeps most fixes and a
fifth of case 2 (~950 survivors against ~300 fixes). swapvid's eye tags say the raw frames carry what the
pair features do not (shading, mottling, one big smooth nucleus vs two). This kernel cuts, for every typed
pair and a sample of unannotated ones, a 26 um x 26 um crop over the annotated plane +-2 and frames t-2..t+2
(25 channels) plus two position masks, trains a small CNN (with the graph features) FIX vs CASE2 out of fold
by movie, and reports the firing rate on every class at fixed FIX recall, with and without the veto.
Outputs: mergegate_pairs.csv (typing, features, out-of-fold probabilities), mergegate_fold{k}.pt,
mergegate_norm.npz. Attach: the competition data, faustine649/cell-tracking (viz_cells.py), kernel outputs
biohub-hparacache (fdet_F4hp_), biohub-hypcache (fdet_F4hyp_), biohub-mergejudge (mergejudge_pairs.csv). T4.
"""
import json
import os
import pathlib
import sys
import time
import traceback

import numpy as np
import pandas as pd

SCRIPT_VERSION = "2026-09-23a-mergegate"
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
from viz_cells import find_root, open_image, zarr_path   # noqa: E402

ROOT = find_root(None)
SCALE = np.array([1.625, 0.40625, 0.40625])
UNET_THRESH = 0.15
N_UNANN, SEED = 5000, 0
HALF, NZ, NT = 32, 2, 2                 # crop half-width (voxels), planes each side, frames each side
CH_IMG = (2 * NT + 1) * (2 * NZ + 1)   # 25
N_FOLDS, EPOCHS, BATCH, LR = 5, 25, 64, 1e-3
BUDGET_CROP_S = 5.0 * 3600


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
FDET_HYP = _walk_files(_npz_prefix("fdet_F4hyp_"))
MJ = _walk_files(lambda fn: fn if fn == "mergejudge_pairs.csv" else None)
print(f"caches: F4hp {len(FDET_HP)}, F4hyp {len(FDET_HYP)}, mergejudge_pairs.csv {'found' if MJ else 'MISSING'}", flush=True)
assert MJ, "attach biohub-mergejudge (mergejudge_pairs.csv)"


# ---------------------------------------------------------------- typing (mergejudge's rules)
def type_pairs(d):
    has_in = d.n_gt_in > 0
    has_out = d.n_gt_kids > 0
    a_right = ((~has_in) | (d.A_in_tp == 1)) & (d.A_out_tp == d.n_gt_kids)
    a_miss = (has_in & (d.A_in_tp == 0)) | (has_out & (d.A_out_tp < d.n_gt_kids))
    b_car = (d.B_carries_in == 1) | (d.B_carries_out == 1)
    b_track = (d.B_back >= 2) & (d.B_fwd >= 2)
    b_loose = (d.B_in_n + d.B_out_n) <= 1
    cls = np.full(len(d), "unann", object)
    one, oe, two = d.lab == 1, d.lab == 2, d.lab == 0
    fix1 = one & (d.A_alive == 1) & a_miss & b_car
    fixdup = one & a_right & b_car
    cls[one] = "one_other"
    cls[fixdup] = "FIXDUP"
    cls[fix1] = "FIX1"
    fix2 = oe & a_miss & b_car
    cls[oe] = "oe_other"
    cls[oe & b_loose & ~b_car] = "CASE3"
    cls[oe & b_track & ~b_car] = "CASE2"
    cls[fix2] = "FIX2"
    cls[two] = "TWO"
    d = d.copy()
    d["cls"] = cls
    d["veto"] = ((d.A_back >= 2) & (d.A_fwd >= 2) & b_track).astype(int)
    return d


FEAT_COLS = ["e1_in", "e1_out", "e1_back", "e1_fwd", "e1_alive", "e1_s",
             "e2_in", "e2_out", "e2_back", "e2_fwd", "e2_alive", "e2_s",
             "d_ab", "dz_pl", "dxy", "saddle", "contest_in", "contest_out"]


def graph_feats(r, cA, cB):
    """Canonical order: end 1 = the longer track (in+out chain), tie -> the higher score. Returns (feats, e1_is_A)."""
    LA, LB = r.A_back + r.A_fwd, r.B_back + r.B_fwd
    e1A = (LA, r.s_a) >= (LB, r.s_b)
    A = (r.A_in_n, r.A_out_n, r.A_back, r.A_fwd, r.A_alive, r.s_a)
    B = (r.B_in_n, r.B_out_n, r.B_back, r.B_fwd, r.B_alive, r.s_b)
    e1, e2 = (A, B) if e1A else (B, A)
    d3 = float(np.linalg.norm((cA - cB) * SCALE))
    dz = float(abs(cA[0] - cB[0]))
    dxy = float(np.linalg.norm((cA[1:] - cB[1:]) * SCALE[1:]))
    return np.array(list(e1) + list(e2) + [d3, dz, dxy, r.saddle, r.contest_in, r.contest_out], np.float32), e1A


def gauss_mask(cy, cx, sigma=2.0):
    yy, xx = np.mgrid[0:2 * HALF, 0:2 * HALF]
    return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2)).astype(np.float16)


def cut_crop(frames, T, t, mid, cE1, cE2):
    """frames: dict f -> (Z, Y, X) array. Returns (27, 2H, 2H) float16."""
    zc, yc, xc = int(round(mid[0])), int(round(mid[1])), int(round(mid[2]))
    Z, Y, X = frames[t].shape
    out = np.zeros((CH_IMG + 2, 2 * HALF, 2 * HALF), np.float32)
    y0, x0 = yc - HALF, xc - HALF
    ys, xs = slice(max(0, y0), min(Y, y0 + 2 * HALF)), slice(max(0, x0), min(X, x0 + 2 * HALF))
    oy, ox = ys.start - y0, xs.start - x0
    k = 0
    for dt in range(-NT, NT + 1):
        f = min(max(t + dt, 0), T - 1)
        vol = frames[f]
        for dz in range(-NZ, NZ + 1):
            z = min(max(zc + dz, 0), Z - 1)
            sl = vol[z, ys, xs]
            out[k, oy:oy + sl.shape[0], ox:ox + sl.shape[1]] = sl
            k += 1
    centre = out[(NT * (2 * NZ + 1)):(NT * (2 * NZ + 1) + 2 * NZ + 1)]
    lo, hi = np.percentile(centre, 20.0), np.percentile(centre, 99.5)
    out[:CH_IMG] = np.clip((out[:CH_IMG] - lo) / max(hi - lo, 1.0), -1.0, 4.0)
    out[CH_IMG] = gauss_mask(cE1[1] - y0, cE1[2] - x0)
    out[CH_IMG + 1] = gauss_mask(cE2[1] - y0, cE2[2] - x0)
    return out.astype(np.float16)


# ---------------------------------------------------------------- stage 1: pairs, features, crops
t_all = time.time()
mj = pd.read_csv(list(MJ.values())[0])
mj = type_pairs(mj)
print("classes:", mj.cls.value_counts().to_dict(), flush=True)
rng = np.random.default_rng(SEED)
keep = mj[mj.cls != "unann"]
un = mj[mj.cls == "unann"]
un = un.iloc[rng.choice(len(un), size=min(N_UNANN, len(un)), replace=False)]
sel = pd.concat([keep, un]).sort_values(["dataset", "t", "q"]).reset_index(drop=True)
print(f"selected {len(sel)} pairs over {sel.dataset.nunique()} movies", flush=True)
movies = sorted(sel.dataset.unique())
FOLD_OF = {m: k % N_FOLDS for k, m in enumerate(movies)}

ROWS, X_G = [], []
X_IMG = np.zeros((len(sel), CH_IMG + 2, 2 * HALF, 2 * HALF), np.float16)     # filled in place: one copy in RAM
n_mismatch = n_amb = 0
for k_, name in enumerate(movies):
    if time.time() - t_all > BUDGET_CROP_S:
        print(f"  crop budget reached at movie {k_}: stopping the crop stage", flush=True)
        break
    t0 = time.time()
    if name not in FDET_HP or name not in FDET_HYP:
        print(f"  {name}: cache missing (hp {name in FDET_HP}, hyp {name in FDET_HYP}), skipped", flush=True)
        continue
    zp, zh = np.load(FDET_HP[name]), np.load(FDET_HYP[name])
    g = sel[sel.dataset == name]
    img = open_image(zarr_path(ROOT, "train", name))
    T = img.shape[0]
    need = sorted({min(max(int(t) + dt, 0), T - 1) for t in g.t.unique() for dt in range(-NT, NT + 1)})
    frames = {f: np.asarray(img[f]) for f in need}
    seeds = {}
    for _, r in g.iterrows():
        t = int(r.t)
        if t not in seeds:
            c, s = np.asarray(zp[f"c{t}"], float).reshape(-1, 3), np.asarray(zp[f"s{t}"], float).reshape(-1)
            m = s >= UNET_THRESH
            seeds[t] = (c[m], s[m])
        C, S = seeds[t]
        par = np.asarray(zh[f"p{t}"], np.int64).reshape(-1, 2)
        if r.q >= len(par):
            n_mismatch += 1
            continue
        a_i, b_i = int(par[r.q, 0]), int(par[r.q, 1])
        if a_i >= len(C) or b_i >= len(C):
            n_mismatch += 1
            continue
        d_ab = float(np.linalg.norm((C[a_i] - C[b_i]) * SCALE))
        if abs(d_ab - r.d_ab) > 0.05:
            n_mismatch += 1
            continue
        if abs(S[a_i] - r.s_a) < 1e-4 and abs(S[b_i] - r.s_b) < 1e-4:
            cA, cB = C[a_i], C[b_i]
        elif abs(S[b_i] - r.s_a) < 1e-4 and abs(S[a_i] - r.s_b) < 1e-4:
            cA, cB = C[b_i], C[a_i]
        else:
            n_amb += 1
            cA, cB = C[a_i], C[b_i]
        feats, e1A = graph_feats(r, cA, cB)
        cE1, cE2 = (cA, cB) if e1A else (cB, cA)
        mid = 0.5 * (cA + cB)
        X_IMG[len(ROWS)] = cut_crop(frames, T, t, mid, cE1, cE2)
        X_G.append(feats)
        ROWS.append(dict(dataset=name, t=t, q=int(r.q), cls=r.cls, veto=int(r.veto), lab=int(r.lab), fold=FOLD_OF[name],
                         Az=float(cA[0]), Ay=float(cA[1]), Ax=float(cA[2]), Bz=float(cB[0]), By=float(cB[1]), Bx=float(cB[2]),
                         e1_is_A=int(e1A), **{c: float(v) for c, v in zip(FEAT_COLS, feats)}))
    del frames, img
    print(f"  [{k_ + 1}/{len(movies)}] {name}: {len(g)} pairs, {len(need)} frames, {time.time() - t0:.0f}s", flush=True)

P = pd.DataFrame(ROWS)
X_IMG = X_IMG[:len(ROWS)]
X_G = np.stack(X_G) if X_G else np.zeros((0, len(FEAT_COLS)), np.float32)
print(f"\nstage 1: {len(P)} pairs cut ({n_mismatch} index/distance mismatches skipped, {n_amb} ambiguous A/B kept), "
      f"crops {X_IMG.shape} {X_IMG.nbytes / 1e9:.2f} GB, {(time.time() - t_all) / 60:.1f} min", flush=True)
print("classes cut:", P.cls.value_counts().to_dict(), flush=True)
P.to_csv(OUT / "mergegate_pairs.csv", index=False)

# graph features: log1p on the counts, standardise on everything cut
XG = X_G.copy()
for j, c in enumerate(FEAT_COLS):
    if c.endswith(("_in", "_out", "_back", "_fwd")):
        XG[:, j] = np.log1p(XG[:, j])
G_MEAN, G_STD = XG.mean(0), XG.std(0) + 1e-6
XG = (XG - G_MEAN) / G_STD
np.savez(OUT / "mergegate_norm.npz", mean=G_MEAN, std=G_STD, feat_cols=np.array(FEAT_COLS))

IS_FIX = P.cls.isin(["FIX1", "FIX2", "FIXDUP"]).values
IS_C2 = (P.cls == "CASE2").values
y = IS_FIX.astype(np.float32)
train_mask = IS_FIX | IS_C2
fold = P.fold.values


# ---------------------------------------------------------------- graph-only baseline
p_graph = np.full(len(P), np.nan)
try:
    from sklearn.ensemble import GradientBoostingClassifier
    for k in range(N_FOLDS):
        tr = train_mask & (fold != k)
        te = fold == k
        if tr.sum() < 50 or y[tr].sum() < 5:
            continue
        clf = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=SEED)
        w = np.where(y[tr] == 1, (1 - y[tr].mean()) / max(y[tr].mean(), 1e-6), 1.0)
        clf.fit(XG[tr], y[tr], sample_weight=w)
        p_graph[te] = clf.predict_proba(XG[te])[:, 1]
    print("graph-only baseline done", flush=True)
except Exception:                                                   # noqa: BLE001
    traceback.print_exc()
P["p_graph"] = p_graph


# ---------------------------------------------------------------- stage 2: CNN, FIX vs CASE2, out of fold by movie
p_img = np.full(len(P), np.nan)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("torch", torch.__version__, "device", dev, flush=True)

    class Net(nn.Module):
        def __init__(self, c_in, n_g):
            super().__init__()

            def blk(a, b):
                return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                                     nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))
            self.b1, self.b2, self.b3, self.b4 = blk(c_in, 32), blk(32, 64), blk(64, 128), blk(128, 128)
            self.g = nn.Sequential(nn.Linear(n_g, 32), nn.ReLU(inplace=True))
            self.head = nn.Sequential(nn.Linear(128 + 32, 64), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(64, 1))

        def forward(self, x, g):
            x = F.max_pool2d(self.b1(x), 2)
            x = F.max_pool2d(self.b2(x), 2)
            x = F.max_pool2d(self.b3(x), 2)
            x = self.b4(x).mean((2, 3))
            return self.head(torch.cat([x, self.g(g)], 1)).squeeze(1)

    def augment(xb):
        k = int(np.random.randint(4))
        xb = torch.rot90(xb, k, (2, 3))
        if np.random.rand() < 0.5:
            xb = torch.flip(xb, (3,))
        return xb

    def predict(model, idx):
        model.eval()
        out = np.zeros(len(idx), np.float32)
        with torch.no_grad():
            for s in range(0, len(idx), 256):
                ii = idx[s:s + 256]
                xb = torch.from_numpy(X_IMG[ii].astype(np.float32)).to(dev)
                gb = torch.from_numpy(XG[ii]).to(dev)
                pr = 0.0
                for k in range(4):                       # test-time rotations
                    pr = pr + torch.sigmoid(model(torch.rot90(xb, k, (2, 3)), gb)).cpu().numpy()
                out[s:s + len(ii)] = pr / 4
        return out

    for k in range(N_FOLDS):
        t0 = time.time()
        tr = np.where(train_mask & (fold != k))[0]
        te = np.where(fold == k)[0]
        if len(tr) < 50 or y[tr].sum() < 5:
            print(f"  fold {k}: too few training rows, skipped", flush=True)
            continue
        torch.manual_seed(SEED + k)
        np.random.seed(SEED + k)
        model = Net(CH_IMG + 2, len(FEAT_COLS)).to(dev)
        pos_w = torch.tensor([(len(tr) - y[tr].sum()) / max(y[tr].sum(), 1.0)], device=dev)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * int(np.ceil(len(tr) / BATCH)))
        for ep in range(EPOCHS):
            model.train()
            perm = np.random.permutation(tr)
            tot = 0.0
            for s in range(0, len(perm), BATCH):
                ii = perm[s:s + BATCH]
                xb = augment(torch.from_numpy(X_IMG[ii].astype(np.float32)).to(dev))
                gb = torch.from_numpy(XG[ii]).to(dev)
                yb = torch.from_numpy(y[ii]).to(dev)
                loss = F.binary_cross_entropy_with_logits(model(xb, gb), yb, pos_weight=pos_w)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                tot += float(loss) * len(ii)
            if ep % 5 == 4 or ep == EPOCHS - 1:
                print(f"    fold {k} epoch {ep + 1}: loss {tot / len(tr):.4f}", flush=True)
        p_img[te] = predict(model, te)
        torch.save(model.state_dict(), OUT / f"mergegate_fold{k}.pt")
        print(f"  fold {k}: train {len(tr)} (FIX {int(y[tr].sum())}), test {len(te)}, {time.time() - t0:.0f}s", flush=True)
except Exception:                                                   # noqa: BLE001
    traceback.print_exc()
    print("CNN stage failed: saving the crops (uint8) so a rerun can train from them", flush=True)
    np.savez_compressed(OUT / "mergegate_crops_u8.npz", x=np.clip((X_IMG.astype(np.float32) + 1) * 51, 0, 255).astype(np.uint8))
P["p_img"] = p_img
P.to_csv(OUT / "mergegate_pairs.csv", index=False)


# ---------------------------------------------------------------- report
def auc(p, pos, neg):
    from sklearn.metrics import roc_auc_score
    m = (pos | neg) & np.isfinite(p)
    return roc_auc_score(pos[m].astype(int), p[m]) if pos[m].sum() and neg[m].sum() else float("nan")


CLASSES = ["FIX1", "FIX2", "FIXDUP", "CASE2", "CASE3", "oe_other", "one_other", "TWO", "unann"]
N_UNANN_POP = int((mj.cls == "unann").sum())
for nm, p in (("graph only", P.p_graph.values), ("image + graph", P.p_img.values)):
    ok = np.isfinite(p)
    if ok.sum() == 0:
        continue
    print(f"\n=== MERGEGATE {nm}: AUC FIX vs CASE2 {auc(p, IS_FIX, IS_C2):.3f}; FIX vs all non-FIX typed "
          f"{auc(p, IS_FIX, ~IS_FIX & (P.cls != 'unann').values):.3f} (out of fold, {ok.sum()} pairs) ===", flush=True)
    for veto_on in (False, True):
        pp = np.where(P.veto.values == 1, -1.0, p) if veto_on else p
        print(f"  {'WITH the both-full-tracks veto' if veto_on else 'no veto'}: firing share per class at a threshold set "
              f"on FIX recall (all FIX, out of fold); last two columns = CASE2 fires per FIX caught, unann fires scaled to the "
              f"{N_UNANN_POP:,} unannotated pairs per FIX caught", flush=True)
        print(f"    {'recall':>6} {'thr':>6} " + "".join(f"{c:>9}" for c in CLASSES) + f"{'C2/fix':>8} {'un/fix':>8}", flush=True)
        n_fix_all = int((IS_FIX & ok).sum())
        fixp = np.sort(p[IS_FIX & ok & (P.veto.values == 0)] if veto_on else p[IS_FIX & ok])[::-1]
        for rec in (0.5, 0.6, 0.7, 0.8, 0.9):
            need = int(np.ceil(rec * n_fix_all))
            if need > len(fixp) or need < 1:
                print(f"    {rec:>6.1f}    unreachable: only {len(fixp)} of {n_fix_all} FIX survive the veto", flush=True)
                continue
            thr = float(fixp[need - 1])
            fire = pp >= thr
            shares = []
            for c in CLASSES:
                m = (P.cls.values == c) & ok
                shares.append(fire[m].mean() if m.sum() else float("nan"))
            n_fix = int(fire[IS_FIX & ok].sum())
            n_c2 = int(fire[IS_C2 & ok].sum())
            m_un = (P.cls.values == "unann") & ok
            un_scaled = fire[m_un].mean() * N_UNANN_POP if m_un.sum() else float("nan")
            print(f"    {rec:>6.1f} {thr:>6.3f} " + "".join(f"{s:>9.3f}" for s in shares)
                  + f"{n_c2 / max(n_fix, 1):>8.2f} {un_scaled / max(n_fix, 1):>8.1f}", flush=True)
print(f"\nMERGEGATE DONE in {(time.time() - t_all) / 60:.1f} min", flush=True)
print("MERGEGATE OK", flush=True)
