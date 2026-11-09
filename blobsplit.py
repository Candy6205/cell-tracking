"""blobsplit.py -- does the raw image show that a detection HIDES a second nucleus?

SCRIPT_VERSION 2026-09-17a-blobsplit

WHY. 2,164 of REF's 5,698 missed edges have an endpoint the detector never found (38%;
oracle ~ +0.019 edge_J). At such a cell the fused heatmap reads 0.10-0.15 but has no local
maximum: the nearest maximum sits 6-7 um away on a neighbour, i.e. the cell's response is
merged into the neighbour's peak. No gate or NMS setting recovers it. The open question:
does the neighbour's blob in the RAW image look different from a single nucleus (bigger,
elongated, peak off-centre, a second intensity maximum), so a splitter could add the
second point? maskfeat measured the same kind of shape features at PAIR sites (two peaks
already present) and found volume / shape nearly useless (AUC 0.51-0.54); this is the
other geometry (one peak, two nuclei) and has not been measured.

SITES, from the 128 6bba movies' GT (train geffs) and the shipped F4HP primaries
(hparacache, score >= 0.15):
  host   the nearest primary (within 12 um) to an annotated cell that has NO primary within
         7 um (the scorer's radius) -- the blob that hides the missed cell; the missed cell's
         position g is kept for DIAGNOSTICS only, never as a feature
  ctrl   a primary within 2.5 um of an annotated cell, no other primary within 8 um and no
         other annotated cell within 10 um: a clean single nucleus (sampled per movie)
  dense  a primary within 2.5 um of an annotated cell WITH another primary within 8 um: a
         nucleus in a crowd that the detector did resolve (sampled per movie)

FEATURES at the detection d, raw crop 10 um half-width, smoothed (0.6 / 1.5 / 1.5 voxels):
  I_d, background (crop median), contrast; at thresholds 0.5 / 0.7 of (I_d - bg): volume of
  d's component (um^3), its longest axis and axis ratio (PCA in um), extent along the longest
  axis, offset of d from the component's intensity-weighted centroid (um), solidity, z
  extent; local intensity maxima inside the 0.5 component (>= 3 um apart): count, the second
  maximum's height ratio and distance from d; gradient magnitude at d; texture cv.
DIAGNOSTICS (host only): I(g) / I(d), the dip along d -> g, a local maximum within 2.5 um of g.

REPORT. Per feature: medians of host / ctrl / dense and AUC host vs ctrl, host vs dense; a
HistGB stack, GroupKFold by movie, AUC and the false-fire rate on ctrl / dense at 50 / 70%
host recall; base-rate arithmetic (hosts ~ 2.3% of primaries): wrong splits per true split.

Attach: faustine649/cell-tracking (viz_cells) + the competition + biohub-hparacache. CPU.
"""
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

SCRIPT_VERSION = "2026-09-17a-blobsplit"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

SCALE = np.array([1.625, 0.40625, 0.40625])
UNET_THRESH = 0.15
MISS_UM = 7.0            # a cell with no primary within this is missed (the scorer's radius)
HOST_UM = 12.0           # the host must be within this of the missed cell
CTRL_MATCH_UM = 2.5
LONE_PRIM_UM = 8.0
LONE_GT_UM = 10.0
N_CTRL_PER_MOVIE = 25
HALF_UM = 10.0
SIGMA_VOX = (0.6, 1.5, 1.5)
BALL_UM = 1.2
FRACS = (0.5, 0.7)
MAX_SEP_UM = 3.0         # local maxima closer than this are one maximum
RNG = np.random.default_rng(0)

CODE_ROOTS = [
    "/kaggle/input/datasets/faustine649/cell-tracking/cell tracking",
    "/kaggle/input/datasets/faustine649/cell-tracking",
    "/kaggle/input/cell-tracking/cell tracking",
    "/kaggle/input/cell-tracking",
]
code = next((c for c in CODE_ROOTS if pathlib.Path(c, "viz_cells.py").exists()), None)
if code is not None:
    sys.path.insert(0, code)

FEAT_NAMES = (["I_d", "I_bg", "contrast", "grad_d", "tex_cv"]
              + [f"{k}_{int(f * 100)}" for f in FRACS for k in ("vol", "ax_long", "ax_ratio", "extent", "off_centroid", "solid", "z_extent")]
              + ["n_max", "max2_ratio", "max2_dist", "max2_off"])
DIAG_NAMES = ["d_host_g", "I_g_ratio", "dip_dg", "max_near_g"]


def _ball_mean(crop, p_vox, r_um):
    z, y, x = (int(round(v)) for v in p_vox)
    rz = max(1, int(round(r_um / SCALE[0])))
    ry = max(1, int(round(r_um / SCALE[1])))
    sub = crop[max(0, z - rz):z + rz + 1, max(0, y - ry):y + ry + 1, max(0, x - ry):x + ry + 1]
    return float(sub.mean()) if sub.size else float("nan")


def _component(mask, p):
    lab, n = ndi.label(mask, structure=np.ones((3, 3, 3), bool))
    pz, py, px = (int(round(v)) for v in p)
    pz, py, px = (min(max(pz, 0), mask.shape[0] - 1), min(max(py, 0), mask.shape[1] - 1), min(max(px, 0), mask.shape[2] - 1))
    lid = lab[pz, py, px]
    if lid == 0:
        # nearest labelled voxel within 2 um
        idx = np.argwhere(lab > 0)
        if len(idx) == 0:
            return None
        d = np.linalg.norm((idx - np.array([pz, py, px])) * SCALE, axis=1)
        k = int(np.argmin(d))
        if d[k] > 2.0:
            return None
        lid = lab[tuple(idx[k])]
    return lab == lid


def _shape(comp, sm, p):
    idx = np.argwhere(comp)
    if len(idx) < 4:
        return dict(vol=float(len(idx) * SCALE.prod()), ax_long=np.nan, ax_ratio=np.nan, extent=np.nan,
                    off_centroid=np.nan, solid=np.nan, z_extent=float(SCALE[0] * (np.ptp(idx[:, 0]) + 1)) if len(idx) else np.nan)
    pts = idx * SCALE
    w = sm[comp].astype(float)
    w = np.clip(w - w.min(), 1e-6, None)
    cen = (pts * w[:, None]).sum(0) / w.sum()
    c = pts - pts.mean(0)
    cov = c.T @ c / len(c)
    ev, evec = np.linalg.eigh(cov)
    ev = np.clip(ev, 1e-9, None)
    long_ax = evec[:, -1]
    proj = c @ long_ax
    bbox = np.prod(idx.max(0) - idx.min(0) + 1) * SCALE.prod()
    return dict(vol=float(len(idx) * SCALE.prod()), ax_long=float(np.sqrt(ev[-1])), ax_ratio=float(np.sqrt(ev[-1] / ev[-2])),
                extent=float(proj.max() - proj.min()), off_centroid=float(np.linalg.norm(cen - np.asarray(p) * SCALE)),
                solid=float(len(idx) * SCALE.prod() / max(bbox, 1e-6)), z_extent=float(SCALE[0] * (np.ptp(idx[:, 0]) + 1)))


def _local_maxima(sm, comp, p):
    """Local maxima of the smoothed crop inside the component, merged within MAX_SEP_UM, sorted by
    height; returns (count, second/first height ratio, distance d -> second, offset of the second
    from d along the longest axis is not needed)."""
    fp = np.ones((3, 5, 5), bool)
    mx = (sm == ndi.maximum_filter(sm, footprint=fp)) & comp
    idx = np.argwhere(mx)
    if len(idx) == 0:
        return 0, np.nan, np.nan, np.nan
    h = sm[mx]
    order = np.argsort(-h)
    kept = []
    for k in order:
        q = idx[k] * SCALE
        if all(np.linalg.norm(q - r) >= MAX_SEP_UM for r, _ in kept):
            kept.append((q, float(h[k])))
    p_um = np.asarray(p, float) * SCALE
    # the maximum nearest d is "d's own"; the highest OTHER one is the second
    own = int(np.argmin([np.linalg.norm(q - p_um) for q, _ in kept]))
    others = [(q, hh) for k, (q, hh) in enumerate(kept) if k != own]
    if not others:
        return len(kept), 0.0, np.nan, np.nan
    q2, h2 = max(others, key=lambda x: x[1])
    return len(kept), h2 / max(kept[own][1], 1e-6), float(np.linalg.norm(q2 - p_um)), float(abs((q2 - p_um)[0]))


def blob_features(crop, d, g=None):
    """crop: raw (Z, Y, X); d: the detection in crop voxels; g (optional): the missed cell, diagnostics only."""
    sm = ndi.gaussian_filter(crop.astype(np.float32), SIGMA_VOX)
    I_d = _ball_mean(sm, d, BALL_UM)
    I_bg = float(np.percentile(sm, 50))
    f = dict(I_d=I_d, I_bg=I_bg, contrast=(I_d - I_bg) / max(abs(I_bg), 1e-6))
    gz, gy, gx = np.gradient(sm, SCALE[0], SCALE[1], SCALE[2])
    f["grad_d"] = _ball_mean(np.sqrt(gz ** 2 + gy ** 2 + gx ** 2), d, BALL_UM)
    comp50 = None
    for fr in FRACS:
        mask = sm > I_bg + fr * (I_d - I_bg)
        comp = _component(mask, d)
        tag = int(fr * 100)
        if comp is None:
            for k in ("vol", "ax_long", "ax_ratio", "extent", "off_centroid", "solid", "z_extent"):
                f[f"{k}_{tag}"] = np.nan
            continue
        st = _shape(comp, sm, d)
        for k, v in st.items():
            f[f"{k}_{tag}"] = v
        if fr == 0.5:
            comp50 = comp
            inside = sm[comp]
            f["tex_cv"] = float(inside.std() / max(inside.mean(), 1e-6)) if inside.size > 10 else np.nan
    if comp50 is not None:
        f["n_max"], f["max2_ratio"], f["max2_dist"], f["max2_off"] = _local_maxima(sm, comp50, d)
    else:
        f["n_max"], f["max2_ratio"], f["max2_dist"], f["max2_off"] = np.nan, np.nan, np.nan, np.nan
    diag = [np.nan] * len(DIAG_NAMES)
    if g is not None:
        g = np.asarray(g, float)
        d_um = float(np.linalg.norm((g - d) * SCALE))
        I_g = _ball_mean(sm, g, BALL_UM)
        mids = [_ball_mean(sm, d + (g - d) * s, BALL_UM) for s in (0.35, 0.5, 0.65)]
        dip = (min(mids) - I_bg) / max(min(I_d, I_g) - I_bg, 1e-6)
        fp = np.ones((3, 5, 5), bool)
        mx = np.argwhere(sm == ndi.maximum_filter(sm, footprint=fp))
        near = bool(len(mx)) and bool((np.linalg.norm((mx - g) * SCALE, axis=1) <= 2.5).any())
        diag = [d_um, I_g / max(I_d, 1e-6), dip, float(near)]
    return [f.get(k, np.nan) for k in FEAT_NAMES], diag


def build_sites(name, gt, prim):
    """gt: dict t -> (rows, pos_um); prim: list per t of (coords_vox, scores). Returns rows
    (kind, t, d_vox, g_vox or None)."""
    out = []
    T = len(prim)
    ctrl_pool, dense_pool = [], []
    for t in range(T):
        if t not in gt:
            continue
        g_pos = gt[t]
        C = np.asarray(prim[t][0], float)
        if len(C) == 0:
            continue
        Cu = C * SCALE
        ptree = cKDTree(Cu)
        gtree = cKDTree(g_pos)
        d_near, i_near = ptree.query(g_pos)
        for k in np.where(d_near > MISS_UM)[0]:
            if d_near[k] <= HOST_UM:
                out.append(("host", t, C[i_near[k]], g_pos[k] / SCALE))
        # singles: matched within 2.5 um, no other primary within 8, no other GT within 10
        for k in np.where(d_near <= CTRL_MATCH_UM)[0]:
            i = int(i_near[k])
            n_prim = len(ptree.query_ball_point(Cu[i], LONE_PRIM_UM)) - 1
            n_gt = len(gtree.query_ball_point(g_pos[k], LONE_GT_UM)) - 1
            if n_prim == 0 and n_gt == 0:
                ctrl_pool.append(("ctrl", t, C[i], None))
            elif n_prim >= 1:
                dense_pool.append(("dense", t, C[i], None))
    for pool in (ctrl_pool, dense_pool):
        if len(pool) > N_CTRL_PER_MOVIE:
            pick = RNG.choice(len(pool), N_CTRL_PER_MOVIE, replace=False)
            pool = [pool[i] for i in pick]
        out.extend(pool)
    return out


if __name__ == "__main__":
    from viz_cells import find_root, geff_path, read_geff, open_image  # noqa: E402
    ROOT = find_root(None)
    TRAIN_DIR = pathlib.Path(ROOT) / "train"
    FDET = {}
    for d_ in pathlib.Path("/kaggle/input").glob("*/**/fdet_F4hp_*.npz"):
        FDET[d_.name[len("fdet_F4hp_"):-len(".npz")]] = d_
    SIX = sorted(n for n in FDET if n.startswith("6bba_"))
    assert len(SIX) >= 128, f"{len(SIX)} fdet_F4hp caches (attach biohub-hparacache)"
    print(f"movies {len(SIX)}", flush=True)
    OUT = pathlib.Path("/kaggle/working")
    fh = open(OUT / "blobsplit_sites.csv", "w", encoding="utf-8", newline="\n")
    fh.write("kind,dataset,t," + ",".join(FEAT_NAMES) + "," + ",".join(DIAG_NAMES) + "\n")
    RES, DIAG, GROUP = defaultdict(list), [], defaultdict(list)
    COUNT = defaultdict(int)
    hz = int(np.ceil(HALF_UM / SCALE[0]))
    hy = int(np.ceil(HALF_UM / SCALE[1]))
    t0 = time.time()
    n_ann = n_missed = 0
    for k_, name in enumerate(SIX):
        tm = time.time()
        z = np.load(FDET[name])
        T = max(int(k[1:]) for k in z.files if k.startswith("c")) + 1
        prim = []
        for t in range(T):
            c = np.asarray(z[f"c{t}"], float)
            s = np.asarray(z[f"s{t}"], float)
            prim.append((c[s >= UNET_THRESH], s[s >= UNET_THRESH]))
        gtg = read_geff(geff_path(ROOT, "train", name))
        g_t = np.asarray(gtg["t"], int)
        g_pos = np.stack([np.asarray(gtg[c_], float) for c_ in ("z", "y", "x")], axis=1) * SCALE
        gt = {int(tt): g_pos[g_t == tt] for tt in np.unique(g_t)}
        n_ann += len(g_t)
        sites = build_sites(name, gt, prim)
        n_missed += sum(1 for s_ in sites if s_[0] == "host")
        img = open_image(TRAIN_DIR / f"{name}.zarr")
        by_t = defaultdict(list)
        for s_ in sites:
            by_t[s_[1]].append(s_)
        for t in sorted(by_t):
            frame = None
            for kind, _t, d, g in by_t[t]:
                z0, y0, x0 = (max(0, int(round(d[0])) - hz), max(0, int(round(d[1])) - hy), max(0, int(round(d[2])) - hy))
                z1, y1, x1 = (int(round(d[0])) + hz + 1, int(round(d[1])) + hy + 1, int(round(d[2])) + hy + 1)
                try:
                    crop = np.asarray(img[t, z0:z1, y0:y1, x0:x1])
                except Exception:                                   # noqa: BLE001
                    if frame is None:
                        frame = np.asarray(img[t])
                    crop = frame[z0:z1, y0:y1, x0:x1]
                off = np.array([z0, y0, x0], float)
                g_c = None
                if g is not None:
                    g_c = g - off
                    if not (np.all(g_c >= 0) and np.all(g_c < np.array(crop.shape))):
                        g_c = None
                vals, diag = blob_features(crop, d - off, g_c)
                fh.write(f"{kind},{name},{t}," + ",".join(f"{v:.5g}" for v in vals) + "," + ",".join(f"{v:.5g}" for v in diag) + "\n")
                RES[kind].append(vals)
                GROUP[kind].append(k_)
                if kind == "host":
                    DIAG.append(diag)
                COUNT[kind] += 1
        fh.flush()
        del img
        print(f"[{k_ + 1}/{len(SIX)}] {name}: sites host {sum(1 for s_ in sites if s_[0] == 'host')} ctrl "
              f"{sum(1 for s_ in sites if s_[0] == 'ctrl')} dense {sum(1 for s_ in sites if s_[0] == 'dense')} "
              f"{time.time() - tm:.0f}s (total {(time.time() - t0) / 60:.1f} min)", flush=True)
    fh.close()
    print(f"\nannotated GT nodes {n_ann:,}; missed at {MISS_UM} um with a host within {HOST_UM} um: {n_missed:,} "
          f"({n_missed / max(1, n_ann):.2%}); sites {dict(COUNT)}", flush=True)
    # ---- summary ----------------------------------------------------------------------------
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold, cross_val_predict
    H, Cc, Dd = (np.asarray(RES[k], float).reshape(-1, len(FEAT_NAMES)) for k in ("host", "ctrl", "dense"))
    dg = np.asarray(DIAG, float).reshape(-1, len(DIAG_NAMES))
    print(f"\n=== BLOBSPLIT: {len(H)} host, {len(Cc)} clean single, {len(Dd)} dense single sites ===", flush=True)
    if len(dg):
        ok = np.isfinite(dg[:, 1])
        print(f"  host diagnostics (the missed cell g, never a feature): d(host, g) median {np.nanmedian(dg[:, 0]):.2f} um; "
              f"I(g)/I(d) median {np.nanmedian(dg[ok, 1]):.3f} (p25 {np.nanpercentile(dg[ok, 1], 25):.3f}, p75 {np.nanpercentile(dg[ok, 1], 75):.3f}); "
              f"dip along d->g median {np.nanmedian(dg[ok, 2]):.3f} (< 0.9 in {np.mean(dg[ok, 2] < 0.9):.2f}); "
              f"a raw local maximum within 2.5 um of g in {np.nanmean(dg[ok, 3]):.2f}", flush=True)
    print(f"\n  {'feature':<16}{'host med':>11}{'single med':>11}{'dense med':>11}{'AUC h/s':>9}{'AUC h/d':>9}", flush=True)
    for k, nm in enumerate(FEAT_NAMES):
        a_, b_, c_ = H[:, k], Cc[:, k], Dd[:, k]
        oa, ob, oc = np.isfinite(a_), np.isfinite(b_), np.isfinite(c_)
        if oa.sum() < 30 or ob.sum() < 30:
            continue
        auc = roc_auc_score(np.r_[np.ones(oa.sum()), np.zeros(ob.sum())], np.r_[a_[oa], b_[ob]])
        auc2 = roc_auc_score(np.r_[np.ones(oa.sum()), np.zeros(oc.sum())], np.r_[a_[oa], c_[oc]]) if oc.sum() >= 30 else float("nan")
        print(f"  {nm:<16}{np.nanmedian(a_):>11.3f}{np.nanmedian(b_):>11.3f}{np.nanmedian(c_):>11.3f}{max(auc, 1 - auc):>9.3f}{max(auc2, 1 - auc2):>9.3f}", flush=True)
    X = np.concatenate([H, Cc, Dd])
    X = np.where(np.isfinite(X), X, -1.0)
    y = np.r_[np.ones(len(H)), np.zeros(len(Cc) + len(Dd))]
    grp = np.r_[GROUP["host"], GROUP["ctrl"], GROUP["dense"]]
    is_c = np.r_[np.zeros(len(H), bool), np.ones(len(Cc), bool), np.zeros(len(Dd), bool)]
    is_d = np.r_[np.zeros(len(H) + len(Cc), bool), np.ones(len(Dd), bool)]
    p = cross_val_predict(HistGradientBoostingClassifier(max_iter=200, random_state=0), X, y,
                          cv=GroupKFold(5), groups=grp, method="predict_proba")[:, 1]
    ph = p[y == 1]
    print(f"\n  stacked, GroupKFold by movie: AUC host vs all singles {roc_auc_score(y, p):.3f}; host vs clean "
          f"{roc_auc_score(np.r_[np.ones(len(H)), np.zeros(len(Cc))], np.r_[ph, p[is_c]]):.3f}; host vs dense "
          f"{roc_auc_score(np.r_[np.ones(len(H)), np.zeros(len(Dd))], np.r_[ph, p[is_d]]):.3f}", flush=True)
    N_PRIM = 2_091_008
    frac_host = n_missed / max(1, n_ann)
    print(f"  base rate: hosts ~ {frac_host:.2%} of primaries (~{frac_host * N_PRIM:,.0f} of {N_PRIM:,}); a wrong split puts a "
          f"second point next to a real cell (a swap site, ~1 edge at risk), a right one recovers ~2 edges", flush=True)
    for rec in (0.3, 0.5, 0.7):
        thr = float(np.quantile(ph, 1 - rec))
        fc, fd = (p[is_c] >= thr).mean(), (p[is_d] >= thr).mean()
        f_all = 0.5 * (fc + fd)
        wrong_per_fix = f_all * (1 - frac_host) / max(rec * frac_host, 1e-9)
        print(f"  at {rec:.0%} host recall (thr {thr:.3f}): fires on {fc:.3f} of clean singles, {fd:.3f} of dense singles "
              f"-> ~{wrong_per_fix:.1f} wrong splits per hidden cell found (break-even ~2)", flush=True)
    print(f"\nBLOBSPLIT DONE in {(time.time() - t0) / 60:.1f} min; blobsplit_sites.csv written", flush=True)
    print("BLOBSPLIT OK", flush=True)
