"""divtrip: can a DIVISION TRIPLE (mother p at t, daughters a and b at t+1) be scored on its own, without the linker?

WHY (2026-09-24). The shipped ILP has no notion of division: a second out-edge costs a flat 0.55 and is judged only by the
edge ranker's probability, whose prior is constant velocity, so at a real division the second daughter's edge reads p ~0.07
and the ILP does not fork; 65% of the shipped forks come from the orphan patch, whose precondition (the mother linked to one
daughter, the other an orphan) rests on the same ranker. A division-aware ILP needs its own variable per triple with a
score of its own. This measures whether such a score exists.

WHAT. Per movie (the 199 train movies with F4HP primaries, score >= 0.15): every triple with a, b within R_PARENT of p,
d_ab <= R_SISTER, and the daughters' midpoint within R_MID of p (real daughters straddle the mother). Features that need no
linker: geometry (d_pa, d_pb, d_ab, asymmetry, midpoint offset, angle at p, z spreads), motion from the nearest primaries
(the mother's predecessor m at t-1 within SNAP_UM: how well each daughter and the daughters' centre of mass continue v_m;
successors at t+2: separation growth), detection scores, densities, and 'orphan-ness' (distance from a, and from b, to the
nearest primary at t other than p: a daughter has none nearby, a passing cell has its own predecessor). Labels: the GT
divisions whose mother and both daughters match a primary within SNAP_UM. Pass 1 samples negatives uniformly per movie,
fits scorer A (HistGB, 5 folds by movie) and reports the geometric ceiling and the recall at the top-K per frame. Pass 2
scores every triple with the out-of-fold model, keeps the top TOPK per frame, adds raw-image features for those (contrast,
peak and half-max radius at p, a, b; ratios) from the decoded frames, fits scorer B on them, and writes the candidate
tables divtrip_<movie>.csv -- the division variables a later ILP bench would take.
Attach: the competition data + faustine649/cell-tracking (viz_cells.py) + biohub-hparacache + biohub-hpara44. CPU.
"""
import os
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import ndimage as _ndi
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

SCRIPT_VERSION = "2026-09-24b-divtrip"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

CODE_ROOTS = [
    "/kaggle/input/datasets/faustine649/cell-tracking/cell tracking",
    "/kaggle/input/datasets/faustine649/cell-tracking",
    "/kaggle/input/cell-tracking/cell tracking",
    "/kaggle/input/cell-tracking",
]
code = next((c for c in CODE_ROOTS if pathlib.Path(c, "viz_cells.py").exists()), None)
assert code is not None, "attach faustine649/cell-tracking"
sys.path.insert(0, code)
from viz_cells import find_root, geff_path, open_image, read_geff, zarr_path   # noqa: E402

ROOT = find_root(None)
OUT = pathlib.Path("/kaggle/working")
INPUT_ROOT = "/kaggle/input"
SCALE = np.array([1.625, 0.40625, 0.40625])
SCORE_MIN = 0.15
R_PARENT, R_SISTER, R_MID = 14.0, 16.0, 8.0      # candidate geometry, um
D_AB_MIN = 2.0                                  # two daughters are not the same point
SNAP_UM = 5.0                                   # GT node -> primary; predecessor / successor search
N_NEG_PER_MOVIE = 6000
TOPK = 20                                       # candidates kept per frame in pass 2
N_FOLDS, SEED = 5, 0
BUDGET_S = 10.0 * 3600
FI_HALF_UM, FI_PEAK_UM, FI_SIZE_UM = 8.0, 2.0, 3.5
FI_SIGMA_VOX = (0.6, 1.5, 1.5)

GEO = ["d_pa", "d_pb", "d_ab", "asym", "mid_off", "ang", "dz_pa", "dz_pb", "dz_ab", "t_frac",
       "has_m", "speed_m", "cont_a", "cont_b", "cont_min", "cont_mid", "has_a2", "has_b2", "sep_grow",
       "sc_p", "sc_a", "sc_b", "sc_min", "dens_p10", "n_t1_12", "orph_a", "orph_b", "orph_min", "orph_max"]
IMG = ["i_p", "c_p", "r_p", "hm_p", "c_a", "c_b", "hm_a", "hm_b", "hm_max_ratio", "hm_sum_ratio", "i_ratio", "br_p"]


def _walk_files(match):
    hits = {}
    for root_, dirs, fnames in os.walk(INPUT_ROOT):
        dirs[:] = [d for d in dirs if "cell-tracking-during-development" not in d and not d.endswith(".zarr") and not d.endswith(".geff")]
        for fn in fnames:
            key = match(fn)
            if key is not None:
                hits.setdefault(key, pathlib.Path(root_) / fn)
    return hits


def _npz_prefix(prefix):
    return lambda fn: fn[len(prefix):-len(".npz")] if fn.startswith(prefix) and fn.endswith(".npz") else None


def primaries(det, t):
    """(voxel coords (N,3), scores (N,)) of the primaries at frame t"""
    C, S = det[f"c{t}"], det[f"s{t}"]
    m = S >= SCORE_MIN
    return np.asarray(C[m], float), np.asarray(S[m], float)


def gt_triples(gt):
    """{t: [(m_pos_um, d1_pos_um, d2_pos_um)]} of the GT divisions with daughters one frame on"""
    g_t = np.asarray(gt["t"], float).round().astype(int)
    pos = np.stack([np.asarray(gt[c], float) for c in ("z", "y", "x")], 1) * SCALE
    row = {int(v): i for i, v in enumerate(gt["node_ids"])}
    gs = defaultdict(list)
    for a, b in gt["edges"]:
        ra, rb = row.get(int(a)), row.get(int(b))
        if ra is not None and rb is not None:
            gs[ra].append(rb)
    out = defaultdict(list)
    for m, ch in gs.items():
        if len(ch) >= 2 and g_t[ch[0]] == g_t[m] + 1 and g_t[ch[1]] == g_t[m] + 1:
            out[int(g_t[m])].append((pos[m], pos[ch[0]], pos[ch[1]]))
    return out


def triples_frame(P, S, Pn, Sn, Pm, Pnn, t, T):
    """all candidate triples of frame t -> t+1 with the linker-free features. P/S at t, Pn/Sn at t+1 (um), Pm at t-1
    (or None), Pnn at t+2 (or None). Returns (idx (K,3) [p, a, b], feats (K, len(GEO)))."""
    if len(P) == 0 or len(Pn) < 2:
        return np.zeros((0, 3), int), np.zeros((0, len(GEO)))
    tree_n = cKDTree(Pn)
    tree_t = cKDTree(P)
    nb = tree_n.query_ball_point(P, R_PARENT)
    pi, ai, bi = [], [], []
    for p, lst in enumerate(nb):
        if len(lst) < 2:
            continue
        lst = np.asarray(lst)
        Q = Pn[lst]
        # pairs i < j
        iu, ju = np.triu_indices(len(lst), 1)
        d_ab = np.linalg.norm(Q[iu] - Q[ju], axis=1)
        mid = 0.5 * (Q[iu] + Q[ju])
        mo = np.linalg.norm(mid - P[p], axis=1)
        ok = (d_ab <= R_SISTER) & (d_ab >= D_AB_MIN) & (mo <= R_MID)
        if ok.any():
            pi.append(np.full(int(ok.sum()), p)); ai.append(lst[iu[ok]]); bi.append(lst[ju[ok]])
    if not pi:
        return np.zeros((0, 3), int), np.zeros((0, len(GEO)))
    pi, ai, bi = np.concatenate(pi), np.concatenate(ai), np.concatenate(bi)
    Pp, Pa, Pb = P[pi], Pn[ai], Pn[bi]
    va, vb = Pa - Pp, Pb - Pp
    d_pa, d_pb, d_ab = np.linalg.norm(va, axis=1), np.linalg.norm(vb, axis=1), np.linalg.norm(Pa - Pb, axis=1)
    asym = np.abs(d_pa - d_pb) / np.maximum(d_pa + d_pb, 1e-6)
    mid_off = np.linalg.norm(0.5 * (Pa + Pb) - Pp, axis=1)
    cosang = np.einsum("ij,ij->i", va, vb) / np.maximum(d_pa * d_pb, 1e-6)
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    # motion: the mother's predecessor (nearest primary at t-1 within SNAP_UM)
    n = len(pi)
    has_m = np.zeros(n); speed_m = np.full(n, np.nan); cont_a = np.full(n, np.nan); cont_b = np.full(n, np.nan); cont_mid = np.full(n, np.nan)
    if Pm is not None and len(Pm):
        dm, jm = cKDTree(Pm).query(P, k=1)
        okm = dm <= SNAP_UM
        v_m = P - Pm[jm]
        hm = okm[pi]
        has_m[hm] = 1
        vm = v_m[pi]
        speed_m[hm] = np.linalg.norm(vm[hm], axis=1)
        cont_a[hm] = np.linalg.norm(va[hm] - vm[hm], axis=1)
        cont_b[hm] = np.linalg.norm(vb[hm] - vm[hm], axis=1)
        cont_mid[hm] = np.linalg.norm(0.5 * (va[hm] + vb[hm]) - vm[hm], axis=1)
    cont_min = np.fmin(cont_a, cont_b)
    # successors at t+2: separation growth
    has_a2 = np.zeros(n); has_b2 = np.zeros(n); sep_grow = np.full(n, np.nan)
    if Pnn is not None and len(Pnn):
        d2, j2 = cKDTree(Pnn).query(Pn, k=1)
        ok2 = d2 <= SNAP_UM
        A2, B2 = Pnn[j2[ai]], Pnn[j2[bi]]
        both = ok2[ai] & ok2[bi]
        has_a2[ok2[ai]] = 1; has_b2[ok2[bi]] = 1
        sep_grow[both] = np.linalg.norm(A2[both] - B2[both], axis=1) - d_ab[both]
    # scores, densities
    sc_p, sc_a, sc_b = S[pi], Sn[ai], Sn[bi]
    dens_p10 = np.asarray([len(x) - 1 for x in tree_t.query_ball_point(P, 10.0)])[pi]
    n_t1_12 = np.asarray([len(x) for x in nb])[pi]
    # orphan-ness: the nearest primary at t other than p, for a and for b
    dd, jj = tree_t.query(Pn, k=2)
    if dd.ndim == 1:                                   # a single primary at t
        dd, jj = np.c_[dd, np.full(len(dd), np.inf)], np.c_[jj, np.full(len(jj), -1)]
    da = np.where(jj[ai, 0] == pi, dd[ai, 1], dd[ai, 0])     # a's nearest primary at t other than p
    db = np.where(jj[bi, 0] == pi, dd[bi, 1], dd[bi, 0])
    feats = np.stack([d_pa, d_pb, d_ab, asym, mid_off, ang, np.abs(va[:, 0]), np.abs(vb[:, 0]), np.abs(Pa[:, 0] - Pb[:, 0]),
                      np.full(n, t / max(1, T - 1)), has_m, speed_m, cont_a, cont_b, cont_min, cont_mid, has_a2, has_b2, sep_grow,
                      sc_p, sc_a, sc_b, np.minimum(np.minimum(sc_p, sc_a), sc_b), dens_p10, n_t1_12, da, db, np.minimum(da, db), np.maximum(da, db)], 1)
    return np.stack([pi, ai, bi], 1), feats


def ceiling_reason(P, Pn, gts):
    """per GT division of a frame: why it has no candidate triple ('ok' when it can)"""
    out = []
    tp = cKDTree(P) if len(P) else None
    tn = cKDTree(Pn) if len(Pn) else None
    for m_pos, d1, d2 in gts:
        if tp is None or tn is None:
            out.append("no_primaries"); continue
        dm, jm = tp.query(m_pos); d1d, j1 = tn.query(d1); d2d, j2 = tn.query(d2)
        if dm > SNAP_UM:
            out.append("mother_unmatched"); continue
        if d1d > SNAP_UM or d2d > SNAP_UM:
            out.append("daughter_unmatched"); continue
        if j1 == j2:
            out.append("daughters_same_primary"); continue
        A, B, M = Pn[j1], Pn[j2], P[jm]
        if np.linalg.norm(A - M) > R_PARENT or np.linalg.norm(B - M) > R_PARENT:
            out.append("daughter_over_R_PARENT"); continue
        dab = np.linalg.norm(A - B)
        if dab > R_SISTER or dab < D_AB_MIN:
            out.append("sister_distance"); continue
        if np.linalg.norm(0.5 * (A + B) - M) > R_MID:
            out.append("midpoint_over_R_MID"); continue
        out.append("ok")
    return out


def label_triples(idx, P, Pn, gts):
    """positive where (p, a, b) match a GT division's (m, d1, d2) within SNAP_UM, daughters in either order"""
    y = np.zeros(len(idx), int)
    if not gts or not len(idx):
        return y, 0
    matched = 0
    tp = cKDTree(P); tn = cKDTree(Pn)
    for m_pos, d1, d2 in gts:
        dm, jm = tp.query(m_pos); d1d, j1 = tn.query(d1); d2d, j2 = tn.query(d2)
        if dm > SNAP_UM or d1d > SNAP_UM or d2d > SNAP_UM or j1 == j2:
            continue
        hit = (idx[:, 0] == jm) & (((idx[:, 1] == j1) & (idx[:, 2] == j2)) | ((idx[:, 1] == j2) & (idx[:, 2] == j1)))
        if hit.any():
            y[hit] = 1
            matched += 1
    return y, matched


# ---- raw-image features for the kept candidates ----------------------------------------------------
def _crop(frame, xyz_vox):
    hz = int(np.ceil(FI_HALF_UM / SCALE[0])); hy = int(np.ceil(FI_HALF_UM / SCALE[1]))
    z, y, x = (int(round(float(v))) for v in xyz_vox)
    z0, y0, x0 = max(0, z - hz), max(0, y - hy), max(0, x - hy)
    crop = frame[z0:z + hz + 1, y0:y + hy + 1, x0:x + hy + 1]
    if crop.size == 0:
        return None, None
    return _ndi.gaussian_filter(crop.astype(np.float32), FI_SIGMA_VOX), np.array([z0, y0, x0], float)


def node_img(frame, xyz_vox):
    """(peak, contrast, radius of gyration, half-max radius) of one node in one decoded frame"""
    sm, off = _crop(frame, xyz_vox)
    if sm is None:
        return (np.nan,) * 4
    c = np.asarray(xyz_vox, float) - off
    zz, yy, xx = np.indices(sm.shape)
    d = np.sqrt(((zz - c[0]) * SCALE[0]) ** 2 + ((yy - c[1]) * SCALE[1]) ** 2 + ((xx - c[2]) * SCALE[2]) ** 2)
    near = d <= FI_PEAK_UM
    if not near.any():
        return (np.nan,) * 4
    bg = float(np.median(sm))
    j = np.unravel_index(int(np.argmax(np.where(near, sm, -np.inf))), sm.shape)
    peak = float(sm[j])
    con = (peak - bg) / max(bg, 1e-6)
    dp = np.sqrt(((zz - j[0]) * SCALE[0]) ** 2 + ((yy - j[1]) * SCALE[1]) ** 2 + ((xx - j[2]) * SCALE[2]) ** 2)
    m = dp <= FI_SIZE_UM
    w = np.clip(sm[m] - bg, 0, None)
    rg = float(np.sqrt((w * dp[m] ** 2).sum() / w.sum())) if w.sum() > 0 else np.nan
    if peak <= bg:
        return peak, con, rg, np.nan
    lab_, _n = _ndi.label(sm >= bg + 0.5 * (peak - bg))
    vol = int((lab_ == lab_[j]).sum()) * float(np.prod(SCALE))
    return peak, float(con), rg, float((3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0))


def _hgb():
    # v2: early_stopping OFF (sklearn's 'auto' turns it on above 10k rows and, with 1 positive in 10,000, the 10% validation
    # split holds ~0 positives: every fold stopped at a near-constant model and the pooled out-of-fold AUC read 0.33);
    # positives weighted to the negatives' mass.
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, min_samples_leaf=20, early_stopping=False, random_state=0)


def _w(y):
    n1 = max(1, int(np.sum(y == 1)))
    return np.where(y == 1, (len(y) - n1) / n1, 1.0)


def fit_oof(X, y, folds, k, tag=""):
    """out-of-fold probabilities, HistGB, folds by movie; prints the per-fold AUC (the pooled one mixes fold scales)"""
    p = np.full(len(y), np.nan)
    per = []
    for f in range(k):
        tr, te = folds != f, folds == f
        if te.sum() == 0 or len(set(y[tr])) < 2:
            continue
        m = _hgb()
        m.fit(X[tr], y[tr], sample_weight=_w(y[tr]))
        p[te] = m.predict_proba(X[te])[:, 1]
        if len(set(y[te])) == 2:
            per.append((f, int(np.sum(y[te] == 1)), roc_auc_score(y[te], p[te])))
    if tag:
        print(f"    {tag}: per-fold AUC " + ", ".join(f"fold {f} ({n} pos) {a:.3f}" for f, n, a in per), flush=True)
    return p


def fit_full(X, y):
    m = _hgb()
    return m.fit(X, y, sample_weight=_w(y))


if __name__ == "__main__":
    t_all = time.time()
    FDET = _walk_files(_npz_prefix("fdet_F4hp_"))
    names = sorted(FDET)
    print(f"movies with F4hp caches: {len(names)}", flush=True)
    assert len(names) >= 190, "attach biohub-hparacache + biohub-hpara44"
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(names))
    FOLD = {nm: int(perm[i] % N_FOLDS) for i, nm in enumerate(names)}

    # ---- pass 1: every triple, the positives, a uniform negative sample --------------------------------
    rows, ylab, fold_of, meta = [], [], [], []
    stats = []
    CEIL = defaultdict(int)
    t_p1 = time.time()
    for k_, name in enumerate(names):
        t0 = time.time()
        det = np.load(FDET[name])
        T = max(int(k[1:]) for k in det.files if k.startswith("c")) + 1
        gt = read_geff(geff_path(ROOT, "train", name))
        gts = gt_triples(gt)
        n_gt = sum(len(v) for v in gts.values())
        PR = {t: primaries(det, t) for t in range(T)}
        um = {t: PR[t][0] * SCALE for t in range(T)}
        n_tri, n_pos, n_match, n_prim = 0, 0, 0, 0
        keep_X, keep_y, keep_meta = [], [], []
        for t in range(T - 1):
            P, S = um[t], PR[t][1]
            Pn, Sn = um[t + 1], PR[t + 1][1]
            n_prim += len(P)
            for r_ in ceiling_reason(P, Pn, gts.get(t, [])):
                CEIL[r_] += 1
            idx, X = triples_frame(P, S, Pn, Sn, um[t - 1] if t >= 1 else None, um[t + 2] if t + 2 < T else None, t, T)
            if not len(idx):
                continue
            y, matched = label_triples(idx, P, Pn, gts.get(t, []))
            n_tri += len(idx); n_pos += int(y.sum()); n_match += matched
            pos = np.where(y == 1)[0]
            neg = np.where(y == 0)[0]
            take = min(len(neg), max(1, int(round(N_NEG_PER_MOVIE / max(1, T - 1)))))   # ~N_NEG_PER_MOVIE negatives per movie
            sel = np.concatenate([pos, rng.choice(neg, take, replace=False)]) if len(neg) else pos
            keep_X.append(X[sel]); keep_y.append(y[sel]); keep_meta.append(np.c_[np.full(len(sel), t), idx[sel]])
        if keep_X:
            rows.append(np.concatenate(keep_X)); ylab.append(np.concatenate(keep_y)); fold_of.append(np.full(sum(len(x) for x in keep_y), FOLD[name]))
            meta.append([(name, int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in np.concatenate(keep_meta)])
        stats.append(dict(movie=name, T=T, prim_per_frame=n_prim / max(1, T - 1), triples_per_frame=n_tri / max(1, T - 1), gt_div=n_gt, gt_matched=n_match, pos=n_pos))
        print(f"  [{k_ + 1}/{len(names)}] {name}: primaries/frame {n_prim / max(1, T - 1):.0f}, triples/frame {n_tri / max(1, T - 1):.0f}, "
              f"GT divisions {n_gt}, in the candidate set {n_match}, {time.time() - t0:.0f}s", flush=True)
        if time.time() - t_all > BUDGET_S * 0.45:
            print(f"[budget] pass 1 stopped after {k_ + 1} movies", flush=True)
            names = names[:k_ + 1]
            break
    X1 = np.concatenate(rows); y1 = np.concatenate(ylab); f1 = np.concatenate(fold_of)
    ST = pd.DataFrame(stats)
    ST.to_csv(OUT / "divtrip_movies.csv", index=False)
    print(f"\n=== DIVTRIP pass 1 ({(time.time() - t_p1) / 60:.1f} min): {len(names)} movies, {ST.triples_per_frame.mean():.0f} triples per frame "
          f"(p10 / p90 over movies {ST.triples_per_frame.quantile(.1):.0f} / {ST.triples_per_frame.quantile(.9):.0f}), "
          f"GT divisions {int(ST.gt_div.sum())} of which {int(ST.gt_matched.sum())} have their triple in the candidate set (geometric ceiling "
          f"{ST.gt_matched.sum() / max(1, ST.gt_div.sum()):.3f}); sample {len(y1):,} rows, {int(y1.sum())} positives ===", flush=True)
    print("  geometric ceiling by reason: " + ", ".join(f"{k}: {v}" for k, v in sorted(CEIL.items(), key=lambda kv: -kv[1])), flush=True)
    pos_mask = y1 == 1
    for j, f in enumerate(GEO):
        v = X1[:, j]; m = np.isfinite(v)
        if m.sum() > 100 and len(set(y1[m])) == 2:
            a = roc_auc_score(y1[m], v[m])
            print(f"    {f:<10} AUC {a:.3f}   pos p25/50/75 {np.nanpercentile(v[pos_mask], 25):7.2f} {np.nanpercentile(v[pos_mask], 50):7.2f} {np.nanpercentile(v[pos_mask], 75):7.2f}"
                  f"   neg {np.nanpercentile(v[~pos_mask], 25):7.2f} {np.nanpercentile(v[~pos_mask], 50):7.2f} {np.nanpercentile(v[~pos_mask], 75):7.2f}", flush=True)
    pA = fit_oof(X1, y1, f1, N_FOLDS, tag="scorer A")
    okA = np.isfinite(pA)
    print(f"  scorer A (geometry + scores + context, out of fold by movie) on the sample: AUC {roc_auc_score(y1[okA], pA[okA]):.4f}", flush=True)
    MODELS_A = {f: fit_full(X1[f1 != f], y1[f1 != f]) for f in range(N_FOLDS)}

    # ---- pass 2: score every triple, keep the top TOPK per frame, image features, scorer B ---------------
    t_p2 = time.time()
    cand_rows = []
    rec = defaultdict(int)
    n_gt_all = 0
    for k_, name in enumerate(names):
        t0 = time.time()
        det = np.load(FDET[name])
        T = max(int(k[1:]) for k in det.files if k.startswith("c")) + 1
        gt = read_geff(geff_path(ROOT, "train", name))
        gts = gt_triples(gt)
        PR = {t: primaries(det, t) for t in range(T)}
        um = {t: PR[t][0] * SCALE for t in range(T)}
        mdl = MODELS_A[FOLD[name]]
        img = open_image(zarr_path(ROOT, "train", name))
        frames = {}

        def frame(t):
            if t not in frames:
                if len(frames) >= 4:
                    del frames[min(frames)]
                frames[t] = np.asarray(img[t])
            return frames[t]

        n_img = 0
        for t in range(T - 1):
            P, S = um[t], PR[t][1]
            Pn, Sn = um[t + 1], PR[t + 1][1]
            idx, X = triples_frame(P, S, Pn, Sn, um[t - 1] if t >= 1 else None, um[t + 2] if t + 2 < T else None, t, T)
            if not len(idx):
                continue
            y, matched = label_triples(idx, P, Pn, gts.get(t, []))
            n_gt_all += matched
            p = mdl.predict_proba(X)[:, 1]
            order = np.argsort(-p)
            for K in (1, 3, 5, 10, 20, 40):
                rec[K] += int(y[order[:K]].sum())
            keep = order[:TOPK]
            # image features per distinct node
            F_t, F_t1 = frame(t), frame(t + 1)
            F_tm = frame(t - 1) if t >= 1 else None
            cache = {}

            def nf(fr, coords_vox, key):
                if key not in cache:
                    cache[key] = node_img(fr, coords_vox)
                return cache[key]

            Pv, Pnv = PR[t][0], PR[t + 1][0]
            for r in keep:
                pi_, ai_, bi_ = idx[r]
                i_p, c_p, r_p, hm_p = nf(F_t, Pv[pi_], ("p", pi_))
                i_a, c_a, _r, hm_a = nf(F_t1, Pnv[ai_], ("n", ai_))
                i_b, c_b, _r, hm_b = nf(F_t1, Pnv[bi_], ("n", bi_))
                br_p = np.nan
                if F_tm is not None and X[r, GEO.index("has_m")] == 1:
                    # the mother's predecessor: nearest primary at t-1 (recomputed cheaply)
                    dm, jm = cKDTree(um[t - 1]).query(P[pi_])
                    if dm <= SNAP_UM:
                        i_pm = nf(F_tm, PR[t - 1][0][jm], ("m", jm))[0]
                        br_p = i_p / i_pm if np.isfinite(i_pm) and i_pm > 0 else np.nan
                okh = np.isfinite(hm_p) and np.isfinite(hm_a) and np.isfinite(hm_b) and hm_p > 0
                imgf = [i_p, c_p, r_p, hm_p, c_a, c_b, hm_a, hm_b,
                        (max(hm_a, hm_b) / hm_p) if okh else np.nan, ((hm_a + hm_b) / hm_p) if okh else np.nan,
                        ((i_a + i_b) / (2 * i_p)) if (np.isfinite(i_a) and np.isfinite(i_b) and np.isfinite(i_p) and i_p > 0) else np.nan, br_p]
                n_img += 3
                cand_rows.append([name, FOLD[name], t, int(pi_), int(ai_), int(bi_), int(y[r]), float(p[r])] + list(X[r]) + imgf)
        del img, frames
        print(f"  [{k_ + 1}/{len(names)}] {name}: kept {TOPK}/frame, image nodes {n_img}, {time.time() - t0:.0f}s", flush=True)
        if time.time() - t_all > BUDGET_S:
            print(f"[budget] pass 2 stopped after {k_ + 1} movies", flush=True)
            break
    C = pd.DataFrame(cand_rows, columns=["movie", "fold", "t", "p", "a", "b", "label", "p_A"] + GEO + IMG)
    C.to_csv(OUT / "divtrip_candidates.csv", index=False)
    print(f"\n=== DIVTRIP pass 2 ({(time.time() - t_p2) / 60:.1f} min): {C.movie.nunique()} movies, {len(C):,} candidates ({TOPK} per frame), "
          f"positives kept {int(C.label.sum())} of {n_gt_all} GT divisions in the candidate sets ===", flush=True)
    print("  recall of the GT divisions (with a matching triple) at the top-K per frame by scorer A: " +
          ", ".join(f"K={K}: {rec[K] / max(1, n_gt_all):.3f}" for K in (1, 3, 5, 10, 20, 40)), flush=True)
    yC = C.label.values; fC = C.fold.values
    for j, f in enumerate(IMG):
        v = C[f].values.astype(float); m = np.isfinite(v)
        if m.sum() > 100 and len(set(yC[m])) == 2:
            print(f"    {f:<13} AUC {roc_auc_score(yC[m], v[m]):.3f}   pos median {np.nanmedian(v[yC == 1]):8.3f}   neg median {np.nanmedian(v[yC == 0]):8.3f}", flush=True)
    XA = C[GEO].values.astype(float); XB = C[GEO + IMG].values.astype(float)
    pA2 = fit_oof(XA, yC, fC, N_FOLDS, tag="scorer A refit"); pB = fit_oof(XB, yC, fC, N_FOLDS, tag="scorer B")
    okA2, okB = np.isfinite(pA2), np.isfinite(pB)
    print(f"  on the kept candidates, out of fold by movie: scorer A (refit) AUC {roc_auc_score(yC[okA2], pA2[okA2]):.4f}; "
          f"scorer B (+ image) AUC {roc_auc_score(yC[okB], pB[okB]):.4f}", flush=True)
    n_frames = int(sum((ST.set_index('movie').loc[m_, 'T'] - 1) for m_ in C.movie.unique()))
    for nm, pp in (("A", pA2), ("B", pB)):
        ok = np.isfinite(pp)
        order = np.argsort(-pp[ok]); ys = yC[ok][order]
        cum = np.cumsum(ys)
        print(f"  scorer {nm}: candidates kept per frame to recover 50% / 70% / 80% / 90% of the kept positives: " +
              " / ".join(f"{(int(np.argmax(cum >= frac * cum[-1])) + 1) / max(1, n_frames):.2f}" for frac in (0.5, 0.7, 0.8, 0.9)) +
              f"   (precision at 70%: {0.7 * cum[-1] / max(1, int(np.argmax(cum >= 0.7 * cum[-1])) + 1):.3f})", flush=True)
    for thr in (0.5, 0.3, 0.2, 0.1, 0.05):
        ok = np.isfinite(pB)
        k_ = pB[ok] > thr
        print(f"    scorer B > {thr}: {int(k_.sum())} candidates ({k_.sum() / max(1, n_frames):.2f} per frame), positives {int(yC[ok][k_].sum())} of {int(yC[ok].sum())}, precision {yC[ok][k_].mean() if k_.sum() else 0:.3f}", flush=True)
    print(f"\nDIVTRIP DONE in {(time.time() - t_all) / 60:.1f} min", flush=True)
    print("DIVTRIP OK", flush=True)
