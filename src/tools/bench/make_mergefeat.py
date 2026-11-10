"""maskfeat.py -> mergefeat.py: the merge classifier over EVERY parent candidate.

WHY. The base rate of the merge question was mis-stated. Among near primary
pairs whose two ends are both annotated (divmiss_prims, 128 6bba movies, 9.5
um): 1,572 are one cell (both within 5 um of the SAME GT node) and 615 two
cells; corrected for annotation sparsity (a one-cell pair needs one annotated
cell, a two-cell pair two) the population is about one one-cell pair per four
two-cell pairs, and within 7 um about one per one. The loss channel of hmerge
was not the labelled two-cell pairs (615, of which 4% pass saddle 0.9) but the
56,301 pairs with ONE annotated end: merging one of those displaces an
annotated node and hands its track to a coin flip; at saddle >= 0.9 the rule
fires on 14% of random pairs = ~7,900 such merges against ~680 swap fixes,
which reproduces hmbench's -0.0082. The stacked classifier (mask + heatmap +
temporal, maskfeat / swapsep 2026-09-15) fires on 0.2-0.5% of random pairs at
50-60% swap recall, 30-70x fewer.

WHAT THIS KERNEL DOES. For every parent hypcache wrote (a pair of primaries
within 9.5 um with heatmap saddle >= 0.8) in the 128 6bba movies with pair
distance <= MAX_D_UM: the raw-image mask features (maskfeat v1's crop and
background, thresholds 0.5 / 0.7 / 0.9), temporal features from the children
caches (persistence of the pair at t +- 1, 2, spacing jitter, the two scores,
density), the saddle, the distance; the GT label of the pair (one cell / two
cells / unlabelled, both ends within 5 um of GT nodes); two fold HistGB models
by movie fold, out of fold; p for every parent. Prints per-feature AUCs, the
out-of-fold AUC, and at thresholds the firing rate on one-end-annotated and
unannotated pairs against the one-cell recall, with the wrong-per-fix estimate.
Writes mergefeat_<movie>.npz (t, q = parent index in hypcache's p{t}, p, lab)
and mergefeat_pairs.csv (labelled rows).

Attach: faustine649/cell-tracking, biohub-hypcache, the competition. CPU,
~3-4 h (v1 maskfeat: 87 ms per site incl. the crop read).
"""
import os

src = open("maskfeat.py", encoding="utf-8").read()
i = src.index("# ---- the features ----")
j = src.index('if __name__ == "__main__" and SITES_B64')
feat_block = src[i:j]
pre = src[:src.index("SITES_B64 = ")]
pre = pre.replace('SCRIPT_VERSION = "2026-09-15b-maskfeat"', 'SCRIPT_VERSION = "2026-09-15a-mergefeat"')
pre = pre.replace("HALF_UM = 6.0                      # v2: crop half-size around the midpoint (v1 10 um: masks spanned several cells)",
                  "HALF_UM = 10.0                     # maskfeat v1's crop (its stack beat v2's)")
pre = pre.replace("FRACS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)", "FRACS = (0.5, 0.7, 0.9)")
feat_block = feat_block.replace("    I_bg = float(np.percentile(sm, 50))     # v2: the crop median (v1 p20 set the masks too low)",
                                "    I_bg = float(np.percentile(sm, 20))     # maskfeat v1's background")
assert 'FRACS = (0.5, 0.7, 0.9)' in pre and "np.percentile(sm, 20)" in feat_block
i0 = pre.index('"""'); i1 = pre.index('"""', i0 + 3) + 3
pre = pre[:i0] + '"""mergefeat.py -- the merge classifier over every hypcache parent (see tools/bench/make_mergefeat.py)."""' + pre[i1:]

MAIN = r'''
# ---- the run --------------------------------------------------------------------
import glob                                                     # noqa: E402
from collections import defaultdict                             # noqa: E402
from scipy.spatial import cKDTree                               # noqa: E402

MAX_D_UM = 8.5          # two-cell pairs sit at 8-9.5 um (median 8.7); one-cell pairs 90% below 8.2
MATCH_UM = 5.0
UNET_THRESH = 0.15
BUDGET_S = 10.5 * 3600

from viz_cells import find_root, geff_path, read_geff, open_image  # noqa: E402
ROOT = find_root(None)
TRAIN_DIR = pathlib.Path(ROOT) / "train"
HYP = {}
for _d in pathlib.Path("/kaggle/input").glob("*/**/fdet_F4hyp_*.npz"):
    HYP[_d.name[len("fdet_F4hyp_"):-len(".npz")]] = _d
SIX = sorted(n for n in HYP if n.startswith("6bba_"))
assert len(SIX) == 128, f"{len(SIX)} 6bba hypcache caches (attach biohub-hypcache)"
FOLD_OF = {n: k % 2 for k, n in enumerate(SIX)}
print(f"hypcache caches: {len(HYP)}; 6bba movies {len(SIX)}", flush=True)
TEMP_NAMES = ["d_ab", "saddle", "s_a", "s_b", "s_min", "s_ratio", "dens10", "d_third", "persist", "both_frames",
              "one_only_frames", "pair_jitter", "t_frac"]
ALL_NAMES = TEMP_NAMES + FEAT_NAMES


def gt_of(name):
    gt = read_geff(geff_path(ROOT, "train", name))
    g_t = np.asarray(gt["t"], int)
    g_pos = np.stack([np.asarray(gt[c], float) for c in ("z", "y", "x")], axis=1) * SCALE
    trees = {}
    for tt in np.unique(g_t):
        rows = np.where(g_t == tt)[0]
        trees[int(tt)] = (rows, cKDTree(g_pos[rows]))
    return trees


def gt_match(trees, t, P_um):
    out = np.full(len(P_um), -1, np.int64)
    if t in trees and len(P_um):
        rows, tree = trees[t]
        dd, gi = tree.query(P_um)
        out[dd <= MATCH_UM] = rows[gi[dd <= MATCH_UM]]
    return out


OUT = pathlib.Path("/kaggle/working")
ROWS = {}          # name -> dict(t, q, X, lab)
STAT = defaultdict(int)
t_all = time.time()
hz = int(np.ceil(HALF_UM / SCALE[0]))
hy = int(np.ceil(HALF_UM / SCALE[1]))
for k_, name in enumerate(SIX):
    if time.time() - t_all > BUDGET_S:
        print(f"BUDGET: stopping before {name}", flush=True)
        break
    tm = time.time()
    z = np.load(HYP[name])
    T = max(int(k[1:]) for k in z.files if k.startswith("c")) + 1
    ch, sc, par, sad = [], [], [], []
    for t in range(T):
        c = np.asarray(z[f"c{t}"], float)
        s = np.asarray(z[f"s{t}"], float)
        p = np.asarray(z[f"p{t}"], np.int64).reshape(-1, 2) if f"p{t}" in z.files else np.zeros((0, 2), np.int64)
        w = np.asarray(z[f"w{t}"], float) if f"w{t}" in z.files else np.zeros(0)
        prim = c[s >= UNET_THRESH]
        sp = s[s >= UNET_THRESH]
        nch = len(prim) - len(p)
        ch.append(prim[:nch])
        sc.append(sp[:nch])
        par.append(p)
        sad.append(w)
    trees_gt = gt_of(name)
    ch_um = [c * SCALE for c in ch]
    ch_tree = [cKDTree(c) if len(c) else None for c in ch_um]
    gm = [gt_match(trees_gt, t, ch_um[t]) for t in range(T)]
    img = open_image(TRAIN_DIR / f"{name}.zarr")
    T_, Q_, X_, L_ = [], [], [], []
    for t in range(T):
        p = par[t]
        if len(p) == 0:
            continue
        C, S = ch[t], sc[t]
        d_all = np.linalg.norm((C[p[:, 0]] - C[p[:, 1]]) * SCALE, axis=1)
        keep = np.where(d_all <= MAX_D_UM)[0]
        STAT["parents"] += len(p)
        STAT["parents_kept"] += len(keep)
        for q in keep:
            a_i, b_i = int(p[q, 0]), int(p[q, 1])
            a, b = C[a_i], C[b_i]
            au, bu = ch_um[t][a_i], ch_um[t][b_i]
            d0 = float(d_all[q])
            f = dict(d_ab=d0, saddle=float(sad[t][q]) if q < len(sad[t]) else np.nan, s_a=float(S[a_i]), s_b=float(S[b_i]))
            f["s_min"] = min(f["s_a"], f["s_b"])
            f["s_ratio"] = f["s_min"] / max(1e-6, max(f["s_a"], f["s_b"]))
            mid = 0.5 * (au + bu)
            tree = ch_tree[t]
            f["dens10"] = len(tree.query_ball_point(mid, 10.0)) - 2
            kk = min(4, len(C))
            dd, ii = tree.query(mid, k=kk)
            others = [d for d, i in zip(np.atleast_1d(dd), np.atleast_1d(ii)) if i not in (a_i, b_i)]
            f["d_third"] = others[0] if others else 20.0
            pers = 0
            n_av = 0
            both = []
            one = 0
            for dt in (-2, -1, 1, 2):
                if not (0 <= t + dt < T) or ch_tree[t + dt] is None:
                    continue
                n_av += 1
                tr2 = ch_tree[t + dt]
                d1, i1 = tr2.query(au)
                d2, i2 = tr2.query(bu)
                pa, pb = d1 <= 4.0, d2 <= 4.0
                if pa and pb and i1 != i2:
                    dp = float(np.linalg.norm(ch_um[t + dt][i1] - ch_um[t + dt][i2]))
                    both.append(abs(dp - d0))
                    pers += abs(dp - d0) < 2.0
                elif pa or pb:
                    one += 1
            f["persist"] = pers / max(1, n_av)
            f["both_frames"] = len(both) / max(1, n_av)
            f["one_only_frames"] = one / max(1, n_av)
            f["pair_jitter"] = float(np.mean(both)) if both else 5.0
            f["t_frac"] = t / max(1, T - 1)
            m = 0.5 * (a + b)
            z0, y0, x0 = (max(0, int(round(m[0])) - hz), max(0, int(round(m[1])) - hy), max(0, int(round(m[2])) - hy))
            z1, y1, x1 = (int(round(m[0])) + hz + 1, int(round(m[1])) + hy + 1, int(round(m[2])) + hy + 1)
            crop = np.asarray(img[t, z0:z1, y0:y1, x0:x1])
            off = np.array([z0, y0, x0], float)
            vals = site_features(crop, a - off, b - off)
            ga, gb = int(gm[t][a_i]), int(gm[t][b_i])
            lab = 1 if (ga >= 0 and ga == gb) else 0 if (ga >= 0 and gb >= 0) else (2 if (ga >= 0 or gb >= 0) else -1)
            # lab: 1 one cell, 0 two cells, 2 one end annotated (the other >= 5 um from any GT node), -1 unannotated
            T_.append(t)
            Q_.append(int(q))
            X_.append([f[k] for k in TEMP_NAMES] + list(vals))
            L_.append(lab)
    X = np.asarray(X_, np.float32).reshape(-1, len(ALL_NAMES))
    L = np.asarray(L_, np.int8)
    ROWS[name] = dict(t=np.asarray(T_, np.int32), q=np.asarray(Q_, np.int32), X=X, lab=L)
    for v_, nm_ in ((1, "one_cell"), (0, "two_cells"), (2, "one_annotated"), (-1, "unannotated")):
        STAT[nm_] += int((L == v_).sum())
    print(f"[{k_ + 1}/{len(SIX)}] {name} fold{FOLD_OF[name]}: parents {sum(len(p) for p in par):,} kept {len(L):,} "
          f"(one cell {int((L == 1).sum())}, two cells {int((L == 0).sum())}, one annotated {int((L == 2).sum())}) "
          f"{time.time() - tm:.0f}s (total {(time.time() - t_all) / 60:.1f} min)", flush=True)
    del img, z
print(f"  parents {STAT['parents']:,}, kept (d <= {MAX_D_UM} um) {STAT['parents_kept']:,}: one cell {STAT['one_cell']:,}, "
      f"two cells {STAT['two_cells']:,}, one end annotated {STAT['one_annotated']:,}, unannotated {STAT['unannotated']:,}", flush=True)

# ---- the classifier: two fold models by movie, out of fold -----------------------------
from sklearn.metrics import roc_auc_score                       # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier     # noqa: E402
names = sorted(ROWS)
Xall = np.concatenate([ROWS[n]["X"] for n in names])
Lall = np.concatenate([ROWS[n]["lab"] for n in names])
print(f"\n=== per-feature AUC one-cell vs two-cell ({int((Lall == 1).sum())} vs {int((Lall == 0).sum())} labelled parents) ===", flush=True)
for k, nm_ in enumerate(ALL_NAMES):
    a_ = Xall[Lall == 1, k]
    b_ = Xall[Lall == 0, k]
    ok_a, ok_b = np.isfinite(a_), np.isfinite(b_)
    if ok_a.sum() < 30 or ok_b.sum() < 30:
        continue
    auc = roc_auc_score(np.r_[np.ones(ok_a.sum()), np.zeros(ok_b.sum())], np.r_[a_[ok_a], b_[ok_b]])
    print(f"  {nm_:<16} one-cell med {np.nanmedian(a_):>9.3f}  two-cell med {np.nanmedian(b_):>9.3f}  AUC {max(auc, 1 - auc):.3f}", flush=True)
MODELS = {}
for k in (0, 1):
    tr = [n for n in names if FOLD_OF[n] != k]
    Xt = np.concatenate([ROWS[n]["X"][np.isin(ROWS[n]["lab"], (0, 1))] for n in tr])
    yt = np.concatenate([ROWS[n]["lab"][np.isin(ROWS[n]["lab"], (0, 1))] for n in tr]).astype(int)
    m = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    m.fit(np.where(np.isfinite(Xt), Xt, -1.0), yt)
    MODELS[k] = m
    for n in names:
        if FOLD_OF[n] == k:
            X_ = ROWS[n]["X"]
            ROWS[n]["p"] = m.predict_proba(np.where(np.isfinite(X_), X_, -1.0))[:, 1].astype(np.float32) if len(X_) else np.zeros(0, np.float32)
    print(f"  fold{k}: trained on {len(tr)} movies / {len(yt):,} labelled rows ({yt.mean():.3f} one-cell)", flush=True)
P = np.concatenate([ROWS[n]["p"] for n in names])
lab01 = np.isin(Lall, (0, 1))
print(f"  out-of-fold AUC one-cell vs two-cell: {roc_auc_score(Lall[lab01], P[lab01]):.4f}", flush=True)
n1 = int((Lall == 1).sum())
print(f"\n  {'thr':>5} {'recall 1cell':>13} {'fires 2cell':>12} {'fires 1-ann':>12} {'fires unann':>12} {'fires all':>10} "
      f"{'est. wrong / fix':>17}", flush=True)
print("  (wrong / fix = merges on one-annotated pairs + on two-cell pairs, per one-cell pair merged; every merge on a "
      "one-annotated pair displaces an annotated node; the unannotated column is what the rule fires on without a judge)", flush=True)
for thr in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98):
    f1 = (P[Lall == 1] >= thr).mean()
    f0 = (P[Lall == 0] >= thr).mean() if (Lall == 0).any() else float("nan")
    f2 = (P[Lall == 2] >= thr).mean() if (Lall == 2).any() else float("nan")
    fu = (P[Lall == -1] >= thr).mean() if (Lall == -1).any() else float("nan")
    fa = (P >= thr).mean()
    wrong = (P[Lall == 2] >= thr).sum() + (P[Lall == 0] >= thr).sum()
    print(f"  {thr:>5.2f} {f1:>13.3f} {f0:>12.3f} {f2:>12.3f} {fu:>12.3f} {fa:>10.3f} {wrong / max(1, (P[Lall == 1] >= thr).sum()):>17.2f}", flush=True)
for n in names:
    np.savez_compressed(OUT / f"mergefeat_{n}.npz", t=ROWS[n]["t"], q=ROWS[n]["q"], p=ROWS[n]["p"], lab=ROWS[n]["lab"])
with open(OUT / "mergefeat_pairs.csv", "w", encoding="utf-8", newline="\n") as fh:
    fh.write("dataset,fold,t,q,lab,p," + ",".join(ALL_NAMES) + "\n")
    for n in names:
        r = ROWS[n]
        for k in np.where(r["lab"] >= 0)[0]:
            fh.write(f"{n},{FOLD_OF[n]},{int(r['t'][k])},{int(r['q'][k])},{int(r['lab'][k])},{float(r['p'][k]):.5f},"
                     + ",".join(f"{float(v):.5g}" for v in r["X"][k]) + "\n")
import pickle                                                    # noqa: E402
with open(OUT / "merge_rank.pkl", "wb") as fh:
    pickle.dump({"models": MODELS, "feature_names": ALL_NAMES, "version": SCRIPT_VERSION, "max_d_um": MAX_D_UM}, fh)
print(f"\nMERGEFEAT DONE in {(time.time() - t_all) / 60:.1f} min; {len(names)} movies", flush=True)
print("MERGEFEAT OK" if len(names) == len(SIX) else f"MERGEFEAT INCOMPLETE {len(names)}/{len(SIX)}", flush=True)
'''
out_src = pre + feat_block + MAIN
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "mergefeat.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(out_src)
print("written mergefeat.py", len(out_src))
