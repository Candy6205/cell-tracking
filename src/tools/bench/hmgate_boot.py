"""offline: the F4 heatmap's size measures (hmsize_forks.csv) in the fork gate, joined on (movie, p, q) with forkprop_sz2's
arm table; the gate fitted the way the bench cross fits it (2 folds by the movie fold on the arm's counted 6bba forks, every
counted 44b6 fork in both training folds with NaN heatmap columns), p > 0.03; per-movie kept counts, approximate score
(adj_J from the REF rows, div_J with FN = GT - TP kept), paired bootstrap over the 128 movies. The judge's FP forks are not
the scorer's division_fp, so the paired differences are the reading, not the levels."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

A = pd.read_csv("bench_csv/forksz2/forkssz2_CR_NOCAP_DV150.csv")
H = pd.read_csv("bench_csv/hmsize/hmsize_forks.csv")
H = H[H.table == "CR_NOCAP_DV150"].drop(columns=["table", "t", "label"])
B = pd.read_csv("bench_csv/forksz/forkssz_44b6img.csv")
R = pd.read_csv("bench_csv/hypz4/hypz4_REF_F4HP.csv").set_index("dataset")
HM = [c for c in H.columns if c not in ("movie", "p", "q")]
A = A.merge(H, on=["movie", "p", "q"], how="left")
print(f"arm forks {len(A)}, with heatmap columns {int(A[HM[0]].notna().sum())}; counted {int((A.label >= 1).sum())}, TP {int((A.label == 2).sum())}")
F20 = ["src", "d_pc1", "d_pq", "d_s", "dv", "p_c1", "p_q", "pf_c1", "pf_q", "pr_q", "sc_p", "sc_c1", "sc_q", "len_back", "len_c1", "len_q", "dens10", "n_orph", "t_frac", "rank_gap"]
IMG11 = ["i_p", "c_p", "r_p", "br_p", "c_c1", "c_q", "r_c1", "r_q", "sym_i", "sym_r", "dau_ratio"]
SZ = ["r_hm_p", "r_hm_c1", "r_hm_q", "rhm_max", "rhm_sum"]
HM_ABS = [c for c in HM if c.endswith(("_p", "_c1", "_q"))]
HM_RAT = [c for c in HM if c.endswith(("_max", "_sum"))]
SETS = {"img31": F20 + IMG11, "img36 (size gate)": F20 + IMG11 + SZ,
        "img31 + hm all (31)": F20 + IMG11 + HM, "img31 + hm abs (21)": F20 + IMG11 + HM_ABS, "img31 + hm ratios (10)": F20 + IMG11 + HM_RAT,
        "img31 + hmom, hr50 (10)": F20 + IMG11 + [c for c in HM if c.startswith(("hmom", "hr50"))],
        "img31 + hmass (10)": F20 + IMG11 + [c for c in HM if c.startswith("hmass")],
        "img36 + hm all (36)": F20 + IMG11 + SZ + HM}
for c in HM:
    B[c] = np.nan
print("\n--- single heatmap columns on the arm's counted forks: AUC TP vs FP (below 0.5 = TP smaller), TP / FP medians ---")
cnt = A.label >= 1
for c in HM:
    v = A.loc[cnt, c]; y = (A.loc[cnt, "label"] == 2).astype(int)
    ok = v.notna()
    if ok.sum() and y[ok].nunique() == 2:
        print(f"  {c:<12} AUC {roc_auc_score(y[ok], v[ok]):.3f}  TP {A.loc[cnt & (A.label == 2), c].median():.3g}  FP {A.loc[cnt & (A.label == 1), c].median():.3g}")
movies = sorted(A.movie.unique())
GT = np.array([R.loc[m, "division_tp"] + R.loc[m, "division_fn"] for m in movies], float)
w = np.array([R.loc[m, "edge_tp"] + R.loc[m, "edge_fp"] + R.loc[m, "edge_fn"] for m in movies], float)
adj = np.array([R.loc[m, "adj_edge_jaccard"] for m in movies], float)
midx = {m: i for i, m in enumerate(movies)}
cntB = B.label >= 1
KEPT, AUC = {}, {}
for name, cols in SETS.items():
    p = np.full(len(A), np.nan)
    for k in (0, 1):
        tr = (A.fold != k) & cnt
        Xtr = np.vstack([A.loc[tr, cols].to_numpy(float), B.loc[cntB, cols].to_numpy(float)])
        ytr = np.concatenate([(A.loc[tr, "label"] == 2).astype(int).to_numpy(), (B.loc[cntB, "label"] == 2).astype(int).to_numpy()])
        m = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, min_samples_leaf=10, random_state=0).fit(Xtr, ytr)
        te = (A.fold == k).to_numpy()
        p[te] = m.predict_proba(A.loc[te, cols].to_numpy(float))[:, 1]
    keep = p > 0.03
    tp = np.zeros(len(movies)); fp = np.zeros(len(movies))
    for mv, lab, kp in zip(A.movie, A.label, keep):
        if kp and lab == 2: tp[midx[mv]] += 1
        if kp and lab == 1: fp[midx[mv]] += 1
    KEPT[name] = (tp, fp)
    AUC[name] = roc_auc_score((A.loc[cnt, "label"] == 2).astype(int), p[cnt.to_numpy()])
    fn = GT - tp
    dj = tp.sum() / (tp.sum() + fp.sum() + fn.sum())
    print(f"  {name:<28} TP kept {int(tp.sum())}/66, judge-FP kept {int(fp.sum())}/659, out-of-fold AUC {AUC[name]:.3f}, approx div_J {dj:.4f}")
rng = np.random.default_rng(9090)
draws = [rng.choice(len(movies), len(movies), replace=True) for _ in range(2000)]


def sc(name, d):
    tp, fp = KEPT[name]
    t, f, g = tp[d].sum(), fp[d].sum(), GT[d].sum()
    return (adj[d] * w[d]).sum() / w[d].sum() + 0.1 * t / max(1.0, t + f + (g - t))


print("\n  paired bootstrap over the 128 movies, 2,000 draws, approximate score, vs img31 and vs img36:")
for name in SETS:
    if name == "img31":
        continue
    d1 = np.array([sc(name, d) - sc("img31", d) for d in draws]); d2 = np.array([sc(name, d) - sc("img36 (size gate)", d) for d in draws])
    f = lambda dd: f"{dd.mean():+.4f} [{np.quantile(dd, .05):+.4f}, {np.quantile(dd, .95):+.4f}]{' YES' if (np.quantile(dd, .05) > 0 or np.quantile(dd, .95) < 0) else ''}"
    print(f"    {name:<28} vs img31 {f(d1)}   vs img36 {f(d2)}")
