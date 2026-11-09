"""offline approximation of forkprop's 6bba gate cross for column sets of the fork gate: the gate fitted as
_gate_and_prune does (2 folds by the movie fold on the arm's counted 6bba forks + every counted 44b6 fork in both
training folds = the joint gate), applied at p > 0.03 to the arm's 6bba forks; per movie TP kept / FP kept by the fork
table's judge labels; div_J pooled with FN = GT - TP kept (GT per movie from the REF rows); score = adj_J (REF rows,
constant across column sets) + 0.1 div_J. Paired bootstrap over the 128 movies. APPROXIMATE: the judge's FP forks
(659) are not the scorer's division_fp (26-36 after the gate), so the absolute numbers are not the cross's; the
paired differences between column sets are the reading."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

A = pd.read_csv("bench_csv/forksz/forkssz_CR_NOCAP_DV150.csv")
B = pd.read_csv("bench_csv/forksz/forkssz_44b6img.csv")
R = pd.read_csv("bench_csv/hypz4/hypz4_REF_F4HP.csv").set_index("dataset")
F20 = ["src", "d_pc1", "d_pq", "d_s", "dv", "p_c1", "p_q", "pf_c1", "pf_q", "pr_q", "sc_p", "sc_c1", "sc_q", "len_back", "len_c1", "len_q", "dens10", "n_orph", "t_frac", "rank_gap"]
IMG11 = ["i_p", "c_p", "r_p", "br_p", "c_c1", "c_q", "r_c1", "r_q", "sym_i", "sym_r", "dau_ratio"]
SETS = {"31": F20 + IMG11, "36": F20 + IMG11 + ["r_hm_p", "r_hm_c1", "r_hm_q", "rhm_max", "rhm_sum"],
        "33": F20 + IMG11 + ["rhm_max", "rhm_sum"]}
cntA, cntB = A.label >= 1, B.label >= 1
movies = sorted(A.movie.unique())
assert all(m in R.index for m in movies)
GT = np.array([R.loc[m, "division_tp"] + R.loc[m, "division_fn"] for m in movies], float)
w = np.array([R.loc[m, "edge_tp"] + R.loc[m, "edge_fp"] + R.loc[m, "edge_fn"] for m in movies], float)
adj = np.array([R.loc[m, "adj_edge_jaccard"] for m in movies], float)
midx = {m: i for i, m in enumerate(movies)}
KEPT = {}
for name, cols in SETS.items():
    p = np.full(len(A), np.nan)
    for k in (0, 1):
        tr = (A.fold != k) & cntA
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
    fn = GT - tp
    dj = tp.sum() / (tp.sum() + fp.sum() + fn.sum())
    aj = (adj * w).sum() / w.sum()
    print(f"  {name} columns: TP kept {int(tp.sum())} of {int((A.label == 2).sum())}, judge-FP kept {int(fp.sum())} of {int((A.label == 1).sum())}, "
          f"approx div_J {dj:.4f}, approx score {aj + 0.1 * dj:.4f}")
rng = np.random.default_rng(9090)
draws = [rng.choice(len(movies), len(movies), replace=True) for _ in range(2000)]


def sc(name, d):
    tp, fp = KEPT[name]
    t, f, g = tp[d].sum(), fp[d].sum(), GT[d].sum()
    return (adj[d] * w[d]).sum() / w[d].sum() + 0.1 * t / max(1.0, t + f + (g - t))


print("\n  paired bootstrap over the 128 6bba movies, 2,000 draws, approximate score (see the header):")
for a, b in (("36", "31"), ("33", "31"), ("33", "36")):
    dd = np.array([sc(a, d) - sc(b, d) for d in draws])
    lo, hi = np.quantile(dd, .05), np.quantile(dd, .95)
    print(f"    {a} vs {b}: {dd.mean():+.4f}  90% [{lo:+.4f}, {hi:+.4f}]  {'YES' if lo > 0 or hi < 0 else 'no'}")
