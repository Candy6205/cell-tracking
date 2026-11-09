"""offline: is the size gate too strict, and does it transfer across domains? Column-set variants of the fork gate on the
CR_NOCAP_DV150 arm's fork tables (6bba: forkssz_CR_NOCAP_DV150.csv; 44b6: forkssz_44b6img.csv), HistGB as fk44prop_sz,
cut p > 0.03. label 2 = TP, 1 = FP, 0 = uncounted."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

A = pd.read_csv("bench_csv/forksz/forkssz_CR_NOCAP_DV150.csv")
B = pd.read_csv("bench_csv/forksz/forkssz_44b6img.csv")
print("6bba forks", len(A), "counted", int((A.label >= 1).sum()), "TP", int((A.label == 2).sum()), "movies", A.movie.nunique())
print("44b6 forks", len(B), "counted", int((B.label >= 1).sum()), "TP", int((B.label == 2).sum()), "movies", B.movie.nunique(), "cols equal", list(A.columns) == list(B.columns))
F20 = ["src", "d_pc1", "d_pq", "d_s", "dv", "p_c1", "p_q", "pf_c1", "pf_q", "pr_q", "sc_p", "sc_c1", "sc_q", "len_back", "len_c1", "len_q", "dens10", "n_orph", "t_frac", "rank_gap"]
IMG11 = ["i_p", "c_p", "r_p", "br_p", "c_c1", "c_q", "r_c1", "r_q", "sym_i", "sym_r", "dau_ratio"]
SZ = ["r_hm_p", "r_hm_c1", "r_hm_q", "rhm_max", "rhm_sum"]
for T in (A, B):
    med = T.groupby("movie")["r_hm_p"].transform("median")
    for c in ("r_hm_p", "r_hm_c1", "r_hm_q"):
        T[c + "_n"] = T[c] / med
    T["dom"] = 0 if T is A else 1
SETS = {"31 (nc35)": F20 + IMG11, "36 (sz)": F20 + IMG11 + SZ, "33 ratios only": F20 + IMG11 + ["rhm_max", "rhm_sum"],
        "36 sizes normalised by movie": F20 + IMG11 + ["r_hm_p_n", "r_hm_c1_n", "r_hm_q_n", "rhm_max", "rhm_sum"],
        "34 abs sizes only": F20 + IMG11 + ["r_hm_p", "r_hm_c1", "r_hm_q"]}

print("\n--- domain shift of the size columns (median over counted forks): 6bba vs 44b6 ---")
for c in SZ + ["r_hm_p_n", "r_hm_c1_n"]:
    a, b = A.loc[A.label >= 1, c], B.loc[B.label >= 1, c]
    print(f"  {c:<10} 6bba TP {A.loc[A.label == 2, c].median():.3f} FP {A.loc[A.label == 1, c].median():.3f} | 44b6 TP {B.loc[B.label == 2, c].median():.3f} FP {B.loc[B.label == 1, c].median():.3f}")


def fit(X, y):
    m = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, min_samples_leaf=10, random_state=0)
    m.fit(X, y)
    return m


def evaluate(train, test, cols, oof_col=None, thr=0.03):
    """returns TP kept, TP total, FP kept, FP total, AUC on test's counted forks; oof_col -> out of fold within one table"""
    tr = train[train.label >= 1]
    te = test[test.label >= 1]
    if oof_col is None:
        m = fit(tr[cols].to_numpy(float), (tr.label == 2).astype(int).to_numpy())
        p = m.predict_proba(te[cols].to_numpy(float))[:, 1]
    else:
        p = np.full(len(te), np.nan)
        for f in sorted(te[oof_col].unique()):
            trf = tr[tr[oof_col] != f]
            m = fit(trf[cols].to_numpy(float), (trf.label == 2).astype(int).to_numpy())
            mask = (te[oof_col] == f).to_numpy()
            p[mask] = m.predict_proba(te.loc[mask, cols].to_numpy(float))[:, 1]
    y = (te.label == 2).to_numpy()
    keep = p > thr
    return int((keep & y).sum()), int(y.sum()), int((keep & ~y).sum()), int((~y).sum()), roc_auc_score(y, p)


AB = pd.concat([A, B], ignore_index=True)
AB["fold2"] = AB["fold"].astype(int)
print("\n--- gate variants at p > 0.03: TP kept / TP, FP kept / FP, AUC ---")
print(f"  {'columns':<30} {'6bba OOF by fold':>22} {'44b6 OOF by fold':>22} {'6bba->44b6':>22} {'44b6->6bba':>22} {'joint OOF: 6bba':>22} {'joint OOF: 44b6':>22}")
for name, cols in SETS.items():
    cells = []
    cells.append(evaluate(A, A, cols, oof_col="fold"))
    cells.append(evaluate(B, B, cols, oof_col="fold"))
    cells.append(evaluate(A, B, cols))
    cells.append(evaluate(B, A, cols))
    # joint: train on both domains, out of fold by the fold column, report each domain's test forks
    tr = AB[AB.label >= 1]
    for dom in (0, 1):
        te = tr[tr.dom == dom]
        p = np.full(len(te), np.nan)
        for f in sorted(te.fold2.unique()):
            trf = tr[tr.fold2 != f]
            m = fit(trf[cols].to_numpy(float), (trf.label == 2).astype(int).to_numpy())
            mask = (te.fold2 == f).to_numpy()
            p[mask] = m.predict_proba(te.loc[mask, cols].to_numpy(float))[:, 1]
        y = (te.label == 2).to_numpy(); keep = p > 0.03
        cells.append((int((keep & y).sum()), int(y.sum()), int((keep & ~y).sum()), int((~y).sum()), roc_auc_score(y, p)))
    print(f"  {name:<30} " + " ".join(f"{c[0]:>3}/{c[1]:<3} {c[2]:>3}/{c[3]:<3} {c[4]:.3f}" for c in cells))
print("  reading: a column set is 'too strict' where it keeps fewer TP than the 31-column gate at the same cut; the cross-domain columns are the transfer test (a new embryo).")
