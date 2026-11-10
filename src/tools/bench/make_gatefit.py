"""gatefit.py: fit the JOINT IMAGE fork gate on Kaggle's sklearn from the two fork tables.

WHY (2026-09-17). forkimg (128 6bba) and fk44img (71 44b6) wrote every fork of the production
graphs with the 20 gate features, the 8 extra ones and the 11 image features (contrast at the
mother and both daughters is the signal: AUC 0.83-0.84 each). forkimg's own pickle block died
(a shadowed name) and a pickle must be written by the sklearn the submission kernel runs, so
this small CPU kernel does the fit: HistGradientBoosting (100 it, lr 0.05, min_samples_leaf 10,
random_state 0), 20 + 11 columns, every COUNTED fork of both domains (TP = 1, FP = 0), the
same recipe as fork_gate_joint.pkl. It also prints, per domain, the out-of-fold curves (2 folds
by the tables' own fold column) of the gate with and without the image columns, so the 44b6
read (26 GT divisions) is on record next to 6bba's.

Attach: biohub-forkimg (forks_dw55.csv), biohub-fk44img (forks_44b6img.csv). CPU, 1 min.
"""
import os

SRC = r'''"""gatefit: the joint image fork gate, fitted on Kaggle (see tools/bench/make_gatefit.py)."""
import glob
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

SCRIPT_VERSION = "2026-09-17a-gatefit"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)
FEAT20 = ["src", "d_pc1", "d_pq", "d_s", "dv", "p_c1", "p_q", "pf_c1", "pf_q",
          "pr_q", "sc_p", "sc_c1", "sc_q", "len_back", "len_c1", "len_q",
          "dens10", "n_orph", "t_frac", "rank_gap"]        # divbench3's order, = submit_hp's GATE_FEATS
FEAT_IMG = ["i_p", "c_p", "r_p", "br_p", "c_c1", "c_q", "r_c1", "r_q", "sym_i", "sym_r", "dau_ratio"]
N_GT = {"6bba": 125, "44b6": 26}


def _find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert len(hits) >= 1, f"{name} not found under /kaggle/input"
    return hits[0]


T6 = pd.read_csv(_find("forks_dw55.csv"))
T4 = pd.read_csv(_find("forks_44b6img.csv"))
for nm, t_ in (("6bba", T6), ("44b6", T4)):
    miss = [c for c in FEAT20 + FEAT_IMG if c not in t_.columns]
    assert not miss, (nm, miss)
    print(f"  {nm}: {len(t_):,} forks, counted {(t_.label >= 1).sum():,}, TP {(t_.label == 2).sum()}, "
          f"image NaN rows {t_.c_p.isna().sum():,}", flush=True)
T6["dom"] = "6bba"
T4["dom"] = "44b6"
ALL = pd.concat([T6, T4], ignore_index=True)
CNT = ALL[ALL.label >= 1].reset_index(drop=True)
y = (CNT.label == 2).astype(int).values


def _fit(X, y_):
    m = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, min_samples_leaf=10, random_state=0)
    m.fit(X, y_)
    return m


def _curve(dom, cols, tag):
    """2-fold out-of-fold by the table's fold column, both domains in the training set; curve on `dom`."""
    p = np.full(len(CNT), np.nan)
    for k in (0, 1):
        tr = (CNT.fold.values != k)
        te = ~tr
        m = _fit(CNT.loc[tr, cols].values, y[tr])
        p[te] = m.predict_proba(CNT.loc[te, cols].values)[:, 1]
    d = (CNT.dom.values == dom)
    auc = roc_auc_score(y[d], p[d]) if len(set(y[d])) == 2 else float("nan")
    print(f"\n  {tag} | {dom}: counted {int(d.sum())}, out-of-fold AUC {auc:.4f}", flush=True)
    print(f"  {'cut':<16} {'TP':>4} {'FP':>5} {'FN':>4} {'div_J':>7}", flush=True)
    for thr in (None, 0.01, 0.02, 0.03, 0.05, 0.1):
        keep = d if thr is None else (d & (p > thr))
        tp = int((y[keep] == 1).sum())
        fp = int((y[keep] == 0).sum())
        fn = N_GT[dom] - tp
        print(f"  {'none' if thr is None else f'p > {thr}':<16} {tp:>4} {fp:>5} {fn:>4} {tp / max(1, tp + fp + fn):>7.4f}", flush=True)


for dom in ("6bba", "44b6"):
    _curve(dom, FEAT20, "20 features (the shipped recipe)")
    _curve(dom, FEAT20 + FEAT_IMG, "20 + 11 image")

print("\n=== the joint image gate: every counted fork of both domains ===", flush=True)
cols = FEAT20 + FEAT_IMG
m_full = _fit(CNT[cols].values, y)
p_all = m_full.predict_proba(ALL[cols].values)[:, 1]
Q = {f: float(np.quantile(p_all, f)) for f in (0.5, 0.6, 0.7, 0.8, 0.9)}
n_train = {"6bba": int((CNT.dom == "6bba").sum()), "44b6": int((CNT.dom == "44b6").sum()), "tp": int(y.sum())}
blob = {"model": m_full, "feature_names": cols, "version": SCRIPT_VERSION,
        "trained_on": "counted F4HP forks, 128 6bba (forkimg) + 71 44b6 (fk44img), 20 gate + 11 image features, TP=1 FP=0",
        "n_train": n_train, "p_quantiles_all_forks": Q, "detector": "F4",
        "image": dict(half_um=8.0, peak_um=2.0, size_um=3.5, sigma_vox=(0.6, 1.5, 1.5)),
        "ilp": {"disapp": None, "floor": 0.48}}
with open("/kaggle/working/fork_gate_jointimg.pkl", "wb") as fh:
    pickle.dump(blob, fh)
print(f"  fork_gate_jointimg.pkl written: n_train {n_train}; p quantiles over all {len(ALL):,} forks: "
      + " ".join(f"{int(f * 100)}%={v:.4f}" for f, v in Q.items()), flush=True)
m_ctrl = _fit(CNT[FEAT20].values, y)
with open("/kaggle/working/fork_gate_joint20_refit.pkl", "wb") as fh:
    pickle.dump({"model": m_ctrl, "feature_names": FEAT20, "version": SCRIPT_VERSION, "n_train": n_train,
                 "trained_on": "the same counted forks, 20 features only (control)", "detector": "F4"}, fh)
print("  fork_gate_joint20_refit.pkl written (control: same forks, no image columns)", flush=True)
print("GATEFIT OK", flush=True)
'''
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gatefit.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(SRC)
print("written gatefit.py", len(SRC))
