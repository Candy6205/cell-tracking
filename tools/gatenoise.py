"""How much does the joint fork gate's counted TP/FP at p > 0.03 move when the fork table
loses a handful of forks? Perturb REF's forks the way the BG arms did (a few counted FP
forks and a few dozen uncounted ones removed), refit the same 2-fold gate + the 67 44b6
forks, count. Then place each BG arm's own table in that distribution."""
import re, sys, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings("ignore")
src = open("bridgegate.py", encoding="utf-8").read()
i = src.index("ROWS44 = ["); j = src.index("]\nassert FEAT44", i) + 1
g = dict(np=np); exec(src[i:j], g); ROWS44 = g["ROWS44"]
X44 = np.array([r[1] for r in ROWS44], float); Y44 = np.array([1 if r[0] == 2 else 0 for r in ROWS44], int)
FEAT20 = ["src", "d_pc1", "d_pq", "d_s", "dv", "p_c1", "p_q", "pf_c1", "pf_q", "pr_q", "sc_p", "sc_c1", "sc_q",
          "len_back", "len_c1", "len_q", "dens10", "n_orph", "t_frac", "rank_gap"]
J20 = list(range(20))
N_GT = 125
THR = 0.03

def gate_count(df, add=True, seed=0):
    X = df[FEAT20].to_numpy(float); lab = df["label"].to_numpy(int); fold = df["fold"].to_numpy(int)
    p = np.full(len(df), np.nan); cnt = lab >= 1
    for k in (0, 1):
        tr = (fold != k) & cnt
        Xt, yt = X[tr], (lab[tr] == 2).astype(int)
        if add:
            Xt = np.vstack([Xt, X44[:, J20]]); yt = np.concatenate([yt, Y44])
        m = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, min_samples_leaf=10, random_state=seed).fit(Xt, yt)
        p[fold == k] = m.predict_proba(X[fold == k])[:, 1]
    keep = p > THR
    tp = int((lab[keep] == 2).sum()); fp = int((lab[keep] == 1).sum())
    return tp, fp, tp / (tp + fp + (N_GT - tp))

REF = pd.read_csv("bench_csv/bridgegate/forks_dw55.csv")
print("REF counted:", (REF.label == 2).sum(), (REF.label == 1).sum(), "uncounted", (REF.label == 0).sum())
base = gate_count(REF); print(f"REF + joint (label count): tp {base[0]} fp {base[1]} div_J {base[2]:.4f}   (cross rescored 49/51 0.2784)")
for arm in ("BG10", "BG20", "BG30", "BG50"):
    A = pd.read_csv(f"bench_csv/bridgegate/forks_{arm}.csv")
    r = gate_count(A)
    print(f"{arm} + joint (label count): tp {r[0]} fp {r[1]} div_J {r[2]:.4f}  counted {(A.label == 2).sum()}/{(A.label == 1).sum()} uncounted {(A.label == 0).sum()}")
rng = np.random.default_rng(0)
print("\nperturbation of REF's table: drop 4 random counted-FP forks + 40 random uncounted forks, refit, count (100 draws)")
res = []
for d in range(100):
    fp_idx = np.where(REF.label == 1)[0]; un_idx = np.where(REF.label == 0)[0]
    drop = np.concatenate([rng.choice(fp_idx, 4, replace=False), rng.choice(un_idx, 40, replace=False)])
    res.append(gate_count(REF.drop(index=drop).reset_index(drop=True)))
res = np.array(res)
print(f"  tp mean {res[:,0].mean():.1f} [{np.percentile(res[:,0],5):.0f}, {np.percentile(res[:,0],95):.0f}]  "
      f"fp mean {res[:,1].mean():.1f} [{np.percentile(res[:,1],5):.0f}, {np.percentile(res[:,1],95):.0f}]  "
      f"div_J mean {res[:,2].mean():.4f} sd {res[:,2].std():.4f} 5-95% [{np.percentile(res[:,2],5):.4f}, {np.percentile(res[:,2],95):.4f}]")
print(f"  score-equivalent spread (0.1 x div_J): sd {0.1*res[:,2].std():.4f}, 5-95% width {0.1*(np.percentile(res[:,2],95)-np.percentile(res[:,2],5)):.4f}")
print(f"  fraction of draws with div_J >= BG20's (label count) {np.mean(res[:,2] >= gate_count(pd.read_csv('bench_csv/bridgegate/forks_BG20.csv'))[2]):.2f}")
print("\nsame gate, different random_state on the untouched REF table (HistGB is deterministic given data; this checks that):")
print("  ", [gate_count(REF, seed=s) for s in (0, 1, 2)])
print("\nperturbation: drop only 40 uncounted forks (the training set is untouched, only the evaluated population changes):")
res2 = np.array([gate_count(REF.drop(index=rng.choice(np.where(REF.label == 0)[0], 40, replace=False)).reset_index(drop=True)) for _ in range(20)])
print("  ", sorted(set(map(tuple, res2[:, :2].astype(int).tolist()))))
