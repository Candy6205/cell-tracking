"""forkjoin -- does the retrained ranker remove the forks the GATE removes?

rkpool left a claim I made and then had to withdraw: that the F4-pool retrain
drops "most of what the joint gate was there to remove". The counts say a
tenth (652 of 7,004 forks; 24 of 249 division FPs against the gate's 210).
But the counts do not answer the question that matters, which is whether the
652 are the SAME forks -- if the retrain quietly kills exactly the ones the
gate was killing, the gate has nothing left to do, and that is the shape of
what was measured: the joint gate is worth +0.0037 on the shipped ranker and
+0.0004 on the retrained one.

Both arms wrote their DIV_W 0.55 fork table, so this is a join, offline, free.

WHAT IT DOES
  1. reproduces divbench3's 20-feature 2-fold gate on the BASE table -- the
     ANCHOR, out-of-fold AUC 0.9283 / 0.8326 and p > 0.03 -> 42/47/83,
     div_J 0.2442. Without that the rest is an unverified model's opinion.
  2. joins the two tables on (movie, p, q) -- the same F4 detections in both
     arms, so a fork is the same fork.
  3. asks of the forks the retrain DROPS: what would the gate have done with
     them? If they sit below the cut, the two mechanisms overlap and the
     gate's collapse is explained. If they sit above it, the retrain is
     removing forks the gate KEEPS, and the collapse needs another story.

N_GT = 125 over the 128 6bba movies (50 TP + 75 FN, ungated).

    py tools/forkjoin.py
"""
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

BASE = sys.argv[1] if len(sys.argv) > 1 else "bench_csv/forks_base/forks_dw55.csv"
F4 = sys.argv[2] if len(sys.argv) > 2 else "bench_csv/forks_f4/forks_dw55.csv"
N_GT = 125
CUT = 0.03


def load(path):
    L = open(path, encoding="utf-8").read().strip().split("\n")
    R = [l.split(",") for l in L[1:]]
    return dict(
        movie=np.array([r[0] for r in R]),
        fold=np.array([int(r[1]) for r in R]),
        # NOT (movie, p, q). `p` and `q` index the SELECTED node set, and the
        # two arms select different nodes -- 6bba_05b6850b keeps 6,165 in the
        # BASE arm and 6,154 in the F4 one -- so the indices shift and a
        # (movie, p, q) join matched 78 of 7,004. The COORDINATES are
        # identical in both arms (same F4 detections), so the fork's own
        # geometry is stable: mother-to-c1, mother-to-q and the sister
        # distance name the same three physical points either way. Label
        # agreement on the matched rows is printed below as the check.
        key=np.array([f"{r[0]}|{r[6]}|{r[7]}|{r[8]}" for r in R]),
        lab=np.array([int(r[4]) for r in R]),
        X=np.array([[float(v) for v in r[5:25]] for r in R]),
        # columns 10/11 are the RANKER's probability on the two fork edges;
        # 12/13/14 are the HEAD's (pf_c1, pf_q, pr_q). The head is the same
        # cached file in both arms, so its columns are a free control.
        p_c1=np.array([float(r[10]) for r in R]),
        p_q=np.array([float(r[11]) for r in R]),
        pf_c1=np.array([float(r[12]) for r in R]),
        pf_q=np.array([float(r[13]) for r in R]),
    )


A, H = load(BASE), load(F4)
print(f"BASE {len(A['lab'])} forks (divbench3: 7,004)   "
      f"F4 {len(H['lab'])} forks")
for tag, D in (("BASE", A), ("F4", H)):
    print(f"  {tag:<5} counted {int((D['lab'] >= 1).sum()):>5}   "
          f"TP {int((D['lab'] == 2).sum()):>3}   "
          f"FP {int((D['lab'] == 1).sum()):>4}")


def auc(y, s):
    y, s = np.asarray(y, int), np.asarray(s, float)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if not y.any() or y.all():
        return float("nan")
    o = np.argsort(s)
    r = np.empty(len(s), float)
    r[o] = np.arange(1, len(s) + 1)
    n1 = int(y.sum())
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def fit_gate(D):
    """divbench3's gate: HistGB(100, lr .05, msl 10, rs 0), 2 folds, trained
    on the OTHER fold's COUNTED forks with TP=1 / FP=0. Returns p for every
    fork in D, out of fold, plus the per-fold AUCs."""
    p = np.full(len(D["lab"]), np.nan)
    aucs = {}
    for k in (0, 1):
        tr = (D["fold"] != k) & (D["lab"] >= 1)
        te = D["fold"] == k
        m = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05,
                                           min_samples_leaf=10,
                                           random_state=0)
        m.fit(D["X"][tr], (D["lab"][tr] == 2).astype(int))
        p[te] = m.predict_proba(D["X"][te])[:, 1]
        c = te & (D["lab"] >= 1)
        aucs[k] = (auc((D["lab"][c] == 2).astype(int), p[c]), int(c.sum()))
    return p, aucs


def cells(lab, keep):
    tp = int(((lab == 2) & keep).sum())
    fp = int(((lab == 1) & keep).sum())
    fn = N_GT - tp
    return tp, fp, fn, tp / max(1e-9, tp + fp + fn)


print("\n=== ANCHOR: divbench3's gate, reproduced on the BASE table ===")
pA, aucA = fit_gate(A)
print(f"  out-of-fold AUC  fold0 {aucA[0][0]:.4f} ({aucA[0][1]} counted)  "
      f"want 0.9283 (122)")
print(f"                   fold1 {aucA[1][0]:.4f} ({aucA[1][1]} counted)  "
      f"want 0.8326 (177)")
tp, fp, fn, dj = cells(A["lab"], np.ones(len(A["lab"]), bool))
print(f"  ungated  {tp}/{fp}/{fn}  div_J {dj:.4f}   want 50/249/75 0.1337")
tp, fp, fn, dj = cells(A["lab"], pA > CUT)
print(f"  p > {CUT}  {tp}/{fp}/{fn}  div_J {dj:.4f}   want 42/47/83 0.2442")
ok = (tp, fp, fn) == (42, 47, 83)
print(f"  -> {'ANCHOR HOLDS' if ok else 'ANCHOR MISSED -- read nothing below'}")

print("\n=== THE JOIN, on (movie, fork geometry) ===")
sA, sH = set(A["key"]), set(H["key"])
assert len(A["key"]) == len(sA), "geometry collides inside the BASE table"
assert len(H["key"]) == len(sH), "geometry collides inside the F4 table"
both = sA & sH
only_a = sA - sH
only_h = sH - sA
print(f"  in both {len(both)}   BASE only {len(only_a)}   F4 only {len(only_h)}")
print(f"  net {len(H['lab']) - len(A['lab'])}, but the CHURN is far larger: "
      f"the retrain drops {len(only_a)} forks and adds {len(only_h)}.")

# The key's own check. A fork's TP/FP status is a property of the physical
# event and the ground truth, not of the ranker, so every matched pair must
# carry the same label. Disagreement means the key is joining different forks.
labH = {k: v for k, v in zip(H["key"], H["lab"])}
mism = [(k, int(v), int(labH[k])) for k, v in zip(A["key"], A["lab"])
        if k in labH and labH[k] != v]
print(f"  label agreement on the {len(both)} matched forks: "
      f"{len(both) - len(mism)}/{len(both)} "
      f"({(1 - len(mism) / max(1, len(both))) * 100:.2f}%)"
      f"{'' if not mism else '  <- ' + str(mism[:3])}")

drop = np.array([k in only_a for k in A["key"]])
kept = ~drop
print(f"\n  what the retrain DROPS, by label:")
for name, v in (("TP (a real division)", 2), ("FP (counted, wrong)", 1),
                ("not counted", 0)):
    d = int(((A["lab"] == v) & drop).sum())
    t = int((A["lab"] == v).sum())
    print(f"    {name:<24} {d:>4} of {t:>5}  ({d / max(1, t) * 100:.1f}%)")

print(f"\n=== WOULD THE GATE HAVE REMOVED THEM ANYWAY? ===")
print(f"  the gate's cut is p > {CUT}; below it the fork is pruned.")
below = pA <= CUT
n_drop = int(drop.sum())
n_drop_below = int((drop & below).sum())
n_keep_below = int((kept & below).sum())
print(f"  of the {n_drop} forks the retrain drops, "
      f"{n_drop_below} ({n_drop_below / max(1, n_drop) * 100:.1f}%) "
      f"were already below the gate's cut")
print(f"  of the {int(kept.sum())} it keeps,       "
      f"{n_keep_below} ({n_keep_below / max(1, int(kept.sum())) * 100:.1f}%) "
      f"were already below it")
print(f"  base rate over all {len(pA)}: "
      f"{int(below.sum()) / len(pA) * 100:.1f}% below the cut")
q = [10, 25, 50, 75, 90]
print(f"  gate p quantiles  {'':<10}" + "".join(f"{f'q{x}':>10}" for x in q))
print(f"    forks the retrain DROPS   " +
      "".join(f"{v:>10.4f}" for v in np.nanpercentile(pA[drop], q)))
print(f"    forks it KEEPS            " +
      "".join(f"{v:>10.4f}" for v in np.nanpercentile(pA[kept], q)))

print(f"\n=== AND WHAT IS LEFT FOR THE GATE ===")
tp0, fp0, fn0, dj0 = cells(A["lab"], np.ones(len(A["lab"]), bool))
tp1, fp1, fn1, dj1 = cells(A["lab"][kept], pA[kept] > -1)
print(f"  BASE ungated                  {tp0}/{fp0}/{fn0}  div_J {dj0:.4f}")
print(f"  BASE minus the dropped forks  {tp1}/{fp1}/{fn1}  div_J {dj1:.4f}"
      f"   <- what the retrain alone buys, ON THE BASE ARM'S OWN GRAPHS")
tp2, fp2, fn2, dj2 = cells(A["lab"], (pA > CUT) & kept)
print(f"  ...then the gate on top        {tp2}/{fp2}/{fn2}  div_J {dj2:.4f}")
tp3, fp3, fn3, dj3 = cells(A["lab"], pA > CUT)
print(f"  the gate alone                 {tp3}/{fp3}/{fn3}  div_J {dj3:.4f}")
print(f"\n=== WHAT THE RETRAIN DID TO THE PROBABILITIES THEMSELVES ===")
print(f"  On the {len(both)} forks present in both arms, the same three "
      f"physical points\n  carry a ranker probability from each arm. rkpool "
      f"read the held-out AUC as\n  FLAT (0.99256 -> 0.99267), so if anything "
      f"moved it is calibration, and this\n  is the direct look at it.")
iA = {k: i for i, k in enumerate(A["key"])}
iH = {k: i for i, k in enumerate(H["key"])}
mk = [k for k in A["key"] if k in iH]
ia = np.array([iA[k] for k in mk])
ih = np.array([iH[k] for k in mk])
mlab = A["lab"][ia]

# The control first: the head is the same cached file in both arms.
for nm in ("pf_c1", "pf_q"):
    da = A[nm][ia] - H[nm][ih]
    da = da[np.isfinite(da)]
    print(f"  CONTROL {nm:<6} (the HEAD, same cache both arms): "
          f"max |d| {np.abs(da).max():.3e} over {len(da)} forks "
          f"{'-- identical, the join is sound' if np.abs(da).max() < 1e-9 else '<-- NOT IDENTICAL, the join is wrong'}")

print(f"\n  {'':<26}{'BASE mean':>12}{'F4 mean':>12}{'d':>10}"
      f"{'higher in F4':>14}")
for nm in ("p_c1", "p_q"):
    for tag, m in (("all matched forks", np.ones(len(mk), bool)),
                   ("  TP (real division)", mlab == 2),
                   ("  FP (counted, wrong)", mlab == 1)):
        a, h = A[nm][ia][m], H[nm][ih][m]
        ok = np.isfinite(a) & np.isfinite(h)
        a, h = a[ok], h[ok]
        if not len(a):
            continue
        print(f"  {nm} {tag:<20}{a.mean():>12.4f}{h.mean():>12.4f}"
              f"{h.mean() - a.mean():>+10.4f}"
              f"{f'{(h > a).mean() * 100:.0f}%':>14}")

print(f"\n  Reading: a shift of the SAME sign in TP and FP is calibration -- "
      f"the whole\n  scale moved and the floor now cuts somewhere else. "
      f"Opposite signs (TP up,\n  FP down) is real separation, which the flat "
      f"held-out AUC argues against.")

print(f"\n  ⚠ the middle two rows are an ARITHMETIC overlay, not a run: "
      f"removing a fork\n    from the BASE arm's table is not the same as "
      f"re-solving the ILP without it,\n    and the F4 arm's real answer is "
      f"52/225/73. Read them as a decomposition of\n    WHICH forks moved, "
      f"not as a prediction of score.")
