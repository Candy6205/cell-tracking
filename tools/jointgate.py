"""jointgate -- rebuild the JOINT fork gate from each candidate's OWN forks.

THE GAP THIS CLOSES. `fork_gate_joint.pkl` (submit-joint, LB 0.944) is trained
on 128 6bba counted forks PLUS 71 44b6 ones, and the 44b6 half comes from
divbench5, generated under the SHIPPED BASE-pool ranker on refine
coordinates. Change the ranker or the coordinates and the 6bba forks change
while the 44b6 training forks do not -- the gate is then fitted on a mixture
of two fork-generating processes. `rkpool` measured the consequence without
naming it: the joint gate's margin over the 6bba-only gate falls from
**+0.0037** on the shipped ranker to **+0.0004** on the retrained one.

Every arm's fork table is on disk, so the coherent gate can be rebuilt here
with no kernel. `gatelab` established that this is legitimate: the pruning
curve is a pure function of (label, gate p), because div_J = TP/(TP+FP+FN)
with FN = N_GT - TP and N_GT fixed. adj_J is NOT available offline and is not
reported.

THE GATE, divbench3's exactly: HistGB(100, lr .05, min_samples_leaf 10,
random_state 0), 2 folds, trained on the OTHER 6bba fold's counted forks plus
whichever 44b6 counted forks the row names, cut at p > 0.03.

THE VARIABLE IS WHICH 44b6 FORKS THE JOINT GATE SEES. Every arm gets three
rows: the 6bba-only gate, the joint gate on the SHIPPED 44b6 forks
(divbench5's -- BASE ranker, refine coordinates, what every recorded run
used), and the joint gate on the arm's OWN 44b6 forks. The first two are
ANCHORED to recorded values; the third is the measurement.

ANCHOR: the shipped pair must reproduce divbench8 -- 6bba-only 42/47/83
div_J 0.2442, joint **46/39/79 div_J 0.2805**. If it does not, read nothing.

    py -u tools/jointgate.py
"""
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

N_GT = 125          # 6bba, over the 128 scored movies (50 TP + 75 FN ungated)
CUT = 0.03
SHIPPED_44B6 = "bench_csv/f44_ref/forks_44b6.csv"

ARMS = [
    # tag, 6bba forks, own 44b6 forks, want(6bba-only), want(joint on SHIPPED)
    ("A ships   (BASE rk, refine)", "bench_csv/forks_base/forks_dw55.csv",
     SHIPPED_44B6, (42, 47, 83), (46, 39, 79)),
    ("B retrain (F4   rk, refine)", "bench_csv/forks_f4/forks_dw55.csv",
     "bench_csv/f44_f4/forks_44b6.csv", (45, 52, 80), (45, 49, 80)),
    ("C hpara   (BASE rk, hpara )", "bench_csv/forks_hpara/forks_dw55.csv",
     "bench_csv/f44_hp/forks_44b6.csv", (45, 53, 80), (49, 51, 76)),
]


def load(path):
    L = open(path, encoding="utf-8").read().strip().split("\n")
    R = [l.split(",") for l in L[1:]]
    return (np.array([int(r[1]) for r in R]),                    # fold
            np.array([int(r[4]) for r in R]),                    # label
            np.array([[float(v) for v in r[5:25]] for r in R]))  # 20 features


def gate(fold6, lab6, X6, extra=None):
    """Out-of-fold p for every 6bba fork. `extra` is (lab, X) of counted forks
    from the other domain, added to BOTH training folds -- which is what makes
    the gate 'joint'."""
    p = np.full(len(lab6), np.nan)
    for k in (0, 1):
        tr = (fold6 != k) & (lab6 >= 1)
        Xa, ya = X6[tr], (lab6[tr] == 2).astype(int)
        if extra is not None:
            le, Xe = extra
            c = le >= 1
            Xa = np.concatenate([Xa, Xe[c]])
            ya = np.concatenate([ya, (le[c] == 2).astype(int)])
        m = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05,
                                           min_samples_leaf=10, random_state=0)
        m.fit(Xa, ya)
        p[fold6 == k] = m.predict_proba(X6[fold6 == k])[:, 1]
    return p


def cells(lab, keep):
    tp = int(((lab == 2) & keep).sum())
    fp = int(((lab == 1) & keep).sum())
    fn = N_GT - tp
    return tp, fp, fn, tp / max(1e-9, tp + fp + fn)


fail = []
print(f"{'arm':<30}{'44b6 forks used':<24}{'tp/fp/fn':>12}{'div_J':>9}"
      f"{'want':>12}{'':>7}  counted", flush=True)
for tag, p6, pown, want6, wantj in ARMS:
    try:
        f6, l6, X6 = load(p6)
    except OSError:
        print(f"{tag:<30}  (no 6bba fork table at {p6})", flush=True)
        continue
    n6 = int((l6 >= 1).sum())
    rows = [("none (6bba-only gate)", None, want6)]
    try:
        _f, lS, XS = load(SHIPPED_44B6)
        rows.append(("SHIPPED (divbench5)", (lS, XS), wantj))
    except OSError:
        pass
    if pown and pown != SHIPPED_44B6:
        try:
            _f, lO, XO = load(pown)
            rows.append(("this arm's OWN", (lO, XO), None))
        except OSError:
            rows.append(("this arm's OWN -- not yet built", "skip", None))
    for name, ex, want in rows:
        if isinstance(ex, str):
            print(f"{tag:<30}{name:<24}{'--':>12}", flush=True)
            continue
        c = cells(l6, gate(f6, l6, X6, ex) > CUT)
        ok = (want is None) or (c[:3] == tuple(want))
        if want and not ok:
            fail.append((tag, name, c[:3], tuple(want)))
        nx = f" + {int((ex[0] >= 1).sum())} 44b6" if ex is not None else ""
        print(f"{tag:<30}{name:<24}{f'{c[0]}/{c[1]}/{c[2]}':>12}{c[3]:>9.4f}"
              f"{(f'{want[0]}/{want[1]}/{want[2]}' if want else 'NEW'):>12}"
              f"{('  OK' if ok else '  MISS') if want else '':>7}"
              f"  {n6} 6bba{nx}", flush=True)

if fail:
    print(f"\n{len(fail)} recorded cell(s) did not reproduce: {fail}")
    print("  The offline gate is not divbench3's; read nothing above.")
    sys.exit(1)
print("\n  every recorded cell reproduces, so the offline gate IS the shipped"
      " one and\n  the OWN rows are readable: each differs from the SHIPPED"
      " row above it in\n  exactly one thing -- which 44b6 forks the joint"
      " gate was fitted on.")
print("""
  div_J only. adj_J needs the real pruning inside the solver and is not
  available offline; divbench3 measured adj_J RISING under every cut, so
  div_J is the lever. Score effect is about 0.1 x d div_J.""")
