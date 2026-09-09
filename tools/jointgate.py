"""jointgate -- rebuild the JOINT fork gate from each candidate's OWN forks.

THE GAP THIS CLOSES. `fork_gate_joint.pkl` (submit-joint, LB 0.944) is trained
on 128 6bba counted forks PLUS 71 44b6 ones, and the 44b6 half comes from
divbench5, generated under the SHIPPED BASE-pool ranker on refine
coordinates. Change the ranker or the coordinates and the 6bba forks change
while the 44b6 training forks do not -- the gate is then fitted on a mixture
of two fork-generating processes. `rkpool` measured the consequence without
naming it: the joint gate's margin over the 6bba-only gate falls from
**+0.0037** on the shipped ranker to **+0.0004** on the retrained one.

Every arm's fork table is on disk, so for the candidates whose BOTH halves
exist the coherent gate can be rebuilt here, with no kernel. `gatelab`
established that this is legitimate: the pruning curve is a pure function of
(label, gate p), because div_J = TP/(TP+FP+FN) with FN = N_GT - TP and N_GT
fixed. adj_J is NOT available offline and is not reported.

THE GATE, divbench3's exactly: HistGB(100, lr .05, msl 10, msl 10, rs 0),
2 folds, trained on the OTHER 6bba fold's counted forks PLUS all the 44b6
counted forks, cut at p > 0.03.

ANCHOR: the shipped pair (rkpool_ref's 6bba forks + hp44full_ref's 44b6 ones)
must reproduce divbench8 -- 6bba-only gate 42/47/83 div_J 0.2442, joint gate
**46/39/79 div_J 0.2805**. If it does not, nothing else here is readable.

    py tools/jointgate.py
"""
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

N_GT = 125          # 6bba, over the 128 scored movies (50 TP + 75 FN ungated)
CUT = 0.03

# (label, 6bba forks, 44b6 forks, recorded 6bba-only cells, recorded joint)
ARMS = [
    ("A ships   (BASE rk, refine)", "bench_csv/forks_base/forks_dw55.csv",
     "bench_csv/f44_ref/forks_44b6.csv", (42, 47, 83), (46, 39, 79)),
    ("B retrain (F4   rk, refine)", "bench_csv/forks_f4/forks_dw55.csv",
     None, (45, 52, 80), None),
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
    from the other domain, added to BOTH training folds -- which is what
    makes the gate 'joint'."""
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
print(f"{'arm':<30}{'gate':<8}{'tp/fp/fn':>14}{'div_J':>9}{'want':>14}{'':>8}")
for tag, p6, p44, want6, wantj in ARMS:
    try:
        f6, l6, X6 = load(p6)
    except OSError:
        print(f"{tag:<30}  (no 6bba fork table at {p6})")
        continue
    n_c = int((l6 >= 1).sum())
    extra = None
    if p44:
        try:
            _f44, l44, X44 = load(p44)
            extra = (l44, X44)
        except OSError:
            print(f"{tag:<30}  (no 44b6 fork table at {p44})")
    for name, ex, want in (("6bba", None, want6), ("joint", extra, wantj)):
        if ex is None and name == "joint":
            print(f"{tag:<30}{name:<8}{'--':>14}{'':>9}"
                  f"{'no coherent 44b6 forks yet (fk44f4 is running)':>14}")
            continue
        c = cells(l6, gate(f6, l6, X6, ex) > CUT)
        w = (f"{want[0]}/{want[1]}/{want[2]}" if want else "new")
        ok = (want is None) or (c[:3] == tuple(want))
        if want and not ok:
            fail.append((tag, name, c[:3], tuple(want)))
        print(f"{tag:<30}{name:<8}{f'{c[0]}/{c[1]}/{c[2]}':>14}{c[3]:>9.4f}"
              f"{w:>14}{'  OK' if ok else '  MISS':>8}"
              f"   ({n_c} counted 6bba"
              + (f" + {int((extra[0] >= 1).sum())} 44b6)" if ex is not None
                 else ")"))

if fail:
    print(f"\n⚠ {len(fail)} recorded cell(s) did not reproduce: {fail}")
    print("  The offline gate is not divbench3's; read nothing above.")
    sys.exit(1)
print("\n  every recorded cell reproduces -- the offline gate is the shipped "
      "one,\n  and the joint rows above are each arm's OWN forks in both "
      "domains.")
print("""
  ⚠ div_J only. adj_J needs the real pruning inside the solver and is not
  available offline; divbench3 measured adj_J RISING under every cut, so
  div_J is the lever. Score effect is about 0.1 x d div_J.""")
