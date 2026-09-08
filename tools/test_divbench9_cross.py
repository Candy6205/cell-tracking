"""Synthetic test of divbench9's new code path, run before the kernel is pushed.

The path is `add=` on `_gate_and_prune`: the 67 counted 44b6 forks appended to
each training fold. Its silent failure is the one an anchor cannot catch,
because in this run the anchors ARE the controls -- if `add` were dropped, the
`joint` column would simply reprint the `gate6` column, all three anchors would
still reproduce, and the cross would report "they overlap completely" when in
fact nothing was crossed.

    py tools/test_divbench9_cross.py divbench9.py
"""
import sys
from collections import defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

TARGET = sys.argv[1] if len(sys.argv) > 1 else "divbench9.py"
SRC = open(TARGET, encoding="utf-8").read()


def cut(start, end, what):
    a = SRC.index(start)
    b = SRC.index(end, a)
    print(f"  extracted {what}: {b - a} chars")
    return SRC[a:b]


print("1. the embedded 44b6 rows are the SAME literal divbench8 measured on")
feat_src = SRC[SRC.index("FEAT20 = ["):SRC.index("FEAT_NAMES = FEAT20 + FEAT_NEW")]
blk = cut("FEAT44 = [", "assert int(Y44.sum()) == 6", "the 44b6 block")
blk += "assert int(Y44.sum()) == 6, int(Y44.sum())\n"
ns = {"np": np}
exec(feat_src, ns)
ns["FEAT_NAMES"] = ns["FEAT20"] + ns["FEAT_NEW"]
exec(blk, ns)
X44, Y44 = ns["X44"], ns["Y44"]
NF = len(ns["FEAT_NAMES"])
assert X44.shape == (67, NF) and Y44.sum() == 6 and (Y44 == 0).sum() == 61
assert ns["FEAT44"] == ns["FEAT_NAMES"] and ns["FEAT20"][0] == "src"
try:
    import hashlib
    h8 = hashlib.sha256(
        open("divbench8.py", encoding="utf-8").read()
        [SRC.index("FEAT44 = ["):0].encode()).hexdigest() if False else None
    s8 = open("divbench8.py", encoding="utf-8").read()
    b8 = s8[s8.index("FEAT44 = ["):s8.index("assert int(Y44.sum()) == 6")]
    b9 = SRC[SRC.index("FEAT44 = ["):SRC.index("assert int(Y44.sum()) == 6")]
    assert b8 == b9, "divbench8 and divbench9 do not carry the SAME 44b6 rows -- the " \
                     "joint column would not be the same measurement"
    print("  OK: byte-identical to divbench8's block, so 'joint' means the same thing")
except FileNotFoundError:
    print("  (divbench8.py not present; skipped the byte-identity check)")

print("\n2. a synthetic arm: 4 movies, 2 folds, a learnable signal")
rng = np.random.default_rng(11)
NAMES = ["m0", "m1", "m2", "m3"]
FOLD_OF = {n: i % 2 for i, n in enumerate(NAMES)}
FORKS = []
for name in NAMES:
    for j in range(70):
        lab = 0 if j >= 40 else (2 if j < 12 else 1)
        f = rng.normal(size=NF)
        f[0] = rng.normal(2.0 if lab == 2 else 0.0)
        FORKS.append((name, FOLD_OF[name], j, 1000 + j, lab, f.tolist()))
LAB = np.array([r[4] for r in FORKS], int)
FLD = np.array([r[1] for r in FORKS], int)
N_TR = {k: int((((FLD != k)) & (LAB >= 1)).sum()) for k in (0, 1)}
KEY = {(r[2], r[3]): i for i, r in enumerate(FORKS)}
del KEY[(0, 1000)]
MOVIE = {n: {"key": KEY} for n in NAMES}
PER = {n: {"name": n, "tag": "REF"} for n in NAMES}
print(f"  {len(FORKS)} forks, {int((LAB >= 1).sum())} counted, "
      f"{int((LAB == 2).sum())} TP; training rows per fold {N_TR}")

FITS, RESCORED, WARNED = [], [], []


class Rec(HistGradientBoostingClassifier):
    def fit(self, X, y, **kw):
        FITS.append((tuple(np.shape(X)), int(np.asarray(y).sum())))
        return super().fit(X, y, **kw)


NS = dict(np=np, defaultdict=defaultdict, roc_auc_score=roc_auc_score,
          HistGradientBoostingClassifier=Rec, FEAT_NAMES=ns["FEAT_NAMES"],
          J20=list(range(20)), SIX_OK=NAMES, FOLD_OF=FOLD_OF,
          warn_once=lambda k, m: WARNED.append((k, m)),
          _rescore=lambda n, s, d, t: (RESCORED.append((n, tuple(sorted(d)), t))
                                       or {"name": n, "tag": t}),
          _with_ratio=lambda rows: {"score": float(len(rows)), "rows": rows})
TAIL = "return float(roc_auc_score(y, p))"
exec(cut("def _auc(y, p):", TAIL, "_auc") + TAIL + "\n", NS)
exec(cut("def _gate_and_prune(", "\n\nTHR = 0.03", "_gate_and_prune"), NS)
gate = NS["_gate_and_prune"]

print("\n3. THE POINT: does `add` reach the estimator, or is `joint` a reprint?")
FITS.clear()
_rc, _rows, _pf, _a = gate("gate6", FORKS, MOVIE, PER, 0.5)
p6 = None
shapes6 = [s for s, _ in FITS]
assert [s[0] for s in shapes6] == [N_TR[0], N_TR[1]], shapes6
print(f"  gate6:  fits {shapes6}")

FITS.clear()
RESCORED.clear()
_rc2, _rows2, _pf2, _a2 = gate("joint", FORKS, MOVIE, PER, 0.5, add=(X44, Y44))
shapesj = [s for s, _ in FITS]
tpj = [t for _, t in FITS]
tp6 = [int((LAB[(FLD != k) & (LAB >= 1)] == 2).sum()) for k in (0, 1)]
assert [s[0] for s in shapesj] == [N_TR[0] + 67, N_TR[1] + 67], shapesj
assert tpj == [t + 6 for t in tp6], (tpj, tp6)
print(f"  joint:  fits {shapesj}   positives {tpj} (= {tp6} + 6)")

drops6 = {}
RESCORED.clear()
gate("g6b", FORKS, MOVIE, PER, 0.5)
drops6 = {n: d for n, d, _t in RESCORED}
RESCORED.clear()
gate("jtb", FORKS, MOVIE, PER, 0.5, add=(X44, Y44))
dropsj = {n: d for n, d, _t in RESCORED}
assert drops6 != dropsj, \
    "FATAL: gate6 and joint pruned exactly the same edges -- the cross would " \
    "report 'they overlap completely' having never crossed anything"
diff = sum(len(set(drops6.get(n, ())) ^ set(dropsj.get(n, ()))) for n in NAMES)
print(f"  the two gates prune DIFFERENT edge sets: {diff} edges differ across "
      f"{len(NAMES)} movies")

print("\n4. an empty add is a no-op, not a crash (the REF column passes add=None)")
RESCORED.clear()
gate("empty", FORKS, MOVIE, PER, 0.5, add=(np.zeros((0, NF)), np.zeros(0, int)))
dropse = {n: d for n, d, _t in RESCORED}
assert dropse == drops6, "add=(empty) changed the result"
print("  OK: add=(empty, empty) reproduces gate6 exactly")

assert any(k == "prune_edge_missing" for k, _m in WARNED), \
    "an unmapped fork edge was skipped without a warning"
print(f"\n5. unmapped fork edges warned rather than passing silently "
      f"({len(WARNED)} warnings)")
print("\nALL CHECKS PASSED for", TARGET)
