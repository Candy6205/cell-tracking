"""Synthetic test of divbench8's NEW code path, run before the kernel is pushed.

The working agreement (2026-09-07): when an assembly adds a NEW CODE PATH
rather than re-parameterising an existing one, test that path on synthetic
data first, because the silent failure -- an arm that runs and returns the
CONTROL -- is exactly the one an anchor cannot catch, since the anchor IS the
control.

divbench8's new path is `_gate_rows`'s training-set augmentation. If `add`
were dropped on the floor, `joint`, `shuf` and `pad` would all print gate6's
numbers, the gate6 anchor would still reproduce divbench3 exactly, and the
run would look perfectly healthy while measuring nothing.

    py tools/test_divbench8_gate.py divbench8.py
"""
import sys
from collections import defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

TARGET = sys.argv[1] if len(sys.argv) > 1 else "divbench8.py"
SRC = open(TARGET, encoding="utf-8").read()


def cut(start, end, what):
    a = SRC.index(start)
    b = SRC.index(end, a)
    print(f"  extracted {what}: {b - a} chars")
    return SRC[a:b]


# ---------------------------------------------------------------------------
print("1. the embedded 44b6 rows")
# ---------------------------------------------------------------------------
feat_src = cut("FEAT20 = [", "\nFEAT_NEW", "FEAT20") + "]"
feat_src = SRC[SRC.index("FEAT20 = ["):SRC.index("FEAT_NAMES = FEAT20 + FEAT_NEW")]
blk_src = cut("FEAT44 = [", 'assert int(Y44.sum()) == 6', "the 44b6 block")
blk_src += 'assert int(Y44.sum()) == 6, int(Y44.sum())\n'
ns = {"np": np}
exec(feat_src, ns)
ns["FEAT_NAMES"] = ns["FEAT20"] + ns["FEAT_NEW"]
exec(blk_src, ns)          # its own asserts are the test
X44, Y44 = ns["X44"], ns["Y44"]
NF = len(ns["FEAT_NAMES"])
assert X44.shape == (67, NF), X44.shape
assert Y44.sum() == 6 and (Y44 == 0).sum() == 61
assert ns["FEAT44"] == ns["FEAT_NAMES"]
assert ns["FEAT20"][0] == "src" and len(ns["FEAT20"]) == 20, ns["FEAT20"][:3]
# the columns must be real numbers, not all-NaN, or the gate learns nothing
finite = np.isfinite(X44[:, :20]).sum(0)
assert (finite > 0).all(), [ns["FEAT20"][i] for i in np.where(finite == 0)[0]]
print(f"  OK: 67 rows x {NF} features, 6 TP / 61 FP, columns match FEAT_NAMES")
print(f"      finite values per shipped feature: min {finite.min()} of 67 "
      f"({ns['FEAT20'][int(np.argmin(finite))]})")

# ---------------------------------------------------------------------------
print("\n2. a synthetic 6bba: 4 movies, 2 folds, a learnable signal")
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)
NAMES = ["m0", "m1", "m2", "m3"]
FOLD_OF = {n: i % 2 for i, n in enumerate(NAMES)}
FORKS = []
for name in NAMES:
    for j in range(70):
        lab = 0 if j >= 40 else (2 if j < 12 else 1)      # 12 TP, 28 FP, 30 uncounted
        f = rng.normal(size=NF)
        f[0] = rng.normal(2.0 if lab == 2 else 0.0)       # column 0 carries the label
        FORKS.append((name, FOLD_OF[name], j, 1000 + j, lab, f.tolist()))
X_ALL = np.array([r[5] for r in FORKS], float)
LAB_ALL = np.array([r[4] for r in FORKS], int)
FLD_ALL = np.array([r[1] for r in FORKS], int)
CNT_ALL = LAB_ALL >= 1
BY_MOVIE = defaultdict(list)
for i, r in enumerate(FORKS):
    BY_MOVIE[r[0]].append((i, r[2], r[3]))
N_TR = {k: int(((FLD_ALL != k) & CNT_ALL).sum()) for k in (0, 1)}
print(f"  {len(FORKS)} forks, {int(CNT_ALL.sum())} counted, {int((LAB_ALL == 2).sum())} TP;"
      f" training rows per fold {N_TR}")

# every fork edge is in the snapshot's key map except two, to exercise the warning
KEY = {(r[2], r[3]): i for i, r in enumerate(FORKS)}
del KEY[(0, 1000)], KEY[(1, 1001)]
MOVIE = {n: {"key": KEY} for n in NAMES}
PER = {n: {"tag": "REF", "name": n} for n in NAMES}

FITS = []
RESCORED = []
WARNED = []


class Rec(HistGradientBoostingClassifier):
    """The real estimator, with the shape of what it is actually handed recorded."""

    def fit(self, X, y, **kw):
        FITS.append((tuple(np.shape(X)), int(np.asarray(y).sum())))
        return super().fit(X, y, **kw)


def _rescore(name, snap, drop_rows, tag):
    RESCORED.append((name, tuple(sorted(drop_rows)), tag))
    return {"tag": tag, "name": name, "dropped": len(drop_rows)}


NS = dict(np=np, defaultdict=defaultdict, roc_auc_score=roc_auc_score,
          HistGradientBoostingClassifier=Rec,
          FORKS=FORKS, X_ALL=X_ALL, LAB_ALL=LAB_ALL, FLD_ALL=FLD_ALL,
          CNT_ALL=CNT_ALL, BY_MOVIE=BY_MOVIE, MOVIE=MOVIE, PER=PER,
          SIX_OK=NAMES, FOLD_OF=FOLD_OF, J20=list(range(20)),
          _rescore=_rescore,
          warn_once=lambda k, m: WARNED.append((k, m)),
          _with_ratio=lambda rows: {"score": float(len(rows)), "rows": rows})
TAIL = "return float(roc_auc_score(y, p))"
exec(cut("def _auc(y, p):", TAIL, "_auc") + TAIL + "\n", NS)
exec(cut("def _gate_rows(", "\n_rng8 = ", "_gate_rows"), NS)
gate = NS["_gate_rows"]

# ---------------------------------------------------------------------------
print("\n3. THE POINT OF THIS TEST: does `add` reach the estimator?")
# ---------------------------------------------------------------------------
FITS.clear()
_rc, _rows, _pf, _aucs, p_plain = gate("plain", 0.5)
shapes_plain = [s for s, _ in FITS]
assert [s[0] for s in shapes_plain] == [N_TR[0], N_TR[1]], shapes_plain
assert all(s[1] == 20 for s in shapes_plain), shapes_plain
print(f"  no add:  fits {shapes_plain}   (expected {[(N_TR[0], 20), (N_TR[1], 20)]})")

FITS.clear()
_rc, _rows, _pf, _aucs, p_add = gate("add", 0.5, add=(X44, Y44))
shapes_add = [s for s, _ in FITS]
assert [s[0] for s in shapes_add] == [N_TR[0] + 67, N_TR[1] + 67], shapes_add
tp_add = [t for _, t in FITS]
tp_plain_expect = [int((LAB_ALL[(FLD_ALL != k) & CNT_ALL] == 2).sum()) for k in (0, 1)]
assert tp_add == [t + 6 for t in tp_plain_expect], (tp_add, tp_plain_expect)
print(f"  with add: fits {shapes_add}   positives {tp_add} "
      f"(= {tp_plain_expect} + 6)   <- the 67 rows and their 6 TP arrived")
assert not np.allclose(p_plain, p_add), \
    "FATAL: adding 67 labelled rows did not change a single probability -- the " \
    "arm would silently return the control"
n_moved = int((~np.isclose(p_plain, p_add)).sum())
print(f"  probabilities changed on {n_moved} of {len(p_plain)} forks")

# ---------------------------------------------------------------------------
print("\n4. the controls have to differ from the candidate, not just from plain")
# ---------------------------------------------------------------------------
y_shuf = Y44.copy()
np.random.default_rng(20260908).shuffle(y_shuf)
assert not np.array_equal(y_shuf, Y44), "the shuffle is a no-op with this seed"
_r, _w, _pf, _a, p_shuf = gate("shuf", 0.5, add=(X44, y_shuf))
assert not np.allclose(p_shuf, p_add), \
    "FATAL: shuffling the added labels changed nothing -- control (E) is dead"
print(f"  shuf differs from joint on "
      f"{int((~np.isclose(p_shuf, p_add)).sum())} forks")

CALLED = []


def pad_same(kk, tr):
    CALLED.append((kk, int(tr.sum())))
    ii = np.where(tr)[0]
    pick = np.random.default_rng(1000 + kk).choice(ii, size=67, replace=True)
    return X_ALL[pick], (LAB_ALL[pick] == 2).astype(int)


FITS.clear()
_r, _w, _pf, _a, p_pad = gate("pad", 0.5, per_fold_add=pad_same)
assert CALLED == [(0, N_TR[0]), (1, N_TR[1])], CALLED
assert [s[0] for s, _ in FITS] == [N_TR[0] + 67, N_TR[1] + 67], FITS
assert not np.allclose(p_pad, p_plain), "FATAL: per_fold_add did not reach the fit"
print(f"  per_fold_add called as {CALLED}, fits grew by 67 in each fold")

# `add` must win over `per_fold_add` when both are given -- the arms never do
# that, but a silent precedence bug here would make two arms identical
FITS.clear()
CALLED.clear()
gate("both", 0.5, add=(X44, Y44), per_fold_add=pad_same)
assert CALLED == [], "per_fold_add ran even though add was given"
print("  add takes precedence over per_fold_add, and per_fold_add is not called")

# ---------------------------------------------------------------------------
print("\n5. the pruning path: only forks at or below the cut are dropped")
# ---------------------------------------------------------------------------
RESCORED.clear()
WARNED.clear()
thr = float(np.median(p_add))
_rc, rows, pf, _a, p_t = gate("cut", thr, add=(X44, Y44))
want = {}
for name in NAMES:
    idx = [i for i, _p, _q in BY_MOVIE[name]
           if np.isfinite(p_t[i]) and p_t[i] <= thr]
    want[name] = tuple(sorted(KEY[(FORKS[i][2], FORKS[i][3])]
                              for i in idx if (FORKS[i][2], FORKS[i][3]) in KEY))
got = {n: d for n, d, _t in RESCORED}
for name in NAMES:
    if want[name]:
        assert got[name] == want[name], (name, len(got.get(name, ())), len(want[name]))
    else:
        assert name not in got, f"{name} was rescored with an empty drop set"
kept = [i for i in range(len(FORKS)) if np.isfinite(p_t[i]) and p_t[i] > thr]
assert kept, "the cut removed everything -- the test proves nothing"
for name in NAMES:
    for i, _p, _q in BY_MOVIE[name]:
        row = KEY.get((FORKS[i][2], FORKS[i][3]))
        if row is not None and np.isfinite(p_t[i]) and p_t[i] > thr:
            assert row not in got.get(name, ()), \
                f"a fork ABOVE the cut was pruned in {name}"
print(f"  cut at p <= {thr:.4f}: "
      f"{sum(len(v) for v in want.values())} edges dropped, "
      f"{len(kept)} forks kept, none of them pruned")
assert len(rows) == len(NAMES) and sum(len(v) for v in pf.values()) == len(NAMES)
print(f"  {len(rows)} per-movie rows returned, split {len(pf[0])}/{len(pf[1])} by fold")

# the two edges deliberately missing from the key map must raise the warning,
# not vanish silently -- the cut is then a LOWER bound and the log has to say so
assert any(k == "prune_edge_missing" for k, _m in WARNED), \
    "fork edges absent from the snapshot's map were dropped without a warning"
print(f"  the {len(WARNED)} unmapped fork edges warned rather than passing silently")

print("\nALL CHECKS PASSED for", TARGET)
