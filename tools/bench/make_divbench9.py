"""divbench6.py -> divbench9.py: the CROSS. Does DIV_W 0.35 and the joint fork
gate ADD, or do they fix the same forks twice?

WHY. Two things landed on 2026-09-08, both measured on the same 128 movies
with the official scorer and real pruning:

  divbench6  DIV_W 0.55 -> 0.35, gated  +0.0015  (both folds, 50/68/75)
  divbench8  the joint gate            +0.0037  (both folds, 46/39/79,
                                                 90% [+0.0016, +0.0060])

They act on the SAME object. `DIV_W` changes which forks the ILP produces;
the gate changes which of the produced forks survive. If the joint gate's
advantage is that it recognises good ILP forks, then lowering DIV_W -- which
makes 3,000 more ILP forks -- either hands it more of exactly what it is good
at (they add) or hands it the same divisions it was already recovering
(they overlap). Nothing measured so far says which, and 2026-08-26 says cross
the knobs whose mechanisms touch rather than sweeping them one at a time.

THE DESIGN. Two solves, four cells, and THREE OF THE FOUR ARE ALREADY
MEASURED, so this run carries three independent anchors and produces exactly
one new number:

                  gate6 (6bba only)          joint (+ 44b6's 67)
  REF  (0.55)     0.9387  42/47/83   [A]     0.9424  46/39/79   [A]
  dw35 (0.35)     0.9402  50/68/75   [A]     ???                 <- the answer

  [A] = anchor. A is divbench3/divbench8, B is divbench8, C is divbench6.

If the bottom-right lands near 0.9424 + 0.0015 = 0.9439, they add. If it
lands near 0.9424, DIV_W buys nothing once the gate is good enough, and the
divbench6 verdict is absorbed rather than stacked. Either is worth knowing
before a second submission is spent.

WHAT IT COSTS. `DIV_W` is solve-time, so REF and dw35 are two full solves over
128 movies; divbench6 did four in 240 min, so expect ~120. The four gate
passes over the snapshots are cheap.

CARRIED OVER FROM divbench8, because divbench6 had none of it: the 67 counted
44b6 forks embedded in the source, `add=` on the gate trainer, and the paired
bootstrap over movies on the official score -- divbench6's +0.0015 came with
no interval and sits inside the bench's own +-0.03 div_J bar, which is the
single weakest thing about it.

ANCHORS: REF ungated 50/249/75, div_J 0.1337, adj_J 0.9133, score 0.9266;
REF+gate6 42/47/83, 0.2442, 0.9143, 0.9387; REF+joint 46/39/79, 0.2805,
0.9143, 0.9424; dw35 ungated 62/371/63, 0.1250, 0.9130, 0.9255; dw35+gate6
50/68/75, 0.2591, 0.9143, 0.9402.
"""
import os

src = open("divbench6.py", encoding="utf-8").read()
SC = ("C:/Users/48688/AppData/Local/Temp/claude/D--programming-cell-tracking/"
      "6567fffd-808a-4ffe-8bc2-6ab712560e47/scratchpad/")


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:80])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-07c-divbench6"', 'SCRIPT_VERSION = "2026-09-08b-divbench9"')

# ---- one arm, not three: the cross needs dw35 and the budget for two gates -
rep('DIVW_ARMS = [("dw45", 0.45), ("dw35", 0.35), ("dw20", 0.20)]',
    'DIVW_ARMS = [("dw35", 0.35)]   # divbench6\'s winner: +0.0015, both folds up')

# ---- the 44b6 counted forks, embedded (identical block to divbench8) -------
BLK = open(SC + "blk44.py", encoding="utf-8").read()
assert BLK.count("\n    (") == 67, BLK.count("\n    (")
rep("""FEAT_NAMES = FEAT20 + FEAT_NEW      # the first 20 are divbench3's, in order""",
    """FEAT_NAMES = FEAT20 + FEAT_NEW      # the first 20 are divbench3's, in order

# ---------------------------------------------------------------------------
# The 67 COUNTED forks from 44b6's 71 movies (61 FP + 6 TP), lifted verbatim
# out of divbench5's forks_44b6.csv -- the same literal divbench8 used, which
# is what makes its +0.0037 and this run's joint column the same measurement.
# ---------------------------------------------------------------------------
""" + BLK + '''assert FEAT44 == FEAT_NAMES, "the 44b6 fork table's columns are not this run's"
X44 = np.array([r_[1] for r_ in ROWS44], float)
Y44 = np.array([1 if r_[0] == 2 else 0 for r_ in ROWS44], int)
assert X44.shape == (67, len(FEAT_NAMES)), X44.shape
assert int(Y44.sum()) == 6, int(Y44.sum())''')

# ---- the gate trainer gains the 44b6 rows ---------------------------------
rep('''def _gate_and_prune(tag, forks_a, movies_a, per_a, thr):''',
    '''def _gate_and_prune(tag, forks_a, movies_a, per_a, thr, add=None):''')
rep('''        mg.fit(Xa[tr][:, J20], (la[tr] == 2).astype(int))''',
    '''        Xtr_, ytr_ = Xa[tr][:, J20], (la[tr] == 2).astype(int)
        if add is not None and len(add[0]):
            Xtr_ = np.vstack([Xtr_, np.asarray(add[0], float)[:, J20]])
            ytr_ = np.concatenate([ytr_, np.asarray(add[1], int)])
        mg.fit(Xtr_, ytr_)''')

# ---- the cross replaces divbench6's single gated table --------------------
rep(r'''print(f"\n=== THE SAME ARMS AFTER THE FORK GATE (trained out of fold on THAT ARM'S own "
      f"counted forks, 20 features, 2 folds -- the arm changes the fork population, so "
      f"the gate has to be refitted per arm) ===", flush=True)
print(f"  {'arm / cut':<24} {'div tp/fp/fn':>14} {'div_J':>7} {'adj_J':>7} {'score':>7} "
      f"{'d adj_J':>8} {'d score':>8} {'vs REF+gate':>12} {'f0':>8} {'f1':>8}", flush=True)''',
    r'''print(f"\n=== THE CROSS: {len(DIVW_ARMS) + 1} DIV_W arms x 2 gates, real pruning at "
      f"p > 0.03, official scorer. Each gate is trained OUT OF FOLD on that arm's own "
      f"counted forks -- the arm changes the fork population -- and the joint column "
      f"adds divbench5's 67 counted 44b6 forks to both training folds ===", flush=True)
print(f"  {'cell':<20} {'div tp/fp/fn':>14} {'div_J':>7} {'adj_J':>7} {'score':>7} "
      f"{'vs REF+gate6':>13} {'f0':>8} {'f1':>8}   AUC / anchor", flush=True)''')

# The splice must start AFTER `def _gate_and_prune`, which sits between the
# header print and the arm loop -- an earlier boundary swallowed the very
# function the replacement block calls.
a = src.index("THR = 0.03", src.index("def _gate_and_prune"))
b = src.index('print("\\n=== reading it ===", flush=True)')
CROSS = r'''THR = 0.03
CELLS = []          # (label, arm, gate) in the order they are printed
for _arm, _fa, _ma, _pa in ([("REF", FORKS, MOVIE, PER)]
                            + [(a_, ARM_FORKS[a_], ARM_MOVIE[a_], ARM_PER[a_])
                               for a_, _d in DIVW_ARMS]):
    for _gn, _add in (("gate6", None), ("joint", (X44, Y44))):
        CELLS.append((f"{_arm} + {_gn}", _arm, _gn, _fa, _ma, _pa, _add))
WANT = {"REF + gate6": "want 42/47/83 0.2442 0.9143 0.9387 (divbench3)",
        "REF + joint": "want 46/39/79 0.2805 0.9143 0.9424 (divbench8)",
        "dw35 + gate6": "want 50/68/75 0.2591 0.9143 0.9402 (divbench6)",
        "dw35 + joint": "<-- THE NEW NUMBER"}
XRES = {}
for lab, _arm, _gn, _fa, _ma, _pa, _add in CELLS:
    rc, rows2, pf2, aucs = _gate_and_prune(lab.replace(" ", ""), _fa, _ma, _pa, THR,
                                           add=_add)
    XRES[lab] = dict(rc=rc, rows=rows2, pf=pf2)
    base = XRES.get("REF + gate6")
    f0, f1 = _with_ratio(pf2[0]), _with_ratio(pf2[1])
    d = rc["score"] - base["rc"]["score"]
    df0 = f0["score"] - _with_ratio(base["pf"][0])["score"]
    df1 = f1["score"] - _with_ratio(base["pf"][1])["score"]
    print(f"  {lab:<20} "
          f"{sum(r['division_tp'] for r in rows2):>4}/"
          f"{sum(r['division_fp'] for r in rows2):>4}/"
          f"{sum(r['division_fn'] for r in rows2):<4} {rc['division_jaccard']:>7.4f} "
          f"{rc['adj_edge_jaccard']:>7.4f} {rc['score']:>7.4f} {d:>+13.4f} "
          f"{df0:>+8.4f} {df1:>+8.4f}   "
          + " ".join(f"{n}:{a:.4f}" for n, a in aucs)
          + "   " + WANT.get(lab, ""), flush=True)

print(f"\n  ADDITIVITY. If DIV_W and the joint gate fix different forks, the bottom "
      f"right equals the sum of the two single moves; if they fix the same ones, it "
      f"equals the larger of them.", flush=True)
_s = {k: XRES[k]["rc"]["score"] for k in XRES}
if "dw35 + joint" in _s:
    _dw = _s["dw35 + gate6"] - _s["REF + gate6"]
    _jt = _s["REF + joint"] - _s["REF + gate6"]
    _both = _s["dw35 + joint"] - _s["REF + gate6"]
    print(f"    DIV_W alone {_dw:+.4f}   joint alone {_jt:+.4f}   "
          f"sum if independent {_dw + _jt:+.4f}   larger alone {max(_dw, _jt):+.4f}",
          flush=True)
    print(f"    MEASURED TOGETHER {_both:+.4f}   -> "
          f"{100.0 * _both / (_dw + _jt) if (_dw + _jt) else float('nan'):.0f}% of the "
          f"independent sum", flush=True)

print(f"\n  PAIRED BOOTSTRAP over the {len(SIX_OK)} movies, 2,000 draws, on the OFFICIAL "
      f"SCORE of the pruned graphs -- the same resample for every cell, so the movie "
      f"draw cancels. divbench6's +0.0015 had no interval and this is the check on it "
      f"===", flush=True)
_rb = np.random.default_rng(9090)
_draws = [_rb.choice(len(SIX_OK), size=len(SIX_OK), replace=True) for _ in range(2000)]
BS = {k: np.array([_with_ratio([XRES[k]["rows"][i] for i in d])["score"] for d in _draws])
      for k in XRES}
print(f"    {'cell':<20} {'mean d vs REF+gate6':>20} {'90% interval':>24}  "
      f"distinguishable", flush=True)
for lab, _a, _g, _f, _m, _p, _x in CELLS:
    if lab == "REF + gate6":
        continue
    d = BS[lab] - BS["REF + gate6"]
    lo, hi = float(np.quantile(d, .05)), float(np.quantile(d, .95))
    print(f"    {lab:<20} {float(np.mean(d)):>+20.4f}   [{lo:>+8.4f}, {hi:>+8.4f}]  "
          f"{'YES' if (lo > 0 or hi < 0) else 'no'}", flush=True)
if "dw35 + joint" in BS:
    d = BS["dw35 + joint"] - BS["REF + joint"]
    lo, hi = float(np.quantile(d, .05)), float(np.quantile(d, .95))
    print(f"    {'dw35+joint vs REF+joint':<20} {float(np.mean(d)):>+13.4f}   "
          f"[{lo:>+8.4f}, {hi:>+8.4f}]  {'YES' if (lo > 0 or hi < 0) else 'no'}"
          f"   <- does DIV_W still buy anything once the gate is the joint one?",
          flush=True)

'''
src = src[:a] + CROSS + src[b:]

# ---- how to read it -------------------------------------------------------
A2 = '''print("  (1) READ THE ARMS AFTER THE GATE.'''
a2 = src.index(A2)
b2 = src.index("warning_summary()", a2)
READ = r'''print("  (1) THREE OF THE FOUR CELLS ARE ANCHORS and are printed with the number "
      "they must reproduce. If REF+gate6, REF+joint or dw35+gate6 misses, the cross is "
      "not readable and the fourth cell means nothing -- it would be the only "
      "unverified number in the table.", flush=True)
print("  (2) THE ANSWER IS THE ADDITIVITY LINE, not the raw score. DIV_W alone was "
      "+0.0015 and the joint gate alone +0.0037. Near 100% of the independent sum "
      "means they fix different forks and both belong in the next submission; near "
      "the larger alone means they fix the same ones and DIV_W is absorbed -- in "
      "which case divbench6's verdict stands as a finding about the SOLVER but buys "
      "nothing on top of a gate that is already good enough.", flush=True)
print("  (3) The bootstrap on dw35+joint against REF+joint is the direct form of the "
      "same question and needs no arithmetic: it is what DIV_W is worth GIVEN the "
      "joint gate. divbench6 measured DIV_W given the OLD gate and got +0.0015 with "
      "no interval; this is the number that would justify shipping DIV_W 0.35.",
      flush=True)
print("  (4) adj_J separately, as always -- it enters the score at 1.0 against div_J's "
      "0.1, and every cell so far has sat at 0.9143. An arm that finally moves it is "
      "doing something different from the others and should be read as such.",
      flush=True)
print("  (5) 2026-09-08: an argmax is only valid while the stages that shaped the "
      "curve are unchanged. DIV_W 0.35 was chosen against the 6bba-only gate. If the "
      "joint gate absorbs it, that is this rule firing AGAIN, one day later, on the "
      "constant this project had just finished reopening.", flush=True)
'''
src = src[:a2] + READ + src[b2:]

rep('DIVBENCH6 DONE in', 'DIVBENCH9 DONE in')
rep('print("DIVBENCH6 OK", flush=True)', 'print("DIVBENCH9 OK", flush=True)')
rep('f"[degraded] DIVBENCH6 INCOMPLETE:', 'f"[degraded] DIVBENCH9 INCOMPLETE:')

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "divbench9.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
