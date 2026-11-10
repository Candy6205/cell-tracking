"""divradius2 -- the same question, done properly: CROSSED, RETRAINED, and
with an interval on every number.

WHAT WAS WRONG WITH divradius (2026-09-09, same day):

1. **It tuned one radius at a time.** The user's standing instruction from
   2026-09-08 was the opposite -- "注意不同候选之间会不会互相影响" -- and the
   gate6 run already contained the warning: at parent 10.0, the sister cut
   did NOTHING (15.0 / 13.0 / 12.0 all read −0.0226 identically), because the
   parent cut had already removed every fork the sister cut would have. That
   is an interaction, it was in my own output, and the joint-gate table
   reported marginals anyway.

2. **It froze the gate.** The gate was trained on the fork population a run
   at 12.0 / 15.0 produced, and the arms only FILTERED that population
   afterwards. But the radius changes what the gate should be fitted on --
   which is exactly how divbench9 handled the DIV_W arms ("each gate is
   trained OUT OF FOLD on that arm's own counted forks -- the arm changes the
   fork population"). Here every arm retrains.

3. **It had no intervals**, and then leaned on a noise floor of
   "0.002-0.003" attributed to `accdiag`. `accdiag` never measured that; the
   number does not exist in the record. What DOES exist is progress.md:830 --
   "the bench's resolution on div_J is about ±0.03 at 90%, i.e. ±0.003 of
   score" -- measured by divbench3's paired bootstrap over GATE VARIANTS.
   That is a looser comparison than this one (a different gate reshuffles
   which forks pass; a smaller radius only removes), so it is the wrong
   ruler too. This file measures its own.

THE BOOTSTRAP IS PAIRED OVER MOVIES and uses the SAME resample for every
arm, so the movie draw cancels and what is left is the radius. FN is
resampled with the movies: each movie carries its own GT division count, so
a draw that picks a division-rich movie twice raises both the TP ceiling and
the FN floor, which is the behaviour the aggregate needs.

ANCHOR: the shipped 12.0 / 15.0 cell with the joint gate must reproduce
divbench8's 46 / 39 / 79, div_J 0.2805.

Usage:  py tools/divradius2.py path/to/forks_dw55.csv [divbench9.py]
"""
import csv
import pathlib
import re
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

FEAT20 = ["src", "d_pc1", "d_pq", "d_s", "dv", "p_c1", "p_q", "pf_c1", "pf_q",
          "pr_q", "sc_p", "sc_c1", "sc_q", "len_back", "len_c1", "len_q",
          "dens10", "n_orph", "t_frac", "rank_gap"]
CUT = 0.03
N_GT_TOTAL = 125          # GT divisions over the 128 6bba movies: 50 TP + 75 FN
ANCHOR = (46, 39, 79)
B = 2000

CSV = sys.argv[1] if len(sys.argv) > 1 else "forks_dw55.csv"
DB9 = sys.argv[2] if len(sys.argv) > 2 else "divbench9.py"

PARENTS = [12.0, 11.0, 10.0, 9.0]
SISTERS = [15.0, 13.0, 12.0, 11.0]


def f(s):
    try:
        return float(s)
    except Exception:
        return np.nan


rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
X = np.array([[f(r[c]) for c in FEAT20] for r in rows], np.float64)
lab = np.array([int(r["label"]) for r in rows])
fold = np.array([int(r["fold"]) for r in rows])
mov = np.array([r["movie"] for r in rows])
d_pq = np.array([f(r["d_pq"]) for r in rows])
d_s = np.array([f(r["d_s"]) for r in rows])
cnt = lab >= 1
MOVIES = sorted(set(mov.tolist()))
ROW_OF = {m: np.where(mov == m)[0] for m in MOVIES}
print(f"{len(rows)} forks over {len(MOVIES)} movies, {int(cnt.sum())} counted "
      f"({int((lab == 2).sum())} TP, {int((lab == 1).sum())} FP)")

# ---- the 44b6 half of the joint gate, out of divbench9's own literal ----
src9 = pathlib.Path(DB9).read_text(encoding="utf-8")
_i = src9.index("ROWS44 = [")
_ns = {"np": np}
exec(src9[_i:src9.index("\n]", _i) + 2], _ns)
FEAT44 = eval(re.search(r"FEAT44 = (\[.*?\])\n", src9, re.S).group(1))
_ix = [FEAT44.index(c) for c in FEAT20]
X44 = np.array([[b[i] for i in _ix] for _, b in _ns["ROWS44"]], np.float64)
y44 = np.array([1 if a == 2 else 0 for a, _ in _ns["ROWS44"]])
print(f"joint gate adds {len(y44)} 44b6 counted forks ({int(y44.sum())} TP) "
      f"to both training folds")

# ---- FN per movie, so the bootstrap can resample it -----------------------
# A movie's GT divisions = the true forks it produced + the ones nothing
# proposed. The per-movie split of the 75 unproposed is not in this table, so
# they are apportioned by the movie's share of counted forks -- an
# approximation that affects the WIDTH of the interval, never the point
# estimate (which uses the exact total).
tp_by_mov = np.array([int((lab[ROW_OF[m]] == 2).sum()) for m in MOVIES])
w = np.array([len(ROW_OF[m]) for m in MOVIES], float)
FN_UNPROPOSED = (N_GT_TOTAL - int((lab == 2).sum())) * w / w.sum()


def gate_p(keep):
    """Out-of-fold gate probability, RETRAINED on the arm's own surviving
    fork population.  `keep` is the arm's radius mask: a fork the arm never
    proposes is not in its training set either."""
    p = np.full(len(rows), np.nan)
    tr_pool = cnt & keep
    for k in (0, 1):
        tr = (fold != k) & tr_pool
        m = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05,
                                           min_samples_leaf=10, random_state=0)
        m.fit(np.vstack([X[tr], X44]),
              np.concatenate([(lab[tr] == 2).astype(int), y44]))
        sel = (fold == k) & keep
        if sel.any():
            p[sel] = m.predict_proba(X[sel])[:, 1]
    return p


ARMS = [(pu, su) for pu in PARENTS for su in SISTERS]
PASS = {}
print(f"\nretraining the joint gate for each of the {len(ARMS)} cells "
      f"(2 folds each)...", flush=True)
for pu, su in ARMS:
    keep = (d_pq <= pu) & (d_s <= su)
    PASS[(pu, su)] = keep & (gate_p(keep) > CUT)


def agg(mask, movies=None, counts=None):
    """div TP / FP / FN / J over a movie multiset."""
    if movies is None:
        tp = int(((lab == 2) & mask).sum())
        fp = int(((lab == 1) & mask).sum())
        fn = N_GT_TOTAL - tp
        return tp, fp, fn, tp / max(1, tp + fp + fn)
    tp = fp = 0.0
    fn = 0.0
    for mi, c in zip(movies, counts):
        r = ROW_OF[MOVIES[mi]]
        tp += c * int(((lab[r] == 2) & mask[r]).sum())
        fp += c * int(((lab[r] == 1) & mask[r]).sum())
        fn += c * (tp_by_mov[mi] + FN_UNPROPOSED[mi])
    fn = fn - tp
    return tp, fp, fn, tp / max(1e-9, tp + fp + fn)


base = agg(PASS[(12.0, 15.0)])
ok = base[:3] == ANCHOR
print(f"\nANCHOR  parent 12.0 / sister 15.0 (SHIPS): {base[0]}/{base[1]}/"
      f"{base[2]}  div_J {base[3]:.4f}   want {'/'.join(map(str, ANCHOR))} "
      f"0.2805 (divbench8) -> {'OK' if ok else 'MISS'}")
assert ok, "not the gate that ships; nothing below is readable"

print(f"\n=== THE CROSS, gate RETRAINED per cell. div_J, and (d score) ===")
print(f"  {'parent \\\\ sister':<18}" + "".join(f"{s:>18.1f}" for s in SISTERS))
for pu in PARENTS:
    line = f"  {pu:<18.1f}"
    for su in SISTERS:
        t_, f_, n_, j_ = agg(PASS[(pu, su)])
        line += f"{f'{j_:.4f} ({0.1 * (j_ - base[3]):+.4f})':>18}"
    print(line)
print("  ROW = parent radius, COLUMN = sister radius. Shipped is the top-left "
      "cell. Read the ROWS against each other, not the marginals: if the "
      "sister column is flat inside a row, the parent cut has already taken "
      "the forks the sister cut would have.")

print(f"\n=== TP / FP / FN per cell ===")
print(f"  {'parent \\\\ sister':<18}" + "".join(f"{s:>16.1f}" for s in SISTERS))
for pu in PARENTS:
    line = f"  {pu:<18.1f}"
    for su in SISTERS:
        t_, f_, n_, _ = agg(PASS[(pu, su)])
        line += f"{f'{t_}/{f_}/{n_}':>16}"
    print(line)

# ---- paired bootstrap over movies, one resample shared by every cell -----
print(f"\n=== PAIRED BOOTSTRAP over the {len(MOVIES)} movies, {B} draws, the "
      f"SAME resample for every cell ===", flush=True)
rng = np.random.default_rng(0)
DRAWS = []
for _ in range(B):
    idx = rng.integers(0, len(MOVIES), len(MOVIES))
    u, c = np.unique(idx, return_counts=True)
    DRAWS.append((u, c.astype(float)))
BJ = {}
for key in ARMS:
    BJ[key] = np.array([agg(PASS[key], u, c)[3] for u, c in DRAWS])
b0 = BJ[(12.0, 15.0)]
print(f"  {'arm':<26}{'d div_J':>10}{'90% interval':>26}{'d score':>10}"
      f"{'P(>0)':>8}")
for key in ARMS:
    if key == (12.0, 15.0):
        continue
    d = BJ[key] - b0
    lo, hi = np.percentile(d, [5, 95])
    tag = f"parent {key[0]} sister {key[1]}"
    print(f"  {tag:<26}{np.mean(d):>+10.4f}"
          f"{f'[{lo:+.4f}, {hi:+.4f}]':>26}{0.1 * np.mean(d):>+10.5f}"
          f"{np.mean(d > 0):>8.3f}")
print(f"\n  For scale, the SHIPPED cell's own sampling spread: div_J "
      f"{base[3]:.4f}, bootstrap 90% "
      f"[{np.percentile(b0, 5):.4f}, {np.percentile(b0, 95):.4f}] "
      f"(width {np.percentile(b0, 95) - np.percentile(b0, 5):.4f}). That is "
      f"the ABSOLUTE spread; the paired deltas above are much tighter "
      f"because the movie draw cancels.")
print("  progress.md:830 recorded this bench's div_J resolution as ±0.03 at "
      "90% -- measured over GATE VARIANTS, which reshuffle which forks pass. "
      "A radius only REMOVES forks, so the paired comparison is tighter and "
      "that number is the wrong ruler here. This is the right one.")
