"""Are the missed cells PERMANENTLY invisible, or do they flicker?

The user watched 6bba_3abfe10a and said many red circles are cells that
were detected in the previous frame and not in the next, and some
"jump back and forth repeatedly".

That distinction decides what the misses are worth:

  permanent    the cell is below threshold for its whole life. A hard
               ceiling; the user's position holds -- dim is dim.
  intermittent the cell sits ON the threshold. It IS detectable, and a
               single dropout is a repairable break, not a lost cell.

And it connects to today's one surviving result. `_short_filter` at the
shipped SHORT_MIN = 6 deletes every predicted track shorter than 6
nodes. A flickering cell is exactly what produces tracks of length 3, 4
and 5 -- so if flicker is common, the filter was deleting the fragments
of real cells, which is why lowering it to 4 gained adj_J in all three
domains.

The repair that exists has a specific shape and a specific limit:

    for t in range(T - 2):
        ends   = tracks ending at t
        starts = tracks starting at t + 2

**`_gap_close` bridges a gap of exactly ONE frame.** A two-frame dropout
has nothing to repair it. So the run-length distribution of the gaps is
the number that matters, not the miss rate.

Measured here, per GT track (following the GT edges, so a track is the
真值 chain, not a predicted one):

  always / never / intermittent   how the misses are distributed
  gap run lengths                 1 is repairable today, 2+ is not
  score at the miss               the detection score of the nearest
                                  peak that was BELOW threshold -- if it
                                  sits just under 0.15 the cell was
                                  marginal, if it is near zero the cell
                                  really was invisible

⚠ Sparse GT: 6bba is 9.71% annotated and 3abfe10a about 2.3%. These are
statements about ANNOTATED cells only. And a GT track here is followed
through the GT's own edges, so it is a real cell's history, which is the
right unit for this question.
"""
import pathlib

import numpy as np
from scipy.spatial import cKDTree

SCALE = np.array([1.625, 0.40625, 0.40625])
UNET_THRESH = 0.15
CAND_THR = 0.05
MATCH_UM = 5.0
GT = pathlib.Path(r"C:\Users\48688\AppData\Local\Temp\gtx")
DET = pathlib.Path(r"C:\Users\48688\AppData\Local\Temp\erout")


def frame_keys(files, prefix):
    return [k for k in files
            if k.startswith(prefix) and k[len(prefix):].isdigit()]


names = sorted(p.stem[len("gt_"):] for p in GT.glob("gt_*.npz")
               if (DET / f"detcache_{p.stem[len('gt_'):]}.npz").exists())
print(f"  {len(names)} datasets with both GT and detections\n")

TOT = dict(always=0, never=0, inter=0)
GAPS = {}
marginal = []
allscore = []

print(f"  {'dataset':<18} {'tracks':>7} {'always':>7} {'never':>6} "
      f"{'inter':>6} {'gap1':>6} {'gap2':>6} {'gap3+':>6}")
for name in names:
    g = np.load(GT / f"gt_{name}.npz")
    d = np.load(DET / f"detcache_{name}.npz")
    T = max(int(k[1:]) for k in frame_keys(d.files, "c")) + 1

    g_t = np.asarray(g["t"], int)
    g_zyx = np.stack([np.asarray(g[c], float)
                      for c in ("z", "y", "x")], axis=1)
    ids = np.asarray(g["node_ids"], np.int64)
    row_of = {int(v): i for i, v in enumerate(ids)}

    # follow GT edges into chains. A division splits the chain; take the
    # first child, since the question is about detection continuity, not
    # lineage.
    succ = {}
    indeg = {}
    for a, b in np.asarray(g["edges"], np.int64):
        ra, rb = row_of.get(int(a)), row_of.get(int(b))
        if ra is None or rb is None:
            continue
        succ.setdefault(ra, []).append(rb)
        indeg[rb] = indeg.get(rb, 0) + 1
    starts = [i for i in range(len(g_t)) if i not in indeg]

    # per-frame kept detections
    keep = {}
    below = {}
    for t in range(T):
        c = np.asarray(d[f"c{t}"], float).reshape(-1, 3)
        s = np.asarray(d[f"s{t}"], float).reshape(-1)
        keep[t] = c[s >= UNET_THRESH]
        m = (s >= CAND_THR) & (s < UNET_THRESH)
        below[t] = (c[m], s[m])

    trees = {t: (cKDTree(keep[t] * SCALE[None, :]) if len(keep[t]) else None)
             for t in range(T)}
    btrees = {t: (cKDTree(below[t][0] * SCALE[None, :])
                  if len(below[t][0]) else None) for t in range(T)}

    cnt = dict(always=0, never=0, inter=0)
    gaps = {}
    for s0 in starts:
        chain = [s0]
        cur = s0
        while succ.get(cur):
            cur = succ[cur][0]
            chain.append(cur)
        if len(chain) < 3:
            continue
        seen = []
        for r_ in chain:
            t = int(g_t[r_])
            tr = trees.get(t)
            if tr is None:
                seen.append(False)
                continue
            dist, _ = tr.query(g_zyx[r_] * SCALE)
            ok = dist <= MATCH_UM
            seen.append(bool(ok))
            if not ok:
                bt = btrees.get(t)
                if bt is not None:
                    bd, bi = bt.query(g_zyx[r_] * SCALE)
                    if bd <= MATCH_UM:
                        marginal.append(float(below[t][1][bi]))
                    else:
                        marginal.append(0.0)
                else:
                    marginal.append(0.0)
        allscore.append(np.mean(seen))
        if all(seen):
            cnt["always"] += 1
        elif not any(seen):
            cnt["never"] += 1
        else:
            cnt["inter"] += 1
            run = 0
            for v in seen:
                if not v:
                    run += 1
                elif run:
                    gaps[run] = gaps.get(run, 0) + 1
                    run = 0
            if run:
                gaps[run] = gaps.get(run, 0) + 1
    for k in cnt:
        TOT[k] += cnt[k]
    for k, v in gaps.items():
        GAPS[k] = GAPS.get(k, 0) + v
    n = sum(cnt.values())
    if n:
        print(f"  {name:<18} {n:>7} {cnt['always']:>7} {cnt['never']:>6} "
              f"{cnt['inter']:>6} {gaps.get(1, 0):>6} {gaps.get(2, 0):>6} "
              f"{sum(v for k, v in gaps.items() if k >= 3):>6}")

n = sum(TOT.values())
print(f"\n  TOTAL over {n} GT tracks of length >= 3")
for k, lab in (("always", "detected in EVERY frame"),
               ("never", "detected in NO frame"),
               ("inter", "INTERMITTENT")):
    print(f"    {lab:<26} {TOT[k]:>6}  {100.0 * TOT[k] / max(1, n):>5.1f}%")

tg = sum(GAPS.values())
print(f"\n  {tg} dropout runs, by length:")
for k in sorted(GAPS):
    tag = "  <- _gap_close repairs ONLY this one" if k == 1 else ""
    if k <= 6:
        print(f"    {k} frame(s): {GAPS[k]:>6}  "
              f"{100.0 * GAPS[k] / max(1, tg):>5.1f}%{tag}")
long = sum(v for k, v in GAPS.items() if k > 6)
if long:
    print(f"    >6 frames: {long:>6}  {100.0 * long / max(1, tg):>5.1f}%")
rep = GAPS.get(1, 0)
print(f"\n  REPAIRABLE BY THE EXISTING PATCH: {rep} of {tg} runs "
      f"({100.0 * rep / max(1, tg):.1f}%)")

if marginal:
    m = np.array(marginal)
    print(f"\n  At the {len(m)} missed frames, the nearest peak in the "
          f"repair band [{CAND_THR}, {UNET_THRESH}):")
    print(f"    none within {MATCH_UM} um: "
          f"{100.0 * (m == 0).mean():>5.1f}%  <- genuinely nothing there")
    mm = m[m > 0]
    if len(mm):
        for q in (10, 25, 50, 75, 90):
            print(f"    score p{q:<3} {np.percentile(mm, q):.4f}")
        print(f"    A cell whose peak sits just under {UNET_THRESH} was "
              f"MARGINAL, not invisible.")

print("\n  READ IT: a high INTERMITTENT share means the misses are "
      "cells sitting on the threshold, not cells that cannot be seen -- "
      "and then the ceiling argument does not apply to them. The gap "
      "run lengths say what could be repaired: `_gap_close` bridges "
      "exactly one missing frame, so everything at 2+ is currently "
      "unrepaired and is what shatters a track into the length-3/4/5 "
      "fragments that SHORT_MIN 6 was deleting.")
