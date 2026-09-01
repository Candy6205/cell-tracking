"""Generate velwhy.py from rankwhy.py by asserted replacement.

THE QUESTION. rankwhy measured that when the correct pair is not the
ranker's first choice, the target it prefers is CLOSER in 97-100% of
cases -- median source-to-preferred 2.13 um against source-to-true 5.89
um on zebrahub -- and on zebrahub 57.2% of those preferred targets are
another track's real successor. The ranker is choosing the near cell and
the true successor is the far one.

The ranker already owns the information that should break that tie. Its
column 4 is `md`, the distance from the source's VELOCITY-PREDICTED
position rather than from the source itself, and column 16 is the
velocity magnitude. So either the model under-weights a motion signal it
has, or the velocity it was handed is not a velocity.

The second is a live possibility with a number behind it. `vels` come
from `pass1_tracks`, a greedy nearest-neighbour chain, and `accdiag`
already measured that chain failing to supply a parent 316,658 times.
A source with no backward link gets velocity 0, and at velocity 0 the
predicted position IS the source position, so `md` collapses to plain
distance and the near cell wins by construction.

WHAT THIS MEASURES, on exactly rankwhy's rank >= 1 population:
  nv            the source's velocity magnitude, and how often it is 0
  rank by d     where the true target sits when candidates are ordered
                by raw distance
  rank by md    where it sits when ordered by motion-predicted distance
  the crossover how often md-ordering puts the true target FIRST while
                the ranker did not

If md-ordering recovers a large share, the motion signal is present and
under-weighted, and the repair is in the model. If it does not, and nv
is 0 for most of these sources, the velocity itself is missing and the
repair is upstream: build the velocities from the ILP's solved tracks
instead of from the greedy chain. That second route is already recorded
as untried in progress.md's biohub-accel entry.

Same harness, same three domains, same anchors as rankwhy.
"""
import ast
import pathlib

SRC = pathlib.Path(__file__).parent / "rankwhy.py"
DST = pathlib.Path(__file__).parent / "velwhy.py"
text = SRC.read_text(encoding="utf-8")


def rep(old, new, n=1):
    global text
    c = text.count(old)
    assert c == n, f"anchor x{c}, wanted x{n}: {old[:70]!r}"
    text = text.replace(old, new)


rep('"""rankwhy -- what does the edge ranker prefer when it puts the right '
    'target second?',
    '"""velwhy -- is the motion signal under-weighted, or is the velocity '
    'missing?')
rep('SCRIPT_VERSION = "2026-09-02b-rankwhy"',
    'SCRIPT_VERSION = "2026-09-02c-velwhy"')

# ---- 1. keep the two geometry columns alongside the probability --------
rep('''    prob_of = {}
    for t in range(T - 1):
        pairs, rows_, _P, _C = feats[t]
        if len(pairs):
            pv = MODEL.predict_proba(rows_[:, MCOLS])[:, 1]
            for (i, j), p in zip(pairs, pv):
                prob_of[(t, i, j)] = float(p)''',
    '''    prob_of = {}
    # column 0 is the plain distance, column 4 the distance from the
    # velocity-PREDICTED position, column 16 the velocity magnitude --
    # the same three numbers the ranker was given.
    d_of = {}
    md_of = {}
    nv_of = {}
    for t in range(T - 1):
        pairs, rows_, _P, _C = feats[t]
        if len(pairs):
            pv = MODEL.predict_proba(rows_[:, MCOLS])[:, 1]
            for k_, ((i, j), p) in enumerate(zip(pairs, pv)):
                prob_of[(t, i, j)] = float(p)
                d_of[(t, i, j)] = float(rows_[k_, 0])
                md_of[(t, i, j)] = float(rows_[k_, 4])
                nv_of[(t, i)] = float(rows_[k_, 16])''')

# ---- 2. the alternative orderings, on rankwhy's own population ---------
rep('''            RROWS.setdefault(domain, []).append((''',
    '''            # where would the true target sit under two orderings the
            # ranker already has as columns?
            cl = [jj for _pp, jj in lst2]
            dd_ = sorted(cl, key=lambda jj: d_of.get((tu, su[1], jj),
                                                     1e9))
            mm_ = sorted(cl, key=lambda jj: md_of.get((tu, su[1], jj),
                                                      1e9))
            rank_d = float(dd_.index(sv[1])) if sv[1] in dd_ else np.nan
            rank_md = float(mm_.index(sv[1])) if sv[1] in mm_ else np.nan
            VROWS.setdefault(domain, []).append((
                float(nv_of.get((tu, su[1]), np.nan)),
                rank_d, rank_md, float(rank_t),
                float(d_of.get((tu, su[1], sv[1]), np.nan)),
                float(md_of.get((tu, su[1], sv[1]), np.nan)),
                float(d_of.get((tu, su[1], j_top), np.nan)),
                float(md_of.get((tu, su[1], j_top), np.nan)),
                float(len(cl))))
            RROWS.setdefault(domain, []).append((''')

# ---- 3. store -----------------------------------------------------------
rep('''NROWS = {}         # domain -> one row per node_absent source''',
    '''NROWS = {}         # domain -> one row per node_absent source
VROWS = {}         # domain -> one row per rank >= 1 case, the orderings
VCOLS = ["nv", "rank_by_d", "rank_by_md", "rank_by_prob", "d_true",
         "md_true", "d_top", "md_top", "n_cand"]''')

# ---- 4. the report ------------------------------------------------------
rep('''print(f"\\nRANKWHY DONE in {(time.time() - t_all) / 60:.1f} min",
      flush=True)''',
    r'''
print("\n\n=== WAS THE VELOCITY EVEN THERE? ===", flush=True)
print("  On exactly the rank >= 1 population above. nv is the source's "
      "velocity magnitude as the ranker received it; at nv = 0 the "
      "predicted position IS the source position and `md` collapses to "
      "plain distance, so the nearest candidate wins by construction. "
      "rank_by_d and rank_by_md are where the TRUE target sits when the "
      "same candidate list is ordered by those two columns instead of "
      "by the model's probability.", flush=True)
for d in DOMAINS:
    V = VROWS.get(d)
    if not V:
        continue
    A = np.asarray(V, np.float64)
    n = len(A)
    nv = A[:, 0]
    zero = float((nv <= 1e-9).sum())
    print(f"\n  -- {d}: {n:,} cases --", flush=True)
    print(f"  velocity is EXACTLY ZERO: {int(zero):,} "
          f"({100.0 * zero / n:.1f}%)", flush=True)
    fin = nv[np.isfinite(nv) & (nv > 1e-9)]
    if len(fin):
        print(f"  where non-zero: median {np.median(fin):.2f} um  "
              f"p90 {np.percentile(fin, 90):.2f} um", flush=True)
    for lbl, col in (("by raw distance", 1), ("by predicted distance", 2)):
        c = A[:, col]
        c = c[np.isfinite(c)]
        if not len(c):
            continue
        first = float((c == 0).sum())
        print(f"  true target ranked FIRST {lbl:<22}: "
              f"{int(first):>6,} ({100.0 * first / len(c):>5.1f}%)",
              flush=True)
    both = A[np.isfinite(A[:, 1]) & np.isfinite(A[:, 2])]
    if len(both):
        fixed = float((both[:, 2] == 0).sum())
        print(f"  the predicted-distance ordering puts it first while "
              f"the model did not: {int(fixed):,} "
              f"({100.0 * fixed / len(both):.1f}% of {len(both):,})",
              flush=True)
        better = float((both[:, 2] < both[:, 1]).sum())
        worse = float((both[:, 2] > both[:, 1]).sum())
        print(f"  predicted-distance beats raw distance in "
              f"{int(better):,} ({100.0 * better / len(both):.1f}%), "
              f"loses in {int(worse):,} "
              f"({100.0 * worse / len(both):.1f}%)", flush=True)
    print(f"  medians   d_true {np.nanmedian(A[:, 4]):>6.2f}  "
          f"md_true {np.nanmedian(A[:, 5]):>6.2f}  "
          f"d_top {np.nanmedian(A[:, 6]):>6.2f}  "
          f"md_top {np.nanmedian(A[:, 7]):>6.2f}  "
          f"n_cand {np.nanmedian(A[:, 8]):>4.1f}", flush=True)

print("\n=== reading it ===", flush=True)
print("  A high zero-velocity share means the ranker was handed no "
      "motion at all for exactly the cells whose motion decides the "
      "answer, and the repair is upstream of the model: build the "
      "velocities from the ILP's solved tracks rather than from "
      "pass1_tracks' greedy chain, which biohub-accel already recorded "
      "as untried. If instead the velocity is present and the "
      "predicted-distance ordering still does not recover the true "
      "target, the motion signal does not separate these cases and "
      "neither route helps.", flush=True)
print(f"\nVELWHY DONE in {(time.time() - t_all) / 60:.1f} min",
      flush=True)''')

DST.write_text(text, encoding="utf-8")
ast.parse(text)
print(f"wrote {DST} ({len(text.splitlines())} lines), ast OK")
