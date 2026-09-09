"""fk44pair -- the RETRAIN's 44b6 score, on all 71 movies instead of 12.

The retrain (`rkpool`) is established on 128 6bba movies: score +0.0023
[+0.0007, +0.0040], P 0.990 at the shipped floor. Its 44b6 evidence is
`rkfloor`'s **12** movies, +0.0011 at floor 0.48, with no interval -- and
`hp44full` established the same day that 12 44b6 movies carry **2** of the
domain's 26 GT divisions and that the small draw misreads this domain.

`fk44f4` is `hp44full_ref` with `RANK_TRAIN="F4"`: the same 71 movies, the
same folds, the same out-of-fold structure, the same ILP; only the table the
fold rankers learn from changes, from BASE primaries to F4 ones.

    hp44full_F4.csv   control -- the fold rankers trained on BASE primaries
    fk44f4_F4.csv     treatment -- trained on the F4 primaries they solve

ANCHOR, the control's, and it is divbench5 v2's REF over all 71 rather than
any 12-movie row: div 6/61/20, div_J 0.0690, adj_J 0.8868, score 0.8937.
`hp44full_ref` already reproduced it exactly this morning, so a miss here
means the CSV is not the one that run wrote.

⚠ THE TREATMENT ARM HAS NO ANCHOR and cannot have one -- it is the new
measurement. What proves its harness is `ANCHOR_D187`, which loads a fixed
pkl the retrain does not touch and stays HARD inside the kernel at 0.8353.
Check that line in the log before reading anything below.

WHAT IT DECIDES. If the retrain is positive here, it has two domains with
adequate power and becomes comparable to hpara, which has three. If it is
null or negative on 71 while positive on 12, that is the `rkfloor` lesson
repeating on the other domain -- a small draw disagreeing with its own
superset -- and the retrain's domain evidence is 6bba alone.

    py tools/fk44pair.py [control.csv] [treatment.csv]
"""
import sys

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
import benchagg as BA  # noqa: E402

REF = sys.argv[1] if len(sys.argv) > 1 else "bench_csv/hp44full_F4.csv"
NEW = sys.argv[2] if len(sys.argv) > 2 else "bench_csv/fk44f4_F4.csv"

ANCHOR = dict(adj_edge_jaccard=0.8868, division_jaccard=0.0690, score=0.8937)
CELLS = (6, 61, 20)
N = 71

A, H = BA.load(REF), BA.load(NEW)
NAMES = sorted(set(A) & set(H))
print(f"{len(A)} control rows, {len(H)} retrain rows, {len(NAMES)} paired")
assert len(NAMES) == N, (len(NAMES), N)

print("\n=== ANCHOR: the CONTROL must reproduce divbench5 v2 over all 71 ===")
gA = BA.check_anchor("BASE", [A[n] for n in NAMES], ANCHOR, CELLS)
gH = BA.agg([H[n] for n in NAMES])

print(f"\n=== UNGATED aggregate, {N} 44b6 movies, floor 0.48 ===")
print(f"  {'':<12}" + "".join(f"{k:>22}" for k in BA.KEYS))
print(f"  {'BASE pool':<12}" + "".join(f"{gA[k]:>22.4f}" for k in BA.KEYS))
print(f"  {'F4 pool':<12}" + "".join(f"{gH[k]:>22.4f}" for k in BA.KEYS))
print(f"  {'delta':<12}" + "".join(f"{gH[k] - gA[k]:>+22.4f}" for k in BA.KEYS))
print(f"  divisions  BASE {BA.div_cells([A[n] for n in NAMES])}   "
      f"F4 {BA.div_cells([H[n] for n in NAMES])}   (26 GT divisions over 71)")

print(f"\n=== PAIRED BOOTSTRAP, the SAME movie resample for both arms ===")
BA.print_bootstrap(BA.paired_bootstrap(A, H, NAMES))

print(f"\n=== PER-MOVIE SIGN ===")
BA.print_per_movie(A, H, NAMES)

print(f"\n=== CONCENTRATION ===")
BA.print_concentration(A, H, NAMES)

print(f"""
=== READING, fixed before the numbers ===
  rkfloor's 12-movie 44b6 read at this same floor was +0.0011, no interval.
  6bba's 128-movie read is +0.0023 [+0.0007, +0.0040], P 0.990.
  1 positive here -> the retrain has two domains with real power. Together
    with its byte direction (-0.50% nodes, the 11/11 predictor) that is the
    strongest case any candidate has had tonight.
  2 null -> the retrain is a 6bba result. hpara, which passes three domains,
    is then the better-supported ship even though its byte delta is ~0.
  3 negative -> rkfloor's 12 disagreed with their own 71, which would be the
    FOURTH time tonight a small draw flipped sign against its superset, and
    the retrain's 44b6 support collapses to nothing.
  ⚠ div_J and adj_J separately. 26 GT divisions over 71 movies is this
  domain's ceiling -- hp44full's div_J interval there was [-0.011, +0.050],
  so expect the division channel to say nothing whatever it prints.""")
