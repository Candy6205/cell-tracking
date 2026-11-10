"""hp44pair -- hpara on ALL 71 44b6 movies, paired.

The 44b6 arm of the hpara ledger has been the weak one and the user said why:
*"思考，把44b6全跑一遍不行吗，你这分裂全挂0"*. The seeded 12-movie draw carries
**2 GT divisions out of the 26 that exist across all 71**, so its division
channel could not move and its adj_J interval was
−0.0002 [−0.0064, +0.0051], P 0.509 -- a null on a sample too small to be
anything else. `hp44full` / `hp44full_ref` run all 71.

    hp44full_F4.csv    control, refine placement (what ships)
    hp44full_F4HP.csv  treatment, parabolic-vertex sub-voxel placement

Same detections, same order, same count, same GT, same solver -- hpara moves
each point INSIDE its voxel and nothing else.

ANCHOR (the control's, from divbench5 v2's recorded REF over all 71, which is
a stronger anchor than divbench5's own 12-movie rows):
    div 6/61/20   div_J 0.0690   adj_J 0.8868   score 0.8937

⚠ WHAT A WIN HERE WOULD AND WOULD NOT MEAN. The three-domain rule says a
change counts only if the sign agrees on 6bba, 44b6 and zebrahub. 6bba reads
+0.0031 [+0.0014, +0.0049] P 0.998 and zebrahub +0.0011 [+0.0005, +0.0017]
P 0.998 on 38 of 41 crops. 44b6 is the one that has never had the statistical
power to agree or disagree. This run gives it that power; it does not change
the rule.

    py tools/hp44pair.py [ref.csv] [hp.csv]
"""
import sys

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
import benchagg as BA  # noqa: E402

REF = sys.argv[1] if len(sys.argv) > 1 else "bench_csv/hp44full_F4.csv"
HP = sys.argv[2] if len(sys.argv) > 2 else "bench_csv/hp44full_F4HP.csv"

ANCHOR = dict(adj_edge_jaccard=0.8868, division_jaccard=0.0690, score=0.8937)
CELLS = (6, 61, 20)
N = 71

A, H = BA.load(REF), BA.load(HP)
NAMES = sorted(set(A) & set(H))
print(f"{len(A)} control rows, {len(H)} hpara rows, {len(NAMES)} paired")
assert len(NAMES) == N, (len(NAMES), N)

print("\n=== ANCHOR: the CONTROL must reproduce divbench5 v2 over all 71 ===")
gA = BA.check_anchor("F4", [A[n] for n in NAMES], ANCHOR, CELLS)
gH = BA.agg([H[n] for n in NAMES])

print(f"\n=== UNGATED aggregate, {N} 44b6 movies ===")
print(f"  {'':<10}" + "".join(f"{k:>22}" for k in BA.KEYS))
print(f"  {'F4':<10}" + "".join(f"{gA[k]:>22.4f}" for k in BA.KEYS))
print(f"  {'F4HP':<10}" + "".join(f"{gH[k]:>22.4f}" for k in BA.KEYS))
print(f"  {'delta':<10}" + "".join(f"{gH[k] - gA[k]:>+22.4f}" for k in BA.KEYS))
print(f"  divisions  F4 {BA.div_cells([A[n] for n in NAMES])}   "
      f"F4HP {BA.div_cells([H[n] for n in NAMES])}")
print(f"  (the 12-movie draw had 2 GT divisions; all {N} carry 26)")

print(f"\n=== PAIRED BOOTSTRAP, the SAME movie resample for both arms ===")
BA.print_bootstrap(BA.paired_bootstrap(A, H, NAMES))

print(f"\n=== PER-MOVIE SIGN ===")
BA.print_per_movie(A, H, NAMES)

print(f"\n=== CONCENTRATION ===")
BA.print_concentration(A, H, NAMES)

print(f"""
=== THE READING, fixed before the numbers ===
  The 12-movie draw read −0.0002 [−0.0064, +0.0051], P 0.509. Three outcomes:
  1 positive, interval excludes 0 -> 44b6 AGREES with 6bba and zebrahub, the
    three-domain rule is satisfied, and hpara becomes a shippable change on
    all three. The 12-movie null was a power problem, as suspected.
  2 still null on 71 -> the effect is genuinely absent on this instrument,
    and the three-domain rule BLOCKS hpara. That is a real answer, not a
    failure: it would be the first change to pass two domains and fail the
    third on adequate power.
  3 negative, interval excludes 0 -> 44b6 disagrees in sign. hpara is a
    per-instrument change, and the ledger's other two arms need re-reading
    for what they share that 44b6 does not (voxel anisotropy is the obvious
    candidate and is measurable from the caches).
  ⚠ Read div_J and adj_J separately. adj_J enters the score at 1.0 against
  div_J's 0.1, and the WHOLE reason for running all 71 was the division
  channel -- an adj_J move with div_J flat means the extra movies bought
  power, not the divisions this was run for.""")
