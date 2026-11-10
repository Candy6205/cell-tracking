"""rkhppair -- the 2x2, paired. Do hpara and the retrain add?

Four cells, same 128 6bba movies, same folds, same ILP configuration, all
ungated REF at DIV_W 0.55 / floor 0.48:

    A  (BASE ranker, refine)      rkpool_BASE.csv        what ships
    B  (F4   ranker, refine)      rkpool_F4.csv          the retrain alone
    C  (BASE ranker, hpara )      hparabench_F4HP.csv    hpara alone
    D  (F4/hp ranker, hpara )     rkhp_F4_F4HP.csv       BOTH, matched

Two of those (A, C) are anchored to recorded values before anything is
compared. D's own harness was proven against C inside the kernel:
`ANCHOR_D187` read 0.9204 / div_J 0.0667 and the `+BIDIR pkl` arm 0.9252 /
0.1389 -- hparabench round 2's F4HP values exactly, on fixed pkls that cannot
see RANK_TRAIN. That is what licenses pairing D against C's recorded CSV.

    py tools/rkhppair.py
"""
import sys

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
import benchagg as BA  # noqa: E402

CELLS = [
    ("A ships   (BASE rk, refine)", "bench_csv/rkpool_BASE.csv",
     dict(adj_edge_jaccard=0.9133, division_jaccard=0.1337, score=0.9266),
     (50, 249, 75)),
    ("B retrain (F4   rk, refine)", "bench_csv/rkpool_F4.csv", None, None),
    ("C hpara   (BASE rk, hpara )", "bench_csv/hparabench_F4HP.csv",
     dict(adj_edge_jaccard=0.9146, division_jaccard=0.1519, score=0.9298),
     (55, 237, 70)),
    ("D BOTH    (F4/hp rk, hpara)", "bench_csv/rkhp_F4_F4HP.csv", None, None),
]

D = {}
NAMES = None
print("=== ANCHORS: the two cells that have recorded values must reproduce "
      "them ===")
for tag, path, want, cells in CELLS:
    D[tag] = BA.load(path)
    NAMES = sorted(set(D[tag]) if NAMES is None else set(NAMES) & set(D[tag]))
assert len(NAMES) == 128, len(NAMES)
for tag, path, want, cells in CELLS:
    if want:
        BA.check_anchor(tag[0], [D[tag][n] for n in NAMES], want, cells)
print("  the aggregation is the scorer's and both records reproduce")

print(f"\n=== THE 2x2, ungated REF, {len(NAMES)} movies ===")
print(f"  {'cell':<30}" + "".join(f"{k:>20}" for k in BA.KEYS)
      + f"{'divisions':>16}")
G = {}
for tag, *_ in CELLS:
    rows = [D[tag][n] for n in NAMES]
    G[tag] = BA.agg(rows)
    c = BA.div_cells(rows)
    print(f"  {tag:<30}" + "".join(f"{G[tag][k]:>20.4f}" for k in BA.KEYS)
          + f"{f'{c[0]}/{c[1]}/{c[2]}':>16}")

A, Bc, C, Dc = [t for t, *_ in CELLS]
print(f"\n  vs A (what ships):   B {G[Bc]['score'] - G[A]['score']:+.4f}   "
      f"C {G[C]['score'] - G[A]['score']:+.4f}   "
      f"D {G[Dc]['score'] - G[A]['score']:+.4f}")
print(f"  if the two were additive D would be "
      f"{G[A]['score'] + (G[Bc]['score'] - G[A]['score']) + (G[C]['score'] - G[A]['score']):.4f}; "
      f"it is {G[Dc]['score']:.4f}")

PAIRS = [(A, Bc, "the retrain alone, against what ships"),
         (A, C, "hpara alone, against what ships"),
         (A, Dc, "BOTH, against what ships"),
         (C, Dc, "BOTH, against hpara alone  <- the pre-registered one"),
         (Bc, Dc, "BOTH, against the retrain alone")]
for lo, hi, why in PAIRS:
    print(f"\n=== {hi.split()[0]} - {lo.split()[0]}: {why} ===")
    BA.print_bootstrap(BA.paired_bootstrap(D[lo], D[hi], NAMES))
