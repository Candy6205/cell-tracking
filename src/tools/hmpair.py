"""hmpair -- does saddle-keyed pair merging (hmerge) change the linked graph?

Two cells, same 128 6bba movies, same folds, same +BIDIR fold rankers, same
ILP configuration, ungated REF at DIV_W 0.55 / floor 0.48:

    C  hpara placement (F4HP)          bench_csv/dupsup/dupsup_REF_F4HP.csv
    M  hpara + hmerge  (F4HM<thr>)     bench_csv/hmbench/hmbench_REF_F4HM<thr>.csv

C is anchored to hparabench round 2 (adj_J 0.9146, div 55/237/70, score
0.9298) before anything is compared. M differs from C in the detection set
only (pairs of primaries with heatmap saddle >= thr within 9.5 um replaced by
one point) and in the two edge heads re-run on that set.

Reading: edge_J is the identity effect alone (no node term); adj_J carries the
node-count term, so the node columns are printed next to it -- fewer predicted
nodes raise adj_J by themselves (the score has no upper clamp) and that part is
not a linking gain.

    py tools/hmpair.py [90|80]
"""
import sys

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
import benchagg as BA  # noqa: E402

THR = sys.argv[1] if len(sys.argv) > 1 else "90"
C_PATH = "bench_csv/dupsup/dupsup_REF_F4HP.csv"
M_PATH = sys.argv[2] if len(sys.argv) > 2 else f"bench_csv/hmbench/hmbench_REF_F4HM{THR}.csv"

C = BA.load(C_PATH)
M = BA.load(M_PATH)
NAMES = sorted(set(C) & set(M))
assert len(NAMES) == 128, (len(NAMES), len(C), len(M))
print("=== ANCHOR: the control must reproduce hparabench round 2 ===")
BA.check_anchor("C", [C[n] for n in NAMES], dict(adj_edge_jaccard=0.9146, division_jaccard=0.1519, score=0.9298),
                (55, 237, 70))

print(f"\n=== ungated REF, {len(NAMES)} movies ===")
print(f"  {'cell':<26}" + "".join(f"{k:>18}" for k in BA.KEYS) + f"{'divisions':>14}{'pred nodes':>12}{'node ratio':>12}")
G = {}
for tag, D in (("C hpara (F4HP)", C), (f"M hpara+hmerge (HM{THR})", M)):
    rows = [D[n] for n in NAMES]
    G[tag] = BA.agg(rows)
    c = BA.div_cells(rows)
    nn = int(sum(r["num_pred_nodes"] for r in rows))
    w = [r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in rows]
    nr = sum(r["total_node_ratio"] * wi for r, wi in zip(rows, w)) / sum(w)
    print(f"  {tag:<26}" + "".join(f"{G[tag][k]:>18.4f}" for k in BA.KEYS)
          + f"{f'{c[0]}/{c[1]}/{c[2]}':>14}{nn:>12,}{nr:>+12.4f}")
    if tag.startswith("C"):
        etp, efp, efn = (int(sum(r[k] for r in rows)) for k in ("edge_tp", "edge_fp", "edge_fn"))
        print(f"  {'':<26}edge tp/fp/fn {etp:,}/{efp:,}/{efn:,}")
    else:
        etp, efp, efn = (int(sum(r[k] for r in rows)) for k in ("edge_tp", "edge_fp", "edge_fn"))
        print(f"  {'':<26}edge tp/fp/fn {etp:,}/{efp:,}/{efn:,}")

print(f"\n=== PAIRED BOOTSTRAP, M - C, 4,000 draws over the {len(NAMES)} movies ===")
BA.print_bootstrap(BA.paired_bootstrap(C, M, NAMES))
print("\n=== per movie, M - C ===")
BA.print_per_movie(C, M, NAMES)
print("\n=== concentration (adj_J): the movies carrying the delta ===")
BA.print_concentration(C, M, NAMES)
