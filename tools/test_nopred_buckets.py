"""Smoke test for nopred.py's bucket logic: fake GT + fake primaries that
hit every bucket once, plus the control-group buckets and the recovery
check. The functions are lifted out of nopred.py by AST, so the code
under test is the code that will run on Kaggle, not a paraphrase
(same pattern as test_gap4_reorder.py).

Run:  py tools/test_nopred_buckets.py
"""
import ast
import pathlib
import sys
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "nopred.py"
SCALE = np.array([1.625, 0.40625, 0.40625], float)
WANT_FN = ("_greedy_match", "_norm", "border_dist", "bucket_one",
           "final_fate", "bucket_dataset")
WANT_CLS = ("GTContext",)
WANT_ASSIGN = ("MATCH_R", "BORDER_UM", "COMP_SHAPE_ZYX", "BUCKETS",
               "ZERO_BUCKETS", "CTRL_BUCKETS", "SUBS", "RC", "MAX_LINK_UM",
               "TIGHT_UM")


def load():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    ns = dict(np=np, defaultdict=defaultdict, cKDTree=cKDTree, SCALE=SCALE,
              warn_once=lambda k, m: print("WARN", k, m))
    got = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in WANT_FN:
            exec(compile(ast.Module([node], []), str(SRC), "exec"), ns)
            got.append(node.name)
        elif isinstance(node, ast.ClassDef) and node.name in WANT_CLS:
            exec(compile(ast.Module([node], []), str(SRC), "exec"), ns)
            got.append(node.name)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in WANT_ASSIGN:
            exec(compile(ast.Module([node], []), str(SRC), "exec"), ns)
            got.append(node.targets[0].id)
    missing = set(WANT_FN + WANT_CLS + WANT_ASSIGN) - set(got)
    assert not missing, f"not found in nopred.py: {missing}"
    return ns


N = load()
BUCKETS, SUBS = N["BUCKETS"], N["SUBS"]
V = 1.0 / SCALE          # voxel step that is 1 um along each axis


def um(z, y, x):
    """um -> voxel coordinates (z, y, x)."""
    return np.array([z, y, x], float) * V


# ---- the scene: T = 3 frames, everything at z = 50 um ---------------------
# GT rows (t, position in um). Field 104 x 104 x 104 um (COMP_SHAPE).
GT = [
    # t = 1 parents (rows 0-5)
    (1, (50, 50, 50)),    # 0 parent of a division: daughters 6 and 7
    (1, (50, 20, 20)),    # 1 parent of 8 -- undetected at t=1 (subthresh)
    (1, (50, 80, 20)),    # 2 parent of 9 -- taken by competitor
    (1, (50, 20, 80)),    # 3 parent of 10 -- out of gate (12 um away)
    (1, (50, 80, 80)),    # 4 parent of 11 (control, linked_to_parent)
    (1, (50, 35, 65)),    # 5 parent of 13 -- cold (no candidate near)
    # t = 2 children (rows 6-14)
    (2, (50, 50, 46)),    # 6 daughter A (zero velocity)
    (2, (50, 50, 54)),    # 7 daughter B (sister; took the parent)
    (2, (50, 20, 22)),    # 8 child of 1 -> parent_undetected / subthresh
    (2, (50, 80, 23)),    # 9 child of 2 -> parent_taken
    (2, (50, 20, 92)),    # 10 child of 3 -> parent_out_of_gate (12 um)
    (2, (50, 80, 82)),    # 11 child of 4 -> control linked_to_parent
    (2, (50, 5, 50)),     # 12 gt_start, 5 um from the y=0 face
    (2, (50, 35, 67)),    # 13 child of 5 -> parent_undetected / cold
    (2, (50, 60, 30)),    # 14 gt_start, interior
]
GT_EDGES = [(0, 6), (0, 7), (1, 8), (2, 9), (3, 10), (4, 11), (5, 13)]
gt_t = np.array([t for t, _ in GT], int)
gt_pos = np.stack([um(*p) for _, p in GT])       # voxel coords
T = 3

# primaries per frame: (index -> position um). Frame 0 empty except one.
P0 = [(50, 50, 48)]
P1 = [(50, 50, 50),    # 0 = GT 0 (division parent)
      (50, 80, 20),    # 1 = GT 2 (taken parent)
      (50, 20, 80),    # 2 = GT 3 (out-of-gate parent)
      (50, 80, 80),    # 3 = GT 4 (control parent)
      (50, 60, 62)]    # 4 unmatched; prev of control primary 11
P2 = [(50, 50, 46),    # 0 = GT 6 daughter A, zero
      (50, 50, 54),    # 1 = GT 7 daughter B, prev = P1[0]
      (50, 20, 22),    # 2 = GT 8, parent undetected (subthresh)
      (50, 80, 23),    # 3 = GT 9, parent taken by primary 4
      (50, 80, 18),    # 4 = competitor, unmatched to GT (no GT within 7? GT 9 is 5 um away!)
      (50, 20, 92),    # 5 = GT 10, out of gate
      (50, 80, 82),    # 6 = GT 11, control linked_to_parent
      (50, 5, 50),     # 7 = GT 12, gt_start border
      (50, 35, 67),    # 8 = GT 13, parent cold
      (50, 60, 30),    # 9 = GT 14, gt_start interior
      (50, 90, 90),    # 10 unmatched (no GT within 7 um)
      (50, 60, 60)]    # 11 control linked_elsewhere: prev = P1[3]? no GT parent... see below
# competitor at (80,18) is 5 um from GT 9 (80,23): the greedy 1:1 match
# gives GT 9 to primary 3 (3 um) first, so primary 4 stays unmatched.
seeds = [(np.stack([um(*p) for p in P0]), np.full(len(P0), 0.5)),
         (np.stack([um(*p) for p in P1]), np.full(len(P1), 0.5)),
         (np.stack([um(*p) for p in P2]), np.full(len(P2), 0.5))]
prev_of = [np.full(len(P0), -1, np.int32), np.full(len(P1), -1, np.int32),
           np.full(len(P2), -1, np.int32)]
prev_of[1][0] = 0            # P1[0] <- P0[0]
prev_of[2][1] = 0            # sister took the division parent
prev_of[2][4] = 1            # competitor took GT 2's primary
prev_of[2][6] = 3            # control: correct link
prev_of[2][11] = 4           # control: primary 11 is unmatched -> bucket unmatched
vels = [np.zeros((len(P0), 3)), np.zeros((len(P1), 3)), np.zeros((len(P2), 3))]
for t in (1, 2):
    for i, pi in enumerate(prev_of[t]):
        if pi >= 0:
            vels[t][i] = (seeds[t][0][i] - seeds[t - 1][0][pi]) * SCALE
# a primary with a predecessor at identical coordinates -> |v| == 0 -> other
P2.append((50, 80, 80))      # 12: same spot as P1[3]
seeds[2] = (np.stack([um(*p) for p in P2]), np.full(len(P2), 0.5))
prev_of[2] = np.append(prev_of[2], 3)
vels[2] = np.vstack([vels[2], np.zeros((1, 3))])

# repair band: one candidate 2 um from GT 1 (subthresh), none near GT 5
cand_trees = [None,
              cKDTree(np.stack([um(50, 20, 18)]) * SCALE[None, :]),
              None]
extent_um = np.array(N["COMP_SHAPE_ZYX"], float) * SCALE

ctx = N["GTContext"](gt_t, gt_pos, np.asarray(GT_EDGES, int), T, seeds, vels,
                     prev_of, cand_trees, extent_um)

# ---- the zero group ---------------------------------------------------
EXPECT_ZERO = {
    0: ("division_daughter", "sister_took_parent"),
    2: ("parent_undetected", "subthresh"),
    3: ("parent_taken", "comp_unmatched"),
    5: ("parent_out_of_gate", "beyond_max_link"),
    7: ("gt_start", "border<10"),
    8: ("parent_undetected", "cold"),
    9: ("gt_start", "interior"),
    10: ("unmatched", "-"),
    12: ("other", "linked_zero_disp"),
}
fails = 0
for i, (b, s) in EXPECT_ZERO.items():
    got = N["bucket_one"](2, i, ctx, True)
    ok = (got[0], got[1]) == (b, s)
    fails += not ok
    print(f"  zero  prim {i:>2}: {got[0]:<26} {got[1]:<20} "
          f"d {tuple(round(float(x), 2) for x in got[2:6])}  "
          f"{'ok' if ok else 'FAIL expected ' + b + '/' + s}")
# spot-check distances
g = N["bucket_one"](2, 7, ctx, True)
assert abs(g[2] - 5.0) < 1e-6 and g[3] == 1.0, f"border dist/axis {g}"
g = N["bucket_one"](2, 5, ctx, True)
assert abs(g[2] - 12.0) < 1e-6, f"out-of-gate raw distance {g}"
g = N["bucket_one"](2, 3, ctx, True)
assert abs(g[2] - 5.0) < 1e-6 and abs(g[3] - 0.0) < 1e-6, f"taken {g}"
# parent_free_but_unlinked is reachable only by a wrong prev_of: force it
prev_of[2][4] = -1
g = N["bucket_one"](2, 3, ctx, True)
ok = g[0] == "parent_free_but_unlinked" and g[1] == "stage1_eligible"
fails += not ok
print(f"  zero  prim  3 (parent freed): {g[0]} / {g[1]}  "
      f"{'ok' if ok else 'FAIL'}")
prev_of[2][4] = 1

# ---- the control group --------------------------------------------------
EXPECT_CTRL = {
    1: ("division_daughter", "linked_to_parent"),
    6: ("linked_to_parent", "-"),
    11: ("unmatched", "-"),
}
for i, (b, s) in EXPECT_CTRL.items():
    got = N["bucket_one"](2, i, ctx, False)
    ok = (got[0], got[1]) == (b, s)
    fails += not ok
    print(f"  ctrl  prim {i:>2}: {got[0]:<26} {got[1]:<20} "
          f"{'ok' if ok else 'FAIL expected ' + b + '/' + s}")
# linked_elsewhere: give primary 6 (GT 11, parent GT 4 = P1[3]) prev P1[2]
prev_of[2][6] = 2
g = N["bucket_one"](2, 6, ctx, False)
ok = g[0] == "linked_elsewhere"
fails += not ok
print(f"  ctrl  prim  6 (relinked): {g[0]}  {'ok' if ok else 'FAIL'}")
prev_of[2][6] = 3

# ---- recovery / final fate -----------------------------------------------
# node ids: seed (t, i) -> id; a final graph with: daughter A linked from
# the parent's node (recovered), GT 8's primary linked from a REPAIR node
# near GT 1 (recovered by position), GT 9's primary with no in-edge (start),
# GT 10's primary absent from the final graph.
node_of_seed = {(1, 0): 1, (2, 0): 2, (2, 2): 3, (2, 3): 4, (2, 5): 5}
nodes0 = {1: {"t": 1, "xyz": seeds[1][0][0]}, 2: {"t": 2, "xyz": seeds[2][0][0]},
          3: {"t": 2, "xyz": seeds[2][0][2]}, 4: {"t": 2, "xyz": seeds[2][0][3]},
          5: {"t": 2, "xyz": seeds[2][0][5]}}
nodes_f = {k: v for k, v in nodes0.items() if k != 5}
nodes_f[99] = {"t": 1, "xyz": um(50, 20, 21)}      # repair node, 1 um from GT 1
edges_f = [(1, 2), (99, 3)]
in_src = defaultdict(list)
for a, b in edges_f:
    in_src[b].append(a)
ff = N["final_fate"]
chk = [
    (ff(2, 0, 0, node_of_seed, nodes0, nodes_f, in_src, ctx.gt_um), (1, 1, 1)),
    (ff(2, 2, 1, node_of_seed, nodes0, nodes_f, in_src, ctx.gt_um), (1, 1, 1)),
    (ff(2, 3, 2, node_of_seed, nodes0, nodes_f, in_src, ctx.gt_um), (1, 0, 0)),
    (ff(2, 5, 3, node_of_seed, nodes0, nodes_f, in_src, ctx.gt_um), (0, 0, None)),
    (ff(2, 7, -1, node_of_seed, nodes0, nodes_f, in_src, ctx.gt_um), (0, 0, None)),
]
for got, want in chk:
    g3 = None if got[2] != got[2] else got[2]
    ok = (got[0], got[1], g3) == want
    fails += not ok
    print(f"  fate {tuple(got)} want {want}  {'ok' if ok else 'FAIL'}")

# ---- bucket_dataset end to end -----------------------------------------
zr, cr, n_t1, n_zero = N["bucket_dataset"](
    ctx, node_of_seed, nodes0, nodes_f, edges_f, np.random.default_rng(0))
print(f"  bucket_dataset: primaries t>=1 {n_t1}, zero {n_zero}, "
      f"zero rows {len(zr)}, ctrl rows {len(cr)}")
assert n_t1 == len(P1) + len(P2) and n_zero == 4 + 9, (n_t1, n_zero)
assert len(zr) == n_zero and len(cr) == min(n_zero, n_t1 - n_zero)
zb = [BUCKETS[int(k)] for k in zr[:, 0]]
for b in ("unmatched", "gt_start", "division_daughter", "parent_undetected",
          "parent_taken", "parent_out_of_gate", "other"):
    assert b in zb, f"{b} never hit end to end"
assert "parent_free_but_unlinked" not in zb
# t = 1 primaries with no predecessor and no GT parent: 1..3 are gt_start
# (GT 1..4 have no parent), 0 has a prev (control)
assert zr.shape[1] == len(N["RC"])
print(f"  end to end: {sorted(set(zb))}")

if fails:
    print(f"\nSMOKE FAIL: {fails} checks")
    sys.exit(1)
print("\nSMOKE OK: every bucket hit once, distances and recovery as expected")
