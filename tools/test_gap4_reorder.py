"""Prove gap4's gate REORDERING preserves behaviour, before paying for it.

gap3 ran the length gate before the candidate lookup; gap4 runs the
lookup first so the length gate can be made conditional on it. The claim
is that this changes nothing when GAP_LEN_MODE == "always", because the
lookup has no side effects.

`gap2`'s BASE arm broke on exactly this kind of claim and it took 204
minutes of Kaggle compute to find out. This costs seconds.

Both `_gap_close` implementations are lifted out of their files by AST
and exec'd into separate namespaces, so the code under test is the code
that will run, not a paraphrase.
"""
import ast
import pathlib
import sys

import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

ROOT = pathlib.Path(r'D:\programming\cell tracking')
SCALE = np.array([1.625, 0.40625, 0.40625], float)
WANT = ('_frag_len', '_prim_tree', '_gap_close')


def load(fname):
    src = (ROOT / fname).read_text(encoding='utf-8')
    tree = ast.parse(src)
    ns = dict(np=np, defaultdict=defaultdict, cKDTree=cKDTree,
              linear_sum_assignment=linear_sum_assignment, SCALE=SCALE)
    ns['_PRIM_TREES'] = {}
    ns['_GAP_LOG'] = defaultdict(lambda: [0, 0, 0, 0])
    got = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in WANT:
            exec(compile(ast.Module([node], []), fname, 'exec'), ns)
            got.append(node.name)
    assert set(got) == set(WANT), (fname, got)
    return ns


def set_consts(ns, **kw):
    ns.update(dict(
        GAP1_GATE_UM=9.0, GAP1_SNAP_UM=3.2, GAP1_MIN_CAND_SCORE=0.10,
        GAP1_CAP_FRAC=0.003, GAP_SPAN=1, GAP_NEED_CAND=True,
        GAP_VEL_COS=-1.0, GAP_MIN_LEN=0, GAP_CLEAR_UM=0.0,
        GAP_LEN_MODE="always"))
    ns.update(kw)


def make_case(rng, n_tracks=45, T=40):
    """A synthetic movie: tracks that walk, break, and leave endpoints
    for _gap_close to join.

    Two things the first version got wrong, both of which made the test
    nearly vacuous for the arms that matter:

      * T was 14 and tracks were 2-14 frames, so a fragment almost never
        reached 8 nodes and the `len8` arm added ZERO edges in 400
        movies -- it was comparing two no-ops.
      * candidates were scattered uniformly over the box, so almost none
        landed inside GAP1_SNAP_UM = 3.2 um of a midpoint and the BASE
        arm added 4 edges in 400 movies.

    Now tracks run long enough to clear a length-8 gate, and roughly
    half the breaks get a candidate planted ON the interpolated
    midpoint, so both branches of the candidate test fire often.
    """
    nodes, edges = {}, []
    nid = 0
    planted = defaultdict(list)     # t -> midpoints to seed a candidate at
    for _ in range(n_tracks):
        t0 = int(rng.integers(0, T - 12))
        length = int(rng.integers(10, T - t0 + 1))
        p = rng.uniform(5, 95, 3)
        v = rng.normal(0, 1.2, 3)
        # a break leaves an `ends` node at t and a `starts` node at t+2.
        # Keep it away from both tips so fragment lengths vary widely.
        brk = int(rng.integers(1, length - 1)) if rng.random() < 0.7 else -1
        prev = None
        p_before = None
        for k in range(length):
            if k == brk:
                p_before = p.copy() - v      # position of the last node
                prev = None
                p = p + v
                if rng.random() < 0.5:
                    planted[t0 + k].append(0.5 * (p_before + p))
                continue
            nodes[nid] = {"t": t0 + k, "xyz": p.copy()}
            if prev is not None:
                edges.append((prev, nid))
            prev = nid
            nid += 1
            p = p + v
    cand, cand_sc, cand_trees = [], [], []
    for t in range(T):
        pts = [np.asarray(m, float) + rng.normal(0, 0.6, 3)
               for m in planted.get(t, [])]
        sc = [float(rng.uniform(0.10, 0.149)) for _ in pts]
        m = int(rng.integers(0, 5))          # plus some genuine noise
        for _ in range(m):
            pts.append(rng.uniform(0, 100, 3))
            sc.append(float(rng.uniform(0.05, 0.149)))
        if pts:
            pts = np.stack(pts)
            sc = np.asarray(sc, float)
        else:
            pts = np.zeros((0, 3))
            sc = np.zeros(0)
        cand.append(pts)
        cand_sc.append(sc)
        cand_trees.append(cKDTree(pts * SCALE[None, :]) if len(pts)
                          else None)
    return nodes, edges, cand, cand_sc, cand_trees, nid


def run(ns, case, **kw):
    nodes, edges, cand, cand_sc, cand_trees, nid = case
    nodes = {k: {"t": v["t"], "xyz": v["xyz"].copy()}
             for k, v in nodes.items()}
    edges = list(edges)
    set_consts(ns, **kw)
    out = ns['_gap_close'](nodes, edges, cand, cand_sc, cand_trees, nid)
    key = sorted((int(a), int(b)) for a, b in edges)
    pos = sorted((int(k), v["t"], tuple(np.round(v["xyz"], 9)))
                 for k, v in nodes.items())
    return out, key, pos


A = load('gap3.py')
B = load('gap4.py')

ARMS = [
    ("BASE", dict(SHORT_MIN=4)),
    ("nocand", dict(GAP_NEED_CAND=False)),
    ("len4", dict(GAP_NEED_CAND=False, GAP_MIN_LEN=4)),
    ("len8", dict(GAP_NEED_CAND=False, GAP_MIN_LEN=8)),
    ("vel0.0", dict(GAP_NEED_CAND=False, GAP_VEL_COS=0.0)),
    ("clear4", dict(GAP_NEED_CAND=False, GAP_CLEAR_UM=4.0)),
    ("cap20", dict(GAP_NEED_CAND=False, GAP1_CAP_FRAC=0.02)),
]

rng = np.random.default_rng(20260827)
fail = 0
touched = defaultdict(int)
for trial in range(400):
    case = make_case(rng)
    for tag, kw in ARMS:
        kw = {k: v for k, v in kw.items() if k != 'SHORT_MIN'}
        ra = run(A, case, **kw)
        rb = run(B, case, **kw)
        touched[tag] += len(rb[1]) - len(case[1])
        if ra != rb:
            fail += 1
            if fail <= 3:
                print(f"  MISMATCH trial {trial} arm {tag}")
                print(f"    gap3 next_id {ra[0]} edges {len(ra[1])}")
                print(f"    gap4 next_id {rb[0]} edges {len(rb[1])}")
print(f"gap3 vs gap4, 400 synthetic movies x {len(ARMS)} arms: "
      f"{'ALL IDENTICAL' if not fail else str(fail) + ' MISMATCHES'}")
print("  edges added per arm (must be non-zero, or the test proved "
      "nothing):")
for tag, _ in ARMS:
    print(f"    {tag:<8} {touched[tag]:>7,}")

# ---- and the NEW behaviour must actually differ ------------------------
def bridges(ns, case, **kw):
    """The set of (ge, gs) pairs actually joined.

    ⚠ Do NOT compare raw edge tuples across arms. Inserted nodes are
    numbered from `next_id`, so an arm that makes a different number of
    bridges numbers ALL of them differently and every edge touching an
    inserted node compares unequal. My first version of this check did
    exactly that and reported the superset property failing 278 times
    out of 300 -- an artefact, not a finding. Mapping back to the
    original endpoints is the comparison that means something.
    """
    nodes, edges, cand, cand_sc, cand_trees, nid = case
    nodes = {k: {"t": v["t"], "xyz": v["xyz"].copy()}
             for k, v in nodes.items()}
    edges = list(edges)
    n0 = len(edges)
    set_consts(ns, **kw)
    ns['_gap_close'](nodes, edges, cand, cand_sc, cand_trees, nid)
    pre, post = {}, {}
    for a, b in edges[n0:]:
        if b >= nid:
            pre[b] = a
        if a >= nid:
            post[a] = b
    return {(pre[k], post[k]) for k in pre if k in post}


rng2 = np.random.default_rng(7)
diff = same = 0
print("\n=== IS THE USER'S POLICY A SUPERSET OF BASE? ===")
print("  'bridge if a candidate exists OR both fragments are long' "
      "contains 'bridge if a candidate exists', so it should never "
      "drop a bridge BASE makes. It does anyway when the BUDGET binds: "
      "max_gaps is handed out in frame order and a policy that bridges "
      "more, earlier, starves the later ones.")
for cap, label in ((0.003, "shipped budget 0.003"),
                   (0.5, "budget past demand")):
    rng3 = np.random.default_rng(7)
    ok = lost = nb = no = 0
    for _ in range(300):
        case = make_case(rng3)
        sb = bridges(B, case, GAP1_CAP_FRAC=cap)
        so = bridges(B, case, GAP_NEED_CAND=False, GAP_MIN_LEN=8,
                     GAP_LEN_MODE="if_nocand", GAP1_CAP_FRAC=cap)
        nb += len(sb)
        no += len(so)
        lost += len(sb - so)
        ok += (not (sb - so))
    print(f"  {label:<22} superset held {ok:>3}/300   "
          f"BASE {nb:>6,} bridges   or8 {no:>6,}   "
          f"BASE bridges LOST {lost:>5,}")

for trial in range(300):
    case = make_case(rng2)
    a = bridges(B, case, GAP_NEED_CAND=False, GAP_MIN_LEN=8,
                GAP_LEN_MODE="always")
    b = bridges(B, case, GAP_NEED_CAND=False, GAP_MIN_LEN=8,
                GAP_LEN_MODE="if_nocand")
    if a != b:
        diff += 1
    else:
        same += 1
print(f"\nif_nocand differs from always in {diff} of {diff + same} "
      f"movies (0 would mean the new mode is inert)")
sys.exit(1 if fail else 0)
