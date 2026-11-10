"""Exercise GAP_SPAN = 2 before paying four hours for it.

`GAP_SPAN` has never run in the CURRENT `_gap_close`. gap2 ran a span
sweep, but that implementation had the `span + 1` normalisation bug and
was rewritten for gap3. So span 2 is the least-exercised path in the
file and it is the entire point of span4.

Three claims to check, all cheap:

  1. span 2 REPRODUCES span 1 on the one-frame gaps. Raising the ceiling
     must not change what was already being repaired -- the loop runs
     `for span in range(1, GAP_SPAN + 1)`, so span 1 goes first and
     should be untouched.
  2. span 2 actually makes TWO-frame bridges, and inserts TWO nodes for
     each. If it makes none the arm is a no-op and the run answers
     nothing.
  3. `D = D / float(span)` is a no-op at span 1 -- the bug gap2 died on.

WRITTEN AS A FILE, NOT A HEREDOC.
"""
import ast
import pathlib
import sys

import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

NL = chr(10)
ROOT = pathlib.Path(__file__).resolve().parents[1]   # src/
SCALE = np.array([1.625, 0.40625, 0.40625], float)
WANT = ('_frag_len', '_prim_tree', '_gap_close')


def load(fname):
    src = (ROOT / fname).read_text(encoding='utf-8')
    ns = dict(np=np, defaultdict=defaultdict, cKDTree=cKDTree,
              linear_sum_assignment=linear_sum_assignment, SCALE=SCALE,
              _PRIM_TREES={}, _GAP_LOG=defaultdict(lambda: [0, 0, 0, 0]))
    got = []
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in WANT:
            exec(compile(ast.Module([node], []), fname, 'exec'), ns)
            got.append(node.name)
    assert set(got) == set(WANT), (fname, got)
    return ns


def make_case(rng, n_tracks=45, T=40):
    """Tracks that break for ONE frame or for TWO, so both span levels
    have something to find."""
    nodes, edges = {}, []
    nid = 0
    two_frame = 0
    for _ in range(n_tracks):
        t0 = int(rng.integers(0, T - 14))
        length = int(rng.integers(12, T - t0 + 1))
        p = rng.uniform(5, 95, 3)
        v = rng.normal(0, 1.0, 3)
        # gap of 1 or 2 frames, or none
        roll = rng.random()
        glen = 1 if roll < 0.4 else (2 if roll < 0.75 else 0)
        if glen == 2:
            two_frame += 1
        brk = int(rng.integers(2, length - glen - 1)) if glen else -1
        prev = None
        for k in range(length):
            if glen and brk <= k < brk + glen:
                prev = None
                p = p + v
                continue
            nodes[nid] = {"t": t0 + k, "xyz": p.copy()}
            if prev is not None:
                edges.append((prev, nid))
            prev = nid
            nid += 1
            p = p + v
    cand, cand_sc, cand_trees = [], [], []
    for _ in range(T):
        m = int(rng.integers(0, 4))
        pts = rng.uniform(0, 100, (m, 3)) if m else np.zeros((0, 3))
        sc = rng.uniform(0.05, 0.149, m) if m else np.zeros(0)
        cand.append(pts)
        cand_sc.append(sc)
        cand_trees.append(cKDTree(pts * SCALE[None, :]) if m else None)
    return nodes, edges, cand, cand_sc, cand_trees, nid, two_frame


def run(ns, case, **kw):
    nodes, edges, cand, cand_sc, cand_trees, nid, _ = case
    nodes = {k: {"t": v["t"], "xyz": v["xyz"].copy()}
             for k, v in nodes.items()}
    edges = list(edges)
    n0 = len(edges)
    ns.update(dict(GAP1_GATE_UM=9.0, GAP1_SNAP_UM=3.2,
                   GAP1_MIN_CAND_SCORE=0.10, GAP1_CAP_FRAC=0.5,
                   GAP_SPAN=1, GAP_NEED_CAND=False, GAP_VEL_COS=-1.0,
                   GAP_MIN_LEN=0, GAP_CLEAR_UM=0.0,
                   GAP_LEN_MODE="always"))
    ns.update(kw)
    ns['_gap_close'](nodes, edges, cand, cand_sc, cand_trees, nid)
    added = edges[n0:]
    pre, post = {}, {}
    for a, b in added:
        if b >= nid:
            pre.setdefault(b, a)
        if a >= nid:
            post.setdefault(a, b)
    # a bridge is ge -> (1 or 2 inserted) -> gs; walk the inserted chain
    chains = []
    for new in sorted(set(pre) | set(post)):
        if new in pre and pre[new] < nid:          # chain head
            cur, span_nodes = new, 1
            while cur in post and post[cur] >= nid:
                cur = post[cur]
                span_nodes += 1
            tail = post.get(cur)
            if tail is not None and tail < nid:
                chains.append((pre[new], tail, span_nodes))
    return chains


B = load('span4.py')

rng = np.random.default_rng(20260828)
tot1 = tot2 = 0
one_only = 0
mismatch = 0
two_frame_avail = 0
for trial in range(200):
    case = make_case(rng)
    two_frame_avail += case[6]
    # budget removed on purpose: at 0.02 of ~1,125 synthetic nodes
    # the allowance is ~22 and span 1 alone spends ~16, so span 2
    # was being starved rather than tested.
    c1 = run(B, case, GAP_SPAN=1)
    c2 = run(B, case, GAP_SPAN=2)
    s1 = {(a, b) for a, b, n in c1}
    s2_one = {(a, b) for a, b, n in c2 if n == 1}
    s2_two = {(a, b) for a, b, n in c2 if n == 2}
    tot1 += len(s1)
    tot2 += len(s2_two)
    one_only += len(s2_one)
    # claim 1: span 2 must reproduce every span-1 bridge
    if s1 != s2_one:
        mismatch += 1
    # claim 2: a 2-frame bridge must insert exactly 2 nodes
    for a, b, n in c2:
        assert n in (1, 2), f"span 2 inserted {n} nodes"

print(f"200 synthetic movies, {two_frame_avail} two-frame gaps planted")
print(f"  span1 one-frame bridges           {tot1:>6,}")
print(f"  span2, its one-frame bridges      {one_only:>6,}   "
      f"{'IDENTICAL' if not mismatch else str(mismatch) + ' MOVIES DIFFER'}")
print(f"  span2, its TWO-frame bridges      {tot2:>6,}   "
      f"{'<- the new capability' if tot2 else '<- ZERO, THE ARM IS A NO-OP'}")

# ---- claim 3: the span-1 gate is not rescaled --------------------------
src = (ROOT / 'span4.py').read_text(encoding='utf-8')
assert 'D = D / float(span)' in src, 'the normalisation line is gone'
# Strip comments first: gap3 left a long note that QUOTES the bug,
# so the string is in the file as prose and matching raw text is a
# false positive.
code = NL.join(ln.split('#')[0] for ln in src.split(NL))
assert 'D = D / float(step)' not in code, (
    'the step-normalised gate is BACK in CODE -- gap2\'s bug, it '
    'doubles GAP1_GATE_UM at span 1')
print("  gate normalisation: divides by SPAN (1 at span 1, a no-op), "
      "not by step -- gap2's bug is absent")

sys.exit(1 if (mismatch or not tot2) else 0)
