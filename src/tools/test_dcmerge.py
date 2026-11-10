"""Prove _dc_merge is add-only before spending an hour on it.

Four properties, all cheap, and the first is the entire safety argument
for this change:

  1. ADD-ONLY -- every one of our points survives, in order, with its
     score unchanged. If this fails, "DeepCenter being worse in other
     respects cannot hurt us" is false.
  2. SEPARATION -- no added point is within DC_MERGE_UM of one of ours.
  3. SCORE -- an added point carries exactly DC_ADD_SCORE, which must sit
     on the primary side of seeds_of and outside the repair pool.
  4. OFF -- DC_TAU = None returns the input unchanged, so BASE is
     byte-identical to cap20.
"""
import ast
import pathlib
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = pathlib.Path(__file__).resolve().parents[1]   # src/
SCALE = np.array([1.625, 0.40625, 0.40625], float)

src = (ROOT / 'dcmerge.py').read_text(encoding='utf-8')
fn = None
for node in ast.parse(src).body:
    if isinstance(node, ast.FunctionDef) and node.name == '_dc_merge':
        fn = ast.get_source_segment(src, node)
assert fn is not None, '_dc_merge not found'
assert fn.count('def ') == 1, 'the extracted segment is not one function'


class _Z:
    """Stands in for the npz: same .files / __getitem__ contract."""

    def __init__(self, d):
        self._d = d
        self.files = list(d)

    def __getitem__(self, k):
        return self._d[k]


def build(ns, tau, cache):
    ns.update(dict(np=np, cKDTree=cKDTree, SCALE=SCALE, UNET_THRESH=0.15,
                   DC_TAU=tau, DC_MERGE_UM=4.0, DC_ADD_SCORE=0.15,
                   _DC_STATS=[0, 0, 0],
                   DCDET={'X': cache} if cache is not None else {}))
    ns['np'].load = lambda f: f          # DCDET holds the object directly
    exec(compile(fn, 'dcmerge', 'exec'), ns)
    return ns['_dc_merge']


rng = np.random.default_rng(20260828)
fail = []
tot_added = tot_ours = 0
for trial in range(300):
    T = 5
    dets, cache = [], {}
    for t in range(T):
        n = int(rng.integers(3, 40))
        c = rng.uniform(0, 60, (n, 3))
        sc = rng.uniform(0.02, 0.9, n)
        dets.append((c, sc))
        m = int(rng.integers(0, 40))
        cache[f"c{t}"] = rng.uniform(0, 60, (m, 3))
        cache[f"s{t}"] = rng.uniform(0.0, 1.0, m)
    z = _Z(cache)

    ns = {}
    merge = build(ns, 0.15, z)
    out = merge(dets, 'X')

    for t in range(T):
        c0, s0 = dets[t]
        c1, s1 = out[t]
        # 1. add-only: ours are a prefix, unchanged
        if not (len(c1) >= len(c0)
                and np.allclose(c1[:len(c0)], c0)
                and np.allclose(s1[:len(s0)], s0)):
            fail.append(('add-only', trial, t))
            continue
        added = c1[len(c0):]
        added_s = s1[len(s0):]
        tot_ours += int((s0 >= 0.15).sum())
        tot_added += len(added)
        # 3. score
        if len(added) and not np.allclose(added_s, 0.15):
            fail.append(('score', trial, t))
        # 2. separation from OUR PRIMARY points
        mine = c0[s0 >= 0.15]
        if len(added) and len(mine):
            d, _ = cKDTree(mine * SCALE[None, :]).query(added * SCALE[None, :])
            if d.min() <= 4.0 - 1e-9:
                fail.append(('separation', trial, t))

    # 4. off
    ns2 = {}
    merge_off = build(ns2, None, z)
    off = merge_off(dets, 'X')
    if off is not dets:
        fail.append(('off-identity', trial, -1))

print(f"300 synthetic movies x 5 frames")
print(f"  our primary points {tot_ours:,}, ADDED {tot_added:,} "
      f"({100.0 * tot_added / max(1, tot_ours):.1f}%)")
if tot_added == 0:
    print("  ⚠ NOTHING WAS ADDED -- the test proved nothing")
kinds = sorted({k for k, _, _ in fail})
print(f"  failures: {len(fail)}" + (f"  kinds: {kinds}" if fail else "  NONE"))
sys.exit(1 if (fail or tot_added == 0) else 0)
