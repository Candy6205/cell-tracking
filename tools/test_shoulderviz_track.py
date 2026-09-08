"""shoulderviz's zoom clip must follow ONE cell. Test that it does.

THE BUG THIS EXISTS FOR (found by the user watching v5, 2026-09-08: "你这个圈
怎么好几帧都在原地"). Up to v5 the red marker in a +-5 frame zoom came from
`miss_bt`, which holds "every cell flagged on THIS frame". A cell is flagged
only on the frame it was missed on, so:

  * on the other frames of the window the red circle simply vanished, and
  * when some OTHER shoulder cell was in view it jumped onto that one.

Measured on the shipped 44b6_a2bb48bb_zoom0.mp4: the red centroid appeared on
3 of 6 frames, at (187,237), (169,219) and (171,327). Read literally that says
the cell moved 100 px in one frame. It says nothing of the kind.

v6 follows the GT lineage with `_walk` instead. The checks below are on that
function, because the thing to guarantee is graph behaviour, not pixels:
one cell per frame, contiguous in time, and never the wrong cell.

    py tools/test_shoulderviz_track.py shoulderviz.py
"""
import sys
from collections import defaultdict

TARGET = sys.argv[1] if len(sys.argv) > 1 else "shoulderviz.py"
SRC = open(TARGET, encoding="utf-8").read()

a = SRC.index("def _walk(")
b = SRC.index("\ndef render(", a)
print(f"  extracted _walk: {b - a} chars")
NS = {}
exec(SRC[a:b], NS)
_walk = NS["_walk"]


def graph(chain_specs):
    """chain_specs: list of [(node, t, y)] chains, linked consecutively."""
    nid_xyz, kids, par = {}, defaultdict(list), defaultdict(list)
    for chain in chain_specs:
        for (n, t, y) in chain:
            nid_xyz[n] = (t, 0.0, float(y), 0.0)
        for (n1, _, _), (n2, _, _) in zip(chain, chain[1:]):
            kids[n1].append(n2)
            par[n2].append(n1)
    return nid_xyz, kids, par


print("\n1. a straight track is followed on EVERY frame of the window")
# one cell drifting 6 px per frame over t=0..10, the miss at t=5
chain = [(100 + t, t, 10 * 1.0 + 6 * t) for t in range(11)]
nid_xyz, kids, par = graph([chain])
got = sorted(_walk(105, 0, 10, nid_xyz, par, kids))
assert len(got) == 11, f"expected all 11 frames, got {len(got)}"
ts = [int(r[0]) for r in got]
assert ts == list(range(11)), ts
ys = [r[2] for r in got]
assert ys == sorted(ys) and ys[-1] - ys[0] == 60.0, ys
print(f"  11 of 11 frames, y drifts {ys[0]:.0f} -> {ys[-1]:.0f}   OK")
print("  (the v5 bug would give 1 frame here -- only the flagged one)")

print("\n2. the window is respected in both directions")
got = _walk(105, 3, 7, nid_xyz, par, kids)
assert sorted(int(r[0]) for r in got) == [3, 4, 5, 6, 7], got
print("  t in [3,7] -> exactly frames 3,4,5,6,7   OK")
got = _walk(100, 0, 10, nid_xyz, par, kids)
assert sorted(int(r[0]) for r in got) == list(range(11)), "walk forward only"
got = _walk(110, 0, 10, nid_xyz, par, kids)
assert sorted(int(r[0]) for r in got) == list(range(11)), "walk backward only"
print("  starting at either end of the chain still returns all 11   OK")

print("\n3. a SECOND, unrelated cell in view is never picked up")
# this is the actual v5 failure: another shoulder cell 300 px away
other = [(200 + t, t, 300.0 + t) for t in range(11)]
nid_xyz, kids, par = graph([chain, other])
got = _walk(105, 0, 10, nid_xyz, par, kids)
ys = [r[2] for r in got]
assert max(ys) < 200.0, \
    f"FATAL: the walk crossed onto the other track ({max(ys)}) -- this is " \
    "exactly the jump the user saw"
assert len(got) == 11
print(f"  focal track y stays in [{min(ys):.0f},{max(ys):.0f}], the decoy at "
      f"300+ is untouched   OK")

print("\n4. one position per frame, so the red circle cannot fork visually")
per_t = defaultdict(int)
for r in got:
    per_t[int(r[0])] += 1
assert set(per_t.values()) == {1}, dict(per_t)
print("  exactly one point on each of 11 frames   OK")

print("\n5. a division inside the window returns BOTH daughters, not neither")
# mother 300..302 (t=0..2), then two daughters from t=3
mom = [(300 + t, t, 50.0) for t in range(3)]
d1 = [(400 + t, t, 40.0) for t in range(3, 7)]
d2 = [(500 + t, t, 60.0) for t in range(3, 7)]
nid_xyz, kids, par = graph([mom, d1, d2])
kids[302] += [403, 503]
par[403].append(302)
par[503].append(302)
got = _walk(301, 0, 6, nid_xyz, par, kids)
per_t = defaultdict(int)
for r in got:
    per_t[int(r[0])] += 1
assert per_t[0] == 1 and per_t[2] == 1, dict(per_t)
assert per_t[3] == 2 and per_t[6] == 2, \
    f"after a division both daughters must be drawn, got {dict(per_t)}"
print(f"  frames 0-2 have 1 point, frames 3-6 have 2   OK  {dict(sorted(per_t.items()))}")

print("\n6. an isolated node with no edges returns just itself")
nid_xyz, kids, par = graph([[(900, 4, 7.0)]])
got = _walk(900, 0, 10, nid_xyz, par, kids)
assert got == [(4, 0.0, 7.0, 0.0)], got
print("  OK")

print("\n7. a cycle cannot hang the walk")
nid_xyz, kids, par = graph([[(1, 0, 1.0), (2, 1, 2.0), (3, 2, 3.0)]])
kids[3].append(1)
par[1].append(3)
got = _walk(1, 0, 5, nid_xyz, par, kids)
assert len(got) == 3, got
print("  a back-edge does not loop forever (the `seen` set holds)   OK")

print("\nALL CHECKS PASSED for", TARGET)
