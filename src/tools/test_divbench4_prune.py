"""Synthetic test of divbench4's compact snapshot and its pruner.

divbench4 keeps four graph snapshots per movie instead of divbench3's one, so
_snapshot was rewritten to hold float32 coordinates, an int32 edge array of
remapped row indices, and a (p, q) -> edge-row map restricted to FORK edges.
_rescore then deletes rows rather than filtering tuples. That bookkeeping is
new code, and a wrong index there silently prunes the wrong edge.

Run: py tools/test_divbench4_prune.py
"""
import pathlib
import sys
from collections import defaultdict

import numpy as np

SRC = pathlib.Path(__file__).resolve().parents[1] / (sys.argv[1] if len(sys.argv) > 1 else "divbench4.py")
lines = SRC.read_text(encoding="utf-8").split("\n")


def grab(start_marker, end_marker):
    a = next(i for i, x in enumerate(lines) if x.startswith(start_marker))
    b = next(i for i, x in enumerate(lines) if i > a and x.startswith(end_marker))
    return "\n".join(lines[a:b])


SAVED = {}


class _dr:
    @staticmethod
    def _save(path, arr, e3):
        SAVED["path"], SAVED["arr"], SAVED["e"] = path, np.asarray(arr), np.asarray(e3)


NS = dict(np=np, defaultdict=defaultdict, pathlib=pathlib, dr=_dr,
          TMP="/tmp/db4test", PRE_STATE={},
          score_one=lambda outp, name: {"n_edges": len(SAVED["e"])})
exec(grab("def _snapshot():", "def _judge("), NS)
_snapshot, _rescore = NS["_snapshot"], NS["_rescore"]

fails = []


def check(name, got, want):
    ok = got == want
    print(f"  {'ok ' if ok else 'FAIL'} {name}: {got}" + ("" if ok else f"  want {want}"))
    if not ok:
        fails.append(name)


# 1 -> 2, 1 -> 3   (a fork at 1)
# 4 -> 5           (a plain link)
# 5 -> 6, 5 -> 7   (a fork at 5)
nodes = {i: {"t": i % 3, "xyz": np.array([i, i * 2.0, i * 3.0])} for i in range(1, 8)}
edges = [(1, 2), (4, 5), (1, 3), (5, 6), (5, 7)]
ids = sorted(nodes)
NS["PRE_STATE"].update(nodes=nodes, final_edges=list(edges), ids=ids)

snap = _snapshot()
print("1. the snapshot")
check("coordinates are float32", snap["arr"].dtype, np.dtype(np.float32))
check("one row per node", snap["arr"].shape, (7, 4))
check("edges are int32 (E,2)", (snap["e"].dtype, snap["e"].shape),
      (np.dtype(np.int32), (5, 2)))
check("edges are remapped to row indices",
      [tuple(map(int, r)) for r in snap["e"]],
      [(0, 1), (3, 4), (0, 2), (4, 5), (4, 6)])
check("the fork map holds ONLY fork edges",
      sorted(snap["key"]), [(1, 2), (1, 3), (5, 6), (5, 7)])
check("and points at the right rows", {k: v for k, v in sorted(snap["key"].items())},
      {(1, 2): 0, (1, 3): 2, (5, 6): 3, (5, 7): 4})
check("the plain link 4->5 is NOT in it", (4, 5) in snap["key"], False)

print("\n2. pruning one fork edge removes exactly that edge")
_rescore("m", snap, {snap["key"][(1, 3)]}, "t")
check("one edge gone", len(SAVED["e"]), 4)
check("the right one gone", [tuple(map(int, r)) for r in SAVED["e"]],
      [(0, 1), (3, 4), (4, 5), (4, 6)])
check("nodes untouched (the node term must not move)", len(SAVED["arr"]), 7)

print("\n3. pruning several, out of order")
_rescore("m", snap, {snap["key"][(5, 7)], snap["key"][(1, 2)]}, "t")
check("two edges gone", [tuple(map(int, r)) for r in SAVED["e"]],
      [(3, 4), (0, 2), (4, 5)])

print("\n4. an empty drop set writes the graph unchanged")
_rescore("m", snap, set(), "t")
check("nothing removed", len(SAVED["e"]), 5)

print("\n5. the snapshot is a COPY: later edits to the live graph do not reach it")
edges.append((2, 4))
NS["PRE_STATE"]["final_edges"] = list(edges)
check("snapshot still has 5 edges", len(snap["e"]), 5)
snap2 = _snapshot()
check("a fresh snapshot has 6", len(snap2["e"]), 6)

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("all checks passed")
