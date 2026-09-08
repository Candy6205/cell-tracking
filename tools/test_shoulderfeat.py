"""Synthetic test of shoulderfeat's two load-bearing pieces, before the push.

There is NO ANCHOR for this run -- nothing has read the geffs this way before
-- so the usual safety net is missing and these are the two ways it could
produce a full, plausible, wrong report:

  1. `_runs`. The run-length table is the one number with an immediate use:
     it decides whether gap_close can reach these cells at all. If the walk
     never backs up to the start of a chain, every run of length L is counted
     as L separate runs of length 1 and the report says "all of them are
     single frames, bridgeable" when the truth is the opposite. That is a
     WRONG ANSWER to the question the run was for, and the in-kernel
     `sum(runs) == len(sh_ids)` assertion does NOT catch it -- both spellings
     sum to the same total.

  2. `decile_table`. If the bin edges collapse (ties, or NaN leaking into the
     percentiles) every bin shows the same rate, the depth question reads
     "no effect", and the run looks like a clean null. So: feed it a signal
     that IS there and check the table finds it, then feed it a null and
     check it does not invent one.

    py tools/test_shoulderfeat.py shoulderfeat.py
"""
import sys
from collections import defaultdict

import numpy as np

TARGET = sys.argv[1] if len(sys.argv) > 1 else "shoulderfeat.py"
SRC = open(TARGET, encoding="utf-8").read()


def cut(start, end):
    a = SRC.index(start)
    b = SRC.index(end, a)
    print(f"  extracted {start.strip()!r}: {b - a} chars")
    return SRC[a:b]


NS = {"np": np, "defaultdict": defaultdict, "print": print}
exec(cut("def _runs(", "\ndef _links("), NS)
exec(cut("def _links(", "\nfor name in MOVIES"), NS)
_runs, _links = NS["_runs"], NS["_links"]


def links(edges, ids):
    return _links(np.asarray(ids, np.int64), np.asarray(edges, np.int64))


print("\n1. _runs on lineages whose answer is known by construction")

# a straight chain 1->2->3->4->5, shoulder on 2,3,4 == ONE run of 3
kids, par = links([(1, 2), (2, 3), (3, 4), (4, 5)], [1, 2, 3, 4, 5])
got = sorted(_runs({2, 3, 4}, kids, par))
assert got == [3], f"a 3-frame hole must be ONE run of 3, got {got}"
print(f"  chain, shoulder on 3 consecutive       -> {got}   OK")

# the failure this test exists for: reaching the chain from its MIDDLE first.
# sorted() hands us 2 first here, but with ids that sort the other way the
# walk must still back up. Same graph, ids reversed in sort order:
kids, par = links([(50, 40), (40, 30), (30, 20), (20, 10)], [10, 20, 30, 40, 50])
got = sorted(_runs({40, 30, 20}, kids, par))
assert got == [3], f"walking a chain from its middle must still give [3], got {got}"
print(f"  same chain, entered from the middle    -> {got}   OK")

# isolated singles must NOT be glued together
kids, par = links([(1, 2), (2, 3), (3, 4), (4, 5)], [1, 2, 3, 4, 5])
got = sorted(_runs({2, 4}, kids, par))
assert got == [1, 1], f"two separated holes are two runs of 1, got {got}"
print(f"  chain, shoulder on 2 and 4 (a gap)     -> {got}   OK")

# a division: parent 2 forks to 3 and 4, all three shoulder.
kids, par = links([(1, 2), (2, 3), (2, 4), (3, 5), (4, 6)], [1, 2, 3, 4, 5, 6])
got = sorted(_runs({2, 3, 4}, kids, par))
assert sum(got) == 3, f"every shoulder cell must be counted once, got {got}"
assert got in ([1, 2], [1, 1, 1]), got
print(f"  fork, shoulder on parent + both kids   -> {got}   "
      f"(the docstring's conservative split)")

# every cell counted exactly once, on a messier graph
kids, par = links([(1, 2), (2, 3), (3, 4), (10, 11), (11, 12), (20, 21)],
                  [1, 2, 3, 4, 10, 11, 12, 20, 21])
sh = {2, 3, 4, 11, 12, 20}
got = sorted(_runs(sh, kids, par))
assert sum(got) == len(sh), (got, len(sh))
assert got == [1, 2, 3], got
print(f"  three separate lineages                -> {got}   OK")

# a shoulder cell with no edges at all (a one-frame GT track)
kids, par = links([], [7])
assert _runs({7}, kids, par) == [1]
print("  a node with no edges                   -> [1]   OK")

print("\n  the failure mode this rules out: a walk that does not back up would")
print("  report [1, 1, 1] for the 3-frame chain -- 'all single frames,")
print("  bridgeable' -- and sum(runs) == len(sh_ids) would still hold.")

# ---------------------------------------------------------------------------
print("\n2. decile_table finds a signal that is there, and no signal that isn't")

NDEC = 10
tbl_src = cut("def decile_table(", "\ndecile_table(ZN")


def run_table(val, sh, dom):
    """Run the kernel's own table function against arrays we control."""
    out = []
    ns = {"np": np, "NDEC": NDEC, "SH": np.asarray(sh, bool),
          "DOM": np.asarray(dom), "print": lambda *a, **k: out.append(
              " ".join(str(x) for x in a))}
    exec(tbl_src, ns)
    ns["decile_table"](np.asarray(val, float), "t")
    rates = []
    for ln in out[2:]:
        f = [p for p in ln.split() if p.endswith("%")]
        if f:
            rates.append(float(f[0].rstrip("%")))
    return rates


rng = np.random.default_rng(0)
N = 6000
z = rng.random(N)
dom = np.array(["6bba"] * N)

# a REAL depth effect: the rate rises from ~0% to ~20% across the stack
sh = rng.random(N) < (0.20 * z)
rates = run_table(z, sh, dom)
assert len(rates) == NDEC, f"{len(rates)} bins, expected {NDEC}"
print(f"  planted a rising 0->20% effect: bins read "
      f"{rates[0]:.1f}% .. {rates[-1]:.1f}%")
assert rates[-1] > rates[0] + 8.0, \
    f"FATAL: a 20-point effect did not show up in the table ({rates})"
print("  OK: the table can see a depth effect")

# a NULL: constant rate, no dependence on z
sh = rng.random(N) < 0.10
rates = run_table(z, sh, dom)
spread = max(rates) - min(rates)
print(f"  planted NO effect:              bins read "
      f"{min(rates):.1f}% .. {max(rates):.1f}%  (spread {spread:.1f} points)")
assert spread < 8.0, f"FATAL: the table invents an effect on flat data ({rates})"
print("  OK: and it does not invent one")

# NaNs must be dropped, not silently binned as a value
v = z.copy()
v[::7] = np.nan
rates = run_table(v, sh, dom)
assert len(rates) == NDEC and all(np.isfinite(rates)), rates
print(f"  1/7 of the values NaN:          still {len(rates)} finite bins   OK")

# ties: a column that is mostly one value must not crash or fake a gradient
v = np.zeros(N)
v[: N // 20] = 1.0
rates = run_table(v, sh, dom)
print(f"  95% ties:                       {len(rates)} bins, "
      f"spread {max(rates) - min(rates):.1f} points (degenerate by "
      f"construction -- it must not CRASH, and it does not)")

print("\nALL CHECKS PASSED for", TARGET)
