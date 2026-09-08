"""Synthetic test of divbench4's two new pieces, run locally before the
kernel finishes so a crash costs a re-push and not a night.

Both are cut OUT OF divbench4.py by line content and exec'd in a stub
namespace, so this tests the shipped text, not a copy of it:

  _add_safe_divisions  -- must be byte-identical in behaviour to divbench3
                          when steal is None, and must actually move the
                          edge (remove r->q, add p->q) when it is not
  fork_rows            -- the eight new feature columns

Run: py tools/test_divbench4_steal.py
"""
import pathlib
import sys
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

SRC = pathlib.Path(__file__).resolve().parents[1] / (sys.argv[1] if len(sys.argv) > 1 else "divbench4.py")
text = SRC.read_text(encoding="utf-8")
lines = text.split("\n")


def grab(start_marker, end_marker):
    a = next(i for i, x in enumerate(lines) if x.startswith(start_marker))
    b = next(i for i, x in enumerate(lines) if i > a and x.startswith(end_marker))
    return "\n".join(lines[a:b])


SCALE = np.array([1.625, 0.40625, 0.40625])


def _dist_um(a, b):
    return float(np.linalg.norm((np.asarray(a, float) - np.asarray(b, float)) * SCALE))


def _chain_len(start, step_map, cap=12):
    n, cur = 0, start
    while n < cap:
        nxt = step_map.get(cur, [])
        if len(nxt) != 1:
            break
        cur = nxt[0]
        n += 1
    return n


NS = dict(np=np, cKDTree=cKDTree, defaultdict=defaultdict, SCALE=SCALE,
          _dist_um=_dist_um, _chain_len=_chain_len,
          DIV_PARENT_UM=12.0, DIV_SISTER_UM=15.0, DIV_CHILD_UM=10.0,
          DIV_DIVERGE_UM=2.25, DIV_W_SISTER=0.15,
          DIV_FRAME_CAP=0.0076, DIV_GLOBAL_CAP=0.00375,
          FORK_LOG={}, PRE_STATE={})
exec(grab("STEAL_LOG = {}", "def _adj_of"), NS)
exec(grab("def _aux_extra", "def walk_lost"), NS)
add_div = NS["_add_safe_divisions"]
fork_rows = NS["fork_rows"]
FORK_LOG = NS["FORK_LOG"]
STEAL_LOG = NS["STEAL_LOG"]
PATCH_CAPS = NS["PATCH_CAPS"]
FEAT_NAMES = ["src", "d_pc1", "d_pq", "d_s", "dv", "p_c1", "p_q", "pf_c1", "pf_q",
              "pr_q", "sc_p", "sc_c1", "sc_q", "len_back", "len_c1", "len_q",
              "dens10", "n_orph", "t_frac", "rank_gap",
              "dv2", "ang", "sc_q_r", "sc_p_r", "dens5", "p_q_marg", "q_ends", "steal"]
NS["FEAT_NAMES"] = FEAT_NAMES

fails = []


def check(name, got, want):
    ok = got == want
    print(f"  {'ok ' if ok else 'FAIL'} {name}: {got}" + ("" if ok else f"  want {want}"))
    if not ok:
        fails.append(name)


def build():
    """A mother M with one child C1 at t=1, plus a node Q at t=1 that already
    has a parent R at t=0. Q sits 6 um from M, C1 8 um from Q, and the two
    lines diverge by more than 2.25 um at t=2. Coordinates are VOXELS; SCALE
    turns them into um (z 1.625, y/x 0.40625), so y/x steps of ~2.5 voxels are
    ~1 um.
    """
    def xyz(z, y, x):
        return np.array([z, y, x], float)
    nodes, edges = {}, []
    # M(1) -> C1(2) -> C1n(3);  R(4) -> Q(5) -> Qn(6)
    nodes[1] = {"t": 0, "xyz": xyz(10, 100, 100)}
    nodes[2] = {"t": 1, "xyz": xyz(10, 100, 108)}      # C1, ~3.25 um from M
    nodes[3] = {"t": 2, "xyz": xyz(10, 100, 116)}
    nodes[4] = {"t": 0, "xyz": xyz(10, 130, 100)}      # R, far from M
    nodes[5] = {"t": 1, "xyz": xyz(10, 100, 84)}       # Q, 6.5 um from M
    nodes[6] = {"t": 2, "xyz": xyz(10, 100, 60)}       # Qn, diverging hard
    nodes[7] = {"t": 0, "xyz": xyz(60, 200, 200)}      # M's own parent
    nodes[9] = {"t": 3, "xyz": xyz(10, 100, 124)}      # C1 two frames on
    nodes[10] = {"t": 3, "xyz": xyz(10, 100, 40)}      # Q two frames on
    edges += [(7, 1), (1, 2), (2, 3), (4, 5), (5, 6), (3, 9), (6, 10)]
    return nodes, edges


print("1. steal=None must add nothing here (Q is not an orphan)")
nodes, edges = build()
FORK_LOG.clear()
STEAL_LOG.clear()
PATCH_CAPS.clear()
n_add, n_st = add_div(nodes, edges)
check("orphan pass adds", (n_add, n_st), (0, 0))
check("edges unchanged", len(edges), 7)

print("\n2. steal=-inf must move the edge: remove R->Q, add M->Q")
nodes, edges = build()
FORK_LOG.clear()
STEAL_LOG.clear()
PATCH_CAPS.clear()
n_add, n_st = add_div(nodes, edges, pfun=lambda a, b: 0.5, steal=float("-inf"))
check("steal count", (n_add, n_st), (0, 1))
check("(4,5) removed", (4, 5) in edges, False)
check("(1,5) added", (1, 5) in edges, True)
check("edge total unchanged", len(edges), 7)
check("FORK_LOG marks it a steal", FORK_LOG.get((1, 5), {}).get("steal"), 1)
check("STEAL_LOG records the loser", STEAL_LOG.get((1, 5), {}).get("r"), 4)

print("\n3. a margin rule must refuse when the mother's edge is the weaker one")
nodes, edges = build()
FORK_LOG.clear()
STEAL_LOG.clear()
PATCH_CAPS.clear()


def pf_weak(a, b):
    return 0.18 if a == 1 else 0.77          # divsteal's Zebrahub medians


n_add, n_st = add_div(nodes, edges, pfun=pf_weak, steal=0.0)
check("margin 0 refuses", (n_add, n_st), (0, 0))
check("counted as a margin refusal", PATCH_CAPS["steal_margin"], 1)
check("(4,5) still there", (4, 5) in edges, True)

print("\n4. the same margin rule must ACCEPT when the mother's edge is stronger")
nodes, edges = build()
FORK_LOG.clear()
STEAL_LOG.clear()
PATCH_CAPS.clear()
n_add, n_st = add_div(nodes, edges, pfun=lambda a, b: 0.80 if a == 1 else 0.20, steal=0.0)
check("margin 0 accepts", (n_add, n_st), (0, 1))
check("margin 0.20 is the boundary", 0.80 - 0.20 > 0.20, True)

print("\n5. an unconditional steal must fire even with NO ranker probability")
nodes, edges = build()
FORK_LOG.clear()
STEAL_LOG.clear()
PATCH_CAPS.clear()
n_add, n_st = add_div(nodes, edges, pfun=lambda a, b: float("nan"), steal=float("-inf"))
check("fires without probabilities", (n_add, n_st), (0, 1))
check("and says so", PATCH_CAPS["steal_no_prob"], 1)
nodes, edges = build()
FORK_LOG.clear()
STEAL_LOG.clear()
PATCH_CAPS.clear()
n_add, n_st = add_div(nodes, edges, pfun=lambda a, b: float("nan"), steal=0.0)
check("a margin arm does NOT", (n_add, n_st), (0, 0))

print("\n6. the steal must refuse a competitor that already forks")
nodes, edges = build()
nodes[8] = {"t": 1, "xyz": np.array([10, 160, 100], float)}
edges.append((4, 8))                                   # R now has two children
FORK_LOG.clear()
STEAL_LOG.clear()
PATCH_CAPS.clear()
n_add, n_st = add_div(nodes, edges, pfun=lambda a, b: 0.5, steal=float("-inf"))
check("refuses a forking competitor", (n_add, n_st), (0, 0))

print("\n7. the ORPHAN path is untouched: drop R and the fork appears as before")
nodes, edges = build()
del nodes[4]
edges.remove((4, 5))
FORK_LOG.clear()
STEAL_LOG.clear()
PATCH_CAPS.clear()
n_add, n_st = add_div(nodes, edges)                    # steal=None
check("orphan fork made with steal off", (n_add, n_st), (1, 0))
check("FORK_LOG says NOT a steal", FORK_LOG.get((1, 5), {}).get("steal"), 0)

print("\n8. the eight new fork features compute on a real fork")
nodes, edges = build()
del nodes[4]
edges.remove((4, 5))
FORK_LOG.clear()
STEAL_LOG.clear()
PATCH_CAPS.clear()
add_div(nodes, edges)
ids = sorted(nodes)
NS["PRE_STATE"].update(nodes=nodes, final_edges=list(edges), ids=ids)
T = 4
seeds = []
for t in range(T):
    g = [i for i in ids if nodes[i]["t"] == t]
    seeds.append((np.array([nodes[i]["xyz"] for i in g], float),
                  np.array([0.2 + 0.1 * j for j in range(len(g))], float)))
seed_of_node = {}
for t in range(T):
    g = [i for i in ids if nodes[i]["t"] == t]
    for j, i in enumerate(g):
        seed_of_node[i] = (t, j)
pmap = {}
for (a, b) in [(1, 2), (1, 5), (2, 3), (5, 6), (7, 1), (3, 9), (6, 10)]:
    sa, sb = seed_of_node[a], seed_of_node[b]
    pmap[(sa[0], sa[1], sb[1])] = (0.6, 0.7, 0.3)
scsort, best2 = NS["_aux_extra"](seeds, pmap)
rows = fork_rows("synthetic", 0, seeds, (pmap, seed_of_node, scsort, best2), [], [])
check("one fork found", len(rows), 1)
f = dict(zip(FEAT_NAMES, rows[0][5]))
print("     " + "  ".join(f"{k}={f[k]:.3f}" for k in
                          ("dv", "dv2", "ang", "sc_q_r", "sc_p_r", "dens5",
                           "p_q_marg", "q_ends", "steal")))
check("ang is obtuse (daughters go opposite ways)", f["ang"] > 150.0, True)
check("dv2 > dv (they keep separating)", f["dv2"] > f["dv"], True)
check("sc_q_r is a fraction", 0.0 <= f["sc_q_r"] <= 1.0, True)
check("q_ends is 0 (the chain runs to the last frame)", f["q_ends"], 0.0)
check("steal is 0 on an orphan fork", f["steal"], 0.0)
check("no unexpected NaN", [k for k, v in f.items()
                            if isinstance(v, float) and v != v], ["p_q_marg"])

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("all checks passed")

# ---- divbench7 only: the patch's per-frame / global caps as parameters ----
if "frame_cap" in text:
    print("\n9. the per-frame cap: two orphan forks in one frame, cap allows one")

    def build2():
        """Two independent mother/child/orphan triples, both at t=0->1, so both
        forks want the same frame. n_frame is tiny, so the shipped cap
        max(1, int(0.0076 * n)) is exactly 1 and the second must be refused."""
        def xyz(z, y, x):
            return np.array([z, y, x], float)
        nodes, edges = {}, []
        base = 0
        for k in range(2):
            o = base + k * 10
            dy = k * 400                      # far apart, no interaction
            nodes[o + 1] = {"t": 0, "xyz": xyz(10, 100 + dy, 100)}
            nodes[o + 2] = {"t": 1, "xyz": xyz(10, 100 + dy, 108)}
            nodes[o + 3] = {"t": 2, "xyz": xyz(10, 100 + dy, 116)}
            nodes[o + 5] = {"t": 1, "xyz": xyz(10, 100 + dy, 84)}
            nodes[o + 6] = {"t": 2, "xyz": xyz(10, 100 + dy, 60)}
            nodes[o + 7] = {"t": 0, "xyz": xyz(60, 200 + dy, 200)}
            edges += [(o + 7, o + 1), (o + 1, o + 2), (o + 2, o + 3), (o + 5, o + 6)]
        return nodes, edges

    nodes, edges = build2()
    FORK_LOG.clear()
    PATCH_CAPS.clear()
    n_add, _ = add_div(nodes, edges)
    check("shipped caps admit one of two", n_add, 1)
    # 12 nodes: max(1, int(0.00375 * 12)) = 1, so the GLOBAL cap binds first
    check("on a 12-node graph the GLOBAL cap is what refuses it",
          PATCH_CAPS["orphan_global"], 1)

    # to see the FRAME cap alone, the graph must be big enough that the global
    # cap allows 2: int(0.00375 * N) >= 2 needs N >= 534, and the frame then
    # holds ~N/3 nodes so int(0.0076 * 178) = 1 still refuses the second.
    nodes, edges = build2()
    for j in range(600):
        nodes[10000 + j] = {"t": j % 3, "xyz": np.array([50.0, 2000.0 + j, 3000.0])}
    FORK_LOG.clear()
    PATCH_CAPS.clear()
    n_add, _ = add_div(nodes, edges)
    check("with 612 nodes the global cap allows 2", n_add, 1)
    check("and the FRAME cap is what refuses the second",
          PATCH_CAPS["orphan_frame"], 1)
    check("the global cap did not fire", PATCH_CAPS["orphan_global"], 0)

    nodes, edges = build2()
    FORK_LOG.clear()
    PATCH_CAPS.clear()
    n_add, _ = add_div(nodes, edges, frame_cap=1.0, global_cap=1.0)
    check("uncapped admits both", n_add, 2)
    check("and refuses nothing by frame", PATCH_CAPS["orphan_frame"], 0)
    check("nor by global", PATCH_CAPS["orphan_global"], 0)

    nodes, edges = build2()
    FORK_LOG.clear()
    PATCH_CAPS.clear()
    n_add, _ = add_div(nodes, edges, frame_cap=1.0, global_cap=0.0)
    check("a zero global cap still allows the floor of 1", n_add, 1)
    check("and says the GLOBAL cap did it", PATCH_CAPS["orphan_global"], 1)

    if fails:
        print(f"\n{len(fails)} FAILED: {fails}")
        sys.exit(1)
    print("\ncap checks passed too")
