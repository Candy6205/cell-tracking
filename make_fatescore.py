"""Generate fatescore.py from mergescore.py by asserted replacement.

THE QUESTION (the user's, 2026-08-31, after M15 scored 0.917): on top of
F4, how many of the added ep05 points does the EDGE RANKER even take
into consideration, and how many does the POST-LINK deletion then throw
away? Follow-up, same night: "最好把每一步都分开来看" -- separate EVERY
stage, and reuse the existing code rather than reinventing it.

Nothing in this project has ever traced a DETECTION through the
pipeline. edgediag (2026-08-29) did the equivalent for GT EDGES,
charging each to the first stage that lost it; this is the node-side
counterpart and it borrows edgediag's two load-bearing techniques
verbatim:

  * the (t, i) -> solved-graph-id map built by ROUNDED COORDINATES
    rather than by assuming the graph numbers its nodes in coords order
    (edgediag lines 1500-1514). Nothing here depends on that convention.
  * a patch pass that RETURNS its result instead of only writing a geff,
    so the post-link stages are observable (edgediag's `patch_final`).

WHY IT MATTERS. M15 grew the output 2.01% and lost 0.008. "Survives to
the output" is not "is used": a node that reaches the geff having never
been linked is pure node-count cost with zero edge_J benefit, and that
is a different disease from "the point is wrong" with a different cure.
The number that decides between them is FINAL-OUTPUT ISOLATED NODES.

STAGES, each reported separately, for BASE / MERGED / REPAIR nodes:
   1 primary          entered the solve at all (score >= UNET_THRESH)
   2 in_pair          the ranker was asked about it
   3 past_floor       some edge of its own cleared ILP_FLOOR
   4 ilp_linked       the ILP selected an edge on it
   5 gap_gain         _gap_close attached an edge to it
   6 short_kill       _short_filter deleted it
   7 div_gain         _add_safe_divisions attached an edge to it
   8 final            present in the written geff
   9 final_isolated   present in the geff with ZERO edges  <-- the answer
  10 mean final degree

Two arms (F4 control, M15), all three domains, ~2.2 h CPU, no GPU quota.
ANCHORS: F4 must reproduce mergescore's 0.9198 / 0.8303 / 0.8115 and M15
its 0.9134 / 0.8363 / 0.8219, or the trace describes a different run
than the one it is being used to explain.
"""
import ast
import pathlib

SRC = pathlib.Path(__file__).parent / "mergescore.py"
DST = pathlib.Path(__file__).parent / "fatescore.py"
text = SRC.read_text(encoding="utf-8")


def rep(old, new, n=1):
    global text
    c = text.count(old)
    assert c == n, f"anchor x{c}, wanted x{n}: {old[:70]!r}"
    text = text.replace(old, new)


rep('SCRIPT_VERSION = "2026-08-31a-mergescore"',
    'SCRIPT_VERSION = "2026-08-31a-fatescore"')

# ---- 1. two arms, plus the anchors they must reproduce ------------------
rep('''RUN_ARMS = [
    ("BASE_D187", "BASE", "D187"),   # harness anchor: 6bba adj_J 0.9293
    ("F4",        "F4",   "D32"),    # the in-run baseline for deltas
    ("M50",       "MERGE:0.50", "D32"),
    ("M30",       "MERGE:0.30", "D32"),
    ("M15",       "MERGE:0.15", "D32"),
]''',
    '''RUN_ARMS = [
    ("F4",  "F4",         "D32"),    # control: the same points, no adds
    ("M15", "MERGE:0.15", "D32"),    # the arm whose points we are tracing
]
# mergescore's recorded rows. If these do not reproduce, the trace
# describes a different run than the one it is being used to explain.
ANCHOR_ADJ = {("F4", "6bba"): 0.9198, ("F4", "44b6"): 0.8303,
              ("F4", "zh"): 0.8115,
              ("M15", "6bba"): 0.9134, ("M15", "44b6"): 0.8363,
              ("M15", "zh"): 0.8219}''')

# ---- 2. merge_dets records the per-frame admitted count -----------------
rep("""def merge_dets(dets, name, tau):""",
    """_ADDED_FRAMES = []      # per-frame count of admitted ep05 points
_FATE = {}              # one dataset's trace, refilled per solve


def _degree(edges):
    d = defaultdict(int)
    for a_, b_ in edges:
        d[a_] += 1
        d[b_] += 1
    return d


def merge_dets(dets, name, tau):""")

rep("""    z = np.load(DCDET_EP05[name])
    out = []""",
    """    z = np.load(DCDET_EP05[name])
    out = []
    _ADDED_FRAMES.clear()""")

rep("""        st[1] += len(dc)
        if len(dc):""",
    """        st[1] += len(dc)
        _ADDED_FRAMES.append(len(dc))
        if len(dc):""")

# ---- 3. solve_one traces every pre-ILP checkpoint in (t, i) space -------
rep("""    del graph, feats, probs
    return nodes0, edges0, (cand, cand_sc, cand_trees)""",
    """    # --- fate trace, part 1: everything up to and including the ILP.
    # Work in (t, i) SEED space -- that is the space `feats` and `probs`
    # already speak -- and cross to graph ids only by rounded
    # coordinates, edgediag's technique. Nothing here assumes the graph
    # numbers its nodes in coords order.
    _id_of = {}
    for _g, _v in nodes0.items():
        _id_of[(int(_v["t"]), tuple(int(round(float(_x)))
                                    for _x in _v["xyz"]))] = _g
    _seed_id, _unmapped = {}, 0
    for _t in range(T):
        _P = seeds[_t][0]
        for _i in range(len(_P)):
            _g = _id_of.get((_t, tuple(int(round(float(_x)))
                                       for _x in _P[_i])))
            if _g is None:
                _unmapped += 1
            else:
                _seed_id[(_t, _i)] = _g
    # Merged points are APPENDED by merge_dets and enter at exactly
    # UNET_THRESH, so seeds_of's `>= UNET_THRESH` mask keeps every one
    # and keeps them last: the tail k_t of frame t are the ep05 adds.
    _nseed = [len(seeds[_t][0]) for _t in range(T)]
    _merged = set()
    for _t in range(T):
        _k = _ADDED_FRAMES[_t] if _t < len(_ADDED_FRAMES) else 0
        for _m in range(_k):
            _merged.add((_t, _nseed[_t] - _k + _m))
    _in_pair, _past_floor = set(), set()
    for _t in range(T - 1):
        _pairs = feats[_t][0]
        for (_i, _j), _pr in zip(_pairs, probs[_t]):
            _in_pair.add((_t, _i))
            _in_pair.add((_t + 1, _j))
            if _pr >= ILP_FLOOR:
                _past_floor.add((_t, _i))
                _past_floor.add((_t + 1, _j))
    _key2ti = {_g: _ti for _ti, _g in _seed_id.items()}
    _deg0 = _degree(edges0)
    _ilp_linked = {_key2ti[_g] for _g in _deg0 if _g in _key2ti}
    _FATE.clear()
    _FATE.update({"merged": _merged, "nseed": _nseed, "T": T,
                  "in_pair": _in_pair, "past_floor": _past_floor,
                  "ilp_linked": _ilp_linked, "seed_id": _seed_id,
                  "key2ti": _key2ti, "unmapped": _unmapped,
                  "ilp_nodes0": set(nodes0)})
    del graph, feats, probs
    return nodes0, edges0, (cand, cand_sc, cand_trees)""")

# ---- 4. patch_and_save snapshots after EACH post-link stage -------------
rep("""    nid2 = len(nodes) + 1
    nid2 = _gap_close(nodes, edges, cand, cand_sc, cand_trees, nid2)
    nodes, edges = _short_filter(nodes, edges)
    _linefit(nodes, edges)
    _add_safe_divisions(nodes, edges)""",
    """    nid2 = len(nodes) + 1
    _d_ilp = _degree(edges)
    nid2 = _gap_close(nodes, edges, cand, cand_sc, cand_trees, nid2)
    _d_gap = _degree(edges)
    _pre_short = set(nodes)
    nodes, edges = _short_filter(nodes, edges)
    _post_short = set(nodes)
    _d_short = _degree(edges)
    _linefit(nodes, edges)
    _add_safe_divisions(nodes, edges)
    _FATE.update({"d_ilp": _d_ilp, "d_gap": _d_gap, "d_short": _d_short,
                  "d_final": _degree(edges),
                  "killed_short": _pre_short - _post_short,
                  "post_short": _post_short,
                  "final_nodes": set(nodes)})""")

# ---- 5. the accumulator, folded in after every solve --------------------
rep("""res = {}
PER = {}
t_all = time.time()""",
    '''res = {}
PER = {}
t_all = time.time()

# (arm, domain) -> group -> counter name -> int
FATE = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
# (arm, domain) -> group -> final degree -> count
DEGHIST = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
# (name, arm) -> frozenset of (t, i) BASE seeds the ILP linked
BASE_LINKED = {}
UNMAPPED = defaultdict(int)


def collect_fate(lbl, domain, name):
    """Fold one solved dataset's trace into the running totals.

    Three groups, kept apart because they are three different
    populations: BASE (F4's own detections), MERGED (the ep05 adds), and
    REPAIR (nodes _gap_close created from the sub-threshold pool, which
    exist in neither seed set)."""
    F = FATE[(lbl, domain)]
    H = DEGHIST[(lbl, domain)]
    merged, seed_id = _FATE["merged"], _FATE["seed_id"]
    killed, final = _FATE["killed_short"], _FATE["final_nodes"]
    d_ilp, d_gap = _FATE["d_ilp"], _FATE["d_gap"]
    d_short, d_fin = _FATE["d_short"], _FATE["d_final"]
    UNMAPPED[(lbl, domain)] += _FATE["unmapped"]
    base_linked = set()
    for t_ in range(_FATE["T"]):
        for i_ in range(_FATE["nseed"][t_]):
            ti = (t_, i_)
            grp = "merged" if ti in merged else "base"
            r = F[grp]
            r["primary"] += 1
            r["in_pair"] += ti in _FATE["in_pair"]
            r["past_floor"] += ti in _FATE["past_floor"]
            lk = ti in _FATE["ilp_linked"]
            r["ilp_linked"] += lk
            if grp == "base" and lk:
                base_linked.add(ti)
            g = seed_id.get(ti)
            if g is None:
                r["absent_from_graph"] += 1
                continue
            r["gap_gain"] += d_gap.get(g, 0) > d_ilp.get(g, 0)
            r["short_kill"] += g in killed
            r["div_gain"] += d_fin.get(g, 0) > d_short.get(g, 0)
            if g in final:
                r["final"] += 1
                deg = d_fin.get(g, 0)
                r["final_deg_sum"] += deg
                r["final_isolated"] += deg == 0
                H[grp][min(deg, 3)] += 1
    # REPAIR nodes: in the final graph but in no seed set.
    seed_keys = set(seed_id.values())
    for g in final:
        if g not in seed_keys:
            r = F["repair"]
            r["primary"] += 1
            r["final"] += 1
            deg = d_fin.get(g, 0)
            r["final_deg_sum"] += deg
            r["final_isolated"] += deg == 0
            H["repair"][min(deg, 3)] += 1
    BASE_LINKED[(name, lbl)] = frozenset(base_linked)''')

# ---- 6. hook the two arm loops ------------------------------------------
# `merge_dets` clears and refills _ADDED_FRAMES itself, so an
# unconditional clear before every arm is enough to keep a non-merge arm
# from inheriting the previous arm's counts.
rep("""            set_ranker(rk)
            dets = load_dets(name, det)""",
    """            set_ranker(rk)
            _ADDED_FRAMES.clear()
            dets = load_dets(name, det)""")

rep("""            nodes0, edges0, pool = solve_one(dets, T)
            patch_and_save(nodes0, edges0, pool,
                           pdir(lbl, domain) / f"{name}.geff")
            del dets, nodes0, edges0, pool""",
    """            nodes0, edges0, pool = solve_one(dets, T)
            patch_and_save(nodes0, edges0, pool,
                           pdir(lbl, domain) / f"{name}.geff")
            collect_fate(lbl, domain, name)
            del dets, nodes0, edges0, pool""")

rep("""            if det == "BASE":
                dets = dets_base""",
    """            _ADDED_FRAMES.clear()
            if det == "BASE":
                dets = dets_base""")

rep("""            patch_and_save(nodes0, edges0, pool, outp)
            rr = td.graph.IndexedRXGraph.from_geff(str(outp))""",
    """            patch_and_save(nodes0, edges0, pool, outp)
            collect_fate(lbl, "zh", name)
            rr = td.graph.IndexedRXGraph.from_geff(str(outp))""")

# ---- 7. the report ------------------------------------------------------
rep('''print("\\n=== reading it ===", flush=True)''',
    '''def _pct(a, b):
    return f"{100.0 * a / b:.1f}%" if b else "  -  "


print("\\n=== ANCHORS: mergescore's rows must reproduce ===", flush=True)
_bad = 0
for (_l, _d), _want in sorted(ANCHOR_ADJ.items()):
    _got = res.get((_l, _d), {}).get("adj_edge_jaccard", float("nan"))
    _ok = abs(_got - _want) < 0.0002
    _bad += not _ok
    print(f"  {_l:<4} {_d:<5} adj_J {_got:.4f} (mergescore {_want:.4f}) "
          f"{'OK' if _ok else 'MISMATCH'}", flush=True)
print("  " + ("all anchors hold" if not _bad else
      f"{_bad} MISMATCHED -- the trace describes a different run"),
      flush=True)
_um = sum(UNMAPPED.values())
print(f"  seeds that never reached the solved graph: {_um:,} "
      f"(these are counted in `primary` and in `absent_from_graph`, "
      f"never in the post-link columns)", flush=True)

print("\\n=== EVERY STAGE, SEPARATELY, BY GROUP ===", flush=True)
print("  base   = F4's own detections. merged = the ep05 points M15 "
      "adds. repair = nodes _gap_close created from the sub-threshold "
      "pool (they are in no seed set, so only their output columns "
      "exist).", flush=True)
print("  Percentages are of that group's OWN primary count, so base and "
      "merged rows are directly comparable.", flush=True)
for _d in ("6bba", "44b6", "zh"):
    for _l, _det, _rk in RUN_ARMS:
        if (_l, _d) not in FATE:
            continue
        print(f"\\n  -- {_l} / {_d} --", flush=True)
        print(f"  {'group':<7} {'primary':>9} {'in_graph':>9} "
              f"{'in_pair':>8} {'floor':>8} {'ILP':>8} {'gap+':>7} "
              f"{'short-':>7} {'div+':>6} {'final':>8} {'ISOLATED':>9} "
              f"{'deg':>6}", flush=True)
        for _g in ("base", "merged", "repair"):
            r = FATE[(_l, _d)][_g]
            n = r["primary"]
            if not n:
                continue
            fin = r["final"]
            print(f"  {_g:<7} {n:>9,} "
                  f"{_pct(n - r['absent_from_graph'], n):>9} "
                  f"{_pct(r['in_pair'], n):>8} "
                  f"{_pct(r['past_floor'], n):>8} "
                  f"{_pct(r['ilp_linked'], n):>8} "
                  f"{_pct(r['gap_gain'], n):>7} "
                  f"{_pct(r['short_kill'], n):>7} "
                  f"{_pct(r['div_gain'], n):>6} "
                  f"{_pct(fin, n):>8} "
                  f"{_pct(r['final_isolated'], n):>9} "
                  f"{r['final_deg_sum'] / fin if fin else 0:>6.2f}",
                  flush=True)
print("\\n  in_graph < 100% would mean the ILP solver PRUNES unselected "
      "nodes, in which case 'ISOLATED' cannot be nonzero and the "
      "isolated-node reading is unavailable -- check that column before "
      "reading anything else.", flush=True)

print("\\n=== FINAL-OUTPUT DEGREE HISTOGRAM (3 means 3 or more) ===",
      flush=True)
for _d in ("6bba", "44b6", "zh"):
    for _l, _det, _rk in RUN_ARMS:
        if (_l, _d) not in DEGHIST:
            continue
        for _g in ("base", "merged", "repair"):
            h = DEGHIST[(_l, _d)][_g]
            tot = sum(h.values())
            if not tot:
                continue
            print(f"  {_l:<4} {_d:<5} {_g:<7} n={tot:>8,}  "
                  + "  ".join(f"deg{k}={_pct(h.get(k, 0), tot)}"
                              for k in (0, 1, 2, 3)), flush=True)

print("\\n=== DID THE ADDED POINTS STEAL LINKS FROM THE BASE POINTS? ===",
      flush=True)
print("  The base points are the SAME points in both arms and carry the "
      "same (t, i) indices, so their ILP-linked sets compare directly. "
      "'lost' = linked under F4 and NOT under M15. This is the "
      "capacity-2 mechanism that would explain 6bba's edge_J falling "
      "0.0037 while its node count rose.", flush=True)
print(f"  {'dataset':<20} {'F4 linked':>12} {'M15':>12} {'lost':>8} "
      f"{'gained':>8} {'net':>9}", flush=True)
_agg = defaultdict(lambda: [0, 0, 0, 0])
for _n in sorted({k[0] for k in BASE_LINKED}):
    if (_n, "F4") not in BASE_LINKED or (_n, "M15") not in BASE_LINKED:
        continue
    _a, _b = BASE_LINKED[(_n, "F4")], BASE_LINKED[(_n, "M15")]
    _lost, _gain = len(_a - _b), len(_b - _a)
    _dm = _n.split("_")[0] if _n.startswith(("6bba", "44b6")) else "zh"
    _ag = _agg[_dm]
    _ag[0] += len(_a)
    _ag[1] += len(_b)
    _ag[2] += _lost
    _ag[3] += _gain
    print(f"  {_n:<20} {len(_a):>12,} {len(_b):>12,} {_lost:>8,} "
          f"{_gain:>8,} {len(_b) - len(_a):>+9,}", flush=True)
print("  " + "-" * 76, flush=True)
for _dm, _ag in sorted(_agg.items()):
    print(f"  {_dm:<20} {_ag[0]:>12,} {_ag[1]:>12,} {_ag[2]:>8,} "
          f"{_ag[3]:>8,} {_ag[1] - _ag[0]:>+9,}   lost "
          f"{_pct(_ag[2], _ag[0])} of F4's links", flush=True)

print("\\n=== how to read this ===", flush=True)
print("  The decisive column is ISOLATED: merged nodes that reach the "
      "geff with ZERO edges are pure node-count cost -- they pay the "
      "penalty and buy no edge_J. If that number is large, M15 failed "
      "because the LINKER could not use the points, which is fixable "
      "without a spurious-detection label. If it is small and the "
      "steal table shows base links lost instead, M15 failed because "
      "the added points DISPLACED real links under capacity-2, which "
      "is a different problem and argues the points are wrong after "
      "all. Both readings are consistent with 0.917; this run "
      "separates them.", flush=True)

print("\\n=== reading it (stale mergescore verdict text follows) ===",
      flush=True)''')

DST.write_text(text, encoding="utf-8")
ast.parse(text)
print(f"wrote {DST} ({len(text.splitlines())} lines), ast OK")
