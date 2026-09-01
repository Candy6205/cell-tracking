"""Generate divgate2.py from divwhy.py by asserted replacement.

THE QUESTION (the user's, 2026-09-02): when the ILP links only ONE
daughter, why does the orphan-fork patch not pick up the other one?
_add_safe_divisions exists for exactly that case and divwhy measured it
firing on 456 / 861 / 128 forks per domain while GT divisions sat in
div_ilp_one_only unrescued.

The patch imposes EIGHT conditions in sequence and none of them has ever
been measured against a real division. Reading them off the source:

  1 the mother has exactly one child in the graph
  2 the mother has exactly one predecessor   <- a track start is skipped
  3 that child is at t+1
  4 mother->child <= DIV_CHILD_UM 10 um
  5 the OTHER daughter is an ORPHAN (in-degree 0) and within
    DIV_PARENT_UM 12 um of the mother
  6 sister distance <= DIV_SISTER_UM 15 um
  7 that orphan is the NEAREST orphan to the linked daughter
  8 both daughters have exactly one child each, and those two children
    are DIV_DIVERGE_UM 2.25 um FURTHER apart than the daughters are

Two are already ruled out by divwhy's own numbers: the geometric gates
(4, 6) pass comfortably -- the div_ilp_one_only bucket runs md_min 2.71
(6bba) / 3.07 (zh) against a 10 um limit and sister 7.11 / 7.64 against
15 -- and the caps do not bind: DIV_GLOBAL_CAP 0.00375 allows 616 / 1236
/ 784 forks against 456 / 861 / 128 actually emitted.

Condition 8 is the one with a number on both sides. div_kinematics
measured real daughters separating at sep_slope 1.155 um/frame (against
0.082 for random pairs). The test demands 2.25 um of EXTRA separation in
a single frame -- about twice the median rate of a real division. If
that is what fires, the patch is calibrated against a divergence real
divisions do not reach.

WHAT THIS ADDS. For every GT division the ILP half-linked (or did not
link at all), replay the patch's conditions in order on the PRE-PATCH
graph and record the FIRST one that rejects it. The pre-patch edge set
is recoverable without touching the pipeline: _add_safe_divisions only
appends, and its edges are exactly the final edges that are absent from
the ILP solution and whose endpoints are both seeds (a _gap_close bridge
always has a non-seed endpoint).

Same harness, same three domains, same anchors as divwhy.
"""
import ast
import pathlib

SRC = pathlib.Path(__file__).parent / "divwhy.py"
DST = pathlib.Path(__file__).parent / "divgate2.py"
text = SRC.read_text(encoding="utf-8")


def rep(old, new, n=1):
    global text
    c = text.count(old)
    assert c == n, f"anchor x{c}, wanted x{n}: {old[:70]!r}"
    text = text.replace(old, new)


rep('"""divwhy -- under what conditions does F4 lose a real division, and '
    'invent a false one?',
    '"""divgate2 -- which of the orphan-fork patch\'s eight conditions '
    'rejects a real division?')
rep('SCRIPT_VERSION = "2026-09-02b-divwhy"',
    'SCRIPT_VERSION = "2026-09-02c-divgate2"')

# ---- 1. remember the graph ids the ledger already resolved -------------
rep('''        dd[b_] += 1
        pm_ = gt_pos[u_] * SCALE''',
    '''        dd[b_] += 1
        if b_ in ("div_ilp_one_only", "div_ilp_neither"):
            _PEND.append((b_, ga_, g1_, g2_))
        pm_ = gt_pos[u_] * SCALE''')
rep('''    dd.setdefault("gt_multi", 0)
    drows = []''',
    '''    dd.setdefault("gt_multi", 0)
    drows = []
    _PEND = []          # (bucket, mother_gid, d1_gid, d2_gid) to replay''')

# ---- 2. replay the patch's conditions on the pre-patch graph -----------
rep('''    # ---- EVERY FORK THE PIPELINE EMITS, true or false -----------------''',
    r'''    # ---- WHY DID THE ORPHAN-FORK PATCH NOT FIRE? ----------------------
    # _add_safe_divisions only APPENDS, so the graph it saw is the final
    # graph minus its own edges: those are the final edges absent from the
    # ILP solution whose endpoints are BOTH seeds -- a _gap_close bridge
    # always has a non-seed endpoint, which is what separates the two.
    _sg = set(seed_id.values())
    patch_e = {(a_, b2_) for a_, b2_ in edges_f
               if (a_, b2_) not in e0set and a_ in _sg and b2_ in _sg}
    pre_e = [(a_, b2_) for a_, b2_ in edges_f if (a_, b2_) not in patch_e]
    pre_set = set(pre_e)
    pre_succ = defaultdict(list)
    pre_pred = defaultdict(list)
    for a_, b2_ in pre_e:
        pre_succ[a_].append(b2_)
        pre_pred[b2_].append(a_)
    pre_by_t = defaultdict(list)
    for g_, v_ in nodes_f.items():
        pre_by_t[int(v_["t"])].append(g_)
    orph_ids = {}
    orph_tree = {}
    for t_, ids_ in pre_by_t.items():
        o_ = [g_ for g_ in ids_ if not pre_pred.get(g_)]
        orph_ids[t_] = o_
        if o_:
            orph_tree[t_] = cKDTree(
                np.asarray([nodes_f[g_]["xyz"] for g_ in o_], float)
                * SCALE)
    gg = GAGG.setdefault(domain, defaultdict(int))
    for b_, ga_, g1_, g2_ in _PEND:
        gg["pending"] += 1
        if ga_ not in nodes_f:
            gg["mother_deleted_post"] += 1
            continue
        # which daughter did the ILP keep, and which one is the patch's job
        kept = [g_ for g_ in (g1_, g2_)
                if g_ in nodes_f and (ga_, g_) in pre_set]
        ch = pre_succ.get(ga_, [])
        if len(ch) != 1:
            gg["1_mother_not_one_child"] += 1
            continue
        c1 = ch[0]
        t_ = int(nodes_f[ga_]["t"])
        q = g2_ if (kept and kept[0] == g1_) else g1_
        if q not in nodes_f:
            gg["0_sister_deleted_post"] += 1
            continue
        if len(pre_pred.get(ga_, [])) != 1:
            gg["2_mother_no_single_pred"] += 1
            continue
        if int(nodes_f[c1]["t"]) != t_ + 1:
            gg["3_child_wrong_frame"] += 1
            continue
        if _dist_um(nodes_f[ga_]["xyz"], nodes_f[c1]["xyz"]) > DIV_CHILD_UM:
            gg["4_mother_child_over_10um"] += 1
            continue
        if pre_pred.get(q):
            gg["5a_sister_not_orphan"] += 1
            continue
        tr_ = orph_tree.get(t_ + 1)
        if tr_ is None:
            gg["5b_no_orphans_in_frame"] += 1
            continue
        d_pq = _dist_um(nodes_f[ga_]["xyz"], nodes_f[q]["xyz"])
        if d_pq > DIV_PARENT_UM:
            gg["5c_sister_over_12um_of_mother"] += 1
            continue
        d_s = _dist_um(nodes_f[c1]["xyz"], nodes_f[q]["xyz"])
        if d_s > DIV_SISTER_UM:
            gg["6_sister_gap_over_15um"] += 1
            continue
        d_q, q_idx = tr_.query(np.asarray(nodes_f[c1]["xyz"], float) * SCALE)
        if orph_ids[t_ + 1][q_idx] != q or d_q > DIV_SISTER_UM:
            gg["7_not_the_nearest_orphan"] += 1
            continue
        c1n, qn = pre_succ.get(c1, []), pre_succ.get(q, [])
        if len(c1n) != 1 or len(qn) != 1:
            gg["8a_daughter_not_one_child"] += 1
            if not c1n or not qn:
                gg["8a_of_which_track_end"] += 1
            continue
        dv = _dist_um(nodes_f[c1n[0]]["xyz"], nodes_f[qn[0]]["xyz"]) - d_s
        GROWS.setdefault(domain, []).append(float(dv))
        if dv < DIV_DIVERGE_UM:
            gg["8b_diverge_under_2.25um"] += 1
            continue
        gg["9_passed_all_gates_then_capped_or_taken"] += 1

    # ---- EVERY FORK THE PIPELINE EMITS, true or false -----------------''')

# ---- 3. stores ----------------------------------------------------------
rep('''KAGG = {}          # domain -> emitted-fork counters''',
    '''KAGG = {}          # domain -> emitted-fork counters
GAGG = {}          # domain -> which patch condition rejected the division
GROWS = {}         # domain -> divergence values that reached condition 8''')

# ---- 4. the report ------------------------------------------------------
rep('''print(f"\\nDIVWHY DONE in {(time.time() - t_all) / 60:.1f} min",
      flush=True)''',
    r'''
print("\n\n=== WHY THE ORPHAN-FORK PATCH DID NOT RESCUE THEM ===",
      flush=True)
print("  Every GT division the ILP half-linked or did not link, replayed "
      "through _add_safe_divisions' conditions IN ORDER on the graph the "
      "patch actually saw. The first condition that rejects it is the one "
      "counted, so the table reads top to bottom as the patch does.",
      flush=True)
print(f"  Constants: DIV_CHILD_UM {DIV_CHILD_UM}, DIV_PARENT_UM "
      f"{DIV_PARENT_UM}, DIV_SISTER_UM {DIV_SISTER_UM}, DIV_DIVERGE_UM "
      f"{DIV_DIVERGE_UM}.", flush=True)
GORDER = ["mother_deleted_post", "0_sister_deleted_post",
          "1_mother_not_one_child", "2_mother_no_single_pred",
          "3_child_wrong_frame", "4_mother_child_over_10um",
          "5a_sister_not_orphan", "5b_no_orphans_in_frame",
          "5c_sister_over_12um_of_mother", "6_sister_gap_over_15um",
          "7_not_the_nearest_orphan", "8a_daughter_not_one_child",
          "8a_of_which_track_end", "8b_diverge_under_2.25um",
          "9_passed_all_gates_then_capped_or_taken"]
for d in DOMAINS:
    gg = GAGG.get(d)
    if not gg:
        continue
    tot = max(1, gg.get("pending", 0))
    print(f"\n  -- {d}: {gg.get('pending', 0):,} unrescued GT divisions --",
          flush=True)
    for c in GORDER:
        if not gg.get(c):
            continue
        mark = "     (of the row above)" if c.endswith("track_end") else ""
        print(f"  {c:<34} {gg[c]:>7,}  {100.0 * gg[c] / tot:>6.2f}%{mark}",
              flush=True)

print("\n=== THE DIVERGENCE TEST, measured ===", flush=True)
print("  For the divisions that survived every earlier condition: how "
      "much FURTHER apart the two daughters' own children are than the "
      f"daughters themselves. The patch demands >= {DIV_DIVERGE_UM} um in "
      "one frame. div_kinematics measured real daughters separating at "
      "1.155 um/frame (median) against 0.082 for random pairs, so a "
      "threshold at 2.25 is about twice the rate a real division "
      "reaches.", flush=True)
for d in DOMAINS:
    v = GROWS.get(d)
    if not v:
        continue
    a = np.asarray(v, float)
    a = a[np.isfinite(a)]
    if not len(a):
        continue
    qs = np.percentile(a, [10, 25, 50, 75, 90])
    print(f"  {d:<10} n={len(a):>6,}  p10 {qs[0]:>6.2f}  p25 {qs[1]:>6.2f}"
          f"  MEDIAN {qs[2]:>6.2f}  p75 {qs[3]:>6.2f}  p90 {qs[4]:>6.2f}"
          f"   share >= {DIV_DIVERGE_UM}: "
          f"{100.0 * float((a >= DIV_DIVERGE_UM).sum()) / len(a):.1f}%",
          flush=True)

print("\n=== reading it ===", flush=True)
print("  A row that dominates is the condition to change, and each one "
      "implies a different repair. 2_mother_no_single_pred and "
      "8a_daughter_not_one_child are STRUCTURAL -- the division sits at "
      "the start or the end of a track fragment and no constant fixes "
      "it. 5a_sister_not_orphan means the ILP gave the second daughter "
      "some other parent, so the fork was not refused, it was spent "
      "elsewhere. 8b is a tuned constant and the only row where changing "
      "a number is the whole fix.", flush=True)
print(f"\nDIVGATE2 DONE in {(time.time() - t_all) / 60:.1f} min",
      flush=True)''')

DST.write_text(text, encoding="utf-8")
ast.parse(text)
print(f"wrote {DST} ({len(text.splitlines())} lines), ast OK")
