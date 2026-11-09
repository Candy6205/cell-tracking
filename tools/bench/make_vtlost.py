"""vtattach.py -> vtlost.py: WHERE DO THE TRUE SHIPPED EDGES GO when the vote candidates attach?

WHY (2026-09-20). vtattach's ledger: VT17XB's funnel counts 182 true edges taken by
annotated candidates while the per-movie CSVs count a net edge_tp of +70; VT17X 387
against +286; VT18XB 64 against -49. Two things can hide in the gap. A candidate can
TAKE OVER a GT match that a shipped node already carried (the funnel then counts a true
edge REF already had), or a candidate can STEAL a link (the ILP picks shipped -> candidate
over shipped -> shipped) so the true edge is gone, with the thief kept (X) or dropped as
dangling after the theft (XB).

MEASUREMENT (per movie, REF's final graph against each arm's final graph, both in hand):
  a GT edge (t -> t+1) is REALIZED in a graph when both GT nodes have a final node within
  7 um and that pair is an edge. lost = realized in REF and not in the arm; gained = the
  converse; kept_via_candidate_takeover = realized in both with a candidate endpoint in
  the arm. Every lost edge is classified on the arm's graphs (solved and final):
    shipped_node_gone            a shipped endpoint of REF's edge is not in the arm's final graph
    endpoint_unmatched           a GT endpoint has no final node within 7 um
    taken_by_candidate_no_edge   a candidate is the nearer node to a GT endpoint and has no edge
    stolen_by_kept_candidate     the source links to a candidate instead (final graph)
    stolen_then_dropped          the source linked to a candidate in the SOLVED graph, the
                                 candidate was dropped as dangling, the source is now unlinked
    stolen_pred_by_candidate /   the same on the target's side
    stolen_pred_then_dropped
    relinked_shipped             both endpoints present, the edge replaced by another shipped edge
    relinked_swap                the same, and the new partner's own true edge is lost too (a swap chain)
    both_unlinked                both endpoints present and unlinked, no candidate in sight
  For the stolen rows: p_true (the shipped pair's ranker p), p_steal (the thief's), d_true,
  d_steal, whether the thief is annotated (GT within 7 um), kept, and its vote score.
  Gained edges: via_candidate_ref_unmatched (a rescued missed cell), via_candidate_ref_matched
  (REF had both nodes and no edge), shipped_relink, shipped_ref_unmatched.

ARMS: VT17X (attach only) and VT17XB (attach only, dangling dropped). CPU ~4.5 h.
"""
import os

src = open("vtattach.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-19c-vtattach"', 'SCRIPT_VERSION = "2026-09-20c-vtlost"')
rep('DUP_ARMS = [("VT17XB", dict(tau=0.17, persist=False, nonew=True, dangle=True)), ("VT18XB", dict(tau=0.18, persist=False, nonew=True, dangle=True))]',
    'DUP_ARMS = [("VT17X", dict(tau=0.17, persist=False, nonew=True, dangle=False)), ("VT17XB", dict(tau=0.17, persist=False, nonew=True, dangle=True))]')

# 1. the ledger: ln_lost + lost_summary, inserted before ln_trace
rep('def ln_trace(name, T, seeds, nnew, keep, pmap, son):\n', r'''import pandas as pd   # the ledger's summary; the harness imports it inside its own functions

LOST_ROWS = []
LOST_COLS = ["arm", "dataset", "kind", "cat", "t", "g1", "g2", "p_true", "p_steal", "d_true", "d_steal", "thief_ann", "thief_kept", "thief_votes", "d_thief_true", "d_thief_gt", "p_new"]
LOST_CNT = defaultdict(lambda: defaultdict(int))


def _final_index(nodes, edges):
    by_t = defaultdict(list)
    for nid, v in nodes.items():
        by_t[int(v["t"])].append(nid)
    tree = {t_: (ids_, cKDTree(np.asarray([nodes[g]["xyz"] for g in ids_], float) * SCALE)) for t_, ids_ in by_t.items()}
    succ, pred = defaultdict(list), defaultdict(list)
    for a_, b_ in edges:
        succ[a_].append(b_)
        pred[b_].append(a_)
    return tree, succ, pred, set(edges)


def _match(tree, t_, pos):
    if t_ not in tree:
        return None
    ids_, tr_ = tree[t_]
    d_, j_ = tr_.query(pos)
    return ids_[int(j_)] if d_ <= LT_MATCH_UM else None


def ln_lost(arm, name, T, seeds_ln, nnew, ref_final, son0, pmap0, pm_a, so_a, e_solved, n_solved):
    """every GT edge (t -> t+1) realized in REF's final graph or the arm's (PRE_STATE): kept / lost / gained, classified."""
    gt = read_geff(geff_path(root, "train", name))
    g_t = np.asarray(gt["t"], int)
    g_pos = np.stack([np.asarray(gt[c_], float) for c_ in ("z", "y", "x")], axis=1) * SCALE
    row_of = {int(v): i for i, v in enumerate(gt["node_ids"])}
    gt_rows = {int(tt): np.where(g_t == tt)[0] for tt in np.unique(g_t)}
    gt_tree = {tt: cKDTree(g_pos[ix]) for tt, ix in gt_rows.items()}
    nodes_r, edges_r = ref_final
    tr_r, succ_r, pred_r, es_r = _final_index(nodes_r, edges_r)
    nodes_x, edges_x = PRE_STATE["nodes"], PRE_STATE["final_edges"]
    tr_x, succ_x, pred_x, es_x = _final_index(nodes_x, edges_x)
    succ_s, pred_s = defaultdict(list), defaultdict(list)
    for a_, b_ in e_solved:
        succ_s[a_].append(b_)
        pred_s[b_].append(a_)
    inv_x = {}
    for nid in nodes_x:
        sd = so_a.get(nid)
        if sd is not None:
            inv_x[(int(sd[0]), int(sd[1]))] = nid
    inv_r = {}
    for nid in nodes_r:
        sd = son0.get(nid)
        if sd is not None:
            inv_r[(int(sd[0]), int(sd[1]))] = nid

    def is_new(nid):
        return ln_is_new(so_a.get(nid), seeds_ln, nnew)

    def seed_x(nid):
        sd = so_a.get(nid)
        return None if sd is None else (int(sd[0]), int(sd[1]))

    def xyz(nid):
        v = nodes_x.get(nid)
        if v is None:
            v = n_solved.get(nid)
        return np.asarray(v["xyz"], float) * SCALE

    def pget(t_, i_, j_):
        v = pm_a.get((t_, i_, j_))
        if v is None:
            v = pmap0.get((t_, i_, j_))
        return float(v[0]) if v is not None else float("nan")

    cnt = LOST_CNT[arm]
    pend = []
    lost_r = set()
    for a_, b_ in gt["edges"]:
        ra, rb = row_of.get(int(a_)), row_of.get(int(b_))
        if ra is None or rb is None:
            continue
        t1, t2 = int(g_t[ra]), int(g_t[rb])
        if t2 != t1 + 1:
            continue
        m1r, m2r = _match(tr_r, t1, g_pos[ra]), _match(tr_r, t2, g_pos[rb])
        m1x, m2x = _match(tr_x, t1, g_pos[ra]), _match(tr_x, t2, g_pos[rb])
        tp_r = m1r is not None and m2r is not None and (m1r, m2r) in es_r
        tp_x = m1x is not None and m2x is not None and (m1x, m2x) in es_x
        if tp_r and tp_x:
            cnt["kept"] += 1
            if is_new(m1x) or is_new(m2x):
                cnt["kept_via_candidate_takeover"] += 1
            continue
        if not tp_r and not tp_x:
            continue
        p_true = p_steal = d_true = d_steal = float("nan")
        thief = None
        side = ""
        s1 = s2 = n1x = n2x = None
        if tp_x:
            kind = "gained"
            if is_new(m1x) or is_new(m2x):
                cat = "via_candidate_ref_unmatched" if (m1r is None or m2r is None) else "via_candidate_ref_matched"
                thief = m1x if is_new(m1x) else m2x
            else:
                cat = "shipped_relink" if (m1r is not None and m2r is not None) else "shipped_ref_unmatched"
        else:
            kind = "lost"
            s1, s2 = son0.get(m1r), son0.get(m2r)
            s1 = None if s1 is None else (int(s1[0]), int(s1[1]))
            s2 = None if s2 is None else (int(s2[0]), int(s2[1]))
            n1x = inv_x.get(s1) if s1 is not None else None
            n2x = inv_x.get(s2) if s2 is not None else None
            if s1 is not None and s2 is not None:
                p_true = pget(s1[0], s1[1], s2[1])
                if n1x is not None and n2x is not None:
                    d_true = float(np.linalg.norm(xyz(n1x) - xyz(n2x)))
            if (s1 is not None and n1x is None) or (s2 is not None and n2x is None):
                cat = "shipped_node_gone"
            elif m1x is None or m2x is None:
                cat = "endpoint_unmatched"
            elif is_new(m1x) or is_new(m2x):
                cat = "taken_by_candidate_no_edge"
                thief = m1x if is_new(m1x) else m2x
            else:
                sx = [c_ for c_ in succ_x.get(m1x, []) if is_new(c_)]
                px = [c_ for c_ in pred_x.get(m2x, []) if is_new(c_)]
                ss = [c_ for c_ in succ_s.get(m1x, []) if is_new(c_)]
                ps = [c_ for c_ in pred_s.get(m2x, []) if is_new(c_)]
                if sx:
                    cat, thief, side = "stolen_by_kept_candidate", sx[0], "succ"
                elif px:
                    cat, thief, side = "stolen_pred_by_candidate", px[0], "pred"
                elif ss and not succ_x.get(m1x):
                    cat, thief, side = "stolen_then_dropped", ss[0], "succ"
                elif ps and not pred_x.get(m2x):
                    cat, thief, side = "stolen_pred_then_dropped", ps[0], "pred"
                elif succ_x.get(m1x) or pred_x.get(m2x):
                    cat = "relinked_shipped"
                else:
                    cat = "both_unlinked"
        thief_ann = thief_kept = -1
        thief_v = d_thief_true = d_thief_gt = float("nan")
        if thief is not None:
            tsd = seed_x(thief)
            tpos = xyz(thief)
            thief_kept = int(thief in nodes_x)
            if tsd is not None:
                if tsd[0] in gt_tree:
                    d_thief_gt = float(gt_tree[tsd[0]].query(tpos)[0])          # the thief to the nearest GT node of its frame
                thief_ann = int(d_thief_gt <= LT_MATCH_UM)
                thief_v = float(np.asarray(seeds_ln[tsd[0]][1], float).reshape(-1)[tsd[1]])
                if side == "succ" and s1 is not None and n1x is not None:
                    p_steal = pget(s1[0], s1[1], tsd[1])
                    d_steal = float(np.linalg.norm(xyz(n1x) - tpos))
                    if n2x is not None:
                        d_thief_true = float(np.linalg.norm(tpos - xyz(n2x)))   # the thief to the true target it displaced (same frame)
                elif side == "pred" and s2 is not None and n2x is not None:
                    p_steal = pget(tsd[0], tsd[1], s2[1])
                    d_steal = float(np.linalg.norm(tpos - xyz(n2x)))
                    if n1x is not None:
                        d_thief_true = float(np.linalg.norm(tpos - xyz(n1x)))   # the thief to the true source it displaced
        if kind == "lost":
            lost_r.add((m1r, m2r))
        pend.append(([arm, name, kind, cat, t1, int(a_), int(b_), p_true, p_steal, d_true, d_steal, thief_ann, thief_kept, thief_v, d_thief_true, d_thief_gt], m1x, m2x, s1, s2))
    # second pass: a relinked edge whose new shipped partner's OWN true edge is also lost is a SWAP, the cascade of one assignment change
    for row, m1x, m2x, s1, s2 in pend:
        p_new = float("nan")
        if row[3] == "relinked_shipped":
            swap = False
            for w_x in succ_x.get(m1x, []):
                if w_x == m2x or is_new(w_x):
                    continue
                sw = seed_x(w_x)
                if sw is not None and s1 is not None and np.isnan(p_new):
                    p_new = pget(s1[0], s1[1], sw[1])
                w_r = inv_r.get(sw) if sw is not None else None
                if w_r is not None and any((u_r, w_r) in lost_r for u_r in pred_r.get(w_r, [])):
                    swap = True
            for v_x in pred_x.get(m2x, []):
                if v_x == m1x or is_new(v_x):
                    continue
                sv = seed_x(v_x)
                if sv is not None and s2 is not None and np.isnan(p_new):
                    p_new = pget(sv[0], sv[1], s2[1])
                v_r = inv_r.get(sv) if sv is not None else None
                if v_r is not None and any((v_r, z_r) in lost_r for z_r in succ_r.get(v_r, [])):
                    swap = True
            if swap:
                row[3] = "relinked_swap"
        row.append(p_new)
        LOST_ROWS.append(row)
        cnt[row[2] + ":" + row[3]] += 1


def lost_summary():
    df = pd.DataFrame(LOST_ROWS, columns=LOST_COLS)
    df.to_csv("/kaggle/working/vtlost_edges.csv", index=False)
    print("", flush=True)
    print(f"=== VTLOST: every GT edge (t -> t+1) realized in REF's final graph or an arm's; {len(df):,} lost / gained rows over the scored movies ===", flush=True)
    for arm in sorted(LOST_CNT):
        cnt = LOST_CNT[arm]
        d_ = df[df.arm == arm]
        n_lost, n_gain = int((d_.kind == "lost").sum()), int((d_.kind == "gained").sum())
        print("", flush=True)
        print(f"  ---- arm {arm}: kept {cnt['kept']:,} (realized through a candidate that took over a GT match REF already carried: {cnt['kept_via_candidate_takeover']:,}); "
              f"lost {n_lost:,}; gained {n_gain:,}; net {n_gain - n_lost:+,} ----", flush=True)
        for kind in ("lost", "gained"):
            sub = d_[d_.kind == kind]
            print(f"  {kind}:", flush=True)
            for cat, g in sub.groupby("cat"):
                extra = ""
                if g.p_steal.notna().any():
                    extra = (f"  p_true med {g.p_true.median():.3f}  p_steal med {g.p_steal.median():.3f}  d_true med {g.d_true.median():.2f}  d_steal med {g.d_steal.median():.2f}"
                             f"  thief annotated {int((g.thief_ann == 1).sum())}/{int((g.thief_ann >= 0).sum())}  thief kept {int((g.thief_kept == 1).sum())}  votes med {g.thief_votes.median():.3f}"
                             f"  thief->true node med {g.d_thief_true.median():.2f}  thief->GT med {g.d_thief_gt.median():.2f}")
                elif (g.thief_ann >= 0).any():
                    extra = (f"  thief annotated {int((g.thief_ann == 1).sum())}/{int((g.thief_ann >= 0).sum())}  thief kept {int((g.thief_kept == 1).sum())}  votes med {g.thief_votes.median():.3f}"
                             f"  thief->GT med {g.d_thief_gt.median():.2f}")
                if g.p_new.notna().any():
                    extra += (f"  p_true med {g.p_true.median():.3f}  p_new (the replacing shipped edge) med {g.p_new.median():.3f}"
                              f"  new edge beats the true one {int((g.p_new > g.p_true).sum())}/{int((g.p_new.notna() & g.p_true.notna()).sum())}")
                print(f"    {cat:32s} {len(g):6,}{extra}", flush=True)
        st = d_[(d_.kind == "lost") & d_.cat.str.startswith("stolen")]
        if len(st):
            both = st[st.p_true.notna() & st.p_steal.notna()]
            if len(both):
                print(f"  stolen edges with both probabilities: {len(both):,}; the thief's p beats the true edge's in {int((both.p_steal > both.p_true).sum()):,}; "
                      f"p_true 10/50/90% = {both.p_true.quantile(0.1):.3f} / {both.p_true.quantile(0.5):.3f} / {both.p_true.quantile(0.9):.3f}; "
                      f"p_steal = {both.p_steal.quantile(0.1):.3f} / {both.p_steal.quantile(0.5):.3f} / {both.p_steal.quantile(0.9):.3f}; "
                      f"true edge below the floor {PROD_FLOOR}: {int((both.p_true < PROD_FLOOR).sum()):,}", flush=True)
            dt = st.d_thief_true.dropna()
            dgt = st.d_thief_gt.dropna()
            if len(dt):
                print(f"  thief to the true node it displaced, 10/25/50/75/90% = {dt.quantile(0.1):.2f} / {dt.quantile(0.25):.2f} / {dt.quantile(0.5):.2f} / {dt.quantile(0.75):.2f} / {dt.quantile(0.9):.2f} um"
                      f" (candidates were admitted only >= 7 um from every shipped primary); within 10 um {int((dt <= 10).sum()):,} of {len(dt):,}; within 12 um {int((dt <= 12).sum()):,};"
                      f" thief to the nearest GT node 10/50/90% = {dgt.quantile(0.1):.2f} / {dgt.quantile(0.5):.2f} / {dgt.quantile(0.9):.2f} um", flush=True)
            pm = st.groupby("dataset").size().sort_values(ascending=False)
            print(f"  stolen per movie: top {', '.join(f'{k} {v}' for k, v in pm.head(5).items())}; movies with any {len(pm)} of {d_.dataset.nunique()}", flush=True)
    print("  reading: kept_via_candidate_takeover is the part of the funnel's G_tp that REF already had; the stolen_* rows are the true edges the attachments displace, "
          "with the ranker's p on both sides; shipped_node_gone / relinked_shipped are the cascade through the short filter and the assignment.", flush=True)


def ln_trace(name, T, seeds, nnew, keep, pmap, son):
''')

# 2. REF's final graph is kept for the ledger
rep('''    patch_final(nodes0, edges0, pool, outp)
    BR_CTX["gt"] = None
''', '''    patch_final(nodes0, edges0, pool, outp)
    REF_FINAL = (PRE_STATE["nodes"], list(PRE_STATE["final_edges"]))
    BR_CTX["gt"] = None
''')

# 3. the arm loop: the solved graph before the drop, the ledger after the funnel
rep('''        pm_a, so_a = aux_a
        LN_STAT[arm]["newnew_withheld"]''', '''        pm_a, so_a = aux_a
        n_solved, e_solved = n_a, list(e_a)
        LN_STAT[arm]["newnew_withheld"]''')
rep('''        LN_STAT[arm]["trace_rows"] += ln_trace(name, T, seeds_ln, nnew_ln, keep, pm_a, so_a)
''', '''        LN_STAT[arm]["trace_rows"] += ln_trace(name, T, seeds_ln, nnew_ln, keep, pm_a, so_a)
        ln_lost(arm, name, T, seeds_ln, nnew_ln, REF_FINAL, seed_of_node, pmap, pm_a, so_a, e_solved, n_solved)
''')
rep('''        del ds_a, n_a, e_a, p_a, aux_a, pm_a, so_a, sc_a, b2_a, keep
''', '''        del ds_a, n_a, e_a, p_a, aux_a, pm_a, so_a, sc_a, b2_a, keep, n_solved, e_solved
''')

# 4. the summary
rep('lt_summary()\n', 'lt_summary()\nlost_summary()\n')

# 5. names
for old, new in (('f"vtattach_REF_{DET_SRC}.csv"', 'f"vtlost_REF_{DET_SRC}.csv"'), ('f"vtattach_arm_{a_}.csv"', 'f"vtlost_arm_{a_}.csv"'),
                 ('"/kaggle/working/vtattach_edges.csv"', '"/kaggle/working/vtlost_trace.csv"'), ('print("VTATTACH OK", flush=True)', 'print("VTLOST OK", flush=True)'),
                 ('print(f"[degraded] VTATTACH INCOMPLETE or anchor miss:', 'print(f"[degraded] VTLOST INCOMPLETE or anchor miss:'),
                 ('=== VTATTACH: every linkable GT edge', '=== VTLOST (funnel): every linkable GT edge')):
    rep(old, new)

assert src.count("ln_lost(") == 2 and src.count("lost_summary()") == 2 and src.count("REF_FINAL") == 2
assert src.index("def ln_lost(") < src.index("def ln_trace(") < src.index("        ln_lost(arm, name")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vtlost.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written vtlost.py", len(src.splitlines()), "lines")
