"""lnbench.py -> lntrace.py: WHERE do the rescue peaks' edges die?

WHY (2026-09-17). lnpersist counted 256 linkable GT edges among the rescue peaks admitted at
score >= 0.20 (an annotated rescue peak whose GT neighbour a shipped primary detects), and
lnbench's LN20 arm turned them into about thirty net true edges (edge_J +0.0003). The user
asked which gate they die at. This run is lnbench with the single arm LN20 and, per movie
after the arm's solve and patches, one row per linkable edge:
  A_below_tau      the peak was not admitted (score < 0.20; kept for the count)
  B_not_candidate  the pair (shipped neighbour, peak) is not in the ranker's pair map
  C_below_floor    the ranker's p is below PROD_FLOOR 0.48, so the edge never entered the ILP
  D_q_dropped      p >= floor but the peak's node is not in the final graph (unlinked, filtered)
  D2_s_dropped     ... the shipped neighbour's node is not in the final graph
  E_competitor     both nodes present, edge not taken, the neighbour is linked to another node
                   (its p, whether that node is itself a rescue peak, its distance to the missed
                   cell, whether it sits on another annotated cell)
  E2_no_link       both present, edge not taken, the neighbour has no link on that side
  F_q_unmatched    edge taken, but another final node is nearer to the annotated cell than the
                   peak (the scorer gives the cell to that node)
  G_tp             edge taken and the peak is the cell's nearest node: a true edge
Funnel per direction (edge from the previous frame / to the next frame), p medians, the
competitor's p and nature. CSV lntrace_edges.csv, one row per linkable edge.

Attach: lnbench's sources. CPU ~3 h (one arm).
"""
import os

src = open("lnbench.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-17a-lnbench"', 'SCRIPT_VERSION = "2026-09-17a-lntrace"')
rep('DUP_ARMS = [("LN20", dict(tau=0.2, persist=False)), ("LN25", dict(tau=0.25, persist=False))]',
    'DUP_ARMS = [("LN20", dict(tau=0.2, persist=False))]')

BLOCK = r'''
# ---- lntrace block ----
LT_MATCH_UM = 7.0
LT_ROWS = []
LT_COLS = ["dataset", "t", "q_i", "q_score", "d_gt", "dir", "nb_t", "s_i", "s_score", "cand", "p", "admitted",
           "q_node", "s_node", "edge_taken", "comp_p", "comp_new", "comp_d_gt", "comp_other", "q_nearest", "cat"]


def ln_trace(name, T, seeds, nnew, keep, pmap, son):
    """one row per linkable GT edge of every annotated rescue peak, on the arm's FINAL graph (PRE_STATE)."""
    gt = read_geff(geff_path(root, "train", name))
    g_t = np.asarray(gt["t"], int)
    g_pos = np.stack([np.asarray(gt[c_], float) for c_ in ("z", "y", "x")], axis=1) * SCALE
    ids = [int(v) for v in gt["node_ids"]]
    row_of = {v: i for i, v in enumerate(ids)}
    nbr = defaultdict(list)
    for a_, b_ in gt["edges"]:
        ra, rb = row_of.get(int(a_)), row_of.get(int(b_))
        if ra is not None and rb is not None:
            nbr[ra].append(rb)
            nbr[rb].append(ra)
    gt_rows = {int(tt): np.where(g_t == tt)[0] for tt in np.unique(g_t)}
    gt_tree = {tt: cKDTree(g_pos[ix]) for tt, ix in gt_rows.items()}
    nodes, edges = PRE_STATE["nodes"], PRE_STATE["final_edges"]
    eset = set(edges)
    inv = {}
    for nid in nodes:
        sd = son.get(nid)
        if sd is not None:
            inv[(int(sd[0]), int(sd[1]))] = nid
    succ, pred = defaultdict(list), defaultdict(list)
    for a_, b_ in edges:
        succ[a_].append(b_)
        pred[b_].append(a_)
    by_t = defaultdict(list)
    for nid, v in nodes.items():
        by_t[int(v["t"])].append(nid)
    ntree = {t_: (ids_, cKDTree(np.asarray([nodes[g]["xyz"] for g in ids_], float) * SCALE)) for t_, ids_ in by_t.items()}
    ship = {}

    def ship_tree(t_):
        if t_ not in ship:
            c_ = np.asarray(seeds[t_][0], float).reshape(-1, 3)
            n_ = len(c_) - nnew[t_]
            ship[t_] = cKDTree(c_[:n_] * SCALE) if n_ else None
        return ship[t_]

    def is_new(sd):
        return int(sd[1] >= len(seeds[sd[0]][0]) - nnew[sd[0]])

    n_rows = 0
    for t in range(T):
        if nnew[t] == 0 or t not in gt_tree:
            continue
        c_ = np.asarray(seeds[t][0], float).reshape(-1, 3)
        s_ = np.asarray(seeds[t][1], float).reshape(-1)
        n0 = len(c_) - nnew[t]
        for k_ in range(nnew[t]):
            qi = n0 + k_
            d_gt, j_ = gt_tree[t].query(c_[qi] * SCALE)
            if d_gt > LT_MATCH_UM:
                continue
            g = int(gt_rows[t][j_])
            q_node = inv.get((t, qi))
            q_near = 0
            if q_node is not None and t in ntree:
                ids_, tr_ = ntree[t]
                dn, jn = tr_.query(g_pos[g])
                q_near = int(ids_[int(jn)] == q_node and dn <= LT_MATCH_UM)
            for gn in nbr[g]:
                tn = int(g_t[gn])
                if tn not in (t - 1, t + 1) or tn < 0 or tn >= T:
                    continue
                st = ship_tree(tn)
                if st is None:
                    continue
                ds, si = st.query(g_pos[gn])
                if ds > LT_MATCH_UM:
                    continue                      # the neighbour is not detected by a shipped primary: not linkable
                si = int(si)
                key = (tn, si, qi) if tn == t - 1 else (t, qi, si)
                pv = pmap.get(key)
                p = float(pv[0]) if pv is not None else float("nan")
                admitted = bool(keep[t][qi])
                s_node = inv.get((tn, si))
                edge_taken = 0
                comp_p, comp_new, comp_d, comp_other = float("nan"), -1, float("nan"), -1
                if q_node is not None and s_node is not None:
                    edge_taken = int(((s_node, q_node) in eset) if tn == t - 1 else ((q_node, s_node) in eset))
                comps = []
                if not edge_taken and s_node is not None:
                    comps = succ.get(s_node, []) if tn == t - 1 else pred.get(s_node, [])
                    if comps:
                        cn = comps[0]
                        csd = son.get(cn)
                        if csd is not None:
                            csd = (int(csd[0]), int(csd[1]))
                            ckey = (tn, si, csd[1]) if tn == t - 1 else (t, csd[1], si)
                            cv = pmap.get(ckey)
                            comp_p = float(cv[0]) if cv is not None else float("nan")
                            comp_new = is_new(csd)
                        cu = np.asarray(nodes[cn]["xyz"], float) * SCALE
                        comp_d = float(np.linalg.norm(cu - g_pos[g]))
                        dd, jj = gt_tree[t].query(cu)
                        comp_other = int(dd <= LT_MATCH_UM and int(gt_rows[t][int(jj)]) != g)
                if not admitted:
                    cat = "A_below_tau"
                elif pv is None:
                    cat = "B_not_candidate"
                elif p < PROD_FLOOR:
                    cat = "C_below_floor"
                elif q_node is None:
                    cat = "D_q_dropped"
                elif s_node is None:
                    cat = "D2_s_dropped"
                elif not edge_taken:
                    cat = "E_competitor" if comps else "E2_no_link"
                elif not q_near:
                    cat = "F_q_unmatched"
                else:
                    cat = "G_tp"
                s_sc = float(np.asarray(seeds[tn][1], float).reshape(-1)[si])
                LT_ROWS.append([name, t, qi, float(s_[qi]), float(d_gt), "prev" if tn == t - 1 else "next", tn, si, s_sc,
                                int(pv is not None), p, int(admitted), int(q_node is not None), int(s_node is not None),
                                edge_taken, comp_p, comp_new, comp_d, comp_other, q_near, cat])
                n_rows += 1
    return n_rows


def lt_summary():
    print(f"\n=== LNTRACE: every linkable GT edge of an annotated rescue peak, {len(LT_ROWS):,} rows over the scored movies "
          f"(one row per GT edge; 'prev' = the edge from the frame before, 'next' = to the frame after) ===", flush=True)
    import pandas as _pd
    df = _pd.DataFrame(LT_ROWS, columns=LT_COLS)
    df.to_csv("/kaggle/working/lntrace_edges.csv", index=False)
    cats = ["A_below_tau", "B_not_candidate", "C_below_floor", "D_q_dropped", "D2_s_dropped", "E_competitor", "E2_no_link",
            "F_q_unmatched", "G_tp"]
    print(f"  {'category':<18} {'all':>6} {'prev':>6} {'next':>6} {'peak score med':>15} {'d_gt med':>9} {'p med':>7}", flush=True)
    for c_ in cats:
        m = df.cat == c_
        if not m.any():
            print(f"  {c_:<18} {0:>6}", flush=True)
            continue
        print(f"  {c_:<18} {int(m.sum()):>6} {int((m & (df.dir == 'prev')).sum()):>6} {int((m & (df.dir == 'next')).sum()):>6} "
              f"{df.q_score[m].median():>15.3f} {df.d_gt[m].median():>9.2f} {df.p[m].median():>7.3f}", flush=True)
    adm = df[df.admitted == 1]
    print(f"\n  admitted peaks' edges {len(adm):,} (lnpersist counted 256 linkable at >= 0.20); peaks with at least one edge taken: "
          f"{adm[adm.edge_taken == 1].groupby(['dataset', 't', 'q_i']).ngroups:,} of {adm.groupby(['dataset', 't', 'q_i']).ngroups:,}", flush=True)
    c = adm[adm.cat == "C_below_floor"]
    if len(c):
        print(f"  C_below_floor: p quantiles 10/25/50/75/90% = " + " / ".join(f"{v:.3f}" for v in c.p.quantile([.1, .25, .5, .75, .9]))
              + f"; shipped neighbour's score med {c.s_score.median():.3f}; peak score med {c.q_score.median():.3f}", flush=True)
    e = adm[adm.cat == "E_competitor"]
    if len(e):
        print(f"  E_competitor: the edge's p med {e.p.median():.3f} vs the competitor's {e.comp_p.median():.3f}; competitor is a rescue peak "
              f"{int((e.comp_new == 1).sum())}, within 7 um of the missed cell (a second detection of it) {int((e.comp_d_gt <= 7).sum())}, "
              f"on another annotated cell {int((e.comp_other == 1).sum())}, on no annotated cell {int(((e.comp_other == 0) & (e.comp_d_gt > 7)).sum())}",
              flush=True)
    f_ = adm[adm.cat == "F_q_unmatched"]
    if len(f_):
        print(f"  F_q_unmatched: {len(f_)} edges taken whose peak is not the cell's nearest final node (d_gt med {f_.d_gt.median():.2f})", flush=True)
    print("  reading: the first non-empty bucket from the top is the gate that kills most of the line; C means the ranker, "
          "E means the ILP's competition, F means a duplicate detection already holds the cell.", flush=True)
# ---- end lntrace block ----
'''
rep('\ndef _judge(name, k, outp, seeds, aux):\n', BLOCK + '\n\ndef _judge(name, k, outp, seeds, aux):\n')
rep('''        patch_final(n_a, e_a, p_a, outa)
        for k_, v_ in PATCH_CAPS.items():
            CAPS_BY_ARM.setdefault(arm, defaultdict(int))[k_] += v_
''', '''        patch_final(n_a, e_a, p_a, outa)
        LN_STAT[arm]["trace_rows"] += ln_trace(name, T, seeds_ln, nnew_ln, keep, pm_a, so_a)
        for k_, v_ in PATCH_CAPS.items():
            CAPS_BY_ARM.setdefault(arm, defaultdict(int))[k_] += v_
''')
rep('\nBR_ARMS_RUN = list(BR_ARMS)\n', '\nlt_summary()\n\nBR_ARMS_RUN = list(BR_ARMS)\n')
rep('print("LNBENCH OK", flush=True)', 'print("LNTRACE OK", flush=True)')
rep('print(f"[degraded] LNBENCH INCOMPLETE or anchor miss:', 'print(f"[degraded] LNTRACE INCOMPLETE or anchor miss:')
rep('f"lnbench_REF_{DET_SRC}.csv"', 'f"lntrace_REF_{DET_SRC}.csv"')
rep('f"lnbench_arm_{a_}.csv"', 'f"lntrace_arm_{a_}.csv"')

assert src.count("ln_trace(") == 2 and src.count("lt_summary()") == 2
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lntrace.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written lntrace.py", len(src))
