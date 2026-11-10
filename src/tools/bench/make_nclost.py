"""nc44.py -> nclost.py: which TRUE edges does the node cost remove on 44b6, and what do they look like?

WHY (2026-09-22). nc44: on the 71 44b6 movies NC20XT removes 2.1% of the nodes and edge_J falls 0.0017 (6bba:
flat). This run keeps two arms (the shipped solve, NC20XT), matches both final graphs to the GT per frame
(Hungarian within 7 um), and for every GT edge the control has and the costed graph lacks records why and
what it looked like in the control:
  why      node_unselected (an endpoint has no ILP edge in the costed solve) / rerouted (both endpoints keep
           ILP edges, this one is not among them) / post_filter (in the costed ILP, gone after the patches) /
           non_ilp (a gap-close or patch edge of the control) / rematch (the costed graph still has the seed
           edge; the GT matching moved)
  features ranker p, the two detection scores, length in um, frame, the control ILP component's node count and
           mean p, frames from the edge to the nearer end of its ILP track, whether an endpoint is exempt
Kept true edges get the same columns (lost = 0). Output nclost_edges.csv; summary tables by reason and
lost-vs-kept quantiles. The report / gate cross of the harness is not run (exit after stage B).
"""
src = open("nc44.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-21a-nc44"', 'SCRIPT_VERSION = "2026-09-22a-nclost"')
rep('              ("F30", dict(rank="contrast", cap=100.0, diverge=1.5, floor=0.30)),\n'
    '              ("F10", dict(rank="contrast", cap=100.0, diverge=1.5, floor=0.10))]\n',
    '              ]\n'
    'NCL_UM = 7.0\n'
    'NCL_ROWS = []\n'
    'NCL_CTL = {}\n'
    'NCL_COLS = ["movie", "t", "lost", "why", "p", "s_u", "s_v", "d_um", "comp_n", "comp_p", "to_end", "exempt"]\n'
    '\n'
    '\n'
    'def ncl_gt(name):\n'
    '    g = read_geff(geff_path(root, "train", name))\n'
    '    g_t = np.asarray(g["t"], int)\n'
    '    g_pos = np.stack([np.asarray(g[c_], float) for c_ in ("z", "y", "x")], axis=1) * SCALE\n'
    '    row_of = {int(v_): i_ for i_, v_ in enumerate(g["node_ids"])}\n'
    '    ge = [(row_of[int(a_)], row_of[int(b_)]) for a_, b_ in g["edges"] if int(a_) in row_of and int(b_) in row_of]\n'
    '    return g_t, g_pos, ge\n'
    '\n'
    '\n'
    'def ncl_match(nodes, g_t, g_pos):\n'
    '    """GT row -> predicted node id, per frame Hungarian on distance within NCL_UM."""\n'
    '    by_t = defaultdict(list)\n'
    '    for nid, v_ in nodes.items():\n'
    '        by_t[int(v_["t"])].append(nid)\n'
    '    out = {}\n'
    '    for t_ in np.unique(g_t):\n'
    '        rows = np.where(g_t == t_)[0]\n'
    '        ids = by_t.get(int(t_), [])\n'
    '        if not len(rows) or not ids:\n'
    '            continue\n'
    '        P = np.asarray([nodes[i_]["xyz"] for i_ in ids], float) * SCALE\n'
    '        D = np.linalg.norm(g_pos[rows][:, None, :] - P[None, :, :], axis=2)\n'
    '        cost = np.where(D <= NCL_UM, D, 1e6)\n'
    '        ri, ci = linear_sum_assignment(cost)\n'
    '        for r_, c_ in zip(ri, ci):\n'
    '            if D[r_, c_] <= NCL_UM:\n'
    '                out[int(rows[r_])] = ids[c_]\n'
    '    return out\n'
    '\n'
    '\n'
    'def ncl_tp(nodes, edges, so, g_t, g_pos, ge):\n'
    '    """{GT edge index: (pred a, pred b)} for the GT edges whose matched nodes are joined, either direction."""\n'
    '    m = ncl_match(nodes, g_t, g_pos)\n'
    '    es = set(edges)\n'
    '    out = {}\n'
    '    for k_, (u_, v_) in enumerate(ge):\n'
    '        a_, b_ = m.get(u_), m.get(v_)\n'
    '        if a_ is None or b_ is None:\n'
    '            continue\n'
    '        if (a_, b_) in es:\n'
    '            out[k_] = (a_, b_)\n'
    '        elif (b_, a_) in es:\n'
    '            out[k_] = (b_, a_)\n'
    '    return out\n'
    '\n'
    '\n'
    'def ncl_comp(ilp_edges):\n'
    '    """per ILP node: (component node count, frames to the nearer end along the chain; forks end a chain walk)."""\n'
    '    succ, pred = defaultdict(list), defaultdict(list)\n'
    '    for a_, b_ in ilp_edges:\n'
    '        succ[a_].append(b_)\n'
    '        pred[b_].append(a_)\n'
    '    nodes_ = set(succ) | set(pred)\n'
    '    comp, seen = {}, set()\n'
    '    for s_ in nodes_:\n'
    '        if s_ in seen:\n'
    '            continue\n'
    '        stack, cc = [s_], []\n'
    '        seen.add(s_)\n'
    '        while stack:\n'
    '            x_ = stack.pop()\n'
    '            cc.append(x_)\n'
    '            for y_ in succ[x_] + pred[x_]:\n'
    '                if y_ not in seen:\n'
    '                    seen.add(y_)\n'
    '                    stack.append(y_)\n'
    '        for x_ in cc:\n'
    '            comp[x_] = (len(cc), tuple(cc))\n'
    '    def _walk(x_, nxt):\n'
    '        n_ = 0\n'
    '        while len(nxt[x_]) == 1 and n_ < 200:\n'
    '            x_ = nxt[x_][0]\n'
    '            n_ += 1\n'
    '        return n_\n'
    '    return comp, succ, pred, _walk\n'
    '\n'
    '\n'
    'def ncl_ledger(name, seeds, pm_c, ctl, nc):\n'
    '    """ctl / nc = dict(nodes, edges (final), so, ilp (ILP edges)). Rows into NCL_ROWS."""\n'
    '    g_t, g_pos, ge = ncl_gt(name)\n'
    '    tp_c = ncl_tp(ctl["nodes"], ctl["edges"], ctl["so"], g_t, g_pos, ge)\n'
    '    tp_n = ncl_tp(nc["nodes"], nc["edges"], nc["so"], g_t, g_pos, ge)\n'
    '    seed_edge_n = {(nc["so"].get(a_), nc["so"].get(b_)) for a_, b_ in nc["edges"]}\n'
    '    ilp_n = {(nc["so"].get(a_), nc["so"].get(b_)) for a_, b_ in nc["ilp"]}\n'
    '    used_n = set()\n'
    '    for a_, b_ in ilp_n:\n'
    '        used_n.add(a_)\n'
    '        used_n.add(b_)\n'
    '    comp, succ, pred, walk = ncl_comp(ctl["ilp"])\n'
    '    ilp_c = set(ctl["ilp"])\n'
    '    cmean = {}\n'
    '    n_lost = 0\n'
    '    for k_, (a_, b_) in tp_c.items():\n'
    '        sa, sb = ctl["so"].get(a_), ctl["so"].get(b_)\n'
    '        lost = 0 if k_ in tp_n else 1\n'
    '        if not lost:\n'
    '            why = "kept"\n'
    '        elif (a_, b_) not in ilp_c or sa is None or sb is None:\n'
    '            why = "non_ilp"\n'
    '        elif (sa, sb) in seed_edge_n:\n'
    '            why = "rematch"\n'
    '        elif sa not in used_n or sb not in used_n:\n'
    '            why = "node_unselected"\n'
    '        elif (sa, sb) in ilp_n:\n'
    '            why = "post_filter"\n'
    '        else:\n'
    '            why = "rerouted"\n'
    '        p_ = s_u = s_v = cn = cp = te = float("nan")\n'
    '        if sa is not None and sb is not None and sb[0] == sa[0] + 1:\n'
    '            p_ = pm_c.get((sa[0], sa[1], sb[1]), (float("nan"),))[0]\n'
    '            s_u = float(np.asarray(seeds[sa[0]][1]).reshape(-1)[sa[1]])\n'
    '            s_v = float(np.asarray(seeds[sb[0]][1]).reshape(-1)[sb[1]])\n'
    '        if a_ in comp:\n'
    '            cn, members = comp[a_]\n'
    '            if members not in cmean:\n'
    '                ps_ = []\n'
    '                for x_ in members:\n'
    '                    for y_ in succ[x_]:\n'
    '                        sx, sy = ctl["so"].get(x_), ctl["so"].get(y_)\n'
    '                        if sx is not None and sy is not None:\n'
    '                            ps_.append(pm_c.get((sx[0], sx[1], sy[1]), (float("nan"),))[0])\n'
    '                cmean[members] = float(np.nanmean(ps_)) if ps_ else float("nan")\n'
    '            cp = cmean[members]\n'
    '            te = min(walk(b_, succ) + 0, walk(a_, pred) + 0)\n'
    '        d_ = float(np.linalg.norm((np.asarray(ctl["nodes"][a_]["xyz"], float) - np.asarray(ctl["nodes"][b_]["xyz"], float)) * SCALE))\n'
    '        ex_ = int((sa in NCL_CTL.get("ex", ())) or (sb in NCL_CTL.get("ex", ())))\n'
    '        NCL_ROWS.append((name, int(ctl["nodes"][a_]["t"]), lost, why, p_, s_u, s_v, d_, cn, cp, te, ex_))\n'
    '        n_lost += lost\n'
    '    return len(tp_c), len(tp_n), n_lost\n')

# capture per arm, ledger after the costed arm
rep("        if arm == STEAL_ARMS[0][0]:\n"
    "            EX_D, EX_T, _n_dau, _n_trk = _hyp_exempt(edges0, seed_of_node, PRE_STATE[\"final_edges\"])\n",
    "        _cap = dict(nodes={g_: {\"t\": v_[\"t\"], \"xyz\": np.asarray(v_[\"xyz\"], float).copy()} for g_, v_ in PRE_STATE[\"nodes\"].items()},\n"
    "                    edges=list(PRE_STATE[\"final_edges\"]), so=dict(_ax[1]), ilp=list(e_x))\n"
    "        if arm == STEAL_ARMS[0][0]:\n"
    "            EX_D, EX_T, _n_dau, _n_trk = _hyp_exempt(edges0, seed_of_node, PRE_STATE[\"final_edges\"])\n"
    "            NCL_CTL.clear()\n"
    "            NCL_CTL.update(cap=_cap, ex=set(EX_T), pm=_ax[0])\n"
    "        else:\n"
    "            try:\n"
    "                _ntc, _ntn, _nl = ncl_ledger(name, seeds, NCL_CTL[\"pm\"], NCL_CTL[\"cap\"], _cap)\n"
    "                print(f\"    nclost {name}: true edges control {_ntc} costed {_ntn} lost {_nl}\", flush=True)\n"
    "            except Exception as _e:                                # noqa: BLE001\n"
    "                warn_once(\"nclost\", f\"{name}: ledger failed ({type(_e).__name__}: {_e})\")\n"
    "        del _cap\n")

# the summary, then stop before the harness report (its gate cross needs per-arm 6bba fork tables)
rep('print("\\n=== ANCHORS / official scorer ===", flush=True)\n',
    'import pandas as _pd                                                     # noqa: E402\n'
    '_df = _pd.DataFrame(NCL_ROWS, columns=NCL_COLS)\n'
    '_df.to_csv("/kaggle/working/nclost_edges.csv", index=False)\n'
    'print(f"\\n=== NCLOST: {len(_df):,} true edges of the control over {_df.movie.nunique()} movies; lost under NC20XT {int(_df.lost.sum()):,} "\n'
    '      f"({_df.lost.mean():.2%}) ===", flush=True)\n'
    'print("  by reason:", flush=True)\n'
    'for w_, n_ in _df[_df.lost == 1].why.value_counts().items():\n'
    '    print(f"    {w_:<16} {n_:>6}  ({n_ / max(1, int(_df.lost.sum())):.1%})", flush=True)\n'
    'print("  lost vs kept, quantiles 10 / 25 / 50 / 75 / 90:", flush=True)\n'
    'for c_ in ("p", "s_u", "s_v", "d_um", "comp_n", "comp_p", "to_end"):\n'
    '    for lab_, m_ in (("lost", _df.lost == 1), ("kept", _df.lost == 0)):\n'
    '        q_ = _df.loc[m_, c_].quantile([.1, .25, .5, .75, .9]).tolist()\n'
    '        print(f"    {c_:<7} {lab_:<5} " + " ".join(f"{x_:>9.3f}" for x_ in q_), flush=True)\n'
    'print("  loss rate by control ILP component size:", flush=True)\n'
    'for lo_, hi_ in ((0, 5), (5, 10), (10, 20), (20, 50), (50, 101), (101, 10 ** 9)):\n'
    '    m_ = (_df.comp_n >= lo_) & (_df.comp_n < hi_)\n'
    '    print(f"    [{lo_:>3}, {hi_ if hi_ < 10 ** 8 else 0:>3}) true edges {int(m_.sum()):>7,}  lost {int(_df.lost[m_].sum()):>6,}  rate {_df.lost[m_].mean() if m_.any() else float(\'nan\'):.3%}", flush=True)\n'
    'print("  loss rate by frames to the nearer ILP track end:", flush=True)\n'
    'for lo_, hi_ in ((0, 1), (1, 2), (2, 4), (4, 8), (8, 10 ** 9)):\n'
    '    m_ = (_df.to_end >= lo_) & (_df.to_end < hi_)\n'
    '    print(f"    [{lo_}, {hi_ if hi_ < 10 ** 8 else 0}) true edges {int(m_.sum()):>7,}  lost {int(_df.lost[m_].sum()):>6,}  rate {_df.lost[m_].mean() if m_.any() else float(\'nan\'):.3%}", flush=True)\n'
    'print("  loss rate by ranker p:", flush=True)\n'
    'for lo_, hi_ in ((0.48, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.97), (0.97, 1.01)):\n'
    '    m_ = (_df.p >= lo_) & (_df.p < hi_)\n'
    '    print(f"    [{lo_}, {hi_}) true edges {int(m_.sum()):>7,}  lost {int(_df.lost[m_].sum()):>6,}  rate {_df.lost[m_].mean() if m_.any() else float(\'nan\'):.3%}", flush=True)\n'
    '_pm = _df.groupby("movie").lost.agg(["sum", "count"])\n'
    '_pm = _pm.sort_values("sum", ascending=False)\n'
    'print("  movies with the most lost true edges: " + ", ".join(f"{i_} {int(r_[\'sum\'])}/{int(r_[\'count\'])}" for i_, r_ in _pm.head(8).iterrows()), flush=True)\n'
    'print(f"  exempt endpoints among the lost: {int(_df.exempt[_df.lost == 1].sum())}", flush=True)\n'
    'print("NCLOST OK", flush=True)\n'
    'sys.exit(0)\n'
    'print("\\n=== ANCHORS / official scorer ===", flush=True)\n')

open("nclost.py", "w", encoding="utf-8", newline="\n").write(src)
print("written nclost.py", src.count("\n"), "lines")
