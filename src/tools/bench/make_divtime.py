"""forkprop.py -> divtime.py: where the GT divisions go under the SHIPPED composition, with fork TIMING.

WHY (2026-09-22). Shipped composition on the 128 6bba movies (forkprop CR_NOCAP_DV150 + jointimg): 61 true /
36 false / 64 missed of 125 GT divisions; ungated 66 / 659 / 59, so 59 divisions are never proposed. hypx's
ledger of REF says 29 are detection-limited and 20 stop at 5a (the second daughter already has a predecessor,
0.7-3 um behind it, ranker p 0.93): the second cell is tracked BEFORE the annotated division frame. The
official division matcher tolerates one frame (the GT parent may match the fork or its predecessor, a GT
daughter lineage the child or the grandchild), not more. This run measures the timing: for every GT division,
the frame offset of the nearest predicted fork on its matched lineage; for the missed ones whether the second
daughter's predicted track starts k frames early and how far it runs from the mother's track; for every
surviving predicted fork, its offset to a GT division on the lineage it matches.

One pass-2 arm (the shipped patch); the ledger runs on the PRUNED graph of the cell CR_NOCAP_DV150 + jointimg.
Outputs divtime_gt.csv (one row per GT division), divtime_forks.csv (one row per surviving fork).
"""
src = open("forkprop.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-17a-forkprop"', 'SCRIPT_VERSION = "2026-09-22a-divtime"')
rep('DA_ARMS = [("CR", dict(diverge=2.25, admit=False, rank="contrast", cap=1.0)),\n'
    '           ("CR_CAP2", dict(diverge=2.25, admit=False, rank="contrast", cap=2.0)),\n'
    '           ("CR_CAP4", dict(diverge=2.25, admit=False, rank="contrast", cap=4.0)),\n'
    '           ("CR_CAP4_DV150", dict(diverge=1.5, admit=False, rank="contrast", cap=4.0)),\n'
    '           ("DIST_CAP4_DV150", dict(diverge=1.5, admit=False, rank="dist", cap=4.0)),\n'
    '           ("CR_NOCAP_DV150", dict(diverge=1.5, admit=False, rank="contrast", cap=100.0))]\n',
    'DA_ARMS = [("CR_NOCAP_DV150", dict(diverge=1.5, admit=False, rank="contrast", cap=100.0))]\n'
    'DT_TAG = "CR_NOCAP_DV150+jointimg"\n'
    'DT_UM = 7.0\n'
    'DT_BACK = 8\n'
    'DT_GT, DT_FK = [], []\n'
    'DT_GT_COLS = ["movie", "t_div", "m_matched", "d1_matched", "d2_matched", "n_rel_forks", "best_off", "cls", "d2_start_off",\n'
    '              "d2_track_len_before", "sep_at_div", "sep_min_before", "d2_pred_is_mother_track"]\n'
    'DT_FK_COLS = ["movie", "t_fork", "off", "on_lineage"]\n'
    '\n'
    '\n'
    'def dt_ledger(name, arr, e3):\n'
    '    """arr (n, 4) t z y x voxels, e3 (m, 2) row indices: the pruned final graph of one movie."""\n'
    '    gt = read_geff(geff_path(root, "train", name))\n'
    '    g_t = np.asarray(gt["t"], int)\n'
    '    g_pos = np.stack([np.asarray(gt[c_], float) for c_ in ("z", "y", "x")], axis=1) * SCALE\n'
    '    row_of = {int(v_): i_ for i_, v_ in enumerate(gt["node_ids"])}\n'
    '    gs, gp = defaultdict(list), {}\n'
    '    for a_, b_ in gt["edges"]:\n'
    '        ra, rb = row_of.get(int(a_)), row_of.get(int(b_))\n'
    '        if ra is not None and rb is not None:\n'
    '            gs[ra].append(rb)\n'
    '            gp[rb] = ra\n'
    '    p_t = arr[:, 0].astype(int)\n'
    '    p_pos = arr[:, 1:].astype(float) * SCALE\n'
    '    ps, pp = defaultdict(list), defaultdict(list)\n'
    '    for a_, b_ in np.asarray(e3, int):\n'
    '        ps[int(a_)].append(int(b_))\n'
    '        pp[int(b_)].append(int(a_))\n'
    '    g2p, p2g = {}, {}\n'
    '    for t_ in np.unique(g_t):\n'
    '        gr = np.where(g_t == t_)[0]\n'
    '        pr = np.where(p_t == t_)[0]\n'
    '        if not len(gr) or not len(pr):\n'
    '            continue\n'
    '        tree = cKDTree(p_pos[pr])\n'
    '        cand = tree.query_ball_point(g_pos[gr], DT_UM)\n'
    '        cols = sorted({c_ for cc in cand for c_ in cc})\n'
    '        if not cols:\n'
    '            continue\n'
    '        ci = {c_: k_ for k_, c_ in enumerate(cols)}\n'
    '        cost = np.full((len(gr), len(cols)), 1e6)\n'
    '        for i_, cc in enumerate(cand):\n'
    '            for c_ in cc:\n'
    '                cost[i_, ci[c_]] = np.linalg.norm(g_pos[gr[i_]] - p_pos[pr[c_]])\n'
    '        ri, cj = linear_sum_assignment(cost)\n'
    '        for i_, j_ in zip(ri, cj):\n'
    '            if cost[i_, j_] <= DT_UM:\n'
    '                g2p[int(gr[i_])] = int(pr[cols[j_]])\n'
    '                p2g[int(pr[cols[j_]])] = int(gr[i_])\n'
    '    forks = [f for f, ss in ps.items() if len(ss) >= 2]\n'
    '    divs = [m for m, ss in gs.items() if len(ss) >= 2]\n'
    '    # lineage of each GT division: ancestors (incl. the mother) and descendants of the daughters, DT_BACK frames each way\n'
    '    lin_of = {}\n'
    '    for m in divs:\n'
    '        L = {m: 0}\n'
    '        cur = m\n'
    '        for _ in range(DT_BACK):\n'
    '            cur = gp.get(cur)\n'
    '            if cur is None:\n'
    '                break\n'
    '            L[cur] = 0\n'
    '        front = list(gs[m])\n'
    '        for _ in range(DT_BACK):\n'
    '            nxt = []\n'
    '            for x_ in front:\n'
    '                L[x_] = 1\n'
    '                nxt.extend(gs.get(x_, []))\n'
    '            front = nxt\n'
    '        lin_of[m] = L\n'
    '    node_div = defaultdict(list)\n'
    '    for m, L in lin_of.items():\n'
    '        for x_ in L:\n'
    '            node_div[x_].append(m)\n'
    '\n'
    '    def fork_divs(f):\n'
    '        """GT divisions whose lineage the fork touches: the fork, up to 3 predecessors, its children and grandchildren."""\n'
    '        probe = [f]\n'
    '        cur = f\n'
    '        for _ in range(3):\n'
    '            if len(pp.get(cur, [])) != 1:\n'
    '                break\n'
    '            cur = pp[cur][0]\n'
    '            probe.append(cur)\n'
    '        for c_ in ps[f]:\n'
    '            probe.append(c_)\n'
    '            probe.extend(ps.get(c_, []))\n'
    '        out = set()\n'
    '        for x_ in probe:\n'
    '            g_ = p2g.get(x_)\n'
    '            if g_ is not None:\n'
    '                out.update(node_div.get(g_, []))\n'
    '        return out\n'
    '\n'
    '    rel = defaultdict(list)\n'
    '    for f in forks:\n'
    '        ds_ = fork_divs(f)\n'
    '        if not ds_:\n'
    '            DT_FK.append((name, int(p_t[f]), float("nan"), 0))\n'
    '            continue\n'
    '        best = min(ds_, key=lambda m_: abs(int(p_t[f]) - int(g_t[m_])))\n'
    '        DT_FK.append((name, int(p_t[f]), int(p_t[f]) - int(g_t[best]), 1))\n'
    '        for m_ in ds_:\n'
    '            rel[m_].append(int(p_t[f]) - int(g_t[m_]))\n'
    '\n'
    '    def start_of(x_):\n'
    '        n_ = 0\n'
    '        while len(pp.get(x_, [])) == 1 and len(ps.get(pp[x_][0], [])) == 1 and n_ < 60:\n'
    '            x_ = pp[x_][0]\n'
    '            n_ += 1\n'
    '        return x_, n_\n'
    '\n'
    '    for m in divs:\n'
    '        t0 = int(g_t[m])\n'
    '        d1, d2 = gs[m][0], gs[m][1]\n'
    '        pm, p1, p2 = g2p.get(m), g2p.get(d1), g2p.get(d2)\n'
    '        offs = rel.get(m, [])\n'
    '        best = min(offs, key=abs) if offs else float("nan")\n'
    '        if offs and abs(best) <= 1:\n'
    '            cls = "on_time"\n'
    '        elif offs and best < 0:\n'
    '            cls = "early"\n'
    '        elif offs:\n'
    '            cls = "late"\n'
    '        elif pm is None:\n'
    '            cls = "mother_unmatched"\n'
    '        elif p1 is None or p2 is None:\n'
    '            cls = "daughter_unmatched"\n'
    '        else:\n'
    '            cls = "no_fork"\n'
    '        so = ln = sep0 = sepm = float("nan")\n'
    '        same = -1\n'
    '        if p1 is not None and p2 is not None:\n'
    '            sep0 = float(np.linalg.norm(p_pos[p1] - p_pos[p2]))\n'
    '            # the daughter whose predicted predecessor chain does NOT pass through the matched mother is "the second"\n'
    '            def chain(x_):\n'
    '                out = []\n'
    '                n_ = 0\n'
    '                while len(pp.get(x_, [])) == 1 and len(ps.get(pp[x_][0], [])) == 1 and n_ < DT_BACK:\n'
    '                    x_ = pp[x_][0]\n'
    '                    out.append(x_)\n'
    '                    n_ += 1\n'
    '                return out\n'
    '            c1, c2 = chain(p1), chain(p2)\n'
    '            sec, oth = (p2, c1) if start_of(p2)[1] <= start_of(p1)[1] else (p1, c2)\n'
    '            csec = c2 if sec == p2 else c1\n'
    '            same = int(pm is not None and pm in csec)\n'
    '            st, n_back = start_of(sec)\n'
    '            so = int(p_t[st]) - (t0 + 1)\n'
    '            ln = n_back\n'
    '            seps = []\n'
    '            for x_, y_ in zip(csec, oth):\n'
    '                if p_t[x_] == p_t[y_]:\n'
    '                    seps.append(float(np.linalg.norm(p_pos[x_] - p_pos[y_])))\n'
    '            sepm = min(seps) if seps else float("nan")\n'
    '        DT_GT.append((name, t0, int(pm is not None), int(p1 is not None), int(p2 is not None), len(offs), best, cls,\n'
    '                      so, ln, sep0, sepm, same))\n'
    '\n'
    '\n')

rep("        row = _rescore(name, movies_a[name], drop, f\"{tag}_p{thr}\") if drop else per_a[name]\n",
    "        row = _rescore(name, movies_a[name], drop, f\"{tag}_p{thr}\") if drop else per_a[name]\n"
    "        if tag == DT_TAG:\n"
    "            try:\n"
    "                _snap = movies_a[name]\n"
    "                _keep = np.ones(len(_snap[\"e\"]), bool)\n"
    "                if drop:\n"
    "                    _keep[np.fromiter(drop, int, len(drop))] = False\n"
    "                dt_ledger(name, np.asarray(_snap[\"arr\"], float), _snap[\"e\"][_keep])\n"
    "            except Exception as _e:                                # noqa: BLE001\n"
    "                warn_once(\"divtime\", f\"{name}: ledger failed ({type(_e).__name__}: {_e})\")\n")

rep('if len(ROWS) == len(SIX) and not INCOMPLETE and not INCOMPLETE2 and _r2:\n',
    'import pandas as _pd                                                     # noqa: E402\n'
    '_g = _pd.DataFrame(DT_GT, columns=DT_GT_COLS)\n'
    '_f = _pd.DataFrame(DT_FK, columns=DT_FK_COLS)\n'
    '_g.to_csv("/kaggle/working/divtime_gt.csv", index=False)\n'
    '_f.to_csv("/kaggle/working/divtime_forks.csv", index=False)\n'
    'print(f"\\n=== DIVTIME: {len(_g)} GT divisions, {len(_f)} surviving predicted forks (pruned graph of {DT_TAG}) ===", flush=True)\n'
    'print("  GT divisions by class: " + ", ".join(f"{k_} {v_}" for k_, v_ in _g.cls.value_counts().items()), flush=True)\n'
    'print("  best fork offset (frames, fork - GT division) over the GT divisions that have a related fork:", flush=True)\n'
    'print("    " + ", ".join(f"{int(k_):+d}: {v_}" for k_, v_ in _g.best_off.dropna().astype(int).value_counts().sort_index().items()), flush=True)\n'
    '_nf = _g[_g.cls.isin(["no_fork", "early", "late"])]\n'
    'print(f"  the {len(_nf)} divisions with both daughters matched and no on-time fork: second daughter\'s predicted track start, frames relative to the daughter frame "\n'
    '      f"(0 = starts at the daughter = an orphan; negative = exists before the division):", flush=True)\n'
    'print("    " + ", ".join(f"{int(k_):+d}: {v_}" for k_, v_ in _nf.d2_start_off.dropna().astype(int).clip(-9, 1).value_counts().sort_index().items()), flush=True)\n'
    'for c_ in ("sep_at_div", "sep_min_before", "d2_track_len_before"):\n'
    '    for lab_, m_ in (("no on-time fork", _g.cls.isin(["no_fork", "early", "late"])), ("on_time", _g.cls == "on_time")):\n'
    '        q_ = _g.loc[m_, c_].quantile([.1, .25, .5, .75, .9]).tolist()\n'
    '        print(f"    {c_:<20} {lab_:<16} " + " ".join(f"{x_:>8.2f}" for x_ in q_), flush=True)\n'
    'print("  surviving forks on a GT lineage, offset histogram: " + ", ".join(f"{int(k_):+d}: {v_}" for k_, v_ in _f.off.dropna().astype(int).value_counts().sort_index().items())\n'
    '      + f"; forks on no GT lineage {int((_f.on_lineage == 0).sum())}", flush=True)\n'
    'print("DIVTIME OK", flush=True)\n'
    'if len(ROWS) == len(SIX) and not INCOMPLETE and not INCOMPLETE2 and _r2:\n')

open("divtime.py", "w", encoding="utf-8", newline="\n").write(src)
print("written divtime.py", src.count("\n"), "lines")
