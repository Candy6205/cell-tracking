"""hypilp.py -> hypx.py: the node cost with REF's fork daughters exempt.

WHY (2026-09-21). hypilp: a cost c per selected node in the shipped ILP (no parents) is a real adj_J gain
through the node term (NC20 +0.0013 [+0.0008, +0.0019], NC35 +0.0027 [+0.0018, +0.0037]; edge_J flat) but
it deletes the weakly linked NEW nodes first, and fresh daughters are exactly those: true divisions 40 ->
38 (NC20) -> 37 (NC35) over the 100 movies, gated div_J 0.248 -> 0.245 -> 0.230, so the score keeps
+0.0010 at both costs. submit_vt (LB 0.953 against 0.954, +0.76% nodes) says the hidden scorer charges
added nodes; the mirror test, deleting nodes, is only clean if it leaves the divisions alone. Here the
two children of every fork REF produces (the ILP's own forks and the orphan patch's) pay no node cost
(NC20X / NC35X), and in one arm neither does the daughter's downstream chain up to the next fork or end
(NC20XT). Read: adj_J gain kept, true divisions back at REF's count.

ARMS (each a fresh full solve of the re-stated ILP, F4HP set, no parents):
  C0      c 0.00                                 -- must reproduce REF (hypilp v3: 100 of 100)
  NC20    c 0.20                                 -- hypilp's NC20
  NC20X   c 0.20, REF's fork daughters exempt
  NC20XT  c 0.20, daughters + their downstream chain exempt
  NC35    c 0.35
  NC35X   c 0.35, daughters exempt
The exemption set is computed per movie from REF's solve (fork children of the ILP edges) and from
REF's patched graph (PRE_STATE after patch_final: the orphan patch's forks), as (t, seed) pairs, and
handed to solve_hyp as `exempt`; a node in it contributes 0 to the node-cost term. The F4HYP union set,
the parents and the parent classification are not loaded (the HY arms were the 8x harder LPs).

Attach: hypilp's sources (the hyp caches stay attached; the top-level asserts on them hold). CPU.
Six solves per movie on top of REF; budget guard 10 h (hypilp: NC solves ~10 s per movie).
"""
src = open("hypilp.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-15a-hypilp"', 'SCRIPT_VERSION = "2026-09-21a-hypx"')

rep('HYP_ARMS = [("C0", dict(cost=0.0, par=False)),\n'
    '            ("NC10", dict(cost=0.10, par=False)), ("NC20", dict(cost=0.20, par=False)),\n'
    '            ("NC35", dict(cost=0.35, par=False)),\n'
    '            ("HY10", dict(cost=0.10, par=True)), ("HY20", dict(cost=0.20, par=True)),\n'
    '            ("HY35", dict(cost=0.35, par=True))]\n',
    'HYP_ARMS = [("C0", dict(cost=0.0, ex=None)),\n'
    '            ("NC20", dict(cost=0.20, ex=None)), ("NC20X", dict(cost=0.20, ex="d")), ("NC20XT", dict(cost=0.20, ex="t")),\n'
    '            ("NC35", dict(cost=0.35, ex=None)), ("NC35X", dict(cost=0.35, ex="d"))]\n')

rep("HYP_SEL = []          # per solve: (t, xyz_parent, xyz_child_a, xyz_child_b) of the selected parents\n",
    "HYP_SEL = []          # per solve: (t, xyz_parent, xyz_child_a, xyz_child_b) of the selected parents\n"
    "EXEMPT_TOT = defaultdict(int)   # hypx: exemption-set sizes summed over the movies\n")

# ---- the exemption set + the exempt argument of solve_hyp ------------------------------------------
rep("def solve_hyp(dets, T, seeds, feats, model, cols, parents, node_cost,\n"
    "              disapp=None, floor=None, divw=None):\n",
    "def _hyp_exempt(edges0, seed_of_node, final_edges):\n"
    '    """hypx: REF\'s fork daughters as (t, seed) pairs. ex_d = the two children of every fork of the ILP solution\n'
    "    (edges0) and of the patched graph (final_edges: the orphan patch's forks included); ex_t = ex_d plus each\n"
    "    child's downstream chain in the patched graph up to the next fork or end. Nodes without a seed (gap-close\n"
    '    insertions) are walked through but not returned. Returns (ex_d, ex_t, n_daughter_nodes, n_track_nodes)."""\n'
    "    def _succ(edges):\n"
    "        s = defaultdict(list)\n"
    "        for a_, b_ in edges:\n"
    "            s[a_].append(b_)\n"
    "        return s\n"
    "    s0, s1 = _succ(edges0), _succ(final_edges)\n"
    "    d = set()\n"
    "    for s in (s0, s1):\n"
    "        for a_, bs in s.items():\n"
    "            if len(bs) >= 2:\n"
    "                d.update(bs)\n"
    "    tn = set(d)\n"
    "    for g in list(d):\n"
    "        cur = g\n"
    "        while len(s1.get(cur, [])) == 1:\n"
    "            cur = s1[cur][0]\n"
    "            if cur in tn:\n"
    "                break\n"
    "            tn.add(cur)\n"
    "    ex_d = {seed_of_node[g] for g in d if seed_of_node.get(g) is not None}\n"
    "    ex_t = {seed_of_node[g] for g in tn if seed_of_node.get(g) is not None}\n"
    "    return ex_d, ex_t, len(d), len(tn)\n"
    "\n"
    "\n"
    "def solve_hyp(dets, T, seeds, feats, model, cols, parents, node_cost,\n"
    "              disapp=None, floor=None, divw=None, exempt=None):\n")
rep("    if node_cost:\n"
    "        obj = obj + node_cost * quicksum(x)\n",
    "    ex = set()\n"
    "    if exempt:\n"
    "        for t_, q_ in exempt:\n"
    "            t_, q_ = int(t_), int(q_)\n"
    "            if t_ in offset and 0 <= q_ < len(seeds[t_][0]):\n"
    "                ex.add(offset[t_] + q_)\n"
    "    DUP_STAT[\"exempt\"] += len(ex)\n"
    "    if node_cost:\n"
    "        obj = obj + node_cost * quicksum(x[g] for g in range(N) if g not in ex)\n")

# ---- the per-movie loop: exemption sets from REF, no union set, one call per arm ---------------------
rep("    patch_final(nodes0, edges0, pool, outp)\n"
    "    BR_CTX[\"gt\"] = None\n",
    "    patch_final(nodes0, edges0, pool, outp)\n"
    "    BR_CTX[\"gt\"] = None\n"
    "    EX_D, EX_T, _n_dau, _n_trk = _hyp_exempt(edges0, seed_of_node, PRE_STATE[\"final_edges\"])\n"
    "    EXEMPT_TOT[\"daughters\"] += _n_dau\n"
    "    EXEMPT_TOT[\"track_nodes\"] += _n_trk\n")
rep("    cells = []\n"
    "    dets_h = load_dets(name, \"F4HYP\")\n"
    "    par_h = load_parents(name)\n"
    "    assert len(dets_h) == T and len(par_h) == T, (name, len(dets_h), len(par_h), T)\n"
    "    seeds_h, feats_h = features_ext(name, dets_h, T, EP_OF[\"F4HYP\"][name],\n"
    "                                    EP2_OF[\"F4HYP\"].get(name))\n"
    "    for t_ in range(T):\n"
    "        assert len(seeds_h[t_][0]) == len(seeds[t_][0]) + len(par_h[t_]), (\n"
    "            f\"{name} t={t_}: union primaries {len(seeds_h[t_][0])} != children \"\n"
    "            f\"{len(seeds[t_][0])} + parents {len(par_h[t_])}\")\n"
    "    gt_trees = _hyp_gt(name)\n",
    "    cells = []\n")
rep("        if cfg[\"par\"]:\n"
    "            n_a, e_a, p_a, aux_a = solve_hyp(dets_h, T, seeds_h, feats_h, MODELS_FOLD[k], COLS,\n"
    "                                             par_h, cfg[\"cost\"], disapp=ILP_DISAPP, floor=PROD_FLOOR)\n"
    "        else:\n"
    "            n_a, e_a, p_a, aux_a = solve_hyp(dets, T, seeds, feats, MODELS_FOLD[k], COLS,\n"
    "                                             None, cfg[\"cost\"], disapp=ILP_DISAPP, floor=PROD_FLOOR)\n",
    "        ex_set = None if cfg.get(\"ex\") is None else (EX_D if cfg[\"ex\"] == \"d\" else EX_T)\n"
    "        n_a, e_a, p_a, aux_a = solve_hyp(dets, T, seeds, feats, MODELS_FOLD[k], COLS,\n"
    "                                         None, cfg[\"cost\"], disapp=ILP_DISAPP, floor=PROD_FLOOR,\n"
    "                                         exempt=ex_set)\n")
rep("        for (t_, xp_, xa_, xb_) in HYP_SEL:\n"
    "            ga_, gb_, gp_ = (_hyp_match(gt_trees, t_, xa_), _hyp_match(gt_trees, t_, xb_),\n"
    "                             _hyp_match(gt_trees, t_, xp_))\n"
    "            if ga_ >= 0 and gb_ >= 0 and ga_ != gb_:\n"
    "                DUP_STAT[\"par_two_cells\"] += 1\n"
    "            elif gp_ >= 0:\n"
    "                DUP_STAT[\"par_one_cell\"] += 1\n"
    "            else:\n"
    "                DUP_STAT[\"par_unlab\"] += 1\n",
    "")
rep("        seeds_x = seeds_h if cfg[\"par\"] else seeds\n", "        seeds_x = seeds\n")
rep("        cells.append(f\"{arm} nodes {len(n_a)} par {DUP_STAT['hyp_par_sel']}/{DUP_STAT['hyp_par_off']} \"\n",
    "        cells.append(f\"{arm} nodes {len(n_a)} ex {DUP_STAT['exempt']} idiv {DUP_STAT['ilp_divisions']} \"\n")
rep("    del dets_h, par_h, seeds_h, feats_h, gt_trees\n", "")

# ---- the per-arm table ------------------------------------------------------------------------------
rep('_dk = ["nodes_offered", "nodes_selected", "edges_selected", "ilp_divisions", "hyp_par_off",\n'
    '       "hyp_par_sel", "par_two_cells", "par_one_cell", "par_unlab", "children_dropped", "solve_s"]\n',
    '_dk = ["nodes_offered", "nodes_selected", "edges_selected", "ilp_divisions", "exempt", "solve_s"]\n')
rep("          + f\"   (parents selected {d_.get('hyp_par_sel', 0) / max(1, d_.get('hyp_par_off', 1)):.3%} of offered)\", flush=True)\n",
    "          + f\"   (exempt {d_.get('exempt', 0) / max(1, d_.get('nodes_offered', 1)):.3%} of offered; nodes selected \"\n"
    "          f\"{d_.get('nodes_selected', 0) / max(1, d_.get('nodes_offered', 1)):.3%})\", flush=True)\n")
rep('_c0 = ARM_PER.get("C0", {})\n',
    'print(f"  exemption sets from REF, summed: fork daughters {EXEMPT_TOT[\'daughters\']:,}, daughters + downstream chains "\n'
    '      f"{EXEMPT_TOT[\'track_nodes\']:,} nodes", flush=True)\n'
    '_c0 = ARM_PER.get("C0", {})\n')

rep('f"hypilp_arm_{a_}.csv"', 'f"hypx_arm_{a_}.csv"')
rep('f"hypilp_REF_{DET_SRC}.csv"', 'f"hypx_REF_{DET_SRC}.csv"')
rep('print("HYPILP OK", flush=True)', 'print("HYPX OK", flush=True)')
rep('print(f"[degraded] HYPILP INCOMPLETE or anchor miss:', 'print(f"[degraded] HYPX INCOMPLETE or anchor miss:')

assert src.count("dets_h") == 0 and src.count("seeds_h") == 0 and src.count("par_h") == 0, "the union set survives in the loop"
open("hypx.py", "w", encoding="utf-8", newline="\n").write(src)
print("written hypx.py", src.count("\n"), "lines")
