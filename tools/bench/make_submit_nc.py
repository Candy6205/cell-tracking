"""submit_prop.py -> submit_nc.py: a node cost in the ILP with the shipped graph's fork daughters (and their
downstream chains) exempt -- hypx's NC20XT.

WHY (2026-09-21). submit_vt (+0.76% nodes, attach-only vote candidates) scored LB 0.953 against 0.954: the
hidden scorer charges added nodes. The mirror is node DELETION with edges and divisions untouched. hypx, 128
6bba movies: a cost of 0.20 per selected node, zero for the two children of every fork of the shipped solve
(ILP forks + orphan-patch forks) and for each child's downstream chain, gives adj_J +0.0013 [+0.0008,
+0.0018], edge_J +0.0003 [-0.0002, +0.0008], ILP divisions 2,371 = REF's, gated divisions 50/47/75 against
49/51/76, score +0.0025 [+0.0011, +0.0040], final nodes -1.0%, 113 movies up / 14 down. Without the
exemption the cost deletes fresh daughters (48/56/77, score -0.0002).

WHAT CHANGES against submit_prop.py (everything else byte-identical):
  * link_ilp keeps its node table, candidate edges and offsets (NC_CTX);
  * track_movie, after the shipped solve: the shipped post-link stages up to the orphan patch run on a COPY
    to find the forks; nc_exempt collects their children and chains as (t, seed); nc_solve re-states the
    shipped ILP in pyscipopt (hypilp's solve_hyp: in + appear = node, out + disappear = node + division,
    division <= node; edge weight -p, disappearance 1.5, division 0.55, appearance 0) with NC_COST per
    selected non-exempt node on the SAME candidate edges; its solution replaces the shipped one and the
    real post-link stages and the fixed gate run on it;
  * any failure (import, SCIP status, time limit NC_TIME_S, wall clock NC_DEADLINE_S) keeps the shipped
    solution for that movie and says so ([degraded-nc]).
Control = submit_prop (ref 56344592, LB 0.954).
"""
src = open("submit_prop.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-17b-submit-prop"', 'SCRIPT_VERSION = "2026-09-21a-submit-nc"')
rep('print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)',
    'print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)\n'
    'print("  submit_nc: submit_prop + a node cost of 0.20 in the ILP, zero for the children of every fork of the shipped\\n"\n'
    '      "  solve and their downstream chains (hypx NC20XT: adj_J +0.0013 [+0.0008, +0.0018], edge_J flat, divisions kept,\\n"\n'
    '      "  final nodes -1.0%). The mirror of submit_vt (+0.76% nodes, LB 0.953 against 0.954).", flush=True)')

rep("ILP_DIV_W = 0.55   # set from the fine sweep's argmax\n",
    "ILP_DIV_W = 0.55   # set from the fine sweep's argmax\n"
    "\n"
    "# ---- NC: node cost with the fork daughters' tracks exempt (hypx NC20XT) ---------------------------\n"
    "NC_COST = 0.20\n"
    "NC_TIME_S = 900.0               # SCIP time limit per movie; not optimal -> the shipped solution stays\n"
    "NC_DEADLINE_S = 10.0 * 3600     # wall clock from here; later movies keep the shipped solution\n"
    "NC_T0 = time.time()\n"
    "NC_CTX = {}                     # link_ilp's node table / candidate edges / offsets of the current movie\n"
    "NC_TOT = defaultdict(int)\n"
    "NC_DEGRADED = []\n")

# ---- the functions, before link_ilp ----------------------------------------------------------------
rep("def link_ilp(dets, T, div_w, head):\n",
    "def nc_exempt(edges0, final_edges):\n"
    '    """hypx _hyp_exempt: the children of every fork of the shipped ILP solution (edges0) and of the patched\n'
    "    graph (final_edges), plus each child's downstream chain in the patched graph up to the next fork or end,\n"
    '    as (t, seed) pairs through SEED_OF_NODE. Gap-close insertions (no seed) are walked through."""\n'
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
    "    return {SEED_OF_NODE[g] for g in tn if SEED_OF_NODE.get(g) is not None}, len(d), len(tn)\n"
    "\n"
    "\n"
    "def nc_solve(exempt, div_w):\n"
    '    """The shipped ILP re-stated in pyscipopt (hypilp solve_hyp, no parents) on NC_CTX\'s node table and\n'
    "    candidate edges, plus NC_COST per selected node outside `exempt`. Returns (nodes, edges, seed_of_node,\n"
    '    stats) in link_ilp\'s shapes -- every detection is a node, selected or not -- or raises."""\n'
    "    from pyscipopt import Model, quicksum\n"
    "    coords, cand, offset = NC_CTX[\"coords\"], NC_CTX[\"cand\"], NC_CTX[\"offset\"]\n"
    "    N = len(coords)\n"
    "    nseed = NC_CTX[\"nseed\"]\n"
    "    ex = set()\n"
    "    for t_, q_ in exempt:\n"
    "        t_, q_ = int(t_), int(q_)\n"
    "        if t_ in offset and 0 <= q_ < nseed[t_]:\n"
    "            ex.add(offset[t_] + q_)\n"
    "    m = Model(\"nc\")\n"
    "    m.hideOutput()\n"
    "    m.setParam(\"limits/time\", NC_TIME_S)\n"
    "    x = [m.addVar(vtype=\"B\", name=f\"x{g}\") for g in range(N)]\n"
    "    a = [m.addVar(vtype=\"B\", name=f\"a{g}\") for g in range(N)]\n"
    "    d = [m.addVar(vtype=\"B\", name=f\"d{g}\") for g in range(N)]\n"
    "    v = [m.addVar(vtype=\"B\", name=f\"v{g}\") for g in range(N)]\n"
    "    y = [m.addVar(vtype=\"B\", name=f\"y{k}\") for k in range(len(cand))]\n"
    "    in_e, out_e = defaultdict(list), defaultdict(list)\n"
    "    for k, (gs, gt_, _p, _z) in enumerate(cand):\n"
    "        out_e[gs].append(k)\n"
    "        in_e[gt_].append(k)\n"
    "    for g in range(N):\n"
    "        m.addCons(quicksum(y[k] for k in in_e[g]) + a[g] == x[g])\n"
    "        m.addCons(quicksum(y[k] for k in out_e[g]) + d[g] == x[g] + v[g])\n"
    "        m.addCons(v[g] <= x[g])\n"
    "    obj = quicksum(-p * y[k] for k, (_gs, _gt, p, _z) in enumerate(cand)) \\\n"
    "        + ILP_DISAPP * quicksum(d) + div_w * quicksum(v) \\\n"
    "        + NC_COST * quicksum(x[g] for g in range(N) if g not in ex)\n"
    "    m.setObjective(obj, \"minimize\")\n"
    "    m.optimize()\n"
    "    status = m.getStatus()\n"
    "    assert status == \"optimal\", f\"SCIP status {status}\"\n"
    "    n_sel = sum(1 for g in range(N) if m.getVal(x[g]) > 0.5)\n"
    "    ys = [k for k in range(len(cand)) if m.getVal(y[k]) > 0.5]\n"
    "    n_div = sum(1 for g in range(N) if m.getVal(v[g]) > 0.5)\n"
    "    del m, x, a, d, v, y\n"
    "    nodes = {g + 1: {\"t\": int(coords[g, 0]), \"xyz\": coords[g, 1:].astype(float)} for g in range(N)}\n"
    "    edges = [(int(cand[k][0]) + 1, int(cand[k][1]) + 1) for k in ys]\n"
    "    son = {g + 1: (int(coords[g, 0]), int(g - offset[int(coords[g, 0])])) for g in range(N)}\n"
    "    return nodes, edges, son, dict(exempt=len(ex), selected=n_sel, edges=len(ys), divisions=n_div)\n"
    "\n"
    "\n"
    "def link_ilp(dets, T, div_w, head):\n")

# ---- link_ilp: keep what the second solve needs ----------------------------------------------------
rep("    graph = PIPE.build_graph(coords, cand_edges)\n",
    "    NC_CTX.clear()\n"
    "    NC_CTX.update(coords=coords, cand=list(cand_edges), offset=dict(offset),\n"
    "                  nseed={t_: len(seeds_of(dets[t_])[0]) for t_ in range(T)})\n"
    "    graph = PIPE.build_graph(coords, cand_edges)\n")

# ---- track_movie: the second solve -------------------------------------------------------------------
rep("    del head\n"
    "    _gap_close(nodes, edges, cand, cand_sc, cand_trees, len(nodes) + 1)\n",
    "    del head\n"
    "    if time.time() - NC_T0 > NC_DEADLINE_S:\n"
    "        NC_DEGRADED.append(ds)\n"
    "        print(f\"    [degraded-nc] {ds}: wall clock {(time.time() - NC_T0) / 3600:.2f} h -- the shipped solution stays\", flush=True)\n"
    "    else:\n"
    "        try:\n"
    "            t_nc = time.time()\n"
    "            _n1 = {g: {\"t\": v_[\"t\"], \"xyz\": v_[\"xyz\"].copy()} for g, v_ in nodes.items()}\n"
    "            _e1 = list(edges)\n"
    "            _gap_close(_n1, _e1, cand, cand_sc, cand_trees, len(_n1) + 1)\n"
    "            _n1, _e1 = _short_filter(_n1, _e1)\n"
    "            _linefit(_n1, _e1)\n"
    "            FORK_LOG.clear()\n"
    "            _fc1 = _FrameCache(zp, meta, T)\n"
    "            _add_safe_divisions(_n1, _e1, _fc1)\n"
    "            FP_STAT.clear()\n"
    "            FORK_LOG.clear()\n"
    "            del _fc1\n"
    "            _ex, _n_dau, _n_trk = nc_exempt(edges, _e1)\n"
    "            _oc0 = defaultdict(int)\n"
    "            for a_, b_ in edges:\n"
    "                _oc0[a_] += 1\n"
    "            _nn, _ee, _son, _st = nc_solve(_ex, ILP_DIV_W)\n"
    "            print(f\"    nc: shipped edges={len(edges)} forks={sum(1 for v_ in _oc0.values() if v_ >= 2)} | daughters={_n_dau} \"\n"
    "                  f\"exempt track nodes={_n_trk} | cost {NC_COST}: selected={_st['selected']} of {len(_nn)} edges={_st['edges']} \"\n"
    "                  f\"forks={_st['divisions']} ({time.time() - t_nc:.0f}s)\", flush=True)\n"
    "            NC_TOT[\"edges_shipped\"] += len(edges)\n"
    "            NC_TOT[\"edges_nc\"] += _st[\"edges\"]\n"
    "            NC_TOT[\"exempt\"] += _st[\"exempt\"]\n"
    "            NC_TOT[\"movies\"] += 1\n"
    "            nodes, edges = _nn, _ee\n"
    "            SEED_OF_NODE.clear()\n"
    "            SEED_OF_NODE.update(_son)\n"
    "            del _n1, _e1, _nn, _ee, _son\n"
    "        except Exception as _e:                      # noqa: BLE001\n"
    "            NC_DEGRADED.append(ds)\n"
    "            print(f\"    [degraded-nc] {ds}: {type(_e).__name__}: {str(_e)[:200]} -- the shipped solution stays\", flush=True)\n"
    "    _gap_close(nodes, edges, cand, cand_sc, cand_trees, len(nodes) + 1)\n")

rep('print("SUBMIT RANKER DONE", flush=True)',
    'print("NC TOTALS: " + " ".join(f"{k}={v}" for k, v in sorted(NC_TOT.items())) + f" | degraded {len(NC_DEGRADED)}: "\n'
    '      f"{NC_DEGRADED[:8]} | elapsed {(time.time() - NC_T0) / 3600:.2f} h", flush=True)\n'
    'print("SUBMIT RANKER DONE", flush=True)')

open("submit_nc.py", "w", encoding="utf-8", newline="\n").write(src)
print("written submit_nc.py", src.count("\n"), "lines")
