"""dupsup.py -> hypilp.py: parent hypotheses INSIDE the ILP, with a per-node cost.

WHAT IT MEASURES. The transferable part of Ultrack is not its network but
the shape of its decision: whether a bright region is one cell or two is
settled by the temporal solver among competing hypotheses, not by NMS before
linking. hmerge tried the merged reading as a blind rule and lost (hmbench
edge_J -0.0082). Here BOTH readings of every saddle-keyed pair are nodes of
the same ILP: hypcache's union set (children = the hpara primaries, parents =
one hypothesis per pair with saddle >= 0.80 at the segment mass centroid,
heads re-run on the union by ephypcache / ep2hypcache), the shipped ranker's
probabilities on every candidate edge, the shipped costs (floor 0.48,
disappearance 1.5, DIV_W 0.55, appearance 0), plus the exclusivity
constraint child + every parent holding it <= 1.

WHY A NODE COST. Edges have positive value and appearances are free, so
two child chains always out-earn one parent chain (up to four edge terms
per frame against two); nothing in the shipped objective can ever pick the
parent. A cost c per selected node makes the parent win exactly when the
children's edges are jointly weak: children 2 edges x p_c - 2c against the
parent p_p - c per frame. c is swept and the SAME c is run without parents,
so the node-cost effect and the parent effect separate:

  C0    c 0.00, no parents  -- the re-stated ILP must reproduce REF exactly
  NC10 / NC20 / NC35   c 0.10 / 0.20 / 0.35, no parents (F4HP set)
  HY10 / HY20 / HY35   the same c with the parents (F4HYP set)

The solver is pyscipopt (support-pack wheel), the shipped solver's
constraints re-stated: in-edges + appear = node; out-edges + disappear =
node + division; division <= node; a node with no in-candidates appears if
selected, one with no out-candidates disappears if selected (as in
tracksdata). Output like solve_one: every child detection stays a node
(selected or not) except the children of a selected parent, which are
dropped; unselected parents are not output. Then the shipped post-link
patches, the official scorer, the fork table and the joint-gate cross, as
dupsup's arms. Every selected parent is classified against the GT: both
children match two different annotated nodes (a wrong merge of two cells),
the parent matches one annotated node (a one-cell reading), or unlabelled.

Attach: dupsup's 16 sources + biohub-hypcache + biohub-ephypcache +
biohub-ep2hypcache. CPU. 7 solves per movie on top of REF; budget guard 10 h.
"""
import os

src = open("dupsup.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-12a-dupsup"', 'SCRIPT_VERSION = "2026-09-15a-hypilp"')

# ---- caches: the union detection set and the heads run on it ----------------------
rep('if not k.startswith("F4_") and not k.startswith("F4hp_")}',
    'if not k.startswith("F4_") and not k.startswith("F4hp_") and not k.startswith("F4hyp_")}', n=2)
rep('EDGEPROB_F4HP = {k[len("F4hp_"):]: v for k, v in _EP_ALL.items()\n                 if k.startswith("F4hp_")}\n',
    'EDGEPROB_F4HP = {k[len("F4hp_"):]: v for k, v in _EP_ALL.items()\n                 if k.startswith("F4hp_")}\n'
    'EDGEPROB_F4HYP = {k[len("F4hyp_"):]: v for k, v in _EP_ALL.items()\n                  if k.startswith("F4hyp_")}\n')
rep('EDGEPROB2_F4HP = {k[len("F4hp_"):]: v for k, v in _EP2_ALL.items()\n                  if k.startswith("F4hp_")}\n',
    'EDGEPROB2_F4HP = {k[len("F4hp_"):]: v for k, v in _EP2_ALL.items()\n                  if k.startswith("F4hp_")}\n'
    'EDGEPROB2_F4HYP = {k[len("F4hyp_"):]: v for k, v in _EP2_ALL.items()\n                   if k.startswith("F4hyp_")}\n')
rep('FDET_F4HP = _walk_files(_npz_prefix("fdet_F4hp_"))\nprint(f"fdet_F4hp caches: {len(FDET_F4HP)}", flush=True)\n',
    'FDET_F4HP = _walk_files(_npz_prefix("fdet_F4hp_"))\nprint(f"fdet_F4hp caches: {len(FDET_F4HP)}", flush=True)\n'
    'FDET_F4HYP = _walk_files(_npz_prefix("fdet_F4hyp_"))\n'
    'print(f"fdet_F4hyp caches: {len(FDET_F4HYP)}; edgeprob_F4hyp {len(EDGEPROB_F4HYP)}; "\n'
    '      f"edgeprob2_F4hyp {len(EDGEPROB2_F4HYP)}", flush=True)\n'
    'assert len(FDET_F4HYP) >= 140, "attach biohub-hypcache (140 fdet_F4hyp_ caches)"\n'
    'assert len(EDGEPROB_F4HYP) >= 140, "attach biohub-ephypcache (the first head on the union set)"\n')
rep('EP_OF = {"BASE": EDGEPROB, "F4": EDGEPROB_F4}\nEP2_OF = {"BASE": EDGEPROB2, "F4": EDGEPROB2_F4}\n',
    'EP_OF = {"BASE": EDGEPROB, "F4": EDGEPROB_F4}\nEP2_OF = {"BASE": EDGEPROB2, "F4": EDGEPROB2_F4}\n'
    '# the union set: first head on every movie; the second head ONLY where the REF arm has one\n'
    '# (dupsup\'s rule: the arms must differ in the node set, never in the pipeline)\n'
    'EP_OF["F4HYP"] = EDGEPROB_F4HYP\n'
    'EP2_OF["F4HYP"] = {k: v for k, v in EDGEPROB2_F4HYP.items() if k in EDGEPROB2_F4}\n'
    'print(f"  F4HYP heads: {len(EP_OF[\'F4HYP\'])} first, {len(EP2_OF[\'F4HYP\'])} second "\n'
    '      f"(second head kept only on the {len(EDGEPROB2_F4)} REF movies that have one)", flush=True)\n')
rep('    z = np.load(CACHES[name] if arm == "BASE" else FDET_F4[name])\n',
    '    z = np.load(CACHES[name] if arm == "BASE" else FDET_F4HYP[name] if arm == "F4HYP"\n'
    '                else FDET_F4[name])\n')

# ---- the arms ---------------------------------------------------------------------
rep('DUP_ARMS = [("SUP55", dict(R=5.5)), ("SUP65", dict(R=6.5)), ("SUP75", dict(R=7.5))]',
    'HYP_ARMS = [("C0", dict(cost=0.0, par=False)),\n'
    '            ("NC10", dict(cost=0.10, par=False)), ("NC20", dict(cost=0.20, par=False)),\n'
    '            ("NC35", dict(cost=0.35, par=False)),\n'
    '            ("HY10", dict(cost=0.10, par=True)), ("HY20", dict(cost=0.20, par=True)),\n'
    '            ("HY35", dict(cost=0.35, par=True))]\n'
    'DUP_ARMS = HYP_ARMS       # hypilp: seven re-stated ILP solves, no suppression')

# ---- the hyp block ------------------------------------------------------------------
HYP_BLOCK = r'''
# ---- hypilp block ----
from pyscipopt import Model, quicksum            # noqa: E402  (support-pack wheel)

HYP_SEL = []          # per solve: (t, xyz_parent, xyz_child_a, xyz_child_b) of the selected parents


def load_parents(name):
    """hypcache's p{t}: per frame the (n_parents, 2) child indices among primaries."""
    z = np.load(FDET_F4HYP[name])
    T = max(int(k[1:]) for k in z.files if k.startswith("c")) + 1
    return [np.asarray(z[f"p{t}"], np.int64).reshape(-1, 2) if f"p{t}" in z.files
            else np.zeros((0, 2), np.int64) for t in range(T)]


def _hyp_gt(name):
    gt = read_geff(geff_path(root, "train", name))
    g_t = np.asarray(gt["t"], int)
    g_pos = np.stack([np.asarray(gt[c], float) for c in ("z", "y", "x")], axis=1) * SCALE
    trees = {}
    for tt in np.unique(g_t):
        rows = np.where(g_t == tt)[0]
        trees[int(tt)] = (rows, cKDTree(g_pos[rows]))
    return trees


def _hyp_match(trees, t, xyz_vox):
    if t not in trees:
        return -1
    rows, tree = trees[t]
    dd, gi = tree.query(np.asarray(xyz_vox, float) * SCALE)
    return int(rows[gi]) if dd <= MATCH_UM else -1


def solve_hyp(dets, T, seeds, feats, model, cols, parents, node_cost,
              disapp=None, floor=None, divw=None):
    """The shipped ILP re-stated in pyscipopt, plus a cost per selected node and,
    when `parents` is given (hypcache's p{t}), the exclusivity constraint
    child + every parent holding it <= 1. parents None + node_cost 0 = REF.

    Returns what solve_one returns: nodes0, edges0, (cand, cand_sc, cand_trees),
    (pmap, seed_of_node). Every child detection is a node of the output, selected
    or not, EXCEPT the children of a selected parent; unselected parents are not
    output. HYP_SEL and DUP_STAT carry the bookkeeping of this solve."""
    t_solve = time.time()
    fl = PROD_FLOOR if floor is None else floor
    dw = DIV_W if divw is None else divw
    dis = ILP_DISAPP if disapp is None else disapp
    probs = [model.predict_proba(rows[:, cols])[:, 1] if len(pairs) else np.zeros(0)
             for pairs, rows, P, C in feats]
    offset, coords_rows, is_par, par_children = {}, [], [], {}
    n_glob = 0
    for t in range(T):
        c, _s = seeds[t]
        offset[t] = n_glob
        n = len(c)
        npar = 0 if parents is None else len(parents[t])
        nch = n - npar
        assert nch >= 0, (t, n, npar)
        for q, xyz in enumerate(c):
            coords_rows.append([t, xyz[0], xyz[1], xyz[2]])
            is_par.append(q >= nch)
        if npar:
            pt = np.asarray(parents[t], int).reshape(-1, 2)
            assert pt.size == 0 or (pt.min() >= 0 and pt.max() < nch), (
                f"t={t}: a parent's child index is outside the children ({nch} children)")
            for q in range(npar):
                par_children[offset[t] + nch + q] = (offset[t] + int(pt[q, 0]),
                                                     offset[t] + int(pt[q, 1]))
        n_glob += n
    coords = np.asarray(np.round(coords_rows), np.int16) if coords_rows else np.zeros((0, 4), np.int16)
    N = n_glob
    cand = []
    for t in range(T - 1):
        pairs, rows, P, C = feats[t]
        for (i_, j_), pr in zip(pairs, probs[t]):
            if pr >= fl:
                cand.append((offset[t] + int(i_), offset[t + 1] + int(j_), float(pr)))
    m = Model("hyp")
    m.hideOutput()
    x = [m.addVar(vtype="B", name=f"x{g}") for g in range(N)]
    a = [m.addVar(vtype="B", name=f"a{g}") for g in range(N)]
    d = [m.addVar(vtype="B", name=f"d{g}") for g in range(N)]
    v = [m.addVar(vtype="B", name=f"v{g}") for g in range(N)]
    y = [m.addVar(vtype="B", name=f"y{k}") for k in range(len(cand))]
    in_e, out_e = defaultdict(list), defaultdict(list)
    for k, (gs, gt_, _p) in enumerate(cand):
        out_e[gs].append(k)
        in_e[gt_].append(k)
    for g in range(N):
        m.addCons(quicksum(y[k] for k in in_e[g]) + a[g] == x[g])
        m.addCons(quicksum(y[k] for k in out_e[g]) + d[g] == x[g] + v[g])
        m.addCons(v[g] <= x[g])
    holders = defaultdict(list)
    for gp, (ga, gb) in par_children.items():
        holders[ga].append(gp)
        holders[gb].append(gp)
    for gch, gps in holders.items():
        m.addCons(x[gch] + quicksum(x[gp] for gp in gps) <= 1)
    obj = quicksum(-p * y[k] for k, (_gs, _gt, p) in enumerate(cand)) \
        + dis * quicksum(d) + dw * quicksum(v)
    if node_cost:
        obj = obj + node_cost * quicksum(x)
    m.setObjective(obj, "minimize")
    m.optimize()
    status = m.getStatus()
    assert status == "optimal", f"SCIP status {status}"
    xs = np.array([m.getVal(x[g]) > 0.5 for g in range(N)], bool) if N else np.zeros(0, bool)
    ys = [k for k in range(len(cand)) if m.getVal(y[k]) > 0.5]
    n_div = sum(1 for g in range(N) if m.getVal(v[g]) > 0.5)
    del m, x, a, d, v, y
    DUP_STAT["nodes_offered"] += N
    DUP_STAT["nodes_selected"] += int(xs.sum())
    DUP_STAT["edges_selected"] += len(ys)
    DUP_STAT["ilp_divisions"] += n_div
    DUP_STAT["solve_s"] += int(time.time() - t_solve)
    # ---- output: children (all), selected parents, minus the children of a selected parent
    HYP_SEL.clear()
    drop = set()
    for gp, (ga, gb) in par_children.items():
        DUP_STAT["hyp_par_off"] += 1
        if xs[gp]:
            drop.update((ga, gb))
            DUP_STAT["hyp_par_sel"] += 1
            HYP_SEL.append((int(coords[gp, 0]), coords[gp, 1:].astype(float),
                            coords[ga, 1:].astype(float), coords[gb, 1:].astype(float)))
    nodes0, gid_of, seed_of_node = {}, {}, {}
    nid = 1
    for g in range(N):
        if is_par[g] and not xs[g]:
            continue
        if g in drop:
            DUP_STAT["children_dropped"] += 1
            continue
        t_ = int(coords[g, 0])
        nodes0[nid] = {"t": t_, "xyz": coords[g, 1:].astype(float)}
        gid_of[g] = nid
        seed_of_node[nid] = (t_, int(g - offset[t_]))
        nid += 1
    edges0 = []
    for k in ys:
        gs, gt_, _p = cand[k]
        assert gs in gid_of and gt_ in gid_of, "a selected edge touches a dropped node"
        edges0.append((gid_of[gs], gid_of[gt_]))
    cand_p, cand_sc, cand_trees = [], [], []
    for t in range(T):
        c_all = np.asarray(dets[t][0], float)
        s_all = np.asarray(dets[t][1], float)
        cm = (s_all >= CAND_THR) & (s_all < UNET_THRESH)
        cand_p.append(c_all[cm])
        cand_sc.append(s_all[cm])
        cand_trees.append(cKDTree(cand_p[-1] * SCALE) if len(cand_p[-1]) else None)
    pmap = {}
    for t in range(T - 1):
        pairs, rows, P, C = feats[t]
        for (i_, j_), pr, row in zip(pairs, probs[t], rows):
            pmap[(t, int(i_), int(j_))] = (float(pr), float(row[C_PFWD]), float(row[C_PREV]))
    del probs
    return nodes0, edges0, (cand_p, cand_sc, cand_trees), (pmap, seed_of_node)
# ---- end hypilp block ----
'''
rep('DUP_STAT = defaultdict(int)\n', 'DUP_STAT = defaultdict(int)\n' + HYP_BLOCK)

# ---- stage B arm loop -----------------------------------------------------------------
rep('    cells = []\n    n_prim = sum(len(seeds[t][0]) for t in range(T))\n'
    '    for arm, cfg in BR_ARMS:\n'
    '        keep = _dup_keep(seeds, cfg["R"])\n'
    '        n_keep = sum(int(k_.sum()) for k_ in keep)\n'
    '        DUP_STAT.clear()\n',
    '    cells = []\n'
    '    dets_h = load_dets(name, "F4HYP")\n'
    '    par_h = load_parents(name)\n'
    '    assert len(dets_h) == T and len(par_h) == T, (name, len(dets_h), len(par_h), T)\n'
    '    seeds_h, feats_h = features_ext(name, dets_h, T, EP_OF["F4HYP"][name],\n'
    '                                    EP2_OF["F4HYP"].get(name))\n'
    '    for t_ in range(T):\n'
    '        assert len(seeds_h[t_][0]) == len(seeds[t_][0]) + len(par_h[t_]), (\n'
    '            f"{name} t={t_}: union primaries {len(seeds_h[t_][0])} != children "\n'
    '            f"{len(seeds[t_][0])} + parents {len(par_h[t_])}")\n'
    '    gt_trees = _hyp_gt(name)\n'
    '    for arm, cfg in BR_ARMS:\n'
    '        DUP_STAT.clear()\n')
rep('        n_a, e_a, p_a, aux_a = solve_one(dets, T, seeds, feats, MODELS_FOLD[k], COLS,\n'
    '                                         disapp=ILP_DISAPP, floor=PROD_FLOOR, keep=keep)\n',
    '        if cfg["par"]:\n'
    '            n_a, e_a, p_a, aux_a = solve_hyp(dets_h, T, seeds_h, feats_h, MODELS_FOLD[k], COLS,\n'
    '                                             par_h, cfg["cost"], disapp=ILP_DISAPP, floor=PROD_FLOOR)\n'
    '        else:\n'
    '            n_a, e_a, p_a, aux_a = solve_hyp(dets, T, seeds, feats, MODELS_FOLD[k], COLS,\n'
    '                                             None, cfg["cost"], disapp=ILP_DISAPP, floor=PROD_FLOOR)\n'
    '        for (t_, xp_, xa_, xb_) in HYP_SEL:\n'
    '            ga_, gb_, gp_ = (_hyp_match(gt_trees, t_, xa_), _hyp_match(gt_trees, t_, xb_),\n'
    '                             _hyp_match(gt_trees, t_, xp_))\n'
    '            if ga_ >= 0 and gb_ >= 0 and ga_ != gb_:\n'
    '                DUP_STAT["par_two_cells"] += 1\n'
    '            elif gp_ >= 0:\n'
    '                DUP_STAT["par_one_cell"] += 1\n'
    '            else:\n'
    '                DUP_STAT["par_unlab"] += 1\n')
# the fork table and the score quantiles must read the seeds the solve used: the union
# set for the parent arms (v1 died: seed_of_node indexed seeds_h, _judge got seeds)
rep('        sc_a, b2_a = _aux_extra(seeds, pm_a)\n',
    '        seeds_x = seeds_h if cfg["par"] else seeds\n        sc_a, b2_a = _aux_extra(seeds_x, pm_a)\n')
rep('        row_a, ds_a, fr_a = _judge(name, k, outa, seeds, (pm_a, so_a, sc_a, b2_a))\n',
    '        row_a, ds_a, fr_a = _judge(name, k, outa, seeds_x, (pm_a, so_a, sc_a, b2_a))\n')
rep('        DUP_BY_ARM[arm]["primaries"] += n_prim\n'
    '        DUP_BY_ARM[arm]["suppressed"] += n_prim - n_keep\n', '')
rep('        cells.append(f"{arm} sup {n_prim - n_keep} nodes {len(n_a)} "',
    '        cells.append(f"{arm} nodes {len(n_a)} par {DUP_STAT[\'hyp_par_sel\']}/{DUP_STAT[\'hyp_par_off\']} "\n'
    '                     f"{DUP_STAT[\'solve_s\']}s "')
rep('        del ds_a, n_a, e_a, p_a, aux_a, pm_a, so_a, sc_a, b2_a, keep\n',
    '        del ds_a, n_a, e_a, p_a, aux_a, pm_a, so_a, sc_a, b2_a, seeds_x\n')
rep('    del dets, seeds, feats, nodes0, edges0, pool, aux, aux0, pmap, seed_of_node, best2\n',
    '    del dets, seeds, feats, nodes0, edges0, pool, aux, aux0, pmap, seed_of_node, best2\n'
    '    del dets_h, par_h, seeds_h, feats_h, gt_trees\n')

# ---- the per-arm bookkeeping table ---------------------------------------------------
rep('print("\\n=== SUPPRESSION PER ARM, summed over the 128 movies ===", flush=True)\n'
    '_dk = ["primaries", "suppressed", "edges_withheld", "nodes_dropped"]',
    'print("\\n=== THE RE-STATED ILP PER ARM, summed over the 128 movies (nodes offered / selected, "\n'
    '      "edges selected, ILP divisions, parents offered / selected, selected parents whose two "\n'
    '      "children match two different annotated cells / whose parent matches one annotated cell / "\n'
    '      "unlabelled, children dropped, solve seconds) ===", flush=True)\n'
    '_dk = ["nodes_offered", "nodes_selected", "edges_selected", "ilp_divisions", "hyp_par_off",\n'
    '       "hyp_par_sel", "par_two_cells", "par_one_cell", "par_unlab", "children_dropped", "solve_s"]')
rep('''    print(f"  {a_:<8}" + "".join(f"{d_.get(k_, 0):>17,}" for k_ in _dk)
          + f"   ({d_.get('suppressed', 0) / max(1, d_.get('primaries', 1)):.3%} of primaries)", flush=True)''',
    '''    print(f"  {a_:<8}" + "".join(f"{d_.get(k_, 0):>14,}" for k_ in _dk)
          + f"   (parents selected {d_.get('hyp_par_sel', 0) / max(1, d_.get('hyp_par_off', 1)):.3%} of offered)", flush=True)
_c0 = ARM_PER.get("C0", {})
_same = [n for n in _c0 if n in PER and all(_c0[n].get(k_) == PER[n].get(k_)
         for k_ in ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn"))]
print(f"  C0 (re-stated ILP, no node cost, no parents) reproduces REF's edge and division counts "
      f"on {len(_same)} of {len(_c0)} movies" + ("" if len(_same) == len(_c0) else
      f"; differing: {[n for n in _c0 if n not in _same][:6]}"), flush=True)''')
rep('print(f"  {\'arm\':<8}" + "".join(f"{k_:>17}" for k_ in _dk), flush=True)',
    'print(f"  {\'arm\':<8}" + "".join(f"{k_:>14}" for k_ in _dk), flush=True)')

# ---- names ------------------------------------------------------------------------
rep('f"dupsup_REF_{DET_SRC}.csv"', 'f"hypilp_REF_{DET_SRC}.csv"')
rep('f"dupsup_arm_{a_}.csv"', 'f"hypilp_arm_{a_}.csv"')
rep('print("DUPSUP OK", flush=True)', 'print("HYPILP OK", flush=True)')
rep('print(f"[degraded] DUPSUP INCOMPLETE or anchor miss:', 'print(f"[degraded] HYPILP INCOMPLETE or anchor miss:')

assert src.count("solve_hyp(") == 3 and "n_keep" not in src.split("def _dup_keep")[1].split("cells = []")[1].split("extra = \"\"")[0]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hypilp.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written hypilp.py", len(src))
