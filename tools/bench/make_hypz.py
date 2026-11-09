"""hypx.py -> hypz.py: the shipped ILP (node cost 0.35, fork tracks exempt) with a DIVISION VARIABLE PER SCORED TRIPLE.

WHY (2026-09-24). divtrip: a division triple (mother p at t, daughters a, b at t+1) scored without the linker reaches
AUC 0.88 out of fold by movie, and the top-20-per-frame pool holds 55% of the GT divisions where the pipeline delivers
42% -- but at 0.3% precision per candidate, so the triple score can only be a PRIOR on the edge costs, never a decision
by itself. The user (2026-09-24): "那么主要就在ILP这里，可以尝试一下 ... 去做吧". The shipped ILP charges a flat
ILP_DIV_W = 0.55 for any second out-edge; the second daughter's edge is often below the 0.48 candidate floor (the ranker
prefers the daughter that continues the mother's motion), so the division is left to the orphan patch, whose
precondition (the ranker linked one daughter and left the other an orphan) is wrong whenever the ranker linked the
division wrong. Here the ILP is given, for every pooled triple whose two daughters sit inside the ranker's 10 um pair
radius, a binary z with its own cost from the triple score, and the two edges of the triple as candidates even below
the floor (usable only through z). The ranker's edge probabilities, the disappearance cost of the cell that would lose
a child, the node cost and the exemption stay exactly as shipped, so the ILP still does the vetoing.

THE MODEL (solve_hyp + trip): for triple j with mother g_p and daughters g_a, g_b (global node ids), cost w_j:
    sum_out y + d_g = x_g + v_g + sum_{j: mother g} z_j      (a division through v OR through one z)
    v_g + sum_{j: mother g} z_j <= x_g
    z_j <= y_pa,  z_j <= y_pb                                (a division is its two edges)
    y_e <= sum_{j uses e} z_j   for an edge e added below the floor (division-only edge)
    objective += sum_j w_j z_j;  v keeps its own cost (DIV_W, or the arm's vcost)
  w_j = clip(DIV_W - lam * (s_j - S0), wmin, wmax), s_j = logit of scorer B's out-of-fold probability, S0 = -9.0
  (about the 92nd percentile of the pooled candidates' logits; the positives' median is -7.2, their p75 -3.9, the pool's
  median -10.75). A triple the scorer likes is a cheap division; one it dislikes costs at least what v costs.

SCORER B is refitted in the kernel from divtrip's candidate table (biohub-divtrip: divtrip_candidates.csv, 389,914
pooled triples with GEO + IMG features, labels, movie folds), out of fold by the table's own 5 movie folds (divtrip's
_hgb / _w), so every movie's score comes from a model that never saw it.

ARMS (each a fresh full solve on the F4HP set, REF's exemption set ex_t, node cost 0.35):
  NC35XT   no z                                  -- the shipped composition (submit_nc35) as the control
  Z05      lam 0.05, wmin 0.10                   -- gentle: logit -3 -> 0.25, 0 -> 0.10
  Z10      lam 0.10, wmin 0.10                   -- logit -7 -> 0.35, -5 -> 0.15, -3 -> 0.10
  Z20      lam 0.20, wmin 0.05                   -- logit -8 -> 0.35, -7 -> 0.15
  Z10V     lam 0.10, wmin 0.10, v cost 0.80 and wmax 0.80  -- unscored / disliked forks dearer than shipped
Per arm the report adds: triples offered / with both pairs in the ranker's set / edges added below the floor, z selected,
their divtrip labels (TP / FP by the 5 um GT match), how many survive the post-link patches, and hypz_z_<arm>.csv with
every selected z (movie, t, p, a, b, w, s, label). The gate cross (each arm's gate refitted out of fold on the arm's own
forks) and the paired bootstrap against REF + joint are hypx's, unchanged -- the arms' z forks pass through the same
patches and the same gate as every other fork, which is the user's question about the later steps eating them.

Attach: hypx's sources + kernel faustine649/biohub-divtrip. CPU; hypx ran 6 arms in 420 min, this is 5 arms + a
2-4 min scorer fit; budget guard 10 h.
"""
src = open("hypx.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-21a-hypx"', 'SCRIPT_VERSION = "2026-09-24a-hypz"')

# ---- the arms ---------------------------------------------------------------------------------------------
rep('HYP_ARMS = [("C0", dict(cost=0.0, ex=None)),\n'
    '            ("NC20", dict(cost=0.20, ex=None)), ("NC20X", dict(cost=0.20, ex="d")), ("NC20XT", dict(cost=0.20, ex="t")),\n'
    '            ("NC35", dict(cost=0.35, ex=None)), ("NC35X", dict(cost=0.35, ex="d"))]\n',
    'Z_S0 = -9.0                    # logit of scorer B at which a triple costs exactly DIV_W\n'
    'HYP_ARMS = [("NC35XT", dict(cost=0.35, ex="t", lam=None)),\n'
    '            ("Z05", dict(cost=0.35, ex="t", lam=0.05, wmin=0.10)),\n'
    '            ("Z10", dict(cost=0.35, ex="t", lam=0.10, wmin=0.10)),\n'
    '            ("Z20", dict(cost=0.35, ex="t", lam=0.20, wmin=0.05)),\n'
    '            ("Z10V", dict(cost=0.35, ex="t", lam=0.10, wmin=0.10, vcost=0.80, wmax=0.80))]\n'
    '\n'
    '\n'
    'def z_cost(s, cfg):\n'
    '    """the division cost of one triple under an arm: DIV_W minus lam per logit above Z_S0, clipped to [wmin, wmax]"""\n'
    '    w = DIV_W - cfg["lam"] * (float(s) - Z_S0)\n'
    '    return float(min(cfg.get("wmax", cfg.get("vcost", DIV_W)), max(cfg["wmin"], w)))\n'
    '\n'
    '\n'
    '# ---- scorer B, refitted out of fold from divtrip\'s candidate table -----------------------------------------\n'
    'import pandas as _pd                                            # noqa: E402\n'
    'TRIP_GEO = ["d_pa", "d_pb", "d_ab", "asym", "mid_off", "ang", "dz_pa", "dz_pb", "dz_ab", "t_frac", "has_m", "speed_m",\n'
    '            "cont_a", "cont_b", "cont_min", "cont_mid", "has_a2", "has_b2", "sep_grow", "sc_p", "sc_a", "sc_b", "sc_min",\n'
    '            "dens_p10", "n_t1_12", "orph_a", "orph_b", "orph_min", "orph_max"]\n'
    'TRIP_IMG = ["i_p", "c_p", "r_p", "hm_p", "c_a", "c_b", "hm_a", "hm_b", "hm_max_ratio", "hm_sum_ratio", "i_ratio", "br_p"]\n'
    '\n'
    '\n'
    'def _trip_hgb():\n'
    '    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, min_samples_leaf=20, early_stopping=False,\n'
    '                                          random_state=0)\n'
    '\n'
    '\n'
    'def _trip_w(y):\n'
    '    n1 = max(1, int(np.sum(y == 1)))\n'
    '    return np.where(y == 1, (len(y) - n1) / n1, 1.0)\n'
    '\n'
    '\n'
    'def load_triples():\n'
    '    """{movie: [(t, p, a, b, s, label)]} from divtrip_candidates.csv, s = logit of scorer B out of fold by the table\'s folds"""\n'
    '    hit = _walk_files(lambda fn: fn if fn == "divtrip_candidates.csv" else None)\n'
    '    assert hit, "attach the kernel faustine649/biohub-divtrip (divtrip_candidates.csv)"\n'
    '    tb = _pd.read_csv(hit["divtrip_candidates.csv"])\n'
    '    for c_ in ["movie", "fold", "t", "p", "a", "b", "label"] + TRIP_GEO + TRIP_IMG:\n'
    '        assert c_ in tb.columns, c_\n'
    '    X = tb[TRIP_GEO + TRIP_IMG].to_numpy(float)\n'
    '    y = tb["label"].to_numpy(int)\n'
    '    f = tb["fold"].to_numpy(int)\n'
    '    p = np.full(len(y), np.nan)\n'
    '    per = []\n'
    '    for k_ in sorted(set(f.tolist())):\n'
    '        tr, te = f != k_, f == k_\n'
    '        m_ = _trip_hgb().fit(X[tr], y[tr], sample_weight=_trip_w(y[tr]))\n'
    '        p[te] = m_.predict_proba(X[te])[:, 1]\n'
    '        if len(set(y[te].tolist())) == 2:\n'
    '            per.append((k_, int(y[te].sum()), roc_auc_score(y[te], p[te])))\n'
    '    assert np.isfinite(p).all()\n'
    '    s = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))\n'
    '    print(f"  scorer B refit on {len(tb):,} pooled triples ({int(y.sum())} positives, {tb.movie.nunique()} movies): "\n'
    '          f"per-fold AUC " + ", ".join(f"fold {k_} ({n_} pos) {a_:.3f}" for k_, n_, a_ in per)\n'
    '          + f"; pooled {roc_auc_score(y, p):.4f}; logit p50 / p90 / p99 {np.quantile(s, .5):.2f} / {np.quantile(s, .9):.2f} / "\n'
    '          f"{np.quantile(s, .99):.2f}, positives p25 / p50 / p75 {np.quantile(s[y == 1], .25):.2f} / "\n'
    '          f"{np.quantile(s[y == 1], .5):.2f} / {np.quantile(s[y == 1], .75):.2f}", flush=True)\n'
    '    for a_, cfg_ in HYP_ARMS:\n'
    '        if cfg_.get("lam") is None:\n'
    '            continue\n'
    '        w_ = np.array([z_cost(v_, cfg_) for v_ in s])\n'
    '        print(f"    {a_}: cost below DIV_W for {int((w_ < DIV_W - 1e-9).sum()):,} triples ({int((w_ < DIV_W - 1e-9)[y == 1].sum())} "\n'
    '              f"positives); at wmin {int((w_ <= cfg_[\'wmin\'] + 1e-9).sum()):,} ({int((w_ <= cfg_[\'wmin\'] + 1e-9)[y == 1].sum())} positives)",\n'
    '              flush=True)\n'
    '    out = defaultdict(list)\n'
    '    for mv, t_, p_, a_, b_, s_, l_ in zip(tb["movie"].tolist(), tb["t"].tolist(), tb["p"].tolist(), tb["a"].tolist(),\n'
    '                                          tb["b"].tolist(), s.tolist(), y.tolist()):\n'
    '        out[mv].append((int(t_), int(p_), int(a_), int(b_), float(s_), int(l_)))\n'
    '    del tb, X\n'
    '    return out\n'
    '\n'
    '\n'
    'TRIP_OF = load_triples()\n'
    'print(f"  triples loaded for {len(TRIP_OF)} movies; {sum(1 for n_ in SIX if n_ in TRIP_OF)} of the {len(SIX)} 6bba "\n'
    '      f"movies covered", flush=True)\n'
    'assert sum(1 for n_ in SIX if n_ in TRIP_OF) >= len(SIX) - 2, "the triple table does not cover the 6bba movies"\n'
    'Z_SEL = []            # per solve: (t, p, a, b, w, s, label) of the selected z triples\n'
    'Z_ROWS = defaultdict(list)   # arm -> [(movie, t, p, a, b, w, s, label, kept_after_patch)]\n')

# ---- solve_hyp: the z variables -----------------------------------------------------------------------------
rep("def solve_hyp(dets, T, seeds, feats, model, cols, parents, node_cost,\n"
    "              disapp=None, floor=None, divw=None, exempt=None):\n",
    "def solve_hyp(dets, T, seeds, feats, model, cols, parents, node_cost,\n"
    "              disapp=None, floor=None, divw=None, exempt=None, trip=None):\n")
rep('    m = Model("hyp")\n',
    "    # ---- hypz: one z per scored triple; its two edges as candidates (division-only when below the floor)\n"
    "    zt = []                                  # (gp, ga, gb, w, s, label, k_a, k_b)\n"
    "    z_of = defaultdict(list)                 # mother global id -> z indices\n"
    "    div_only = {}                            # cand index -> z indices allowed to use it\n"
    "    Z_SEL.clear()\n"
    "    if trip:\n"
    "        kidx = {(gs, gt_): k for k, (gs, gt_, _p) in enumerate(cand)}\n"
    "        pm_t = {}\n"
    "        for t in range(T - 1):\n"
    "            pairs, rows, P, C = feats[t]\n"
    "            for (i_, j_), pr in zip(pairs, probs[t]):\n"
    "                pm_t[(t, int(i_), int(j_))] = float(pr)\n"
    "        for (t_, ip, ia, ib, w_, s_, l_) in trip:\n"
    "            DUP_STAT['z_offered'] += 1\n"
    "            if t_ >= T - 1 or ip >= len(seeds[t_][0]) or max(ia, ib) >= len(seeds[t_ + 1][0]) or ia == ib:\n"
    "                DUP_STAT['z_bad_index'] += 1\n"
    "                continue\n"
    "            pa, pb = pm_t.get((t_, ip, ia)), pm_t.get((t_, ip, ib))\n"
    "            if pa is None or pb is None:\n"
    "                DUP_STAT['z_no_pair'] += 1\n"
    "                DUP_STAT['z_no_pair_tp'] += int(l_)\n"
    "                continue\n"
    "            gp, ga, gb = offset[t_] + ip, offset[t_ + 1] + ia, offset[t_ + 1] + ib\n"
    "            ks = []\n"
    "            for gc_, pc_ in ((ga, pa), (gb, pb)):\n"
    "                k_ = kidx.get((gp, gc_))\n"
    "                if k_ is None:\n"
    "                    k_ = len(cand)\n"
    "                    cand.append((gp, gc_, pc_))\n"
    "                    kidx[(gp, gc_)] = k_\n"
    "                    div_only[k_] = []\n"
    "                    DUP_STAT['z_edges_added'] += 1\n"
    "                ks.append(k_)\n"
    "            j_ = len(zt)\n"
    "            for k_ in ks:\n"
    "                if k_ in div_only:\n"
    "                    div_only[k_].append(j_)\n"
    "            zt.append((gp, ga, gb, float(w_), float(s_), int(l_), ks[0], ks[1]))\n"
    "            z_of[gp].append(j_)\n"
    "        DUP_STAT['z_vars'] += len(zt)\n"
    "        DUP_STAT['z_vars_tp'] += sum(1 for r_ in zt if r_[5] == 1)\n"
    '    m = Model("hyp")\n')
rep("    y = [m.addVar(vtype=\"B\", name=f\"y{k}\") for k in range(len(cand))]\n",
    "    y = [m.addVar(vtype=\"B\", name=f\"y{k}\") for k in range(len(cand))]\n"
    "    z = [m.addVar(vtype=\"B\", name=f\"z{j}\") for j in range(len(zt))]\n")
rep("        m.addCons(quicksum(y[k] for k in out_e[g]) + d[g] == x[g] + v[g])\n"
    "        m.addCons(v[g] <= x[g])\n",
    "        m.addCons(quicksum(y[k] for k in out_e[g]) + d[g] == x[g] + v[g] + quicksum(z[j] for j in z_of.get(g, ())))\n"
    "        m.addCons(v[g] + quicksum(z[j] for j in z_of.get(g, ())) <= x[g])\n"
    "    for j, (_gp, _ga, _gb, _w, _s, _l, ka, kb) in enumerate(zt):\n"
    "        m.addCons(z[j] <= y[ka])\n"
    "        m.addCons(z[j] <= y[kb])\n"
    "    for k, js in div_only.items():\n"
    "        m.addCons(y[k] <= quicksum(z[j] for j in js))\n")
rep("        + dis * quicksum(d) + dw * quicksum(v)\n",
    "        + dis * quicksum(d) + dw * quicksum(v) \\\n"
    "        + quicksum(zt[j][3] * z[j] for j in range(len(zt)))\n")
rep("    n_div = sum(1 for g in range(N) if m.getVal(v[g]) > 0.5)\n"
    "    del m, x, a, d, v, y\n",
    "    n_div = sum(1 for g in range(N) if m.getVal(v[g]) > 0.5)\n"
    "    for j, (gp, ga, gb, w_, s_, l_, _ka, _kb) in enumerate(zt):\n"
    "        if m.getVal(z[j]) > 0.5:\n"
    "            t_ = int(coords[gp, 0])\n"
    "            Z_SEL.append((t_, int(gp - offset[t_]), int(ga - offset[t_ + 1]), int(gb - offset[t_ + 1]), w_, s_, l_))\n"
    "    DUP_STAT['z_selected'] += len(Z_SEL)\n"
    "    DUP_STAT['z_sel_tp'] += sum(1 for r_ in Z_SEL if r_[6] == 1)\n"
    "    n_div += len(Z_SEL)\n"
    "    del m, x, a, d, v, y, z\n")

# ---- the per-movie loop: hand the arm its triples, track the z forks through the patches -----------------------
rep("        n_a, e_a, p_a, aux_a = solve_hyp(dets, T, seeds, feats, MODELS_FOLD[k], COLS,\n"
    "                                         None, cfg[\"cost\"], disapp=ILP_DISAPP, floor=PROD_FLOOR,\n"
    "                                         exempt=ex_set)\n",
    "        trip_a = None\n"
    "        if cfg.get(\"lam\") is not None:\n"
    "            trip_a = [(t_, p_, a_, b_, z_cost(s_, cfg), s_, l_) for (t_, p_, a_, b_, s_, l_) in TRIP_OF.get(name, [])]\n"
    "        n_a, e_a, p_a, aux_a = solve_hyp(dets, T, seeds, feats, MODELS_FOLD[k], COLS,\n"
    "                                         None, cfg[\"cost\"], disapp=ILP_DISAPP, floor=PROD_FLOOR,\n"
    "                                         exempt=ex_set, divw=cfg.get(\"vcost\"), trip=trip_a)\n"
    "        z_sel_a = list(Z_SEL)\n")
rep("        PATCH_CAPS.clear()\n"
    "        patch_final(n_a, e_a, p_a, outa)\n",
    "        PATCH_CAPS.clear()\n"
    "        patch_final(n_a, e_a, p_a, outa)\n"
    "        if z_sel_a:\n"
    "            _nid = {v_: k_ for k_, v_ in so_a.items()}\n"
    "            _fe = set(PRE_STATE[\"final_edges\"])\n"
    "            _fn = PRE_STATE[\"nodes\"]\n"
    "            for (t_, p_, a_, b_, w_, s_, l_) in z_sel_a:\n"
    "                np_, na_, nb_ = _nid.get((t_, p_)), _nid.get((t_ + 1, a_)), _nid.get((t_ + 1, b_))\n"
    "                kept = int(np_ is not None and (np_, na_) in _fe and (np_, nb_) in _fe)\n"
    "                if kept:\n"
    "                    reason = 'kept'\n"
    "                elif np_ is None or na_ is None or nb_ is None or any(n_ not in _fn for n_ in (np_, na_, nb_)):\n"
    "                    reason = 'node_dropped'                 # the short-track filter removed a node of the fork\n"
    "                else:\n"
    "                    reason = 'edge_dropped'\n"
    "                DUP_STAT['z_kept_after_patch'] += kept\n"
    "                DUP_STAT['z_kept_tp'] += int(kept and l_ == 1)\n"
    "                Z_ROWS[arm].append((name, t_, p_, a_, b_, w_, s_, l_, kept, reason, -1 if np_ is None else int(np_)))\n")
rep("        cells.append(f\"{arm} nodes {len(n_a)} ex {DUP_STAT['exempt']} idiv {DUP_STAT['ilp_divisions']} \"\n",
    "        cells.append(f\"{arm} nodes {len(n_a)} ex {DUP_STAT['exempt']} idiv {DUP_STAT['ilp_divisions']} \"\n"
    "                     f\"z {DUP_STAT['z_selected']}/{DUP_STAT['z_vars']} (tp {DUP_STAT['z_sel_tp']}) \"\n")

# ---- the per-arm table ------------------------------------------------------------------------------------------
rep('_dk = ["nodes_offered", "nodes_selected", "edges_selected", "ilp_divisions", "exempt", "solve_s"]\n',
    '_dk = ["nodes_offered", "nodes_selected", "edges_selected", "ilp_divisions", "exempt", "solve_s", "z_offered", "z_no_pair",\n'
    '       "z_no_pair_tp", "z_vars", "z_vars_tp", "z_edges_added", "z_selected", "z_sel_tp", "z_kept_after_patch", "z_kept_tp"]\n')
rep("    by_movie = defaultdict(list)\n",
    "    _arm_of = tag.split(\"+\")[0]\n"
    "    if tag.endswith(\"+joint\") and _arm_of in Z_ROWS:            # hypz: the z forks' gate probabilities (the joint gate ships)\n"
    "        _idx = {(r_[0], r_[2]): i_ for i_, r_ in enumerate(forks_a)}\n"
    "        for r_ in Z_ROWS[_arm_of]:\n"
    "            i_ = _idx.get((r_[0], r_[10]))\n"
    "            if i_ is not None:\n"
    "                Z_GATE[(_arm_of, r_[0], r_[10])] = (float(pa[i_]), int(la[i_]))\n"
    "    by_movie = defaultdict(list)\n")
rep("Z_SEL = []            # per solve: (t, p, a, b, w, s, label) of the selected z triples\n",
    "Z_SEL = []            # per solve: (t, p, a, b, w, s, label) of the selected z triples\n"
    "Z_GATE = {}           # (arm, movie, mother node id) -> (joint-gate probability, the judge's fork label 2 TP / 1 FP / 0)\n")
rep('for a_, _c in BR_ARMS_RUN:\n    _csv_a = pathlib.Path("/kaggle/working") / f"hypx_arm_{a_}.csv"\n',
    'print("\\n=== THE z FORKS PER ARM: where a true division goes. offered = a z variable existed (both daughters inside the "\n'
    '      "ranker\'s 10 um pairs; no pair = outside them); selected = the ILP took it; kept = both edges still in the graph after "\n'
    '      "the post-link patches (node_dropped: the short-track filter removed a node of the fork; edge_dropped: the edge itself); "\n'
    '      "gate = the joint gate\'s verdict at p > 0.03 on the kept ones. TP by divtrip\'s 5 um label; in brackets the judge\'s "\n'
    '      "own verdict on the kept forks (2 TP / 1 FP) ===", flush=True)\n'
    'for a_, _c in BR_ARMS_RUN:\n'
    '    rz = Z_ROWS.get(a_, [])\n'
    '    d_ = DUP_BY_ARM[a_]\n'
    '    if not d_.get("z_vars"):\n'
    '        continue\n'
    '    ntp = sum(1 for r_ in rz if r_[7] == 1)\n'
    '    nk = sum(1 for r_ in rz if r_[8])\n'
    '    nktp = sum(1 for r_ in rz if r_[8] and r_[7] == 1)\n'
    '    lost_node = sum(1 for r_ in rz if r_[9] == "node_dropped")\n'
    '    lost_node_tp = sum(1 for r_ in rz if r_[9] == "node_dropped" and r_[7] == 1)\n'
    '    lost_edge = sum(1 for r_ in rz if r_[9] == "edge_dropped")\n'
    '    lost_edge_tp = sum(1 for r_ in rz if r_[9] == "edge_dropped" and r_[7] == 1)\n'
    '    g_seen = [(r_, Z_GATE.get((a_, r_[0], r_[10]))) for r_ in rz if r_[8]]\n'
    '    g_pass = sum(1 for r_, g_ in g_seen if g_ is not None and g_[0] > THR)\n'
    '    g_pass_tp = sum(1 for r_, g_ in g_seen if g_ is not None and g_[0] > THR and r_[7] == 1)\n'
    '    g_cut_tp = sum(1 for r_, g_ in g_seen if g_ is not None and g_[0] <= THR and r_[7] == 1)\n'
    '    g_none = sum(1 for r_, g_ in g_seen if g_ is None)\n'
    '    j_tp = sum(1 for r_, g_ in g_seen if g_ is not None and g_[1] == 2)\n'
    '    j_fp = sum(1 for r_, g_ in g_seen if g_ is not None and g_[1] == 1)\n'
    '    ws = np.array([r_[5] for r_ in rz]) if rz else np.zeros(1)\n'
    '    print(f"  {a_:<8} offered {d_[\'z_vars\']:,} (TP {d_.get(\'z_vars_tp\', 0)}; no pair {d_.get(\'z_no_pair\', 0):,}, TP "\n'
    '          f"{d_.get(\'z_no_pair_tp\', 0)}) -> selected {len(rz):,} (TP {ntp}, precision {ntp / max(1, len(rz)):.3f}) -> kept "\n'
    '          f"{nk} (TP {nktp}; lost node_dropped {lost_node} / TP {lost_node_tp}, edge_dropped {lost_edge} / TP {lost_edge_tp}) "\n'
    '          f"-> gate pass {g_pass} (TP {g_pass_tp}; cut TP {g_cut_tp}; {g_none} not in the fork table) [judge on kept: "\n'
    '          f"TP {j_tp} FP {j_fp}]; cost p10 / p50 / p90 {np.quantile(ws, .1):.2f} / {np.quantile(ws, .5):.2f} / "\n'
    '          f"{np.quantile(ws, .9):.2f}; by cost bin (n, TP): "\n'
    '          + ", ".join(f"<{hi_:.2f}: {int(((ws >= lo_) & (ws < hi_)).sum())}, "\n'
    '                      f"{sum(1 for r_ in rz if lo_ <= r_[5] < hi_ and r_[7] == 1)}"\n'
    '                      for lo_, hi_ in ((0.0, 0.15), (0.15, 0.30), (0.30, 0.45), (0.45, 0.56), (0.56, 9.0))), flush=True)\n'
    '    _csv_z = pathlib.Path("/kaggle/working") / f"hypz_z_{a_}.csv"\n'
    '    with open(_csv_z, "w", encoding="utf-8", newline="\\n") as _fh:\n'
    '        _fh.write("movie,t,p,a,b,w,s,label,kept,reason,mother_id,gate_p,judge_label\\n")\n'
    '        for r_ in rz:\n'
    '            g_ = Z_GATE.get((a_, r_[0], r_[10]))\n'
    '            _fh.write(",".join(str(v_) for v_ in r_) + ("," + ",".join(str(v_) for v_ in g_) if g_ else ",,") + "\\n")\n'
    'for a_, _c in BR_ARMS_RUN:\n    _csv_a = pathlib.Path("/kaggle/working") / f"hypx_arm_{a_}.csv"\n')

rep('f"hypx_arm_{a_}.csv"', 'f"hypz_arm_{a_}.csv"')
rep('f"hypx_REF_{DET_SRC}.csv"', 'f"hypz_REF_{DET_SRC}.csv"')
rep('print("HYPX OK", flush=True)', 'print("HYPZ OK", flush=True)')
rep('print(f"[degraded] HYPX INCOMPLETE or anchor miss:', 'print(f"[degraded] HYPZ INCOMPLETE or anchor miss:')

assert src.count("trip=") == 2 and src.count("Z_SEL") >= 4 and src.count("z_cost(") == 3 and src.count("Z_GATE") == 4, (
    src.count("trip="), src.count("Z_SEL"), src.count("z_cost("))
open("hypz.py", "w", encoding="utf-8", newline="\n").write(src)
print("written hypz.py", src.count("\n"), "lines")
