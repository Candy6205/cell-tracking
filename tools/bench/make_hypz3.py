"""hypz.py -> hypz3.py: the division-variable ILP with three more levers -- a better-fed triple scorer, a wider pool,
and the z daughters exempt from the node cost.

WHY (2026-09-25). hypz: Z10 (a division variable per pooled triple, cost 0.55 - 0.10 per logit above -9, floor 0.10)
reads +0.0024 over the shipped composition after the joint gate, all of it division. Of the 64 true triples that had a
variable, the ILP declined 29; those 29 have the same geometry as the 35 it took, but 18 of them got NO discount from
scorer B (logit below -9, the pool's bulk) -- the ILP was right, the scorer was blind. And the top-20-per-frame pool
(scorer A, which leans on orphan-ness) drops 38 of the 121 reachable true triples before the ILP sees them. The user
(2026-09-25): "去跑吧" on scorer v2 / wider pool / z-daughter exemption.

ARMS (all: node cost 0.35, REF's fork tracks exempt, lam 0.10, wmin 0.10; S0 per scorer = the logit above which 1.15
triples per frame get a discount, which is what -9 was for scorer B on the pool):
  Z10    scorer B on the divtrip pool (top 20 by A)                 -- hypz's pick, the reference arm of this run
  Z10R   scorer B's columns + the ranker's two edge probabilities (p_pa, p_pb, min, max), fitted on the 6bba pool rows
         (the ranker's out-of-fold probabilities exist only there), out of fold by divtrip's folds; pool = the same
  Z10A   scorer A2 = divtrip's pass 1 recomputed in-kernel (every triple of the 199 movies, the positives + ~6,000
         negatives per movie: 121 positives, geometry only), out of fold by divtrip's folds; pool = top 20 per frame by A2
  Z10W   scorer A2, pool = top 60 per frame by A2
  Z10X   scorer B, pool 20, TWO-PASS exemption: the z solution's fork daughters + their downstream chains are added
         to the exemption set and the movie is solved again (the shipped nc_solve's own two-pass shape)
The z fate table, the gate cross and the bootstrap are hypz's. New prints: A2's and R's per-fold AUC, S0 per scorer,
the A pools' true-triple counts at 20 / 60 per frame against the csv pool's.

Attach: hypz's sources (hypx's 19 + biohub-divtrip). CPU; hypz v2 ran 5 arms in 372 min; this adds ~20 min of
pre-passes and one extra solve per movie (Z10X). Budget guard 10 h.
"""
src = open("hypz.py", encoding="utf-8").read()
dt = open("divtrip.py", encoding="utf-8").read()
GEOM = dt[dt.index("def gt_triples(gt):"):dt.index("# ---- raw-image features for the kept candidates")]
assert "def triples_frame(" in GEOM and "def label_triples(" in GEOM and "def _crop" not in GEOM


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-24a-hypz"', 'SCRIPT_VERSION = "2026-09-25a-hypz3"')

# ---- the arms and the per-scorer S0 ----------------------------------------------------------------------------
rep('HYP_ARMS = [("NC35XT", dict(cost=0.35, ex="t", lam=None)),\n'
    '            ("Z05", dict(cost=0.35, ex="t", lam=0.05, wmin=0.10)),\n'
    '            ("Z10", dict(cost=0.35, ex="t", lam=0.10, wmin=0.10)),\n'
    '            ("Z20", dict(cost=0.35, ex="t", lam=0.20, wmin=0.05)),\n'
    '            ("Z10V", dict(cost=0.35, ex="t", lam=0.10, wmin=0.10, vcost=0.80, wmax=0.80))]\n',
    'Z_PER_FRAME = 1.15             # triples per frame that get a discount: S0 per scorer is set so this holds\n'
    'HYP_ARMS = [("Z10", dict(cost=0.35, ex="t", lam=0.10, wmin=0.10, src="B")),\n'
    '            ("Z10R", dict(cost=0.35, ex="t", lam=0.10, wmin=0.10, src="R")),\n'
    '            ("Z10A", dict(cost=0.35, ex="t", lam=0.10, wmin=0.10, src="A", K=20)),\n'
    '            ("Z10W", dict(cost=0.35, ex="t", lam=0.10, wmin=0.10, src="A", K=60)),\n'
    '            ("Z10X", dict(cost=0.35, ex="t", lam=0.10, wmin=0.10, src="B", zx=True))]\n')
rep('def z_cost(s, cfg):\n'
    '    """the division cost of one triple under an arm: DIV_W minus lam per logit above Z_S0, clipped to [wmin, wmax]"""\n'
    '    w = DIV_W - cfg["lam"] * (float(s) - Z_S0)\n',
    'def z_cost(s, cfg, s0=None):\n'
    '    """the division cost of one triple under an arm: DIV_W minus lam per logit above s0 (Z_S0 when not given), clipped to [wmin, wmax]"""\n'
    '    w = DIV_W - cfg["lam"] * (float(s) - (Z_S0 if s0 is None else s0))\n')
# keep the 6bba pool table for the R scorer
rep("    out = defaultdict(list)\n"
    "    for mv, t_, p_, a_, b_, s_, l_ in zip(tb[\"movie\"].tolist(), tb[\"t\"].tolist(), tb[\"p\"].tolist(), tb[\"a\"].tolist(),\n",
    "    global TRIP_TB\n"
    "    TRIP_TB = tb[tb[\"movie\"].isin(set(SIX))].reset_index(drop=True)      # hypz3: the R scorer refits on these rows\n"
    "    out = defaultdict(list)\n"
    "    for mv, t_, p_, a_, b_, s_, l_ in zip(tb[\"movie\"].tolist(), tb[\"t\"].tolist(), tb[\"p\"].tolist(), tb[\"a\"].tolist(),\n")

# ---- divtrip's geometry, the A2 scorer, the R scorer, the pools, S0 -- before the main loop --------------------
rep("for name in SIX:\n    if time.time() - t_all > BUDGET_S:\n",
    "# ---- hypz3: divtrip's triple geometry (ported verbatim), scorer A2, scorer R, the A pools, S0 per scorer ----------\n"
    "R_PARENT, R_SISTER, R_MID = 14.0, 16.0, 8.0      # divtrip's candidate geometry, um\n"
    "D_AB_MIN = 2.0\n"
    "SNAP_UM = 5.0\n"
    "N_NEG_PER_MOVIE = 6000\n"
    "TRIP_FOLDS, TRIP_SEED = 5, 0\n"
    "GEO = TRIP_GEO\n"
    "\n\n" + GEOM + "\n"
    "def _zlogit(p):\n"
    "    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)\n"
    "    return np.log(p / (1 - p))\n"
    "\n\n"
    "def _trip_names():\n"
    "    names = sorted(FDET_F4HP)\n"
    "    assert len(names) == 199, f'{len(names)} fdet_F4hp caches; divtrip had 199 (attach biohub-hparacache + biohub-hpara44)'\n"
    "    rng = np.random.RandomState(TRIP_SEED)\n"
    "    perm = rng.permutation(len(names))\n"
    "    return names, {nm: int(perm[i] % TRIP_FOLDS) for i, nm in enumerate(names)}\n"
    "\n\n"
    "def _prims_um(name):\n"
    "    z = np.load(FDET_F4HP[name])\n"
    "    T = max(int(k[1:]) for k in z.files if k.startswith('c')) + 1\n"
    "    PR = {}\n"
    "    for t in range(T):\n"
    "        c, s = z[f'c{t}'], z[f's{t}']\n"
    "        m = s >= UNET_THRESH\n"
    "        PR[t] = (np.asarray(c[m], float) * SCALE, np.asarray(s[m], float))\n"
    "    return T, PR\n"
    "\n\n"
    "def _frame_triples(PR, T, t, gts):\n"
    "    P, S = PR[t]\n"
    "    Pn, Sn = PR[t + 1]\n"
    "    idx, X = triples_frame(P, S, Pn, Sn, PR[t - 1][0] if t >= 1 else None, PR[t + 2][0] if t + 2 < T else None, t, T)\n"
    "    if not len(idx):\n"
    "        return idx, X, np.zeros(0, int)\n"
    "    y, _m = label_triples(idx, P, Pn, gts.get(t, []))\n"
    "    return idx, X, y\n"
    "\n\n"
    "def fit_scorer_A2():\n"
    "    \"\"\"divtrip's pass 1 in-kernel: every triple of the 199 movies, the positives + ~6,000 negatives per movie, HistGB out of\n"
    "    fold by divtrip's 5 movie folds. Returns (models by fold, fold_of).\"\"\"\n"
    "    names, fold_of = _trip_names()\n"
    "    rng = np.random.RandomState(TRIP_SEED)\n"
    "    X_, y_, f_ = [], [], []\n"
    "    t0 = time.time()\n"
    "    for name in names:\n"
    "        T, PR = _prims_um(name)\n"
    "        gts = gt_triples(read_geff(geff_path(root, 'train', name)))\n"
    "        for t in range(T - 1):\n"
    "            idx, X, y = _frame_triples(PR, T, t, gts)\n"
    "            if not len(idx):\n"
    "                continue\n"
    "            pos = np.where(y == 1)[0]\n"
    "            neg = np.where(y == 0)[0]\n"
    "            take = min(len(neg), max(1, int(round(N_NEG_PER_MOVIE / max(1, T - 1)))))\n"
    "            sel = np.concatenate([pos, rng.choice(neg, take, replace=False)]) if len(neg) else pos\n"
    "            X_.append(X[sel])\n"
    "            y_.append(y[sel])\n"
    "            f_.append(np.full(len(sel), fold_of[name]))\n"
    "    X = np.concatenate(X_)\n"
    "    y = np.concatenate(y_)\n"
    "    f = np.concatenate(f_)\n"
    "    models, per = {}, []\n"
    "    for k_ in range(TRIP_FOLDS):\n"
    "        tr, te = f != k_, f == k_\n"
    "        models[k_] = _trip_hgb().fit(X[tr], y[tr], sample_weight=_trip_w(y[tr]))\n"
    "        if len(set(y[te].tolist())) == 2:\n"
    "            per.append((k_, int(y[te].sum()), roc_auc_score(y[te], models[k_].predict_proba(X[te])[:, 1])))\n"
    "    print(f'  scorer A2 (geometry only, divtrip pass 1 in-kernel, {time.time() - t0:.0f}s): {len(y):,} rows, {int(y.sum())} '\n"
    "          f'positives over {len(names)} movies; per-fold AUC ' + ', '.join(f'fold {k_} ({n_} pos) {a_:.3f}' for k_, n_, a_ in per),\n"
    "          flush=True)\n"
    "    return models, fold_of\n"
    "\n\n"
    "def build_pool_A(models, fold_of, K):\n"
    "    \"\"\"per 6bba movie: every triple scored by the movie's out-of-fold A2 model, the top K per frame kept as\n"
    "    (t, p, a, b, logit, label, rank); also every logit (for S0) and the frame count.\"\"\"\n"
    "    out, logs, n_frames, n_true_all = {}, [], 0, 0\n"
    "    t0 = time.time()\n"
    "    for name in SIX:\n"
    "        T, PR = _prims_um(name)\n"
    "        gts = gt_triples(read_geff(geff_path(root, 'train', name)))\n"
    "        m_ = models[fold_of[name]]\n"
    "        rows = []\n"
    "        for t in range(T - 1):\n"
    "            n_frames += 1\n"
    "            idx, X, y = _frame_triples(PR, T, t, gts)\n"
    "            if not len(idx):\n"
    "                continue\n"
    "            n_true_all += int(y.sum())\n"
    "            s = _zlogit(m_.predict_proba(X)[:, 1])\n"
    "            logs.append(s.astype(np.float32))\n"
    "            order = np.argsort(-s)[:K]\n"
    "            for r_, i_ in enumerate(order):\n"
    "                rows.append((t, int(idx[i_, 0]), int(idx[i_, 1]), int(idx[i_, 2]), float(s[i_]), int(y[i_]), int(r_)))\n"
    "        out[name] = rows\n"
    "    allc = np.concatenate(logs) if logs else np.zeros(0, np.float32)\n"
    "    print(f'  A pools ({time.time() - t0:.0f}s): {len(allc):,} triples over {n_frames:,} frames of the {len(SIX)} 6bba movies, '\n"
    "          f'{n_true_all} true; in the top 20 / 60 per frame: '\n"
    "          f\"{sum(1 for rs in out.values() for r_ in rs if r_[6] < 20 and r_[5] == 1)} / \"\n"
    "          f\"{sum(1 for rs in out.values() for r_ in rs if r_[5] == 1)} (the csv pool: \"\n"
    "          f\"{sum(1 for n_ in SIX for r_ in TRIP_OF.get(n_, []) if r_[5] == 1)})\", flush=True)\n"
    "    return out, allc, n_frames\n"
    "\n\n"
    "def fit_scorer_R():\n"
    "    \"\"\"scorer B's columns + the ranker's two edge probabilities on the 6bba pool rows, out of fold by divtrip's folds.\n"
    "    Returns ({name: [(t, p, a, b, logit, label)]}, logits of every row).\"\"\"\n"
    "    t0 = time.time()\n"
    "    tb = TRIP_TB\n"
    "    ppa = np.full(len(tb), np.nan)\n"
    "    ppb = np.full(len(tb), np.nan)\n"
    "    by_movie = {n_: np.where(tb['movie'].to_numpy() == n_)[0] for n_ in SIX}\n"
    "    for name in SIX:\n"
    "        ix = by_movie[name]\n"
    "        if not len(ix) or name not in FEATS:\n"
    "            continue\n"
    "        seeds_, feats_ = FEATS[name]\n"
    "        m_ = MODELS_FOLD[FOLD_OF[name]]\n"
    "        pm = {}\n"
    "        for t_, (pairs, rows, P, C) in enumerate(feats_):\n"
    "            if not len(pairs):\n"
    "                continue\n"
    "            pr = m_.predict_proba(rows[:, COLS])[:, 1]\n"
    "            for (i_, j_), p_ in zip(pairs, pr):\n"
    "                pm[(t_, int(i_), int(j_))] = float(p_)\n"
    "        tt, pp, aa, bb = (tb[c].to_numpy()[ix] for c in ('t', 'p', 'a', 'b'))\n"
    "        for q_, (t_, p_, a_, b_) in enumerate(zip(tt, pp, aa, bb)):\n"
    "            ppa[ix[q_]] = pm.get((int(t_), int(p_), int(a_)), np.nan)\n"
    "            ppb[ix[q_]] = pm.get((int(t_), int(p_), int(b_)), np.nan)\n"
    "        del pm\n"
    "    X = np.column_stack([tb[TRIP_GEO + TRIP_IMG].to_numpy(float), ppa, ppb, np.fmin(ppa, ppb), np.fmax(ppa, ppb)])\n"
    "    y = tb['label'].to_numpy(int)\n"
    "    f = tb['fold'].to_numpy(int)\n"
    "    p = np.full(len(y), np.nan)\n"
    "    per = []\n"
    "    for k_ in sorted(set(f.tolist())):\n"
    "        tr, te = f != k_, f == k_\n"
    "        if te.sum() == 0 or len(set(y[tr].tolist())) < 2:\n"
    "            continue\n"
    "        m_ = _trip_hgb().fit(X[tr], y[tr], sample_weight=_trip_w(y[tr]))\n"
    "        p[te] = m_.predict_proba(X[te])[:, 1]\n"
    "        if len(set(y[te].tolist())) == 2:\n"
    "            per.append((k_, int(y[te].sum()), roc_auc_score(y[te], p[te])))\n"
    "    ok = np.isfinite(p)\n"
    "    print(f'  scorer R (B + ranker p_pa / p_pb / min / max, {time.time() - t0:.0f}s): {len(y):,} 6bba pool rows, {int(y.sum())} '\n"
    "          f'positives, pairs found {np.isfinite(ppa).mean():.3f} / {np.isfinite(ppb).mean():.3f}; per-fold AUC '\n"
    "          + ', '.join(f'fold {k_} ({n_} pos) {a_:.3f}' for k_, n_, a_ in per)\n"
    "          + (f'; pooled {roc_auc_score(y[ok], p[ok]):.4f}' if ok.sum() and len(set(y[ok].tolist())) == 2 else ''), flush=True)\n"
    "    s = _zlogit(np.where(ok, p, 1e-6))\n"
    "    out = defaultdict(list)\n"
    "    for mv, t_, p_, a_, b_, s_, l_ in zip(tb['movie'].tolist(), tb['t'].tolist(), tb['p'].tolist(), tb['a'].tolist(),\n"
    "                                          tb['b'].tolist(), s.tolist(), y.tolist()):\n"
    "        out[mv].append((int(t_), int(p_), int(a_), int(b_), float(s_), int(l_)))\n"
    "    return out, s\n"
    "\n\n"
    "def _s0(logits, n_frames):\n"
    "    \"\"\"the logit above which Z_PER_FRAME triples per frame get a discount\"\"\"\n"
    "    v = np.sort(np.asarray(logits, float))[::-1]\n"
    "    k_ = min(len(v) - 1, int(round(Z_PER_FRAME * n_frames)))\n"
    "    return float(v[k_]) if len(v) else Z_S0\n"
    "\n\n"
    "print('\\n=== hypz3 pre-passes: scorer A2, the A pools, scorer R, S0 per scorer ===', flush=True)\n"
    "A2_MODELS, TRIP_FOLD_OF = fit_scorer_A2()\n"
    "TRIP_A, _logs_A, _nf_A = build_pool_A(A2_MODELS, TRIP_FOLD_OF, 60)\n"
    "TRIP_R, _logs_R = fit_scorer_R()\n"
    "_b6 = [r_ for n_ in SIX for r_ in TRIP_OF.get(n_, [])]\n"
    "_nf_B = len({(n_, r_[0]) for n_ in SIX for r_ in TRIP_OF.get(n_, [])})\n"
    "S0_OF = {'B': _s0([r_[4] for r_ in _b6], _nf_B), 'R': _s0(_logs_R, _nf_B), 'A': _s0(_logs_A, _nf_A)}\n"
    "print(f\"  S0 per scorer (logit above which {Z_PER_FRAME} triples per frame are discounted): B {S0_OF['B']:.2f} (hypz used {Z_S0}), \"\n"
    "      f\"R {S0_OF['R']:.2f}, A {S0_OF['A']:.2f}; discounted true triples in the 6bba movies: B \"\n"
    "      f\"{sum(1 for r_ in _b6 if r_[5] == 1 and r_[4] > S0_OF['B'])} of {sum(1 for r_ in _b6 if r_[5] == 1)}, R \"\n"
    "      f\"{sum(1 for n_ in SIX for r_ in TRIP_R.get(n_, []) if r_[5] == 1 and r_[4] > S0_OF['R'])}, A (top 60) \"\n"
    "      f\"{sum(1 for n_ in SIX for r_ in TRIP_A.get(n_, []) if r_[5] == 1 and r_[4] > S0_OF['A'])} of \"\n"
    "      f\"{sum(1 for n_ in SIX for r_ in TRIP_A.get(n_, []) if r_[5] == 1)}\", flush=True)\n"
    "del _logs_A, _logs_R, _b6\n"
    "gc.collect()\n"
    "\n"
    "for name in SIX:\n    if time.time() - t_all > BUDGET_S:\n")

# ---- the loop: the source of each arm's triples, the two-pass exemption ---------------------------------------------
rep("        trip_a = None\n"
    "        if cfg.get(\"lam\") is not None:\n"
    "            trip_a = [(t_, p_, a_, b_, z_cost(s_, cfg), s_, l_) for (t_, p_, a_, b_, s_, l_) in TRIP_OF.get(name, [])]\n",
    "        trip_a = None\n"
    "        if cfg.get(\"lam\") is not None:\n"
    "            src_ = cfg.get(\"src\", \"B\")\n"
    "            if src_ == \"A\":\n"
    "                base_ = [(t_, p_, a_, b_, s_, l_) for (t_, p_, a_, b_, s_, l_, r_) in TRIP_A.get(name, []) if r_ < cfg[\"K\"]]\n"
    "            elif src_ == \"R\":\n"
    "                base_ = TRIP_R.get(name, [])\n"
    "            else:\n"
    "                base_ = TRIP_OF.get(name, [])\n"
    "            trip_a = [(t_, p_, a_, b_, z_cost(s_, cfg, S0_OF[src_]), s_, l_) for (t_, p_, a_, b_, s_, l_) in base_]\n")
rep("                                         exempt=ex_set, divw=cfg.get(\"vcost\"), trip=trip_a)\n"
    "        z_sel_a = list(Z_SEL)\n",
    "                                         exempt=ex_set, divw=cfg.get(\"vcost\"), trip=trip_a)\n"
    "        if cfg.get(\"zx\") and trip_a:\n"
    "            # hypz3: the z solution's fork daughters + downstream chains join the exemption set; solve again\n"
    "            _exz = _hyp_exempt(e_a, aux_a[1], e_a)[1]\n"
    "            _n_first = DUP_STAT[\"z_selected\"]\n"
    "            DUP_STAT.clear()\n"
    "            n_a, e_a, p_a, aux_a = solve_hyp(dets, T, seeds, feats, MODELS_FOLD[k], COLS,\n"
    "                                             None, cfg[\"cost\"], disapp=ILP_DISAPP, floor=PROD_FLOOR,\n"
    "                                             exempt=(set(ex_set) | set(_exz)), divw=cfg.get(\"vcost\"), trip=trip_a)\n"
    "            DUP_STAT[\"zx_first_pass_z\"] += _n_first\n"
    "            DUP_STAT[\"zx_added_exempt\"] += len(_exz)\n"
    "        z_sel_a = list(Z_SEL)\n")
rep('_dk = ["nodes_offered", "nodes_selected", "edges_selected", "ilp_divisions", "exempt", "solve_s", "z_offered", "z_no_pair",\n'
    '       "z_no_pair_tp", "z_vars", "z_vars_tp", "z_edges_added", "z_selected", "z_sel_tp", "z_kept_after_patch", "z_kept_tp"]\n',
    '_dk = ["nodes_offered", "nodes_selected", "edges_selected", "ilp_divisions", "exempt", "solve_s", "z_offered", "z_no_pair",\n'
    '       "z_no_pair_tp", "z_vars", "z_vars_tp", "z_edges_added", "z_selected", "z_sel_tp", "z_kept_after_patch", "z_kept_tp",\n'
    '       "zx_first_pass_z", "zx_added_exempt"]\n')
rep('f"hypz_arm_{a_}.csv"', 'f"hypz3_arm_{a_}.csv"')
rep('f"hypz_z_{a_}.csv"', 'f"hypz3_z_{a_}.csv"')
rep('f"hypz_REF_{DET_SRC}.csv"', 'f"hypz3_REF_{DET_SRC}.csv"')
rep('print("HYPZ OK", flush=True)', 'print("HYPZ3 OK", flush=True)')
rep('print(f"[degraded] HYPZ INCOMPLETE or anchor miss:', 'print(f"[degraded] HYPZ3 INCOMPLETE or anchor miss:')

assert src.count("S0_OF[") == 7 and src.count("TRIP_A") >= 4 and src.count("TRIP_R") >= 3 and src.count("zx") >= 4, (
    src.count("S0_OF["), src.count("TRIP_A"), src.count("TRIP_R"), src.count("zx"))
open("hypz3.py", "w", encoding="utf-8", newline="\n").write(src)
print("written hypz3.py", src.count("\n"), "lines")
