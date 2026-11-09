"""forkprop_sz.py -> forkprop_z.py: the 6bba fork harness with the DIVISION-VARIABLE ILP as the solve behind the shipped
patch, so the image gate can be refitted on the z composition's own fork population.

WHY (2026-09-25). hypz / hypz3: Z10 (a division variable per pooled triple, cost 0.55 - 0.10 per logit of scorer B above
-9, floor 0.10, its two edges admitted below the 0.48 floor as division-only candidates) reads +0.002 over the shipped
composition after a 20-column gate refitted on its forks. The submission's gate is the 31 / 36-column image gate, fitted
by forkprop_sz + fk44prop_sz on the shipped composition's forks; the z composition adds ~2,000 forks per 128 movies
(1.7% of them on annotated divisions, 34% true among the judged), a population the shipped gate never saw. The user
(2026-09-25): "不缺就搭吧".

WHAT. forkprop_sz.py verbatim (the size columns stay), with:
  * hypz's z machinery ported (z_cost, load_triples + the scorer-B refit out of fold by divtrip's folds, _hyp_exempt,
    solve_hyp with trip, the Z_SEL bookkeeping); S0 fixed at -9.0 (what ships; hypz's 0.9491 run).
  * pass 1, per movie, after REF's solve + patch: the exemption set EX_T from REF (fork daughters + downstream chains, the
    shipped nc composition's), then TWO z solves, pickled beside REF's: Z = node cost 0.35 + EX_T + Z10's triples (hypz's
    Z10); ZX = a first solve at node cost 0 with the triples, its z-fork daughters + chains added to EX_T, then the costed
    solve (hypz4's Z10X0; whichever hypz4 favours ships, the gate exists for both).
  * pass 2: three arms, all the shipped patch (contrast-ranked, caps x100, DIV_DIVERGE_UM 1.5): CR_NOCAP_DV150 on REF's
    solve (forkprop_sz's cell, the anchor of this run), ZCR_NOCAP_DV150 on Z, ZXCR_NOCAP_DV150 on ZX. Fork tables
    forkssz_<arm>.csv (20 + 8 + 16 columns) for fk44prop_z's gate fit; the cross and the bootstrap as before.
Attach: forkprop_sz's sources + kernel faustine649/biohub-divtrip (the pooled triples). CPU: forkprop_sz's pass 1 was
106 min and its pass 2 411 min for 6 arms; here pass 1 + 3 SCIP solves per movie (~40 s) and 3 arms: ~6.5 h. Budget 10 h.
"""
src = open("forkprop_sz.py", encoding="utf-8").read()
hz = open("hypz.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


def hslice(a, b):
    i = hz.index(a)
    return hz[i:hz.index(b, i)]


rep('SCRIPT_VERSION = "2026-09-24a-forkprop_sz"', 'SCRIPT_VERSION = "2026-09-25a-forkprop_z"')

# ---- the z machinery, ported from hypz.py -------------------------------------------------------------------------
Z_BLOCK = (
    "\n# ---- forkprop_z: the division-variable ILP (hypz.py's machinery, ported verbatim) --------------------------------\n"
    "from pyscipopt import Model, quicksum            # noqa: E402  (support-pack wheel)\n"
    "DUP_STAT = defaultdict(int)\n"
    "HYP_SEL = []\n"
    "Z_SEL = []            # per solve: (t, p, a, b, w, s, label) of the selected z triples\n"
    "Z_STAT = {a_: defaultdict(int) for a_ in ('Z', 'ZX')}\n"
    "Z_S0 = -9.0                    # logit of scorer B at which a triple costs exactly DIV_W (hypz's Z10)\n"
    "Z_CFG = dict(cost=0.35, lam=0.10, wmin=0.10)\n"
    "HYP_ARMS = [('Z10', dict(lam=0.10, wmin=0.10))]   # load_triples prints the discount counts per entry\n"
    + hslice("def z_cost(", "TRIP_OF = load_triples()")
    + "TRIP_OF = load_triples()\n"
    "print(f\"  triples loaded for {len(TRIP_OF)} movies; {sum(1 for n_ in SIX if n_ in TRIP_OF)} of the {len(SIX)} 6bba \"\n"
    "      f\"movies covered\", flush=True)\n"
    "assert sum(1 for n_ in SIX if n_ in TRIP_OF) >= len(SIX) - 2, 'the triple table does not cover the 6bba movies'\n"
    "\n\n"
    + hslice("def _hyp_exempt(", "def solve_hyp(")
    + hslice("def solve_hyp(", "# ---- end hypilp block ----")
    + "\n\n"
    "def z_solves(name, dets, T, seeds, feats, model, edges0, seed_of_node, final_edges):\n"
    "    \"\"\"the two z compositions of one movie: Z (node cost 0.35, REF's exemption, Z10's triples) and ZX (a first solve\n"
    "    at cost 0, its z-fork daughters + chains added to REF's exemption, then the costed solve). Returns {tag: (nodes,\n"
    "    edges, pool, pmap, seed_of_node)} in solve_one's shapes; Z_STAT[tag] carries the counts.\"\"\"\n"
    "    ex_t = _hyp_exempt(edges0, seed_of_node, final_edges)[1]\n"
    "    trip = [(t_, p_, a_, b_, z_cost(s_, Z_CFG), s_, l_) for (t_, p_, a_, b_, s_, l_) in TRIP_OF.get(name, [])]\n"
    "    out = {}\n"
    "    for tag in ('Z', 'ZX'):\n"
    "        DUP_STAT.clear()\n"
    "        if tag == 'Z':\n"
    "            n_, e_, p_, aux_ = solve_hyp(dets, T, seeds, feats, model, COLS, None, Z_CFG['cost'], disapp=ILP_DISAPP,\n"
    "                                         floor=PROD_FLOOR, exempt=ex_t, trip=trip)\n"
    "        else:\n"
    "            n0, e0, p0, aux0 = solve_hyp(dets, T, seeds, feats, model, COLS, None, 0.0, disapp=ILP_DISAPP,\n"
    "                                         floor=PROD_FLOOR, exempt=None, trip=trip)\n"
    "            ex_z = _hyp_exempt(e0, aux0[1], e0)[1]\n"
    "            Z_STAT[tag]['first_pass_z'] += DUP_STAT['z_selected']\n"
    "            Z_STAT[tag]['first_pass_z_tp'] += DUP_STAT['z_sel_tp']\n"
    "            DUP_STAT.clear()\n"
    "            n_, e_, p_, aux_ = solve_hyp(dets, T, seeds, feats, model, COLS, None, Z_CFG['cost'], disapp=ILP_DISAPP,\n"
    "                                         floor=PROD_FLOOR, exempt=(set(ex_t) | set(ex_z)), trip=trip)\n"
    "            Z_STAT[tag]['added_exempt'] += len(ex_z)\n"
    "            del n0, e0, p0, aux0\n"
    "        for k_, v_ in DUP_STAT.items():\n"
    "            Z_STAT[tag][k_] += v_\n"
    "        Z_STAT[tag]['movies'] += 1\n"
    "        out[tag] = (n_, e_, p_, aux_[0], aux_[1])\n"
    "    return out\n"
    "\n\n")
rep("for name in SIX:\n    if time.time() - t_all > BUDGET_S:\n", Z_BLOCK + "for name in SIX:\n    if time.time() - t_all > BUDGET_S:\n")

# ---- pass 1: the z solves after REF's patch, pickled beside REF's solve ---------------------------------------------
rep("    with open(BR_DIR / f\"{name}.pkl\", \"wb\") as _fh:\n"
    "        pickle.dump((nodes0, edges0, pool[0], pool[1], seeds, pmap, seed_of_node), _fh,\n"
    "                    protocol=pickle.HIGHEST_PROTOCOL)\n",
    "    with open(BR_DIR / f\"{name}.pkl\", \"wb\") as _fh:\n"
    "        pickle.dump((nodes0, edges0, pool[0], pool[1], seeds, pmap, seed_of_node), _fh,\n"
    "                    protocol=pickle.HIGHEST_PROTOCOL)\n"
    "    _tz = time.time()\n"
    "    _zs = z_solves(name, dets, T, seeds, feats, MODELS_FOLD[k], edges0, seed_of_node, PRE_STATE[\"final_edges\"])\n"
    "    for _tag, (_nz, _ez, _pz, _pmz, _soz) in _zs.items():\n"
    "        with open(BR_DIR / f\"{name}_{_tag}.pkl\", \"wb\") as _fh:\n"
    "            pickle.dump((_nz, _ez, _pz[0], _pz[1], seeds, _pmz, _soz), _fh, protocol=pickle.HIGHEST_PROTOCOL)\n"
    "    cells.append(f\"z solves {time.time() - _tz:.0f}s: Z edges {len(_zs['Z'][1])} ZX edges {len(_zs['ZX'][1])} (REF {len(edges0)})\")\n"
    "    del _zs\n")

# ---- pass 2: three arms, each replayed on its own solve ----------------------------------------------------------------
rep('DA_ARMS = [("CR", dict(diverge=2.25, admit=False, rank="contrast", cap=1.0)),\n'
    '           ("CR_CAP2", dict(diverge=2.25, admit=False, rank="contrast", cap=2.0)),\n'
    '           ("CR_CAP4", dict(diverge=2.25, admit=False, rank="contrast", cap=4.0)),\n'
    '           ("CR_CAP4_DV150", dict(diverge=1.5, admit=False, rank="contrast", cap=4.0)),\n'
    '           ("DIST_CAP4_DV150", dict(diverge=1.5, admit=False, rank="dist", cap=4.0)),\n'
    '           ("CR_NOCAP_DV150", dict(diverge=1.5, admit=False, rank="contrast", cap=100.0))]\n',
    'DA_ARMS = [("CR_NOCAP_DV150", dict(diverge=1.5, admit=False, rank="contrast", cap=100.0, src="REF")),\n'
    '           ("ZCR_NOCAP_DV150", dict(diverge=1.5, admit=False, rank="contrast", cap=100.0, src="Z")),\n'
    '           ("ZXCR_NOCAP_DV150", dict(diverge=1.5, admit=False, rank="contrast", cap=100.0, src="ZX"))]\n')
rep("    with open(BR_DIR / f\"{name}.pkl\", \"rb\") as _fh:\n"
    "        nodes0, edges0, cand_, cand_sc_, seeds, pmap, seed_of_node = pickle.load(_fh)\n"
    "    cand_trees_ = [cKDTree(c_ * SCALE) if len(c_) else None for c_ in cand_]\n"
    "    pool = (cand_, cand_sc_, cand_trees_)\n"
    "    k = FOLD_OF[name]\n"
    "    scsort, best2 = _aux_extra(seeds, pmap)\n"
    "    aux = (pmap, seed_of_node, scsort, best2)\n",
    "    SOLVES = {}\n"
    "    for _src in (\"REF\", \"Z\", \"ZX\"):\n"
    "        with open(BR_DIR / (f\"{name}.pkl\" if _src == \"REF\" else f\"{name}_{_src}.pkl\"), \"rb\") as _fh:\n"
    "            _n0, _e0, _c, _cs, seeds, _pm, _so = pickle.load(_fh)\n"
    "        _ct = [cKDTree(c_ * SCALE) if len(c_) else None for c_ in _c]\n"
    "        _ss, _b2 = _aux_extra(seeds, _pm)\n"
    "        SOLVES[_src] = (_n0, _e0, (_c, _cs, _ct), _pm, _so, (_pm, _so, _ss, _b2))\n"
    "    nodes0, edges0, pool, pmap, seed_of_node, aux = SOLVES[\"REF\"]\n"
    "    cand_, cand_sc_, cand_trees_ = pool\n"
    "    k = FOLD_OF[name]\n")
rep("        DA_CTX.clear()\n"
    "        DA_CTX.update(seeds=seeds, seed_of_node=seed_of_node, added=[])\n",
    "        nodes0, edges0, pool, pmap, seed_of_node, aux = SOLVES[cfg.get(\"src\", \"REF\")]\n"
    "        DA_CTX.clear()\n"
    "        DA_CTX.update(seeds=seeds, seed_of_node=seed_of_node, added=[])\n")
rep("    del nodes0, edges0, pool, cand_, cand_sc_, cand_trees_, seeds, pmap, seed_of_node, aux\n",
    "    del nodes0, edges0, pool, cand_, cand_sc_, cand_trees_, seeds, pmap, seed_of_node, aux, SOLVES\n")

# ---- the report: the z solves' counts -----------------------------------------------------------------------------
rep('print("\\n=== THE PATCH PER ARM (orphan pass + admit pass), summed over the movies replayed ===", flush=True)\n',
    'print("\\n=== THE z SOLVES, summed over the movies (Z: cost 0.35 + REF\'s exemption; ZX: first pass at cost 0, its z daughters "\n'
    '      "exempt in the costed pass). TP by divtrip\'s 5 um label ===", flush=True)\n'
    'for _tag in ("Z", "ZX"):\n'
    '    d_ = Z_STAT[_tag]\n'
    '    print(f"  {_tag:<3} movies {d_[\'movies\']} | triples offered {d_[\'z_offered\']:,}, variables {d_[\'z_vars\']:,} (TP {d_[\'z_vars_tp\']}), "\n'
    '          f"edges added below the floor {d_[\'z_edges_added\']:,} | z selected {d_[\'z_selected\']:,} (TP {d_[\'z_sel_tp\']}) | "\n'
    '          f"nodes selected {d_[\'nodes_selected\']:,} of {d_[\'nodes_offered\']:,}, edges {d_[\'edges_selected\']:,}, ILP divisions "\n'
    '          f"{d_[\'ilp_divisions\']:,}, exempt {d_[\'exempt\']:,}, solve {d_[\'solve_s\']:,}s"\n'
    '          + (f" | first pass z {d_[\'first_pass_z\']:,} (TP {d_[\'first_pass_z_tp\']}), exempt nodes added {d_[\'added_exempt\']:,}"\n'
    '             if _tag == "ZX" else ""), flush=True)\n'
    'print("\\n=== THE PATCH PER ARM (orphan pass + admit pass), summed over the movies replayed ===", flush=True)\n')
rep('print("FORKPROPSZ OK", flush=True)', 'print("FORKPROPZ OK", flush=True)')
rep('print(f"[degraded] BRIDGERANK INCOMPLETE or anchor miss:', 'print(f"[degraded] FORKPROPZ INCOMPLETE or anchor miss:')

assert src.count("def solve_hyp(") == 1 and src.count("def _hyp_exempt(") == 1 and src.count("z_solves(") == 2 \
    and src.count("SOLVES[") == 3 and src.count("forkssz_") >= 2, (src.count("SOLVES["), src.count("forkssz_"))
open("forkprop_z.py", "w", encoding="utf-8", newline="\n").write(src)
print("written forkprop_z.py", src.count("\n"), "lines")
