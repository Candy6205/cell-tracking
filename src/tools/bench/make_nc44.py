"""fk44prop.py -> nc44.py: the node cost (hypx NC20XT) and the ILP candidate floor on the 71 44b6 movies.

WHY (2026-09-21). submit_nc's visible run: the two 6bba copies lose 0.6-0.8% of their nodes under NC20XT, as
hypx measured, but 44b6_0b24845f loses 6.9% (edges 20,124 -> 18,801); hypx benched 6bba only. The same movie
has 40% of its detections unselected by the shipped ILP, the regime the floor 0.10 -> 0.48 move read -0.0131
on 44b6 when it shipped (submit-f48; LB +0.001). Both questions on the 44b6 half, shipped composition:
  CR_NOCAP_DV150   control: the shipped solve (floor 0.48) under the shipped orphan patch
  NC20XT           node cost 0.20, the control's fork daughters + chains exempt (hypx), floor 0.48
  F30 / F10        the shipped solver at floor 0.30 / 0.10
Every arm runs the shipped patch (contrast rank, caps x100, diverge 1.5), is judged with its own solve's
probabilities, and goes through the harness's gate cross. Per movie the share of offered nodes that carry
an edge after the solve is logged (a GT-free key for a per-movie floor). Paired bootstrap of every arm
against the control on the ungated per-movie rows (adj_J, edge_J, score).
"""
src = open("fk44prop.py", encoding="utf-8").read()
hx = open("hypx.py", encoding="utf-8").read()
HYP_BLOCK = hx[hx.index("# ---- hypilp block ----"):hx.index("# ---- end hypilp block ----")] + "# ---- end hypilp block ----\n"


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-17b-fk44prop"', 'SCRIPT_VERSION = "2026-09-21a-nc44"')
rep('STEAL_ARMS = [("CR_NOCAP_DV150", dict(rank="contrast", cap=100.0, diverge=1.5))]\n',
    'STEAL_ARMS = [("CR_NOCAP_DV150", dict(rank="contrast", cap=100.0, diverge=1.5)),\n'
    '              ("NC20XT", dict(rank="contrast", cap=100.0, diverge=1.5, cost=0.20)),\n'
    '              ("F30", dict(rank="contrast", cap=100.0, diverge=1.5, floor=0.30)),\n'
    '              ("F10", dict(rank="contrast", cap=100.0, diverge=1.5, floor=0.10))]\n'
    'DUP_STAT = defaultdict(int)\n'
    'FDET_F4HYP = {}\n'
    'SELFRAC = defaultdict(dict)      # arm -> movie -> share of offered nodes carrying an edge after the solve\n'
    'NC_SOLVE_S = defaultdict(int)\n')
rep("def solve_one(dets, T, seeds, feats, model, cols, disapp=None, floor=None):\n",
    HYP_BLOCK + "\n\ndef solve_one(dets, T, seeds, feats, model, cols, disapp=None, floor=None):\n")

rep("        FP_RANK, FP_CAP, DIV_DIVERGE_UM = cfg[\"rank\"], cfg[\"cap\"], cfg[\"diverge\"]\n"
    "        patch_final(nodes0, edges0, pool, outa)\n",
    "        FP_RANK, FP_CAP, DIV_DIVERGE_UM = cfg[\"rank\"], cfg[\"cap\"], cfg[\"diverge\"]\n"
    "        _ts = time.time()\n"
    "        if cfg.get(\"cost\"):\n"
    "            DUP_STAT.clear()\n"
    "            n_x, e_x, p_x, _ax = solve_hyp(dets, T, seeds, feats, MODELS_FOLD[k], COLS, None, cfg[\"cost\"],\n"
    "                                           disapp=ILP_DISAPP, floor=PROD_FLOOR, exempt=EX_T)\n"
    "        elif cfg.get(\"floor\") is not None:\n"
    "            n_x, e_x, p_x, _ax = solve_one(dets, T, seeds, feats, MODELS_FOLD[k], COLS,\n"
    "                                           disapp=ILP_DISAPP, floor=cfg[\"floor\"])\n"
    "        else:\n"
    "            n_x, e_x, p_x, _ax = nodes0, edges0, pool, aux0\n"
    "        NC_SOLVE_S[arm] += int(time.time() - _ts)\n"
    "        _sx, _b2x = _aux_extra(seeds, _ax[0])\n"
    "        aux_x = (_ax[0], _ax[1], _sx, _b2x)\n"
    "        _used = set()\n"
    "        for a_, b_ in e_x:\n"
    "            _used.add(a_)\n"
    "            _used.add(b_)\n"
    "        SELFRAC[arm][name] = len(_used) / max(1, len(n_x))\n"
    "        patch_final(n_x, e_x, p_x, outa)\n"
    "        if arm == STEAL_ARMS[0][0]:\n"
    "            EX_D, EX_T, _n_dau, _n_trk = _hyp_exempt(edges0, seed_of_node, PRE_STATE[\"final_edges\"])\n")
rep("    cells = []\n    for arm, cfg in STEAL_ARMS:\n",
    "    cells = []\n    EX_D, EX_T = set(), set()      # filled after the control arm's patch\n    for arm, cfg in STEAL_ARMS:\n")
rep("        row_a, ds_a, fr_a = _judge(name, k, outa, seeds, aux)\n"
    "        IMG_CTX.clear()\n",
    "        row_a, ds_a, fr_a = _judge(name, k, outa, seeds, aux_x)\n"
    "        IMG_CTX.clear()\n")
rep("        cells.append(f\"{arm} +{n_st} {row_a['division_tp']}/{row_a['division_fp']}/\"\n",
    "        cells.append(f\"{arm} sel {SELFRAC[arm][name]:.3f} n {row_a['num_pred_nodes']} +{n_st} {row_a['division_tp']}/{row_a['division_fp']}/\"\n")

rep('print(f"\\nFK44PROP DONE in',
    'print("\\n=== NC44: every arm against the control (CR_NOCAP_DV150), UNGATED per-movie rows, paired bootstrap 2,000 draws ===", flush=True)\n'
    '_ctl = STEAL_ARMS[0][0]\n'
    '_names = [n_ for n_ in ARM_PER[_ctl] if all(n_ in ARM_PER[a_] for a_, _c in STEAL_ARMS)]\n'
    '_rng = np.random.RandomState(20260921)\n'
    '_draws = [_rng.randint(0, len(_names), len(_names)) for _ in range(2000)]\n'
    'def _agg(arm, idx):\n'
    '    return _with_ratio([ARM_PER[arm][_names[i_]] for i_ in idx])\n'
    '_full = list(range(len(_names)))\n'
    '_c0 = _agg(_ctl, _full)\n'
    'print(f"  {len(_names)} movies; control adj_J {_c0[\'adj_edge_jaccard\']:.4f} edge_J {_c0.get(\'edge_jaccard\', float(\'nan\')):.4f} node_r {_c0[\'total_node_ratio\']:+.4f} "\n'
    '      f"div {_c0[\'division_tp\']}/{_c0[\'division_fp\']}/{_c0[\'division_fn\']} score {_c0[\'score\']:.4f}; solve s {dict(NC_SOLVE_S)}", flush=True)\n'
    'for a_, _c in STEAL_ARMS[1:]:\n'
    '    _a0 = _agg(a_, _full)\n'
    '    print(f"  {a_:<8} adj_J {_a0[\'adj_edge_jaccard\']:.4f} edge_J {_a0.get(\'edge_jaccard\', float(\'nan\')):.4f} node_r {_a0[\'total_node_ratio\']:+.4f} "\n'
    '          f"div {_a0[\'division_tp\']}/{_a0[\'division_fp\']}/{_a0[\'division_fn\']} score {_a0[\'score\']:.4f}", flush=True)\n'
    '    for key_ in ("adj_edge_jaccard", "edge_jaccard", "score"):\n'
    '        ds_ = np.array([_agg(a_, ix).get(key_, float(\'nan\')) - _agg(_ctl, ix).get(key_, float(\'nan\')) for ix in _draws])\n'
    '        lo_, hi_ = (np.nanpercentile(ds_, [5, 95]) if np.isfinite(ds_).any() else (float(\'nan\'), float(\'nan\')))\n'
    '        print(f"    {a_:<8} {key_:<18} {_a0.get(key_, float(\'nan\')) - _c0.get(key_, float(\'nan\')):+.4f}  [{lo_:+.4f}, {hi_:+.4f}]  {\'YES\' if lo_ > 0 or hi_ < 0 else \'no\'}", flush=True)\n'
    '    up_ = sum(1 for n_ in _names if _adj_of(ARM_PER[a_][n_]) > _adj_of(ARM_PER[_ctl][n_]))\n'
    '    dn_ = sum(1 for n_ in _names if _adj_of(ARM_PER[a_][n_]) < _adj_of(ARM_PER[_ctl][n_]))\n'
    '    print(f"    {a_:<8} movies up {up_} / down {dn_}", flush=True)\n'
    'with open("/kaggle/working/nc44_selfrac.csv", "w") as _fh:\n'
    '    _fh.write("movie," + ",".join(a_ for a_, _c in STEAL_ARMS) + "\\n")\n'
    '    for n_ in _names:\n'
    '        _fh.write(n_ + "," + ",".join(f"{SELFRAC[a_].get(n_, float(\'nan\')):.4f}" for a_, _c in STEAL_ARMS) + "\\n")\n'
    'print(f"\\nFK44PROP DONE in')
rep('f"fk44prop_arm_{a_}.csv"', 'f"nc44_arm_{a_}.csv"')
rep('f"fk44prop_{DET_SRC}.csv"', 'f"nc44_{DET_SRC}.csv"')
rep('print("FK44PROP OK", flush=True)', 'print("NC44 OK", flush=True)')

open("nc44.py", "w", encoding="utf-8", newline="\n").write(src)
print("written nc44.py", src.count("\n"), "lines")
