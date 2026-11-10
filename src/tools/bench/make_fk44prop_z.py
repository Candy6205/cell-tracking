"""fk44prop_sz.py -> fk44prop_z.py: the 44b6 fork harness with the division-variable ILP behind the shipped patch, and
the submission gates fitted on the z compositions' forks in both domains.

WHY (2026-09-25). forkprop_z gives the 6bba fork tables of the z compositions (ZCR_NOCAP_DV150, ZXCR_NOCAP_DV150); the
image gate that ships is fitted on the 6bba tables + the 44b6 forks of the same composition (fk44prop's shape). This is
that step for the z compositions.

WHAT. fk44prop_sz.py verbatim, with: hypz's z machinery (as in forkprop_z; TRIP_OF covers the 71 44b6 movies too);
per movie, after REF's solve + patch, the two z solves (Z: cost 0.35 + REF's exemption; ZX: first pass at cost 0, its
z daughters exempt in the costed pass); STEAL_ARMS = CR_NOCAP_DV150 on REF's solve (fk44prop_sz's cell), ZCR_NOCAP_DV150
on Z, ZXCR_NOCAP_DV150 on ZX -- each the shipped patch replayed on its own solve, judged with image features; the 6bba
tables read from biohub-forkprop-z (forkssz_<arm>.csv); one gate pickle per arm: fork_gate_sz.pkl (the REF arm, must
reproduce fk44prop_sz's), fork_gate_z.pkl (Z), fork_gate_zx.pkl (ZX), each carrying the z provenance (lam, wmin, S0,
cost, top-k, the scorer's origin) for the submission's asserts.
Attach: fk44prop_sz's sources with biohub-forkprop-sz replaced by biohub-forkprop-z, + biohub-divtrip. CPU: fk44prop_sz
ran 263 min with one arm; three arms + 3 SCIP solves per movie: ~6 h. Budget 10 h.
"""
src = open("fk44prop_sz.py", encoding="utf-8").read()
hz = open("hypz.py", encoding="utf-8").read()
fz = open("forkprop_z.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


def hslice(a, b):
    i = hz.index(a)
    return hz[i:hz.index(b, i)]


rep('SCRIPT_VERSION = "2026-09-24a-fk44prop_sz"', 'SCRIPT_VERSION = "2026-09-25a-fk44prop_z"')

# ---- the z machinery (forkprop_z's block, with the coverage assert on BENCH) --------------------------------------
Z_BLOCK = fz[fz.index("# ---- forkprop_z: the division-variable ILP"):fz.index("for name in SIX:\n    if time.time() - t_all > BUDGET_S:\n")]
Z_BLOCK = "\n" + Z_BLOCK.replace(
    "print(f\"  triples loaded for {len(TRIP_OF)} movies; {sum(1 for n_ in SIX if n_ in TRIP_OF)} of the {len(SIX)} 6bba \"\n"
    "      f\"movies covered\", flush=True)\n"
    "assert sum(1 for n_ in SIX if n_ in TRIP_OF) >= len(SIX) - 2, 'the triple table does not cover the 6bba movies'\n",
    "print(f\"  triples loaded for {len(TRIP_OF)} movies; {sum(1 for n_ in BENCH if n_ in TRIP_OF)} of the {len(BENCH)} 44b6 \"\n"
    "      f\"movies covered\", flush=True)\n"
    "assert sum(1 for n_ in BENCH if n_ in TRIP_OF) >= len(BENCH) - 2, 'the triple table does not cover the 44b6 movies'\n")
assert "BENCH if n_ in TRIP_OF" in Z_BLOCK and "def z_solves(" in Z_BLOCK and "def solve_hyp(" in Z_BLOCK
rep("for name in BENCH:\n    if time.time() - t_all > BUDGET_S:\n", Z_BLOCK + "for name in BENCH:\n    if time.time() - t_all > BUDGET_S:\n")

# ---- the arms: one per solve -------------------------------------------------------------------------------------
rep('STEAL_ARMS = [("CR_NOCAP_DV150", dict(rank="contrast", cap=100.0, diverge=1.5))]\n',
    'STEAL_ARMS = [("CR_NOCAP_DV150", dict(rank="contrast", cap=100.0, diverge=1.5, src="REF")),\n'
    '              ("ZCR_NOCAP_DV150", dict(rank="contrast", cap=100.0, diverge=1.5, src="Z")),\n'
    '              ("ZXCR_NOCAP_DV150", dict(rank="contrast", cap=100.0, diverge=1.5, src="ZX"))]\n'
    'GATE_PKL = {"CR_NOCAP_DV150": "fork_gate_sz.pkl", "ZCR_NOCAP_DV150": "fork_gate_z.pkl", "ZXCR_NOCAP_DV150": "fork_gate_zx.pkl"}\n')
# pass 1: the z solves after REF's patch
rep("    patch_final(nodes0, edges0, pool, outp)\n"
    "    for k_, v_ in PATCH_CAPS.items():\n"
    "        CAPS_BY_ARM.setdefault(\"REF\", defaultdict(int))[k_] += v_\n",
    "    patch_final(nodes0, edges0, pool, outp)\n"
    "    for k_, v_ in PATCH_CAPS.items():\n"
    "        CAPS_BY_ARM.setdefault(\"REF\", defaultdict(int))[k_] += v_\n"
    "    _tz = time.time()\n"
    "    _zs = z_solves(name, dets, T, seeds, feats, MODELS_FOLD[k], edges0, seed_of_node, PRE_STATE[\"final_edges\"])\n"
    "    SOLVES = {\"REF\": (nodes0, edges0, pool, aux)}\n"
    "    for _tag, (_nz, _ez, _pz, _pmz, _soz) in _zs.items():\n"
    "        _ss, _b2 = _aux_extra(seeds, _pmz)\n"
    "        SOLVES[_tag] = (_nz, _ez, _pz, (_pmz, _soz, _ss, _b2))\n"
    "    _z_line = f\"z solves {time.time() - _tz:.0f}s: Z edges {len(_zs['Z'][1])} ZX edges {len(_zs['ZX'][1])} (REF {len(edges0)})\"\n"
    "    del _zs\n")
rep("    cells = []\n    for arm, cfg in STEAL_ARMS:\n",
    "    cells = [_z_line]\n    for arm, cfg in STEAL_ARMS:\n        nodes0, edges0, pool, aux = SOLVES[cfg.get(\"src\", \"REF\")]\n")
rep("        cells.append(f\"{arm} +{n_st} {row_a['division_tp']}/{row_a['division_fp']}/\"\n"
    "                     f\"{row_a['division_fn']} {_adj_of(row_a):.4f}\")\n"
    "        del ds_a\n",
    "        cells.append(f\"{arm} +{n_st} {row_a['division_tp']}/{row_a['division_fp']}/\"\n"
    "                     f\"{row_a['division_fn']} {_adj_of(row_a):.4f}\")\n"
    "        del ds_a\n"
    "    nodes0, edges0, pool, aux = SOLVES[\"REF\"]\n"
    "    del SOLVES\n")
# the 6bba tables come from biohub-forkprop-z
rep('assert hit, f"attach the kernel faustine649/biohub-forkprop-sz ({fn})"', 'assert hit, f"attach the kernel faustine649/biohub-forkprop-z ({fn})"')
rep('_SIX_CSV = {"REF": "forkssz_dw55.csv", **{a_: f"forkssz_{a_}.csv" for a_, _c in STEAL_ARMS}}   # from biohub-forkprop-sz',
    '_SIX_CSV = {"REF": "forkssz_dw55.csv", **{a_: f"forkssz_{a_}.csv" for a_, _c in STEAL_ARMS}}   # from biohub-forkprop-z')
# one gate pickle per arm, with the z provenance
rep('            "ilp": {"disapp": ILP_DISAPP, "floor": PROD_FLOOR}}\n'
    '    with open("/kaggle/working/fork_gate_sz.pkl", "wb") as fh:\n'
    '        pickle.dump(blob, fh)\n'
    '    print(f"\\n  fork_gate_sz.pkl written: n_train {n_train}; p quantiles over all {len(p_every):,} forks of the arm "\n',
    '            "ilp": {"disapp": ILP_DISAPP, "floor": PROD_FLOOR},\n'
    '            "z": (None if cfg_.get("src", "REF") == "REF" else\n'
    '                  {"src": cfg_["src"], "cost": Z_CFG["cost"], "lam": Z_CFG["lam"], "wmin": Z_CFG["wmin"], "s0": Z_S0,\n'
    '                   "topk": 20, "scorer": "divtrip scorer B (geometry + image), refit out of fold here; divscore.pkl ships",\n'
    '                   "two_pass": cfg_["src"] == "ZX"})}\n'
    '    with open(f"/kaggle/working/{GATE_PKL[a_]}", "wb") as fh:\n'
    '        pickle.dump(blob, fh)\n'
    '    print(f"\\n  {GATE_PKL[a_]} written: n_train {n_train}; p quantiles over all {len(p_every):,} forks of the arm "\n')
# a second gate per z arm WITHOUT the five raw-image half-max size columns (submit_nc35's 31-column set): the size
# gate read LB 0.954 against nc35's 0.956 (submission 56530848), so the z submission ships on the 31-column base by
# default and the 36-column gate stays as the alternative. Out-of-fold AUC by fold for both column sets, for the record.
rep("""    print(f"\\n  {GATE_PKL[a_]} written: n_train {n_train}; p quantiles over all {len(p_every):,} forks of the arm "
          f"in both domains: " + " ".join(f"{int(f * 100)}%={v:.4f}" for f, v in Q.items()), flush=True)
""", """    print(f"\\n  {GATE_PKL[a_]} written: n_train {n_train}; p quantiles over all {len(p_every):,} forks of the arm "
          f"in both domains: " + " ".join(f"{int(f * 100)}%={v:.4f}" for f, v in Q.items()), flush=True)
    if blob["z"] is not None:
        J31 = J20 + [FEAT_NAMES.index(f_) for f_ in FEAT_IMG if not (f_.startswith("r_hm") or f_.startswith("rhm"))]
        assert len(J31) == 31, len(J31)
        m31 = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, min_samples_leaf=10, random_state=0)
        m31.fit(Xall[:, J31], yall)
        p31 = m31.predict_proba(X_every[:, J31])[:, 1]
        Q31 = {f: float(np.quantile(p31, f)) for f in (0.5, 0.6, 0.7, 0.8, 0.9)}
        fold_all = np.concatenate([c6.fold.to_numpy(int), np.array([r_[1] for r_ in fk_], int)[cnt]])
        auc_ = {}
        for nm_, cols_ in (("36", JIMG_COLS), ("31", J31)):
            po_ = np.full(len(yall), np.nan)
            for f_ in sorted(set(fold_all.tolist())):
                tr_, te_ = fold_all != f_, fold_all == f_
                if yall[tr_].sum() == 0 or te_.sum() == 0:
                    continue
                mo_ = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, min_samples_leaf=10, random_state=0)
                mo_.fit(Xall[tr_][:, cols_], yall[tr_])
                po_[te_] = mo_.predict_proba(Xall[te_][:, cols_])[:, 1]
            ok_ = np.isfinite(po_)
            auc_[nm_] = float(_auc(yall[ok_], po_[ok_])) if ok_.sum() and yall[ok_].min() != yall[ok_].max() else float("nan")
        blob31 = dict(blob)
        blob31["model"] = m31
        blob31["feature_names"] = [FEAT_NAMES[j] for j in J31]
        blob31["trained_on"] = blob["trained_on"].replace("20 gate + 16 image features (5 raw-image half-max size)", "20 gate + 11 image features (no size columns)")
        blob31["p_quantiles_all_forks"] = Q31
        blob31["auc_oof_by_fold"] = auc_
        _n31 = GATE_PKL[a_].replace(".pkl", "31.pkl")
        with open(f"/kaggle/working/{_n31}", "wb") as fh:
            pickle.dump(blob31, fh)
        print(f"  {_n31} written (the same forks, 31 columns = submit_nc35's set); p quantiles "
              + " ".join(f"{int(f * 100)}%={v:.4f}" for f, v in Q31.items())
              + f"; out-of-fold AUC by fold on the pooled counted forks: 36 columns {auc_['36']:.4f}, 31 columns {auc_['31']:.4f}", flush=True)
""")
rep('print("FK44PROPSZ OK", flush=True)', 'print("FK44PROPZ OK", flush=True)')
# the z solves' counts in the report
rep('print("    reading: the arm ships only if its gated cell does not lose on 44b6 (interval not below zero on score "\n'
    '      "AND adj_J); a 44b6 loss the size of the 6bba gain cancels it on a half-and-half test set.", flush=True)\n',
    'print("    reading: the arm ships only if its gated cell does not lose on 44b6 (interval not below zero on score "\n'
    '      "AND adj_J); a 44b6 loss the size of the 6bba gain cancels it on a half-and-half test set.", flush=True)\n'
    'print("\\n=== THE z SOLVES on 44b6, summed over the movies (Z: cost 0.35 + REF\'s exemption; ZX: first pass at cost 0, its z "\n'
    '      "daughters exempt in the costed pass). TP by divtrip\'s 5 um label ===", flush=True)\n'
    'for _tag in ("Z", "ZX"):\n'
    '    d_ = Z_STAT[_tag]\n'
    '    print(f"  {_tag:<3} movies {d_[\'movies\']} | triples offered {d_[\'z_offered\']:,}, variables {d_[\'z_vars\']:,} (TP {d_[\'z_vars_tp\']}), "\n'
    '          f"edges added below the floor {d_[\'z_edges_added\']:,} | z selected {d_[\'z_selected\']:,} (TP {d_[\'z_sel_tp\']}) | "\n'
    '          f"nodes selected {d_[\'nodes_selected\']:,} of {d_[\'nodes_offered\']:,}, edges {d_[\'edges_selected\']:,}, ILP divisions "\n'
    '          f"{d_[\'ilp_divisions\']:,}, exempt {d_[\'exempt\']:,}, solve {d_[\'solve_s\']:,}s"\n'
    '          + (f" | first pass z {d_[\'first_pass_z\']:,} (TP {d_[\'first_pass_z_tp\']}), exempt nodes added {d_[\'added_exempt\']:,}"\n'
    '             if _tag == "ZX" else ""), flush=True)\n')

assert src.count("def solve_hyp(") == 1 and src.count("z_solves(") == 2 and src.count("SOLVES[") == 3 \
    and src.count("GATE_PKL[a_]") == 3 and src.count("blob31") >= 5 and src.count('PRE_STATE["final_edges"] = list(edges)') == 1, (
        src.count("SOLVES["), src.count("GATE_PKL[a_]"))
open("fk44prop_z.py", "w", encoding="utf-8", newline="\n").write(src)
print("written fk44prop_z.py", src.count("\n"), "lines")
