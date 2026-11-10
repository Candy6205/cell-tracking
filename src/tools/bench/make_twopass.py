"""forkprop_z.py -> twopass.py: the untried two-pass design -- multi-frame columns from the SOLVED tracks, a second solve.

WHY (2026-09-26). The record (2026-08-25 "multi-frame linking, three attempts"; 2026-08-28 accdiag): every multi-frame
feature so far came from the greedy backward chain (pass1_tracks), and the acceleration columns lost (-0.0019 adj_J,
permutation importance 0) because the chain's extra positions double the predictor's noise variance. The design left
untried: solve once, build the multi-frame columns from the ILP's globally solved tracks, retrain the ranker with them,
solve again. The user (2026-09-26): "去试试吧".

WHAT. forkprop_z.py's harness (features + head join on all 199 train movies, stage A rankers, pass 1 solves on the 128
6bba movies with the shipped patch, pass 2 arm replays, the gate cross with paired bootstraps) with the z machinery
unused and, in its place:
  * label_rows also returns the (t, i, j) keys of its labelled rows;
  * track_cols(): for every candidate pair of a movie, from pass 1's solved graph (REF's own solve, before the patch):
    has_tr (0 / 1 / 2 solved backward steps behind the source), md_tr (distance of the target from the source's
    solved-velocity prediction), dmd_tr (d - md_tr), cos_tr (cosine between the candidate step and the solved velocity),
    len_back_tr (solved chain length behind the source, capped at 12), md_acc_tr (the same distance with the solved
    acceleration added: the record predicts this one worse), cos_fwd_tr (cosine between the candidate step and the
    target's solved onward step);
  * after pass 1: two fold rankers per fold trained on the 6bba other-fold movies' F4HP labelled rows -- R6 (the
    shipped 21 columns; the control, so the only difference is the columns) and TP (21 + the 7 track columns); their
    held-out AUCs, the track columns' single-column AUCs;
  * pass 2: per movie the track columns are rebuilt from the pickled REF solve, R6 and TP are solved (the ILP as
    shipped), and the shipped patch + the 36-column joint image gate replay on REF / R6 / TP; the cross and the paired
    bootstraps as in forkprop_z, plus TP against R6 directly (gated and ungated, score / adj_J / edge_J).
Attach: forkprop_z's sources. CPU ~9 h (forkprop_z took 519 min with 2 extra solves per movie; the same shape here).
"""
src = open("forkprop_z.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:100])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-25a-forkprop_z"', 'SCRIPT_VERSION = "2026-09-26a-twopass"')

# ---- label_rows returns its keys --------------------------------------------------------------------------------
rep("        gt_by_t[int(g_t[i])].append(i)\n    X_list, y_list = [], []\n",
    "        gt_by_t[int(g_t[i])].append(i)\n    X_list, y_list, K_list = [], [], []\n")
rep("            X_list.append(row)\n            y_list.append(1 if j in j_true else 0)\n",
    "            X_list.append(row)\n            y_list.append(1 if j in j_true else 0)\n            K_list.append((int(t), int(i), int(j)))\n")
rep("    if not X_list:\n        return np.zeros((0, NCOL_ALL), np.float32), np.zeros(0, int)\n    return np.stack(X_list).astype(np.float32), np.asarray(y_list)\n",
    "    if not X_list:\n        return np.zeros((0, NCOL_ALL), np.float32), np.zeros(0, int), []\n    return np.stack(X_list).astype(np.float32), np.asarray(y_list), K_list\n")
rep("    X_, y_ = label_rows(name, feats)\n    LAB[name] = (X_, y_)\n", "    X_, y_, _K = label_rows(name, feats)\n    LAB[name] = (X_, y_)\n")
rep("del LAB4\n", "# twopass: LAB4 (F4HP labelled rows with keys) is kept for the two-pass rankers\n")

# ---- the arms: REF / R6 / TP -----------------------------------------------------------------------------------
rep('''DA_ARMS = [("CR_NOCAP_DV150", dict(diverge=1.5, admit=False, rank="contrast", cap=100.0, src="REF")),
           ("ZCR_NOCAP_DV150", dict(diverge=1.5, admit=False, rank="contrast", cap=100.0, src="Z")),
           ("ZXCR_NOCAP_DV150", dict(diverge=1.5, admit=False, rank="contrast", cap=100.0, src="ZX"))]
''', '''DA_ARMS = [("CR_NOCAP_DV150", dict(diverge=1.5, admit=False, rank="contrast", cap=100.0, src="REF")),
           ("R6CR_NOCAP_DV150", dict(diverge=1.5, admit=False, rank="contrast", cap=100.0, src="R6")),
           ("TPCR_NOCAP_DV150", dict(diverge=1.5, admit=False, rank="contrast", cap=100.0, src="TP"))]
''')

# ---- the track columns ----------------------------------------------------------------------------------------------
TRACK = '''
# ---- twopass: multi-frame columns from the SOLVED graph ------------------------------------------------------------
TR_NAMES = ["has_tr", "md_tr", "dmd_tr", "cos_tr", "len_back_tr", "md_acc_tr", "cos_fwd_tr"]
N_TR = len(TR_NAMES)
COLS_TP = COLS + list(range(NCOL_ALL, NCOL_ALL + N_TR))
TRK_LAB = {}               # name -> track columns of LAB4[name]'s rows (pass 1)
TP_STAT = {a_: defaultdict(int) for a_ in ("R6", "TP")}


def track_cols(feats, T, nodes0, edges0, seeds, seed_of_node):
    """per candidate pair (t, i, j), in feats order, the multi-frame columns read off the solved graph (nodes0, edges0):
    the source's solved predecessor chain gives its velocity (one step) and acceleration (two steps); the target's
    solved successor gives its onward step. Positions from the seeds (unrounded), in um."""
    pred, succ = {}, defaultdict(list)
    for a_, b_ in edges0:
        pred[b_] = a_
        succ[a_].append(b_)
    node_of = {s_: g_ for g_, s_ in seed_of_node.items() if s_ is not None}

    def um(g_):
        s_ = seed_of_node.get(g_)
        if s_ is None:
            return np.asarray(nodes0[g_]["xyz"], float) * SCALE
        return np.asarray(seeds[s_[0]][0][s_[1]], float) * SCALE

    out = []
    for t in range(T - 1):
        pairs, rows, P, C = feats[t]
        arr = np.full((len(pairs), N_TR), np.nan, np.float32)
        if len(pairs):
            Pu = np.asarray(P, float) * SCALE
            Cu = np.asarray(C, float) * SCALE
            cache = {}
            for r_, (i_, j_) in enumerate(pairs):
                i_, j_ = int(i_), int(j_)
                if i_ not in cache:
                    g = node_of.get((t, i_))
                    v = a = None
                    n_back = 0
                    if g is not None:
                        g1 = pred.get(g)
                        if g1 is not None:
                            v = um(g) - um(g1)
                            n_back = 1
                            g2 = pred.get(g1)
                            if g2 is not None:
                                a = v - (um(g1) - um(g2))
                                n_back = 2
                                cur = g2
                                while n_back < 12 and pred.get(cur) is not None:
                                    cur = pred[cur]
                                    n_back += 1
                    cache[i_] = (v, a, n_back)
                v, a, n_back = cache[i_]
                step = Cu[j_] - Pu[i_]
                d = float(np.linalg.norm(step))
                arr[r_, 0] = 0.0 if v is None else (1.0 if a is None else 2.0)
                arr[r_, 4] = float(n_back)
                if v is not None:
                    md = float(np.linalg.norm(Cu[j_] - (Pu[i_] + v)))
                    arr[r_, 1] = md
                    arr[r_, 2] = d - md
                    nv = float(np.linalg.norm(v))
                    arr[r_, 3] = float(step @ v / (d * nv)) if d > 1e-9 and nv > 1e-9 else np.nan
                    if a is not None:
                        arr[r_, 5] = float(np.linalg.norm(Cu[j_] - (Pu[i_] + v + 0.5 * a)))
                gj = node_of.get((t + 1, j_))
                if gj is not None and len(succ.get(gj, [])) == 1:
                    fwd = um(succ[gj][0]) - um(gj)
                    nf = float(np.linalg.norm(fwd))
                    arr[r_, 6] = float(step @ fwd / (d * nf)) if d > 1e-9 and nf > 1e-9 else np.nan
        out.append(arr)
    return out


def tr_for_keys(trk, feats, keys):
    """the track columns of the labelled rows, by their (t, i, j) keys"""
    idx = {}
    for t in range(len(feats)):
        pairs = feats[t][0]
        if len(pairs):
            idx[t] = {(int(i_), int(j_)): r_ for r_, (i_, j_) in enumerate(pairs)}
    out = np.full((len(keys), N_TR), np.nan, np.float32)
    for r_, (t, i_, j_) in enumerate(keys):
        m_ = idx.get(t)
        if m_ is not None and (i_, j_) in m_:
            out[r_] = trk[t][m_[(i_, j_)]]
    return out


def feats_with_tr(feats, trk):
    return [(pairs, (np.hstack([rows, trk[t]]) if len(pairs) else rows), P, C) for t, (pairs, rows, P, C) in enumerate(feats)]


'''
rep("for name in SIX:\n    if time.time() - t_all > BUDGET_S:\n        INCOMPLETE.append(name)\n",
    TRACK + "for name in SIX:\n    if time.time() - t_all > BUDGET_S:\n        INCOMPLETE.append(name)\n")
rep("    seeds, feats = FEATS.pop(name)\n", "    seeds, feats = FEATS[name]                 # twopass: kept for the pass-2 solves\n")

# ---- pass 1: the track columns of the labelled rows, in place of the z solves ------------------------------------
rep('''    _tz = time.time()
    _zs = z_solves(name, dets, T, seeds, feats, MODELS_FOLD[k], edges0, seed_of_node, PRE_STATE["final_edges"])
    for _tag, (_nz, _ez, _pz, _pmz, _soz) in _zs.items():
        with open(BR_DIR / f"{name}_{_tag}.pkl", "wb") as _fh:
            pickle.dump((_nz, _ez, _pz[0], _pz[1], seeds, _pmz, _soz), _fh, protocol=pickle.HIGHEST_PROTOCOL)
    cells.append(f"z solves {time.time() - _tz:.0f}s: Z edges {len(_zs['Z'][1])} ZX edges {len(_zs['ZX'][1])} (REF {len(edges0)})")
    del _zs
''', '''    _tz = time.time()
    _trk = track_cols(feats, T, nodes0, edges0, seeds, seed_of_node)
    TRK_LAB[name] = tr_for_keys(_trk, feats, LAB4[name][2])
    _fin = np.isfinite(TRK_LAB[name][:, 1])
    cells.append(f"track cols {time.time() - _tz:.0f}s: labelled rows {len(_fin)}, with a solved predecessor {int(_fin.sum())}")
    del _trk
''')

# ---- between the passes: the two-pass rankers -----------------------------------------------------------------------
TRAIN = '''
# ---- twopass: the R6 (control) and TP rankers, per fold, on the 6bba other-fold F4HP labelled rows ---------------
print(f"\\n=== TWO-PASS RANKERS: per fold, R6 = the shipped 21 columns and TP = 21 + {N_TR} track columns "
      f"({', '.join(TR_NAMES)}), both trained on the OTHER fold's 6bba F4HP labelled rows (the 44b6 rows have no solve, "
      f"so R6 is the control that differs from TP by the columns alone) ===", flush=True)
MODELS_R6, MODELS_TP = {}, {}
_tp_ok = [n for n in SIX if n in TRK_LAB]
_Xall = np.concatenate([np.hstack([LAB4[n][0], TRK_LAB[n]]) for n in _tp_ok]).astype(np.float32)
_yall = np.concatenate([LAB4[n][1] for n in _tp_ok])
print(f"  labelled F4HP rows {len(_yall):,} ({_yall.mean():.3f} pos) over {len(_tp_ok)} movies; a solved predecessor on "
      f"{int(np.isfinite(_Xall[:, NCOL_ALL + 1]).sum()):,} rows, two steps on {int((_Xall[:, NCOL_ALL] == 2).sum()):,}, an onward step on "
      f"{int(np.isfinite(_Xall[:, NCOL_ALL + 6]).sum()):,}", flush=True)
print("  single-column AUC of the track columns on the labelled rows (finite rows; below 0.5 = informative with the sign flipped):", flush=True)
for _c, _nm in enumerate(TR_NAMES):
    _v = _Xall[:, NCOL_ALL + _c]
    _m = np.isfinite(_v)
    if _m.sum() and _yall[_m].min() != _yall[_m].max():
        print(f"    {_nm:<12} AUC {_auc(_yall[_m], _v[_m]):.4f} on {int(_m.sum()):,} rows", flush=True)
for _c, _nm in ((4, "md_um (greedy chain)"), (5, "d_minus_md (greedy chain)")):
    print(f"    {_nm:<28} AUC {_auc(_yall, _Xall[:, _c]):.4f} (for scale)", flush=True)
for k in (0, 1):
    t0 = time.time()
    _tr = [n for n in _tp_ok if FOLD_OF[n] != k]
    _he = [n for n in _tp_ok if FOLD_OF[n] == k]
    Xa = np.concatenate([np.hstack([LAB4[n][0], TRK_LAB[n]]) for n in _tr]).astype(np.float32)
    ya = np.concatenate([LAB4[n][1] for n in _tr])
    Xh = np.concatenate([np.hstack([LAB4[n][0], TRK_LAB[n]]) for n in _he]).astype(np.float32)
    yh = np.concatenate([LAB4[n][1] for n in _he])
    line = f"  fold{k}: trained on {len(_tr)} movies / {len(ya):,} rows, held-out {len(_he)} movies / {len(yh):,} rows:"
    for tag, cols, bank in (("R6", COLS, MODELS_R6), ("TP", COLS_TP, MODELS_TP)):
        m = HistGradientBoostingClassifier(max_iter=300, random_state=0)
        m.fit(Xa[:, cols], ya)
        bank[k] = m
        line += f"  {tag} held-out AUC {_auc(yh, m.predict_proba(Xh[:, cols])[:, 1]):.4f}"
    print(line + f"  {time.time() - t0:.0f}s", flush=True)
    del Xa, ya, Xh, yh
del _Xall, _yall
gc.collect()

'''
rep("DIVW_ARMS = [(a_, DIV_W) for a_, _c in BR_ARMS_RUN]\nBR_STAT_ARM = {a_: defaultdict(int) for a_, _c in BR_ARMS_RUN}\n",
    TRAIN + "DIVW_ARMS = [(a_, DIV_W) for a_, _c in BR_ARMS_RUN]\nBR_STAT_ARM = {a_: defaultdict(int) for a_, _c in BR_ARMS_RUN}\n")

# ---- pass 2: the R6 and TP solves in place of the Z / ZX pickles -----------------------------------------------------
rep('''    for _src in ("REF", "Z", "ZX"):
        with open(BR_DIR / (f"{name}.pkl" if _src == "REF" else f"{name}_{_src}.pkl"), "rb") as _fh:
            _n0, _e0, _c, _cs, seeds, _pm, _so = pickle.load(_fh)
        _ct = [cKDTree(c_ * SCALE) if len(c_) else None for c_ in _c]
        _ss, _b2 = _aux_extra(seeds, _pm)
        SOLVES[_src] = (_n0, _e0, (_c, _cs, _ct), _pm, _so, (_pm, _so, _ss, _b2))
''', '''    for _src in ("REF",):
        with open(BR_DIR / f"{name}.pkl", "rb") as _fh:
            _n0, _e0, _c, _cs, seeds, _pm, _so = pickle.load(_fh)
        _ct = [cKDTree(c_ * SCALE) if len(c_) else None for c_ in _c]
        _ss, _b2 = _aux_extra(seeds, _pm)
        SOLVES[_src] = (_n0, _e0, (_c, _cs, _ct), _pm, _so, (_pm, _so, _ss, _b2))
    # twopass: the track columns from REF's solve, then the R6 (control) and TP solves
    _tsol = time.time()
    _dets2 = load_dets(name, "F4")
    _T2 = len(_dets2)
    _seeds2, _feats2 = FEATS[name]
    _n0r, _e0r, _cr, _csr, _sr, _pmr, _sor = SOLVES["REF"][0], SOLVES["REF"][1], None, None, seeds, SOLVES["REF"][3], SOLVES["REF"][4]
    _trk2 = track_cols(_feats2, _T2, _n0r, _e0r, _seeds2, _sor)
    _feats_tp = feats_with_tr(_feats2, _trk2)
    for _src, _mdl, _cols, _ff in (("R6", MODELS_R6[FOLD_OF[name]], COLS, _feats2), ("TP", MODELS_TP[FOLD_OF[name]], COLS_TP, _feats_tp)):
        _t1 = time.time()
        _n0, _e0, _pool, _aux0 = solve_one(_dets2, _T2, _seeds2, _ff, _mdl, _cols, disapp=ILP_DISAPP, floor=PROD_FLOOR)
        _pm, _so = _aux0
        _ss, _b2 = _aux_extra(_seeds2, _pm)
        SOLVES[_src] = (_n0, _e0, _pool, _pm, _so, (_pm, _so, _ss, _b2))
        _oc = defaultdict(int)
        for _a, _b in _e0:
            _oc[_a] += 1
        TP_STAT[_src]["movies"] += 1
        TP_STAT[_src]["edges"] += len(_e0)
        TP_STAT[_src]["forks"] += sum(1 for v_ in _oc.values() if v_ >= 2)
        TP_STAT[_src]["solve_s"] += int(time.time() - _t1)
    TP_STAT["REF"] = TP_STAT.get("REF", defaultdict(int))
    TP_STAT["REF"]["edges"] += len(SOLVES["REF"][1])
    _ocr = defaultdict(int)
    for _a, _b in SOLVES["REF"][1]:
        _ocr[_a] += 1
    TP_STAT["REF"]["forks"] += sum(1 for v_ in _ocr.values() if v_ >= 2)
    TP_STAT["REF"]["movies"] += 1
    _tp_line = (f"solves {time.time() - _tsol:.0f}s: REF edges {len(SOLVES['REF'][1])}, R6 {len(SOLVES['R6'][1])}, TP {len(SOLVES['TP'][1])}")
    del _dets2, _trk2, _feats_tp
''')
rep("    cells = []\n    for arm, cfg in BR_ARMS_RUN:\n        DIV_DIVERGE_UM = cfg[\"diverge\"]\n",
    "    cells = [_tp_line]\n    for arm, cfg in BR_ARMS_RUN:\n        DIV_DIVERGE_UM = cfg[\"diverge\"]\n")

# ---- the z report -> the two-pass solve report ----------------------------------------------------------------------
i0 = src.index('print("\\n=== THE z SOLVES, summed over the movies')
i1 = src.index('print("\\n=== THE PATCH PER ARM (orphan pass + admit pass)')
src = src[:i0] + '''print("\\n=== THE SOLVES, summed over the movies replayed (REF: the shipped fold ranker, 21 columns, trained on 199 movies; "
      "R6: 21 columns trained on the other fold's 6bba F4HP rows; TP: R6's rows + the track columns from REF's solve) ===", flush=True)
for _tag in ("REF", "R6", "TP"):
    d_ = TP_STAT.get(_tag, {})
    print(f"  {_tag:<4} movies {d_.get('movies', 0)} | edges {d_.get('edges', 0):,} | ILP forks {d_.get('forks', 0):,}"
          + (f" | solve {d_.get('solve_s', 0):,}s" if _tag != "REF" else ""), flush=True)
''' + src[i1:]

# ---- TP against R6 directly -----------------------------------------------------------------------------------------
rep('''print("    reading: an arm counts only if its jointimg cell beats REF + jointimg with the interval off zero; the "''',
    '''print("\\n  TWO-PASS AGAINST ITS CONTROL: TPCR vs R6CR (the same rows, the same shipped patch and gate; the columns are the only "
      "difference), paired bootstrap over the movies, gated and ungated ===", flush=True)
for _pair in (("TPCR_NOCAP_DV150 + jointimg", "R6CR_NOCAP_DV150 + jointimg"), ("TPCR_NOCAP_DV150 + joint", "R6CR_NOCAP_DV150 + joint"),
              ("R6CR_NOCAP_DV150 + jointimg", "CR_NOCAP_DV150 + jointimg")):
    if _pair[0] not in BS or _pair[1] not in BS:
        continue
    for nm_, bank in (("score", BS), ("adj_J", _BSA)):
        d = bank[_pair[0]] - bank[_pair[1]]
        lo, hi = float(np.quantile(d, .05)), float(np.quantile(d, .95))
        print(f"    {_pair[0]:<28} vs {_pair[1]:<28} {nm_:<6} {float(np.mean(d)):>+9.4f}   [{lo:>+8.4f}, {hi:>+8.4f}]  "
              f"{'YES' if (lo > 0 or hi < 0) else 'no'}", flush=True)
print("    reading: the two-pass claim holds only if TPCR beats R6CR with the interval off zero on adj_J (the linking term); "
      "R6CR vs CR says what training on 6bba-only rows costs against the shipped 199-movie ranker.", flush=True)
print("    reading: an arm counts only if its jointimg cell beats REF + jointimg with the interval off zero; the "''')

rep('with open("/kaggle/working/forkssz_dw55.csv", "w") as fh:', 'with open("/kaggle/working/forktp_dw55.csv", "w") as fh:')
rep('    with open(f"/kaggle/working/forkssz_{arm}.csv", "w") as fh:', '    with open(f"/kaggle/working/forktp_{arm}.csv", "w") as fh:')
rep('    print("FORKPROPZ OK", flush=True)', '    print("TWOPASS OK", flush=True)')

assert src.count("def track_cols(") == 1 and src.count("MODELS_TP[") == 1 and src.count("z_solves(") == 1 and "SOLVES['TP']" in src
open("twopass.py", "w", encoding="utf-8", newline="\n").write(src)
print("written twopass.py", src.count("\n"), "lines")
