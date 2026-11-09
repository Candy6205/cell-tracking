"""twopass.py -> flowgate.py: the public 0.953 notebook's flow gate, in our pipeline, on its own.

WHY (2026-09-27). The public notebook anvithpothula/biohub-0-953-lb-original adds a flow-guided relink to the 0.947 stack
("agent/frontier943_flow"): a first matching, then for every cell the median displacement of its 12 nearest matched
neighbours within 40 um (excluding within 1.5 um, at least 4) predicts where it goes, and a pair is allowed if the target
lies within 7 um of the cell's raw position OR of its predicted position. Our record has flow only as a ranker column
(flowvel, 2026-09-03: every magnitude under 0.001); the gate is a different use and has not been tested. The user
(2026-09-27): "流场门也去试试吧，或许分开试".

WHAT. twopass.py's harness (pass 1: the shipped fold ranker + ILP on the 128 6bba movies; pass 2: re-solves + the shipped
patch + the 36-column image gate; the cross and the paired bootstraps) with the two-pass rankers removed and, in place:
  * flow_field(): per frame, per seed, the median displacement of the FLOW_K nearest sources of pass 1's SOLVED edges
    within FLOW_R_UM (excluding within FLOW_EXCL_UM), NaN below FLOW_MIN neighbours;
  * feats_gated(): a candidate pair (i, j) stays if d_raw <= R or d_pred <= R, or (no flow and d_raw <= FLOW_RELAX_UM);
    our pairs are generated within GATE_UM 10 um, so the gate prunes the (R, 10] band unless the flow explains the step;
  * pass 1: per labelled F4HP row d_raw, d_pred and the fold ranker's probability -> the base rate of the gate by label
    (true edges lost against false candidates removed, all rows and the rows above the ILP floor), before any solve;
  * pass 2: the SAME shipped fold ranker solved on the gated candidates at R = 7 (the notebook's) and R = 9;
    arms CR (REF's solve), FG7CR, FG9CR under the shipped patch; FG against CR paired (score, adj_J; gated and ungated).
Attach: twopass's sources. CPU ~9 h.
"""
src = open("twopass.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:100])
    src = src.replace(old, new)


def cut(a, b, new):
    """replace src[a-marker .. b-marker) by new (b stays)"""
    global src
    assert src.count(a) == 1 and src.count(b) >= 1, (src.count(a), src.count(b), a[:80], b[:80])
    i0 = src.index(a)
    i1 = src.index(b, i0)
    src = src[:i0] + new + src[i1:]


rep('SCRIPT_VERSION = "2026-09-26a-twopass"', 'SCRIPT_VERSION = "2026-09-27a-flowgate"')

rep('''           ("R6CR_NOCAP_DV150", dict(diverge=1.5, admit=False, rank="contrast", cap=100.0, src="R6")),
           ("TPCR_NOCAP_DV150", dict(diverge=1.5, admit=False, rank="contrast", cap=100.0, src="TP"))]
''', '''           ("FG7CR_NOCAP_DV150", dict(diverge=1.5, admit=False, rank="contrast", cap=100.0, src="FG7")),
           ("FG9CR_NOCAP_DV150", dict(diverge=1.5, admit=False, rank="contrast", cap=100.0, src="FG9"))]
''')

# ---- the flow field and the gate, in place of the track columns -------------------------------------------------------
FLOW = '''# ---- flowgate: the neighbourhood flow of pass 1's solved edges, and the gate on the candidate pairs ----------------
FLOW_K = 12                # the notebook's: the 12 nearest matched neighbours
FLOW_R_UM = 40.0           # within 40 um
FLOW_EXCL_UM = 1.5         # excluding within 1.5 um (the cell itself)
FLOW_MIN = 4               # at least 4, else no prediction
FLOW_RELAX_UM = 10.0       # no prediction: the relaxed raw gate (= our pair radius GATE_UM)
FG_RADII = (("FG7", 7.0), ("FG9", 9.0))
FG_LAB = {}                # name -> (d_raw, d_pred, p) of LAB4[name]'s rows (pass 1)
TP_STAT = {a_: defaultdict(int) for a_, _r in FG_RADII}
assert GATE_UM == FLOW_RELAX_UM, GATE_UM


def flow_field(T, edges0, seeds, seed_of_node):
    """per frame t an array (len(seeds[t]), 3) in um: the median displacement of the FLOW_K nearest sources of the solved
    edges t -> t + 1 within FLOW_R_UM of the seed (excluding within FLOW_EXCL_UM); NaN below FLOW_MIN neighbours"""
    pts = [[] for _ in range(T)]
    dsp = [[] for _ in range(T)]
    for a_, b_ in edges0:
        sa, sb = seed_of_node.get(a_), seed_of_node.get(b_)
        if sa is None or sb is None or sb[0] != sa[0] + 1:
            continue
        pa = np.asarray(seeds[sa[0]][0][sa[1]], float) * SCALE
        pb = np.asarray(seeds[sb[0]][0][sb[1]], float) * SCALE
        pts[sa[0]].append(pa)
        dsp[sa[0]].append(pb - pa)
    out = []
    for t in range(T):
        P = np.asarray(seeds[t][0], float).reshape(-1, 3) * SCALE
        f = np.full((len(P), 3), np.nan)
        if len(P) and len(pts[t]) >= FLOW_MIN:
            S = np.asarray(pts[t])
            D = np.asarray(dsp[t])
            kq = min(FLOW_K + 4, len(S))
            dd, jj = cKDTree(S).query(P, k=kq, distance_upper_bound=FLOW_R_UM)
            dd = dd.reshape(len(P), -1)
            jj = jj.reshape(len(P), -1)
            for i in range(len(P)):
                m = np.isfinite(dd[i]) & (dd[i] > FLOW_EXCL_UM)
                sel = jj[i][m][:FLOW_K]
                if len(sel) >= FLOW_MIN:
                    f[i] = np.median(D[sel], axis=0)
        out.append(f)
    return out


def pair_dists(P, C, pairs, flow_t):
    """d_raw and d_pred (inf where the source has no flow) of the pairs of one frame, um"""
    Pu = np.asarray(P, float).reshape(-1, 3) * SCALE
    Cu = np.asarray(C, float).reshape(-1, 3) * SCALE
    ii = np.fromiter((i for i, _ in pairs), np.int64, len(pairs))
    jj = np.fromiter((j for _, j in pairs), np.int64, len(pairs))
    d_raw = np.linalg.norm(Cu[jj] - Pu[ii], axis=1)
    fl = flow_t[ii]
    has = np.isfinite(fl[:, 0])
    d_pred = np.full(len(pairs), np.inf)
    d_pred[has] = np.linalg.norm(Cu[jj[has]] - (Pu[ii[has]] + fl[has]), axis=1)
    return d_raw, d_pred, has


def gate_keep(d_raw, d_pred, has, R):
    return (d_raw <= R) | (d_pred <= R) | (~has & (d_raw <= FLOW_RELAX_UM))


def feats_gated(feats, flow, R):
    """the candidate pairs that pass the flow gate at radius R, rows aligned; and the counts"""
    out, st = [], defaultdict(int)
    for t, (pairs, rows, P, C) in enumerate(feats):
        if not len(pairs):
            out.append((pairs, rows, P, C))
            continue
        d_raw, d_pred, has = pair_dists(P, C, pairs, flow[t])
        keep = gate_keep(d_raw, d_pred, has, R)
        st["pairs"] += len(pairs)
        st["kept"] += int(keep.sum())
        st["no_flow"] += int((~has).sum())
        st["kept_by_flow"] += int(((d_raw > R) & (d_pred <= R)).sum())
        out.append(([pairs[k_] for k_ in np.where(keep)[0]], rows[keep], P, C))
    return out, st


def fg_for_keys(flow, seeds, keys):
    """(d_raw, d_pred) of the labelled rows' (t, i, j); d_pred NaN where the source has no flow"""
    out = np.full((len(keys), 2), np.nan, np.float32)
    for r_, (t_, i_, j_) in enumerate(keys):
        pu = np.asarray(seeds[t_][0][i_], float) * SCALE
        cu = np.asarray(seeds[t_ + 1][0][j_], float) * SCALE
        out[r_, 0] = np.linalg.norm(cu - pu)
        f = flow[t_][i_]
        if np.isfinite(f[0]):
            out[r_, 1] = np.linalg.norm(cu - (pu + f))
    return out


'''
cut("# ---- twopass: multi-frame columns from the SOLVED graph",
    "for name in SIX:\n    if time.time() - t_all > BUDGET_S:\n        INCOMPLETE.append(name)\n", FLOW)

# ---- pass 1: the gate's base rate on the labelled rows ----------------------------------------------------------------
rep('''    _trk = track_cols(feats, T, nodes0, edges0, seeds, seed_of_node)
    TRK_LAB[name] = tr_for_keys(_trk, feats, LAB4[name][2])
    _fin = np.isfinite(TRK_LAB[name][:, 1])
    cells.append(f"track cols {time.time() - _tz:.0f}s: labelled rows {len(_fin)}, with a solved predecessor {int(_fin.sum())}")
    del _trk
''', '''    _flow = flow_field(T, edges0, seeds, seed_of_node)
    _dd = fg_for_keys(_flow, seeds, LAB4[name][2])
    _pp = (MODELS_FOLD[k].predict_proba(LAB4[name][0][:, COLS])[:, 1] if len(_dd) else np.zeros(0))
    FG_LAB[name] = np.column_stack([_dd, _pp]).astype(np.float32) if len(_dd) else np.zeros((0, 3), np.float32)
    _fin = np.isfinite(FG_LAB[name][:, 1])
    cells.append(f"flow {time.time() - _tz:.0f}s: labelled rows {len(_fin)}, with a flow prediction {int(_fin.sum())}")
    del _flow, _dd, _pp
''')

# ---- between the passes: the base-rate report, in place of the two-pass rankers ---------------------------------------
BASE = '''# ---- flowgate: the gate's base rate on the labelled rows, before any solve ------------------------------------------
print(f"\\n=== FLOW GATE BASE RATE on the 6bba F4HP labelled rows: a pair passes if d_raw <= R or d_pred <= R (d_pred = the "
      f"distance of the target from the source + the median displacement of its {FLOW_K} nearest solved neighbours within "
      f"{FLOW_R_UM:g} um), or the source has no flow and d_raw <= {FLOW_RELAX_UM:g}; pairs exist only within {GATE_UM:g} um ===", flush=True)
_fg_ok = [n for n in SIX if n in FG_LAB]
_G = np.concatenate([FG_LAB[n] for n in _fg_ok])
_yall = np.concatenate([LAB4[n][1] for n in _fg_ok])
_has = np.isfinite(_G[:, 1])
_dp = np.where(_has, _G[:, 1], np.inf)
print(f"  labelled rows {len(_yall):,} ({_yall.mean():.3f} pos) over {len(_fg_ok)} movies; with a flow prediction {int(_has.sum()):,} "
      f"({_has.mean() * 100:.1f}%)", flush=True)
for _lab, _nm in ((1, "true pairs"), (0, "false pairs")):
    _m = _yall == _lab
    _band = _m & (_G[:, 0] > 7.0)
    print(f"  {_nm:<11} n {int(_m.sum()):>7,} | d_raw p50 {np.median(_G[_m, 0]):.2f} p90 {np.percentile(_G[_m, 0], 90):.2f} | "
          f"d_pred p50 {np.median(_G[_m & _has, 1]):.2f} p90 {np.percentile(_G[_m & _has, 1], 90):.2f} | in the (7, 10] band "
          f"{int(_band.sum()):,} ({_band.sum() / max(_m.sum(), 1) * 100:.2f}%), of those within 7 um of the prediction "
          f"{int((_band & (_dp <= 7.0)).sum()):,}, without flow {int((_band & ~_has).sum()):,}", flush=True)
print(f"  single-column AUC (flipped = smaller is truer): d_raw {1 - _auc(_yall, _G[:, 0]):.4f}, d_pred on the rows with flow "
      f"{1 - _auc(_yall[_has], _G[_has, 1]):.4f} (d_raw on the same rows {1 - _auc(_yall[_has], _G[_has, 0]):.4f}), "
      f"md_um (the ranker's greedy-chain prediction) {1 - _auc(_yall, np.concatenate([LAB4[n][0][:, 4] for n in _fg_ok])):.4f}", flush=True)
print(f"  {'gate':<6} {'rows':<22} {'true kept':>10} {'true lost':>10} {'false kept':>11} {'false removed':>14}", flush=True)
for _tag, _R in FG_RADII:
    _k = gate_keep(_G[:, 0], _dp, _has, _R)
    for _nm, _sel in (("all labelled rows", np.ones(len(_yall), bool)), (f"p >= {PROD_FLOOR} (ILP sees)", _G[:, 2] >= PROD_FLOOR)):
        _t = _sel & (_yall == 1)
        _f = _sel & (_yall == 0)
        print(f"  {_tag:<6} {_nm:<22} {int((_t & _k).sum()):>10,} {int((_t & ~_k).sum()):>10,} {int((_f & _k).sum()):>11,} "
              f"{int((_f & ~_k).sum()):>14,}", flush=True)
print("  reading: the gate can only help if, among the rows the ILP sees, it removes false candidates without losing true "
      "ones; the labelled false pairs are the annotated cells' wrong partners, the unannotated candidates are not counted here.", flush=True)
del _G, _yall, _has, _dp
gc.collect()

'''
cut("# ---- twopass: the R6 (control) and TP rankers, per fold, on the 6bba other-fold F4HP labelled rows",
    "DIVW_ARMS = [(a_, DIV_W) for a_, _c in BR_ARMS_RUN]\nBR_STAT_ARM = {a_: defaultdict(int) for a_, _c in BR_ARMS_RUN}\n", BASE)

# ---- pass 2: the gated solves ---------------------------------------------------------------------------------------
cut("    # twopass: the track columns from REF's solve, then the R6 (control) and TP solves\n",
    "    nodes0, edges0, pool, pmap, seed_of_node, aux = SOLVES[\"REF\"]\n",
    '''    # flowgate: the flow of REF's solve, then the shipped fold ranker solved on the gated candidates
    _tsol = time.time()
    _dets2 = load_dets(name, "F4")
    _T2 = len(_dets2)
    _seeds2, _feats2 = FEATS[name]
    _flow2 = flow_field(_T2, SOLVES["REF"][1], _seeds2, SOLVES["REF"][4])
    for _src, _R in FG_RADII:
        _t1 = time.time()
        _ff, _gs = feats_gated(_feats2, _flow2, _R)
        _n0, _e0, _pool, _aux0 = solve_one(_dets2, _T2, _seeds2, _ff, MODELS_FOLD[FOLD_OF[name]], COLS, disapp=ILP_DISAPP, floor=PROD_FLOOR)
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
        for k_, v_ in _gs.items():
            TP_STAT[_src][k_] += v_
        del _ff
    TP_STAT["REF"] = TP_STAT.get("REF", defaultdict(int))
    TP_STAT["REF"]["edges"] += len(SOLVES["REF"][1])
    _ocr = defaultdict(int)
    for _a, _b in SOLVES["REF"][1]:
        _ocr[_a] += 1
    TP_STAT["REF"]["forks"] += sum(1 for v_ in _ocr.values() if v_ >= 2)
    TP_STAT["REF"]["movies"] += 1
    _tp_line = (f"solves {time.time() - _tsol:.0f}s: REF edges {len(SOLVES['REF'][1])}, "
                + ", ".join(f"{_s} {len(SOLVES[_s][1])}" for _s, _r in FG_RADII))
    del _dets2, _flow2
''')

# ---- the solve report -------------------------------------------------------------------------------------------------
cut('print("\\n=== THE SOLVES, summed over the movies replayed', 'print("\\n=== THE PATCH PER ARM (orphan pass + admit pass)',
    '''print("\\n=== THE SOLVES, summed over the movies replayed (REF: the shipped fold ranker on every candidate pair; FG7 / FG9: the "
      "same ranker on the pairs that pass the flow gate at 7 / 9 um, the flow read off REF's solve) ===", flush=True)
for _tag in ["REF"] + [a_ for a_, _r in FG_RADII]:
    d_ = TP_STAT.get(_tag, {})
    print(f"  {_tag:<4} movies {d_.get('movies', 0)} | edges {d_.get('edges', 0):,} | ILP forks {d_.get('forks', 0):,}"
          + (f" | pairs {d_.get('pairs', 0):,} kept {d_.get('kept', 0):,} ({d_.get('kept', 0) / max(d_.get('pairs', 0), 1) * 100:.2f}%), "
             f"kept by the flow alone {d_.get('kept_by_flow', 0):,}, sources without flow (pairs) {d_.get('no_flow', 0):,} | "
             f"solve {d_.get('solve_s', 0):,}s" if _tag != "REF" else ""), flush=True)
''')

# ---- the gate against its control ---------------------------------------------------------------------------------------
cut('print("\\n  TWO-PASS AGAINST ITS CONTROL: TPCR vs R6CR', 'print("    reading: an arm counts only if its jointimg cell beats REF + jointimg',
    '''print("\\n  THE FLOW GATE AGAINST ITS CONTROL: FG7CR / FG9CR vs CR (the same ranker, the same shipped patch and gate; the "
      "candidate set is the only difference), paired bootstrap over the movies, gated and ungated ===", flush=True)
for _pair in (("FG7CR_NOCAP_DV150 + jointimg", "CR_NOCAP_DV150 + jointimg"), ("FG7CR_NOCAP_DV150 + joint", "CR_NOCAP_DV150 + joint"),
              ("FG9CR_NOCAP_DV150 + jointimg", "CR_NOCAP_DV150 + jointimg"), ("FG9CR_NOCAP_DV150 + joint", "CR_NOCAP_DV150 + joint")):
    if _pair[0] not in BS or _pair[1] not in BS:
        continue
    for nm_, bank in (("score", BS), ("adj_J", _BSA)):
        d = bank[_pair[0]] - bank[_pair[1]]
        lo, hi = float(np.quantile(d, .05)), float(np.quantile(d, .95))
        print(f"    {_pair[0]:<29} vs {_pair[1]:<26} {nm_:<6} {float(np.mean(d)):>+9.4f}   [{lo:>+8.4f}, {hi:>+8.4f}]  "
              f"{'YES' if (lo > 0 or hi < 0) else 'no'}", flush=True)
print("    reading: the flow-gate claim holds only if an FG arm beats CR with the interval off zero on adj_J (the linking term).", flush=True)
''')

rep('with open("/kaggle/working/forktp_dw55.csv", "w") as fh:', 'with open("/kaggle/working/forkfg_dw55.csv", "w") as fh:')
rep('    with open(f"/kaggle/working/forktp_{arm}.csv", "w") as fh:', '    with open(f"/kaggle/working/forkfg_{arm}.csv", "w") as fh:')
rep('    print("TWOPASS OK", flush=True)', '    print("FLOWGATE OK", flush=True)')

for bad in ("track_cols", "TRK_LAB", "MODELS_R6", "MODELS_TP", "COLS_TP", "feats_with_tr", "tr_for_keys", "R6CR", "TPCR", "TR_NAMES"):
    assert bad not in src, bad
assert src.count("def flow_field(") == 1 and src.count("feats_gated(") == 2 and "FG7CR_NOCAP_DV150" in src
open("flowgate.py", "w", encoding="utf-8", newline="\n").write(src)
print("written flowgate.py", src.count("\n"), "lines")
