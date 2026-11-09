"""twopass.py -> flowrank.py: the neighbourhood flow as RANKER COLUMNS (not as a gate), a second solve.

WHY (2026-09-28). flowgate (2026-09-27) killed the flow as a pruning of our candidates and left one measured residue: the
distance of the target from the flow-predicted position separates true from false pairs better than the ranker's own
greedy-chain prediction (single-column AUC 0.9795 against md_um 0.9608; in the 7-10 um band 66% of the true pairs lie
within 7 um of the prediction against 3% of the false). twopass (2026-09-26) found seven solved-chain columns add nothing,
but its strongest column (md_tr 0.964) was no better than md_um; this one is. The user (2026-09-28): "那把2也做一下？".

WHAT. twopass.py with the track columns replaced by six flow columns per candidate pair, read off pass 1's solved edges
(REF's own solve; the 12 nearest solved sources within 40 um, excluding within 1.5 um, at least 4 -- the public notebook's
recipe): has_fl, d_pred (target to source + flow), dmd_fl (d_raw - d_pred), cos_fl (the candidate step against the flow),
fl_mag (|flow|), fl_disp (median deviation of the neighbours' displacements from the flow: how coherent the
neighbourhood is). Everything else is twopass's: R6 (the shipped 21 columns, the control) and FL (21 + 6) per fold on the
other fold's 6bba F4HP labelled rows, pass 2 solves with both, arms CR / R6CR / FLCR under the shipped patch and the
36-column image gate, FL against R6 paired (score, adj_J; gated and ungated).
Attach: twopass's sources. CPU ~9 h.
"""
src = open("twopass.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:100])
    src = src.replace(old, new)


def cut(a, b, new):
    global src
    assert src.count(a) == 1, (src.count(a), a[:80])
    i0 = src.index(a)
    i1 = src.index(b, i0)
    src = src[:i0] + new + src[i1:]


rep('SCRIPT_VERSION = "2026-09-26a-twopass"', 'SCRIPT_VERSION = "2026-09-28a-flowrank"')

# ---- the flow columns in place of the track columns (same shapes: one (len(pairs), N_TR) array per frame) ------------
cut('TR_NAMES = ["has_tr", "md_tr", "dmd_tr", "cos_tr", "len_back_tr", "md_acc_tr", "cos_fwd_tr"]', "def tr_for_keys(trk, feats, keys):",
    '''TR_NAMES = ["has_fl", "d_pred", "dmd_fl", "cos_fl", "fl_mag", "fl_disp"]
N_TR = len(TR_NAMES)
COLS_TP = COLS + list(range(NCOL_ALL, NCOL_ALL + N_TR))
TRK_LAB = {}               # name -> flow columns of LAB4[name]'s rows (pass 1)
TP_STAT = {a_: defaultdict(int) for a_ in ("R6", "TP")}
FLOW_K = 12                # the public notebook's recipe: the 12 nearest matched neighbours
FLOW_R_UM = 40.0           # within 40 um
FLOW_EXCL_UM = 1.5         # excluding within 1.5 um (the cell itself)
FLOW_MIN = 4               # at least 4, else no prediction


def flow_field(T, edges0, seeds, seed_of_node):
    """per frame t: flow (len(seeds[t]), 3) um = the median displacement of the FLOW_K nearest sources of the solved edges
    t -> t + 1 within FLOW_R_UM (excluding within FLOW_EXCL_UM), NaN below FLOW_MIN neighbours; and disp (len,) um = the
    median distance of those neighbours' displacements from the flow"""
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
    flow, disp = [], []
    for t in range(T):
        P = np.asarray(seeds[t][0], float).reshape(-1, 3) * SCALE
        f = np.full((len(P), 3), np.nan)
        g = np.full(len(P), np.nan)
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
                    g[i] = np.median(np.linalg.norm(D[sel] - f[i], axis=1))
        flow.append(f)
        disp.append(g)
    return flow, disp


def track_cols(feats, T, nodes0, edges0, seeds, seed_of_node):
    """per candidate pair (t, i, j), in feats order, the flow columns read off the solved graph's edges (nodes0 unused):
    has_fl, d_pred, dmd_fl, cos_fl, fl_mag, fl_disp; NaN (has_fl 0) where the source has no flow"""
    flow, disp = flow_field(T, edges0, seeds, seed_of_node)
    out = []
    for t, (pairs, rows, P, C) in enumerate(feats):
        arr = np.full((len(pairs), N_TR), np.nan, np.float32)
        if len(pairs):
            Pu = np.asarray(P, float).reshape(-1, 3) * SCALE
            Cu = np.asarray(C, float).reshape(-1, 3) * SCALE
            ii = np.fromiter((i for i, _ in pairs), np.int64, len(pairs))
            jj = np.fromiter((j for _, j in pairs), np.int64, len(pairs))
            step = Cu[jj] - Pu[ii]
            d_raw = np.linalg.norm(step, axis=1)
            fl = flow[t][ii]
            has = np.isfinite(fl[:, 0])
            arr[:, 0] = has
            if has.any():
                d_pred = np.linalg.norm(step[has] - fl[has], axis=1)
                mag = np.linalg.norm(fl[has], axis=1)
                arr[has, 1] = d_pred
                arr[has, 2] = d_raw[has] - d_pred
                den = mag * d_raw[has]
                cs = np.full(int(has.sum()), np.nan)
                ok = den > 0.04
                cs[ok] = (step[has][ok] * fl[has][ok]).sum(1) / den[ok]
                arr[has, 3] = cs
                arr[has, 4] = mag
                arr[has, 5] = disp[t][ii][has]
        out.append(arr)
    return out


''')

# ---- the training block's prints ------------------------------------------------------------------------------------
rep('''print(f"\\n=== TWO-PASS RANKERS: per fold, R6 = the shipped 21 columns and TP = 21 + {N_TR} track columns "
      f"({', '.join(TR_NAMES)}), both trained on the OTHER fold's 6bba F4HP labelled rows (the 44b6 rows have no solve, "
      f"so R6 is the control that differs from TP by the columns alone) ===", flush=True)''',
    '''print(f"\\n=== FLOW RANKERS: per fold, R6 = the shipped 21 columns and FL = 21 + {N_TR} flow columns "
      f"({', '.join(TR_NAMES)}), both trained on the OTHER fold's 6bba F4HP labelled rows (the 44b6 rows have no solve, "
      f"so R6 is the control that differs from FL by the columns alone) ===", flush=True)''')
rep('''print(f"  labelled F4HP rows {len(_yall):,} ({_yall.mean():.3f} pos) over {len(_tp_ok)} movies; a solved predecessor on "
      f"{int(np.isfinite(_Xall[:, NCOL_ALL + 1]).sum()):,} rows, two steps on {int((_Xall[:, NCOL_ALL] == 2).sum()):,}, an onward step on "
      f"{int(np.isfinite(_Xall[:, NCOL_ALL + 6]).sum()):,}", flush=True)''',
    '''print(f"  labelled F4HP rows {len(_yall):,} ({_yall.mean():.3f} pos) over {len(_tp_ok)} movies; a flow prediction on "
      f"{int(np.isfinite(_Xall[:, NCOL_ALL + 1]).sum()):,} rows", flush=True)''')
rep('    for tag, cols, bank in (("R6", COLS, MODELS_R6), ("TP", COLS_TP, MODELS_TP)):',
    '    _imp = None\n    for tag, cols, bank in (("R6", COLS, MODELS_R6), ("FL", COLS_TP, MODELS_TP)):')
rep('''    cells.append(f"track cols {time.time() - _tz:.0f}s: labelled rows {len(_fin)}, with a solved predecessor {int(_fin.sum())}")''',
    '''    cells.append(f"flow cols {time.time() - _tz:.0f}s: labelled rows {len(_fin)}, with a flow prediction {int(_fin.sum())}")''')

# ---- names in the report: TP -> FL (the internal source tag stays "TP") ------------------------------------------------
rep('           ("TPCR_NOCAP_DV150", dict(diverge=1.5, admit=False, rank="contrast", cap=100.0, src="TP"))]',
    '           ("FLCR_NOCAP_DV150", dict(diverge=1.5, admit=False, rank="contrast", cap=100.0, src="TP"))]')
rep('''    _tp_line = (f"solves {time.time() - _tsol:.0f}s: REF edges {len(SOLVES['REF'][1])}, R6 {len(SOLVES['R6'][1])}, TP {len(SOLVES['TP'][1])}")''',
    '''    _tp_line = (f"solves {time.time() - _tsol:.0f}s: REF edges {len(SOLVES['REF'][1])}, R6 {len(SOLVES['R6'][1])}, FL {len(SOLVES['TP'][1])}")''')
cut('print("\\n=== THE SOLVES, summed over the movies replayed', 'print("\\n=== THE PATCH PER ARM (orphan pass + admit pass)',
    '''print("\\n=== THE SOLVES, summed over the movies replayed (REF: the shipped fold ranker, 21 columns, trained on 199 movies; "
      "R6: 21 columns trained on the other fold's 6bba F4HP rows; FL: R6's rows + the flow columns from REF's solve) ===", flush=True)
for _tag, _nm in (("REF", "REF"), ("R6", "R6"), ("TP", "FL")):
    d_ = TP_STAT.get(_tag, {})
    print(f"  {_nm:<4} movies {d_.get('movies', 0)} | edges {d_.get('edges', 0):,} | ILP forks {d_.get('forks', 0):,}"
          + (f" | solve {d_.get('solve_s', 0):,}s" if _tag != "REF" else ""), flush=True)
''')
cut('print("\\n  TWO-PASS AGAINST ITS CONTROL: TPCR vs R6CR', 'print("    reading: an arm counts only if its jointimg cell beats REF + jointimg',
    '''print("\\n  THE FLOW COLUMNS AGAINST THEIR CONTROL: FLCR vs R6CR (the same rows, the same shipped patch and gate; the columns are "
      "the only difference), paired bootstrap over the movies, gated and ungated ===", flush=True)
for _pair in (("FLCR_NOCAP_DV150 + jointimg", "R6CR_NOCAP_DV150 + jointimg"), ("FLCR_NOCAP_DV150 + joint", "R6CR_NOCAP_DV150 + joint"),
              ("FLCR_NOCAP_DV150 + jointimg", "CR_NOCAP_DV150 + jointimg"), ("R6CR_NOCAP_DV150 + jointimg", "CR_NOCAP_DV150 + jointimg")):
    if _pair[0] not in BS or _pair[1] not in BS:
        continue
    for nm_, bank in (("score", BS), ("adj_J", _BSA)):
        d = bank[_pair[0]] - bank[_pair[1]]
        lo, hi = float(np.quantile(d, .05)), float(np.quantile(d, .95))
        print(f"    {_pair[0]:<28} vs {_pair[1]:<28} {nm_:<6} {float(np.mean(d)):>+9.4f}   [{lo:>+8.4f}, {hi:>+8.4f}]  "
              f"{'YES' if (lo > 0 or hi < 0) else 'no'}", flush=True)
print("    reading: the flow-column claim holds only if FLCR beats R6CR with the interval off zero on adj_J (the linking term); "
      "FLCR vs CR says whether it also beats the shipped 199-movie ranker, which is what a submission would replace.", flush=True)
''')

rep('with open("/kaggle/working/forktp_dw55.csv", "w") as fh:', 'with open("/kaggle/working/forkfl_dw55.csv", "w") as fh:')
rep('    with open(f"/kaggle/working/forktp_{arm}.csv", "w") as fh:', '    with open(f"/kaggle/working/forkfl_{arm}.csv", "w") as fh:')
rep('    print("TWOPASS OK", flush=True)', '    print("FLOWRANK OK", flush=True)')

for bad in ("TPCR", "md_acc_tr", "len_back_tr", "NCOL_ALL + 6"):
    assert bad not in src, bad
assert src.count("def flow_field(") == 1 and src.count("def track_cols(") == 1 and src.count("def tr_for_keys(") == 1 and "FLCR_NOCAP_DV150" in src
open("flowrank.py", "w", encoding="utf-8", newline="\n").write(src)
print("written flowrank.py", src.count("\n"), "lines")
