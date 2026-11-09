"""lntrace.py -> lnrank.py: the RANKER RETRAINED WITH THE RESCUE PEAKS in its training rows.

WHY (2026-09-17). lntrace: of the rescue peaks' 256 linkable GT edges, 128 die at the ranker
(p median 0.07, nine in ten below 0.31) and only 2 at the ILP. The shipped fold rankers are
trained on BASE-primary rows, where a dim annotated cell that the detector never found never
appears as a positive; so a pair (shipped neighbour -> rescue peak), dim to dim, is a kind of
row the ranker has never been shown as true. Before any detector work can pay, this has to be
settled: if a ranker that HAS seen such rows still refuses them (or accepts too many noise
peaks), recovered cells cannot be linked and the detector line is moot; if it links them, the
detector line has a path.

WHAT. lntrace's harness (128 6bba, REF anchors, arm LN20 with the trace) plus:
  * in the feature stage, for every 6bba movie, labelled rows on the RESCUE set (F4HP + lnrescue
    peaks) restricted to pairs whose two primaries are admitted at score >= 0.20 (LABLN);
  * stage A trains a second pair of out-of-fold rankers, MODELS_LN, on the 71 44b6 BASE rows +
    the other fold's LABLN rows (same estimator, same columns), and prints the held-out AUC on
    the held fold's LABLN and LAB4 rows;
  * arm LN20R = the rescue set solved with MODELS_LN (LN20 keeps the shipped rankers), both traced:
    the trace funnel per arm says whether C_below_floor moved to G_tp, and the official score with
    the paired bootstrap says what the noise peaks cost.
Confound stated up front: LN20R's rankers are trained on F4HP-placed rows with peaks, the shipped
ones on BASE rows; rkprod measured the F4-rows-only retrain at +0.0023 local / flat on the LB.

Attach: lnbench's sources. CPU ~6.5 h.
"""
import os

src = open("lntrace.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-17a-lntrace"', 'SCRIPT_VERSION = "2026-09-18a-lnrank"')
rep('DUP_ARMS = [("LN20", dict(tau=0.2, persist=False))]',
    'DUP_ARMS = [("LN20", dict(tau=0.2, persist=False)), ("LN20R", dict(tau=0.2, persist=False, model="LN"))]')

# ---- move the lnbench + lntrace blocks before the feature stage (ln_load / ln_keep are needed there) ----
i0 = src.index("# ---- lnbench block ----")
i1 = src.index("# ---- end lntrace block ----") + len("# ---- end lntrace block ----\n")
blocks = src[i0:i1]
src = src[:i0] + src[i1:]
rep('\n_jobs = [(n, "44b6") for n in FOUR] + [(n, "6bba") for n in SIX]\n',
    "\n" + blocks + '\nLABLN = {}                 # name -> (X, y) labelled rows on the rescue set, admitted pairs only, 6bba\n'
    '_jobs = [(n, "44b6") for n in FOUR] + [(n, "6bba") for n in SIX]\n')

# ---- label_rows gets a keep filter ------------------------------------------------------------------------
rep('def label_rows(name, feats):', 'def label_rows(name, feats, keep=None):')
rep('''        for (i, j), row in zip(pairs, rows):
            if i not in m_p:
                continue
            kids = child_of.get(m_p[i], [])''',
    '''        for (i, j), row in zip(pairs, rows):
            if keep is not None and not (keep[t][i] and keep[t + 1][j]):
                continue
            if i not in m_p:
                continue
            kids = child_of.get(m_p[i], [])''')

# ---- the feature stage: rescue rows per 6bba movie -------------------------------------------------------
rep('''            LAB4[name] = label_rows(name, feats4)
''', '''            LAB4[name] = label_rows(name, feats4)
            _dln, _nln = ln_load(name)
            _sln, _fln = features_ext(name, _dln, T4, LN_EP1[name],
                                      LN_EP2.get(name) if EP2_OF["F4"].get(name) is not None else None)
            _kln = ln_keep(_sln, _nln, 0.20)[0]
            LABLN[name] = label_rows(name, _fln, keep=_kln)
            n4 = (f" LN rows={len(LABLN[name][1])} pos={int(LABLN[name][1].sum())}"
                  f" (F4 rows={len(LAB4[name][1])} pos={int(LAB4[name][1].sum())})")
            del _dln, _nln, _sln, _fln, _kln
''')

# ---- stage A: the rescue-trained fold rankers --------------------------------------------------------------
rep('''del LAB
gc.collect()
''', '''print("\\n=== STAGE A2: the same two out-of-fold rankers trained with the RESCUE rows (44b6 BASE rows + the other "
      "fold's 6bba rows on the rescue set, pairs among primaries admitted at score >= 0.20) ===", flush=True)
MODELS_LN = {}
for k in (0, 1):
    t0 = time.time()
    train_names = FOUR + [n for n in SIX if FOLD_OF[n] != k]
    held = [n for n in SIX if FOLD_OF[n] == k]
    Xa = np.concatenate([(LAB[n] if n in FOUR else LABLN[n])[0] for n in train_names]).astype(np.float32)[:, COLS]
    ya = np.concatenate([(LAB[n] if n in FOUR else LABLN[n])[1] for n in train_names])
    m = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    m.fit(Xa, ya)
    Xh = np.concatenate([LABLN[n][0] for n in held]).astype(np.float32)[:, COLS]
    yh = np.concatenate([LABLN[n][1] for n in held])
    Xh4 = np.concatenate([LAB4[n][0] for n in held]).astype(np.float32)[:, COLS]
    yh4 = np.concatenate([LAB4[n][1] for n in held])
    MODELS_LN[k] = m
    print(f"  fold{k}: trained on {len(train_names)} movies / {len(ya)} rows ({ya.mean():.3f} pos; the shipped rankers' "
          f"6bba rows {sum(len(LAB[n][1]) for n in train_names if n not in FOUR)}); held-out fold on rescue rows "
          f"({len(yh)} rows) AUC {_auc(yh, m.predict_proba(Xh)[:, 1]):.4f}, shipped ranker on the same rows "
          f"{_auc(yh, MODELS_FOLD[k].predict_proba(Xh)[:, 1]):.4f}; on F4HP rows ({len(yh4)}) {_auc(yh4, m.predict_proba(Xh4)[:, 1]):.4f} "
          f"vs shipped {_auc(yh4, MODELS_FOLD[k].predict_proba(Xh4)[:, 1]):.4f}  {time.time() - t0:.0f}s", flush=True)
    del Xa, ya, Xh, yh, Xh4, yh4
    gc.collect()
_nln = sum(len(LABLN[n][1]) for n in LABLN)
_pln = sum(int(LABLN[n][1].sum()) for n in LABLN)
_n4 = sum(len(LAB4[n][1]) for n in LAB4)
_p4 = sum(int(LAB4[n][1].sum()) for n in LAB4)
print(f"  6bba rows: rescue set {_nln:,} ({_pln:,} positive) vs F4HP set {_n4:,} ({_p4:,} positive): "
      f"{_nln - _n4:+,} rows, {_pln - _p4:+,} positives come from the admitted rescue peaks", flush=True)
del LAB, LABLN
gc.collect()
''')

# ---- stage B: the arm's rankers, the trace per arm ---------------------------------------------------------
rep('        n_a, e_a, p_a, aux_a = solve_one(dets_ln, T, seeds_ln, feats_ln, MODELS_FOLD[k], COLS,',
    '        n_a, e_a, p_a, aux_a = solve_one(dets_ln, T, seeds_ln, feats_ln, MODELS_LN[k] if cfg.get("model") == "LN" else MODELS_FOLD[k], COLS,')
rep('        LN_STAT[arm]["trace_rows"] += ln_trace(name, T, seeds_ln, nnew_ln, keep, pm_a, so_a)',
    '        LN_STAT[arm]["trace_rows"] += ln_trace(name, T, seeds_ln, nnew_ln, keep, pm_a, so_a, arm)')
rep('def ln_trace(name, T, seeds, nnew, keep, pmap, son):', 'def ln_trace(name, T, seeds, nnew, keep, pmap, son, arm="LN20"):')
rep('LT_COLS = ["dataset", "t", "q_i",', 'LT_COLS = ["arm", "dataset", "t", "q_i",')
rep('                LT_ROWS.append([name, t, qi, float(s_[qi]),', '                LT_ROWS.append([arm, name, t, qi, float(s_[qi]),')

# lt_summary: the report per arm
rep('''    df = _pd.DataFrame(LT_ROWS, columns=LT_COLS)
    df.to_csv("/kaggle/working/lntrace_edges.csv", index=False)
    cats = [''', '''    df_all = _pd.DataFrame(LT_ROWS, columns=LT_COLS)
    df_all.to_csv("/kaggle/working/lnrank_edges.csv", index=False)
    for arm_ in sorted(df_all.arm.unique()):
        print(f"\\n  --- arm {arm_} ({'the rescue-trained rankers' if arm_.endswith('R') else 'the shipped rankers'}) ---", flush=True)
        _lt_report(df_all[df_all.arm == arm_].reset_index(drop=True))


def _lt_report(df):
    cats = [''')
rep('print("LNTRACE OK", flush=True)', 'print("LNRANK OK", flush=True)')
rep('print(f"[degraded] LNTRACE INCOMPLETE or anchor miss:', 'print(f"[degraded] LNRANK INCOMPLETE or anchor miss:')
rep('f"lntrace_REF_{DET_SRC}.csv"', 'f"lnrank_REF_{DET_SRC}.csv"')
rep('f"lntrace_arm_{a_}.csv"', 'f"lnrank_arm_{a_}.csv"')

assert src.index("def ln_load(") < src.index('_jobs = [(n, "44b6")') and src.count("MODELS_LN[k]") == 2
assert src.count("def _lt_report(df):") == 1 and "lntrace_edges.csv" not in src
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lnrank.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written lnrank.py", len(src))
