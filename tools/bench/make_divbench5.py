"""divbench4.py -> divbench5.py: the same division bench on the OTHER imaging
batch -- all 71 44b6 train movies on F4 primaries.

WHAT THIS IS FOR, and what it is not. 44b6 carries about 26 GT divisions
against 6bba's 125, so div_J here is nearly unmeasurable and NOTHING may be
selected on it. What 44b6 can answer, and what the shipped gate has never
been asked:

  (1) does the 6bba-trained fork gate HURT the other batch? divbench3 said
      no on a 12-movie draw with 2 GT divisions (adj_J +0.0020, all 11 of
      its counted false forks gone). adj_J is measured over ~100k edges per
      movie, so on 71 movies that answer is precise even though div_J is
      not. This is the 15% of the score weight the gate has never been
      checked against at scale.
  (2) does the gate's RANKING transfer? Its out-of-fold AUC on 6bba is
      0.83-0.93. A 44b6 AUC near 0.5 would mean the gate is 6bba-specific
      and the LB gain came from the 85% alone.
  (3) is the RAW detection score the reason if it does not? gatelab
      (2026-09-07) found `sc_q` carries the gate almost alone and it is a
      raw score whose distribution moves with the imaging batch. divbench4
      added `sc_q_r` / `sc_p_r`, the per-movie quantile versions; 44b6 is
      the only place the difference between them can show.
  (4) it writes `forks_44b6.csv`, so a joint gate over 125 + 26 events can
      be built and bootstrapped offline (gatelab), not in a kernel.

RANKER LEAK. The production +BIDIR ranker's pool contains 59 of the 71, so
this bench cannot use it for the solve. Two out-of-fold rankers as in
divbench3, each trained on all 128 6bba + the other half of 44b6.

ANCHORS, both recorded: D187 at floor 0.10 on the 12 manifest-excluded 44b6
= F4-44b6 adj_J 0.8353 (divsteal, HARD); the production +BIDIR pkl at floor
0.48 on the same 12 = 0.8384 (divbench3's 44b6 REF row and ilpsweep's F48
row, soft -- those 12 are outside its pool).

No steal arms here: the 5a cases are still written out (a table costs
nothing), but 44b6 is slow and its division count cannot price a steal.
"""
import re

src = open("divbench4.py", encoding="utf-8").read()
NL = "\n"


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:80])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-07a-divbench4"', 'SCRIPT_VERSION = "2026-09-07b-divbench5"')

# ---- the bench is 44b6 now. Rename first so nothing reads backwards. -------
src = re.sub(r"\bSIX\b", "BENCH", src)
src = re.sub(r"\bFOUR\b", "OTHER", src)

rep('''BENCH = sorted(n for n in ALL_TRAIN if n.startswith("6bba_"))
OTHER = sorted(n for n in ALL_TRAIN if n.startswith("44b6_"))
assert len(BENCH) == 128 and len(OTHER) == 71, (len(BENCH), len(OTHER))''',
    '''BENCH = sorted(n for n in ALL_TRAIN if n.startswith("44b6_"))
OTHER = sorted(n for n in ALL_TRAIN if n.startswith("6bba_"))
assert len(BENCH) == 71 and len(OTHER) == 128, (len(BENCH), len(OTHER))
BENCH_SET = set(BENCH)''')
rep('_jobs = [(n, "44b6") for n in OTHER] + [(n, "6bba") for n in BENCH]',
    '_jobs = [(n, "6bba") for n in OTHER] + [(n, "44b6") for n in BENCH]')
rep('''    if role == "6bba" or name in SET_44B6:''',
    '''    if name in BENCH_SET:''')
rep('''          f"{' fold' + str(FOLD_OF[name]) if role == '6bba' else ''}: T={T} "''',
    '''          f"{' fold' + str(FOLD_OF[name]) if name in BENCH_SET else ''}: T={T} "''')
rep('''    _missing = [n for n in BENCH if n not in FDET_F4]
assert not _missing, f"no fdet_F4 cache for {len(_missing)} 6bba movies: {_missing[:5]}"
_missing = [n for n in BENCH if n not in EDGEPROB_F4]
assert not _missing, f"no edgeprob F4 cache for {len(_missing)} 6bba movies: {_missing[:5]}"'''
    .lstrip(),
    '''_missing = [n for n in BENCH if n not in FDET_F4]
assert not _missing, (f"no fdet_F4 cache for {len(_missing)} 44b6 movies: "
                      f"{_missing[:5]} -- attach biohub-fuse44b6 AND biohub-fusecache")
_missing = [n for n in BENCH if n not in EDGEPROB_F4]
assert not _missing, (f"no edgeprob F4 cache for {len(_missing)} 44b6 movies: "
                      f"{_missing[:5]} -- attach biohub-ep44b6 AND biohub-edgeprob")''')
rep('''=== FEATURES + JOIN: {len(_jobs)} passes, all 199 train movies on "
      f"BASE (xl) primaries + BASE edgeprob; every 6bba movie keeps its full "
      f"features for the solve ===''',
    '''=== FEATURES + JOIN: {len(_jobs)} passes, all 199 train movies on "
      f"BASE (xl) primaries + BASE edgeprob; every 44b6 movie keeps its full "
      f"F4 features for the solve ===''')
rep('''=== STAGE A: two out-of-fold +BIDIR rankers (18 + p_fwd p_rev h_fused; "
      "poolsize hyperparameters), each trained on the 71 44b6 movies + the "
      "other 64 6bba movies ===''',
    '''=== STAGE A: two out-of-fold +BIDIR rankers (18 + p_fwd p_rev h_fused; "
      "poolsize hyperparameters), each trained on the 128 6bba movies + the "
      "other half of 44b6 -- the production pkl's pool holds 59 of these 71, "
      "so it cannot solve this bench ===''')
rep('''          f"({len(held)} 6bba movies, {len(yh)} rows) AUC {held_auc:.4f}  "''',
    '''          f"({len(held)} 44b6 movies, {len(yh)} rows) AUC {held_auc:.4f}  "''')

# ---- no steal arms -------------------------------------------------------
rep('''STEAL_ARMS = [("steal_all", float("-inf")), ("steal_m0", 0.0), ("steal_m20", 0.20)]''',
    '''STEAL_ARMS = []       # 44b6 has ~26 GT divisions: it cannot price a steal.
                      # The 5a cases are still written out (steal5a_44b6.csv).''')

# ---- anchors: the 12 manifest-excluded 44b6, not the 6bba holdout ---------
rep('    if name in HOLDOUT_6BBA:', '    if name in SET_44B6:')
rep('''print("  ANCHOR: D187 on the 12-movie holdout at floor 0.10 must read adj_J 0.9216 "
      "(F4-6bba hard, div_J 0.0606). Soft: the production +BIDIR pkl at floor 0.48 "
      "on the same 12 = ilpsweep's F48 row 0.9261. REF over the 128 must read "
      "50/249/75, div_J 0.1337, adj_J 0.9133, score 0.9266 (divbench3).", flush=True)''',
    '''print("  ANCHOR: D187 at floor 0.10 on the 12 manifest-excluded 44b6 must read "
      "adj_J 0.8353 (F4-44b6 HARD, divsteal). Soft: the production +BIDIR pkl at "
      "floor 0.48 on the same 12 = 0.8384 (divbench3's 44b6 REF row, ilpsweep F48). "
      "REF over all 71 has never been measured -- it is the new reference.",
      flush=True)''')
rep('''if len(ANCHOR_ROWS) == len(HOLDOUT_6BBA):''', '''if len(ANCHOR_ROWS) == len(SET_44B6):''')
rep('''    ok = abs(got - 0.9216) <= ANCHOR_TOL
    print(f"  ANCHOR_D187  F4-6bba(12)  adj_J {got:.4f}  div_J {ra['division_jaccard']:.4f}"
          f"  want 0.9216 HARD (div_J 0.0606) -> {'OK' if ok else 'MISS'} ({got - 0.9216:+.4f})",
          flush=True)
    if not ok:
        ANCHOR_FAIL.append(("ANCHOR_D187", "F4-6bba", got, 0.9216))
    rs = _with_ratio(SOFT_ROWS)
    print(f"  +BIDIR pkl   F4-6bba(12)  adj_J {rs['adj_edge_jaccard']:.4f}  div_J "
          f"{rs['division_jaccard']:.4f}  want 0.9261 soft (ilpsweep F48) -> "
          f"{'OK' if abs(rs['adj_edge_jaccard'] - 0.9261) <= ANCHOR_TOL else 'MISS'}",
          flush=True)
else:
    ANCHOR_FAIL.append(("ANCHOR_D187", "F4-6bba", float("nan"), 0.9216))''',
    '''    ok = abs(got - 0.8353) <= ANCHOR_TOL
    print(f"  ANCHOR_D187  F4-44b6(12)  adj_J {got:.4f}  div_J {ra['division_jaccard']:.4f}"
          f"  want 0.8353 HARD (divsteal) -> {'OK' if ok else 'MISS'} ({got - 0.8353:+.4f})",
          flush=True)
    if not ok:
        ANCHOR_FAIL.append(("ANCHOR_D187", "F4-44b6", got, 0.8353))
    rs = _with_ratio(SOFT_ROWS)
    print(f"  +BIDIR pkl   F4-44b6(12)  adj_J {rs['adj_edge_jaccard']:.4f}  div_J "
          f"{rs['division_jaccard']:.4f}  want 0.8384 soft (divbench3 44b6 REF) -> "
          f"{'OK' if abs(rs['adj_edge_jaccard'] - 0.8384) <= ANCHOR_TOL else 'MISS'}",
          flush=True)
else:
    ANCHOR_FAIL.append(("ANCHOR_D187", "F4-44b6", float("nan"), 0.8353))''')
rep('''ref_ok = (TP, FP, FN) == (50, 249, 75)
print(f"  REF   F4-6bba(128) div {TP}/{FP}/{FN} div_J {base_J:.4f} adj_J "
      f"{RES['adj_edge_jaccard']:.4f} score {RES['score']:.4f}  want 50/249/75 "
      f"0.1337 0.9133 0.9266 (divbench3) -> {'OK' if ref_ok else 'MISS'}", flush=True)
if not ref_ok:
    ANCHOR_FAIL.append(("REF_FORKS", "F4-6bba", TP * 1000 + FP, 50 * 1000 + 249))''',
    '''print(f"  REF   F4-44b6({len(ROWS)}) div {TP}/{FP}/{FN} div_J {base_J:.4f} adj_J "
      f"{RES['adj_edge_jaccard']:.4f} score {RES['score']:.4f}  (NEW -- 6bba's 128 "
      f"read 50/249/75, 0.1337, 0.9133, 0.9266)", flush=True)''')
rep('''BASE_WALK = {"mother_absent": 9, "daughter_absent": 22, "5a_sister_not_orphan": 23,
             "8b_diverge_under_2.25um": 11,
             "9_passed_all_gates_then_capped_or_taken": 3, "recovered": 50}''',
    '''BASE_WALK = {"mother_absent": 9, "daughter_absent": 22, "5a_sister_not_orphan": 23,
             "8b_diverge_under_2.25um": 11,
             "9_passed_all_gates_then_capped_or_taken": 3, "recovered": 50}
# the db3 column is 6bba's 125 events, printed for shape only -- 44b6 has ~26''')

# ---- output names --------------------------------------------------------
rep('"/kaggle/working/steal5a.csv"', '"/kaggle/working/steal5a_44b6.csv"')
rep('"/kaggle/working/forks_f4_v4.csv"', '"/kaggle/working/forks_44b6.csv"')
rep('''=== EVERY FORK THE 128 REF GRAPHS CARRY: {len(FORKS)} (divbench3: 7,004); "
      f"verdict x source; forks_f4_v4.csv + one csv per steal arm written ===''',
    '''=== EVERY FORK THE {len(ROWS)} 44b6 REF GRAPHS CARRY: {len(FORKS)} "
      f"(6bba's 128 carried 7,004); verdict x source; forks_44b6.csv written ===''')
rep('''for tag, cols, names_ in (("f4v4_20", J20, FEAT20),
                          ("f4v4_28", list(range(len(FEAT_NAMES))), FEAT_NAMES)):''',
    '''for tag, cols, names_ in (("44b6_20", J20, FEAT20),
                          ("44b6_28", list(range(len(FEAT_NAMES))), FEAT_NAMES)):''')
rep('''f"  fork_gate_{tag}.pkl written; p quantiles over all {len(FORKS)} forks: "''',
    '''f"  fork_gate_{tag}.pkl written (44b6-trained); p quantiles over all "
          f"{len(FORKS)} forks: "''')

# ---- THE question: the 6bba gate on 44b6 forks ----------------------------
TRANSFER = r'''
# ---- the 6bba gate on 44b6 forks: the transfer test ------------------------
print("\n=== THE 6bba-TRAINED GATE APPLIED TO 44b6 FORKS (the question this bench "
      "exists for). fork_gate_f4.pkl is the PRODUCTION gate -- divbench3's, trained "
      "on all 299 counted 6bba forks, shipped in submit_gate.py, LB 0.943. It has "
      "seen no 44b6 movie. divbench4's candidates are used too IF they happen to be "
      "mounted; they are not required ===", flush=True)
SIXGATE = {}
for tag in ("f4", "f4v4_20", "f4v4_28"):
    hit = _walk_files(lambda fn, _t=tag: fn if fn == f"fork_gate_{_t}.pkl" else None)
    if not hit:
        warn_once(f"no_{tag}", f"fork_gate_{tag}.pkl not mounted -- that gate's "
                               f"transfer test is SKIPPED, not silently replaced")
        continue
    with open(next(iter(hit.values())), "rb") as fh:
        SIXGATE[tag] = pickle.load(fh)
    print(f"  loaded fork_gate_{tag}.pkl (version "
          f"{SIXGATE[tag].get('version')}, {len(SIXGATE[tag]['feature_names'])} features)",
          flush=True)

if not SIXGATE:
    warn_once("no_gate_at_all", "NO 6bba gate was mounted -- the transfer test, the "
                                "whole point of this bench, did not run")
P6 = {}
for tag, blob in SIXGATE.items():
    names_ = blob["feature_names"]
    missing = [f for f in names_ if f not in FEAT_NAMES]
    assert not missing, f"the 6bba gate wants features this run does not build: {missing}"
    cols = [FEAT_NAMES.index(f) for f in names_]
    P6[tag] = blob["model"].predict_proba(X_f[:, cols])[:, 1]
    if counted.sum() and (lab[counted] == 2).any() and (lab[counted] == 1).any():
        a6 = _auc((lab[counted] == 2).astype(int), P6[tag][counted])
    else:
        a6 = float("nan")
    print(f"\n  {tag}: AUC on 44b6's {int(counted.sum())} counted forks "
          f"({int((lab == 2).sum())} TP / {int((lab == 1).sum())} FP): {a6:.4f}"
          f"   -- the same gate read 0.9283 / 0.8326 out of fold on 6bba", flush=True)
    print(HDR, flush=True)
    _curve_line(np.ones(len(lab), bool), "none")
    for thr in (0.01, 0.02, 0.03, 0.05, 0.1):
        _curve_line(P6[tag] > thr, f"6bba gate, fixed p > {thr}")
    q = blob.get("p_quantiles_all_forks", {})
    for f_, v_ in sorted(q.items()):
        _curve_line(P6[tag] > v_, f"6bba gate, its own q{int(float(f_) * 100)} (p > {v_:.4f})")
    print(f"  share of 44b6's {len(FORKS)} forks the cut p > 0.03 would remove: "
          f"{100.0 * float((P6[tag] <= 0.03).mean()):.1f}%  (on 6bba it removed 80%, "
          f"and on the submit-gate visible 4 it removed 85%)", flush=True)

print("\n=== PRUNING APPLIED FOR REAL on the 44b6 graphs with the 6bba gate, official "
      "scorer -- adj_J is the readout here, div_J has ~26 events behind it ===",
      flush=True)
print(f"  {'cut':<34} {'div tp/fp/fn':>14} {'div_J':>7} {'adj_J':>7} {'edge_J':>7} "
      f"{'node_r':>8} {'score':>7} {'d adj_J':>8} {'d score':>8} {'f0 dadj':>8} "
      f"{'f1 dadj':>8}", flush=True)
fork_by_movie = defaultdict(list)
for i_, r_ in enumerate(FORKS):
    fork_by_movie[r_[0]].append((i_, r_[2], r_[3]))
f0_ref = _with_ratio([PER[n] for n in BENCH if FOLD_OF[n] == 0 and n in PER])
f1_ref = _with_ratio([PER[n] for n in BENCH if FOLD_OF[n] == 1 and n in PER])
print(f"  {'REF (no gate)':<34} {TP:>4}/{FP:>4}/{FN:<4} {base_J:>7.4f} "
      f"{RES['adj_edge_jaccard']:>7.4f} {RES.get('edge_jaccard', float('nan')):>7.4f} "
      f"{RES.get('total_node_ratio', float('nan')):>+8.4f} {RES['score']:>7.4f} "
      f"{0.0:>+8.4f} {0.0:>+8.4f} {0.0:>+8.4f} {0.0:>+8.4f}", flush=True)
for tag in list(P6):
    for thr in (0.03,):
        rows_c, pf = [], {0: [], 1: []}
        for name in BENCH:
            if name not in MOVIE:
                continue
            key = MOVIE[name]["key"]
            want = [(p_, q_) for i_, p_, q_ in fork_by_movie[name] if P6[tag][i_] <= thr]
            miss = [e_ for e_ in want if e_ not in key]
            if miss:
                warn_once("prune_edge_missing",
                          f"{name}: {len(miss)} fork edges to prune are not in the "
                          f"snapshot's fork map -> NOT removed, the cut is a LOWER bound")
            drop = {key[e_] for e_ in want if e_ in key}
            row = _rescore(name, MOVIE[name], drop, f"{tag}_p{thr}") if drop else PER[name]
            rows_c.append(row)
            pf[FOLD_OF[name]].append(row)
        rc = _with_ratio(rows_c)
        f0, f1 = _with_ratio(pf[0]), _with_ratio(pf[1])
        print(f"  {tag + ' p>' + str(thr):<34} "
              f"{sum(r['division_tp'] for r in rows_c):>4}/"
              f"{sum(r['division_fp'] for r in rows_c):>4}/"
              f"{sum(r['division_fn'] for r in rows_c):<4} {rc['division_jaccard']:>7.4f} "
              f"{rc['adj_edge_jaccard']:>7.4f} "
              f"{rc.get('edge_jaccard', float('nan')):>7.4f} "
              f"{rc.get('total_node_ratio', float('nan')):>+8.4f} {rc['score']:>7.4f} "
              f"{rc['adj_edge_jaccard'] - RES['adj_edge_jaccard']:>+8.4f} "
              f"{rc['score'] - RES['score']:>+8.4f} "
              f"{f0['adj_edge_jaccard'] - f0_ref['adj_edge_jaccard']:>+8.4f} "
              f"{f1['adj_edge_jaccard'] - f1_ref['adj_edge_jaccard']:>+8.4f}", flush=True)
print("  (divbench3's 12-movie 44b6 draw read adj_J 0.8384 -> 0.8404, +0.0020, with "
      "all 11 of its counted false forks gone and its 2 GT divisions already lost)",
      flush=True)
'''
a = src.index('print("\\n=== reading it ===", flush=True)')
a = src.rfind(NL, 0, a) + 1
src = src[:a] + TRANSFER.lstrip(NL) + NL + src[a:]

rep('''print("  (1) The false-fork side: the eight new features are read from their single-"
      "feature AUCs and from the 28-vs-20 gate rows at the same cut, on the same "
      "forks. Offline on divbench3's table the paired bootstrap over movies gives a "
      "90% interval of about +-0.03 on div_J, so a 28-feature gate has to beat 20 by "
      "more than that before it is anything but noise; the pre-registered reading is "
      "that nothing under +0.03 div_J counts, whatever its sign.", flush=True)
print("  (2) The steal: a submission candidate iff its 'd score' is positive in BOTH "
      "folds against REF+gate at the same cut. The unconditional arm is a control, "
      "not a candidate -- if it comes back near zero that is a reason to look for the "
      "cancellation between its good and bad cases, not to close the line.", flush=True)
print("  (3) The 5a table says whether the mechanism can work at all: if the mother's "
      "edge almost never beats the ILP's, the margin rule fires on nothing and the "
      "lever is the LINKER, not the patch.", flush=True)''',
    '''print("  (1) The gate SHIPS. The question is not whether to keep it but whether it "
      "costs 44b6 anything: read adj_J and its two per-fold deltas in the real-pruning "
      "table, over 71 movies and ~7 million edges. div_J on ~26 events is context, not "
      "evidence -- do not select on it.", flush=True)
print("  (2) Transfer: the 6bba gate's AUC on 44b6's counted forks against 6bba's own "
      "0.83-0.93 out of fold. Near 0.5 means the gate is 6bba-specific and the LB "
      "+0.010 came from the 85% alone; the 20-vs-28 pair says whether the per-movie "
      "score RANKS travel better than the raw scores, which is the one mechanism named "
      "in advance (gatelab, 2026-09-07).", flush=True)
print("  (3) forks_44b6.csv is the deliverable for a joint gate: 125 + ~26 events, "
      "built and bootstrapped offline, not in a kernel.", flush=True)''')

rep('DIVBENCH4 DONE in', 'DIVBENCH5 DONE in')
rep('print("DIVBENCH4 OK", flush=True)', 'print("DIVBENCH5 OK", flush=True)')
rep('f"[degraded] DIVBENCH4 INCOMPLETE:', 'f"[degraded] DIVBENCH5 INCOMPLETE:')

# ---- v1 post-mortem: 44b6 has 67 counted forks and SIX true ones ----------
# The internal gate cannot be trained on that, which is the expected answer,
# but it crashed the run before the transfer test. Make it a printed finding.
rep('''    ok_g = np.isfinite(p_g)
    _curve_line(ok_g, "none")
    for thr in (0.01, 0.02, 0.03, 0.05, 0.1):
        _curve_line(ok_g & (p_g > thr), f"fixed p > {thr}")
    for frac in (0.7, 0.8, 0.9):
        t_ = np.quantile(p_g[ok_g], frac)
        _curve_line(ok_g & (p_g > t_), f"quantile {int(frac * 100)}% (p > {t_:.4f})")''',
    '''    ok_g = np.isfinite(p_g)
    if not ok_g.any():
        print(f"  no out-of-fold probability exists for a single fork: every fold "
              f"was refused for want of true forks. NOT an error -- this bench has "
              f"{int((lab == 2).sum())} of them. Skipping the curve.", flush=True)
        continue
    _curve_line(ok_g, "none")
    for thr in (0.01, 0.02, 0.03, 0.05, 0.1):
        _curve_line(ok_g & (p_g > thr), f"fixed p > {thr}")
    for frac in (0.7, 0.8, 0.9):
        t_ = np.quantile(p_g[ok_g], frac)
        _curve_line(ok_g & (p_g > t_), f"quantile {int(frac * 100)}% (p > {t_:.4f})")''')

# a locally-trained production gate on six true forks is not a model
rep('''for tag, cols, names_ in (("44b6_20", J20, FEAT20),
                          ("44b6_28", list(range(len(FEAT_NAMES))), FEAT_NAMES)):
    mg_full = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05,
                                             min_samples_leaf=10, random_state=0)''',
    '''_N_TP = int((lab == 2).sum())
for tag, cols, names_ in (("44b6_20", J20, FEAT20),
                          ("44b6_28", list(range(len(FEAT_NAMES))), FEAT_NAMES)):
    if _N_TP < 10 or int(counted.sum()) < 40:
        print(f"  fork_gate_{tag}.pkl NOT written: {_N_TP} true forks and "
              f"{int(counted.sum())} counted ones is not a training set. A 44b6 gate "
              f"needs the 6bba forks too, and that join is a gatelab job on "
              f"forks_44b6.csv + forks_f4_v4.csv, not a kernel.", flush=True)
        continue
    mg_full = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05,
                                             min_samples_leaf=10, random_state=0)''')

# labels the SIX -> BENCH rename could not reach, because they are inside strings
rep('''print(f"\\n=== STAGE B: {len(BENCH)} 6bba movies on F4 primaries, one production solve "''',
    '''print(f"\\n=== STAGE B: {len(BENCH)} 44b6 movies on F4 primaries, one production solve "''')
rep('''print("\\n=== THE PATCH\'S OWN BUDGET, per arm, summed over the 128 movies "''',
    '''print(f"\\n=== THE PATCH\'S OWN BUDGET, per arm, summed over the {len(BENCH)} movies "''')
rep('''      f"AWAY ({len(STEAL5A)} cases, F4, 128 6bba movies) ===", flush=True)''',
    '''      f"AWAY ({len(STEAL5A)} cases, F4, {len(ROWS)} 44b6 movies) ===", flush=True)''')

open("divbench5.py", "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
