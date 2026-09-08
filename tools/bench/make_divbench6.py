"""divbench4.py -> divbench6.py: reopen the ILP's own fork price now that a
fork gate exists to clean up behind it.

THE ARGUMENT, in one paragraph. `DIV_W = 0.55` decides when the solver takes
an ORPHAN daughter: with `edge_weight = -1.0 * edge_prob` and
`appearance_weight = 0.0`, it forks iff `p_mother_orphan > DIV_W`. It was set
to 0.55 by `biohub-ranker-fine` on 2026-08-23 as the argmax of a curve
measured on TWELVE division events, and that entry says in its own words that
"0.45-0.65 are all defensible and 0.55 is simply the best measured". Lower
values lost by OVER-FORKING -- 1,024 forks and 57 counted false ones at 0.20,
adj_J down 0.0027 -- and over-forking is precisely and only what the fork
gate removes: divbench3 cuts 80% of forks while keeping 84% of the true ones,
and it was worth +0.010 at LB. There was no gate in August. Two more reasons
from divbench3's own table: the solver's forks are MORE precise than the
patch's (0.196 against 0.150 on counted forks) and it makes far fewer of them
(2,490 against 4,514), so lowering the price moves fork production from the
geometric patch to the better-calibrated source.

WHAT THIS IS NOT. `DIV_W` does not touch the 5a case. When the daughter
already has a parent, taking her ends that parent's track at
`disappearance_weight` 1.5, so the fork needs `p_M - p_R > DIV_W + 1.5 =
2.05` and probabilities stop at 1: 5a is structural. divbench4's steal pass
is the instrument for that, and `ilpsweep` already measured loosening
disappearance (1.0 lost 12 of 12 on 6bba).

DESIGN. `DIV_W` is read by `solve_one` when the ILPSolver is built, so the
2026-08-28 rule forbids putting it in a post-link sweep -- every arm needs
its own solve. Four solves per movie (REF + three), 128 movies. Each arm is
patched, scored by the official scorer, and its forks are gated by a gate
trained OUT OF FOLD ON THAT ARM'S OWN FORKS, because the fork population is
what the arm changes. Read the arms AFTER the gate: that is the shipped
pipeline.

ANCHOR: the REF arm is div_w 0.55 and must reproduce divbench4 = divbench3 --
50/249/75, div_J 0.1337, adj_J 0.9133, score 0.9266, plus D187 F4-6bba(12)
0.9216 HARD.
"""
src = open("divbench4.py", encoding="utf-8").read()
NL = "\n"


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:80])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-07a-divbench4"', 'SCRIPT_VERSION = "2026-09-07c-divbench6"')

# ---- DIV_W becomes a parameter of the solve ------------------------------
rep('''def solve_one(dets, T, seeds, feats, model, cols, disapp=None, floor=None):''',
    '''def solve_one(dets, T, seeds, feats, model, cols, disapp=None, floor=None,
              divw=None):''')
rep('''            division_weight=DIV_W)''',
    '''            division_weight=(DIV_W if divw is None else divw))''')

# ---- the arms ------------------------------------------------------------
rep('''# (arm name, ranker margin the mother's edge must beat the ILP's by). -inf is
# the unconditional control: it measures the SUM of its good and bad cases,
# which is not a ceiling (2026-08-26), and it says whether the gates bind.
STEAL_ARMS = [("steal_all", float("-inf")), ("steal_m0", 0.0), ("steal_m20", 0.20)]''',
    '''STEAL_ARMS = []       # divbench6 sweeps the SOLVER, not the patch
# (arm name, DIV_W). The shipped value is 0.55; ranker-fine's 12-event curve
# read forks 111 / 190 / 394 / 1024 at 0.55 / 0.45 / 0.35 / 0.20, so these
# three span an order of magnitude in fork production. 0.65 and above is not
# swept: that direction makes FEWER forks and the gate cannot help there.
DIVW_ARMS = [("dw45", 0.45), ("dw35", 0.35), ("dw20", 0.20)]''')

# ---- stage B: each arm is its own solve ----------------------------------
rep('''    cells = []
    for arm, margin in STEAL_ARMS:
        outa = pathlib.Path(f"{TMP}/arm_{arm}") / f"{name}.geff"
        outa.parent.mkdir(parents=True, exist_ok=True)
        PATCH_CAPS.clear()
        patch_final(nodes0, edges0, pool, outa, pfun=pfun, steal=margin)
        for k_, v_ in PATCH_CAPS.items():
            CAPS_BY_ARM.setdefault(arm, defaultdict(int))[k_] += v_
        n_st = len(STEAL_LOG)
        row_a, ds_a, fr_a = _judge(name, k, outa, seeds, aux)
        ARM_ROWS[arm].append(row_a)
        ARM_PER[arm][name] = row_a
        ARM_FORKS[arm].extend(fr_a)
        ARM_MOVIE[arm][name] = _snapshot()
        cells.append(f"{arm} +{n_st} {row_a['division_tp']}/{row_a['division_fp']}/"
                     f"{row_a['division_fn']} {_adj_of(row_a):.4f}")
        del ds_a''',
    '''    cells = []
    for arm, dw in DIVW_ARMS:
        outa = pathlib.Path(f"{TMP}/arm_{arm}") / f"{name}.geff"
        outa.parent.mkdir(parents=True, exist_ok=True)
        n_a, e_a, p_a, aux_a = solve_one(dets, T, seeds, feats, MODELS_FOLD[k], COLS,
                                         disapp=ILP_DISAPP, floor=PROD_FLOOR, divw=dw)
        pm_a, so_a = aux_a
        sc_a, b2_a = _aux_extra(seeds, pm_a)
        PATCH_CAPS.clear()
        patch_final(n_a, e_a, p_a, outa)
        for k_, v_ in PATCH_CAPS.items():
            CAPS_BY_ARM.setdefault(arm, defaultdict(int))[k_] += v_
        row_a, ds_a, fr_a = _judge(name, k, outa, seeds, (pm_a, so_a, sc_a, b2_a))
        ARM_ROWS[arm].append(row_a)
        ARM_PER[arm][name] = row_a
        ARM_FORKS[arm].extend(fr_a)
        ARM_MOVIE[arm][name] = _snapshot()
        n_ilp = sum(1 for r_ in fr_a if r_[5][0] == 0)
        cells.append(f"{arm} forks {len(fr_a)}({n_ilp} ilp) "
                     f"{row_a['division_tp']}/{row_a['division_fp']}/"
                     f"{row_a['division_fn']} {_adj_of(row_a):.4f}")
        del ds_a, n_a, e_a, p_a, aux_a, pm_a, so_a, sc_a, b2_a''')
rep('''ARM_FORKS = {a: [] for a, _ in STEAL_ARMS}
ARM_ROWS = {a: [] for a, _ in STEAL_ARMS}
ARM_PER = {a: {} for a, _ in STEAL_ARMS}
ARM_MOVIE = {a: {} for a, _ in STEAL_ARMS}''',
    '''ARM_FORKS = {a: [] for a, _ in DIVW_ARMS}
ARM_ROWS = {a: [] for a, _ in DIVW_ARMS}
ARM_PER = {a: {} for a, _ in DIVW_ARMS}
ARM_MOVIE = {a: {} for a, _ in DIVW_ARMS}''')
rep('''      f"each (out-of-fold +BIDIR, disappearance {ILP_DISAPP}, floor {PROD_FLOOR}); "
      f"REF patch + {len(STEAL_ARMS)} steal arms, every fork logged and judged, every "
      f"lost division walked and every 5a case written out ===", flush=True)''',
    '''      f"each at DIV_W {DIV_W} (out-of-fold +BIDIR, disappearance {ILP_DISAPP}, "
      f"floor {PROD_FLOOR}) plus {len(DIVW_ARMS)} arms that RE-SOLVE at "
      f"{', '.join(str(d) for _a, d in DIVW_ARMS)}; every fork logged and judged ===",
      flush=True)''')

# ---- the arm report: per-arm out-of-fold gate, then the real pruning -------
OLD_A = '''print(f"\\n=== THE STEAL ARMS: the patch allowed to take a daughter the ILP had already '''
a = src.index(OLD_A)
a = src.rfind(NL, 0, a) + 1
b = src.index('print("\\n=== reading it ===", flush=True)')
b = src.rfind(NL, 0, b) + 1
ARMS_BLOCK = r'''print(f"\n=== THE DIV_W ARMS, UNGATED (each is its own ILP solve; official scorer, "
      f"all {len(SIX_OK)} movies) ===", flush=True)
print(f"  {'arm':<16} {'forks':>7} {'ilp':>7} {'patch':>7} {'div tp/fp/fn':>14} {'div_J':>7} "
      f"{'adj_J':>7} {'edge_J':>7} {'node_r':>8} {'score':>7} {'d adj_J':>8} {'d div_J':>8} "
      f"{'d score':>8} {'f0 dsc':>7} {'f1 dsc':>7}", flush=True)
b_ref = RES
f0_ref = _with_ratio([PER[n] for n in SIX_OK if FOLD_OF[n] == 0])
f1_ref = _with_ratio([PER[n] for n in SIX_OK if FOLD_OF[n] == 1])


def _armline(tag, rows_a, forks_a, per_a, dtp=None):
    rc = _with_ratio(rows_a)
    f0 = _with_ratio([per_a[n] for n in SIX_OK if FOLD_OF[n] == 0 and n in per_a])
    f1 = _with_ratio([per_a[n] for n in SIX_OK if FOLD_OF[n] == 1 and n in per_a])
    nf = len(forks_a) if forks_a is not None else -1
    ni = sum(1 for r_ in forks_a if r_[5][0] == 0) if forks_a else 0
    print(f"  {tag:<16} {nf:>7,} {ni:>7,} {nf - ni:>7,} "
          f"{sum(r['division_tp'] for r in rows_a):>4}/"
          f"{sum(r['division_fp'] for r in rows_a):>4}/"
          f"{sum(r['division_fn'] for r in rows_a):<4} "
          f"{rc['division_jaccard']:>7.4f} {rc['adj_edge_jaccard']:>7.4f} "
          f"{rc.get('edge_jaccard', float('nan')):>7.4f} "
          f"{rc.get('total_node_ratio', float('nan')):>+8.4f} {rc['score']:>7.4f} "
          f"{rc['adj_edge_jaccard'] - b_ref['adj_edge_jaccard']:>+8.4f} "
          f"{rc['division_jaccard'] - base_J:>+8.4f} "
          f"{rc['score'] - b_ref['score']:>+8.4f} "
          f"{f0['score'] - f0_ref['score']:>+7.4f} {f1['score'] - f1_ref['score']:>+7.4f}",
          flush=True)
    return rc


_armline(f"REF (dw {DIV_W})", ROWS, FORKS, PER)
for arm, dw in DIVW_ARMS:
    _armline(f"{arm} (dw {dw})", ARM_ROWS[arm], ARM_FORKS[arm], ARM_PER[arm])

print(f"\n=== THE SAME ARMS AFTER THE FORK GATE (trained out of fold on THAT ARM'S own "
      f"counted forks, 20 features, 2 folds -- the arm changes the fork population, so "
      f"the gate has to be refitted per arm) ===", flush=True)
print(f"  {'arm / cut':<24} {'div tp/fp/fn':>14} {'div_J':>7} {'adj_J':>7} {'score':>7} "
      f"{'d adj_J':>8} {'d score':>8} {'vs REF+gate':>12} {'f0':>8} {'f1':>8}", flush=True)


def _gate_and_prune(tag, forks_a, movies_a, per_a, thr):
    """Fit a 2-fold gate on this arm's counted forks, prune, rescore."""
    Xa = np.array([r_[5] for r_ in forks_a], float) if forks_a else np.zeros((0, len(FEAT_NAMES)))
    la = np.array([r_[4] for r_ in forks_a], int)
    fa = np.array([r_[1] for r_ in forks_a], int)
    pa = np.full(len(forks_a), np.nan)
    cnt = la >= 1
    aucs = []
    for kk in (0, 1):
        tr = (fa != kk) & cnt
        if tr.sum() < 20 or (la[tr] == 2).sum() < 5:
            warn_once("arm_gate_small", f"{tag}: fold{kk} had {int(tr.sum())} counted forks")
            continue
        mg = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05,
                                            min_samples_leaf=10, random_state=0)
        mg.fit(Xa[tr][:, J20], (la[tr] == 2).astype(int))
        te = fa == kk
        pa[te] = mg.predict_proba(Xa[te][:, J20])[:, 1]
        tc = te & cnt
        if tc.sum() and (la[tc] == 2).any() and (la[tc] == 1).any():
            aucs.append((int(tc.sum()), _auc((la[tc] == 2).astype(int), pa[tc])))
    by_movie = defaultdict(list)
    for i_, r_ in enumerate(forks_a):
        by_movie[r_[0]].append((i_, r_[2], r_[3]))
    rows_c, pf = [], {0: [], 1: []}
    for name in SIX_OK:
        key = movies_a[name]["key"]
        want = [(p_, q_) for i_, p_, q_ in by_movie[name]
                if np.isfinite(pa[i_]) and pa[i_] <= thr]
        miss = [e_ for e_ in want if e_ not in key]
        if miss:
            warn_once("prune_edge_missing",
                      f"{name}: {len(miss)} fork edges to prune are not in the "
                      f"snapshot's fork map -> NOT removed, the cut is a LOWER bound")
        drop = {key[e_] for e_ in want if e_ in key}
        row = _rescore(name, movies_a[name], drop, f"{tag}_p{thr}") if drop else per_a[name]
        rows_c.append(row)
        pf[FOLD_OF[name]].append(row)
    return _with_ratio(rows_c), rows_c, pf, aucs


THR = 0.03
REF_G, ref_rows, ref_pf, ref_aucs = _gate_and_prune("REF", FORKS, MOVIE, PER, THR)
f0g_ref, f1g_ref = _with_ratio(ref_pf[0]), _with_ratio(ref_pf[1])
print(f"  {'REF + gate p>' + str(THR):<24} "
      f"{sum(r['division_tp'] for r in ref_rows):>4}/"
      f"{sum(r['division_fp'] for r in ref_rows):>4}/"
      f"{sum(r['division_fn'] for r in ref_rows):<4} {REF_G['division_jaccard']:>7.4f} "
      f"{REF_G['adj_edge_jaccard']:>7.4f} {REF_G['score']:>7.4f} "
      f"{REF_G['adj_edge_jaccard'] - b_ref['adj_edge_jaccard']:>+8.4f} "
      f"{REF_G['score'] - b_ref['score']:>+8.4f} {0.0:>+12.4f} "
      f"{f0g_ref['score']:>8.4f} {f1g_ref['score']:>8.4f}"
      f"   out-of-fold AUC " + " ".join(f"{n}:{a:.4f}" for n, a in ref_aucs)
      + "   want 42/47/83 0.2442 (divbench3)", flush=True)
for arm, dw in DIVW_ARMS:
    rc, rows2, pf2, aucs = _gate_and_prune(arm, ARM_FORKS[arm], ARM_MOVIE[arm],
                                           ARM_PER[arm], THR)
    f0, f1 = _with_ratio(pf2[0]), _with_ratio(pf2[1])
    print(f"  {arm + ' + gate p>' + str(THR):<24} "
          f"{sum(r['division_tp'] for r in rows2):>4}/"
          f"{sum(r['division_fp'] for r in rows2):>4}/"
          f"{sum(r['division_fn'] for r in rows2):<4} {rc['division_jaccard']:>7.4f} "
          f"{rc['adj_edge_jaccard']:>7.4f} {rc['score']:>7.4f} "
          f"{rc['adj_edge_jaccard'] - b_ref['adj_edge_jaccard']:>+8.4f} "
          f"{rc['score'] - b_ref['score']:>+8.4f} "
          f"{rc['score'] - REF_G['score']:>+12.4f} "
          f"{f0['score'] - f0g_ref['score']:>+8.4f} {f1['score'] - f1g_ref['score']:>+8.4f}"
          f"   out-of-fold AUC " + " ".join(f"{n}:{a:.4f}" for n, a in aucs), flush=True)
print("  (the last two columns are per-fold deltas against REF+gate, which is what "
      "ships; the ungated table above is the August comparison and is NOT the "
      "decision)", flush=True)

'''
src = src[:a] + ARMS_BLOCK + src[b:]

# SIX_OK: the movies that actually completed, used by every arm table
rep('''RES = _with_ratio(ROWS)''',
    '''SIX_OK = [n for n in SIX if n in PER]
RES = _with_ratio(ROWS)''')

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
    '''print("  (1) READ THE ARMS AFTER THE GATE. The ungated table reproduces the August "
      "shape -- lower DIV_W makes more forks, div_J falls, adj_J falls a little -- and "
      "that is the comparison that closed this line when no gate existed. The decision "
      "is the gated table: an arm is a submission candidate iff its score beats "
      "REF+gate in BOTH folds.", flush=True)
print("  (2) Size: divbench3's bench resolves about +-0.03 of div_J at 90% (gatelab, "
      "paired bootstrap over movies), which is +-0.003 of score. adj_J is measured far "
      "more precisely, so an arm that trades -0.002 of adj_J for +0.03 of div_J is a "
      "real loss, not a wash. Read both columns, not the sum alone.", flush=True)
print("  (3) The mechanism to check before believing any of it: the ilp / patch fork "
      "counts in the ungated table. If lowering DIV_W does not actually move fork "
      "production from the patch to the solver, the arm did not test what it claims. "
      "ranker-fine measured 111 / 190 / 394 / 1024 ILP forks at 0.55 / 0.45 / 0.35 / "
      "0.20 on 12 movies with the old pipeline.", flush=True)
print("  (4) PRE-REGISTERED, from divbench7 (2026-09-08). That run raised the PATCH's "
      "own per-frame cap, made 1,862 more forks, recovered 4 more true divisions -- "
      "and lost, with the per-arm gate's out-of-fold AUC FALLING as the cap rose "
      "(0.9283/0.8326 at REF, 0.8936/0.8409 at 3x, 0.8841/0.8471 uncapped). The "
      "explanation on record is that the patch's budget is already handed out in "
      "QUALITY order (proposals sorted by d_pq + 0.15*d_s), so what it refuses is the "
      "worst tail. DIV_W is the OPPOSITE direction on that same axis: it moves "
      "production to the solver, whose counted forks are MORE precise than the "
      "patch's (0.196 against 0.150, divbench3). So: if source quality is what "
      "governs, the per-arm AUC should HOLD or RISE as DIV_W falls. IF IT FALLS HERE "
      "TOO, the governing variable is fork COUNT rather than source quality, and both "
      "levers close together -- that reading is fixed in advance either way.",
      flush=True)''')

rep('DIVBENCH4 DONE in', 'DIVBENCH6 DONE in')
rep('print("DIVBENCH4 OK", flush=True)', 'print("DIVBENCH6 OK", flush=True)')
rep('f"[degraded] DIVBENCH4 INCOMPLETE:', 'f"[degraded] DIVBENCH6 INCOMPLETE:')
rep('"/kaggle/working/forks_f4_v4.csv"', '"/kaggle/working/forks_dw55.csv"')
rep('''for arm, _m in STEAL_ARMS:
    with open(f"/kaggle/working/forks_f4_{arm}.csv", "w") as fh:''',
    '''for arm, _m in DIVW_ARMS:
    with open(f"/kaggle/working/forks_{arm}.csv", "w") as fh:''')
rep('''for tag, cols, names_ in (("f4v4_20", J20, FEAT20),
                          ("f4v4_28", list(range(len(FEAT_NAMES))), FEAT_NAMES)):''',
    '''for tag, cols, names_ in (("dw55_20", J20, FEAT20),):''')

open("divbench6.py", "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
