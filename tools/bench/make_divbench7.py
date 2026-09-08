"""divbench4.py -> divbench7.py: the division patch's own budget, crossed
with the fork gate.

WHY. divbench4's budget table, the first time this stage's caps were ever
counted, over 128 movies: 4,514 forks ADDED, **1,672 refused by the PER-FRAME
cap**, 8 by the global cap. 27% of the proposals that pass every geometry
gate are refused because their frame is full. divbench5 counts 861 more on
44b6's 71 movies, so it is both batches.

This is the 2026-08-27 shape exactly: `gap2` and `gap3` swept the bridge span
and four bridge gates behind a repair budget that bound in 24 of 32 datasets,
and raising the budget beat every gate they had selected. The rule says every
verdict about the thing behind a binding budget is provisional -- which here
means divgate2's condition walk and divbench's tighter-geometry arms.

WHY NOW rather than in August. Over-production is newly safe: the fork gate
removes 80% of forks while keeping 84% of the true ones and scored +0.010 at
LB. "Raise the cap, let the gate filter" is a composition that did not exist
before 2026-09-06.

WHAT IT COSTS. `DIV_FRAME_CAP` and `DIV_GLOBAL_CAP` are read inside
`_add_safe_divisions`, which `patch_final` runs AFTER the solve, so they are
POST-LINK constants (2026-08-28 rule: find the line that READS the constant
and check whether it runs inside the arm loop -- this one does). One solve per
movie, five patches. Same cost as divbench4.

THE CROSS. The cap and the gate act on the same failure from opposite ends --
the cap rations forks by frame order, the gate rejects them by quality -- so
2026-08-26 says cross them rather than sweeping one at a time. Every arm is
scored ungated AND after the gate at p > 0.03.

ARMS. REF is (0.0076, 0.00375). Then 3x, 8x, and uncapped. The uncapped arm
is the control that says how much the cap is actually rationing; per
2026-08-26 it is not a ceiling, because a gated version can sit above it.

ANCHOR: REF must reproduce 50/249/75, div_J 0.1337, adj_J 0.9133, score
0.9266, and REF + gate at p > 0.03 must reproduce 42/47/83, div_J 0.2442.
"""
src = open("divbench4.py", encoding="utf-8").read()
NL = "\n"


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:80])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-07a-divbench4"', 'SCRIPT_VERSION = "2026-09-07d-divbench7"')

# ---- the caps become parameters of the patch ------------------------------
rep('''def _add_safe_divisions(nodes, edges, pfun=None, steal=None):''',
    '''def _add_safe_divisions(nodes, edges, pfun=None, steal=None,
                        frame_cap=None, global_cap=None):''')
rep('''    max_global = max(1, int(DIV_GLOBAL_CAP * len(nodes)))
    added = 0''',
    '''    gcap = DIV_GLOBAL_CAP if global_cap is None else global_cap
    fcap = DIV_FRAME_CAP if frame_cap is None else frame_cap
    max_global = max(1, int(gcap * len(nodes)))
    added = 0''')
rep('''        if per_frame[t] >= max(1, int(DIV_FRAME_CAP * n_frame[t])):
            PATCH_CAPS["orphan_frame"] += 1
            continue
        FORK_LOG[(p, q)] = dict(c1=c1, d_pq=d_pq, d_s=d_s, dv=dv_,
                                n_orph=len(orphan_ids.get(t + 1, [])), steal=0)''',
    '''        if per_frame[t] >= max(1, int(fcap * n_frame[t])):
            PATCH_CAPS["orphan_frame"] += 1
            continue
        FORK_LOG[(p, q)] = dict(c1=c1, d_pq=d_pq, d_s=d_s, dv=dv_,
                                n_orph=len(orphan_ids.get(t + 1, [])), steal=0)''')
rep('''    max_steal = max(1, int(DIV_GLOBAL_CAP * len(nodes)))''',
    '''    max_steal = max(1, int(gcap * len(nodes)))''')
rep('''        if per_frame[t] >= max(1, int(DIV_FRAME_CAP * n_frame[t])):
            PATCH_CAPS["steal_frame"] += 1
            continue''',
    '''        if per_frame[t] >= max(1, int(fcap * n_frame[t])):
            PATCH_CAPS["steal_frame"] += 1
            continue''')
rep('''def patch_final(nodes0, edges0, pool, out_path, divs=True, pfun=None, steal=None):''',
    '''def patch_final(nodes0, edges0, pool, out_path, divs=True, pfun=None, steal=None,
                frame_cap=None, global_cap=None):''')
rep('''    if divs:
        _add_safe_divisions(nodes, edges, pfun=pfun, steal=steal)''',
    '''    if divs:
        _add_safe_divisions(nodes, edges, pfun=pfun, steal=steal,
                            frame_cap=frame_cap, global_cap=global_cap)''')

# ---- the arms -------------------------------------------------------------
rep('''# (arm name, ranker margin the mother's edge must beat the ILP's by). -inf is
# the unconditional control: it measures the SUM of its good and bad cases,
# which is not a ceiling (2026-08-26), and it says whether the gates bind.
STEAL_ARMS = [("steal_all", float("-inf")), ("steal_m0", 0.0), ("steal_m20", 0.20)]''',
    '''STEAL_ARMS = []       # divbench7 sweeps the patch's BUDGET, not the steal
# (arm name, DIV_FRAME_CAP, DIV_GLOBAL_CAP). REF is (0.0076, 0.00375) and
# refused 1,672 proposals by the frame cap and 8 by the global one over the
# 128 movies. "nocap" is the control that says how much is being rationed; it
# is NOT a ceiling -- a gated version can sit above it (2026-08-26).
CAP_ARMS = [("cap3x", 0.0228, 0.01125), ("cap8x", 0.0608, 0.03),
            ("nocap", 1.0, 1.0)]''')

# ---- stage B: one solve, one patch per arm --------------------------------
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
    for arm, fcap, gcap in CAP_ARMS:
        outa = pathlib.Path(f"{TMP}/arm_{arm}") / f"{name}.geff"
        outa.parent.mkdir(parents=True, exist_ok=True)
        PATCH_CAPS.clear()
        patch_final(nodes0, edges0, pool, outa, frame_cap=fcap, global_cap=gcap)
        for k_, v_ in PATCH_CAPS.items():
            CAPS_BY_ARM.setdefault(arm, defaultdict(int))[k_] += v_
        n_add = PATCH_CAPS.get("orphan_added", 0)
        row_a, ds_a, fr_a = _judge(name, k, outa, seeds, aux)
        ARM_ROWS[arm].append(row_a)
        ARM_PER[arm][name] = row_a
        ARM_FORKS[arm].extend(fr_a)
        ARM_MOVIE[arm][name] = _snapshot()
        cells.append(f"{arm} +{n_add} {row_a['division_tp']}/{row_a['division_fp']}/"
                     f"{row_a['division_fn']} {_adj_of(row_a):.4f}")
        del ds_a''')
rep('''ARM_FORKS = {a: [] for a, _ in STEAL_ARMS}
ARM_ROWS = {a: [] for a, _ in STEAL_ARMS}
ARM_PER = {a: {} for a, _ in STEAL_ARMS}
ARM_MOVIE = {a: {} for a, _ in STEAL_ARMS}''',
    '''ARM_FORKS = {a: [] for a, _f, _g in CAP_ARMS}
ARM_ROWS = {a: [] for a, _f, _g in CAP_ARMS}
ARM_PER = {a: {} for a, _f, _g in CAP_ARMS}
ARM_MOVIE = {a: {} for a, _f, _g in CAP_ARMS}''')
rep('''      f"REF patch + {len(STEAL_ARMS)} steal arms, every fork logged and judged, every "
      f"lost division walked and every 5a case written out ===", flush=True)''',
    '''      f"REF patch (frame cap {DIV_FRAME_CAP}, global {DIV_GLOBAL_CAP}) + "
      f"{len(CAP_ARMS)} arms that RE-PATCH the same solve at "
      f"{', '.join(a for a, _f, _g in CAP_ARMS)}; every fork logged and judged ===",
      flush=True)''')
rep('''for a_ in ["REF"] + [x[0] for x in STEAL_ARMS]:''',
    '''for a_ in ["REF"] + [x[0] for x in CAP_ARMS]:''')
rep('''for arm, _m in STEAL_ARMS:
    with open(f"/kaggle/working/forks_f4_{arm}.csv", "w") as fh:''',
    '''for arm, _f, _g in CAP_ARMS:
    with open(f"/kaggle/working/forks_f4_{arm}.csv", "w") as fh:''')

# ---- the arm report -------------------------------------------------------
OLD_A = '''print(f"\\n=== THE STEAL ARMS: the patch allowed to take a daughter the ILP had already '''
a = src.index(OLD_A)
a = src.rfind(NL, 0, a) + 1
b = src.index('print("\\n=== reading it ===", flush=True)')
b = src.rfind(NL, 0, b) + 1
ARMS_BLOCK = r'''SIX_OK = [n for n in SIX if n in PER]
print(f"\n=== THE CAP ARMS, UNGATED (one solve per movie, one patch per arm; official "
      f"scorer, all {len(SIX_OK)} movies) ===", flush=True)
print(f"  {'arm':<10} {'forks':>7} {'added':>8} {'frame-refused':>14} {'div tp/fp/fn':>14} "
      f"{'div_J':>7} {'adj_J':>7} {'node_r':>8} {'score':>7} {'d adj_J':>8} {'d div_J':>8} "
      f"{'d score':>8} {'f0 dsc':>7} {'f1 dsc':>7}", flush=True)
b_ref = RES
f0_ref = _with_ratio([PER[n] for n in SIX_OK if FOLD_OF[n] == 0])
f1_ref = _with_ratio([PER[n] for n in SIX_OK if FOLD_OF[n] == 1])


def _armline(tag, rows_a, forks_a, per_a, caps):
    rc = _with_ratio(rows_a)
    f0 = _with_ratio([per_a[n] for n in SIX_OK if FOLD_OF[n] == 0 and n in per_a])
    f1 = _with_ratio([per_a[n] for n in SIX_OK if FOLD_OF[n] == 1 and n in per_a])
    print(f"  {tag:<10} {len(forks_a):>7,} {caps.get('orphan_added', 0):>8,} "
          f"{caps.get('orphan_frame', 0):>14,} "
          f"{sum(r['division_tp'] for r in rows_a):>4}/"
          f"{sum(r['division_fp'] for r in rows_a):>4}/"
          f"{sum(r['division_fn'] for r in rows_a):<4} "
          f"{rc['division_jaccard']:>7.4f} {rc['adj_edge_jaccard']:>7.4f} "
          f"{rc.get('total_node_ratio', float('nan')):>+8.4f} {rc['score']:>7.4f} "
          f"{rc['adj_edge_jaccard'] - b_ref['adj_edge_jaccard']:>+8.4f} "
          f"{rc['division_jaccard'] - base_J:>+8.4f} "
          f"{rc['score'] - b_ref['score']:>+8.4f} "
          f"{f0['score'] - f0_ref['score']:>+7.4f} {f1['score'] - f1_ref['score']:>+7.4f}",
          flush=True)
    return rc


_armline("REF", ROWS, FORKS, PER, CAPS_BY_ARM.get("REF", {}))
for arm, fcap, gcap in CAP_ARMS:
    _armline(arm, ARM_ROWS[arm], ARM_FORKS[arm], ARM_PER[arm], CAPS_BY_ARM.get(arm, {}))

print(f"\n=== THE CROSS: the same arms AFTER the fork gate at p > 0.03. The gate is "
      f"trained out of fold on THAT ARM'S own counted forks -- raising the cap changes "
      f"the fork population, so a REF-trained gate would be measuring the wrong thing "
      f"===", flush=True)
print(f"  {'arm / cut':<22} {'div tp/fp/fn':>14} {'div_J':>7} {'adj_J':>7} {'score':>7} "
      f"{'d adj_J':>8} {'d score':>8} {'vs REF+gate':>12} {'f0':>8} {'f1':>8}", flush=True)


def _gate_and_prune(tag, forks_a, movies_a, per_a, thr):
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
print(f"  {'REF + gate':<22} "
      f"{sum(r['division_tp'] for r in ref_rows):>4}/"
      f"{sum(r['division_fp'] for r in ref_rows):>4}/"
      f"{sum(r['division_fn'] for r in ref_rows):<4} {REF_G['division_jaccard']:>7.4f} "
      f"{REF_G['adj_edge_jaccard']:>7.4f} {REF_G['score']:>7.4f} "
      f"{REF_G['adj_edge_jaccard'] - b_ref['adj_edge_jaccard']:>+8.4f} "
      f"{REF_G['score'] - b_ref['score']:>+8.4f} {0.0:>+12.4f} "
      f"{f0g_ref['score']:>8.4f} {f1g_ref['score']:>8.4f}"
      f"   AUC " + " ".join(f"{n}:{a:.4f}" for n, a in ref_aucs)
      + "   want 42/47/83 0.2442 (divbench3)", flush=True)
for arm, fcap, gcap in CAP_ARMS:
    rc, rows2, pf2, aucs = _gate_and_prune(arm, ARM_FORKS[arm], ARM_MOVIE[arm],
                                           ARM_PER[arm], THR)
    f0, f1 = _with_ratio(pf2[0]), _with_ratio(pf2[1])
    print(f"  {arm + ' + gate':<22} "
          f"{sum(r['division_tp'] for r in rows2):>4}/"
          f"{sum(r['division_fp'] for r in rows2):>4}/"
          f"{sum(r['division_fn'] for r in rows2):<4} {rc['division_jaccard']:>7.4f} "
          f"{rc['adj_edge_jaccard']:>7.4f} {rc['score']:>7.4f} "
          f"{rc['adj_edge_jaccard'] - b_ref['adj_edge_jaccard']:>+8.4f} "
          f"{rc['score'] - b_ref['score']:>+8.4f} "
          f"{rc['score'] - REF_G['score']:>+12.4f} "
          f"{f0['score'] - f0g_ref['score']:>+8.4f} {f1['score'] - f1g_ref['score']:>+8.4f}"
          f"   AUC " + " ".join(f"{n}:{a:.4f}" for n, a in aucs), flush=True)
print("  (the last two columns are per-fold deltas against REF+gate, which is what "
      "ships; the ungated table above is how this line was read in August, when there "
      "was no gate, and is NOT the decision)", flush=True)

'''
src = src[:a] + ARMS_BLOCK + src[b:]

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
    '''print("  (1) FIRST read the 'frame-refused' column: if it does not fall to ~0 at "
      "cap8x and nocap, the arms did not lift the constraint and nothing else in the "
      "table means anything.", flush=True)
print("  (2) The decision is the GATED table. An arm is a submission candidate iff its "
      "score beats REF+gate in BOTH folds. The ungated table is how this line was read "
      "in August; it is here to show the shape of the trade, not to decide.", flush=True)
print("  (3) Size: the bench resolves about +-0.03 of div_J at 90% (gatelab, paired "
      "bootstrap over movies) = +-0.003 of score, while adj_J is measured far more "
      "precisely. An arm that trades -0.002 of adj_J for +0.03 of div_J is a real "
      "loss. Read both columns, never the sum alone.", flush=True)
print("  (4) nocap is a CONTROL, not a candidate: it measures the sum of its good and "
      "bad cases and a gated version can sit ABOVE it (2026-08-26). If nocap comes "
      "back near zero, look for the cancellation.", flush=True)''')

rep('DIVBENCH4 DONE in', 'DIVBENCH7 DONE in')
rep('print("DIVBENCH4 OK", flush=True)', 'print("DIVBENCH7 OK", flush=True)')
rep('f"[degraded] DIVBENCH4 INCOMPLETE:', 'f"[degraded] DIVBENCH7 INCOMPLETE:')
rep('"/kaggle/working/forks_f4_v4.csv"', '"/kaggle/working/forks_f4_ref7.csv"')
rep('''for tag, cols, names_ in (("f4v4_20", J20, FEAT20),
                          ("f4v4_28", list(range(len(FEAT_NAMES))), FEAT_NAMES)):''',
    '''for tag, cols, names_ in (("ref7_20", J20, FEAT20),):''')

open("divbench7.py", "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
