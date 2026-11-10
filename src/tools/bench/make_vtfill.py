"""vtlost.py -> vtfill.py: the vote candidates as PATCHES on REF's solution, no re-solve.

WHY (2026-09-20). vtlost's ledger on the attach-only arm (VT17XB): the candidates
themselves lose 22 true edges and gain 118 (net +96); the RE-SOLVE around them churns
267 shipped-to-shipped true edges away and 238 back (net -29, sign varying by movie,
the worst movies churn-only). So keep REF's solved graph exactly as it is and insert a
candidate only where it patches a break:

  HOLE   a track END at t-1 (no successor) and a track START at t+1 (no predecessor)
         both link to the candidate at t with the ranker's p >= floor_hole; best p sum
         first, one-to-one on ends, starts and candidates.
  END    (optional) a leftover candidate extends a track end or a track start with ONE
         edge at p >= floor_end. vtattach's X arm showed one-sided attachments carry
         true edges (+217 over XB) but cost 40k nodes; a high floor is the test of
         whether the ranker separates them.
  GAP    (optional) the admitted candidates are also appended to the sub-threshold pool
         the shipped gap-close bridges with (their encoded score 0.17-0.19 passes
         GAP1_MIN_CAND_SCORE 0.10; today they are excluded because the pool takes only
         scores in [CAND_THR 0.05, UNET_THRESH 0.15)).

The insertion runs after REF's ILP and before patch_final, so gap-close, the short
filter, linefit and the safe divisions run on the patched graph as they do on REF's.
The candidate pairs' p come from the vote heads' features through the same ranker
(solve_one in predict-only mode: no ILP). The ledger (ln_lost) and the funnel run per
arm; the ledger gains two columns, whether REF's source already had a successor and
whether REF's target already had a predecessor (a gained edge that replaces a WRONG
REF edge against one that fills an absence).

ARMS:  FILL      hole floor 0.48
       FILLE85   hole 0.48 + end floor 0.85
       FILLE95   hole 0.48 + end floor 0.95
       FILLGAP   hole 0.48 + the candidates in the gap-close pool
CPU: features 40 min + REF solves + 4 cheap arms (no ILP) + the cross: ~4.5 h.
"""
import os

src = open("vtlost.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-20c-vtlost"', 'SCRIPT_VERSION = "2026-09-20a-vtfill"')
rep('DUP_ARMS = [("VT17X", dict(tau=0.17, persist=False, nonew=True, dangle=False)), ("VT17XB", dict(tau=0.17, persist=False, nonew=True, dangle=True))]',
    'DUP_ARMS = [("FILL", dict(tau=0.17, persist=False, hole=0.48, end=None, gap=False)), '
    '("FILLE85", dict(tau=0.17, persist=False, hole=0.48, end=0.85, gap=False)), '
    '("FILLE95", dict(tau=0.17, persist=False, hole=0.48, end=0.95, gap=False)), '
    '("FILLGAP", dict(tau=0.17, persist=False, hole=0.48, end=None, gap=True))]   # vtfill: patches on REF\'s solution, no re-solve')

# 1. solve_one in predict-only mode: the ranker's p on every pair, no ILP
rep('''def solve_one(dets, T, seeds, feats, model, cols, disapp=None, floor=None,
              divw=None, keep=None):
''', '''def solve_one(dets, T, seeds, feats, model, cols, disapp=None, floor=None,
              divw=None, keep=None, predict_only=False):
''')
rep('''    else:
        probs = [model.predict_proba(rows[:, cols])[:, 1]
                 if len(pairs) else np.zeros(0)
                 for pairs, rows, P, C in feats]
    coords_rows, offset, n_glob = [], {}, 0
''', '''    else:
        probs = [model.predict_proba(rows[:, cols])[:, 1]
                 if len(pairs) else np.zeros(0)
                 for pairs, rows, P, C in feats]
    if predict_only:                       # vtfill: the pmap alone, no graph, no ILP
        pmap = {}
        for t in range(T - 1):
            pairs, rows, P, C = feats[t]
            for (i_, j_), pr, row in zip(pairs, probs[t], rows):
                pmap[(t, int(i_), int(j_))] = (float(pr), float(row[C_PFWD]), float(row[C_PREV]))
        return None, None, None, (pmap, None)
    coords_rows, offset, n_glob = [], {}, 0
''')

# 2. the patch functions, before ln_trace
rep('def ln_trace(name, T, seeds, nnew, keep, pmap, son):\n', r'''def vt_fill(nodes0, edges0, son0, seeds_ln, nnew, keep, pm_a, floor_hole, floor_end):
    """REF's solved graph with the admitted vote candidates inserted as patches, NO re-solve.
    A candidate at t fills a HOLE when a track end at t-1 and a track start at t+1 both link to it at p >= floor_hole
    (best p sum first, one-to-one on ends, starts and candidates). When floor_end is not None a leftover candidate may
    extend a track end or a track start with ONE edge at p >= floor_end. Returns (nodes, edges, son, stats)."""
    nodes = {g_: {"t": int(v_["t"]), "xyz": np.asarray(v_["xyz"], float).copy()} for g_, v_ in nodes0.items()}
    edges = list(edges0)
    son = dict(son0)
    succ, pred = defaultdict(list), defaultdict(list)
    for a_, b_ in edges:
        succ[a_].append(b_)
        pred[b_].append(a_)
    inv = {(int(sd[0]), int(sd[1])): g_ for g_, sd in son.items() if sd is not None}
    T = len(seeds_ln)
    n0 = [len(np.asarray(s_[1]).reshape(-1)) - int(nn_) for s_, nn_ in zip(seeds_ln, nnew)]
    pin, pout = defaultdict(list), defaultdict(list)
    for (t_, i_, j_), v_ in pm_a.items():
        if t_ + 1 >= T:
            continue
        if i_ < n0[t_] and j_ >= n0[t_ + 1]:
            pin[(t_ + 1, j_)].append((i_, float(v_[0])))        # shipped at t -> candidate at t+1
        elif i_ >= n0[t_] and j_ < n0[t_ + 1]:
            pout[(t_, i_)].append((j_, float(v_[0])))           # candidate at t -> shipped at t+1
    props = []
    for t in range(T):
        for k_ in range(int(nnew[t])):
            qi = n0[t] + k_
            if keep is not None and not keep[t][qi]:
                continue
            ins = sorted(((p, inv.get((t - 1, ie))) for ie, p in pin.get((t, qi), []) if p >= floor_hole), reverse=True)
            outs = sorted(((p, inv.get((t + 1, js))) for js, p in pout.get((t, qi), []) if p >= floor_hole), reverse=True)
            ins = [(p, e) for p, e in ins if e is not None and not succ.get(e)]        # track ENDS only
            outs = [(p, s_) for p, s_ in outs if s_ is not None and not pred.get(s_)]   # track STARTS only
            if ins and outs:
                props.append((2, ins[0][0] + outs[0][0], t, qi, ins[0][1], outs[0][1]))
            elif floor_end is not None:
                if ins and ins[0][0] >= floor_end:
                    props.append((1, ins[0][0], t, qi, ins[0][1], None))
                elif outs and outs[0][0] >= floor_end:
                    props.append((1, outs[0][0], t, qi, None, outs[0][1]))
    props.sort(reverse=True)
    st = defaultdict(int)
    st["proposed_hole"] = sum(1 for p_ in props if p_[0] == 2)
    st["proposed_end"] = sum(1 for p_ in props if p_[0] == 1)
    nid = max(nodes) if nodes else 0
    for n_e, _sc, t, qi, e, s_ in props:
        if (e is not None and succ.get(e)) or (s_ is not None and pred.get(s_)):
            st["refused_taken"] += 1
            continue
        nid += 1
        nodes[nid] = {"t": int(t), "xyz": np.round(np.asarray(seeds_ln[t][0], float).reshape(-1, 3)[qi]).astype(float)}
        son[nid] = (int(t), int(qi))
        if e is not None:
            edges.append((e, nid))
            succ[e].append(nid)
            pred[nid].append(e)
        if s_ is not None:
            edges.append((nid, s_))
            succ[nid].append(s_)
            pred[s_].append(nid)
        st["hole" if n_e == 2 else "end"] += 1
    return nodes, edges, son, st


def vt_pool(pool, seeds_ln, nnew, keep):
    """REF's sub-threshold gap-close pool with the admitted vote candidates appended (their encoded score as is)."""
    cand, cand_sc, _ = pool
    cand2, sc2, tr2 = [], [], []
    for t, (c_, s_) in enumerate(zip(cand, cand_sc)):
        n_ = len(np.asarray(seeds_ln[t][1]).reshape(-1)) - int(nnew[t])
        cc = np.asarray(seeds_ln[t][0], float).reshape(-1, 3)[n_:]
        ss = np.asarray(seeds_ln[t][1], float).reshape(-1)[n_:]
        m_ = np.ones(len(cc), bool) if keep is None else np.asarray(keep[t], bool)[n_:]
        c2 = np.vstack([np.asarray(c_, float).reshape(-1, 3), cc[m_]]) if len(cc) else np.asarray(c_, float).reshape(-1, 3)
        s2 = np.concatenate([np.asarray(s_, float).reshape(-1), ss[m_]]) if len(cc) else np.asarray(s_, float).reshape(-1)
        cand2.append(c2)
        sc2.append(s2)
        tr2.append(cKDTree(c2 * SCALE) if len(c2) else None)
    return cand2, sc2, tr2


def ln_trace(name, T, seeds, nnew, keep, pmap, son):
''')

# 3. the ledger: did REF's source / target already carry an edge (a WRONG one) where the arm gains a true edge?
rep('"d_thief_true", "d_thief_gt", "p_new"]', '"d_thief_true", "d_thief_gt", "ref_src_succ", "ref_tgt_pred", "p_new"]')
rep('''        pend.append(([arm, name, kind, cat, t1, int(a_), int(b_), p_true, p_steal, d_true, d_steal, thief_ann, thief_kept, thief_v, d_thief_true, d_thief_gt], m1x, m2x, s1, s2))
''', '''        ref_src_succ = -1 if m1r is None else int(bool(succ_r.get(m1r)))
        ref_tgt_pred = -1 if m2r is None else int(bool(pred_r.get(m2r)))
        pend.append(([arm, name, kind, cat, t1, int(a_), int(b_), p_true, p_steal, d_true, d_steal, thief_ann, thief_kept, thief_v, d_thief_true, d_thief_gt,
                      ref_src_succ, ref_tgt_pred], m1x, m2x, s1, s2))
''')
rep('''                if g.p_new.notna().any():
                    extra += (f"  p_true med {g.p_true.median():.3f}  p_new (the replacing shipped edge) med {g.p_new.median():.3f}"
''', '''                if kind == "gained":
                    extra += (f"  in REF the source already had a successor (a WRONG edge replaced) {int((g.ref_src_succ == 1).sum())}, the target a predecessor {int((g.ref_tgt_pred == 1).sum())},"
                              f" neither (an absence filled) {int(((g.ref_src_succ == 0) & (g.ref_tgt_pred == 0)).sum())}")
                if g.p_new.notna().any():
                    extra += (f"  p_true med {g.p_true.median():.3f}  p_new (the replacing shipped edge) med {g.p_new.median():.3f}"
''')

# 4. the arm loop: one predict-only pass for the candidate pairs' p, then four cheap patch arms on REF's graph
i0 = src.index("    for arm, cfg in BR_ARMS:\n        keep, n_new, n_kept = ln_keep(")
i1 = src.index("    del dets_ln, nnew_ln, seeds_ln, feats_ln\n")
NEW_LOOP = '''    _, _, _, (pm_a, _) = solve_one(dets_ln, T, seeds_ln, feats_ln, MODELS_FOLD[k], COLS,
                                   disapp=ILP_DISAPP, floor=PROD_FLOOR, predict_only=True)
    sc_a, b2_a = _aux_extra(seeds_ln, pm_a)
    for arm, cfg in BR_ARMS:
        keep, n_new, n_kept = ln_keep(seeds_ln, nnew_ln, cfg["tau"], cfg.get("persist", False))
        LN_STAT[arm]["new"] += n_new
        LN_STAT[arm]["kept"] += n_kept
        DUP_STAT.clear()
        outa = pathlib.Path(f"{TMP}/arm_{arm}") / f"{name}.geff"
        outa.parent.mkdir(parents=True, exist_ok=True)
        n_a, e_a, so_a, fst = vt_fill(nodes0, edges0, seed_of_node, seeds_ln, nnew_ln, keep, pm_a, cfg["hole"], cfg.get("end"))
        for k_, v_ in fst.items():
            LN_STAT[arm][k_] += v_
        n_solved, e_solved = n_a, list(e_a)
        p_a = vt_pool(pool, seeds_ln, nnew_ln, keep) if cfg.get("gap", False) else pool
        PATCH_CAPS.clear()
        patch_final(n_a, e_a, p_a, outa)
        LT_ARM = arm
        LN_STAT[arm]["trace_rows"] += ln_trace(name, T, seeds_ln, nnew_ln, keep, pm_a, so_a)
        ln_lost(arm, name, T, seeds_ln, nnew_ln, REF_FINAL, seed_of_node, pmap, pm_a, so_a, e_solved, n_solved)
        LN_STAT[arm]["final_new_nodes"] += sum(1 for g_ in PRE_STATE["nodes"] if ln_is_new(so_a.get(g_), seeds_ln, nnew_ln))
        LN_STAT[arm]["final_nodes"] += len(PRE_STATE["nodes"])
        for k_, v_ in PATCH_CAPS.items():
            CAPS_BY_ARM.setdefault(arm, defaultdict(int))[k_] += v_
        row_a, ds_a, fr_a = _judge(name, k, outa, seeds_ln, (pm_a, so_a, sc_a, b2_a))
        ARM_ROWS[arm].append(row_a)
        ARM_PER[arm][name] = row_a
        ARM_FORKS[arm].extend(fr_a)
        ARM_MOVIE[arm][name] = _snapshot()
        DUP_BY_ARM[arm]["primaries"] += n_prim
        DUP_BY_ARM[arm]["suppressed"] += n_new - n_kept
        for k_, v_ in DUP_STAT.items():
            DUP_BY_ARM[arm][k_] += v_
        cells.append(f"{arm} new {n_kept}/{n_new} holes {fst['hole']} ends {fst['end']} nodes {len(n_a)} "
                     f"{row_a['division_tp']}/{row_a['division_fp']}/{row_a['division_fn']} "
                     f"{_adj_of(row_a):.4f}")
        del ds_a, n_a, e_a, p_a, so_a, keep, n_solved, e_solved, fst
    del pm_a, sc_a, b2_a
'''
src = src[:i0] + NEW_LOOP + src[i1:]

# 5. the per-arm summary
i0 = src.index("for a_, _c in BR_ARMS:\n    d_ = LN_STAT[a_]\n    print(f\"  {a_:<6} tau {_c['tau']:.2f}")
i1 = src.index("(vttrace: ~144k of 224k at 2 votes, ~47k of 82k at 3)\", flush=True)\n") + len("(vttrace: ~144k of 224k at 2 votes, ~47k of 82k at 3)\", flush=True)\n")
src = src[:i0] + '''for a_, _c in BR_ARMS:
    d_ = LN_STAT[a_]
    print(f"  {a_:<8} tau {_c['tau']:.2f} hole floor {_c['hole']} end floor {_c.get('end')} gap pool {_c.get('gap', False)}: "
          f"new {d_['new']:,}, kept {d_['kept']:,}; proposed holes {d_['proposed_hole']:,} ends {d_['proposed_end']:,}, refused (slot taken) {d_['refused_taken']:,}; "
          f"inserted holes {d_['hole']:,} ends {d_['end']:,}; in the FINAL graph {d_['final_new_nodes']:,} of {d_['final_nodes']:,} nodes "
          f"(vtattach XB: 8,000 hole-fillers of 2,099,577 after the re-solve)", flush=True)
''' + src[i1:]
rep('print("\\n=== VTBENCH: NEW PEAKS PER ARM, summed over the scored movies (new = the vote candidates appended by mapvote; "\n      "kept = admitted at the arm\'s score threshold, the rest withheld from the ILP and dropped) ===", flush=True)',
    'print("\\n=== VTFILL: PATCHES PER ARM, summed over the scored movies (new = the vote candidates appended by mapvote; kept = admitted at tau; "\n      "the rest are never looked at; REF\'s solved graph is not re-solved) ===", flush=True)')

# 6. names
for old, new in (('f"vtlost_REF_{DET_SRC}.csv"', 'f"vtfill_REF_{DET_SRC}.csv"'), ('f"vtlost_arm_{a_}.csv"', 'f"vtfill_arm_{a_}.csv"'),
                 ('"/kaggle/working/vtlost_trace.csv"', '"/kaggle/working/vtfill_trace.csv"'), ('"/kaggle/working/vtlost_edges.csv"', '"/kaggle/working/vtfill_edges.csv"'),
                 ('print("VTLOST OK", flush=True)', 'print("VTFILL OK", flush=True)'),
                 ('print(f"[degraded] VTLOST INCOMPLETE or anchor miss:', 'print(f"[degraded] VTFILL INCOMPLETE or anchor miss:'),
                 ('=== VTLOST (funnel): every linkable GT edge', '=== VTFILL (funnel): every linkable GT edge'),
                 ('=== VTLOST: every GT edge (t -> t+1) realized', '=== VTFILL (ledger): every GT edge (t -> t+1) realized')):
    rep(old, new)

assert src.count("vt_fill(") == 2 and src.count("vt_pool(") == 2 and src.count("predict_only=True") == 1
assert "LN_NEWMASK = ln_newmask(" not in src and "ln_drop_dangling(n_a" not in src
assert src.index("def vt_fill(") < src.index("def ln_trace(") < src.index("        n_a, e_a, so_a, fst = vt_fill(")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vtfill.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written vtfill.py", len(src.splitlines()), "lines")
