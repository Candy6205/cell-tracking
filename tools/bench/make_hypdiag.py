"""hypilp.py -> hypdiag.py: WHERE do the parent hypotheses lose their true edges?

hypilp v3 (2026-09-15): parents (HY10) cost 601 true edges for 279 false ones
removed and halved the ILP's divisions, although the labelled parents were
almost all one-cell readings (772 vs 6). Before any constraint is added, the
losses have to be located. This kernel runs the same harness on DIAG_SIX (16
movies: the five HY10 lost most on, the five it gained most on, six in the
middle of the size range), arms C0 and HY10 only, and after both solves
writes an EDGE LEDGER against the GT (nearest GT node within MATCH_UM per
predicted node, label_rows' rule):

  lost   GT edges (g -> h) carried by a C0 edge and by no HY10 edge, classed:
    endpoint_unmatched   g or h has no HY10 node within 5 um (the child was
                         dropped for a parent that sits too far, or moved)
    parent_no_edge       g or h is matched by a selected PARENT, but the parent
                         has no edge to the other endpoint (the chain breaks at
                         the parent)
    children_rewired     both endpoints are still matched by children, the
                         edge went elsewhere (the solve changed around the site)
    each also flagged at_division when g has two GT children
  gained GT edges carried by HY10 and not by C0: via_parent / via_children
  parents  every selected parent: chained (its in- or out-edge is another
           parent) or isolated; in-source is a candidate mother of BOTH children
           (u -> a and u -> b are candidate edges: the division case); the
           children's best in / out probability (pmap), selected vs unselected
           parents, so a 'weak children' gate can be sized.

Attach: hypilp's sources. CPU, ~2.5 h (features on all 199 movies + 16 x 3
solves). Output: the ledger per movie and summed, `hypdiag_ledger.csv`.
"""
import os

src = open("hypilp.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-15a-hypilp"', 'SCRIPT_VERSION = "2026-09-15b-hypdiag"')

# ---- the movies and the arms -----------------------------------------------------
rep('FOLD_OF = {n: i % 2 for i, n in enumerate(SIX)}      # alternate by sorted name\n',
    'FOLD_OF = {n: i % 2 for i, n in enumerate(SIX)}      # alternate by sorted name\n'
    '# hypdiag: stage B on these 16 only (features + stage A stay on all 199 so REF is round 2\'s)\n'
    'DIAG_SIX = ["6bba_a90a0b9c", "6bba_1f58c2f6", "6bba_786893ac", "6bba_7af54fde", "6bba_07477033",\n'
    '            "6bba_55c70843", "6bba_474be664", "6bba_b693381b", "6bba_67ebd073", "6bba_3abfe10a",\n'
    '            "6bba_05b6850b", "6bba_05db0fb1", "6bba_062c8d37", "6bba_268e1230", "6bba_2312ac41",\n'
    '            "6bba_283bf9f1"]\n'
    'assert all(n in FOLD_OF for n in DIAG_SIX), [n for n in DIAG_SIX if n not in FOLD_OF]\n')
rep("for name in SIX:\n    if time.time() - t_all > BUDGET_S:\n        INCOMPLETE.append(name)\n        continue\n",
    "for name in DIAG_SIX:\n    if time.time() - t_all > BUDGET_S:\n        INCOMPLETE.append(name)\n        continue\n")
i = src.index('HYP_ARMS = [("C0", dict(cost=0.0, par=False)),')
j = src.index('DUP_ARMS = HYP_ARMS', i)
src = src[:i] + 'HYP_ARMS = [("C0", dict(cost=0.0, par=False)), ("HY10", dict(cost=0.10, par=True))]\n' + src[j:]

# ---- the ledger --------------------------------------------------------------------
LEDGER = r'''
# ---- hypdiag block ----
DIAG_SOL = {}                 # arm -> (nodes0, edges0, seed_of_node, pmap) of the current movie
LEDGER_ROWS = []              # (movie, kind, klass, at_division, n)
LEDGER_TOT = defaultdict(int)
PAR_ROWS = []                 # per selected / unselected parent: (movie, selected, chained, mother_both, min_child_best_in, min_child_best_out)


def _diag_gt(name):
    gt = read_geff(geff_path(root, "train", name))
    g_t = np.asarray(gt["t"], int)
    g_pos = np.stack([np.asarray(gt[c], float) for c in ("z", "y", "x")], axis=1) * SCALE
    row_of = {int(v): i for i, v in enumerate(gt["node_ids"])}
    kids = defaultdict(list)
    for a, b in gt["edges"]:
        ra, rb = row_of.get(int(a)), row_of.get(int(b))
        if ra is not None and rb is not None:
            kids[ra].append(rb)
    trees = {}
    for tt in np.unique(g_t):
        rows = np.where(g_t == tt)[0]
        trees[int(tt)] = (rows, cKDTree(g_pos[rows]))
    return trees, kids, g_t


def _diag_match(trees, nodes):
    """node id -> GT row within MATCH_UM (nearest), or -1."""
    out = {}
    by_t = defaultdict(list)
    for nid, v in nodes.items():
        by_t[int(v["t"])].append(nid)
    for t, ids in by_t.items():
        if t not in trees:
            for nid in ids:
                out[nid] = -1
            continue
        rows, tree = trees[t]
        P = np.stack([np.asarray(nodes[nid]["xyz"], float) for nid in ids]) * SCALE
        dd, gi = tree.query(P)
        for k, nid in enumerate(ids):
            out[nid] = int(rows[gi[k]]) if dd[k] <= MATCH_UM else -1
    return out


def _tp_edges(edges, gmatch, kids):
    """GT edges (g, h) carried by at least one predicted edge whose endpoints match g and h."""
    out = {}
    for a_, b_ in edges:
        g, h = gmatch.get(a_, -1), gmatch.get(b_, -1)
        if g >= 0 and h >= 0 and h in kids.get(g, ()):
            out.setdefault((g, h), []).append((a_, b_))
    return out


def diag_ledger(name, T, seeds_h, par_h, sol_c0, sol_hy):
    trees, kids, g_t = _diag_gt(name)
    n_c0, e_c0, so_c0, _pm_c0 = sol_c0
    n_hy, e_hy, so_hy, pm_hy = sol_hy
    m_c0, m_hy = _diag_match(trees, n_c0), _diag_match(trees, n_hy)
    tp_c0, tp_hy = _tp_edges(e_c0, m_c0, kids), _tp_edges(e_hy, m_hy, kids)
    nch = [len(seeds_h[t][0]) - len(par_h[t]) for t in range(T)]

    def is_par(nid):
        s_ = so_hy.get(nid)
        return s_ is not None and s_[1] >= nch[s_[0]]
    matched_hy = defaultdict(list)          # GT row -> HY nodes
    for nid, g in m_hy.items():
        if g >= 0:
            matched_hy[g].append(nid)
    for (g, h) in set(tp_c0) - set(tp_hy):
        div = len(kids.get(g, ())) >= 2
        mg, mh = matched_hy.get(g, []), matched_hy.get(h, [])
        if not mg or not mh:
            klass = "endpoint_unmatched"
        elif any(is_par(x) for x in mg) or any(is_par(x) for x in mh):
            klass = "parent_no_edge"
        else:
            klass = "children_rewired"
        LEDGER_TOT[("lost", klass, div)] += 1
        LEDGER_ROWS.append((name, "lost", klass, int(div)))
    for (g, h) in set(tp_hy) - set(tp_c0):
        div = len(kids.get(g, ())) >= 2
        via = any(is_par(a_) or is_par(b_) for a_, b_ in tp_hy[(g, h)])
        klass = "via_parent" if via else "via_children"
        LEDGER_TOT[("gained", klass, div)] += 1
        LEDGER_ROWS.append((name, "gained", klass, int(div)))
    LEDGER_TOT[("tp", "C0", False)] += len(tp_c0)
    LEDGER_TOT[("tp", "HY10", False)] += len(tp_hy)
    # ---- the parents: context of the selected ones against the unselected --------------
    in_of, out_of = defaultdict(list), defaultdict(list)
    for a_, b_ in e_hy:
        out_of[a_].append(b_)
        in_of[b_].append(a_)
    sel_by_seed = {so_hy[nid]: nid for nid in n_hy if so_hy.get(nid) is not None}
    best_in, best_out = defaultdict(float), defaultdict(float)
    for (t_, i_, j_), v in pm_hy.items():
        p = v[0]
        if not np.isfinite(p):
            continue
        if p > best_out[(t_, i_)]:
            best_out[(t_, i_)] = p
        if p > best_in[(t_ + 1, j_)]:
            best_in[(t_ + 1, j_)] = p
    for t in range(T):
        pt = np.asarray(par_h[t], int).reshape(-1, 2)
        for q in range(len(pt)):
            a_, b_ = int(pt[q, 0]), int(pt[q, 1])
            pid = nch[t] + q
            nid = sel_by_seed.get((t, pid))
            selected = nid is not None
            chained = mother_both = False
            if selected:
                nb = in_of.get(nid, []) + out_of.get(nid, [])
                chained = any(is_par(x) for x in nb)
                for u in in_of.get(nid, []):
                    su = so_hy.get(u)
                    if su is not None and su[0] == t - 1:
                        pa = pm_hy.get((t - 1, su[1], a_), (np.nan,))[0]
                        pb = pm_hy.get((t - 1, su[1], b_), (np.nan,))[0]
                        if np.isfinite(pa) and np.isfinite(pb) and pa >= PROD_FLOOR and pb >= PROD_FLOOR:
                            mother_both = True
            mi = min(best_in[(t, a_)], best_in[(t, b_)])
            mo = min(best_out[(t, a_)], best_out[(t, b_)])
            PAR_ROWS.append((name, int(selected), int(chained), int(mother_both), mi, mo))
    n_sel = sum(1 for r in PAR_ROWS if r[0] == name and r[1])
    n_ch = sum(1 for r in PAR_ROWS if r[0] == name and r[1] and r[2])
    n_mb = sum(1 for r in PAR_ROWS if r[0] == name and r[1] and r[3])
    lost = {k: v for k, v in LEDGER_TOT.items()}
    print(f"  ledger {name}: TP edges C0 {len(tp_c0)} HY10 {len(tp_hy)}; lost "
          f"{len(set(tp_c0) - set(tp_hy))} gained {len(set(tp_hy) - set(tp_c0))}; parents selected {n_sel}, "
          f"chained {n_ch}, in-source is a candidate mother of both children {n_mb}", flush=True)
    del lost
# ---- end hypdiag block ----
'''
rep('# ---- end hypilp block ----\n', '# ---- end hypilp block ----\n' + LEDGER)

# ---- stash the two solutions, ledger after the arms ---------------------------------
rep('        del ds_a, n_a, e_a, p_a, aux_a, pm_a, so_a, sc_a, b2_a, seeds_x\n',
    '        DIAG_SOL[arm] = (n_a, e_a, so_a, pm_a)\n'
    '        del ds_a, n_a, e_a, p_a, aux_a, pm_a, so_a, sc_a, b2_a, seeds_x\n')
rep('    extra = ""\n    if name in HOLDOUT_6BBA:',
    '    if "C0" in DIAG_SOL and "HY10" in DIAG_SOL:\n'
    '        diag_ledger(name, T, seeds_h, par_h, DIAG_SOL["C0"], DIAG_SOL["HY10"])\n'
    '    DIAG_SOL.clear()\n'
    '    extra = ""\n    if name in HOLDOUT_6BBA:')

# ---- the summary -------------------------------------------------------------------
SUMMARY = r'''
print("\n=== HYPDIAG LEDGER over the diagnosed movies (GT edges carried by C0 and not by HY10 = lost; "
      "by HY10 and not by C0 = gained; at_division = the GT source node has two GT children) ===", flush=True)
print(f"  TP GT edges carried: C0 {LEDGER_TOT[('tp', 'C0', False)]:,}  HY10 {LEDGER_TOT[('tp', 'HY10', False)]:,}", flush=True)
for kind in ("lost", "gained"):
    ks = sorted({k[1] for k in LEDGER_TOT if k[0] == kind})
    for kl in ks:
        n_all = LEDGER_TOT[(kind, kl, False)] + LEDGER_TOT[(kind, kl, True)]
        print(f"  {kind:<7} {kl:<20} {n_all:>7,}   of which at a GT division {LEDGER_TOT[(kind, kl, True)]:>5,}", flush=True)
_pr = np.array([(r[1], r[2], r[3], r[4], r[5]) for r in PAR_ROWS], float) if PAR_ROWS else np.zeros((0, 5))
if len(_pr):
    sel = _pr[:, 0] == 1
    print(f"\n  parents offered {len(_pr):,}, selected {int(sel.sum()):,}; of the selected: chained to another parent "
          f"{int(_pr[sel, 1].sum()):,} ({_pr[sel, 1].mean():.1%}), in-source is a candidate mother of both children "
          f"{int(_pr[sel, 2].sum()):,} ({_pr[sel, 2].mean():.1%})", flush=True)
    for lab, col in (("min over the two children of their best IN probability", 3),
                     ("min over the two children of their best OUT probability", 4)):
        qs = (0.1, 0.25, 0.5, 0.75, 0.9)
        a_ = " / ".join(f"{np.quantile(_pr[sel, col], q):.2f}" for q in qs)
        b_ = " / ".join(f"{np.quantile(_pr[~sel, col], q):.2f}" for q in qs)
        print(f"  {lab}: selected {a_}   unselected {b_}   (quantiles 10/25/50/75/90)", flush=True)
    for thr in (0.5, 0.7, 0.8, 0.9):
        weak = np.minimum(_pr[:, 3], _pr[:, 4]) < thr
        print(f"  'weak children' gate min(best in, best out) < {thr}: keeps {weak.mean():.1%} of offered parents, "
              f"{(weak & sel).sum() / max(1, sel.sum()):.1%} of the selected", flush=True)
with open("/kaggle/working/hypdiag_ledger.csv", "w", encoding="utf-8", newline="\n") as _fh:
    _fh.write("dataset,kind,klass,at_division\n")
    for r in LEDGER_ROWS:
        _fh.write(",".join(str(x) for x in r) + "\n")
with open("/kaggle/working/hypdiag_parents.csv", "w", encoding="utf-8", newline="\n") as _fh:
    _fh.write("dataset,selected,chained,mother_both,min_child_best_in,min_child_best_out\n")
    for r in PAR_ROWS:
        _fh.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.4f},{r[5]:.4f}\n")
print("  hypdiag_ledger.csv + hypdiag_parents.csv written", flush=True)
for _arm_, _per_ in (("REF", PER),) + tuple((a_, ARM_PER[a_]) for a_, _c in BR_ARMS):
    _csv_a = pathlib.Path("/kaggle/working") / f"hypdiag_{_arm_}.csv"
    _keys_a = sorted({k for r in _per_.values() for k in r})
    with open(_csv_a, "w", encoding="utf-8", newline="\n") as _fh:
        _fh.write("dataset,fold," + ",".join(_keys_a) + "\n")
        for _n in sorted(_per_):
            _fh.write(f"{_n},{FOLD_OF.get(_n, -1)}," + ",".join(repr(_per_[_n].get(_k)) for _k in _keys_a) + "\n")
warning_summary()
print(f"\nHYPDIAG DONE in {(time.time() - t_all) / 60:.1f} min, {len(ROWS)} of {len(DIAG_SIX)} movies", flush=True)
# the report below (gate cross, bootstrap) needs the 128-movie fork population and crashed
# on 16 movies in v1 (np.quantile of an empty gate); the diagnostic ends here.
print("HYPDIAG OK" if len(ROWS) == len(DIAG_SIX) and not INCOMPLETE else "HYPDIAG INCOMPLETE", flush=True)
raise SystemExit(0)
'''
# v1 placed the summary at the very end, behind the gate cross, which crashed on 16 movies
# before reaching it; the summary now runs right after stage B and the kernel stops there.
rep('\nBR_ARMS_RUN = list(BR_ARMS)\n', SUMMARY + '\nBR_ARMS_RUN = list(BR_ARMS)\n')
rep('f"hypilp_REF_{DET_SRC}.csv"', 'f"hypdiag_REF_{DET_SRC}.csv"')
rep('f"hypilp_arm_{a_}.csv"', 'f"hypdiag_arm_{a_}.csv"')
rep('print("HYPILP OK", flush=True)', 'print("HYPDIAG OK", flush=True)')
rep('print(f"[degraded] HYPILP INCOMPLETE or anchor miss:', 'print(f"[degraded] HYPDIAG INCOMPLETE or anchor miss:')
rep("if len(ROWS) == len(SIX) and not INCOMPLETE and _r2:", "if len(ROWS) == len(DIAG_SIX) and not INCOMPLETE:")

assert src.count("diag_ledger(") == 2
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hypdiag.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written hypdiag.py", len(src))
