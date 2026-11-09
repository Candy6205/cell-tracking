"""bridgerank.py -> dupsup.py: suppress duplicate detections before the ILP.

WHY (swapplace + swapimg, 2026-09-12). The 1,643 swap edges of the shipped
arm sit where ONE annotated cell carries TWO detections about 7 um apart
(6.1-8.4 um IQR), each 3.6-4.6 um off the annotated centroid, the centroid
between them (median 1.9 um from their midpoint), the raw intensity flat
across the pair (82% one maximum along a->b, dip 0.02) and no raw peak at
the centroid (17% within 2 um). Which detection the scorer calls the cell
turns on a fraction of a z plane, so the linker's nearest choice and the
scorer's match disagree. Two annotated nuclei are almost never that close
(154 pairs within 7 um among 113,121 GT nodes; GT sisters 11.5 um apart,
9.6% under 6.5), so a pair of primaries within ~7 um is nearly always one
cell. NMS_UM 4.0 lets these pairs through.

THE ARM. A second, wider suppression on the hpara primaries, greedy by
score in 3D physical distance at radius R, applied to the ILP's candidate
edges (a suppressed primary gets no edge) and to the solved nodes (a
suppressed node is dropped before the post-link stages). The features, the
head cache and the (t, i, j) indexing are untouched: the pair set is the
shipped one, only the ILP sees fewer of it. This keeps the cache join
intact at the price that a kept detection stays where it was (3.6-4.6 um
off centre) -- moving it to the pair's midpoint needs a new head cache and
is the follow-up if this arm gains.

ARMS (each a full re-solve, like hparabench's dw35): SUP55 / SUP65 / SUP75 =
R 5.5 / 6.5 / 7.5 um, removing 1.7% / 5.1% / 11.8% of primaries (local count
on the divmiss caches). READ edge_J, adj_J and the node term SEPARATELY: the
scorer rewards under-detection without a clamp (adj = J (1 - 0.1 ratio),
REF ratio -0.023), so removing 12% of nodes buys about +0.01 of adj_J from
the node term alone even if no edge changes; edge_J is the identity effect.
The paired bootstrap prints score, adj_J and edge_J.

REMOVED from bridgerank: the bridge collection, pass 2a/2b, the pickles.
`_gap_close` runs as shipped. Anchors unchanged: REF 55/237/70 adj_J 0.9146,
REF + joint 0.9433. Attach: rkedge's 16 sources. CPU, ~5.5 h (four solves
per movie).
"""
import os

src = open("bridgerank.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:120])
    src = src.replace(old, new)


def cut(start, end, new):
    global src
    assert src.count(start) == 1, (src.count(start), start[:80])
    i = src.index(start)
    j = src.index(end, i)
    src = src[:i] + new + src[j:]


rep('SCRIPT_VERSION = "2026-09-11b-bridgerank"', 'SCRIPT_VERSION = "2026-09-12a-dupsup"')

# ---- arms + the suppression -------------------------------------------------
cut("BR_ARMS = [", "DIVW_ARMS = [(a_, DIV_W) for a_, _c in BR_ARMS]",
    '''DUP_ARMS = [("SUP55", dict(R=5.5)), ("SUP65", dict(R=6.5)), ("SUP75", dict(R=7.5))]
BR_ARMS = DUP_ARMS        # the report sections iterate BR_ARMS_RUN / DIVW_ARMS
DUP_STAT = defaultdict(int)


def _dup_keep(seeds, R):
    """Per frame: a boolean mask over the primaries, greedy suppression by
    score within R um (3D physical distance). The primaries, their order and
    their indices are untouched; the mask only decides which ones the ILP may
    use."""
    keep = []
    for t in range(len(seeds)):
        c_ = np.asarray(seeds[t][0], float).reshape(-1, 3)
        s_ = np.asarray(seeds[t][1], float).reshape(-1)
        n = len(c_)
        k = np.ones(n, bool)
        if n >= 2:
            P = c_ * SCALE
            tr = cKDTree(P)
            for i_ in np.argsort(-s_, kind="stable"):
                if not k[i_]:
                    continue
                for j_ in tr.query_ball_point(P[i_], R):
                    if j_ != i_ and s_[j_] <= s_[i_]:
                        k[j_] = False
        keep.append(k)
    return keep


''')

# ---- solve_one: a keep mask ---------------------------------------------------
rep('''def solve_one(dets, T, seeds, feats, model, cols, disapp=None, floor=None,
              divw=None):''',
    '''def solve_one(dets, T, seeds, feats, model, cols, disapp=None, floor=None,
              divw=None, keep=None):''')
rep('''        for (i_, j_), pr in zip(pairs, probs[t]):
            if pr >= (ILP_FLOOR if floor is None else floor):
                cand_edges.append((offset[t] + i_, offset[t + 1] + j_,
                                   float(pr), 0.0))''',
    '''        for (i_, j_), pr in zip(pairs, probs[t]):
            if pr >= (ILP_FLOOR if floor is None else floor):
                if keep is not None and not (keep[t][i_] and keep[t + 1][j_]):
                    DUP_STAT["edges_withheld"] += 1
                    continue
                cand_edges.append((offset[t] + i_, offset[t + 1] + j_,
                                   float(pr), 0.0))''')
rep('''    del graph, probs
    return nodes0, edges0, (cand, cand_sc, cand_trees), (pmap, seed_of_node)''',
    '''    if keep is not None:
        # a suppressed primary never had an edge; drop its node so no
        # post-link stage (gap close, orphan patch) can pick it up
        drop = {g_ for g_, s_ in seed_of_node.items()
                if s_ is not None and not keep[s_[0]][s_[1]]}
        nodes0 = {g_: v_ for g_, v_ in nodes0.items() if g_ not in drop}
        edges0 = [(a_, b_) for a_, b_ in edges0 if a_ not in drop and b_ not in drop]
        seed_of_node = {g_: s_ for g_, s_ in seed_of_node.items() if g_ not in drop}
        DUP_STAT["nodes_dropped"] += len(drop)
    del graph, probs
    return nodes0, edges0, (cand, cand_sc, cand_trees), (pmap, seed_of_node)''')

# ---- pass 1: no bridge collection, the arms re-solve here, no pickles ----------
rep('''    BR_COLLECT = True
    patch_final(nodes0, edges0, pool, outp)
    BR_COLLECT = False
    _brs = dict(BR_STAT)
''', '''    patch_final(nodes0, edges0, pool, outp)
''')
cut('''    cells = [f"br pairs s2 {_brs.get('collect_span2_pairs', 0)} "''',
    '''    extra = ""
    if name in HOLDOUT_6BBA:''',
    '''    cells = []
    n_prim = sum(len(seeds[t][0]) for t in range(T))
    for arm, cfg in BR_ARMS:
        keep = _dup_keep(seeds, cfg["R"])
        n_keep = sum(int(k_.sum()) for k_ in keep)
        DUP_STAT.clear()
        outa = pathlib.Path(f"{TMP}/arm_{arm}") / f"{name}.geff"
        outa.parent.mkdir(parents=True, exist_ok=True)
        n_a, e_a, p_a, aux_a = solve_one(dets, T, seeds, feats, MODELS_FOLD[k], COLS,
                                         disapp=ILP_DISAPP, floor=PROD_FLOOR, keep=keep)
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
        DUP_BY_ARM[arm]["primaries"] += n_prim
        DUP_BY_ARM[arm]["suppressed"] += n_prim - n_keep
        for k_, v_ in DUP_STAT.items():
            DUP_BY_ARM[arm][k_] += v_
        cells.append(f"{arm} sup {n_prim - n_keep} nodes {len(n_a)} "
                     f"{row_a['division_tp']}/{row_a['division_fp']}/{row_a['division_fn']} "
                     f"{_adj_of(row_a):.4f}")
        del ds_a, n_a, e_a, p_a, aux_a, pm_a, so_a, sc_a, b2_a, keep

''')
rep("ARM_MOVIE = {a: {} for a, _ in DIVW_ARMS}\n",
    "ARM_MOVIE = {a: {} for a, _ in DIVW_ARMS}\nDUP_BY_ARM = {a: defaultdict(int) for a, _ in DIVW_ARMS}\n")

# ---- pass 2a + 2b: gone ---------------------------------------------------------
cut("# ---- BRIDGERANK PASS 2a: the bridge model, two folds by movie --------------",
    "# ---- report ----------------------------------------------------------------",
    '''BR_ARMS_RUN = list(BR_ARMS)
print("\\n=== SUPPRESSION PER ARM, summed over the 128 movies ===", flush=True)
_dk = ["primaries", "suppressed", "edges_withheld", "nodes_dropped"]
print(f"  {'arm':<8}" + "".join(f"{k_:>17}" for k_ in _dk), flush=True)
for a_, _c in BR_ARMS_RUN:
    d_ = DUP_BY_ARM[a_]
    print(f"  {a_:<8}" + "".join(f"{d_.get(k_, 0):>17,}" for k_ in _dk)
          + f"   ({d_.get('suppressed', 0) / max(1, d_.get('primaries', 1)):.3%} of primaries)", flush=True)

''')
assert "BR_COLLECT = True" not in src and "pickle.dump((nodes0" not in src

# ---- bootstrap: edge_J as a third bank --------------------------------------------
rep('''_BSA = {k: np.array([_with_ratio([XRES[k]["rows"][i] for i in d])["adj_edge_jaccard"] for d in _draws])
        for k in XRES}
for a_, _c in BR_ARMS_RUN:
    lab = f"{a_} + joint"
    if lab not in BS:
        continue
    for nm_, bank in (("score", BS), ("adj_J", _BSA)):''',
    '''_BSA = {k: np.array([_with_ratio([XRES[k]["rows"][i] for i in d])["adj_edge_jaccard"] for d in _draws])
        for k in XRES}


def _ej(rows_):
    tp_ = sum(r["edge_tp"] for r in rows_)
    return tp_ / max(1, tp_ + sum(r["edge_fp"] for r in rows_) + sum(r["edge_fn"] for r in rows_))


_BSE = {k: np.array([_ej([XRES[k]["rows"][i] for i in d]) for d in _draws]) for k in XRES}
print("    (edge_J = plain micro Jaccard, no node term: the identity effect alone)", flush=True)
for a_, _c in BR_ARMS_RUN:
    lab = f"{a_} + joint"
    if lab not in BS:
        continue
    for nm_, bank in (("score", BS), ("adj_J", _BSA), ("edge_J", _BSE)):''')

# ---- names ----------------------------------------------------------------------
rep('f"bridgerank_arm_{a_}.csv"', 'f"dupsup_arm_{a_}.csv"')
rep('f"bridgerank_REF_{DET_SRC}.csv"', 'f"dupsup_REF_{DET_SRC}.csv"')
rep('print("BRIDGERANK OK", flush=True)', 'print("DUPSUP OK", flush=True)')
rep("if len(ROWS) == len(SIX) and not INCOMPLETE and not INCOMPLETE2 and _r2:",
    "if len(ROWS) == len(SIX) and not INCOMPLETE and _r2:")
rep('print(f"[degraded] BRIDGERANK INCOMPLETE or anchor miss:', 'print(f"[degraded] DUPSUP INCOMPLETE or anchor miss:')
cut('print("\\n=== reading it ===", flush=True)', "warning_summary()\nprint(f\"\\nDIVBENCH9 DONE", "")
assert "=== reading it ===" not in src and "bridge_pairs_zh" not in src
assert src.count("keep=keep") == 1 and src.count("_dup_keep(") == 2
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dupsup.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written dupsup.py", len(src))
