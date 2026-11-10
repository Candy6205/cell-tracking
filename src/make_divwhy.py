"""Generate divwhy.py from f4diag.py by asserted replacement.

THE QUESTION (the user's, 2026-09-01, the second half of the edgewhy
pair): for the F4 pipeline that ships at LB 0.925, under what conditions
does a real division go missing, and under what conditions does the
pipeline invent one?

WHY IT IS WORTH A KERNEL OF ITS OWN. f4diag measured that F4 more than
HALVES 6bba's division jaccard -- 0.1316 -> 0.0606 -- against a +0.0168
gain on 44b6's edges. That cost has never been attributed to a stage,
and div_J is 0.1 of the score. Nothing in this project has ever charged
a GT division to the stage that lost it; div_forensics (2026-08-23) came
closest and ran on 12 divisions of the pre-ranker pipeline.

THE LEDGER, one bucket per GT division, first losing stage wins:
  div_mother_missed / div_daughter_missed  -- detection
  div_no_candidate                          -- a daughter outside the
                                               10 um gate
  div_below_floor                           -- ranker prob < ILP_FLOOR
  div_node_absent                           -- endpoint not in the graph
  div_ilp_neither / div_ilp_one_only        -- the solver took one
                                               daughter or none
  div_killed_post                           -- the fork existed and a
                                               post-link stage removed it
  div_recovered_ilp / div_recovered_patch   -- a fork in the FINAL
                                               graph, from the solver
                                               or from the patch

THE ONE CONDITION WITH A KNOWN THRESHOLD. With appearance_weight 0.0 the
solver takes a fork only when the second daughter's edge probability
exceeds DIV_W (0.55 as shipped). div_ilp_one_only should therefore
pile up just below 0.55, and the p_min column tests exactly that -- a
prediction with a number in it, not a story.

Also counts the OTHER direction: every fork the pipeline emits, true or
false, split by whether the ILP chose it or _add_safe_divisions added
it. 26 false divisions bought 5 real ones on the 6bba holdout at last
count, and which stage makes the false ones has never been recorded.

EDGES ARE A SEPARATE KERNEL (edgewhy). Same harness, same three domains,
run alongside this one.

ARM: F4 only. ANCHORS, all recorded: F4-6bba adj_J 0.9216 / div_J 0.0606
and F4-44b6 0.8353 (f4diag, 2026-09-01), F4-zh 0.8115 (fatescore).
zebrahub carries this run: 6bba and 44b6 hold roughly a dozen annotated
divisions between them, and zh holds thousands.
"""
import ast
import pathlib

SRC = pathlib.Path(__file__).parent / "f4diag.py"
DST = pathlib.Path(__file__).parent / "divwhy.py"
text = SRC.read_text(encoding="utf-8")


def rep(old, new, n=1):
    global text
    c = text.count(old)
    assert c == n, f"anchor x{c}, wanted x{n}: {old[:70]!r}"
    text = text.replace(old, new)


rep('"""edgediag -- where do the ~7 missing points of edge_J actually go?',
    '"""divwhy -- under what conditions does F4 lose a real division, and '
    'invent a false one?')
rep('SCRIPT_VERSION = "2026-09-01a-f4diag"',
    'SCRIPT_VERSION = "2026-09-02b-divwhy"')

# ---- 1. covariate helpers, built once per dataset -----------------------
rep('''    # seed (t, i) -> solved-graph id, matched by rounded coordinates:''',
    r'''    # ---- covariate helpers (divwhy) -----------------------------------
    prim_tree = [cKDTree(seeds[t_][0] * SCALE[None, :])
                 if len(seeds[t_][0]) else None for t_ in range(T)]
    cand_by_src = defaultdict(list)
    for (t_, i_, j_), p_ in prob_of.items():
        cand_by_src[(t_, i_)].append((p_, j_))
    for k_ in cand_by_src:
        cand_by_src[k_].sort(reverse=True)

    def _dens(t_, pt_um, r=10.0):
        """primaries within r um -- crowding as the linker sees it."""
        tr = prim_tree[t_] if 0 <= t_ < T else None
        return float(len(tr.query_ball_point(pt_um, r))) if tr else 0.0

    # seed (t, i) -> solved-graph id, matched by rounded coordinates:''')

# ---- 2. the division ledger, appended after the GT-edge pass ------------
rep('''    # ---- node-side ledger ---------------------------------------------''',
    r'''    # ---- GT DIVISIONS, charged to the first stage that loses them -----
    # A GT division is a GT node with two children in the NEXT frame.
    # The bucket order mirrors the edge ledger so the two tables read the
    # same way, and the same seed/prob/graph objects decide it -- nothing
    # here re-derives anything the edge pass already established.
    gt_out = defaultdict(list)
    for u_, v_ in gt_edges:
        if int(gt_t[int(v_)]) == int(gt_t[int(u_)]) + 1:
            gt_out[int(u_)].append(int(v_))
    dd = DAGG.setdefault(domain, {c: 0 for c in DCATS})
    dd.setdefault("gt_divisions", 0)
    dd.setdefault("gt_multi", 0)
    drows = []
    for u_, kids in gt_out.items():
        if len(kids) < 2:
            continue
        dd["gt_divisions"] += 1
        if len(kids) > 2:
            dd["gt_multi"] += 1
        v1, v2 = int(kids[0]), int(kids[1])
        tu_ = int(gt_t[u_])
        su_ = m_seed.get(u_)
        s1_, s2_ = m_seed.get(v1), m_seed.get(v2)
        pa_ = pb_ = np.nan
        if su_ is None:
            b_ = "div_mother_missed"
        elif s1_ is None or s2_ is None:
            b_ = "div_daughter_missed"
        else:
            _x = prob_of.get((tu_, su_[1], s1_[1]))
            _y = prob_of.get((tu_, su_[1], s2_[1]))
            pa_ = np.nan if _x is None else float(_x)
            pb_ = np.nan if _y is None else float(_y)
            if _x is None or _y is None:
                b_ = "div_no_candidate"
            elif min(_x, _y) < ILP_FLOOR:
                b_ = "div_below_floor"
            else:
                ga_ = seed_id.get(su_)
                g1_, g2_ = seed_id.get(s1_), seed_id.get(s2_)
                if ga_ is None or g1_ is None or g2_ is None:
                    b_ = "div_node_absent"
                else:
                    # The FINAL graph decides recovery, not the ILP.
                    # _add_safe_divisions runs AFTER linking and can add
                    # the second daughter the solver refused, so asking
                    # e0set first files a rescued division under
                    # div_ilp_one_only. v1 of this script did exactly
                    # that and reported div_recovered ~ 0 while the
                    # emitted-fork table counted 20 true forks on zh.
                    n0_ = int((ga_, g1_) in e0set) + int((ga_, g2_) in e0set)
                    nf_ = int((ga_, g1_) in efset) + int((ga_, g2_) in efset)
                    if nf_ == 2:
                        b_ = ("div_recovered_ilp" if n0_ == 2
                              else "div_recovered_patch")
                    elif n0_ == 2:
                        b_ = "div_killed_post"
                    elif n0_ == 1:
                        b_ = "div_ilp_one_only"
                    else:
                        b_ = "div_ilp_neither"
        dd[b_] += 1
        pm_ = gt_pos[u_] * SCALE
        q1_ = gt_pos[v1] * SCALE
        q2_ = gt_pos[v2] * SCALE
        d1_ = float(np.linalg.norm(q1_ - pm_))
        d2_ = float(np.linalg.norm(q2_ - pm_))
        both = [x for x in (pa_, pb_) if np.isfinite(x)]
        # MERGE TEST. When one daughter is missed, how far is it from the
        # primary that DID match its sister? NMS_UM is 4.0, so a missed
        # daughter sitting inside that radius of its sister's detection
        # is one peak serving two cells, not a cell the detector failed
        # to see. This is the column that separates the two.
        mg_ = np.nan
        if b_ == "div_daughter_missed":
            fnd = s1_ if s1_ is not None else s2_
            if fnd is None:
                dd["both_daughters_missed"] = \
                    dd.get("both_daughters_missed", 0) + 1
            else:
                mis = q2_ if s1_ is not None else q1_
                fp_ = np.asarray(seeds[fnd[0]][0][fnd[1]], float) * SCALE
                mg_ = float(np.linalg.norm(mis - fp_))
        drows.append((float(DCATS.index(b_)),
                      float(np.linalg.norm(q1_ - q2_)),
                      max(d1_, d2_), min(d1_, d2_),
                      _dens(tu_, pm_),
                      float(min(both)) if len(both) == 2 else np.nan,
                      float(max(both)) if both else np.nan,
                      (float(seeds[tu_][1][su_[1]]) if su_ is not None
                       else np.nan),
                      float(pm_[0]), float(tu_) / max(1, T),
                      float(len(cand_by_src.get((tu_, su_[1]), ())))
                      if su_ is not None else np.nan,
                      mg_))
    if drows:
        DROWS.setdefault(domain, []).append(np.asarray(drows, np.float32))

    # ---- EVERY FORK THE PIPELINE EMITS, true or false -----------------
    gt_pair = {}
    for u_, kids in gt_out.items():
        if len(kids) >= 2:
            gt_pair[u_] = set(int(k_) for k_ in kids)
    seed_gids = set(seed_id.values())
    out_f = defaultdict(list)
    for a_, b2_ in edges_f:
        out_f[a_].append(b2_)
    node2gt_f = {}
    krows = []
    kk = KAGG.setdefault(domain, defaultdict(int))
    fin_by_t2 = defaultdict(list)
    for g_, v_ in nodes_f.items():
        fin_by_t2[int(v_["t"])].append(g_)
    for t_, ids_ in fin_by_t2.items():
        rows2_ = gt_by_t.get(t_, [])
        if not rows2_:
            continue
        g_um2 = gt_pos[rows2_] * SCALE[None, :]
        f_um2 = np.array([nodes_f[g_]["xyz"] for g_ in ids_], float) \
            * SCALE[None, :]
        mm2 = _greedy_match(g_um2, f_um2)
        inv2 = {dj: rows2_[gi] for gi, dj in mm2.items()}
        for k2, g_ in enumerate(ids_):
            if k2 in inv2:
                node2gt_f[g_] = inv2[k2]
    for a_, kids in out_f.items():
        if len(kids) < 2:
            continue
        kk["forks"] += 1
        ua_ = node2gt_f.get(a_)
        got = set(node2gt_f.get(k_) for k_ in kids)
        true_ = 0.0
        if ua_ is not None and ua_ in gt_pair:
            true_ = 1.0 if len(gt_pair[ua_] & got) >= 2 else 0.0
        kk["true" if true_ else "false"] += 1
        if all((a_, k_) in e0set for k_ in kids):
            prov_ = 0.0
            kk["prov_ilp"] += 1
        elif any(k_ not in seed_gids or a_ not in seed_gids
                 for k_ in kids):
            prov_ = 1.0
            kk["prov_bridge"] += 1
        else:
            prov_ = 2.0
            kk["prov_divpatch"] += 1
        qs = [np.asarray(nodes_f[k_]["xyz"], float) * SCALE for k_ in kids]
        pa2 = np.asarray(nodes_f[a_]["xyz"], float) * SCALE
        krows.append((true_, prov_,
                      float(np.linalg.norm(qs[0] - qs[1])),
                      _dens(int(nodes_f[a_]["t"]), pa2),
                      float(len(kids)),
                      float(max(np.linalg.norm(q_ - pa2) for q_ in qs)),
                      1.0 if ua_ is not None else 0.0))
    if krows:
        KROWS.setdefault(domain, []).append(np.asarray(krows, np.float32))

    # ---- node-side ledger ---------------------------------------------''')

# ---- 3. row stores ------------------------------------------------------
rep('''NAGG = {}          # domain -> node-side counters''',
    '''NAGG = {}          # domain -> node-side counters
DCATS = ["div_mother_missed", "div_daughter_missed", "div_no_candidate",
         "div_below_floor", "div_node_absent", "div_ilp_neither",
         "div_ilp_one_only", "div_killed_post", "div_recovered_ilp",
         "div_recovered_patch"]
DAGG = {}          # domain -> GT-division bucket counters
DROWS = {}         # domain -> [array of per-GT-division covariate rows]
KAGG = {}          # domain -> emitted-fork counters
KROWS = {}         # domain -> [array of per-emitted-fork rows]
DCOLS = ["bucket", "sister_um", "md_max_um", "md_min_um", "dens10",
         "p_min", "p_max", "sc_mother", "z_um", "t_frac", "n_cand",
         "merge_um"]
KCOLS = ["is_true", "prov", "sister_um", "dens10", "n_kids", "reach_um",
         "parent_matched"]''')

# ---- 4. one arm, three domains ------------------------------------------
rep('''for _arm in ("BASE", "F4"):
  for _dm, names in (("6bba", SET_6BBA), ("44b6", SET_44B6)):''',
    '''for _arm in ("F4",):
  for _dm, names in (("6bba", SET_6BBA), ("44b6", SET_44B6)):''')

# ---- 5. zebrahub back in, on F4 detections. It carries this run ---------
rep('''# ---- zebrahub: dropped, the question named 6bba and 44b6 ----------------
if False:''',
    '''# ---- zebrahub: IN, and it CARRIES this run. The competition train
# split holds 151 annotated divisions across 199 datasets, so the 24
# sampled here hold roughly a dozen; Zebrahub holds thousands and its GT
# covers every cell, so a false fork there is a real false fork.
if SET_ZH:''')
rep('''        dets, T, gt_arr, gt_e, n_cross = zh_load(name)
        diag_one("zh", name, dets, T, gt_arr[:, 0].astype(int),''',
    '''        _db, T, gt_arr, gt_e, n_cross = zh_load(name)
        _z4 = np.load(FDET_F4[name])
        dets = [(_z4[f"c{t}"], _z4[f"s{t}"]) for t in range(T)]
        del _db
        diag_one("F4-zh", name, dets, T, gt_arr[:, 0].astype(int),''')
rep('''    res["zh"] = summarise(zrows)
    PER2["zh"] = None''',
    '''    res["F4-zh"] = summarise(zrows)
    PER2["F4-zh"] = None''')
rep('''    elif d == "zh" and _ZROWS:''',
    '''    elif d.endswith("zh") and _ZROWS:''')
rep('''DOMAINS = [d for d in ("BASE-6bba", "F4-6bba", "BASE-44b6", "F4-44b6")
           if d in res]''',
    '''DOMAINS = [d for d in ("F4-6bba", "F4-44b6", "F4-zh") if d in res]''')
rep('''print("  BASE-6bba must read adj_J 0.9293 (dcpool and edgediag, same "
      "machinery). BASE-44b6 must read 0.8185 (fusescore). F4 is then "
      "ONE documented change away from an anchored harness: the "
      "detection cache. Both arms run D187, which is what ref 55889500 "
      "ships.", flush=True)''',
    '''print("  F4-6bba must read adj_J 0.9216 and div_J 0.0606, F4-44b6 "
      "0.8353 (f4diag, 2026-09-01, same machinery and caches), F4-zh "
      "0.8115 (mergescore/fatescore). All three on D187, which is what "
      "ref 55889500 ships. If an anchor misses, the division tables "
      "below describe a pipeline nobody has scored.", flush=True)''')

# ---- 6. the division report ---------------------------------------------
rep('''print(f"\\nEDGEDIAG DONE in {(time.time() - t_all) / 60:.1f} min",
      flush=True)''',
    r'''
# ------------------------------------------------------------------------
# DIVISIONS. Everything above is the edge pipeline; everything below is
# the 0.1 term nobody has ever charged to a stage.
# ------------------------------------------------------------------------
def _cat(store, d):
    v = store.get(d)
    return np.concatenate(v, axis=0) if v else None


def _med(arr, col):
    v = arr[:, col]
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else float("nan")


def _fmt(x, w=7, p=2):
    return f"{x:>{w}.{p}f}" if np.isfinite(x) else f"{'-':>{w}}"


print("\n\n=== WHERE EVERY GT DIVISION GOES (first losing stage) ===",
      flush=True)
print("  A GT division is an annotated node with two children in the "
      "next frame. mother_missed / daughter_missed are detection. "
      "no_candidate = a daughter sits outside the 10 um gate. "
      "below_floor = the ranker scored a mother-daughter pair under "
      "0.10. ilp_one_only = the solver linked ONE daughter and refused "
      "the fork -- with appearance_weight 0.0 it takes a second child "
      "only when that edge's probability exceeds DIV_W "
      f"({DIV_W}). killed_post = the fork existed and a post-link "
      "stage removed it.", flush=True)
for d in DOMAINS:
    dd = DAGG.get(d)
    if not dd:
        continue
    tot = max(1, dd.get("gt_divisions", 0))
    print(f"\n  -- {d}: {dd.get('gt_divisions', 0):,} GT divisions "
          f"({dd.get('gt_multi', 0):,} with more than two children, "
          f"counted on the first two) --", flush=True)
    for c in DCATS:
        print(f"  {c:<22} {dd[c]:>8,}  {100.0 * dd[c] / tot:>6.2f}%",
              flush=True)

print("\n=== THE CONDITIONS, per bucket ===", flush=True)
print("  sister = how far apart the two daughters are in the frame "
      "AFTER the division. md_max = the longer mother-to-daughter step. "
      "dens10 = primaries within 10 um of the mother. p_min / p_max = "
      "the ranker's two mother-daughter probabilities. n_cand = how many "
      "candidates the mother had at all. merge_um (daughter_missed only) "
      "= how far the missed daughter is from the primary that matched "
      "its sister; under NMS_UM 4.0 means one peak served both. The "
      "two `div_recovered_*` rows are the control.",
      flush=True)
for d in DOMAINS:
    A = _cat(DROWS, d)
    if A is None:
        continue
    print(f"\n  -- {d}: {len(A):,} GT divisions --", flush=True)
    print(f"  {'bucket':<22} {'n':>7} {'sister':>7} {'md_max':>7} "
          f"{'md_min':>7} {'dens10':>7} {'p_min':>7} {'p_max':>7} "
          f"{'sc_mum':>7} {'n_cand':>7}", flush=True)
    for ci, c in enumerate(DCATS):
        S = A[A[:, 0] == ci]
        if not len(S):
            continue
        print(f"  {c:<22} {len(S):>7,} {_fmt(_med(S, 1))} "
              f"{_fmt(_med(S, 2))} {_fmt(_med(S, 3))} {_fmt(_med(S, 4))} "
              f"{_fmt(_med(S, 5), 7, 3)} {_fmt(_med(S, 6), 7, 3)} "
              f"{_fmt(_med(S, 7), 7, 3)} {_fmt(_med(S, 10))}", flush=True)

print("\n=== THE FORK PRICE: p_min against DIV_W ===", flush=True)
print(f"  The solver takes the second daughter only when that edge's "
      f"probability clears {DIV_W}. If the loss is the PRICE, "
      f"div_ilp_one_only piles up just below it and the recovered rows "
      f"above. If the bins do not separate there, the price is not what "
      f"is losing them and lowering DIV_W would not buy the "
      f"divisions back.", flush=True)
for d in DOMAINS:
    A = _cat(DROWS, d)
    if A is None:
        continue
    S = A[np.isfinite(A[:, 5])]
    if not len(S):
        continue
    REC = [float(DCATS.index(c)) for c in
           ("div_recovered_ilp", "div_recovered_patch")]
    one = float(DCATS.index("div_ilp_one_only"))
    print(f"\n  -- {d}: {len(S):,} GT divisions with both pairs scored --",
          flush=True)
    print(f"  {'p_min':<12} {'n':>7} {'recovered':>10} {'one_only':>10} "
          f"{'other':>8}", flush=True)
    for lo, hi in ((0.0, 0.10), (0.10, 0.30), (0.30, 0.55),
                   (0.55, 0.80), (0.80, 1.01)):
        R = S[(S[:, 5] >= lo) & (S[:, 5] < hi)]
        if not len(R):
            continue
        nr = float(sum((R[:, 0] == k).sum() for k in REC))
        no = float((R[:, 0] == one).sum())
        print(f"  {lo:.2f}-{hi:.2f}   {len(R):>7,} "
              f"{100.0 * nr / len(R):>9.1f}% {100.0 * no / len(R):>9.1f}% "
              f"{100.0 * (len(R) - nr - no) / len(R):>7.1f}%", flush=True)

print("\n=== RECOVERY BY SISTER SEPARATION ===", flush=True)
print("  The two daughters right after the division. Very close means "
      "one detection may cover both; far means the pair may fall outside "
      "the gate.", flush=True)
for d in DOMAINS:
    A = _cat(DROWS, d)
    if A is None:
        continue
    REC = [float(DCATS.index(c)) for c in
           ("div_recovered_ilp", "div_recovered_patch")]
    print(f"\n  -- {d} --", flush=True)
    print(f"  {'sister um':<12} {'n':>7} {'recovered':>10} "
          f"{'detect-lost':>12} {'link-lost':>10}", flush=True)
    det_lost = {float(DCATS.index(c)) for c in
                ("div_mother_missed", "div_daughter_missed")}
    for lo, hi in ((0, 3), (3, 5), (5, 8), (8, 12), (12, 1e9)):
        R = A[(A[:, 1] >= lo) & (A[:, 1] < hi)]
        if not len(R):
            continue
        nr = float(sum((R[:, 0] == k).sum() for k in REC))
        nd = float(sum((R[:, 0] == k).sum() for k in det_lost))
        lbl = f"{lo:g}-{hi:g}" if hi < 1e8 else f"{lo:g}+"
        print(f"  {lbl:<12} {len(R):>7,} {100.0 * nr / len(R):>9.1f}% "
              f"{100.0 * nd / len(R):>11.1f}% "
              f"{100.0 * (len(R) - nr - nd) / len(R):>9.1f}%", flush=True)

print("\n\n=== EVERY FORK THE PIPELINE EMITS ===", flush=True)
print("  prov = which stage made it: ILP chose both edges from ranked "
      "candidates, BRIDGE means a _gap_close insert is one of the two "
      "children, DIVPATCH means _add_safe_divisions added the second "
      "daughter after linking. true = the parent matched a GT division "
      "and at least two children matched its GT daughters.", flush=True)
PROVN = ["ILP", "BRIDGE", "DIVPATCH"]
for d in DOMAINS:
    kk = KAGG.get(d)
    K = _cat(KROWS, d)
    if not kk or K is None:
        continue
    tot = max(1, kk.get("forks", 0))
    print(f"\n  -- {d}: {kk.get('forks', 0):,} emitted forks, "
          f"{kk.get('true', 0):,} true / {kk.get('false', 0):,} false --",
          flush=True)
    print(f"  {'prov':<10} {'n':>8} {'share':>7} {'true':>8} "
          f"{'sister':>7} {'dens10':>7} {'reach':>7} {'parent GT':>10}",
          flush=True)
    for pi, pn in enumerate(PROVN):
        S = K[K[:, 1] == pi]
        if not len(S):
            continue
        nt = float(S[:, 0].sum())
        pm = float(S[:, 6].sum())
        print(f"  {pn:<10} {len(S):>8,} {100.0 * len(S) / tot:>6.1f}% "
              f"{100.0 * nt / len(S):>7.1f}% {_fmt(_med(S, 2))} "
              f"{_fmt(_med(S, 3))} {_fmt(_med(S, 5))} "
              f"{100.0 * pm / len(S):>9.1f}%", flush=True)

print("\n=== reading the divisions ===", flush=True)
print("  If most GT divisions die at mother_missed / daughter_missed, "
      "the division term is a DETECTION problem and no solver knob "
      "touches it. If they die at ilp_one_only with p_min just under "
      f"{DIV_W}, it is a PRICE problem and DIV_W is the knob -- "
      "but the emitted-fork table prices that move, because lowering the "
      "fork price buys false forks at whatever rate the false column "
      "already shows. If they die at no_candidate with a large sister "
      "separation, it is the 10 um gate.", flush=True)
print("  ⚠ 6bba and 44b6 hold roughly a dozen annotated divisions "
      "between them in this sample -- read their rows as counts, not "
      "rates. zh carries the statistics and is the only domain where a "
      "false fork is really false (about 2.5% of its cells are "
      "unlabelled, per zhcomplete).", flush=True)
print(f"\nDIVWHY DONE in {(time.time() - t_all) / 60:.1f} min",
      flush=True)''')

DST.write_text(text, encoding="utf-8")
ast.parse(text)
print(f"wrote {DST} ({len(text.splitlines())} lines), ast OK")
