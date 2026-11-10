"""Generate edgewhy.py from f4diag.py by asserted replacement.

THE QUESTION (the user's, 2026-09-01): for the F4 pipeline that ships at
LB 0.925, under WHAT CONDITIONS does an edge go missing (漏边) and under
what conditions does the linker connect the wrong pair (连错边)?

f4diag already counts both: every GT edge charged to the first stage
that loses it, and every final edge classified against GT. What it does
NOT record is the circumstances -- how far the cell moved, how crowded
its neighbourhood was, how close the nearest distractor detection sat,
what probability the ranker gave the correct pair and where it ranked.
Counts say which stage; covariates say when.

WHAT THIS ADDS
  per GT edge:    bucket + displacement, local primary density, distance
                  from the target to the nearest OTHER primary, source
                  depth, frame position, both detection scores, the
                  ranker probability of the CORRECT pair, its rank among
                  the source's candidates, and the top probability the
                  source got.
  per final edge: class + length, ranker probability, source density,
                  PROVENANCE (ILP / gap-close bridge / division patch),
                  and for a wrong link, how far it landed from the
                  correct target.
  reported as:    per-bucket medians against the `recovered` control,
                  plus loss rate binned by displacement and by density,
                  which is the "under what conditions" answer directly.

DIVISIONS ARE A SEPARATE KERNEL (divwhy). Same harness, same three
domains, run alongside this one.

ARM: F4 only -- BASE is dropped, so this is one arm x three domains
instead of f4diag's two arms x two domains.
ANCHORS, all three already recorded: F4-6bba adj_J 0.9216 and F4-44b6
0.8353 (f4diag, 2026-09-01), F4-zh 0.8115 (fatescore/mergescore).
zebrahub is IN: it is the only domain whose GT covers every cell, so it
is the only place a wrong link can be told from an unannotated one.
"""
import ast
import pathlib

SRC = pathlib.Path(__file__).parent / "f4diag.py"
DST = pathlib.Path(__file__).parent / "edgewhy.py"
text = SRC.read_text(encoding="utf-8")


def rep(old, new, n=1):
    global text
    c = text.count(old)
    assert c == n, f"anchor x{c}, wanted x{n}: {old[:70]!r}"
    text = text.replace(old, new)


rep('"""edgediag -- where do the ~7 missing points of edge_J actually go?',
    '"""edgewhy -- under what conditions does F4 lose an edge or link the '
    'wrong pair?')
rep('SCRIPT_VERSION = "2026-09-01a-f4diag"',
    'SCRIPT_VERSION = "2026-09-02a-edgewhy"')

# ---- 1. covariate helpers, built once per dataset -----------------------
rep('''    # seed (t, i) -> solved-graph id, matched by rounded coordinates:''',
    r'''    # ---- covariate helpers (edgewhy) ----------------------------------
    # Everything here is derived from the same seeds/probs the buckets
    # already use, so no new pass over the data and no new assumption.
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

    def _gap2(t_, pt_um):
        """distance to the SECOND nearest primary: the nearest distractor
        once the cell's own detection is accounted for."""
        tr = prim_tree[t_] if 0 <= t_ < T else None
        if tr is None or tr.n < 2:
            return float("nan")
        d_, _ = tr.query(pt_um, k=2)
        return float(d_[1])

    # seed (t, i) -> solved-graph id, matched by rounded coordinates:''')

# ---- 2. the GT-edge loop, now recording the conditions ------------------
rep('''    for u, v in gt_edges:
        u, v = int(u), int(v)
        tu, tv = int(gt_t[u]), int(gt_t[v])
        a["gt_edges"] += 1
        if tv != tu + 1:
            a["nonadjacent"] += 1        # the metric discards these
            continue
        su, sv = m_seed.get(u), m_seed.get(v)
        if su is None or sv is None:
            miss_sub = (su is None and near_pool.get(u)) or \\
                       (sv is None and near_pool.get(v))
            a["det_miss_subthresh" if miss_sub else "det_miss_cold"] += 1
            continue
        p = prob_of.get((tu, su[1], sv[1]))
        if p is None:
            a["no_candidate"] += 1
            continue
        if p < ILP_FLOOR:
            a["below_floor"] += 1
            continue
        ga, gb = seed_id.get(su), seed_id.get(sv)
        if ga is None or gb is None:
            a["node_absent"] += 1
            continue
        if (ga, gb) not in e0set:
            a["ilp_rejected"] += 1
            continue
        if (ga, gb) not in efset:
            a["killed_post"] += 1
            continue
        a["recovered"] += 1''',
    r'''    gt_out = defaultdict(list)
    for u_, v_ in gt_edges:
        if int(gt_t[int(v_)]) == int(gt_t[int(u_)]) + 1:
            gt_out[int(u_)].append(int(v_))
    erows = []
    for u, v in gt_edges:
        u, v = int(u), int(v)
        tu, tv = int(gt_t[u]), int(gt_t[v])
        a["gt_edges"] += 1
        if tv != tu + 1:
            a["nonadjacent"] += 1        # the metric discards these
            continue
        su, sv = m_seed.get(u), m_seed.get(v)
        if su is None or sv is None:
            miss_sub = (su is None and near_pool.get(u)) or \
                       (sv is None and near_pool.get(v))
            bucket = "det_miss_subthresh" if miss_sub else "det_miss_cold"
        else:
            p = prob_of.get((tu, su[1], sv[1]))
            if p is None:
                bucket = "no_candidate"
            elif p < ILP_FLOOR:
                bucket = "below_floor"
            else:
                ga, gb = seed_id.get(su), seed_id.get(sv)
                if ga is None or gb is None:
                    bucket = "node_absent"
                elif (ga, gb) not in e0set:
                    bucket = "ilp_rejected"
                elif (ga, gb) not in efset:
                    bucket = "killed_post"
                else:
                    bucket = "recovered"
        a[bucket] += 1
        pu = gt_pos[u] * SCALE
        pv = gt_pos[v] * SCALE
        p_true = np.nan
        rank_t = np.nan
        p_top = np.nan
        if su is not None:
            lst = cand_by_src.get((tu, su[1]))
            if lst:
                p_top = float(lst[0][0])
                if sv is not None:
                    for k_ in range(len(lst)):
                        if lst[k_][1] == sv[1]:
                            rank_t = float(k_)
                            p_true = float(lst[k_][0])
                            break
        erows.append((float(CATS.index(bucket)),
                      float(np.linalg.norm(pv - pu)),
                      _dens(tu, pu),
                      _gap2(tv, pv),
                      float(pu[0]),
                      float(tu) / max(1, T),
                      (float(seeds[tu][1][su[1]]) if su is not None
                       else np.nan),
                      (float(seeds[tv][1][sv[1]]) if sv is not None
                       else np.nan),
                      p_true, rank_t, p_top,
                      1.0 if len(gt_out.get(u, ())) >= 2 else 0.0))
    if erows:
        EROWS.setdefault(domain, []).append(np.asarray(erows, np.float32))''')

# ---- 3. the final-edge loop, now recording provenance and conditions ----
rep('''    for aa, bb in edges_f:
        f["final_edges"] += 1
        ua, ub = node2gt.get(aa), node2gt.get(bb)
        if ua is not None and ub is not None:
            if (ua, ub) in gt_edge_set:
                f["tp_like"] += 1
            else:
                f["fp_confusion"] += 1
        elif ua is not None or ub is not None:
            f["fp_half"] += 1
        else:
            f["fp_invisible"] += 1''',
    r'''    seed_gids = set(seed_id.values())
    gid2ti = {g_: ti_ for ti_, g_ in seed_id.items()}
    frows = []
    for aa, bb in edges_f:
        f["final_edges"] += 1
        ua, ub = node2gt.get(aa), node2gt.get(bb)
        if ua is not None and ub is not None:
            cls = "tp_like" if (ua, ub) in gt_edge_set else "fp_confusion"
        elif ua is not None or ub is not None:
            cls = "fp_half"
        else:
            cls = "fp_invisible"
        f[cls] += 1
        pa_ = np.asarray(nodes_f[aa]["xyz"], float) * SCALE
        pb_ = np.asarray(nodes_f[bb]["xyz"], float) * SCALE
        # PROVENANCE. Three stages can put an edge in the final graph and
        # they are not the same kind of claim: the ILP chose it from
        # ranked candidates, _gap_close invented it to bridge a dropout,
        # _add_safe_divisions added it as a second daughter.
        if (aa, bb) in e0set:
            prov = 0.0                                   # ILP
        elif aa not in seed_gids or bb not in seed_gids:
            prov = 1.0                                   # gap-close bridge
        else:
            prov = 2.0                                   # division patch
        pp_ = np.nan
        ti_a, ti_b = gid2ti.get(aa), gid2ti.get(bb)
        if ti_a is not None and ti_b is not None:
            _x = prob_of.get((ti_a[0], ti_a[1], ti_b[1]))
            if _x is not None:
                pp_ = float(_x)
        err = np.nan
        if cls == "fp_confusion":
            tg = gt_out.get(ua, [])
            if tg:
                err = float(min(np.linalg.norm(gt_pos[k_] * SCALE - pb_)
                                for k_ in tg))
        frows.append((float(FPC.index(cls)),
                      float(np.linalg.norm(pb_ - pa_)),
                      pp_, prov,
                      _dens(int(nodes_f[aa]["t"]), pa_),
                      err))
    if frows:
        FROWS.setdefault(domain, []).append(np.asarray(frows, np.float32))''')

# ---- 4. row stores ------------------------------------------------------
rep('''NAGG = {}          # domain -> node-side counters''',
    '''NAGG = {}          # domain -> node-side counters
EROWS = {}         # domain -> [array of per-GT-edge covariate rows]
FROWS = {}         # domain -> [array of per-final-edge covariate rows]
ECOLS = ["bucket", "disp_um", "dens10", "gap2_um", "z_um", "t_frac",
         "sc_src", "sc_tgt", "p_true", "rank_true", "p_top", "is_div"]
FCOLS = ["class", "len_um", "p", "prov", "dens10", "err_um"]''')

# ---- 5. one arm, three domains ------------------------------------------
rep('''for _arm in ("BASE", "F4"):
  for _dm, names in (("6bba", SET_6BBA), ("44b6", SET_44B6)):''',
    '''for _arm in ("F4",):
  for _dm, names in (("6bba", SET_6BBA), ("44b6", SET_44B6)):''')

# ---- 6. zebrahub back in, on F4 detections ------------------------------
rep('''# ---- zebrahub: dropped, the question named 6bba and 44b6 ----------------
if False:''',
    '''# ---- zebrahub: IN. The only domain whose GT covers every cell, so the
# only one where a wrong link can be distinguished from a link into a
# cell nobody labelled. ~2.5% of its cells are unlabelled (zhcomplete),
# so its FP rates are slightly overstated and its FN rates are not.
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
    '''print("  F4-6bba must read adj_J 0.9216 and F4-44b6 0.8353 (f4diag, "
      "2026-09-01, same machinery and the same caches); F4-zh must read "
      "0.8115 (mergescore/fatescore). All three on D187, which is what "
      "ref 55889500 ships. If any anchor misses, the covariate tables "
      "below describe a pipeline nobody has scored.", flush=True)''')

# ---- 7. the new report --------------------------------------------------
rep('''print(f"\\nEDGEDIAG DONE in {(time.time() - t_all) / 60:.1f} min",
      flush=True)''',
    r'''
# ------------------------------------------------------------------------
# THE CONDITIONS. Everything above is counts; everything below is when.
# ------------------------------------------------------------------------
def _cat(store, d):
    v = store.get(d)
    return np.concatenate(v, axis=0) if v else None


def _med(arr, col):
    v = arr[:, col]
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else float("nan")


def _pct(arr, col, q):
    v = arr[:, col]
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if len(v) else float("nan")


def _fmt(x, w=7, p=2):
    return f"{x:>{w}.{p}f}" if np.isfinite(x) else f"{'-':>{w}}"


DISP_BINS = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 1e9]
DENS_BINS = [0.0, 2.0, 5.0, 10.0, 20.0, 1e9]


def _bin_label(b, i):
    lo, hi = b[i], b[i + 1]
    return f"{lo:g}-{hi:g}" if hi < 1e8 else f"{lo:g}+"


print("\n\n=== MISSED EDGES: THE CONDITIONS ===", flush=True)
print("  One row per bucket. disp = how far the cell actually moved "
      "between the two frames. dens10 = primaries within 10 um of the "
      "source, i.e. how many competitors the linker had. gap2 = distance "
      "from the true target to the NEAREST OTHER primary; small means a "
      "distractor sits on top of the right answer. p_true = the ranker's "
      "probability for the CORRECT pair, rank = where that pair sat in "
      "the source's candidate list (0 = the ranker's own first choice), "
      "p_top = the probability of whatever the ranker liked best. "
      "`recovered` is the control: read every other row against it.",
      flush=True)
for d in DOMAINS:
    A = _cat(EROWS, d)
    if A is None:
        continue
    print(f"\n  -- {d}: {len(A):,} adjacent GT edges --", flush=True)
    print(f"  {'bucket':<20} {'n':>7} {'disp':>7} {'p90disp':>8} "
          f"{'dens10':>7} {'gap2':>7} {'z_um':>7} {'sc_src':>7} "
          f"{'p_true':>7} {'rank':>6} {'p_top':>7} {'%div':>6}",
          flush=True)
    for ci, c in enumerate(CATS):
        S = A[A[:, 0] == ci]
        if not len(S):
            continue
        dv = S[:, 11]
        print(f"  {c:<20} {len(S):>7,} {_fmt(_med(S, 1))} "
              f"{_fmt(_pct(S, 1, 90), 8)} {_fmt(_med(S, 2))} "
              f"{_fmt(_med(S, 3))} {_fmt(_med(S, 4))} "
              f"{_fmt(_med(S, 6))} {_fmt(_med(S, 8))} "
              f"{_fmt(_med(S, 9), 6, 1)} {_fmt(_med(S, 10))} "
              f"{100.0 * float(dv.sum()) / len(S):>5.1f}%", flush=True)

print("\n=== LOSS RATE BY DISPLACEMENT (how far the cell moved) ===",
      flush=True)
print("  `lost` = every bucket except recovered. The columns after it "
      "are that bin's share of the loss by cause, so a row reads: at "
      "this speed, this fraction of GT edges is lost, and this is what "
      "loses them.", flush=True)
for d in DOMAINS:
    A = _cat(EROWS, d)
    if A is None:
        continue
    rec = float(CATS.index("recovered"))
    print(f"\n  -- {d} --", flush=True)
    print(f"  {'disp um':<10} {'n':>8} {'lost':>7}   " + " ".join(
        f"{c[:11]:>11}" for c in CATS if c != "recovered"), flush=True)
    for i in range(len(DISP_BINS) - 1):
        S = A[(A[:, 1] >= DISP_BINS[i]) & (A[:, 1] < DISP_BINS[i + 1])]
        if not len(S):
            continue
        lost = float((S[:, 0] != rec).sum())
        parts = []
        for c in CATS:
            if c == "recovered":
                continue
            k = float((S[:, 0] == CATS.index(c)).sum())
            parts.append(f"{100.0 * k / len(S):>10.2f}%")
        print(f"  {_bin_label(DISP_BINS, i):<10} {len(S):>8,} "
              f"{100.0 * lost / len(S):>6.2f}%   " + " ".join(parts),
              flush=True)

print("\n=== LOSS RATE BY CROWDING (primaries within 10 um) ===",
      flush=True)
for d in DOMAINS:
    A = _cat(EROWS, d)
    if A is None:
        continue
    rec = float(CATS.index("recovered"))
    print(f"\n  -- {d} --", flush=True)
    print(f"  {'dens10':<10} {'n':>8} {'lost':>7}   " + " ".join(
        f"{c[:11]:>11}" for c in CATS if c != "recovered"), flush=True)
    for i in range(len(DENS_BINS) - 1):
        S = A[(A[:, 2] >= DENS_BINS[i]) & (A[:, 2] < DENS_BINS[i + 1])]
        if not len(S):
            continue
        lost = float((S[:, 0] != rec).sum())
        parts = []
        for c in CATS:
            if c == "recovered":
                continue
            k = float((S[:, 0] == CATS.index(c)).sum())
            parts.append(f"{100.0 * k / len(S):>10.2f}%")
        print(f"  {_bin_label(DENS_BINS, i):<10} {len(S):>8,} "
              f"{100.0 * lost / len(S):>6.2f}%   " + " ".join(parts),
              flush=True)

print("\n=== WAS THE CORRECT PAIR EVEN RANKED FIRST? ===", flush=True)
print("  Over GT edges whose BOTH endpoints were detected and whose "
      "correct pair was a candidate. rank 0 means the ranker's own top "
      "choice was right, so anything lost from rank 0 was lost by the "
      "ILP's global solve rather than by the score.", flush=True)
for d in DOMAINS:
    A = _cat(EROWS, d)
    if A is None:
        continue
    S = A[np.isfinite(A[:, 9])]
    if not len(S):
        continue
    rec = float(CATS.index("recovered"))
    print(f"\n  -- {d}: {len(S):,} GT edges with a scored correct pair --",
          flush=True)
    print(f"  {'rank':<8} {'n':>8} {'share':>7} {'lost':>7} "
          f"{'med p_true':>11} {'med p_top':>10}", flush=True)
    for lo, hi, lbl in ((0, 1, "0"), (1, 2, "1"), (2, 3, "2"),
                        (3, 5, "3-4"), (5, 1e9, "5+")):
        R = S[(S[:, 9] >= lo) & (S[:, 9] < hi)]
        if not len(R):
            continue
        lost = float((R[:, 0] != rec).sum())
        print(f"  {lbl:<8} {len(R):>8,} {100.0 * len(R) / len(S):>6.2f}% "
              f"{100.0 * lost / len(R):>6.2f}% {_fmt(_med(R, 8), 11, 3)} "
              f"{_fmt(_med(R, 10), 10, 3)}", flush=True)

print("\n\n=== WRONG EDGES: THE CONDITIONS ===", flush=True)
print("  prov = which stage created the edge: ILP chose it from ranked "
      "candidates, BRIDGE is a _gap_close insert across a one-frame "
      "dropout, DIVPATCH is _add_safe_divisions' second daughter. "
      "err_um (fp_confusion only) = how far the chosen target sits from "
      "the correct one, so a small err is a near-miss between two "
      "adjacent cells and a large one is a different cell entirely.",
      flush=True)
PROV = ["ILP", "BRIDGE", "DIVPATCH"]
for d in DOMAINS:
    F = _cat(FROWS, d)
    if F is None:
        continue
    print(f"\n  -- {d}: {len(F):,} final edges --", flush=True)
    print(f"  {'class':<15} {'n':>9} {'len_um':>7} {'p90len':>8} "
          f"{'p':>7} {'dens10':>7} {'err_um':>7}   " + " ".join(
              f"{p:>9}" for p in PROV), flush=True)
    for ci, c in enumerate(FPC):
        S = F[F[:, 0] == ci]
        if not len(S):
            continue
        pv = []
        for k in range(3):
            n_ = float((S[:, 3] == k).sum())
            pv.append(f"{100.0 * n_ / len(S):>8.2f}%")
        print(f"  {c:<15} {len(S):>9,} {_fmt(_med(S, 1))} "
              f"{_fmt(_pct(S, 1, 90), 8)} {_fmt(_med(S, 2), 7, 3)} "
              f"{_fmt(_med(S, 4))} {_fmt(_med(S, 5))}   " + " ".join(pv),
              flush=True)

print("\n=== WRONG-LINK ERROR DISTANCE (fp_confusion only) ===", flush=True)
print("  How far the wrong target sits from the right one. Under ~5 um "
      "is a swap between neighbouring cells; far out is a link across "
      "the field.", flush=True)
for d in DOMAINS:
    F = _cat(FROWS, d)
    if F is None:
        continue
    S = F[(F[:, 0] == float(FPC.index("fp_confusion")))
          & np.isfinite(F[:, 5])]
    if not len(S):
        continue
    print(f"\n  -- {d}: {len(S):,} wrong links with a known correct "
          f"target --", flush=True)
    for lo, hi in ((0, 3), (3, 5), (5, 8), (8, 12), (12, 1e9)):
        k = float(((S[:, 5] >= lo) & (S[:, 5] < hi)).sum())
        lbl = f"{lo:g}-{hi:g} um" if hi < 1e8 else f"{lo:g}+ um"
        print(f"  {lbl:<12} {int(k):>7,} {100.0 * k / len(S):>6.2f}%",
              flush=True)

print("\n=== reading the conditions ===", flush=True)
print("  A bucket whose disp is far above `recovered` is a SPEED "
      "failure and wants a wider gate or a motion model. A bucket whose "
      "dens10 is far above it, or whose gap2 is far below it, is a "
      "CROWDING failure and wants a better score, not a wider gate -- a "
      "wider gate adds more of the distractors that caused it. A bucket "
      "whose p_true is high while rank is 0 was not lost by the ranker "
      "at all and only the ILP or the post-link filters can be "
      "responsible.", flush=True)
print("  ⚠ zh is the only domain whose GT covers every cell, so it is "
      "the only place fp_confusion and fp_invisible mean what they say; "
      "on 6bba and 44b6 almost every edge has no annotated endpoint. "
      "44b6 below_floor stays undercounted (D187 in-sample).", flush=True)
print(f"\nEDGEWHY DONE in {(time.time() - t_all) / 60:.1f} min",
      flush=True)''')

DST.write_text(text, encoding="utf-8")
ast.parse(text)
print(f"wrote {DST} ({len(text.splitlines())} lines), ast OK")
