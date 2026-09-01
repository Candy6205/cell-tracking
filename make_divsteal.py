"""Generate divsteal.py from divgate2.py by asserted replacement.

THE QUESTION. divgate2 measured that 56% of the GT divisions the ILP
half-links are rejected by the orphan-fork patch at condition 5a: the
second daughter is NOT an orphan, because the solver gave her some other
parent. The patch only ever adopts orphans, so it cannot see her.

The obvious repair is to let the patch STEAL -- re-parent a node that
already has a parent. Before anyone builds that, it has to be priced,
and the price is one number: is the edge that would be broken a TRUE
edge?

  * If the competing parent's edge is a real GT edge, stealing destroys
    a true positive to buy a division. Edges carry 0.9 of the score and
    divisions 0.1, so that trade is 9:1 against.
  * If it is a false link, stealing fixes two errors at once and the
    trade is free.

Zebrahub decides this: its GT covers every cell, so "is the competing
edge real" is answerable there and nowhere else. On 6bba and 44b6 the
competing edge usually has no annotated endpoint at all, which is
recorded as `known=0` rather than guessed.

WHAT THIS ADDS, at divgate2's 5a branch and nowhere else:
  p_mother  the ranker's probability for the edge the patch WOULD make
  p_other   the ranker's probability for the edge the ILP DID make
  true_other / known   whether that competing edge is a GT edge, and
            whether both its endpoints are annotated at all
  d_mother / d_other   the two distances

If p_mother sits far below p_other the solver preferred the link it
made, and a steal rule would be overruling the ranker rather than
correcting the patch's blind spot.

Same harness, same three domains, same anchors as divwhy and divgate2.
"""
import ast
import pathlib

SRC = pathlib.Path(__file__).parent / "divgate2.py"
DST = pathlib.Path(__file__).parent / "divsteal.py"
text = SRC.read_text(encoding="utf-8")


def rep(old, new, n=1):
    global text
    c = text.count(old)
    assert c == n, f"anchor x{c}, wanted x{n}: {old[:70]!r}"
    text = text.replace(old, new)


rep('"""divgate2 -- which of the orphan-fork patch\'s eight conditions '
    'rejects a real division?',
    '"""divsteal -- would stealing the second daughter break a true edge?')
rep('SCRIPT_VERSION = "2026-09-02c-divgate2"',
    'SCRIPT_VERSION = "2026-09-02d-divsteal"')

# ---- 1. the mappings the 5a branch needs, built before the replay ------
rep('''    gg = GAGG.setdefault(domain, defaultdict(int))''',
    r'''    # ---- what the 5a branch needs: seed ids backwards, GT edges as a
    # set, and a node -> GT row map on the PRE-PATCH graph.
    gid2ti = {g_: ti_ for ti_, g_ in seed_id.items()}
    gt_pairs = {(int(u_), int(v_)) for u_, kids_ in gt_out.items()
                for v_ in kids_}
    node2gt_pre = {}
    for t_, ids_ in pre_by_t.items():
        rows3_ = gt_by_t.get(t_, [])
        if not rows3_:
            continue
        g_um3 = gt_pos[rows3_] * SCALE[None, :]
        f_um3 = np.array([nodes_f[g_]["xyz"] for g_ in ids_], float) \
            * SCALE[None, :]
        mm3 = _greedy_match(g_um3, f_um3)
        inv3 = {dj: rows3_[gi] for gi, dj in mm3.items()}
        for k3, g_ in enumerate(ids_):
            if k3 in inv3:
                node2gt_pre[g_] = inv3[k3]

    def _pair_prob(src_gid, dst_gid):
        """the ranker's probability for an edge between two primaries."""
        a3, b3 = gid2ti.get(src_gid), gid2ti.get(dst_gid)
        if a3 is None or b3 is None or b3[0] != a3[0] + 1:
            return np.nan
        x3 = prob_of.get((a3[0], a3[1], b3[1]))
        return np.nan if x3 is None else float(x3)

    gg = GAGG.setdefault(domain, defaultdict(int))''')

# ---- 2. the 5a branch, instrumented ------------------------------------
rep('''        if pre_pred.get(q):
            gg["5a_sister_not_orphan"] += 1
            continue''',
    r'''        if pre_pred.get(q):
            gg["5a_sister_not_orphan"] += 1
            other = pre_pred[q][0]
            ua3, ub3 = node2gt_pre.get(other), node2gt_pre.get(q)
            known = 1.0 if (ua3 is not None and ub3 is not None) else 0.0
            true_other = 1.0 if (known and (ua3, ub3) in gt_pairs) else 0.0
            SROWS.setdefault(domain, []).append((
                _pair_prob(ga_, q),
                _pair_prob(other, q),
                true_other, known,
                _dist_um(nodes_f[ga_]["xyz"], nodes_f[q]["xyz"]),
                _dist_um(nodes_f[other]["xyz"], nodes_f[q]["xyz"]),
                1.0 if len(pre_pred.get(q, ())) > 1 else 0.0))
            continue''')

# ---- 3. store -----------------------------------------------------------
rep('''GROWS = {}         # domain -> divergence values that reached condition 8''',
    '''GROWS = {}         # domain -> divergence values that reached condition 8
SROWS = {}         # domain -> one row per 5a case (the steal question)
SCOLS = ["p_mother", "p_other", "true_other", "known", "d_mother",
         "d_other", "multi_parent"]''')

# ---- 4. the report ------------------------------------------------------
rep('''print(f"\\nDIVGATE2 DONE in {(time.time() - t_all) / 60:.1f} min",
      flush=True)''',
    r'''
print("\n\n=== THE STEAL QUESTION: what would be broken ===", flush=True)
print("  One row per GT division rejected at 5a, i.e. every case where "
      "the second daughter already had a parent. p_mother is the edge "
      "the patch WOULD make, p_other the edge the ILP DID make. "
      "true_other says the competing edge is a real GT edge, so stealing "
      "would destroy a true positive; known says both its endpoints are "
      "annotated at all, and where known is low the true_other rate is "
      "not measurable -- only zebrahub answers this.", flush=True)
for d in DOMAINS:
    S = SROWS.get(d)
    if not S:
        continue
    A = np.asarray(S, np.float64)
    n = len(A)
    kn = A[A[:, 3] > 0.5]
    both = A[np.isfinite(A[:, 0]) & np.isfinite(A[:, 1])]
    print(f"\n  -- {d}: {n:,} divisions rejected at 5a --", flush=True)
    print(f"  competing edge annotated at both ends: {len(kn):,} "
          f"({100.0 * len(kn) / max(1, n):.1f}%)", flush=True)
    if len(kn):
        print(f"  of those, the competing edge is a TRUE GT edge: "
              f"{int(kn[:, 2].sum()):,} "
              f"({100.0 * float(kn[:, 2].sum()) / len(kn):.1f}%)",
              flush=True)
    print(f"  the daughter had more than one parent: "
          f"{int(A[:, 6].sum()):,}", flush=True)
    if len(both):
        pm, po = both[:, 0], both[:, 1]
        print(f"  p_mother  median {np.median(pm):.3f}  "
              f"p25 {np.percentile(pm, 25):.3f}  "
              f"p75 {np.percentile(pm, 75):.3f}", flush=True)
        print(f"  p_other   median {np.median(po):.3f}  "
              f"p25 {np.percentile(po, 25):.3f}  "
              f"p75 {np.percentile(po, 75):.3f}", flush=True)
        print(f"  p_mother > p_other in {100.0 * float((pm > po).sum()) / len(both):.1f}% "
              f"of {len(both):,} cases where both are scored", flush=True)
        for thr in (0.0, 0.1, 0.2, 0.3):
            k = float((pm - po > thr).sum())
            print(f"    margin p_mother - p_other > {thr:.1f}: "
                  f"{int(k):,} ({100.0 * k / len(both):.1f}%)", flush=True)
    else:
        print("  neither probability was scored in any case -- the "
              "competing edge is not a ranker candidate", flush=True)
    fm = A[np.isfinite(A[:, 4])]
    if len(fm):
        print(f"  d_mother median {np.median(fm[:, 4]):.2f} um   "
              f"d_other median {np.median(fm[:, 5]):.2f} um", flush=True)

print("\n=== the trade, stated ===", flush=True)
print("  A steal rule is worth building only if the competing edge is "
      "usually FALSE. Edges carry 0.9 of the score and divisions 0.1, so "
      "breaking one true edge to gain one division is 9:1 against on the "
      "edge term alone, before the division is even scored. If instead "
      "p_mother sits below p_other in most cases, the ranker preferred "
      "the link the solver made and the disagreement is with the ranker, "
      "not with the patch.", flush=True)
print(f"\nDIVSTEAL DONE in {(time.time() - t_all) / 60:.1f} min",
      flush=True)''')

DST.write_text(text, encoding="utf-8")
ast.parse(text)
print(f"wrote {DST} ({len(text.splitlines())} lines), ast OK")
