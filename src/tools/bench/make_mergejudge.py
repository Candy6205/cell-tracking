"""dupsup.py -> mergejudge.py: the EDGE-LEVEL judge of every hypcache parent pair on
REF's own final graph.

WHY. mergefeat (2026-09-15) found that among hypcache's parents at d <= 8.5 um the pairs
with both ends annotated are 94% one cell (1,226 vs 74): the two-cell class is not the
loss channel, and the 9,598 pairs with ONE annotated end cannot be typed by both-ends
labels. hmerge's -0.0082 came from firing on those. This kernel types them from REF's
solution, pair by pair, with the annotated detection A and its pair mate B:

  geometry   d(A, G), d(B, G), d(midpoint, G) for G = the GT node A matched (5 um); the
             merged point would sit at the midpoint, and the scorer matches at 7 um
  A's edges  is A's GT in-edge carried by a REF edge (u -> A, u matching G's GT parent)?
             how many of G's GT children are carried by REF edges out of A?
  B's edges  does B carry one of G's GT neighbours (the swap signature: a merge would
             hand that edge to the merged node)? does B have in / out edges to OTHER
             nodes, and how long is the chain through B backward / forward (up to 5)?
  contest    A and B both have an in-edge (from different nodes) / both have out-edges:
             after a merge one node carries one in-edge, the other track loses

The summary counts, by label (one cell / two cells / one end annotated), the shares:
A already right on every GT edge it has; A missing a GT edge that B carries (fixable);
B a track of its own (case 2, a real unannotated cell); B loose (case 3, a ghost);
midpoint farther than 5 / 7 um from G. Rows for every parent pair (unannotated ones
with the chain features only) go to mergejudge_pairs.csv.

Attach: dupsup's 16 sources + biohub-hypcache + biohub-mergefeat. CPU: features ~1 h +
128 REF solves (~2.5 h) + the gate cross (~1 h); no arms.
"""
import os

src = open("dupsup.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-12a-dupsup"', 'SCRIPT_VERSION = "2026-09-16a-mergejudge"')
rep('DUP_ARMS = [("SUP55", dict(R=5.5)), ("SUP65", dict(R=6.5)), ("SUP75", dict(R=7.5))]',
    'DUP_ARMS = []             # mergejudge: REF only')

# ---- the parents and mergefeat's rows ------------------------------------------------
rep('FDET_F4HP = _walk_files(_npz_prefix("fdet_F4hp_"))\nprint(f"fdet_F4hp caches: {len(FDET_F4HP)}", flush=True)\n',
    'FDET_F4HP = _walk_files(_npz_prefix("fdet_F4hp_"))\nprint(f"fdet_F4hp caches: {len(FDET_F4HP)}", flush=True)\n'
    'MJ_HYP = _walk_files(_npz_prefix("fdet_F4hyp_"))          # hypcache: p{t} = children among primaries, w{t} = saddles\n'
    'MJ_MF = _walk_files(_npz_prefix("mergefeat_"))            # mergefeat: t, q, p, lab per scored parent\n'
    'print(f"mergejudge: hypcache caches {len(MJ_HYP)}, mergefeat files {len(MJ_MF)}", flush=True)\n'
    'assert len([n for n in MJ_HYP if n.startswith("6bba_")]) >= 128, "attach biohub-hypcache"\n')

JUDGE = r'''
# ---- mergejudge block ----
MJ_MAX_D = 8.5
MJ_SCORE_UM = 7.0
MJ_ROWS = []
MJ_COLS = ["dataset", "t", "q", "lab", "mf_lab", "mf_p", "d_ab", "saddle", "s_a", "s_b", "d_AG", "d_BG", "d_MG",
           "n_gt_in", "A_in_tp", "n_gt_kids", "A_out_tp", "A_in_n", "A_out_n", "A_back", "A_fwd",
           "B_carries_in", "B_carries_out", "B_in_other", "B_out_other", "B_in_n", "B_out_n", "B_back", "B_fwd",
           "contest_in", "contest_out", "A_alive", "B_alive"]


def _mj_chain(nid, nb, limit=5):
    """Consecutive frames reachable from nid along single links (stop at a fork or a dead end)."""
    n = 0
    cur = nid
    while n < limit:
        nxt = nb.get(cur, [])
        if len(nxt) != 1:
            break
        cur = nxt[0]
        n += 1
    return n


def merge_judge(name, T, seeds, seed_of_node, nodes_f, edges_f, trees, kids):
    """Rows for every hypcache parent pair (children within MJ_MAX_D) on REF's final graph."""
    zh = np.load(MJ_HYP[name])
    zm = np.load(MJ_MF[name]) if name in MJ_MF else None
    mf = {}
    if zm is not None:
        for t_, q_, p_, l_ in zip(zm["t"], zm["q"], zm["p"], zm["lab"]):
            mf[(int(t_), int(q_))] = (float(p_), int(l_))
    gpred = {}
    for g_, hs in kids.items():
        for h_ in hs:
            gpred[h_] = g_
    node_of = {s_: nid for nid, s_ in seed_of_node.items() if s_ is not None and nid in nodes_f}
    in_of, out_of = defaultdict(list), defaultdict(list)
    for a_, b_ in edges_f:
        out_of[a_].append(b_)
        in_of[b_].append(a_)
    # GT match of every final node at MATCH_UM (mergefeat's label rule)
    m_f = {}
    by_t = defaultdict(list)
    for nid, v in nodes_f.items():
        by_t[int(v["t"])].append(nid)
    for t, ids in by_t.items():
        if t not in trees:
            for nid in ids:
                m_f[nid] = -1
            continue
        rows, tree = trees[t]
        P = np.stack([np.asarray(nodes_f[nid]["xyz"], float) for nid in ids]) * SCALE
        dd, gi = tree.query(P)
        for k_, nid in enumerate(ids):
            m_f[nid] = int(rows[gi[k_]]) if dd[k_] <= MATCH_UM else -1
    g_pos = {}
    for t, (rows, tree) in trees.items():
        for r_, x_ in zip(rows, tree.data):
            g_pos[int(r_)] = np.asarray(x_, float)
    n_rows = 0
    for t in range(T):
        if f"p{t}" not in zh.files:
            continue
        par = np.asarray(zh[f"p{t}"], np.int64).reshape(-1, 2)
        sad = np.asarray(zh[f"w{t}"], float) if f"w{t}" in zh.files else np.full(len(par), np.nan)
        if len(par) == 0:
            continue
        C, S = seeds[t]
        Cu = C * SCALE[None, :]
        for q in range(len(par)):
            a_i, b_i = int(par[q, 0]), int(par[q, 1])
            if a_i >= len(C) or b_i >= len(C):
                warn_once("mj_child_index", f"{name}: a parent child index is outside the F4HP primaries")
                continue
            d_ab = float(np.linalg.norm(Cu[a_i] - Cu[b_i]))
            if d_ab > MJ_MAX_D:
                continue
            na, nb_ = node_of.get((t, a_i)), node_of.get((t, b_i))
            ga = m_f.get(na, -1) if na is not None else -1
            gb = m_f.get(nb_, -1) if nb_ is not None else -1
            if na is None or nb_ is None:
                # a child the short filter removed: match it directly
                if t in trees:
                    rows, tree = trees[t]
                    for who, idx in ((0, a_i), (1, b_i)):
                        dd, gi = tree.query(Cu[idx])
                        g_ = int(rows[gi]) if dd <= MATCH_UM else -1
                        if who == 0 and na is None:
                            ga = g_
                        if who == 1 and nb_ is None:
                            gb = g_
            lab = 1 if (ga >= 0 and ga == gb) else 0 if (ga >= 0 and gb >= 0) else (2 if (ga >= 0 or gb >= 0) else -1)
            # A = the annotated end (the closer one when both match the same GT node)
            if lab == 1:
                G = ga
                swap = np.linalg.norm(Cu[b_i] - g_pos[G]) < np.linalg.norm(Cu[a_i] - g_pos[G])
            elif lab == 2:
                swap = ga < 0
                G = gb if swap else ga
            else:
                G = ga if ga >= 0 else -1
                swap = False
            if swap:
                a_i, b_i, na, nb_ = b_i, a_i, nb_, na
            mfp, mfl = mf.get((t, q), (np.nan, -9))
            row = dict(dataset=name, t=t, q=q, lab=lab, mf_lab=mfl, mf_p=mfp, d_ab=d_ab,
                       saddle=float(sad[q]) if q < len(sad) else np.nan, s_a=float(S[a_i]), s_b=float(S[b_i]),
                       A_alive=int(na is not None), B_alive=int(nb_ is not None))
            row.update(A_in_n=len(in_of.get(na, [])) if na is not None else 0,
                       A_out_n=len(out_of.get(na, [])) if na is not None else 0,
                       A_back=_mj_chain(na, in_of) if na is not None else 0,
                       A_fwd=_mj_chain(na, out_of) if na is not None else 0,
                       B_in_n=len(in_of.get(nb_, [])) if nb_ is not None else 0,
                       B_out_n=len(out_of.get(nb_, [])) if nb_ is not None else 0,
                       B_back=_mj_chain(nb_, in_of) if nb_ is not None else 0,
                       B_fwd=_mj_chain(nb_, out_of) if nb_ is not None else 0)
            ins_a = set(in_of.get(na, [])) if na is not None else set()
            ins_b = set(in_of.get(nb_, [])) if nb_ is not None else set()
            row["contest_in"] = int(bool(ins_a) and bool(ins_b) and not (ins_a & ins_b))
            row["contest_out"] = int(row["A_out_n"] > 0 and row["B_out_n"] > 0)
            if G >= 0:
                gp = gpred.get(G)
                gk = kids.get(G, [])
                mid = 0.5 * (Cu[a_i] + Cu[b_i])
                row.update(d_AG=float(np.linalg.norm(Cu[a_i] - g_pos[G])), d_BG=float(np.linalg.norm(Cu[b_i] - g_pos[G])),
                           d_MG=float(np.linalg.norm(mid - g_pos[G])), n_gt_in=int(gp is not None), n_gt_kids=len(gk))
                row["A_in_tp"] = int(gp is not None and any(m_f.get(u, -1) == gp for u in in_of.get(na, [])))
                row["A_out_tp"] = int(sum(1 for h_ in gk if any(m_f.get(v, -1) == h_ for v in out_of.get(na, []))))
                row["B_carries_in"] = int(gp is not None and any(m_f.get(u, -1) == gp for u in in_of.get(nb_, [])))
                row["B_carries_out"] = int(any(m_f.get(v, -1) in gk for v in out_of.get(nb_, [])) if gk else 0)
                row["B_in_other"] = int(any(m_f.get(u, -1) != gp for u in in_of.get(nb_, [])))
                row["B_out_other"] = int(any(m_f.get(v, -1) not in gk for v in out_of.get(nb_, [])))
            else:
                row.update(d_AG=np.nan, d_BG=np.nan, d_MG=np.nan, n_gt_in=0, n_gt_kids=0, A_in_tp=0, A_out_tp=0,
                           B_carries_in=0, B_carries_out=0, B_in_other=int(bool(ins_b)), B_out_other=int(row["B_out_n"] > 0))
            MJ_ROWS.append([row[c] for c in MJ_COLS])
            n_rows += 1
    print(f"  mergejudge {name}: {n_rows:,} parent pairs within {MJ_MAX_D} um judged on REF's final graph", flush=True)


def mj_summary():
    import pandas as pd
    df = pd.DataFrame(MJ_ROWS, columns=MJ_COLS)
    df.to_csv("/kaggle/working/mergejudge_pairs.csv", index=False)
    print(f"\n=== MERGEJUDGE: {len(df):,} hypcache parent pairs within {MJ_MAX_D} um on REF's final graph, "
          f"{df.dataset.nunique()} movies; mergejudge_pairs.csv written ===", flush=True)
    both = df[df.mf_lab != -9]
    print(f"  label agreement with mergefeat on its scored rows: {(both.lab == both.mf_lab).mean():.3f} of {len(both):,}", flush=True)
    for lab, nm in ((1, "one cell (both ends match one GT node)"), (2, "one end annotated"), (0, "two cells"), (-1, "unannotated")):
        d = df[df.lab == lab]
        if len(d) == 0:
            continue
        print(f"\n  --- {nm}: {len(d):,} pairs ---", flush=True)
        print(f"  A alive in the final graph {d.A_alive.mean():.3f}, B alive {d.B_alive.mean():.3f}; chains (median frames) "
              f"A back {d.A_back.median():.0f} fwd {d.A_fwd.median():.0f}, B back {d.B_back.median():.0f} fwd {d.B_fwd.median():.0f}", flush=True)
        print(f"  B has an in-edge {(d.B_in_n > 0).mean():.3f}, an out-edge {(d.B_out_n > 0).mean():.3f}; "
              f"B a track of its own (back >= 2 and fwd >= 2) {((d.B_back >= 2) & (d.B_fwd >= 2)).mean():.3f}; "
              f"B loose (<= 1 edge) {((d.B_in_n + d.B_out_n) <= 1).mean():.3f}", flush=True)
        print(f"  contest: both have in-edges from different nodes {d.contest_in.mean():.3f}, both have out-edges {d.contest_out.mean():.3f}", flush=True)
        if lab >= 0:
            has_in = d.n_gt_in > 0
            has_out = d.n_gt_kids > 0
            a_right = ((~has_in) | (d.A_in_tp == 1)) & (d.A_out_tp == d.n_gt_kids)
            a_miss = (has_in & (d.A_in_tp == 0)) | (has_out & (d.A_out_tp < d.n_gt_kids))
            b_carries = (d.B_carries_in == 1) | (d.B_carries_out == 1)
            print(f"  A's GT edges: in-edge exists {has_in.mean():.3f} carried by A {(d.A_in_tp[has_in] == 1).mean():.3f}; "
                  f"out-edges exist {has_out.mean():.3f} all carried {(d.A_out_tp[has_out] == d.n_gt_kids[has_out]).mean():.3f}", flush=True)
            print(f"  A already right on every GT edge {a_right.mean():.3f}; A missing a GT edge {a_miss.mean():.3f}, "
                  f"of those B carries the GT neighbour (fixable by a merge) {b_carries[a_miss].mean() if a_miss.any() else float('nan'):.3f} "
                  f"= {int((a_miss & b_carries).sum()):,} pairs; B carries a GT neighbour while A is right {int((a_right & b_carries).sum()):,}", flush=True)
            print(f"  distances to G (um, median): A {d.d_AG.median():.2f}, B {d.d_BG.median():.2f}, midpoint {d.d_MG.median():.2f}; "
                  f"midpoint > 5 um {(d.d_MG > 5).mean():.3f}, > 7 um (the scorer's radius) {(d.d_MG > MJ_SCORE_UM).mean():.3f}; "
                  f"A itself > 5 um {(d.d_AG > 5).mean():.3f}", flush=True)
            if lab == 2:
                two = (d.B_back >= 2) & (d.B_fwd >= 2) & ~b_carries
                ghost = ((d.B_in_n + d.B_out_n) <= 1) & ~b_carries
                print(f"  typing of the one-end pairs: case 2 (B a track of its own, not carrying A's GT neighbour) {two.mean():.3f} = {int(two.sum()):,}; "
                      f"case 3 (B loose) {ghost.mean():.3f} = {int(ghost.sum()):,}; fixable swap {(a_miss & b_carries).mean():.3f} = {int((a_miss & b_carries).sum()):,}; "
                      f"other {(~two & ~ghost & ~(a_miss & b_carries)).mean():.3f}", flush=True)
                print(f"  where a merge can only lose: A right and (B a track or midpoint > 7 um) {(a_right & (two | (d.d_MG > MJ_SCORE_UM))).mean():.3f} "
                      f"= {int((a_right & (two | (d.d_MG > MJ_SCORE_UM))).sum()):,} pairs", flush=True)
# ---- end mergejudge block ----
'''
rep('\ndef _judge(name, k, outp, seeds, aux):\n', JUDGE + '\n\ndef _judge(name, k, outp, seeds, aux):\n')

# ---- the hook: REF's final graph, judged before the GT structures are dropped ----------
rep('    patch_final(nodes0, edges0, pool, outp)\n    BR_CTX["gt"] = None\n',
    '    _mj_nodes, _mj_edges = patch_final(nodes0, edges0, pool, outp)\n'
    '    merge_judge(name, T, seeds, seed_of_node, _mj_nodes, _mj_edges, _trees, _kids)\n'
    '    del _mj_nodes, _mj_edges\n'
    '    BR_CTX["gt"] = None\n')

# ---- the summary right after stage B --------------------------------------------------
rep('\nBR_ARMS_RUN = list(BR_ARMS)\n', '\nmj_summary()\n\nBR_ARMS_RUN = list(BR_ARMS)\n')

# ---- names ------------------------------------------------------------------------------
rep('print("DUPSUP OK", flush=True)', 'print("MERGEJUDGE OK", flush=True)')
rep('print(f"[degraded] DUPSUP INCOMPLETE or anchor miss:', 'print(f"[degraded] MERGEJUDGE INCOMPLETE or anchor miss:')
rep('f"dupsup_REF_{DET_SRC}.csv"', 'f"mergejudge_REF_{DET_SRC}.csv"')
rep('f"dupsup_arm_{a_}.csv"', 'f"mergejudge_arm_{a_}.csv"')

assert src.count("merge_judge(") == 2 and src.count("mj_summary()") == 2
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "mergejudge.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written mergejudge.py", len(src))
