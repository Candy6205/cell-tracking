"""vttrace.py -> vtattach.py: the vote candidates may only ATTACH to existing tracks.

WHY (2026-09-19). vttrace: offered to the shipped linker, the vote candidates (mapvote,
>= 2 single maps agree, no shipped primary within 7 um) yield 408 true edges (edge_J
+0.0035 [+0.0022, +0.0050], the largest identity gain of any detection-side device) and
lose adj_J -0.0025 through the node term: 144k of the 224k admitted candidates survive
the short-component filter because texture candidates link to texture candidates frame
after frame (lnpersist's standing structure). Every true edge in the funnel joins a
candidate to a SHIPPED primary. So: withhold every candidate-to-candidate pair from the
ILP. A candidate then either attaches to an existing track (and is charged as a node) or
stays isolated and falls to the short-component filter. A second arm also drops a
candidate that attached by a single edge (a dangling extension at a track end) before the
post-link patches, keeping only candidates that fill a hole between two shipped nodes.

ARMS (each a fresh solve on the vote set, lnbench's harness, lntrace's per-arm funnel):
  VT17X   >= 2 votes admitted, candidate-candidate pairs withheld
  VT17XB  the same, plus dangling candidates (degree <= 1 after the solve) dropped
  VT18X   >= 3 votes admitted, candidate-candidate pairs withheld
v3 (2026-09-19c): VT17XB and VT18XB only; v2 measured VT17X / VT18X and its XB arm was invalid (see step 2b).
Read: the per-arm count of new nodes in the FINAL graph against vttrace's 144k / 47k,
G_tp against 408 / 137, and the paired bootstrap vs REF + joint on score / adj_J / edge_J.

Attach: vttrace's sources (lnbench's + biohub-mapvote / epvtcache / ep2vtcache). CPU ~5 h.
"""
import os
import sys

src = open("vttrace.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-19a-vttrace"', 'SCRIPT_VERSION = "2026-09-19c-vtattach"')
rep('DUP_ARMS = [("VT17", dict(tau=0.17, persist=False)), ("VT18", dict(tau=0.18, persist=False))]   # vtbench: the vote set, new peaks admitted at score >= tau',
    'DUP_ARMS = [("VT17XB", dict(tau=0.17, persist=False, nonew=True, dangle=True)), '
    '("VT18XB", dict(tau=0.18, persist=False, nonew=True, dangle=True))]   # vtattach v3: the hole-filling candidates only (v2 measured VT17X / VT18X)')

# 1. the solver: withhold candidate-to-candidate pairs when the arm says so
rep('''                if keep is not None and not (keep[t][i_] and keep[t + 1][j_]):
                    DUP_STAT["edges_withheld"] += 1
                    continue
''', '''                if keep is not None and not (keep[t][i_] and keep[t + 1][j_]):
                    DUP_STAT["edges_withheld"] += 1
                    continue
                if LN_NO_NEWNEW and LN_NEWMASK is not None and LN_NEWMASK[t][i_] and LN_NEWMASK[t + 1][j_]:
                    DUP_STAT["newnew_withheld"] += 1
                    continue
''')

# 2. the block: the mask globals and the dangling-candidate drop
rep('LN_STAT = defaultdict(lambda: defaultdict(int))\n', '''LN_STAT = defaultdict(lambda: defaultdict(int))
LN_NEWMASK = None            # per frame: True for the vote candidates (the last nnew[t] primaries); set per movie
LN_NO_NEWNEW = False         # set per arm: withhold candidate-to-candidate pairs from the ILP


def ln_newmask(seeds, nnew):
    """per frame, True for the appended candidates (the last nnew[t] primaries)."""
    out = []
    for s_, nn_ in zip(seeds, nnew):
        n_ = len(np.asarray(s_[1]).reshape(-1))
        m_ = np.zeros(n_, bool)
        if nn_:
            m_[n_ - nn_:] = True
        out.append(m_)
    return out


def ln_is_new(sd, seeds, nnew):
    return sd is not None and int(sd[1]) >= len(np.asarray(seeds[int(sd[0])][1]).reshape(-1)) - nnew[int(sd[0])]


def ln_drop_dangling(nodes, edges, son, seeds, nnew):
    """Drop every candidate node with at most one edge after the solve (a dangling extension), with its edge.
    Shipped nodes are never dropped. Returns (nodes, edges, n_dropped)."""
    deg = defaultdict(int)
    for a_, b_ in edges:
        deg[a_] += 1
        deg[b_] += 1
    drop = {g_ for g_ in nodes if ln_is_new(son.get(g_), seeds, nnew) and deg[g_] <= 1}
    nodes2 = {g_: v_ for g_, v_ in nodes.items() if g_ not in drop}
    edges2 = [(a_, b_) for a_, b_ in edges if a_ not in drop and b_ not in drop]
    return nodes2, edges2, len(drop)
''')

# 3a. the REF solve of every movie runs with the constraint OFF: the globals are set per arm below and would otherwise
#     carry the previous movie's mask (and its frame sizes) into the next movie's REF solve (v1 died there: IndexError)
rep('''    k = FOLD_OF[name]
    nodes0, edges0, pool, aux0 = solve_one(dets, T, seeds, feats, MODELS_FOLD[k], COLS,
''', '''    k = FOLD_OF[name]
    LN_NEWMASK, LN_NO_NEWNEW = None, False
    nodes0, edges0, pool, aux0 = solve_one(dets, T, seeds, feats, MODELS_FOLD[k], COLS,
''')

# 2b. patch_final allocates the gap-close node ids from len(nodes)+1. The ids are 1..N before any drop (max == len, so REF
#     is unchanged); after ln_drop_dangling they have holes and len+1 COLLIDES with a surviving id: v2's XB arm overwrote
#     shipped nodes with interpolants (div 54/3,353/71, 24.7k shipped nodes gone). Allocate past the largest id instead.
rep('''    nid2 = len(nodes) + 1
''', '''    nid2 = (max(nodes) if nodes else 0) + 1   # past the LARGEST id: after a node drop len+1 would collide with a survivor
''')

# 3. the arm loop: set the mask and the flag before the solve, drop dangling candidates after it, count survivors
rep('''        DUP_STAT.clear()
        outa = pathlib.Path(f"{TMP}/arm_{arm}") / f"{name}.geff"
''', '''        DUP_STAT.clear()
        LN_NEWMASK = ln_newmask(seeds_ln, nnew_ln)
        LN_NO_NEWNEW = bool(cfg.get("nonew", False))
        outa = pathlib.Path(f"{TMP}/arm_{arm}") / f"{name}.geff"
''')
rep('''        pm_a, so_a = aux_a
        sc_a, b2_a = _aux_extra(seeds_ln, pm_a)
''', '''        pm_a, so_a = aux_a
        LN_STAT[arm]["newnew_withheld"] += int(DUP_STAT.get("newnew_withheld", 0))
        LN_STAT[arm]["solved_new_nodes"] += sum(1 for g_ in n_a if ln_is_new(so_a.get(g_), seeds_ln, nnew_ln))
        if cfg.get("dangle", False):
            n_a, e_a, n_dg = ln_drop_dangling(n_a, e_a, so_a, seeds_ln, nnew_ln)
            LN_STAT[arm]["dangling_dropped"] += n_dg
        sc_a, b2_a = _aux_extra(seeds_ln, pm_a)
''')
rep('''        LT_ARM = arm
        LN_STAT[arm]["trace_rows"] += ln_trace(name, T, seeds_ln, nnew_ln, keep, pm_a, so_a)
''', '''        LT_ARM = arm
        LN_STAT[arm]["trace_rows"] += ln_trace(name, T, seeds_ln, nnew_ln, keep, pm_a, so_a)
        LN_STAT[arm]["final_new_nodes"] += sum(1 for g_ in PRE_STATE["nodes"] if ln_is_new(so_a.get(g_), seeds_ln, nnew_ln))
        LN_STAT[arm]["final_nodes"] += len(PRE_STATE["nodes"])
''')

# 4. the summary line per arm
rep('''          f"new {d_['new']:,}, kept {d_['kept']:,} ({d_['kept'] / max(1, d_['new']):.1%})", flush=True)
''', '''          f"new {d_['new']:,}, kept {d_['kept']:,} ({d_['kept'] / max(1, d_['new']):.1%}); candidate-candidate pairs withheld "
          f"{d_['newnew_withheld']:,}; candidates in the solved graph {d_['solved_new_nodes']:,}, dangling dropped {d_['dangling_dropped']:,}, "
          f"in the FINAL graph {d_['final_new_nodes']:,} of {d_['final_nodes']:,} nodes (vttrace: ~144k of 224k at 2 votes, ~47k of 82k at 3)", flush=True)
''')

# 5. names
for old, new in (('f"vttrace_REF_{DET_SRC}.csv"', 'f"vtattach_REF_{DET_SRC}.csv"'), ('f"vttrace_arm_{a_}.csv"', 'f"vtattach_arm_{a_}.csv"'),
                 ('"/kaggle/working/vttrace_edges.csv"', '"/kaggle/working/vtattach_edges.csv"'), ('print("VTTRACE OK", flush=True)', 'print("VTATTACH OK", flush=True)'),
                 ('print(f"[degraded] VTTRACE INCOMPLETE or anchor miss:', 'print(f"[degraded] VTATTACH INCOMPLETE or anchor miss:'),
                 ('=== VTTRACE: every linkable GT edge', '=== VTATTACH: every linkable GT edge')):
    rep(old, new)

assert src.count("LN_NO_NEWNEW") == 4 and src.count("ln_drop_dangling(") == 2 and src.count("ln_newmask(") == 2
assert src.index("def ln_drop_dangling(") < src.index("def solve_one(") or src.index("def ln_drop_dangling(") < src.index("for arm, cfg in BR_ARMS:")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vtattach.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written vtattach.py", len(src.splitlines()), "lines")
