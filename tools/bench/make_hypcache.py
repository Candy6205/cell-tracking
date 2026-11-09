"""hmerge.py -> hypcache.py: the hpara detections PLUS parent hypotheses, cached
for the 140 bench movies.

WHY. hmerge replaced each saddle-keyed pair by one point and lost (hmbench:
edge_J -0.0082): a blind merge fires on 119k pairs to fix 1.8k swap edges.
Ultrack's way is to keep BOTH readings of such a pair -- the two detections
and the one merged point -- as hypotheses and let the temporal ILP choose,
with an exclusivity constraint (a child and its parent are never both
selected). This kernel writes the detection set that makes that possible:
every hpara detection as before (children; the sub-threshold repair pool
untouched) and, appended AFTER all of them, one PARENT per pair of primaries
within HM_D um whose heatmap saddle is >= 0.80 (hsaddle: 64% of swap pairs,
10% of two-cell pairs), placed at the segment mass centroid, refined like
every detection, score = max of the two. Not greedy: a primary may sit in
several parents; the ILP's exclusivity constraint (child + every parent
holding it <= 1) settles it.

OUTPUT: fdet_F4hyp_<name>.npz in the detcache layout (c{t}/s{t}: children
then parents), plus p{t} = (n_parents, 2) int32 -- each parent's two children
as indices AMONG PRIMARIES in seeds_of order (score >= 0.15, original order;
parents come after all children in that order because they are appended
after every detection, sub-threshold ones included) -- and w{t} = the
saddles. The edge heads are then re-run on the union set (ephypcache /
ep2hypcache) and the bench (hypilp) solves the ILP with the constraint.
Anchor: every frame must reproduce the shipped fdet_F4_ cache exactly, as
in hmerge. Attach: hmerge's sources. GPU T4, ~3.5 h (hmerge: 203 min).
"""
import os

src = open("hmerge.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


i = src.index('SCRIPT_VERSION = "')
j = src.index('"', i + len('SCRIPT_VERSION = "')) + 1
src = src[:i] + 'SCRIPT_VERSION = "2026-09-14a-hypcache"' + src[j:]

rep('HM_ARMS = [("hm90", 0.90), ("hm80", 0.80)]',
    'HM_ARMS = [("hyp", 0.80)]                   # hypcache: parents for every pair with saddle >= 0.80')

# ---- hm_apply (greedy replace) -> hyp_apply (append parents, keep children) --------
a = src.index("def hm_apply(h, vol, coords, scores, cand, thr):")
b = src.index("    return new_c[keep], new_s[keep], n\n", a) + len("    return new_c[keep], new_s[keep], n\n")
src = src[:a] + '''def hyp_apply(h, vol, coords, scores, cand, thr):
    """Every pair with saddle >= thr becomes a PARENT hypothesis appended after all
    detections: position = the segment mass centroid, refined like every detection;
    score = max of the two. The children stay. Not greedy: a primary may belong to
    several parents. Returns (coords, scores, parents (n, 2) int32 = the two
    children's indices AMONG PRIMARIES in seeds_of order, saddles float32, n)."""
    coords = np.asarray(coords, float)
    scores = np.asarray(scores, float)
    prim_rank = np.cumsum(scores >= UNET_THRESH) - 1
    add_c, add_s, par, sad_l = [], [], [], []
    for sad, i_, j_ in sorted(cand, key=lambda x: -x[0]):
        if sad < thr:
            break
        seg = segment_centroid(h, to_pooled(coords[i_]), to_pooled(coords[j_]))
        add_c.append(_refine(vol, from_pooled(seg)))
        add_s.append(max(scores[i_], scores[j_]))
        par.append((int(prim_rank[i_]), int(prim_rank[j_])))
        sad_l.append(float(sad))
    if add_c:
        coords = np.concatenate([coords, np.asarray(add_c, float).reshape(-1, 3)])
        scores = np.concatenate([scores, np.asarray(add_s, float)])
    return (coords, scores, np.asarray(par, np.int32).reshape(-1, 2),
            np.asarray(sad_l, np.float32), len(par))
''' + src[b:]

rep("            c2, s2, nmg = hm_apply(h, vol, ch, s, cand, thr)\n"
    "            outs[a_][f\"c{t}\"] = np.asarray(c2, np.float32)\n"
    "            outs[a_][f\"s{t}\"] = np.asarray(s2, np.float32)\n",
    "            c2, s2, p2, w2, nmg = hyp_apply(h, vol, ch, s, cand, thr)\n"
    "            outs[a_][f\"c{t}\"] = np.asarray(c2, np.float32)\n"
    "            outs[a_][f\"s{t}\"] = np.asarray(s2, np.float32)\n"
    "            outs[a_][f\"p{t}\"] = p2\n"
    "            outs[a_][f\"w{t}\"] = w2\n")
rep('STAT[a_]["merged"] += nmg', 'STAT[a_]["parents"] += nmg')
rep('''          + ", ".join(f"{a_} merged {STAT[a_]['merged']:,}/{STAT[a_]['primaries']:,} primaries "
                      f"({STAT[a_]['merged'] / max(1, STAT[a_]['primaries']):.3%}, cumulative)"''',
    '''          + ", ".join(f"{a_} parents {STAT[a_]['parents']:,}/{STAT[a_]['primaries']:,} primaries "
                      f"({STAT[a_]['parents'] / max(1, STAT[a_]['primaries']):.3%}, cumulative)"''')
rep('''          f"merged {d_['merged']:,} ({d_['merged'] / max(1, d_['primaries']):.3%} of primaries, "
          f"{d_['merged'] / max(1, d_['pairs']):.3%} of pairs)", flush=True)''',
    '''          f"parents {d_['parents']:,} ({d_['parents'] / max(1, d_['primaries']):.3%} of primaries, "
          f"{d_['parents'] / max(1, d_['pairs']):.3%} of pairs)", flush=True)''')
rep('print(f"\\n=== HMERGE: {len(NAMES)} movies, ALL frames; pairs of primaries within {HM_D} um "\n'
    '      f"merged at the segment mass centroid when the heatmap saddle >= "',
    'print(f"\\n=== HYPCACHE: {len(NAMES)} movies, ALL frames; every pair of primaries within {HM_D} um "\n'
    '      f"gets a PARENT hypothesis at the segment mass centroid when the heatmap saddle >= "')
rep('print(f"\\n=== HMERGE TOTALS over {done} movies ===", flush=True)',
    'print(f"\\n=== HYPCACHE TOTALS over {done} movies ===", flush=True)')
rep('print(f"\\nHMERGE DONE in', 'print(f"\\nHYPCACHE DONE in')
rep('print("HMERGE OK" if not ANCHOR_FAILS and done == len(NAMES) else "HMERGE INCOMPLETE OR ANCHOR FAILED", flush=True)',
    'print("HYPCACHE OK" if not ANCHOR_FAILS and done == len(NAMES) else "HYPCACHE INCOMPLETE OR ANCHOR FAILED", flush=True)')

assert "hm_apply(" not in src and src.count("hyp_apply(") == 2
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hypcache.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written hypcache.py", len(src))
