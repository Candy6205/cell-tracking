"""dupsup.py -> mgpass.py: the TWO-PASS merge bench. Solve REF, decide the merges from
REF's own final graph, merge, re-solve, score.

WHY. mergejudge (2026-09-16) typed the one-end parent pairs on REF's graph: 47.6% are two
real cells tracked side by side (a merge leaves one node for two tracks: hmerge's loss
channel), 35.9% have a mate with <= 1 edge, 3.7% are fixable swaps. "Both ends carry full
tracks" (both have in-edges from different nodes and both have out-edges) marks 79% of the
two-real-cell pairs and 27% of the one-cell pairs. So the merge decision is made on the
graph, not by a classifier:

  MG0   every hypcache parent pair with children within MG_MAX_D um is merged (greedy by
        saddle, each child once) -- the control, hmerge's rule on this population
  MGB   merged only when both children are alive in REF's final graph and NOT both full
        tracks (the veto)

For each arm and movie: the merged detection set is hypcache's union cache with the
selected parents kept and their children removed (mergecache's index map), the two union
heads (ephypcache / ep2hypcache) remapped to the merged indices and written as temporary
caches, features_ext + solve_one + patch_final + the official scorer as for any arm. REF is
untouched (round-2 anchors hold); the report's cross / bootstrap machinery reads the arms.

Attach: dupsup's 16 sources + biohub-hypcache + biohub-ephypcache + biohub-ep2hypcache.
CPU: features ~1 h + REF ~70 min + 2 arms x 128 x ~50 s (~3.5 h) + the cross ~1 h.
"""
import os

src = open("dupsup.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-12a-dupsup"', 'SCRIPT_VERSION = "2026-09-16a-mgpass"')
rep('DUP_ARMS = [("SUP55", dict(R=5.5)), ("SUP65", dict(R=6.5)), ("SUP75", dict(R=7.5))]',
    'DUP_ARMS = [("MG0", dict(veto=False)), ("MGB", dict(veto=True))]   # mgpass: merge arms, see the docstring')
# the union heads carry the F4hyp_ tag; keep them out of the BASE head dict
rep('if not k.startswith("F4_") and not k.startswith("F4hp_")}',
    'if not k.startswith("F4_") and not k.startswith("F4hp_") and not k.startswith("F4hyp_")}', n=2)
rep('FDET_F4HP = _walk_files(_npz_prefix("fdet_F4hp_"))\nprint(f"fdet_F4hp caches: {len(FDET_F4HP)}", flush=True)\n',
    'FDET_F4HP = _walk_files(_npz_prefix("fdet_F4hp_"))\nprint(f"fdet_F4hp caches: {len(FDET_F4HP)}", flush=True)\n'
    'MG_HYP = _walk_files(_npz_prefix("fdet_F4hyp_"))          # hypcache: union c/s (parents appended), p{t}, w{t}\n'
    'MG_EP1 = _walk_files(_npz_prefix("edgeprob_F4hyp_"))      # head 1 over the union primaries\n'
    'MG_EP2 = _walk_files(_npz_prefix("edgeprob2_F4hyp_"))     # head 2 over the union primaries\n'
    'print(f"mgpass: hypcache {len(MG_HYP)}, union head1 {len(MG_EP1)}, union head2 {len(MG_EP2)}", flush=True)\n'
    '_mg6 = [n for n in MG_HYP if n.startswith("6bba_")]\n'
    'assert len(_mg6) >= 128 and all(n in MG_EP1 for n in _mg6), "attach biohub-hypcache + biohub-ephypcache (+ ep2hypcache)"\n')

BLOCK = r'''
# ---- mgpass block ----
MG_MAX_D = 8.5
MG_DIR = pathlib.Path("/tmp/mg")
MG_DIR.mkdir(parents=True, exist_ok=True)
MG_STAT = defaultdict(lambda: defaultdict(int))


def merged_index_map(s_all, p_par, q_sel):
    """mergecache.merged_index_map: keep (bool over the union array) and new_of (old primary
    index -> new primary index or -1) once the selected parents replace their children and
    the unselected parents go."""
    prim_pos = np.where(s_all >= UNET_THRESH)[0]
    n_prim = len(prim_pos)
    n_par = len(p_par)
    n_ch = n_prim - n_par
    keep_prim = np.ones(n_prim, bool)
    for q in range(n_par):
        keep_prim[n_ch + q] = False
    for q in q_sel:
        keep_prim[n_ch + q] = True
        keep_prim[p_par[q, 0]] = False
        keep_prim[p_par[q, 1]] = False
    keep = np.ones(len(s_all), bool)
    keep[prim_pos[~keep_prim]] = False
    new_of = np.full(n_prim, -1, np.int64)
    new_of[keep_prim] = np.arange(int(keep_prim.sum()))
    return keep, new_of


def remap_head(z, new_of_t, T):
    """mergecache.remap_head: (t, i, j) rows re-indexed to the merged primaries, rows touching a
    removed primary dropped, n_per_frame recomputed."""
    et = np.asarray(z["t"], np.int64)
    ei = np.asarray(z["i"], np.int64)
    ej = np.asarray(z["j"], np.int64)
    ni = np.array([new_of_t[t][i] for t, i in zip(et, ei)], np.int64) if len(et) else np.zeros(0, np.int64)
    nj = np.array([new_of_t[t + 1][j] for t, j in zip(et, ej)], np.int64) if len(et) else np.zeros(0, np.int64)
    ok = (ni >= 0) & (nj >= 0)
    out = {"n_per_frame": np.array([int((new_of_t[t] >= 0).sum()) for t in range(T)], np.int64),
           "t": et[ok], "i": ni[ok], "j": nj[ok]}
    for k in z.files:
        if k in ("n_per_frame", "t", "i", "j"):
            continue
        v = z[k]
        out[k] = v[ok] if (hasattr(v, "shape") and v.shape[:1] == et.shape) else v
    return out


def mg_select(T, seeds, seed_of_node, nodes_f, edges_f, zh, cfg):
    """Per frame, the parents to merge, decided on REF's final graph. veto=True: both children
    alive and not both full tracks (in-edges from different nodes AND both with out-edges).
    Greedy by saddle, each child in at most one merge."""
    node_of = {s_: nid for nid, s_ in seed_of_node.items() if s_ is not None and nid in nodes_f}
    in_of, out_of = defaultdict(set), defaultdict(set)
    for a_, b_ in edges_f:
        out_of[a_].add(b_)
        in_of[b_].add(a_)
    st = defaultdict(int)
    sel_t = []
    for t in range(T):
        par = np.asarray(zh[f"p{t}"], np.int64).reshape(-1, 2) if f"p{t}" in zh.files else np.zeros((0, 2), np.int64)
        sad = np.asarray(zh[f"w{t}"], float) if f"w{t}" in zh.files else np.zeros(len(par))
        C = np.asarray(seeds[t][0], float) * SCALE[None, :]
        cand = []
        for q in range(len(par)):
            a_i, b_i = int(par[q, 0]), int(par[q, 1])
            if a_i >= len(C) or b_i >= len(C):
                warn_once("mg_child_index", "a hypcache child index is outside the F4HP primaries")
                continue
            if float(np.linalg.norm(C[a_i] - C[b_i])) > MG_MAX_D:
                continue
            st["offered"] += 1
            na, nb_ = node_of.get((t, a_i)), node_of.get((t, b_i))
            if cfg.get("veto"):
                if na is None or nb_ is None:
                    st["skipped_dead"] += 1
                    continue
                ins_a, ins_b = in_of.get(na, set()), in_of.get(nb_, set())
                both_full = bool(ins_a) and bool(ins_b) and not (ins_a & ins_b) and bool(out_of.get(na)) and bool(out_of.get(nb_))
                if both_full:
                    st["vetoed_both_full"] += 1
                    continue
            cand.append((float(sad[q]) if q < len(sad) else 0.0, q, a_i, b_i))
        used = set()
        sel = []
        for _s, q, a_i, b_i in sorted(cand, key=lambda x: -x[0]):
            if a_i in used or b_i in used:
                st["greedy_conflict"] += 1
                continue
            used.add(a_i)
            used.add(b_i)
            sel.append(q)
        st["merged"] += len(sel)
        sel_t.append(sel)
    return sel_t, st


def mg_build(name, arm, T, dets_ref, sel_t, zh, use_ep2):
    """The merged detection list + the two union heads remapped and written as temporary
    caches. Asserts that the union's prefix IS the F4HP cache (hyp_apply appends, never
    reorders)."""
    dets_m, new_of_t = [], []
    for t in range(T):
        c = np.asarray(zh[f"c{t}"], float)
        s = np.asarray(zh[f"s{t}"], float)
        par = np.asarray(zh[f"p{t}"], np.int64).reshape(-1, 2) if f"p{t}" in zh.files else np.zeros((0, 2), np.int64)
        c_ref = np.asarray(dets_ref[t][0], float).reshape(-1, 3)
        n_ref = len(c_ref)
        assert len(c) == n_ref + len(par) and np.allclose(c[:n_ref], c_ref), (
            f"{name} t={t}: the hypcache union ({len(c)} = {n_ref} + {len(par)} parents?) is not the F4HP cache + parents")
        keep, new_of = merged_index_map(s, par, sel_t[t])
        dets_m.append((c[keep], s[keep]))
        new_of_t.append(new_of)
    h1 = remap_head(np.load(MG_EP1[name]), new_of_t, T)
    npf = np.array([int((np.asarray(d[1]) >= UNET_THRESH).sum()) for d in dets_m])
    assert np.array_equal(npf, h1["n_per_frame"]), (name, arm, "n_per_frame mismatch after the merge")
    p1 = MG_DIR / f"{arm}_{name}_ep1.npz"
    np.savez(p1, **h1)
    p2 = None
    if use_ep2 and name in MG_EP2:
        h2 = remap_head(np.load(MG_EP2[name]), new_of_t, T)
        p2 = MG_DIR / f"{arm}_{name}_ep2.npz"
        np.savez(p2, **h2)
    return dets_m, p1, p2
# ---- end mgpass block ----
'''
rep('\ndef _judge(name, k, outp, seeds, aux):\n', BLOCK + '\n\ndef _judge(name, k, outp, seeds, aux):\n')

# ---- REF's final graph is the decision input ----------------------------------------------
rep('    patch_final(nodes0, edges0, pool, outp)\n    BR_CTX["gt"] = None\n',
    '    _mg_nodes, _mg_edges = patch_final(nodes0, edges0, pool, outp)\n    BR_CTX["gt"] = None\n')

# ---- the arm loop body: merged set instead of suppression --------------------------------
i = src.index('    for arm, cfg in BR_ARMS:\n        keep = _dup_keep(seeds, cfg["R"])')
j_end = '        del ds_a, n_a, e_a, p_a, aux_a, pm_a, so_a, sc_a, b2_a, keep\n'
j = src.index(j_end, i) + len(j_end)
ARM = '''    _zh = np.load(MG_HYP[name])
    for arm, cfg in BR_ARMS:
        sel_t, st = mg_select(T, seeds, seed_of_node, _mg_nodes, _mg_edges, _zh, cfg)
        for k_, v_ in st.items():
            MG_STAT[arm][k_] += v_
        dets_m, ep1_m, ep2_m = mg_build(name, arm, T, dets, sel_t, _zh, EP2_OF["F4"].get(name) is not None)
        seeds_m, feats_m = features_ext(name, dets_m, T, ep1_m, ep2_m)
        outa = pathlib.Path(f"{TMP}/arm_{arm}") / f"{name}.geff"
        outa.parent.mkdir(parents=True, exist_ok=True)
        n_a, e_a, p_a, aux_a = solve_one(dets_m, T, seeds_m, feats_m, MODELS_FOLD[k], COLS,
                                         disapp=ILP_DISAPP, floor=PROD_FLOOR)
        pm_a, so_a = aux_a
        sc_a, b2_a = _aux_extra(seeds_m, pm_a)
        PATCH_CAPS.clear()
        patch_final(n_a, e_a, p_a, outa)
        for k_, v_ in PATCH_CAPS.items():
            CAPS_BY_ARM.setdefault(arm, defaultdict(int))[k_] += v_
        row_a, ds_a, fr_a = _judge(name, k, outa, seeds_m, (pm_a, so_a, sc_a, b2_a))
        ARM_ROWS[arm].append(row_a)
        ARM_PER[arm][name] = row_a
        ARM_FORKS[arm].extend(fr_a)
        ARM_MOVIE[arm][name] = _snapshot()
        DUP_BY_ARM[arm]["primaries"] += n_prim
        DUP_BY_ARM[arm]["suppressed"] += st["merged"]
        cells.append(f"{arm} merged {st['merged']}/{st['offered']} nodes {len(n_a)} "
                     f"{row_a['division_tp']}/{row_a['division_fp']}/{row_a['division_fn']} "
                     f"{_adj_of(row_a):.4f}")
        for _p in (ep1_m, ep2_m):
            if _p is not None:
                _p.unlink(missing_ok=True)
        del ds_a, n_a, e_a, p_a, aux_a, pm_a, so_a, sc_a, b2_a, dets_m, seeds_m, feats_m, sel_t
    del _zh, _mg_nodes, _mg_edges
'''
src = src[:i] + ARM + src[j:]

# ---- the merge counts per arm, right after stage B --------------------------------------------
SUMMARY = r'''
print("\n=== MGPASS: MERGES PER ARM, summed over the scored movies (offered = hypcache parent pairs with children "
      f"within {MG_MAX_D} um; MGB skips a pair with a child the short filter removed and vetoes both-full-track pairs) ===", flush=True)
_mk = ["offered", "skipped_dead", "vetoed_both_full", "greedy_conflict", "merged"]
print(f"  {'arm':<6}" + "".join(f"{k_:>18}" for k_ in _mk), flush=True)
for a_, _c in BR_ARMS:
    d_ = MG_STAT[a_]
    print(f"  {a_:<6}" + "".join(f"{d_[k_]:>18,}" for k_ in _mk)
          + f"   (merged {d_['merged'] / max(1, d_['offered']):.1%} of offered)", flush=True)
'''
rep('\nBR_ARMS_RUN = list(BR_ARMS)\n', SUMMARY + '\nBR_ARMS_RUN = list(BR_ARMS)\n')

# ---- names -----------------------------------------------------------------------------------
rep('print("DUPSUP OK", flush=True)', 'print("MGPASS OK", flush=True)')
rep('print(f"[degraded] DUPSUP INCOMPLETE or anchor miss:', 'print(f"[degraded] MGPASS INCOMPLETE or anchor miss:')
rep('f"dupsup_REF_{DET_SRC}.csv"', 'f"mgpass_REF_{DET_SRC}.csv"')
rep('f"dupsup_arm_{a_}.csv"', 'f"mgpass_arm_{a_}.csv"')

assert src.count("mg_select(") == 2 and src.count("mg_build(") == 2 and "_dup_keep(seeds, cfg" not in src
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "mgpass.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written mgpass.py", len(src))
