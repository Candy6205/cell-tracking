"""dupsup.py -> lnbench.py: the bench of the RESCUE detection set (F4HP + the locally
normalised pass's new peaks), with the new peaks admitted at a score threshold.

WHY (2026-09-17). lnrescue wrote fdet_F4ln_<name>.npz: every shipped F4HP detection
untouched plus the BGSUB15 peaks that lie >= 7 um from every shipped detection, appended
last (nnew{t} of them per frame, score = the normalised pass's fused value). locnorm found
that such peaks recover ~19% of the annotated cells the shipped pass misses while about
half of them are noise, with the recovered cells' scores around 0.30. The question the
bench answers: at which admission threshold do the recovered edges outweigh the noise
nodes' cost (node term, false links, swap-like twins)?

ARMS. REF = the shipped F4HP solve (round-2 anchors hold). For each tau in TAUS an arm
LN<tt>: the F4ln detection set with features from the F4ln heads (eplncache / ep2lncache,
head 2 only where REF has one), solved with the SAME fold rankers, with solve_one's keep
mask withholding every new peak whose score is below tau or, in a P arm, without another new
peak within 4 um in an adjacent frame (its edges never enter the ILP and its node is dropped), then the shipped patches, the official scorer, the fork table, the
joint-gate cross and the paired bootstrap against REF + joint. Features on the F4ln set are
built once per movie and shared by the arms.

Usage: py tools/bench/make_lnbench.py 0.15P,0.20,0.25   (trailing P = temporal persistence required)
Attach: dupsup's 16 sources + biohub-lnrescue + biohub-eplncache + biohub-ep2lncache.
CPU: features ~1 h + REF ~70 min + 128 x (features_ext ~15 s + arms x ~35 s) + the cross ~1 h.
"""
import os
import sys

# arm spec "0.20" = admit new peaks at score >= 0.20; "0.15P" = score >= 0.15 AND a new peak within 4 um in an adjacent frame
ARM_SPECS = (sys.argv[1] if len(sys.argv) > 1 else "0.15P,0.20,0.25").split(",")
TAUS = [(float(x.rstrip("P")), x.endswith("P")) for x in ARM_SPECS]
src = open("dupsup.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-12a-dupsup"', 'SCRIPT_VERSION = "2026-09-17a-lnbench"')
arms = ", ".join(f'("LN{int(round(t * 100)):02d}{"P" if p_ else ""}", dict(tau={t}, persist={p_}))' for t, p_ in TAUS)
rep('DUP_ARMS = [("SUP55", dict(R=5.5)), ("SUP65", dict(R=6.5)), ("SUP75", dict(R=7.5))]',
    f'DUP_ARMS = [{arms}]   # lnbench: the rescue set, new peaks admitted at score >= tau')
rep('if not k.startswith("F4_") and not k.startswith("F4hp_")}',
    'if not k.startswith("F4_") and not k.startswith("F4hp_") and not k.startswith("F4ln_")}', n=2)
rep('FDET_F4HP = _walk_files(_npz_prefix("fdet_F4hp_"))\nprint(f"fdet_F4hp caches: {len(FDET_F4HP)}", flush=True)\n',
    'FDET_F4HP = _walk_files(_npz_prefix("fdet_F4hp_"))\nprint(f"fdet_F4hp caches: {len(FDET_F4HP)}", flush=True)\n'
    'LN_FDET = _walk_files(_npz_prefix("fdet_F4ln_"))          # lnrescue: F4HP + new peaks appended, nnew{t}\n'
    'LN_EP1 = _walk_files(_npz_prefix("edgeprob_F4ln_"))\n'
    'LN_EP2 = _walk_files(_npz_prefix("edgeprob2_F4ln_"))\n'
    'print(f"lnbench: rescue caches {len(LN_FDET)}, head1 {len(LN_EP1)}, head2 {len(LN_EP2)}", flush=True)\n'
    '_ln6 = [n for n in LN_FDET if n.startswith("6bba_")]\n'
    'assert len(_ln6) >= 128 and all(n in LN_EP1 for n in _ln6), "attach biohub-lnrescue + biohub-eplncache (+ ep2lncache)"\n')

BLOCK = r'''
# ---- lnbench block ----
LN_STAT = defaultdict(lambda: defaultdict(int))


def ln_load(name):
    """The rescue detection list and, per frame, how many new peaks sit at the end of the primaries."""
    z = np.load(LN_FDET[name])
    T = max(int(k[1:]) for k in z.files if k.startswith("c")) + 1
    dets = [(np.asarray(z[f"c{t}"], np.float32).reshape(-1, 3), np.asarray(z[f"s{t}"], np.float32).reshape(-1)) for t in range(T)]
    nnew = [int(z[f"nnew{t}"]) if f"nnew{t}" in z.files else 0 for t in range(T)]
    return dets, nnew


LN_PERSIST_UM = 4.0


def ln_keep(seeds, nnew, tau, persist=False):
    """keep mask over the primaries: shipped primaries always; a new peak iff score >= tau and, with
    persist, another new peak lies within LN_PERSIST_UM in the previous or the next frame."""
    T = len(seeds)
    newpts = []
    for t in range(T):
        c_ = np.asarray(seeds[t][0], float).reshape(-1, 3)
        newpts.append(c_[-nnew[t]:] * SCALE if nnew[t] else np.zeros((0, 3)))
    trees = [cKDTree(pn) if len(pn) else None for pn in newpts]
    keep = []
    n_new = n_kept = 0
    for t in range(T):
        s_ = np.asarray(seeds[t][1], float).reshape(-1)
        k = np.ones(len(s_), bool)
        if nnew[t]:
            assert nnew[t] <= len(s_), (t, nnew[t], len(s_))
            adm = s_[-nnew[t]:] >= tau
            if persist:
                near = np.zeros(nnew[t], bool)
                for tt in (t - 1, t + 1):
                    if 0 <= tt < T and trees[tt] is not None:
                        d, _ = trees[tt].query(newpts[t])
                        near |= d <= LN_PERSIST_UM
                adm &= near
            k[-nnew[t]:] = adm
            n_new += nnew[t]
            n_kept += int(adm.sum())
        keep.append(k)
    return keep, n_new, n_kept
# ---- end lnbench block ----
'''
rep('\ndef _judge(name, k, outp, seeds, aux):\n', BLOCK + '\n\ndef _judge(name, k, outp, seeds, aux):\n')

i = src.index('    for arm, cfg in BR_ARMS:\n        keep = _dup_keep(seeds, cfg["R"])')
j_end = '        del ds_a, n_a, e_a, p_a, aux_a, pm_a, so_a, sc_a, b2_a, keep\n'
j = src.index(j_end, i) + len(j_end)
ARM = '''    dets_ln, nnew_ln = ln_load(name)
    assert len(dets_ln) == T, (name, len(dets_ln), T)
    for t_ in range(T):
        c_ref = np.asarray(dets[t_][0], np.float32).reshape(-1, 3)
        assert len(dets_ln[t_][0]) >= len(c_ref) and np.allclose(dets_ln[t_][0][:len(c_ref)], c_ref), (
            f"{name} t={t_}: the rescue cache is not the F4HP cache + appended peaks")
    seeds_ln, feats_ln = features_ext(name, dets_ln, T, LN_EP1[name],
                                      LN_EP2.get(name) if EP2_OF["F4"].get(name) is not None else None)
    for arm, cfg in BR_ARMS:
        keep, n_new, n_kept = ln_keep(seeds_ln, nnew_ln, cfg["tau"], cfg.get("persist", False))
        LN_STAT[arm]["new"] += n_new
        LN_STAT[arm]["kept"] += n_kept
        DUP_STAT.clear()
        outa = pathlib.Path(f"{TMP}/arm_{arm}") / f"{name}.geff"
        outa.parent.mkdir(parents=True, exist_ok=True)
        n_a, e_a, p_a, aux_a = solve_one(dets_ln, T, seeds_ln, feats_ln, MODELS_FOLD[k], COLS,
                                         disapp=ILP_DISAPP, floor=PROD_FLOOR, keep=keep)
        pm_a, so_a = aux_a
        sc_a, b2_a = _aux_extra(seeds_ln, pm_a)
        PATCH_CAPS.clear()
        patch_final(n_a, e_a, p_a, outa)
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
        cells.append(f"{arm} new {n_kept}/{n_new} nodes {len(n_a)} "
                     f"{row_a['division_tp']}/{row_a['division_fp']}/{row_a['division_fn']} "
                     f"{_adj_of(row_a):.4f}")
        del ds_a, n_a, e_a, p_a, aux_a, pm_a, so_a, sc_a, b2_a, keep
    del dets_ln, nnew_ln, seeds_ln, feats_ln
'''
src = src[:i] + ARM + src[j:]

SUMMARY = r'''
print("\n=== LNBENCH: NEW PEAKS PER ARM, summed over the scored movies (new = the rescue peaks appended by lnrescue; "
      "kept = admitted at the arm's score threshold, the rest withheld from the ILP and dropped) ===", flush=True)
for a_, _c in BR_ARMS:
    d_ = LN_STAT[a_]
    print(f"  {a_:<6} tau {_c['tau']:.2f}{' + persistence (a new peak within 4 um at t-1 or t+1)' if _c.get('persist') else ''}: "
          f"new {d_['new']:,}, kept {d_['kept']:,} ({d_['kept'] / max(1, d_['new']):.1%})", flush=True)
'''
rep('\nBR_ARMS_RUN = list(BR_ARMS)\n', SUMMARY + '\nBR_ARMS_RUN = list(BR_ARMS)\n')
rep('print("DUPSUP OK", flush=True)', 'print("LNBENCH OK", flush=True)')
rep('print(f"[degraded] DUPSUP INCOMPLETE or anchor miss:', 'print(f"[degraded] LNBENCH INCOMPLETE or anchor miss:')
rep('f"dupsup_REF_{DET_SRC}.csv"', 'f"lnbench_REF_{DET_SRC}.csv"')
rep('f"dupsup_arm_{a_}.csv"', 'f"lnbench_arm_{a_}.csv"')

assert src.count("ln_keep(") == 2 and src.count("ln_load(") == 2 and "_dup_keep(seeds, cfg" not in src
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lnbench.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written lnbench.py", len(src), "arms", TAUS)
