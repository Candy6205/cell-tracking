"""dupsup.py -> hrbench.py: the bench of hrefine's RE-PLACED detection set (every F4HP primary moved by the
placement head, none added or removed) against the shipped F4HP placement.

WHY (2026-09-23). hrefine trained a CNN on the raw crop around each detection to predict the offset to the
annotated centroid (out of fold by movie) and wrote fdet_F4hr_<name>.npz: the F4HP caches with every primary
re-placed, sub-threshold peaks untouched, order kept. ephrcache / ep2hrcache re-ran both edge heads on those
points. This bench is make_lnbench's machinery with the rescue admission removed: ONE arm, HR, the F4hr set
with its own heads and every primary kept, solved with the same fold rankers (fitted on F4HP rows -- a
positive result is trustworthy, a negative one is not decisive), the shipped patches, the official scorer, the
fork table, the joint-gate cross and the paired bootstrap against REF + joint.

Usage: py tools/bench/make_hrbench.py
Attach: dupsup's 16 sources + biohub-hrefine + biohub-ephrcache + biohub-ep2hrcache. CPU.
"""
import os
import sys

# arm spec "0.20" = admit new peaks at score >= 0.20; "0.15P" = score >= 0.15 AND a new peak within 4 um in an adjacent frame
ARM_SPECS = ["0.0"]   # hrbench: one arm, every primary kept
TAUS = [(float(x.rstrip("P")), x.endswith("P")) for x in ARM_SPECS]
src = open("dupsup.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-12a-dupsup"', 'SCRIPT_VERSION = "2026-09-23a-hrbench"')
arms = '("HR", dict(tau=0.0, persist=False))'   # hrbench: the re-placed set, nothing withheld
rep('DUP_ARMS = [("SUP55", dict(R=5.5)), ("SUP65", dict(R=6.5)), ("SUP75", dict(R=7.5))]',
    f'DUP_ARMS = [{arms}]   # hrbench: the re-placed set (hrefine), every primary kept')
rep('if not k.startswith("F4_") and not k.startswith("F4hp_")}',
    'if not k.startswith("F4_") and not k.startswith("F4hp_") and not k.startswith("F4hr_")}', n=2)
rep('FDET_F4HP = _walk_files(_npz_prefix("fdet_F4hp_"))\nprint(f"fdet_F4hp caches: {len(FDET_F4HP)}", flush=True)\n',
    'FDET_F4HP = _walk_files(_npz_prefix("fdet_F4hp_"))\nprint(f"fdet_F4hp caches: {len(FDET_F4HP)}", flush=True)\n'
    'LN_FDET = _walk_files(_npz_prefix("fdet_F4hr_"))          # hrefine: F4HP re-placed, same primaries, same order\n'
    'LN_EP1 = _walk_files(_npz_prefix("edgeprob_F4hr_"))\n'
    'LN_EP2 = _walk_files(_npz_prefix("edgeprob2_F4hr_"))\n'
    'print(f"hrbench: re-placed caches {len(LN_FDET)}, head1 {len(LN_EP1)}, head2 {len(LN_EP2)}", flush=True)\n'
    '_ln6 = [n for n in LN_FDET if n.startswith("6bba_")]\n'
    'assert len(_ln6) >= 128 and all(n in LN_EP1 for n in _ln6), "attach biohub-hrefine + biohub-ephrcache (+ ep2hrcache)"\n')

BLOCK = r'''
# ---- lnbench block ----
LN_STAT = defaultdict(lambda: defaultdict(int))
HR_MOVE = []          # per-detection shift, um, over every frame the bench solved


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
        s_ref = np.asarray(dets[t_][1], np.float32).reshape(-1)
        assert len(dets_ln[t_][0]) == len(c_ref) and np.array_equal(
            np.asarray(dets_ln[t_][1], np.float32).reshape(-1), s_ref), (
            f"{name} t={t_}: the re-placed cache does not carry the F4HP primaries (same count, same scores)")
        HR_MOVE.append(np.linalg.norm((np.asarray(dets_ln[t_][0], np.float32).reshape(-1, 3) - c_ref) * SCALE, axis=1))
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
_mv = np.concatenate(HR_MOVE) if HR_MOVE else np.zeros(1)
print(f"\n=== HRBENCH: the HR arm re-places every primary (hrefine); shift over {len(_mv):,} detections: "
      f"p50 {np.median(_mv):.3f} p90 {np.percentile(_mv, 90):.3f} max {_mv.max():.3f} um; nothing withheld ===", flush=True)
'''
rep('\nBR_ARMS_RUN = list(BR_ARMS)\n', SUMMARY + '\nBR_ARMS_RUN = list(BR_ARMS)\n')
rep('print("DUPSUP OK", flush=True)', 'print("HRBENCH OK", flush=True)')
rep('print(f"[degraded] DUPSUP INCOMPLETE or anchor miss:', 'print(f"[degraded] HRBENCH INCOMPLETE or anchor miss:')
rep('f"dupsup_REF_{DET_SRC}.csv"', 'f"hrbench_REF_{DET_SRC}.csv"')
rep('f"dupsup_arm_{a_}.csv"', 'f"hrbench_arm_{a_}.csv"')

assert src.count("ln_keep(") == 2 and src.count("ln_load(") == 2 and "_dup_keep(seeds, cfg" not in src
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hrbench.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written hrbench.py", len(src), "arms", TAUS)
