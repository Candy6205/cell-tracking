"""mapvote.py -> zhvote.py: the four-map vote on the 41 Zebrahub crops, judged by DENSE GT.

WHY (2026-09-20). On the 128 6bba train movies the vote candidates (maxima of >= 2 of the
four single maps, no shipped primary within 7 um) are annotated at 0.0070 against the
shipped primaries' 0.0548 (0.13x) and recover 31% of the missed annotated cells; every
device built on them sits within +/-0.001. But the four detectors were trained on all 199
train videos (forum 742064), so the 6bba numbers are IN-SAMPLE: the detector has memorised
those movies and misses little there. Zebrahub was never seen by any of the four weights
and its crops carry every cell (dense GT), so the same measurement there reads the
candidates' TRUE purity out of sample and the shipped detector's true miss rate, both of
which the sparse in-sample bench cannot show. Dense GT includes dim cells the competition's
sparse annotation would not, so the absolute miss rate is pessimistic; the RATIOS
(candidate share / shipped share, recovered / missed) are what is compared to 6bba.

READ per crop and pooled: shipped primaries, their dense precision at 7 um (share with a
GT node within 7 um), GT nodes, missed GT (no primary within 7 um), candidates by votes
with their GT share and the ratio to the shipped share, missed GT recovered within 7 um by
k >= 2 / 3 / 4, f4 value and distance to GT at the annotated candidates.
6bba (mapvote): shipped share 0.0548; k = 2 / 3 / 4 share 0.0084 / 0.0090 / 0.0064 =
0.15x / 0.16x / 0.12x; k >= 2 admits +10% nodes at 0.13x and recovers 637 of 2,067 (31%).

OUTPUT: zhvote_<crop>.npz (the shipped hpara peaks + the appended candidates in the
fdet_F4vt_ layout, the crop's GT arrays copied through) and zhvote_cands.csv (one row per
candidate: crop, t, k, v, f, d_gt, gt_missed). Attach: mapvote's model datasets +
faustine649/biohub-zebrahub-extract (crops). T4, ~70 min (4,100 frames).
"""
import os

src = open("mapvote.py", encoding="utf-8").read()
head = src[:src.index("FDET = {}\n")]
assert "def select_new(" in head and "def branch_maps4(" in head and "SCORE_OF_K = " in head


def rep(old, new, n=1):
    global head
    assert head.count(old) == n, (head.count(old), old[:90])
    head = head.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-19a-mapvote"', 'SCRIPT_VERSION = "2026-09-20a-zhvote"')

loop = r'''
# ---------------------------------------------------------------- Zebrahub
CROPS = None
for _c in ["/kaggle/input/biohub-zebrahub-extract/crops",
           "/kaggle/input/notebooks/faustine649/biohub-zebrahub-extract/crops"]:
    if pathlib.Path(_c).is_dir():
        CROPS = pathlib.Path(_c)
        break
if CROPS is None:
    _hits = sorted(p for p in pathlib.Path("/kaggle/input").glob("**/crops") if p.is_dir() and any(p.glob("*.npz")))
    CROPS = _hits[0] if _hits else None
assert CROPS is not None, "zebrahub crops not mounted (attach faustine649/biohub-zebrahub-extract)"
FILES = sorted(CROPS.glob("*.npz"))
assert FILES, "crops directory is empty"
print(f"crops: {len(FILES)} under {CROPS}", flush=True)
OUT = pathlib.Path("/kaggle/working")
KS = (2, 3, 4)
GT_KEYS = ("nodes", "track_id", "parent_track_id", "origin_um", "voxel_um", "t0")


def zh_gt(z, T):
    """(g_t, g_pos_um) of the crop's GT nodes inside [0, T): nodes are (t, z, y, x) in crop voxels."""
    nodes = np.asarray(z["nodes"], float).reshape(-1, 4)
    keep = (nodes[:, 0] >= 0) & (nodes[:, 0] < T)
    nodes = nodes[keep]
    return nodes[:, 0].astype(int), nodes[:, 1:] * SCALE


def frame_stats(prim_um, cn_vox, kn, gt_um, ks=KS, r_um=RESCUE_UM):
    """dense-GT bookkeeping for one frame: shipped primaries (um), new candidates (voxels) with their votes, GT (um).
    Returns (stat dict, d_gt per candidate, gt_missed per candidate)."""
    st = defaultdict(float)
    prim_um = np.asarray(prim_um, float).reshape(-1, 3)
    cn_vox = np.asarray(cn_vox, float).reshape(-1, 3)
    kn = np.asarray(kn, int).reshape(-1)
    gt_um = np.asarray(gt_um, float).reshape(-1, 3)
    st["shipped"] += len(prim_um)
    st["gt"] += len(gt_um)
    dg = np.full(len(cn_vox), np.inf)
    gmiss = np.zeros(len(cn_vox), bool)
    if len(gt_um):
        gtree = cKDTree(gt_um)
        if len(prim_um):
            st["shipped_ann"] += int((gtree.query(prim_um)[0] <= r_um).sum())
            d_gp = cKDTree(prim_um).query(gt_um)[0]
        else:
            d_gp = np.full(len(gt_um), np.inf)
        miss = d_gp > r_um
        st["gt_miss"] += int(miss.sum())
        if len(cn_vox):
            cu = cn_vox * SCALE
            dg, jg = gtree.query(cu)
            gmiss = miss[jg] & (dg <= r_um)
            for kmin in ks:
                m_ = kn >= kmin
                if m_.any() and miss.any():
                    d_ = cKDTree(cu[m_]).query(gt_um[miss])[0]
                    st[f"miss_hit{kmin}"] += int((d_ <= r_um).sum())
    for kk in ks:
        m_ = kn == kk
        st[f"new_{kk}"] += int(m_.sum())
        st[f"newann_{kk}"] += int((m_ & (dg <= r_um)).sum())
        st[f"newmiss_{kk}"] += int((m_ & gmiss).sum())
    return st, dg, gmiss


if __name__ == "__main__":
    TOT = defaultdict(float)
    DGT = defaultdict(list)
    F4V = defaultdict(list)
    ROWS = []
    PER = []
    t_all = time.time()
    done = 0
    for i_, f in enumerate(FILES):
        if time.time() - t_all > BUDGET_S:
            print(f"BUDGET: stopping before {f.stem}", flush=True)
            break
        tm = time.time()
        z = np.load(f)
        img = z["image"]
        if img.ndim != 4:
            print(f"  WARN {f.stem}: image ndim {img.ndim}, expected 4 -- skipped", flush=True)
            continue
        T = int(img.shape[0])
        g_t, g_pos = zh_gt(z, T)
        out = {}
        stat = defaultdict(float)
        for t in range(T):
            vol = np.asarray(img[t])
            hs = branch_maps4(vol)
            f4 = sum(hs) / 4.0
            cr, ch, s0 = detect_both(hs, vol, (1.0, 1.0, 1.0, 1.0))
            c0 = np.asarray(ch, np.float32).reshape(-1, 3)
            s0 = np.asarray(s0, np.float32).reshape(-1)
            prim = c0[s0 >= UNET_THRESH] * SCALE
            C, K, Vm, Ff = vote_candidates(hs, f4)
            del hs, f4
            if len(C):
                refined = np.array([_refine(vol, c) for c in _unpool(C)], float).reshape(-1, 3)
                sel = select_new(refined, K, Vm, Ff, prim)
            else:
                refined, sel = np.zeros((0, 3)), np.zeros(0, int)
            del vol
            cn = refined[sel].astype(np.float32)
            kn = K[sel].astype(np.int8)
            sn = np.asarray([SCORE_OF_K[int(min(kk, 4))] for kk in kn], np.float32)
            out[f"c{t}"] = np.concatenate([c0, cn]) if len(cn) else c0
            out[f"s{t}"] = np.concatenate([s0, sn]) if len(sn) else s0
            out[f"nnew{t}"] = np.int32(len(cn))
            out[f"k{t}"] = kn
            out[f"f{t}"] = Ff[sel].astype(np.float32)
            out[f"v{t}"] = Vm[sel].astype(np.float32)
            st, dg, gmiss = frame_stats(prim, cn, kn, g_pos[g_t == t])
            st["raw_clusters"] += len(C)
            for k2, v2 in st.items():
                stat[k2] += v2
            for j_ in range(len(cn)):
                ROWS.append([f.stem, t, int(kn[j_]), float(Vm[sel][j_]), float(Ff[sel][j_]), float(dg[j_]), int(gmiss[j_])])
                if dg[j_] <= RESCUE_UM:
                    DGT[int(kn[j_])].append(float(dg[j_]))
                F4V[int(kn[j_])].append(float(Ff[sel][j_]))
            if t % 25 == 0:
                gc.collect()
        for k_ in GT_KEYS:
            if k_ in z.files:
                out[k_] = z[k_]
        np.savez_compressed(OUT / f"zhvote_{f.stem}.npz", **out)
        done += 1
        for k2, v2 in stat.items():
            TOT[k2] += v2
        n_new = sum(stat[f"new_{kk}"] for kk in KS)
        n_new_ann = sum(stat[f"newann_{kk}"] for kk in KS)
        PER.append(dict(crop=f.stem, T=T, shipped=stat["shipped"], shipped_prec=stat["shipped_ann"] / max(1, stat["shipped"]), gt=stat["gt"],
                        gt_miss=stat["gt_miss"], new=n_new, new_ann=n_new_ann, rec2=stat["miss_hit2"], rec3=stat["miss_hit3"]))
        print(f"  [{done}/{len(FILES)}] {f.stem}: T={T}, shipped {stat['shipped']:,.0f} (GT within 7 um: {stat['shipped_ann'] / max(1, stat['shipped']):.3f}), "
              f"GT {stat['gt']:,.0f} missed {stat['gt_miss']:,.0f} ({stat['gt_miss'] / max(1, stat['gt']):.1%}); clusters {stat['raw_clusters']:,.0f}, "
              f"new {n_new:,} (+{n_new / max(1, stat['shipped']):.1%}; GT share {n_new_ann / max(1, n_new):.3f}); by votes "
              + " ".join(f"[{kk}: {stat[f'new_{kk}']:.0f}/{stat[f'newann_{kk}']:.0f}]" for kk in KS)
              + f"; recovered k>=2 {stat['miss_hit2']:.0f} k>=3 {stat['miss_hit3']:.0f}  {time.time() - tm:.0f}s (total {(time.time() - t_all) / 60:.1f} min)", flush=True)
        del img, out, z
        gc.collect()

    def _q(vals, f_):
        v = np.asarray(vals, float)
        return float(np.quantile(v, f_)) if len(v) else float("nan")

    import pandas as pd
    pd.DataFrame(ROWS, columns=["crop", "t", "k", "v", "f", "d_gt", "gt_missed"]).to_csv(OUT / "zhvote_cands.csv", index=False)
    pd.DataFrame(PER).to_csv(OUT / "zhvote_crops.csv", index=False)
    base = TOT["shipped_ann"] / max(1, TOT["shipped"])
    print(f"\n=== ZHVOTE over {done} crops, DENSE GT: shipped primaries {TOT['shipped']:,.0f}, share with a GT node within 7 um {base:.4f} "
          f"(6bba sparse: 0.0548); GT nodes {TOT['gt']:,.0f}, missed at 7 um {TOT['gt_miss']:,.0f} ({TOT['gt_miss'] / max(1, TOT['gt']):.2%}; 6bba sparse: 2,067 of 113,121 = 1.83%); "
          f"vote clusters before exclusion {TOT['raw_clusters']:,.0f} ===", flush=True)
    print(f"  {'votes':<8}{'count':>9}{'+% of shipped':>14}{'GT<=7um':>9}{'share':>8}{'x shipped':>11}{'on a MISSED cell':>18}{'f4 med':>8}{'f4 p90':>8}{'d->GT med':>10}   6bba: share / x shipped", flush=True)
    for kk, ref in zip(KS, ("0.0084 / 0.15x", "0.0090 / 0.16x", "0.0064 / 0.12x")):
        n, a, m = TOT[f"new_{kk}"], TOT[f"newann_{kk}"], TOT[f"newmiss_{kk}"]
        print(f"  {kk:<8}{n:>9,.0f}{n / max(1, TOT['shipped']):>14.2%}{a:>9,.0f}{a / max(1, n):>8.4f}{(a / max(1, n)) / max(base, 1e-9):>10.2f}x"
              f"{m:>10,.0f} ({m / max(1, n):.3f}){_q(F4V[kk], .5):>8.3f}{_q(F4V[kk], .9):>8.3f}{_q(DGT[kk], .5):>10.2f}   {ref}", flush=True)
    for lab, kmin, ref in (("k >= 2", 2, "6bba: +10.0% nodes at 0.13x, recovered 637 of 2,067 (30.8%)"), ("k >= 3", 3, "6bba: +3.7% at 0.16x, 203 (9.8%)"), ("k >= 4", 4, "")):
        n = sum(TOT[f"new_{kk}"] for kk in KS if kk >= kmin)
        a = sum(TOT[f"newann_{kk}"] for kk in KS if kk >= kmin)
        print(f"  admit {lab}: {n:,.0f} candidates (+{n / max(1, TOT['shipped']):.2%} of shipped), GT within 7 um {a:,.0f} ({a / max(1, n):.4f}, "
              f"{(a / max(1, n)) / max(base, 1e-9):.2f}x shipped); missed GT cells recovered within 7 um: {TOT[f'miss_hit{kmin}']:,.0f} of {TOT['gt_miss']:,.0f} "
              f"({TOT[f'miss_hit{kmin}'] / max(1, TOT['gt_miss']):.1%})   {ref}", flush=True)
    print("  reading: 'x shipped' is the domain-free comparison (the candidates' GT share over the shipped primaries' GT share); 0.13x on the in-sample "
          "6bba movies. A markedly higher ratio here says the vote finds more cells where the detector is out of sample; 'on a MISSED cell' is the part "
          "of the candidates that stands where the shipped detector found nothing within 7 um (the only part a linker can turn into new true edges).", flush=True)
    print(f"\nZHVOTE DONE in {(time.time() - t_all) / 60:.1f} min", flush=True)
    print("ZHVOTE OK" if done == len(FILES) else f"ZHVOTE INCOMPLETE {done}/{len(FILES)}", flush=True)
'''

out_src = head + loop
assert out_src.count("if __name__") == 1
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "zhvote.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(out_src)
print("written zhvote.py", len(out_src.splitlines()), "lines")
