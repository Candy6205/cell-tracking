"""hparacache.py -> mapvote.py: the VOTE detection cache -- the shipped F4HP detections plus
the points where two or more of the four single heatmaps have a local maximum that the
mean map does not.

WHY (2026-09-19). hmshoulder: at 45% of the annotated cells F4 misses, at least one of the
four single maps (xl b0 / b1 / b2, ep05 DeepCenter) has a local maximum at the 0.05 floor
within 3 um of the cell; the mean map has one at 12% -- averaging erases maxima the
branches carry. Two or more maps agree at 18% of the misses, three or more at 6%. Whether
those agreed maxima are cells where no annotation says so is unpriced: lnrescue's
background-subtracted peaks were cells at an eighth of the shipped detections' rate, and
hplace saw all four maps put extra maxima in the same wrong places at swap sites. This
kernel writes the candidates as a detection set so that (1) their annotated share by vote
count can be read against the shipped detections' share and (2) the bench (lnbench's
harness with the heads re-run on this set) can price them and lntrace can say whether the
ranker links them.

WHAT IT DOES. Per frame the shipped four-map pass (branch_maps4, detect_both; the hpara
half must reproduce fdet_F4hp_'s c{t} exactly). On each single map, peak_local_max at
VOTE_FLOOR with the shipped parabolic sub-voxel shift. Maxima of DISTINCT maps within
AGREE_UM of each other form one candidate (greedy from the highest value; nearest per map);
a candidate needs MIN_VOTES maps. Its position is the members' mean, unpooled and refined
on the raw image exactly as every shipped detection is (_refine). It is NEW iff no shipped
primary (score >= 0.15) lies within RESCUE_UM; new candidates then pass the shipped 4 um
NMS among themselves (votes first, then value). Encoded score SCORE_OF_K[k] (0.17 / 0.18 /
0.19 for 2 / 3 / 4 votes): above UNET_THRESH so the harness keeps them, distinct so the
bench can admit by vote count with lnbench's tau.

OUTPUT per 6bba movie: fdet_F4vt_<name>.npz with c{t} / s{t} = the fdet_F4hp_ arrays with
the new candidates appended after them, nnew{t}, and for the new ones k{t} (votes), f{t}
(the mean map's value at the candidate), v{t} (the members' mean single-map value). The
log prints, per movie and in total, the candidates by vote count with their annotated
share (a GT node within 7 um) against the shipped detections' share, the missed annotated
cells' recovery rate by vote count, and the annotated candidates' distance to their GT node.

Attach: hplace's dataset sources + kernel source biohub-hparacache. GPU T4, one 4-map pass
per frame, 128 movies ~2 h.
"""
import os
import sys

src = open("hparacache.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:80])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-08a-hparacache"', 'SCRIPT_VERSION = "2026-09-19a-mapvote"')
cut = "names_all = sorted(p.name[:-len(\".zarr\")] for p in TRAIN_DIR.glob(\"*.zarr\"))"
assert src.count(cut) == 1
head = src[:src.index(cut)]
for must in ("def _refine(", "def _nms_keep(", "def _hpara_shift(", "def branch_maps4(", "def detect_both(", "read_geff", "peak_local_max"):
    assert must in head, must

tail = r'''
# ---- MAPVOTE ----------------------------------------------------------------------------------------
VOTE_FLOOR = 0.05        # peak-finding floor on each single map
AGREE_UM = 2.5           # maxima of different maps within this of each other are one candidate
RESCUE_UM = 7.0          # a candidate within this of a shipped primary (s >= 0.15) is not new
NEW_NMS_UM = 4.0         # the shipped NMS radius, among the new candidates
MIN_VOTES = 2
SCORE_OF_K = {2: 0.17, 3: 0.18, 4: 0.19}
BUDGET_S = 11.0 * 3600
PSCALE_ARR = np.array([SCALE[0], SCALE[1] * POOL, SCALE[2] * POOL])


def _unpool(zyx):
    c = np.asarray(zyx, float).reshape(-1, 3).copy()
    c[:, 1] = c[:, 1] * POOL + (POOL - 1) / 2
    c[:, 2] = c[:, 2] * POOL + (POOL - 1) / 2
    return c


def single_maxima(m):
    """(pooled sub-voxel coords, values) of every local maximum of one map at VOTE_FLOOR, hpara-shifted."""
    pk = peak_local_max(m, min_distance=1, threshold_abs=VOTE_FLOOR, exclude_border=False)
    if len(pk) == 0:
        return np.zeros((0, 3)), np.zeros(0)
    v = m[pk[:, 0], pk[:, 1], pk[:, 2]].astype(float)
    return np.asarray(_hpara_shift(m, pk), float).reshape(-1, 3), v


def vote_candidates(hs, f4):
    """Maxima of distinct maps within AGREE_UM of each other, greedy from the highest value, nearest per map.
    Returns (pooled mean coords (n, 3), votes (n,), mean member value (n,), f4 value at the mean (n,)) for
    the clusters with >= MIN_VOTES maps."""
    P, V, L = [], [], []
    for i, m in enumerate(hs):
        p, v = single_maxima(m)
        P.append(p)
        V.append(v)
        L.append(np.full(len(v), i, int))
    P, V, L = np.concatenate(P), np.concatenate(V), np.concatenate(L)
    empty = (np.zeros((0, 3)), np.zeros(0, int), np.zeros(0), np.zeros(0))
    if len(P) == 0:
        return empty
    Pu = P * PSCALE_ARR
    tree = cKDTree(Pu)
    claimed = np.zeros(len(P), bool)
    cc, kk, vv = [], [], []
    for i in np.argsort(-V, kind="stable"):
        if claimed[i]:
            continue
        near = [j for j in tree.query_ball_point(Pu[i], AGREE_UM) if j != i and not claimed[j] and L[j] != L[i]]
        best = {}
        for j in near:
            d = float(np.linalg.norm(Pu[j] - Pu[i]))
            if L[j] not in best or d < best[L[j]][0]:
                best[L[j]] = (d, j)
        idx = [i] + [j for _d, j in best.values()]
        claimed[idx] = True
        if len(idx) >= MIN_VOTES:
            cc.append(P[idx].mean(0))
            kk.append(len(idx))
            vv.append(float(V[idx].mean()))
    if not cc:
        return empty
    C = np.asarray(cc, float)
    Z, Y, X = f4.shape
    ci = np.clip(np.round(C).astype(int), 0, [Z - 1, Y - 1, X - 1])
    ff = f4[ci[:, 0], ci[:, 1], ci[:, 2]].astype(float)
    return C, np.asarray(kk, int), np.asarray(vv, float), ff


def select_new(refined_vox, k, v, f, prim_um):
    """Keep the candidates with no shipped primary within RESCUE_UM, then the shipped NMS among them
    (votes first, then value). Returns the kept index array."""
    refined_vox = np.asarray(refined_vox, float).reshape(-1, 3)
    if len(refined_vox) == 0:
        return np.zeros(0, int)
    keep = np.ones(len(refined_vox), bool)
    if len(prim_um):
        d, _ = cKDTree(prim_um).query(refined_vox * SCALE)
        keep &= d > RESCUE_UM
    idx = np.where(keep)[0]
    if len(idx) == 0:
        return idx
    order_score = k[idx] * 10.0 + v[idx]
    kept = _nms_keep(refined_vox[idx], order_score, NEW_NMS_UM)
    return idx[np.asarray(kept, bool)] if np.asarray(kept).dtype == bool else idx[np.asarray(kept, int)]


FDET = {}
for _d in pathlib.Path("/kaggle/input").glob("*/**/fdet_F4hp_*.npz"):
    FDET[_d.name[len("fdet_F4hp_"):-len(".npz")]] = _d
SIX = sorted(n for n in FDET if n.startswith("6bba_"))
assert len(SIX) >= 128, f"{len(SIX)} fdet_F4hp caches (attach biohub-hparacache)"
OUT = pathlib.Path("/kaggle/working")
KS = (2, 3, 4)
TOT = defaultdict(float)
DGT = defaultdict(list)          # annotated new candidates: distance to the GT node, by k
F4V = defaultdict(list)          # f4 value at the new candidates, by k
t_all = time.time()
done = n_bad = 0
ANCHOR_FAILS = []
for k_, name in enumerate(SIX):
    if time.time() - t_all > BUDGET_S:
        print(f"BUDGET: stopping before {name}", flush=True)
        break
    tm = time.time()
    z = np.load(FDET[name])
    T = max(int(k[1:]) for k in z.files if k.startswith("c")) + 1
    gtg = read_geff(geff_path(ROOT, "train", name))
    g_t = np.asarray(gtg["t"], int)
    g_pos = np.stack([np.asarray(gtg[c_], float) for c_ in ("z", "y", "x")], axis=1) * SCALE
    img = open_image(TRAIN_DIR / f"{name}.zarr")
    out = {}
    stat = defaultdict(float)
    for t in range(T):
        c0 = np.asarray(z[f"c{t}"], np.float32).reshape(-1, 3)
        s0 = np.asarray(z[f"s{t}"], np.float32).reshape(-1)
        prim = c0[s0 >= UNET_THRESH] * SCALE
        vol = np.asarray(img[t])
        hs = branch_maps4(vol)
        f4 = sum(hs) / 4.0
        cr, ch, s = detect_both(hs, vol, (1.0, 1.0, 1.0, 1.0))
        n_ = np.asarray(ch, np.float32)
        if c0.shape != n_.shape or not np.array_equal(c0, n_):
            n_bad += 1
            if len(ANCHOR_FAILS) < 6:
                ANCHOR_FAILS.append((name, t, c0.shape, n_.shape))
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
        gt_t = g_pos[g_t == t]
        gtree = cKDTree(gt_t) if len(gt_t) else None
        stat["shipped"] += len(prim)
        stat["raw_clusters"] += len(C)
        if gtree is not None and len(prim):
            stat["shipped_ann"] += int((gtree.query(prim)[0] <= RESCUE_UM).sum())
        if gtree is not None:
            d_gp = cKDTree(prim).query(gt_t)[0] if len(prim) else np.full(len(gt_t), np.inf)
            miss = d_gp > RESCUE_UM
            stat["gt"] += len(gt_t)
            stat["gt_miss"] += int(miss.sum())
            if miss.any() and len(cn):
                ctree_all = cKDTree(cn * SCALE)
                for kmin in KS:
                    m_ = kn >= kmin
                    if m_.any():
                        d_ = cKDTree(cn[m_] * SCALE).query(gt_t[miss])[0]
                        stat[f"miss_hit{kmin}"] += int((d_ <= RESCUE_UM).sum())
                del ctree_all
        if len(cn):
            dg = gtree.query(cn * SCALE)[0] if gtree is not None else np.full(len(cn), np.inf)
            for kk in KS:
                m_ = kn == kk
                stat[f"new_{kk}"] += int(m_.sum())
                stat[f"newann_{kk}"] += int((m_ & (dg <= RESCUE_UM)).sum())
                if m_.any():
                    F4V[kk].extend(Ff[sel][m_].tolist())
                    DGT[kk].extend(dg[m_ & (dg <= RESCUE_UM)].tolist())
        if t % 25 == 0:
            gc.collect()
    np.savez_compressed(OUT / f"fdet_F4vt_{name}.npz", **out)
    done += 1
    for k2, v2 in stat.items():
        TOT[k2] += v2
    n_new = sum(stat[f"new_{kk}"] for kk in KS)
    n_new_ann = sum(stat[f"newann_{kk}"] for kk in KS)
    print(f"  [{done}/{len(SIX)}] {name}: T={T}, shipped {stat['shipped']:,.0f} (annotated {stat['shipped_ann'] / max(1, stat['shipped']):.3f}), "
          f"clusters {stat['raw_clusters']:,.0f}, new {n_new:,} (+{n_new / max(1, stat['shipped']):.1%}; annotated {n_new_ann / max(1, n_new):.3f}); by votes "
          + " ".join(f"[{kk}: {stat[f'new_{kk}']:.0f}/{stat[f'newann_{kk}']:.0f}]" for kk in KS)
          + f"; missed GT {stat['gt_miss']:.0f}, recovered k>=2 {stat['miss_hit2']:.0f} k>=3 {stat['miss_hit3']:.0f}; anchor misses {n_bad}"
          f"  {time.time() - tm:.0f}s (total {(time.time() - t_all) / 60:.1f} min)", flush=True)
    if done == 1:
        assert n_bad == 0, f"ANCHOR FAILED on the first movie: {ANCHOR_FAILS[:3]}"
        print("  ANCHOR OK on movie 1: every frame reproduces fdet_F4hp_ exactly", flush=True)
    del img, out, z
    gc.collect()


def _q(vals, f):
    v = np.asarray(vals, float)
    return float(np.quantile(v, f)) if len(v) else float("nan")


print(f"\n=== MAPVOTE over {done} movies: shipped primaries {TOT['shipped']:,.0f}, annotated share {TOT['shipped_ann'] / max(1, TOT['shipped']):.4f}; "
      f"vote clusters before exclusion {TOT['raw_clusters']:,.0f}; anchor misses {n_bad} ===", flush=True)
print(f"  {'votes':<8}{'count':>9}{'+% of shipped':>14}{'annotated':>10}{'ann share':>10}{'x shipped share':>16}{'f4 med':>8}{'f4 p90':>8}{'d->GT med':>10}", flush=True)
base = TOT["shipped_ann"] / max(1, TOT["shipped"])
for kk in KS:
    n, a = TOT[f"new_{kk}"], TOT[f"newann_{kk}"]
    print(f"  {kk:<8}{n:>9,.0f}{n / max(1, TOT['shipped']):>14.2%}{a:>10,.0f}{a / max(1, n):>10.4f}{(a / max(1, n)) / max(base, 1e-9):>16.2f}"
          f"{_q(F4V[kk], .5):>8.3f}{_q(F4V[kk], .9):>8.3f}{_q(DGT[kk], .5):>10.2f}", flush=True)
for lab, kmin in (("k >= 2", 2), ("k >= 3", 3), ("k >= 4", 4)):
    n = sum(TOT[f"new_{kk}"] for kk in KS if kk >= kmin)
    a = sum(TOT[f"newann_{kk}"] for kk in KS if kk >= kmin)
    print(f"  admit {lab}: {n:,.0f} candidates (+{n / max(1, TOT['shipped']):.2%} of shipped), annotated {a:,.0f} ({a / max(1, n):.4f}, "
          f"{(a / max(1, n)) / max(base, 1e-9):.2f}x shipped); missed annotated cells recovered within 7 um: {TOT[f'miss_hit{kmin}']:,.0f} of {TOT['gt_miss']:,.0f} "
          f"({TOT[f'miss_hit{kmin}'] / max(1, TOT['gt_miss']):.1%}); annotated GT nodes {TOT['gt']:,.0f}", flush=True)
print("  reading: 'x shipped share' near 1 means the agreed maxima are cells as often as the shipped detections are; "
      "well below 1 means the vote admits mostly texture. The recovery column is the ceiling the bench can reach; the "
      "bench (heads on this set + lnbench's harness, lntrace's accounting) says how many the ranker links.", flush=True)
print(f"\nMAPVOTE DONE in {(time.time() - t_all) / 60:.1f} min", flush=True)
print("MAPVOTE OK" if done == len(SIX) and n_bad == 0 else f"MAPVOTE INCOMPLETE {done}/{len(SIX)} anchor misses {n_bad}", flush=True)
'''

out_src = head + tail
for must in ("def vote_candidates(", "def select_new(", "def single_maxima(", "fdet_F4vt_", "MAPVOTE OK", 'SCRIPT_VERSION = "2026-09-19a-mapvote"'):
    assert out_src.count(must) >= 1, must
assert out_src.count("AGREE_UM = 2.5") == 1 and out_src.count("MIN_VOTES = 2") == 1
out = sys.argv[1] if len(sys.argv) > 1 else "mapvote.py"
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(out_src)
print(f"wrote {out}: {len(out_src.splitlines())} lines")
