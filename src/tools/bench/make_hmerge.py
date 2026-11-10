"""hparacache.py -> hmerge.py: the hpara detections with saddle-keyed pair
merging, cached for the 140 bench movies.

WHY (hsaddle, 2026-09-12). Between the two peaks of a swap pair the fused
F4 heatmap is a ridge (saddle median 0.88, 78% above 0.7); between two
annotated cells it dips (0.45, 20% above 0.7). A merge keyed on that saddle
at >= 0.9 and a pair distance <= 9.5 um takes 43% of swap pairs and 4% of
two-cell pairs, and the merged point -- the heatmap-mass centroid within
2.5 um of the a->b segment -- lands 1.65 um from the annotated centroid
against 3.64 um for either detection. This kernel writes that detection set
so the round-2 bench can score it (after the two edge heads are re-run on
it: the pair set changes, so the cached (t, i, j) rows do not join).

WHAT IT DOES, per frame of each of hparacache's 140 movies: the shipped
heatmap pass (branch_maps4, equal F4 weights) and detect_both, anchored
frame by frame against the shipped fdet_F4_ cache through the refine half;
then, among the PRIMARIES (score >= UNET_THRESH 0.15) placed by hpara, every
pair within HM_D um gets its saddle = min(h along a->b, 25 samples) /
min(h_a, h_b) on the pooled grid. Pairs are merged greedily, highest saddle
first, each primary at most once, when saddle >= the arm's threshold: the
two are replaced by ONE detection at the segment mass centroid (h - 0.05
clipped at 0, within 2.5 um of the segment), unpooled and passed through the
same `_refine` every detection gets, with score max(s_a, s_b). Sub-threshold
peaks (the repair pool) are untouched. Two arms from one heatmap pass:
  HM90  saddle >= 0.90   (hsaddle: 43% of swap pairs, 4% of two-cell pairs)
  HM80  saddle >= 0.80   (64% / 10%)
OUTPUT: fdet_F4hm90_<name>.npz and fdet_F4hm80_<name>.npz in the detcache
layout (c{t}/s{t}), plus per-movie counts of primaries, pairs and merges.
Attach: hsaddle's sources (support pack, dctrain-175 weights, the two
xiaoleilian weight datasets, cell-tracking; fusecache + fuse6bba for the
anchor caches; the competition). GPU T4, ~3.5 h.
"""
import os

src = open("hparacache.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:80])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-08a-hparacache"', 'SCRIPT_VERSION = "2026-09-12a-hmerge"')
rep("from scipy.spatial import cKDTree\n", "from scipy import ndimage as ndi\nfrom scipy.spatial import cKDTree\n")

cut = "for nm in NAMES:\n    img = open_image(TRAIN_DIR / f\"{nm}.zarr\")"
assert src.count(cut) == 1
head = src[:src.index(cut)]
# the "CACHING 1 blend arms" banner above the loop describes hparacache; replace it
_b0 = head.index('print(f"\\n=== CACHING {len(ARMS)} blend arms x "')
_b1 = head.index("for nm in NAMES", _b0) if "for nm in NAMES" in head[_b0:] else len(head)
head = head[:_b0]

tail = r'''
# ---- HMERGE: saddle-keyed merging of hpara primaries ---------------------------
HM_D = 9.5                                  # pair gate, um (hsaddle's site gate)
HM_ARMS = [("hm90", 0.90), ("hm80", 0.80)]
HM_SEG_UM = 2.5                             # segment-centroid radius
UNPOOL_OFF = (POOL - 1) / 2.0
PSCALE = np.array([SCALE[0], SCALE[1] * POOL, SCALE[2] * POOL])


def to_pooled(zyx):
    zyx = np.asarray(zyx, float)
    return np.array([zyx[0], (zyx[1] - UNPOOL_OFF) / POOL, (zyx[2] - UNPOOL_OFF) / POOL])


def from_pooled(p):
    p = np.asarray(p, float)
    return np.array([p[0], p[1] * POOL + UNPOOL_OFF, p[2] * POOL + UNPOOL_OFF])


def hm_pairs(h, coords, scores, thr_prim=UNET_THRESH, D=HM_D):
    """Every pair of primaries within D um: (saddle, i, j), i < j indices into
    coords. Saddle = min of h along the pooled segment / min of the two end
    values (25 samples, trilinear)."""
    prim = np.where(np.asarray(scores) >= thr_prim)[0]
    if len(prim) < 2:
        return []
    P = np.asarray(coords, float)[prim] * SCALE
    pairs = cKDTree(P).query_pairs(D)
    if not pairs:
        return []
    out = []
    ts = np.linspace(0.0, 1.0, 25)
    for i_, j_ in pairs:
        a = to_pooled(coords[prim[i_]])
        b = to_pooled(coords[prim[j_]])
        pts = a[None, :] * (1 - ts)[:, None] + b[None, :] * ts[:, None]
        prof = ndi.map_coordinates(h, pts.T, order=1, mode="nearest")
        ref = float(min(prof[0], prof[-1]))
        sad = float(prof.min() / ref) if ref > 0 else 0.0
        out.append((sad, int(prim[i_]), int(prim[j_])))
    return out


def segment_centroid(h, pa, pb, r_um=HM_SEG_UM, floor=CAND_THR):
    pa, pb = np.asarray(pa, float), np.asarray(pb, float)
    r = int(np.ceil(r_um / PSCALE[0])) + 1
    lo = np.floor(np.minimum(pa, pb)).astype(int) - r
    hi = np.ceil(np.maximum(pa, pb)).astype(int) + r + 1
    Z, Y, X = h.shape
    zs = slice(max(0, lo[0]), min(Z, hi[0]))
    ys = slice(max(0, lo[1]), min(Y, hi[1]))
    xs = slice(max(0, lo[2]), min(X, hi[2]))
    sub = h[zs, ys, xs]
    zz, yy, xx = np.mgrid[zs, ys, xs]
    P = np.stack([zz, yy, xx], -1).astype(float) * PSCALE
    A_, B_ = pa * PSCALE, pb * PSCALE
    ab = B_ - A_
    L2 = float(ab @ ab)
    tt = np.clip(((P - A_) @ ab) / L2, 0.0, 1.0) if L2 > 0 else np.zeros(P.shape[:-1])
    d = np.linalg.norm(P - (A_ + tt[..., None] * ab), axis=-1)
    w = np.clip(sub - floor, 0, None) * (d <= r_um)
    if w.sum() <= 0:
        return 0.5 * (pa + pb)
    return np.array([(w * zz).sum(), (w * yy).sum(), (w * xx).sum()]) / w.sum()


def hm_apply(h, vol, coords, scores, cand, thr):
    """Greedy merge, highest saddle first, each detection at most once.
    Returns (coords, scores, n_merged): the merged point replaces detection i
    (the pair's first index) and detection j is dropped."""
    coords = np.asarray(coords, float)
    scores = np.asarray(scores, float)
    used = set()
    drop = set()
    new_c = coords.copy()
    new_s = scores.copy()
    n = 0
    for sad, i_, j_ in sorted(cand, key=lambda x: -x[0]):
        if sad < thr:
            break
        if i_ in used or j_ in used:
            continue
        seg = segment_centroid(h, to_pooled(coords[i_]), to_pooled(coords[j_]))
        c_full = from_pooled(seg)
        new_c[i_] = _refine(vol, c_full)
        new_s[i_] = max(scores[i_], scores[j_])
        used.add(i_)
        used.add(j_)
        drop.add(j_)
        n += 1
    keep = np.array([k for k in range(len(coords)) if k not in drop], int)
    return new_c[keep], new_s[keep], n


t0 = time.time()
done = 0
STAT = {a_: defaultdict(int) for a_, _t in HM_ARMS}
print(f"\n=== HMERGE: {len(NAMES)} movies, ALL frames; pairs of primaries within {HM_D} um "
      f"merged at the segment mass centroid when the heatmap saddle >= "
      f"{' / '.join(str(t_) for _a, t_ in HM_ARMS)} ===", flush=True)
for nm in NAMES:
    img = open_image(TRAIN_DIR / f"{nm}.zarr")
    T = img.shape[0]
    ref_old = np.load(OLD[nm])
    outs = {a_: {} for a_, _t in HM_ARMS}
    bad = nfr = 0
    tm = time.time()
    for t in range(T):
        vol = np.asarray(img[t])
        hs = branch_maps4(vol)
        h = sum(hs) / 4.0
        cr, ch, s = detect_both(hs, vol, (1.0, 1.0, 1.0, 1.0))
        key = f"c{t}"
        if key in ref_old.files:
            o = np.asarray(ref_old[key], np.float32)
            n_ = np.asarray(cr, np.float32)
            nfr += 1
            if o.shape != n_.shape or not np.array_equal(o, n_):
                bad += 1
                if len(ANCHOR_FAILS) < 6:
                    ANCHOR_FAILS.append((nm, t, o.shape, n_.shape))
        cand = hm_pairs(h, ch, s)
        n_prim = int((s >= UNET_THRESH).sum())
        for a_, thr in HM_ARMS:
            c2, s2, nmg = hm_apply(h, vol, ch, s, cand, thr)
            outs[a_][f"c{t}"] = np.asarray(c2, np.float32)
            outs[a_][f"s{t}"] = np.asarray(s2, np.float32)
            STAT[a_]["primaries"] += n_prim
            STAT[a_]["pairs"] += len(cand)
            STAT[a_]["merged"] += nmg
        del hs, h, vol
        if t % 25 == 0:
            gc.collect()
    for a_, _t in HM_ARMS:
        np.savez_compressed(OUT / f"fdet_F4{a_}_{nm}.npz", **outs[a_])
    done += 1
    print(f"  {done}/{len(NAMES)} {nm}: T={T}, anchor {nfr - bad}/{nfr} frames exact, "
          + ", ".join(f"{a_} merged {STAT[a_]['merged']:,}/{STAT[a_]['primaries']:,} primaries "
                      f"({STAT[a_]['merged'] / max(1, STAT[a_]['primaries']):.3%}, cumulative)"
                      for a_, _t in HM_ARMS)
          + f"  {time.time() - tm:.0f}s (total {(time.time() - t0) / 60:.1f} min)", flush=True)
    if done == 1:
        assert bad == 0, (f"ANCHOR FAILED on the first movie: {bad} of {nfr} frames differ from the "
                          f"shipped fdet_F4_ cache. {ANCHOR_FAILS[:3]}")
        print("  ANCHOR OK on movie 1: every frame reproduces fdet_F4_ exactly", flush=True)
    ref_old.close()
    del outs
    gc.collect()

print(f"\n=== HMERGE TOTALS over {done} movies ===", flush=True)
for a_, thr in HM_ARMS:
    d_ = STAT[a_]
    print(f"  {a_} (saddle >= {thr}): primaries {d_['primaries']:,}, pairs within {HM_D} um {d_['pairs']:,}, "
          f"merged {d_['merged']:,} ({d_['merged'] / max(1, d_['primaries']):.3%} of primaries, "
          f"{d_['merged'] / max(1, d_['pairs']):.3%} of pairs)", flush=True)
if ANCHOR_FAILS:
    print(f"\n  !! ANCHOR DRIFT on {len(ANCHOR_FAILS)}+ frames after movie 1. First few: {ANCHOR_FAILS[:4]}", flush=True)
else:
    print("\n  ANCHOR: every frame of every movie reproduced the shipped fdet_F4_ cache exactly.", flush=True)
print(f"\nHMERGE DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
print("HMERGE OK" if not ANCHOR_FAILS and done == len(NAMES) else "HMERGE INCOMPLETE OR ANCHOR FAILED", flush=True)
'''
out_src = head + tail
assert out_src.count("ANCHOR_FAILS = []") == 1 and "OUT = pathlib.Path" in out_src
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hmerge.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(out_src)
print("written hmerge.py", len(out_src))
