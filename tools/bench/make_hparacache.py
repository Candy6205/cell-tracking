"""fuse6bba.py -> hparacache.py: the same detections, placed by `hpara`.

WHY. refinecmp (2026-09-08) compared five rules for WHERE to put a detection,
all starting from the same peaks. `hpara` -- the vertex of the parabola
through the heatmap value and its two neighbours, one axis at a time, on the
pooled grid, before the shipped `_refine` runs on the raw volume -- beat the
shipped rule on BOTH domains:

    6bba   jitter p50 1.29 -> 0.89 um, recall +0.0008, |offset| -0.10
    44b6   jitter p50 1.51 -> 1.40 um, recall +0.0000, |offset| -0.08

Jitter is frame-to-frame wobble of a matched detection, i.e. motion the
pipeline INVENTS. On 6bba the real displacement is 2.72 um p50, so this cuts
the manufactured share of apparent motion from 47% to 33%. The ranker's
distance and velocity features are computed from exactly these coordinates.

The user, 2026-09-08: "那个hpara换成流水线臂可以先做".

WHY A CACHE KERNEL AT ALL. `fdet_F4_<name>.npz` stores only the final
coordinates and scores -- no pooled peak index, no heatmap -- so `hpara`
CANNOT be recomputed from it. The heatmap has to be produced again, which
means the GPU. Only the 24 bench movies are re-cached (the 12-movie 6bba
holdout + dctrain's seeded 12 44b6), not all 187: at fuse6bba's 82 s/movie
that is about 35 minutes instead of four hours.

THE ONE DESIGN CHOICE, and it makes the arm CONSERVATIVE. NMS runs on
physical distances between refined coordinates, so sub-voxel shifts can flip
a suppression near the 4.0 um radius -- which is where refinecmp's +0.0008
recall came from. That would change WHICH detections exist, and the cached
edge-head probabilities are joined to the bench's rows by (t, i, j): a
different detection set silently breaks that join. So NMS is run ONCE, on the
`refine` coordinates, and the survivors are then placed both ways. The two
arms therefore hold detection COUNT, ORDER and IDENTITY fixed and differ only
in position -- which is exactly refinecmp's framing, and it forgoes hpara's
small recall gain. If the arm wins anyway, it wins on placement alone.

ANCHOR, and it is a strong one: the `refine` half of every frame must
reproduce the attached `fdet_F4_<name>.npz` EXACTLY. Same weights, same
peaks, same NMS -- any drift means the heatmap pass itself changed and every
number downstream is void. Asserted hard on the first movie and counted on
the rest.

OUTPUT: fdet_F4hp_<name>.npz in the detcache c{t}/s{t} layout, so the bench
reads it with no code change.

Attach: the same sources as fuse6bba (weights + cell-tracking + competition),
PLUS faustine649/biohub-fusecache as a kernel_source for the anchor.
"""
import os

src = open("fuse6bba.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:80])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-06a-fuse6bba"',
    'SCRIPT_VERSION = "2026-09-08a-hparacache"')

# ---- NMS that reports WHICH peaks survived --------------------------------
rep('''def _physical_nms(coords, scores, radius_um, scale=SCALE):
    if len(coords) <= 1:
        return coords, scores
    pts = coords * scale[None, :]
    order = np.argsort(-scores)
    tree = cKDTree(pts)
    killed = np.zeros(len(coords), bool)
    keep = []
    for i in order:
        if killed[i]:
            continue
        keep.append(int(i))
        killed[tree.query_ball_point(pts[i], r=radius_um)] = True
    keep = np.array(keep)
    return coords[keep], scores[keep]''',
    '''def _nms_keep(coords, scores, radius_um, scale=SCALE):
    """The indices `_physical_nms` would have kept, in the same order.

    Factored out so both placement arms can be built from ONE suppression --
    see the module docstring of make_hparacache.py. `_physical_nms` below is
    now defined in terms of this, so the two cannot drift apart.
    """
    if len(coords) <= 1:
        return np.arange(len(coords))
    pts = coords * scale[None, :]
    order = np.argsort(-scores)
    tree = cKDTree(pts)
    killed = np.zeros(len(coords), bool)
    keep = []
    for i in order:
        if killed[i]:
            continue
        keep.append(int(i))
        killed[tree.query_ball_point(pts[i], r=radius_um)] = True
    return np.array(keep)


def _physical_nms(coords, scores, radius_um, scale=SCALE):
    keep = _nms_keep(coords, scores, radius_um, scale)
    if len(coords) <= 1:
        return coords, scores
    return coords[keep], scores[keep]''')

# ---- the placement rule, verbatim from refinecmp --------------------------
rep('''def detect_weighted(hs, vol, weights):''',
    '''def _para1(a, b, c):
    """Sub-voxel offset of the vertex of the parabola through (-1,a) (0,b)
    (1,c). Zero when the three are colinear or b is not the largest.

    Copied verbatim from refinecmp.py, which is where the measurement that
    justifies this kernel was made. Do not "simplify" it: the `d >= -1e-12`
    guard is what stops a flat window or an upward cusp from dragging the
    detection into a MINIMUM.
    """
    d = a - 2.0 * b + c
    if d >= -1e-12:
        return 0.0
    return float(np.clip(0.5 * (a - c) / d, -0.5, 0.5))


def _hpara_shift(h, pk):
    """pk: (n,3) pooled integer peaks -> (n,3) pooled float coordinates."""
    Zp, Yp, Xp = h.shape
    zyx = pk.astype(float)
    for i in range(len(pk)):
        z0, y0, x0 = (int(v) for v in pk[i])
        b = float(h[z0, y0, x0])
        for ax, (hi, c0) in enumerate(((Zp, z0), (Yp, y0), (Xp, x0))):
            if c0 - 1 < 0 or c0 + 1 >= hi:
                continue
            im = [z0, y0, x0]
            ip = [z0, y0, x0]
            im[ax] -= 1
            ip[ax] += 1
            zyx[i][ax] += _para1(float(h[tuple(im)]), b, float(h[tuple(ip)]))
    return zyx


def detect_both(hs, vol, weights):
    """(refine coords, hpara coords, scores) for ONE shared NMS.

    Returns two placements of the SAME surviving detections, so the arms
    differ in position and in nothing else.
    """
    w = np.asarray(weights, np.float64)
    h = sum(wi * hi for wi, hi in zip(w, hs)) / w.sum()
    pk = peak_local_max(h, min_distance=1, threshold_abs=DETECT_THRESH,
                        exclude_border=False)
    if len(pk) == 0:
        z3 = np.zeros((0, 3))
        return z3, z3, np.zeros(0)
    sc = h[pk[:, 0], pk[:, 1], pk[:, 2]].astype(float)

    def _unpool(zyx):
        c = zyx.copy()
        c[:, 1] = c[:, 1] * POOL + (POOL - 1) / 2
        c[:, 2] = c[:, 2] * POOL + (POOL - 1) / 2
        return c

    ref = np.array([_refine(vol, c) for c in _unpool(pk.astype(float))])
    hpa = np.array([_refine(vol, c) for c in _unpool(_hpara_shift(h, pk))])
    keep = _nms_keep(ref, sc, NMS_UM)
    return ref[keep], hpa[keep], sc[keep]


def detect_weighted(hs, vol, weights):''')

# ---- the 24 bench movies, both domains ------------------------------------
rep('''assert len(NAMES_6BBA) == 116, len(NAMES_6BBA)''',
    '''assert len(NAMES_6BBA) == 116, len(NAMES_6BBA)

# dctrain.py:158-163 -- the seeded 44b6 draw the whole bench is built on.
# Hardcoded here AND cross-checked against the names that actually carry an
# fdet_F4_ cache, because a silent change of this list would compare two
# different movie sets and still print a plausible number.
SET_44B6 = [
    "44b6_0b24845f", "44b6_0db75fae", "44b6_33b596bf", "44b6_5740d24b",
    "44b6_87bba6c4", "44b6_8f5ab931", "44b6_9bfa6a0a", "44b6_a2bb48bb",
    "44b6_c15fded2", "44b6_c96cfa10", "44b6_d2f34f90", "44b6_e29f0176",
]
assert len(SET_44B6) == 12 and len(set(SET_44B6)) == 12

# 2026-09-08: EXTENDED FROM 24 TO 140. The 24-movie run answered "does the
# placement change anything" and could not answer "by how much": hparabench
# scores all 128 6bba movies, only 12 of which had hpara coordinates, so 116
# rows were identical between the arms and the score delta landed inside the
# holdout's own 0.002-0.003 noise -- 90% interval [-0.0050, +0.0015], which
# is equally consistent with a real regression and with a small gain. The
# only way to resolve that is to give every scored movie hpara coordinates.
# The 12 44b6 stay in even though this bench never scores them: 18 extra
# minutes, and dropping them would break the downstream override asserts.
NAMES = HOLDOUT_6BBA + NAMES_6BBA + SET_44B6
assert len(NAMES) == 140 and len(set(NAMES)) == 140, len(NAMES)
print(f"NAMES: {len(HOLDOUT_6BBA)} holdout + {len(NAMES_6BBA)} pool 6bba "
      f"+ {len(SET_44B6)} 44b6 = {len(NAMES)}", flush=True)

# ---- the anchor caches ----------------------------------------------------
OLD = {}
for _d in pathlib.Path("/kaggle/input").glob("*/**/fdet_F4_*.npz"):
    OLD[_d.name[len("fdet_F4_"):-len(".npz")]] = _d
print(f"anchor caches found: {len(OLD)} fdet_F4_*.npz", flush=True)
_missing = [n for n in NAMES if n not in OLD]
assert not _missing, (
    f"{len(_missing)} of the {len(NAMES)} bench movies have no fdet_F4_ "
    f"cache to "
    f"anchor against -- attach faustine649/biohub-fusecache as a "
    f"kernel_source. Missing: {_missing[:4]}")
ANCHOR_FAILS = []''')

# ---- the loop -------------------------------------------------------------
rep('''for nm in NAMES_6BBA:
    img = open_image(TRAIN_DIR / f"{nm}.zarr")
    T = img.shape[0]
    outs = {tag: {} for tag, _ in ARMS}
    for t in range(T):
        vol = np.asarray(img[t])
        hs = branch_maps4(vol)
        for tag, wts in ARMS:
            c, s = detect_weighted(hs, vol, wts)
            outs[tag][f"c{t}"] = np.asarray(c, np.float32)
            outs[tag][f"s{t}"] = np.asarray(s, np.float32)
        del hs, vol
        if t % 25 == 0:
            gc.collect()
    for tag, _ in ARMS:
        np.savez_compressed(OUT / f"fdet_{tag}_{nm}.npz", **outs[tag])
    done += 1
    n15 = {tag: sum(int((outs[tag][f"s{t_}"] >= 0.15).sum())
                    for t_ in range(T)) for tag, _ in ARMS}
    print(f"  {done}/{len(NAMES_6BBA)} {nm}: T={T}, peaks>=0.15 "
          + " ".join(f"{k}={v:,}" for k, v in n15.items())
          + f" ({time.time() - t0:.0f}s)", flush=True)
    del outs
    gc.collect()''',
    '''for nm in NAMES:
    img = open_image(TRAIN_DIR / f"{nm}.zarr")
    T = img.shape[0]
    ref_old = np.load(OLD[nm])
    out = {}
    bad = nfr = 0
    shift = []
    for t in range(T):
        vol = np.asarray(img[t])
        hs = branch_maps4(vol)
        cr, ch, s = detect_both(hs, vol, (1.0, 1.0, 1.0, 1.0))
        out[f"c{t}"] = np.asarray(ch, np.float32)
        out[f"s{t}"] = np.asarray(s, np.float32)
        # ANCHOR: the refine half must reproduce the shipped cache exactly
        key = f"c{t}"
        if key in ref_old.files:
            o = np.asarray(ref_old[key], np.float32)
            n = np.asarray(cr, np.float32)
            nfr += 1
            if o.shape != n.shape or not np.array_equal(o, n):
                bad += 1
                if len(ANCHOR_FAILS) < 6:
                    d = (float(np.abs(o - n).max())
                         if o.shape == n.shape else float("nan"))
                    ANCHOR_FAILS.append((nm, t, o.shape, n.shape, d))
        if len(ch):
            shift.append(np.linalg.norm((ch - cr) * SCALE[None, :], axis=1))
        del hs, vol
        if t % 25 == 0:
            gc.collect()
    np.savez_compressed(OUT / f"fdet_F4hp_{nm}.npz", **out)
    done += 1
    sh = np.concatenate(shift) if shift else np.zeros(1)
    n15 = sum(int((out[f"s{t_}"] >= 0.15).sum()) for t_ in range(T))
    print(f"  {done}/{len(NAMES)} {nm}: T={T}, peaks>=0.15 {n15:,}, "
          f"anchor {nfr - bad}/{nfr} frames exact, "
          f"|hpara - refine| p50 {np.median(sh):.3f} p90 "
          f"{np.percentile(sh, 90):.3f} max {sh.max():.3f} um "
          f"({time.time() - t0:.0f}s)", flush=True)
    if done == 1:
        assert bad == 0, (
            f"ANCHOR FAILED on the first movie: {bad} of {nfr} frames differ "
            f"from the shipped fdet_F4_ cache. The heatmap pass itself has "
            f"changed; nothing downstream of this is comparable. "
            f"{ANCHOR_FAILS[:3]}")
        print("  ANCHOR OK on movie 1: every frame reproduces "
              "fdet_F4_ exactly", flush=True)
    ref_old.close()
    del out
    gc.collect()

if ANCHOR_FAILS:
    print(f"\\n  !! ANCHOR DRIFT on {len(ANCHOR_FAILS)}+ frames after movie 1 "
          f"-- the hpara caches are NOT comparable to the shipped ones. "
          f"First few: {ANCHOR_FAILS[:4]}", flush=True)
else:
    print("\\n  ANCHOR: every frame of every movie reproduced the shipped "
          "fdet_F4_ cache exactly.", flush=True)''')

rep('print(f"\\nFUSE6BBA DONE in {(time.time() - t0) / 60:.1f} min", flush=True)',
    'print(f"\\nHPARACACHE DONE in {(time.time() - t0) / 60:.1f} min", flush=True)\n'
    'print("HPARACACHE OK" if not ANCHOR_FAILS else "HPARACACHE ANCHOR FAILED",\n'
    '      flush=True)')

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "hparacache.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
