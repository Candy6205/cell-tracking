"""submit_joint.py -> submit_hp.py: the SAME pipeline with the detections
placed at the PARABOLIC VERTEX of the heatmap instead of at the pooled peak's
integer voxel.

Nothing else changes -- not the heatmaps, not the peak set, not the threshold,
not the ranker, not the ILP, not DIV_W, not the fork gate, not the patches.
The suppression still runs on the REFINE coordinates and both placements are
indexed by the SAME `keep`, so this kernel emits the same number of nodes in
the same order as the LB 0.944 kernel, at slightly different positions.

WHY IT IS WORTH A SUBMISSION, in the unit the 2026-08-28 rule asks for -- a
mechanism, not a local number. `peak_local_max` returns an INTEGER voxel of a
4x-pooled heatmap. That is a quantisation, and it is provably one: over the
bench movies the vertex of the parabola through the peak and its two
neighbours sits a median of **0.43 um** away from the voxel centre the shipped
code uses, on every axis of every detection, always. The scorer matches
predicted to GT nodes by distance, so a systematic 0.43 um offset is a defect
you can state without mentioning the score at all -- and `_refine` downstream
cannot undo it, because it starts its search from the point this code hands it.

WHAT IT MEASURED, three domains, official scorer, paired bootstrap over movies:

  6bba, 128 movies      score +0.0031  90% [+0.0014, +0.0049]  P 0.998
  44b6, 71 movies       score +0.0013  90% [-0.0027, +0.0057]  P 0.688
  zebrahub, 38 crops    score +0.0011  90% [+0.0005, +0.0017]  P 0.998

Two positive, one null, none negative -- the three-domain rule is met. In the
configuration this kernel actually ships (REF + the joint fork gate) it reads
**0.9433 against what shipped at 0.9424**, and it is the only one of the two
live candidates that gains there without any constant being moved with it.

⚠ DO NOT COMBINE THIS WITH THE RETRAINED RANKER. rkhp ran the 2x2 on 128 6bba
movies: hpara alone 0.9298, the retrain alone 0.9290, **both together 0.9282**,
where additive would have been 0.9321. Adding the retrain on top of hpara is
-0.0016 with a 90% interval of [-0.0030, -0.0001] -- the one interval in that
table that excludes zero on the wrong side. The two changes go in two separate
submissions or not at all.

ATTACH exactly what submit_joint attached; this kernel loads no new file.
"""
import os

src = open("submit_joint.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:80])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-08a-submit-joint"',
    'SCRIPT_VERSION = "2026-09-10a-submit-hp"')

# ---- 1. expose the suppression's keep indices -----------------------------
# Both placements must survive or die TOGETHER, so the two arms differ in
# position and in nothing else. `_physical_nms` is redefined in terms of
# `_nms_keep` rather than duplicated, so the two cannot drift apart.
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
    """The indices `_physical_nms` would keep, in the same (score) order."""
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

# ---- 2. the placement rule, verbatim from refinecmp / hparacache ----------
rep('''def detect(vol):''',
    '''def _para1(a, b, c):
    """Sub-voxel offset of the vertex of the parabola through (-1,a) (0,b)
    (1,c). Zero when the three are colinear or b is not the largest.

    Copied verbatim from refinecmp.py, where the measurement that justifies
    this kernel was made. Do not "simplify" it: the `d >= -1e-12` guard is
    what stops a flat window or an upward cusp from dragging the detection
    into a MINIMUM.
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


def detect(vol):''')

# ---- 3. the one behavioural line -----------------------------------------
# The shift is applied in POOLED space, where `h` lives, and the result is
# unpooled by the same transform the integer peaks went through. Suppression
# runs on `ref` -- the shipped coordinates -- so the surviving SET and its
# ORDER are byte-for-byte the ones the 0.944 kernel produced; only the
# positions written out differ.
rep('''    coords = pk.astype(float)
    coords[:, 1] = coords[:, 1] * POOL + (POOL - 1) / 2
    coords[:, 2] = coords[:, 2] * POOL + (POOL - 1) / 2
    ref = np.array([_refine(vol, c)[0] for c in coords])
    return _physical_nms(ref, sc, NMS_UM)''',
    '''    def _unpool(zyx):
        c = zyx.copy()
        c[:, 1] = c[:, 1] * POOL + (POOL - 1) / 2
        c[:, 2] = c[:, 2] * POOL + (POOL - 1) / 2
        return c

    ref = np.array([_refine(vol, c)[0] for c in _unpool(pk.astype(float))])
    hpa = np.array([_refine(vol, c)[0] for c in _unpool(_hpara_shift(h, pk))])
    keep = _nms_keep(ref, sc, NMS_UM)
    return hpa[keep], sc[keep]''')

# ---- 4. say so in the log ------------------------------------------------
rep('''print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)''',
    '''print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)
print("  hpara placement: detections sit at the PARABOLIC VERTEX of the pooled "
      "heatmap, not\\n  at the peak voxel. Suppression still runs on the refine "
      "coordinates, so the node\\n  COUNT and ORDER are the 0.944 kernel's and "
      "only the positions move (median 0.43 um).\\n  Local, official scorer, "
      "paired over movies: 6bba/128 +0.0031 [+0.0014, +0.0049];\\n  44b6/71 "
      "+0.0013 [-0.0027, +0.0057]; zebrahub/38 +0.0011 [+0.0005, +0.0017].\\n"
      "  Shipped configuration (REF + joint gate): 0.9433 against 0.9424.",
      flush=True)''')

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "submit_hp.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
