"""zh_detect.py -> zhhpcache.py: hpara placement for the THIRD domain.

WHY. The user, 2026-09-08: "比赛的验证集两个都不是，所以最好两个都跑，甚至
zebrahub也要跑". The hidden set is neither 6bba nor 44b6, so what matters is
whether a change holds ACROSS domains, not how it scores on one. Every hpara
number so far is 6bba-only, and 6bba is the easy domain -- 97 cells/frame
against 44b6's 327, no z-edge sag, 9.71% of cells annotated against 0.77%.

WHAT DIFFERS FROM hparacache, and it is the one thing that could have made
this comparison dishonest: `zh_detect` runs NMS on the REFINED coordinates
(`_physical_nms(ref, sc, NMS_UM)`), so the survivor set depends on placement.
Left alone, the two arms would keep DIFFERENT detections and the measurement
would confound placement with detection. `_nms_keep` decides the survivors
ONCE, on refine, and both placements take that set -- the same rule
hparacache uses for its peak set.

ANCHOR, and it is a strong one: the refine placement is computed alongside
and compared frame by frame against the shipped `zhdet_<stem>.npz` from
biohub-zh-detect. Every frame must match exactly. If it does not, the
heatmap pass itself has drifted and the hpara column is meaningless.

OUTPUT: `zhdet_hp_<stem>.npz`, same keys as the shipped cache (c{t}, s{t},
plus the label arrays carried through), so anything that reads zhdet_ reads
this by changing one prefix.

Attach: zh_detect's own sources plus faustine649/biohub-zh-detect for the
anchor.
"""
import os

src = open("zh_detect.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-08-24b-zh-detect"',
    'SCRIPT_VERSION = "2026-09-08a-zhhpcache"')

# ---- the anchor caches and the shift tally ------------------------------
rep('''files = sorted(CROPS.glob("*.npz"))''',
    '''OLD = {}
for _p in pathlib.Path("/kaggle/input").glob("*/**/zhdet_*.npz"):
    if _p.name.startswith("zhdet_hp_"):
        continue
    OLD[_p.name[len("zhdet_"):-len(".npz")]] = np.load(_p)
print(f"anchor caches found: {len(OLD)} zhdet_*.npz", flush=True)
assert OLD, ("attach faustine649/biohub-zh-detect -- without it the refine "
             "arm cannot be checked and the hpara column is unverifiable")
ANCHOR_FAILS, NFR, NBAD, SHIFT = [], [0], [0], []

files = sorted(CROPS.glob("*.npz"))''')

rep('''def _physical_nms(coords, scores, radius_um, scale=SCALE):
    if len(coords) <= 1:
        return coords, scores''',
    '''def _nms_keep(coords, scores, radius_um, scale=SCALE):
    """_physical_nms's own loop, returning the KEPT INDICES instead of the
    rows. Both placements must survive or die together: if each arm ran its
    own NMS on its own coordinates the two would keep different detections
    and the comparison would confound placement with detection."""
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


def _para1(a, b, c):
    """Vertex of the parabola through (-1,a) (0,b) (1,c). Copied verbatim
    from hparacache.py, itself verbatim from refinecmp.py. The
    `d >= -1e-12` guard is what stops a flat window or an upward cusp from
    dragging the detection into a MINIMUM -- do not simplify it away."""
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
            if c0 <= 0 or c0 >= hi - 1:
                continue
            im = [z0, y0, x0]
            ip = [z0, y0, x0]
            im[ax] -= 1
            ip[ax] += 1
            zyx[i][ax] += _para1(float(h[tuple(im)]), b, float(h[tuple(ip)]))
    return zyx


def _physical_nms(coords, scores, radius_um, scale=SCALE):
    if len(coords) <= 1:
        return coords, scores''')

rep('''    ref = np.array([_refine(vol, c)[0] for c in coords])
    return _physical_nms(ref, sc, NMS_UM)''',
    '''    ref = np.array([_refine(vol, c)[0] for c in coords])
    return _physical_nms(ref, sc, NMS_UM)


def detect_both(vol):
    """(refine coords, hpara coords, scores) from ONE peak set and ONE NMS.

    The shipped `detect` runs NMS on the REFINED coordinates, so the survivor
    set depends on placement. Deciding it once, on refine, is what keeps the
    two arms scoring the same cells -- the same rule hparacache uses.
    """
    hs = []
    for models, ws, pp in _BRANCHES:
        x = pool_norm(vol, pp)
        acc = None
        for i in _FLIP_VIEWS:
            xv = np.ascontiguousarray(_flip(x, i))
            with torch.no_grad():
                lg = None
                for m, w in zip(models, ws):
                    l_ = m(torch.from_numpy(xv)[None, None].to(DEVICE)
                           )[0, 0].float().cpu().numpy()
                    lg = w * l_ if lg is None else lg + w * l_
            lg = _flip(lg, i)
            acc = lg if acc is None else acc + lg
        hs.append(1.0 / (1.0 + np.exp(-(acc / len(_FLIP_VIEWS)))))
    h = hs[0] if len(hs) == 1 else np.mean(hs, axis=0)
    pk = peak_local_max(h, min_distance=1, threshold_abs=DETECT_THRESH,
                        exclude_border=False)
    if len(pk) == 0:
        z = np.zeros((0, 3))
        return z, z.copy(), np.zeros(0)
    sc = h[pk[:, 0], pk[:, 1], pk[:, 2]].astype(float)
    coords = pk.astype(float)
    coords[:, 1] = coords[:, 1] * POOL + (POOL - 1) / 2
    coords[:, 2] = coords[:, 2] * POOL + (POOL - 1) / 2
    ref = np.array([_refine(vol, c)[0] for c in coords])
    hp = _hpara_shift(h, pk)
    hp[:, 1] = hp[:, 1] * POOL + (POOL - 1) / 2
    hp[:, 2] = hp[:, 2] * POOL + (POOL - 1) / 2
    keep = _nms_keep(ref, sc, NMS_UM)
    return ref[keep], hp[keep], sc[keep]''')

# ---- write BOTH placements, and anchor the refine one on the shipped cache
rep('''        c, s = detect(np.asarray(img[t]))
        out[f"c{t}"] = np.asarray(c, np.float32)
        out[f"s{t}"] = np.asarray(s, np.float32)''',
    '''        c, chp, s = detect_both(np.asarray(img[t]))
        out[f"c{t}"] = np.asarray(chp, np.float32)      # hpara is the OUTPUT
        out[f"s{t}"] = np.asarray(s, np.float32)
        _key = f"c{t}"
        _old = OLD.get(f.stem)
        if _old is not None and _key in _old.files:
            _o = np.asarray(_old[_key], np.float32)
            _n = np.asarray(c, np.float32)              # refine is the ANCHOR
            NFR[0] += 1
            if _o.shape != _n.shape or not np.array_equal(_o, _n):
                NBAD[0] += 1
                if len(ANCHOR_FAILS) < 6:
                    ANCHOR_FAILS.append(
                        (f.stem, t, _o.shape, _n.shape,
                         float(np.abs(_o - _n).max())
                         if _o.shape == _n.shape else float("nan")))
        if len(c):
            SHIFT.append(np.linalg.norm((chp - c) * SCALE[None, :], axis=1))''')

rep('''    np.savez_compressed(OUT / f"zhdet_{f.stem}.npz", **out)''',
    '''    np.savez_compressed(OUT / f"zhdet_hp_{f.stem}.npz", **out)''')

# ---- report the anchor and the displacement ----------------------------
rep('''print("ZH DETECT DONE"''', '''print(f"\\nANCHOR: {NFR[0] - NBAD[0]} of {NFR[0]} frames reproduce the "
      f"shipped zhdet_ cache exactly on the REFINE placement", flush=True)
if ANCHOR_FAILS:
    print(f"  [degraded] first mismatches: {ANCHOR_FAILS}", flush=True)
if SHIFT:
    _s = np.concatenate(SHIFT)
    print(f"|hpara - refine|: p50 {np.median(_s):.3f} p90 "
          f"{np.percentile(_s, 90):.3f} max {_s.max():.3f} um "
          f"({len(_s):,} detections)", flush=True)
print("ZHHPCACHE " + ("OK" if NBAD[0] == 0 and NFR[0] > 0 else "ANCHOR FAILED"),
      flush=True)
print("ZH DETECT DONE"''')

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "zhhpcache.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
