"""hparacache.py -> hplace.py: which of the four heatmaps PLACES a cell most
accurately, at swap sites and at ordinary cells?

WHY (2026-09-13). The swap bucket is a placement problem (swapplace: 3.6 um
from the annotated centroid at swap sites against 1.2 um elsewhere, on the
four-map MEAN). Whether any single map places better than the mean has never
been measured; the maps may agree on WHICH cells exist and still differ by
microns on WHERE they are. Blind merging of near pairs on the mean
(hmerge) lost edges, so this is the remaining measurable question before
anything else is built on placement.

WHAT IT DOES. The shipped heatmap pass (branch_maps4: xl b0, b1, b2 and ep05
DeepCenter, each the sigmoid of its 4-flip logit mean; f4 = their mean),
anchored frame by frame against the shipped fdet_F4_ cache through
detect_both's refine half, on the 1,354 frames that carry a site. Sites
(embedded gzip+base64 CSV, built locally from bench_csv/hsaddle and
bench_csv/divmiss):
  swap  1,643  the GT child A of a swap pair (hsaddle's swap sites)
  ctrl  3,963  annotated cells from the SAME frames, 3 per frame, seeded,
               each with a shipped primary within 5 um
Per site and per map m in (b0, b1, b2, dc, f4), on the pooled grid
(isotropic 1.625 um):
  local maxima of m (peak_local_max, min_distance 1, >= DETECT_THRESH)
  within 7 um of A (the scorer's match radius); n7_m = how many;
  the nearest one, sub-voxel shifted by the shipped parabolic step
  (_hpara_shift): d_m = its distance to A in um, dz_m / dxy_m its z and xy
  parts; v_m = the map value at the peak; dr_m = the distance after the
  shipped image-mass _refine on top (what the pipeline would place).
  A map with no maximum within 7 um gets NaN and is counted as "missed".
Reference columns: d_det = distance from A to the nearest SHIPPED detection
(hpara + refine, the fdet_F4hp_ placement), n_prim9 = shipped detections
within 9.5 um of A (crowdedness by detections; annotation is too sparse for
a GT-based measure).
OUTPUT: hplace_sites.csv and a summary by kind x map: median / p75 / share
> 2.5 um of d_m, missed share, n7 >= 2 share, which map is nearest per
site, paired median d_m - d_f4. v2 (2026-09-13b) adds the chosen peak's
pooled coordinates per map (pz, py, px) and the four maps' peaks AGAINST
EACH OTHER: pairwise distance, per-site max, the share of sites where the
four peaks sit within 1.5 um (wrong together) or spread over 3 um (wrong in
different places, the mean landing between them), with d -> A in each. Attach: hsaddle's sources. GPU T4, ~20 min.
"""
import os
import sys

src = open("hparacache.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:80])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-08a-hparacache"', 'SCRIPT_VERSION = "2026-09-13b-hplace"')
rep("import gc\nimport glob\n", "import base64\nimport gc\nimport glob\nimport gzip\n")

cut = "names_all = sorted(p.name[:-len(\".zarr\")] for p in TRAIN_DIR.glob(\"*.zarr\"))"
assert src.count(cut) == 1
head = src[:src.index(cut)]

tail = r'''
# ---- HPLACE: per-map placement against the annotated centroid ----------------
SITES_B64 = "@@SITES@@"
UNPOOL_OFF = (POOL - 1) / 2.0          # detect_both: c_pooled * POOL + 1.5
PSCALE = np.array([SCALE[0], SCALE[1] * POOL, SCALE[2] * POOL])   # pooled grid, um per voxel
assert abs(PSCALE[0] - PSCALE[1]) < 1e-9, PSCALE                  # isotropic 1.625 um
MAPS = ["b0", "b1", "b2", "dc", "f4"]
R_MATCH = 7.0                          # the scorer's match radius, um
R_CROWD = 9.5


def load_sites():
    raw = gzip.decompress(base64.b64decode(SITES_B64)).decode()
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    hdr = lines[0].split(",")
    rows = []
    for ln in lines[1:]:
        v = ln.split(",")
        r = dict(zip(hdr, v))
        for k in hdr[3:]:
            r[k] = float(r[k]) if r[k] not in ("", "nan") else float("nan")
        r["t"] = int(r["t"])
        rows.append(r)
    return hdr, rows


def to_pooled(zyx):
    zyx = np.asarray(zyx, float)
    return np.array([zyx[0], (zyx[1] - UNPOOL_OFF) / POOL, (zyx[2] - UNPOOL_OFF) / POOL])


def from_pooled(p):
    p = np.asarray(p, float)
    return np.array([p[0], p[1] * POOL + UNPOOL_OFF, p[2] * POOL + UNPOOL_OFF])


def um_p(p, q):
    return float(np.linalg.norm((np.asarray(p, float) - np.asarray(q, float)) * PSCALE))


def um_full(p, q):
    return float(np.linalg.norm((np.asarray(p, float) - np.asarray(q, float)) * SCALE))


def place_one(m, vol, pA, thr=DETECT_THRESH, R=R_MATCH):
    """Nearest local maximum of map m within R um of pooled point pA.
    Returns dict(d, dz, dxy, v, n7, dr) with NaN when there is none."""
    Z, Y, X = m.shape
    r = int(np.ceil(R / PSCALE[0])) + 1
    z0, y0, x0 = (int(round(v)) for v in pA)
    zs = slice(max(0, z0 - r), min(Z, z0 + r + 1))
    ys = slice(max(0, y0 - r), min(Y, y0 + r + 1))
    xs = slice(max(0, x0 - r), min(X, x0 + r + 1))
    sub = m[zs, ys, xs]
    nan = float("nan")
    if sub.size == 0:
        return dict(d=nan, dz=nan, dxy=nan, v=nan, n7=0, dr=nan, pz=nan, py=nan, px=nan)
    pk = peak_local_max(sub, min_distance=1, threshold_abs=thr, exclude_border=False)
    if len(pk) == 0:
        return dict(d=nan, dz=nan, dxy=nan, v=nan, n7=0, dr=nan, pz=nan, py=nan, px=nan)
    pk = pk + np.array([zs.start, ys.start, xs.start])
    d = np.linalg.norm((pk - pA[None, :]) * PSCALE[None, :], axis=1)
    inR = d <= R
    if not inR.any():
        return dict(d=nan, dz=nan, dxy=nan, v=nan, n7=0, dr=nan, pz=nan, py=nan, px=nan)
    n7 = int(inR.sum())
    k = int(np.argmin(np.where(inR, d, np.inf)))
    p = _hpara_shift(m, pk[k:k + 1])[0]
    dv = (p - pA) * PSCALE
    out = dict(d=float(np.linalg.norm(dv)), dz=float(abs(dv[0])), dxy=float(np.hypot(dv[1], dv[2])),
               v=float(m[tuple(pk[k])]), n7=n7, pz=float(p[0]), py=float(p[1]), px=float(p[2]))
    pr = _refine(vol, from_pooled(p))
    out["dr"] = um_full(pr, from_pooled(pA))
    return out


SUB = ["d", "dz", "dxy", "v", "n7", "dr", "pz", "py", "px"]
KEYS = ["d_det", "n_prim9"] + [f"{s}_{mp}" for mp in MAPS for s in SUB]

# ---- the anchor caches ---------------------------------------------------------
OLD = {}
for _d in pathlib.Path("/kaggle/input").glob("*/**/fdet_F4_*.npz"):
    OLD[_d.name[len("fdet_F4_"):-len(".npz")]] = _d
print(f"anchor caches found: {len(OLD)} fdet_F4_*.npz", flush=True)

hdr, rows = load_sites()
by_frame = defaultdict(list)
for r in rows:
    by_frame[(r["dataset"], r["t"])].append(r)
by_movie = defaultdict(list)
for (nm, t) in by_frame:
    by_movie[nm].append(t)
print(f"sites {len(rows)}: " + ", ".join(f"{k} {sum(1 for r in rows if r['kind'] == k)}" for k in ("swap", "ctrl"))
      + f"; frames {len(by_frame)} in {len(by_movie)} movies", flush=True)
_missing = [n for n in by_movie if n not in OLD]
assert not _missing, f"no fdet_F4_ anchor cache for {_missing[:4]} -- attach faustine649/biohub-fusecache + fuse6bba"

OUT = pathlib.Path("/kaggle/working")
fh = open(OUT / "hplace_sites.csv", "w", encoding="utf-8", newline="\n")
fh.write("kind,dataset,t,gz,gy,gx,d_nn," + ",".join(KEYS) + "\n")
RES = defaultdict(list)
t0 = time.time()
n_frames = n_bad = 0
ANCHOR_FAILS = []
for k_, nm in enumerate(sorted(by_movie)):
    img = open_image(TRAIN_DIR / f"{nm}.zarr")
    ref_old = np.load(OLD[nm])
    tm = time.time()
    for t in sorted(by_movie[nm]):
        vol = np.asarray(img[t])
        hs = branch_maps4(vol)
        maps = dict(zip(MAPS[:4], hs))
        maps["f4"] = sum(hs) / 4.0
        cr, ch, s = detect_both(hs, vol, (1.0, 1.0, 1.0, 1.0))
        key = f"c{t}"
        if key in ref_old.files:
            o = np.asarray(ref_old[key], np.float32)
            n_ = np.asarray(cr, np.float32)
            if o.shape != n_.shape or not np.array_equal(o, n_):
                n_bad += 1
                if len(ANCHOR_FAILS) < 6:
                    ANCHOR_FAILS.append((nm, t, o.shape, n_.shape))
        prim = ch[s >= UNET_THRESH] if len(ch) else np.zeros((0, 3))
        tree = cKDTree(prim * SCALE) if len(prim) else None
        for r in by_frame[(nm, t)]:
            A = np.array([r["gz"], r["gy"], r["gx"]], float)
            pA = to_pooled(A)
            o_ = {}
            if tree is not None:
                dd, _ = tree.query(A * SCALE, k=1)
                o_["d_det"] = float(dd)
                o_["n_prim9"] = len(tree.query_ball_point(A * SCALE, r=R_CROWD))
            else:
                o_["d_det"] = float("nan")
                o_["n_prim9"] = 0
            for mp in MAPS:
                pl = place_one(maps[mp], vol, pA)
                for s_ in SUB:
                    o_[f"{s_}_{mp}"] = pl[s_]
            RES[r["kind"]].append(o_)
            fh.write(f"{r['kind']},{nm},{t},{r['gz']:.3f},{r['gy']:.3f},{r['gx']:.3f},{r['d_nn']:.3f},"
                     + ",".join(f"{o_[k]:.4g}" if o_[k] == o_[k] else "nan" for k in KEYS) + "\n")
        n_frames += 1
        del hs, maps, vol
    fh.flush()
    ref_old.close()
    gc.collect()
    print(f"  [{k_ + 1}/{len(by_movie)}] {nm}: {len(by_movie[nm])} frames, anchor misses so far {n_bad}, "
          f"{time.time() - tm:.0f}s (total {(time.time() - t0) / 60:.1f} min)", flush=True)
    if k_ == 0:
        assert n_bad == 0, f"ANCHOR FAILED on the first movie: {ANCHOR_FAILS[:3]}"
        print("  ANCHOR OK on movie 1: every computed frame reproduces fdet_F4_ exactly", flush=True)
fh.close()


def q(vals, f):
    v = np.array([x for x in vals if x == x], float)
    return float(np.quantile(v, f)) if len(v) else float("nan")


def share(vals, pred):
    v = [x for x in vals if x == x]
    return float(np.mean([pred(x) for x in v])) if v else float("nan")


print(f"\n=== HPLACE: {n_frames} frames, {sum(len(v) for v in RES.values())} sites; anchor misses {n_bad} ===", flush=True)
for kind, lab in (("swap", "swap sites (GT child A of a swap pair)"), ("ctrl", "ordinary annotated cells, same frames")):
    R = RES[kind]
    print(f"\n  --- {lab}: n = {len(R)}; shipped detection -> A median {q([o['d_det'] for o in R], .5):.2f} um, "
          f"> 2.5 um {share([o['d_det'] for o in R], lambda x: x > 2.5):.2f}; detections within 9.5 um: "
          f"median {q([o['n_prim9'] for o in R], .5):.0f}, >= 2 share {share([o['n_prim9'] for o in R], lambda x: x >= 2):.2f}", flush=True)
    print(f"  {'map':<5}{'missed':>8}{'d med':>8}{'d p75':>8}{'>2.5um':>8}{'dz med':>8}{'dxy med':>9}{'dr med':>8}{'v med':>7}{'n7>=2':>7}{'d-d_f4 med':>12}", flush=True)
    for mp in MAPS:
        d_ = [o[f"d_{mp}"] for o in R]
        miss = share([1.0 if x != x else 0.0 for x in d_], lambda x: x == 1.0)
        pair = [o[f"d_{mp}"] - o["d_f4"] for o in R if o[f"d_{mp}"] == o[f"d_{mp}"] and o["d_f4"] == o["d_f4"]]
        print(f"  {mp:<5}{miss:>8.3f}{q(d_, .5):>8.2f}{q(d_, .75):>8.2f}{share(d_, lambda x: x > 2.5):>8.2f}"
              f"{q([o[f'dz_{mp}'] for o in R], .5):>8.2f}{q([o[f'dxy_{mp}'] for o in R], .5):>9.2f}"
              f"{q([o[f'dr_{mp}'] for o in R], .5):>8.2f}{q([o[f'v_{mp}'] for o in R], .5):>7.2f}"
              f"{share([o[f'n7_{mp}'] for o in R], lambda x: x >= 2):>7.2f}{q(pair, .5):>+12.2f}", flush=True)
    # v2: the four maps' peaks against EACH OTHER (pooled coords, isotropic 1.625 um)
    pair_d, cen_d, maxpair = [], [], []
    for o in R:
        P = [np.array([o[f"pz_{mp}"], o[f"py_{mp}"], o[f"px_{mp}"]]) for mp in MAPS[:4] if o[f"pz_{mp}"] == o[f"pz_{mp}"]]
        if len(P) == 4:
            dd = [um_p(P[a], P[b]) for a in range(4) for b in range(a + 1, 4)]
            pair_d.extend(dd)
            maxpair.append(max(dd))
            pf = np.array([o[f"pz_f4"], o[f"py_f4"], o[f"px_f4"]])
            cen_d.append(um_p(np.mean(P, axis=0), pf))
    print(f"  four maps' peaks against each other (sites with all four found: {len(maxpair)}): pairwise distance "
          f"median {q(pair_d, .5):.2f} um, p75 {q(pair_d, .75):.2f}; per-site max pairwise median {q(maxpair, .5):.2f}, "
          f"p75 {q(maxpair, .75):.2f}, > 3 um share {share(maxpair, lambda x: x > 3):.2f}, <= 1.5 um share "
          f"{share(maxpair, lambda x: x <= 1.5):.2f}; the four peaks' mean -> f4's own peak median {q(cen_d, .5):.2f}", flush=True)
    win = defaultdict(int)
    for o in R:
        ds = [(o[f"d_{mp}"], mp) for mp in MAPS[:4] if o[f"d_{mp}"] == o[f"d_{mp}"]]
        if ds:
            win[min(ds)[1]] += 1
        else:
            win["none"] += 1
    tot = max(1, sum(win.values()))
    print("  nearest single map per site: " + ", ".join(f"{k} {v / tot:.2f}" for k, v in sorted(win.items())), flush=True)
    for lab2, pred in (("<= 1 detection within 9.5 um", lambda n: n <= 1), (">= 2 detections within 9.5 um", lambda n: n >= 2)):
        Rs = [o for o in R if pred(o["n_prim9"])]
        if not Rs:
            continue
        print(f"  {lab2} (n = {len(Rs)}): d med " + ", ".join(f"{mp} {q([o[f'd_{mp}'] for o in Rs], .5):.2f}" for mp in MAPS)
              + f"; det {q([o['d_det'] for o in Rs], .5):.2f}", flush=True)
    # v2: the tight / scattered split -- are the maps wrong TOGETHER or in different places?
    for lab2, pred in (("four peaks within 1.5 um of each other", lambda x: x <= 1.5), ("four peaks spread > 3 um", lambda x: x > 3)):
        Rs = []
        for o in R:
            P = [np.array([o[f"pz_{mp}"], o[f"py_{mp}"], o[f"px_{mp}"]]) for mp in MAPS[:4] if o[f"pz_{mp}"] == o[f"pz_{mp}"]]
            if len(P) == 4 and pred(max(um_p(P[a], P[b]) for a in range(4) for b in range(a + 1, 4))):
                Rs.append(o)
        if Rs:
            print(f"  {lab2} (n = {len(Rs)}): d -> A med " + ", ".join(f"{mp} {q([o[f'd_{mp}'] for o in Rs], .5):.2f}" for mp in MAPS)
                  + f"; n7>=2 f4 {share([o['n7_f4'] for o in Rs], lambda x: x >= 2):.2f}", flush=True)
print("\n  reading: d is the map's own nearest maximum (parabolic sub-voxel shift, no image refine) to the "
      "annotated centroid; GT z is integer planes (1.625 um), so dz carries ~0.8 um of quantisation. A map that "
      "places swap-site cells within ~2 um where f4 sits at ~3.5 would be a placement source for dense regions; "
      "if all five sit together, placement error is in the image or the annotation, not the fusion.", flush=True)
print(f"\nHPLACE DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
print("HPLACE OK" if n_bad == 0 else f"HPLACE ANCHOR DRIFT on {n_bad} frames", flush=True)
'''

if __name__ == "__main__":
    b64 = open(sys.argv[1], encoding="ascii").read().strip()
    assert tail.count("@@SITES@@") == 1
    out_src = head + tail.replace("@@SITES@@", b64)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hplace.py")
    open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(out_src)
    print("written hplace.py", len(out_src))
