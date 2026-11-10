"""hparacache.py -> hsaddle.py: does the HEATMAP separate the two peaks of a
swap pair, or is it one ridge?

WHY (swapimg + dupsup, 2026-09-12). At the 1,643 swap sites one annotated
centroid A is flanked by two hpara detections a and b about 7 um apart; the
raw image is flat across the pair and so cannot tell one nucleus from two
touching ones, and blind suppression of near pairs (dupsup) destroyed true
edges. The remaining question is whether the fused F4 HEATMAP -- the
network's own output, sharper than the image -- has a saddle between a and
b (two centres: leave them) or a continuous ridge (one centre: merge them
at the ridge's mass centroid). If the heatmap saddle separates swap pairs
from pairs of two annotated cells, a merge rule has its discriminator and
the next step is a merged detection cache; if not, placement has to come
from a retrained target.

WHAT IT DOES. The shipped heatmap pass (branch_maps4, equal F4 weights,
anchored frame by frame against the shipped fdet_F4_ cache through
detect_both's refine half), on the 1,674 frames that carry a site. Sites,
embedded (gzip + base64 CSV, built locally from bench_csv/rkedge and
bench_csv/divmiss):
  swap    1,643  frame t+1: a = true-child detection, b = false target,
                 A = the GT child
  gtpair    511  two annotated nodes within 9.5 um in one frame, both with
                 a distinct primary within 5 um: a, b = their primaries,
                 A = the first node (two cells by construction: the
                 control the merge rule must NOT fire on)
  rand    1,348  a random primary pair 6-9.5 um apart from a swap frame
                 (the population a rule would touch)
Per site, on the pooled heatmap h (isotropic 1.625 um grid):
  h_a, h_b, h_A      trilinear values at a, b, A
  saddle             min of h along a->b (25 samples) / min(h_a, h_b)
  n_peaks            maxima along a->b with prominence >= 10% of min(h_a, h_b)
  d_c4, d_c5         distance between the two peaks' heatmap-mass centroids
                     (h - 0.05 clipped at 0, within R = 4 / 5 um of each
                     peak), um: the pair merges under NMS 4.0 if < 4
  dcA_4, dcA_5       distance from the midpoint of the two centroids to A
  dcaA_4             distance from a's own centroid to A (placement)
  d_ab, d_aA, d_bA   the detections' own distances
OUTPUT: hsaddle_sites.csv and a summary by kind. Attach: the support pack,
faustine649/biohub-dctrain-175-weights, faustine649/cell-tracking,
faustine649/biohub-fusecache (anchor caches), the competition. GPU T4,
~1 h (1,674 frames).
"""
import os
import sys

src = open("hparacache.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:80])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-08a-hparacache"', 'SCRIPT_VERSION = "2026-09-12a-hsaddle"')
rep("import gc\nimport glob\n", "import base64\nimport gc\nimport glob\nimport gzip\n")
rep("from scipy.spatial import cKDTree\n", "from scipy import ndimage as ndi\nfrom scipy.spatial import cKDTree\n")

cut = "names_all = sorted(p.name[:-len(\".zarr\")] for p in TRAIN_DIR.glob(\"*.zarr\"))"
assert src.count(cut) == 1
head = src[:src.index(cut)]

tail = r'''
# ---- HSADDLE: the heatmap between the two peaks of a pair -------------------
SITES_B64 = "@@SITES@@"
UNPOOL_OFF = (POOL - 1) / 2.0          # detect_both: c_pooled * POOL + 1.5
PSCALE = np.array([SCALE[0], SCALE[1] * POOL, SCALE[2] * POOL])   # pooled grid, um per voxel
assert abs(PSCALE[0] - PSCALE[1]) < 1e-9, PSCALE                  # isotropic 1.625 um


def load_sites():
    raw = gzip.decompress(base64.b64decode(SITES_B64)).decode()
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    hdr = lines[0].split(",")
    rows = []
    for ln in lines[1:]:
        v = ln.split(",")
        r = dict(zip(hdr, v))
        for k in hdr[2:]:
            r[k] = float(r[k]) if r[k] not in ("", "nan") else float("nan")
        r["t"] = int(r["t"])
        rows.append(r)
    return hdr, rows


def to_pooled(zyx):
    """full-resolution voxel coords (z, y, x) -> pooled grid coords."""
    zyx = np.asarray(zyx, float)
    return np.array([zyx[0], (zyx[1] - UNPOOL_OFF) / POOL, (zyx[2] - UNPOOL_OFF) / POOL])


def hval(h, p):
    return float(ndi.map_coordinates(h, np.asarray(p, float).reshape(3, 1), order=1, mode="nearest")[0])


def profile_peaks(prof, ref, frac=0.10):
    """Number of separate blobs along a 1-D profile: local maxima (plateaus
    collapsed) accepted highest first, a new one only if the profile dips
    at least frac * ref below BOTH it and every accepted maximum somewhere
    between them. Endpoints count as maxima. Never below 1."""
    prof = np.asarray(prof, float)
    n = len(prof)
    if ref <= 0 or n == 0:
        return 1
    cands = []
    for k in range(n):
        if k > 0 and prof[k] < prof[k - 1]:
            continue
        if k < n - 1 and prof[k] < prof[k + 1]:
            continue
        if cands and k - cands[-1] <= 1:
            cands[-1] = k
        else:
            cands.append(k)
    cands.sort(key=lambda k: -prof[k])
    acc = []
    for k in cands:
        ok = True
        for a in acc:
            lo, hi = (k, a) if k < a else (a, k)
            dip = min(prof[k], prof[a]) - prof[lo:hi + 1].min()
            if dip < frac * ref:
                ok = False
                break
        if ok:
            acc.append(k)
    return max(1, len(acc))


def mass_centroid(h, p, R_um, floor=CAND_THR):
    """Heatmap-mass centroid within R_um of pooled point p; falls back to p."""
    r = int(np.ceil(R_um / PSCALE[0]))
    z0, y0, x0 = (int(round(v)) for v in p)
    Z, Y, X = h.shape
    zs = slice(max(0, z0 - r), min(Z, z0 + r + 1))
    ys = slice(max(0, y0 - r), min(Y, y0 + r + 1))
    xs = slice(max(0, x0 - r), min(X, x0 + r + 1))
    sub = h[zs, ys, xs]
    zz, yy, xx = np.mgrid[zs, ys, xs]
    d = np.sqrt(((zz - p[0]) * PSCALE[0]) ** 2 + ((yy - p[1]) * PSCALE[1]) ** 2 + ((xx - p[2]) * PSCALE[2]) ** 2)
    w = np.clip(sub - floor, 0, None) * (d <= R_um)
    if w.sum() <= 0:
        return np.asarray(p, float)
    return np.array([(w * zz).sum(), (w * yy).sum(), (w * xx).sum()]) / w.sum()


def um_p(p, q):
    return float(np.linalg.norm((np.asarray(p) - np.asarray(q)) * PSCALE))


def segment_centroid(h, pa, pb, r_um, floor=CAND_THR):
    """Heatmap-mass centroid of the voxels within r_um of the segment pa->pb:
    the single point a saddle-based merge would place for the pair."""
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


def measure(h, a, b, A):
    """a, b, A in full-res voxel coords (A may be NaN). Returns a dict."""
    pa, pb = to_pooled(a), to_pooled(b)
    out = dict(h_a=hval(h, pa), h_b=hval(h, pb), d_ab=um_p(pa, pb))
    ref = min(out["h_a"], out["h_b"])
    n = 25
    ts = np.linspace(0, 1, n)
    pts = pa[None, :] * (1 - ts)[:, None] + pb[None, :] * ts[:, None]
    prof = ndi.map_coordinates(h, pts.T, order=1, mode="nearest")
    out["saddle"] = float(prof.min() / ref) if ref > 0 else float("nan")
    out["n_peaks"] = profile_peaks(prof, ref)
    for R in (4.0, 5.0):
        ca, cb = mass_centroid(h, pa, R), mass_centroid(h, pb, R)
        out[f"d_c{int(R)}"] = um_p(ca, cb)
        if np.all(np.isfinite(A)):
            pA = to_pooled(A)
            out[f"dcA_{int(R)}"] = um_p(0.5 * (ca + cb), pA)
            out[f"dcaA_{int(R)}"] = um_p(ca, pA)
        else:
            out[f"dcA_{int(R)}"] = float("nan")
            out[f"dcaA_{int(R)}"] = float("nan")
    seg = segment_centroid(h, pa, pb, 2.5)
    if np.all(np.isfinite(A)):
        pA = to_pooled(A)
        out["h_A"] = hval(h, pA)
        out["d_aA"] = um_p(pa, pA)
        out["d_bA"] = um_p(pb, pA)
        out["dseg_A"] = um_p(seg, pA)
    else:
        out["h_A"] = out["d_aA"] = out["d_bA"] = out["dseg_A"] = float("nan")
    return out


KEYS = ["h_a", "h_b", "h_A", "d_ab", "d_aA", "d_bA", "saddle", "n_peaks", "d_c4", "d_c5",
        "dcA_4", "dcA_5", "dcaA_4", "dcaA_5", "dseg_A"]

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
print(f"sites {len(rows)}: " + ", ".join(f"{k} {sum(1 for r in rows if r['kind'] == k)}" for k in ("swap", "gtpair", "rand"))
      + f"; frames {len(by_frame)} in {len(by_movie)} movies", flush=True)
_missing = [n for n in by_movie if n not in OLD]
assert not _missing, f"no fdet_F4_ anchor cache for {_missing[:4]} -- attach faustine649/biohub-fusecache"

OUT = pathlib.Path("/kaggle/working")
fh = open(OUT / "hsaddle_sites.csv", "w", encoding="utf-8", newline="\n")
fh.write("kind,dataset,t," + ",".join(KEYS) + "\n")
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
        h = sum(hs) / 4.0                                   # F4: equal weights
        cr, ch, s = detect_both(hs, vol, (1.0, 1.0, 1.0, 1.0))
        key = f"c{t}"
        if key in ref_old.files:
            o = np.asarray(ref_old[key], np.float32)
            n_ = np.asarray(cr, np.float32)
            if o.shape != n_.shape or not np.array_equal(o, n_):
                n_bad += 1
                if len(ANCHOR_FAILS) < 6:
                    ANCHOR_FAILS.append((nm, t, o.shape, n_.shape))
        for r in by_frame[(nm, t)]:
            a = (r["az"], r["ay"], r["ax"])
            b = (r["bz"], r["by"], r["bx"])
            A = np.array([r["gz"], r["gy"], r["gx"]], float)
            o_ = measure(h, a, b, A)
            RES[r["kind"]].append(o_)
            fh.write(f"{r['kind']},{nm},{t}," + ",".join(f"{o_[k]:.4g}" if o_[k] == o_[k] else "nan" for k in KEYS) + "\n")
        n_frames += 1
        del hs, h, vol
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


print(f"\n=== HSADDLE: {n_frames} frames, {sum(len(v) for v in RES.values())} sites; anchor misses {n_bad} ===", flush=True)
cols = (("swap", 26, "swap (a, b around GT child A)"), ("gtpair", 30, "two annotated cells (control)"), ("rand", 24, "random 6-9.5 um pair"))
print(f"  {'measure':<50}" + "".join(f"{lab:>{w}}" for _k, w, lab in cols), flush=True)


def line(label, fn):
    print(f"  {label:<50}" + "".join(f"{fn(RES[k]):>{w}}" for k, w, _l in cols), flush=True)


line("n", lambda R: f"{len(R)}")
line("a-b distance, um (med)", lambda R: f"{q([o['d_ab'] for o in R], .5):.2f}")
line("h at a / at b (med)", lambda R: f"{q([o['h_a'] for o in R], .5):.2f} / {q([o['h_b'] for o in R], .5):.2f}")
line("h at A (med)", lambda R: f"{q([o['h_A'] for o in R], .5):.2f}")
line("saddle = min(h on a->b) / min(h_a, h_b): p10/25/50/75/90",
     lambda R: "/".join(f"{q([o['saddle'] for o in R], f):.2f}" for f in (.1, .25, .5, .75, .9)))
line("saddle >= 0.5 / 0.7 / 0.9 (share)",
     lambda R: " / ".join(f"{share([o['saddle'] for o in R], (lambda x, c=c: x >= c)):.2f}" for c in (0.5, 0.7, 0.9)))
line("one maximum along a->b (share)", lambda R: f"{share([o['n_peaks'] for o in R], lambda x: x == 1):.2f}")
line("centroid pair distance R4, um (med); < 4 um (share)",
     lambda R: f"{q([o['d_c4'] for o in R], .5):.2f}; {share([o['d_c4'] for o in R], lambda x: x < 4):.2f}")
line("centroid pair distance R5, um (med); < 4 um (share)",
     lambda R: f"{q([o['d_c5'] for o in R], .5):.2f}; {share([o['d_c5'] for o in R], lambda x: x < 4):.2f}")
line("merged midpoint -> A, um (med) R4 / R5",
     lambda R: f"{q([o['dcA_4'] for o in R], .5):.2f} / {q([o['dcA_5'] for o in R], .5):.2f}")
line("segment mass centroid (2.5 um of a->b) -> A, um (med / p75)",
     lambda R: f"{q([o['dseg_A'] for o in R], .5):.2f} / {q([o['dseg_A'] for o in R], .75):.2f}")
line("a's own centroid -> A, um (med) R4 / R5; a -> A",
     lambda R: f"{q([o['dcaA_4'] for o in R], .5):.2f} / {q([o['dcaA_5'] for o in R], .5):.2f}; {q([o['d_aA'] for o in R], .5):.2f}")
print("\n  reading: a rule 'merge when saddle >= s and the R-centroids fall within 4 um' has precision = swap share "
      "over (swap share + gtpair share) at that s; gtpair is two cells by construction, so its share is the "
      "collateral rate. If the swap and gtpair saddle distributions overlap, the heatmap does not carry the "
      "distinction and the fix is a retrained target.", flush=True)
print(f"\nHSADDLE DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
print("HSADDLE OK" if n_bad == 0 else f"HSADDLE ANCHOR DRIFT on {n_bad} frames", flush=True)
'''

if __name__ == "__main__":
    b64 = open(sys.argv[1], encoding="ascii").read().strip()
    assert tail.count("@@SITES@@") == 1
    out_src = head + tail.replace("@@SITES@@", b64)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hsaddle.py")
    open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(out_src)
    print("written hsaddle.py", len(out_src))
