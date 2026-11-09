"""Writes swapimg.py: the raw-image diagnostic at the linker's swap sites.

Usage: py -X utf8 tools/bench/make_swapimg.py <sites.b64>
The site list (gzip + base64 CSV: kind, dataset, t, gz, gy, gx, az, ay, ax,
bz, by, bx, sa, sb) is embedded verbatim into the kernel script. It was built
from bench_csv/rkedge/rkedge_edges.csv (swap pairs) and
bench_csv/divmiss/divmiss_prims + divmiss_gt (coordinates); see progress.md
"2026-09-12 swapimg" for the construction.
"""
import os
import sys

HEAD = r'''"""swapimg: what does the RAW IMAGE look like where the linker swaps?

WHY (swapplace, 2026-09-12). The 1,643 swap edges of the shipped arm sit
where the detections are 3.6 um from the annotated centroid (1.2 um
elsewhere): two detections about 5 um apart around one annotated cell, and
which one the scorer calls "the cell" turns on a fraction of a z plane. Two
readings remain and they lead to different fixes:
  (a) the raw image holds TWO nuclei there (the annotation names one of them):
      the detector's heatmap target must keep neighbours apart
      (proximity-adjusted centroids) -> retrain;
  (b) the raw image holds ONE nucleus and the heatmap put two peaks on its
      shoulders: merge the pair and re-place it on the image -> post-process.

WHAT IT DOES. For every swap site (frame t+1: GT child A, true-child
detection a, false-target detection b), every swap source (frame t: GT
parent, detection a) and 1,500 baseline sites (a true-positive edge's target
A with its detection a and the nearest other primary b), read a crop of the
raw volume around A (+-4 planes, +-24 voxels = 6.5 um x 9.75 um), smooth
(sigma 0.6 plane, 2 voxels), and measure:
  n_max6      smoothed local maxima within 6 um of A (footprint 3x9x9,
              above p20 + 0.25 (p99.5 - p20) of the crop)
  d_max_A     distance from A to the nearest such maximum (um)
  d_cent_A    distance from A to the background-subtracted intensity
              centroid within 4 um of A
  d_a_A, d_b_A  the detections' distances to A
  d_max_a, d_max_b  each detection's distance to its nearest maximum
  n_prof      maxima along the straight a->b profile (25 samples, prominence
              >= 10% of the profile range): 1 = one blob, 2 = two
  dip         depth of the deepest dip along the a->b profile below the
              lower end, as a fraction of the profile range
  I_A, I_a, I_b  smoothed intensity at A, a, b
The site list is embedded (gzip+base64 CSV built from bench_csv/rkedge and
bench_csv/divmiss). Output: swapimg_sites.csv and the summary below.
Attach: the competition data + faustine649/cell-tracking (viz_cells.py).
CPU, no internet.
"""
import base64
import gzip
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np
from scipy import ndimage as ndi

SCRIPT_VERSION = "2026-09-12a-swapimg"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

CODE_ROOTS = [
    "/kaggle/input/datasets/faustine649/cell-tracking/cell tracking",
    "/kaggle/input/datasets/faustine649/cell-tracking",
    "/kaggle/input/cell-tracking/cell tracking",
    "/kaggle/input/cell-tracking",
]
code = next((c for c in CODE_ROOTS if pathlib.Path(c, "viz_cells.py").exists()), None)
assert code is not None, "attach faustine649/cell-tracking"
sys.path.insert(0, code)
from viz_cells import find_root, open_image, zarr_path   # noqa: E402

ROOT = find_root(None)
OUT = pathlib.Path("/kaggle/working")
SCALE = np.array([1.625, 0.40625, 0.40625])
HZ, HY = 4, 24                 # crop half-sizes (planes, voxels)
SIG = (0.6, 2.0, 2.0)
FOOT = (3, 9, 9)
R_MAX_UM = 6.0
R_CENT_UM = 4.0

SITES_B64 = "@@SITES@@"


def load_sites():
    raw = gzip.decompress(base64.b64decode(SITES_B64)).decode()
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]   # the CSV was written on Windows: \r\n
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


def um(a, b):
    return float(np.linalg.norm((np.asarray(a, float) - np.asarray(b, float)) * SCALE))


def profile_peaks(prof, frac=0.10, ref=None):
    """Indices of the maxima of a 1-D profile with prominence >= frac of
    `ref` (the crop's intensity range; the profile's own range when None); a
    plateau counts once. Prominence of an interior maximum = its height above
    the higher of the two valley floors on either side; an end point is a
    maximum if it is above its neighbour by the same margin. Measuring against
    the crop's range, not the profile's, keeps a nearly flat profile across
    one bright nucleus from counting its noise as two peaks."""
    prof = np.asarray(prof, float)
    n = len(prof)
    rng = float(prof.max() - prof.min()) if ref is None else float(ref)
    if rng <= 0:
        return [0]
    peaks = []
    for k in range(n):
        if k > 0 and prof[k] < prof[k - 1]:
            continue
        if k < n - 1 and prof[k] < prof[k + 1]:
            continue
        left = prof[:k].min() if k > 0 else prof[k]
        right = prof[k + 1:].min() if k < n - 1 else prof[k]
        if 0 < k < n - 1:
            prom = prof[k] - max(left, right)
        else:
            prom = prof[k] - (right if k == 0 else left)
        if prom >= frac * rng:
            if not peaks or k - peaks[-1] > 1:
                peaks.append(k)
            else:
                peaks[-1] = k          # plateau: keep one
    if not peaks:                      # nothing prominent: one blob, its top
        peaks = [int(np.argmax(prof))]
    return peaks


def analyse(crop, off, A, a, b):
    """crop: raw (z, y, x) float32; off: crop origin in volume voxels; A, a, b:
    volume voxel coords (b may be None). Returns a dict of the measures."""
    sm = ndi.gaussian_filter(crop.astype(np.float32), SIG)
    lo_, hi_ = np.percentile(sm, 20.0), np.percentile(sm, 99.5)
    thr = lo_ + 0.25 * (hi_ - lo_)
    mx = (sm == ndi.maximum_filter(sm, size=FOOT)) & (sm > thr)
    pk = np.argwhere(mx).astype(float)                     # crop coords
    Ac = np.asarray(A, float) - off
    ac = np.asarray(a, float) - off
    out = {}
    if len(pk):
        d = np.linalg.norm((pk - Ac) * SCALE, axis=1)
        out["n_max6"] = int((d <= R_MAX_UM).sum())
        out["d_max_A"] = float(d.min())
        out["d_max_a"] = float(np.linalg.norm((pk - ac) * SCALE, axis=1).min())
    else:
        out["n_max6"] = 0
        out["d_max_A"] = np.nan
        out["d_max_a"] = np.nan
    zz, yy, xx = np.indices(sm.shape)
    dd = np.sqrt(((zz - Ac[0]) * SCALE[0]) ** 2 + ((yy - Ac[1]) * SCALE[1]) ** 2
                 + ((xx - Ac[2]) * SCALE[2]) ** 2)
    w = np.clip(sm - lo_, 0, None) * (dd <= R_CENT_UM)
    if w.sum() > 0:
        c = np.array([(w * zz).sum(), (w * yy).sum(), (w * xx).sum()]) / w.sum()
        out["d_cent_A"] = float(np.linalg.norm((c - Ac) * SCALE))
    else:
        out["d_cent_A"] = np.nan

    def _val(p):
        return float(ndi.map_coordinates(sm, np.asarray(p, float).reshape(3, 1), order=1, mode="nearest")[0])
    out["I_A"] = _val(Ac)
    out["I_a"] = _val(ac)
    out["d_a_A"] = um(a, A)
    if b is not None and np.all(np.isfinite(b)):
        bc = np.asarray(b, float) - off
        out["d_b_A"] = um(b, A)
        out["d_max_b"] = float(np.linalg.norm((pk - bc) * SCALE, axis=1).min()) if len(pk) else np.nan
        out["I_b"] = _val(bc)
        n = 25
        ts = np.linspace(0, 1, n)
        pts = ac[None, :] * (1 - ts)[:, None] + bc[None, :] * ts[:, None]
        prof = ndi.map_coordinates(sm, pts.T, order=1, mode="nearest")
        ref = float(hi_ - lo_)
        out["n_prof"] = len(profile_peaks(prof, ref=ref))
        ends = min(prof[0], prof[-1])
        out["dip"] = float((ends - prof.min()) / ref) if ref > 0 else 0.0
        out["d_ab"] = um(a, b)
    else:
        for k in ("d_b_A", "d_max_b", "I_b", "n_prof", "dip", "d_ab"):
            out[k] = np.nan
    return out


KEYS = ["n_max6", "d_max_A", "d_cent_A", "d_a_A", "d_b_A", "d_max_a", "d_max_b", "n_prof", "dip",
        "d_ab", "I_A", "I_a", "I_b"]


def q(vals, f):
    v = np.array([x for x in vals if x == x], float)
    return float(np.quantile(v, f)) if len(v) else float("nan")


def share(vals, pred):
    v = [x for x in vals if x == x]
    return float(np.mean([pred(x) for x in v])) if v else float("nan")


def summary(RES, n_done):
    print(f"\n=== SWAPIMG: {n_done} sites, raw image around the annotated centroid A ===", flush=True)
    cols = (("swap", 40, "swap (A=child, a=true child det, b=false target)"),
            ("swapsrc", 30, "swap source (A=parent, a)"),
            ("base", 42, "baseline (TP target A, a, b=nearest other primary)"))
    print(f"  {'measure':<46}" + "".join(f"{lab:>{w}}" for _k, w, lab in cols), flush=True)

    def line(label, fn):
        print(f"  {label:<46}" + "".join(f"{fn(RES[k]):>{w}}" for k, w, _l in cols), flush=True)
    line("n sites", lambda R: f"{len(R)}")
    line("maxima within 6 um of A: 0 / 1 / 2+ (share)",
         lambda R: " / ".join(f"{share([o['n_max6'] for o in R], (lambda x, c=c: x == c if c < 2 else x >= 2)):.2f}" for c in (0, 1, 2)))
    line("A -> nearest maximum, um (med / p75)",
         lambda R: f"{q([o['d_max_A'] for o in R], .5):.2f} / {q([o['d_max_A'] for o in R], .75):.2f}")
    line("A -> nearest maximum <= 2 um (share)", lambda R: f"{share([o['d_max_A'] for o in R], lambda x: x <= 2):.2f}")
    line("A -> intensity centroid (4 um), um (med)", lambda R: f"{q([o['d_cent_A'] for o in R], .5):.2f}")
    line("detection a -> A, um (med)", lambda R: f"{q([o['d_a_A'] for o in R], .5):.2f}")
    line("detection a -> its nearest maximum, um (med)", lambda R: f"{q([o['d_max_a'] for o in R], .5):.2f}")
    line("detection b -> A, um (med)", lambda R: f"{q([o['d_b_A'] for o in R], .5):.2f}")
    line("detection b -> its nearest maximum, um (med)", lambda R: f"{q([o['d_max_b'] for o in R], .5):.2f}")
    line("a-b distance, um (med)", lambda R: f"{q([o['d_ab'] for o in R], .5):.2f}")
    line("profile a->b: 1 maximum / 2+ (share)",
         lambda R: " / ".join(f"{share([o['n_prof'] for o in R], (lambda x, c=c: x == c if c < 2 else x >= 2)):.2f}" for c in (1, 2)))
    line("profile dip, fraction of range (med / p75)",
         lambda R: f"{q([o['dip'] for o in R], .5):.2f} / {q([o['dip'] for o in R], .75):.2f}")
    line("I(A) / I(a) (med)", lambda R: f"{q([o['I_A'] / o['I_a'] for o in R if o['I_a'] > 0], .5):.2f}")
    line("I(b) / I(a) (med)", lambda R: f"{q([o['I_b'] / o['I_a'] for o in R if o['I_a'] > 0 and o['I_b'] == o['I_b']], .5):.2f}")


if __name__ == "__main__":
    t_all = time.time()
    hdr, rows = load_sites()
    print(f"sites: {len(rows)}  kinds: "
          + ", ".join(f"{k} {sum(1 for r in rows if r['kind'] == k)}" for k in ("swap", "swapsrc", "base")), flush=True)
    by_movie = defaultdict(list)
    for r in rows:
        by_movie[r["dataset"]].append(r)
    fh = open(OUT / "swapimg_sites.csv", "w", encoding="utf-8", newline="\n")
    fh.write("kind,dataset,t,gz,gy,gx," + ",".join(KEYS) + "\n")
    RES = defaultdict(list)
    n_done = 0
    for k_, (name, sites) in enumerate(sorted(by_movie.items())):
        t0 = time.time()
        img = open_image(zarr_path(ROOT, "train", name))
        T, Z, Y, X = img.shape
        for r in sites:
            t = r["t"]
            A = (r["gz"], r["gy"], r["gx"])
            a = (r["az"], r["ay"], r["ax"])
            b = (r["bz"], r["by"], r["bx"]) if np.isfinite(r["bz"]) else None
            z0, z1 = max(0, int(round(A[0])) - HZ), min(Z, int(round(A[0])) + HZ + 1)
            y0, y1 = max(0, int(round(A[1])) - HY), min(Y, int(round(A[1])) + HY + 1)
            x0, x1 = max(0, int(round(A[2])) - HY), min(X, int(round(A[2])) + HY + 1)
            crop = np.asarray(img[t, z0:z1, y0:y1, x0:x1], dtype=np.float32)
            off = np.array([z0, y0, x0], float)
            o = analyse(crop, off, A, a, b)
            RES[r["kind"]].append(o)
            fh.write(f"{r['kind']},{name},{t},{A[0]:.1f},{A[1]:.1f},{A[2]:.1f},"
                     + ",".join(f"{o[k]:.4g}" if o[k] == o[k] else "nan" for k in KEYS) + "\n")
            n_done += 1
        fh.flush()
        print(f"  [{k_ + 1}/{len(by_movie)}] {name}: {len(sites)} sites, shape {tuple(img.shape)}, "
              f"{time.time() - t0:.0f}s", flush=True)
        del img
    fh.close()
    summary(RES, n_done)
    print("\n  reading: if the swap sites show 2+ maxima and a two-peaked a->b profile as often as the "
          "baseline's nearest-neighbour pairs, the image holds two nuclei (fix = detector target); if they "
          "show one maximum with both detections off it and a one-peaked profile, one nucleus carries two "
          "peaks (fix = merge and re-place). d_max_A says whether the raw image's own maximum is nearer to "
          "the annotation than the detections are.", flush=True)
    print(f"\nSWAPIMG DONE in {(time.time() - t_all) / 60:.1f} min", flush=True)
    print("SWAPIMG OK", flush=True)
'''

if __name__ == "__main__":
    b64 = open(sys.argv[1], encoding="ascii").read().strip()
    assert "@@SITES@@" in HEAD and HEAD.count("@@SITES@@") == 1
    src = HEAD.replace("@@SITES@@", b64)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "swapimg.py")
    open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
    print("written swapimg.py", len(src))
