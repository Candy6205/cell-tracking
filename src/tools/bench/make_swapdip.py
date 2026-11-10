"""make_swapdip.py -> swapdip.py: the raw a -> b intensity dip at swap pairs, clean two-cell pairs and random near pairs.

WHY (2026-09-23). swapvid's eye tags at the far swap sites (a-b >= 8 um, same plane) put the clear two-nucleus
sites at dip >= 0.22 (8 of 9) and the one-nucleus sites at dip <= 0.06 (7 of 9). swapimg had read the raw
profile as useless because its baseline (a TP target + the nearest other primary) was as flat as the swaps;
that baseline holds one-nucleus pairs too (swapvid #14, #21). The clean two-cell reference is hsaddle's
gtpair set (511 pairs, both ends annotated, each with its own primary), never measured on the raw image.
This run measures the same dip (swapimg's definition) with the crop centred on the pair's midpoint at
  swap     hsaddle's 1,643 swap pairs (frame t+1: true-child detection a, false target b)
  gtpair   the 511 both-annotated pairs
  rand     hsaddle's 1,348 random primary pairs 6-9.5 um apart
  rand_z1  2,000 fresh random primary pairs 6-9.5 um apart within 1 plane in z (the box a same-plane rule lives in)
and prints, per kind, the dip distribution and the share below 0.10 / 0.15 / 0.20, split by a-b band and plane
difference. The firing count of "merge when dip < thr" = rate on rand(_z1) x the population count (printed by
the generator). Output swapdip_pairs.csv. Attach the competition data + faustine649/cell-tracking. CPU, no internet.
"""
import base64
import gzip
import io
import pathlib

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

SP = pathlib.Path(r"C:\Users\48688\AppData\Local\Temp\claude\D--programming-cell-tracking\a1e8b4c4-4a60-41bb-b52c-09b7aaecb127\scratchpad")
SCALE = np.array([1.625, 0.40625, 0.40625])
N_RAND_Z1, SEED = 2000, 0

raw = open(SP / "hsaddle_sites.b64").read().strip()
try:
    txt = gzip.decompress(base64.b64decode(raw)).decode()
except Exception:                                                   # noqa: BLE001
    txt = base64.b64decode(raw).decode()
hs = pd.read_csv(io.StringIO(txt))
hs = hs[["kind", "dataset", "t", "az", "ay", "ax", "bz", "by", "bx"]].copy()
print(hs.kind.value_counts().to_dict())

# fresh random pairs, 6-9.5 um apart, |dz| <= 1 plane, spread over the movies
rng = np.random.default_rng(SEED)
files = sorted(pathlib.Path("bench_csv/divmiss/divmiss_prims").glob("*.npz"))
per = int(np.ceil(N_RAND_Z1 / len(files)))
rows = []
for f in files:
    z = np.load(f)
    t, zyx = z["t"], z["zyx"]
    xyz = zyx * SCALE
    cand = []
    for tt in rng.choice(np.unique(t), size=min(10, len(np.unique(t))), replace=False):
        m = np.where(t == tt)[0]
        pr = cKDTree(xyz[m]).query_pairs(9.5, output_type="ndarray")
        if len(pr) == 0:
            continue
        d = np.linalg.norm(xyz[m][pr[:, 0]] - xyz[m][pr[:, 1]], axis=1)
        dz = np.abs(zyx[m][pr[:, 0], 0] - zyx[m][pr[:, 1], 0])
        ok = (d >= 6) & (dz <= 1.5)
        for i, j in pr[ok]:
            cand.append((int(tt), m[i], m[j]))
    if not cand:
        continue
    for k in rng.choice(len(cand), size=min(per, len(cand)), replace=False):
        tt, i, j = cand[k]
        rows.append(("rand_z1", f.stem, tt, *zyx[i], *zyx[j]))
rz = pd.DataFrame(rows, columns=hs.columns).head(N_RAND_Z1)
sites = pd.concat([hs, rz], ignore_index=True)
print(sites.kind.value_counts().to_dict())
csv = sites.to_csv(index=False, float_format="%.3f", lineterminator="\n")
b64 = base64.b64encode(gzip.compress(csv.encode())).decode()

TEMPLATE = r'''"""swapdip: the raw intensity dip along a -> b at swap pairs, clean two-cell pairs and random near pairs.

Crop centred on the pair's midpoint (+-4 planes, +-24 voxels), smoothed (sigma 0.6 plane, 2 voxels); the
profile a -> b (25 samples) with swapimg's dip = (lower end - profile minimum) / (crop p99.5 - p20), n_prof =
prominent maxima along it, I_a / I_b, and swapz's dz_ab = plane distance between the z-profile maxima through
a and through b. Summary per kind, split by a-b band and plane difference. Output swapdip_pairs.csv.
Attach the competition data + faustine649/cell-tracking (viz_cells.py). CPU, no internet.
"""
import base64
import gzip
import io
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

SCRIPT_VERSION = "2026-09-23a-swapdip"
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
HZ, HY = 4, 24
SIG = (0.6, 2.0, 2.0)
N_PROF = 25
SITES_B64 = "__B64__"


def profile_peaks(prof, frac=0.10, ref=None):
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
        prom = prof[k] - max(left, right) if 0 < k < n - 1 else prof[k] - (right if k == 0 else left)
        if prom >= frac * rng:
            if not peaks or k - peaks[-1] > 1:
                peaks.append(k)
            else:
                peaks[-1] = k
    if not peaks:
        peaks = [int(np.argmax(prof))]
    return peaks


def measure(crop, off, a, b):
    sm = ndi.gaussian_filter(crop.astype(np.float32), SIG)
    lo_, hi_ = np.percentile(sm, 20.0), np.percentile(sm, 99.5)
    ref = float(hi_ - lo_)
    ac, bc = np.asarray(a, float) - off, np.asarray(b, float) - off
    ts = np.linspace(0, 1, N_PROF)
    pts = ac[None, :] * (1 - ts)[:, None] + bc[None, :] * ts[:, None]
    prof = ndi.map_coordinates(sm, pts.T, order=1, mode="nearest")
    out = {}
    out["I_a"], out["I_b"] = float(prof[0]), float(prof[-1])
    out["dip"] = float((min(prof[0], prof[-1]) - prof.min()) / ref) if ref > 0 else 0.0
    out["dip_hi"] = float((max(prof[0], prof[-1]) - prof.min()) / ref) if ref > 0 else 0.0
    out["n_prof"] = len(profile_peaks(prof, ref=ref))
    out["ref"] = ref
    out["bg"] = float(lo_)

    def _zprof(p):
        y_, x_ = int(np.clip(round(p[1]), 0, sm.shape[1] - 1)), int(np.clip(round(p[2]), 0, sm.shape[2] - 1))
        return sm[:, y_, x_]
    za, zb = _zprof(ac), _zprof(bc)
    pa_, pb_ = profile_peaks(za, ref=ref), profile_peaks(zb, ref=ref)
    out["dz_ab"] = float(abs(pa_[int(np.argmax(za[pa_]))] - pb_[int(np.argmax(zb[pb_]))]))
    return out


KEYS = ["d_ab", "dz_pl", "dip", "dip_hi", "n_prof", "I_a", "I_b", "ref", "bg", "dz_ab"]

if __name__ == "__main__":
    t_all = time.time()
    sites = pd.read_csv(io.StringIO(gzip.decompress(base64.b64decode(SITES_B64)).decode()))
    print(f"{len(sites)} pairs: {sites.kind.value_counts().to_dict()}", flush=True)
    fh = open(OUT / "swapdip_pairs.csv", "w", encoding="utf-8", newline="\n")
    fh.write("kind,dataset,t," + ",".join(KEYS) + "\n")
    n_done = 0
    for k_, (name, grp) in enumerate(sites.groupby("dataset", sort=True)):
        t0 = time.time()
        img = open_image(zarr_path(ROOT, "train", name))
        T, Z, Y, X = img.shape
        for _, r in grp.iterrows():
            a = np.array([r.az, r.ay, r.ax], float)
            b = np.array([r.bz, r.by, r.bx], float)
            mid = 0.5 * (a + b)
            z0, z1 = max(0, int(round(mid[0])) - HZ), min(Z, int(round(mid[0])) + HZ + 1)
            y0, y1 = max(0, int(round(mid[1])) - HY), min(Y, int(round(mid[1])) + HY + 1)
            x0, x1 = max(0, int(round(mid[2])) - HY), min(X, int(round(mid[2])) + HY + 1)
            crop = np.asarray(img[int(r.t), z0:z1, y0:y1, x0:x1], dtype=np.float32)
            off = np.array([z0, y0, x0], float)
            o = measure(crop, off, a, b)
            o["d_ab"] = float(np.linalg.norm((a - b) * SCALE))
            o["dz_pl"] = float(abs(a[0] - b[0]))
            fh.write(f"{r.kind},{name},{int(r.t)}," + ",".join(f"{o[k]:.4g}" for k in KEYS) + "\n")
            n_done += 1
        fh.flush()
        print(f"  [{k_ + 1}/{sites.dataset.nunique()}] {name}: {len(grp)} pairs, {time.time() - t0:.0f}s", flush=True)
        del img
    fh.close()
    d = pd.read_csv(OUT / "swapdip_pairs.csv")
    print("\n=== SWAPDIP: dip along a -> b, by kind ===", flush=True)
    print(f"  {'kind':<8} {'n':>5} {'d_ab':>5} {'dip med':>8} {'p25':>6} {'p75':>6} {'<0.10':>6} {'<0.15':>6} {'<0.20':>6} {'1peak':>6}", flush=True)

    def line(lab, g):
        if len(g) == 0:
            return
        print(f"  {lab:<8} {len(g):>5} {g.d_ab.median():>5.1f} {g.dip.median():>8.3f} {g.dip.quantile(.25):>6.3f} {g.dip.quantile(.75):>6.3f} "
              f"{(g.dip < .10).mean():>6.2f} {(g.dip < .15).mean():>6.2f} {(g.dip < .20).mean():>6.2f} {(g.n_prof == 1).mean():>6.2f}", flush=True)
    for kind in ("swap", "gtpair", "rand", "rand_z1"):
        line(kind, d[d.kind == kind])
    for lab, sel in (("same plane (|dz| <= 1)", d.dz_pl <= 1.5), ("|dz| >= 3 planes", d.dz_pl >= 2.5),
                     ("a-b 6-8 um", (d.d_ab >= 6) & (d.d_ab < 8)), ("a-b 8-9.5 um", d.d_ab >= 8),
                     ("same plane & a-b >= 8", (d.dz_pl <= 1.5) & (d.d_ab >= 8)), ("same plane & a-b 6-8", (d.dz_pl <= 1.5) & (d.d_ab < 8))):
        print(f"  -- {lab}", flush=True)
        for kind in ("swap", "gtpair", "rand", "rand_z1"):
            line(kind, d[(d.kind == kind) & sel])
    print("  swap: dip_hi (relative to the brighter end) < 0.15:", round(float((d[d.kind == 'swap'].dip_hi < .15).mean()), 2),
          " gtpair:", round(float((d[d.kind == 'gtpair'].dip_hi < .15).mean()), 2), flush=True)
    print(f"\nSWAPDIP DONE in {(time.time() - t_all) / 60:.1f} min, {n_done} pairs", flush=True)
    print("SWAPDIP OK", flush=True)
'''
src = TEMPLATE.replace("__B64__", b64)
open("swapdip.py", "w", encoding="utf-8", newline="\n").write(src)
print("written swapdip.py", src.count("\n"), "lines")
