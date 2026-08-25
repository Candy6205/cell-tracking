"""Sheet, strand, or ball -- and if a sheet, which way is it facing?

geom2 reported `planarity = 1 - lam0/lam2` and called 0.899 "a locally
flat sheet". The user then showed a frame of 6bba_aeee7805: the cells
form a narrow diagonal BAND, not a sheet filling the field. That metric
cannot tell the two apart --

    sheet   lam0 << lam1 ~= lam2
    strand  lam0 ~= lam1 << lam2      <- also gives 1 - lam0/lam2 ~ 1
    ball    lam0 ~= lam1 ~= lam2

so it answers "is it flat OR thin", which is not the question.

The standard discriminators use the MIDDLE eigenvalue:

    linearity  (lam2 - lam1) / lam2     1 = strand
    planarity  (lam1 - lam0) / lam2     1 = sheet
    sphericity  lam0 / lam2             1 = ball

These sum to 1, so one row of three numbers says which shape it is.

THE THIRD POSSIBILITY, which the movie cannot distinguish either. The
render is a maximum-intensity projection ALONG Z. A monolayer lying flat
(normal pointing along z) fills the frame; the SAME monolayer standing
up (normal in the y-x plane) projects to a narrow band. So a band on
screen is consistent with both a strand and an edge-on sheet.

`|normal_z|` below separates them: the mean absolute z-component of the
local smallest eigenvector, i.e. the sheet's own facing direction.

    |normal_z| ~ 1   sheet lies flat, fills the projection
    |normal_z| ~ 0   sheet stands up, projects to a band

Supporting number already on record: aeee7805 has the LARGEST z spread
of the twelve (33.63 um, above the ~30 a uniform fill would give), which
is what a sheet crossing the full depth looks like.

⚠ Everything here is measured on DETECTIONS, not ground truth. If the
detector missed interior cells in a solid tissue, that tissue would look
more sheet-like than it is -- a bias that works AGAINST calling anything
solid, so a solid verdict is conservative.
"""
import pathlib

import numpy as np
from scipy.spatial import cKDTree

SCALE = np.array([1.625, 0.40625, 0.40625])
UNET_THRESH = 0.15
CACHE = pathlib.Path(r"C:\Users\48688\AppData\Local\Temp\erout")
K = 12

HOLDOUT = [
    "6bba_268e1230", "6bba_2312ac41", "6bba_283bf9f1", "6bba_3a1849c2",
    "6bba_3abfe10a", "6bba_5c824876", "6bba_61dd1e0d", "6bba_7af54fde",
    "6bba_7b5d3b2c", "6bba_afb141ff", "6bba_aeee7805", "6bba_c27cba08",
]
TRUE_DENS = {
    "6bba_268e1230": 51, "6bba_2312ac41": 120, "6bba_283bf9f1": 209,
    "6bba_3a1849c2": 70, "6bba_3abfe10a": 521, "6bba_5c824876": 246,
    "6bba_61dd1e0d": 101, "6bba_7af54fde": 96, "6bba_7b5d3b2c": 58,
    "6bba_afb141ff": 61, "6bba_aeee7805": 50, "6bba_c27cba08": 64,
}


def frame_keys(files, prefix):
    return [k for k in files
            if k.startswith(prefix) and k[len(prefix):].isdigit()]


rows = []
for name in HOLDOUT:
    f = CACHE / f"detcache_{name}.npz"
    if not f.exists():
        continue
    z = np.load(f)
    T = max(int(k[1:]) for k in frame_keys(z.files, "c")) + 1
    lin, pla, sph, nz, frac_yx = [], [], [], [], []
    for t in range(0, T, 10):
        c = np.asarray(z[f"c{t}"], float).reshape(-1, 3)
        s = np.asarray(z[f"s{t}"], float).reshape(-1)
        c = c[s >= UNET_THRESH]
        if len(c) <= K:
            continue
        P = c * SCALE[None, :]
        tree = cKDTree(P)
        step = max(1, len(P) // 300)
        for i in range(0, len(P), step):
            _d, idx = tree.query(P[i], k=K + 1)
            idx = np.asarray(idx)[1:]
            Q = P[idx] - P[idx].mean(0)
            ev, evec = np.linalg.eigh(Q.T @ Q / len(idx))
            ev = np.clip(ev, 0, None)
            if ev[2] <= 0:
                continue
            lin.append((ev[2] - ev[1]) / ev[2])
            pla.append((ev[1] - ev[0]) / ev[2])
            sph.append(ev[0] / ev[2])
            nz.append(abs(float(evec[0, 0])))   # z-component of normal
        # how much of the FRAME the cells cover, in projection -- this is
        # the thing the eye reads off the movie
        yx = np.unique((c[:, 1:] // 16).astype(int), axis=0)
        frac_yx.append(len(yx) / (16 * 16))
    if not lin:
        continue
    rows.append((name, TRUE_DENS.get(name, 0), float(np.mean(lin)),
                 float(np.mean(pla)), float(np.mean(sph)),
                 float(np.mean(nz)), float(np.mean(frac_yx))))

print(f"  {'dataset':<18} {'true/fr':>8} {'strand':>7} {'sheet':>7} "
      f"{'ball':>7} {'|norm_z|':>9} {'yx cover':>9}  shape")
for r in sorted(rows, key=lambda r: r[1]):
    which = max((("strand", r[2]), ("sheet", r[3]), ("ball", r[4])),
                key=lambda p: p[1])[0]
    facing = ""
    if which == "sheet":
        facing = " lying flat" if r[5] > 0.6 else (
            " STANDING UP" if r[5] < 0.4 else " tilted")
    print(f"  {r[0]:<18} {r[1]:>8} {r[2]:>7.3f} {r[3]:>7.3f} "
          f"{r[4]:>7.3f} {r[5]:>9.3f} {r[6]:>9.3f}  {which}{facing}")

if len(rows) > 2:
    d = np.array([r[1] for r in rows], float)
    for lab, col in (("strand", 2), ("sheet", 3), ("ball", 4),
                     ("|norm_z|", 5), ("yx cover", 6)):
        v = np.array([r[col] for r in rows], float)
        print(f"  corr(true density, {lab:<9}) = "
              f"{float(np.corrcoef(d, v)[0, 1]):+.4f}")

print("\n  READ IT: strand + sheet + ball sum to 1 for each row, so the "
      "largest names the local shape. For a sheet, |norm_z| says which "
      "way it faces -- near 0 means it stands up and projects to a "
      "narrow band, which is what a strand also looks like on screen. "
      "`yx cover` is the fraction of the frame the cells occupy in "
      "projection: it is what the eye reads, and it cannot tell a "
      "strand from an edge-on sheet either.")
