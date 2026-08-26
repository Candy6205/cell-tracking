"""The MODE, not the median. And the uncensored distribution.

The user asked why merge.py reported a median. Fair: the claim being
tested is "misses pile up at one cell diameter, which is the merge
signature", and a pile-up is a MODE. The median is just the middle of
whatever shape is there.

And merge.py's binning started at MATCH_UM = 5.0, so everything below it
was already counted as a hit and never plotted. That censors the left
half of the distribution. A peak in the 5-6 um bin -- the first bin that
exists -- could be a real mode or could be the right tail of a
distribution whose true peak sits below 5 and is invisible.

So this plots the FULL distance distribution for every GT node, hit or
missed, in 0.5 um bins, and marks where the matcher's cut falls.

Two readings, and they say different things:

  one broad peak below 5 um with a thin tail crossing it
      -> the misses are ordinary matching near-misses, not a separate
         population. Fitting would not target them specifically.
  a SECOND peak around 7-10 um
      -> a distinct population sitting at neighbour-centre spacing,
         which is the merge signature and what a mixture fit addresses.
"""
import pathlib

import numpy as np
from scipy.spatial import cKDTree

SCALE = np.array([1.625, 0.40625, 0.40625])
UNET_THRESH = 0.15
MATCH_UM = 5.0
GT = pathlib.Path(r"C:\Users\48688\AppData\Local\Temp\gtx2")
DET = pathlib.Path(r"C:\Users\48688\AppData\Local\Temp\erout")

HOLDOUT = [
    "6bba_268e1230", "6bba_2312ac41", "6bba_283bf9f1", "6bba_3a1849c2",
    "6bba_3abfe10a", "6bba_5c824876", "6bba_61dd1e0d", "6bba_7af54fde",
    "6bba_7b5d3b2c", "6bba_afb141ff", "6bba_aeee7805", "6bba_c27cba08",
]


def frame_keys(files, prefix):
    return [k for k in files
            if k.startswith(prefix) and k[len(prefix):].isdigit()]


allsome = []
nn_spacing = []
for name in HOLDOUT:
    gf, df = GT / f"gt_{name}.npz", DET / f"detcache_{name}.npz"
    if not (gf.exists() and df.exists()):
        continue
    g = np.load(gf)
    d = np.load(df)
    T = max(int(k[1:]) for k in frame_keys(d.files, "c")) + 1
    g_t = np.asarray(g["t"], int)
    g_zyx = np.stack([np.asarray(g[c], float)
                      for c in ("z", "y", "x")], axis=1)
    for t in range(T):
        gi = np.where(g_t == t)[0]
        if not len(gi):
            continue
        c = np.asarray(d[f"c{t}"], float).reshape(-1, 3)
        s = np.asarray(d[f"s{t}"], float).reshape(-1)
        keep = c[s >= UNET_THRESH] * SCALE[None, :]
        if len(keep) < 2:
            continue
        gu = g_zyx[gi] * SCALE[None, :]
        dist, _ = cKDTree(keep).query(gu)
        allsome.append(dist)
        # spacing between SURVIVING peaks: the scale a merge would sit at
        dd, _ = cKDTree(keep).query(keep, k=2)
        nn_spacing.append(dd[:, 1])

D = np.concatenate(allsome)
NN = np.concatenate(nn_spacing)

print(f"  {len(D):,} annotated GT nodes")
print(f"  distance to the nearest SURVIVING peak, ALL of them\n")
print(f"  {'um':>10}  {'count':>7}  {'%':>6}")
BIN = 0.5
edges = np.arange(0, 20.0 + BIN, BIN)
h, _ = np.histogram(D, bins=edges)
top = h.max()
for k in range(len(h)):
    lo, hi = edges[k], edges[k + 1]
    bar = "#" * int(round(40.0 * h[k] / top)) if top else ""
    mark = "  <== MATCH_UM, everything右 is a MISS" \
        if abs(lo - MATCH_UM) < 1e-9 else ""
    print(f"  {lo:4.1f}-{hi:<4.1f}  {h[k]:>7,}  {100.0 * h[k] / len(D):5.1f}"
          f"  {bar}{mark}")
over = (D > 20).sum()
print(f"  {'>20':>10}  {over:>7,}  {100.0 * over / len(D):5.1f}")

mode_bin = int(np.argmax(h))
print(f"\n  MODE of the whole distribution: "
      f"{edges[mode_bin]:.1f}-{edges[mode_bin + 1]:.1f} um")
miss = D[D > MATCH_UM]
hm, _ = np.histogram(miss, bins=edges)
mb = int(np.argmax(hm))
print(f"  MODE of the MISSES only:       "
      f"{edges[mb]:.1f}-{edges[mb + 1]:.1f} um")
print(f"  median of the misses:          {np.median(miss):.2f} um"
      f"   (what merge.py reported)")

print(f"\n  For scale -- spacing between SURVIVING peaks "
      f"(nearest neighbour):")
for q in (10, 25, 50, 75, 90):
    print(f"    p{q:<3} {np.percentile(NN, q):6.2f} um")
print(f"  A merged pair's surviving peak should sit at roughly this")
print(f"  spacing from the cell it absorbed. Compare it to the mode of")
print(f"  the misses: if they agree, merge. If the misses sit well")
print(f"  INSIDE the typical spacing, they are matching near-misses.")
