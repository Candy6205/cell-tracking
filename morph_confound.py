"""morph_confound.py -- is morphology change an EDGE signal, or the
detection signal leaking through the negative class?

biohub-morph-cheap measured |d diam|/diam at 0.0268 across true GT edges
against 0.2423 across false candidate pairs, AUC 0.825, and I reported it
as morphology being an edge feature. The user did not believe it, and the
arithmetic says they are right to doubt:

  matched detections   diam median 8.83   (biohub-morph-probe step 2a)
  unmatched            diam median 6.87
  |8.83 - 6.87| / 8.83 = 0.222

against the 0.2423 actually observed on false pairs. Nearly identical. So
the likely story is not "a cell keeps its size from frame to frame" but
"the wrong partner is usually a SPURIOUS detection, and spurious
detections are systematically smaller" -- the step-2a detection-quality
signal showing up a second time, through how the negatives were drawn.

The test that separates them is to split the negative class:

  N-real   candidate IS GT-matched -- a real cell, just the wrong link
  N-fake   candidate is unmatched  -- a spurious detection

Separation on N-fake alone means detection quality. Separation on N-real
means genuine identity preservation, which is what an edge feature has to
be.

PART A, competition data: run the split and PRINT THE COUNTS. It is
expected to fail for lack of N-real -- sparse GT means the only matched
cell near a matched cell is nearly always its own successor, which is
what left morph_probe with three negatives. Reporting the size of the
problem is the point.

PART B, Zebrahub: dense GT, so nearly every detection matches and N-real
is plentiful. This is where the question is actually answerable. Size
needs masks and Zebrahub has none cached, but morph-cheap showed crop
intensity tracks the mask measurement at rho 0.965 while size does not
(0.713) -- so PART B uses INTENSITY, which carries the same confound
(matched detections are 31% brighter) and is therefore an equally valid
discriminator between the two stories.

CPU, no quota, no detector.
"""

SCRIPT_VERSION = "2026-08-25g-morph-confound"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

CODE_ROOTS = [
    "/kaggle/input/datasets/faustine649/cell-tracking/cell tracking",
    "/kaggle/input/datasets/faustine649/cell-tracking",
    "/kaggle/input/cell-tracking/cell tracking",
    "/kaggle/input/cell-tracking",
]
_code = next((c for c in CODE_ROOTS if Path(c, "viz_cells.py").exists()),
             None)
assert _code is not None, "attach faustine649/cell-tracking"
sys.path.insert(0, _code)
from viz_cells import read_geff, geff_path                     # noqa: E402

SCALE = np.array([1.625, 0.40625, 0.40625])
POOL = 4
MATCH_UM = 5.0
GATE_UM = 15.0
CROP_R = 3
UNET_THRESH = 0.15
MAX_CROPS = 8          # Part B: enough for counts, cheap enough to finish


def _pool_xy(vol, f=POOL):
    Z, Y, X = vol.shape
    Y2, X2 = (Y // f) * f, (X // f) * f
    v = vol[:, :Y2, :X2].astype(np.float32, copy=False)
    return v.reshape(Z, Y2 // f, f, X2 // f, f).mean(axis=(2, 4))


def _norm(p):
    lo = float(np.percentile(p, 50))
    hi = float(np.percentile(p, 99.5))
    return np.clip((p - lo) / (hi - lo + 1e-6), -0.5, 6.0).astype(np.float32)


def crop_mean(img, zyx, r=CROP_R):
    """Crop intensity mean. morph-cheap: rho 0.965 against the mask-based
    regionprops intensity_mean, so this stands in for it without masks."""
    Z, Y, X = img.shape
    z0, y0, x0 = (int(round(v)) for v in zyx)
    c = img[max(0, z0 - r):min(Z, z0 + r + 1),
            max(0, y0 - r):min(Y, y0 + r + 1),
            max(0, x0 - r):min(X, x0 + r + 1)]
    return float(c.mean()) if c.size else np.nan


def stratified(y, grp, v, tag):
    """y: 1 true edge. grp: 'real' or 'fake' for negatives. v: |delta|."""
    y = np.asarray(y)
    v = np.asarray(v, float)
    grp = np.asarray(grp)
    t = v[y == 1]
    print(f"\n  --- {tag} ---", flush=True)
    print(f"  true edges                    n={len(t):>5}  "
          f"median {np.median(t) if len(t) else float('nan'):.4f}",
          flush=True)
    for g, label in (("real", "N-real (both real cells, wrong link)"),
                     ("fake", "N-fake (partner is spurious)")):
        m = (y == 0) & (grp == g)
        f = v[m]
        if len(f) < 10:
            print(f"  {label:<29} n={len(f):>5}  TOO FEW to judge",
                  flush=True)
            continue
        sub = np.concatenate([t, f])
        lab = np.concatenate([np.ones(len(t)), np.zeros(len(f))])
        auc = roc_auc_score(lab, -sub)
        print(f"  {label:<29} n={len(f):>5}  median {np.median(f):.4f}  "
              f"AUC vs true {auc:.4f}", flush=True)


# ================= PART A: competition data ==============================
print("\n=== PART A: competition data (sparse GT) ===", flush=True)
hits = glob.glob("/kaggle/input/**/hoct_masks.npz", recursive=True)
CANDIROOT = ["/kaggle/input/biohub-cell-tracking-during-development",
             "/kaggle/input/competitions/"
             "biohub-cell-tracking-during-development"]
ROOT = next((p for p in CANDIROOT if Path(p, "train").exists()), None)
if not hits or ROOT is None:
    print("  masks or competition data missing -> Part A skipped; Part B "
          "is the one that answers the question anyway", flush=True)
else:
    from skimage.measure import regionprops_table
    z = np.load(hits[0])
    labels = z["labels"]
    POOLED = np.asarray(z["pooled"], float)
    NAME = str(z["name"][0])
    T = labels.shape[0]

    def _read_meta(zp):
        import json as _j
        with open(Path(zp) / "0" / "zarr.json") as f:
            return _j.load(f)

    def _load_volume(zp, t, m):
        try:
            import zarr
            return np.asarray(zarr.open(str(zp), mode="r")["0"][t])
        except Exception:
            import blosc2
            buf = blosc2.decompress(
                open(Path(zp) / "0" / "c" / str(t) / "0" / "0" / "0",
                     "rb").read())
            return np.frombuffer(
                buf, dtype=np.dtype(m["data_type"])).reshape(
                    tuple(m["shape"])[1:])

    zp = Path(ROOT) / "train" / f"{NAME}.zarr"
    mm = _read_meta(zp)
    images = np.stack([_norm(_pool_xy(_load_volume(zp, t, mm)))
                       for t in range(T)])
    diam, tv, pv = [], [], []
    for t in range(T):
        if labels[t].max() == 0:
            continue
        tb = regionprops_table(labels[t], intensity_image=images[t],
                               properties=("label", "centroid",
                                           "equivalent_diameter_area"))
        for i in range(len(tb["label"])):
            diam.append(float(tb["equivalent_diameter_area"][i]))
            tv.append(t)
            pv.append((float(tb["centroid-0"][i]),
                       float(tb["centroid-1"][i]),
                       float(tb["centroid-2"][i])))
    diam = np.asarray(diam)
    tv = np.asarray(tv, int)
    pv = np.asarray(pv)
    pum = pv * POOLED[None, :]

    gt = read_geff(geff_path(Path(ROOT), "train", NAME))
    g_t = np.asarray(gt["t"], int)
    g_pos = np.stack([np.asarray(gt[c], float)
                      for c in ("z", "y", "x")], axis=1) * SCALE
    row_of = {int(v): i for i, v in enumerate(gt["node_ids"])}
    gte = set()
    for a, b in gt["edges"]:
        ra, rb = row_of.get(int(a)), row_of.get(int(b))
        if ra is not None and rb is not None:
            gte.add((ra, rb))
    gbt = defaultdict(list)
    for i in range(len(g_t)):
        gbt[int(g_t[i])].append(i)
    mt = {}
    for t in range(T):
        idx = gbt.get(t, [])
        sel = np.where(tv == t)[0]
        if not idx or not len(sel):
            continue
        tr = cKDTree(g_pos[idx])
        d, gi = tr.query(pum[sel])
        for k, s in enumerate(sel):
            if d[k] <= MATCH_UM:
                mt[int(s)] = idx[int(gi[k])]

    y, grp, val = [], [], []
    for t in range(T - 1):
        a_sel = np.where(tv == t)[0]
        b_sel = np.where(tv == t + 1)[0]
        if not len(a_sel) or not len(b_sel):
            continue
        tr = cKDTree(pum[b_sel])
        for ai in a_sel:
            if int(ai) not in mt:
                continue
            for bj in tr.query_ball_point(pum[ai], GATE_UM):
                bi = int(b_sel[bj])
                is_e = bi in mt and (mt[int(ai)], mt[bi]) in gte
                y.append(1 if is_e else 0)
                grp.append("real" if bi in mt else "fake")
                val.append(abs(diam[bi] - diam[int(ai)])
                           / max(diam[int(ai)], 1e-6))
    stratified(y, grp, val, f"{NAME}, |d diam| / diam")
    print("\n  Sparse GT is why N-real is thin: near a matched cell the "
          "only other matched cell is nearly always its own successor.",
          flush=True)


# ================= PART B: Zebrahub, dense GT ============================
print("\n\n=== PART B: Zebrahub (DENSE GT) -- the answerable one ===",
      flush=True)
ZH = {f.stem[len("zhdet_"):]: f
      for f in Path("/kaggle/input").glob("*/**/zhdet_*.npz")}
CROPS = None
for c in Path("/kaggle/input").glob("*/**/crops"):
    CROPS = c
    break
print(f"  {len(ZH)} zhdet caches; crops at {CROPS}", flush=True)
if not ZH or CROPS is None:
    print("  attach biohub-zh-detect and biohub-zebrahub-extract -> "
          "Part B SKIPPED, and with it the whole question", flush=True)
else:
    Y, G, V = [], [], []
    done = 0
    for nm, f in sorted(ZH.items()):
        if done >= MAX_CROPS:
            break
        cf = CROPS / f"{nm}.npz"
        if not cf.exists():
            continue
        zc = np.load(cf)
        if "image" not in zc.files:
            continue
        img = zc["image"]
        zd = np.load(f)
        nodes = np.asarray(zd["nodes"], float)
        tid = np.asarray(zd["track_id"], np.int64)
        Tn = min(img.shape[0],
                 max(int(k[1:]) for k in zd.files if k.startswith("c")) + 1)
        imgs = {t: _norm(_pool_xy(np.asarray(img[t]))) for t in range(Tn)}
        POOLED_B = np.array([SCALE[0], SCALE[1] * POOL, SCALE[2] * POOL])

        det, dtv, dint = [], [], []
        for t in range(Tn):
            c = np.asarray(zd[f"c{t}"], float).reshape(-1, 3)
            s = np.asarray(zd[f"s{t}"], float).reshape(-1)
            c = c[s >= UNET_THRESH]
            for xyz in c:
                det.append(xyz)
                dtv.append(t)
                dint.append(crop_mean(imgs[t], xyz / np.array([1, POOL,
                                                               POOL])))
        if not det:
            continue
        det = np.asarray(det)
        dtv = np.asarray(dtv, int)
        dint = np.asarray(dint, float)
        dum = det * SCALE[None, :]

        gbt2 = defaultdict(list)
        for r in range(len(nodes)):
            gbt2[int(nodes[r, 0])].append(r)
        mt2 = {}
        for t in range(Tn):
            idx = gbt2.get(t, [])
            sel = np.where(dtv == t)[0]
            if not idx or not len(sel):
                continue
            tr = cKDTree(nodes[idx, 1:] * SCALE[None, :])
            d, gi = tr.query(dum[sel])
            for k, s_ in enumerate(sel):
                if d[k] <= MATCH_UM:
                    mt2[int(s_)] = idx[int(gi[k])]

        for t in range(Tn - 1):
            a_sel = np.where(dtv == t)[0]
            b_sel = np.where(dtv == t + 1)[0]
            if not len(a_sel) or not len(b_sel):
                continue
            tr = cKDTree(dum[b_sel])
            for ai in a_sel:
                if int(ai) not in mt2:
                    continue
                ka = int(tid[mt2[int(ai)]])
                for bj in tr.query_ball_point(dum[ai], GATE_UM):
                    bi = int(b_sel[bj])
                    is_e = (bi in mt2 and int(tid[mt2[bi]]) == ka)
                    if not np.isfinite(dint[int(ai)]) or \
                            not np.isfinite(dint[bi]):
                        continue
                    Y.append(1 if is_e else 0)
                    G.append("real" if bi in mt2 else "fake")
                    V.append(abs(dint[bi] - dint[int(ai)]))
        done += 1
        print(f"  {done}/{MAX_CROPS} {nm}: {len(mt2)}/{len(det)} "
              f"detections matched", flush=True)
        del zc, zd, img, imgs
    stratified(Y, G, V, f"Zebrahub x{done}, |d crop intensity|")
    print("\n  READ THIS: separation on N-fake but NOT on N-real means "
          "morphology is a DETECTION-quality signal that leaked into the "
          "edge test through how negatives were drawn -- morph-cheap's "
          "AUC 0.825 would then not be an edge feature at all. "
          "Separation on N-real is the identity signal an edge feature "
          "needs.", flush=True)

print("\nMORPH CONFOUND DONE", flush=True)
