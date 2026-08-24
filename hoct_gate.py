"""hoct_gate.py -- is the host lab's higher-order tracker worth
integrating, or did the domain gap eat it?

Forum /726521: Jordao Bragantini, posting as HOST and a co-author, points
at HOCT (arxiv 2607.11754, github.com/royerlab/hoct) -- the Royer lab,
i.e. the lab that produced this competition's data. Edge-centric
transformer, five-frame window, pretrained `general_v0`. The paper's two
stated problems are exactly ours: cell divisions entangling lineage paths,
and "edges sharing a node have near-random label agreement" -- which is
what ranker_scale measured, per-dataset ranker AUC FLAT against density
while adj_J slides hard.

Against it: trained on Cell Tracking Challenge plus a bacteria benchmark,
NOT light-sheet embryos, and the training code is withheld so it cannot be
fine-tuned. The one recorded attempt (rank-24 competitor, /728551) fed it
UNIFORM-BALL masks and it over-linked into spurious forks -- consistent
with Bragantini's own caveat that results depend on mask quality.

This is a GATE, not an integration, and a RECONNAISSANCE run: the README
publishes no function signatures, so the API is introspected and printed
before it is used, and every extraction attempt announces what it tried.
A wrong guess still leaves the API documented rather than wasting the run.

Masks come from a WATERSHED on the pooled detection heatmap rather than
spheres, for two reasons: it keeps real cell shape, which is what the
model's geometric prior needs, and the pooled grid is ISOTROPIC by
construction -- POOL=4 on Y/X turns (1.625, 0.40625, 0.40625) into
(1.625, 1.625, 1.625) -- so the 4x anisotropy mismatch costs nothing to
remove.

Read-out: HOCT's edge score against our GT edges, versus a negative-
distance baseline on the SAME edges. Near 0.5, or under the baseline, and
the domain gap ate it -- stop before packaging weights for offline
submission. Comfortably above, and the feature route is worth building:
HOCT's score as one more column in OUR in-domain ranker, which sidesteps
the missing training code entirely because what gets recalibrated is how
much to trust it, not the model.

One dataset, 30 frames. Internet ON -- this cannot be a submission path
as written.
"""
import gc
import glob
import json
import pathlib
import pickle
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

SCRIPT_VERSION = "2026-08-24f-hoct-gate"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

DEVICE = "cpu"
if torch.cuda.is_available():
    try:
        _p = nn.Conv3d(1, 1, 3).to("cuda")
        _ = _p(torch.zeros(1, 1, 4, 4, 4, device="cuda")).cpu()
        DEVICE = "cuda"
        del _p
    except Exception as e:                                     # noqa: BLE001
        print("GPU present but conv3d unusable -> CPU:", str(e)[:80])

# ---- detect-side constants (xl-repro verbatim) ------------------------
SCALE = np.array([1.625, 0.40625, 0.40625])
POOL = 4
REPAIR = True
CAND_THR = 0.05
UNET_THRESH = 0.15
NMS_UM = 4.0
DETECT_THRESH = min(UNET_THRESH, CAND_THR) if REPAIR else UNET_THRESH

# ---- pipeline constants (ranker-diag verbatim: the 0.9345 config) -----
GATE_UM = 10.0
MAX_LINK_UM = 10.0
TIGHT_UM = 6.0
LINK_SIM_W = 2.0
SHORT_MIN = 6
LINEFIT_WEIGHT = 0.8
LINEFIT_WINDOW = 2
GAP1_GATE_UM = 9.0
GAP1_SNAP_UM = 3.2
GAP1_MIN_CAND_SCORE = 0.10
GAP1_CAP_FRAC = 0.003
DIV_PARENT_UM = 12.0
DIV_SISTER_UM = 15.0
DIV_CHILD_UM = 10.0
DIV_DIVERGE_UM = 2.25
DIV_W_SISTER = 0.15
DIV_FRAME_CAP = 0.0076
DIV_GLOBAL_CAP = 0.00375
ILP_FLOOR = 0.10                            # candidates fed to the ILP
ILP_DISAPP = 1.5
ILP_DIV_W = 0.55   # fine sweep argmax
ILP_WINDOW = 25    # wincal: 0.9382 vs global 0.9368, 2.08x faster

CANDIROOT = ["/kaggle/input/biohub-cell-tracking-during-development",
             "/kaggle/input/competitions/biohub-cell-tracking-during-development"]
ROOT = next((p for p in CANDIROOT if Path(p, "train").exists()), None)
assert ROOT is not None, "competition not attached"
TEST_DIR = Path(ROOT) / "test"

# ---- xl detect stack (xl-repro verbatim) ------------------------------

def _find_weight(name):
    cands = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert cands, f"{name} not found"
    return cands[0]

def _block(ci, co):
    return nn.Sequential(nn.Conv3d(ci, co, 3, padding=1), nn.BatchNorm3d(co),
                         nn.ReLU(inplace=True),
                         nn.Conv3d(co, co, 3, padding=1), nn.BatchNorm3d(co),
                         nn.ReLU(inplace=True))

class UNet3D(nn.Module):
    def __init__(self, base=24):
        super().__init__()
        self.e1 = _block(1, base)
        self.e2 = _block(base, base * 2)
        self.e3 = _block(base * 2, base * 4)
        self.pool = nn.MaxPool3d(2)
        self.bott = _block(base * 4, base * 8)
        self.u3 = nn.ConvTranspose3d(base * 8, base * 4, 2, stride=2)
        self.d3 = _block(base * 8, base * 4)
        self.u2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.d2 = _block(base * 4, base * 2)
        self.u1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.d1 = _block(base * 2, base)
        self.out = nn.Conv3d(base, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.bott(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


def _load_model(name):
    ck = torch.load(_find_weight(name), map_location=DEVICE)
    m = UNet3D(base=ck.get("base", 24)).to(DEVICE)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    print("loaded", name, "| val_recall", ck.get("val_recall"), flush=True)
    return m

MODELS = [_load_model("unet3d_bright.pt"), _load_model("unet3d_traintophat.pt")]

def read_array_meta(zp):
    with open(Path(zp) / "0" / "zarr.json") as f:
        m = json.load(f)
    return dict(shape=tuple(m["shape"]), dtype=np.dtype(m["data_type"]))

_ZC = {}

def load_volume(zp, t, meta=None):
    try:
        import zarr
        k = str(zp)
        if k not in _ZC:
            _ZC[k] = zarr.open(k, mode="r")["0"]
        return np.asarray(_ZC[k][t])
    except Exception:
        import blosc2
        if meta is None:
            meta = read_array_meta(zp)
        buf = blosc2.decompress(
            open(Path(zp) / "0" / "c" / str(t) / "0" / "0" / "0", "rb").read())
        return np.frombuffer(buf, dtype=meta["dtype"]).reshape(meta["shape"][1:])

def pool_xy(vol, f=POOL):
    Z, Y, X = vol.shape
    Y2, X2 = (Y // f) * f, (X // f) * f
    v = vol[:, :Y2, :X2].astype(np.float32, copy=False)
    return v.reshape(Z, Y2 // f, f, X2 // f, f).mean(axis=(2, 4))

def pool_norm(vol, preproc=""):
    p = pool_xy(vol)
    if preproc == "tophat":
        from scipy.ndimage import grey_opening
        p = np.clip(p - grey_opening(p, size=(1, 7, 7)), 0.0, None)
    lo = float(np.percentile(p, 50))
    hi = float(np.percentile(p, 99.5))
    return np.clip((p - lo) / (hi - lo + 1e-6), -0.5, 6.0).astype(np.float32)

def _refine(vol, zyx, rz=2, ryx=5):
    Z, Y, X = vol.shape
    z, y, x = (int(round(v)) for v in zyx)
    z0, z1 = max(0, z - rz), min(Z, z + rz + 1)
    y0, y1 = max(0, y - ryx), min(Y, y + ryx + 1)
    x0, x1 = max(0, x - ryx), min(X, x + ryx + 1)
    crop = vol[z0:z1, y0:y1, x0:x1].astype(np.float32)
    bg = float(crop.min())
    w = np.clip(crop - bg, 0, None)
    s = float(w.sum())
    if s <= 0:
        return np.array([z, y, x], float), 0.0
    zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1]
    return (np.array([(zz * w).sum(), (yy * w).sum(), (xx * w).sum()]) / s,
            float(crop.max() - bg))

def _physical_nms(coords, scores, radius_um, scale=SCALE):
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
    return coords[keep], scores[keep]

_BRANCHES = [([MODELS[0]], [1.0], ""), ([MODELS[1]], [1.0], "tophat"),
             ([_load_model("unet3d_v2_tophat_b32.pt")], [1.0], "tophat")]
_FLIP_VIEWS = (0, 1, 2, 3)

def _flip(x, i):
    if i & 1:
        x = np.flip(x, -2)
    if i & 2:
        x = np.flip(x, -1)
    return x

def detect(vol):
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
        return np.zeros((0, 3)), np.zeros(0)
    sc = h[pk[:, 0], pk[:, 1], pk[:, 2]].astype(float)
    coords = pk.astype(float)
    coords[:, 1] = coords[:, 1] * POOL + (POOL - 1) / 2
    coords[:, 2] = coords[:, 2] * POOL + (POOL - 1) / 2
    ref = np.array([_refine(vol, c)[0] for c in coords])
    return _physical_nms(ref, sc, NMS_UM)


# ------------------------------------------------------------- graph ops

def avail_T(zp):
    meta = read_array_meta(zp)
    T = meta["shape"][0]
    present = [t for t in range(T)
               if (Path(zp) / "0" / "c" / str(t) / "0" / "0" / "0").exists()]
    return max(present) + 1 if present else 0

# ---------------- cache xl detections for the rest of train --------------
# The census settled it: 151 GT divisions across all 199 train datasets, and
# only 31 of them sit in the 44 datasets we have detection caches for. This
# caches the other 155 with the SAME xl detector (same three branches, same
# flip views, same thresholds), so the pools can simply be merged.
#
# Two payoffs, both independent of the division work:
#   - ~127 usable in-domain divisions instead of 26 (the anchor set that the
#     Zebrahub / synthetic question needs in order to be answerable at all)
#   - the edge ranker's training pool goes from 32 datasets to 187
#
# Output format matches biohub-edge-ranker exactly: one detcache_<name>.npz
# per dataset with keys c{t} (coords Nx3) and s{t} (scores N), so the
# existing `CACHES` glob picks both up with no code change.

TRAIN_DIR = Path(ROOT) / "train"


# ---- the HOCT gate -----------------------------------------------------
# Forum /726521: Jordao Bragantini, posting as HOST and a co-author, points
# at higher-order cell tracking -- arxiv 2607.11754, code and a pretrained
# `general_v0` at github.com/royerlab/hoct. Same lab as this competition's
# data. Edge-centric transformer, 5-frame window, and the paper's two
# stated problems are OUR two: "cell divisions entangle distinct lineage
# paths" and "edges sharing a node have near-random label agreement" --
# the second being exactly what ranker_scale measured (per-dataset ranker
# AUC FLAT against density while adj_J slides).
#
# BUT: it was trained on Cell Tracking Challenge + a bacteria benchmark,
# NOT light-sheet embryos, and the training code is withheld so it cannot
# be fine-tuned. The one recorded attempt (rank-24 competitor, /728551)
# fed it UNIFORM-BALL masks and it over-linked into spurious forks.
#
# This run is a GATE, not an integration. Two questions, in order:
#   1. what is the actual Python API?  The README gives no signatures, so
#      everything below is introspected at runtime and PRINTED before it
#      is used. A wrong guess still leaves the API documented.
#   2. do HOCT's edge scores beat a distance baseline against our GT?
#      Near 0.5 and the domain gap ate everything -- stop before any
#      packaging work.
#
# Masks come from a WATERSHED on the pooled detection heatmap, not from
# spheres. Two reasons: it respects real cell shape, which is what the
# model's geometric prior needs, and the pooled grid is ISOTROPIC by
# construction -- POOL=4 on Y/X turns (1.625, 0.40625, 0.40625) into
# (1.625, 1.625, 1.625) -- so the 4x anisotropy mismatch disappears for
# free.

import os                                                      # noqa: E402
import inspect                                                 # noqa: E402
import traceback                                               # noqa: E402

r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hoct"],
                   capture_output=True, text=True)
print(f"pip install hoct exit {r.returncode}", flush=True)
if r.returncode != 0:
    print(r.stdout[-2000:], flush=True)
    print(r.stderr[-2000:], flush=True)

import hoct                                                    # noqa: E402
print(f"hoct {getattr(hoct, '__version__', '?')} at {hoct.__file__}",
      flush=True)

print("\n=== API SURFACE (the README publishes no signatures) ===",
      flush=True)
print("dir(hoct):", [n for n in dir(hoct) if not n.startswith("_")],
      flush=True)
for nm in ("load_model", "predict"):
    fn = getattr(hoct, nm, None)
    if fn is None:
        print(f"  {nm}: ABSENT", flush=True)
        continue
    try:
        print(f"  {nm}{inspect.signature(fn)}", flush=True)
    except (ValueError, TypeError) as exc:
        print(f"  {nm}: signature unavailable ({exc})", flush=True)
    doc = (inspect.getdoc(fn) or "").strip()
    if doc:
        print("    " + doc[:900].replace("\n", "\n    "), flush=True)

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
from viz_cells import read_geff, geff_path                      # noqa: E402
from skimage.segmentation import watershed                      # noqa: E402
from skimage.measure import label as cc_label                   # noqa: E402
from sklearn.metrics import roc_auc_score                       # noqa: E402

NAME = "6bba_268e1230"       # one dataset; this is recon, not measurement
N_FRAMES = 30
MASK_FLOOR = 0.05            # heatmap value bounding a cell's region
MATCH_UM = 5.0
POOLED = np.array([SCALE[0], SCALE[1] * POOL, SCALE[2] * POOL])
print(f"\n  pooled voxel spacing {tuple(POOLED)} um -- isotropic: "
      f"{np.allclose(POOLED, POOLED[0])}", flush=True)


def heatmap(vol):
    """detect()'s averaged heatmap, returned before peak-finding."""
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
            acc = _flip(lg, i) if acc is None else acc + _flip(lg, i)
        hs.append(1.0 / (1.0 + np.exp(-(acc / len(_FLIP_VIEWS)))))
    return np.mean(hs, axis=0)


print(f"\n=== BUILDING WATERSHED MASKS: {NAME}, {N_FRAMES} frames ===",
      flush=True)
zp = TRAIN_DIR / f"{NAME}.zarr"
assert zp.exists(), f"no zarr at {zp}"
meta = read_array_meta(zp)
T = min(N_FRAMES, avail_T(zp))
labels_t, centres_t = [], []
t0 = time.time()
for t in range(T):
    h = heatmap(load_volume(zp, t, meta))
    pk = peak_local_max(h, min_distance=1, threshold_abs=UNET_THRESH,
                        exclude_border=False)
    mk = np.zeros(h.shape, np.int32)
    for i, p in enumerate(pk):
        mk[p[0], p[1], p[2]] = i + 1
    lab = watershed(-h, mk, mask=h >= MASK_FLOOR).astype(np.int32)
    labels_t.append(lab)
    # a region's centre in POOLED voxels, so it maps to um by POOLED
    centres_t.append(pk.astype(float))
    if t == 0:
        print(f"  frame 0: {len(pk)} peaks -> {lab.max()} labels, "
              f"{int((lab > 0).sum())} voxels covered, mean region "
              f"{(lab > 0).sum() / max(lab.max(), 1):.1f} voxels",
              flush=True)
labels = np.stack(labels_t)
print(f"  labels {labels.shape} {labels.dtype}, "
      f"{time.time() - t0:.0f}s", flush=True)

print("\n=== RUNNING HOCT ===", flush=True)
graph = None
try:
    model = hoct.load_model(device="cuda" if torch.cuda.is_available()
                            else "cpu")
    print(f"  model loaded: {type(model)}", flush=True)
    for env in ("HOCT_CACHE_DIR", "XDG_CACHE_HOME"):
        print(f"  {env}={os.environ.get(env)}", flush=True)
    r2 = subprocess.run(["find", "/root", "/tmp", "-name", "*.pt",
                         "-size", "+1M"], capture_output=True, text=True)
    print("  cached weights (needed later for OFFLINE packaging):",
          flush=True)
    print("   ", (r2.stdout or "(none found)").strip()[:600], flush=True)

    sig = inspect.signature(hoct.predict)
    print(f"  predict{sig}", flush=True)
    kw = {}
    for cand in ("full_graph", "return_candidates", "solve", "full"):
        if cand in sig.parameters:
            kw[cand] = (False if cand == "solve" else True)
    print(f"  calling predict with extra kwargs {kw}", flush=True)
    graph = hoct.predict(model, labels=labels, **kw)
    print(f"  returned {type(graph)}", flush=True)
    print("  attributes:", [n for n in dir(graph)
                            if not n.startswith("_")][:40], flush=True)
except Exception:
    print("  HOCT CALL FAILED -- the API surface above is still the "
          "deliverable of this run:", flush=True)
    traceback.print_exc()

if graph is None:
    print("\nHOCT GATE: NO GRAPH. Read the signatures above and retry.",
          flush=True)
    print("\nHOCT GATE DONE", flush=True)
    raise SystemExit(0)

# ---- score whatever came back against our ground truth ------------------
print("\n=== GATE: HOCT edge score vs a distance baseline, on OUR GT ===",
      flush=True)
gt = read_geff(geff_path(Path(ROOT), "train", NAME))
g_t = np.asarray(gt["t"], int)
g_pos = np.stack([np.asarray(gt[c], float)
                  for c in ("z", "y", "x")], axis=1) * SCALE
row_of = {int(v): i for i, v in enumerate(gt["node_ids"])}
gt_edges = set()
for a, b in gt["edges"]:
    ra, rb = row_of.get(int(a)), row_of.get(int(b))
    if ra is not None and rb is not None:
        gt_edges.add((ra, rb))
gt_by_t = defaultdict(list)
for i in range(len(g_t)):
    gt_by_t[int(g_t[i])].append(i)

# our detections in um, per frame, and their GT match
match = {}
for t in range(T):
    idx = gt_by_t.get(t, [])
    if not idx or not len(centres_t[t]):
        continue
    tree = cKDTree(g_pos[idx])
    dd, gi = tree.query(centres_t[t] * POOLED[None, :])
    for di in range(len(centres_t[t])):
        if dd[di] <= MATCH_UM:
            match[(t, di)] = idx[int(gi[di])]

print(f"  {len(match)} of our detections matched a GT node "
      f"(<= {MATCH_UM} um)", flush=True)


def extract(g):
    """(nodes_tzyx, edges, scores) out of whatever HOCT returned.

    The README documents only `to_geff`, so try that first -- it lands in
    a format read_geff already handles -- then fall back to networkx and
    tracksdata shapes. Every attempt says what it tried."""
    try:
        p = "/kaggle/working/hoct_out.geff"
        g.to_geff(p)
        d = read_geff(p)
        nodes = np.stack([np.asarray(d[c], float)
                          for c in ("t", "z", "y", "x")], axis=1)
        rid = {int(v): i for i, v in enumerate(d["node_ids"])}
        ed = [(rid[int(a)], rid[int(b)]) for a, b in d["edges"]
              if int(a) in rid and int(b) in rid]
        print(f"  to_geff worked: {len(nodes)} nodes, {len(ed)} edges",
              flush=True)
        print("  (geff carries no edge score -- solved edges only, so the "
              "AUC below uses a BINARY label as the score and is a "
              "sanity check, not the gate)", flush=True)
        return nodes, ed, None
    except Exception as exc:
        print(f"  to_geff route failed: {exc!r}", flush=True)
    try:
        import networkx as nx
        if isinstance(g, nx.Graph) or hasattr(g, "nodes"):
            nodes, order = [], {}
            for i, (n, a) in enumerate(g.nodes(data=True)):
                order[n] = i
                nodes.append([a.get("t", 0), a.get("z", 0),
                              a.get("y", 0), a.get("x", 0)])
            ed, sc = [], []
            for a, b, at in g.edges(data=True):
                ed.append((order[a], order[b]))
                s = next((at[k] for k in ("score", "prob", "weight",
                                          "edge_prob", "logit")
                          if k in at), None)
                sc.append(s)
            print(f"  networkx route: {len(nodes)} nodes, {len(ed)} edges, "
                  f"score keys present: {sc[0] is not None}", flush=True)
            return np.asarray(nodes, float), ed, (
                None if sc and sc[0] is None else np.asarray(sc, float))
    except Exception as exc:
        print(f"  networkx route failed: {exc!r}", flush=True)
    print("  NO ROUTE WORKED. The signatures and attribute list above are "
          "what this run buys; write the extractor against them.",
          flush=True)
    return None, None, None


nodes, ed, sc = extract(graph)
if nodes is None or not ed:
    print("\nHOCT GATE: could not read the graph. Not a verdict on the "
          "model.", flush=True)
    print("\nHOCT GATE DONE", flush=True)
    raise SystemExit(0)

# map HOCT's nodes to GT by coordinate. Its coordinates are in POOLED
# voxels if it echoed our label grid back; try that first and report the
# match rate so a wrong assumption is visible rather than silent.
best = None
for tag, scl in (("pooled voxels", POOLED), ("already um", np.ones(3))):
    hit = {}
    for i, row in enumerate(nodes):
        t = int(round(row[0]))
        idx = gt_by_t.get(t, [])
        if not idx:
            continue
        d = np.sqrt((((g_pos[idx] - row[1:] * scl) ** 2)).sum(1))
        j = int(np.argmin(d))
        if d[j] <= MATCH_UM:
            hit[i] = idx[j]
    print(f"  assuming HOCT coords are {tag}: {len(hit)}/{len(nodes)} "
          f"nodes matched GT", flush=True)
    if best is None or len(hit) > len(best[1]):
        best = (tag, hit)
tag, hit = best
print(f"  -> using '{tag}' ({len(hit)} matched)", flush=True)

y, s_h, s_d = [], [], []
for k, (a, b) in enumerate(ed):
    if a not in hit or b not in hit:
        continue
    y.append(1 if (hit[a], hit[b]) in gt_edges else 0)
    s_h.append(float(sc[k]) if sc is not None else 1.0)
    s_d.append(-float(np.linalg.norm(
        (nodes[b][1:] - nodes[a][1:]) *
        (POOLED if tag == "pooled voxels" else np.ones(3)))))
y = np.asarray(y)
print(f"\n  {len(y)} HOCT edges with both endpoints matched to GT; "
      f"{int(y.sum())} of them are real GT edges "
      f"({y.mean() if len(y) else float('nan'):.3f})", flush=True)
if len(y) > 20 and 0 < y.sum() < len(y):
    if sc is not None:
        print(f"  HOCT score      AUC {roc_auc_score(y, s_h):.4f}",
              flush=True)
    else:
        print("  HOCT score      unavailable (solved edges only)",
              flush=True)
    print(f"  -distance base  AUC {roc_auc_score(y, s_d):.4f}", flush=True)
    print("\n  GATE: near 0.5 means the domain gap ate it -- stop before "
          "any packaging work. Comfortably above the distance baseline "
          "means the feature route (HOCT score as one column in OUR "
          "in-domain ranker) is worth building.", flush=True)
else:
    print("  too few usable edges to judge", flush=True)

print("\nHOCT GATE DONE", flush=True)
