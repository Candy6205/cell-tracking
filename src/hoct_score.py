"""hoct_score.py -- the HOCT gate at last, on CPU, for free.

Four GPU attempts went into this API. Each one bought a fact and none
bought the gate number, so v4's real job was to persist the watershed
masks -- the detector is the only GPU-bound step -- and it did. This
kernel attaches that output and needs no accelerator at all.

Everything the earlier runs had to guess is now known:

  predict(model, *, graph=None, labels=None, ..., scale, window_size=5,
          return_solution=True)
      return_solution=False returns None. The scores land on a graph the
      CALLER owns, so build it, hold it, and read it afterwards.

  create_graph(labels, *, distance_threshold, n_neighbors, delta_t,
               scale=None, images=None, ...)
      v4 passed none of the three required keyword-only arguments.

  scale must have FOUR elements, (t, z, y, x). v2 passed three.

  edge_attr_keys() is a METHOD, and the score is called **`similarity`**
      -- not edge_prob, prob, score, weight or logit, all of which
      earlier runs guessed. That is why every previous gate fell back to
      the solved graph, where the only "score" is a constant 1 and the
      AUC has no negatives.

The gate: HOCT's `similarity` against our GT edges, versus a negative-
distance baseline on the SAME candidate edges. Near 0.5, or under the
baseline, and the domain gap ate it -- HOCT trained on Cell Tracking
Challenge plus a bacteria benchmark, not light-sheet embryos, and its
training code is withheld so it cannot be fine-tuned.

Worth noting whatever the verdict: `create_graph` computes per-node
morphology from the masks -- equivalent_diameter_area, inertia_tensor,
intensity min/max/mean/std, border_dist. Our pipeline has NONE of that;
every feature it owns is a coordinate or a detection score. "The mother
got smaller just before dividing" is not expressible anywhere in our
code and is sitting right there in this graph.
"""

SCRIPT_VERSION = "2026-08-25c-hoct-score"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

import glob
import pathlib
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

# v1 died here: a bare `pip install hoct` upgraded numpy and scipy's
# compiled _ckdtree then refused to import against the new ABI. This is
# the toolchain trap the other kernels dodge with --no-deps ("pip cannot
# half-upgrade numpy"). hoct_gate got away with it only because the
# inherited preamble imported numpy and scipy BEFORE installing.
# Two guards: import first, and pin numpy on the install line itself.
NP_VER = np.__version__
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hoct",
                    f"numpy=={NP_VER}"], capture_output=True, text=True)
print(f"pip install hoct (numpy pinned {NP_VER}) exit {r.returncode}",
      flush=True)
if r.returncode != 0:
    print(r.stderr[-1500:], flush=True)
assert np.__version__ == NP_VER,     f"numpy moved {NP_VER} -> {np.__version__} despite the pin"

import hoct                                                    # noqa: E402
from hoct.features.graph import create_graph                   # noqa: E402

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
MATCH_UM = 5.0
DIST_UM = 15.0
N_NEIGH = 5
DELTA_T = 3

CANDIROOT = ["/kaggle/input/biohub-cell-tracking-during-development",
             "/kaggle/input/competitions/"
             "biohub-cell-tracking-during-development"]
ROOT = next((p for p in CANDIROOT if Path(p, "train").exists()), None)
assert ROOT is not None, "competition not attached (needed for the GT geff)"

hits = glob.glob("/kaggle/input/**/hoct_masks.npz", recursive=True)
assert hits, "attach biohub-hoct-gate for hoct_masks.npz"
z = np.load(hits[0])
labels = z["labels"]
centres = z["centres"]
offsets = z["offsets"]
POOLED = np.asarray(z["pooled"], float)
NAME = str(z["name"][0])
T = labels.shape[0]
print(f"masks {labels.shape} {labels.dtype} from {hits[0]}", flush=True)
print(f"  dataset {NAME}, {T} frames, pooled spacing {tuple(POOLED)} um",
      flush=True)


# ---- the images v2 forgot -----------------------------------------------
# v2 passed labels ONLY, and the node features came back with
# intensity_min/max/mean/std all exactly 0.0. HOCT ran blind to image
# content, so "loses to a distance baseline by 0.24 AUC" described a
# blinded model, not the model. create_graph takes `images` the same shape
# as `labels`; pool the raw volumes onto the same grid. Pure numpy.
POOL = 4


def _pool_xy(vol, f=POOL):
    Z, Y, X = vol.shape
    Y2, X2 = (Y // f) * f, (X // f) * f
    v = vol[:, :Y2, :X2].astype(np.float32, copy=False)
    return v.reshape(Z, Y2 // f, f, X2 // f, f).mean(axis=(2, 4))


def _norm(p):
    lo = float(np.percentile(p, 50))
    hi = float(np.percentile(p, 99.5))
    return np.clip((p - lo) / (hi - lo + 1e-6), -0.5, 6.0).astype(np.float32)


def _read_meta(zp):
    import json as _j
    with open(Path(zp) / "0" / "zarr.json") as f:
        m = _j.load(f)
    return dict(shape=tuple(m["shape"]), dtype=np.dtype(m["data_type"]))


def _load_volume(zp, t, meta):
    try:
        import zarr
        return np.asarray(zarr.open(str(zp), mode="r")["0"][t])
    except Exception:
        import blosc2
        buf = blosc2.decompress(
            open(Path(zp) / "0" / "c" / str(t) / "0" / "0" / "0",
                 "rb").read())
        return np.frombuffer(buf, dtype=meta["dtype"]).reshape(
            meta["shape"][1:])


images = None
try:
    _zp = Path(ROOT) / "train" / f"{NAME}.zarr"
    _meta = _read_meta(_zp)
    images = np.stack([_norm(_pool_xy(_load_volume(_zp, t, _meta)))
                       for t in range(T)])
    assert images.shape == labels.shape, \
        f"images {images.shape} != labels {labels.shape}"
    print(f"  images {images.shape} pooled and normalised, range "
          f"[{images.min():.3f}, {images.max():.3f}]", flush=True)
except Exception:
    images = None
    print("  WARNING could not build images -> running BLIND again, which "
          "reproduces v2 rather than testing anything:", flush=True)
    traceback.print_exc()


# ---- build the candidate graph, then score it ---------------------------
print("\n=== HOCT on CPU ===", flush=True)
model = hoct.load_model(device="cpu")
graph = create_graph(labels,
                     images=images,
                     distance_threshold=DIST_UM,
                     n_neighbors=N_NEIGH,
                     delta_t=DELTA_T,
                     scale=(1.0,) + tuple(float(v) for v in POOLED))
print(f"  candidate graph: {type(graph).__name__}", flush=True)
ek = list(graph.edge_attr_keys())
print(f"  edge_attr_keys before predict: {ek}", flush=True)
hoct.predict(model, graph=graph, return_solution=False)
ek = list(graph.edge_attr_keys())
print(f"  edge_attr_keys after  predict: {ek}", flush=True)

SCORE_KEY = "similarity"
if SCORE_KEY not in ek:
    alt = [k for k in ek if k not in ("distance", "delta_t", "solution")]
    print(f"  WARNING '{SCORE_KEY}' absent -> falling back to {alt[:1]}; "
          f"if that list is empty the run below reports nothing usable "
          f"and the key must be read off the list above", flush=True)
    SCORE_KEY = alt[0] if alt else None
assert SCORE_KEY, f"no score-like edge attribute among {ek}"

na = graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
nid = [int(v) for v in na["node_id"].to_list()]
order = {v: i for i, v in enumerate(nid)}
nodes = np.stack([np.asarray(na[c].to_list(), float)
                  for c in ("t", "z", "y", "x")], axis=1)
ea = graph.edge_attrs(attr_keys=["source_id", "target_id", SCORE_KEY,
                                 "distance"])
src = [int(v) for v in ea["source_id"].to_list()]
tgt = [int(v) for v in ea["target_id"].to_list()]
sim = np.asarray(ea[SCORE_KEY].to_list(), float)
dist = np.asarray(ea["distance"].to_list(), float)
print(f"  {len(nodes)} nodes, {len(src)} candidate edges, "
      f"{SCORE_KEY} in [{sim.min():.4f}, {sim.max():.4f}]", flush=True)


# ---- our ground truth ---------------------------------------------------
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
gt_by_t = {}
for i in range(len(g_t)):
    gt_by_t.setdefault(int(g_t[i]), []).append(i)

hit = {}
for i, row in enumerate(nodes):
    idx = gt_by_t.get(int(round(row[0])), [])
    if not idx:
        continue
    d = np.sqrt((((g_pos[idx] - row[1:] * POOLED) ** 2)).sum(1))
    j = int(np.argmin(d))
    if d[j] <= MATCH_UM:
        hit[i] = idx[j]
print(f"  {len(hit)}/{len(nodes)} HOCT nodes matched a GT node "
      f"(<= {MATCH_UM} um)", flush=True)

y, s_h, s_d = [], [], []
for k in range(len(src)):
    a, b = order.get(src[k]), order.get(tgt[k])
    if a is None or b is None or a not in hit or b not in hit:
        continue
    y.append(1 if (hit[a], hit[b]) in gt_edges else 0)
    s_h.append(float(sim[k]))
    s_d.append(-float(dist[k]))
y = np.asarray(y)
print(f"\n  {len(y)} candidate edges with BOTH endpoints matched to GT; "
      f"{int(y.sum())} are real GT edges "
      f"({y.mean() if len(y) else float('nan'):.3f} positive)", flush=True)

if len(y) > 20 and 0 < y.sum() < len(y):
    a_h = roc_auc_score(y, s_h)
    a_d = roc_auc_score(y, s_d)
    print(f"\n  HOCT '{SCORE_KEY}'   AUC {a_h:.4f}", flush=True)
    print(f"  -distance baseline  AUC {a_d:.4f}", flush=True)
    print(f"  HOCT minus baseline {a_h - a_d:+.4f}", flush=True)
    print("\n  GATE: at or under the baseline means the domain gap ate it "
          "-- HOCT saw Cell Tracking Challenge and bacteria, not "
          "light-sheet embryos, and cannot be fine-tuned. Comfortably "
          "above and the feature route is worth building: its score as "
          "one column in OUR in-domain ranker, which sidesteps the "
          "withheld training code because what gets recalibrated is how "
          "much to trust it.", flush=True)
else:
    print("  not enough usable edges, or no negatives among them, to "
          "compute an AUC -- read the counts above before concluding "
          "anything about the model", flush=True)


# ---- the morphology our pipeline has never had --------------------------
print("\n=== NODE FEATURES create_graph COMPUTED FROM THE MASKS ===",
      flush=True)
print("  Our pipeline owns coordinates and a detection score. Nothing "
      "else. These come free with the graph:", flush=True)
try:
    keys = [k for k in graph.node_attr_keys()
            if k not in ("t", "z", "y", "x", "node_id", "mask", "bbox",
                         "solution", "tracklet_id")]
    print(f"  {keys}", flush=True)
    probe = [k for k in ("equivalent_diameter_area", "intensity_mean",
                         "border_dist", "orphan_prob") if k in keys]
    if probe:
        nav = graph.node_attrs(attr_keys=probe)
        for k in probe:
            v = np.asarray(nav[k].to_list(), float)
            v = v[np.isfinite(v)]
            if len(v):
                print(f"    {k:<26} median {np.median(v):.4f}  "
                      f"p10 {np.quantile(v, 0.1):.4f}  "
                      f"p90 {np.quantile(v, 0.9):.4f}", flush=True)
except Exception:
    print("  could not read them:", flush=True)
    traceback.print_exc()

print("\nHOCT SCORE DONE", flush=True)
