"""
edgeprob2_cache.py -- the SECOND-SEED official edge head's opinion of OUR
(xl) node set. Generated from edgeprob_cache.py (v3, 2026-09-03a) by
make_edgeprob2.py; the injection, the datasets, the pairs and the output
layout are byte-for-byte edgeprob_cache.py's. What differs:

  * WEIGHTS come from pilkwang/biohub-temporal-unet3d-seed314159-v1
    (weights/unet_transformer/split_0/edge_predictor_best.pth + config.json),
    the independently trained seed the public 0.948 notebook fuses as its
    "secondary" model (detection weight 0.80, edge weight 0.20). Same
    architecture, same repo; the pack's repo/ is still what is imported.
  * Output prefix edgeprob2_ (edgeprob2_<name>.npz, edgeprob2_F4_<name>.npz,
    edgeprob2_counts.json with a "head" field), so both heads' caches can
    be mounted side by side and joined on (t, i, j).
  * The split_manifest.json cross-check reads only a manifest with an
    "all" key (dctrain's); the seed pack ships its own split_manifest.json
    with a different layout, which is printed for the record instead:
    which of the 24 evaluation datasets the second seed trained on.

--- edgeprob_cache.py's docstring follows ---
edgeprob_cache.py -- the OFFICIAL edge head's opinion of OUR (xl) node set.

Why this exists
---------------
The current pipeline links xl detections with an 18-feature geometric
ranker and never sees the end-to-end-trained edge head (progress.md
2026-08-24, "the edge ranker never sees edge_prob"). The head's probs cached
by `cache_candidates.py` are indexed to the OFFICIAL detector's nodes, so
they cannot be joined onto xl pairs. This script re-runs the official head
over the xl primaries -- a GPU pass, because the head indexes U-Net features
at the node locations -- and dumps, per dataset, the head's probability for
every candidate pair the CPU ranker will ever look at.

How the injection works (read from the official predict_unet_transformer.py)
---------------------------------------------------------------------------
`predict_video` (official script, lines 297-497) finds cells by calling the
module-level `_detect_cells_pooled(det_logits, t, det_threshold, pool_kernel)`
(lines 254-294) once per frame and expects back an (N, 4) array of
[t, z, y, x] "in downsampled space" (its own docstring, line 280). Everything
downstream of that call -- `_index_features` (feature lookup at the node's
grid voxel), `extract_pos_features` (sinusoidal position over the downsampled
image shape), `predict_edges` (which gets `p_coords * ds_arr_t`, i.e. the
coords scaled back to original voxels) -- consumes exactly that array. So:

  * `P._detect_cells_pooled` is replaced by a function that ignores the
    heatmap and returns our primaries for frame t, converted to the grid
    the head indexes with (see COORDINATE FRAME below);
  * `model.predict_edges` is wrapped (weights untouched) so the raw
    (n_src, n_tgt) logit matrix of every consecutive pair is captured at
    the moment it is produced; the wrapper also asserts that the coords
    the head was handed round-trip to the coords we injected;
  * `cfg.threshold = 2.0` so predict_video's own O(n^2) Python candidate
    loop keeps nothing -- the probabilities are taken from the captured
    matrices with the official activation (`softmax` over SOURCES, dim 0)
    and gated to pairs within GATE_UM, the same gate `pair_features` uses.

COORDINATE FRAME. `detcache_<name>.npz` stores `c{t}` as full-resolution
voxel coordinates (z, y, x): cache_train.py maps the 4x-pooled heatmap peak
back with `* POOL + (POOL-1)/2` (lines 240-241), re-centres on the
full-resolution volume in `_refine` (174-188), and `_physical_nms` scales by
SCALE um/voxel (190-204). The official head reads each frame STRIDED,
`zarr_arr[t, ::dz, ::dy, ::dx]` (`_load_frame`, line 220), with
`downsample` from the checkpoint's config.json (`load_model`, 162-200;
default (1, 4, 4)). Grid index g therefore sits on full-res voxel g*d, so
full-res (z, y, x) -> grid (z/dz, y/dy, x/dx). Floats are injected, not
rounded: training fed the head float GT coords `/ ds` (`get_window_data`,
train_unet_transformer.py line 224) and `_index_features` truncates with
`.long()` (line 466-468), so floats reproduce the training convention.

The 2026-08-22 chimera v1 died of a double downsample: `open_dataset`
already returns the DOWNSAMPLED `image_shape` (io.py 95-96,
`ceil(raw / d)`), so nothing here divides a shape or a coordinate twice.

REVERSE PASS. `predict_video` reads frames by index through `_load_frame`
and asks `_detect_cells_pooled` for frame t; remapping both through
t -> T-1-t runs the identical model over the reversed movie, so the head
scores t+1 -> t and its softmax normalises over the t+1 side. The public
0.948 notebook fuses forward/reverse by a harmonic mean at weight 0.15;
the fusion is left to the CPU side, both probabilities are dumped.

v3 (2026-09-03a) widens the pass in two directions. (a) ALL 199 train
movies with BASE (xl) primaries, so a ranker can be retrained on any pool
with the head's probability as a feature and every holdout is covered.
(b) A second pass over the 24 evaluation datasets (HOLDOUT_6BBA + the
seeded 44b6 draw) using the F4 primaries -- fusecache's
`fdet_F4_<name>.npz` (same c{t}/s{t} layout, floor 0.05; primaries are
again score >= UNET_THRESH), the detector that ships at 0.925 -- written
as `edgeprob_F4_<name>.npz`, so the CPU side can score the head on the
node set it will actually link. Neither pass skips a file that already
exists: this kernel's output is self-contained and never depends on the
v2 output being mounted.

Output, one file per dataset, `edgeprob_<name>.npz` (BASE) or
`edgeprob_F4_<name>.npz` (F4 primaries):

    t       (M,) int32    source frame
    i       (M,) int32    index into seeds_of(dets[t])[0]      (primaries)
    j       (M,) int32    index into seeds_of(dets[t+1])[0]
    p_fwd   (M,) float32  official softmax prob, forward pass  (t -> t+1)
    p_rev   (M,) float32  official softmax prob, reverse pass  (t+1 -> t);
                          NaN when REVERSE is False
    l_fwd   (M,) float32  raw logit, forward  (lets the CPU side try sigmoid)
    l_rev   (M,) float32  raw logit, reverse  (NaN when REVERSE is False)
    n_per_frame (T,) int32  primaries injected per frame -- the CPU script
                          MUST assert this equals its own seeds_of counts
                          before trusting (t, i, j)

plus `edgeprob_counts.json` with per-dataset counts and timings and a
`groups` field listing the base and F4 dataset names.

(i, j) index the primaries in the SAME order `rankwhy.seeds_of` produces
them: `c{t}` rows filtered by `s{t} >= UNET_THRESH`, order preserved.
Pairs are those within GATE_UM (physical, SCALE um/voxel) -- the same gate
`pair_features` uses -- so the CPU join is a plain (t, i, j) lookup.

Push with the official CLI (the MCP connector cannot attach data sources):

    py kag.py kernels push -p .

Needs, in kernel-metadata.json: enable_gpu true, machine_shape
NvidiaTeslaT4 (trap 3/9), dataset_sources pilkwang's support pack,
competition_sources the competition data (images are read), kernel_sources
biohub-edge-ranker + biohub-cache-train (the detcaches) + biohub-poolsize
(edge_ranker_d32.pkl, for the 44b6 draw and the D32 pool) + biohub-fusecache
(the fdet_F4_<name>.npz caches for the 24 evaluation datasets).

OUTPUT DISCIPLINE (rules 2 and 10): one line per dataset, nothing per
frame, nothing per pair.
"""

from __future__ import annotations

import json
import os
import pathlib
import pickle
import random
import shutil
import subprocess
import sys
import time

import numpy as np

SCRIPT_VERSION = "2026-09-08a-ep2hpcache"

# ---------------------------------------------------------------- knobs

# MUST equal rankwhy.py / cache_train.py; (t, i, j) is meaningless otherwise.
UNET_THRESH = 0.15
# MUST equal rankwhy.GATE_UM: pairs the CPU ranker considers.
GATE_UM = 10.0
# The xl pipeline's constant (cache_train.py:49); pair_features gates with
# this, not with the zarr's own scale, so the gate here uses the same.
SCALE = np.array([1.625, 0.40625, 0.40625])

REVERSE = True          # also score t+1 -> t on the reversed movie
SEED_44B6 = 20260827    # dctrain.py:158 -- the user's standing rule: random, seeded
N_44B6 = 12
N_POOL32 = 32

# Two base datasets + one F4 dataset. v2 came back clean on 56 datasets,
# so the default is the full pass: 199 base + 24 F4 = 223, ~2.8 h of T4.
SMOKE = False

# predict_video's own candidate gate. Set ABOVE 1 so its Python loop keeps
# nothing; the probabilities come from the captured logit matrices instead.
NO_CANDIDATES = 2.0

BUDGET_S = 10.0 * 3600  # Kaggle kills a GPU kernel at 12 h; stop cleanly first
                        # (223 datasets x ~45 s = ~2.8 h expected)

# ---------------------------------------------------------------- setup

print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

PACK = pathlib.Path("/kaggle/input/datasets/pilkwang/"
                    "biohub-tracking-support-pack-50ep-v1")
if not PACK.exists():                    # some mounts drop the "datasets/" level
    PACK = pathlib.Path("/kaggle/input/biohub-tracking-support-pack-50ep-v1")
if not PACK.exists():
    entries = sorted(e.name for e in os.scandir("/kaggle/input")) \
        if pathlib.Path("/kaggle/input").exists() else []
    raise SystemExit(
        f"support pack not attached. /kaggle/input holds {len(entries)} "
        f"entries: {entries}. If this is empty the notebook was pushed "
        f"without data sources -- use `py kag.py kernels push -p .` with "
        f"kernel-metadata.json, not the MCP connector."
    )

# Wheels that would overwrite something the Kaggle image already ships and that
# torch/scipy are compiled against. Installing these is what broke numpy.
KEEP_OUT = {"numpy", "scipy", "torch", "pandas", "pyarrow", "numba", "llvmlite",
            "matplotlib", "pillow", "networkx", "scikit-learn", "scikit-image",
            "setuptools", "wheel", "pip"}

WH = pathlib.Path("/kaggle/working/wheelhouse")
shutil.rmtree(WH, ignore_errors=True)
WH.mkdir(parents=True)
for w in sorted(PACK.rglob("*.whl")):
    if w.name.split("-")[0].replace("_", "-").lower() not in KEEP_OUT:
        shutil.copy(w, WH / w.name)
staged = sorted(WH.glob("*.whl"))
r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps"]
    + [str(w) for w in staged],
    capture_output=True, text=True)
print(f"pip exit {r.returncode} ({len(staged)} wheels)", flush=True)
if r.returncode != 0:
    print(r.stderr[-1500:], flush=True)
shutil.rmtree(WH, ignore_errors=True)

import torch                                                  # noqa: E402
from scipy.spatial import cKDTree                             # noqa: E402

REPO = PACK / "repo"
for p in (REPO / "src", REPO / "scripts"):
    assert p.exists(), p
    sys.path.insert(0, str(p))

import predict_unet_transformer as P                          # noqa: E402

# The injection relies on these two module-level names being what
# predict_video calls. Fail here if the pack's copy differs.
assert callable(getattr(P, "_detect_cells_pooled", None)), \
    "predict_unet_transformer has no _detect_cells_pooled -- injection point gone"
assert callable(getattr(P, "_load_frame", None)), \
    "predict_unet_transformer has no _load_frame -- reverse pass impossible"

SECONDARY = pathlib.Path("/kaggle/input/datasets/pilkwang/"
                         "biohub-temporal-unet3d-seed314159-v1")
if not SECONDARY.exists():
    SECONDARY = pathlib.Path("/kaggle/input/biohub-temporal-unet3d-seed314159-v1")
assert SECONDARY.exists(), (
    "second-seed pack not attached: add pilkwang/biohub-temporal-unet3d-"
    "seed314159-v1 to dataset_sources and push with kag.py")
WEIGHTS = SECONDARY / "weights/unet_transformer/split_0/edge_predictor_best.pth"
assert WEIGHTS.exists(), WEIGHTS
assert (WEIGHTS.parent / "config.json").exists(), (
    "the seed pack has no config.json beside its weights; load_model needs it")
HEAD = "seed314159"
import hashlib as _hashlib                                    # noqa: E402
WEIGHTS_SHA256 = _hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
print(f"head {HEAD}: {WEIGHTS} sha256 {WEIGHTS_SHA256}", flush=True)
# the public 0.948 notebook pins this checksum for its secondary model
if WEIGHTS_SHA256 != "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f":
    print("  WARN weight sha256 differs from the one the 0.948 notebook pins "
          "-- a different version of the seed pack is mounted", flush=True)
_PRIMARY_WEIGHTS = PACK / "weights/unet_transformer/split_0/edge_predictor_best.pth"
assert _PRIMARY_WEIGHTS.exists(), _PRIMARY_WEIGHTS
assert _hashlib.sha256(_PRIMARY_WEIGHTS.read_bytes()).hexdigest() != WEIGHTS_SHA256, \
    "the seed pack's weights are byte-identical to the pack's -- not a second seed"

COMP = pathlib.Path("/kaggle/input/competitions/"
                    "biohub-cell-tracking-during-development")
if not COMP.exists():
    COMP = pathlib.Path("/kaggle/input/biohub-cell-tracking-during-development")
assert COMP.exists(), "competition data not attached (the head reads the images)"
TRAIN = COMP / "train"
assert TRAIN.exists(), TRAIN


def _walk_files(match):
    """Bounded search of /kaggle/input: prunes the competition tree (millions
    of zarr chunks; a blind walk appears to hang -- toolchain trap 5)."""
    hits = {}
    for root_, dirs, fnames in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs
                   if "cell-tracking-during-development" not in d
                   and not d.endswith(".zarr") and not d.endswith(".geff")]
        for fn in fnames:
            key = match(fn)
            if key is not None:
                hits.setdefault(key, pathlib.Path(root_) / fn)
    return hits


CACHES = _walk_files(lambda fn: fn[len("detcache_"):-len(".npz")]
                     if fn.startswith("detcache_") and fn.endswith(".npz")
                     else None)
assert CACHES, ("no detcache_*.npz under /kaggle/input -- attach "
                "faustine649/biohub-edge-ranker and "
                "faustine649/biohub-cache-train as kernel sources")
print(f"detcaches mounted: {len(CACHES)}", flush=True)

# ---------------------------------------------------------------- datasets
# BASE pass: all 199 train movies. The three groups below are still
# computed and PRINTED (informational, and the counts JSON records them):
# the 12-dataset 6bba holdout every recorded number comes from; 12 44b6
# drawn at random with dctrain's seed from the 44b6 OUTSIDE the D32 ranker
# pool (the user's standing rule: never sorted()[:12]); and the D32 pool
# itself. F4 pass: the 24 evaluation datasets (6bba holdout + 44b6 draw).

HOLDOUT_6BBA = [
    "6bba_268e1230", "6bba_2312ac41", "6bba_283bf9f1", "6bba_3a1849c2",
    "6bba_3abfe10a", "6bba_5c824876", "6bba_61dd1e0d", "6bba_7af54fde",
    "6bba_7b5d3b2c", "6bba_afb141ff", "6bba_aeee7805", "6bba_c27cba08",
]

_pk = _walk_files(lambda fn: fn if fn == "edge_ranker_d32.pkl" else None)
assert _pk, ("edge_ranker_d32.pkl not found -- attach "
             "faustine649/biohub-poolsize as a kernel source; without the "
             "D32 pool the 44b6 draw cannot avoid the ranker's training set")
with open(next(iter(_pk.values())), "rb") as _f:
    _blob = pickle.load(_f)
POOL32 = sorted(set(_blob.get("pool") or []))
assert len(POOL32) == N_POOL32, (
    f"the pkl's pool has {len(POOL32)} datasets, expected {N_POOL32}")
assert not set(POOL32) & set(HOLDOUT_6BBA), "D32 pool overlaps the holdout"

ALL_TRAIN = sorted(p.name[:-len(".zarr")] for p in TRAIN.glob("*.zarr"))
assert len(ALL_TRAIN) == 199, f"expected 199 train movies, found {len(ALL_TRAIN)}"
assert all(n in ALL_TRAIN for n in HOLDOUT_6BBA + POOL32), \
    "a holdout / pool name is not in train/"

# dctrain.py:158-163 verbatim, so this draw equals dctrain's excluded 12.
eligible_44b6 = sorted(n for n in ALL_TRAIN
                       if n.startswith("44b6_") and n not in set(POOL32))
assert len(eligible_44b6) >= N_44B6, \
    f"only {len(eligible_44b6)} 44b6 datasets outside D32's pool"
SET_44B6 = sorted(random.Random(SEED_44B6).sample(eligible_44b6, N_44B6))

# Optional cross-check against dctrain's own split_manifest.json when its
# weights happen to be mounted. NOT required -- do not attach a source for it.
_man_all = {}
for _r0, _d0, _f0 in os.walk("/kaggle/input"):
    _d0[:] = [d for d in _d0 if "cell-tracking-during-development" not in d
              and not d.endswith(".zarr") and not d.endswith(".geff")]
    if "split_manifest.json" in _f0:
        _pp = pathlib.Path(_r0) / "split_manifest.json"
        try:
            _jj = json.loads(_pp.read_text())
        except Exception:                                        # noqa: BLE001
            continue
        if isinstance(_jj, dict) and "all" in _jj:
            _man_all[str(_pp)] = _jj
_man = _man_all
if _man:
    _seen = set(next(iter(_man.values()))["all"])
    assert not (set(SET_44B6) & _seen), (
        "the seeded 44b6 draw overlaps dctrain's training pool -- the draw "
        "no longer reproduces dctrain.py's; do not trust this 44b6 arm")
    print("  44b6 draw cross-checked against split_manifest.json: disjoint",
          flush=True)
else:
    print("  split_manifest.json not mounted -> 44b6 draw NOT cross-checked "
          "against dctrain's (compare the printed names by hand)", flush=True)

_groups = HOLDOUT_6BBA + SET_44B6 + POOL32
assert len(_groups) == len(set(_groups)) == 56, "groups overlap or are short"
print(f"6bba holdout ({len(HOLDOUT_6BBA)}): {' '.join(HOLDOUT_6BBA)}", flush=True)
print(f"44b6 seed={SEED_44B6} from {len(eligible_44b6)} outside-D32 "
      f"({len(SET_44B6)}): {' '.join(SET_44B6)}", flush=True)
print(f"D32 pool ({len(POOL32)}): {' '.join(POOL32)}", flush=True)


def _manifest_names(path):
    """Every dataset-looking name in a split manifest, whatever its layout."""
    out = {}
    try:
        jj = json.loads(path.read_text())
    except Exception as exc:                                     # noqa: BLE001
        print(f"  {path}: unreadable ({exc})", flush=True)
        return out

    def walk(o, key):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, k)
        elif isinstance(o, list):
            names = [str(x) for x in o if isinstance(x, str)
                     and len(x) > 5 and x[4] == "_"]
            if names:
                out.setdefault(key, set()).update(names)
    walk(jj, "root")
    return out


for _lab, _mp in (("second seed", WEIGHTS.parent / "split_manifest.json"),
                  ("pack split_0", _PRIMARY_WEIGHTS.parent / "split_manifest.json")):
    if not _mp.exists():
        print(f"  {_lab}: no split_manifest.json beside the weights", flush=True)
        continue
    _mn = _manifest_names(_mp)
    for _k, _v in sorted(_mn.items()):
        _o6 = sorted(_v & set(HOLDOUT_6BBA))
        _o4 = sorted(_v & set(SET_44B6))
        print(f"  {_lab} manifest[{_k}]: {len(_v)} names; overlaps our "
              f"6bba holdout on {len(_o6)}, our 44b6 draw on {len(_o4)}",
              flush=True)

# BASE pass: every train movie, sorted.
names = list(ALL_TRAIN)
assert len(names) == 199, len(names)

# F4 pass: the 24 evaluation datasets, primaries from fusecache's
# fdet_F4_<name>.npz (same c{t}/s{t} keys as a detcache; rankwhy's
# load_dets(name, arm="F4")). Discovered by name after "fdet_F4_".
EVAL_24 = sorted(HOLDOUT_6BBA + SET_44B6)
assert len(EVAL_24) == len(set(EVAL_24)) == 24, "evaluation set is not 24"
CACHES_F4 = _walk_files(lambda fn: fn[len("fdet_F4hp_"):-len(".npz")]
                        if fn.startswith("fdet_F4hp_") and fn.endswith(".npz")
                        else None)
print(f"fdet_F4 caches mounted: {len(CACHES_F4)}", flush=True)
assert len(CACHES_F4) >= 24, ("fewer than 24 fdet_F4hp_*.npz under "
                              "/kaggle/input -- attach biohub-fusecache "
                              "for the F4 detections")
names_f4 = [n for n in ALL_TRAIN if n in set(CACHES_F4)]
assert len(names_f4) == len(CACHES_F4), (len(names_f4), len(CACHES_F4))
_six_ = [n for n in ALL_TRAIN if n.startswith("6bba_")]
_gap_ = [n for n in _six_ if n not in set(names_f4)]
assert not _gap_ and len(names_f4) >= 128, (
    f"{len(_gap_)} of the 128 6bba movies have no fdet_F4hp_ cache "
    f"({_gap_[:5]}); the bench scores all 128 and needs a row for each")
SHIPPED_EP2 = _walk_files(lambda fn: fn[len("edgeprob2_F4_"):-len(".npz")]
                          if fn.startswith("edgeprob2_F4_")
                          and fn.endswith(".npz") else None)
print(f"shipped edgeprob2_F4_ caches for the n_per_frame check: "
      f"{len(SHIPPED_EP2)}", flush=True)
NPF_BAD = []
NPF_NOCHK = []

if SMOKE:
    names = [HOLDOUT_6BBA[0], SET_44B6[0]]
    names_f4 = [SET_44B6[0]]
    print("[SMOKE] 2 base + 1 F4 datasets only -- set SMOKE = False for "
          "the full pass", flush=True)

missing = [n for n in names if n not in CACHES]
assert not missing, f"no detcache for: {missing}"
missing = [n for n in names if not (TRAIN / f"{n}.zarr").exists()]
assert not missing, f"not in train/: {missing}"
missing = [n for n in names_f4 if n not in CACHES_F4]
assert not missing, (f"no fdet_F4 cache for {len(missing)} of the 24 "
                     f"evaluation datasets: {missing} -- attach "
                     f"faustine649/biohub-fusecache; if the 44b6 names above "
                     f"differ from fusecache's manifest-excluded 12, the "
                     f"seeded draw no longer reproduces dctrain's")
missing = [n for n in names_f4 if n not in names]
assert not missing, f"F4 datasets outside the base list: {missing}"

# One job per (arm, dataset): the base pass first, then the F4 pass.
jobs = [("F4", n) for n in names_f4]      # base dropped, see above
assert len(jobs) == len(names_f4) == 24, (len(jobs), len(names_f4))
print(f"jobs: 0 base (dropped) + {len(names_f4)} F4hp = {len(jobs)}",
      flush=True)

# ---------------------------------------------------------------- GPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, flush=True)
if device.type != "cuda":
    raise SystemExit("[degraded] no GPU -- aborting rather than running "
                     "volumetric inference on CPU")
cap = torch.cuda.get_device_capability(0)
print(f"gpu: {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}", flush=True)
if cap[0] < 7:
    raise SystemExit(
        f"[degraded] this machine is sm_{cap[0]}{cap[1]}; the image's torch "
        f"supports sm_70 and up. Set machine_shape to NvidiaTeslaT4 in "
        f"kernel-metadata.json and push again."
    )

model, window_size, downsample = P.load_model(WEIGHTS, device)
assert window_size == 2, (
    f"window_size={window_size}: the wrapper below assumes one predict_edges "
    f"call per consecutive pair in ascending order, which holds for W=2 only")
DS = np.asarray(downsample, dtype=np.float32)          # (dz, dy, dx)
assert DS.shape == (3,) and np.all(DS >= 1), downsample

# det_tta only affects the detection logits, which are discarded here
# (predict_video lines 379-392: the TTA flips touch det_logits, never
# unet_out). Off saves three U-Net passes per window.
cfg = P.PredictConfig(det_threshold=0.99, det_tta=False, use_ilp=True,
                      threshold=NO_CANDIDATES)
print(f"downsample={tuple(downsample)} window={window_size} "
      f"edge_activation={cfg.edge_activation} threshold={cfg.threshold} "
      f"det_tta={cfg.det_tta} caps={cfg.max_parents_per_node}/"
      f"{cfg.max_children_per_node}", flush=True)
assert cfg.edge_activation == "softmax", cfg.edge_activation

# ---------------------------------------------------------------- injection

_ORIG_DETECT = P._detect_cells_pooled
_ORIG_LOAD = P._load_frame
_ORIG_EDGES = model.predict_edges


def seeds_of(coords, scores):
    """rankwhy.seeds_of verbatim: primaries are score >= UNET_THRESH, order kept."""
    coords = np.asarray(coords, float).reshape(-1, 3)
    scores = np.asarray(scores, float).reshape(-1)
    m = scores >= UNET_THRESH
    return coords[m], scores[m]


def load_primaries(name, arm="base"):
    """[(N_t, 3) full-res float64] for t = 0..T-1, in seeds_of order.

    arm 'base': detcache_<name>.npz (xl primaries);
    arm 'F4':   fdet_F4_<name>.npz (fusecache's F4 blend, same layout).
    """
    assert arm in ("base", "F4"), arm
    z = np.load(CACHES[name] if arm == "base" else CACHES_F4[name])
    T = 0
    while f"c{T}" in z.files:
        T += 1
    assert T > 1, f"{name}: detcache has {T} frame(s)"
    assert all(f"s{t}" in z.files for t in range(T)), f"{name}: s{{t}} missing"
    return [seeds_of(z[f"c{t}"], z[f"s{t}"])[0] for t in range(T)]


def gated_pairs(prim):
    """Per t: (ii, jj) of pairs within GATE_UM -- pair_features' gate."""
    out = []
    for t in range(len(prim) - 1):
        Pu = prim[t] * SCALE[None, :]
        Cu = prim[t + 1] * SCALE[None, :]
        if len(Pu) == 0 or len(Cu) == 0:
            out.append((np.zeros(0, np.int32), np.zeros(0, np.int32)))
            continue
        hits = cKDTree(Cu).query_ball_point(Pu, GATE_UM)
        ii = np.concatenate([np.full(len(h), i, np.int32)
                             for i, h in enumerate(hits)]) \
            if any(len(h) for h in hits) else np.zeros(0, np.int32)
        jj = np.concatenate([np.asarray(h, np.int32) for h in hits]) \
            if any(len(h) for h in hits) else np.zeros(0, np.int32)
        out.append((ii, jj))
    return out


class Run:
    """State shared between the three patched functions for ONE pass.

    direction 'fwd': predict_video's frame index f is the movie's t.
    direction 'rev': predict_video's frame index f is the movie's T-1-f,
    so its pair (f, f+1) is the movie's (t+1 -> t) with t = T-2-f.
    """

    def __init__(self, name, prim, pairs, direction):
        self.name, self.prim, self.pairs, self.dir = name, prim, pairs, direction
        self.T = len(prim)
        self.n = np.array([len(p) for p in prim], np.int32)
        # frames in predict_video order -> full-res coords
        self.full = [prim[self.movie_t(f)].astype(np.float32)
                     for f in range(self.T)]
        # the movie t of each predict_edges call, in call order; a pair is
        # skipped by predict_video when either frame is empty (line 418-419)
        self.expected = [self.movie_t(f) if direction == "fwd" else
                         self.movie_t(f) - 1
                         for f in range(self.T - 1)
                         if len(self.full[f]) and len(self.full[f + 1])]
        self.ptr = 0
        self.p = {}
        self.l = {}

    def movie_t(self, f):
        return f if self.dir == "fwd" else self.T - 1 - f

    # -- P._detect_cells_pooled replacement ---------------------------------
    def detect(self, det_logits, f, det_threshold, pool_kernel):
        c = self.full[f]
        if len(c) == 0:
            return np.empty((0, 4), dtype=np.float32)
        grid = c / DS[None, :]                       # full-res -> U-Net grid
        return np.concatenate(
            [np.full((len(c), 1), f, np.float32), grid.astype(np.float32)],
            axis=1)

    # -- P._load_frame replacement (identity in the forward pass) -----------
    def load_frame(self, zarr_arr, f, target_shape, downsample_):
        return _ORIG_LOAD(zarr_arr, self.movie_t(f), target_shape, downsample_)

    # -- model.predict_edges wrapper ---------------------------------------
    def edges(self, feat_src, feat_tgt, coords_src, coords_tgt, *rest):
        out = _ORIG_EDGES(feat_src, feat_tgt, coords_src, coords_tgt, *rest)
        assert self.ptr < len(self.expected), (
            f"{self.name} {self.dir}: predict_edges called more often than "
            f"there are non-empty consecutive pairs ({len(self.expected)})")
        t = self.expected[self.ptr]
        self.ptr += 1
        src_t, tgt_t = (t, t + 1) if self.dir == "fwd" else (t + 1, t)
        raw = out[0]
        assert tuple(raw.shape) == (self.n[src_t], self.n[tgt_t]), (
            f"{self.name} {self.dir} t={t}: logits {tuple(raw.shape)} vs "
            f"injected ({self.n[src_t]}, {self.n[tgt_t]})")
        # Round trip at the consumption point: the head was handed
        # grid * downsample, which must be the full-res coords injected.
        for got, exp_t in ((coords_src[0], src_t), (coords_tgt[0], tgt_t)):
            exp = torch.from_numpy(self.prim[exp_t].astype(np.float32)).to(got.device)
            err = float((got - exp).abs().max()) if len(exp) else 0.0
            assert err <= 0.5, (
                f"{self.name} {self.dir} t={exp_t}: coords handed to "
                f"predict_edges differ from injected by {err:.3f} voxel")
        ii, jj = self.pairs[t]
        probs = torch.softmax(raw, dim=0)            # official: over sources
        ii_t = torch.as_tensor(ii, dtype=torch.long, device=raw.device)
        jj_t = torch.as_tensor(jj, dtype=torch.long, device=raw.device)
        if self.dir == "fwd":
            sel = (ii_t, jj_t)
        else:
            sel = (jj_t, ii_t)                       # rows are t+1 nodes here
        self.p[t] = probs[sel].float().cpu().numpy() if len(ii) \
            else np.zeros(0, np.float32)
        self.l[t] = raw[sel].float().cpu().numpy() if len(ii) \
            else np.zeros(0, np.float32)
        return out

    def install(self):
        P._detect_cells_pooled = self.detect
        P._load_frame = self.load_frame
        model.predict_edges = self.edges

    @staticmethod
    def uninstall():
        P._detect_cells_pooled = _ORIG_DETECT
        P._load_frame = _ORIG_LOAD
        model.predict_edges = _ORIG_EDGES

    def check_return(self, coords, edges):
        """predict_video's own return must round-trip our injection."""
        assert len(edges) == 0, (
            f"{self.name} {self.dir}: predict_video kept {len(edges)} edges at "
            f"threshold {NO_CANDIDATES}; softmax probs above 1 are impossible")
        assert self.ptr == len(self.expected), (
            f"{self.name} {self.dir}: {self.ptr} predict_edges calls, expected "
            f"{len(self.expected)}")
        coords = np.asarray(coords)
        assert len(coords) == int(self.n.sum()), (
            f"{self.name} {self.dir}: predict_video reports {len(coords)} nodes, "
            f"injected {int(self.n.sum())}")
        # predict_video returns int16(trunc(grid * ds)); compare per frame in
        # ITS frame order against trunc of what we injected.
        off = 0
        for f in range(self.T):
            k = len(self.full[f])
            blk = coords[off:off + k]
            off += k
            if k == 0:
                continue
            assert np.all(blk[:, 0] == f), \
                f"{self.name} {self.dir}: frame column mismatch at f={f}"
            grid = (self.full[f] / DS[None, :]).astype(np.float32)
            exp = np.trunc(grid * DS[None, :])
            err = float(np.abs(blk[:, 1:].astype(np.float32) - exp).max())
            assert err <= 0.5, (
                f"{self.name} {self.dir} f={f}: returned coords differ from "
                f"injected by {err:.2f} voxel (truncation accounted for)")


# ---------------------------------------------------------------- run

OUT = pathlib.Path("/kaggle/working/edgeprob2")
OUT.mkdir(exist_ok=True)

t_all = time.time()
failed: list[str] = []
counts: dict[str, dict] = {}
skipped_time = 0

for k, (arm, name) in enumerate(jobs, 1):
    # label: "<name>" for the base pass, "F4_<name>" for the F4 pass --
    # also the output stem after "edgeprob_" and the counts/failed key.
    label = name if arm == "base" else f"F4hp_{name}"
    if time.time() - t_all > BUDGET_S:
        skipped_time = len(jobs) - k + 1
        print(f"\nWARN wall-clock budget reached: {skipped_time} datasets NOT "
              f"cached ({label} onwards). PARTIAL output.", flush=True)
        break
    t0 = time.time()
    try:
        prim = load_primaries(name, arm)
        T = len(prim)
        pairs = gated_pairs(prim)
        runs = {}
        for direction in (("fwd", "rev") if REVERSE else ("fwd",)):
            run = Run(label, prim, pairs, direction)
            run.install()
            try:
                coords, edges = P.predict_video(
                    model, TRAIN / name, device, cfg=cfg,
                    window_size=window_size, max_frames=T,
                    unet_batch_size=4, downsample=downsample)
            finally:
                Run.uninstall()
            run.check_return(coords, edges)
            runs[direction] = run
    except Exception as exc:                                  # noqa: BLE001
        Run.uninstall()
        failed.append(label)
        print(f"[degraded] {label} FAILED ({type(exc).__name__}: "
              f"{str(exc)[:300]}) -- no edgeprob written; every CPU join "
              f"below will silently lack this dataset. Continuing.",
              flush=True)
        continue

    fwd = runs["fwd"]
    tt, ii, jj, pf, lf, pr, lr = [], [], [], [], [], [], []
    for t in range(T - 1):
        i_, j_ = pairs[t]
        if len(i_) == 0:
            continue
        # a gated pair exists only when both frames are non-empty, and then
        # predict_edges ran for it -- so the lookups below cannot KeyError
        tt.append(np.full(len(i_), t, np.int32))
        ii.append(i_)
        jj.append(j_)
        pf.append(fwd.p[t])
        lf.append(fwd.l[t])
        if REVERSE:
            pr.append(runs["rev"].p[t])
            lr.append(runs["rev"].l[t])
    M = int(sum(len(x) for x in ii))
    if M == 0:
        failed.append(label)
        print(f"[degraded] {label}: zero gated pairs -- skipped", flush=True)
        continue
    cat = lambda xs: np.concatenate(xs).astype(np.float32)    # noqa: E731
    p_rev = cat(pr) if REVERSE else np.full(M, np.nan, np.float32)
    l_rev = cat(lr) if REVERSE else np.full(M, np.nan, np.float32)
    p_fwd = cat(pf)
    assert np.all((p_fwd >= 0) & (p_fwd <= 1)), f"{label}: p_fwd out of [0,1]"
    if REVERSE:
        assert np.all((p_rev >= 0) & (p_rev <= 1)), f"{label}: p_rev out of [0,1]"

    np.savez_compressed(
        OUT / f"edgeprob2_{label}.npz",
        t=np.concatenate(tt), i=np.concatenate(ii).astype(np.int32),
        j=np.concatenate(jj).astype(np.int32),
        p_fwd=p_fwd, p_rev=p_rev, l_fwd=cat(lf), l_rev=l_rev,
        n_per_frame=fwd.n)
    _ship = SHIPPED_EP2.get(name)
    if _ship is None:
        NPF_NOCHK.append(name)
        pass   # expected off the 24; counted, not failed
    else:
        _z = np.load(_ship)
        _old = np.asarray(_z["n_per_frame"], np.int64)
        _new = np.asarray(fwd.n, np.int64)
        if _old.shape != _new.shape or not np.array_equal(_old, _new):
            NPF_BAD.append(name)
            print(f"  !! {name}: n_per_frame DIFFERS from the shipped "
                  f"edgeprob2_F4_ cache -- hparacache was supposed to keep "
                  f"every detection, so the arms are NOT scoring the same "
                  f"cells", flush=True)
        _z.close()
    mb = (OUT / f"edgeprob2_{label}.npz").stat().st_size / 2 ** 20
    dt = time.time() - t0
    counts[label] = dict(arm=arm, name=name, T=T, nodes=int(fwd.n.sum()),
                         pairs=M, calls_fwd=fwd.ptr,
                         calls_rev=runs["rev"].ptr if REVERSE else 0,
                         seconds=round(dt, 1))
    # above_0.5: how many pairs the official gate (threshold 0.5) would keep.
    print(f"[{k}/{len(jobs)}] {label}: T={T} nodes={int(fwd.n.sum())} "
          f"pairs={M} fwd>0.5={int((p_fwd > 0.5).sum())} "
          f"rev>0.5={int((p_rev > 0.5).sum()) if REVERSE else 'n/a'} "
          f"p_fwd_med={float(np.median(p_fwd)):.3f} {mb:.1f}MB {dt:.0f}s",
          flush=True)

(OUT / "edgeprob2_counts.json").write_text(json.dumps(
    dict(script_version=SCRIPT_VERSION, head=HEAD, weights_sha256=WEIGHTS_SHA256,
         unet_thresh=UNET_THRESH,
         gate_um=GATE_UM, scale=SCALE.tolist(), downsample=list(map(int, DS)),
         reverse=REVERSE, seed_44b6=SEED_44B6, set_44b6=SET_44B6,
         pool32=POOL32, holdout_6bba=HOLDOUT_6BBA,
         groups=dict(base=[], F4=names_f4), failed=failed,
         skipped_time=skipped_time, datasets=counts), indent=1))

written = sorted(OUT.glob("edgeprob2_*.npz"))
n_f4 = sum(1 for p in written
           if p.name.startswith("edgeprob2_F4hp_"))
n_base = len(written) - n_f4
n_done = len(written)
total_mb = sum(p.stat().st_size for p in written) / 2 ** 20
print(f"\n{n_done} cached ({n_base} base + {n_f4} F4), {total_mb:.1f} MB, "
      f"{time.time() - t_all:.0f}s", flush=True)
if failed:
    print(f"[degraded] {len(failed)} dataset(s) produced nothing: {failed}",
          flush=True)
if skipped_time:
    print(f"INCOMPLETE: {skipped_time} datasets not attempted (budget)",
          flush=True)
if n_done == 0:
    raise SystemExit("[degraded] nothing was cached -- do NOT read this run "
                     "as a success")
complete = (not failed and not skipped_time
            and n_base == 0 and n_f4 == len(names_f4))
if complete:
    print(f"n_per_frame check: {len(names_f4) - len(NPF_BAD)} of "
          f"{len(names_f4) - len(NPF_NOCHK)} checkable datasets "
          f"reproduce the shipped counts"
          + (f" -- MISMATCHES: {NPF_BAD}" if NPF_BAD else " exactly")
          + f"; {len(NPF_NOCHK)} had no shipped edgeprob2_F4_ to check "
            f"against (expected: only the 24 eval movies were ever cached)",
          flush=True)
    print(f"EP2HPCACHE {'OK' if not NPF_BAD else 'NPF MISMATCH'} "
          f"({n_base + n_f4} of {len(jobs)})", flush=True)
else:
    print(f"EP2HPCACHE INCOMPLETE ({n_base} base + {n_f4} F4 of {len(jobs)}; "
          f"{len(failed)} failed, {skipped_time} not attempted) -- NOT OK",
          flush=True)
