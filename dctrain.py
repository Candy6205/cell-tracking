"""dctrain.py -- retrain DeepCenter on a 175-dataset split we control.

SCRIPT_VERSION 2026-08-28a-dctrain

WHY. The published DeepCenter was trained on the 71 `44b6_` movies and
model-selected on the 128 `6bba_` -- not by design but by a split bug
visible in its own shipped trainer: `split_samples` groups movies by
name prefix, the prefix has exactly two values, and "hold out 10% of
groups" becomes "hold out one entire imaging batch". Consequently every
domain we evaluate in is either its training set (44b6, forever nan) or
its model-selection set (6bba, tainted), and `dcmerge` v2's opposite
signs (zh +0.0379 / 6bba -0.0413) could never be settled. The user's
call, 2026-08-28: retrain it on a split we choose, then re-measure.

THE SPLIT (199 = 167 train + 8 val + 24 excluded-for-eval):

  * EXCLUDED, never seen in training or model selection:
      - the 12-dataset `6bba` holdout, verbatim from poolsize.py --
        the recorded-anchor domain;
      - 12 `44b6` drawn with random.Random(20260827).sample from the
        44b6 names OUTSIDE D32's training pool. Random per the user's
        standing instruction ("所有测试集的44b6都要随机抽"); restricted
        to outside-D32 because evaluating a detection change needs the
        D32 ranker, and its pool must stay out of the eval arm
        (working-agreement: the ranker leak).
      - Zebrahub needs no exclusion: no competition movie is Zebrahub.
    With these out, all THREE eval domains are clean -- the first
    detector in this project that can face the full three-domain rule.
  * VAL (8 = 4 `44b6` + 4 `6bba`, random.Random(20260828) from the
    remaining pool): val_loss selects best.pt. Excluded from training
    so model selection is not on training frames, and NOT part of any
    eval domain so selection does not taint the eval.
  * TRAIN: the other 167.

THE RECIPE. Verbatim the author's shipped defaults (their run wrapper's
env defaults equal the trainer's argparse defaults, checked line by
line): pool_factor 4, base 24, sigma 1.0, PU weights 12/1/0.05,
dynamic-range norm 50..99.5 pct clipped [-0.5, 6], batch 8, lr 1e-3, no
weight decay, no grad clip, random flips, no jitter. Two deliberate
deviations, both about the KNOWN failure mode (their 500-epoch run
diverged; best.pt landed at epoch 2):
  * --snapshot-interval 5 --snapshot-final, so every 5-epoch state and
    its gate diagnostics survive whatever the loss curve does;
  * epochs 30, not 50: 167 movies/epoch vs their 71 is 2.35x the data
    exposure per epoch, so 30 here is ~70 of theirs. checkpoint_last.pt
    persists in the kernel output; a --resume push continues if the
    history says it was still improving.

PLUMBING. The shipped trainer is imported from the DeepCenter dataset
mount and used as-is; only `discover_samples` and `split_samples` are
replaced (explicit name lists; labels via viz_cells.read_geff, the
zarr-free reader, instead of the trainer's tracksdata/zarr readers,
neither of which exists on a bare Kaggle image). Its frame reader
already matches the competition layout -- one blosc chunk per frame at
<zarr>/0/c/<t>/0/0/0 -- via blosc2, which the Kaggle image ships.

Attach: competition data, faustine649/cell-tracking (code),
pilkwang/biohub-deepcenter-unet3d-center-prior-v1 (trainer source),
faustine649/biohub-poolsize (edge_ranker_d32.pkl, for the D32 pool).
GPU: T4 pinned via machine_shape (toolchain traps 3/9).
"""
from __future__ import annotations

import importlib.util
import pathlib
import pickle
import random
import sys
import time

import numpy as np

SCRIPT_VERSION = "2026-08-28a-dctrain"
print(f"SCRIPT_VERSION {SCRIPT_VERSION}", flush=True)

# ---------------------------------------------------------------- GPU gate
import torch  # noqa: E402

assert torch.cuda.is_available(), (
    "no CUDA device -- this is a training kernel, enable_gpu must be true")
cap = torch.cuda.get_device_capability(0)
assert cap[0] >= 7, (
    f"GPU capability {cap} < sm_70 (a P100): the image's torch cannot run "
    f"on it. kernel-metadata.json must pin machine_shape NvidiaTeslaT4.")
print(f"GPU: {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}", flush=True)

# ------------------------------------------------- our code (zarr-free IO)
CODE_ROOTS = [
    "/kaggle/input/datasets/faustine649/cell-tracking/cell tracking",
    "/kaggle/input/datasets/faustine649/cell-tracking",
    "/kaggle/input/cell-tracking/cell tracking",
    "/kaggle/input/cell-tracking",
]
code = next((c for c in CODE_ROOTS
             if pathlib.Path(c, "viz_cells.py").exists()), None)
assert code is not None, (
    "viz_cells.py not found -- attach faustine649/cell-tracking")
sys.path.insert(0, code)
from viz_cells import find_root, geff_path, read_geff  # noqa: E402

root = find_root(None)
TRAIN_DIR = pathlib.Path(root) / "train"
print(f"competition root: {root}", flush=True)

# ------------------------------------------------- the shipped trainer
DC_ROOTS = [
    "/kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
    "/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1",
]
trainer_py = next(
    (p for c in DC_ROOTS
     for p in [pathlib.Path(c, "source_scripts",
                            "train_full_frame_center_detector.py")]
     if p.exists()), None)
assert trainer_py is not None, (
    "train_full_frame_center_detector.py not found -- attach "
    "pilkwang/biohub-deepcenter-unet3d-center-prior-v1")
spec = importlib.util.spec_from_file_location("dc_trainer", trainer_py)
dc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc)
print(f"trainer loaded from {trainer_py}", flush=True)

# ------------------------------------------------- D32 pool (ranker leak)
# Bounded search: top-level mounts only, skipping the competition tree
# (trap 20: a blind recursive walk over the zarr chunks appears to hang).
POOL32: set[str] = set()
for top in sorted(pathlib.Path("/kaggle/input").iterdir()):
    if "cell-tracking-during-development" in top.name:
        continue
    for p in top.rglob("edge_ranker_d32.pkl"):
        with open(p, "rb") as f:
            blob = pickle.load(f)
        POOL32 = set(blob.get("pool") or [])
        print(f"D32 pool read from {p}: {len(POOL32)} datasets", flush=True)
        break
    if POOL32:
        break
assert POOL32, (
    "edge_ranker_d32.pkl not found -- attach faustine649/biohub-poolsize. "
    "Without the D32 pool the 44b6 exclusion cannot avoid the ranker's "
    "training set and the eval arm would be leak-tainted (silent-degradation "
    "rule: fail here, not there).")

# ------------------------------------------------- the split, printed whole
names = sorted(p.name[:-len(".zarr")] for p in TRAIN_DIR.glob("*.zarr"))
assert len(names) == 199, f"expected 199 train movies, found {len(names)}"

HOLDOUT_6BBA = [
    "6bba_268e1230", "6bba_2312ac41", "6bba_283bf9f1", "6bba_3a1849c2",
    "6bba_3abfe10a", "6bba_5c824876", "6bba_61dd1e0d", "6bba_7af54fde",
    "6bba_7b5d3b2c", "6bba_afb141ff", "6bba_aeee7805", "6bba_c27cba08",
]
assert all(n in names for n in HOLDOUT_6BBA), "holdout name missing from train/"

SEED_44B6 = 20260827
eligible_44b6 = sorted(n for n in names
                       if n.startswith("44b6_") and n not in POOL32)
assert len(eligible_44b6) >= 12, (
    f"only {len(eligible_44b6)} 44b6 datasets outside D32's pool")
SET_44B6 = sorted(random.Random(SEED_44B6).sample(eligible_44b6, 12))

EXCLUDED = set(HOLDOUT_6BBA) | set(SET_44B6)
pool = [n for n in names if n not in EXCLUDED]
assert len(pool) == 175, f"pool is {len(pool)}, expected 175"

SEED_VAL = 20260828
rng = random.Random(SEED_VAL)
VAL = sorted(rng.sample([n for n in pool if n.startswith("44b6_")], 4)
             + rng.sample([n for n in pool if n.startswith("6bba_")], 4))
TRAIN_NAMES = [n for n in pool if n not in set(VAL)]
assert len(TRAIN_NAMES) == 167 and len(VAL) == 8

print(f"\nsplit: {len(TRAIN_NAMES)} train + {len(VAL)} val + "
      f"{len(EXCLUDED)} excluded-for-eval = {len(names)}", flush=True)
print(f"excluded 6bba holdout ({len(HOLDOUT_6BBA)}): "
      + " ".join(HOLDOUT_6BBA), flush=True)
print(f"excluded 44b6 seed={SEED_44B6} from {len(eligible_44b6)} "
      f"outside-D32 ({len(SET_44B6)}): " + " ".join(SET_44B6), flush=True)
print(f"val seed={SEED_VAL} ({len(VAL)}): " + " ".join(VAL), flush=True)
print(f"train ({len(TRAIN_NAMES)}): " + " ".join(TRAIN_NAMES), flush=True)

# ------------------------------------------------- labels via read_geff
def _centers_by_t(name: str) -> dict[int, np.ndarray]:
    gt = read_geff(geff_path(root, "train", name))
    out: dict[int, list[list[float]]] = {}
    for t, z, y, x in zip(gt["t"], gt["z"], gt["y"], gt["x"]):
        out.setdefault(int(t), []).append([float(z), float(y), float(x)])
    return {t: np.asarray(c, dtype=np.float32) for t, c in out.items() if c}


def my_discover(data_dir, cfg):
    """Explicit lists instead of directory listing; viz_cells labels."""
    samples = []
    t0 = time.time()
    for i, name in enumerate(TRAIN_NAMES + VAL):
        zarr_path = TRAIN_DIR / f"{name}.zarr"
        shape, dtype = dc.read_zarr_meta(zarr_path)
        centers = _centers_by_t(name)
        assert centers, f"{name}: read_geff returned no labelled frames"
        samples.append({
            "name": name, "zarr": zarr_path,
            "geff": geff_path(root, "train", name),
            "shape": shape, "dtype": dtype, "centers_by_t": centers,
        })
        if (i + 1) % 40 == 0 or i + 1 == len(TRAIN_NAMES) + len(VAL):
            print(f"  labels {i + 1}/{len(TRAIN_NAMES) + len(VAL)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return samples


def my_split(samples, val_fraction, seed):
    val_set = set(VAL)
    train = [s for s in samples if s["name"] not in val_set]
    val = [s for s in samples if s["name"] in val_set]
    assert len(train) == 167 and len(val) == 8
    return train, val


dc.discover_samples = my_discover
dc.split_samples = my_split

# ------------------------------------------------- smoke test before 5h run
_smoke = TRAIN_DIR / f"{TRAIN_NAMES[0]}.zarr"
_shape, _dtype = dc.read_zarr_meta(_smoke)
_frame = dc.read_frame(_smoke, 0, _shape, _dtype)
assert _frame.shape == _shape[1:] and _frame.max() > 0, (
    f"smoke read failed: {_frame.shape} vs {_shape[1:]}, max {_frame.max()}")
print(f"smoke frame {TRAIN_NAMES[0]}: shape {_frame.shape} dtype "
      f"{_frame.dtype} max {int(_frame.max())}", flush=True)

# ------------------------------------------------- run, author's defaults
sys.argv = [
    "train_full_frame_center_detector.py",
    "--data-dir", str(TRAIN_DIR),
    "--output-dir", "/kaggle/working/weights/full_frame_center_175",
    "--epochs", "30",
    "--overwrite",
    "--snapshot-interval", "5",
    "--snapshot-final",
]
args = dc.parse_args()
t0 = time.time()
dc.train(args)
print(f"\nDCTRAIN DONE in {(time.time() - t0) / 3600:.2f} h", flush=True)
