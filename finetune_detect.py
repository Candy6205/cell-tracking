"""
finetune_detect.py -- can training move the detector's REAL performance?

The ruler (v2)
--------------
The 6bba holdout is void for this question (duplicate-inflated, possibly
inside the weights' training split, saturated -- progress.md 2026-08-21
"validation trust"). The number to move is the Zebrahub dense benchmark:
`biohub-zh-detect-bench` measured the public weights at **recall 0.650 /
precision 0.637** at det gate 0.99 over ~206k dense labels. Re-run that
bench with the fine-tuned weights; that delta is the result. The holdout
is only checked afterwards for "did linking collapse".

What changed in v2, and why (progress.md, same day)
---------------------------------------------------
1. **Photometric + sharpness augmentation.** The hidden test's instrument
   settings are unknown (nobody has seen it; only indirect evidence).
   Brightness varies 8x inside the competition and 10x across Zebrahub
   movies, and the detector is measurably brittle to it (dim ZSNS004 is
   the worst bench movie). Sharpness/PSF and noise can differ too and are
   unmeasured. So train under random gain/gamma/noise/blur: learn shape,
   not one instrument's look.
2. **Two-domain training.** Zebrahub dense crops PLUS competition
   sparse-GT frames (non-holdout datasets only). Features that hold on
   two instruments stand a better chance on a third, and the competition
   frames anchor the model to the domain the test is closest to.
3. **Per-domain loss weighting.** Sparse competition GT keeps the repo's
   neg_weight 0.1 (unlabeled cells must not be punished). Zebrahub labels
   are dense, so absence is informative -- but ~10% are noise, so 0.5,
   not 1.0.

Normalisation: the same 0.1%-99.9% per-source quantile stretch everywhere.

Not trained: the edge head (Zebrahub's frame interval is shorter; motion
would be learned at the wrong scale).
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np

SCRIPT_VERSION = "2026-08-21c"
# v3: v2 was OOM-Killed at crop 36 of 41 -- decompressing every npz into
# RAM costs ~800 MB each, ~33 GB total. Now each crop decompresses once to
# a memmap in /kaggle/tmp (not shipped as output) and frames are read
# per-sample.

EPOCHS = 2
LR = 1e-5
VAL_FRAC = 0.15
SPARSE_NEG_WEIGHT = 0.1     # repo default, for sparse competition GT
DENSE_NEG_WEIGHT = 0.5      # Zebrahub: dense labels, ~10% noise
COMP_FRACTION = 0.3         # share of competition samples in each epoch
MAX_STEPS = 1500            # per epoch, subsampled
N_COMP = 16                 # competition datasets mixed in (non-holdout)
N_COMP_VAL = 4              # further datasets held out for comp-domain val
FRAMES_PER_COMP = 12        # labelled frames sampled per competition movie

HOLDOUT_6BBA = {            # NEVER train on these -- the local holdout
    "6bba_268e1230", "6bba_2312ac41", "6bba_283bf9f1", "6bba_3a1849c2",
    "6bba_3abfe10a", "6bba_5c824876", "6bba_61dd1e0d", "6bba_7af54fde",
    "6bba_7b5d3b2c", "6bba_afb141ff", "6bba_aeee7805", "6bba_c27cba08",
}

print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)


def first(*paths):
    for p in paths:
        q = pathlib.Path(p)
        if q.exists():
            return q
    return None


PACK = first("/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1",
             "/kaggle/input/biohub-tracking-support-pack-50ep-v1")
assert PACK is not None, "support pack not attached"

CROPS = None
for d in pathlib.Path("/kaggle/input").rglob("crops"):
    if d.is_dir() and any(d.glob("*.npz")):
        CROPS = d
        break
assert CROPS is not None, "zebrahub crops not attached"

CODE = first("/kaggle/input/datasets/faustine649/cell-tracking/cell tracking",
             "/kaggle/input/datasets/faustine649/cell-tracking",
             "/kaggle/input/cell-tracking/cell tracking",
             "/kaggle/input/cell-tracking")
assert CODE is not None and (CODE / "viz_cells.py").exists(), \
    "code dataset with viz_cells.py not attached"
sys.path.insert(0, str(CODE))

KEEP_OUT = {"numpy", "scipy", "torch", "pandas", "pyarrow", "numba", "llvmlite",
            "matplotlib", "pillow", "networkx", "scikit-learn", "scikit-image",
            "setuptools", "wheel", "pip"}
WH = pathlib.Path("/kaggle/working/wheelhouse")
shutil.rmtree(WH, ignore_errors=True)
WH.mkdir(parents=True)
for w in sorted(PACK.rglob("*.whl")):
    if w.name.split("-")[0].replace("_", "-").lower() not in KEEP_OUT:
        shutil.copy(w, WH / w.name)
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index",
                    "--no-deps"] + [str(w) for w in sorted(WH.glob("*.whl"))],
                   capture_output=True, text=True)
print(f"pip exit {r.returncode}", flush=True)
shutil.rmtree(WH, ignore_errors=True)

import torch                                                   # noqa: E402
from scipy.ndimage import gaussian_filter                      # noqa: E402

for p in (PACK / "repo" / "src", PACK / "repo" / "scripts"):
    sys.path.insert(0, str(p))

import predict_unet_transformer as P                           # noqa: E402
from train_unet_transformer import compute_detection_loss      # noqa: E402
from viz_cells import (find_root, geff_path, list_datasets,    # noqa: E402
                       open_image, read_geff)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda", "no GPU"
cap = torch.cuda.get_device_capability(0)
print(f"gpu: {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}", flush=True)
assert cap[0] >= 7, "the image's torch needs sm_70+; pin NvidiaTeslaT4"

WEIGHTS = PACK / "weights/unet_transformer/split_0/edge_predictor_best.pth"
model, window_size, downsample = P.load_model(WEIGHTS, device)
dz, dy, dx = downsample
print(f"window_size={window_size} downsample={downsample}", flush=True)

rng = np.random.default_rng(0)

# ---------------------------------------------------------------- sources
# Each source: dict(get(t) -> full-res frame float32 raw, by_t, q, neg_w).

sources = []
samples = []                # (source_index, t)


def add_source(name, get, by_t, q, neg_w, n_frames):
    sources.append({"name": name, "get": get, "by_t": by_t, "q": q,
                    "neg_w": neg_w})
    si = len(sources) - 1
    added = 0
    for t in sorted(by_t):
        if t + 1 < n_frames and by_t.get(t):
            samples.append((si, t))
            added += 1
    return added


# -- Zebrahub crops (dense labels), decompressed once to /kaggle/tmp
# memmaps: holding all 41 in RAM is ~33 GB and OOM-killed v2.
TMPD = pathlib.Path("/kaggle/tmp/zh_mm")
TMPD.mkdir(parents=True, exist_ok=True)
for p in sorted(CROPS.glob("*.npz")):
    z = np.load(p)
    nodes = z["nodes"]
    if len(nodes) == 0:
        continue
    mm = TMPD / f"{p.stem}.npy"
    np.save(mm, z["image"])
    del z
    img = np.load(mm, mmap_mode="r")
    s = np.asarray(img[:: max(1, len(img) // 8)]).astype(np.float32).ravel()
    q = tuple(float(v) for v in np.quantile(s, [0.001, 0.999]))
    by_t = {}
    for t, zz, yy, xx in nodes:
        by_t.setdefault(int(t), []).append((zz, yy, xx))
    n = add_source(p.stem,
                   (lambda im: lambda t: np.asarray(im[t]))(img),
                   by_t, q, DENSE_NEG_WEIGHT, len(img))
    print(f"  ZH {p.stem}: {n} frame pairs, q=({q[0]:.0f},{q[1]:.0f})",
          flush=True)

n_zh_samples = len(samples)

# -- competition sparse frames (non-holdout datasets, stride-picked)
root = find_root(None)
names = [n for n in list_datasets(root, "train") if n not in HOLDOUT_6BBA]
b44 = [n for n in names if n.startswith("44b6")]
b6b = [n for n in names if n.startswith("6bba")]
pick = (b44[:: max(1, len(b44) // ((N_COMP + N_COMP_VAL) // 2))]
        + b6b[:: max(1, len(b6b) // ((N_COMP + N_COMP_VAL) // 2))])
pick = pick[: N_COMP + N_COMP_VAL]
comp_val_names = set(pick[N_COMP:])
print(f"\ncompetition datasets: {len(pick)} picked "
      f"({len(comp_val_names)} for val)", flush=True)

comp_val_samples = []
for name in pick:
    try:
        img = open_image(root / "train" / f"{name}.zarr")
        gt = read_geff(geff_path(root, "train", name))
    except Exception as e:                                    # noqa: BLE001
        print(f"  [degraded] {name}: {type(e).__name__}: {e}", flush=True)
        continue
    T = img.shape[0]
    by_t = {}
    for i in range(len(gt["t"])):
        by_t.setdefault(int(gt["t"][i]), []).append(
            (float(gt["z"][i]), float(gt["y"][i]), float(gt["x"][i])))
    ts = sorted(t for t in by_t if t + 1 < T)
    ts = ts[:: max(1, len(ts) // FRAMES_PER_COMP)][:FRAMES_PER_COMP]
    by_t = {t: by_t[t] for t in ts}
    fr = np.asarray(img[ts[0]], dtype=np.float32).ravel() if ts else None
    if fr is None:
        continue
    q = tuple(float(v) for v in np.quantile(fr, [0.001, 0.999]))

    def make_get(im):
        return lambda t: np.asarray(im[t])

    dest = comp_val_samples if name in comp_val_names else samples
    before = len(samples)
    if name in comp_val_names:
        sources.append({"name": name, "get": make_get(img), "by_t": by_t,
                        "q": q, "neg_w": SPARSE_NEG_WEIGHT})
        si = len(sources) - 1
        comp_val_samples += [(si, t) for t in by_t]
        n = len(by_t)
    else:
        n = add_source(name, make_get(img), by_t, q, SPARSE_NEG_WEIGHT, T)
    print(f"  COMP {name}: {n} frames"
          f"{' (val)' if name in comp_val_names else ''}", flush=True)

n_comp_samples = len(samples) - n_zh_samples
print(f"\nsamples: {n_zh_samples} ZH + {n_comp_samples} comp train, "
      f"{len(comp_val_samples)} comp val", flush=True)
assert n_zh_samples and n_comp_samples, "one domain is empty"

# ZH val split
zh_idx = np.arange(n_zh_samples)
rng.shuffle(zh_idx)
n_val = max(1, int(n_zh_samples * VAL_FRAC))
zh_val = zh_idx[:n_val].tolist()
zh_train = zh_idx[n_val:].tolist()
comp_train = list(range(n_zh_samples, len(samples)))

# ---------------------------------------------------------------- batches


def augment(pair):
    """Photometric + sharpness jitter, on the stretched pair in-place.
    gain/offset/gamma/noise cover instrument photometrics; blur covers
    PSF/sharpness -- both unknown for the hidden test."""
    gain = rng.uniform(0.6, 1.6)
    off = rng.uniform(-0.05, 0.05)
    gamma = rng.uniform(0.7, 1.4)
    pair = np.clip(pair * gain + off, 0.0, None) ** gamma
    sig = rng.uniform(0.0, 1.2)
    if sig > 0.3:
        pair = np.stack([gaussian_filter(f, sigma=(sig / 2, sig, sig))
                         for f in pair])
    ns = rng.uniform(0.0, 0.04)
    if ns > 0.005:
        pair = pair + rng.normal(0.0, ns, size=pair.shape)
    return np.clip(pair, 0.0, None)


def batch_of(i, train, sample_list=None):
    si, t = (sample_list or samples)[i]
    c = sources[si]
    q_low, q_high = c["q"]
    pair = np.stack([c["get"](t), c["get"](t + 1)]).astype(np.float32)
    pair = pair[:, ::dz, ::dy, ::dx]
    pair = np.clip((pair - q_low) / (q_high - q_low + 1e-6), 0.0, None)
    if train:
        pair = augment(pair)
    pts = np.asarray(c["by_t"][t], dtype=np.float32)
    pts = pts / np.array([dz, dy, dx], dtype=np.float32)
    x = torch.from_numpy(np.ascontiguousarray(pair, dtype=np.float32))
    return (x[None].to(device),
            torch.from_numpy(pts)[None].to(device),
            torch.ones(1, len(pts), dtype=torch.bool, device=device),
            c["neg_w"])


def evaluate(idx, sample_list=None):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in idx:
            imgs, pts, mask, neg_w = batch_of(i, train=False,
                                              sample_list=sample_list)
            _, det = model.encode(imgs)
            tot += float(compute_detection_loss(det[0], pts, mask,
                                                neg_weight=neg_w))
            n += 1
    model.train()
    return tot / max(1, n)


# ---------------------------------------------------------------- train

params = list(model.unet.parameters()) + list(model.detect_head.parameters())
opt = torch.optim.AdamW(params, lr=LR)
print(f"\ntraining {sum(p.numel() for p in params):,} parameters "
      f"(unet + detect_head), lr={LR}, comp fraction {COMP_FRACTION}",
      flush=True)

zh_before = evaluate(zh_val)
cv_before = evaluate(range(len(comp_val_samples)), comp_val_samples)
print(f"val loss before: ZH {zh_before:.5f}  COMP {cv_before:.5f}",
      flush=True)

n_comp_per_ep = int(MAX_STEPS * COMP_FRACTION)
n_zh_per_ep = MAX_STEPS - n_comp_per_ep

for ep in range(EPOCHS):
    model.train()
    ep_idx = ([int(i) for i in rng.choice(zh_train, n_zh_per_ep,
                                          replace=len(zh_train) < n_zh_per_ep)]
              + [int(i) for i in rng.choice(comp_train, n_comp_per_ep,
                                            replace=len(comp_train) < n_comp_per_ep)])
    rng.shuffle(ep_idx)
    run, t0 = 0.0, time.time()
    for k, i in enumerate(ep_idx, 1):
        imgs, pts, mask, neg_w = batch_of(i, train=True)
        _, det = model.encode(imgs)
        loss = compute_detection_loss(det[0], pts, mask, neg_weight=neg_w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        run += float(loss)
        if k % 100 == 0 or k == len(ep_idx):
            el = time.time() - t0
            print(f"  ep{ep+1} {k}/{len(ep_idx)} loss {run/k:.5f} "
                  f"{el:.0f}s eta {el/k*(len(ep_idx)-k):.0f}s", flush=True)
    zh_v = evaluate(zh_val)
    cv_v = evaluate(range(len(comp_val_samples)), comp_val_samples)
    print(f"ep{ep+1} val: ZH {zh_v:.5f} ({zh_v-zh_before:+.5f})  "
          f"COMP {cv_v:.5f} ({cv_v-cv_before:+.5f})", flush=True)

zh_after = evaluate(zh_val)
cv_after = evaluate(range(len(comp_val_samples)), comp_val_samples)
print(f"\nval loss ZH   {zh_before:.5f} -> {zh_after:.5f} "
      f"({zh_after-zh_before:+.5f})", flush=True)
print(f"val loss COMP {cv_before:.5f} -> {cv_after:.5f} "
      f"({cv_after-cv_before:+.5f})", flush=True)
if cv_after > cv_before * 1.1:
    print("[degraded] competition-domain loss rose >10% -- the fine-tune "
          "is drifting off the test's home domain; do not ship without "
          "checking the holdout.", flush=True)

out = pathlib.Path("/kaggle/working/edge_predictor_finetuned.pth")
sd = torch.load(WEIGHTS, map_location="cpu")
key = "model" if isinstance(sd, dict) and "model" in sd else None
state = {k: v.cpu() for k, v in model.state_dict().items()}
if key:
    sd["model"] = state
    torch.save(sd, out)
else:
    torch.save(state, out)
print(f"wrote {out} ({out.stat().st_size/2**20:.0f} MB)", flush=True)
print("\nNext: re-run biohub-zh-detect-bench with these weights (baseline "
      "recall 0.650 / prec 0.637 @0.99), then the holdout for link_rate, "
      "then package as a dataset and submit if the gain is large.",
      flush=True)
print("FINETUNE DONE", flush=True)
