"""
finetune_detect.py -- can training move the detection ceiling?

The question
------------
Post-processing is exhausted. `threshold` x `disappearance` converged at an
interior optimum (0.40 / 3.5 -> adj_J 0.9387), and those two knobs only ever
MOVE loss between detection and linking. `det_threshold` cannot raise the
ceiling at all:

    det    detect_ceiling
    0.99      0.9825
    0.98      0.9827
    0.95      0.9821

The 180 GT edges whose endpoint was never detected are not peaks in the heatmap
at any threshold. So the ceiling is the model's, and the only way past it is a
better detector.

Zebrahub supplies the supervision the competition's own labels cannot:
401,780 annotated nodes at up to 1211 per frame, against 133,318 across the
whole competition training set at 6.7 per frame.

The single number this run has to move: **detect_ceiling above 0.9825**.

What could go wrong, and is measured
------------------------------------
The U-Net is SHARED. `unet_out` feeds the detection head and the edge
transformer alike, so fine-tuning it on a detection loss alone can degrade
linking. This trains it anyway -- the detection head by itself is one 1x1
convolution and cannot learn new features -- but the run is worthless unless
`link_rate` is checked afterwards as well. Both come out of the same scoring
pass.

Normalisation matters more than usual here: a different microscope. Train and
predict both apply the same 0.1%-99.9% quantile stretch, so the crops get the
same treatment computed from their own statistics, which is what puts the two
instruments on one scale.

Not trained here: the edge head. Zebrahub's frame interval is shorter --
displacement p50 1.24 um against the competition's 1.82 -- so anything learned
about MOTION would be learned at the wrong scale.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np

SCRIPT_VERSION = "2026-08-21a"

EPOCHS = 2
LR = 1e-5                  # low: the weights are converged, this is a nudge
BATCH = 1                  # (B, W=2, 64, 256, 256) with gradients is heavy
VAL_FRAC = 0.15
NEG_WEIGHT = 0.1           # the repo's own default

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
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index",
                    "--no-deps"] + [str(w) for w in staged],
                   capture_output=True, text=True)
print(f"pip exit {r.returncode} ({len(staged)} wheels)", flush=True)
shutil.rmtree(WH, ignore_errors=True)

import torch                                                   # noqa: E402

for p in (PACK / "repo" / "src", PACK / "repo" / "scripts"):
    assert p.exists(), p
    sys.path.insert(0, str(p))

import predict_unet_transformer as P                           # noqa: E402
from train_unet_transformer import compute_detection_loss      # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda", "no GPU"
cap = torch.cuda.get_device_capability(0)
print(f"gpu: {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}", flush=True)
assert cap[0] >= 7, "the image's torch needs sm_70+; pin NvidiaTeslaT4"

WEIGHTS = PACK / "weights/unet_transformer/split_0/edge_predictor_best.pth"
model, window_size, downsample = P.load_model(WEIGHTS, device)
print(f"window_size={window_size} downsample={downsample}", flush=True)

# ---------------------------------------------------------------- data

dz, dy, dx = downsample
samples = []          # (crop_index, frame) pairs that actually carry labels
crops = []

for p in sorted(CROPS.glob("*.npz")):
    z = np.load(p)
    nodes = z["nodes"]
    if len(nodes) == 0:
        print(f"  skip {p.stem}: no labels", flush=True)
        continue
    img = z["image"]
    # The same 0.1%-99.9% stretch predict_video applies, from this crop's own
    # statistics. A different microscope otherwise lands on a different scale
    # and the fine-tune would be learning the offset rather than the cells.
    s = img[:: max(1, len(img) // 8)].astype(np.float32).ravel()
    q_low, q_high = np.quantile(s, [0.001, 0.999])
    crops.append({"img": img, "nodes": nodes, "q": (float(q_low),
                                                    float(q_high)),
                  "name": p.stem})
    by_t = {}
    for t, zz, yy, xx in nodes:
        by_t.setdefault(int(t), []).append((zz, yy, xx))
    crops[-1]["by_t"] = by_t
    for t in range(len(img) - 1):
        if by_t.get(t):
            samples.append((len(crops) - 1, t))
    print(f"  {p.stem}: {len(img)} frames, {len(nodes):,} nodes, "
          f"q=({q_low:.0f}, {q_high:.0f})", flush=True)

assert samples, "no labelled frames"
rng = np.random.default_rng(0)
order = rng.permutation(len(samples))
n_val = max(1, int(len(samples) * VAL_FRAC))
val_idx, train_idx = order[:n_val], order[n_val:]
print(f"\n{len(samples)} labelled frame pairs -> {len(train_idx)} train, "
      f"{len(val_idx)} val", flush=True)


def batch_of(i):
    ci, t = samples[i]
    c = crops[ci]
    q_low, q_high = c["q"]
    imgs = np.stack([c["img"][t], c["img"][t + 1]]).astype(np.float32)
    imgs = imgs[:, ::dz, ::dy, ::dx]
    imgs = np.clip((imgs - q_low) / (q_high - q_low + 1e-6), 0.0, None)
    pts = np.asarray(c["by_t"][t], dtype=np.float32)
    pts = pts / np.array([dz, dy, dx], dtype=np.float32)
    return (torch.from_numpy(imgs)[None].to(device),
            torch.from_numpy(pts)[None].to(device),
            torch.ones(1, len(pts), dtype=torch.bool, device=device))


def evaluate(idx):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in idx:
            imgs, pts, mask = batch_of(i)
            _, det = model.encode(imgs)
            tot += float(compute_detection_loss(det[0], pts, mask,
                                                neg_weight=NEG_WEIGHT))
            n += 1
    model.train()
    return tot / max(1, n)


# ---------------------------------------------------------------- train

# The detection head alone is a single 1x1 convolution: it can rescale what the
# UNet already reports but cannot make a cell visible that the features do not
# encode. The ceiling is a features problem, so the UNet trains too -- and that
# is exactly why link_rate has to be re-measured afterwards.
params = list(model.unet.parameters()) + list(model.detect_head.parameters())
opt = torch.optim.AdamW(params, lr=LR)
print(f"\ntraining {sum(p.numel() for p in params):,} parameters "
      f"(unet + detect_head), lr={LR}", flush=True)

before = evaluate(val_idx)
print(f"val detection loss before: {before:.5f}", flush=True)

for ep in range(EPOCHS):
    model.train()
    rng.shuffle(train_idx)
    run, t0 = 0.0, time.time()
    for k, i in enumerate(train_idx, 1):
        imgs, pts, mask = batch_of(i)
        _, det = model.encode(imgs)
        loss = compute_detection_loss(det[0], pts, mask, neg_weight=NEG_WEIGHT)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        run += float(loss)
        if k % 50 == 0 or k == len(train_idx):
            el = time.time() - t0
            print(f"  ep{ep+1} {k}/{len(train_idx)} loss {run/k:.5f} "
                  f"{el:.0f}s eta {el/k*(len(train_idx)-k):.0f}s", flush=True)
    v = evaluate(val_idx)
    print(f"ep{ep+1} val detection loss {v:.5f}  ({v-before:+.5f} vs start)",
          flush=True)

after = evaluate(val_idx)
print(f"\nval detection loss {before:.5f} -> {after:.5f} "
      f"({after-before:+.5f})", flush=True)
if after >= before:
    print("[degraded] the loss did not fall. Either the learning rate is wrong "
          "or the features already encode what these labels say, and nothing "
          "downstream will improve.", flush=True)

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
print("\nNext, and the run means nothing without it: re-cache the 12 held-out "
      "datasets with these weights and score. detect_ceiling must beat 0.9825, "
      "AND link_rate must not have fallen from 0.9839.", flush=True)
print("FINETUNE DONE", flush=True)
