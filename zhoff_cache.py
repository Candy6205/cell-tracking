"""zhoff_cache.py -- the OFFICIAL detector's nodes on the 8 Zebrahub evaluation
crops, cached in the detcache layout for a pipeline test on out-of-sample,
densely labelled data.

SCRIPT_VERSION 2026-09-04a-zhoff

The user, 2026-09-04: "难道它的检测真的是神？" and "去吧". What the record
has: on zebrahub, the only venue no detector has seen and every cell is
labelled, the 0.948 notebook's dual-seed blend scores F1 0.696 against
F4's 0.790 at its own best gate (detfuse3: +37% detections, precision
0.59). But detection F1 is measured before tracking deletes anything,
and the notebook's post-processing (one-to-one linking, prune isolated,
min track 6 with rescue) is built to delete. offdet cached the official
nodes only for the 24 competition sets, where the detector is
in-sample. This kernel caches them on the 8 zebrahub eval crops
(fusescore's SET_ZH: sorted zhdet stems [:8]) so `zhoffrank.py` can run
official nodes -> our D187 linker -> patches at SHORT_MIN 4 and 6 against
F4 on the same crops (F4-zh anchor 0.8118).

Per crop and frame, exactly predict_video's detection path on the
crop's [::1, ::4, ::4] strided volume: quantile normalisation (the
crop's own 0.001 / 0.999 over all frames -- the official path uses the
dataset's; per-frame quantiles are what detfuse3 used), the 2-frame
window in which the frame is FIRST seen ((t, t+1) at index 0; the last
frame at index 1 of (T-2, T-1)), 4-view flip TTA on the logits,
`_detect_cells_pooled` with the 3 um max-pool kernel (3, 3, 3), sigmoid
score at the peak, coordinates scaled back by (1, 4, 4).

Arms: OFFP (primary seed) and OFF28 (the notebook's blend: the second
seed's logits re-centred to the primary's per-frame mean and re-scaled
by the clamped std ratio, 0.2 / 0.8, with its retention guard: if the
blend keeps < 90% of the primary's peak count at the gate, the primary
alone is used for that frame), each at gates 0.965 and 0.99. Output
zhoff_<ARM><gate>_<crop>.npz with c{t} (full-res voxel coords, float32)
and s{t} (sigmoid), n_per_frame, plus zhoff_counts.json.

Attach: pilkwang/biohub-tracking-support-pack-50ep-v1,
pilkwang/biohub-temporal-unet3d-seed314159-v1, kernel output
faustine649/biohub-zebrahub-extract. T4.
"""
import gc
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

SCRIPT_VERSION = "2026-09-04a-zhoff"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, flush=True)

_WARNED = set()


def warn_once(key, msg):
    if key not in _WARNED:
        _WARNED.add(key)
        print(f"  WARN {msg}", flush=True)


# ---- the official model code (pack repo) and its wheels -----------------
PACK = pathlib.Path("/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1")
if not PACK.exists():
    PACK = pathlib.Path("/kaggle/input/biohub-tracking-support-pack-50ep-v1")
assert PACK.exists(), "support pack not attached (push with kag.py)"
SECONDARY = pathlib.Path("/kaggle/input/datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1")
if not SECONDARY.exists():
    SECONDARY = pathlib.Path("/kaggle/input/biohub-temporal-unet3d-seed314159-v1")
assert SECONDARY.exists(), "second-seed pack not attached"
KEEP_OUT = {"numpy", "scipy", "torch", "pandas", "pyarrow", "numba", "llvmlite",
            "matplotlib", "pillow", "networkx", "scikit-learn", "scikit-image",
            "setuptools", "wheel", "pip"}
WH = pathlib.Path("/kaggle/working/wheelhouse")
shutil.rmtree(WH, ignore_errors=True)
WH.mkdir(parents=True)
for w in sorted(PACK.rglob("*.whl")):
    if w.name.split("-")[0].replace("_", "-").lower() not in KEEP_OUT:
        shutil.copy(w, WH / w.name)
_r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index",
                     "--no-deps"] + [str(w) for w in sorted(WH.glob("*.whl"))],
                    capture_output=True, text=True)
print(f"pip exit {_r.returncode}", flush=True)
if _r.returncode != 0:
    print(_r.stderr[-1500:], flush=True)
shutil.rmtree(WH, ignore_errors=True)
for _p in (PACK / "repo" / "src", PACK / "repo" / "scripts"):
    assert _p.exists(), _p
    sys.path.insert(0, str(_p))
import predict_unet_transformer as P                            # noqa: E402

DEV = torch.device(DEVICE)
OFF = {}
for _lab, _wp in (("p", PACK / "weights/unet_transformer/split_0/edge_predictor_best.pth"),
                  ("s", SECONDARY / "weights/unet_transformer/split_0/edge_predictor_best.pth")):
    assert _wp.exists(), _wp
    _m, _win, _dsm = P.load_model(_wp, DEV)
    assert _win == 2, (_lab, _win)
    assert tuple(_dsm) == (1, 4, 4), (_lab, _dsm)
    OFF[_lab] = _m
    print(f"official detector {_lab}: {_wp} window {_win} downsample {tuple(_dsm)}",
          flush=True)
DS = (1, 4, 4)
SCALE = (1.625, 0.40625, 0.40625)          # the competition grid the crops sit on
VOXEL_DOWN = tuple(s * d for s, d in zip(SCALE, DS))
POOL_K = P.pool_kernel_from_um(3.0, VOXEL_DOWN)   # PredictConfig.pool_kernel_um default
print(f"pool kernel {POOL_K} on voxel {VOXEL_DOWN}", flush=True)
FLIPS = ((-1,), (-2,), (-2, -1))                    # predict_video's det_tta views
GATES = (0.965, 0.99)
SEC_W = 0.80                                        # BIOHUB_SECONDARY_DETECTION_WEIGHT
MIN_RETENTION = 0.90                                # BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION

# ---- the crops -----------------------------------------------------------
CROPS = None
for _c in ("/kaggle/input/notebooks/faustine649/biohub-zebrahub-extract/crops",
           "/kaggle/input/biohub-zebrahub-extract/crops"):
    if pathlib.Path(_c).exists():
        CROPS = pathlib.Path(_c)
        break
if CROPS is None:
    hits = list(pathlib.Path("/kaggle/input").glob("*/**/crops"))
    CROPS = hits[0] if hits else None
assert CROPS is not None, "zebrahub crops not mounted"
files = {f.stem: f for f in CROPS.glob("*.npz")}
SET_ZH = sorted(files)[:8]                # fusescore / zhrank / gap2: sorted zhdet stems [:8]
EXPECT = ["ZSNS001_tail_z2y0x3", "ZSNS001_tail_z2y1x2", "ZSNS001_tail_z2y2x3",
          "ZSNS001_tail_z2y3x0", "ZSNS001_tail_z2y3x2", "ZSNS001_z2y1x4",
          "ZSNS001_z2y1x5", "ZSNS001_z2y1x6"]
assert SET_ZH == EXPECT, f"the 8 eval crops differ from zhrank's: {SET_ZH}"
print(f"crops: {len(files)} under {CROPS}; eval {SET_ZH}", flush=True)
OUT = pathlib.Path("/kaggle/working")


def strided(v):
    return torch.from_numpy(np.ascontiguousarray(
        np.asarray(v)[:, ::DS[1], ::DS[2]]).astype(np.float32))


def encode_tta(model, imgs):
    """W logits (each (1, 1, Z, Y, X)), the mean over the identity and the
    three flip views, as predict_video does with det_tta."""
    with torch.no_grad():
        _u, det = model.encode(imgs)
        det = [d.clone() for d in det]
        del _u
        for dims in FLIPS:
            _u, det_f = model.encode(imgs.flip(dims))
            for f in range(len(det)):
                det[f] = det[f] + det_f[f].flip(dims)
            del _u, det_f
        for f in range(len(det)):
            det[f] = det[f] / (1 + len(FLIPS))
    return det


def peaks(logit, t, gate):
    """(coords full-res float32 (n, 3), sigmoid scores (n,)) from one frame's
    (1, 1, Z, Y, X) logits, predict_video's own peak extraction."""
    arr = P._detect_cells_pooled(logit[0], t, gate, POOL_K)
    if len(arr) == 0:
        return np.zeros((0, 3), np.float32), np.zeros(0, np.float32)
    g = arr[:, 1:].astype(np.int64)
    lg = logit[0, 0][g[:, 0], g[:, 1], g[:, 2]]
    sc = torch.sigmoid(lg).float().cpu().numpy().astype(np.float32)
    full = (g.astype(np.float32) * np.asarray(DS, np.float32)[None, :]).astype(np.float32)
    return full, sc


COUNTS = {}
t_all = time.time()
for k_, stem in enumerate(SET_ZH):
    t0 = time.time()
    z = np.load(files[stem])
    img = z["image"]
    assert img.ndim == 4, (stem, img.shape)
    T = img.shape[0]
    assert T >= 2, (stem, T)
    # the crop's own 0.001 / 0.999 quantiles over all frames (subsampled)
    sample = np.asarray(img[::max(1, T // 10)]).astype(np.float32)
    q_low, q_high = (float(v) for v in np.quantile(sample, [0.001, 0.999]))
    del sample
    outs = {(arm, g): {} for arm in ("OFFP", "OFF28") for g in GATES}
    guard = {g: [0, 0] for g in GATES}          # frames where the primary was kept, frames
    for t in range(T):
        if t < T - 1:
            fr, idx = (t, t + 1), 0
        else:
            fr, idx = (T - 2, T - 1), 1
        imgs = torch.stack([strided(img[a]) for a in fr])
        imgs = ((imgs - q_low) / (q_high - q_low + 1e-6)).clamp(0.0)
        imgs = imgs.unsqueeze(0).to(DEV)
        det_p = encode_tta(OFF["p"], imgs)[idx]
        det_s = encode_tta(OFF["s"], imgs)[idx]
        del imgs
        # the notebook's alignment (per frame): re-centre and re-scale the
        # secondary to the primary, clamp the std ratio to [0.5, 2]
        p_mean, s_mean = det_p.mean(), det_s.mean()
        p_std = det_p.float().std(unbiased=False).clamp_min(1e-4)
        s_std = det_s.float().std(unbiased=False).clamp_min(1e-4)
        ratio = (p_std / s_std).clamp(0.5, 2.0)
        s_aligned = (det_s - s_mean) * ratio + p_mean
        blended = (1.0 - SEC_W) * det_p + SEC_W * s_aligned
        for g in GATES:
            c, s = peaks(det_p, t, g)
            outs[("OFFP", g)][t] = (c, s)
            n_p = len(c)
            cb, sb = peaks(blended, t, g)
            retention = (len(cb) / n_p) if n_p else 1.0
            guard[g][1] += 1
            if n_p > 0 and retention < MIN_RETENTION:
                guard[g][0] += 1
                outs[("OFF28", g)][t] = (c, s)
            else:
                outs[("OFF28", g)][t] = (cb, sb)
        del det_p, det_s, s_aligned, blended
        if t % 25 == 0:
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
    line = []
    for (arm, g), d in outs.items():
        pay = {"n_per_frame": np.array([len(d[t][0]) for t in range(T)], np.int64),
               "q_low": np.float64(q_low), "q_high": np.float64(q_high),
               "script_version": np.str_(SCRIPT_VERSION)}
        for t in range(T):
            pay[f"c{t}"] = d[t][0]
            pay[f"s{t}"] = d[t][1]
        tag = f"{arm}{int(round(g * 1000))}"
        np.savez_compressed(OUT / f"zhoff_{tag}_{stem}.npz", **pay)
        npf = float(pay["n_per_frame"].mean())
        COUNTS[f"{tag}/{stem}"] = {"T": T, "dets_per_frame": npf,
                                   "guard_primary_frames": guard[g][0] if arm == "OFF28" else 0}
        line.append(f"{tag} {npf:.1f}/fr")
    print(f"[{k_ + 1}/{len(SET_ZH)}] {stem}: T={T} q=({q_low:.1f}, {q_high:.1f}) "
          + "  ".join(line)
          + f"  guard kept primary on {guard[0.965][0]} / {guard[0.99][0]} frames "
          f"(0.965 / 0.99)  {time.time() - t0:.0f}s", flush=True)
    del z, img, outs
    gc.collect()

json.dump({"script_version": SCRIPT_VERSION, "gates": list(GATES), "sec_w": SEC_W,
           "min_retention": MIN_RETENTION, "pool_kernel": list(POOL_K),
           "tta_views": 1 + len(FLIPS), "quantiles": "crop-level 0.001/0.999",
           "counts": COUNTS},
          open(OUT / "zhoff_counts.json", "w"), indent=1)
n_files = len(list(OUT.glob("zhoff_*.npz")))
print(f"\nZHOFF DONE in {(time.time() - t_all) / 60:.1f} min: {n_files} caches "
      f"({len(SET_ZH)} crops x 4 arms)", flush=True)
assert n_files == 4 * len(SET_ZH), n_files
print("ZHOFF OK", flush=True)
