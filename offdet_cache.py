"""
offdet_cache.py -- the OFFICIAL detector's nodes, with the official head's
bidirectional probabilities over them, for the 24 evaluation datasets.

WHY (2026-09-04). detfuse3 v4's gate curves: at equal detections per
frame the official detector (pack split_0 TemporalUNet3D) recalls 8-12%
FEWER zebrahub cells than F4, but on the sparse competition sets it
recalls MORE annotated cells -- 44b6: 297 of 297 at gate 0.99 with 9%
fewer detections than F4 (F4: 0.9596); 6bba: 0.9943 vs 0.9761 at +10%
detections. The record's LB verdict against this detector (official
family 0.898 vs xl family 0.917, 2026-08-22/24) changed the detector AND
the linker at once. rank2 (2026-09-04) showed the bidirectional head
alone links within 0.002 / 0.006 of ranker+head, so the linker can now
be held fixed without retraining: official detections -> bidirectional
head -> our ILP and patches, against F4 -> the same (HEAD1, 0.9245 /
0.8453). This kernel caches the GPU half.

WHAT IT DOES. For each of the 24 evaluation datasets (HOLDOUT_6BBA + the
seeded 44b6 draw) and each detection gate in GATES:

  forward pass  predict_video with the OFFICIAL detection
                (det_threshold = gate, det_tta True, pool kernel 3 um), its
                `_detect_cells_pooled` WRAPPED (not replaced) to record
                every frame's peaks and their sigmoid scores, and
                `model.predict_edges` wrapped to keep the full softmax
                matrix (over sources, dim 0) of every consecutive pair;
  reverse pass  the same node set injected (edgeprob_cache.Run, frames
                remapped f -> T-1-f, det_tta off since detection is
                bypassed), full matrices kept.

Output `offdet_<gate>_<name>.npz`, readable BOTH as a detection cache
(detcache layout, so the CPU harness's load_dets reads it unchanged)
and as an edgeprob cache (so its _load_head reads it unchanged):

    c{t}  (N_t, 3) float32  full-res voxel coords = grid * downsample
                            (the official pipeline's own convention:
                            predict_video returns trunc(grid * ds))
    s{t}  (N_t,)   float32  sigmoid(det logit) at the peak, >= gate
    t, i, j, p_fwd, p_rev, l_fwd, l_rev, n_per_frame  as edgeprob_*.npz,
          over pairs within GATE_UM (10 um, physical) -- (i, j) index
          c{t} / c{t+1} directly (every node is a primary: s >= gate >=
          UNET_THRESH)

plus offdet_counts.json. No repair pool exists below the gate: the CPU
side's gap bridges fall back to midpoints (GAP_NEED_CAND False, shipped).

Push with `py kag.py kernels push -p .` -- GPU T4, dataset_sources the
support pack, competition_sources the competition, kernel_sources
biohub-poolsize (edge_ranker_d32.pkl defines the 44b6 draw).
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

SCRIPT_VERSION = "2026-09-04a-offdet"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

UNET_THRESH = 0.15
GATE_UM = 10.0
SCALE = np.array([1.625, 0.40625, 0.40625])
SEED_44B6 = 20260827
N_44B6 = 12
N_POOL32 = 32
GATES = (0.965, 0.99)           # the 0.948 notebook's gate; dets/fr ~ F4's
NO_CANDIDATES = 2.0
BUDGET_S = 10.0 * 3600
SMOKE = False

PACK = pathlib.Path("/kaggle/input/datasets/pilkwang/"
                    "biohub-tracking-support-pack-50ep-v1")
if not PACK.exists():
    PACK = pathlib.Path("/kaggle/input/biohub-tracking-support-pack-50ep-v1")
assert PACK.exists(), "support pack not attached -- push with kag.py"

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

assert callable(getattr(P, "_detect_cells_pooled", None))
assert callable(getattr(P, "_load_frame", None))
WEIGHTS = PACK / "weights/unet_transformer/split_0/edge_predictor_best.pth"
assert WEIGHTS.exists(), WEIGHTS

COMP = pathlib.Path("/kaggle/input/competitions/"
                    "biohub-cell-tracking-during-development")
if not COMP.exists():
    COMP = pathlib.Path("/kaggle/input/biohub-cell-tracking-during-development")
TRAIN = COMP / "train"
assert TRAIN.exists(), TRAIN


def _walk_files(match):
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


# ---------------------------------------------------------------- datasets
HOLDOUT_6BBA = [
    "6bba_268e1230", "6bba_2312ac41", "6bba_283bf9f1", "6bba_3a1849c2",
    "6bba_3abfe10a", "6bba_5c824876", "6bba_61dd1e0d", "6bba_7af54fde",
    "6bba_7b5d3b2c", "6bba_afb141ff", "6bba_aeee7805", "6bba_c27cba08",
]
_pk = _walk_files(lambda fn: fn if fn == "edge_ranker_d32.pkl" else None)
assert _pk, "edge_ranker_d32.pkl not found -- attach biohub-poolsize"
with open(next(iter(_pk.values())), "rb") as _f:
    POOL32 = sorted(set(pickle.load(_f).get("pool") or []))
assert len(POOL32) == N_POOL32
ALL_TRAIN = sorted(p.name[:-len(".zarr")] for p in TRAIN.glob("*.zarr"))
assert len(ALL_TRAIN) == 199
eligible_44b6 = sorted(n for n in ALL_TRAIN
                       if n.startswith("44b6_") and n not in set(POOL32))
SET_44B6 = sorted(random.Random(SEED_44B6).sample(eligible_44b6, N_44B6))
EVAL_24 = sorted(HOLDOUT_6BBA + SET_44B6)
assert len(EVAL_24) == 24
print(f"44b6 seed={SEED_44B6} ({len(SET_44B6)}): {' '.join(SET_44B6)}", flush=True)
names = list(EVAL_24)
if SMOKE:
    names = [HOLDOUT_6BBA[0], SET_44B6[0]]
    print("[SMOKE] 2 datasets", flush=True)
jobs = [(g, n) for g in GATES for n in names]
print(f"jobs: {len(names)} datasets x {len(GATES)} gates = {len(jobs)}", flush=True)

# ---------------------------------------------------------------- GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise SystemExit("[degraded] no GPU")
print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)
model, window_size, downsample = P.load_model(WEIGHTS, device)
assert window_size == 2, window_size
DS = np.asarray(downsample, dtype=np.float32)
assert DS.shape == (3,) and np.all(DS >= 1), downsample
print(f"downsample={tuple(downsample)} window={window_size}", flush=True)

_ORIG_DETECT = P._detect_cells_pooled
_ORIG_LOAD = P._load_frame
_ORIG_EDGES = model.predict_edges


def gated_pairs(prim):
    out = []
    for t in range(len(prim) - 1):
        Pu = prim[t] * SCALE[None, :]
        Cu = prim[t + 1] * SCALE[None, :]
        if len(Pu) == 0 or len(Cu) == 0:
            out.append((np.zeros(0, np.int32), np.zeros(0, np.int32)))
            continue
        hits = cKDTree(Cu).query_ball_point(Pu, GATE_UM)
        if not any(len(h) for h in hits):
            out.append((np.zeros(0, np.int32), np.zeros(0, np.int32)))
            continue
        ii = np.concatenate([np.full(len(h), i, np.int32) for i, h in enumerate(hits)])
        jj = np.concatenate([np.asarray(h, np.int32) for h in hits])
        out.append((ii, jj))
    return out


class DetectRun:
    """Forward pass with the OFFICIAL detection kept: records each frame's
    peaks (grid coords -> full-res) and sigmoid scores, and the full
    softmax / raw matrix of every consecutive pair in call order."""

    def __init__(self, name, gate):
        self.name, self.gate = name, gate
        # T is not known up front (no open_dataset here): frames are stored
        # as they are detected; predict_video detects every frame once.
        self.full = {}                  # f -> full-res float32 (N_t, 3)
        self.score = {}
        self.T = 0
        self.next_f = 0                 # next pair (f, f+1) to expect
        self.mat = {}
        self.raw = {}

    def _have(self, f):
        return f in self.full

    def detect(self, det_logits, f, det_threshold, pool_kernel):
        arr = _ORIG_DETECT(det_logits, f, det_threshold, pool_kernel)
        assert det_threshold == self.gate, (det_threshold, self.gate)
        assert f not in self.full, f"{self.name}: frame {f} detected twice"
        self.T = max(self.T, f + 1)
        if len(arr) == 0:
            self.full[f] = np.zeros((0, 3), np.float32)
            self.score[f] = np.zeros(0, np.float32)
            return arr
        g = arr[:, 1:].astype(np.int64)
        lg = det_logits[0][g[:, 0], g[:, 1], g[:, 2]]
        self.score[f] = torch.sigmoid(lg).float().cpu().numpy().astype(np.float32)
        assert np.all(self.score[f] > self.gate - 1e-6), (
            f"{self.name} f={f}: a kept peak scores below the gate")
        self.full[f] = (g.astype(np.float32) * DS[None, :]).astype(np.float32)
        return arr

    def edges(self, feat_src, feat_tgt, coords_src, coords_tgt, *rest):
        out = _ORIG_EDGES(feat_src, feat_tgt, coords_src, coords_tgt, *rest)
        # predict_video skips a pair when either frame is empty; both frames
        # of the pair it is scoring have been detected by now
        f = self.next_f
        assert self._have(f) and self._have(f + 1), (
            f"{self.name}: predict_edges for pair {f} before its detections")
        while len(self.full[f]) == 0 or len(self.full[f + 1]) == 0:
            f += 1
            assert self._have(f) and self._have(f + 1), (
                f"{self.name}: predict_edges for pair {f} before its detections")
        raw = out[0]
        assert tuple(raw.shape) == (len(self.full[f]), len(self.full[f + 1])), (
            f"{self.name} t={f}: logits {tuple(raw.shape)} vs detected "
            f"({len(self.full[f])}, {len(self.full[f + 1])})")
        for got, ft in ((coords_src[0], f), (coords_tgt[0], f + 1)):
            exp = torch.from_numpy(self.full[ft]).to(got.device)
            err = float((got - exp).abs().max()) if len(exp) else 0.0
            assert err <= 0.5, f"{self.name} t={ft}: coords differ by {err:.3f}"
        self.mat[f] = torch.softmax(raw, dim=0).float().cpu().numpy()
        self.raw[f] = raw.float().cpu().numpy()
        self.next_f = f + 1
        return out

    def install(self):
        P._detect_cells_pooled = self.detect
        model.predict_edges = self.edges


class InjectRun:
    """edgeprob_cache.Run, reverse direction, full matrices kept."""

    def __init__(self, name, prim, direction="rev"):
        self.name, self.prim, self.dir = name, prim, direction
        self.T = len(prim)
        self.n = np.array([len(p) for p in prim], np.int32)
        self.full = [prim[self.movie_t(f)].astype(np.float32) for f in range(self.T)]
        self.expected = [self.movie_t(f) - 1 for f in range(self.T - 1)
                         if len(self.full[f]) and len(self.full[f + 1])]
        self.ptr = 0
        self.mat = {}
        self.raw = {}

    def movie_t(self, f):
        return self.T - 1 - f

    def detect(self, det_logits, f, det_threshold, pool_kernel):
        c = self.full[f]
        if len(c) == 0:
            return np.empty((0, 4), dtype=np.float32)
        grid = c / DS[None, :]
        return np.concatenate([np.full((len(c), 1), f, np.float32),
                               grid.astype(np.float32)], axis=1)

    def load_frame(self, zarr_arr, f, target_shape, downsample_):
        return _ORIG_LOAD(zarr_arr, self.movie_t(f), target_shape, downsample_)

    def edges(self, feat_src, feat_tgt, coords_src, coords_tgt, *rest):
        out = _ORIG_EDGES(feat_src, feat_tgt, coords_src, coords_tgt, *rest)
        assert self.ptr < len(self.expected), f"{self.name} rev: too many calls"
        t = self.expected[self.ptr]
        self.ptr += 1
        raw = out[0]
        assert tuple(raw.shape) == (self.n[t + 1], self.n[t]), (
            f"{self.name} rev t={t}: {tuple(raw.shape)} vs ({self.n[t + 1]}, {self.n[t]})")
        for got, exp_t in ((coords_src[0], t + 1), (coords_tgt[0], t)):
            exp = torch.from_numpy(self.prim[exp_t].astype(np.float32)).to(got.device)
            err = float((got - exp).abs().max()) if len(exp) else 0.0
            assert err <= 0.5, f"{self.name} rev t={exp_t}: coords differ by {err:.3f}"
        self.mat[t] = torch.softmax(raw, dim=0).float().cpu().numpy()   # rows = t+1
        self.raw[t] = raw.float().cpu().numpy()
        return out

    def install(self):
        P._detect_cells_pooled = self.detect
        P._load_frame = self.load_frame
        model.predict_edges = self.edges


def uninstall():
    P._detect_cells_pooled = _ORIG_DETECT
    P._load_frame = _ORIG_LOAD
    model.predict_edges = _ORIG_EDGES


OUT = pathlib.Path("/kaggle/working/offdet")
OUT.mkdir(exist_ok=True)
t_all = time.time()
failed, counts = [], {}
skipped_time = 0

for k, (gate, name) in enumerate(jobs, 1):
    label = f"{gate:g}_{name}"
    if time.time() - t_all > BUDGET_S:
        skipped_time = len(jobs) - k + 1
        print(f"WARN budget: {skipped_time} jobs not run", flush=True)
        break
    t0 = time.time()
    try:
        cfg_f = P.PredictConfig(det_threshold=gate, det_tta=True, use_ilp=True,
                                threshold=NO_CANDIDATES)
        fwd = DetectRun(label, gate)
        fwd.install()
        try:
            coords, edges = P.predict_video(model, TRAIN / name, device, cfg=cfg_f,
                                            window_size=window_size, max_frames=None,
                                            unet_batch_size=4, downsample=downsample)
        finally:
            uninstall()
        assert len(edges) == 0, f"{label}: predict_video kept {len(edges)} edges"
        T = fwd.T
        assert T >= 2 and all(fwd._have(f) for f in range(T)), (
            f"{label}: frames detected {sorted(fwd.full)[:5]}... of T={T}")
        fwd.full = [fwd.full[f] for f in range(T)]
        fwd.score = [fwd.score[f] for f in range(T)]
        n = np.array([len(x) for x in fwd.full], np.int32)
        assert len(coords) == int(n.sum()), (label, len(coords), int(n.sum()))
        prim = [x.astype(np.float64) for x in fwd.full]
        cfg_r = P.PredictConfig(det_threshold=0.99, det_tta=False, use_ilp=True,
                                threshold=NO_CANDIDATES)
        rev = InjectRun(label, prim)
        rev.install()
        try:
            coords_r, edges_r = P.predict_video(model, TRAIN / name, device, cfg=cfg_r,
                                                window_size=window_size, max_frames=T,
                                                unet_batch_size=4, downsample=downsample)
        finally:
            uninstall()
        assert len(edges_r) == 0 and rev.ptr == len(rev.expected), (
            f"{label} rev: {len(edges_r)} edges, {rev.ptr}/{len(rev.expected)} calls")
        pairs = gated_pairs(prim)
        tt, ii, jj, pf, pr, lf, lr = [], [], [], [], [], [], []
        for t in range(T - 1):
            i_, j_ = pairs[t]
            if len(i_) == 0:
                continue
            assert t in fwd.mat and t in rev.mat, f"{label}: no matrix for pair {t}"
            tt.append(np.full(len(i_), t, np.int32)); ii.append(i_); jj.append(j_)
            pf.append(fwd.mat[t][i_, j_]); lf.append(fwd.raw[t][i_, j_])
            pr.append(rev.mat[t][j_, i_]); lr.append(rev.raw[t][j_, i_])
        M = int(sum(len(x) for x in ii))
        cat = lambda xs: np.concatenate(xs).astype(np.float32)   # noqa: E731
        out = {f"c{t}": fwd.full[t] for t in range(T)}
        out.update({f"s{t}": fwd.score[t] for t in range(T)})
        out.update(t=np.concatenate(tt) if M else np.zeros(0, np.int32),
                   i=np.concatenate(ii).astype(np.int32) if M else np.zeros(0, np.int32),
                   j=np.concatenate(jj).astype(np.int32) if M else np.zeros(0, np.int32),
                   p_fwd=cat(pf) if M else np.zeros(0, np.float32),
                   p_rev=cat(pr) if M else np.zeros(0, np.float32),
                   l_fwd=cat(lf) if M else np.zeros(0, np.float32),
                   l_rev=cat(lr) if M else np.zeros(0, np.float32),
                   n_per_frame=n)
        np.savez_compressed(OUT / f"offdet_{label}.npz", **out)
        dt = time.time() - t0
        counts[label] = dict(gate=gate, name=name, T=T, nodes=int(n.sum()),
                             pairs=M, seconds=round(dt, 1))
        print(f"[{k}/{len(jobs)}] {label}: T={T} nodes={int(n.sum())} "
              f"({n.sum() / T:.1f}/fr) pairs={M} fwd>0.5="
              f"{int((out['p_fwd'] > 0.5).sum())} {dt:.0f}s", flush=True)
    except Exception as exc:                                  # noqa: BLE001
        uninstall()
        failed.append(label)
        print(f"[degraded] {label} FAILED ({type(exc).__name__}: {str(exc)[:300]})",
              flush=True)
        continue

(OUT / "offdet_counts.json").write_text(json.dumps(
    dict(script_version=SCRIPT_VERSION, unet_thresh=UNET_THRESH, gate_um=GATE_UM,
         scale=SCALE.tolist(), downsample=list(map(int, DS)), reverse=True,
         gates=list(GATES), set_44b6=SET_44B6, holdout_6bba=HOLDOUT_6BBA,
         failed=failed, skipped_time=skipped_time, datasets=counts), indent=1))
written = sorted(OUT.glob("offdet_*.npz"))
print(f"\n{len(written)} cached, {time.time() - t_all:.0f}s", flush=True)
if failed or skipped_time or len(written) != len(jobs):
    print(f"OFFDET INCOMPLETE: {len(written)} of {len(jobs)}, failed {failed}, "
          f"skipped {skipped_time} -- NOT OK", flush=True)
else:
    print(f"OFFDET OK ({len(written)} of {len(jobs)})", flush=True)
