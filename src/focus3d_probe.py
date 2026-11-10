"""focus3d_probe: FOCUS-3D on a few frames of one movie -- time, counts, recall, and one-instance-or-two at the pair sites."""
import base64
import gzip
import io
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

SCRIPT_VERSION = "2026-09-23a-focus3d_probe"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

CODE_ROOTS = [
    "/kaggle/input/datasets/faustine649/cell-tracking/cell tracking",
    "/kaggle/input/datasets/faustine649/cell-tracking",
    "/kaggle/input/cell-tracking/cell tracking",
    "/kaggle/input/cell-tracking",
]
code = next((c for c in CODE_ROOTS if pathlib.Path(c, "viz_cells.py").exists()), None)
assert code is not None, "attach faustine649/cell-tracking"
sys.path.insert(0, code)
from viz_cells import find_root, geff_path, open_image, read_geff, zarr_path   # noqa: E402

ROOT = find_root(None)
OUT = pathlib.Path("/kaggle/working")
SCALE = np.array([1.625, 0.40625, 0.40625])
MOVIE = "6bba_ebff6e76"
FRAMES = [6, 8, 9, 13, 14, 18, 22, 23]
NPRIM = {6: 716, 8: 700, 9: 703, 13: 699, 14: 684, 18: 673, 22: 642, 23: 648}
CELL_RADIUS_PX = 12.0          # ~5 um nuclei at 0.40625 um/px
Z_RATIO = float(SCALE[0] / SCALE[1])
SITES_B64 = "H4sIAGZjs2oC/41X7W4cNwz832cRBJH6IPk0xR3iGEGBIkgM1M7TZ4baM1z09txfC8u7I3I4HPL++vb3l/Ll8nL5+fRSXsrlV7m8lctruf4q17dyfS3Pv8rzW3l+/ePnP5fvZV2vlz+frl+/ridbZRUddQ0psuoyL9JHDfypXq230qs3HM5ZR7eiq7bWirT9wKt43kf16tILvhMBWF/V3IpoXT14ulxx6nWtUUQSbvZ8aI9zVFzoE/c3xIo7el3acdVyBrVqLCtmtUUrt0gtny4PMBGU9xK9hiJFtToiCqIwHSXIxigqVmUK3yVcbFTV9gB21mZGAgMpLrzpOESOkVxPQR7Tk54xdrBzP9YpqhdEt0ILKBqgs2uVNhFH8kB2J0+BM/HqAadHsfoD1Fa1oUzN8SXCQ2EMsQvTxLFKdUpjRI2JZHxnf0Me8xzZqoNUlHbZ3AUHIAo++ygTLAzPgjtgY6tADxXYOSj0NwTUrup9FqoQorJqy/mHuPBfoV52oDvzMk5pjc1gMDCCCD9n1YGJMkHKMZNAA/U3WvtnPZCoOvAF3p1zgSitjoTJn+BYWT2wPEHv5GsHrX7Qus6RmfkonaJqlGrgGrIqmfpBBOW3Se2b1K3ee4hoVbgASw9pNpXiSL0r7jFASgMYmsHxDsR8tNbRtr4egMIwBLUJpGRZqQFPAYO+ovisKiCpVzEc7khjfSjZCWjWXfl9GL0FasAdKO9cM9ufpaOLoKu3mRxB4vQRLmRls/NuAQKqEn1RwG3rSQYJ1Fg3UjfmeZnwAVIdYAx5I+FsFDqVQ/4kEH61T/tkxJuAo/j6AJUxZZxGS4EzDdgX/lAqQqDVSVH6Lc4NPNsDRDTm1KTUG0MBtWMVPBw3oOxtaZI3oNcbpfOzRhVKZaFRpfUsCuEh1In0cQj7CAjV0V4wr6Or2gHup6BOqQgqHeBtpfrZTR2tG079KBwxLcXeRWWfGABB6XE0eoiLfWCHxeNY6bAoJC1wsWfXe6/q4dx26gKajenoHvTCAn3KeWPpywMBk3nTnJDN4+bWfX1GLnCRZF+cIpAtVYVvJJvcnRPClYF57Tzc8d6sJeQBLFzPOWKgK1g3PL9BpZ05ILyGOlFgGGtwNLWP/J4rDPXtHC3osSWQq26gkSJesXK0tBlpKtqQmh0TUT7K4gQaLxm/RT9AvMOh3cmHoactJxfSD1RgB3hMhfMOU3bT6JJRAHjhA2xEkfUa7IlBAizeoXaU61H+KCzVKhh6MbcK6IwImloWXOW9Z48FHbd/nAE5OAn9/PL98u3H/50zdBQMQW4pS7l09PSIkzlzF5tzAZPAshE6vADuUtXSdTgK0GDGVYPrEbQ9js3g6IqIB8iDzsnlAOFNODXfNhaOe6jtIeF799hV3UTsR5MHyE4bZHVCcyNklzpbDR8jIW81IGHFLtYptcN3bx29HiKf2QR8l1XFoORoyxUO6j6xibvgNAruV56L1YThinOhy30LAgA40uHcMGiGz/h3Hbchn4B3LnManiGyTRiLgJxcRnjnqJ2NiJ3aZOzV78OOd0T+44KfGv/ZdA2EcgFroABRQGg849KEXxEdKgflgktKuQfgfEu4zrKprOdEFMxJ5dYAMtG2MyKtI/Aj4j5KUIv8N94OzT1SJFck/ppZsErYdh7KCQLXFegGjTP22OqSYeHQdx9wy6KvoCYnECOdG0a5mELwL3hXrg/Y8BCazklVCCI8gcitlQ0AS5xUL8aW5w6RweC0U2mYzV3OyNC8i78Jsn0wt/jrBg2KxWC0voVmOQcMzJ6A9Nx1F9Mae6fsO36UiwpCmRazkpzqAPkN7nrXookOAAA="

# ---- FOCUS-3D code: qiweiyin/focus3d-nuclei-runtime when attached (runtime + config + weights), else the public space ----
SPACE = OUT / "focus3d_space"
for root_, dirs, fnames in os.walk("/kaggle/input"):
    dirs[:] = [d for d in dirs if "cell-tracking-during-development" not in d and not d.endswith(".zarr") and not d.endswith(".geff")]
    if "inference_win.py" in fnames and "focus3d_runtime" in root_:
        SPACE = pathlib.Path(root_.split("focus3d_runtime")[0])
        print(f"runtime from {SPACE}", flush=True)
        break
if not SPACE.exists():
    t0 = time.time()
    subprocess.run(["git", "clone", "--depth", "1", "https://huggingface.co/spaces/Qinghua-thu/FOCUS-3D", str(SPACE)],
                   check=True, env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"})
    print(f"cloned the space in {time.time() - t0:.0f}s", flush=True)
sys.path.insert(0, str(SPACE / "focus3d_runtime"))
for _mod, _pkg in (("zarr", "zarr"), ("tifffile", "tifffile"), ("yaml", "PyYAML"), ("skimage", "scikit-image"), ("tqdm", "tqdm")):
    try:
        __import__(_mod)
    except ImportError:
        print(f"installing {_pkg}", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", _pkg], check=True)
import torch                                                        # noqa: E402
from focus3d.segmentation.FOCUS3D.inference_win import build_predictor, infer_volume, setup_cfg   # noqa: E402
from focus3d.segmentation.FOCUS3D.mask2former.maskformer_model_win import build_maskformer_model_from_cfg   # noqa: E402

CONFIG = SPACE / "configs" / "3d_test.yaml"
weights = None
for root_, dirs, fnames in os.walk("/kaggle/input"):
    dirs[:] = [d for d in dirs if "cell-tracking-during-development" not in d and not d.endswith(".zarr") and not d.endswith(".geff")]
    for fn in fnames:
        if fn == "model_final_nuclei.pth":
            weights = pathlib.Path(root_) / fn
REAL = weights is not None
print(f"torch {torch.__version__} cuda {torch.cuda.is_available()}; weights: {weights if REAL else 'NONE -> random init, TIMING ONLY'}", flush=True)
t0 = time.time()
cfg = setup_cfg(CONFIG, weights if REAL else "/nonexistent.pth", device="cuda")
if REAL:
    model = build_predictor(cfg)
else:
    model = build_maskformer_model_from_cfg(cfg)
    model.eval()
    model.to("cuda")
n_par = sum(p.numel() for p in model.parameters())
print(f"model built in {time.time() - t0:.0f}s: {n_par / 1e6:.0f} M parameters", flush=True)

# ---- data ----
img = open_image(zarr_path(ROOT, "train", MOVIE))
T, Z, Y, X = img.shape
gt = read_geff(geff_path(ROOT, "train", MOVIE))
g_t = np.asarray(gt["t"], float).round().astype(int)
g_um = np.stack([np.asarray(gt[c], float) for c in ("z", "y", "x")], 1) * SCALE
est = gt["extra"].get("estimated_number_of_nodes")
print(f"{MOVIE}: shape {img.shape}, GT nodes {len(g_t)}, estimated nodes {est} (~{float(est) / T:.0f} per frame)" if est else f"{MOVIE}: shape {img.shape}", flush=True)
sites = pd.read_csv(io.StringIO(gzip.decompress(base64.b64decode(SITES_B64)).decode()))

rows, labels_out, pair_rows = [], {}, []
tmp = pathlib.Path("/tmp/f3d")
tmp.mkdir(exist_ok=True)
# two settings: A = the space's defaults with our radius (resamples XY to 320, overlapping windows);
# B = qiweiyin's validated Kaggle settings (radius 15 -> no resampling, non-overlapping windows, background = frame median)
CONFIGS = {"B": dict(cell_radius=15.0, stride=[32, 96, 96], bg="median", score=0.7),
           "C": dict(cell_radius=20.0, stride=[32, 96, 96], bg="median", score=0.7),
           "D": dict(cell_radius=25.0, stride=[32, 96, 96], bg="median", score=0.7),
           "E": dict(cell_radius=20.0, stride=[32, 96, 96], bg="median", score=0.85)}
for k_, (t, cname) in enumerate((t, c) for t in FRAMES for c in CONFIGS):
    cf = CONFIGS[cname]
    vol = np.asarray(img[t]).astype(np.uint16)
    bg = float(np.percentile(vol, 10.0)) if cf["bg"] == "p10" else float(np.median(vol))
    tif = tmp / f"{MOVIE}_t{t}.tif"
    if not tif.exists():
        tifffile.imwrite(str(tif), vol, metadata={"axes": "ZYX"}, photometric="minisblack")
    t0 = time.time()
    res = infer_volume(image_path=tif, config_file=CONFIG, weights_path=weights if REAL else "/nonexistent.pth", model=model,
                       device="cuda", output_dir=tmp / f"out_{cname}_t{t}", z_ratio=Z_RATIO, lower_percentile=1.0, upper_percentile=99.0,
                       data_loader_num_workers=0, cell_radius=cf["cell_radius"], background_threshold=bg, stride=cf["stride"],
                       batch_size=12, score_thresh=cf["score"], mask_thresh=0.5, min_edge_area=64, topk_postprocess=300, save_intermediate=False)
    dt = time.time() - t0
    lab = np.asarray(res["instance_map"]).astype(np.int32)
    assert lab.shape == vol.shape, (lab.shape, vol.shape)
    ids = np.unique(lab[lab > 0])
    n_inst = len(ids)
    # centroids
    if n_inst:
        cz = ndi.center_of_mass(np.ones_like(lab, float), lab, ids)
        cen = np.asarray(cz, float) * SCALE
        sizes = np.bincount(lab.ravel())[ids] * SCALE.prod()
    else:
        cen, sizes = np.zeros((0, 3)), np.zeros(0)
    m = g_t == t
    G = g_um[m]
    rec5 = rec3 = dmed = np.nan
    if n_inst and m.sum():
        d, _ = cKDTree(cen).query(G)
        rec5, rec3, dmed = float((d <= 5).mean()), float((d <= 3).mean()), float(np.median(d))
    rows.append(dict(cfg=cname, t=t, seconds=dt, instances=n_inst, gt_nodes=int(m.sum()), primaries=NPRIM.get(t, -1),
                     vol_med_um3=float(np.median(sizes)) if n_inst else np.nan, recall5=rec5, recall3=rec3, gt_to_nearest_med=dmed))
    print(f"  [{k_ + 1}/{len(FRAMES) * len(CONFIGS)}] {cname} t={t}: {dt:.1f}s, instances {n_inst} (GT {int(m.sum())}, primaries {NPRIM.get(t, -1)}), "
          f"recall@5um {rec5:.2f} @3um {rec3:.2f}, GT->nearest {dmed:.2f} um, median volume {rows[-1]['vol_med_um3']:.0f} um^3", flush=True)
    labels_out[f"{cname}_t{t}"] = lab.astype(np.uint16)
    # the pair sites at this frame: which instance holds a, b, and A
    def _lab_at(p):
        z_, y_, x_ = [int(round(v)) for v in p]
        if 0 <= z_ < Z and 0 <= y_ < Y and 0 <= x_ < X:
            return int(lab[z_, y_, x_])
        return -1
    for _, r in sites[sites.t == t].iterrows():
        la, lb = _lab_at((r.az, r.ay, r.ax)), _lab_at((r.bz, r.by, r.bx)) if np.isfinite(r.bz) else -1
        lg = _lab_at((r.gz, r.gy, r.gx)) if np.isfinite(r.gz) else -1
        pair_rows.append(dict(cfg=cname, kind=r.kind, t=t, lab_a=la, lab_b=lb, lab_A=lg,
                              same=int(la > 0 and la == lb), both_in=int(la > 0 and lb > 0), a_bg=int(la == 0), b_bg=int(lb == 0)))
np.savez_compressed(OUT / "focus3d_probe_labels.npz", **labels_out)
R = pd.DataFrame(rows)
R.to_csv(OUT / "focus3d_probe_frames.csv", index=False)
P = pd.DataFrame(pair_rows)
P.to_csv(OUT / "focus3d_probe_pairs.csv", index=False)
print(f"\n=== FOCUS3D PROBE ({'REAL WEIGHTS' if REAL else 'RANDOM WEIGHTS: timing only, counts meaningless'}) ===", flush=True)
for cname, cf in CONFIGS.items():
    Rc, Pc = R[R.cfg == cname], P[P.cfg == cname]
    print(f"\n  config {cname}: radius {cf['cell_radius']}, stride {cf['stride']}, background {cf['bg']}, score {cf['score']}", flush=True)
    print(f"  seconds per volume: median {Rc.seconds.median():.1f} (first {Rc.seconds.iloc[0]:.1f}); -> 12,800 volumes = {Rc.seconds.median() * 12800 / 3600:.0f} h", flush=True)
    print(f"  instances per frame median {Rc.instances.median():.0f}; GT nodes {Rc.gt_nodes.median():.0f}; shipped primaries {Rc.primaries.median():.0f}"
          + (f"; estimated nodes per frame {float(est) / T:.0f}" if est else "") + f"; median instance volume {Rc.vol_med_um3.median():.0f} um^3", flush=True)
    print(f"  annotated centroid inside 5 um / 3 um of an instance centroid: {Rc.recall5.mean():.3f} / {Rc.recall3.mean():.3f}; GT -> nearest median {Rc.gt_to_nearest_med.median():.2f} um", flush=True)
    for kind in ("swap", "gtpair", "rand"):
        q = Pc[Pc.kind == kind]
        if len(q):
            print(f"  {kind:<7} pairs {len(q)}: a and b in the SAME instance {q.same.mean():.2f}; both inside some instance {q.both_in.mean():.2f}; "
                  f"a on background {q.a_bg.mean():.2f}, b on background {q.b_bg.mean():.2f}", flush=True)
print("FOCUS3D PROBE OK", flush=True)
