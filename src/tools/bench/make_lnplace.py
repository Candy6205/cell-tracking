"""lnplace.py: can the rescue peaks be RE-PLACED on the raw image to where the annotated cell is?

WHY (2026-09-18). lnrank: rankers retrained with the rescue peaks' rows still refuse the same 128 edges; what
separates the 74 linked rescue edges from the refused ones is placement, 2.8 um against 5.7 um from the
annotated centroid. lntrace recorded every such peak with its category. Before any placement-aware detector
work (expensive), the cheap question: does the RAW image have a maximum where the annotation is, and does a
raw re-placement of the peak (the smoothed crop's maximum within 3 / 5 / 7 um, or the intensity centroid
around it) bring the refused peaks to the ~3 um the linked ones sit at, without pulling the linked ones away
and without landing on a shipped primary (a duplicate)? A control on real detections (shipped primaries
matched to GT) says what the same re-placement does to good detections. Two smoothing scales: forkimg's
contrast crop (sigma 0.6 / 1.5 / 1.5 vox, ~1 um) and a coarse one (1.2 / 4 / 4 vox, ~2 um, a nuclear radius).

v2 -> v3: q_i in lnrank_edges.csv indexes the SEEDS (primaries with score >= UNET_THRESH 0.15, seeds_of), not
the raw cache; v2 indexed the cache and measured the wrong points (its peak rows are void; its control rows
stand). v3 applies seeds_of first and checks every recomputed d_gt against the csv.

WHAT. Standalone CPU kernel, no solve: lnrank_edges.csv (arm LN20, one row per linkable GT edge of an
annotated rescue peak) -> one row per PEAK with its class (linked / ranker_refused / below_tau / other);
per movie: the rescue detection list (fdet_F4ln_, lnrescue), the GT geff, the raw image; per peak the
distances above at both scales, the local-maximum test around the GT centroid at contrast floors, the
contrast at the peak and at the GT; summary per class; lnplace_peaks.csv, lnplace_control.csv.

Attach: biohub-lnrescue (fdet_F4ln_*.npz), biohub-lnrank (lnrank_edges.csv), the competition data, the
cell-tracking code dataset, the support pack. CPU ~5 min.
"""
import os

ln = open("lnrank.py", encoding="utf-8").read()


def between(s, a, b, include_b=False):
    i = s.index(a)
    j = s.index(b, i + len(a))
    return s[i:j + (len(b) if include_b else 0)]


pre = between(ln, "from __future__ import annotations\n", "SCALE = np.array([1.625, 0.40625, 0.40625])\n", include_b=True)
pre = pre.replace('SCRIPT_VERSION = "2026-09-18a-lnrank"', 'SCRIPT_VERSION = "2026-09-18b-lnplace"')
assert 'SCRIPT_VERSION = "2026-09-18b-lnplace"' in pre and "from viz_cells import" in pre
warn = between(ln, "_WARN_COUNT = defaultdict(int)\n", "def pass1_tracks(dets):")
walk = between(ln, "def _walk_files(match):", "# ---- the official head's probabilities")
lnload = between(ln, "def ln_load(name):", "LN_PERSIST_UM = 4.0")

BODY = r'''
root = find_root(None)
print(f"root {root}", flush=True)
import pandas as pd                                                          # noqa: E402
from scipy import ndimage as _ndi                                            # noqa: E402
from scipy.ndimage import maximum_filter                                     # noqa: E402
from viz_cells import open_image as _open_image                              # noqa: E402

LN_FDET = _walk_files(_npz_prefix("fdet_F4ln_"))
print(f"rescue caches (fdet_F4ln_): {len(LN_FDET)}", flush=True)
assert len(LN_FDET) >= 100, "attach biohub-lnrescue"
_csv = [p_ for p_ in pathlib.Path("/kaggle/input").glob("**/lnrank_edges.csv")]
assert _csv, "attach biohub-lnrank (lnrank_edges.csv)"
df = pd.read_csv(_csv[0])
df = df[df.arm == "LN20"].reset_index(drop=True)
print(f"lnrank_edges.csv: {len(df)} LN20 rows over {df.dataset.nunique()} movies; categories "
      f"{df.cat.value_counts().to_dict()}", flush=True)

R_LIST = (3.0, 5.0, 7.0)
LM_UM = 3.0                       # a raw local maximum this close to the GT centroid = the cell is there in the raw image
LM_THR = ((0.0, "d_lm0"), (0.25, "d_lm25"), (0.5, "d_lm50"), (1.0, "d_lm100"))   # the maximum's contrast floor
DUP_UM = 3.0                      # a re-placed peak this close to a shipped primary would be a duplicate
UNET_THRESH = 0.15                # seeds_of: the harness keeps primaries at score >= this; q_i indexes THAT list
HALF_UM = 8.0
SIGMAS = (("f", (0.6, 1.5, 1.5)), ("c", (1.2, 4.0, 4.0)))    # fine = forkimg's contrast crop (~1 um); coarse ~ a nuclear radius


def seeds_of(dets_t):
    coords, scores = dets_t
    coords = np.asarray(coords, float).reshape(-1, 3)
    scores = np.asarray(scores, float).reshape(-1)
    m = scores >= UNET_THRESH
    return coords[m], scores[m]


def _crop(img, t, xyz_vox, sigma):
    """smoothed crop (half HALF_UM) around a voxel position and its offset; (None, None) out of range."""
    if t < 0 or t >= img.shape[0]:
        return None, None
    hz = int(np.ceil(HALF_UM / SCALE[0]))
    hy = int(np.ceil(HALF_UM / SCALE[1]))
    z, y, x = (int(round(float(v))) for v in xyz_vox)
    z0, y0, x0 = max(0, z - hz), max(0, y - hy), max(0, x - hy)
    try:
        crop = np.asarray(img[t, z0:z + hz + 1, y0:y + hy + 1, x0:x + hy + 1])
    except Exception:                                                        # noqa: BLE001
        crop = np.asarray(img[t])[z0:z + hz + 1, y0:y + hy + 1, x0:x + hy + 1]
    if crop.size == 0:
        return None, None
    return _ndi.gaussian_filter(crop.astype(np.float32), sigma), np.array([z0, y0, x0], float)


def peak_class(cats):
    cats = set(cats)
    if "G_tp" in cats:
        return "linked"
    if "C_below_floor" in cats:
        return "ranker_refused"
    if cats == {"A_below_tau"}:
        return "below_tau"
    return "other"


peaks = {}
for (name, t, qi), g_ in df.groupby(["dataset", "t", "q_i"]):
    peaks[(name, int(t), int(qi))] = dict(cls=peak_class(g_.cat.tolist()), q_score=float(g_.q_score.iloc[0]),
                                          d_gt_csv=float(g_.d_gt.iloc[0]), admitted=int(g_.admitted.iloc[0]),
                                          n_edges=len(g_), cats="|".join(sorted(set(g_.cat))))
print(f"peaks: {len(peaks)}; per class " + str(pd.Series([v['cls'] for v in peaks.values()]).value_counts().to_dict()),
      flush=True)


def _grid_um(sm, off):
    idx = np.indices(sm.shape).reshape(3, -1).T.astype(float)
    return (idx + off) * SCALE, idx


def replace_on_raw(img, t, c_vox, sigma):
    """the smoothed crop around c: its maximum within R um of c for each R, the intensity centroid (3.5 um)
    around the 5-um maximum, the contrast at c; None when the crop is unreadable."""
    out = {}
    sm, off = _crop(img, t, c_vox, sigma)
    if sm is None:
        return None
    pos_um, idx = _grid_um(sm, off)
    v = sm.reshape(-1)
    c_um = np.asarray(c_vox, float) * SCALE
    d = np.linalg.norm(pos_um - c_um, axis=1)
    bg = float(np.median(v))
    j0 = int(np.argmin(d))
    out["con_peak"] = (float(v[j0]) - bg) / max(bg, 1e-6)
    for R in R_LIST:
        near = d <= R
        j = int(np.argmax(np.where(near, v, -np.inf)))
        out[f"pos_r{int(R)}"] = pos_um[j]
        out[f"con_r{int(R)}"] = (float(v[j]) - bg) / max(bg, 1e-6)
    j5 = int(np.argmax(np.where(d <= 5.0, v, -np.inf)))
    dp = np.linalg.norm(pos_um - pos_um[j5], axis=1)
    m = dp <= 3.5
    w = np.clip(v[m] - bg, 0, None)
    out["pos_cen"] = (w[:, None] * pos_um[m]).sum(0) / w.sum() if w.sum() > 0 else pos_um[j5]
    return out


def raw_maximum_near(img, t, g_um, sigma):
    """crop centred on the GT centroid: the nearest raw local maximum (26-neighbourhood) within 7 um at each
    contrast floor over the crop median, the contrast at the GT voxel; None when unreadable."""
    g_vox = np.asarray(g_um, float) / SCALE
    sm, off = _crop(img, t, g_vox, sigma)
    if sm is None:
        return None
    pos_um, idx = _grid_um(sm, off)
    v = sm.reshape(-1)
    bg = float(np.median(v))
    lm = (sm >= maximum_filter(sm, size=3, mode="nearest")).reshape(-1) & (v > bg)
    d = np.linalg.norm(pos_um - np.asarray(g_um, float), axis=1)
    con = (v - bg) / max(bg, 1e-6)
    j0 = int(np.argmin(d))
    out = dict(con_gt=float(con[j0]))
    # smoothed noise has a local maximum every 1-2 um, so a maximum only counts above a contrast floor
    for thr_, key_ in LM_THR:
        cand = np.where(lm & (d <= 7.0) & (con >= thr_))[0]
        out[key_] = float(d[cand].min()) if len(cand) else np.nan
        if key_ == "d_lm50":
            out["con_lm"] = float(con[cand[np.argmin(d[cand])]]) if len(cand) else np.nan
    out["d_gt_lm"] = out["d_lm50"]
    return out


def measure(img, t, c_vox, g_um, ship_tree):
    """every column of one point (a rescue peak or a control detection) at both scales; None when unreadable."""
    c_um = np.asarray(c_vox, float) * SCALE
    row = {}
    for tag, sig in SIGMAS:
        rp = replace_on_raw(img, t, c_vox, sig)
        rm = raw_maximum_near(img, t, g_um, sig)
        if rp is None or rm is None:
            return None
        suf = "" if tag == "f" else "_c"
        row[f"con_peak{suf}"] = rp["con_peak"]
        for k_, v_ in rm.items():
            row[f"{k_}{suf}"] = v_
        for R in R_LIST:
            p_ = rp[f"pos_r{int(R)}"]
            row[f"d_gt_r{int(R)}{suf}"] = float(np.linalg.norm(p_ - g_um))
            row[f"moved_r{int(R)}{suf}"] = float(np.linalg.norm(p_ - c_um))
            row[f"con_r{int(R)}{suf}"] = rp[f"con_r{int(R)}"]
            row[f"d_ship_r{int(R)}{suf}"] = float(ship_tree.query(p_)[0]) if ship_tree is not None else np.nan
        row[f"d_gt_cen{suf}"] = float(np.linalg.norm(rp["pos_cen"] - g_um))
        row[f"moved_cen{suf}"] = float(np.linalg.norm(rp["pos_cen"] - c_um))
        row[f"d_ship_cen{suf}"] = float(ship_tree.query(rp["pos_cen"])[0]) if ship_tree is not None else np.nan
    row["d_ship0"] = float(ship_tree.query(c_um)[0]) if ship_tree is not None else np.nan
    return row


ROWS, CTRL = [], []
t_all = time.time()
rng = np.random.default_rng(0)
by_movie = defaultdict(list)
for k_, v_ in peaks.items():
    by_movie[k_[0]].append((k_[1], k_[2], v_))
n_skip = n_bad_idx = 0
for mi, name in enumerate(sorted(by_movie)):
    t0 = time.time()
    if name not in LN_FDET:
        warn_once("no_fdet", f"{name}: no fdet_F4ln_ cache")
        continue
    gt = read_geff(geff_path(root, "train", name))
    g_t = np.asarray(gt["t"], int)
    g_pos = np.stack([np.asarray(gt[c_], float) for c_ in ("z", "y", "x")], axis=1) * SCALE
    gt_rows = {int(tt): np.where(g_t == tt)[0] for tt in np.unique(g_t)}
    gt_tree = {tt: cKDTree(g_pos[ix]) for tt, ix in gt_rows.items()}
    dets, nnew = ln_load(name)
    seeds = [seeds_of(dets[t]) for t in range(len(dets))]
    try:
        img = _open_image(pathlib.Path(root) / "train" / f"{name}.zarr")
    except Exception as _e:                                                  # noqa: BLE001
        warn_once("no_img", f"{name}: image not opened ({_e})")
        continue
    ship_tree = {}

    def _ship(t):
        if t not in ship_tree:
            c_ = seeds[t][0]
            n_ = len(c_) - nnew[t]
            ship_tree[t] = cKDTree(c_[:n_] * SCALE) if n_ > 0 else None
        return ship_tree[t]

    for t, qi, meta in by_movie[name]:
        c_all = seeds[t][0]
        if qi >= len(c_all) or t not in gt_tree:
            n_bad_idx += 1
            continue
        c_vox = c_all[qi]
        c_um = c_vox * SCALE
        d0, j_ = gt_tree[t].query(c_um)
        g_um = g_pos[gt_rows[t][int(j_)]]
        row = measure(img, t, c_vox, g_um, _ship(t))
        if row is None:
            n_skip += 1
            continue
        row.update(dataset=name, t=t, q_i=qi, **meta, d_gt0=float(d0), d_gt_csv_check=float(abs(d0 - meta["d_gt_csv"])))
        ROWS.append(row)
    # the control: shipped primaries of the same frames matched to GT within 5 um, up to 30 per movie
    frames = sorted({t for t, _q, _m in by_movie[name]})
    cands = []
    for t in frames:
        if t not in gt_tree:
            continue
        c_all = seeds[t][0]
        n0 = len(c_all) - nnew[t]
        if n0 <= 0:
            continue
        dd, jj = gt_tree[t].query(c_all[:n0] * SCALE)
        for i_ in np.where(dd <= 5.0)[0]:
            cands.append((t, int(i_), float(dd[i_]), g_pos[gt_rows[t][int(jj[i_])]]))
    if cands:
        for ci in rng.choice(len(cands), size=min(30, len(cands)), replace=False):
            t, i_, d0, g_um = cands[ci]
            crow = measure(img, t, seeds[t][0][i_], g_um, _ship(t))
            if crow is None:
                continue
            crow.update(dataset=name, t=t, i=i_, d_gt0=d0)
            CTRL.append(crow)
    if mi % 16 == 0 or mi == len(by_movie) - 1:
        print(f"  [{mi + 1}/{len(by_movie)}] {name}: {len(by_movie[name])} peaks, {time.time() - t0:.1f}s "
              f"(total {(time.time() - t_all) / 60:.1f} min)", flush=True)

P = pd.DataFrame(ROWS)
C = pd.DataFrame(CTRL)
P.to_csv("/kaggle/working/lnplace_peaks.csv", index=False)
C.to_csv("/kaggle/working/lnplace_control.csv", index=False)
n_agree = int((P.d_gt_csv_check < 0.01).sum())
print(f"\n{len(P)} peaks measured ({n_skip} unreadable, {n_bad_idx} with an index beyond the seed list), {len(C)} control "
      f"detections; the recomputed d_gt agrees with lnrank's csv within 0.01 um for {n_agree} of {len(P)} peaks", flush=True)
assert n_agree >= 0.95 * len(P), "the peaks are not the csv's peaks: indexing is wrong (v2's bug)"


def _q(s, f=0.5):
    s = np.asarray(s, float)
    s = s[np.isfinite(s)]
    return float(np.quantile(s, f)) if len(s) else float("nan")


def _frac(m):
    m = np.asarray(m)
    return 100.0 * float(np.mean(m)) if len(m) else float("nan")


CLASSES = ("ranker_refused", "linked", "below_tau", "other", "ALL", "control")


def _sel(cls):
    return C if cls == "control" else (P if cls == "ALL" else P[P.cls == cls])


for tag, sig in SIGMAS:
    suf = "" if tag == "f" else "_c"
    print(f"\n=== RE-PLACEMENT ON THE RAW IMAGE, smoothing sigma {sig} vox ({'fine, forkimg' if tag == 'f' else 'coarse, ~nuclear radius'}); "
          f"per class (medians in um; '<=3' = share within 3 um of the GT centroid) ===", flush=True)
    hdr = (f"  {'class':<15} {'n':>4} {'d_gt0':>6} {'<=3':>5} | {'r3':>6} {'<=3':>5} | {'r5':>6} {'<=3':>5} {'moved':>6} {'dup':>5} | "
           f"{'r7':>6} {'<=3':>5} | {'cen':>6} {'<=3':>5} {'dup':>5} | {'lm<=3':>6} {'d_lm':>6} | {'c_peak':>6} {'c_gt':>6} {'c_r5':>6}")
    print(hdr, flush=True)
    print("  " + "-" * (len(hdr) - 2), flush=True)
    for cls in CLASSES:
        s = _sel(cls)
        if not len(s):
            continue
        print(f"  {cls:<15} {len(s):>4} {_q(s.d_gt0):>6.2f} {_frac(s.d_gt0 <= 3):>4.0f}% | "
              f"{_q(s[f'd_gt_r3{suf}']):>6.2f} {_frac(s[f'd_gt_r3{suf}'] <= 3):>4.0f}% | "
              f"{_q(s[f'd_gt_r5{suf}']):>6.2f} {_frac(s[f'd_gt_r5{suf}'] <= 3):>4.0f}% {_q(s[f'moved_r5{suf}']):>6.2f} "
              f"{_frac(s[f'd_ship_r5{suf}'] < DUP_UM):>4.0f}% | "
              f"{_q(s[f'd_gt_r7{suf}']):>6.2f} {_frac(s[f'd_gt_r7{suf}'] <= 3):>4.0f}% | "
              f"{_q(s[f'd_gt_cen{suf}']):>6.2f} {_frac(s[f'd_gt_cen{suf}'] <= 3):>4.0f}% {_frac(s[f'd_ship_cen{suf}'] < DUP_UM):>4.0f}% | "
              f"{_frac(s[f'd_lm50{suf}'] <= LM_UM):>5.0f}% {_q(s[f'd_lm50{suf}']):>6.2f} | "
              f"{_q(s[f'con_peak{suf}']):>6.2f} {_q(s[f'con_gt{suf}']):>6.2f} {_q(s[f'con_r5{suf}']):>6.2f}", flush=True)
    print("  a raw local maximum within 3 um of the GT centroid, by the maximum's contrast floor over the crop median (share):", flush=True)
    for cls in CLASSES:
        s = _sel(cls)
        if not len(s):
            continue
        print(f"    {cls:<15} n={len(s):>4}  " + "  ".join(f"con>={thr_:<4g} {_frac(s[f'{key_}{suf}'] <= LM_UM):>4.0f}%" for thr_, key_ in LM_THR)
              + f"   contrast at the point {_q(s[f'con_peak{suf}']):.2f}, at the GT voxel {_q(s[f'con_gt{suf}']):.2f}", flush=True)
print("  columns: d_gt0 = the point as placed (lnrescue's peak, or the shipped detection for the control); r3/r5/r7 = the raw "
      "maximum within that radius of the point; cen = the intensity centroid (3.5 um) around the 5-um maximum; moved = how far "
      "r5 moved the point; dup = the re-placed point lands within 3 um of a shipped primary (for the control: of ANOTHER shipped "
      "primary or itself, so read it only for the peaks); lm<=3 = a raw local maximum with contrast >= 0.5 within 3 um of the "
      "GT centroid exists; d_lm = the nearest such maximum within 7 um; c_* = contrast over the crop median at the point / at "
      "the GT voxel / at r5.", flush=True)

print("\n=== the refused peaks against the linked ones (the placement question), both scales ===", flush=True)
for tag, sig in SIGMAS:
    suf = "" if tag == "f" else "_c"
    for cls in ("ranker_refused", "linked", "control"):
        s = _sel(cls)
        if len(s):
            print(f"  [{tag}] {cls:<15} n={len(s):>4}  d_gt 0: {_q(s.d_gt0):.2f} -> r5: {_q(s[f'd_gt_r5{suf}']):.2f} -> cen: "
                  f"{_q(s[f'd_gt_cen{suf}']):.2f}; within 3 um: {_frac(s.d_gt0 <= 3):.0f}% -> {_frac(s[f'd_gt_r5{suf}'] <= 3):.0f}% -> "
                  f"{_frac(s[f'd_gt_cen{suf}'] <= 3):.0f}%; a raw maximum (con >= 0.5) within 3 um of the GT: "
                  f"{_frac(s[f'd_lm50{suf}'] <= LM_UM):.0f}%; r5 within 3 um of a shipped primary: {_frac(s[f'd_ship_r5{suf}'] < DUP_UM):.0f}%",
                  flush=True)
print("  reading: if the refused peaks have a raw maximum at the GT and r5/cen bring them to ~3 um without landing on shipped "
      "primaries (and the control is not made worse), a re-placement step before the ranker is the next bench; if the raw image "
      "has no maximum at the GT for them, no detector can place them there and the line closes on evidence.", flush=True)
warning_summary()
print(f"\nLNPLACE DONE in {(time.time() - t_all) / 60:.1f} min", flush=True)
print("LNPLACE OK", flush=True)
'''

src = pre + "\n" + warn + "\n" + walk + "\n" + lnload + BODY
assert src.count("def seeds_of(") == 1 and src.count("def ln_load(") == 1 and src.count("def _crop(") == 1
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lnplace.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written lnplace.py", len(src))
