"""
division_features.py -- does anything actually distinguish a dividing cell?

Four hypotheses, all measured against controls drawn from the SAME frame:

  1. geometry   sister separation over the frames after the split, and how much
                of it is along Z (the axis with 4x worse resolution)
  2. edge length parent->daughter distance, against the distribution of ordinary
                ground-truth edges
  3. intensity  chromatin condenses during mitosis, so a dividing nucleus should
                be BRIGHTER. Orientation-independent, trivial to compute.
  4. elongation metaphase/anaphase stretches the nucleus. Measured as the 3D
                intensity-weighted second-moment tensor in PHYSICAL units, so it
                does not depend on which way the division axis points -- a 2D
                projection would miss any division along Z.

Every feature is reported as an AUC against controls: 0.5 = useless, 1.0 =
perfectly separating. Anything near 0.5 should be dropped, not tuned.

Usage (CPU):
    import division_features as df
    df.run(limit=40)          # datasets to scan; 199 is the whole train split

OUTPUT DISCIPLINE (rules 2 and 10): fixed-size tables, per-event data goes to a
CSV, and `run` returns a small dict -- never the per-event records.
"""

from __future__ import annotations

import csv
import os

import numpy as np

from viz_cells import (find_root, geff_path, list_datasets, open_image,
                       read_geff, read_scale, zarr_path)

try:
    from viz_cells import warn_once, warning_summary
except ImportError:                                        # pragma: no cover
    def warn_once(key, msg):
        print(f"[degraded] {msg}", flush=True)

    def warning_summary(reset=True):
        pass

SCRIPT_VERSION = "2026-08-19a"

RZ, RXY = 2, 8          # analysis window: +-3.25 um in Z, +-3.25 um in XY
N_CONTROL = 12          # control cells per division, same frame
FOLLOW = 5              # frames of sister separation to follow


def _auc(pos, neg):
    """P(a random positive exceeds a random negative). 0.5 = no signal."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort().astype(float) + 1        # ranks, ties ignored
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def _features(vol, z, y, x, scale):
    """(background-subtracted intensity sum, peak, elongation) around one cell."""
    Z, Y, X = vol.shape
    z0, z1 = max(0, z - RZ), min(Z, z + RZ + 1)
    y0, y1 = max(0, y - RXY), min(Y, y + RXY + 1)
    x0, x1 = max(0, x - RXY), min(X, x + RXY + 1)
    w = vol[z0:z1, y0:y1, x0:x1].astype(np.float32)
    if w.size < 8:
        return np.nan, np.nan, np.nan
    bg = np.percentile(w, 10)
    m = np.clip(w - bg, 0, None)
    tot = float(m.sum())
    if tot <= 0:
        return 0.0, float(w.max() - bg), np.nan

    gz, gy, gx = np.meshgrid(np.arange(z0, z1) * scale[0],
                             np.arange(y0, y1) * scale[1],
                             np.arange(x0, x1) * scale[2], indexing="ij")
    p = np.stack([gz.ravel(), gy.ravel(), gx.ravel()], axis=1)
    wt = m.ravel()
    mu = (p * wt[:, None]).sum(0) / tot
    d = p - mu
    cov = (d * wt[:, None]).T @ d / tot
    ev = np.linalg.eigvalsh(cov)                       # ascending
    ev = np.clip(ev, 1e-9, None)
    elong = float(np.sqrt(ev[2] / ev[0]))              # long axis / short axis
    return tot, float(w.max() - bg), elong


def run(root=None, split="train", limit=None, out_dir="/kaggle/working",
        log_every=10, seed=0):
    r = find_root(root)
    names = list_datasets(r, split)
    if limit:
        step = max(1, len(names) // int(limit))
        names = names[::step][: int(limit)]
    rng = np.random.default_rng(seed)
    print(f"scanning {len(names)} datasets", flush=True)

    sep = {k: [] for k in range(1, FOLLOW + 1)}        # sister separation, um
    sepz = {k: [] for k in range(1, FOLLOW + 1)}       # |dz| share of it
    div_edges, all_edges = [], []
    P = {"tot": [], "peak": [], "elong": []}           # dividing parents
    C = {"tot": [], "peak": [], "elong": []}           # controls, same frames
    rows, n_div, n_img = [], 0, 0

    for di, name in enumerate(names, 1):
        gp = geff_path(r, split, name)
        if not gp.exists():
            continue
        gt = read_geff(gp)
        sc = np.asarray(read_scale(zarr_path(r, split, name)))
        row = {int(v): i for i, v in enumerate(gt["node_ids"])}
        pos = np.stack([gt["z"], gt["y"], gt["x"]], axis=1)
        tt = gt["t"].astype(int)

        kids = {}
        for a, b in gt["edges"]:
            ra, rb = row.get(int(a)), row.get(int(b))
            if ra is None or rb is None:
                continue
            kids.setdefault(ra, []).append(rb)
            all_edges.append(float(np.linalg.norm((pos[rb] - pos[ra]) * sc)))

        divs = {p: c for p, c in kids.items() if len(c) >= 2}
        if not divs:
            continue

        def follow(nd, steps):
            path, cur = [nd], nd
            for _ in range(steps):
                ch = kids.get(cur, [])
                if len(ch) != 1:
                    break
                cur = ch[0]; path.append(cur)
            return path

        img, frames = None, {}
        for p, ch in divs.items():
            n_div += 1
            c1, c2 = ch[:2]
            t0 = int(tt[p])
            for c in (c1, c2):
                div_edges.append(float(np.linalg.norm((pos[c] - pos[p]) * sc)))
            A, B = follow(c1, FOLLOW), follow(c2, FOLLOW)
            rec = {"dataset": name, "t": t0, "z": int(pos[p][0]),
                   "y": int(pos[p][1]), "x": int(pos[p][2])}
            for k in range(min(len(A), len(B))):
                if k + 1 > FOLLOW:
                    break
                d = (pos[A[k]] - pos[B[k]]) * sc
                dist = float(np.linalg.norm(d))
                sep[k + 1].append(dist)
                if dist > 0:
                    sepz[k + 1].append(abs(float(d[0])) / dist)
                rec[f"sep_t+{k+1}"] = round(dist, 3)

            # --- image features, parent vs controls in the SAME frame ---
            try:
                if img is None:
                    img = open_image(zarr_path(r, split, name))
                if t0 not in frames:
                    frames[t0] = np.asarray(img[t0])
                    n_img += 1
                vol = frames[t0]
                ftot, fpk, fel = _features(vol, int(pos[p][0]), int(pos[p][1]),
                                           int(pos[p][2]), sc)
                excl = {p, c1, c2}
                cand = [i for i in np.nonzero(tt == t0)[0] if i not in excl]
                if len(cand) > N_CONTROL:
                    cand = list(rng.choice(cand, N_CONTROL, replace=False))
                ct, cp, ce = [], [], []
                for i in cand:
                    a, b, c = _features(vol, int(pos[i][0]), int(pos[i][1]),
                                        int(pos[i][2]), sc)
                    ct.append(a); cp.append(b); ce.append(c)
                if ct:
                    # normalise by the frame's own controls -> unit-free
                    P["tot"].append(ftot / (np.nanmedian(ct) + 1e-9))
                    P["peak"].append(fpk / (np.nanmedian(cp) + 1e-9))
                    P["elong"].append(fel)
                    C["tot"] += list(np.asarray(ct) / (np.nanmedian(ct) + 1e-9))
                    C["peak"] += list(np.asarray(cp) / (np.nanmedian(cp) + 1e-9))
                    C["elong"] += [v for v in ce]
                    rec.update(tot_ratio=round(P["tot"][-1], 3),
                               peak_ratio=round(P["peak"][-1], 3),
                               elong=round(fel, 3) if fel == fel else "")
            except Exception as e:
                warn_once("img_features",
                          f"{name}: could not read the image for the intensity/"
                          f"elongation features ({type(e).__name__}: {e}) -- "
                          f"those two rows below are computed from FEWER events "
                          f"than the geometric ones")
            rows.append(rec)

        if log_every and (di == 1 or di % log_every == 0 or di == len(names)):
            print(f"  [{di}/{len(names)}] {n_div} divisions so far", flush=True)

    if not n_div:
        print("no divisions found -- nothing to measure")
        return None

    print(f"\n=== {n_div} divisions, {n_img} frames read ===")

    print("\n1. sister separation after the split")
    print(f"{'frame':>8}{'n':>6}{'p10':>8}{'p50':>8}{'p90':>8}   "
          f"{'median |dz| share':>18}")
    for k in range(1, FOLLOW + 1):
        v = np.asarray(sep[k])
        if len(v) < 3:
            continue
        zs = np.median(sepz[k]) if sepz[k] else float("nan")
        print(f"{'t+'+str(k):>8}{len(v):6d}"
              f"{np.percentile(v,10):8.2f}{np.percentile(v,50):8.2f}"
              f"{np.percentile(v,90):8.2f}{zs:19.2f}")
    v1 = np.asarray(sep[1])
    if len(v1):
        print(f"  below the 5.0 um NMS kernel at t+1: "
              f"{(v1 < 5).mean():.1%} of divisions")

    print("\n2. edge length: parent->daughter vs ordinary GT edges")
    de, ae = np.asarray(div_edges), np.asarray(all_edges)
    print(f"{'':>16}{'n':>8}{'p50':>8}{'p90':>8}{'p99':>8}")
    print(f"{'division edges':>16}{len(de):8d}{np.percentile(de,50):8.2f}"
          f"{np.percentile(de,90):8.2f}{np.percentile(de,99):8.2f}")
    print(f"{'all GT edges':>16}{len(ae):8d}{np.percentile(ae,50):8.2f}"
          f"{np.percentile(ae,90):8.2f}{np.percentile(ae,99):8.2f}")
    print(f"  a division edge is longer than {_auc(de, ae):.0%} of ordinary edges "
          f"(AUC; 50% would mean no difference)")

    print("\n3./4. image features, dividing cell vs controls in the same frame")
    print(f"{'feature':>14}{'n_div':>7}{'n_ctrl':>8}{'median div':>12}"
          f"{'median ctrl':>13}{'AUC':>8}")
    for key, label in (("tot", "intensity sum"), ("peak", "peak intensity"),
                       ("elong", "elongation")):
        p, c = np.asarray(P[key], float), np.asarray(C[key], float)
        p, c = p[np.isfinite(p)], c[np.isfinite(c)]
        if len(p) < 3 or len(c) < 3:
            continue
        print(f"{label:>14}{len(p):7d}{len(c):8d}{np.median(p):12.3f}"
              f"{np.median(c):13.3f}{_auc(p, c):8.2f}")
    print("\n  AUC 0.5 = the feature cannot tell a dividing cell from a normal "
          "one.\n  Drop anything below ~0.65; it will not survive as a gate.")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "division_features.csv")
    cols = ["dataset", "t", "z", "y", "x"] + \
           [f"sep_t+{k}" for k in range(1, FOLLOW + 1)] + \
           ["tot_ratio", "peak_ratio", "elong"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nwrote {path}  ({len(rows)} divisions) -- send this file, "
          f"do not print it")
    warning_summary()
    return {"n_divisions": n_div, "n_datasets": len(names),
            "auc_edge_length": float(round(_auc(de, ae), 3)),
            **{f"auc_{k}": (float(round(_auc(P[k], C[k]), 3))
                            if len(P[k]) > 2 else None) for k in P}}


if __name__ == "__main__":
    import argparse
    a, _ = argparse.ArgumentParser().parse_known_args()
    run()
