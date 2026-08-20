#!/usr/bin/env python
"""
cell_count.py -- estimate each dataset's true cell count N_true from the image.

Why this exists:
  True cell density spans 27x across the train set (38 -> 1015 cells/frame), so
  no single detection threshold can fit all of them. Annotation density is
  NEGATIVELY correlated with true density (annotators labelled a larger
  fraction of the sparse, easy videos), so you cannot infer N_true from the GT
  -- and the test split has no GT at all.

  So: regress N_true from image features. The train split gives 199 labelled
  (image -> N_true) samples, where N_true is the geff metadata field
  estimated_number_of_nodes. Predict N_true for each test dataset, then
  calibrate the detection threshold per dataset so N_pred/N_true lands in
  0.9-1.0.

The main trap: random CV lies.
  Train contains only two imaging batches (44b6_ / 6bba_); test is likely a
  third. The more features you add, the prettier random CV looks and the worse
  cross-batch generalization gets. So model selection here uses
  leave-one-group-out (LOGO) only, and all absolute-intensity features are
  excluded by default (they encode laser power / exposure, not biology).

Usage:
  !pip install -q zarr geff scikit-learn
  python cell_count.py features --split train --out feats_train.csv --limit 12
  python cell_count.py features --split train --out feats_train.csv
  python cell_count.py features --split test  --out feats_test.csv
  python cell_count.py fit     --train feats_train.csv
  python cell_count.py predict --train feats_train.csv --test feats_test.csv

In a notebook:
  import cell_count
  cell_count.run_features(split="train", out="feats_train.csv", limit=12)
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from viz_cells import (
    DEFAULT_SCALE,
    _progress,
    warn_once,
    warning_summary,
    find_root,
    geff_path,
    group_of,
    list_datasets,
    open_image,
    read_geff,
    read_scale,
    zarr_path,
)

SCRIPT_VERSION = "2026-08-18f"

# Dimensionless features -- invariant to laser power / exposure. Used by default.
DIMLESS = [
    "fg_frac_k2", "fg_frac_k3", "fg_frac_k4",
    "maxima_per_100um3_s3", "maxima_per_100um3_s5", "maxima_per_100um3_s7",
    "p99_over_p50", "p95_over_p50", "p90_over_p50", "p75_over_p50",
    "iqr_over_med", "cv_downsampled", "skewness",
]

# Absolute-intensity features: excluded by default, kept only as a control.
ABSOLUTE = ["abs_p50", "abs_p99", "abs_std", "abs_max"]

TARGET = "n_true"

# Candidate feature subsets, smallest first. fit() picks the best LOGO score.
FEATURE_SETS = {
    "f1": ["maxima_per_100um3_s5"],
    "f3": ["maxima_per_100um3_s5", "fg_frac_k3", "p99_over_p50"],
    "f5": ["maxima_per_100um3_s3", "maxima_per_100um3_s5", "maxima_per_100um3_s7",
           "fg_frac_k3", "p99_over_p50"],
    "f8": ["maxima_per_100um3_s3", "maxima_per_100um3_s5", "maxima_per_100um3_s7",
           "fg_frac_k2", "fg_frac_k3", "fg_frac_k4",
           "p99_over_p50", "cv_downsampled"],
    "all_dimless": DIMLESS,
    "all_plus_abs": DIMLESS + ABSOLUTE,   # control; expected to be worst on LOGO
}


# ---------------------------------------------------------------- features

def _frame_features(vol: np.ndarray, scale: tuple = DEFAULT_SCALE) -> dict:
    """
    vol   : (Z, Y, X) single frame
    scale : this dataset's (z, y, x) um/voxel -- read per dataset, never hardcode

    Intensity statistics are all expressed as ratios or fractions so they do not
    encode absolute brightness.
    """
    from scipy import ndimage

    v = vol.astype(np.float32)
    p25, p50, p75, p90, p95, p99, _ = np.percentile(
        v, [25, 50, 75, 90, 95, 99, 99.9]
    )
    denom = max(p50, 1e-6)
    spread = max(p99 - p50, 1e-6)

    out = {
        "abs_p50": float(p50),
        "abs_p99": float(p99),
        "abs_std": float(v.std()),
        "abs_max": float(v.max()),
        "p99_over_p50": float(p99 / denom),
        "p95_over_p50": float(p95 / denom),
        "p90_over_p50": float(p90 / denom),
        "p75_over_p50": float(p75 / denom),
        "iqr_over_med": float((p75 - p25) / denom),
    }

    mu = float(v.mean())
    sd = float(v.std()) + 1e-6
    out["skewness"] = float(((v - mu) ** 3).mean() / sd**3)

    # Foreground volume fraction at thr = p50 + k*(p99-p50)/3. Dimensionless.
    for k in (2, 3, 4):
        out[f"fg_frac_k{k}"] = float((v > p50 + k * spread / 3.0).mean())

    # Coefficient of variation after downsampling: structure, not shot noise.
    ds = v[:: max(1, v.shape[0] // 16), :: 8, :: 8]
    out["cv_downsampled"] = float(ds.std() / (ds.mean() + 1e-6))

    # Local-maxima count, normalized to "per (100 um)^3".
    vol_um3 = (v.shape[0] * scale[0] * v.shape[1] * scale[1] * v.shape[2] * scale[2])
    norm = vol_um3 / 1e6
    thr_max = p50 + spread
    for sigma_um in (3, 5, 7):
        # Physically isotropic neighbourhood: Z is ~4x coarser, so the Z window
        # must be correspondingly smaller in voxels.
        sz = max(1, int(round(sigma_um / scale[0])))
        sy = max(1, int(round(sigma_um / scale[1])))
        mx = ndimage.maximum_filter(v, size=(sz | 1, sy | 1, sy | 1), mode="nearest")
        n = int(((v == mx) & (v > thr_max)).sum())
        out[f"maxima_per_100um3_s{sigma_um}"] = float(n / max(norm, 1e-6))

    return out


def dataset_features(img, scale: tuple = DEFAULT_SCALE, n_frames: int = 5,
                     max_voxels: int = 12_000_000) -> dict:
    """Median of per-frame features over a few evenly spaced frames."""
    T = img.shape[0]
    ts = np.unique(np.linspace(0, T - 1, min(n_frames, T)).astype(int))
    per = []
    for t in ts:
        vol = np.asarray(img[int(t)])
        if vol.ndim > 3:
            vol = vol.reshape(vol.shape[-3:])
        if vol.size > max_voxels:  # center-crop instead of blowing up memory
            Z, Y, X = vol.shape
            f = (max_voxels / vol.size) ** (1 / 3)
            cy, cx = int(Y * f) // 2, int(X * f) // 2
            vol = vol[:, Y // 2 - cy: Y // 2 + cy, X // 2 - cx: X // 2 + cx]
        per.append(_frame_features(vol, scale))
    return {k: float(np.median([d[k] for d in per])) for k in per[0]}


def run_features(root=None, split="train", out=None, limit=None, n_frames=5,
                 log_every=25):
    """
    log_every: progress line every N datasets (0 = silent). This loop reads
    image data, so it is slow -- but still do not print per dataset; a Kaggle
    cell can die on too much output.
    """
    r = find_root(root)
    names = list_datasets(r, split)
    if limit:
        names = names[: int(limit)]
        print(f"[warning] --limit {limit}: names are sorted, so this covers only "
              f"the earliest prefix group. Fine as a smoke test, useless for "
              f"cross-batch (LOGO) validation.")
    out = out or f"feats_{split}.csv"

    rows = []
    for i, name in enumerate(names, 1):
        row = {"dataset": name, "group": group_of(name)}
        try:
            zp = zarr_path(r, split, name)
            img = open_image(zp)
            sc = read_scale(zp)  # per dataset, never hardcoded
            row["T"] = int(img.shape[0])
            row.update(dataset_features(img, sc, n_frames=n_frames))
        except Exception as e:
            row["error"] = str(e)[:150]
            print(f"  [error] {name}: {e}", flush=True)
            rows.append(row)
            continue

        gp = geff_path(r, split, name)
        if gp.exists():
            try:
                est = read_geff(gp)["extra"].get("estimated_number_of_nodes")
                if est:
                    row[TARGET] = float(est)
                    row["n_true_per_frame"] = float(est) / max(row["T"], 1)
            except Exception as e:
                row["gt_error"] = str(e)[:120]

        _progress(i, len(names), log_every, name)
        rows.append(row)

    cols = (["dataset", "group", "T", TARGET, "n_true_per_frame"]
            + DIMLESS + ABSOLUTE + ["error", "gt_error"])
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\n[saved] {out}  ({len(rows)} rows)")
    warning_summary()
    return out


# ---------------------------------------------------------------- table I/O

def load_table(path: str):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r_ in csv.DictReader(f):
            d = {}
            for k, v in r_.items():
                if k in ("dataset", "group", "error", "gt_error"):
                    d[k] = v
                else:
                    try:
                        d[k] = float(v) if v not in ("", None) else np.nan
                    except Exception:
                        d[k] = np.nan
            rows.append(d)
    return rows


def _matrix(rows, feats):
    return np.array([[r_.get(f, np.nan) for f in feats] for r_ in rows], dtype=float)


def _clean(rows, feats):
    return [r_ for r_ in rows
            if np.isfinite(r_.get(TARGET, np.nan))
            and all(np.isfinite(r_.get(f, np.nan)) for f in feats)]


# ---------------------------------------------------------------- evaluation

def penalty_multiplier(n_pred: np.ndarray, n_true: np.ndarray) -> np.ndarray:
    """
    Official: adjusted = max(0, jaccard * (1 - 0.1 * (N_pred - N_true)/N_true))
    There is no upper clamp, so under-prediction is rewarded and
    over-prediction is penalized twice. Returned for reporting only -- see the
    note in run_fit() on why this must NOT drive model selection.
    """
    ratio = (n_pred - n_true) / np.maximum(n_true, 1e-9)
    return np.maximum(0.0, 1.0 - 0.1 * ratio)


def _fit_predict(Xtr, ytr, Xte, model="ridge"):
    """Regress in log space -- counts span 27x, log is far more linear."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if model == "rf":
        m = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0)
    else:
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))
    m.fit(Xtr, np.log(np.maximum(ytr, 1.0)))
    return np.exp(m.predict(Xte))


def evaluate_set(rows, feats, model="ridge"):
    """Return (random 5-fold MdAPE, {group: LOGO MdAPE}, worst penalty mult, n)."""
    from sklearn.model_selection import KFold

    sub = _clean(rows, feats)
    if len(sub) < 8:
        return None
    X = _matrix(sub, feats)
    y = np.array([r_[TARGET] for r_ in sub], dtype=float)
    g = np.array([r_["group"] for r_ in sub])

    # Random 5-fold -- reported as a control only; it lies across imaging batches.
    pred = np.zeros_like(y)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        pred[te] = _fit_predict(X[tr], y[tr], X[te], model)
    cv_err = float(np.median(np.abs(pred - y) / y))

    logo = {}
    worst_mult = 1.0
    for grp in np.unique(g):
        tr, te = g != grp, g == grp
        if tr.sum() < 5 or te.sum() < 2:
            continue
        p = _fit_predict(X[tr], y[tr], X[te], model)
        logo[str(grp)] = float(np.median(np.abs(p - y[te]) / y[te]))
        worst_mult = min(worst_mult, float(np.median(penalty_multiplier(p, y[te]))))
    return cv_err, logo, worst_mult, len(sub)


def run_fit(train="feats_train.csv", model="ridge"):
    rows = load_table(train)
    have = [r_ for r_ in rows if np.isfinite(r_.get(TARGET, np.nan))]
    print(f"[data] {len(rows)} rows, {len(have)} with N_true")
    groups = sorted({r_["group"] for r_ in have})
    print(f"[imaging groups] {groups}  <- LOGO leaves one of these out")
    if len(groups) < 2:
        print("[warning] only one group present: LOGO cannot run, so the numbers "
              "below say nothing about cross-batch generalization. Extract "
              "features for ALL datasets before trusting this table.")

    y = np.array([r_[TARGET] for r_ in have], dtype=float)
    base = float(np.median(y))
    base_err = float(np.median(np.abs(base - y) / y))
    print(f"[baseline] predict the global median {base:.0f} -> MdAPE "
          f"{100*base_err:.1f}%")
    print("           a model is only worth using if it beats this.\n")

    print(f"{'set':<13}{'k':>3}{'n':>5}{'randCV':>9}{'LOGOworst':>11}"
          f"  {'LOGO per group':<28}{'penalty':>8}")
    print("-" * 92)

    best = None
    for nameset, feats in FEATURE_SETS.items():
        res = evaluate_set(rows, feats, model)
        if res is None:
            print(f"{nameset:<13}  not enough samples, skipped")
            continue
        cv, logo, worst_mult, n = res
        worst_err = max(logo.values()) if logo else float("nan")
        logo_s = " ".join(f"{k}:{100*v:.1f}%" for k, v in logo.items()) or "n/a"
        print(f"{nameset:<13}{len(feats):>3}{n:>5}{100*cv:>8.1f}%"
              f"{100*worst_err:>10.1f}%  {logo_s:<28}{worst_mult:>8.3f}")
        # Selection criterion: SMALLEST worst-group LOGO error. Not random CV,
        # and deliberately not the penalty multiplier -- that metric rewards
        # under-prediction, so a model that crushes the count downward scores
        # > 1 while being useless: calibrating a detector to it makes the
        # threshold far too strict and floods the result with false negatives.
        # Ties go to the smaller feature set (less able to memorize the batch).
        key = (-worst_err, -len(feats))
        if best is None or key > best[0]:
            best = (key, nameset, feats, cv, logo, worst_mult)

    if best is None:
        print("\nNo usable feature set.")
        return None

    _, nameset, feats, cv, logo, worst_mult = best
    worst_err = max(logo.values()) if logo else float("nan")
    print(f"\n>> selected {nameset} ({len(feats)} features): {feats}")
    print(f"   random CV MdAPE   = {100*cv:.1f}%   <- do not trust this")
    print("   LOGO per group    = "
          + (" ".join(f"{k}:{100*v:.1f}%" for k, v in logo.items()) or "n/a"))
    print(f"   LOGO worst group  = {100*worst_err:.1f}%   <- expect this on test")
    print(f"   penalty mult      = {worst_mult:.3f}  (reported only, not a "
          f"selection criterion)")
    if logo and worst_err > 3 * cv:
        print("   [!] LOGO is >3x worse than random CV -> strong domain shift; "
              "the model is memorizing the imaging batch.")
    if logo and worst_err > base_err:
        print(f"   [!!] the worst group ({100*worst_err:.1f}%) is worse than just "
              f"predicting the global median ({100*base_err:.1f}%).")
        print("        Do NOT use the model. Use the median, or fall back to "
              "sweeping a threshold per dataset.")
    return nameset, feats


def run_predict(train="feats_train.csv", test="feats_test.csv",
                out="n_true_pred.csv", model="ridge", featureset=None):
    tr_rows = load_table(train)
    te_rows = load_table(test)

    if featureset:
        feats = FEATURE_SETS[featureset]
    else:
        picked = run_fit(train, model)
        if picked is None:
            return None
        _, feats = picked
        print()

    sub = _clean(tr_rows, feats)
    X = _matrix(sub, feats)
    y = np.array([r_[TARGET] for r_ in sub], dtype=float)

    ok_i = [i for i, r_ in enumerate(te_rows)
            if all(np.isfinite(r_.get(f, np.nan)) for f in feats)]
    ok_set = set(ok_i)
    ok = [te_rows[i] for i in ok_i]
    bad = [r_ for i, r_ in enumerate(te_rows) if i not in ok_set]

    pred = _fit_predict(X, y, _matrix(ok, feats), model) if ok else np.array([])
    fallback = float(np.median(y))

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "T", "n_true_pred", "n_true_pred_per_frame", "source"])
        for r_, p in zip(ok, pred):
            T = r_.get("T", np.nan)
            w.writerow([r_["dataset"], T, f"{p:.0f}",
                        f"{p/max(T,1):.2f}" if np.isfinite(T) else "", "model"])
        for r_ in bad:
            warn_once("target_fallback_median",
                      f"{r_['dataset']}: features missing -> predicting the "
                      f"global median {fallback:.0f} instead of a real "
                      f"estimate (marked fallback_median in the CSV)")
            w.writerow([r_["dataset"], r_.get("T", np.nan), f"{fallback:.0f}", "",
                        "fallback_median"])
    print(f"[saved] {out}  ({len(ok)} from model + {len(bad)} fallback)")
    if len(pred):
        print(f"  predicted range: {pred.min():.0f} .. {pred.max():.0f}  "
              f"(train truth {y.min():.0f} .. {y.max():.0f})")
        if pred.max() > 3 * y.max() or pred.min() < y.min() / 3:
            print("  [!] predictions extrapolate well outside the train range.")
    return out


# ---------------------------------------------------------------- CLI

def main(argv=None):
    p = argparse.ArgumentParser(description="Estimate N_true from images")
    p.add_argument("mode", nargs="?", default="fit",
                   choices=["features", "fit", "predict"])
    p.add_argument("--root", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--frames", type=int, default=5, help="frames sampled per dataset")
    p.add_argument("--log-every", type=int, default=25,
                   help="progress line every N datasets; 0 = silent")
    p.add_argument("--train", default="feats_train.csv")
    p.add_argument("--test", default="feats_test.csv")
    p.add_argument("--model", default="ridge", choices=["ridge", "rf"])
    p.add_argument("--featureset", default=None, choices=list(FEATURE_SETS))

    a, unknown = p.parse_known_args(argv)
    if unknown:
        print(f"[ignoring unknown args] {unknown}", file=sys.stderr)

    if a.mode == "features":
        run_features(a.root, a.split, a.out, a.limit, a.frames, a.log_every)
    elif a.mode == "fit":
        run_fit(a.train, a.model)
    else:
        run_predict(a.train, a.test, a.out or "n_true_pred.csv", a.model,
                    a.featureset)


if __name__ == "__main__":
    main()
