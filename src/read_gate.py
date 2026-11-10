"""
Read the small metadata files inside the DeepCenter artefact.

check_weights.py answered "who trained on what". This answers "how was the
detection gate tuned, and is the checkpoint the notebook loads any good".
CPU only, seconds to run. Attach both weight datasets.

OUTPUT DISCIPLINE (engineering-rules.md rule 2): every file here is under
50 KB and is summarised, not dumped. Total output is ~80 lines.
WALK DISCIPLINE (rule 9): fixed paths only, no rglob over /kaggle/input.
"""

import csv
import json
import os

DC = "/kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1"
SP = "/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1"
GATE = f"{DC}/weights/full_frame_center"


def _p(msg=""):
    print(msg, flush=True)


def _load(path):
    """Returns None and WARNS on failure -- never silently skips (rule 1)."""
    if not os.path.exists(path):
        _p(f"[degraded] missing: {path} -- section skipped, "
           f"any conclusion below is drawn without it")
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        _p(f"[degraded] unreadable {os.path.basename(path)}: {e}")
        return None


def _dump(d, keys, indent="  "):
    for k in keys:
        if k not in d:
            _p(f"{indent}{k}: <absent>")
            continue
        v = d[k]
        if isinstance(v, dict):
            _p(f"{indent}{k}:")
            for kk, vv in v.items():
                _p(f"{indent}  {kk}: {str(vv)[:160]}")
        else:
            _p(f"{indent}{k}: {str(v)[:300]}")


def section_config():
    _p("=== deepcenter training config ===")
    d = _load(f"{GATE}/config.json")
    if d:
        _p("  " + ", ".join(f"{k}={v}" for k, v in sorted(d.items())))


def section_contract():
    _p("\n=== how the gate is applied (ARTIFACT_MANIFEST) ===")
    d = _load(f"{DC}/ARTIFACT_MANIFEST.json")
    if d:
        _dump(d, ["model", "coordinate_contract", "gate_and_blend_contract",
                  "training_contract"])
    _p("\n=== snapshot manifest ===")
    d = _load(f"{GATE}/SNAPSHOT_MANIFEST.json")
    if d:
        _dump(d, [k for k in d if k not in ("files", "config")])
        if isinstance(d.get("config"), dict):
            _p("  config: " + ", ".join(f"{k}={v}" for k, v in
                                        sorted(d["config"].items()))[:400])


def section_threshold():
    """The sweep the author used to pick BIOHUB_DET_THRESHOLD = 0.96875."""
    _p("\n=== gate threshold sweep ===")
    d = _load(f"{GATE}/gate_summary.json")
    if d:
        for k in ("evaluated_frames", "match_radius_um", "peak_min_distance",
                  "peak_sample_limit", "notes"):
            if k in d:
                _p(f"  {k}: {str(d[k])[:300]}")
        thr, met = d.get("thresholds"), d.get("threshold_metrics")
        if isinstance(thr, list) and isinstance(met, list):
            cols = sorted(met[0]) if isinstance(met[0], dict) else []
            _p("  " + "thr".ljust(10) + "".join(c[:11].ljust(13) for c in cols))
            for t, m in zip(thr, met):
                if not isinstance(m, dict):
                    continue
                row = "".join(
                    (f"{m[c]:.4f}" if isinstance(m[c], float) else str(m[c])
                     )[:11].ljust(13) for c in cols)
                _p(f"  {str(t)[:9].ljust(10)}{row}")
    path = f"{GATE}/gate_threshold_metrics.csv"
    if os.path.exists(path):
        _p(f"\n  -- {os.path.basename(path)} --")
        with open(path) as f:
            for i, line in enumerate(f):
                if i > 20:
                    _p("  ... truncated")
                    break
                _p("  " + line.rstrip()[:200])
    else:
        _p(f"[degraded] missing {path}")


def section_history():
    """Is checkpoint_last.pt (epoch 500) better or worse than best.pt (epoch 2)?"""
    _p("\n=== training history: every 50th epoch ===")
    path = f"{GATE}/history.csv"
    if not os.path.exists(path):
        _p(f"[degraded] missing {path}")
        return
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        _p("[degraded] history.csv is empty")
        return
    cols = [c for c in rows[0] if c in
            ("epoch", "train_loss", "val_loss", "score", "minutes")] or list(rows[0])
    _p("  " + "".join(c[:11].ljust(13) for c in cols))
    keep = rows[::50] + [rows[-1]]
    for r in keep:
        _p("  " + "".join(str(r.get(c, ""))[:11].ljust(13) for c in cols))
    if "val_loss" in rows[0]:
        vals = [(float(r["val_loss"]), int(float(r["epoch"]))) for r in rows
                if r.get("val_loss") not in (None, "")]
        best_v, best_e = min(vals)
        last_v, last_e = vals[-1]
        _p(f"\n  best val_loss {best_v:.5f} at epoch {best_e}")
        _p(f"  last val_loss {last_v:.5f} at epoch {last_e}"
           f"  ({last_v / best_v:.1f}x worse)")
        _p("  The notebook loads checkpoint_last.pt and asserts epoch == 500.")


def section_support():
    _p("\n=== support pack manifest ===")
    d = _load(f"{SP}/ARTIFACT_MANIFEST.json")
    if d:
        _dump(d, ["model", "models", "notes", "compatibility", "source"])


SPLIT_PAT = ("split", "movie_limit", "val_fraction", "--train", "--val",
             "44b6", "6bba", "group", "manifest", "shuffle", "seed")


def _grep(path, pats, max_hits=30, width=140):
    """Print only the lines that mention one of `pats`. Never the whole file."""
    if not os.path.exists(path):
        _p(f"[degraded] missing {path} -- cannot show how the split was made")
        return
    _p(f"  -- {os.path.basename(path)} --")
    hits = 0
    with open(path, errors="replace") as f:
        for n, line in enumerate(f, 1):
            low = line.lower()
            if not any(t.lower() in low for t in pats):
                continue
            hits += 1
            if hits > max_hits:
                continue
            _p(f"  {n:>5}: {line.strip()[:width]}")
    if hits > max_hits:
        _p(f"  ... {hits - max_hits} more matching lines not shown")
    if hits == 0:
        _p("  (no matching lines)")


def section_howsplit():
    """WHY is train the 71 44b6 movies and val the 128 6bba ones?
    The launch script and the trainer's own split code are both shipped."""
    _p("\n=== how the 71/128 split was actually produced ===")
    _grep(f"{DC}/source_scripts/run_full_frame_center_training.sh", SPLIT_PAT)
    _p("")
    _grep(f"{DC}/source_scripts/train_full_frame_center_detector.py", SPLIT_PAT)


if __name__ == "__main__":
    for p in (DC, SP):
        if not os.path.isdir(p):
            _p(f"[degraded] not attached: {p}")
    section_config()
    section_contract()
    section_threshold()
    section_history()
    section_support()
    section_howsplit()
