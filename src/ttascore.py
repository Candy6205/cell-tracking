"""ttascore.py -- the READOUT half of ttaprobe, redone on CPU off its caches.

WHY THIS EXISTS. `biohub-ttaprobe` v3 COMPLETED: 36 passes (12 holdout 6bba
movies x noTTA/tta4/tta8), 69 min of T4, 23,760 extra U-Net passes, every
probability cache written. Its AUC table is WRONG, and the bug is one
subscript:

    prim = load_primaries(name, "F4")        # [ (N_t, 3) ] per frame
    mp = match(np.asarray(prim[t_][0], ...)  # <-- row 0 of the frame

`load_primaries` returns a LIST OF ARRAYS, not a list of (coords, scores)
tuples, so `prim[t_][0]` is the FIRST DETECTION of frame t, not the frame.
Only detection index 0 could ever be matched, as source or as target. The
table it printed rests on 565 rows where the same labelling rule over the
same 12 movies gives ~15,000 (accdiag/acchp, progress.md). 4% of the
population, and the surviving 4% is not a random sample of it -- it is
"whatever landed at index 0".

The probabilities on disk are unaffected: the bug is in the labeller, which
runs after every cache is written. So this redoes the labelling and the AUC
on CPU, off those caches, and costs no GPU.

WHAT IT ANSWERS, unchanged from the reading fixed before the GPU run: does
averaging the U-Net features over D4 views make the EDGE PROBABILITY more
informative? Single-column AUC against poolsize's own labels -- no bench, no
ILP, no thresholds.

  * flat AUC with the probabilities moved -> the mechanism works and buys
    nothing on our features; the ~20 GPU hours for real caches are REFUSED.
  * tta8 > tta4 > noTTA -> worth the caches, and the view count matters,
    which is their claim.
  * AUC DOWN -> their +0.005 came from something else in that step, or it
    needs the eight-view DETECTION ensemble we do not run.

THE ANCHOR that would have caught the bug. Before labelling anything, this
rebuilds `gated_pairs` from its own primaries and asserts, per movie:
  (a) the per-frame pair COUNT equals the cache's, and
  (b) every cached (i, j) is within GATE_UM under those primaries.
If the indexing is off by anything, both fail. It also refuses to print a
table on fewer than 5,000 labelled rows, which is the check that was
missing.

Attach: faustine649/biohub-ttaprobe (36 caches), biohub-fusecache +
biohub-fuse6bba (fdet_F4_), the support pack (wheels for geff/zarr),
faustine649/cell-tracking (viz_cells), competition data. CPU only.
"""
import sys
import pathlib
import shutil
import subprocess
import time
from collections import defaultdict

import numpy as np

SCRIPT_VERSION = "2026-09-09b-ttascore"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

SCALE = np.array([1.625, 0.40625, 0.40625])
UNET_THRESH = 0.15
MATCH_UM = 5.0
GATE_UM = 10.0
ARMS = ["noTTA", "tta4", "tta8"]
VIEWS = {"noTTA": 1, "tta4": 4, "tta8": 8}
MIN_ROWS = 5000

BASE = pathlib.Path("/kaggle/input")

# ---- wheels: read_geff needs geff/zarr, which the image does not ship -----
# Named explicitly.  An rglob over /kaggle/input would descend into the
# competition's .zarr stores and never come back (toolchain trap 5).
PACK = BASE / "datasets/pilkwang/biohub-tracking-support-pack-50ep-v1"
if not PACK.exists():                    # some mounts drop the "datasets/" level
    PACK = BASE / "biohub-tracking-support-pack-50ep-v1"
if not PACK.exists():
    entries = sorted(e.name for e in BASE.iterdir()) if BASE.exists() else []
    raise SystemExit(
        f"support pack not attached. /kaggle/input holds {len(entries)} "
        f"entries: {entries}. If this is empty the notebook was pushed "
        f"without data sources -- use `py kag.py kernels push -p .`, not the "
        f"MCP connector.")
WHEELS = sorted(PACK.rglob("*.whl"))
assert WHEELS, f"no wheels under {PACK}"
KEEP_OUT = {"numpy", "scipy", "torch", "pandas", "pyarrow", "numba", "llvmlite",
            "matplotlib", "pillow", "networkx", "scikit-learn", "scikit-image",
            "setuptools", "wheel", "pip"}
WH = pathlib.Path("/kaggle/working/wheelhouse")
shutil.rmtree(WH, ignore_errors=True)
WH.mkdir(parents=True)
for w in WHEELS:
    if w.name.split("-")[0].replace("_", "-").lower() not in KEEP_OUT:
        shutil.copy(w, WH / w.name)
staged = sorted(WH.glob("*.whl"))
r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps"]
    + [str(w) for w in staged], capture_output=True, text=True)
print(f"pip exit {r.returncode} ({len(staged)} wheels)", flush=True)
if r.returncode != 0:
    print(r.stderr[-1500:], flush=True)
shutil.rmtree(WH, ignore_errors=True)

from scipy.spatial import cKDTree                              # noqa: E402
from sklearn.metrics import roc_auc_score                      # noqa: E402

CODE_ROOTS = [
    "/kaggle/input/datasets/faustine649/cell-tracking/cell tracking",
    "/kaggle/input/datasets/faustine649/cell-tracking",
    "/kaggle/input/cell-tracking/cell tracking",
    "/kaggle/input/cell-tracking",
]
_code = next((c for c in CODE_ROOTS
              if pathlib.Path(c, "viz_cells.py").exists()), None)
assert _code is not None, "faustine649/cell-tracking is not mounted"
sys.path.insert(0, _code)
from viz_cells import read_geff                                # noqa: E402


def _walk(pred):
    """Shallow walk of /kaggle/input for *.npz, never descending into a
    .zarr or .geff (millions of chunk files -- toolchain trap 5)."""
    out, stack = {}, [BASE]
    while stack:
        d = stack.pop()
        if d.name.endswith((".zarr", ".geff")):
            continue
        try:
            for f in d.glob("*.npz"):
                k = pred(f.name)
                if k is not None:
                    out[k] = f
            if len(d.parts) - len(BASE.parts) < 6:
                stack += [p for p in d.iterdir() if p.is_dir()]
        except (PermissionError, OSError):
            pass
    return out


def _fdet_key(n):
    if n.startswith("fdet_F4hp_") or not n.startswith("fdet_F4_"):
        return None
    return n[len("fdet_F4_"):-len(".npz")]


FDET = _walk(_fdet_key)
print(f"fdet_F4 caches: {len(FDET)}", flush=True)

PROB = {}
for a in ARMS:
    pre = "edgeprob_" + a + "_"
    PROB[a] = _walk(lambda n, p=pre: n[len(p):-len(".npz")]
                    if n.startswith(p) else None)
    print(f"  {a}: {len(PROB[a])} probability caches", flush=True)

NAMES = sorted(set.intersection(*[set(PROB[a]) for a in ARMS]) & set(FDET))
print(f"movies in ALL arms and carrying an fdet_F4 cache: {len(NAMES)}",
      flush=True)
assert len(NAMES) == 12, (len(NAMES), NAMES)

COMP = next((p for p in (BASE / "biohub-cell-tracking-during-development",
                         BASE / "competitions" /
                         "biohub-cell-tracking-during-development")
             if p.exists()), None)
assert COMP is not None, "competition data not mounted"
TRAIN = COMP / "train"


def primaries(name):
    """[(N_t, 3) full-res float] per frame, seeds_of order -- exactly what
    ttaprobe's `load_primaries(name, 'F4')` returns, and the order its
    cached (t, i, j) index against.  seeds_of is score >= UNET_THRESH with
    the order kept, nothing else."""
    z = np.load(FDET[name])
    T = 0
    while f"c{T}" in z.files:
        T += 1
    out = []
    for t in range(T):
        c = np.asarray(z[f"c{t}"], float).reshape(-1, 3)
        s = np.asarray(z[f"s{t}"], float).reshape(-1)
        out.append(c[s >= UNET_THRESH])
    z.close()
    return out


def anchor_gate(name, z, prim):
    """The check ttaprobe did not have.  Rebuild the GATE_UM gate from these
    primaries and demand the cache agrees: same pair count per frame, and
    every cached pair actually inside the gate.  A wrong primaries index
    fails both."""
    et = np.asarray(z["t"], int)
    ei = np.asarray(z["i"], int)
    ej = np.asarray(z["j"], int)
    n_cached = np.bincount(et, minlength=len(prim) - 1)
    bad_n = bad_d = 0
    worst = 0.0
    for t in range(len(prim) - 1):
        Pu, Cu = prim[t] * SCALE[None, :], prim[t + 1] * SCALE[None, :]
        if len(Pu) == 0 or len(Cu) == 0:
            n_gate = 0
        else:
            hits = cKDTree(Cu).query_ball_point(Pu, GATE_UM)
            n_gate = int(sum(len(h) for h in hits))
        if n_gate != int(n_cached[t]):
            bad_n += 1
        sel = np.where(et == t)[0]
        if len(sel):
            assert ei[sel].max() < len(Pu) and ej[sel].max() < len(Cu), (
                f"{name} t={t}: cached index out of range for these primaries")
            d = np.linalg.norm(Pu[ei[sel]] - Cu[ej[sel]], axis=1)
            worst = max(worst, float(d.max()))
            bad_d += int((d > GATE_UM + 1e-6).sum())
    assert bad_n == 0, (f"{name}: {bad_n} frames whose gated-pair COUNT "
                        f"differs from the cache -- primaries do not match")
    assert bad_d == 0, (f"{name}: {bad_d} cached pairs outside the gate "
                        f"(worst {worst:.3f} um) -- primaries do not match")
    return worst


def label_pairs(name, z, prim):
    """poolsize's labelling rule, `label_rows` verbatim.

    A row counts only when the SOURCE primary matched a GT node that HAS
    children; the label is 1 iff the TARGET primary matched one of them.
    The population every other number on this project uses.
    """
    gt = read_geff(TRAIN / f"{name}.geff")
    gpos = np.stack([np.asarray(gt[c], float)
                     for c in ("z", "y", "x")], axis=1) * SCALE
    gtt = np.asarray(gt["t"], int)
    row_of = {int(v): i for i, v in enumerate(gt["node_ids"])}
    child = {}
    for a_, b_ in gt["edges"]:
        ra, rb = row_of.get(int(a_)), row_of.get(int(b_))
        if ra is not None and rb is not None:
            child.setdefault(ra, []).append(rb)
    by_t = defaultdict(list)
    for i_ in range(len(gtt)):
        by_t[int(gtt[i_])].append(i_)

    def match(pos_um, idx):
        if not idx or len(pos_um) == 0:
            return {}
        tr = cKDTree(gpos[idx])
        out = {}
        for d_, x_ in enumerate(pos_um):
            dd, gi = tr.query(x_)
            if dd <= MATCH_UM:
                out[d_] = idx[int(gi)]
        return out

    et = np.asarray(z["t"], int)
    ei = np.asarray(z["i"], int)
    ej = np.asarray(z["j"], int)
    keep, lab = [], []
    for t_ in sorted(set(et.tolist())):
        mp = match(prim[t_] * SCALE[None, :], by_t.get(t_, []))
        mc = match(prim[t_ + 1] * SCALE[None, :], by_t.get(t_ + 1, []))
        det_of = {g: d for d, g in mc.items()}
        for r_ in np.where(et == t_)[0]:
            i_, j_ = int(ei[r_]), int(ej[r_])
            if i_ not in mp:
                continue
            kids = child.get(mp[i_], [])
            if not kids:
                continue
            tgt = {det_of.get(b_) for b_ in kids}
            tgt.discard(None)
            keep.append(int(r_))
            lab.append(1 if j_ in tgt else 0)
    return np.asarray(keep, int), np.asarray(lab, int)


COLS = ["p_fwd", "p_rev", "prod"]
P = {a: {c: [] for c in COLS} for a in ARMS}
Y = {a: [] for a in ARMS}
MOV = []
t0 = time.time()
print("", flush=True)
print("=== labelling, with the gate anchored per movie ===", flush=True)
for n in NAMES:
    prim = primaries(n)
    base_key = None
    worst = float("nan")
    per = {}
    for a in ARMS:
        z = np.load(PROB[a][n])
        key = (np.asarray(z["t"], np.int64), np.asarray(z["i"], np.int64),
               np.asarray(z["j"], np.int64))
        if base_key is None:
            worst = anchor_gate(n, z, prim)
            base_key = key
        else:
            # Every arm ran the SAME primaries through the SAME gate, so the
            # (t, i, j) row set must be identical.  If it is not, the arms
            # are not scoring the same pairs and nothing below compares.
            assert all(np.array_equal(base_key[c], key[c]) for c in range(3)), (
                f"{n}/{a}: (t, i, j) rows differ from the noTTA arm")
        k, y = label_pairs(n, z, prim)
        f = np.asarray(z["p_fwd"], float)
        rv = np.asarray(z["p_rev"], float)
        P[a]["p_fwd"].append(f[k])
        P[a]["p_rev"].append(rv[k])
        P[a]["prod"].append((f * rv)[k])
        Y[a].append(y)
        per[a] = y
        z.close()
    MOV.append(n)
    print(f"  {n}: {len(base_key[0]):>7,} gated pairs, "
          f"{len(per[ARMS[0]]):>6,} labelled "
          f"({per[ARMS[0]].mean():.4f} pos), worst gate dist {worst:.3f} um",
          flush=True)

NROW = sum(len(y) for y in Y[ARMS[0]])
print("", flush=True)
print(f"labelled rows: {NROW:,} over {len(MOV)} movies "
      f"({time.time() - t0:.0f}s)", flush=True)
assert NROW >= MIN_ROWS, (
    f"only {NROW:,} labelled rows over {len(MOV)} movies -- the same rule on "
    f"the same holdout gives ~15,000 (accdiag).  This is the ttaprobe bug "
    f"again; do not read a table off it.")


def pooled(a, col, idx=None):
    src = range(len(MOV)) if idx is None else idx
    p = np.concatenate([P[a][col][m] for m in src])
    y = np.concatenate([Y[a][m] for m in src])
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, p))


print("", flush=True)
print("=== EDGE-FEATURE TTA: does averaging the U-Net features over D4 views "
      "make the edge probability more informative? ===", flush=True)
print(f"  {NROW:,} labelled pairs, "
      f"{np.concatenate(Y[ARMS[0]]).mean():.4f} positive, 12 holdout 6bba "
      f"movies.  Same rows in every arm.", flush=True)
print("", flush=True)
hdr = f"  {'arm':<8}{'views':>6}"
for c in COLS:
    hdr += f"{'AUC ' + c:>12}{'vs noTTA':>10}"
hdr += f"{'|dp| p50':>10}{'|dp| p90':>10}"
print(hdr, flush=True)
base = {}
base_fwd = None
for a in ARMS:
    line = f"  {a:<8}{VIEWS[a]:>6}"
    for c in COLS:
        v = pooled(a, c)
        if a == ARMS[0]:
            base[c] = v
        line += f"{v:>12.5f}{v - base[c]:>+10.5f}"
    f = np.concatenate(P[a]["p_fwd"])
    if base_fwd is None:
        base_fwd = f
        line += f"{'--':>10}{'--':>10}"
    else:
        dp = np.abs(f - base_fwd)
        line += f"{np.percentile(dp, 50):>10.5f}{np.percentile(dp, 90):>10.5f}"
    print(line, flush=True)

print("", flush=True)
print("=== per movie, AUC of the product (what the solver reads) ===",
      flush=True)
print(f"  {'movie':<16}{'rows':>8}{'pos':>8}"
      f"{'noTTA':>10}{'tta4':>10}{'tta8':>10}   sign", flush=True)
win4 = win8 = 0
for m, n in enumerate(MOV):
    v = [pooled(a, "prod", [m]) for a in ARMS]
    s = ("+" if v[1] > v[0] else "-") + ("+" if v[2] > v[0] else "-")
    win4 += int(v[1] > v[0])
    win8 += int(v[2] > v[0])
    print(f"  {n:<16}{len(Y[ARMS[0]][m]):>8,}{Y[ARMS[0]][m].mean():>8.4f}"
          f"{v[0]:>10.5f}{v[1]:>10.5f}{v[2]:>10.5f}   {s}", flush=True)
print(f"  movies where TTA beats noTTA: tta4 {win4}/{len(MOV)}, "
      f"tta8 {win8}/{len(MOV)}", flush=True)

print("", flush=True)
print("=== movie-level paired bootstrap of the pooled delta, 2000 draws ===",
      flush=True)
rng = np.random.default_rng(0)
B = 2000
for a in ARMS[1:]:
    d = []
    for _ in range(B):
        idx = rng.integers(0, len(MOV), len(MOV))
        v0, v1 = pooled(ARMS[0], "prod", idx), pooled(a, "prod", idx)
        if v0 == v0 and v1 == v1:
            d.append(v1 - v0)
    d = np.asarray(d)
    lo, hi = np.percentile(d, [5, 95])
    print(f"  {a} - noTTA (prod): {np.mean(d):+.5f}  90% CI "
          f"[{lo:+.5f}, {hi:+.5f}]   P(>0) = {np.mean(d > 0):.3f}",
          flush=True)

print("", flush=True)
print("READ IT against the rule fixed before the GPU run.  Their +0.005 is a "
      "leaderboard delta on a full pipeline; nothing here reproduces that.  "
      "What this says is whether the edge probability improves AT ALL.",
      flush=True)
print("  * flat AUC WITH the probabilities moving -> the mechanism works and "
      "buys nothing on our features; the ~20 GPU hours are REFUSED.",
      flush=True)
print("  * flat AUC AND |dp| ~ 0 -> the wrapper did nothing.  Broken run, "
      "not a result.", flush=True)
print("  * tta8 > tta4 > noTTA -> worth the caches, and the view count "
      "matters, which is their claim.", flush=True)
print("TTASCORE OK", flush=True)
