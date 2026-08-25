"""Rebuild the Colab environment in ONE detached job.

The first build was three separate synchronous cells (setup1/2/3) that
each had to be babysat. This does the whole thing in one pass with a
log, so it can be launched and polled.

Every version here is a FIX THAT WAS ALREADY PAID FOR, not a guess --
see docs/kaggle-toolchain.md section 11:

  tracksdata 0.1.0rc6.dev3+g980c2d30a   Kaggle's exact build. Install
      the PyPI release first for the dependency tree, then force the
      wheel over the top with --no-deps.
  polars 1.42.0    Colab ships older; tracksdata's requirement was
      already "satisfied" so pip left it, and the pipeline died on
      `polars has no attribute 'Float16'`.
  rustworkx 0.18.0, sqlalchemy 2.0.51   tracksdata stores graphs
      through both.
  geff 1.2.0.1.1, zarr 3.2.1   the geff reader/writer pair.

The zarr double-import trap: importing zarr, then reinstalling it, then
using it gives two `Buffer` classes with the same name and different
identity. Nothing here imports zarr before the installs finish, and the
verification runs in a FRESH interpreter (this whole file is one), so
the trap cannot fire.
"""
import pathlib
import subprocess
import sys
import time

t0 = time.time()
print("ENV BUILD START", time.strftime("%H:%M:%S", time.gmtime()),
      flush=True)


def sh(*a, **kw):
    r = subprocess.run(list(a), capture_output=True, text=True, **kw)
    tail = [x for x in (r.stdout + r.stderr).strip().split("\n") if x]
    print(f"  rc={r.returncode}  {tail[-1][:120] if tail else ''}",
          flush=True)
    return r.returncode


def pip(*a):
    return sh(sys.executable, "-m", "pip", "install", "-q", *a)


# ---- 1. the competition repo ------------------------------------------
print("\n1. git clone the competition repo", flush=True)
if pathlib.Path("/content/repo").exists():
    print("  already present", flush=True)
else:
    sh("git", "clone", "--depth", "1",
       "https://github.com/royerlab/kaggle-cell-tracking-competition.git",
       "/content/repo")

# ---- 2. the data pack --------------------------------------------------
print("\n2. unpack the data", flush=True)
DATA = pathlib.Path("/content/data")
DATA.mkdir(exist_ok=True)
sh("tar", "-xzf", "/content/ctpack.tgz", "-C", str(DATA))
det = sorted(DATA.glob("detcache_*.npz"))
gts = sorted(DATA.glob("gt_*.npz"))
pkl = sorted(DATA.glob("*.pkl"))
print(f"  detcaches {len(det)}, gt {len(gts)}, ranker {len(pkl)}",
      flush=True)
# division_recovery.py and viz_cells.py must be importable as top level
for name in ("division_recovery.py", "viz_cells.py"):
    src = DATA / name
    if src.exists():
        pathlib.Path("/content", name).write_bytes(src.read_bytes())
        print(f"  placed /content/{name}", flush=True)
    else:
        print(f"  MISSING {name} -- the pack is incomplete", flush=True)

# ---- 3. the stack ------------------------------------------------------
print("\n3. tracksdata: PyPI release for the dependency tree", flush=True)
pip("tracksdata==0.1.0rc8")
print("   ... then Kaggle's exact build over the top", flush=True)
WHEEL = ("/content/tracksdata-0.1.0rc6.dev3+g980c2d30a-py3-none-any.whl")
if pathlib.Path("/content/td.whl").exists():
    pathlib.Path(WHEEL).write_bytes(
        pathlib.Path("/content/td.whl").read_bytes())
pip("--force-reinstall", "--no-deps", WHEEL)

print("\n4. pin the support pack's versions", flush=True)
pip("polars==1.42.0", "rustworkx==0.18.0", "sqlalchemy==2.0.51",
    "geff==1.2.0.1.1", "zarr==3.2.1")

# ---- 5. verify ---------------------------------------------------------
print("\n5. imports", flush=True)
sys.path.insert(0, "/content/repo/src")
sys.path.insert(0, "/content/repo/scripts")
sys.path.insert(0, "/content")

ok = True
for mod in ("polars", "tracksdata", "predict_unet_transformer",
            "tracking_cellmot.io", "tracking_cellmot.metrics",
            "division_recovery", "viz_cells"):
    try:
        m = __import__(mod)
        print("  OK      %-28s %s"
              % (mod, getattr(m, "__version__", "")), flush=True)
    except Exception as e:
        ok = False
        print("  FAIL    %-28s %s: %s"
              % (mod, type(e).__name__, str(e)[:90]), flush=True)

if ok:
    import tracksdata as td
    from tracking_cellmot.metrics import evaluate, per_sample_metrics
    print(f"\n  tracksdata {td.__version__}", flush=True)
    print("  wanted     0.1.0rc6.dev3+g980c2d30a", flush=True)
    print(f"  ILPSolver: {hasattr(td.solvers, 'ILPSolver')}", flush=True)
    print(f"  scorer:    {evaluate is not None and per_sample_metrics is not None}",
          flush=True)
    # which solver backend is actually available -- Kaggle falls back to
    # SCIP because there is no Gurobi licence, and the two can break ties
    # differently, which is exactly what the acceptance test is for.
    try:
        import ilpy
        print(f"  ilpy preference: {ilpy.Preference.Any}", flush=True)
    except Exception as e:
        print(f"  ilpy probe failed: {e}", flush=True)
    print(f"\nENV READY after {time.time() - t0:.0f}s", flush=True)
else:
    print(f"\nENV FAILED after {time.time() - t0:.0f}s", flush=True)
