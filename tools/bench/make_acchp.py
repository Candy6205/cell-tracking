"""accdiag.py -> acchp.py: does hpara's smoother placement reach the FEATURES?

WHY THIS EXISTS. accdiag closed the acceleration line with an argument that
is arithmetic, not a measurement:

    pred     = 2.0*x_t - 1.0*x_{t-1}                   var  5.00 * sigma^2
    pred_acc = 2.5*x_t - 2.0*x_{t-1} + 0.5*x_{t-2}     var 10.50 * sigma^2

    "A second-order extrapolator needs a smoother position estimate than
     single-frame centroids provide."

`hpara` is exactly a smoother position estimate -- jitter p50 1.29 -> 0.89
um on 6bba, refinecmp. So the entry names the condition and hparacache
supplies it, on the SAME 12 movies accdiag ran.

WHAT IS PREDICTED, before the run. The variance ratio 10.50/5.00 does not
contain sigma, so shrinking sigma shrinks BOTH predictors' noise by the same
factor and cannot change their ORDER. `md_acc` should stay worse than `md`.
A reversal would mean the ratio argument is wrong somewhere.

WHAT IS ACTUALLY WORTH THE SLOT, and it is not md_acc. This is a
SINGLE-COLUMN AUC on raw features -- no model, no ILP, no thresholds. So it
answers the question hparabench cannot: does the better placement improve
the FEATURES AT ALL, before anything fitted on refine coordinates sees them?
`d` (raw distance to candidate) and `md` (distance from the constant-velocity
prediction) read 0.9681 / 0.9636 on BASE coordinates. If hpara leaves them
flat, the placement gain never reaches the pipeline and hparabench's null is
explained without appealing to the ranker at all.

THE CUT

  * detections come from `fdet_F4_` / `fdet_F4hp_` instead of `detcache_`,
    and `load_dets` takes an arm;
  * the D187 ranker, the 44b6 sample and STAGE A/B are cut -- they need
    caches this run does not mount, and none of them feeds the diagnostic;
  * the diagnostic runs TWICE, once per coordinate set, so the control is
    in the same process on the same movies.

ANCHOR. None against accdiag's own numbers -- those were measured on BASE
detections and this runs on F4. The check that IS available: the two arms
must produce the SAME NUMBER of labelled rows to within a few tenths of a
percent. hpara neither adds nor removes a detection, so a large divergence
means the label matcher moved, not the placement.

Attach: biohub-edge-ranker (support pack), biohub-fuse6bba + biohub-fusecache
(the fdet_F4_ primaries), biohub-hparacache (the fdet_F4hp_ ones).
"""
import os
import re

src = open("accdiag.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-08-28j-accdiag"',
    'SCRIPT_VERSION = "2026-09-08a-acchp"')

# ---- detections: two arms of fdet_*, not one of detcache_ ---
rep("""CACHES = {}
stack = [BASE]""",
    '''ARM_CACHES = {"F4": {}, "F4HP": {}}
stack = [BASE]''')

rep('''    for f in d.glob("detcache_*.npz"):
        CACHES[f.stem[len("detcache_"):]] = f''',
    '''    for _arm, _pfx in (("F4HP", "fdet_F4hp_"), ("F4", "fdet_F4_")):
        for f in d.glob(_pfx + "*.npz"):
            ARM_CACHES[_arm][f.stem[len(_pfx):]] = f''')

rep('''print(f"detection caches found: {len(CACHES)}", flush=True)
assert len(CACHES) >= 40, "attach biohub-edge-ranker as kernel source"


def load_dets(name):
    z = np.load(CACHES[name])''',
    '''# "fdet_F4_" needs an underscore where "fdet_F4hp_" has an 'h', so the two
# globs cannot overlap -- asserted rather than trusted, because a leak would
# put hpara rows silently into the control arm.
assert not set(ARM_CACHES["F4"]) & set("hp_" + k for k in ARM_CACHES["F4HP"])
for _arm in ("F4", "F4HP"):
    print(f"{_arm} detection caches: {len(ARM_CACHES[_arm])}", flush=True)
CACHES = ARM_CACHES["F4"]
assert len(ARM_CACHES["F4"]) >= 12 and len(ARM_CACHES["F4HP"]) >= 12, (
    "attach biohub-hparacache for fdet_F4hp_ and biohub-fuse6bba / "
    "biohub-fusecache for fdet_F4_")


def load_dets(name, arm="F4"):
    z = np.load(ARM_CACHES[arm][name])''')

# ---- cut the ranker, the 44b6 sample and STAGE A/B ----------------------
# None of it feeds the diagnostic, and all of it needs caches this run does
# not mount (the D187 pkl, 60+ 44b6 detcaches). Verified after the cut by
# comparing the names the tail READS against the names bound before it.
_i = src.index("# ---- the D187 ranker, loaded not retrained ---")
_j = src.index("# ======================= the diagnostic ==============")
src = src[:_i] + ("# [acchp] STAGE A/B, the D187 ranker and the 44b6 sample "
                  "are cut here:\n# none of them feeds the diagnostic and "
                  "all need caches this run\n# does not mount. See the "
                  "maker's docstring.\n\n") + src[_j:]

# ---- run the diagnostic once per arm ------------------------------------
rep('''_names = [n for n in HOLDOUT if n in CACHES]
print(f"  datasets: {len(_names)}", flush=True)''',
    '''_names = [n for n in HOLDOUT
          if n in ARM_CACHES["F4"] and n in ARM_CACHES["F4HP"]]
print(f"  datasets in BOTH arms: {len(_names)}", flush=True)
assert len(_names) == len(HOLDOUT), (
    f"only {len(_names)} of {len(HOLDOUT)} holdout movies have both a "
    f"fdet_F4_ and a fdet_F4hp_ cache -- the arms would not be paired")
RESULTS = {}
for ARM in ("F4", "F4HP"):
  print(f"{chr(10)}########## ARM {ARM} ##########", flush=True)''')

rep('''    _dets = load_dets(_name)
    _T = len(_dets)
    # ---- 1 & 2: how often is it computable, and how big is the stage? --''',
    '''    _dets = load_dets(_name, ARM)
    _T = len(_dets)
    # ---- 1 & 2: how often is it computable, and how big is the stage? --''')

# everything from the per-movie loop to the end of the AUC table is indented
# one level so it runs under `for ARM`, and the accumulators reset per arm.
rep('''_hist = [0, 0, 0]
_glob, _cell = [], []
X_raw, X_dem, Y = [], [], []
_t0 = time.time()
''',
    '''  _hist = [0, 0, 0]
  _glob, _cell = [], []
  X_raw, X_dem, Y = [], [], []
  _t0 = time.time()
''')


# ---- stage 2: put the whole diagnostic inside the per-arm loop ----------
# The rep above only indented the accumulators; everything from the
# per-movie loop through the AUC table must run per arm too, or the tables
# would be printed once, from whichever arm happened to run last.
_lines = src.split("\n")
_a = next(i for i, l in enumerate(_lines) if l.startswith("for _k, _name in enumerate(_names):"))
_b = next(i for i, l in enumerate(_lines) if l.startswith('print(f"  for scale, the strongest BASE columns: '))
_end = _b
while not _lines[_end].rstrip().endswith("flush=True)"):
    _end += 1
for _i in range(_a, _end + 1):
    if _lines[_i].strip():
        _lines[_i] = "  " + _lines[_i]
_lines.insert(_end + 1, '''  RESULTS[ARM] = dict(
      n=len(Y), pos=float(Y.mean()),
      d=roc_auc_score(Y, X_raw[:, 0]), md=roc_auc_score(Y, X_raw[:, 4]),
      md_acc=roc_auc_score(Y, X_raw[:, NFEAT_BASE + 2]),
      score_c=roc_auc_score(Y, X_raw[:, 7]),
      stage=float(np.median(_glob)) if _glob else float("nan"),
      cell=float(np.median(_cell)) if _cell else float("nan"))
  del X_raw, X_dem, Y
  gc.collect()''')
src = "\n".join(_lines)

# ---- the side-by-side table, which is the whole point -------------------
rep('''print(f"{chr(10)}=== reading it ===", flush=True)''',
    '''print(f"{chr(10)}=== F4 (refine) vs F4HP (hpara), SAME movies, SAME "
      f"labels, ONLY the placement differs ===", flush=True)
_r, _h = RESULTS.get("F4"), RESULTS.get("F4HP")
if _r and _h:
    print(f"  labelled rows: F4 {_r['n']:,} ({_r['pos']:.4f} pos), "
          f"F4HP {_h['n']:,} ({_h['pos']:.4f} pos)", flush=True)
    _drift = abs(_h["n"] - _r["n"]) / max(1, _r["n"])
    print(f"  row-count drift {100 * _drift:.3f}%  "
          + ("-- hpara adds no detections, so this should be a few tenths "
             "of a percent at most; more means the LABEL MATCHER moved and "
             "the columns below are not comparable"
             if _drift > 0.01 else "-- as expected"), flush=True)
    print(f"{chr(10)}  {'column':<10} {'F4 refine':>10} {'F4HP hpara':>11} "
          f"{'delta':>9}   note", flush=True)
    for _c, _note in (("d", "raw distance to the candidate"),
                      ("md", "distance from the CONSTANT-VELOCITY prediction"),
                      ("md_acc", "distance from the ACCELERATION-corrected one"),
                      ("score_c", "detection score of the candidate")):
        print(f"  {_c:<10} {_r[_c]:>10.4f} {_h[_c]:>11.4f} "
              f"{_h[_c] - _r[_c]:>+9.4f}   {_note}", flush=True)
    print(f"{chr(10)}  PREDICTED BEFORE THE RUN: md_acc stays below md in "
          f"BOTH arms. The variance ratio 10.50/5.00 does not contain "
          f"sigma, so a smoother position shrinks both predictors equally "
          f"and cannot reorder them. A reversal falsifies that argument.",
          flush=True)
    print(f"  THE COLUMN THAT MATTERS MORE: `d` and `md`. Flat there means "
          f"the placement gain never reaches the features at all, which "
          f"would explain hparabench's null without involving the ranker.",
          flush=True)
    print(f"  stage/cell acceleration ratio: F4 "
          f"{_r['stage'] / max(1e-9, _r['cell']):.3f}  F4HP "
          f"{_h['stage'] / max(1e-9, _h['cell']):.3f}", flush=True)
else:
    print(f"  [degraded] only {sorted(RESULTS)} completed -- no comparison",
          flush=True)

print(f"{chr(10)}=== reading it ===", flush=True)''')

rep("print(\"ACCDIAG DONE\", flush=True)",
    "print(\"ACCHP OK\" if len(RESULTS) == 2 else "
    "f\"[degraded] ACCHP INCOMPLETE: {sorted(RESULTS)}\", flush=True)")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "acchp.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written stage 1", len(src))
