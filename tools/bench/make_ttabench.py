"""hparabench_ref.py -> ttabench.py: what does edge-feature TTA do to the SCORE.

The user, on being told the head's probabilities got worse and the 20 GPU
hours were therefore refused: *"边概率头变了和它不值之间的关联是？"* -- there
was none. `ttascore` measured the head's own AUC. The ranker never ran, the
ILP never ran, nothing was scored. And working-agreement.md records that
offline edge AUC anti-predicted this pipeline three times running.

Then, on being told a 12-movie score test would probably come back "cannot
tell": *"只看12部的分数会死？"* and *"你都知道测不出来了，还在这里诚实的测不
出来"*. Both fair. Predicting the width of an interval is not measuring it,
and this comparison is far more tightly paired than the one that prediction
was borrowed from:

  hparabench round 1 changed the COORDINATES, so detections moved, the gate
  re-formed, the candidate pairs changed, and the arms shared almost nothing.
  Here the detections are byte-identical, the (t, i, j) rows are identical --
  `ttaprobe` held `det_logits` at view 0 exactly so they would be -- and the
  ONLY difference is the value in `p_fwd` / `p_rev`. Nothing else in the
  pipeline can move except through those two columns.

WHAT IT RUNS. hparabench_ref unchanged, except that the 12 holdout movies'
`edgeprob_F4_` caches are replaced by `ttaprobe`'s `edgeprob_tta4_` ones.
128 6bba movies are scored; 12 of them differ; the paired per-movie
comparison is on those 12, and the CSV carries all 128 so the dilution can
be read as well as the CHANGED subset.

THE FREE ANCHOR, and it decides whether one run is enough. `ttaprobe`'s
`noTTA` arm is the identity path of the same wrapper over the same primaries
with the same model, so `edgeprob_noTTA_<m>` must reproduce
`edgeprob_F4_<m>` EXACTLY. If it does, `hparabench_ref`'s own CSV is a valid
control and this one run answers the question. If it does not, the wrapper
is not transparent at view 1, the tta4 arm is not comparable to the F4
baseline, and a separate noTTA arm has to be run -- the script says which,
loudly, before spending three hours.

⚠ The SECOND head is not TTA'd. `ttaprobe` cached `edgeprob_` only, so
`edgeprob2_F4_` stays at its single-view value on the 12 movies that have
one. That is stated rather than hidden: this measures TTA on the first edge
head, which is the head the public notebook's one-character diff touches.

Attach: hparabench_ref's sources PLUS faustine649/biohub-ttaprobe. CPU only.
"""
import os

src = open("hparabench_ref.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-09a-hparabench"',
    'SCRIPT_VERSION = "2026-09-09a-ttabench"')
rep('DET_SRC = "F4"', 'DET_SRC = "F4"      # ttabench is a REFINE-coordinate run')

# ---- split the TTA caches out, and override -----------------------------
rep('''EDGEPROB_F4 = {k[len("F4_"):]: v for k, v in _EP_ALL.items()
               if k.startswith("F4_")}
del _EP_ALL''',
    '''EDGEPROB_F4 = {k[len("F4_"):]: v for k, v in _EP_ALL.items()
               if k.startswith("F4_")}

# ttaprobe writes edgeprob_<arm>_<movie>.npz for arm in noTTA / tta4 / tta8.
# Those match the "edgeprob_" prefix too and would otherwise land in the BASE
# dict under a key nothing looks up.
TTA_ARM = "tta4"
EP_TTA = {a: {k[len(a) + 1:]: v for k, v in _EP_ALL.items()
              if k.startswith(a + "_")}
          for a in ("noTTA", "tta4", "tta8")}
EDGEPROB = {k: v for k, v in EDGEPROB.items()
            if not any(k.startswith(a + "_") for a in EP_TTA)}
del _EP_ALL
print("  ttaprobe caches: " + ", ".join(f"{a} {len(d)}"
                                        for a, d in EP_TTA.items()), flush=True)
assert len(EP_TTA[TTA_ARM]) == 12, (
    f"{len(EP_TTA[TTA_ARM])} {TTA_ARM} caches, want 12 -- attach "
    f"faustine649/biohub-ttaprobe")''')

# ---- the anchor: view 1 must be the identity ----------------------------
rep('''print(f"edgeprob caches mounted: {len(EDGEPROB)} BASE + {len(EDGEPROB_F4)} "''',
    '''# THE ANCHOR. ttaprobe's noTTA arm is the same model on the same primaries
# with the D4 average switched off, so it must reproduce edgeprob_F4_ bit for
# bit. If it does, hparabench_ref's CSV is a valid control for this run and
# one pass answers the question; if it does not, the wrapper is not
# transparent at view 1 and a separate noTTA arm is required. Checked BEFORE
# three hours of CPU, and reported either way.
_ident, _diff, _worst = [], [], 0.0
for _m in sorted(EP_TTA["noTTA"]):
    if _m not in EDGEPROB_F4:
        continue
    _a, _b = np.load(EP_TTA["noTTA"][_m]), np.load(EDGEPROB_F4[_m])
    _same = (np.array_equal(_a["t"], _b["t"]) and np.array_equal(_a["i"], _b["i"])
             and np.array_equal(_a["j"], _b["j"]))
    if not _same:
        _diff.append((_m, "rows differ"))
        continue
    _d = max(float(np.abs(np.asarray(_a[c], float)
                          - np.asarray(_b[c], float)).max())
             for c in ("p_fwd", "p_rev"))
    _worst = max(_worst, _d)
    (_ident if _d == 0.0 else _diff).append((_m, _d))
    _a.close(); _b.close()
print(f"  ANCHOR view-1 identity: {len(_ident)} of {len(_ident) + len(_diff)} "
      f"movies reproduce edgeprob_F4_ EXACTLY, worst |dp| {_worst:.3e}",
      flush=True)
if _diff:
    print(f"  \\u26a0 {len(_diff)} movies DIFFER: {_diff[:3]}. hparabench_ref is "
          f"then NOT a valid control for this run -- the tta4 delta below "
          f"would carry the wrapper's own offset as well as the TTA. Run the "
          f"noTTA arm (TTA_ARM = 'noTTA') and pair against that instead.",
          flush=True)
else:
    print("  -> hparabench_ref's hparabench_F4.csv IS the control; pair its "
          "12 holdout rows against this run's.", flush=True)

_ov = sorted(set(EDGEPROB_F4) & set(EP_TTA[TTA_ARM]))
assert len(_ov) == 12, (f"{len(_ov)} of the {TTA_ARM} caches match an "
                        f"edgeprob_F4_ entry, want 12: {_ov}")
EDGEPROB_F4.update(EP_TTA[TTA_ARM])
print(f"  edgeprob: {len(_ov)} movies switched to the {TTA_ARM} head "
      f"({' '.join(n.split('_')[-1] for n in _ov)})", flush=True)
print("  \\u26a0 the SECOND head is NOT TTA'd -- ttaprobe cached edgeprob_ "
      "only -- so edgeprob2_F4_ stays single-view wherever it exists. This "
      "measures TTA on the FIRST edge head, which is the head the public "
      "notebook's one-character diff touches.", flush=True)

print(f"edgeprob caches mounted: {len(EDGEPROB)} BASE + {len(EDGEPROB_F4)} "''')

# ---- the two anchors that CANNOT hold in this arm -----------------------
# Same situation as hparabench's F4HP run: both are built on HOLDOUT_6BBA,
# which is exactly the 12 movies whose probabilities this arm changes.
rep('''assert not ANCHOR_FAIL, f"hard anchor miss: {ANCHOR_FAIL}"''',
    '''_moved = [f for f in ANCHOR_FAIL if f[0] in ("ANCHOR_D187", "REF_FORKS")]
ANCHOR_FAIL = [f for f in ANCHOR_FAIL if f not in _moved]
for _f in _moved:
    print(f"  [expected] {_f[0]} {_f[1]} moved to {_f[2]} from {_f[3]} -- both "
          f"anchors are built on HOLDOUT_6BBA, the 12 movies whose edge "
          f"probabilities this arm changes, so neither CAN hold; the move is "
          f"the measurement", flush=True)
assert not ANCHOR_FAIL, f"hard anchor miss: {ANCHOR_FAIL}"''')

# ---- per-movie CSV, so the 12 can be paired offline ---------------------
rep('''_csv = pathlib.Path("/kaggle/working") / f"hparabench_{DET_SRC}.csv"''',
    '''_csv = pathlib.Path("/kaggle/working") / f"ttabench_{TTA_ARM}.csv"''')

rep('HPARABENCH OK', 'TTABENCH OK')
rep('HPARABENCH INCOMPLETE', 'TTABENCH INCOMPLETE')

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
out = os.path.abspath(os.path.join(base, "ttabench.py"))
open(out, "w", encoding="utf-8", newline="\n").write(src)
print("written ttabench.py", len(src))
