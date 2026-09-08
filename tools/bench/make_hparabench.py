"""divbench9.py -> hparabench.py: the same bench, run on hpara coordinates.

THE QUESTION. refinecmp measured that `hpara` -- the parabolic vertex of the
heatmap, applied on the pooled grid before the shipped `_refine` -- places a
detection better than the shipped rule on both domains, cutting frame-to-frame
jitter 1.29 -> 0.89 um on 6bba and 1.51 -> 1.40 on 44b6 with recall flat or
up. Jitter is motion the pipeline INVENTS, and the ranker's distance and
velocity features are computed from exactly those coordinates. Does the better
placement convert into SCORE? The user, 2026-09-08: "hpara换成流水线臂可以先做".

THE DESIGN, and why it is a one-line diff rather than a new arm. Adding a
third det arm inside divbench9 would need stage A's feature pass, stage B's
solve, the fork tables and every anchor to grow a dimension. Instead the
DETECTION SOURCE is switched at the top: `FDET_F4` is repointed at
`fdet_F4hp_<name>.npz` and everything downstream -- features, ranker, ILP,
patches, scorer -- runs unchanged on the new coordinates. The run is then done
TWICE, once per source, and the per-movie rows of both are dumped so the
paired bootstrap can be done on the pair.

  DET_SRC = "F4"     reads fdet_F4_,   must reproduce divbench9's anchors
                     EXACTLY (0.9216 hard). This is the control, and it is
                     the same code path, so a difference cannot come from
                     the harness.
  DET_SRC = "F4HP"   reads fdet_F4hp_ from biohub-hparacache. Its anchors
                     are EXPECTED to move -- that is the measurement -- so
                     they are printed and warned, never asserted.

WHAT IS HELD FIXED, stated because it is an approximation. hparacache runs
NMS once, on the refine coordinates, and places the survivors both ways, so
the two runs have identical detection COUNT, ORDER and IDENTITY. That keeps
the cached edge-head probabilities (`edgeprob_F4_<name>.npz`, joined by
(t, i, j) and checked against `n_per_frame`) exactly valid. The head's own
crops would shift by up to 0.8 um if it were re-run; it is NOT re-run here.
So this arm measures what better COORDINATES do to the ranker, the ILP and
the patches, with the head held at its refine-based value. It also forgoes
hpara's +0.0008 recall, which came from NMS acting on better coordinates.

AND A THIRD, found while fixing the control run's sources: the bench LOADS
`edge_ranker_ep175v3_bidir.pkl` rather than fitting it, and that pkl was
fitted by `edgeprob-rank2` on the 175-movie pool's REFINE-based features.
So the ranker is fixed, was fitted on refine coordinates, and is applied
here to hpara ones. Re-fitting it means re-caching all 187 movies (about
four GPU hours against 35 minutes for the 24) and re-running
`edgeprob-rank2`. That is exactly the practical question -- can the
placement be swapped in without retraining anything -- so a POSITIVE result
is trustworthy and a NEGATIVE one is not decisive.

All three approximations push the same way: they make the arm conservative.

OUTPUT: hparabench_<DET_SRC>.csv, one row per movie with the per-sample
metrics, so the two runs can be paired offline (24 movies, same order, same
folds) and bootstrapped.

Attach: divbench9's sources, PLUS faustine649/biohub-hparacache as a
kernel_source when DET_SRC is F4HP.
"""
import os

src = open("divbench9.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:80])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-08b-divbench9"',
    'SCRIPT_VERSION = "2026-09-08a-hparabench"')

# ---- the switch -----------------------------------------------------------
rep('''# Score the BASE primaries too (v1's arm)? Doubles stage B (~+66 min).
SCORE_BASE_ARM = False''',
    '''# Score the BASE primaries too (v1's arm)? Doubles stage B (~+66 min).
SCORE_BASE_ARM = False

# WHICH detection cache the "F4" path reads. "F4" is divbench9 verbatim and
# carries the hard anchor; "F4HP" swaps in hparacache's hpara placement of the
# SAME detections and is expected to move the anchors -- that movement is the
# measurement. See the module docstring of make_hparabench.py.
DET_SRC = "F4HP"
assert DET_SRC in ("F4", "F4HP"), DET_SRC
print(f"DET_SRC = {DET_SRC}", flush=True)''')

# ---- repoint the cache ----------------------------------------------------
rep('''FDET_F4 = _walk_files(_npz_prefix("fdet_F4_"))
print(f"fdet_F4 caches: {len(FDET_F4)}", flush=True)
assert len(FDET_F4) >= 24, "attach biohub-fusecache for the F4 detections"''',
    '''FDET_F4 = _walk_files(_npz_prefix("fdet_F4_"))
print(f"fdet_F4 caches: {len(FDET_F4)}", flush=True)
assert len(FDET_F4) >= 24, "attach biohub-fusecache for the F4 detections"
FDET_F4HP = _walk_files(_npz_prefix("fdet_F4hp_"))
print(f"fdet_F4hp caches: {len(FDET_F4HP)}", flush=True)
if DET_SRC == "F4HP":
    assert len(FDET_F4HP) >= 24, (
        "DET_SRC is F4HP but no fdet_F4hp_*.npz is attached -- add "
        "faustine649/biohub-hparacache as a kernel_source")
    # OVERRIDE, not replace. Only the 24 EVAL movies are re-cached with
    # hpara; the 175-movie pool stays on refine (re-caching all 187 is four
    # GPU hours against 35 minutes), and the bench asserts an fdet_F4_ cache
    # for all 128 6bba movies further down. v1 assigned FDET_F4 = FDET_F4HP
    # and died on 116 missing 6bba pool caches.
    _both = sorted(set(FDET_F4) & set(FDET_F4HP))
    FDET_F4 = {**FDET_F4, **FDET_F4HP}
    print(f"  hpara caches override {len(_both)} of {len(FDET_F4)} F4 "
          f"entries; the rest of the pool stays on refine", flush=True)
    assert len(_both) == 24, (
        f"expected the 24 eval movies to be overridden, got {len(_both)} -- "
        f"a mismatch here would silently score some eval movies on refine "
        f"coordinates and call it the hpara arm")''')

# ---- and the two heads, overridden the same way ---------------------------
rep('''EP_OF = {"BASE": EDGEPROB, "F4": EDGEPROB_F4}
EP2_OF = {"BASE": EDGEPROB2, "F4": EDGEPROB2_F4}''',
    '''if DET_SRC == "F4HP":
    for _nm, _base, _hp in (("edgeprob", EDGEPROB_F4, EDGEPROB_F4HP),
                            ("edgeprob2", EDGEPROB2_F4, EDGEPROB2_F4HP)):
        _ov = sorted(set(_base) & set(_hp))
        assert len(_ov) == 24, (
            f"{_nm}: {len(_ov)} of 24 eval movies have an F4hp cache -- "
            f"attach biohub-ephpcache and biohub-ep2hpcache. Running "
            f"without them scores hpara COORDINATES against a head that "
            f"read the refine ones, and the (t,i,j) join fails outright.")
        _base.update(_hp)
        print(f"  {_nm}: {len(_ov)} eval entries switched to the hpara head",
              flush=True)

EP_OF = {"BASE": EDGEPROB, "F4": EDGEPROB_F4}
EP2_OF = {"BASE": EDGEPROB2, "F4": EDGEPROB2_F4}''')

# ---- both heads must follow the coordinates -------------------------------
# The harness joins each head's cached rows to its candidate pairs by
# (t, i, j), and the pairs are GATED ON DISTANCE (GATE_UM = 10.0). hpara moves
# every point by a median 0.43 um, so pairs cross that line in both directions
# and the cached pair set is not the harness's. v2 died on
# "44b6_0b24845f t=0: 40 of 990 harness pairs have no edgeprob row".
# ephpcache / ep2hpcache re-run BOTH heads on the hpara primaries; doing only
# the first would move the failure one line down, since both go through
# `_load_head`. This also REMOVES the "head frozen at its refine value"
# approximation the original design carried.
rep('''EDGEPROB = {k: v for k, v in _EP_ALL.items() if not k.startswith("F4_")}''',
    '''EDGEPROB = {k: v for k, v in _EP_ALL.items()
            if not k.startswith("F4_") and not k.startswith("F4hp_")}
EDGEPROB_F4HP = {k[len("F4hp_"):]: v for k, v in _EP_ALL.items()
                 if k.startswith("F4hp_")}''')

rep('''EDGEPROB2 = {k: v for k, v in _EP2_ALL.items() if not k.startswith("F4_")}''',
    '''EDGEPROB2 = {k: v for k, v in _EP2_ALL.items()
             if not k.startswith("F4_") and not k.startswith("F4hp_")}
EDGEPROB2_F4HP = {k[len("F4hp_"):]: v for k, v in _EP2_ALL.items()
                  if k.startswith("F4hp_")}''')

# ---- the anchors move on purpose in the HP run ----------------------------
rep('''ANCHOR_TOL = 0.0005''',
    '''if DET_SRC == "F4HP":
    # The whole point of the run is that these numbers move. Asserting them
    # would abort the measurement; dropping them silently would hide a
    # harness change, so they are demoted to SOFT and still printed.
    ANCHOR_SOFT.update(ANCHOR_HARD)
    ANCHOR_HARD = {}
    print("  DET_SRC=F4HP: the F4 anchors are demoted to soft. A large move "
          "in ANCHOR_D187-6bba (0.9216) is the measurement; a move of more "
          "than a couple of points is more likely a broken join -- check "
          "the JOIN counters before believing it.", flush=True)
ANCHOR_TOL = 0.0005''')

# ---- dump the per-movie rows so the two runs can be paired ---------------
rep('''assert not ANCHOR_FAIL, f"hard anchor miss: {ANCHOR_FAIL}"''',
    '''_csv = pathlib.Path("/kaggle/working") / f"hparabench_{DET_SRC}.csv"
_keys = sorted({k for r in PER.values() for k in r})
with open(_csv, "w", encoding="utf-8", newline="\\n") as _fh:
    _fh.write("dataset,fold," + ",".join(_keys) + "\\n")
    for _n in sorted(PER):
        _r = PER[_n]
        _fh.write(f"{_n},{FOLD_OF.get(_n, -1)},"
                  + ",".join(repr(_r.get(_k)) for _k in _keys) + "\\n")
print(f"\\n  {_csv.name} written ({len(PER)} movies, {len(_keys)} columns). "
      f"Pair it with the other DET_SRC run -- same movies, same folds, same "
      f"order -- and bootstrap the per-movie score difference; a single "
      f"aggregate cannot show whether one movie carries the whole delta, "
      f"which is exactly the failure this comparison is exposed to.",
      flush=True)

assert not ANCHOR_FAIL, f"hard anchor miss: {ANCHOR_FAIL}"''')

rep('DIVBENCH9 OK', 'HPARABENCH OK')
rep('DIVBENCH9 INCOMPLETE', 'HPARABENCH INCOMPLETE')

# Two files: the CONTROL (DET_SRC=F4, divbench9's hard anchor still asserted,
# so it validates the harness and the new CSV dump on its own) and the ARM.
# They are pushed as two kernel ids and run concurrently -- both are CPU, so
# neither touches the GPU quota, and pairing two runs of ONE script is safer
# than growing divbench9 a third det-arm dimension.
base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
for fname, det in (("hparabench.py", "F4HP"), ("hparabench_ref.py", "F4")):
    body = src.replace('DET_SRC = "F4HP"', f'DET_SRC = "{det}"')
    assert body.count(f'DET_SRC = "{det}"') == 1, (det, body.count(det))
    open(os.path.abspath(os.path.join(base, fname)), "w",
         encoding="utf-8", newline="\n").write(body)
    print("written", fname, len(body), "DET_SRC", det)
