"""zhpara.py -> zhpara2.py: the 3 crops v3 did not finish, plus the 8 anchors.

WHAT v3 DID AND WHERE IT STOPPED, diagnosed from artefacts because there was
no traceback to read.

    the Kaggle log is 0 BYTES              -> the process was SIGKILLed, not
                                              raised (eprank3 and rkpool both
                                              produced full tracebacks the
                                              same night)
    819 files, 7.6 MB under ps_zh_gt       -> disk exhaustion is DEAD as an
                                              explanation; it was measured,
                                              not assumed
    39 GT geffs written, 38 crops in the   -> it died INSIDE crop 39
      checkpoint CSV
    exactly 1 leftover prediction geff,    -> and after that crop's REF arm had
      ps_zh_REF/ZSNS001_z2y2x7.geff           saved. The rmtree runs at the END
                                              of a crop, so a single leftover
                                              names the crop that never
                                              finished.

So: **killed inside `ZSNS001_z2y2x7`, after the REF arm's geff was written.**
What is NOT established is why. RAM is the leading candidate (SIGKILL with no
Python-level error is what the Linux OOM killer produces) but it has not been
measured, and one fact argues against a simple size story: crops LARGER in GT
finished, `ZSNS001_tail_z2y3x2` at 56,167 GT nodes among them. v3 ordered by
CACHE FILE SIZE, not by GT count.

WHAT THIS KERNEL IS FOR, in that order of priority:

  1. **Measure.** Print RSS and peak RSS (/proc/self/status VmRSS / VmHWM) at
     every stage of every crop. If it is killed again, the last line before
     the kill names the stage, which is the one thing v3 could not say.
  2. **Finish the domain.** 38 of 41 crops are already paired and the missing
     3 are unknown in size but were ordered late. adj_J is weight-averaged, so
     they can move the number and the direction is unknown.

WHAT IT RUNS: the 8 anchor crops (so the run is readable -- an unanchored
partial is not) plus the crops that carry both caches and are NOT among v3's
38. The 38 are hardcoded from v3's own checkpoint CSV and the remainder is
asserted to be exactly 3, so a silent change in the crop set fails here
instead of quietly measuring something else.

⚠ The three anchors are inherited unchanged and must all hold again: REF over
the 8 must read edge_J 0.8036, adj_J 0.8093, div_J 0.0230; the hpara cache
must carry byte-identical GT; the per-frame detection counts must match.

Attach: zhpara's sources. CPU only.
"""
import os

src = open("zhpara.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-09a-zhpara"',
    'SCRIPT_VERSION = "2026-09-09a-zhpara2"')

# ---- the crop list: the 8 anchors + whatever v3 did not finish ---------
rep('''_size = {k: v.stat().st_size for k, v in ZH_REF.items()}
BOTH = ANCHOR8 + sorted((n for n in BOTH if n not in set(ANCHOR8)),
                        key=lambda n: _size.get(n, 0))''',
    '''# v3's checkpoint CSV, verbatim: the 38 crops it scored in BOTH arms before
# it was killed inside crop 39. Hardcoded rather than read from an attached
# output, because zhpara ERRORED and a failed kernel cannot be a kernel_source.
DONE_V3 = [
    'ZSNS001_tail_z2y0x3', 'ZSNS001_tail_z2y1x2', 'ZSNS001_tail_z2y2x3',
    'ZSNS001_tail_z2y3x0', 'ZSNS001_tail_z2y3x2', 'ZSNS001_z2y1x4',
    'ZSNS001_z2y1x5', 'ZSNS001_z2y1x6', 'ZSNS001_z2y2x6', 'ZSNS001_z2y3x7',
    'ZSNS001_z2y4x7', 'ZSNS003_z1y1x4', 'ZSNS003_z1y2x1', 'ZSNS003_z1y2x2',
    'ZSNS003_z1y2x6', 'ZSNS003_z1y3x1', 'ZSNS003_z1y5x0', 'ZSNS003_z1y5x6',
    'ZSNS003_z1y6x1', 'ZSNS003_z1y7x2', 'ZSNS004_z2y0x3', 'ZSNS004_z2y0x4',
    'ZSNS004_z2y0x5', 'ZSNS004_z2y1x5', 'ZSNS004_z2y1x7', 'ZSNS004_z2y3x2',
    'ZSNS004_z2y4x3', 'ZSNS004_z2y4x7', 'ZSNS004_z2y5x5', 'ZSNS005_z1y1x5',
    'ZSNS005_z1y3x2', 'ZSNS005_z1y3x9', 'ZSNS005_z1y4x2', 'ZSNS005_z1y4x8',
    'ZSNS005_z1y5x2', 'ZSNS005_z1y5x8', 'ZSNS005_z1y6x6', 'ZSNS005_z1y6x8',
]
assert len(DONE_V3) == 38 and len(set(DONE_V3)) == 38, len(DONE_V3)
assert set(ANCHOR8) <= set(DONE_V3), (
    "an anchor crop is not among v3's 38 -- the anchor set changed and v3's "
    "CSV cannot be joined to this run")
_size = {k: v.stat().st_size for k, v in ZH_REF.items()}
REMAIN = sorted((n for n in BOTH if n not in set(DONE_V3)),
                key=lambda n: _size.get(n, 0))
assert len(REMAIN) == 3, (
    f"expected 3 crops left after v3's 38, got {len(REMAIN)}: {REMAIN} -- "
    f"the crop set changed since v3 ran and pairing its CSV to this run "
    f"would join two different samples")
assert "ZSNS001_z2y2x7" in REMAIN, (
    f"ZSNS001_z2y2x7 is the crop v3 was killed inside (its GT geff was "
    f"written, its row never was, and its REF prediction geff was the only "
    f"one left behind) -- it must be re-run, not skipped. REMAIN={REMAIN}")
# The 8 anchors are re-solved so the run carries its own anchor; they are all
# small and already known to fit.
BOTH = ANCHOR8 + REMAIN
print(f"  v3 scored {len(DONE_V3)} of 41 crops before it was SIGKILLed "
      f"inside crop 39. This run does the {len(REMAIN)} it did not finish "
      f"({' '.join(REMAIN)}) plus the {len(ANCHOR8)} anchor crops, "
      f"{len(BOTH)} in all.", flush=True)
print(f"  cache sizes MB: " + " ".join(
    f"{n.split('_')[-1]}={_size.get(n, 0) / 1e6:.0f}" for n in REMAIN),
      flush=True)''')

# ---- the measurement v3 could not make ---------------------------------
rep('''def _checkpoint():''',
    '''def _rss(where):
    """Current and peak RSS, printed at every stage of every crop.

    v3 was killed with a ZERO-BYTE log: no traceback, no last line, nothing
    to say which stage was holding the memory. This is the cheapest way to
    make the next kill informative -- the last line printed names the stage.
    /proc/self/status is Linux-only, which Kaggle is; anywhere else this is
    a no-op rather than a crash.
    """
    try:
        with open("/proc/self/status") as _f:
            _d = dict(l.split(":", 1) for l in _f if ":" in l)
        cur = _d.get("VmRSS", "?").strip()
        peak = _d.get("VmHWM", "?").strip()
    except OSError:
        cur = peak = "n/a"
    print(f"      [rss] {where:<22} now {cur:>12}  peak {peak:>12}",
          flush=True)


def _checkpoint():''')

rep('''    _d0, T, gt_arr, gt_e, n_cross = zh_load(name)''',
    '''    _rss(f"crop start")
    _d0, T, gt_arr, gt_e, n_cross = zh_load(name)
    _rss("after REF zh_load")''')

rep('''    _d1, T1, gt_arr1, gt_e1, _nc1 = zh_load(name)''',
    '''    _d1, T1, gt_arr1, gt_e1, _nc1 = zh_load(name)
    _rss("after HP zh_load")''')

rep('''        dets, T2, _ga, _ge, _nc = zh_load(name)
        nodes0, edges0, pool = solve_one(dets, T2)''',
    '''        dets, T2, _ga, _ge, _nc = zh_load(name)
        _rss(f"{tag} after zh_load")
        nodes0, edges0, pool = solve_one(dets, T2)
        _rss(f"{tag} after solve_one")''')

rep('''        er = compute_metric(g, gtg, scale=tuple(SCALE))''',
    '''        _rss(f"{tag} after from_geff")
        er = compute_metric(g, gtg, scale=tuple(SCALE))
        _rss(f"{tag} after compute_metric")''')

rep('print("ZHPARA OK", flush=True)', 'print("ZHPARA2 OK", flush=True)')
rep('print(f"ZHPARA INCOMPLETE ({len(done)} of {len(BOTH)} crops)", flush=True)',
    'print(f"ZHPARA2 INCOMPLETE ({len(done)} of {len(BOTH)} crops)", flush=True)')

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "zhpara2.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
