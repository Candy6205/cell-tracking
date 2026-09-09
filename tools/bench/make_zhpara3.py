"""zhpara2.py -> zhpara3.py: measure the kill instead of printing about it.

WHAT zhpara2 ESTABLISHED, and what it failed to.

  established: **it is not accumulation.** zhpara2 held only 8 small crops of
  state and still died on crop 9 -- the first of the 3 that v3 never
  finished. A per-crop peak, not a leak.
  established: **the harness is deterministic across runs.** zhpara2's 8
  anchor crops reproduce v3's rows BIT FOR BIT -- 8 crops x 2 arms x 12
  columns, 0 values differ, worst |d| = 0.000e+00 -- and its own three
  anchors read 0.8036 / 0.8093 / 0.0230 exactly. So rows from separate runs
  may be joined, which is what makes this kernel legitimate at 5 crops.
  FAILED: **the RSS trace.** zhpara2 printed VmRSS/VmHWM at every stage and
  the Kaggle log came back **0 bytes**, exactly as v3's had. A SIGKILLed
  session on Kaggle loses stdout ENTIRELY -- not truncated, gone.

That is the same lesson `_checkpoint()` already encodes, applied to results
and not to diagnostics: **on a session that can be killed, only a FILE in
/kaggle/working survives.** So `_rss` now appends to `/kaggle/working/rss.log`
and flushes and fsyncs on every call. If this run is killed too, that file's
last line names the crop and the stage, which is the one thing two killed
runs have not been able to say.

THE ONE BLIND CHANGE, labelled as blind. `patch_and_save(nodes0, edges0,
pool, outp)` writes the geff and then `from_geff` reads it back, but
`nodes0 / edges0 / pool` are not freed until after `compute_metric`. So the
node and edge arrays, the on-disk store and the rebuilt graph are all alive
at once for no reason. Freeing them straight after the save is safe and
cannot change a number. **It is a guess at the peak, not a diagnosis** -- the
rss.log is the diagnosis, and it will say whether this mattered.

WHAT IT RUNS: 2 tripwire crops + the 3 v3 never finished. The tripwires are
the two smallest of the 8 anchor crops with their v3 values asserted PER
CROP, which is a sharper check than the 8-crop aggregate and costs about two
minutes. A run that reproduces them and then completes the 3 gives rows that
join v3's 38 directly.

Attach: zhpara2's sources. CPU only.
"""
import os

src = open("zhpara2.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-09a-zhpara2"',
    'SCRIPT_VERSION = "2026-09-09b-zhpara3"')

# ---- 2 tripwire crops instead of 8 anchors ----------------------------
rep('''# The 8 anchors are re-solved so the run carries its own anchor; they are all
# small and already known to fit.
BOTH = ANCHOR8 + REMAIN''',
    '''# TWO tripwire crops, not eight. zhpara2 proved the harness is bit-identical
# across runs on these very crops, so the 8-crop aggregate anchor buys nothing
# a per-crop equality check does not, and it costs six crops of runtime before
# the interesting ones are reached. These two are the smallest of the eight;
# their v3 values are asserted to six decimals below, per crop and per arm.
TRIPWIRE = ["ZSNS001_z2y1x4", "ZSNS001_z2y1x6"]
V3_ADJ = {("ZSNS001_z2y1x4", "REF"): 0.819402,
          ("ZSNS001_z2y1x4", "HP"): 0.820898,
          ("ZSNS001_z2y1x6", "REF"): 0.764233,
          ("ZSNS001_z2y1x6", "HP"): 0.763003}
assert set(TRIPWIRE) <= set(ANCHOR8), TRIPWIRE
BOTH = TRIPWIRE + REMAIN''')

# ---- the RSS trace goes to a FILE, because stdout does not survive -----
rep('''    try:
        with open("/proc/self/status") as _f:
            _d = dict(l.split(":", 1) for l in _f if ":" in l)
        cur = _d.get("VmRSS", "?").strip()
        peak = _d.get("VmHWM", "?").strip()
    except OSError:
        cur = peak = "n/a"
    print(f"      [rss] {where:<22} now {cur:>12}  peak {peak:>12}",
          flush=True)''',
    '''    try:
        with open("/proc/self/status") as _f:
            _d = dict(l.split(":", 1) for l in _f if ":" in l)
        cur = _d.get("VmRSS", "?").strip()
        peak = _d.get("VmHWM", "?").strip()
    except OSError:
        cur = peak = "n/a"
    line = f"{time.time() - t_zh:8.0f}s  {where:<28} now {cur:>12}  peak {peak:>12}"
    print(f"      [rss] {line}", flush=True)
    # AND to a file, fsynced. zhpara v3 and zhpara2 were both SIGKILLed and
    # both came back with a ZERO-BYTE Kaggle log: on a killed session stdout
    # is gone entirely, so a printed trace cannot be read afterwards. This
    # file is the only thing that survives, and its last line is the answer.
    try:
        import os as _os          # zhpara's import block carries no `os`
        with open("/kaggle/working/rss.log", "a", encoding="utf-8") as _fh:
            _fh.write(line + "\\n")
            _fh.flush()
            _os.fsync(_fh.fileno())
    except OSError:
        pass''')

# ---- the one blind change, and it cannot alter a number ---------------
rep('''        patch_and_save(nodes0, edges0, pool, outp)
        rr = td.graph.IndexedRXGraph.from_geff(str(outp))''',
    '''        patch_and_save(nodes0, edges0, pool, outp)
        # BLIND, and safe: the geff is on disk now, so the node/edge arrays
        # and the repair pool are dead weight while from_geff rebuilds the
        # graph and compute_metric matches it against the GT. v2 held all of
        # it until after the metric. This cannot change a number; whether it
        # changes the peak is what rss.log is for.
        del nodes0, edges0, pool
        gc.collect()
        _rss(f"{tag} freed, pre from_geff")
        rr = td.graph.IndexedRXGraph.from_geff(str(outp))''')

rep('''        del dets, nodes0, edges0, pool, g
        gc.collect()''',
    '''        del dets, g
        gc.collect()''')

# ---- the per-crop tripwire assert -------------------------------------
rep('''    NGT[name] = int(len(gt_arr))
    done.append(name)''',
    '''    NGT[name] = int(len(gt_arr))
    if (name, "REF") in V3_ADJ:
        for _t in ("REF", "HP"):
            _got = _adj_of(ZROWS[_t][name])
            _want = V3_ADJ[(name, _t)]
            assert abs(_got - _want) <= 1e-5, (
                f"TRIPWIRE {name} {_t}: adj_J {_got:.6f} != v3's {_want:.6f}. "
                f"zhpara2 reproduced v3 bit for bit on this crop, so a "
                f"difference here means the harness moved and the 3 new "
                f"crops cannot be joined to v3's 38.")
        print(f"    tripwire {name}: both arms reproduce v3 to 1e-5",
              flush=True)
    done.append(name)''')

rep('print("ZHPARA2 OK", flush=True)', 'print("ZHPARA3 OK", flush=True)')
rep('print(f"ZHPARA2 INCOMPLETE ({len(done)} of {len(BOTH)} crops)", flush=True)',
    'print(f"ZHPARA3 INCOMPLETE ({len(done)} of {len(BOTH)} crops)", flush=True)')

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "zhpara3.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
