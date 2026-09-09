"""zhpara3.py -> zhpara4.py: is it a quadratic allocation inside compute_metric?

WHAT zhpara3 ESTABLISHED. Its `rss.log` survived the kill -- the first time in
four zebrahub deaths that any evidence did -- and it names the stage:

    97s  REF after zh_load          1,201,936 kB
   312s  REF after solve_one        2,398,420 kB   peak 2,409,328 kB
   323s  REF freed, pre from_geff   2,132,648 kB
   323s  REF after from_geff        2,132,736 kB
                                    <- killed; the next sample would have been
                                       "REF after compute_metric"

So: **killed inside `compute_metric` on `ZSNS001_z2y2x7`.** The same trace
shows that crop's `solve_one` taking **215 s** against 9-13 s on the two
tripwire crops, so it is roughly twenty times the work of a small one.

WHAT DOES NOT ADD UP, and it is the reason for this kernel. RSS entering the
metric was **2.1 GB**, with a peak of 2.4 GB. A Kaggle CPU box has about
30 GB. Nothing in that trace is running out of memory gradually, so
"it OOMed" as a description of a slow climb is wrong.

THE HYPOTHESIS, and it fits every fact. `compute_metric` matches the
predicted graph against the ground-truth graph, and `zhpara` imports
`scipy.optimize.linear_sum_assignment`. A DENSE cost matrix over
N_pred x N_gt nodes is quadratic: at 60,000 x 60,000 in float64 that is
28.8 GB in ONE allocation. That would die between two RSS samples, leave no
Python traceback (the OOM killer sends SIGKILL), and produce the zero-byte
log every one of these runs has produced -- while the last recorded RSS
stays modest, which is exactly what was observed.

⚠ IT IS A HYPOTHESIS. Two others are still live and this kernel does not
distinguish them by itself: a segfault inside a native extension
(rustworkx / scipy) would also kill silently at modest RSS, and so would a
cgroup limit far below the advertised 30 GB. What this run does is measure
the ONE number that would settle the quadratic story.

WHAT IT MEASURES, all of it written to `/kaggle/working/rss.log` because a
killed session loses stdout entirely (working-agreement.md):

  * `MemTotal` and `MemAvailable` once at the start -- the ceiling this box
    actually has, which has been assumed and never read;
  * before EVERY `compute_metric` call: `g` and `gtg` node and edge counts,
    and the dense-matrix size `N_pred * N_gt * 8` in GB.

The two tripwire crops give the same numbers for graphs that DO fit, so the
comparison is inside this one run: if `ZSNS001_z2y2x7` shows a dense size in
the tens of GB and the tripwires show megabytes, the mechanism is settled and
the fix is a sparse or blocked matcher, not a bigger box.

Same 5 crops, same per-crop tripwire asserts against v3's values. CPU only,
about 10 minutes to the point of interest.

Attach: zhpara3's sources.
"""
import os

src = open("zhpara3.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-09b-zhpara3"',
    'SCRIPT_VERSION = "2026-09-09c-zhpara4"')

# ---- read the ceiling instead of assuming it ---------------------------
rep('''def _rss(where):''',
    '''def _meminfo():
    """The box's actual limits, written once. Every zebrahub post-mortem in
    this record has ASSUMED ~30 GB and none has read it."""
    try:
        with open("/proc/meminfo") as _f:
            _d = dict(l.split(":", 1) for l in _f if ":" in l)
        return (f"MemTotal {_d.get('MemTotal', '?').strip()}  "
                f"MemAvailable {_d.get('MemAvailable', '?').strip()}")
    except OSError:
        return "meminfo unavailable"


def _rss(where):''')

rep('''ZROWS = {tag: {} for tag, _ in ARM_SRC}''',
    '''ZROWS = {tag: {} for tag, _ in ARM_SRC}
_MEM = _meminfo()
print(f"  box: {_MEM}", flush=True)
try:
    import os as _os0
    with open("/kaggle/working/rss.log", "a", encoding="utf-8") as _f0:
        _f0.write(f"BOX {_MEM}\\n")
        _f0.flush()
        _os0.fsync(_f0.fileno())
except OSError:
    pass''')

# ---- the number that settles the quadratic story -----------------------
rep('''        _rss(f"{tag} after from_geff")
        er = compute_metric(g, gtg, scale=tuple(SCALE))''',
    '''        # THE MEASUREMENT. zhpara3 died between "after from_geff" and
        # "after compute_metric" at a modest 2.1 GB, so a gradual climb is
        # not what killed it. A dense N_pred x N_gt cost matrix is, and this
        # prints its size in GB before the call that would allocate it.
        try:
            _np_, _ne_ = g.num_nodes(), g.num_edges()
            _ng_, _ge_ = gtg.num_nodes(), gtg.num_edges()
            _dense = _np_ * _ng_ * 8 / 1e9
            _rss(f"{tag} pred {_np_}n/{_ne_}e gt {_ng_}n/{_ge_}e "
                 f"dense {_dense:.1f}GB")
        except Exception as _e:            # never let the probe kill the run
            _rss(f"{tag} graph-size probe failed: {type(_e).__name__}")
        _rss(f"{tag} after from_geff")
        er = compute_metric(g, gtg, scale=tuple(SCALE))''')

rep('print("ZHPARA3 OK", flush=True)', 'print("ZHPARA4 OK", flush=True)')
rep('print(f"ZHPARA3 INCOMPLETE ({len(done)} of {len(BOTH)} crops)", flush=True)',
    'print(f"ZHPARA4 INCOMPLETE ({len(done)} of {len(BOTH)} crops)", flush=True)')

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "zhpara4.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
