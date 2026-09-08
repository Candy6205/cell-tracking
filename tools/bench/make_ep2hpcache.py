"""edgeprob2_cache.py -> ep2hpcache.py: the SECOND edge head on hpara points.

The twin of `make_ephpcache.py`, and it exists for the same reason: the
harness joins BOTH heads' cached rows to its candidate pairs by (t, i, j),
and both go through the same `_load_head`, so re-running only the first head
would move the failure by one line instead of fixing it. `hparabench`'s F4HP
arm died on

    44b6_0b24845f t=0: 40 of 990 harness pairs have no edgeprob row

because the pair set is GATED ON DISTANCE (`GATE_UM = 10.0`) and hpara moves
every point by a median 0.43 um, so some pairs cross that line in each
direction. The detections are identical; the PAIRS are not.

The cut is smaller than ephpcache's because edgeprob2_cache already runs on
exactly the 24 evaluation movies (`names_f4 = list(EVAL_24)`):

  * `CACHES_F4` discovers `fdet_F4hp_*` instead of `fdet_F4_`;
  * the output label becomes `F4hp_<name>`, so the file is
    `edgeprob2_F4hp_<name>.npz`.

Nothing that computes a probability is touched -- the arm must differ from
the control in COORDINATES only.

NO ANCHOR (the outputs differ by construction). The check that IS available:
`n_per_frame` must reproduce the shipped `edgeprob2_F4_<name>.npz` exactly,
because hparacache neither added nor removed a detection.

Attach: edgeprob2_cache's own sources, plus faustine649/biohub-hparacache
for the hpara primaries and faustine649/biohub-edgeprob2 for the check.
"""
import os

src = open("edgeprob2_cache.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-03b-seed2"',
    'SCRIPT_VERSION = "2026-09-08a-ep2hpcache"')

rep('''CACHES_F4 = _walk_files(lambda fn: fn[len("fdet_F4_"):-len(".npz")]
                        if fn.startswith("fdet_F4_") and fn.endswith(".npz")''',
    '''CACHES_F4 = _walk_files(lambda fn: fn[len("fdet_F4hp_"):-len(".npz")]
                        if fn.startswith("fdet_F4hp_") and fn.endswith(".npz")''')

rep('''assert len(CACHES_F4) >= 24, ("fewer than 24 fdet_F4_*.npz under ''',
    '''assert len(CACHES_F4) >= 24, ("fewer than 24 fdet_F4hp_*.npz under ''')

rep('''    label = name if arm == "base" else f"F4_{name}"''',
    '''    label = name if arm == "base" else f"F4hp_{name}"''')

# ---- the n_per_frame cross-check against the shipped second-head cache ----
rep('''        OUT / f"edgeprob2_{label}.npz",''',
    '''        OUT / f"edgeprob2_{label}.npz",''')

rep('''    mb = (OUT / f"edgeprob2_{label}.npz").stat().st_size / 2 ** 20''',
    '''    _ship = SHIPPED_EP2.get(name)
    if _ship is None:
        NPF_BAD.append(f"{name}(no shipped cache)")
        print(f"  !! {name}: no shipped edgeprob2_F4_ to check n_per_frame "
              f"against -- the only available check on this run is off",
              flush=True)
    else:
        _z = np.load(_ship)
        _old = np.asarray(_z["n_per_frame"], np.int64)
        _new = np.asarray(fwd.n, np.int64)
        if _old.shape != _new.shape or not np.array_equal(_old, _new):
            NPF_BAD.append(name)
            print(f"  !! {name}: n_per_frame DIFFERS from the shipped "
                  f"edgeprob2_F4_ cache -- hparacache was supposed to keep "
                  f"every detection, so the arms are NOT scoring the same "
                  f"cells", flush=True)
        _z.close()
    mb = (OUT / f"edgeprob2_{label}.npz").stat().st_size / 2 ** 20''')

rep('''names_f4 = list(EVAL_24)''',
    '''names_f4 = list(EVAL_24)
SHIPPED_EP2 = _walk_files(lambda fn: fn[len("edgeprob2_F4_"):-len(".npz")]
                          if fn.startswith("edgeprob2_F4_")
                          and fn.endswith(".npz") else None)
print(f"shipped edgeprob2_F4_ caches for the n_per_frame check: "
      f"{len(SHIPPED_EP2)}", flush=True)
NPF_BAD = []''')

rep('''    print(f"EDGEPROB2 OK ({n_base} base + {n_f4} F4 of {len(jobs)})",
          flush=True)''',
    '''    print(f"n_per_frame check: {len(names_f4) - len(NPF_BAD)} of "
          f"{len(names_f4)} datasets reproduce the shipped counts"
          + (f" -- MISMATCHES: {NPF_BAD}" if NPF_BAD else " exactly"),
          flush=True)
    print(f"EP2HPCACHE {'OK' if not NPF_BAD else 'NPF MISMATCH'} "
          f"({n_base + n_f4} of {len(jobs)})", flush=True)''')
rep('EDGEPROB2 INCOMPLETE', 'EP2HPCACHE INCOMPLETE')

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "ep2hpcache.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
