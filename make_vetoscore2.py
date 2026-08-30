"""Generate vetoscore2.py from fusescore.py by asserted text replacement.

vetoscore2: the FAITHFUL veto -- drop xl detections whose ep05 HEATMAP
response (hdval caches from dcsample, +-1 pooled-voxel window max) is
below tau. vetoscore tested a peak-list proxy instead and its 4 um NMS
made the proxy veto 4-6x too many points; this is the rule the user
actually designed and dc_gate.py actually measured (~90% veto precision
at 0.2-0.9% collateral).
"""
import ast
import pathlib

SRC = pathlib.Path(__file__).parent / "fusescore.py"
DST = pathlib.Path(__file__).parent / "vetoscore2.py"
text = SRC.read_text(encoding="utf-8")


def rep(old, new, n=1):
    global text
    c = text.count(old)
    assert c == n, f"anchor x{c}, wanted x{n}: {old[:70]!r}"
    text = text.replace(old, new)


rep("SCRIPT_VERSION 2026-08-29a-fusescore",
    "SCRIPT_VERSION 2026-08-30a-vetoscore2")
rep('SCRIPT_VERSION = "2026-08-29a-fusescore"',
    'SCRIPT_VERSION = "2026-08-30a-vetoscore2"')

rep("""THE QUESTION. detfuse2 measured the ep05-DeepCenter blend at +0.0155
detection F1 on the honest denominator with recall AND precision up and
FEWER detections per frame -- the first detection change to satisfy the
no-surplus criterion. This runs the three winning blends (F4 / W1101 /
DCheavy, cached by fusecache) through the FULL pipeline -- ranker, ILP,
post-link patches, official metric -- in three domains.""",
    """THE QUESTION. The faithful DeepCenter veto. vetoscore's peak-list
proxy (no ep05 PEAK within 3 um) vetoed 12-18%% of all points -- 4-6x
the local prediction -- because dcdet's 4 um NMS erases the peaks of
real cells shadowed by stronger neighbours, and it lost 0.05-0.07
adj_J in all three domains. THIS run uses the rule the user designed
and dc_gate.py measured (~90%% veto precision, 0.2-0.9%% collateral):
drop an xl detection iff ep05's HEATMAP response at that point
(dcsample's hdval caches, +-1 pooled-voxel window max) is < tau, for
tau in 0.02 / 0.05 / 0.10. Positions of survivors untouched, nothing
added. Same three-domain harness, same pre-registered same-sign rule.
The user's niche-overlap caveat is on record: the pipeline already
deletes points post-link (_short_filter), so the veto's value, if any,
must come from cleaning the candidate set BEFORE the edge ranker sees
it.""")

rep("""print("fdet caches:", {k: len(v) for k, v in sorted(FDET.items())},
      flush=True)
""",
    """print("fdet caches:", {k: len(v) for k, v in sorted(FDET.items())},
      flush=True)

# dcsample's hdval caches: ep05's raw heatmap response at every cached
# xl detection, row-aligned with detcache_/zhdet_ c{t}. The veto drops
# points with hd < tau; survivors keep exact position and score.
HDVAL = _walk_npz("hdval_")
print(f"hdval caches: {len(HDVAL)}", flush=True)
_VETO_STATS = defaultdict(lambda: [0, 0, 0])  # (tau,dom)->[pts,veto,mismatch]


def veto_dets(dets, name, tau):
    \"\"\"Drop xl detections whose ep05 heatmap response is < tau. A frame
    whose hdval row count mismatches the detcache keeps ALL its points
    (counted, so a systematic mismatch is visible in the stats).\"\"\"
    assert name in HDVAL, f"hdval missing for {name}"
    dom_ = name.split("_")[0] if name.startswith(("6bba", "44b6")) \\
        else "zh"
    st = _VETO_STATS[(tau, dom_)]
    z = np.load(HDVAL[name])
    out = []
    for t, (c, s) in enumerate(dets):
        c = np.asarray(c, float).reshape(-1, 3)
        s = np.asarray(s, float).reshape(-1)
        st[0] += len(c)
        hv = (np.asarray(z[f"h{t}"], float).reshape(-1)
              if f"h{t}" in z.files else None)
        if len(c) == 0:
            out.append((c, s))
            continue
        if hv is None or len(hv) != len(c):
            st[2] += 1
            out.append((c, s))
            continue
        keep = hv >= tau
        st[1] += int((~keep).sum())
        out.append((c[keep], s[keep]))
    return out
""")

rep('''def load_dets(name, det="BASE"):
    """BASE reads the shipped detcache; a blend tag reads fusecache's
    fdet file for that arm. Same layout either way."""
    path = CACHES[name] if det == "BASE" else FDET[det][name]
    z = np.load(path)
    T = max(int(k[1:]) for k in z.files if k.startswith("c")) + 1
    return [(z[f"c{t}"], z[f"s{t}"]) for t in range(T)]''',
    '''def load_dets(name, det="BASE"):
    """BASE reads the shipped detcache; VETO:<tau> reads the SAME cache
    and drops the points whose ep05 heatmap response is < tau."""
    z = np.load(CACHES[name])
    T = max(int(k[1:]) for k in z.files if k.startswith("c")) + 1
    dets = [(z[f"c{t}"], z[f"s{t}"]) for t in range(T)]
    if det.startswith("VETO:"):
        dets = veto_dets(dets, name, float(det.split(":", 1)[1]))
    return dets''')

rep('''    ("F4",        "F4",   "D32"),
    ("W1101",     "W1101", "D32"),
    ("DCheavy",   "DCheavy", "D32"),
]''',
    '''    ("V02",       "VETO:0.02", "D32"),
    ("V05",       "VETO:0.05", "D32"),
    ("V10",       "VETO:0.10", "D32"),
]''')

rep('''for _lbl, _det, _rk in RUN_ARMS:
    if _det != "BASE":
        _missing = [n for n in SET_6BBA + SET_44B6
                    if n not in FDET.get(_det, {})]
        assert not _missing, (
            f"{_det}: fdet cache missing for {_missing[:4]} -- attach "
            f"biohub-fusecache")''',
    '''_missing = [n for n in SET_6BBA + SET_44B6 if n not in HDVAL]
assert not _missing, (
    f"hdval missing for {_missing[:4]} -- attach biohub-dcsample")''')

rep('''    print(f"  domains: zebrahub {len(SET_ZH)}", flush=True)''',
    '''    print(f"  domains: zebrahub {len(SET_ZH)}", flush=True)
    _mzh = [n for n in SET_ZH if n not in HDVAL]
    assert not _mzh, f"hdval missing for zh {_mzh}"''')

rep('''            if det == "BASE":
                dets = dets_base
            else:
                z_ = np.load(FDET[det][name])
                dets = [(z_[f"c{t}"], z_[f"s{t}"]) for t in range(T)]''',
    '''            if det == "BASE":
                dets = dets_base
            else:
                dets = veto_dets(dets_base, name,
                                 float(det.split(":", 1)[1]))''')

rep('=== FUSESCORE: {len(RUN_ARMS)} arms x 3 domains',
    '=== VETOSCORE2: {len(RUN_ARMS)} arms x 3 domains')
rep('print("  PRE-REGISTERED RULE: a blend counts only if its adj_J '
    'delta vs "\n      "BASE (both on D32) has the SAME SIGN in all '
    'three domains.",\n      flush=True)',
    'print("  PRE-REGISTERED RULE: a veto arm counts only if its adj_J '
    'delta vs "\n      "BASE (both on D32) has the SAME SIGN in all '
    'three domains. The "\n      "user accepts a small recall '
    'sacrifice for fewer points.", flush=True)')

rep('print("\\n=== per-dataset, BASE vs DCheavy (both D32) ===", '
    'flush=True)',
    'print("\\n=== per-dataset, BASE vs V02 (both D32) ===", flush=True)')
rep('    pb, pn = PER.get(("BASE", d)), PER.get(("DCheavy", d))',
    '    pb, pn = PER.get(("BASE", d)), PER.get(("V02", d))')

rep('''print("\\n=== reading it ===", flush=True)''',
    '''print("\\n=== veto counts, per tau x domain ===", flush=True)
for (tau_, dom_) in sorted(_VETO_STATS):
    p_, v_, m_ = _VETO_STATS[(tau_, dom_)]
    print(f"  tau {tau_:.2f} {dom_:<5} {v_:>8,}/{p_:>9,} vetoed "
          f"({v_ / max(1, p_):.2%}), mismatched frames kept whole: {m_}",
          flush=True)

print("\\n=== reading it ===", flush=True)''')

rep('''print("  This is the arbiter the detection tables could not be: the "
      "official metric, three domains, clean 44b6, leak-safe ranker. A "
      "SAME-SIGN-POSITIVE blend here is a submission candidate with a "
      "mechanism behind it (the edgediag cold-miss bucket plus the "
      "detfuse2 rescue rate); anything else closes the blend line at "
      "the pipeline level.", flush=True)''',
    '''print("  FIRST check the veto counts against dc_gate's prediction "
      "(tau 0.05 should delete ~3%% of points, not 12-18%%) -- if they "
      "disagree, the caches lie and the adj_J rows are void. If they "
      "agree: a SAME-SIGN-POSITIVE arm is a submission candidate that "
      "deletes ~90%%-fake points pre-ranker; same-sign-negative "
      "settles that pre-ranker deletion at these taus does not pay in "
      "this pipeline (the user's niche-overlap reading: post-link "
      "_short_filter already owns this niche), and the veto signal "
      "itself remains available as a ranker FEATURE, untested.",
      flush=True)''')

rep('print(f"\\nFUSESCORE DONE in', 'print(f"\\nVETOSCORE2 DONE in')

DST.write_text(text, encoding="utf-8")
ast.parse(text)
print(f"wrote {DST} ({len(text.splitlines())} lines), ast OK")
