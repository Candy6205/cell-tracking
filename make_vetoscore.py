"""Generate vetoscore.py from fusescore.py by asserted text replacement.

vetoscore: same three-domain pipeline harness, but the judged arms DROP
xl detections DeepCenter-ep05 is silent about (no ep05 peak >= tau
within 3 um), instead of blending heatmaps. Positions of surviving
points are untouched; nothing is added.
"""
import ast
import pathlib

SRC = pathlib.Path(__file__).parent / "fusescore.py"
DST = pathlib.Path(__file__).parent / "vetoscore.py"
text = SRC.read_text(encoding="utf-8")


def rep(old, new, n=1):
    global text
    c = text.count(old)
    assert c == n, f"anchor x{c}, wanted x{n}: {old[:70]!r}"
    text = text.replace(old, new)


# -- version stamp (docstring + assignment) -------------------------------
rep("SCRIPT_VERSION 2026-08-29a-fusescore",
    "SCRIPT_VERSION 2026-08-30a-vetoscore")
rep('SCRIPT_VERSION = "2026-08-29a-fusescore"',
    'SCRIPT_VERSION = "2026-08-30a-vetoscore"')

# -- question paragraph in the docstring ----------------------------------
rep("""THE QUESTION. detfuse2 measured the ep05-DeepCenter blend at +0.0155
detection F1 on the honest denominator with recall AND precision up and
FEWER detections per frame -- the first detection change to satisfy the
no-surplus criterion. This runs the three winning blends (F4 / W1101 /
DCheavy, cached by fusecache) through the FULL pipeline -- ranker, ILP,
post-link patches, official metric -- in three domains.""",
    """THE QUESTION. The user's split of the fusion idea: DeepCenter as a
VETO ONLY. dc_gate.py measured locally (judge_rows.npz): among
candidates xl keeps, those where ep05 is silent are ~90% fake at a
0.2-0.9%% collateral kill rate (6bba comp positives: 0.19%% at tau
0.02). Unlike blending, a veto never moves an xl peak and never adds a
point -- it only deletes DC-silent ones -- so the three paths by which
DCheavy lost 6bba edge_J (moved/lost/added peaks) are all closed by
construction. This is the pipeline test: veto arms at tau 0.05/0.10/
0.15 (cache floor is 0.05), full solve, official metric, three
domains. Tenth reminder: detection-level numbers have anti-predicted
adj_J nine times; only this harness decides.""")

# -- veto machinery after the fdet print ----------------------------------
rep("""print("fdet caches:", {k: len(v) for k, v in sorted(FDET.items())},
      flush=True)
""",
    """print("fdet caches:", {k: len(v) for k, v in sorted(FDET.items())},
      flush=True)

# dccache3's epoch-5 DeepCenter detections, used as a VETO oracle: an xl
# detection with NO ep05 peak (score >= tau) within VETO_UM is dropped
# BEFORE the pipeline sees it. Surviving points keep their exact
# position and score; nothing is added.
DCDET_EP05 = _walk_npz("dcdet_ep05_")
print(f"dcdet_ep05 caches: {len(DCDET_EP05)}", flush=True)
VETO_UM = 3.0
_VETO_STATS = defaultdict(lambda: [0, 0, 0])  # (tau,dom)->[pts,veto,dc0]


def veto_dets(dets, name, tau):
    \"\"\"Drop xl detections ep05 is silent about. A frame where ep05 has
    ZERO peaks >= tau keeps ALL its xl points -- a dead oracle must not
    veto a whole frame.\"\"\"
    assert name in DCDET_EP05, f"dcdet_ep05 missing for {name}"
    dom_ = name.split("_")[0] if name.startswith(("6bba", "44b6")) \\
        else "zh"
    st = _VETO_STATS[(tau, dom_)]
    z = np.load(DCDET_EP05[name])
    out = []
    for t, (c, s) in enumerate(dets):
        c = np.asarray(c, float).reshape(-1, 3)
        s = np.asarray(s, float).reshape(-1)
        dc = np.zeros((0, 3))
        if f"c{t}" in z.files:
            dc = np.asarray(z[f"c{t}"], float).reshape(-1, 3)
            ds = np.asarray(z[f"s{t}"], float).reshape(-1)
            dc = dc[ds >= tau]
        st[0] += len(c)
        if len(c) == 0:
            out.append((c, s))
            continue
        if len(dc) == 0:
            st[2] += 1
            out.append((c, s))
            continue
        d, _ = cKDTree(dc * SCALE[None, :]).query(c * SCALE[None, :])
        keep = d <= VETO_UM
        st[1] += int((~keep).sum())
        out.append((c[keep], s[keep]))
    return out
""")

# -- load_dets learns the VETO tag ----------------------------------------
rep('''def load_dets(name, det="BASE"):
    """BASE reads the shipped detcache; a blend tag reads fusecache's
    fdet file for that arm. Same layout either way."""
    path = CACHES[name] if det == "BASE" else FDET[det][name]
    z = np.load(path)
    T = max(int(k[1:]) for k in z.files if k.startswith("c")) + 1
    return [(z[f"c{t}"], z[f"s{t}"]) for t in range(T)]''',
    '''def load_dets(name, det="BASE"):
    """BASE reads the shipped detcache; VETO:<tau> reads the SAME cache
    and drops the points DeepCenter ep05 is silent about."""
    z = np.load(CACHES[name])
    T = max(int(k[1:]) for k in z.files if k.startswith("c")) + 1
    dets = [(z[f"c{t}"], z[f"s{t}"]) for t in range(T)]
    if det.startswith("VETO:"):
        dets = veto_dets(dets, name, float(det.split(":", 1)[1]))
    return dets''')

# -- arms -----------------------------------------------------------------
rep('''    ("F4",        "F4",   "D32"),
    ("W1101",     "W1101", "D32"),
    ("DCheavy",   "DCheavy", "D32"),
]''',
    '''    ("V05",       "VETO:0.05", "D32"),
    ("V10",       "VETO:0.10", "D32"),
    ("V15",       "VETO:0.15", "D32"),
]''')

# -- coverage assert replaces the fdet assert -----------------------------
rep('''for _lbl, _det, _rk in RUN_ARMS:
    if _det != "BASE":
        _missing = [n for n in SET_6BBA + SET_44B6
                    if n not in FDET.get(_det, {})]
        assert not _missing, (
            f"{_det}: fdet cache missing for {_missing[:4]} -- attach "
            f"biohub-fusecache")''',
    '''_missing = [n for n in SET_6BBA + SET_44B6 if n not in DCDET_EP05]
assert not _missing, (
    f"dcdet_ep05 missing for {_missing[:4]} -- attach biohub-dccache3")''')

# -- zh coverage assert right after the zh domain print -------------------
rep('''    print(f"  domains: zebrahub {len(SET_ZH)}", flush=True)''',
    '''    print(f"  domains: zebrahub {len(SET_ZH)}", flush=True)
    _mzh = [n for n in SET_ZH if n not in DCDET_EP05]
    assert not _mzh, f"dcdet_ep05 missing for zh {_mzh}"''')

# -- zh arm dispatch ------------------------------------------------------
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

# -- banner ---------------------------------------------------------------
rep('=== FUSESCORE: {len(RUN_ARMS)} arms x 3 domains',
    '=== VETOSCORE: {len(RUN_ARMS)} arms x 3 domains')
rep('print("  PRE-REGISTERED RULE: a blend counts only if its adj_J '
    'delta vs "\n      "BASE (both on D32) has the SAME SIGN in all '
    'three domains.",\n      flush=True)',
    'print("  PRE-REGISTERED RULE: a veto arm counts only if its adj_J '
    'delta vs "\n      "BASE (both on D32) has the SAME SIGN in all '
    'three domains. The "\n      "user accepts a small recall '
    'sacrifice for fewer points.", flush=True)')

# -- per-dataset detail on the safest veto arm ----------------------------
rep('print("\\n=== per-dataset, BASE vs DCheavy (both D32) ===", '
    'flush=True)',
    'print("\\n=== per-dataset, BASE vs V05 (both D32) ===", flush=True)')
rep('    pb, pn = PER.get(("BASE", d)), PER.get(("DCheavy", d))',
    '    pb, pn = PER.get(("BASE", d)), PER.get(("V05", d))')

# -- veto stats before the reading ----------------------------------------
rep('''print("\\n=== reading it ===", flush=True)''',
    '''print("\\n=== veto counts, per tau x domain ===", flush=True)
for (tau_, dom_) in sorted(_VETO_STATS):
    p_, v_, z0_ = _VETO_STATS[(tau_, dom_)]
    print(f"  tau {tau_:.2f} {dom_:<5} {v_:>8,}/{p_:>9,} vetoed "
          f"({v_ / max(1, p_):.2%}), dc-empty frames kept whole: {z0_}",
          flush=True)

print("\\n=== reading it ===", flush=True)''')

# -- reading text ---------------------------------------------------------
rep('''print("  This is the arbiter the detection tables could not be: the "
      "official metric, three domains, clean 44b6, leak-safe ranker. A "
      "SAME-SIGN-POSITIVE blend here is a submission candidate with a "
      "mechanism behind it (the edgediag cold-miss bucket plus the "
      "detfuse2 rescue rate); anything else closes the blend line at "
      "the pipeline level.", flush=True)''',
    '''print("  The veto is the half of the user's gate hypothesis that "
      "survived the local test (rescue died: only 24-30%% of rescued "
      "points are real). A SAME-SIGN-POSITIVE veto arm is a submission "
      "candidate: it deletes ~90%%-fake points without touching any "
      "surviving position. Anything else settles the CLAIM that "
      "DC-silence at these taus prices deletion correctly -- the "
      "measured veto precision itself stands and other uses of it "
      "(ranker feature, pool filter) remain untested.", flush=True)''')

rep('print(f"\\nFUSESCORE DONE in', 'print(f"\\nVETOSCORE DONE in')

DST.write_text(text, encoding="utf-8")
ast.parse(text)
print(f"wrote {DST} ({len(text.splitlines())} lines), ast OK")
