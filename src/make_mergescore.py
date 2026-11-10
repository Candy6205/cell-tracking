"""Generate mergescore.py from fusescore.py by asserted text replacement.

mergescore: dcmerge REOPENED under the amended three-domain rule, with
OUR ep05 instead of the published DeepCenter. Base arm is F4 (the
shipped 0.925 detections, fdet caches); merge arms add ep05 points
(score >= tau, farther than 4 um from every F4 point) as PRIMARY nodes.
44b6 + zh vote; 6bba is reported but NON-VOTING (memorization-tainted
for detection-side changes, per the 2026-08-31 amendment). The question
44b6 answers: does the added points' edge conversion beat the LINEAR
node tax under competition-style annotation.
"""
import ast
import pathlib

SRC = pathlib.Path(__file__).parent / "fusescore.py"
DST = pathlib.Path(__file__).parent / "mergescore.py"
text = SRC.read_text(encoding="utf-8")


def rep(old, new, n=1):
    global text
    c = text.count(old)
    assert c == n, f"anchor x{c}, wanted x{n}: {old[:70]!r}"
    text = text.replace(old, new)


rep("SCRIPT_VERSION 2026-08-29a-fusescore",
    "SCRIPT_VERSION 2026-08-31a-mergescore")
rep('SCRIPT_VERSION = "2026-08-29a-fusescore"',
    'SCRIPT_VERSION = "2026-08-31a-mergescore"')

rep("""THE QUESTION. detfuse2 measured the ep05-DeepCenter blend at +0.0155
detection F1 on the honest denominator with recall AND precision up and
FEWER detections per frame -- the first detection change to satisfy the
no-surplus criterion. This runs the three winning blends (F4 / W1101 /
DCheavy, cached by fusecache) through the FULL pipeline -- ranker, ILP,
post-link patches, official metric -- in three domains.""",
    """THE QUESTION. dcmerge REOPENED. 2026-08-28's dcmerge added the
PUBLISHED DeepCenter's points as primaries: zh +0.022..+0.038 monotone,
6bba negative monotone, 44b6 permanently nan (all 71 in that model's
training set) -- discarded on sign disagreement. Two things changed:
(1) the 6bba veto on detection-side changes is memorization-tainted
(F4 lost 6bba locally and won the LB +0.003, ref 55889500), so 6bba is
NON-VOTING here; (2) our ep05 excludes 12 44b6 sets from training, so
44b6 -- competition-style annotation, the domain that prices the
user's node-tax objection -- can finally vote. Base arm is F4 (the
shipped 0.925 detections); merge arms add ep05 points (score >= tau,
farther than MERGE_UM from every F4 point) as primaries at score
UNET_THRESH, dcmerge's admission rule verbatim. The user's framing on
record: zh's dense GT subsidises added points (no cell-count ceiling
hit), so the DECIDING vote is 44b6 -- added-point edge conversion vs
the linear node tax the LB charges (+10.3% output cost -0.002, twice
measured).""")

rep("""print("fdet caches:", {k: len(v) for k, v in sorted(FDET.items())},
      flush=True)
""",
    """print("fdet caches:", {k: len(v) for k, v in sorted(FDET.items())},
      flush=True)

# dccache3's epoch-5 detections, the ADD source for the merge arms.
DCDET_EP05 = _walk_npz("dcdet_ep05_")
print(f"dcdet_ep05 caches: {len(DCDET_EP05)}", flush=True)
MERGE_UM = 4.0    # dcmerge's DC_MERGE_UM: same separation our NMS enforces
_MERGE_STATS = defaultdict(lambda: [0, 0])   # (tau,dom) -> [f4 pts, added]


def merge_dets(dets, name, tau):
    \"\"\"dcmerge's admission rule on an F4 base: an ep05 point enters iff
    its own score >= tau and it is farther than MERGE_UM from every F4
    point at that frame; it enters at score UNET_THRESH, i.e. as a
    PRIMARY node that never touches the repair band.\"\"\"
    assert name in DCDET_EP05, f"dcdet_ep05 missing for {name}"
    dom_ = name.split("_")[0] if name.startswith(("6bba", "44b6")) \\
        else "zh"
    st = _MERGE_STATS[(tau, dom_)]
    z = np.load(DCDET_EP05[name])
    out = []
    for t, (c, s) in enumerate(dets):
        c = np.asarray(c, float).reshape(-1, 3)
        s = np.asarray(s, float).reshape(-1)
        st[0] += len(c)
        dc = np.zeros((0, 3))
        if f"c{t}" in z.files:
            dc = np.asarray(z[f"c{t}"], float).reshape(-1, 3)
            ds = np.asarray(z[f"s{t}"], float).reshape(-1)
            dc = dc[ds >= tau]
        if len(dc) and len(c):
            d, _ = cKDTree(c * SCALE[None, :]).query(dc * SCALE[None, :])
            dc = dc[d > MERGE_UM]
        st[1] += len(dc)
        if len(dc):
            out.append((np.concatenate([c, dc], 0),
                        np.concatenate([s, np.full(len(dc),
                                                   UNET_THRESH)], 0)))
        else:
            out.append((c, s))
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
    """BASE reads the shipped detcache; F4 reads fusecache's fdet;
    MERGE:<tau> reads the F4 dets and adds ep05 points above tau."""
    if det.startswith("MERGE:"):
        return merge_dets(load_dets(name, "F4"), name,
                          float(det.split(":", 1)[1]))
    path = CACHES[name] if det == "BASE" else FDET[det][name]
    z = np.load(path)
    T = max(int(k[1:]) for k in z.files if k.startswith("c")) + 1
    return [(z[f"c{t}"], z[f"s{t}"]) for t in range(T)]''')

rep('''    ("BASE",      "BASE", "D32"),    # the in-run baseline for deltas
    ("F4",        "F4",   "D32"),
    ("W1101",     "W1101", "D32"),
    ("DCheavy",   "DCheavy", "D32"),
]''',
    '''    ("F4",        "F4",   "D32"),    # the in-run baseline for deltas
    ("M50",       "MERGE:0.50", "D32"),
    ("M30",       "MERGE:0.30", "D32"),
    ("M15",       "MERGE:0.15", "D32"),
]''')

rep('''for _lbl, _det, _rk in RUN_ARMS:
    if _det != "BASE":
        _missing = [n for n in SET_6BBA + SET_44B6
                    if n not in FDET.get(_det, {})]
        assert not _missing, (
            f"{_det}: fdet cache missing for {_missing[:4]} -- attach "
            f"biohub-fusecache")''',
    '''_missing = [n for n in SET_6BBA + SET_44B6
            if n not in FDET.get("F4", {}) or n not in DCDET_EP05]
assert not _missing, (
    f"F4/dcdet_ep05 cache missing for {_missing[:4]} -- attach "
    f"biohub-fusecache AND biohub-dccache3")''')

rep('''    print(f"  domains: zebrahub {len(SET_ZH)}", flush=True)''',
    '''    print(f"  domains: zebrahub {len(SET_ZH)}", flush=True)
    _mzh = [n for n in SET_ZH
            if n not in FDET.get("F4", {}) or n not in DCDET_EP05]
    assert not _mzh, f"F4/dcdet_ep05 missing for zh {_mzh}"''')

rep('''            if det == "BASE":
                dets = dets_base
            else:
                z_ = np.load(FDET[det][name])
                dets = [(z_[f"c{t}"], z_[f"s{t}"]) for t in range(T)]''',
    '''            if det == "BASE":
                dets = dets_base
            else:
                z_ = np.load(FDET["F4"][name])
                dets = [(z_[f"c{t}"], z_[f"s{t}"]) for t in range(T)]
                if det.startswith("MERGE:"):
                    dets = merge_dets(dets, name,
                                      float(det.split(":", 1)[1]))''')

rep('=== FUSESCORE: {len(RUN_ARMS)} arms x 3 domains',
    '=== MERGESCORE: {len(RUN_ARMS)} arms x 3 domains')
rep('print("  PRE-REGISTERED RULE: a blend counts only if its adj_J '
    'delta vs "\n      "BASE (both on D32) has the SAME SIGN in all '
    'three domains.",\n      flush=True)',
    'print("  PRE-REGISTERED RULE (amended 2026-08-31): a merge arm '
    'counts only "\n      "if its adj_J delta vs F4 (both on D32) has '
    'the SAME SIGN in 44b6 "\n      "AND zh. 6bba is reported but '
    'NON-VOTING: memorization-tainted "\n      "for detection-side '
    'changes (F4 lost 6bba locally, won the LB).", flush=True)')

# deltas are taken against F4, not BASE
rep('''=== adj_J DELTA vs BASE (both D32), per domain ===''',
    '''=== adj_J DELTA vs F4 (both D32), per domain ===''')
rep('''for lbl, det, rk in RUN_ARMS:
    if lbl in ("BASE", "BASE_D187"):
        continue
    ds = [res[(lbl, d)]["adj_edge_jaccard"]
          - res[("BASE", d)]["adj_edge_jaccard"] for d in DOMAINS]
    same = all(x > 0 for x in ds) or all(x < 0 for x in ds)
    verdict = ("SAME SIGN, POSITIVE" if same and ds[0] > 0 else
               "same sign, negative" if same else
               "signs disagree -- domain-specific, discard")''',
    '''for lbl, det, rk in RUN_ARMS:
    if lbl in ("F4", "BASE_D187"):
        continue
    ds = [res[(lbl, d)]["adj_edge_jaccard"]
          - res[("F4", d)]["adj_edge_jaccard"] for d in DOMAINS]
    vote = ds[1:]     # 44b6 and zh; 6bba (ds[0]) reported, non-voting
    same = all(x > 0 for x in vote) or all(x < 0 for x in vote)
    verdict = ("CLEAN DOMAINS POSITIVE" if same and vote[0] > 0 else
               "clean domains negative" if same else
               "clean domains disagree -- discard")''')
rep('''print("\\n=== div_J DELTA vs BASE (D32) ===", flush=True)
for lbl, det, rk in RUN_ARMS:
    if lbl in ("BASE", "BASE_D187"):
        continue
    ds = [res[(lbl, d)]["division_jaccard"]
          - res[("BASE", d)]["division_jaccard"] for d in DOMAINS]''',
    '''print("\\n=== div_J DELTA vs F4 (D32) ===", flush=True)
for lbl, det, rk in RUN_ARMS:
    if lbl in ("F4", "BASE_D187"):
        continue
    ds = [res[(lbl, d)]["division_jaccard"]
          - res[("F4", d)]["division_jaccard"] for d in DOMAINS]''')

rep('print("\\n=== per-dataset, BASE vs DCheavy (both D32) ===", '
    'flush=True)',
    'print("\\n=== per-dataset, F4 vs M30 (both D32) ===", flush=True)')
rep('    pb, pn = PER.get(("BASE", d)), PER.get(("DCheavy", d))',
    '    pb, pn = PER.get(("F4", d)), PER.get(("M30", d))')

rep('''print("\\n=== reading it ===", flush=True)''',
    '''print("\\n=== points added, per tau x domain ===", flush=True)
for (tau_, dom_) in sorted(_MERGE_STATS):
    p_, a_ = _MERGE_STATS[(tau_, dom_)]
    print(f"  tau {tau_:.2f} {dom_:<5} +{a_:>8,} on {p_:>9,} F4 points "
          f"(+{a_ / max(1, p_):.2%} nodes)", flush=True)

print("\\n=== reading it ===", flush=True)''')

rep('''print("  This is the arbiter the detection tables could not be: the "
      "official metric, three domains, clean 44b6, leak-safe ranker. A "
      "SAME-SIGN-POSITIVE blend here is a submission candidate with a "
      "mechanism behind it (the edgediag cold-miss bucket plus the "
      "detfuse2 rescue rate); anything else closes the blend line at "
      "the pipeline level.", flush=True)''',
    '''print("  The deciding vote is 44b6: competition-style annotation, "
      "our biggest node deficit (-14.8%%), and ep05-clean. zh's dense "
      "GT subsidises added points (the user's observation), so a "
      "zh-only win repeats dcmerge and settles nothing. CLEAN DOMAINS "
      "POSITIVE with a modest node growth = LB candidate on top of "
      "0.925; clean-domains-negative = the point-add line closes with "
      "ep05 too, and what it settles is that the node tax beats the "
      "conversion rate even where the deficit is largest.", flush=True)''')

rep('print(f"\\nFUSESCORE DONE in', 'print(f"\\nMERGESCORE DONE in')

rep('''print("  BASE_D187 6bba must read adj_J 0.9293 (dcpool). BASE (D32) is "
      "a NEW baseline -- no recorded anchor exists for it at this "
      "configuration; it is the in-run reference the blends are judged "
      "against.", flush=True)
for lbl in ("BASE_D187", "BASE"):''',
    '''print("  BASE_D187 6bba must read adj_J 0.9293 (dcpool). F4 x D32 is "
      "the in-run reference the merge arms are judged against; its own "
      "recorded row (fusescore) is 6bba 0.9198 / 44b6 0.8303 / zh "
      "0.8115 and must reproduce.", flush=True)
for lbl in ("BASE_D187", "F4"):''')

DST.write_text(text, encoding="utf-8")
ast.parse(text)
print(f"wrote {DST} ({len(text.splitlines())} lines), ast OK")
