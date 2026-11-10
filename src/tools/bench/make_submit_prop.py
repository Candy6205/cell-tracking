"""submit_img.py -> submit_prop.py: forkprop's orphan patch (no cap, diverge 1.5, contrast-ranked proposals)
under a joint image gate fitted on that patch's own forks in both domains.

WHY (2026-09-17). forkprop (128 6bba movies): the orphan-fork patch that admits every geometric proposal
(caps x100, DIV_DIVERGE_UM 1.5, proposals ranked by the image contrast of mother + daughters) and lets the
joint image gate prune it beats what ships (REF + jointimg, LB 0.951) by +0.0050 [+0.0012, +0.0089] of
score, 61/36/64 against 55/42/70, adj_J unchanged; every capped arm is within its interval of zero and
the 20-column gate cannot do it. fk44prop measures the 44b6 half and fits the gate on the arm's forks of
both domains (fork_gate_prop.pkl); this file ships that pair.

WHAT CHANGES against submit_img.py (everything else byte-identical):
  * DIV_DIVERGE_UM 2.25 -> 1.5; FP_CAP 100 multiplies both caps of the orphan pass; FP_RANK "contrast":
    the proposal key is (-(c_p + c_c1 + c_q), d_pq + 0.15 d_s), the contrast read with the submission's
    own _fi_node over a three-frame cache (proposals visited in frame order), -1 when unreadable --
    forkprop's fp_contrast;
  * GATE_NAME fork_gate_prop.pkl (kernel source biohub-fk44prop), the provenance asserts on its n_train
    and on the patch it was fitted under (rank / cap / diverge / the DIV_* constants);
  * the patch prints proposals / added / crops per movie.
Control = submit_img (ref 56291359, LB 0.951).
"""
import os

src = open("submit_img.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-17a-submit-img"', 'SCRIPT_VERSION = "2026-09-17b-submit-prop"')
rep('print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)',
    'print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)\n'
    'print("  submit_prop: submit_img + forkprop\'s orphan patch (no cap, DIV_DIVERGE_UM 1.5, proposals ranked by\\n"\n'
    '      "  image contrast) under the joint image gate fitted on that patch\'s own forks in both domains\\n"\n'
    '      "  (fork_gate_prop.pkl, fk44prop). forkprop, 128 6bba movies, official scorer: 61/36/64 against\\n"\n'
    '      "  55/42/70, score +0.0050 [+0.0012, +0.0089] over REF + jointimg (LB 0.951), adj_J unchanged.", flush=True)')

# ---- the gate ------------------------------------------------------------------------------------
rep('GATE_NAME = "fork_gate_jointimg.pkl"', 'GATE_NAME = "fork_gate_prop.pkl"')
rep('''assert _gk is not None, (f"{GATE_NAME} not found under /kaggle/input -- attach "
                         f"faustine649/biohub-gatefit as a kernel source")''',
    '''assert _gk is not None, (f"{GATE_NAME} not found under /kaggle/input -- attach "
                         f"faustine649/biohub-fk44prop as a kernel source")''')
rep('''assert _nt.get("6bba") == 292 and _nt.get("44b6", 0) >= 40 and _nt.get("tp", 0) >= 56, (
    f"fork_gate_jointimg.pkl was trained on {_nt}, not forkimg's 292 counted 6bba forks plus "
    f"fk44img's 44b6 ones -- this is not the gate forkimg measured")''',
    '''assert _nt.get("6bba") == 725 and _nt.get("44b6", 0) >= 40 and _nt.get("tp", 0) >= 70, (
    f"fork_gate_prop.pkl was trained on {_nt}, not forkprop's 725 counted 6bba forks of the CR_NOCAP_DV150 "
    f"patch plus fk44prop's 44b6 ones -- this is not the gate forkprop / fk44prop measured")''')
rep('''print("  (divbench8, 2026-09-08: this gate against the shipped 6bba-only one, on the "
      "128 6bba train movies with real pruning and the official scorer -- 46/39/79 "
      "against 42/47/83, div_J 0.2805 against 0.2442, adj_J 0.9143 both, score 0.9424 "
      "against 0.9387, both folds up, paired 90% [+0.0016, +0.0060]. Controls: the "
      "same 67 rows label-shuffled -0.0030, 67 bootstrapped 6bba rows -0.0017.)",
      flush=True)''',
    '''print("  (forkprop, 2026-09-17: this patch + a gate fitted out of fold on its own 6bba forks against "
      "REF + jointimg on the 128 6bba train movies, real pruning, official scorer -- 61/36/64 against "
      "55/42/70, div_J 0.3789 against 0.3293, adj_J 0.9155 both, score 0.9534 against 0.9484, paired "
      "90% [+0.0012, +0.0089]. The gate shipped here is fk44prop's: the same patch's counted forks of "
      "both domains, all movies.)", flush=True)''')

# ---- the patch knobs -----------------------------------------------------------------------------
rep('DIV_DIVERGE_UM = 2.25\n', 'DIV_DIVERGE_UM = 1.5      # submit_img: 2.25. forkprop: 1.5 with no cap, under the image gate\n')
rep('DIV_GLOBAL_CAP = 0.00375\n',
    'DIV_GLOBAL_CAP = 0.00375\n'
    'FP_RANK = "contrast"      # forkprop: proposals by -(c_p + c_c1 + c_q), then d_pq + 0.15 d_s; submit_img: distance only\n'
    'FP_CAP = 100.0            # forkprop: x100 on both caps of the orphan pass = no proposal refused by a cap\n'
    'FP_STAT = defaultdict(int)\n'
    '_gp = _gb.get("patch") or {}\n'
    'assert (_gp.get("rank") == FP_RANK and float(_gp.get("cap", -1)) == FP_CAP\n'
    '        and float(_gp.get("diverge", -1)) == DIV_DIVERGE_UM and float(_gp.get("DIV_W_SISTER", -1)) == DIV_W_SISTER\n'
    '        and float(_gp.get("DIV_FRAME_CAP", -1)) == DIV_FRAME_CAP and float(_gp.get("DIV_GLOBAL_CAP", -1)) == DIV_GLOBAL_CAP\n'
    '        and float(_gp.get("DIV_CHILD_UM", -1)) == DIV_CHILD_UM and float(_gp.get("DIV_PARENT_UM", -1)) == DIV_PARENT_UM\n'
    '        and float(_gp.get("DIV_SISTER_UM", -1)) == DIV_SISTER_UM), (\n'
    '    f"{GATE_NAME} was fitted under another patch: {_gp} against rank {FP_RANK}, cap {FP_CAP}, diverge {DIV_DIVERGE_UM}")\n'
    'print(f"  patch: rank {FP_RANK}, caps x{FP_CAP:g}, DIV_DIVERGE_UM {DIV_DIVERGE_UM}; the gate was fitted under "\n'
    '      f"{_gp.get(\'arm\')} (rank {_gp.get(\'rank\')}, caps x{float(_gp.get(\'cap\', 0)):g}, diverge {_gp.get(\'diverge\')})", flush=True)\n')

# ---- the orphan pass -----------------------------------------------------------------------------
rep('def _add_safe_divisions(nodes, edges):\n', '''def _contrast(fc, nodes, g, memo):
    """the image contrast of one node (forkprop's fp_contrast): -1 without an image or when the crop is unreadable."""
    if g in memo:
        return memo[g]
    con = -1.0
    if fc is None:
        FP_STAT["no_img"] += 1
    else:
        FP_STAT["crops"] += 1
        try:
            _pk, c_, _rg = _fi_node(fc.get(int(nodes[g]["t"])), nodes[g]["xyz"])
            con = float(c_) if np.isfinite(c_) else -1.0
        except Exception:                                          # noqa: BLE001
            FP_STAT["contrast_fail"] += 1
    memo[g] = con
    return con


def _add_safe_divisions(nodes, edges, fc=None):
''')
rep('''            proposals.append((d_pq + DIV_W_SISTER * d_s, t, p, q, c1, d_pq, d_s, dv_))
    proposals.sort()
    used_p = set()''',
    '''            proposals.append((t, p, q, c1, d_pq, d_s, dv_))
    proposals.sort(key=lambda r_: r_[0])          # frame order: the frame cache holds three frames
    memo = {}
    keyed = []
    for t, p, q, c1, d_pq, d_s, dv_ in proposals:
        if FP_RANK == "contrast":
            key_ = (-(_contrast(fc, nodes, p, memo) + _contrast(fc, nodes, c1, memo) + _contrast(fc, nodes, q, memo)),
                    d_pq + DIV_W_SISTER * d_s)
        else:
            key_ = (d_pq + DIV_W_SISTER * d_s,)
        keyed.append((key_, t, p, q, c1, d_pq, d_s, dv_))
    proposals = sorted(keyed)
    FP_STAT["proposals"] += len(proposals)
    used_p = set()''')
rep('''    max_global = max(1, int(DIV_GLOBAL_CAP * len(nodes)))
    added = 0''',
    '''    max_global = max(1, int(DIV_GLOBAL_CAP * FP_CAP * len(nodes)))
    added = 0''')
rep('''        if per_frame[t] >= max(1, int(DIV_FRAME_CAP * n_frame[t])):
            continue''',
    '''        if per_frame[t] >= max(1, int(DIV_FRAME_CAP * FP_CAP * n_frame[t])):
            continue''')

# ---- the call site -------------------------------------------------------------------------------
rep('''    FORK_LOG.clear()
    _add_safe_divisions(nodes, edges)
    n_before = len(edges)
''', '''    FORK_LOG.clear()
    _fc = _FrameCache(zp, meta, T)
    _n0 = len(edges)
    _add_safe_divisions(nodes, edges, _fc)
    print(f"    patch: proposals={FP_STAT['proposals']} forks added={len(edges) - _n0} "
          f"contrast crops={FP_STAT['crops']} unreadable={FP_STAT['contrast_fail']} "
          f"(frame reads {_fc.reads})", flush=True)
    FP_STAT.clear()
    del _fc
    n_before = len(edges)
''')

assert src.count("_contrast(fc, nodes, ") == 4 and src.count("FP_CAP *") == 2
assert src.index("def _contrast(") < src.index("def _add_safe_divisions(") < src.index("class _FrameCache:")
assert src.index("_gb = pickle.load(_f)") < src.index('_gp = _gb.get("patch")')
assert src.index("def _fi_node(") < src.index("def track_movie(")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "submit_prop.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written submit_prop.py", len(src))
