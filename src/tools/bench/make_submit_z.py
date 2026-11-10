"""submit_sz.py -> submit_z.py: the shipped submission with a DIVISION VARIABLE PER SCORED TRIPLE in the node-cost ILP.

WHY (2026-09-25). hypz / hypz3: giving the shipped ILP one binary per pooled division triple, priced by a triple scorer
(cost 0.55 - 0.10 per logit above -9, floor 0.10) with the triple's two edges admitted below the 0.48 candidate floor
as division-only candidates, reads +0.002 over the shipped composition after the gate on the 128 6bba movies (Z10:
55-56 true divisions against 50 of 125), the patches remove none of the z forks and the refitted gate cuts 2-3 true
ones. The user (2026-09-25): "不缺就搭吧". The hidden test has no triple table, so the submission computes the triples
itself with divscore.pkl's two scorers (biohub-divscore) and takes its gate from fk44prop_z (fork_gate_z.pkl /
fork_gate_zx.pkl, fitted on the z composition's forks in both domains).

WHAT. submit_sz.py verbatim (node cost 0.35, fork tracks exempt, the uncapped contrast-ranked orphan patch, the 36-column
image + size gate at p > 0.03) with:
  * divscore.pkl loaded and checked (feature lists, top-k, S0, the geometry and image constants);
  * divtrip's triple geometry (triples_frame) and raw-image node features (node_img) verbatim; per movie, before the
    node-cost solve: every triple of the ILP's own seeds (a, b within 14 um of p, sisters 2-16 um, midpoint within
    8 um), scorer A on all of them, the top 20 per frame, image features from the submission's frame cache, scorer B,
    the cost per triple -- about 30 s per movie, on every movie (no wall-clock skip: the user wants a timeout to fail
    the run, not fall back silently);
  * nc_solve(trip=...): the z variables (sum_out y + d = x + v + sum z; v + sum z <= x; z <= y_pa, z <= y_pb; a
    below-floor edge usable only through its z), the objective + sum w z; optionally the two-pass shape (Z_TWO_PASS:
    a first solve without the node cost, its z-fork daughters + chains added to the exemption set);
  * the gate fork_gate_z.pkl (or fork_gate_zx.pkl), its provenance asserted against Z_CFG / S0 / top-k / two_pass.
Attach: submit_sz's sources with biohub-fk44prop-sz -> biohub-fk44prop-z, + biohub-divscore. GPU T4.
"""
import sys

TWO_PASS = "--two-pass" in sys.argv          # hypz4's verdict decides; default the single solve (hypz's Z10)
SIZE_GATE = "--size-gate" in sys.argv        # default: the 31-column gate (submit_nc35's set, LB 0.956); the size gate read 0.954
NO_FALLBACK = "--no-fallback" in sys.argv    # the user (2026-09-25): once a z submission ran without a timeout, remove every fallback and submit again
src = open("submit_sz.py", encoding="utf-8").read()
dt = open("divtrip.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


def dslice(a, b):
    i = dt.index(a)
    return dt[i:dt.index(b, i)]


GEOM = dslice("def triples_frame(", "def label_triples(")
NODE_IMG = dslice("def _crop(", "def _hgb(")
assert "def triples_frame(" in GEOM and "def node_img(" in NODE_IMG and "_crop(" in NODE_IMG

rep('SCRIPT_VERSION = "2026-09-24a-submit-sz"', f'SCRIPT_VERSION = "{"2026-09-27a" if NO_FALLBACK else "2026-09-25b"}-submit-z{"x" if TWO_PASS else ""}{"sz" if SIZE_GATE else ""}{"nf" if NO_FALLBACK else ""}"')
rep('GATE_NAME = "fork_gate_sz.pkl"', f'GATE_NAME = "fork_gate_{"zx" if TWO_PASS else "z"}{"" if SIZE_GATE else "31"}.pkl"')
if not SIZE_GATE:
    # the gate's column set = submit_nc35's 31 (the five size columns are computed and ignored)
    rep('              "r_hm_p", "r_hm_c1", "r_hm_q", "rhm_max", "rhm_sum"]      # submit_sz: raw-image half-max size\n',
        '              "r_hm_p", "r_hm_c1", "r_hm_q", "rhm_max", "rhm_sum"]      # submit_sz: raw-image half-max size\n'
        'GATE_FEATS_ALL = list(GATE_FEATS)\n'
        'GATE_FEATS = GATE_FEATS[:31]      # submit_z: the 31-column gate (submit_nc35\'s set, LB 0.956; the size gate read 0.954)\n')
    rep('FEAT_IMG = GATE_FEATS[20:]\n', 'FEAT_IMG = GATE_FEATS_ALL[20:]     # all 16 image columns are computed; the gate reads its 11\n')
    rep('assert "r_hm_p" in _gb["feature_names"] and len(_gb["feature_names"]) == 36, _gb["feature_names"]\n',
        'assert "r_hm_p" not in _gb["feature_names"] and len(_gb["feature_names"]) == 31, _gb["feature_names"]\n')
rep('''                         f"faustine649/biohub-fk44prop-sz as a kernel source")''',
    '''                         f"faustine649/biohub-fk44prop-z as a kernel source")''')
rep('''assert 600 <= _nt.get("6bba", 0) <= 900 and _nt.get("44b6", 0) >= 40 and _nt.get("tp", 0) >= 70, (
    f"fork_gate_sz.pkl was trained on {_nt}, not forkprop_sz's counted 6bba forks of the CR_NOCAP_DV150 "
    f"patch plus fk44prop_sz's 44b6 ones -- this is not the gate forkprop_sz / fk44prop_sz measured")''',
    '''assert 600 <= _nt.get("6bba", 0) <= 2500 and _nt.get("44b6", 0) >= 40 and _nt.get("tp", 0) >= 70, (
    f"{GATE_NAME} was trained on {_nt}, not forkprop_z's counted 6bba forks of the z composition under the "
    f"CR_NOCAP_DV150 patch plus fk44prop_z's 44b6 ones -- this is not the gate forkprop_z / fk44prop_z measured")''')

# ---- the triple scorers, the geometry, the image features, the candidate stage ------------------------------------
Z_BLOCK = '''
# ---- submit_z: the division-variable ILP's inputs ----------------------------------------------------------------
Z_TWO_PASS = %s
Z_CFG = dict(cost=0.35, lam=0.10, wmin=0.10)      # hypz's Z10: w = clip(DIV_W - lam (logit - S0), wmin, DIV_W)
Z_TOT = defaultdict(int)
DIVSCORE_NAME = "divscore.pkl"
_dk = None
for _root, _dirs, _files in _os.walk("/kaggle/input"):
    if DIVSCORE_NAME in _files:
        _dk = _os.path.join(_root, DIVSCORE_NAME)
        break
assert _dk is not None, f"{DIVSCORE_NAME} not found under /kaggle/input -- attach faustine649/biohub-divscore as a kernel source"
with open(_dk, "rb") as _f:
    _ds = pickle.load(_f)
GEO = list(_ds["geo"])
IMG = list(_ds["img"])
SCORER_A, SCORER_B = _ds["A"], _ds["B"]
Z_S0 = float(_ds["s0"])
Z_TOPK = int(_ds["topk"])
_dg = _ds["geometry"]
R_PARENT, R_SISTER, R_MID, D_AB_MIN, SNAP_UM = (float(_dg[k_]) for k_ in ("R_PARENT", "R_SISTER", "R_MID", "D_AB_MIN", "SNAP_UM"))
assert float(_dg["SCORE_MIN"]) == UNET_THRESH, (_dg, UNET_THRESH)
_di = _ds["image"]
assert (float(_di["half_um"]), float(_di["peak_um"]), float(_di["size_um"])) == (FI_HALF_UM, FI_PEAK_UM, FI_SIZE_UM) \\
    and tuple(float(v) for v in _di["sigma_vox"]) == FI_SIGMA_VOX, f"divscore's image constants differ: {_di}"
assert GEO == ["d_pa", "d_pb", "d_ab", "asym", "mid_off", "ang", "dz_pa", "dz_pb", "dz_ab", "t_frac", "has_m", "speed_m",
               "cont_a", "cont_b", "cont_min", "cont_mid", "has_a2", "has_b2", "sep_grow", "sc_p", "sc_a", "sc_b", "sc_min",
               "dens_p10", "n_t1_12", "orph_a", "orph_b", "orph_min", "orph_max"], GEO
assert IMG == ["i_p", "c_p", "r_p", "hm_p", "c_a", "c_b", "hm_a", "hm_b", "hm_max_ratio", "hm_sum_ratio", "i_ratio", "br_p"], IMG
assert Z_TOPK == 20 and Z_S0 == -9.0 and hasattr(SCORER_A, "predict_proba") and hasattr(SCORER_B, "predict_proba")
_gz = _gb.get("z") or {}
assert _gz and _gz.get("src") == ("ZX" if Z_TWO_PASS else "Z") and bool(_gz.get("two_pass")) == Z_TWO_PASS \\
    and float(_gz.get("cost", -1)) == Z_CFG["cost"] == NC_COST and float(_gz.get("lam", -1)) == Z_CFG["lam"] \\
    and float(_gz.get("wmin", -1)) == Z_CFG["wmin"] and float(_gz.get("s0", 1)) == Z_S0 and int(_gz.get("topk", 0)) == Z_TOPK, (
    f"{GATE_NAME} was fitted under another z composition: {_gz} against {Z_CFG}, S0 {Z_S0}, top-k {Z_TOPK}, two-pass {Z_TWO_PASS}")
print(f"divscore {DIVSCORE_NAME} from {_dk} | version {_ds.get('version')} | n_train {_ds.get('n_train')} | out-of-fold AUC "
      f"{_ds.get('auc_oof')} | top-k {Z_TOPK} | S0 {Z_S0} | z {Z_CFG} two-pass {Z_TWO_PASS} | gate's z provenance {_gz}", flush=True)


def z_cost(s):
    w = DIV_W_ZBASE - Z_CFG["lam"] * (float(s) - Z_S0)
    return float(min(DIV_W_ZBASE, max(Z_CFG["wmin"], w)))


def _zlogit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


''' % ("True" if TWO_PASS else "False") + GEOM + NODE_IMG + '''
def triple_candidates(dets, T, fc):
    """the ILP's division triples of one movie: every triple of the seeds (divtrip's geometry), scorer A, the top Z_TOPK
    per frame, raw-image features from the frame cache, scorer B -> [(t, ip, ia, ib, w, s, 0)] in the seed index space
    link_ilp / nc_solve use; and the stage's counts."""
    seeds = [seeds_of(dets[t]) for t in range(T)]
    um = {t: np.asarray(seeds[t][0], float) * SCALE for t in range(T)}
    out, st = [], defaultdict(int)
    for t in range(T - 1):
        P, S = um[t], seeds[t][1]
        Pn, Sn = um[t + 1], seeds[t + 1][1]
        idx, X = triples_frame(P, S, Pn, Sn, um[t - 1] if t >= 1 else None, um[t + 2] if t + 2 < T else None, t, T)
        if not len(idx):
            continue
        st["triples"] += len(idx)
        pA = SCORER_A.predict_proba(X)[:, 1]
        keep = np.argsort(-pA)[:Z_TOPK]
        F_t, F_t1 = fc.get(t), fc.get(t + 1)
        F_tm = fc.get(t - 1) if t >= 1 else None
        cache = {}

        def nf(fr, coords_vox, key):
            if key not in cache:
                cache[key] = node_img(fr, coords_vox) if fr is not None else (np.nan,) * 4
            return cache[key]

        Pv, Pnv = np.asarray(seeds[t][0], float), np.asarray(seeds[t + 1][0], float)
        rows, meta = [], []
        for r in keep:
            pi_, ai_, bi_ = (int(v) for v in idx[r])
            i_p, c_p, r_p, hm_p = nf(F_t, Pv[pi_], ("p", pi_))
            i_a, c_a, _r, hm_a = nf(F_t1, Pnv[ai_], ("n", ai_))
            i_b, c_b, _r, hm_b = nf(F_t1, Pnv[bi_], ("n", bi_))
            br_p = np.nan
            if F_tm is not None and X[r, GEO.index("has_m")] == 1:
                dm, jm = cKDTree(um[t - 1]).query(P[pi_])
                if dm <= SNAP_UM:
                    i_pm = nf(F_tm, np.asarray(seeds[t - 1][0], float)[jm], ("m", int(jm)))[0]
                    br_p = i_p / i_pm if np.isfinite(i_pm) and i_pm > 0 else np.nan
            okh = np.isfinite(hm_p) and np.isfinite(hm_a) and np.isfinite(hm_b) and hm_p > 0
            imgf = [i_p, c_p, r_p, hm_p, c_a, c_b, hm_a, hm_b,
                    (max(hm_a, hm_b) / hm_p) if okh else np.nan, ((hm_a + hm_b) / hm_p) if okh else np.nan,
                    ((i_a + i_b) / (2 * i_p)) if (np.isfinite(i_a) and np.isfinite(i_b) and np.isfinite(i_p) and i_p > 0) else np.nan,
                    br_p]
            rows.append(np.concatenate([X[r], np.asarray(imgf, float)]))
            meta.append((t, pi_, ai_, bi_))
        if not rows:
            continue
        sB = _zlogit(SCORER_B.predict_proba(np.asarray(rows, float))[:, 1])
        st["image_nodes"] += len(cache)
        for (t_, pi_, ai_, bi_), s_ in zip(meta, sB.tolist()):
            w_ = z_cost(s_)
            st["pool"] += 1
            st["discounted"] += int(w_ < DIV_W_ZBASE - 1e-9)
            out.append((t_, pi_, ai_, bi_, w_, float(s_), 0))
    return out, dict(st)


'''
rep("NC_DEGRADED = []\n", "NC_DEGRADED = []\nDIV_W_ZBASE = 0.55                # the z cost's ceiling = the plain second-edge toll (ILP_DIV_W)\n" + Z_BLOCK)

# ---- the gate-failure fallback removed: a gate that throws on a movie fails the run instead of keeping every fork
# of that movie silently (the user, 2026-09-25: "先把兜底删了") ------------------------------------------------------
rep("""    try:
        edges, (n_forks, n_drop) = _gate_prune(nodes, edges, dets, zp, meta)
    except Exception as _e:                      # noqa: BLE001
        print(f"    WARN fork gate failed ({type(_e).__name__}: {_e}) -- forks kept", flush=True)
        n_forks, n_drop = -1, 0
""", """    edges, (n_forks, n_drop) = _gate_prune(nodes, edges, dets, zp, meta)   # submit_z: no fallback -- a gate failure fails the run
""")

# ---- nc_exempt with an explicit seed map (the two-pass exemption of the first solve's z daughters) ---------------------
rep("def nc_exempt(edges0, final_edges):\n", "def nc_exempt(edges0, final_edges, son=None):\n")
rep("    return {SEED_OF_NODE[g] for g in tn if SEED_OF_NODE.get(g) is not None}, len(d), len(tn)\n",
    "    _m = SEED_OF_NODE if son is None else son\n"
    "    return {_m[g] for g in tn if _m.get(g) is not None}, len(d), len(tn)\n")

# ---- nc_solve: the z variables ----------------------------------------------------------------------------------------
rep("def nc_solve(exempt, div_w):\n", "def nc_solve(exempt, div_w, trip=None, node_cost=None):\n")
rep('    coords, cand, offset = NC_CTX["coords"], NC_CTX["cand"], NC_CTX["offset"]\n',
    '    coords, cand, offset = NC_CTX["coords"], list(NC_CTX["cand"]), NC_CTX["offset"]\n')
rep('    m = Model("nc")\n',
    "    # ---- submit_z: one z per scored triple; its two edges as candidates (division-only when below the floor)\n"
    "    zt = []                                  # (gp, ga, gb, w, k_a, k_b)\n"
    "    z_of = defaultdict(list)\n"
    "    div_only = {}\n"
    "    zst = defaultdict(int)\n"
    "    if trip:\n"
    "        kidx = {(gs, gt_): k for k, (gs, gt_, _p, _z) in enumerate(cand)}\n"
    "        for (t_, ip, ia, ib, w_, s_, _l) in trip:\n"
    "            zst['offered'] += 1\n"
    "            if t_ not in offset or t_ + 1 not in offset or ip >= nseed[t_] or max(ia, ib) >= nseed[t_ + 1] or ia == ib:\n"
    "                zst['bad_index'] += 1\n"
    "                continue\n"
    "            pa = PMAP.get((t_, ip, ia))\n"
    "            pb = PMAP.get((t_, ip, ib))\n"
    "            if pa is None or pb is None:\n"
    "                zst['no_pair'] += 1\n"
    "                continue\n"
    "            gp, ga, gb = offset[t_] + ip, offset[t_ + 1] + ia, offset[t_ + 1] + ib\n"
    "            ks = []\n"
    "            for gc_, pc_ in ((ga, float(pa[0])), (gb, float(pb[0]))):\n"
    "                k_ = kidx.get((gp, gc_))\n"
    "                if k_ is None:\n"
    "                    k_ = len(cand)\n"
    "                    cand.append((gp, gc_, pc_, 0.0))\n"
    "                    kidx[(gp, gc_)] = k_\n"
    "                    div_only[k_] = []\n"
    "                    zst['edges_added'] += 1\n"
    "                ks.append(k_)\n"
    "            j_ = len(zt)\n"
    "            for k_ in ks:\n"
    "                if k_ in div_only:\n"
    "                    div_only[k_].append(j_)\n"
    "            zt.append((gp, ga, gb, float(w_), ks[0], ks[1]))\n"
    "            z_of[gp].append(j_)\n"
    "        zst['z_vars'] = len(zt)\n"
    '    m = Model("nc")\n')
rep('    y = [m.addVar(vtype="B", name=f"y{k}") for k in range(len(cand))]\n',
    '    y = [m.addVar(vtype="B", name=f"y{k}") for k in range(len(cand))]\n'
    '    z = [m.addVar(vtype="B", name=f"z{j}") for j in range(len(zt))]\n')
rep("        m.addCons(quicksum(y[k] for k in out_e[g]) + d[g] == x[g] + v[g])\n"
    "        m.addCons(v[g] <= x[g])\n",
    "        m.addCons(quicksum(y[k] for k in out_e[g]) + d[g] == x[g] + v[g] + quicksum(z[j] for j in z_of.get(g, ())))\n"
    "        m.addCons(v[g] + quicksum(z[j] for j in z_of.get(g, ())) <= x[g])\n"
    "    for j, (_gp, _ga, _gb, _w, ka, kb) in enumerate(zt):\n"
    "        m.addCons(z[j] <= y[ka])\n"
    "        m.addCons(z[j] <= y[kb])\n"
    "    for k, js in div_only.items():\n"
    "        m.addCons(y[k] <= quicksum(z[j] for j in js))\n")
rep("        + NC_COST * quicksum(x[g] for g in range(N) if g not in ex)\n",
    "        + (NC_COST if node_cost is None else node_cost) * quicksum(x[g] for g in range(N) if g not in ex) \\\n"
    "        + quicksum(zt[j][3] * z[j] for j in range(len(zt)))\n")
rep("    del m, x, a, d, v, y\n",
    "    zst['z_selected'] = sum(1 for j in range(len(zt)) if m.getVal(z[j]) > 0.5)\n"
    "    n_div += zst['z_selected']\n"
    "    del m, x, a, d, v, y, z\n")
rep("    return nodes, edges, son, dict(exempt=len(ex), selected=n_sel, edges=len(ys), divisions=n_div)\n",
    "    return nodes, edges, son, dict(exempt=len(ex), selected=n_sel, edges=len(ys), divisions=n_div, **{f'z_{k_}': v_ for k_, v_ in zst.items()})\n")

# ---- track_movie: the triples before the node-cost solve; the optional two-pass -------------------------------------
rep("            _fc1 = _FrameCache(zp, meta, T)\n",
    "            _t_tr = time.time()\n"
    "            _fc0 = _FrameCache(zp, meta, T)\n"
    "            _trip, _zst = triple_candidates(dets, T, _fc0)      # no wall-clock skip: the user (2026-09-25) wants a timeout to fail, not a silent fallback\n"
    "            print(f\"    triples: {_zst} (frame reads {_fc0.reads}, {time.time() - _t_tr:.0f}s)\", flush=True)\n"
    "            del _fc0\n"
    "            for k_, v_ in _zst.items():\n"
    "                Z_TOT[k_] += v_\n"
    "            _fc1 = _FrameCache(zp, meta, T)\n")
rep("            _nn, _ee, _son, _st = nc_solve(_ex, ILP_DIV_W)\n",
    "            if Z_TWO_PASS and _trip:\n"
    "                _n0, _e0, _son0, _st0 = nc_solve(set(), ILP_DIV_W, trip=_trip, node_cost=0.0)\n"
    "                _exz, _, _ = nc_exempt(_e0, _e0, son=_son0)\n"
    "                Z_TOT['two_pass_z'] += int(_st0.get('z_z_selected', 0))\n"
    "                Z_TOT['two_pass_exempt'] += len(_exz)\n"
    "                _ex = set(_ex) | set(_exz)\n"
    "                del _n0, _e0, _son0, _st0\n"
    "            _nn, _ee, _son, _st = nc_solve(_ex, ILP_DIV_W, trip=_trip)\n"
    "            Z_TOT['z_selected'] += int(_st.get('z_z_selected', 0))\n"
    "            Z_TOT['z_vars'] += int(_st.get('z_z_vars', 0))\n")
rep('                  f"forks={_st[\'divisions\']} ({time.time() - t_nc:.0f}s)", flush=True)\n',
    '                  f"forks={_st[\'divisions\']} z={_st.get(\'z_z_selected\', 0)}/{_st.get(\'z_z_vars\', 0)} "\n'
    '                  f"(edges added {_st.get(\'z_edges_added\', 0)}, no pair {_st.get(\'z_no_pair\', 0)}) ({time.time() - t_nc:.0f}s)", flush=True)\n')
rep('print("NC TOTALS: " + " ".join(f"{k}={v}" for k, v in sorted(NC_TOT.items())) + f" | degraded {len(NC_DEGRADED)}: "\n',
    'print("Z TOTALS: " + " ".join(f"{k}={v}" for k, v in sorted(Z_TOT.items())) + " | every movie ran the triple stage (no wall-clock skip)", flush=True)\n'
    'print("NC TOTALS: " + " ".join(f"{k}={v}" for k, v in sorted(NC_TOT.items())) + f" | degraded {len(NC_DEGRADED)}: "\n')


if NO_FALLBACK:
    # ---- no fallback anywhere: the node-cost stage runs on every movie (no wall-clock guard), a failure or a non-optimal SCIP
    # ---- fails the run instead of keeping the shipped solution, and a head-pass failure fails the run instead of NaN columns
    rep("NC_TIME_S = 900.0               # SCIP time limit per movie; not optimal -> the shipped solution stays\n",
        "NC_TIME_S = 900.0               # SCIP time limit per movie; not optimal -> the run FAILS (no fallback)\n")
    rep("NC_DEADLINE_S = 10.0 * 3600     # wall clock from here; later movies keep the shipped solution\n", "")
    head = ("    if time.time() - NC_T0 > NC_DEADLINE_S:\n"
            "        NC_DEGRADED.append(ds)\n"
            "        print(f\"    [degraded-nc] {ds}: wall clock {(time.time() - NC_T0) / 3600:.2f} h -- the shipped solution stays\", flush=True)\n"
            "    else:\n"
            "        try:\n")
    tail = ("        except Exception as _e:                      # noqa: BLE001\n"
            "            NC_DEGRADED.append(ds)\n"
            "            print(f\"    [degraded-nc] {ds}: {type(_e).__name__}: {str(_e)[:200]} -- the shipped solution stays\", flush=True)\n")
    assert src.count(head) == 1 and src.count(tail) == 1
    i = src.index(head); j = src.index(tail) + len(tail)
    body = src[i + len(head):src.index(tail)]                  # the try body, indented 12
    body = "".join((ln[4:] if ln.startswith("    ") else ln) for ln in body.splitlines(True))
    src = (src[:i]
           + "    # submit_z*nf: no wall-clock guard, no degrade -- every movie runs the node-cost stage and any failure fails the run\n"
           + "    try:\n" + body
           + "    except Exception as _e:                      # noqa: BLE001\n"
           + "        print(f\"    [FAIL-nc] {ds}: {type(_e).__name__}: {str(_e)[:200]} -- no fallback: the run fails\", flush=True)\n"
           + "        raise\n"
           + src[j:])
    rep("        _HeadRun.uninstall()\n"
        "        HEAD_FAILED.append(ds)\n"
        "        print(f\"[degraded] {ds}: head pass failed ({type(exc).__name__}: \"\n"
        "              f\"{str(exc)[:300]}) -- falling back to the 18-column path is \"\n"
        "              f\"NOT available with this pkl, so this movie is scored with \"\n"
        "              f\"p_fwd=p_rev=h_fused=NaN\", flush=True)\n"
        "        return None\n",
        "        _HeadRun.uninstall()\n"
        "        print(f\"[FAIL] {ds}: head pass failed ({type(exc).__name__}: {str(exc)[:300]}) -- no fallback: the run fails\", flush=True)\n"
        "        raise\n")
    assert "NC_DEADLINE_S" not in src and "the shipped solution stays" not in src and "[degraded-nc]" not in src

assert src.count("def triples_frame(") == 1 and src.count("def node_img(") == 1 and src.count("triple_candidates(") == 2 \
    and src.count("Z_TWO_PASS") >= 4 and src.count("_ndi") >= 3
name = f"submit_{'zx' if TWO_PASS else 'z'}{'sz' if SIZE_GATE else ''}{'nf' if NO_FALLBACK else ''}.py"
open(name, "w", encoding="utf-8", newline="\n").write(src)
print("written", name, src.count("\n"), "lines")
