"""submit_prop.py -> submit_vt.py: the vote candidates as attach-only primaries (vtattach XB + zhattach's f4 gate).

WHY (2026-09-20). mapvote: local maxima of the four single heatmaps that agree within 2.5 um but are not
shipped detections recover 31% of the annotated cells the shipped pass misses (in sample) and 40% out of
sample (zhvote, Zebrahub). Linked freely they cost more nodes than they earn (vttrace adj_J -0.0025).
vtattach XB: a candidate may only attach to shipped tracks (candidate-to-candidate ILP pairs withheld) and
is dropped when the solve leaves it with <= 1 edge; in sample adj_J +0.0006 [-0.0001, +0.0013], edge_J
+0.0009 [+0.0002, +0.0016], 8,000 nodes on 2.1M. zhattach, out of sample on 41 Zebrahub crops scored on the
GT thinned to the competition's 5.5% node share: XB adj_J +0.0021 [+0.0011, +0.0032], XB + f4 >= 0.10
+0.0023 [+0.0013, +0.0034]; the gain tracks the detector's miss rate and the f4 gate costs nothing.
Read on the LB as a direction: a negative says the node term is charged for these nodes; a positive
confirms the device and says nothing about the node term.

WHAT CHANGES against submit_prop.py (everything else byte-identical):
  * detect() also runs mapvote's vote on the four maps it already computes: single-map maxima at 0.05
    (hpara-shifted), clusters of >= 2 distinct maps within 2.5 um, none within 7 um of a shipped primary,
    NMS 4 um among the new ones, f4 (the mean map) >= 0.10; admitted candidates are appended to the frame's
    detections at SCORE_OF_K (0.17 / 0.18 / 0.19), i.e. as primaries the head and the ranker see;
  * link_ilp withholds candidate-to-candidate pairs from the ILP (VT_ATTACH_ONLY);
  * track_movie drops every candidate left with <= 1 edge after the solve (VT_DROP_DANGLING) and hands
    _gap_close max(nodes) + 1 as the first free id (len + 1 collides after a drop: vtattach v2);
  * a wall-clock guard: after VT_DEADLINE_S from script start the remaining movies run without candidates
    (announced), so the port can never turn the submission into a timeout;
  * per-movie `vt:` line and VT TOTALS at the end.
Control = submit_prop (ref 56344592, LB 0.954).
"""
src = open("submit_prop.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-17b-submit-prop"', 'SCRIPT_VERSION = "2026-09-20a-submit-vt"')
rep('print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)',
    'print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)\n'
    'print("  submit_vt: submit_prop + the vote candidates as ATTACH-ONLY primaries (vtattach XB, zhattach XBF):\\n"\n'
    '      "  maxima of >= 2 of the four single heatmaps within 2.5 um, none within 7 um of a shipped detection, f4 >= 0.10,\\n"\n'
    '      "  scored 0.17-0.19; the ILP gets no candidate-to-candidate pair; a candidate with <= 1 edge after the solve is\\n"\n'
    '      "  dropped. In sample adj_J +0.0006 [-0.0001, +0.0013]; Zebrahub, GT thinned to 5.5%: +0.0023 [+0.0013, +0.0034].",\n'
    '      flush=True)')

# ---- constants ------------------------------------------------------------------------------------
rep("ILP_DIV_W = 0.55   # set from the fine sweep's argmax\n",
    "ILP_DIV_W = 0.55   # set from the fine sweep's argmax\n"
    "\n"
    "# ---- VT: vote candidates (mapvote -> vtattach XB -> zhattach XBF) ----------------------------------\n"
    "# Local maxima of the four single maps at VOTE_FLOOR, hpara-shifted; maxima of >= MIN_VOTES distinct maps\n"
    "# within AGREE_UM are one candidate; a candidate within RESCUE_UM of a shipped primary is not new; the\n"
    "# shipped NMS radius among the new ones; f4 (the mean map) >= ADMIT_F4. Candidates enter the frame's\n"
    "# detections as primaries scored SCORE_OF_K (above UNET_THRESH: the head and the ranker see them), the\n"
    "# ILP is denied candidate-to-candidate pairs, and a candidate left with <= 1 edge after the solve is\n"
    "# dropped before the post-link patches. After VT_DEADLINE_S of wall clock the remaining movies run\n"
    "# without candidates, so this port cannot turn the submission into a timeout.\n"
    "VOTE_FLOOR = 0.05\n"
    "AGREE_UM = 2.5\n"
    "RESCUE_UM = 7.0\n"
    "NEW_NMS_UM = NMS_UM\n"
    "MIN_VOTES = 2\n"
    "SCORE_OF_K = {2: 0.17, 3: 0.18, 4: 0.19}\n"
    "ADMIT_F4 = 0.10\n"
    "VT_ATTACH_ONLY = True\n"
    "VT_DROP_DANGLING = True\n"
    "VT_DEADLINE_S = 10.0 * 3600\n"
    "T_START = time.time()\n"
    "VOTE_ON = True\n"
    "PSCALE_ARR = np.array([SCALE[0], SCALE[1] * POOL, SCALE[2] * POOL])\n"
    "NNEW = []                        # per frame of the current movie: appended candidates (the LAST seeds of the frame)\n"
    "NSEED = {}                       # per frame: how many primaries seeds_of() yields (set in link_ilp)\n"
    "VT_STAT = defaultdict(int)       # per movie\n"
    "VT_TOT = defaultdict(int)        # over the run\n")

# ---- the vote, before detect() --------------------------------------------------------------------
rep("def detect(vol):\n",
    "def _vt_unpool(zyx):\n"
    "    c = np.asarray(zyx, float).reshape(-1, 3).copy()\n"
    "    c[:, 1] = c[:, 1] * POOL + (POOL - 1) / 2\n"
    "    c[:, 2] = c[:, 2] * POOL + (POOL - 1) / 2\n"
    "    return c\n"
    "\n"
    "\n"
    "def _vt_single_maxima(m):\n"
    '    """(pooled sub-voxel coords, values) of every local maximum of one map at VOTE_FLOOR, hpara-shifted."""\n'
    "    pk = peak_local_max(m, min_distance=1, threshold_abs=VOTE_FLOOR, exclude_border=False)\n"
    "    if len(pk) == 0:\n"
    "        return np.zeros((0, 3)), np.zeros(0)\n"
    "    v = m[pk[:, 0], pk[:, 1], pk[:, 2]].astype(float)\n"
    "    return np.asarray(_hpara_shift(m, pk), float).reshape(-1, 3), v\n"
    "\n"
    "\n"
    "def vt_vote_candidates(hs, f4):\n"
    '    """mapvote verbatim: maxima of distinct maps within AGREE_UM of each other, greedy from the highest\n'
    "    value, nearest per map. Returns (pooled mean coords (n, 3), votes (n,), mean member value (n,), f4 at\n"
    '    the mean (n,)) for the clusters with >= MIN_VOTES maps."""\n'
    "    P, V, L = [], [], []\n"
    "    for i, m in enumerate(hs):\n"
    "        p, v = _vt_single_maxima(m)\n"
    "        P.append(p)\n"
    "        V.append(v)\n"
    "        L.append(np.full(len(v), i, int))\n"
    "    P, V, L = np.concatenate(P), np.concatenate(V), np.concatenate(L)\n"
    "    empty = (np.zeros((0, 3)), np.zeros(0, int), np.zeros(0), np.zeros(0))\n"
    "    if len(P) == 0:\n"
    "        return empty\n"
    "    Pu = P * PSCALE_ARR\n"
    "    tree = cKDTree(Pu)\n"
    "    claimed = np.zeros(len(P), bool)\n"
    "    cc, kk, vv = [], [], []\n"
    "    for i in np.argsort(-V, kind=\"stable\"):\n"
    "        if claimed[i]:\n"
    "            continue\n"
    "        near = [j for j in tree.query_ball_point(Pu[i], AGREE_UM) if j != i and not claimed[j] and L[j] != L[i]]\n"
    "        best = {}\n"
    "        for j in near:\n"
    "            d = float(np.linalg.norm(Pu[j] - Pu[i]))\n"
    "            if L[j] not in best or d < best[L[j]][0]:\n"
    "                best[L[j]] = (d, j)\n"
    "        idx = [i] + [j for _d, j in best.values()]\n"
    "        claimed[idx] = True\n"
    "        if len(idx) >= MIN_VOTES:\n"
    "            cc.append(P[idx].mean(0))\n"
    "            kk.append(len(idx))\n"
    "            vv.append(float(V[idx].mean()))\n"
    "    if not cc:\n"
    "        return empty\n"
    "    C = np.asarray(cc, float)\n"
    "    Z, Y, X = f4.shape\n"
    "    ci = np.clip(np.round(C).astype(int), 0, [Z - 1, Y - 1, X - 1])\n"
    "    ff = f4[ci[:, 0], ci[:, 1], ci[:, 2]].astype(float)\n"
    "    return C, np.asarray(kk, int), np.asarray(vv, float), ff\n"
    "\n"
    "\n"
    "def vt_select_new(refined_vox, k, v, f, prim_um):\n"
    '    """mapvote verbatim: keep the candidates with no shipped primary within RESCUE_UM, then the shipped NMS\n'
    '    among them (votes first, then value). Returns the kept index array."""\n'
    "    refined_vox = np.asarray(refined_vox, float).reshape(-1, 3)\n"
    "    if len(refined_vox) == 0:\n"
    "        return np.zeros(0, int)\n"
    "    keep = np.ones(len(refined_vox), bool)\n"
    "    if len(prim_um):\n"
    "        d, _ = cKDTree(prim_um).query(refined_vox * SCALE)\n"
    "        keep &= d > RESCUE_UM\n"
    "    idx = np.where(keep)[0]\n"
    "    if len(idx) == 0:\n"
    "        return idx\n"
    "    order_score = k[idx] * 10.0 + v[idx]\n"
    "    kept = np.asarray(_nms_keep(refined_vox[idx], order_score, NEW_NMS_UM))\n"
    "    return idx[kept.astype(bool)] if kept.dtype == bool else idx[kept.astype(int)]\n"
    "\n"
    "\n"
    "def vt_is_new(t, i):\n"
    '    """seed i of frame t is an appended candidate: the candidates are the LAST NNEW[t] seeds of the frame."""\n'
    "    t = int(t)\n"
    "    return bool(NNEW) and NNEW[t] > 0 and int(i) >= NSEED[t] - NNEW[t]\n"
    "\n"
    "\n"
    "def vt_drop_dangling(nodes, edges):\n"
    '    """vtattach ln_drop_dangling: drop every candidate node with <= 1 edge after the solve, with its edge.\n'
    '    Shipped nodes are never dropped. Returns (nodes, edges, n_dropped)."""\n'
    "    deg = defaultdict(int)\n"
    "    for a, b in edges:\n"
    "        deg[a] += 1\n"
    "        deg[b] += 1\n"
    "    drop = set()\n"
    "    for g in nodes:\n"
    "        sd = SEED_OF_NODE.get(g)\n"
    "        if sd is not None and vt_is_new(sd[0], sd[1]) and deg[g] <= 1:\n"
    "            drop.add(g)\n"
    "    nodes2 = {g: v for g, v in nodes.items() if g not in drop}\n"
    "    edges2 = [(a, b) for a, b in edges if a not in drop and b not in drop]\n"
    "    return nodes2, edges2, len(drop)\n"
    "\n"
    "\n"
    "def detect(vol):\n")

# ---- detect(): append the admitted candidates ------------------------------------------------------
rep("    if len(pk) == 0:\n"
    "        return np.zeros((0, 3)), np.zeros(0)\n"
    "    sc = h[pk[:, 0], pk[:, 1], pk[:, 2]].astype(float)\n",
    "    if len(pk) == 0:\n"
    "        NNEW.append(0)\n"
    "        return np.zeros((0, 3)), np.zeros(0)\n"
    "    sc = h[pk[:, 0], pk[:, 1], pk[:, 2]].astype(float)\n")
rep("    keep = _nms_keep(ref, sc, NMS_UM)\n"
    "    return hpa[keep], sc[keep]\n",
    "    keep = _nms_keep(ref, sc, NMS_UM)\n"
    "    coords_out, sc_out = hpa[keep], sc[keep]\n"
    "    n_new = 0\n"
    "    if VOTE_ON:\n"
    "        C, K, Vm, Ff = vt_vote_candidates(hs, h)\n"
    "        VT_STAT[\"clusters\"] += int(len(C))\n"
    "        if len(C):\n"
    "            refined = np.array([_refine(vol, c)[0] for c in _vt_unpool(C)], float).reshape(-1, 3)\n"
    "            prim_um = coords_out[sc_out >= UNET_THRESH] * SCALE\n"
    "            sel = vt_select_new(refined, K, Vm, Ff, prim_um)\n"
    "            VT_STAT[\"new\"] += int(len(sel))\n"
    "            sel = sel[Ff[sel] >= ADMIT_F4]\n"
    "            if len(sel):\n"
    "                cn = refined[sel]\n"
    "                sn = np.asarray([SCORE_OF_K[int(min(kk, 4))] for kk in K[sel]], float)\n"
    "                coords_out = np.concatenate([coords_out, cn])\n"
    "                sc_out = np.concatenate([sc_out, sn])\n"
    "                n_new = int(len(sel))\n"
    "    VT_STAT[\"admitted\"] += n_new\n"
    "    NNEW.append(n_new)\n"
    "    return coords_out, sc_out\n")

# ---- link_ilp: seed counts + the attach-only rule ----------------------------------------------------
rep("    PMAP.clear()\n"
    "    SEED_OF_NODE.clear()\n"
    "    vels = pass1_velocities(dets)\n",
    "    PMAP.clear()\n"
    "    SEED_OF_NODE.clear()\n"
    "    NSEED.clear()\n"
    "    vels = pass1_velocities(dets)\n")
rep("        c, s_ = seeds_of(dets[t])\n"
    "        offset[t] = n_glob\n"
    "        n_glob += len(c)\n",
    "        c, s_ = seeds_of(dets[t])\n"
    "        offset[t] = n_glob\n"
    "        NSEED[t] = len(c)\n"
    "        n_glob += len(c)\n")
rep("        for (i, j), p in zip(pairs, pr):\n"
    "            if p >= ILP_FLOOR:\n"
    "                cand_edges.append((offset[t] + i, offset[t + 1] + j,\n"
    "                                   float(p), 0.0))\n",
    "        for (i, j), p in zip(pairs, pr):\n"
    "            if p >= ILP_FLOOR:\n"
    "                if VT_ATTACH_ONLY and vt_is_new(t, i) and vt_is_new(t + 1, j):\n"
    "                    VT_STAT[\"withheld\"] += 1          # attach only: no candidate-to-candidate pair\n"
    "                    continue\n"
    "                cand_edges.append((offset[t] + i, offset[t + 1] + j,\n"
    "                                   float(p), 0.0))\n")

# ---- track_movie: per-movie reset, the dangling drop, the id fix ----------------------------------
rep("    meta = read_array_meta(zp)\n"
    "    dets = []\n"
    "    for t in range(T):\n"
    "        dets.append(detect(load_volume(zp, t, meta)))\n",
    "    meta = read_array_meta(zp)\n"
    "    NNEW.clear()\n"
    "    VT_STAT.clear()\n"
    "    dets = []\n"
    "    for t in range(T):\n"
    "        dets.append(detect(load_volume(zp, t, meta)))\n")
rep("    del head\n"
    "    _gap_close(nodes, edges, cand, cand_sc, cand_trees, len(nodes) + 1)\n",
    "    del head\n"
    "    n_solved_new = sum(1 for g in nodes if SEED_OF_NODE.get(g) is not None and vt_is_new(*SEED_OF_NODE[g]))\n"
    "    n_dg = 0\n"
    "    if VT_DROP_DANGLING and any(NNEW):\n"
    "        nodes, edges, n_dg = vt_drop_dangling(nodes, edges)\n"
    "    print(f\"    vt: on={VOTE_ON} clusters={VT_STAT['clusters']} new={VT_STAT['new']} admitted={VT_STAT['admitted']} \"\n"
    "          f\"withheld={VT_STAT['withheld']} solved={n_solved_new} dangling_dropped={n_dg} kept={n_solved_new - n_dg}\",\n"
    "          flush=True)\n"
    "    for k_, v_ in VT_STAT.items():\n"
    "        VT_TOT[k_] += v_\n"
    "    VT_TOT[\"solved\"] += n_solved_new\n"
    "    VT_TOT[\"dangling_dropped\"] += n_dg\n"
    "    VT_TOT[\"kept\"] += n_solved_new - n_dg\n"
    "    # past the LARGEST id: after a node drop len + 1 would collide with a survivor (vtattach v2)\n"
    "    _gap_close(nodes, edges, cand, cand_sc, cand_trees, (max(nodes) if nodes else 0) + 1)\n")

# ---- main loop: the wall-clock guard; totals ----------------------------------------------------------
rep("    t0 = time.time()\n"
    "    (nrows, erows), nodes, edges = track_movie(zp, ds, T)\n",
    "    t0 = time.time()\n"
    "    if VOTE_ON and time.time() - T_START > VT_DEADLINE_S:\n"
    "        VOTE_ON = False\n"
    "        print(f\"  VT DEADLINE: {(time.time() - T_START) / 3600:.2f} h elapsed; {ds} and the remaining movies run \"\n"
    "              f\"WITHOUT candidates\", flush=True)\n"
    "    (nrows, erows), nodes, edges = track_movie(zp, ds, T)\n")
rep('print("SUBMIT RANKER DONE", flush=True)',
    'print("VT TOTALS: " + " ".join(f"{k}={v}" for k, v in sorted(VT_TOT.items())) + f" | vote_on_at_end={VOTE_ON} "\n'
    '      f"elapsed={(time.time() - T_START) / 3600:.2f} h", flush=True)\n'
    'print("SUBMIT RANKER DONE", flush=True)')

assert "len(nodes) + 1)" not in src.split("def track_movie")[1].split("parts = []")[0], "the old gap-close id survives"
open("submit_vt.py", "w", encoding="utf-8", newline="\n").write(src)
print("written submit_vt.py", src.count("\n"), "lines")
