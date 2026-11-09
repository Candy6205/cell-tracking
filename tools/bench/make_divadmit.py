"""bridgerank.py -> divadmit.py: the two priced division items, on the
round-2 harness, replayed on one solve per movie.

WHY. The 2026-09-11 loss map (rkedge + divmiss + bridgerank v2) leaves two
priced, untested moves on the division side of the shipped arm (REF: 55
TP / 237 FP / 70 FN ungated; 49/51/76 under the joint gate, score 0.9433):

  (a) daughter_absent: 21 GT divisions have a daughter with no node of the
      pre-patch graph within MATCH_UM 5.0. divmiss puts a PRIMARY (s >=
      0.15) within 5 um of 234 of the 250 GT daughters, and 14 daughters
      have no peak at all, so roughly 8 of the 21 lost a daughter the
      DETECTOR had and the ILP (or the short filter) dropped. The orphan
      patch cannot reach them: it draws second daughters from graph nodes
      with no predecessor. The ADMIT pass draws them from the dropped
      primaries instead, as childless daughters -- the scorer's daughter
      lineage is {child, *successors}, so a matched childless child counts.
  (b) 8b_diverge_under_2.25um: 11 GT divisions pass every orphan-patch gate
      except DIV_DIVERGE_UM. The constant was set against the OLD gate; the
      joint gate now prunes forks, so the divergence gate is re-crossed at
      1.5 / 0.75 / 0.0 with the joint gate doing the pruning.

ARMS (post-link only, replayed on the pickled REF solve of each movie):
  DV150, DV075, DV000   DIV_DIVERGE_UM 1.5 / 0.75 / 0.0, no admit
  ADM                   DIV_DIVERGE_UM 2.25 (shipped) + the admit pass
  ADM_DV075             0.75 + admit
Each arm's forks are judged by the official scorer, its gate is trained out
of fold on its own counted forks (the arm changes the fork population), and
the paired bootstrap of arm + joint against REF + joint decides.

ADMIT PASS (inside `_add_safe_divisions`, after the orphan pass, so REF is
untouched by construction; `DA_ADMIT` False is the shipped patch): for
every single-child parent p (same gates as the orphan pass: one
predecessor, child c1 at t+1 within DIV_CHILD_UM), the dropped primaries at
t+1 within DIV_PARENT_UM of p; q must be the dropped primary NEAREST to c1
(the orphan pass's mutual-nearest rule) within DIV_SISTER_UM; q is refused
when it lies within DA_MIN_SEP_UM 3.0 of c1 or of ANY node the graph has at
t+1 (a second detection of a cell the graph already tracks). Ranked by
d_pq + DIV_W_SISTER * d_s, the shared per-frame cap and a global cap of the
orphan pass's size. The admitted node gets the seed's coordinate (rounded,
as solve_one rounds), a fresh id above every existing one, and a
seed_of_node entry so fork_rows finds its score and its p; the entries are
removed after each arm so an id reused by another arm cannot read a stale
seed. No divergence gate (the daughter has no successor; dv is NaN in the
fork table, len_q 0, q_ends 1 -- HistGB takes NaN natively).

REMOVED from bridgerank: the bridge-pair collection in pass 1 (BR_COLLECT
stays False), pass 2a (the bridge model; no zebrahub CSV is attached), the
bridges-per-arm table. `_gap_close` runs as shipped (GAP_SPAN 1, BR_MODEL
None). Anchors unchanged: REF 55/237/70 adj_J 0.9146, REF + joint 0.9433.

OUTPUT: divadmit_REF_F4HP.csv, divadmit_arm_<arm>.csv, forks_<arm>.csv,
steal5a.csv, fork_gate_dw55_20.pkl. Attach: rkedge's 16 sources. CPU,
~3 h (pass 1 ~65 min, pass 2 ~80 min for five arms, cross ~25 min).
"""
import os

src = open("bridgerank.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:120])
    src = src.replace(old, new)


def cut(start, end, new):
    """replace src[start_pat : end_pat) with new (end pattern kept)."""
    global src
    assert src.count(start) == 1, (src.count(start), start[:80])
    i = src.index(start)
    j = src.index(end, i)
    src = src[:i] + new + src[j:]


rep('SCRIPT_VERSION = "2026-09-11b-bridgerank"', 'SCRIPT_VERSION = "2026-09-12a-divadmit"')

# ---- arms -------------------------------------------------------------------
cut("BR_ARMS = [", "DIVW_ARMS = [(a_, DIV_W) for a_, _c in BR_ARMS]",
    '''DA_ARMS = [("DV150", dict(diverge=1.5, admit=False)),
           ("DV075", dict(diverge=0.75, admit=False)),
           ("DV000", dict(diverge=0.0, admit=False)),
           ("ADM", dict(diverge=2.25, admit=True)),
           ("ADM_DV075", dict(diverge=0.75, admit=True))]
BR_ARMS = DA_ARMS         # the replay loop below iterates BR_ARMS_RUN
DA_ADMIT = False          # shipped patch; set per arm in pass 2
DA_MIN_SEP_UM = 3.0       # an admitted daughter this close to an existing node is a duplicate detection
DA_CTX = {}               # seeds, seed_of_node, added ids -- set per arm
''')

# ---- the admit pass, inside _add_safe_divisions, after the orphan pass -------
rep('''    PATCH_CAPS["orphan_added"] += added
    if steal is None:
        return added, 0
''', '''    PATCH_CAPS["orphan_added"] += added
    # ---- divadmit: second daughters from the primaries the graph dropped ----
    if DA_ADMIT and DA_CTX.get("seeds") is not None:
        seeds_da = DA_CTX["seeds"]
        son = DA_CTX["seed_of_node"]
        present = {son[g] for g in nodes if son.get(g) is not None}
        node_tree = {}
        for t, ids in by_t.items():
            node_tree[t] = cKDTree(np.asarray([nodes[g]["xyz"] for g in ids], float) * SCALE)
        drop = {}          # t -> (indices into seeds[t][0], coords in um, tree)
        for t in range(len(seeds_da)):
            c_ = np.asarray(seeds_da[t][0], float).reshape(-1, 3)
            if not len(c_):
                continue
            keep = np.array([i_ for i_ in range(len(c_)) if (t, i_) not in present], int)
            if not len(keep):
                continue
            cu = c_[keep] * SCALE
            drop[t] = (keep, cu, cKDTree(cu))
            PATCH_CAPS["admit_candidates"] += len(keep)
        aprop = []
        for p, n in nodes.items():
            ch = succ.get(p, [])
            if len(ch) != 1 or p in used_p:
                continue
            c1 = ch[0]
            t = n["t"]
            if nodes[c1]["t"] != t + 1 or len(pred.get(p, [])) != 1:
                continue
            if _dist_um(n["xyz"], nodes[c1]["xyz"]) > DIV_CHILD_UM:
                continue
            dr = drop.get(t + 1)
            if dr is None:
                continue
            keep, cu, tr = dr
            pu = np.asarray(n["xyz"], float) * SCALE
            c1u = np.asarray(nodes[c1]["xyz"], float) * SCALE
            hits = tr.query_ball_point(pu, r=DIV_PARENT_UM)
            if not hits:
                continue
            d_near, i_near = tr.query(c1u)
            if d_near > DIV_SISTER_UM or int(i_near) not in hits:
                continue
            hi = int(i_near)
            d_pq = float(np.linalg.norm(cu[hi] - pu))
            d_s = float(np.linalg.norm(cu[hi] - c1u))
            if d_s < DA_MIN_SEP_UM:
                PATCH_CAPS["admit_dup"] += 1
                continue
            d_any, _i = node_tree[t + 1].query(cu[hi])
            if d_any < DA_MIN_SEP_UM:
                PATCH_CAPS["admit_dup"] += 1
                continue
            aprop.append((d_pq + DIV_W_SISTER * d_s, t, p, (t + 1, int(keep[hi])), c1, d_pq, d_s))
        aprop.sort()
        PATCH_CAPS["admit_proposals"] += len(aprop)
        max_admit = max(1, int(DIV_GLOBAL_CAP * len(nodes)))
        nxt = (max(nodes) + 1) if nodes else 1
        used_seed = set()
        n_adm = 0
        for score, t, p, sq, c1, d_pq, d_s in aprop:
            if n_adm >= max_admit:
                PATCH_CAPS["admit_global"] += 1
                break
            if p in used_p or sq in used_seed:
                continue
            if per_frame[t] >= max(1, int(DIV_FRAME_CAP * n_frame[t])):
                PATCH_CAPS["admit_frame"] += 1
                continue
            xyz = np.round(np.asarray(seeds_da[sq[0]][0], float).reshape(-1, 3)[sq[1]]).astype(float)
            q = nxt
            nxt += 1
            nodes[q] = {"t": t + 1, "xyz": xyz}
            son[q] = sq
            DA_CTX["added"].append(q)
            by_t[t + 1].append(q)
            FORK_LOG[(p, q)] = dict(c1=c1, d_pq=d_pq, d_s=d_s, dv=float("nan"),
                                    n_orph=len(drop[t + 1][0]), steal=0, admit=1)
            edges.append((p, q))
            succ[p].append(q)
            pred[q].append(p)
            used_p.add(p)
            used_q.add(q)
            used_seed.add(sq)
            per_frame[t] += 1
            n_adm += 1
        PATCH_CAPS["admit_added"] += n_adm
    if steal is None:
        return added, 0
''')

# ---- pass 1: no bridge collection ------------------------------------------
rep('''    BR_COLLECT = True
    patch_final(nodes0, edges0, pool, outp)
    BR_COLLECT = False
''', '''    patch_final(nodes0, edges0, pool, outp)
''')

# ---- pass 2a: gone -----------------------------------------------------------
cut("# ---- BRIDGERANK PASS 2a: the bridge model, two folds by movie --------------",
    "# ---- BRIDGERANK PASS 2b: the four arms, replayed on the pickled solves -----",
    "BR_FOLD_MODEL = {}\nBR_MODEL_OK = False\n\n")
rep('BR_ARMS_RUN = [(a_, c_) for a_, c_ in BR_ARMS if (not c_["model"]) or BR_MODEL_OK]',
    'BR_ARMS_RUN = list(BR_ARMS)')
cut('print(f"\\n=== PASS 2: {len(BR_ARMS_RUN)} post-link arms replayed on the pickled solve of "',
    "tb2 = time.time()",
    '''print(f"\\n=== PASS 2: {len(BR_ARMS_RUN)} post-link arms replayed on the pickled solve of "
      f"each movie: " + ", ".join(f"{a_} (DIV_DIVERGE_UM {c_['diverge']}"
                                  f"{', admit dropped primaries' if c_['admit'] else ''})"
                                  for a_, c_ in BR_ARMS_RUN) + " ===", flush=True)
''')

# ---- the arm loop: set the two knobs instead of the bridge ones ---------------
rep('''    for arm, cfg in BR_ARMS_RUN:
        GAP_SPAN = cfg["span"]
        BR_MODEL = BR_FOLD_MODEL[k] if cfg["model"] else None
        BR_TAU = cfg["tau"]
        BR_STAT.clear()
''', '''    for arm, cfg in BR_ARMS_RUN:
        DIV_DIVERGE_UM = cfg["diverge"]
        DA_ADMIT = cfg["admit"]
        DA_CTX.clear()
        DA_CTX.update(seeds=seeds, seed_of_node=seed_of_node, added=[])
        BR_STAT.clear()
''')
rep('''        ARM_MOVIE[arm][name] = _snapshot()
        cells.append(f"{arm} b2 {BR_STAT.get('span2_bridged', 0)} b3 {BR_STAT.get('span3_bridged', 0)} "
''', '''        ARM_MOVIE[arm][name] = _snapshot()
        for _g in DA_CTX.get("added", []):
            seed_of_node.pop(_g, None)
        cells.append(f"{arm} adm {PATCH_CAPS.get('admit_added', 0)} orph {PATCH_CAPS.get('orphan_added', 0)} "
''')
rep('''    GAP_SPAN = 1
    BR_MODEL = None
    BR_TAU = 0.0
    print(f"  {name} fold{k} REF''', '''    DIV_DIVERGE_UM = 2.25
    DA_ADMIT = False
    DA_CTX.clear()
    print(f"  {name} fold{k} REF''')
assert src.count("DIV_DIVERGE_UM = 2.25") == 2   # the knob line and the reset

# ---- the per-arm table: patch counters instead of bridge counters ------------
cut('print("\\n=== BRIDGES PER ARM, summed over the movies replayed ===", flush=True)',
    "# ---- report ----------------------------------------------------------------",
    '''print("\\n=== THE PATCH PER ARM (orphan pass + admit pass), summed over the movies replayed ===", flush=True)
_bk = ["orphan_added", "orphan_frame", "orphan_global", "admit_candidates", "admit_proposals",
       "admit_dup", "admit_added", "admit_frame", "admit_global"]
print(f"  {'arm':<10}" + "".join(f"{k_:>17}" for k_ in _bk), flush=True)
for a_ in ["REF"] + [x[0] for x in BR_ARMS_RUN]:
    d_ = CAPS_BY_ARM.get(a_, {})
    print(f"  {a_:<10}" + "".join(f"{d_.get(k_, 0):>17,}" for k_ in _bk), flush=True)

''')

# ---- names -----------------------------------------------------------------
rep('f"bridgerank_arm_{a_}.csv"', 'f"divadmit_arm_{a_}.csv"')
rep('f"bridgerank_REF_{DET_SRC}.csv"', 'f"divadmit_REF_{DET_SRC}.csv"')
rep('print("BRIDGERANK OK", flush=True)', 'print("DIVADMIT OK", flush=True)')
# the stale divbench9 "reading it" paragraph (dw35 arms that this run does not have)
cut('print("\\n=== reading it ===", flush=True)', "warning_summary()\nprint(f\"\\nDIVBENCH9 DONE", "")
assert "=== reading it ===" not in src and "bridgerank_REF_" not in src

assert "bridge_pairs_zh" not in src and "BR_FOLD_MODEL[k]" not in src
assert src.count("DA_ADMIT = cfg") == 1 and src.count('PATCH_CAPS["admit_added"]') == 1
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "divadmit.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written divadmit.py", len(src))
