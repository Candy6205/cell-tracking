"""skipilp.py -> bridgegate.py: a learned veto on the shipped span-1 bridges.

WHAT IT MEASURES. The public 0.947 notebook's last gain came from vetoing its
synthetic gap midpoints with a second detector's response (DeepCenter, 8-view
TTA). Our span-1 gap close inserts a midpoint for every (end, start) pair
within 9 um up to the 2% cap with no check at all (GAP_NEED_CAND was measured
worse). skipilp's labelled span-2 pairs price the situation: among pairs within
9 um whose two ends are both free (no consecutive candidate above the floor at
either end -- the bridge situation), 280 are true continuations (211 of them
real holes) and 216 are not; the out-of-fold span-2 selector at p < 0.2 kills
55% of the false ones and 6% of the true holes, at p < 0.3 62% / 12%. So the
veto here is that selector (skipilp's stage A2: 23 features incl. the
midpoint's nearest primary / candidate, the best two-step path, velocity
consistency; two fold HistGB models, out of fold), applied to every span-1
bridge the shipped rule would insert: the bridge is refused when the pair's
p is below BRIDGE_TAU. A bridge whose endpoint is itself an inserted node
(no seed) is never vetoed (the shipped behaviour).

ARMS, all post-link replays on the SAME REF solve (no re-solve): BG00 (tau 0,
must equal REF exactly; carries the labelled composition of every bridge),
BG10 / BG20 / BG30 / BG50. Each goes through the rest of the shipped patches,
the official scorer, the fork table and the joint-gate cross like dupsup's
arms. Stage A2 is skipilp's (28 min); stage B is REF + 5 cheap replays.

Attach: dupsup's 16 sources (skipilp's metadata). CPU, ~3 h.
"""
import os

src = open("skipilp.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-14a-skipilp"', 'SCRIPT_VERSION = "2026-09-15a-bridgegate"')

# ---- the arms ---------------------------------------------------------------------
rep('DUP_ARMS = SKIP_ARMS      # skipilp: four skip-edge solves, no suppression',
    'BRIDGE_ARMS = [("BG00", dict(tau=0.0)), ("BG10", dict(tau=0.10)), ("BG20", dict(tau=0.20)),\n'
    '               ("BG30", dict(tau=0.30)), ("BG50", dict(tau=0.50))]\n'
    'DUP_ARMS = BRIDGE_ARMS    # bridgegate: post-link replays of the span-1 gap close with a veto')
rep('SKIP_CHOSEN = []     # (t, i, j) of the skip edges the last solve kept, filled by _skip_insert\n',
    'SKIP_CHOSEN = []     # (t, i, j) of the skip edges the last solve kept, filled by _skip_insert\n'
    'BRIDGE_TAU = 0.0     # bridgegate: a span-1 bridge whose pair p is below this is refused (0 = shipped)\n'
    'BRIDGE_P = {}        # (t, i, j) -> out-of-fold selector p of the current movie\n'
    'BRIDGE_LAB = {}      # (t, i, j) -> (lab, mid) where labelled\n')

# ---- the veto, at the bridge acceptance point --------------------------------------
rep('                n_demand += 1\n',
    '                n_demand += 1\n'
    '                if span == 1:\n'
    '                    _sa = BR_CTX.get("seed_of_node", {}).get(ge)\n'
    '                    _sb = BR_CTX.get("seed_of_node", {}).get(gs)\n'
    '                    _pb = None\n'
    '                    _key = None\n'
    '                    if _sa is not None and _sb is not None and _sb[0] == _sa[0] + 2:\n'
    '                        _key = (int(_sa[0]), int(_sa[1]), int(_sb[1]))\n'
    '                        _pb = BRIDGE_P.get(_key)\n'
    '                    DUP_STAT["bridge_offered"] += 1\n'
    '                    if _pb is None:\n'
    '                        _tag = "noseed"\n'
    '                    else:\n'
    '                        _lab = BRIDGE_LAB.get(_key, (-1, -1))\n'
    '                        _tag = ("pos_hole" if _lab == (1, 1) else "pos" if _lab[0] == 1\n'
    '                                else "neg" if _lab[0] == 0 else "unlab")\n'
    '                    if _pb is not None and _pb < BRIDGE_TAU:\n'
    '                        DUP_STAT["bridge_vetoed"] += 1\n'
    '                        DUP_STAT[f"bridge_vetoed_{_tag}"] += 1\n'
    '                        continue\n'
    '                    DUP_STAT[f"bridge_kept_{_tag}"] += 1\n')

# ---- the arm loop: replays, no re-solve --------------------------------------------
i = src.index('    sp = SKIP_PROPS.pop(name)\n')
j = src.index('    extra = ""\n', i)
src = src[:i] + '''    sp = SKIP_PROPS.pop(name)
    BRIDGE_P = {(int(t_), int(i_), int(j_)): float(p_)
                for t_, i_, j_, p_ in zip(sp["t"], sp["i"], sp["j"], sp["p"])}
    BRIDGE_LAB = {(int(t_), int(i_), int(j_)): (int(l_), int(m_))
                  for t_, i_, j_, l_, m_ in zip(sp["t"], sp["i"], sp["j"], sp["lab"], sp["mid"]) if l_ >= 0}
    for arm, cfg in BR_ARMS:
        BRIDGE_TAU = float(cfg["tau"])
        DUP_STAT.clear()
        outa = pathlib.Path(f"{TMP}/arm_{arm}") / f"{name}.geff"
        outa.parent.mkdir(parents=True, exist_ok=True)
        PATCH_CAPS.clear()
        n_a, e_a = patch_final(nodes0, edges0, pool, outa)
        BRIDGE_TAU = 0.0
        for k_, v_ in PATCH_CAPS.items():
            CAPS_BY_ARM.setdefault(arm, defaultdict(int))[k_] += v_
        row_a, ds_a, fr_a = _judge(name, k, outa, seeds, aux)
        ARM_ROWS[arm].append(row_a)
        ARM_PER[arm][name] = row_a
        ARM_FORKS[arm].extend(fr_a)
        ARM_MOVIE[arm][name] = _snapshot()
        for k_, v_ in DUP_STAT.items():
            DUP_BY_ARM[arm][k_] += v_
        cells.append(f"{arm} veto {DUP_STAT['bridge_vetoed']}/{DUP_STAT['bridge_offered']} nodes {len(n_a)} "
                     f"{row_a['division_tp']}/{row_a['division_fp']}/{row_a['division_fn']} "
                     f"{_adj_of(row_a):.4f}")
        del ds_a, n_a, e_a, fr_a
    BRIDGE_P = {}
    BRIDGE_LAB = {}

''' + src[j:]

# ---- the per-arm bookkeeping table ---------------------------------------------------
rep('_dk = ["skip_offered", "skip_chosen", "skip_dropped_fork", "skip_inserted", "skip_snapped",\n'
    '       "skip_chosen_pos", "skip_chosen_pos_hole", "skip_chosen_neg", "skip_chosen_unlab"]',
    '_dk = ["bridge_offered", "bridge_vetoed", "bridge_kept_pos_hole", "bridge_kept_pos", "bridge_kept_neg",\n'
    '       "bridge_kept_unlab", "bridge_kept_noseed", "bridge_vetoed_pos_hole", "bridge_vetoed_pos",\n'
    '       "bridge_vetoed_neg", "bridge_vetoed_unlab"]')
rep('print(f"  {\'arm\':<8}" + "".join(f"{k_[5:]:>17}" for k_ in _dk), flush=True)',
    'print(f"  {\'arm\':<8}" + "".join(f"{k_[7:]:>14}" for k_ in _dk), flush=True)')
rep('''    print(f"  {a_:<8}" + "".join(f"{d_.get(k_, 0):>17,}" for k_ in _dk)
          + f"   (chosen {d_.get('skip_chosen', 0) / max(1, d_.get('skip_offered', 1)):.3%} of offered)", flush=True)''',
    '''    print(f"  {a_:<8}" + "".join(f"{d_.get(k_, 0):>14,}" for k_ in _dk)
          + f"   (vetoed {d_.get('bridge_vetoed', 0) / max(1, d_.get('bridge_offered', 1)):.3%} of offered)", flush=True)''')
rep('print("\\n=== SKIP EDGES PER ARM, summed over the 128 movies (offered = candidate edges given to '
    'the ILP; chosen = solved t->t+2 edges; inserted = chosen minus those dropped for sharing a '
    'source with another edge; snapped = inserted node sits on a sub-threshold candidate; '
    'pos/neg/unlab = GT label of the chosen pair, hole = positive whose middle GT node no primary '
    'matched) ===", flush=True)',
    'print("\\n=== SPAN-1 BRIDGES PER ARM, summed over the 128 movies (offered = (end, start) pairs the '
    'shipped rule would bridge; vetoed = refused at the arm\'s tau; kept/vetoed by GT label of the pair: '
    'pos_hole = true continuation whose middle GT node no primary matched, pos = true continuation, neg = '
    'not a continuation, unlab = an unannotated endpoint, noseed = an endpoint is itself an inserted '
    'node) ===", flush=True)')

# ---- names ------------------------------------------------------------------------
rep('f"skipilp_REF_{DET_SRC}.csv"', 'f"bridgegate_REF_{DET_SRC}.csv"')
rep('f"skipilp_arm_{a_}.csv"', 'f"bridgegate_arm_{a_}.csv"')
rep('print("SKIPILP OK", flush=True)', 'print("BRIDGEGATE OK", flush=True)')
rep('print(f"[degraded] SKIPILP INCOMPLETE or anchor miss:', 'print(f"[degraded] BRIDGEGATE INCOMPLETE or anchor miss:')

assert "skip=skip_a" not in src and src.count("BRIDGE_TAU") >= 4
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "bridgegate.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written bridgegate.py", len(src))
