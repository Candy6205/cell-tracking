"""hypz3.py -> hypz4.py: the z-daughter exemption done right -- the first pass without the node cost.

WHY (2026-09-25). hypz3's Z10X solved the movie with the node cost 0.35, took that solution's z daughters + downstream
chains into the exemption set, and solved again: identical to Z10 in every count, because a z daughter selected under
the cost had already paid it -- exempting a node that is already in the solution changes no decision. The shipped
pipeline's shape is different: its first solve carries NO node cost (the td solver), and the exemption protects that
solution's fork daughters when the costed solve would drop them. hypz's record notes the bench under-favours z for the
same reason (its exemption set comes from the control's forks). The user (2026-09-25): "那要重跑吗" -> this one arm.

ARMS: Z10 (the reference, unchanged) and Z10X0 (first pass: node cost 0, no exemption, the Z10 triples; second pass:
node cost 0.35, exemption = REF's set + the first pass's z-fork daughters and their downstream chains). The A2 / R
pre-passes of hypz3 are skipped (no arm uses them); the S0 for scorer B is hypz3's rule (1.15 per frame).
Attach: hypz3's sources. CPU, ~3 h (two arms, one of them two solves).
"""
src = open("hypz3.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-25a-hypz3"', 'SCRIPT_VERSION = "2026-09-25b-hypz4"')
rep('HYP_ARMS = [("Z10", dict(cost=0.35, ex="t", lam=0.10, wmin=0.10, src="B")),\n'
    '            ("Z10R", dict(cost=0.35, ex="t", lam=0.10, wmin=0.10, src="R")),\n'
    '            ("Z10A", dict(cost=0.35, ex="t", lam=0.10, wmin=0.10, src="A", K=20)),\n'
    '            ("Z10W", dict(cost=0.35, ex="t", lam=0.10, wmin=0.10, src="A", K=60)),\n'
    '            ("Z10X", dict(cost=0.35, ex="t", lam=0.10, wmin=0.10, src="B", zx=True))]\n',
    'HYP_ARMS = [("Z10", dict(cost=0.35, ex="t", lam=0.10, wmin=0.10, src="B")),\n'
    '            ("Z10X0", dict(cost=0.35, ex="t", lam=0.10, wmin=0.10, src="B", zx=True))]\n')
# skip the unused pre-passes
rep("A2_MODELS, TRIP_FOLD_OF = fit_scorer_A2()\n"
    "TRIP_A, _logs_A, _nf_A = build_pool_A(A2_MODELS, TRIP_FOLD_OF, 60)\n"
    "TRIP_R, _logs_R = fit_scorer_R()\n",
    "TRIP_A, _logs_A, _nf_A = {}, np.zeros(0, np.float32), 1          # hypz4: no arm uses the A pools or scorer R\n"
    "TRIP_R, _logs_R = {}, np.zeros(0)\n")
rep("S0_OF = {'B': _s0([r_[4] for r_ in _b6], _nf_B), 'R': _s0(_logs_R, _nf_B), 'A': _s0(_logs_A, _nf_A)}\n",
    "S0_OF = {'B': _s0([r_[4] for r_ in _b6], _nf_B), 'R': Z_S0, 'A': Z_S0}\n")
# the first pass of a zx arm: no node cost, no exemption
rep("        n_a, e_a, p_a, aux_a = solve_hyp(dets, T, seeds, feats, MODELS_FOLD[k], COLS,\n"
    "                                         None, cfg[\"cost\"], disapp=ILP_DISAPP, floor=PROD_FLOOR,\n"
    "                                         exempt=ex_set, divw=cfg.get(\"vcost\"), trip=trip_a)\n"
    "        if cfg.get(\"zx\") and trip_a:\n"
    "            # hypz3: the z solution's fork daughters + downstream chains join the exemption set; solve again\n",
    "        n_a, e_a, p_a, aux_a = solve_hyp(dets, T, seeds, feats, MODELS_FOLD[k], COLS,\n"
    "                                         None, 0.0 if cfg.get(\"zx\") else cfg[\"cost\"], disapp=ILP_DISAPP, floor=PROD_FLOOR,\n"
    "                                         exempt=None if cfg.get(\"zx\") else ex_set, divw=cfg.get(\"vcost\"), trip=trip_a)\n"
    "        if cfg.get(\"zx\") and trip_a:\n"
    "            # hypz4: the FIRST pass ran without the node cost (the shipped pipeline's shape); its z-fork daughters +\n"
    "            # downstream chains join REF's exemption set and the movie is solved again with the cost\n")
rep('f"hypz3_arm_{a_}.csv"', 'f"hypz4_arm_{a_}.csv"')
rep('f"hypz3_z_{a_}.csv"', 'f"hypz4_z_{a_}.csv"')
rep('f"hypz3_REF_{DET_SRC}.csv"', 'f"hypz4_REF_{DET_SRC}.csv"')
rep('print("HYPZ3 OK", flush=True)', 'print("HYPZ4 OK", flush=True)')
rep('print(f"[degraded] HYPZ3 INCOMPLETE or anchor miss:', 'print(f"[degraded] HYPZ4 INCOMPLETE or anchor miss:')
assert src.count("0.0 if cfg.get(\"zx\")") == 1 and src.count("= fit_scorer_A2()") == 0
open("hypz4.py", "w", encoding="utf-8", newline="\n").write(src)
print("written hypz4.py", src.count("\n"), "lines")
