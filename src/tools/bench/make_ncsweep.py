"""forkprop_z.py -> ncsweep.py: the node cost above 0.35, on the Z composition, before any submission.

WHY (2026-09-28). Three submissions sit on one line (node cost 0 -> 0.20 -> 0.35: LB 0.954 -> 0.955 -> 0.956) and the
record's residue says the cost above 0.35 has no local number (hypx's sweep stopped at NC35X). hypx's arms split by movie
difficulty gain in every quartile at 0.35, most in the hardest. The user (2026-09-28): "节点代价你得先在本地测0.5和0.65，
我不能容忍本地没有数据的参数上提交管线".

WHAT. forkprop_z.py's harness (the 128 6bba movies, F4HP primaries, the shipped fold rankers, the division-variable ILP
with divscore's triples, the shipped patch, the 36-column joint image gate fitted out of fold on each arm's own forks, the
official scorer, paired bootstraps) with the ZX arm removed and the Z solve run at three node costs:
    Z    node cost 0.35 (what submit_zsz ships; the control; must reproduce forkprop_z's Z cells)
    Z50  node cost 0.50
    Z65  node cost 0.65
the triples' costs, the exemption (REF's fork daughters and their chains) and everything after the solve unchanged.
Arms CR (REF's solve, the anchor), ZCR, Z50CR, Z65CR. Reads: Z50 / Z65 against Z paired over the movies (score, adj_J;
gated and ungated), the node and edge counts per arm, and the per-movie rows for the split by difficulty.
Attach: forkprop_z's sources. CPU ~9 h.
"""
src = open("forkprop_z.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:100])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-25a-forkprop_z"', 'SCRIPT_VERSION = "2026-09-28a-ncsweep"')

rep('''           ("ZXCR_NOCAP_DV150", dict(diverge=1.5, admit=False, rank="contrast", cap=100.0, src="ZX"))]
''', '''           ("Z50CR_NOCAP_DV150", dict(diverge=1.5, admit=False, rank="contrast", cap=100.0, src="Z50")),
           ("Z65CR_NOCAP_DV150", dict(diverge=1.5, admit=False, rank="contrast", cap=100.0, src="Z65"))]
NC_ARMS = (("Z", 0.35), ("Z50", 0.50), ("Z65", 0.65))        # ncsweep: the node cost per z solve
''')
rep("Z_STAT = {a_: defaultdict(int) for a_ in ('Z', 'ZX')}\n", "Z_STAT = {a_: defaultdict(int) for a_ in ('Z', 'Z50', 'Z65')}\n")

# ---- the three costed solves ------------------------------------------------------------------------------------------
i0 = src.index("    for tag in ('Z', 'ZX'):\n        DUP_STAT.clear()\n")
i1 = src.index("        for k_, v_ in DUP_STAT.items():\n            Z_STAT[tag][k_] += v_\n", i0)
src = src[:i0] + '''    assert Z_CFG['cost'] == NC_ARMS[0][1], (Z_CFG, NC_ARMS)
    for tag, cost_ in NC_ARMS:
        DUP_STAT.clear()
        _ts = time.time()
        n_, e_, p_, aux_ = solve_hyp(dets, T, seeds, feats, model, COLS, None, cost_, disapp=ILP_DISAPP,
                                     floor=PROD_FLOOR, exempt=ex_t, trip=trip)
        Z_STAT[tag]['wall_s'] += int(time.time() - _ts)
''' + src[i1:]

rep('''    cells.append(f"z solves {time.time() - _tz:.0f}s: Z edges {len(_zs['Z'][1])} ZX edges {len(_zs['ZX'][1])} (REF {len(edges0)})")
''', '''    cells.append(f"z solves {time.time() - _tz:.0f}s: " + " ".join(f"{_t} nodes {len(_zs[_t][0])} edges {len(_zs[_t][1])}" for _t, _c in NC_ARMS)
                 + f" (REF nodes {len(nodes0)} edges {len(edges0)})")
''')
rep('    for _src in ("REF", "Z", "ZX"):\n', '    for _src in ("REF", "Z", "Z50", "Z65"):\n')

i0 = src.index('print("\\n=== THE z SOLVES, summed over the movies')
i1 = src.index('print("\\n=== THE PATCH PER ARM (orphan pass + admit pass)')
src = src[:i0] + '''print("\\n=== THE z SOLVES, summed over the movies (the same triples and exemption; the node cost is the only difference). "
      "TP by divtrip's 5 um label ===", flush=True)
for _tag, _cost in NC_ARMS:
    d_ = Z_STAT[_tag]
    print(f"  {_tag:<4} node cost {_cost:.2f} | movies {d_['movies']} | z variables {d_['z_vars']:,} (TP {d_['z_vars_tp']}), z selected "
          f"{d_['z_selected']:,} (TP {d_['z_sel_tp']}) | nodes selected {d_['nodes_selected']:,} of {d_['nodes_offered']:,} "
          f"({d_['nodes_selected'] / max(d_['nodes_offered'], 1) * 100:.2f}%), edges {d_['edges_selected']:,}, ILP divisions "
          f"{d_['ilp_divisions']:,}, exempt {d_['exempt']:,}, solve {d_['solve_s']:,}s (wall {d_['wall_s']:,}s)", flush=True)
''' + src[i1:]

rep('''print("    reading: an arm counts only if its jointimg cell beats REF + jointimg with the interval off zero; the "''',
    '''print("\\n  THE NODE COST AGAINST 0.35: Z50CR / Z65CR vs ZCR (the same triples, patch and gate recipe; the node cost is the only "
      "difference), paired bootstrap over the movies, gated and ungated ===", flush=True)
for _pair in (("Z50CR_NOCAP_DV150 + jointimg", "ZCR_NOCAP_DV150 + jointimg"), ("Z50CR_NOCAP_DV150 + joint", "ZCR_NOCAP_DV150 + joint"),
              ("Z65CR_NOCAP_DV150 + jointimg", "ZCR_NOCAP_DV150 + jointimg"), ("Z65CR_NOCAP_DV150 + joint", "ZCR_NOCAP_DV150 + joint"),
              ("Z65CR_NOCAP_DV150 + jointimg", "Z50CR_NOCAP_DV150 + jointimg")):
    if _pair[0] not in BS or _pair[1] not in BS:
        continue
    for nm_, bank in (("score", BS), ("adj_J", _BSA)):
        d = bank[_pair[0]] - bank[_pair[1]]
        lo, hi = float(np.quantile(d, .05)), float(np.quantile(d, .95))
        print(f"    {_pair[0]:<29} vs {_pair[1]:<29} {nm_:<6} {float(np.mean(d)):>+9.4f}   [{lo:>+8.4f}, {hi:>+8.4f}]  "
              f"{'YES' if (lo > 0 or hi < 0) else 'no'}", flush=True)
print("    reading: a node cost above 0.35 is worth a submission only if its arm beats ZCR on the score with the interval off zero; "
      "the hidden scorer has so far paid more for fewer nodes than the local one (three submissions on one line).", flush=True)
print("    reading: an arm counts only if its jointimg cell beats REF + jointimg with the interval off zero; the "''')

rep('with open("/kaggle/working/forkssz_dw55.csv", "w") as fh:', 'with open("/kaggle/working/forknc_dw55.csv", "w") as fh:')
rep('    with open(f"/kaggle/working/forkssz_{arm}.csv", "w") as fh:', '    with open(f"/kaggle/working/forknc_{arm}.csv", "w") as fh:')
rep('    print("FORKPROPZ OK", flush=True)', '    print("NCSWEEP OK", flush=True)')

for bad in ('"ZX"', "'ZX'", "ZXCR"):
    assert bad not in src, (bad, src.index(bad))
assert src.count("for tag, cost_ in NC_ARMS:") == 1 and "Z65CR_NOCAP_DV150" in src
open("ncsweep.py", "w", encoding="utf-8", newline="\n").write(src)
print("written ncsweep.py", src.count("\n"), "lines")
