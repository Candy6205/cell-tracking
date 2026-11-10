"""submit_nc.py -> submit_nc35.py: the shipped submission with the node cost raised from 0.20 to 0.35.

WHY (2026-09-23). submit_nc (NC20XT: node cost 0.20, REF's fork daughters and their downstream chains exempt) removed 1.7% of
the nodes and scored LB 0.955 against the 0.954 control; every node-adding submission lost. hypx measured cost 0.35 with the
daughters exempt (NC35X) at adj_J +0.0023 [+0.0013, +0.0032] against REF, edge_J -0.0006, ILP nodes -5.3%, score +0.0021
[-0.0001, +0.0042]; the forum (2026-09-23 reading) reports the node-count term as the one every LB movement follows. This is
the same two-pass port with NC_COST = 0.35 (the XT exemption kept: it only matters through the divisions). Nothing else changes.
"""
src = open("submit_nc.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-21a-submit-nc"', 'SCRIPT_VERSION = "2026-09-23a-submit-nc35"')
rep("NC_COST = 0.20\n", "NC_COST = 0.35\n")
open("submit_nc35.py", "w", encoding="utf-8", newline="\n").write(src)
print("written submit_nc35.py", src.count("\n"), "lines")
