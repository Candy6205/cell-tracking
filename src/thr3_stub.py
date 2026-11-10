"""Stub pushed to STOP the running biohub-thr3 session, 2026-08-25.

Kaggle runs one session per kernel, so pushing a new version replaces
the running one. The real script is `thr3.py` in this repository, at
commit 758b47c; re-push it by pointing kernel-metadata.json's
`code_file` back at `thr3.py`.

WHY IT WAS STOPPED, so this is not read as the experiment failing. It
had run 2 h 46 min against my estimate of ~170 min, and the estimate was
wrong for a reason worth keeping: I sized it with `poolsize`'s measured
67 s per dataset, which was measured on `6bba` ONLY, while half of this
run's competition datasets are `44b6` at **327 cells per frame against
6bba's 97**. The ILP costs about n^1.5 in candidate pairs, so a 44b6
dataset can run 5-8x longer. **Timing measured on one population does
not transfer to another** -- the same mistake, in a new place, that this
whole session has been about.

The user's call (option B): the queue behind it held a five-minute GT
export that unlocks a third compute worker, and `thr3`'s question --
whether `thr-sweep`'s +0.0033 is a `6bba` artefact -- cannot be settled
locally anyway. The leaderboard has refused two local gains of +0.0019
and +0.0073, so a third local number at that scale decides nothing.

Re-run it when a slot is free and nothing cheaper is waiting.
"""
print("thr3 stopped deliberately; see the docstring. Real script: thr3.py")
