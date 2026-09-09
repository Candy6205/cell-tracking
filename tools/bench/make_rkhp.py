"""rkpool.py -> rkhp.py: the fourth cell. Do hpara and the retrain ADD?

Two changes passed tonight, each measured alone against the same control:

    (BASE ranker, refine dets)  0.9266   what ships          rkpool_ref
    (F4 ranker,   refine dets)  0.9290   the retrain alone   rkpool     +0.0024
    (BASE ranker, hpara dets)   0.9298   hpara alone         hparabench +0.0032
    (?  ranker,   hpara dets)   ?        BOTH                THIS RUN

**They touch the same thing** -- which coordinates the ranker sees -- so there
is no reason to expect the sum. hpara moves every detection sub-voxel; the
retrain is the fix for a ranker trained on coordinates it is not deployed on.
Ship hpara with the F4-pool pkl and you have RE-CREATED the mismatch the
retrain removed, one placement further along. The matched version is to train
on hpara rows and solve hpara, and that is what this runs.

WHY IT IS ONE CONSTANT AND NO NEW GPU. `hparabench_ref.py` already carries the
whole `DET_SRC == "F4HP"` override, and it is careful: it replaces the F4
detection and BOTH head caches with their hpara counterparts, cut down to the
keys the refine arm already has, so the arms can never differ in the PIPELINE
rather than the placement. rkpool added `RANK_TRAIN` on top, and because the
override rewrites `FDET_F4` / `EDGEPROB_F4` in place, `RANK_TRAIN="F4"` in an
`F4HP` run means "train on hpara rows" for free. `hpara44` + `ephp44` (both
COMPLETE, 2026-09-09) finished the coverage: **fdet_F4hp 199, edgeprob_F4hp
199**, so the whole 175-movie pool can now be trained on hpara -- round 2 had
140 caches and could not have done this.

WHY ONE KERNEL AND NOT TWO. The control is `hparabench` round 2's F4HP arm,
whose per-movie CSV is on disk. Reusing a recorded CSV is normally forbidden
here, and the thing that licenses it this time is a proof rather than a hope:
`ANCHOR_D187` and `+BIDIR pkl` load FIXED pkls, and rkpool showed empirically
that both are byte-identical across `RANK_TRAIN` arms (0.9216 / 0.0606 and
0.9261 / 0.1351 in both). So round 2's F4HP values for them are HARD anchors
in this run:

    ANCHOR_D187  F4-6bba(12)  adj_J 0.9204  div_J 0.0667   HARD
    +BIDIR pkl   F4-6bba(12)  adj_J 0.9252  div_J 0.1389   soft

If those hold, this run's harness is round 2's and the two CSVs may be paired.
If they miss, the 59 extra hpara caches perturbed something and the pairing is
off -- which is exactly what the check is for.

    REF over the 128 to compare against: 55/237/70, div_J 0.1519,
    adj_J 0.9146, score 0.9298 (hparabench round 2).

PRE-REGISTERED READING, and all four branches are informative.
  1 ADDITIVE (~0.9322) -> the two fix different things. Ship both, and the
    joint candidate is worth about +0.0056 local over what ships.
  2 ABSORBED (~0.9298, the larger alone) -> hpara's gain WAS the mismatch.
    That is a real finding about hpara: it would mean the +0.0031 is the
    ranker coping better with cleaner coordinates, not the geometry helping
    the linker, and the zebrahub arm (different solver, no ranker) would then
    be the odd one out needing an explanation.
  3 BETWEEN -> partial overlap, and the honest ship is the better single one
    unless the joint clears both by more than the bench's noise.
  4 BELOW EITHER -> they interfere. Then DIV_W and the fork gate have to be
    re-crossed a THIRD time, because rkpool already showed the retrain moves
    the fork population by 9% and hpara moves it too.

⚠ WHAT THIS DOES NOT ANSWER. It is the UNGATED REF comparison on 128 6bba.
rkpool's lesson was that the shipped cell (DIV_W 0.55 + joint gate) can move
the other way from the ungated one, and the gate cross printed by this kernel
is trained inside this arm on this arm's own forks. Read the cross as
direction, and do not ship off it without the 44b6 fork population
regenerated under the same ranker.

Attach: hp44full's 16 sources exactly (rkpool's plus the five hpara cache
kernels). CPU only, about 3 h.
"""
import os

src = open("rkpool.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


assert 'RANK_TRAIN = "F4"' in src, "cut from rkpool.py (the F4 arm), not the ref"

rep('SCRIPT_VERSION = "2026-09-09a-rkpool"', 'SCRIPT_VERSION = "2026-09-09a-rkhp"')

# ---- 1. the one constant ------------------------------------------------
rep('DET_SRC = "F4"\nassert DET_SRC in ("F4", "F4HP"), DET_SRC',
    'DET_SRC = "F4HP"\nassert DET_SRC in ("F4", "F4HP"), DET_SRC')

# ---- 2. the anchors that license pairing against round 2's CSV ----------
rep('''    print("  DET_SRC=F4HP: the F4 anchors are demoted to soft. A large move "
          "in ANCHOR_D187-6bba (0.9216) is the measurement; a move of more "
          "than a couple of points is more likely a broken join -- check "
          "the JOIN counters before believing it.", flush=True)''',
    '''    print("  DET_SRC=F4HP: the F4 anchors are demoted to soft. A large move "
          "in ANCHOR_D187-6bba (0.9216) is the measurement; a move of more "
          "than a couple of points is more likely a broken join -- check "
          "the JOIN counters before believing it.", flush=True)
    # ...AND THEN RE-ARMED AT ROUND 2's VALUES. This run does not have its own
    # control arm: it is paired against hparabench round 2's F4HP CSV, which
    # is only legitimate if the harness is round 2's. ANCHOR_D187 and the
    # +BIDIR pkl arm load FIXED pkls, and rkpool showed both are byte
    # identical across RANK_TRAIN arms -- so on the SAME coordinates they must
    # reproduce round 2 exactly, whatever the fold rankers were trained on.
    # A miss means the 59 hpara caches hpara44/ephp44 added (140 -> 199)
    # perturbed the scored path, and the pairing is invalid.
    ANCHOR_SOFT.pop(("ANCHOR_D187", "F4", "6bba"), None)
    ANCHOR_HARD[("ANCHOR_D187", "F4", "6bba")] = 0.9204
    ANCHOR_SOFT[("+BIDIR", "F4", "6bba")] = 0.9252
    ANCHOR_DIV[("ANCHOR_D187", "F4", "6bba")] = 0.0667
    ANCHOR_DIV[("+BIDIR", "F4", "6bba")] = 0.1389
    print("  ...and re-armed at hparabench round 2's F4HP values: "
          "ANCHOR_D187 0.9204 HARD, +BIDIR pkl 0.9252 soft. These two load "
          "fixed pkls and cannot see RANK_TRAIN, so on round 2's coordinates "
          "they must reproduce round 2. That is what licenses pairing this "
          "CSV against hparabench_F4HP.csv instead of re-running the control.",
          flush=True)''')

# ---- 3. the REF row is the measurement; print what it must beat ---------
rep('''print("  hard anchors reproduced: the harness is the recorded one." if not ANCHOR_FAIL
      else "  HARD ANCHOR MISS", flush=True)''',
    '''print("  hard anchors reproduced: the harness is the recorded one." if not ANCHOR_FAIL
      else "  HARD ANCHOR MISS", flush=True)
print(f"  THE FOURTH CELL. Compare the REF row above against the three that "
      f"are on record, all on these same 128 movies:\\n"
      f"      (BASE ranker, refine)  50/249/75  div_J 0.1337  adj_J 0.9133  "
      f"score 0.9266   rkpool_ref\\n"
      f"      (F4   ranker, refine)  52/225/73  div_J 0.1486  adj_J 0.9141  "
      f"score 0.9290   rkpool     +0.0024\\n"
      f"      (BASE ranker, hpara )  55/237/70  div_J 0.1519  adj_J 0.9146  "
      f"score 0.9298   hparabench +0.0032\\n"
      f"      (F4   ranker, hpara )  <- this run. ~0.9322 is additive, "
      f"~0.9298 is absorbed, below 0.9290 is interference.", flush=True)''')

# ---- 4. the CSV must not overwrite either parent's ----------------------
rep('f"rkpool_{RANK_TRAIN}.csv"', 'f"rkhp_{RANK_TRAIN}_{DET_SRC}.csv"')

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "rkhp.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written rkhp.py", len(src), "DET_SRC F4HP, RANK_TRAIN F4")
