"""hparacache.py -> hpara44.py: hpara coordinates for the 59 44b6 movies that
do not have them, so hpara can be scored on ALL 71 instead of 12.

THE USER, on being handed 44b6 as a "null" measured on 12 movies:
*"思考，把44b6全跑一遍不行吗，你这分裂全挂0"*. Both halves land.

**The divisions really are all zero, structurally.** The 12-movie seeded draw
(`SET_44B6`) contains **2 GT divisions**, and both were FN in both arms:
F4 0/11/2, F4HP 0/16/2. div_J is 0.0000 on both sides, so 44b6's score IS its
adj_J and the +0.0182 div_J that carries a third of hpara's 6bba gain has no
way to express itself. The full 71 carry **26** (progress.md:3532). Not many
against 6bba's 125, but 26 can be read and 2 cannot.

**And the record already knows the 12-movie draw misreads 44b6.** `divbench5`
v2 ran all 71 (117.9 min, both hard anchors exact) for the fork gate and
recorded, in its own words: divbench3's 12-movie draw said +0.0020 adj_J on 2
GT divisions; the full 71 said **+0.0014 on 26**. The small draw was 43% high
on the same quantity.

WHAT IS MISSING, and it is only this. All 71 44b6 movies have F4 primaries
(`fusecache` 12 + `fuse44b6` 59) and F4 head probabilities (`edgeprob` 12 +
`ep44b6` 59). Only **12** have hpara ones: `hparacache` v3 built
`NAMES = HOLDOUT_6BBA + NAMES_6BBA + SET_44B6` = 12 + 116 + 12 = 140, and the
44b6 twelve are the seeded draw. This kernel builds the other 59. The head
pass over them is a separate kernel (`ephp44`), exactly as `ep44b6` followed
`fuse44b6`.

SIZING, from the runs it is cut from. `fuse44b6` built these same 59 movies
in **90.1 min** (91 s each, 44b6's F4 node sets running about 1.8x 6bba's).
hpara adds one parabola per surviving peak on the pooled grid and no second
heatmap pass, and `hparacache` came in at fuse6bba's own rate, so expect
roughly 90 min again.

THE ANCHOR IS INHERITED AND IS THE STRONG ONE: every frame's `refine` half
must reproduce the attached `fdet_F4_<name>.npz` exactly. Same weights, same
peaks, same NMS -- NMS is run ONCE on the refine coordinates and the
survivors are placed both ways, so detection count, order and identity are
held fixed and only position moves. For these 59 the anchor caches come from
`biohub-fuse44b6` rather than `biohub-fusecache`, which is the only reason
this file exists rather than a one-line list edit.

⚠ Same caveat as `fuse44b6`: these 59 are INSIDE dctrain's 175-movie pool, so
F4's fourth heatmap (ep05 DeepCenter) is in-sample on them. **No detection
claim may be made from these caches.** They exist so that the hpara pipeline
arm can be scored out-of-fold on 71 movies instead of 12.

Attach: hparacache's sources, with faustine649/biohub-fuse44b6 added as a
kernel_source for the anchor. GPU.
"""
import os

src = open("hparacache.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:80])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-08a-hparacache"',
    'SCRIPT_VERSION = "2026-09-09a-hpara44"')

# ---- the 59, defined exactly as fuse44b6 defined them ------------------
rep('''NAMES = HOLDOUT_6BBA + NAMES_6BBA + SET_44B6
assert len(NAMES) == 140 and len(set(NAMES)) == 140, len(NAMES)
print(f"NAMES: {len(HOLDOUT_6BBA)} holdout + {len(NAMES_6BBA)} pool 6bba "
      f"+ {len(SET_44B6)} 44b6 = {len(NAMES)}", flush=True)''',
    '''# make_fuse44b6.py's list, character for character: every 44b6 movie that is
# NOT one of the 12 manifest-excluded ones (fusecache's EX_44B6 = dctrain's
# held-out 12 = edgeprob_cache's seeded SET_44B6 = the 12 hparacache already
# built). Cross-checked against dctrain's pool the same way, because a silent
# change here would cache a different movie set and still print a plausible
# number.
NAMES = [n for n in names_all
         if n.startswith("44b6_") and n not in set(EX_44B6)]
assert len(NAMES) == 59, len(NAMES)
assert all(n in _SEEN for n in NAMES), \\
    "a 44b6 movie outside dctrain's pool slipped into the build list"
assert not (set(NAMES) & set(SET_44B6)), \\
    "SET_44B6 must already be cached by hparacache; rebuilding it would move "\\
    "every anchor that reads those 12"
print(f"44b6 to build ({len(NAMES)}): {' '.join(NAMES)}", flush=True)
print(f"44b6 already cached by hparacache ({len(SET_44B6)}): "
      f"{' '.join(SET_44B6)}", flush=True)
print(f"  after this kernel all 71 44b6 movies carry hpara primaries, so the "
      f"hpara arm can be scored on 71 instead of 12 -- the draw that "
      f"divbench5 v2 measured as 43% high on the same quantity.", flush=True)''')

rep('''    f"{len(_missing)} of the {len(NAMES)} bench movies have no fdet_F4_ "
    f"cache to "
    f"anchor against -- attach faustine649/biohub-fusecache as a "
    f"kernel_source. Missing: {_missing[:4]}")''',
    '''    f"{len(_missing)} of the {len(NAMES)} 44b6 movies have no fdet_F4_ "
    f"cache to anchor against -- attach faustine649/biohub-fuse44b6 (it "
    f"built exactly these 59) as a kernel_source. Missing: {_missing[:4]}")''')

rep('print(f"\\nHPARACACHE DONE', 'print(f"\\nHPARA44 DONE')
rep('print("HPARACACHE OK" if not ANCHOR_FAILS else "HPARACACHE ANCHOR FAILED",',
    'print("HPARA44 OK" if not ANCHOR_FAILS else "HPARA44 ANCHOR FAILED",')

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "hpara44.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
