# Progress log — Biohub Cell Tracking During Development

> Reader: a future AI session. Strictly reverse-chronological; the newest
> entry and the state block below are always at the top. Stable lookup facts
> (metric, survey, mechanics, code map, traps) live in `reference.md` — this
> file records events and measurements only. When a later entry overturns an
> earlier claim, the earlier text carries a `SUPERSEDED` mark in place.
> Engineering conventions: `engineering-rules.md` (rule 1: degradation is
> never silent).

## HANDOFF (2026-08-26, replacing the 2026-08-24 one below)

**BEST IS NOW 0.944, ref 56086066** (2026-09-08) — submit-joint:
the fork gate retrained on BOTH domains' counted forks
(`fork_gate_joint.pkl`, p > 0.03) on the 0.943 pipeline, **+0.001**.
Its pre-registered band called that WEAK TRANSFER: keep it. Read
the public score as a DIRECTION, sign only -- local +0.0037 landing
as LB +0.001 is not a finding, because movie-to-movie weight inside
one domain spans 4.3x (p90/p10 labelled edges, n=128) and the public
subset is unknown. The reading to be alarmed by is local-positive
with LB NEGATIVE. `submit_joint.py` is the
production script. Previous best 0.943, ref 56055405 (2026-09-07) — submit-gate: the
fork gate (divbench3, fork_gate_f4.pkl at p > 0.03) on the 0.933
pipeline, **+0.010, the largest single LB gain on record**, with
0.04% less hidden output (266,152,167 bytes); above the public
notebook's 0.935. `submit_gate.py` is the production script. Previous
best 0.933, ref 56030644 (2026-09-05) — submit-f48: the ILP candidate
floor 0.10 -> 0.48, +0.001 (266,263,495 bytes). Before that 0.932,
ref 55974245 (2026-09-03) — edgeprob: the
official edge head's probability (forward + reverse + harmonic mean) as
three extra ranker columns, ranker retrained on D175. +0.007 over F4's
0.925, the largest single LB gain on record, with 0.22% LESS hidden
output. Previous best 0.925, ref 55889500 (2026-08-31, F4). Full history:

| ref | change | its own control | LB | verdict |
|---|---|---|---|---|
| 55707206 | D32 ranker, global solve | — | 0.921 | held the record 08-23 to 08-28 |
| 55757064 | ranker pool 32 → 187 | 55707206 = 0.921 | 0.920 | indistinguishable, not a loss |
| 55772253 | `SHORT_MIN` 6 → 4 | 55757064 = **0.920** | **0.920** | a TRUE ZERO, not a loss |
| 55780773 | + gate 0.15 → 0.12 | 55772253 = 0.920 | **0.918** | the only clear loss on record |
| 55817630 | gap budget 0.003 → 0.02, candidate requirement dropped | 55772253 = 0.920 | 0.922 | +0.002 |
| **55889500** | **F4: ep05 DeepCenter as 4th equal-weight heatmap** | 55817630 = 0.922 | **0.925** | **+0.003, NEW BEST — largest single gain on record** |
| 55904403 | W1101: branch 3 to half weight | 55889500 = 0.925 | 0.924 | −0.001, and it PRICES 6bba (below) |
| 55911058 | M15: ep05's missed points added as primaries | 55889500 = 0.925 | **0.917** | **−0.008, WORST ON RECORD** |
| **55974245** | **edgeprob: official edge head p_fwd/p_rev/h_fused as ranker cols, D175 +BIDIR ranker** | 55889500 = 0.925 | **0.932** | **+0.007, NEW BEST — largest single gain on record** |
| 56000581 | submit_off: OFFICIAL detector nodes (gate 0.99, flip TTA) + head-only h_fused into our ILP, no ranker | 55974245 = 0.932 | **0.895** | **−0.037, WORST ON RECORD; the local +0.004 / +0.057 was training-set overlap** |
| 56014734 | nb948-asis: the public "0.948" notebook forked UNCHANGED (not our pipeline) | — | 0.935 | NOT 0.948; 204,966,451 bytes = 24% less output than ours |
| **56030644** | **submit-f48: ILP candidate floor 0.10 -> 0.48, nothing else** | 55974245 = 0.932 | **0.933** | **+0.001, NEW BEST, with −1.43% output; the hidden set carries almost no weak chains of ours** |
| **56055405** | **submit-gate: the fork gate (fork_gate_f4.pkl, p > 0.03) on the 0.933 pipeline, nothing else** | 56030644 = 0.933 | **0.943** | **+0.010, NEW BEST, LARGEST GAIN ON RECORD; −0.04% output; the bench said +0.012 and LB agreed** |
| **56086066** | **submit-joint: the same gate RETRAINED on 128 6bba + 71 44b6 counted forks** | 56055405 = 0.943 | **0.944** | **+0.001, NEW BEST; its own pre-registered band = weak transfer. Sign is the reading; the gap to local +0.0037 is not** |

Hidden-test output size, `total_bytes` from the submission metadata,
paired against each change's own control:

| change | Δ hidden output | Δ LB |
|---|---|---|
| ranker pool 32 → 187 | +0.07% | −0.001 |
| gap4 nocand_cap20 | +0.42% | +0.002 |
| F4 | +0.96% | +0.003 |
| W1101 | +0.96% | −0.001 |
| `SHORT_MIN` 6 → 4 | +1.61% | 0.000 |
| M15 | +3.71% | −0.008 |
| gate 0.15 → 0.12 | +3.94% | −0.002 |
| **edgeprob** | **−0.22%** | **+0.007** |
| submit_off (official detector) | +3.70% | −0.037 |
| nb948-asis (the public notebook, vs our F4 pipeline) | **−24.1%** | +0.003 |
| **submit-f48 (ILP floor 0.48)** | **−1.43%** | **+0.001** |
| **submit-gate (fork gate p > 0.03)** | **−0.04%** | **+0.010** |

Absolute bytes: 55707206 262,623,879 · 55757064 262,803,696 · 55772253
267,030,255 · 55817630 268,154,477 · 55889500 270,740,009 · 55904403
273,340,932 · 55780773 277,555,127 · 55911058 280,776,965 · **55974245
270,137,616** · 56000581 280,141,150 · 56014734 (public notebook) **204,966,451** · 56030644 266,263,495 · **56055405 266,152,167**.

⏳ **THREE RUNS FROM 2026-09-02 ARE UNREAD (待分析).** `edgewhy`,
`divwhy` (read v2, not v1) and `divgate2` are recorded below with their
tables and anchors and no conclusion filed against them. Four more ran the same
night and are recorded the same way: `divsweep` (DIV_DIVERGE_UM 2.25 /
1.50 / 1.00 / 0.50 / 0.00, post-link, scored), `divsteal`, `rankwhy`
and `velwhy`. Seven entries in all, all dated 2026-09-02. **The user
read all seven on 2026-09-02 evening** (session notes are in the
edgeprob entry's "Reading").

⭐ **2026-09-03: `edgeprob` = ref 55974245 = LB 0.932, NEW BEST (+0.007
over 55889500 = 0.925).** The official edge head's probability,
forward + reverse, as ranker columns (+BIDIR) with a D175-trained
ranker, on F4 primaries: **+0.0045 F4-6bba / +0.0162 F4-44b6 adj_J vs
the shipped D187 ranker, 44b6 12 of 12 datasets up**, both groups clean
holdouts, anchors exact. Entry just below the HANDOFF section. Kernels:
`biohub-edgeprob` v3 (head probs, GPU), `biohub-edgeprob-rank` v2
(ranker + eval, CPU; holds `edge_ranker_ep175_bidir.pkl`),
`biohub-submit-ep` v1 (`submit_ep.py`, generated by `make_submit_ep.py`).

`nopred` (2026-09-03, first entry below this HANDOFF): the
zero-velocity cells are mostly undetected parents and spurious nodes;
the pre-pass never leaves a free parent unlinked (0 in all domains).
Closes the velocity line (twopass, flowvel, nopred); residue in the entry.

`seed2` (2026-09-03/04, entry just below): the public 0.948 notebook's
second model (seed 314159) as a second edge head: −0.002 / −0.005 to
−0.010 vs the shipped +BIDIR, both groups negative in every arm; as a
fifth detector map: −0.02 to −0.05 zh F1. The head alone, no ranker,
bidirectional: 0.9245 / 0.8453, above the old coordinate ranker, 0.0016 /
0.0062 under ranker+head. Not submitted. Both official heads were trained
on all 199 train movies (their split manifest), so local head numbers are
in-sample for the heads.

⭐ `nb948-asis` (2026-09-05, entry below seed2): **the public "0.948"
notebook, forked unchanged, scores 0.935 with 204,966,451 bytes -- 24%
less output than our 0.932.** The 0.016 margin this record chased on
2026-09-03/04 does not exist in the public code; the margin is 0.003,
carried on a quarter fewer rows. The user's reading rule: below 270 M
= our rejection still has room. That is the open line. **Where the
24% goes, measured (2026-09-05, same entry): 90% of it leaves inside
the official ILP export** (appearance 0 / disappearance 2 / floor
0.48; only solution nodes are exported; −15% of raw on the visible 4,
−39% on 44b6_0b24845f, uniform across frames), the whole
post-processing accounts for 10%. Our ILP (disappearance 1.5, floor
0.10) keeps 23,371 of the official detector's nodes on that movie
where theirs keeps 18,618. **edgecmp (2026-09-05, entry below): edge
by edge on the visible 4 the notebook is 0.026 BELOW us (0.8952 vs
0.9214) while in-sample; the scorer sees 1.8% of our edges (0.2% on
44b6); on 44b6_0b24845f 0 of our 14,323 extra nodes and 0 of its 5,517
sit on an annotated cell. Its hidden margin is the node term (signed,
no upper clamp: 10% below the estimate = +1% of J; it is 43% below on
0b24845f), worth ~+0.02-0.025 on the hidden set against its ~0.02
edge deficit. Intersection −0.021, union −0.027: no merge. The room
is a deletion that carries counted edges at <1/10 the average rate;
LB is the only venue (submit-f48).** **divbench (2026-09-05, entry
below): divisions on 125 GT events out-of-fold, BASE primaries -- REF
54 / 258 / 71, div_J 0.141 (0.014 of score), patch worth +0.002,
tighter geometry −0.0195, divergence 1.0 / 0 −0.06 / −0.08 with TP
LOST; the geometry is a local optimum; the lever is 258 counted false
forks for 54 true; folds differ by 0.065 so ±0.03 is noise.**
**STATE AT THE 2026-09-06 STOP (the user: record the gate submission
when it lands, then stop; the rest next week).** **Best 0.943 = ref
56055405 (`submit_gate.py`, scored 2026-09-07): the fork gate at
p > 0.03 on the 0.933 pipeline, +0.010, bytes −0.04% -- the bench's
+0.012 transferred to LB.** Previous 0.933 = ref 56030644. Assets for next week:
F4 detections for all 128 6bba train movies (biohub-fusecache 12 +
biohub-fuse6bba 116), the head's F4 caches for all 128
(biohub-edgeprob 24 + biohub-ep6bba 116), `fork_gate_f4.pkl` and
`forks_f4.csv` in biohub-divbench3's output, the bench scripts
divbench.py / divbench2.py / divbench3.py (generated by make_*.py in
the session scratchpad -- copy them into tools/ before the scratchpad
is gone). Open lines, in order: (1) the gate's LB reading (below);
(2) if it transfers, the same bench for 44b6 (71 movies' F4 caches
are the missing piece, ~2 h GPU + ~1 h GPU) and a gate trained on
both domains; (3) the 5a steal case (23 of 71 lost divisions on F4,
the ILP gave the second daughter another parent) is the next FN
lever, priced on the same bench; (4) all local changes are
uncommitted (the user has not asked for a commit).

⭐⭐⭐ **divbench3 (2026-09-06, entry below): the fork gate on
PRODUCTION primaries (F4), pruning applied for real, official
scorer: +0.0121 of local score at p > 0.03 (div_J 0.134 -> 0.244,
adj_J +0.0010), both 6bba folds up (+0.0132 / +0.0096), the 44b6
draw +0.0020 with all 11 of its counted false forks gone. Anchors
exact. `fork_gate_f4.pkl` in biohub-divbench3's output; submit_gate
= submit_f48 + the gate at p > 0.03 is the next submission.**
⭐⭐ **divbench2 (2026-09-06, entry below): the false forks are
separable at fork time. A 2-fold gate trained on the forks the
pipeline emits (labelled by the scorer) reads AUC 0.92 / 0.90 out of
fold; cutting 70% of forks keeps 52 of 54 true, sheds 173 of 258
false, div_J 0.141 -> 0.248 = +0.011 of local score, both folds up
(the orphan daughter's detection score alone is AUC 0.885). Bench is
BASE primaries; production is F4 -- the F4 caches for the 128 6bba
train movies are the next build, then divbench3 on F4, then the
gate into submit_f48.** **submit-f48 = ref 56030644 =
LB 0.933, NEW BEST (+0.001), 266,263,495 bytes (−1.43%)**: on the
hidden set the floor removed 1.4% of our output (4.8% on the visible
4), so our hidden output carries almost no weak chains -- the
notebook's 24% is not available to us by deletion, and the byte
pattern holds (every byte-removing change gained, every byte-adding
one lost). **ilpsweep (2026-09-05, same
day, entry below): that lever, on our nodes, CLOSED** -- the notebook's ILP
settings score +0.0002 / −0.0115 / −0.0010, delete only 6-15% of our
nodes (not 39%), and the chains they delete are 62% / 91% / 58% true
edges where the scorer looks; looser (disappearance 1.0) loses 12 of
12 on 6bba. REF is the local optimum of that knob.

`zhoff` / `zhoffrank` (2026-09-04, entry below seed2): the official
detector's nodes on the 8 Zebrahub eval crops through our D187 linker
score 0.49 (0.965) / 0.40 (0.99) against F4's 0.81, 8 of 8 crops down,
under SHORT_MIN 4 and 6 alike; at its production gate it UNDER-detects
zebrahub by 27-41%, worst on dim crops, so the "over-detect then
delete" interaction cannot be tested there. SHORT_MIN 6 never pays.
Fourth clean loss for that detector. The interaction on the hidden
embryo (where it over-detects: +3.7% bytes) is LB-only: submit_off with
SHORT_MIN 6, one submission, untried. For detector questions zebrahub
overstates the shift relative to the hidden test; for linking /
post-processing it stays the honest venue.

`gap2strict` (2026-09-04, biohub-gap2 v3, entry below seed2): the 0.948
notebook's 3-frame bridge under its own four conditions makes 709
bridges on the 24 sets (span2 made 35,711) and moves +0.0005 / −0.0017
/ +0.0014, signs disagree: zero. The head's probabilities are LESS
polarised than our ranker's (true ≥ 0.9: 72% vs 92% on 6bba), so
"their edge scores suit post-processing better" is false on our nodes.
Of the 0.948 notebook, what remains untested is its detection (dual-seed
logit blend at gate 0.965 with candidate retention) and its short-track
rule (min 6 with rescue); its DeepCenter veto runs on the dead
checkpoint_last.pt.

`zhrank` (2026-09-04, entry below seed2): the 18-column ranker trained
on Zebrahub loses 0.010 / 0.012 to the D175-trained one on 6bba / 44b6
and wins 0.0014 at home (the user's 水土不服 fear, measured); mixing the
pools halves the loss, still negative. On zebrahub itself 218 k sparse
competition rows rank as well as 1.8 M dense rows (AUC 0.9960 both).
Re-normalising the shipped ranker's probabilities over competitors
(NORM / MULT) loses 0.002 / 0.006, a redistribution across movies.
Residue: 44b6_0b24845f jumps +0.07 under unrelated changes, unexplained.

`motionlink` (2026-09-04, entry below seed2): the 0.948 notebook's
velocity-Hungarian as our final linker loses 0.010 / 0.033 to the
ranker + ILP (23 of 24 movies down); the two agree on 96-97% of links
and the ILP's refusals decide the rest. The linker is not where that
notebook's margin comes from; its detection blend and post-processing
remain untested.

`p1head` (2026-09-04, entry below seed2): the greedy pre-pass with the
edge head's probability as its cost, the user's "贪心乱连" question. Same
number of wrong parents as the distance cost (188 vs 183 on 6bba, 146 vs
156 on 44b6), 3-6x more cells left without one; pipeline −0.001 / +0.003,
signs disagree, the 44b6 side is one movie. Zero. Residue in the entry.

`offdet` (2026-09-04, entry below p1head): the OFFICIAL detector's nodes
(gate 0.99, flip TTA) linked by the official head alone through our ILP
and patches: local 0.9297 / 0.9089 vs the shipped 0.9261 / 0.8515,
+0.004 / +0.057, but the official detector trained on all 199 train
movies with the sparse GT as target (the user's catch), so the local win
is not evidence. **SUBMITTED as ref 56000581 (2026-09-04, `submit_off.py`,
kernel `biohub-submit-off` v1), control 55974245 = 0.932: LB 0.895,
−0.037, the worst on record**, with +3.70% more hidden output bytes
while the visible-4 had 3% FEWER nodes than F4. The local win was the
training-set overlap; the official detector line is closed at LB, twice
now (0.898 on 2026-08-22 with its own linker, 0.895 with ours). Residue
in the entry.

⚠ **DO NOT ask the user to set the final selection** (the user,
2026-09-01). There are two slots and Kaggle already defaults them to the
highest-scoring submissions. Selecting manually only pins the choice, so
a later higher score has to be remembered and updated by hand. Leaving
it on the default is strictly safer while the competition is open.

⚠ **The daily submission limit is FIVE, not two.** `max_daily_submissions:
5`. Days of rationing at two were based on confusing the daily cap with
the final-selection cap.

⚠ Read 55772253 against **55757064 = 0.920**, not against 0.921. It
differs from 55757064 by one non-comment line. Calling it "0.921 →
0.920" invents a regression that did not happen.

### The single most important open question

**Does a three-domain-agreeing result appear on the leaderboard?**

**ANSWERED 2026-08-28: yes, once — and the SIZE of the local delta was
not what predicted it.** The holdout is 1 for 4. The four failures
(windowing +0.0019, ranker pool +0.0073, `SHORT_MIN` +0.0025/+0.0054/
+0.0045, gate 0.12 +0.0055/+0.0045/+0.0121) were all CONSTANT TUNING
where the local number was the whole argument. The one success had the
SMALLEST weakest-domain delta of the five (+0.0013) and the only
measured MECHANISM: `wide44` proved the repair budget bound in 30 of 83
datasets and left 64% of demand unrepaired. See the 0.922 entry at the
top.

**And as of `thr5` there is no longer a mechanism excusing that.** The
node-count term is linear with slope `-0.1 * dN/N_true`, independent of
the current ratio, so local and hidden pay the same rate and the local
deltas were already net of it. The asymmetry the 2026-08-26 node-count
entry proposed does not exist. What remains is the plain reading: **a
+0.002 to +0.007 local gain, however many domains agree, is not
evidence that the hidden test will move.**

The live candidate is `gap4`'s **`nocand_cap20`**, SUBMITTED as ref
55817630 (+0.0013 / **+0.0085** / +0.0056 — the 44b6 figure is
`wide44`'s, over all 71 datasets; gap4's +0.0113 came from a
twelve-dataset prefix and overstated by a third): drop the candidate requirement AND raise `GAP1_CAP_FRAC`
0.003 -> 0.02. No gate of any kind. It supersedes `len8`, which ties it
on the weakest domain (+0.0013) but is 2.6x worse in 44b6 and **3x
worse in Zebrahub, the only dense-GT domain**, whose rates describe all
cells rather than annotated ones.

⚠ **gap3 and thr5 both ran on D187, and only ONE of them has a usable
44b6 arm.** Every 44b6 dataset is inside D187's training pool, so the
question is whether the leak cancels in the delta:

- **gap3 is a POST-LINK sweep.** All eight arms share one ranker, one
  detection set and one ILP solution, and differ only in what
  `_gap_close` does afterwards. The in-sample optimism inflates every
  arm identically and cancels in the delta. **Three usable domains.**
- **thr5 changes the DETECTIONS.** Each gate feeds the ranker a
  different candidate set, so the in-sample bias differs per arm and
  does NOT cancel. Its own log said so at startup. **Two usable
  domains: 6bba and Zebrahub.**

### What 2026-08-25/26 closed, and what it opened

| line | verdict |
|---|---|
| post-link constants (14 changes) | **one survivor**: `SHORT_MIN` 6 → 4 |
| `SHORT_MIN` curve, 1→7 | a curve, not a spike; deleting length-4 and length-5 tracks loses in **all three** domains |
| NMS radius / density | **closed.** 2 of 3 domains, none significant |
| detector depth | **closed.** signs disagree, nothing significant |
| gap closing, the SPAN | **RE-CLOSED 2026-08-28 by `span4`**, this time at a budget that does not bind. span 2 fails the three-domain rule (6bba −0.0014), its 6bba `edge_J` equals BASE's while adding 1.4% nodes, and it costs −0.0173 of `div_J` there. The reopening was correct; the answer did not change |
| gap closing, `GAP1_GATE_UM` 9 → 12 | **closed 2026-08-28.** Passes the rule but is below `cap20` in all three domains: 28% more bridges, slightly worse everywhere. 9.0 is right |
| **the gap-closing BUDGET** | **SHIPPED. `GAP1_CAP_FRAC` 0.003 -> 0.02 with no gate at all: +0.0013 / +0.0085 / +0.0056 locally and **LB 0.920 -> 0.922**, ref 55817630. The only knob in this line that ever mattered** |
| gap closing, a LEARNED gate | **closed 2026-08-28 by `rank5`.** The two-frame ranker scores 0.9693 AUC against distance-alone 0.8829, and rejecting on it still loses in all three domains, monotonically. FIVE gates now, none selecting. `_short_filter` is why: no bridge deletes the fragment WHOLE, a wrong bridge keeps its correct edges |
| detection gate, LOOSER (0.12) | **tried.** Local +0.0055/+0.0045/+0.0121, LB 0.918 against control 0.920. Reverted |
| detection gate, TIGHTER (0.18–0.26) | **closed 2026-08-26 by `thr5`.** Monotone loss in both independent domains; Zebrahub −0.2204 at 0.26. It deletes real cells, not marginal false positives |
| the node-count term as an explanation | **closed 2026-08-26 by `thr5`.** Linear, constant slope, no local/hidden asymmetry |
| morphology | closed 2026-08-25 (`MASS_ratio` median 0.994) |

**The flicker finding, and BOTH routes out of it, now closed.**
`flicker.py` found that **no annotated cell is permanently invisible**
— 0.0% of 379 GT tracks are never detected. ⚠ **The other two figures
here were the 5.0 µm ones and are corrected below**: at the scorer's
7.0 µm radius, 6bba tracks intermittent is **14.2%** (not 28.8%), there
are **95** dropout runs (not 196), and **34 are longer than the one
frame `_gap_close` can bridge** (not 94) — about 36% of dropouts, not
48%. The absolute number of unreachable multi-frame gaps on the 12
holdout datasets is **34**. There are exactly two ways to recover a
multi-frame dropout, and **both have been run**:

| route | what it does | run | verdict |
|---|---|---|---|
| **GUESS the position** | interpolate 2 nodes across the gap | `span4` | 6bba **−0.0014**; its `edge_J` equals BASE's while adding 1.4% nodes, and `div_J` **−0.0173** — a longer bridge eats real daughters. **Closed** |
| **MEASURE the position** | a second detector supplies the missing peak | `dcmerge` v2 | zh **+0.0379**, 6bba **−0.0413**, monotone in both. **Closed as an unconditional add** |

⚠ **A correction, 2026-08-28.** This block previously said the losses
were "currently unaddressed by anything that has been tried", citing
only `gap2`. That was wrong: **span 2 IS the repair for the >1-frame
dropouts and it was measured and rejected**, and `dcmerge` then measured
the other route. What `gap2` adds is only why a cheaper fix is
unavailable — at 79.8% of missed frames the heatmap response *vanishes*
rather than dipping below threshold, so no threshold recovers it.

**The losses are real and identified. What is left open is narrow: a
CONDITIONAL foreign detection, admitted only where our own detector is
weak, which needs a test-time weakness signal nobody has proposed.**

**ANSWERED 2026-08-29, and the answer is null: the conditional version
was run (`dcpool`, epoch-5 retrained DeepCenter into the repair pool,
the track break itself as the weakness signal) and moved nothing --
20% of bridge inserts took measured positions and adj_J was unchanged
to the fourth decimal in all three domains. Bridge POSITION was never
the loss; see the dcpool entry. The multi-frame-dropout losses remain
real and remain unaddressed by every route now tried.**

### Read these first

1. **`biohub-div-train` v3 ANSWERED (2026-08-24): yes.** Zebrahub only
   0.8083 on held-out 6bba, 44b6 only (18 pos) 0.4295. The external-data
   line is open. Details and the caveat about the mixture sweep are in
   open item 1 below.
2. **Windowing is REVERTED. Use the global solve.** Three LB points:
   global 0.921, windowed 0.920, windowed-with-seam-constraint 0.920.
   The runtime worry that motivated windowing never materialised -- the
   global solve COMPLETED on the hidden test. Do not re-introduce it
   without evidence that the 12 h cap actually binds.
3. **`total_bytes` in the submission metadata is the only measurement of
   behaviour on the HIDDEN test.** Check it on every submission. It is
   what revealed that the windowed run emitted 10% more output there
   while emitting 2% LESS on the visible 4.

### The three big facts this session established

- **Divisions cannot be learned from competition data.** 151 GT
  divisions in the entire 199-dataset train split, from only 2 embryos
  (26 in 44b6, 125 in 6bba). **124 of them land on detections.**
- **The linker is finished.** The solver already takes every division
  the candidate set contains; widening the gate (isotropic and
  z-anisotropic, both properly swept) only moves divisions from
  "outside gate" to "below floor" and never raises REACHABLE past 5.
- **Zebrahub is another embryo domain, not another problem.** Every
  geometric gap between it and the competition is the size of a
  gap between two embryos within either dataset.

### Assets now in place

| what | where |
|---|---|
| xl detections, ALL 199 train datasets | `biohub-cache-train` (65 MB, 65 s/dataset) |
| xl detections, 41 Zebrahub crops + labels | `biohub-zh-detect` (26 MB) |
| 468,617 validated Zebrahub divisions | `biohub-zh-divisions` (counts only) |
| trained edge ranker | `biohub-ranker-fine` -> `edge_ranker.pkl` |

### Open, in rough order of expected value

1. **PARKED 2026-08-24, user's call. Every CHEAP route is spent; the
   remaining ones all need pixels.** Do not re-run the light experiments.

   What was settled, in order: div-train (Zebrahub alone carries it,
   held-out 6bba AUC 0.808 vs 0.430 for 44b6's 18 positives, below
   chance); div-kinematics (the DAUGHTERS' future is real signal --
   `sep_slope` 1.155 vs 0.082 um/frame, a 14x gap -- while the mother's
   pre-division slow-down does NOT exist in this data, `decel` 0.905 vs
   0.934); div-gate (cannot audit already-selected forks; pruning is
   monotonically worse than doing nothing); div-forks (cannot select
   them either -- rerank buys 1 of 23 removable FPs, generate is worse).

   **Not "unreachable" -- "the model is not good enough".** ~+0.027 sits
   in the rerank direction and the model captured 1/23 of it.

   Why the rest is expensive: **every feature ever built for this reads
   `detcache_*.npz`, i.e. coordinates and detection scores. Nothing reads
   pixels.** So "the mother got smaller / rounder / brighter just before
   dividing" -- the obvious remaining signal, and the one mitotic
   rounding actually produces -- is not expressible at all. Getting it
   means re-reading the raw zarr volumes per candidate, the `det_ablate`
   cost profile (minutes per dataset), not a cheap CPU pass over caches.
   `division_features.py` exists for exactly this question and **has no
   recorded result anywhere** -- written, never run, or run unlogged.

   The other untested lever is data volume: `biohub-zh-divisions`
   validated **468,617** divisions; training uses **2001**, because only
   41 crops were extracted. `PER_CROP = 400` is not binding, the crop
   COUNT is. That one IS cheap -- `zebrahub_extract.py` already exists --
   but nobody has measured whether AP scales with it.
2. **Explain the 10% output gap** between global and windowed on the
   hidden test. The seam hypothesis is DISPROVEN (below). One untested
   idea: windowing charges `disappearance_weight` once per window per
   track instead of once per track, biasing toward over-linking.
3. `biohub-zh-pipeline` died of OOM after 10 of 41 crops and its
   `div_J` came out nan. Its 9 completed crops already answered the
   transfer question; finishing it is optional.
4. **CLOSED, negative**: scale-invariant features for the edge ranker do
   nothing (entry below). The -0.73 density penalty is NOT in the
   ranker's feature space -- per-dataset AUC is flat against density
   while adj_J is not.

## 2026-09-03/04 -- seed2, CLOSED negative: the second official model (pilkwang seed 314159) adds nothing as a second edge head (−0.002 / −0.005..−0.010) or as a detector map (−0.02..−0.05 F1); the head WITHOUT a ranker is 0.002 / 0.006 under ranker+head

The user, 2026-09-03, after edgeprob's LB 0.932: "都去做吧，不过为什么只是
拿来做一个特征呢". Three kernels, pre-registered here before any result
is read.

What the public 0.948 notebook (rishabhr0y, post-patch) actually does
with its second model, read from its source this session: the same
architecture as the official split_0 weights, trained by pilkwang with
seed 314159 (`pilkwang/biohub-temporal-unet3d-seed314159-v1`, edge head
sha256 9bac2fa0...). Fused in TWO places: (a) detection -- the
secondary's heatmap logits re-centred and re-scaled to the primary's
per-frame mean / std (ratio clamped 0.5-2), then
0.20 * primary + 0.80 * secondary ("our swept detection-fusion peak, was
0.475"); (b) edges -- the secondary's edge logits aligned the same way and
blended at 0.20 in mode `low_margin_consensus` (the branch body was not
read; by name and by its 0.35 parameter, the second model speaks where
the first's top-1 / top-2 margin is small). Then the bidirectional
harmonic fusion on top. NOT equal-weight averaging.

1. **`edgeprob2_cache.py` -> `biohub-edgeprob2` v1 (GPU T4, pushed
   2026-09-03).** edgeprob_cache.py v3 with the seed-314159 weights and
   the output prefix `edgeprob2_`; 199 BASE + 24 F4 datasets, both
   directions, same (t, i, j) rows. Prints which of our 24 evaluation
   datasets the second seed's split manifest lists (leakage check, the
   same question the primary head carries). Expected ~2.8 h.
2. **`edgeprob_rank2.py` -> `biohub-edgeprob-rank2` (CPU, pushed when 1
   completes).** v2's harness, D175 pool, F4 evaluation, 36-column join
   (23 harness + head 1's 5 + head 2's 5 + p_avg_fwd / p_avg_rev /
   h_avg). Arms: ANCHOR_D187 (hard 0.9216), +BIDIR (v2's ranker retrained,
   soft anchors 0.9261 / 0.8515, div_J 0.1515 / 0.0000), +BIDIR2 (second
   head alone), +BOTH, +AVG, and two arms with NO RANKER: HEAD1 (head 1's
   h_fused straight into the ILP as p) and HEAD12 (h_avg). Reference for
   every paired delta: +BIDIR (what ships). Pre-registered readings: a
   ranker arm above +BIDIR in both groups at >= +0.003 earns a submission
   with both heads inline; a head-only arm at or above +BIDIR says the 18
   coordinate columns add nothing once the head is bidirectional (the
   2026-08-22 chimera lost 0.011 single-seed, single-direction).
3. **`detfuse3.py` -> `biohub-detfuse3` v1 (GPU T4, pushed 2026-09-03).**
   detfuse2's measurement (zh dense-GT F1 at 5 / 7 um, stride-10 frames;
   24 competition sets, sparse) with the two official detection heatmaps
   (pack split_0 = off_p, seed 314159 = off_s) as maps 5 and 6, run the
   official way (strided [::4], zarr-attr quantiles, 2-frame window,
   4-view TTA, sigmoid) and cropped onto our pooled grid (0.6 um offset,
   inside the refine window). Arms: FULL3 / F4 (anchors 0.7774 / 0.7903),
   F5p, F5s, F6, F6h, OFFp, OFFs, OFF28 (the notebook's 0.2 / 0.8 in
   probability space). Cold-miss rescue against F4's misses. A winning
   arm buys fusecache-style caches + both heads on the new primaries +
   a three-domain pipeline test, per detfuse2's protocol; detection F1
   alone decides nothing (DCheavy won detfuse2's F1 and F4 shipped).

Results, as they land (item 3 first; it ran in 27 min):

### detfuse3 results (v3, 26.6 min T4, 2026-09-03) -- NEGATIVE on the honest denominator: every blend with an official map is below F4, and the official detectors alone are below the shipped three-branch mean

Anchors digit-exact: zh FULL3 0.7774, F4 0.7903 (F1* at 7 um).
(v1 and v2 died on a constant used before its definition; nothing ran.)

zh, dense GT, 101,615 points over 410 frames, 7 um, own best gate:

| arm (b0,b1,b2,dc,off_p,off_s) | gate* | dets/fr | recall | prec | F1* | F1@.15 |
|---|---|---|---|---|---|---|
| FULL3 (1,1,1,0,0,0) | 0.15 | 260.9 | 0.7979 | 0.7579 | 0.7774 | 0.7774 |
| **F4 (1,1,1,1,0,0) ships** | 0.15 | 257.7 | 0.8060 | 0.7752 | **0.7903** | 0.7903 |
| F5p (+ official primary) | 0.30 | 250.5 | 0.7725 | 0.7642 | 0.7683 | 0.7484 |
| F5s (+ second seed) | 0.15 | 310.5 | 0.8444 | 0.6740 | 0.7496 | 0.7496 |
| F6 (+ both) | 0.20 | 312.1 | 0.8370 | 0.6646 | 0.7409 | 0.7381 |
| F6h (+ both at 0.5) | 0.20 | 297.0 | 0.8335 | 0.6955 | 0.7583 | 0.7534 |
| OFFp (official primary alone) | 0.60 | 327.7 | 0.8248 | 0.6237 | 0.7103 | 0.7069 |
| OFFs (second seed alone) | 0.40 | 319.7 | 0.7827 | 0.6067 | 0.6836 | 0.6813 |
| OFF28 (0.2 / 0.8, the notebook's mix) | 0.15 | 352.3 | 0.8427 | 0.5927 | 0.6960 | 0.6960 |

At 5 um the order is the same (F4 0.7685; F5p 0.7458, F5s 0.7247, F6
0.7154, F6h 0.7353; OFFp 0.6787, OFFs 0.6526, OFF28 0.6661).

- The official detector, either seed, is **0.08-0.11 F1 below the
  shipped F4 on the only all-cells denominator**, and 0.07-0.09 below
  the three-branch mean it was meant to help. It buys recall (OFF28
  0.8427 vs F4 0.8060) with 37% more detections per frame and a
  precision of 0.59.
- Blended in, it drags every arm down: F5p −0.022, F5s −0.041, F6
  −0.049, F6h −0.032 against F4. Half weight is not enough to undo it.
- Cold-miss rescue against F4's 16,226 zh misses: window-max >= 0.15
  within 3.25 um in the 4-map mean 85.6%, off_p 93.0%, off_s 86.7%,
  any 98.1%; at >= 0.05 the 4-map mean already reaches 99.6%. So F4's
  misses are not cold in the heatmap: the map responds and the peak /
  NMS stage loses them. An extra map that responds there does not
  make them peaks; it makes more peaks elsewhere.
- Competition side (sparse GT, 12 + 12 datasets, stride 10): as in
  detfuse2 the annotated-only precision makes F1 unreadable, but
  dets/fr is honest: the official maps alone produce 168.8 / 368.8
  detections per frame on 6bba / 44b6 against F4's 48.9 / 26.1, and
  recall over annotated cells 0.996 / 1.000 against F4's 0.698 / 0.242
  at F4's own best gate 0.40. Adding one official map (F5p / F5s)
  lifts dets/fr 48.9 -> 63.9 and 26.1 -> 41.2. Surplus, the node-count
  penalty's food; the no-surplus criterion (detfuse2) is failed.

Caveats, both of which cut the same way and neither of which is
tested: (a) on zh the official maps were normalised by per-frame
0.001 / 0.999 quantiles, not the dataset's (the crops carry none);
(b) the blend is equal-weight in probability space, where the 0.948
notebook aligns logits to the primary's per-frame mean / std and
weights 0.2 / 0.8. OFF28 alone at its own best gate is already 0.094
below F4, so an alignment that changes only the blending cannot be
expected to cross that; it is residue, not a next step.

Killed claim: "the official detection heatmaps, added to the four the
pipeline averages, surface the undetected parents nopred charged to
the detector". Residue: the peak / NMS stage loses cells the maps
respond to (85.6% of F4's zh misses have a >= 0.15 response within
3.25 um), which is the NMS / peak line the 2026-08-25 nms4 entry
closed ("2 of 3, none significant"); and the logit-aligned blend,
untested. No pipeline test
is bought; per detfuse2's protocol the line stops here.


### detfuse3 v4 (2026-09-04, 25.6 min T4): the full gate curves -- on zebrahub the official detector loses at equal detections; on the sparse competition sets it recalls MORE annotated cells at equal or fewer detections

Same run with gates extended to 0.99 and arms FULL3 / F4 / OFFp / OFFs
/ OFF28 (the F5/F6 blends were settled by v3). Anchors exact again.
The official detector's sigmoid is saturated: on zh its dets/fr falls
only 366 -> 250 from gate 0.05 to 0.99, so v3's "own best gate" rows
compared it at 320-350 dets/fr against F4's 258. At EQUAL dets/fr (7 um):

| domain | F4 (gate 0.15, ships) | official primary at the gate that matches F4's dets/fr |
|---|---|---|
| zh, dense GT | 257.7 dets/fr, recall 0.8060, prec 0.7752, F1 0.7903 | gate 0.98-0.99: 250-269 dets/fr, recall 0.688-0.723, F1 0.684-0.694 |
| 6bba, 1,047 annotated cells | 137.9 dets/fr, recall 0.9761 | gate 0.99: 151.6 dets/fr, recall 0.9943 (+10% dets, +1.8% recall); F4 at gate 0.05 reaches 150.9 / 0.9885 |
| 44b6, 297 annotated cells | 310.1 dets/fr, recall 0.9596 | gate 0.99: 282.7 dets/fr, recall 1.0000 (−9% dets, +4.0% recall); gate 0.965: 315.1 / 1.0000 |

On zebrahub the verdict stands: at the same number of detections the
official detector finds 8-12% fewer cells. On the competition domains
the sign flips: the official detector at 0.99 recalls every one of the
297 annotated 44b6 cells with 9% FEWER detections per frame than F4,
and 0.994 of the 6bba ones at +10% detections. Small n (297 / 1,047
annotated points on stride-10 frames), annotated cells only, precision
unreadable -- but recall over annotated cells is exactly what the
scorer counts, and 44b6 is where the pipeline is weakest (0.85) and
where edgediag charged 7% of GT edges to detection misses.

The record's standing verdict on this detector (2026-08-22 Branch C,
corrected 08-24): the temporal UNet3D LOST to xl on zebrahub (0.650 /
0.637 vs 0.766 / 0.736 recall / precision, dim-quartile recall 0.52 vs
0.73) and on LB (official family 0.898 vs xl family 0.917). That LB
comparison was official-detector + official single-direction head vs
xl + our ranker: two things changed at once. What has never been
measured is the official detector with the linker held fixed -- and
rank2 just showed the bidirectional head alone links within 0.002 /
0.006 of ranker+head, so the linker CAN be held fixed without retraining
anything: official detections at 0.965 / 0.99 -> bidirectional head ->
our ILP and patches, against F4 -> the same. That is the pipeline test
this curve buys (offdet, below). Killed claim, narrowed: "the official
detector is worse than F4" holds on zebrahub at equal detections;
on the competition domains' annotated cells it is not established.

### nb948-asis (2026-09-04/05): the public "0.948" notebook forked UNCHANGED scores LB 0.935 with 204,966,451 bytes -- 24% LESS output than our 0.932, and NOT 0.948

The user, 2026-09-04: "其实我的意思是你把公开notebook整个提交上去看看最终文件大小".
`rishabhr0y/biohub-948-sew20` pulled with its metadata and pushed as
`faustine649/biohub-nb948-asis` v1 (notebook, T4, no internet; sources
pilkwang/biohub-deepcenter-unet3d-center-prior-v1,
pilkwang/biohub-temporal-unet3d-seed314159-v1,
pilkwang/biohub-tracking-support-pack-50ep-v1; its own docker image
dropped). Its guard cells validate the produced submission and record
a receipt; nothing aborts a plain run. What the submission's
`total_bytes` reads against ours: F4 pipeline (ref 55974245, 0.932)
270,137,616; official nodes + our patches at SHORT_MIN 4 (ref 56000581,
0.895) 280,141,150. The notebook's bytes far below 280 M = its
post-processing deletes a large share of the official detector's
hidden-embryo surplus (the interaction, measured where it happens);
near or above 280 M = it carries as many nodes as our official-node
run and still scores 0.948, so the difference is in which nodes and
edges, not how many. Its LB should reproduce ~0.948 (public score of
the source notebook); if it does not, the fork is not the scored
version and the bytes do not apply. The SHORT_MIN 6 probe below was my
misreading of the request; its kernel runs, its submission is on hold.

**The user's reading rule (2026-09-04), the one that matters, against
OUR 270,137,616 (F4, 0.932):** the notebook's bytes BELOW 270 M = a
0.948 pipeline carries fewer nodes and edges than ours, so our
rejection still has room ("否定还有更大的空间": the deletion side is
where to look); ABOVE 270 M = a 0.948 pipeline carries more than ours,
so conditional loosening of nodes ("有条件的放宽节点") remains viable
despite M15 (−0.008) and the gate 0.12 loss (−0.002). The 280 M
comparison (official nodes under our patches) is the secondary
reading on the interaction.

**Kernel v1 landed (GPU, ~1.4 h with its own validator), submitted as
ref 56014734 (2026-09-04).** Visible-4 output: 234,871 rows, 119,517
nodes, 115,354 edges -- FEWER nodes than our F4 pipeline's 128,157
(ref 55889500) and than official nodes under our patches (124,334 /
121,524 at SHORT_MIN 4 / 6). Per movie, its stages: 44b6_0b24845f
20,236 nodes after motion relink -> 20,591 after safe divisions ->
18,618 after the short-track filter (450 components removed);
6bba_05db0fb1 69,829 -> 70,278 -> 69,359. Its retention guard fell
back to the primary seed on most frames of 44b6_0b24845f (blend keeps
79-89% of the primary's peaks there). Its pipeline manifest: dual seed
loaded, bidirectional 0.15 harmonic, **DeepCenter veto loaded from
`weights/full_frame_center/best.pt`** (the preset's checkpoint_last
path is overridden by its default; so the veto runs on the live epoch-2
weights, and my earlier "dead checkpoint" note in the offdet / gap2
entries is wrong -- corrected here). Its own validator, 4 TRAIN movies
(in-sample for its detector): adj_J 44b6_12dfb391 0.9352, 44b6_267148e4
0.8107, 6bba_062c8d37 0.9832, 6bba_07e24132 0.8311.

**LB 0.935, total_bytes 204,966,451 (scored 2026-09-05).** Two facts,
both bigger than the question asked:

1. **The forked code does not score 0.948.** It scores 0.935, +0.003
   over our 0.932. The notebook's own markdown calls its base "946"
   and its proposal "ACCEPT iff LB > 0.936, else REVERT"; whatever
   scored 0.948 on that account is not the code that is public (a
   different version, or a different notebook). Every "0.948" in this
   record's entries of 2026-09-03/04 refers to a margin of 0.016 that,
   for the code we can read, is 0.003. The parts we tested (linker,
   3-frame bridge, division rule, polarisation, detector) were tested
   against a phantom margin; the tests stand, the target does not.
2. **It carries 24.1% fewer bytes than we do on the hidden embryo:
   204,966,451 vs 270,137,616 (ours, F4) and 280,141,150 (official
   nodes under our patches).** On the visible 4 it had only 6.7% fewer
   nodes than F4 (119,517 vs 128,157); on the hidden embryo the gap is
   3.6x larger. So its post-processing deletes far more of the
   official detector's hidden-embryo output than of its visible-4
   output (280 M -> 205 M is −27% on the same detector), and the
   result scores 0.003 ABOVE our 32%-larger output. Under the user's
   reading rule (below 270 M): **our rejection still has room** -- a
   pipeline with a quarter fewer rows is not worse. Where our surplus
   is (the ILP keeps a node it links; the short filter at 4 keeps
   fragments of 4+; the gate 0.15 admits the sub-0.965 tail the
   official detector never sees) is now the question, and the three
   LB data points on our own side agree with the sign: every
   submission that added bytes lost or stalled (M15 +3.7% −0.008; gate
   0.12 +3.9% −0.002; W1101 +1.0% −0.001) and the only one that
   removed bytes gained (edgeprob −0.2% +0.007). Caveat: fewer rows
   can also mean fewer true edges; their 0.935 does not say our rows
   are wrong, it says 24% of ours are not needed for 0.935.

Residue for the next session: (a) which of our rows the scorer does
not need cannot be read from bytes; a local proxy is node_r, which is
−0.02 / −0.11 on 6bba / 44b6 (we are BELOW estimate there), so the
hidden embryo's estimate-vs-output relation differs from both local
groups and only LB can price a deletion; (b) the cheapest LB probe of
"more rejection" on our own stack is the one lever never tried at LB
in the tightening direction: the detection gate above 0.15 (thr5
closed 0.18-0.26 locally, monotone loss on zh and 6bba, but zh
overstates detector-side shifts and 6bba is the memorisation domain),
or a higher ILP floor / disappearance cost; (c) the SHORT_MIN 6 probe
(submit_off v2, on hold) would remove ~2% of official-node bytes,
nowhere near the notebook's −27%, so it is not the deletion that
matters.

Its run_stats.csv (visible 4: 44b6_0113de3b / 44b6_0b24845f /
6bba_05b6850b / 6bba_05db0fb1), the user's question "它的DeepCenter否定是
怎么样的": the veto is add-only (never touches detector nodes or linker
edges), scored as the max of DeepCenter's sigmoid (best.pt, epoch 2;
checkpoint_last is rejected by its epoch check) in a z±1 / yx±2 window
on the 4x-pooled grid. One-frame bridges: short ones bypass (40 / 214
/ 17 / 317), synthetic middles of bridges ≥ 8.5 um are checked (12 /
109 / 3 / 107) and REJECTED at 0.25 in 3 / 90 / 1 / 89 cases -- 25% /
83% / 33% / 83% of long bridges refused. Safe divisions: geometric
candidates 92 / 91 / 9 / 166 (after the divergence rule had already
refused 217 / 280 / 32 / 1088 and mutual-NN 13 / 9 / 3 / 77), DeepCenter
refuses 11 / 32 / 0 / 67 more at 0.12, 67 / 53 / 9 / 90 added. Bridges
made 49 / 233 / 19 / 335 (all synthetic; reuse of isolated nodes 0
everywhere), gap2 12 / 61 / 3 / 57, short-track removal 261 / 1,954 /
174 / 892 nodes. None of these is a quarter of the output; the −24%
is upstream of all of them. (My first guess here, "the relink leaving
nodes isolated, then pruned", was wrong: prune-isolated removes 6 / 19
/ 1 / 27 nodes. The measured answer is the block below.)

**Where the 24% goes (2026-09-05, measured, the user: "所以这24%在哪里没的，
你去查呗"). It goes at ONE place: the official ILP. Its solver returns
only the nodes it linked, and that is what gets exported.** Read
from the fork's own output (`detector_coordinates_*.jsonl` = the
notebook's pre-graph, pre-ILP coordinate manifest, "rows" +
per-frame counts; the ILP-exported `.geff` under
`tracking_repo/predictions/unknown/unet_transformer/split_0/`
(zarr v3, read by hand: nodes/ids, nodes/props/solution,
edges/props/solution, edge_prob); `run_stats.csv` for every later
stage). Visible 4, order 44b6_0113de3b / 44b6_0b24845f /
6bba_05b6850b / 6bba_05db0fb1:

| stage (the notebook) | nodes | vs raw |
|---|---|---|
| raw detections, gate 0.965, dual seed, before the graph | 26,532 / 33,336 / 7,370 / 76,396 = 143,634 | — |
| ILP export (the .geff): only nodes in the ILP solution | 25,678 / 20,236 / 6,206 / 69,829 = 121,949 | −3.2% / **−39.3%** / −15.8% / −8.6% = **−15.1%** |
| after relink + bridges + safe-div + prune + short filter 6 | 25,484 / 18,618 / 6,056 / 69,359 = 119,517 | a further −2.0% |

Every node in the exported geff carries solution = True and has at
least one edge; isolated nodes 0 / 0 / 0 / 0. So of the 24,117 rows
the notebook loses between raw detection and its final file, 21,685
(90%) leave inside the official ILP export, before any of the
post-processing this record spent a week decomposing; the whole
post-processing (prune-isolated 6 / 19 / 1 / 27, short-track 261 /
1,954 / 174 / 892, minus bridges added) accounts for the other 10%.
Per frame the deletion on 44b6_0b24845f is uniform, not local: kept
fraction 0.64 in frames 0-9, 0.65 / 0.63 / 0.58 / 0.58 in the middle
bands, 0.54 in frames 90-99; worst single frames 295 -> 121 and 465
-> 201. It is a property of the movie: the solution edges' median
probability is 0.842 there against 0.975 / 0.911 / 0.904 on the other
three.

Why the ILP deletes that much: appearance 0, disappearance 2, edge
cost −p (p = the harmonic forward/reverse probability, candidates
only above 0.48). A track's cost is 2 − Σp, so a track survives only
if the sum of its edge probabilities exceeds 2: no track of 3 or
fewer nodes can survive at all, a 4-node track needs every edge
above ~0.67 on average, and a longer chain of weak edges is dropped
whole. Ours: appearance 0, disappearance 1.5, floor 0.10 (submit_off
/ F4 alike): a 3-node track with two edges above 0.75 survives, and
the short filter then removes components under 4 nodes. Like for
like (official detector nodes, our ILP + patches, submit_off v1 = ref
56000581): raw 25,810 / 29,736 / 7,226 / 72,311 = 135,083 at gate
0.99 -> final 25,409 / 23,371 / 6,519 / 69,035 = 124,334, i.e. −1.6%
/ −21.4% / −9.8% / −4.5% = −8.0%, against the notebook's −3.9% /
−44.2% / −17.8% / −9.2% = −16.8% from a looser gate (0.965). On the
dense movie we keep 23,371 of the official detector's nodes, it keeps
18,618. F4 final for reference: 24,416 / 27,424 / 6,212 / 68,436 =
126,488 (ref 55974245's kernel).

What that says about the hidden bytes: official nodes at 0.99 + our
ILP = 280,141,150; official nodes at 0.965 + its ILP = 204,966,451
(−27%). The gate went DOWN (more raw nodes) and the file still shrank
by a quarter, so on the hidden embryo its ILP rejects an even larger
share than the −27% shows, and the hidden embryo behaves like
44b6_0b24845f (weak edges, dense), not like the other three. This is
the concrete "room in our rejection" the user's rule pointed at: the
lever is the ILP's disappearance cost / candidate floor (ours 1.5 /
0.10, its 2.0 / 0.48), i.e. refusing to keep short-or-weak chains
inside the ILP, not the short filter (4 -> 6 is −2%) and not the
post-processing. Untested on our stack at either venue; nothing
here says the rows it drops are wrong rows, only that they are not
needed for 0.935. Residue: a disappearance / floor sweep is the next
thing to price locally (three-domain rule), then at LB if the local
sign holds.

### ⭐⭐⭐⭐ gatelab (2026-09-07, OFFLINE on divbench3's own forks_f4.csv, no kernel): of the 0.244 -> 0.400 room on the false-fork side, the THRESHOLD is worth +0.0009 of score and everything else is the RANKING -- and no model, fold count or feature subset beats the shipped gate by more than the bench's own noise

The user's second open item was "the wrong-fork side's ceiling div_J
0.40 still has room over 0.244". This prices that room without spending
a kernel. `forks_f4.csv` (7,004 forks, 299 counted, 50 TP / 249 FP, the
20 features and the scorer's verdict) was pulled out of
biohub-divbench3's output; div_J under any cut is a pure function of
(label, gate p) with FN = 125 - TP, so every gate variant can be scored
offline. Script `tools/bench/gatelab*.py` (session scratchpad copies in
tools/bench).

**ANCHOR, exact.** The 2-fold HistGB(100, lr 0.05, msl 10) trained on
the other fold's counted forks reproduces divbench3 to the digit:
out-of-fold AUC **0.9283** (fold0, 122 counted) / **0.8326** (fold1,
177), and the curve q70 44/79/81 div_J 0.2157, q80 42/46/83 0.2456,
p > 0.03 **42/47/83 0.2442**, oracle 0.4000. So the offline harness is
the recorded one.

**Where the 0.1558 of missing div_J actually is:**

| | div_J | of score |
|---|---|---|
| shipped: 2-fold ranking, cut p > 0.03 | 0.2442 | -- |
| the BEST cut on that same ranking (p > 0.0381) | **0.2531** | **+0.0009** |
| oracle: every counted false fork gone, every true one kept | 0.4000 | +0.0266 |

**The cut is exhausted: 0.0089 of the 0.1558 gap. The other 0.1469 is
the ranking.** divbench3 chose 0.03 on a flat part of the curve and
that judgement was right -- there is almost nothing there.

**div_J depends ONLY on the ranking within the 299 counted forks.** The
other 6,705 never enter it (they touch no annotated cell); they cost
edges in adj_J and nothing else. What that ranking can do, walked down
by gate p:

| true forks kept | false forks that come with them | div_J |
|---|---|---|
| 50 (all) | 182 | 0.1629 |
| 48 | 142 | 0.1798 |
| 46 | 104 | 0.2009 |
| 44 | 68 | 0.2280 |
| **42 (the shipped cut)** | **45** | **0.2471** |
| 40 | 37 | 0.2469 |

To reach div_J 0.30 the gate has to keep 44-46 true forks while holding
false ones under about 20 -- roughly a **2.5x cut in false forks at the
same recall**.

**Nothing I can change offline gets there, and the bench cannot see the
differences anyway.** Paired bootstrap over movies (2,000 draws, the
same resampled movies for both arms, per-movie GT counts recovered from
divbench3's log so N_GT is exact):

| gate | div_J at p > 0.03 | paired d vs shipped | 90% interval | distinguishable |
|---|---|---|---|---|
| 2-fold, 20 features (SHIPPED) | 0.2442 (42 TP / 47 FP) | -- | -- | -- |
| 3-fold, 20 features | 0.2573 (44 / 46) | +0.0121 | [−0.0141, +0.0389] | **no** |
| 5-fold, 20 features | 0.2514 (44 / 50) | +0.0067 | [−0.0161, +0.0316] | **no** |
| 3-fold, 6 features | 0.2459 (45 / 58) | +0.0010 | [−0.0245, +0.0289] | **no** |
| 3-fold, 2 features (sc_q, dv) | 0.2487 (47 / 64) | +0.0035 | [−0.0232, +0.0302] | **no** |

**The bench's resolution on div_J is about ±0.03 at 90%**, i.e. ±0.003
of score. Every arm above sits inside it. HistGB is deterministic here
(five random_state values give identical numbers), so this is sampling
noise, not seed noise.

Two things that are still worth carrying:

- **The production gate is better than the number divbench3 recorded.**
  It trains on all 299 counted forks; the recorded 0.2442 came from
  gates trained on ~150 each. Every k >= 3 variant reads higher
  (0.2514-0.2573). Not significant, but it is the right sign and it
  means the +0.010 that landed at LB came from a gate the bench was
  understating, not overstating.
- **Fifteen of the twenty features are dead weight.** Leave-one-out on
  the 3-fold pooled AUC (0.9158 with all 20): only `sc_q` (−0.0360) and
  `dv` (−0.0333) matter, with `t_frac` (−0.0094) and `sc_p` (−0.0064)
  behind them. Dropping any of `sc_c1`, `len_q`, `len_c1`, `pr_q`,
  `p_c1`, `rank_gap`, `dens10`, `d_s`, `len_back`, `p_q`, `pf_q`,
  `d_pq`, `d_pc1`, `src` **raises** it. 20 features on 299 rows with 50
  positives is over-parameterised, and `sc_q` -- the single strongest,
  AUC 0.842 alone -- is a RAW detection score whose distribution moves
  with the movie and with the imaging batch. That is the transfer risk
  for a 44b6 gate, named before 44b6 is measured.
- Both fork sources gate equally well (patch AUC 0.9203 on 187 counted,
  ILP 0.9101 on 112), so one gate for both costs nothing. No split.

**WHY more model capacity does not help: what survives the cut already
looks like a true division.** At p > 0.03 the gate keeps 42 true forks,
kills 8, and lets 47 false ones through. Medians (gatelab5):

| feature | kept TRUE | surviving FALSE | all FALSE |
|---|---|---|---|
| sc_q (orphan's detection score) | 0.418 | 0.379 | 0.313 |
| dv (divergence, um) | 3.30 | 3.04 | 2.50 |
| d_s (sister separation, um) | 8.95 | 8.23 | 7.12 |
| p_q (ranker p, mother->orphan) | 0.545 | 0.500 | 0.435 |
| sc_p (mother's score) | 0.463 | 0.425 | 0.382 |
| n_orph (orphans in the frame) | 3.0 | 4.5 | 10.0 |

**The surviving false forks sit next to the kept true ones on every
column, while the false population as a whole sits well below both.**
The gate has already taken the separable ones; the remainder is a
sub-population these twenty numbers do not distinguish. They are also
not concentrated -- 47 forks spread over 35 of the 128 movies, only two
movies carrying three or more -- so there is no bad-movie mechanism to
find either.

**The 8 true forks it kills are not marginal and they are not random.**
Their gate p has median 0.0063 and maximum 0.0256, well under the cut.
Four of the eight have the second daughter 9-11 um from the mother with
the ranker's probability for that edge at 0.073 / 0.025 / nan / 0.599,
and one of them has **no probability at all**:

- **`DIV_PARENT_UM` is 12.0 um and `GATE_UM` -- the ranker's own pair
  gate -- is 10.0.** The division patch reaches two micrometres further
  than the ranker ever scores, so a fork made out there carries NaN in
  `p_q`, `pf_q`, `pr_q` and `rank_gap`, four of the twenty columns, and
  the gate can only see detection scores and geometry.
- 23 of the 299 counted forks (7.7%) are in that state: **3 of 50 true,
  20 of 249 false**.
- And the region beyond the ranker's reach is ENRICHED in real
  divisions: 14.0% of true forks have d_pq > 10 um against 8.8% of
  false ones. Real daughters separate further, which is exactly where
  the gate is blindest.

That is a named, fixable defect rather than a tuning question, and it
was not visible until the fork table was read row by row.

Tested claim: that the remaining 0.244 -> 0.400 can be reached by
choosing a better cut, model, fold scheme or feature subset over the
existing 20 features. **Closed on this bench: the cut is worth +0.0009
and every other variant is inside ±0.03 div_J.** Residue: the room is
real and it is in the ranking; the only levers left are features the
table does not have (divbench4 adds eight), the 10-vs-12 um mismatch
above, and events the bench does not have (44b6's 26 divisions). And
the bench has a hard ceiling -- 151 GT divisions exist in the entire
train set, so no division mechanism can ever be selected locally at a
resolution finer than this.

### submit-joint SUBMITTED as ref 56086066 (2026-09-08); RESULT IS IN, LB 0.944 -- see the LANDED entry above for the verdict against this entry's pre-registered bands

The joint fork gate is at the leaderboard. Kernel `biohub-submit-joint`
v1 ran clean on GPU and wrote `submission.csv`, 234,761 rows.

**The pkl guard fired correctly and the right gate loaded:**
`fork_gate_joint.pkl | version 2026-09-08a-divbench8 | n_train
{'6bba': 299, '44b6': 67, 'tp': 56}`. The assertion added to
`submit_joint.py` exists precisely so a stale or wrong pickle cannot
ship under the right filename.

**Visible-4 output, against the 0.943 kernel it is cut from** -- the
only difference between the two runs is which pickle scores the forks:

| movie | shipped gate, forks judged / removed | joint gate |
|---|---|---|
| 44b6_0113de3b | 64 / 41 | 64 / 41 |
| 44b6_0b24845f | 75 / 75 | 75 / 75 |
| 6bba_05b6850b | 23 / 14 | 23 / **15** |
| 6bba_05db0fb1 | 135 / 123 | 135 / **114** |
| total | 297 / 253 (85%) | 297 / **245** (82.5%) |

nodes **120,411 both** (the gate never touches nodes), edges 114,350
against 114,342, rows 234,761 against 234,753. **The edge and row
deltas are +8 and +8, exactly the eight forks the joint gate keeps that
the shipped one cut.** Everything downstream is consistent with the
bench: the joint gate is slightly less aggressive overall while being
more accurate about which forks to spare.

(The log ends with a `ModuleNotFoundError: tracking_cellmot` traceback.
That is the kernel's own optional self-validator, which cannot import
the scorer in the submission image; the script catches it and prints
"SCORING FAILED (submission.csv is already written and valid)" before
"SUBMIT RANKER DONE". Not a failure -- but worth naming here so the
next reader does not spend ten minutes on it.)

**Pre-registered reading of the LB, fixed before the score is known.**
Local 6bba is +0.0037 at 85% of the score weight, 44b6 unmeasured under
real pruning but ahead offline, so the prediction is **about +0.003 on
0.943**. The comparison that matters is against ref 56055405 = 0.943,
which is this kernel's own control in the strictest sense available on
this project -- same code, same inputs, one pickle different.

- **>= +0.002**: the joint gate transfers, and "train the gate on every
  batch you have labels for" becomes a general move rather than a
  one-off. It would also be the second time in a row that the division
  line paid at LB.
- **0.000 to +0.002**: transfers weakly. Consistent with the hidden set
  being mostly 6bba, where the 67 added rows are out-of-domain help;
  not a reason to revert, but the +0.0037 would be revised down as an
  estimate of what this kind of change is worth.
- **negative**: the local gain is 6bba-specific in a way the paired
  bootstrap could not see, and the 85/15 weighting is the suspect --
  it assumes 44b6 unchanged, and the joint gate is the one thing that
  has 44b6's own labels inside it.

The 2026-08-28 table would then have a sixth row, and this is the first
entry added to it whose argument was a mechanism (the gate's ordering,
plus a label-shuffle control) rather than a local number.

### ⭐⭐⭐⭐ zprofile VERDICT (2026-09-08, biohub-zprofile, 1 min CPU): THE ZERO-PADDING EXPLANATION IS DEAD -- the z-edge sag exists on 44b6 and NOT on 6bba, and one network cannot have a domain-specific padding artifact. The mirror-padding experiment is cancelled. Plus the hpara pipeline arm, pushed

Two lines opened out of refinecmp and shoulderfeat, on the user's ordering:
"那个hpara换成流水线臂可以先做，因为z边缘那个岔路我没有听懂".

**`hparacache` LANDED (2026-09-08, 35.9 min T4): the anchor is perfect.**
24 of 24 movies, **every frame of every movie reproduced the shipped
`fdet_F4_` cache exactly** -- so the heatmap pass is bit-identical and the
only thing that differs between the arms is where the detection was put. The
hpara displacement is |hpara - refine| p50 **0.43 um**, p90 0.75, max ~1.34,
uniform across all 24 movies (per-movie p50 spans 0.427-0.488). That is well
inside both the 4.0 um NMS radius and the 7 um match radius, which is why
holding NMS on the refine coordinates costs so little.

**1. `hparacache` (GPU, 35.9 min) + `hparabench` / `hparabench_ref` (CPU).**
`fdet_F4_<name>.npz` stores only final coordinates and scores, so `hpara`
-- which needs the heatmap -- cannot be recomputed from it. Only the 24 bench
movies are re-cached, at fuse6bba's 82 s/movie.

THE DESIGN CHOICE, and it makes the arm conservative: NMS runs ONCE, on the
`refine` coordinates, and the survivors are placed both ways. That holds
detection COUNT, ORDER and IDENTITY fixed, so the cached edge-head
probabilities stay validly joined by (t, i, j) -- and it forgoes hpara's
+0.0008 recall, which came from NMS acting on better coordinates. The arm
therefore measures placement ALONE.

`hparabench` is divbench9 with 51 lines changed: `DET_SRC` repoints `FDET_F4`
at the hpara caches, the F4 anchors demote to soft in that run only, and the
per-movie rows are dumped to CSV so the two runs pair for a bootstrap. The
control (`hparabench_ref`, `DET_SRC=F4`) keeps divbench9's 0.9216 hard
anchor, so it validates the harness on its own.

`tools/test_hparacache.py` plants a Gaussian at a known sub-voxel position:
the shift cuts placement error 0.646 -> 0.044 um on clean blobs. That test
exists because the kernel's anchor CANNOT catch a no-op shift -- the anchor
IS the control arm, and a no-op would pass it while making both arms
byte-identical.

`hparabench_ref` burned FOUR pushes on its source list, which is a
process failure worth naming: v1 missed `biohub-edgeprob2`, v2 missed the
pool's `fdet_F4_` caches, v3 missed `biohub-edgeprob-rank2`. Every one came
from copying the list out of divbench9's DOCSTRING, which records what the
sources were when it was written. v4 pulls divbench9's real metadata
(`py kag.py kernels pull <ref> -m`) and copies it verbatim -- and that list
contains two sources nobody would guess (`biohub-zh-detect`,
`biohub-edgeprob-rank2`) and excludes two that look obviously required
(`fuse44b6`, `ep44b6`). The rule is now in `working-agreement.md`.
(An earlier draft of this paragraph added "`kernel_sources` also appears to
cap at 10" -- WRONG, and struck: 12 sources pushed fine. The real constraint
is that a named source kernel must have COMPLETED a run; naming one that has
not only prints a warning and pushes WITHOUT it, which is the silent failure
to watch for.)

**The pair gate, and the two extra cache kernels it cost (2026-09-08).**
The first F4HP push died on

    44b6_0b24845f t=0: 40 of 990 harness pairs have no edgeprob row

and the cause is not a bug in either arm. Candidate pairs are DISTANCE-GATED
at `GATE_UM = 10.0`, and hpara moves every detection by a median 0.43 um, so
a handful of pairs cross that line in each direction per frame. The
DETECTIONS are identical -- `hparacache`'s anchor proves that, 24/24 movies
reproducing `fdet_F4_` exactly -- but the PAIR SET is not, and the cached
edge probabilities are keyed by `(t, i, j)`. Both heads therefore had to be
re-cached on hpara coordinates before the bench could join anything:

| kernel | writes | cost | check |
|---|---|---|---|
| `biohub-ephpcache` | `edgeprob_F4hp_*` x24 | 15.2 min T4 | n_per_frame 24/24 exact |
| `biohub-ep2hpcache` | `edgeprob2_F4hp_*` x24 | 15.1 min T4 | n_per_frame 24/24 exact |

`n_per_frame` is the only check available (the probabilities differ by
construction), and it is the RIGHT one: hparacache neither added nor removed
a detection, so a per-frame primary count that differs would mean the two
arms are not scoring the same cells.

`ep2hpcache` needed two versions for a reason worth naming, since it is the
same shape as the source-list failures above: **v1 inherited
`edgeprob2_cache`'s job list**, `[("base", n) for n in names] + [("F4", n)
for n in names_f4]`, which puts 199 base movies AHEAD of the 24 wanted --
70 minutes in it was still in the base pass, against 16 minutes for the whole
job once cut. Cutting it exposed two more inherited-tail bugs, both from
hard-coded label prefixes: `n_f4` counted `edgeprob2_F4_` when the label is
`F4hp_`, and `complete` demanded `n_base == len(names)` = 199 outputs the
version never writes. `ephpcache` has the first of those and its landed run
therefore printed `EPHPCACHE INCOMPLETE ... NOT OK` on a run that cached all
24 and passed its check 24/24 -- **a false negative; the outputs are good**,
readable only because that line prints unconditionally there. Both makers
are fixed; `ephpcache` was NOT re-run, since nothing about its outputs would
change. General lesson: when a kernel is CUT from another, the tail's
accounting is written for the ORIGINAL's arms and is the first thing to
re-read.

**What v3's failure taught, and it is the caveat that governs this arm.**
The bench does NOT retrain the ranker: it LOADS
`edge_ranker_ep175v3_bidir.pkl`, fitted by `edgeprob-rank2` on the
175-movie pool's features -- which are refine-based. So the ranker is
FIXED, was fitted on REFINE coordinates, and is here applied to HPARA ones.
Re-fitting it would mean re-caching all 187 movies (about four GPU hours
against 35 minutes for the 24) and re-running `edgeprob-rank2` on top.
That is exactly the practical question -- can the placement be swapped in
without retraining anything -- so a POSITIVE result is trustworthy. A
NEGATIVE one is NOT decisive: it would leave open that a ranker fitted on
hpara coordinates does better.

**2. `zprofile` (CPU, ~1 min): the diagnostic that splits the z-edge fork.**
shoulderfeat's finding has two mechanisms behind it:

  (a) the cell is physically half outside the imaged volume -- nothing
      recovers it, at any architecture;
  (b) the detector degrades at the z border of its own input -- fixable.

**The padding facts, checked in the code rather than asserted.** Every 3D
convolution in the repo is `nn.Conv3d(..., 3, padding=1)`, and
**`padding_mode` appears zero times**, so all of them use PyTorch's default
zeros. That covers both architectures in play -- `UNet3D` (the three xl
branches) and `_DCDeepCenterUNet3D` (the OFFICIAL DeepCenter, and our ep05
retrain of it) -- and therefore also the public 0.948 notebook, which runs
the official weights. The whole volume goes through in one forward pass
(`branch_maps4`), so the only boundary is the real one.

Accumulating the receptive field: 2 convs per block, `MaxPool3d(2)` pooling
**z as well**, 64 -> 32 -> 16 -> 8. Encoder reach in original planes is
2 + 4 + 8 + 16 = 30, and the decoder adds about as much again -- against a
stack that is only **64 planes deep**. In z no voxel is far from a fabricated
neighbourhood. In y/x, hundreds of voxels wide, only a thin rim is. That
asymmetry is the shape of shoulderfeat's finding.

**A SEPARATE HOLE THIS EXPOSED, unrelated to the shoulder cells and not yet
tested: a `state_dict` does not carry `padding_mode`.** If the original
authors trained the official DeepCenter with reflect padding, our
reimplementation would load their weights without a single error and run
inference with the wrong padding. Checkable by comparing our heatmap against
the public notebook's on one volume.

**What `zprofile` measures.** gtsplit's CSVs already carry `h_gt` -- the F4
heatmap value at the annotated voxel -- for all 13,646 GT cells, not just the
313. Joining those to each cell's z from the geff, against the true stack
depth read from the zarr metadata:

  * h_gt sagging at BOTH z edges among cells the detector FOUND is branch
    (b): systematic, output-level, and a z-calibration curve or reflect
    padding is the cheap undo;
  * h_gt flat in z among matched cells, with only the missed ones weak, kills
    the attenuation story and makes (a) the better reading.

The matched cells are 97.6% / 96.0% of GT and are the control -- their
profile is a readout of the DETECTOR against depth, independent of the
question. **The caveat recorded before the numbers**: a real biological
gradient would also bend this curve, so only a SYMMETRIC sag at both ends
points at the network -- tissue has no reason to be symmetric about the
middle of the imaging volume.

**THE RESULT, and it kills branch (b).** All 24 movies are Z=64 and every
GT cell joined. Among the cells the detector FOUND:

| | 6bba | 44b6 |
|---|---|---|
| h_gt, within 4 planes of an edge | 0.385 (n=2,096) | 0.205 (n=316) |
| h_gt, 12+ planes in | 0.366 (n=6,309) | 0.306 (n=2,064) |
| **ratio** | **1.054 -- no sag at all** | **0.672 -- 33% down** |
| miss rate, 0-4 planes from an edge | 2.67% | 8.89% |
| miss rate, 10+ planes in | 2.13% | 1.90% |
| **ratio** | **1.26x** | **4.68x** |

**One network, one architecture, one padding mode, Z=64 in every movie of
both domains -- and the effect is present in one domain and absent in the
other. A zero-padding artifact cannot do that**: the convolution does not
know which embryo it is looking at. On 6bba the edge cells are if anything
BRIGHTER than the interior ones. Branch (b) is refuted, and with it the whole
fix space -- reflect padding, z-calibration, depth-dependent thresholds,
partial convolutions, z-crop augmentation. **The mirror-padding experiment is
cancelled before it was pushed.**

44b6's sag is symmetric in RAW z (z 0-5 -> 0.219, z 55-63 -> 0.194, middle
0.30-0.32), which also rules out simple light attenuation -- that would be
monotone, one end dark. Symmetric means both ends of the stack graze the
sample, i.e. a roughly centred embryo whose top and bottom planes cut
partially-imaged cells. That is branch (a): physical, and nothing recovers
those cells.

**AND IT SHARPENS shoulderfeat's OWN FINDING.** The 17-of-18 sign test stands
-- it is a fact about medians -- but the miss-rate table shows the edge is
worth only **1.26x on 6bba** against **4.68x on 44b6**. The z-edge story is
overwhelmingly a 44b6 story, which shoulderfeat's per-movie table did not make
visible because it reported position, not rate. Anyone reading "the shoulder
cells sit at a z edge" as a pipeline-wide lever should read this row first.

**The fix space, HAD it turned out to be (b)**, since the user asked whether
mirror padding is the only option (it is not, and it is the weakest):
inference-only -- reflect-pad z and crop back, a measured z-calibration curve
on the heatmap, or a depth-dependent detection threshold; retraining --
random z-crop augmentation, reflect padding in the architecture, or partial
convolutions. Reflect padding at inference is a DISTRIBUTION SHIFT for a
model trained with zeros, which is exactly why it is an experiment and not a
fix, and why its control is that the interior must not move.


### ⭐⭐⭐⭐ shoulderfeat (2026-09-08, biohub-shoulderfeat v2, geffs only, no images, no GPU): THE 313 SHOULDER CELLS SIT AT A z EDGE OF THE STACK -- in 17 of 18 movies they are nearer an edge than that movie's own cells (sign test p = 1.5e-5), at BOTH ends, not deep. And only 32% of them are isolated single frames; the rest are multi-frame holes `gap_close` cannot span

The user, 2026-09-08: "好烦，这个点除了在山坡上以外就没有任何特征吗！"
Everything found so far needed the heatmap -- which is the thing that
fails on these cells. So ask the tracking graph instead: a feature in the
geometry or the lineage would be knowable with no heatmap and no labels.

**THE FINDING, and it is per-movie so no pooling artifact can produce it.**
z is normalised inside each movie (p1..p99 of that movie's own GT z), then
folded to "distance from the middle of the stack":

| movie | shoulder | \|z50 sh − .5\| | \|z50 all − .5\| |
|---|---|---|---|
| 6bba_3a1849c2 | 77 | 0.123 | 0.008 |
| 44b6_a2bb48bb | 44 | 0.390 | 0.159 |
| 6bba_3abfe10a | 40 | 0.331 | 0.144 |
| 6bba_268e1230 | 33 | 0.123 | 0.025 |
| 44b6_c96cfa10 | 22 | 0.473 | 0.176 |
| 6bba_7b5d3b2c | 6 | 0.500 | 0.076 |
| 44b6_0db75fae | 1 | 0.530 | 0.026 |

**17 of 18 movies put their shoulder cells nearer a z edge than their own
cells, 0 against, 1 tie. Sign test p = 1.5e-5.** It is BOTH ends, not
"deep": `44b6_c96cfa10` sits at 0.027 (the near edge), `6bba_c27cba08` at
0.951 (the far edge). A monotone depth table cannot express that, which is
why the folded axis is the one to read:

Distance to the nearer z edge (0 = at an edge, 0.5 = mid-stack), on the
13,646 measured cells:

| bin | 6bba rate | 44b6 rate |
|---|---|---|
| 0.00-0.02 | 2.91% | **9.05%** |
| 0.02-0.06 | 1.12% | 7.27% |
| 0.06-0.11 | 2.47% | 3.90% |
| 0.17-0.23 | 2.36% | 2.01% |
| 0.33-0.38 | 1.91% | 1.22% |
| 0.44-0.50 | **1.30%** | **1.18%** |

**44b6 is a clean 7.7x gradient and nearly monotone. 6bba is 2.2x and
noisy.** In raw um the same U shape: 44b6 is 14.00% in 0-9.75 um and 7.60%
in 95.88-102.38 um against 0.36-2.3% in the middle.

**Why this matters more than "they are dim".** It explains the dimness
mechanically -- a cell against the end of the stack is only partly imaged,
so its response is weak and never forms its own maximum -- and it is
knowable AT INFERENCE with no labels at all, which nothing else about
these cells has been. (An earlier draft tied this to 17 shoulder cells
said to be "outside the volume" -- that reading is WITHDRAWN, see the
SUPERSEDED note in the shoulderviz entry: those are a y/x index
off-by-one in our own measurement, and 30 of the 48 such cells were
matched by the detector, so they are in the image.)

**THE ALTERNATIVE THIS DOES NOT SETTLE, and it is the actionable one.**
Two mechanisms produce the same table: (a) the cell is physically half
outside the imaged volume, in which case nothing is recoverable; (b) the
3D UNets degrade at the z border of their own input, in which case it is a
padding choice and is FIXABLE. The geffs cannot tell these apart. The z
sweeps rendered by `shoulderviz` v3 can, cell by cell: if the cell looks
whole in the raw planes and is still missed, it is (b).

**THE SECOND USEFUL NUMBER: run lengths.** Consecutive shoulder frames on
one lineage, 155 runs over the 313 cells:

| run | runs | cells | share |
|---|---|---|---|
| 1 frame | 99 | 99 | 31.6% |
| 2 | 26 | 52 | 16.6% |
| 3 | 11 | 33 | 10.5% |
| 4 | 7 | 28 | 8.9% |
| 5-8 | 10 | 68 | 21.8% |
| 12 | 1 | 12 | 3.8% |
| 21 | 1 | 21 | 6.7% |

**The run lengths are in REAL consecutive frames** -- checked when the user
asked "这个标注是隔几帧有一个？", which is the question that would have voided
this whole column. gtpeak ran at `COMP_STRIDE = 1`, and the GT itself is
annotated on EVERY frame of its own range: 23 of the 24 measured movies have
zero missing frames, the single exception being `44b6_87bba6c4`, which has no
annotated cell at t=2. Several movies annotate only a sub-range (`3abfe10a`
t=18-99, `0b24845f` 11-50, `33b596bf` 21-70, `c15fded2` 21-74, `e29f0176`
30-80, `0db75fae` 0-88), which shortens what a run CAN be but does not stretch
a frame. What is sparse in this data is SPACE, not time: 9.71% of 6bba's cells
and 0.77% of 44b6's carry an annotation at all.

**Only 99 cells (31.6%) are one-frame holes.** At `GAP_SPAN = 1` a bridge
can only ever reach those; 68% are holes of two or more frames, longest 21.
So "use gap_close to pick up the shoulder cells" has a hard ceiling of
about a third of them before anything else is considered. (The walk splits
a run at a division rather than following both branches, which can only
make runs look SHORTER -- so 31.6% is an upper bound on the isolated share,
not a lower one.)

Track position, against the correct 24-movie control: shoulder cells are
at a track START 8.9% / 4.5% of the time against 3.6% / 1.9% for all GT,
and at a track END 7.6% / 7.9% against 3.7% / 2.0% -- about double, which
is what an edge-of-volume population should look like.

**NOT A FINDING, recorded so nobody mines it later**: 0 of 313 shoulder
cells sits next to a division. The base rate is 0.4% / 0.3%, so chance
predicts about 1.2. Zero is entirely consistent with that and means
nothing.

**Time**: 44b6 rises from 0.33% in the third and fourth deciles to
5.0-5.7% in the last four. 6bba is flat around 2% with a 3.39% last
decile. Confounded with z here (the embryo moves in the volume), untested.

**Crowding at the GT level was measured and is uninformative**, as
predicted before the run: GT annotates 9.71% of 6bba and 0.77% of 44b6, so
the nearest ANNOTATED neighbour is a fact about the annotation. Both
domains show U shapes in it with no interpretable ordering. It neither
supports nor refutes the local-crowding picture the `shoulderviz` crops
suggested, and the earlier MOVIE-level detection-density refutation stands
untouched -- these are three different questions.

**v1 OF THIS RUN HAD A DENOMINATOR BUG, corrected in v2.** v1 iterated
every train movie in the two domains -- 199 of them -- but only 24 were
ever measured by gtpeak. The other 175 contributed ~120,000 cells labelled
"not shoulder" because nobody looked, diluting every rate roughly tenfold
and mixing in whatever z and t distributions those movies happen to have.
**v1's decile tables are void; its per-movie table was never affected**,
because each row uses only its own movie's cells -- which is why the
headline finding above survived the bug and was in fact found from that
table. The lesson is the generic one: a label set that covers part of the
data makes every unlabelled row a silent negative.

### ⭐⭐⭐⭐ submit-joint LANDED: LB 0.944, ref 56086066 (2026-09-08) -- NEW BEST, +0.001 over its own control. The pre-registered reading says WEAK TRANSFER, and the local bench over-valued this change by about 3x

The fork gate retrained on both domains' counted forks (`fork_gate_joint.pkl`,
128 6bba + 71 44b6 movies, p > 0.03) on the 0.943 pipeline, nothing else
changed. Control: ref 56055405 = 0.943, the same code with the 6bba-only gate.

**LB 0.944, so +0.001.**

**The reading was fixed before the score existed** (entry below), and it lands
squarely in the middle band:

| band | meaning | |
|---|---|---|
| >= +0.002 | the joint gate transfers; "train on every batch you have labels for" becomes a general move | |
| **0.000 to +0.002** | **transfers weakly** | **<- +0.001** |
| negative | the local gain was 6bba-specific and the 85/15 weighting was the suspect | |

So it is kept, and the estimate is revised down, exactly as the band said it
would be: **not a reason to revert, but +0.0037 local is not worth +0.0037 on
the hidden set.** divbench8 measured +0.0037 with a 90% interval of
[+0.0016, +0.0060] and the entry predicted "about +0.003" after the 85/15
domain weighting. LB delivered a third of that.

**HOW TO READ AN LB DELTA AT ALL -- the standing rule, replacing two
earlier drafts of this paragraph that both got it wrong.** The first said
"stop treating a +0.003 local delta as worth a submission slot"; the second
retreated to "read a bench delta as an upper bound on the LB delta". Both
still treat the SIZE of an LB move as information. It mostly is not, and the
user's correction is the rule to keep:

> 不用太因为LB分数而消极。因为不同的影片哪怕是同一组的都有很大的差别，更不
> 要说验证集完全是盲的，我们无从推断算出public score的那部分验证集是取的哪
> 一部分。所以理想情况下，验证集的分布应该和训练集是一样的，那么public score
> 就当一个方向就可以。如果本地全正，LB负了才是要警惕的。

**Read the public score as a DIRECTION. Only the sign carries information.**

- **ALARM: local positive across the board, LB NEGATIVE.** That is the
  reading that says something is wrong -- the bench and the hidden set
  disagree in kind, not in degree.
- **NOT alarm: LB positive but smaller than the local delta.** Nothing to
  diagnose. Do not spend a paragraph on it, do not downgrade the bench, and
  do not let it argue against submitting the next small gain.

**Why the magnitude cannot be read, measured rather than asserted.** Movies
inside ONE domain are not interchangeable. Over the 128 6bba movies in
divbench9's own run log: candidate pairs run 3,581 to 272,577 (**76x**,
p90/p10 23.6x) and LABELLED GT EDGES run 295 to 6,716 (**22.8x**, p90/p10
4.3x). The competition score sums edge TP/FP/FN across movies and divides
once, so a movie's weight in the aggregate IS its edge count -- which means
two different subsets of the same pool can weight the same change by 4x
before any modelling difference enters. The public split is one such unknown
subset, and it is small. In the ideal case its distribution matches the
training pool's, which is exactly why the sign transfers and the magnitude
need not.

So: **local positive and LB positive = a real gain**, kept, and that is the
whole reading. divbench8's [+0.0016, +0.0060] contained the outcome; the
point estimate being nearer the top of its own interval is not a finding.

**The 2026-08-28 LB table gains a sixth row**, and it is the second
consecutive win from the division line:

| ref | change | control | LB | verdict |
|---|---|---|---|---|
| 56055405 | fork gate, 6bba-only | 56030644 = 0.933 | 0.943 | +0.010, largest on record |
| **56086066** | **fork gate retrained on 6bba + 44b6** | **56055405 = 0.943** | **0.944** | **+0.001, NEW BEST, weak transfer** |

`submit_joint.py` is the production script. The only difference from
`submit_gate.py` is the pickle name plus an assert that the loaded gate really
carries n_train 6bba 299 / 44b6 67 / tp 56 -- which is what stopped a
silently-6bba-only gate from being submitted as the joint one.


### ⭐⭐⭐⭐⭐ rkhp RESULT (2026-09-09, 188 min CPU, both recorded cells reproduce EXACTLY): THE TWO CHANGES DO NOT ADD -- they INTERFERE. Both together reads **0.9282**, below hpara alone (0.9298) and below the retrain alone (0.9290), where additive would have been 0.9321. And the best cell of the four has the **LARGEST** train/deploy mismatch, which kills "removing the mismatch" as the retrain's mechanism

The user asked the exact question this answers: *"万一重训的排序器和新坐标不
适应怎么办"*. It does not adapt, and the failure is not the one either of us
expected.

#### The 2x2, ungated REF, 128 6bba movies, DIV_W 0.55, floor 0.48

| cell | ranker trained on | detections | div tp/fp/fn | div_J | adj_J | **score** |
|---|---|---|---|---|---|---|
| **A** ships | BASE primaries | refine | 50/249/75 | 0.1337 | 0.9133 | **0.9266** |
| **B** retrain | F4 (refine) | refine | 52/225/73 | 0.1486 | 0.9141 | **0.9290** |
| **C** hpara | BASE primaries | **hpara** | 55/237/70 | 0.1519 | 0.9146 | **0.9298** |
| **D** both | **hpara** | **hpara** | 50/222/75 | 0.1441 | 0.9138 | **0.9282** |

A and C are anchored to their recorded values and reproduce them to four
decimals, including the division cells. Paired bootstrap, same movie
resample for both arms of each comparison:

| comparison | Δ score | 90% interval | P(>0) |
|---|---|---|---|
| B − A, the retrain alone | +0.00231 | [+0.00067, +0.00396] | 0.990 |
| C − A, hpara alone | **+0.00309** | [+0.00137, +0.00492] | **0.998** |
| D − A, both | +0.00153 | [−0.00047, +0.00360] | 0.890 |
| **D − C, the retrain ON TOP of hpara** | **−0.00156** | **[−0.00304, −0.00007]** | **0.043** |
| D − B, hpara on top of the retrain | −0.00078 | [−0.00324, +0.00168] | 0.295 |

**Adding the retrain on top of hpara is distinguishably NEGATIVE** — the one
interval in the table that excludes zero on the wrong side, and it does so on
edge_J and adj_J as well as score. **Doing both is not distinguishable from
shipping nothing** (D − A crosses zero).

#### The mechanism I have been carrying is dead

Line the four cells up by how badly the ranker's training coordinates match
its deployment coordinates:

| cell | mismatch | score |
|---|---|---|
| C | **largest** — trained on BASE, deployed on hpara | **0.9298, best** |
| B | none — trained and deployed on refine-F4 | 0.9290 |
| D | none — trained and deployed on hpara | 0.9282 |
| A | moderate — trained on BASE, deployed on refine-F4 | 0.9266, worst |

**The best cell has the biggest mismatch and the two matched cells sit in the
middle.** If "the ranker is trained on coordinates it is not deployed on"
were the defect the retrain repairs, D would beat C. It loses to it, and
distinguishably. So the rkpool entry's framing — accurate as a description of
what was changed — is **not** an explanation of why it gained, and this entry
withdraws it as one.

⚠ And no replacement is offered, because the obvious candidates are measured
and none of them order the cells:

* **not the ranking.** Held-out AUC is flat across all four: 0.99256 (A),
  0.99267 (B), 0.99250 (D), all on ~184k rows. Nothing here is about the
  ranker's ability to order edges.
* **not the training pool.** rkhp trains on 153,375 / 141,329 rows against
  rkpool's 153,002 / 140,982 — hpara moves the gated pair set by 0.24%.
* **not output size.** Nodes kept: A 2,091,678; B −10,506; C −670; D −2,700.
  The score order is A < D < B < C; the node order is B < D < C < A. They do
  not agree, so "both changes drop nodes and D overshoots" is also false.

What is left is that two changes which each improve the shipped pipeline
make it worse together, through the ILP rather than through the ranking, and
the record should say that plainly rather than name a fourth mechanism it has
not measured.

#### The gated cross, which is what would actually ship

Point estimates only — the CSVs carry the ungated rows, so there are no
cross-arm intervals here, and each arm's fork gate is trained inside that arm
on its own forks:

| cell | REF ungated | REF+gate6 | **REF+joint = SHIPS** | dw35+gate6 | dw35+joint |
|---|---|---|---|---|---|
| A ships | 0.9266 | 0.9387 | **0.9424** | 0.9402 | 0.9426 |
| B retrain | 0.9290 | 0.9406 | **0.9410** | 0.9419 | **0.9449** |
| C hpara | 0.9298 | 0.9408 | **0.9433** | 0.9429 | 0.9428 |
| D both | 0.9282 | 0.9387 | **0.9404** | 0.9405 | 0.9419 |

**D is at or below A in every column.** And the two survivors want different
things: in the shipped configuration only **C gains** (+0.0009, one constant
changed), while **B needs DIV_W 0.35** to be worth anything and then gives
the highest cell anywhere (0.9449, +0.0025 over what ships).

#### ⚠ A check I built did not fire, and I read it by eye instead

This kernel was licensed to pair against round 2's CSV by re-arming
`ANCHOR_HARD` at round 2's F4HP values. **`ANCHOR_HARD` and `ANCHOR_SOFT` are
never read anywhere in this script lineage** — they are declared, documented,
mutated by the F4HP demotion block, and consulted by nothing; the real
anchors are literals next to the print. So the re-arm was a no-op, the log
printed `want 0.9216 -> MISS (-0.0012)`, and the check I relied on never ran.

The conclusion is unaffected, because the numbers are right there and match
round 2 exactly — `ANCHOR_D187` 0.9204 / div_J 0.0667 and `+BIDIR pkl` 0.9252
/ div_J 0.1389, both to four decimals, on fixed pkls that cannot see
`RANK_TRAIN`. But it was verified by reading, not by the mechanism built for
it, and a mechanism that silently does nothing is worse than none. See the
working-agreement note added with this entry.

### ⭐⭐⭐⭐ rkhp PUSHED (2026-09-09, one CPU kernel, no new GPU): DO hpara AND THE RETRAIN ADD? They touch the same thing -- which coordinates the ranker sees -- and shipping hpara with the F4-pool pkl would RE-CREATE the mismatch the retrain just removed. The reading is fixed here, before the number

Tonight produced two changes that each beat the same control on the same 128
movies, and they have never been run together:

| ranker trained on | detections solved | score | run |
|---|---|---|---|
| BASE primaries | refine | 0.9266 | `rkpool_ref` — what ships |
| **F4 primaries** | refine | **0.9290** | `rkpool`, +0.0024 |
| BASE primaries | **hpara** | **0.9298** | `hparabench` round 2, +0.0032 |
| **F4/hpara primaries** | **hpara** | **?** | **`rkhp`** |

**There is no reason to expect the sum.** The retrain's whole content is
"train the ranker on the coordinates it is deployed on". hpara *changes those
coordinates*. So a pkl fitted on refine-F4 rows and deployed on hpara points
is the same disease one placement further along, and the matched joint arm --
train on hpara rows, solve hpara -- is the one worth measuring.

#### Why it is one constant and no new GPU

`hparabench_ref.py` already carries the entire `DET_SRC == "F4HP"` override,
and it is the careful kind: it swaps the F4 detection cache *and both head
caches* for their hpara counterparts, first cutting the hpara dict down to
the keys the refine arm already has, so the two arms can never differ in the
PIPELINE instead of the placement. rkpool then added `RANK_TRAIN` on top.
Because the override rewrites `FDET_F4` / `EDGEPROB_F4` **in place**,
`RANK_TRAIN="F4"` inside an `F4HP` run already means "train on hpara rows".

What made it impossible until today was coverage: round 2 had **140** hpara
caches. `hpara44` and `ephp44` (both COMPLETE this morning) took it to
**fdet_F4hp 199, edgeprob_F4hp 199**, so the whole 175-movie training pool
can now be put on hpara.

#### Why ONE kernel, when this record forbids reusing a recorded CSV

The control is round 2's F4HP arm and its per-movie CSV is on disk. Reusing
it is normally not allowed here -- attaching different datasets has perturbed
a control before, which is why `rkpool_ref` was RUN rather than read off
divbench3. What licenses it this time is a proof rather than a hope:

**`ANCHOR_D187` and the `+BIDIR pkl` arm load FIXED pkls, and `rkpool`
demonstrated empirically that both are byte-identical across `RANK_TRAIN`
arms** — 0.9216 / 0.0606 and 0.9261 / 0.1351, the same to four decimals in
both. So on round 2's coordinates they must reproduce round 2's numbers no
matter what the fold rankers learned from, and they are re-armed as anchors:

    ANCHOR_D187  F4-6bba(12)  adj_J 0.9204  div_J 0.0667   HARD
    +BIDIR pkl   F4-6bba(12)  adj_J 0.9252  div_J 0.1389   soft

If they hold, the harness is round 2's and the CSVs may be paired. If they
miss, the 59 hpara caches added since round 2 perturbed the scored path and
the pairing is void — which is the entire reason the check exists.

#### Pre-registered reading, all four branches informative

1. **ADDITIVE** (~0.9322) — the two fix different things. The joint candidate
   is then worth about +0.0056 local over what ships, the largest single step
   on record since the pipeline was assembled.
2. **ABSORBED** (~0.9298, the larger alone) — then **hpara's gain WAS the
   mismatch**, and that reinterprets hpara rather than the retrain: the
   +0.0031 would be the ranker coping better with cleaner coordinates, not
   the geometry helping the linker. The zebrahub arm would become the odd one
   out needing its own explanation, since it runs dcmerge's solver with no
   ranker at all and still read +0.0011 P 0.998.
3. **BETWEEN** — partial overlap; ship the better single one unless the joint
   clears both by more than the bench's noise.
4. **BELOW EITHER** — they interfere, and `DIV_W` and the fork gate have to be
   re-crossed a third time. rkpool already showed the retrain moves the fork
   population by 9.3%, and hpara moves it too.

#### And the byte pattern already separates them (`tools/bytedir.py`)

The project's one LB predictor that is independent of the local metric --
11 recorded (Δ bytes, Δ LB) pairs, every byte-removing change gaining --
reads the two candidates very differently on the same 128 movies:

| candidate | Δ nodes | Δ nodes % | Δ matched edges % | the pattern says |
|---|---|---|---|---|
| retrain (F4 ranker, refine) | **−10,506** | **−0.50%** | −0.11% | **GAIN** |
| hpara (BASE ranker, hpara) | −670 | −0.03% | +0.06% | **no call** |

**The retrain removes output; hpara does not.** That fits each mechanism --
the retrain changes which detections the ILP keeps, while hpara moves points
*inside* their voxels on an identical peak set, so it has little reason to
change the node count at all. It also means the two candidates are not
equally supported: the retrain has the independent predictor behind it and
hpara has only the local bench, which for `submit-f48` was the one that was
wrong. Direction only -- these are training-movie nodes, not test-output
bytes.

⚠ **This is the UNGATED REF comparison.** rkpool's lesson was that the shipped
cell (DIV_W 0.55 + the joint gate) can move the *other way* from the ungated
one -- there, four of five cells gained and the deployed one lost 0.0014. The
gate cross this kernel prints is trained inside this arm on this arm's own
forks. Read it as direction. Nothing ships off it until the 44b6 fork
population is regenerated under the same ranker, which is the outstanding
item from the rkpool entry and is unchanged by this run.

### ⭐⭐⭐⭐⭐ rkfloor (2026-09-09, 103.3 min CPU, all FOUR anchors exact): eprank3's TWO negative numbers have TWO DIFFERENT causes. 44b6's was the FLOOR -- it turns positive at 0.30/0.48/0.60. 6bba's was the POPULATION -- the same 12 movies at the same floor 0.48 read −0.0019 while the 128 containing them read +0.0009. Neither was the retrain

The question the rkpool entry left open, closed. `rkfloor` crosses
floor x pkl on one movie set: nothing is retrained, both v4 pkls are loaded
from disk, and the only thing that varies is what the ILP is offered.

    0.10 / v4_BASE / F4-6bba  0.9261  ANCHOR want 0.9261 -> OK
    0.48 / v4_BASE / F4-6bba  0.9261  ANCHOR want 0.9261 -> OK
    0.10 / v4_BASE / F4-44b6  0.8515  ANCHOR want 0.8515 -> OK
    0.48 / v4_BASE / F4-44b6  0.8384  ANCHOR want 0.8384 -> OK

#### The cross

| floor | 44b6, BASE | 44b6, F4 | d | 6bba, BASE | 6bba, F4 | d |
|---|---|---|---|---|---|---|
| 0.10 | 0.8515 | 0.8493 | **−0.0022** | 0.9261 | 0.9250 | −0.0011 |
| 0.30 | 0.8390 | 0.8447 | **+0.0057** | 0.9266 | 0.9248 | −0.0018 |
| 0.48 | 0.8384 | 0.8396 | **+0.0011** | 0.9261 | 0.9242 | −0.0019 |
| 0.60 | 0.8319 | 0.8329 | **+0.0010** | 0.9241 | 0.9234 | −0.0007 |

**On 44b6 the floor flips the sign.** eprank3's −0.0022 is the 0.10 cell,
top-left, and it is the only negative one in that column. At the floor that
actually ships the retrain is positive on 44b6.

**On 6bba the floor does not.** All four cells are negative, so re-tuning the
floor does not rescue the retrain *on these 12 movies*. Which sets up the
comparison that matters:

| same domain, same floor 0.48, same pkls | F4 − BASE |
|---|---|
| 12-movie holdout (`rkfloor`) | **−0.0019** |
| the 128 that contain them (`rkpool`) | **+0.0009** [+0.00003, +0.00167] |

**The 12-movie holdout disagrees in sign with the 128 containing it, at an
identical operating point.** That is the third time this record has caught
that specific failure (hparabench round 2, eprank3, now this), and it is the
whole of eprank3's 6bba result. Both of eprank3's negatives are now
accounted for, by two different causes, and neither is the retrain.

#### What the calibration actually looks like

Printed on the same gated rows, which is the comparison the two pkls could
not make on their own:

| domain | ranker | p5 | p25 | p50 | p75 | p95 | >= 0.48 |
|---|---|---|---|---|---|---|---|
| F4-6bba | v4_BASE | 0.0009 | 0.0036 | 0.0953 | 0.9562 | 0.9970 | 43.60% |
| F4-6bba | v4_F4 | 0.0004 | 0.0032 | 0.0927 | 0.9602 | 0.9970 | 43.46% |
| F4-44b6 | v4_BASE | 0.0010 | 0.0033 | 0.0263 | 0.8362 | 0.9877 | 32.49% |
| F4-44b6 | v4_F4 | 0.0004 | 0.0020 | 0.0246 | 0.8381 | 0.9850 | 32.24% |

The retrained ranker is **sharper, not shifted**: lower in the bottom tail,
higher at p75, and the share clearing the shipped floor moves by **0.14 and
0.25 percentage points**. That is the whole row-level effect. rkpool measured
a **9.3%** drop in forks offered from it, so the ILP compounds a quarter-point
change at the row level into a tenth of the fork population.

#### ⚠ The floor argmax is NOT actionable and this run says so itself

Read as a sweep, both rankers prefer **0.10** on 44b6 and BASE prefers 0.30 on
6bba -- none of them 0.48. That is the pre-registration's third branch firing,
and it must not be acted on. `submit-f48` put floor 0.48 on the leaderboard
*because it won there* (+0.001, −1.43% bytes) while reading **+0.0000 /
−0.0131 locally**. Local and LB already disagree about this exact knob, so a
12-movie local argmax is the weakest possible evidence about it. These cells
answer "does re-tuning the floor rescue the retrain" -- no on 6bba, not needed
on 44b6 -- and nothing else.

### ⭐⭐⭐⭐ hp44full / hp44full-ref (2026-09-09, two CPU kernels, the control reproduces all 71 movies' recorded REF exactly): 44b6's hpara null is STILL a null with all 71 movies and all 26 GT divisions -- but it moved from −0.0002 to **+0.0013**, its interval now contains 6bba's +0.0031, and the division channel finally moved at all. The three-domain ledger is unchanged in verdict and much better supported

The user: *"思考，把44b6全跑一遍不行吗，你这分裂全挂0"*. Correct, and it is
done. The seeded 12-movie draw carried **2** of this domain's 26 GT
divisions, so its division channel could not move. All 71 carry all 26.

#### The control is exact

    ANCHOR_D187  F4-44b6(12)  adj_J 0.8353  want 0.8353 HARD (divsteal) -> OK
    ANCHOR71     REF over all 71: 6/61/20  div_J 0.0690  adj_J 0.8868
                 score 0.8937   want 6/61/20 0.0690 0.8868 0.8937 -> OK

and in the treatment arm both are reported as `[expected]` misses, because
hpara re-places the very movies they are built on.

#### The measurement, 71 movies, paired bootstrap

| | F4 (ships) | F4HP (hpara) | delta | 90% interval | P(>0) |
|---|---|---|---|---|---|
| divisions tp/fp/fn | 6/61/20 | **9/79/17** | +3/+18/−3 | | |
| division_jaccard | 0.0690 | 0.0857 | **+0.0174** | [−0.0109, +0.0497] | 0.823 |
| adj_edge_jaccard | 0.8868 | 0.8863 | −0.0005 | [−0.0033, +0.0025] | 0.393 |
| edge_jaccard | 0.8733 | 0.8731 | −0.0002 | [−0.0030, +0.0027] | 0.447 |
| **score** | 0.8937 | **0.8948** | **+0.0013** | [−0.0027, +0.0057] | 0.688 |

32/71 movies up on adj_J. The eight largest movers carry 40% of the absolute
movement over 71 movies, against rkpool's 27% over 128 -- this domain is the
noisier one per movie, as it has been throughout.

#### The ledger, with 44b6 replaced

| domain | Δ score | 90% interval | P(>0) | sample |
|---|---|---|---|---|
| 6bba | **+0.0031** | [+0.0014, +0.0049] | 0.998 | 128 movies |
| zebrahub | **+0.0011** | [+0.0005, +0.0017] | 0.998 | 38 crops, dense GT |
| **44b6, was** | −0.0002 | [−0.0064, +0.0051] | 0.509 | 12 movies, **2 GT divisions** |
| **44b6, now** | **+0.0013** | **[−0.0027, +0.0057]** | 0.688 | **71 movies, 26 divisions** |

The verdict does not change -- **two positive with intervals excluding zero,
one null, none negative** -- but three things about it improve:

1. **The null's point estimate flipped sign**, −0.0002 to +0.0013, and now
   points the same way as the other two.
2. **Its interval contains 6bba's +0.0031** comfortably. So 44b6 is
   *consistent with* the effect measured elsewhere; it is not evidence
   against it. Same on the edge channel alone -- 44b6's adj_J interval
   [−0.0033, +0.0025] contains 6bba's adj_J +0.0013 [+0.0004, +0.0022].
3. **The division channel moved for the first time**, +0.0174 against 6bba's
   +0.0182 -- the point estimates nearly coincide. On 12 movies with 2 GT
   divisions this number did not exist.

⚠ **I pre-registered branch 2 as "still null on 71 -> the three-domain rule
BLOCKS hpara". That branch mis-stated the project's own rule.** The rule as
recorded is *"the sign must agree"* and the ledger's own reading of it was
*"two positive, one null, none negative -- the rule is met"*. A null has never
blocked anything here, and I wrote a branch that would have blocked hpara on
an outcome the rule permits. The numbers are what they are; the branch was
wrong before they arrived.

#### The ceiling this domain has

The division interval is [−0.0109, +0.0497] on **all 71 movies**. It cannot
be narrowed by running more of them, because **26 ground-truth divisions is
all that exist** in this domain. Any future 44b6 division claim inherits that
width. Worth writing down once, since three separate experiments have now
been designed as if more 44b6 movies would settle a division question.

The zebrahub arm's caveat stands unchanged: it runs dcmerge's solver, so it
answers "does placement help the LINKER", not the shipped submission. And its
own missing 3 crops were shown not to bias it (`tools/zhsize.py`, this night).

### ⭐⭐⭐⭐ zhpara4 (2026-09-09, ~6 min to the kill): THE BOX IS 32.9 GB WITH 31.7 GB FREE AND THE LOG SAYS `Killed`, SO BOTH RIVALS ARE DEAD -- AND SO IS MY OWN HYPOTHESIS. A crop needing a **20.95 GB** dense matrix SCORED; the one that kills needs **17.7 GB**. Separately: the three lost crops do NOT bias the zebrahub hpara verdict

Two of the three things this run was built to settle came back clean, and the
third came back against me.

#### What is now measured rather than assumed

**The ceiling.** `MemTotal 32,870,492 kB` (32.9 GB), `MemAvailable
31,686,692 kB`. Every zebrahub post-mortem in this record has *assumed* about
30 GB and none had read it. There is **no cgroup limit far below the
advertised size** -- rival 1 is dead.

**The signal.** The log survived this time (21,397 chars, not zero) and its
last line is `Killed`. That is SIGKILL. A native segfault prints
`Segmentation fault` -- rival 2 is dead. It is the OOM killer, inside
`compute_metric`, exactly where zhpara3's `rss.log` put it.

#### And the hypothesis this run existed to confirm is CONTRADICTED

The prediction was a dense `N_pred x N_gt` float64 cost matrix. The probe
fired and the numbers are real:

    tripwire crop 1   pred  8,966n / gt  5,051n   dense  0.4 GB   scored
    tripwire crop 2   pred  6,164n / gt  5,320n   dense  0.3 GB   scored
    ZSNS001_z2y2x7    pred 40,936n / gt 54,073n   dense 17.7 GB   KILLED

which looks like confirmation until the 38 crops that DID score are put on
the same axis (`tools/zhsize.py`, from zhpara v3's own CSV, which carries
`n_gt` and `num_pred_nodes`):

| crop | n_gt | n_pred | dense | outcome |
|---|---|---|---|---|
| ZSNS001_tail_z2y3x2 | 56,167 | 46,633 | **20.95 GB** | **scored** |
| ZSNS005_z1y6x8 | 47,527 | 51,282 | 19.50 GB | scored |
| ZSNS001_tail_z2y0x3 | 47,455 | 43,611 | 16.56 GB | scored |
| **ZSNS001_z2y2x7** | 54,073 | 40,936 | **17.7 GB** | **killed, twice** |

**A larger allocation succeeded and a smaller one failed, in the same run.**
The 38 that scored span 0.18 - 20.95 GB, so the lost crop is not even an
extrapolation -- it sits *inside* the range that works. And it is not a
matter of accumulated state: it killed zhpara v3 after 38 crops and zhpara4
after 2, entering the metric at 2.1 GB both times.

So `N_pred x N_gt x 8` does not predict the kill, and a dense cost matrix
over the nodes is not what is being allocated -- or not the only thing.

⚠ **I wrote "the hypothesis that fits every fact" in the previous entry.** It
fit every fact I had bothered to collect. The 38 successful crops' node
counts were sitting in a CSV in the kernel output the whole time and would
have falsified it before the run was pushed.

#### What is left, stated as small as the evidence allows

An OOM kill inside `compute_metric`, reproducible on one specific crop across
two runs at very different resident memory, **not** ordered by node counts.
Whatever is quadratic is quadratic in something else -- candidate pairs
within a matching radius, a per-component structure, an edge-level rather
than node-level matrix. Naming it needs a probe *inside* the metric, not
around it, and this is the third run spent on the question.

**The cost of not knowing is now small**, which is the reason to stop here:

#### The three lost crops do NOT bias the hpara verdict

They are the three largest by cache size (39/41, 40/41, 41/41), so they are
missing BY SIZE, not at random. That only matters if the effect depends on
size, and on the 38 that scored it does not:

| | value |
|---|---|
| Spearman(dense size, d adj_J), n=38 | **−0.093** |
| Spearman(dense size, \|d adj_J\|) | −0.219 |
| weighted d adj_J, largest 12 crops | **+0.00077** |
| weighted d adj_J, smallest 12 crops | **+0.00065** |
| weighted d adj_J, all 38 | +0.00098 |

Same sign in both thirds, no size trend, and the largest third -- the third
adjacent to the missing crops -- carries the effect as strongly as the
smallest. **The zebrahub arm of the hpara ledger (+0.0011 [+0.0005, +0.0017],
P 0.998, 38 of 41 crops) stands**, with the caveat it already carried: it
runs dcmerge's solver, so it answers "does placement help the LINKER", not
"does it help the shipped submission".

`tools/klog.py` needed a fix before any of this was readable: kagglesdk
writes the downloaded log with `open(path, "w")` and no encoding, so on this
box it gets gbk and **any kernel whose stdout contains `⚠` fails to
download** -- which is every script in this project. It now re-execs itself
under `-X utf8`.

### ⭐⭐⭐⭐⭐ rkpool RESULT (2026-09-09, 194.7 + 167.8 min CPU, the control reproduces SEVEN recorded numbers exactly): ON 128 MOVIES AT THE SHIPPED FLOOR, RETRAINING THE RANKER ON F4 COORDINATES **WINS** -- score +0.0023 [+0.0007, +0.0040], P(>0) 0.990, 77/128 movies up. eprank3's 12-movie verdict is overturned. AND dropping the new pkl into the shipped pipeline unchanged **LOSES 0.0014**, because DIV_W and the fork gate were tuned against the old calibration

The user's own idea, from *"哦，排序器水土不服，那先重训也行"*. Chased to the
128-movie paired test, it is worth about as much as the joint fork gate --
and it cannot be shipped by swapping one file.

#### The control is exact, on seven numbers

`rkpool_ref` (RANK_TRAIN=BASE) is the shipped path with two more cache
datasets attached. Every inherited anchor holds:

    ANCHOR_D187  F4-6bba(12)  adj_J 0.9216  div_J 0.0606   HARD -> OK (+0.0000)
    +BIDIR pkl   F4-6bba(12)  adj_J 0.9261                 soft -> OK
    REF over 128  div 50/249/75  div_J 0.1337  adj_J 0.9133  score 0.9266  -> OK
    REF + gate6   42/47/83  0.2442  0.9143  0.9387   (divbench3)  -> OK
    REF + joint   46/39/79  0.2805  0.9143  0.9424   (divbench8)  -> OK
    dw35 + gate6  50/68/75  0.2591  0.9143  0.9402   (divbench6)  -> OK

So attaching `fuse44b6` + `ep44b6` and rebinding `LAB_TRAIN` perturbed
nothing on the control side, and the delta below is the retrain.

#### The measurement, 128 6bba movies out of fold, UNGATED, floor 0.48

| | BASE pool (ships) | F4 pool (retrained) | delta |
|---|---|---|---|
| fold0 / fold1 training rows | 154,107 / 141,994 | 153,002 / 140,982 | −1,105 / −1,012 |
| val AUC, own primaries | 0.9910 / 0.9929 | 0.9908 / 0.9925 | −0.0002 / −0.0004 |
| **held-out AUC on F4 rows, pooled** | **0.99256** | **0.99267** | **+0.00011** |
| forks offered / ILP / patch | 7,004 / 2,490 / 4,514 | 6,352 / 1,876 / 4,476 | **−652 forks** |
| divisions tp/fp/fn | 50 / 249 / 75 | 52 / 225 / 73 | **+2 / −24 / −2** |
| div_J | 0.1337 | 0.1486 | **+0.0149** |
| adj_J | 0.9133 | 0.9141 | +0.0009 |
| edge_J | 0.9104 | 0.9109 | +0.0005 |
| **score** | **0.9266** | **0.9290** | **+0.0024** |

Paired bootstrap, 4,000 draws, the same movie resample for both arms
(`tools/rkpoolpair.py`, whose aggregation is asserted against the control's
own log before anything is compared):

| metric | mean d | 90% interval | P(>0) |
|---|---|---|---|
| edge_jaccard | +0.00053 | [−0.00039, +0.00139] | 0.834 |
| adj_edge_jaccard | +0.00085 | [+0.00003, +0.00167] | **0.953** |
| division_jaccard | +0.01460 | [+0.00140, +0.02889] | **0.962** |
| **score** | **+0.00231** | **[+0.00067, +0.00396]** | **0.990** |

**Two thirds of the gain is divisions** (0.1 x 0.0146 = +0.00146 against
adj_J's +0.00085), and it is not one movie: 77/128 up on adj_J, median
+0.00014, the eight largest movers carry 27% of the total absolute
movement, and deleting the single largest does not flip the sign.

#### The shape: the ranking barely moved, the graph did

The held-out AUC is flat to the fourth decimal (+0.00011) while the score
moves +0.0024. That is the same shape eprank3 found and hparabench round 2
found before it -- **the ordering of edges is not what changed.** What
changed is which candidates clear floor 0.48 and what the ILP then does:

* **652 fewer forks offered** (−9.3%), and 614 fewer of them reaching the
  ILP. Division FP falls 249 -> 225 while TP *rises* 50 -> 52.
* **10,506 fewer nodes kept** across the 128 graphs (2,091,678 -> 2,081,172,
  −0.50%), 26/128 movies up, median −13.5 per movie. Node recall is
  essentially unchanged (0.97323 -> 0.97279).

So the retrained ranker is not better at ranking; it is better *calibrated*
at 0.48, and the ILP is offered a cleaner candidate set.

⚠ **WITHDRAWN AS AN EXPLANATION, 2026-09-09, by rkhp.** The 2x2 shows the
BEST cell is the one with the LARGEST train/deploy mismatch (trained on BASE
primaries, deployed on hpara, 0.9298), and both MATCHED cells sit below it.
If the mismatch were the defect, the matched hpara cell would win; it loses
distinguishably. The description of WHAT was changed stands; the account of
WHY it gained does not. See the rkhp RESULT entry above.

⚠ **Byte-pattern note, unprompted by any local number.** Fewer nodes and
fewer edges mean a SMALLER hidden output. All 11 recorded (Δ bytes, Δ LB)
pairs say byte-removing changes gain. This is the first candidate in a while
where the local metric and the byte pattern agree in sign.

#### It overturns eprank3, and TWO things differ, not one

eprank3 read this same retrain as **−0.0011 / −0.0022** with both intervals
crossing zero. rkpool reads **+0.0009** with the interval excluding it. The
sign flipped. But eprank3 and rkpool differ in two ways at once:

| | eprank3 | rkpool |
|---|---|---|
| movies | 12 (the holdout) | **128** |
| ILP floor | **0.10** (inherited; no `PROD_FLOOR` at all) | **0.48** (production) |

Both are known to matter. hparabench round 2 established the day before that
a 12-movie holdout can disagree in SIGN with the 128 containing it (−0.0012
vs +0.0013), and the floor is the exact knob a calibration change acts
through. **Which of the two flipped it is not settled here.** `rkfloor`
crosses floor x pkl on one movie set and is the separator; it is running.

Until it reports, the defensible statement is the one the bigger, correctly
floored experiment makes: **at the operating point that ships, on the whole
scored population, the retrain gains.**

⭐ **rkfloor REPORTED, and it is BOTH -- one cause each.** On 44b6 the floor
flips the sign (−0.0022 at 0.10, **+0.0011** at 0.48). On 6bba it does not,
and the 12-movie holdout reads −0.0019 at the same floor 0.48 where the 128
containing it read **+0.0009** -- so 6bba's negative was the POPULATION. See
the rkfloor entry above. Neither of eprank3's two negatives was the retrain.

#### ⚠ AND YET IT CANNOT BE SHIPPED AS A FILE SWAP

The pre-registration warned that `PROD_FLOOR`, `ILP_DISAPP`, `DIV_W` and the
fork gate were all tuned against the mismatched calibration. That is not a
caveat any more, it is the result. The same cross, both arms:

| cell | BASE (ships) | F4 (retrained) | delta |
|---|---|---|---|
| REF, ungated | 0.9266 | 0.9290 | +0.0024 |
| REF + gate6 | 0.9387 | 0.9406 | +0.0019 |
| **REF + joint — WHAT SHIPS TODAY** | **0.9424** | **0.9410** | **−0.0014** |
| dw35 + gate6 | 0.9402 | 0.9419 | +0.0017 |
| **dw35 + joint** | 0.9426 | **0.9449** | **+0.0023** |

**Swap the pkl and change nothing else and the pipeline gets worse.** The
gain survives in four of five cells and dies in precisely the one that is
deployed.

Why: the joint gate is worth +0.0037 on the old ranker and **+0.0004** on
the new one, while DIV_W 0.35 goes the other way -- worth +0.0002 (interval
crossing zero) on the old ranker and **+0.0041 [+0.0003, +0.0079]** on the
new one. Additivity moves from 75% of the independent sum to 250%. ⚠ **The obvious mechanism is measured and it is TOO SMALL.** I
wrote "the retrain removes the forks the gate was there to remove" and then
checked it: the retrain drops **652 of 7,004** forks (9.3%) and **24 of the
249** division FPs, while the gate at p > 0.03 drops **over 5,600** forks and
**210** FPs. A tenth is not an absorption.

What the gate's own quantiles DO show is a cleaner fork population -- every
decile of its probability rises (50% 0.0028 -> 0.0036, 80% 0.0275 -> 0.0463,
90% 0.2558 -> 0.3923) and its pooled out-of-fold AUC rises 0.8798 -> 0.8995.

##### The per-fork join, done (`tools/forkjoin.py`, offline, no kernel)

Both arms wrote their DIV_W 0.55 fork table, so the overlap question is a
join. **It cannot be keyed on `(movie, p, q)`** -- `p` and `q` index the
SELECTED node set and the two arms select different nodes (6bba_05b6850b
keeps 6,165 against 6,154), so that key matched 78 of 7,004. The coordinates
are identical in both arms, so the fork's own geometry is the stable key, and
it checks out: **label agreement 4,049/4,049, 100.00%**, on a key that knows
nothing about labels. The gate is reproduced first and hits divbench3 exactly
(AUC 0.9283 / 0.8326, p > 0.03 -> 42/47/83, div_J 0.2442).

**1. The churn is four times the net.** Net −652 forks, but the retrain
*drops 2,955 and adds 2,303*. It reshapes the fork population rather than
trimming it, which no aggregate count showed.

**2. The overlap hypothesis is dead, measured.** Of the forks the retrain
drops, **83.1%** sat below the gate's cut -- against **77.5%** of the ones it
keeps and a base rate of **79.8%**. The gate-p medians of dropped and kept
forks are 0.0050 and 0.0049. The retrain is not removing the gate's targets;
to the gate's eye it removes forks at almost exactly the base rate.

**3. What it IS selective about is the label, not the gate's opinion.** It
drops **18% of the TPs and 42% of the FPs** (9 of 50 against 104 of 249), and
the forks it *adds* are better than the ones it removes: +11 TP / +80 FP in,
−9 TP / −104 FP out. Net +2 TP, −24 FP -- the whole division gain, decomposed.

**4. The join is EXACT, not merely plausible.** The fork table also carries
the official head's probabilities (`pf_c1`, `pf_q`), and the head is the same
cached file in both arms, so on a correct join those columns must be
identical. They are: **max |d| = 0.000e+00** over 3,898 and 3,621 matched
forks. Two independent checks now hold on a key built from geometry alone.

**5. And so the ranker's own probabilities can be compared, fork by fork.**
rkpool read the held-out AUC as flat, so anything that moved should be
calibration. It is not that simple:

| | BASE | F4 | d |
|---|---|---|---|
| `p_c1` (mother -> first daughter), TP | 0.8525 | 0.8575 | **+0.0050** |
| `p_c1`, FP | 0.8604 | 0.8581 | **−0.0024** |
| `p_q` (the edge that MAKES it a fork), TP | 0.4591 | 0.4442 | −0.0149 |
| `p_q`, FP | 0.2602 | 0.2505 | −0.0097 |

On the primary edge the signs are **opposite** -- true forks up, false ones
down -- which is separation, not calibration. On the fork-making edge both
fall, which is a downward shift of the scale, and that is the same direction
as "652 fewer forks offered". So the retrain does a little of each, on
different edges.

⚠ **Read row 5 as conditioned.** These 4,049 forks are by construction the
ones BOTH arms offered; the 2,955 dropped and 2,303 added are absent, and
they are where the division gain actually lives. The table says what happened
to the forks that survived, not why the others did not.

##### ⚠ AND THE THING I SAID COLLAPSED IS NOT THE THING THAT COLLAPSED

Written above: "the joint gate is worth +0.0037 on the old ranker and +0.0004
on the new one". True as arithmetic, wrong as a description. Those are the
*joint* gate's margin **over the 6bba-only gate**. The gate itself is worth
the same in both arms:

| | BASE | F4 |
|---|---|---|
| ungated -> + gate6 | 0.9266 -> 0.9387 = **+0.0121** | 0.9290 -> 0.9406 = **+0.0116** |
| gate6 -> joint | 0.9387 -> 0.9424 = **+0.0037** | 0.9406 -> 0.9410 = **+0.0004** |

So the fork gate is undamaged by the retrain. What the retrain destroys is
the *extra* value the JOINT gate got from adding 71 44b6 movies' counted
forks to its training set -- and that has a mechanism already named in this
entry: those 44b6 forks come from divbench5, i.e. from the **BASE** ranker's
fork population, while the 6bba forks being gated now come from the F4 one.
The retrain creates a new train/deploy mismatch one level up, in the gate.
Re-running the 44b6 forks under the F4 ranker is the fix, and it is step 2
below.

⚠ Two reasons this row is weaker than the ungated one, and both are real:

1. **No cross-arm interval exists for the gated cells.** The CSV carries the
   UNGATED per-movie rows only, so +0.0023 / −0.0014 here are point
   estimates. The within-arm bootstraps are printed in each log and are not
   the same comparison.
2. **The fork gate is retrained inside each arm** on that arm's own forks,
   so these cells carry gate-training noise on top (pooled gate AUC 0.8798
   -> 0.8995), and the *joint* gate's 44b6 training forks still come from
   the BASE-ranker fork population -- a mismatch **the retrain itself
   creates**, one level up. Fixing that means re-running divbench5's 44b6
   forks under the new ranker.

#### What to do with it

Best available local cell is **F4 pkl + DIV_W 0.35 + joint gate = 0.9449**
against today's shipped 0.9424, +0.0025. Three of the four moving parts
would change at once, which is exactly the kind of submission this record
keeps warning about. The order that respects the interaction:

1. `rkfloor` first -- it says whether floor 0.48 is even the retrained
   ranker's argmax. If it is not, the 0.9449 cell is measured at the wrong
   floor too.
2. Re-run the 44b6 fork population under the F4 ranker, so the joint gate is
   joint in the same sense it was before.
3. Only then a submission, with the byte delta recorded before it goes.

#### Housekeeping

`biohub-rkpool` shows **ERROR**, and the error is mine and harmless: the
inherited `assert not ANCHOR_FAIL` fires at the very end on `REF_FORKS`
(52/225 against the control's 50/249). Those anchors exist to prove the
CONTROL is the recorded harness; in the treatment arm they *must* miss. They
should have been gated on `RANK_TRAIN`. Because the assert is accumulated and
checked once at the end, all 128 movies were scored and `rkpool_F4.csv` was
written before it fired -- nothing was lost. `tools/klog.py --fetch` (new)
pulled the 14 kB CSV out of the output in seconds instead of pulling the
whole working directory.

**The artifact the user asked for is verified, not just claimed.**
`edge_ranker_ep175v4_bidir_f4.pkl`, 348,720 bytes, pulled out of
`biohub-eprank3`'s output and sitting in `bench_csv/`. That is the retrain,
landed -- trained on all 175 pool movies' F4 rows, the single-pkl form of
what rkpool measured out of fold.

### ⭐⭐⭐ THE ZEBRAHUB KILL, DIAGNOSED (2026-09-09, zhpara v3 / zhpara2 / zhpara3): it dies **inside `compute_metric`** on `ZSNS001_z2y2x7`, at 2.1 GB RSS on a box with about 30 GB — so it is not a gradual leak, and the leading candidate is a single quadratic allocation. Four kills, and this is the first one that left any evidence at all

Zebrahub has now killed four runs -- `biohub-zh-pipeline` at 10 of 41 crops
(progress.md:425), `zhpara` v1 at 9, `zhpara` v3 at 39, `zhpara2` at 9 -- and
until tonight **none of them left a readable error.** Kaggle returns a
**zero-byte log** for a SIGKILLed session: stdout is not truncated, it is
gone. `zhpara2` printed a full `VmRSS` / `VmHWM` trace for exactly this
purpose and every line of it died with the process.

#### What each run ruled out

| run | what it establishes |
|---|---|
| `zhpara` v3 (38/41) | 39 GT geffs against 38 CSV rows -> it died INSIDE crop 39; the single leftover prediction geff names it **`ZSNS001_z2y2x7`** and shows the REF arm had already saved. `_checkpoint()` kept 38 crops of paired data. |
| artefact count | 819 files / 7.6 MB under `ps_zh_gt` -> **disk exhaustion is dead**, measured rather than assumed. It was the hypothesis I was about to fix. |
| `zhpara2` (8 anchors, died on crop 9) | **not accumulation.** It held only 8 small crops of state and still died on the first big one. And its 8 crops reproduce v3's rows **bit for bit** -- 8 x 2 arms x 12 columns, 0 differ, worst \|d\| 0.000e+00 -- so the harness is deterministic and rows from separate runs may be joined. |
| `zhpara3` (`rss.log`, fsynced) | **the stage.** |

#### The trace that survived

`_rss` writing to `/kaggle/working/rss.log` with `flush` + `fsync` is the
whole difference. On `ZSNS001_z2y2x7`:

     97s  REF after zh_load          1,201,936 kB
    312s  REF after solve_one        2,398,420 kB   peak 2,409,328 kB
    323s  REF freed, pre from_geff   2,132,648 kB
    323s  REF after from_geff        2,132,736 kB
                                     <- killed. The next sample would have
                                        been "REF after compute_metric".

That crop's `solve_one` takes **215 s** against 9-13 s on the two tripwire
crops, so it is roughly twenty times the work of a small one.

#### What does not add up, and why it changes the fix

**RSS entering the metric was 2.1 GB, peak 2.4 GB, on a box with about
30 GB.** Nothing here is climbing gradually into a ceiling, so "it OOMed" as
a description of a slow leak is simply wrong, and every memory-shaving fix
aimed at that story would have been aimed at nothing. (The one blind change
`zhpara3` did make -- freeing `nodes0 / edges0 / pool` right after
`patch_and_save` -- is visible in the trace as 2,398 MB -> 2,132 MB, and it
was not the problem.)

⚠ **DEAD, 2026-09-09, by zhpara4 -- see the entry above.** A crop needing a
**20.95 GB** dense matrix scored and the one that kills needs **17.7 GB**, so
`N_pred x N_gt` does not order the outcome. The 38 successful crops' node
counts were in the kernel output when this paragraph was written. What
survives from here is the *shape* -- one allocation, not a slow climb -- and
nothing about which one.

~~**The hypothesis that fits every fact** is a single quadratic allocation:
`compute_metric` matches the predicted graph against the ground-truth graph,
and `zhpara` imports `scipy.optimize.linear_sum_assignment`. A DENSE
`N_pred x N_gt` float64 cost matrix at 60,000 x 60,000 is **28.8 GB in one
allocation** -- it would die between two RSS samples, leave no Python
traceback, and produce exactly the zero-byte log all four runs produced,
while the last recorded RSS stayed modest.

⚠ **Two rivals are still live** and nothing so far distinguishes them: a
segfault inside a native extension (rustworkx / scipy) kills just as
silently at modest RSS, and so would a cgroup limit far below the advertised
30 GB.~~

⭐ **BOTH RIVALS ARE NOW DEAD, and so is the hypothesis they were rivals to.**
`MemTotal` reads 32.9 GB with 31.7 GB available -- no hidden cgroup limit --
and the log's last line is `Killed`, which is SIGKILL, not the `Segmentation
fault` a native crash prints. See the zhpara4 entry above. `zhpara4` measures the two numbers that separate them -- `MemTotal` /
`MemAvailable` (assumed in every zebrahub post-mortem in this record and
never once read) and the pred/GT node and edge counts with the dense size in
GB, written before every `compute_metric` call. The two tripwire crops
supply the same numbers for graphs that DO fit, so the comparison lives
inside one run.

**If the quadratic story holds, the fix is not a bigger box** -- it is a
sparse or blocked matcher, and it would explain why this domain has killed
runs since the beginning while the competition domains never have: zebrahub
is the only one with DENSE ground truth, so `N_gt` is every cell rather than
the 0.77-9.71% an annotator labelled.


### ⭐⭐⭐⭐ eprank3 / eprank3-ref RESULT (2026-09-09, 102.2 min + ~100 min CPU, the control reproduces ALL THREE recorded anchors exactly): ⚠ SUPERSEDED IN SIGN by the rkpool entry above (128 movies, floor 0.48: +0.0009 adj_J, +0.0024 score, P 0.990). What stands is the number as measured -- 12 movies at floor 0.10. RETRAINING THE PRODUCTION RANKER ON THE COORDINATES IT IS DEPLOYED ON MAKES IT WORSE **on that holdout at that floor** -- adj_J −0.0011 on 6bba and −0.0022 on 44b6, 10 of 12 and 9 of 12 datasets down or flat. the direction is real on both domains but the MECHANISM I first gave (branch 3, BASE-as-augmentation) is RETRACTED by this run's own flat AUC, and the measurement froze the floor the way divradius froze the gate

The user: *"你去把重训落地"*. It is landed --
`edge_ranker_ep175v4_bidir_f4.pkl` exists -- and the first read says do not
ship it.

#### The control is exact, and there is a free internal control too

`eprank3_ref` (POOL_ARM=BASE) reproduces every recorded number for this
harness, so cutting the three head-2 arms and adding the `ep2_path is None`
guard moved nothing:

    ANCHOR_D187  F4-6bba   adj_J 0.9216   want 0.9216 HARD -> OK (+0.0000)
    +BIDIR       F4-6bba   adj_J 0.9261   want 0.9261 soft -> OK (-0.0000)
    +BIDIR       F4-44b6   adj_J 0.8515   want 0.8515 soft -> OK (+0.0000)

and it re-read `HEAD1` 0.9245 / 0.8453 and `HEAD12` 0.9238 / 0.8438 as well.

**The free control**: `ANCHOR_D187`, `HEAD1` and `HEAD12` use no retrained
ranker, so they must be BYTE-IDENTICAL between the two arms. They are, on
all 24 datasets. The only thing that moved is `+BIDIR`, which is the only
thing that was retrained. `ANCHOR_D187`'s hard anchor also holds in the F4
arm, so the harness is the recorded one on both sides.

#### The measurement

| | BASE pool (ships) | F4 pool (retrained) | delta |
|---|---|---|---|
| D175 training rows | 217,688 (0.497 pos) | 216,209 (0.501 pos) | −1,479 |
| `+BIDIR` val AUC | 0.9926 | 0.9926 | **0.0000** |
| edge AUC F4-6bba | 0.9929 | 0.9929 | **0.0000** |
| edge AUC F4-44b6 | 0.9871 | 0.9872 | +0.0001 |
| **stage B adj_J F4-6bba** | **0.9261** | **0.9250** | **−0.0011** |
| stage B div_J F4-6bba | 0.1515 | 0.1316 | −0.0199 |
| stage B score F4-6bba | 0.9412 | 0.9381 | −0.0031 |
| **stage B adj_J F4-44b6** | **0.8515** | **0.8493** | **−0.0022** |

**The AUC is FLAT and the score is DOWN** -- the same shape hparabench round
2 found for hpara, with the sign reversed (there: AUC flat, score up). So the
damage is not in the ranker's ability to order edges; it is downstream, where
the ILP's floor, the disappearance cost and the fork patch read that
probability.

Per dataset, `+BIDIR` only:

| | weighted adj_J (what is scored) | unweighted mean | 90% interval | up |
|---|---|---|---|---|
| F4-6bba | **−0.0011** | −0.00129 | [−0.00303, +0.00031] | **2/12** |
| F4-44b6 | **−0.0022** | +0.00214 | [−0.00725, +0.01596] | **3/12** |

⚠ **44b6's unweighted mean disagrees in SIGN with its weighted one, and one
movie is why.** `44b6_0b24845f` reads 0.9167 -> 0.9961, **+0.0794**, against
a median move of −0.0002. That dataset is already on record as unstable --
`edgeprob_rank2` measured it at 0.9167 / 0.9937 / 0.9979 / 0.8148 across four
linker arms, a ±0.18 swing. Set it aside and the per-dataset picture agrees
with the weighted one.

#### The mechanism, stated the way the arms allow

`HEAD1` is identical in both arms, so the gap between `+BIDIR` and `HEAD1` is
exactly what the 18 coordinate columns are worth on top of the head:

| ranker's gain over the head alone | BASE pool | F4 pool |
|---|---|---|
| F4-6bba | +0.0016 | **+0.0005** |
| F4-44b6 | +0.0062 | **+0.0040** |

**Training on the deployment coordinates makes the coordinate features
contribute LESS, on both domains.**

⚠ **RETRACTED, same night, by this run's own AUC.** The first version of
this paragraph read that as the pre-registered third branch -- BASE primaries
are the same scene measured more noisily, so fitting on them is augmentation
and removing the mismatch removes the regularisation. **The data does not
support that, and it does not support plain overfitting either.** Both
stories are about GENERALISATION, which is what edge AUC measures, and the
AUC is flat in five independent readings:

| | BASE pool | F4 pool |
|---|---|---|
| fold0 held-out AUC (rkpool, 128 movies) | 0.9935 | 0.9935 |
| fold1 held-out AUC (rkpool, 128 movies) | 0.9917 | 0.9918 |
| val AUC on its own primaries (eprank3) | 0.9926 | 0.9926 |
| F4-6bba (eprank3) | 0.9929 | 0.9929 |
| F4-44b6 (eprank3) | 0.9871 | 0.9872 |

**The retrained ranker orders edges exactly as well.** Whatever it lost is
not ranking ability, so neither overfitting nor lost regularisation can be
the account.

**What DID move are the threshold-sensitive outputs**, and that is a
different shape:

| 6bba | BASE | F4 | relative |
|---|---|---|---|
| `edge_J` | 0.9242 | 0.9230 | −0.13% |
| `div_J` | 0.1515 | 0.1316 | **−13%** |
| `node_r` | −0.0199 | −0.0203 | fewer nodes kept |
| `node_r` 44b6 | −0.1113 | −0.1120 | fewer nodes kept |

Two thirds of the 6bba score loss is divisions (0.1 x −0.0199 = −0.0020 of
−0.0031), and divisions are the most threshold-sensitive thing in the
pipeline: the safe-division patch and the capacity-2 ILP both accept on
probability VALUES. The ILP does not read the ranking; it reads the value
against `PROD_FLOOR` 0.48 and `ILP_DISAPP` 1.5.

⚠ **And the simplest version of the calibration story is measured and
FALSE.** Both pkls were unpickled and their intercepts compared:

    BASE pool   baseline logit -0.012710  -> p 0.496823
    F4 pool     baseline logit +0.003145  -> p 0.500786

0.016 logit apart, which is exactly the training base-rate difference
(0.497 -> 0.501 positive) and therefore class balance rather than anything
learned -- **and it points the wrong way**, since a higher intercept raises
probabilities while what was observed is FEWER nodes surviving. So it is not
a simple offset. If calibration is the mechanism it is a change in the trees'
shape, which cannot be read off the pkls and needs both models' probabilities
on the same rows.

⚠ **THE MEASUREMENT IS ALSO CONFOUNDED, and by my own hand -- at a floor
that does not even ship.** `eprank3` inherits `edgeprob_rank2`'s solver
constants, and **it has no `PROD_FLOOR` at all**:

    eprank3.py:203      ILP_FLOOR = 0.10      # candidates fed to the ILP
    eprank3.py:1849         if pr >= ILP_FLOOR:
    $ grep -c PROD_FLOOR eprank3.py
    0

So the whole comparison ran at **floor 0.10**, while what ships is **0.48**
(ref 56030644), and `ilpsweep` measured that move as 6bba +0.0000 but 44b6
**−0.0131** -- the two operating points are not equivalent on that domain.
−0.0011 / −0.0022 was therefore read at a scale belonging neither to the new
ranker nor to production.

Freezing it at all is **the identical error the user corrected three
experiments earlier**, where `divradius` froze the fork gate and read +0.0088
for a sister radius that became −0.0005 once the gate was retrained per cell.
The lesson went into the record and then straight back out of it. (My first
write-up of this even named the wrong constant, `PROD_FLOOR = 0.48`, before
reading the source.) So the result admits two readings this run cannot
separate: a real loss, or a real ranker measured at the wrong operating
point.

**`rkpool` is at the right one.** `rkpool.py:1848` sets
`PROD_FLOOR = 0.48` and stage B solves with `floor=PROD_FLOOR`, so the
128-movie decision is taken at the shipped configuration as well as the
larger sample.

**`rkfloor` settles it and is cheap**: both v4 pkls already exist, so nothing
is retrained; it needs features for the 24 evaluation movies only, crosses
floor x pkl, and prints each ranker's probability QUANTILES on the same
gated rows -- the calibration measurement that cannot be made from the pkls.
If the loss is the operating point, the F4 ranker has its own optimum floor
and shrinks or reverses there; if it is real, no floor recovers it.

**Four anchor cells across two floors**, all recorded, and they validate the
sweep machinery itself before any new number is read:

| BASE pkl | F4-6bba | F4-44b6 | source |
|---|---|---|---|
| floor 0.10 | 0.9261 | 0.8515 | eprank3-ref, ilpsweep REF |
| floor 0.48 | 0.9261 | 0.8384 | ilpsweep's F48 delta (+0.0000 / −0.0131) |

(An earlier draft of this paragraph wrote "0.9261 / 0.8515" for floor 0.48.
0.8515 is the floor-0.10 value; 0.8384 is 0.48's. 6bba is 0.9261 at both
because its F48 delta is +0.0000, which is exactly what hid the difference.)

⚠ And floor 0.48 ships because it **won on LB** (+0.001, ref submit-f48,
−1.43% bytes), not because it is the local optimum -- locally it is
+0.0000 / −0.0131. So "which floor is best" is already a question where
local and LB disagree, and `rkfloor` can only answer the local half.

Also on record from the same run, and relevant to the parallel-blend
question: the `+BIDIR` permutation drop is **h_fused +0.1765** against
p_fwd +0.0019 and p_rev +0.0020. The ranker leans hard on the head's fused
column. That weakens the reading of `HEAD1`'s 6/6 per-movie split as "two
near-equal, disagreeing predictors" -- it is more likely one signal at two
calibrations. It does not make the ranker a wrapper around the head, though:
the 18 columns alone (`ANCHOR_D187`, no head) still read AUC 0.9894 and
adj_J 0.9216.

#### The 128-movie control is verified too (`rkpool_ref`, 134.6 min stage B)

`rkpool_ref` -- the same BASE-vs-F4 question on 128 6bba movies out of fold --
completed and reproduced **four** recorded numbers exactly:

    ANCHOR_D187  F4-6bba(12)   adj_J 0.9216  want 0.9216 HARD -> OK (+0.0000)
    +BIDIR pkl   F4-6bba(12)   adj_J 0.9261  want 0.9261 soft -> OK
    REF          F4-6bba(128)  50/249/75  div_J 0.1337  adj_J 0.9133
                               score 0.9266 (divbench3) -> OK
    POOLED F4 edge AUC 0.99256 over 183,966 rows  (hparabench round 2: 0.99256)

**And the fold AUCs are flat between the arms at 128-movie scale too**:
BASE 0.9935 / 0.9917 against F4 0.9935 / 0.9918 on the held-out folds. So the
"AUC flat, score moves" shape is not an artefact of the 12-movie read; the
ranker orders edges equally well either way and the damage is downstream.
`rkpool` (the F4 arm) is what remains.

#### ⚠ What this does NOT settle

**This is a 12-movie evaluation, and this record established the day before
that a 12-movie holdout can disagree in SIGN with the 128 that contain it**
(hparabench round 2: adj_J −0.0012 on the 12 against +0.0013 on the 128).
`F4-6bba` IS that holdout. Both intervals here include zero. `rkpool` /
`rkpool_ref` -- 128 movies, out of fold, paired bootstrap -- is the decisive
test and is running.

⭐ **IT REPORTED, AND IT REVERSED THIS.** +0.00085 adj_J [+0.00003, +0.00167],
+0.0024 score, 77/128 movies up. This entry's caveat was the correct one and
its headline was not. See the rkpool RESULT entry above; note also that
rkpool ran at floor **0.48** and this ran at **0.10**, so sample size is not
the only difference between them.

What can be said now: the weighted metric is negative on **both** domains,
2/12 and 3/12 datasets up, the harness is proven identical on both sides, and
nothing about the direction is carried by a single dataset except the one
44b6 outlier that is already on record as unstable.

**Consequence for the TTA line, stated before its data lands.** The
augmentation reading, if it survives `rkpool`, applies to `ttacache2` too:
retraining the ranker on tta4 features is the same move -- fit on the exact
features you will deploy -- and would be expected to lose for the same
reason. The caches are nearly built and the arm will still be run, because a
mechanism inferred from one experiment is a hypothesis, not a result. But the
prior on "retraining rescues TTA" is now worse than it was when the user
proposed it, and saying so before the number arrives is the point of writing
it here.


### ⭐⭐⭐⭐⭐ zhpara v3 (2026-09-09, ~3.8 h CPU, killed at crop 39 of 41 with 38 crops of PAIRED data saved, all three anchors EXACT): hpara is POSITIVE on zebrahub too -- score +0.0011, 90% [+0.0005, +0.0017], P(>0) = 0.998. TWO of three domains positive, one null, none negative

The third domain, and the only one with **dense** ground truth: zebrahub
labels every cell, so its denominator is the real cell population rather than
what an annotator could see. That is why its absolute scores are 0.68 and not
0.93, and why it is the one place hpara's effect can be read without a
sparse-GT caveat.

**The anchor is exact on all three metrics.** The REF arm over the 8
dcmerge-v2 crops:

    edge_jaccard      got 0.8036  want 0.8036  OK (+0.0000)
    adj_edge_jaccard  got 0.8093  want 0.8093  OK (+0.0000)
    division_jaccard  got 0.0230  want 0.0230  OK (+0.0000)

#### The measurement, 38 crops, paired crop by crop

| | REF (refine) | HP (hpara) | delta | 90% interval | P(>0) |
|---|---|---|---|---|---|
| `edge_J` | 0.6826 | 0.6836 | **+0.00101** | [+0.00040, +0.00161] | 0.995 |
| `adj_J` | 0.6828 | 0.6839 | **+0.00102** | [+0.00042, +0.00162] | 0.996 |
| `div_J` | 0.0064 | 0.0072 | +0.00077 | [−0.00014, +0.00182] | 0.922 |
| **score** | 0.6835 | 0.6846 | **+0.00110** | **[+0.00049, +0.00171]** | **0.998** |

27 of 38 crops up on adj_J, median +0.00116 -- broad, not one crop.

**And the division counts replay 6bba's mechanism exactly**: REF 44/853/5929
-> HP 49/823/5924. **TP +5, FP −30, FN −5, all three in the right
direction.** Same story hparabench round 2 found on 6bba (7,004 -> 6,709
forks proposed while true ones rose): every test in `_add_safe_divisions` is
a DISTANCE, so cleaner coordinates make the distance tests discriminate
better. Two instruments, two GT regimes, one mechanism.

#### The three-domain ledger

| domain | Δ score | 90% interval | P(>0) | sample |
|---|---|---|---|---|
| 6bba | **+0.0031** | [+0.0014, +0.0049] | 0.998 | 128 movies |
| ~~44b6~~ | ~~−0.0002~~ | ~~[−0.0064, +0.0051]~~ | ~~0.509~~ | ~~12 movies, **2 GT divisions**~~ |
| **44b6 (SUPERSEDED 2026-09-09 by hp44full)** | **+0.0013** | [−0.0027, +0.0057] | 0.688 | **71 movies, 26 GT divisions** |
| zebrahub | **+0.0011** | [+0.0005, +0.0017] | 0.998 | 38 crops, dense GT |

**Two positive with intervals excluding zero, one null, none negative.** The
record's rule -- three domains, or it does not count -- is met, and this is
the first candidate of the night to meet it.

⭐ **Still true on all 71 44b6 movies, and better supported.** The null's point
estimate flipped to +0.0013, its interval now CONTAINS 6bba's +0.0031, and the
division channel moved for the first time (+0.0174 against 6bba's +0.0182).
See the hp44full entry above.

#### Three limits kept in view

1. **38 of 41, and the missing 3 are the LARGEST.** v3 ordered the crops
   smallest-first behind the 8 anchor crops precisely so a kill would cost
   least, and it worked -- `_checkpoint()` rewrote the CSV after every crop,
   so 38 paired crops survived a run that died. But adj_J is weight-averaged,
   large crops carry the most weight, and those are exactly the three that
   are missing. **The number will move when they land; the direction is
   unknown.** Finishing them is a ~30-minute job on 3 crops, not a re-run.
2. **zebrahub's div_J interval includes zero** (P = 0.922). The edge half is
   what clears here; the division half is directionally right and not
   significant.
3. **⚠ ZEBRAHUB IS A DIFFERENT PIPELINE, and the kernel says so itself.**
   `zhpara` runs dcmerge's solver: **no edge head and no fork gate.** So
   +0.0011 answers "does better placement help the LINKER", not "does it help
   the shipped submission" -- `hparabench` answers the second. That is
   exactly what the three-domain rule asks a third domain for (does the
   effect generalise across instruments and GT regimes), and it is NOT a
   number that can be added to a submission's expected gain.
4. **44b6's "null" is a property of its 12-movie draw**, which carries 2 GT
   divisions and scored div_J 0.0000 in both arms. `hpara44` (GPU) +
   `ephp44` are running/queued to give all 71 movies hpara coordinates, so
   that domain can be re-read on 26 GT divisions instead of 2. `divbench5` v2
   already showed the 12-movie draw reads 43% high on this kind of quantity
   (+0.0020 on 2 events against the full 71's +0.0014 on 26).


### ⭐⭐⭐⭐ rkpool / rkpool-ref PUSHED (2026-09-09, two CPU kernels): THE RANKER THAT SHIPS WAS TRAINED ON BASE DETECTIONS AND IS DEPLOYED ON F4 ONES. The mismatch has never been measured, it costs no GPU to remove, and it makes ttabench's −0.0051 unreadable. THE READING IS FIXED HERE, BEFORE THE NUMBERS

The user, on being told that ttabench measured TTA through a ranker fitted on
non-TTA columns: *"哦，排序器水土不服，那先重训也行"*. Following that down
found something larger than the TTA question.

#### The mismatch, in two lines of source

`edgeprob_rank2.py:1608` builds the production training pool:

    _jobs = [(n, "BASE", "pool") for n in POOL175]

So `edge_ranker_ep175v3_bidir.pkl` -- the pkl every submission loads --
learned the map from features to truth on **BASE** primaries (the xl
three-branch mean, LB 0.922) and is deployed on **F4** ones (that mean plus
ep05 DeepCenter as a fourth equal-weight map, LB 0.925, ref 55889500).
`hparabench_ref` inherits the same shape: stage A's two fold rankers train
from `LAB`, built with `EP_OF["BASE"]`, and stage B solves F4.

**Every score in this record was produced by a ranker reading coordinates it
was not fitted on**, and no arm has ever isolated what that costs.

#### It also breaks ttabench's number, and by more than I said

The ttabench entry above reports adj_J −0.0051 / edge_J −0.0074 for
edge-feature TTA. Its override replaced `EDGEPROB_F4`; **training reads
`EDGEPROB`**. Not one tta4 row entered the fold rankers. They were 100%
noTTA-trained and then handed tta4 columns at solve time, so that delta is
train/test mismatch plus TTA, with no way to separate them. (I first told the
user "90%+ noTTA"; it is all of it.) **The measurement stands as what it
measured -- TTA bolted onto a ranker fitted without it -- and that is not the
question the public notebook's +0.005 poses.** The matched test needs tta4 in
the TRAINING rows, which `ttacache` (140 movies, running) makes possible on
6bba.

#### Why it costs no GPU, and why it could not be run before 2026-09-07

A ranker trained on F4 needs F4 primaries and F4 head probabilities for the
whole pool. `fuse44b6` (59 of 59, 90.1 min) and `ep44b6` (59 of 59, 45.3 min)
landed exactly those for the 59 44b6 train movies that lacked them, so all
199 movies now carry both. hparabench simply never attached them -- its log
reads `199 BASE + 140 F4`. With `biohub-fuse44b6` + `biohub-ep44b6` attached,
the F4 pool is the same 175 movies the BASE pool is: **no pool-size
confound, no new GPU.**

#### The two arms

Identical movies, folds, columns, estimator, seed and stage B. One line
differs -- which table stage A concatenates from.

| kernel | `RANK_TRAIN` | the fold rankers learn on | stage B solves |
|---|---|---|---|
| `rkpool_ref` | `BASE` | the xl three-branch mean | F4 |
| `rkpool` | `F4` | that mean + ep05 DeepCenter | F4 |

The held-out-fold AUC is read on **F4 rows in both arms**, deliberately: F4 is
what stage B solves, so for the BASE arm that is a cross-domain read, and it
is what the mismatch costs in AUC before the ILP ever sees it.

**ANCHORS.** `rkpool_ref` is hparabench_ref's path with `LAB_TRAIN` bound back
to `LAB`, so divbench3 must come back to the digit: REF over the 128 =
50/249/75, div_J 0.1337, adj_J 0.9133, score 0.9266, plus the gate cross cells
0.2442 / 0.2805 / 0.2591. Those asserts are inherited unchanged and not
restated -- if attaching two more cache datasets perturbed the BASE path they
fail on their own, which is why the control arm is being RUN rather than
reused from round 2's CSV.

#### Pre-registered reading, all three branches informative

1. **F4 beats BASE, interval excludes zero** -> the mismatch costs real score.
   Retrain the production pkl on F4 and every submission gains for free. But
   then `PROD_FLOOR` 0.48, `ILP_DISAPP` 1.5, `DIV_W` 0.55 and the fork gate
   were all tuned against the mismatched ranker's calibration and every one of
   them has to be re-checked -- the divradius2 lesson, at pipeline scale.
2. **Flat** -> HistGB absorbs this shift. Then the TTA retrain is unlikely to
   rescue TTA either (same mechanism, smaller shift), and the case for the
   parallel-blend arm below weakens with it.
3. (⚠ **BRANCH 1 IS WHAT FIRED**, on 128 movies at floor 0.48. This branch
   fired only on the 12-movie read at floor 0.10.) **BASE beats F4** -> the
   BASE pool is acting as augmentation: training on
   coordinates slightly noisier than the deployment ones regularises. That
   would be a finding in its own right and would argue AGAINST retraining for
   TTA.
   ⚠ **This branch's DIRECTION fired on eprank3 and its EXPLANATION did
   not.** Augmentation is a claim about generalisation and the held-out AUC is
   flat in five readings, so the pre-registration offered no branch for what
   actually happened -- same ranking, different downstream behaviour, at
   constants tuned for the other calibration. Pre-registering three branches
   does not make them exhaustive, and this is the second time tonight the
   outcome fell outside the table (hparabench round 2's "AUC flat, score UP"
   was the first).

#### Not run, and not previously rejected: the head and the ranker as PEERS

The user: *"边头和排序器同级呢，虽然之前好像否定过这个"*. Checked. Three
things were tried and rejected -- `+BIDIR2` / `+BOTH` / `+AVG` (more head
columns INSIDE the ranker, 1/11, 5/7, 4/7 per movie) and `HEAD1` / `HEAD12`
(the head REPLACING the ranker, 0.9245 / 0.8453). A convex blend of the two
probabilities at the same level was never run, and `make_submit_ep2.py:72`
forbids it structurally:

    assert (RANKER_NAME is None) != (HEAD_ONLY_COL is None)

Two things make it worth a arm rather than a paragraph. `HEAD1` loses only
0.0016 / 0.0062 yet splits **6/6 per movie on both domains** -- two
near-equal predictors that disagree movie by movie is the shape an average
helps, unlike `+BIDIR2`'s 1/11. And the record's own caveat
(`progress.md:4692`) is that our ILP's costs "were tuned against the ranker's
calibration, not h_fused's", so HEAD1's −0.0016 was measured at a floor
belonging to the other scale: **it is a lower bound on the head's value, not
an estimate of it.** A blend arm therefore has to sweep the floor with the
weight, not freeze it.

⚠ Held back until `rkpool` reports, because the two interact directly: if
training on F4 moves the ranker's probabilities, every blend weight measured
against the old ranker is stale.

⚠ **What none of this says is that the official head likes our nodes.**
`progress.md:104`: both official edge heads were trained on all 199 train
movies, so every local head number is in-sample. `progress.md:46`:
`submit_off` (official nodes + head only, no ranker) read +0.004 / +0.057
locally and scored **LB 0.895, the worst on record**. That run changed the
detector too, so it does not isolate the head -- but it is the reason a
head-heavy configuration does not go to LB on local numbers alone.


### ⭐⭐⭐⭐ divradius + divradius2 (2026-09-09, laptop CPU, BOTH gate anchors EXACT): the last OPEN row of the public-notebook comparison closes -- their parent radius costs −0.0042 score, 90% [−0.0079, −0.0006]. The first pass ALSO produced a free +0.0009 candidate that turned out to be an artifact of freezing the gate, and the "named, fixable defect" at `DIV_PARENT_UM` 12 vs `GATE_UM` 10 gets a ceiling of +0.0022

My own entry a few hours ago claimed all six differences from the public
0.946 notebook were closed. **That was wrong: the comparison table's own
verdict column says `division parent / sister um` (theirs 9.0 / 14.0, ours
12.0 / 15.0) is OPEN**, and it had never been measured. It is measured now,
and it cost nothing -- `hparabench_ref` already writes `forks_dw55.csv`, one
row per fork with all 20 gate features, `d_pq`, `d_s` and the label.

**The gate is reproduced locally, and both anchors are exact.** Same recipe
as divbench9 -- `HistGradientBoostingClassifier(max_iter=100,
learning_rate=0.05, min_samples_leaf=10, random_state=0)`, 20 features, two
alternating movie folds, out of fold, cut `p > 0.03`:

| gate | got | recorded | |
|---|---|---|---|
| 6bba only | 42 / 47 / 83, div_J 0.2442 | divbench3 `REF + gate6` 0.2442 | **exact** |
| **+ divbench5's 67 counted 44b6 forks in both folds** | **46 / 39 / 79, div_J 0.2805** | divbench8 `REF + joint` 0.2805 | **exact** |

The second is what ships (LB 0.944, ref 56086066), so every row below is
against the gate that is actually in front of these radii.

#### The result: their parent radius costs us four thousandths

| arm | TP/FP/FN | div_J | d div_J | **d score** | counted cut | TP cut |
|---|---|---|---|---|---|---|
| **12.0 / 15.0 (SHIPS)** | 46/39/79 | **0.2805** | +0.0000 | +0.00000 | 0 | 0 |
| `DIV_PARENT_UM` 11.0 | 43/38/82 | 0.2638 | −0.0167 | −0.00167 | 10 | 3 |
| `DIV_PARENT_UM` **10.0** (= `GATE_UM`) | 39/30/86 | 0.2516 | −0.0289 | −0.00289 | 29 | 7 |
| `DIV_PARENT_UM` **9.0** (theirs) | 37/28/88 | 0.2418 | **−0.0387** | **−0.00387** | 49 | 10 |
| `DIV_SISTER_UM` **14.0** (theirs) | 46/39/79 | 0.2805 | +0.0000 | +0.00000 | 1 | 0 |
| `DIV_SISTER_UM` 13.0 | 46/37/79 | 0.2840 | +0.0035 | +0.00035 | 6 | 0 |
| `DIV_SISTER_UM` **12.0** | 46/34/79 | **0.2893** | **+0.0088** | **+0.00088** | 15 | **0** |
| `DIV_SISTER_UM` 11.0 | 43/31/82 | 0.2756 | −0.0048 | −0.00048 | 30 | 3 |
| theirs, both: 9.0 / 14.0 | 37/28/88 | 0.2418 | −0.0387 | −0.00387 | 49 | 10 |

**Their 9.0 / 14.0 would cost us −0.004 score.** The parent half is a clear
loss at every cut, the sister half is an exact zero. **REJECTED, measured**
-- and that empties the comparison table: nothing in the public 0.946
notebook is left to port.

#### The mechanism the record proposed, and why it is wrong

The record (2026-09-08, the fork-table read) named a defect: `DIV_PARENT_UM`
is 12.0 while `GATE_UM` -- the ranker's own pair gate -- is 10.0, so a fork
proposed out there carries NaN in `p_q`, `pf_q`, `pr_q` and `rank_gap`, four
of the twenty gate features. 23 of 299 counted forks are in that state, and
the band is enriched in real divisions. The obvious repair is to pull the
patch's reach back to 10.0 so the gate always has its features. **It loses
0.003, and the survival table says why:**

| band | counted | TP | FP | TP kept | FP kept | **TP survival** | **FP survival** |
|---|---|---|---|---|---|---|---|
| `d_pq <= 10`, inside the gate | 270 | 43 | 227 | 39 | 30 | **0.91** | **0.13** |
| `d_pq > 10`, four NaN features | 29 | 7 | 22 | 7 | 9 | **1.00** | **0.41** |

**The gate is not blind out there. It is less selective** -- 41% of false
forks survive against 13% inside -- **and it keeps every single true one.**
So the band contains 7 real divisions and 9 surviving false ones, and
cutting it at the radius throws away all 7 to remove all 9. In a metric
where a lost TP moves to FN, that is the wrong direction twice over.

**The ceiling on fixing it properly, which is the number this line was
missing.** Suppose the missing features were supplied and the gate then
played that band perfectly -- every one of the 7 true forks kept, all 9
surviving false ones cut. That is 46 / 30 / 79, div_J 0.3020, **+0.0215
div_J = +0.0022 score, exactly, on these 128 movies.**

⚠ The first draft of this paragraph called that "about one noise width" and
cited `accdiag` for a floor of 0.002-0.003. **Both halves were wrong.**
`accdiag` measured no such thing, and there is no floor: eight `random_state`
values give the identical 46 / 39 / 79, so repeat-run noise here is **zero**.
+0.0022 is a real ceiling, not a number lost in noise. What it is competing
against is the COST -- a re-cache of the edge head at a wider pair gate for
128 movies plus a bench run -- for a change whose best possible outcome is
+0.0022 and whose realistic one is a fraction of that, since the gate would
have to play the band perfectly. That is the reason to defer it, and it is a
cost argument, not a significance one. (Distinct from widening the gate for
the LINKER, which `gate-sweep`, `gate2` and `zgate` all closed.)

#### ⚠ `DIV_SISTER_UM` 12.0 WAS AN ARTIFACT OF FREEZING THE GATE. It does not survive `divradius2`

The first draft of this entry offered `DIV_SISTER_UM` 15.0 -> 12.0 as a free
+0.0009 score at zero true-fork cost. **It is not free and it is not
+0.0009.** The user, reading this entry: *"你这不还是每个参数分开调"*.

The defect was structural, not arithmetic. `divradius` trained the gate ONCE,
on the fork population a run at 12.0 / 15.0 produces, and every arm then
merely FILTERED that population. But the radius changes what the gate should
be fitted on -- which is precisely how divbench9 handled its `DIV_W` arms
(*"each gate is trained OUT OF FOLD on that arm's own counted forks -- the arm
changes the fork population"*). `tools/divradius2.py` retrains per cell:

| `DIV_SISTER_UM` 12.0 | div_J vs shipped | 90% interval |
|---|---|---|
| gate FROZEN on the 12.0/15.0 population (the first draft) | **+0.0088** | not computed |
| gate RETRAINED on the arm's own forks | **−0.0005** | [−0.0116, +0.0102] |

**The whole gain was the frozen gate.** Fewer far-sister forks in training
changes what the gate learns, and once it is allowed to relearn, the FPs the
radius was removing come back through the threshold.

#### The cross, which is what should have been run first

Gate retrained in every cell, `div_J`:

| parent \ sister | 15.0 | 13.0 | 12.0 | 11.0 |
|---|---|---|---|---|
| **12.0 (SHIPS)** | **0.2805** | 0.2771 | 0.2805 | 0.2622 |
| 11.0 | 0.2593 | 0.2531 | 0.2437 | 0.2437 |
| 10.0 | 0.2378 | 0.2378 | 0.2407 | 0.2422 |
| 9.0 | 0.2390 | 0.2390 | 0.2468 | 0.2357 |

**And there IS an interaction, in the shape the marginals hid.** In the top
row the sister radius does nothing from 15 down to 12; from the second row
down every cell has already fallen 0.02-0.04 and the sister column barely
moves inside it. **The parent cut has already taken the forks the sister cut
would have taken.** Sweeping sister alone reads "mildly useful", sweeping
parent alone reads "mildly bad", and only the square shows the parent radius
dominating both.

Paired bootstrap over the 128 movies, the same resample for every cell, on
the arms that matter:

| arm | d div_J | 90% interval | d score | P(>0) |
|---|---|---|---|---|
| sister 12.0 alone | −0.0005 | [−0.0116, +0.0102] | −0.00005 | 0.425 |
| parent 11.0 | −0.0214 | [−0.0478, +0.0044] | −0.00214 | 0.086 |
| parent 10.0 (= `GATE_UM`) | −0.0430 | [−0.0799, −0.0092] | −0.00430 | 0.021 |
| **theirs, 9.0 / 13.0** | **−0.0417** | **[−0.0790, −0.0056]** | **−0.00417** | **0.029** |

Their parent radius stays rejected, and now with an interval instead of a
bare point estimate. Everything else here is a null.

**Why the paired form is not optional.** The shipped cell's own ABSOLUTE
bootstrap spread is div_J [0.2145, 0.3486] -- a width of 0.134, because 128
movies carry only 125 GT divisions between them. Unpaired, nothing on this
axis could ever be measured. Paired, the movie draw cancels and the deltas
above are 0.03-0.08 wide.

**What the separation still says, and it is why the sister cut looked free:**

| | TP p50 | TP p90 | FP p50 | FP p90 |
|---|---|---|---|---|
| `d_pq` | 6.10 | 10.73 | 6.00 | 9.79 |
| `d_s` | 8.88 | **10.43** | 7.12 | **11.17** |

True daughters ARE further apart than false pairs while spreading TIGHTER
(TP p90 10.43 against FP p90 11.17), so a cut at 12 really does sit above
every true fork and inside the false tail. That much is a fact about the
data. It is just not worth anything once the gate is allowed to see the
population it is judging.

#### Two more free nulls off the same table, before anyone spends a run on them

**A per-SOURCE gate cut buys nothing.** Forks come from two stages -- the
ILP's own (`src` 0, 112 counted, 22 true) and the geometric patch's (`src` 1,
187 counted, 28 true) -- and they share one threshold. Splitting it, best of
a 10 x 10 grid: ILP 0.05 / patch 0.03, **+0.00017 score**, which is WORSE
than simply moving the single cut. The reason is in the survival rates:
at 0.03 the gate keeps 0.91 of true ILP forks against 0.93 of true patch
ones, and 0.13 of false against 0.17. **The two sources are the same problem
to this gate**, which is what you would expect given that `src` is already
one of its twenty features.

**And the cut itself is at its plateau.** Sweeping the single threshold:
0.03 (ships) 0.2805, 0.04 **0.2848 (+0.00043 score)**, 0.05 0.2675, 0.02
0.2686. The peak is one grid step wide on 299 forks and 50 true events;
divbench3 already priced this axis at +0.0009 and closed it. Recorded so the
next reader does not re-run the sweep to rediscover a fifth of a noise width.

#### ⚠ WHAT THIS APPROXIMATES, stated because the whole entry rests on it

`tools/divradius.py` filters the fork proposals a run at the shipped radii
already made; it does not re-run `_add_safe_divisions` at a smaller radius.
The two differ where the caps bind -- `DIV_GLOBAL_CAP`, `DIV_FRAME_CAP` and
the `used_p` / `used_q` exclusivity mean a dropped proposal can free a slot
for another one. Proposals sort by `d_pq + 0.15 * d_s`, so the far ones sort
LAST and win contested slots least often, which bounds how much that can
matter. It does not make it zero. **The −0.004 verdict on their 9.0 is far
outside anything that could flip; the +0.0009 on sister 12.0 is not, and is
labelled a candidate rather than a result for exactly that reason.**

Also: 6bba only, 299 counted forks, 50 true. And the joint gate was retrained
here on the 12.0-radius fork population; an arm at 10.0 would retrain on its
own. Retraining cannot recover a fork that was never proposed, so that works
against the tightened arms only in the direction they already lose.


### ⭐⭐⭐⭐⭐ hparabench ROUND 2 (2026-09-09, both arms 179 min CPU, control anchors EXACT on divbench3 / divbench6 / divbench8): FULL COVERAGE OVERTURNS ROUND 1. hpara is +0.0031 score on 6bba, 90% [+0.0014, +0.0049], and the AUC is FLAT -- so the gain is not the ranker, it is the geometry. Plus: the 12-movie holdout disagrees in SIGN with the 128 that contain it

Round 1 scored 128 movies of which TWELVE carried hpara coordinates, read
`edge_J −0.0018` off the changed subset, and concluded the placement does not
convert. `hparacache` v3 + `ephpcache` + `ep2hpcache` now cover 140, so all
128 scored movies differ between the arms. The conclusion reverses.

**The control is exact on all three cross cells** -- `REF+gate6` 42/47/83
0.2442, `REF+joint` 46/39/79 0.2805, `dw35+gate6` 50/68/75 0.2591, each
reproducing divbench3 / divbench8 / divbench6 to the digit. The harness did
not move.

#### The measurement, ungated, 128 movies, paired movie by movie

`tools/hparapair2.py`: 4,000 draws, the SAME resample for both arms so the
movie draw cancels, and the aggregation asserted against each run's own log
before anything is compared (`summarise`'s micro-average for edge_J and
div_J, its weight-average by `tp+fp+fn` for adj_J).

| | F4 (refine) | F4HP (hpara) | delta | 90% interval | P(>0) |
|---|---|---|---|---|---|
| `edge_J` | 0.9104 | 0.9118 | **+0.00134** | [+0.00041, +0.00228] | 0.990 |
| `adj_J` | 0.9133 | 0.9146 | **+0.00128** | [+0.00037, +0.00221] | 0.988 |
| `div_J` | 0.1337 | 0.1519 | **+0.01810** | [+0.00322, +0.03387] | 0.977 |
| **score** | 0.9266 | 0.9298 | **+0.00309** | **[+0.00137, +0.00492]** | **0.998** |

**Every interval excludes zero, and it is not one movie.** 68 of 128 up on
adj_J; the six biggest movers carry 19% of the total absolute movement.

#### The discriminator says FLAT, and that combination was not in the table

| | pooled out-of-fold AUC | rows |
|---|---|---|
| F4 | 0.99256 | 183,966 |
| F4HP | **0.99244** | 184,174 |

−0.00012. **The ranker reads hpara coordinates exactly as well as refine
ones**, which kills the distribution-shift hypothesis round 1 raised (the
ranker was fitted on refine features and never saw coordinates this clean).
The pre-registered 2x2 had no row for "AUC flat, score UP" -- it assumed a
gain would have to arrive through the ranker.

**Where it does arrive, and it is visible in the counts.** Forks proposed
fall 7,004 -> 6,709 (**−4%**) while true ones RISE, 50/249/75 -> 55/237/70.
Every test in `_add_safe_divisions` is a DISTANCE -- `DIV_PARENT_UM`,
`DIV_SISTER_UM`, `DIV_DIVERGE_UM`, and `GAP1_GATE_UM` downstream. `refinecmp`
measured hpara cutting frame-to-frame jitter 31%. Cleaner coordinates make
distance tests discriminate better, so the patch proposes FEWER forks and
more of them are real. The edges follow for the same reason.

#### Gated -- what actually ships -- the gain is smaller and half of it reverses

| joint gate, p > 0.03 | div TP/FP/FN | div_J | adj_J | score |
|---|---|---|---|---|
| F4 | 46/39/79 | 0.2805 | 0.9143 | 0.9424 |
| F4HP | 49/51/76 | 0.2784 | **0.9155** | **0.9433** |
| delta | +3/+12/−3 | **−0.0021** | **+0.0012** | **+0.0009** |

The adj_J half survives intact (+0.0012 gated against +0.00128 ungated); the
div_J half reverses, because **the fork gate was trained on the refine
coordinate fork population and hpara changes that population.** ⚠ There is
no interval on the gated +0.0009 -- the CSV carries the ungated per-movie
rows only. What can be said is that adj_J carries weight 1.0 against div_J's
0.1, so the gated delta is essentially the adj_J delta, and that term's
interval is [+0.0004, +0.0022].

#### ⭐⭐⭐⭐⭐ The 12-movie holdout disagrees in SIGN with the 128 that contain it

Both arms print the D187 anchor cell and the production-floor cell, and both
are computed on `HOLDOUT_6BBA` -- a subset of the 128:

| hpara − refine | adj_J | div_J | score |
|---|---|---|---|
| 12-movie holdout, D187 anchor config | **−0.0012** | +0.0061 | ≈ −0.0006 |
| 12-movie holdout, production floor 0.48 | **−0.0009** | +0.0038 | ≈ −0.0005 |
| **128 movies** | **+0.0013** [+0.0004, +0.0022] | +0.0182 | **+0.0031** [+0.0014, +0.0049] |

**The same change, measured on 12 movies and on the 128 that contain them,
comes out with opposite signs**, and the 128-movie interval excludes zero.
This is not sampling noise in the loose sense -- the pipeline is
deterministic, eight `random_state` values give identical output -- it is
that those 12 movies are not a representative draw of the 128 for this
quantity.

**And that holdout is what almost everything in this project is tuned
against.** `divbench6`'s DIV_W argmax, round 1's hpara verdict, `divradius`'s
sister radius, every AUC in `accdiag` and `ttascore`: all 12 movies. The
practical rule that falls out is narrow and worth keeping: **when a change
touches the whole 128, measure it on the whole 128.** A 12-movie read of a
128-movie change is not a cheap approximation of it; here it was the wrong
sign.

#### 44b6 IS IN, and it is a NULL (2026-09-09, hp44 6.8 min + hp44_ref 6.4 min, the control's hard anchor EXACT)

`hp44` / `hp44_ref`: the production ranker (`edge_ranker_ep175v3_bidir`, pool
D175) at the production configuration, on the 12 44b6 movies D175 held out.
The control's own assert: **`ANCHOR +BIDIR F4-44b6 adj_J 0.8384 want 0.8384 ->
OK (+0.0000)`** -- the second cross-harness, four-day reproduction of
`ilpsweep`'s measured F48 delta of −0.0131.

| 44b6, 12 movies | edge_J | adj_J | div_J | score |
|---|---|---|---|---|
| F4 (refine) | 0.8264 | 0.8384 | 0.0000 | 0.8384 |
| F4HP (hpara) | 0.8267 | 0.8386 | 0.0000 | 0.8386 |
| paired bootstrap mean | +0.00002 | **−0.00015** | 0 | **−0.00015** |
| 90% interval | | **[−0.00644, +0.00514]** | | P(>0) **0.509** |

**A coin flip.** 4 of 12 movies up, median −0.00080. The weight-averaged
aggregate (+0.0002) and the bootstrap mean (−0.00015) disagree in sign
because the aggregate is carried by the two heaviest movies (`8f5ab931`
w=658 +0.0099, `c96cfa10` w=650 +0.0090) while a movie resample lets the
many small negatives speak. Both sit on zero.

**Why div_J cannot contribute here, which is a specific reason and not a
shrug:** these 12 movies contain **zero scored GT divisions** (F4 0/11/2,
F4HP 0/16/2 -- hpara proposes 5 more false forks and still finds no true
one). div_J is 0.0000 in both arms, so 44b6's score IS its adj_J, and the
+0.0182 div_J that carries a third of hpara's 6bba gain has no way to
express itself on this domain.

| domain | hpara − refine | interval | reading |
|---|---|---|---|
| 6bba, 128 movies | **+0.0031** | [+0.0014, +0.0049] | positive, excludes zero |
| 44b6, 12 movies | −0.0002 | [−0.0064, +0.0051] | **null** |
| zebrahub | `zhpara` running | | |

#### What is still missing before this can be submitted

- **44b6: DONE, and null** -- see the section above. It had never been
  measured, and not because of hpara: `divbench9` defines
  `GROUPS = (("6bba", …), ("44b6", …))` and `DOMAINS` and **uses neither**
  (dead code from an ancestor), so the bench has never scored 44b6. `SET_44B6`
  exists in it only to hold 12 movies out of the ranker pool and to feed the
  joint gate. (I told the user both arms would print 44b6 rows; they do not.)
  `hp44` / `hp44_ref` ran `MODEL_BIDIR` (pool D175, which held those 12 out)
  at the production configuration instead.
- **zebrahub: DONE, and POSITIVE** -- score +0.0011, 90% [+0.0005, +0.0017],
  P(>0) 0.998 on 38 of 41 crops, all three REF anchors exact. See the zhpara
  entry above. That makes it **two positive domains, one null, none
  negative**, which meets the three-domain rule. Two things still outstanding
  before a submission: the 3 largest zebrahub crops (zhpara died at 39 of 41
  and adj_J is weight-averaged), and the 44b6 re-read on 71 movies once
  `hpara44` + `ephp44` land.
- Three domains, or it does not count. One domain positive is a property of
  that domain.


### ⭐⭐⭐⭐ ttabench (2026-09-09, 178 min CPU, the view-1 identity anchor EXACT on 12/12): edge-feature TTA costs adj_J −0.0051 and edge_J −0.0074 through the FULL pipeline, both intervals excluding zero. The link I had asserted without measuring is now measured -- and my prediction that it could not be measured was wrong

⚠ **AMENDED 2026-09-09, later the same night, by the rkpool entry above.**
The fold rankers in this run were **100% noTTA-trained**: the override
replaced `EDGEPROB_F4`, and stage A trains from `LAB`, which is built with
`EP_OF["BASE"]`. So the arms differ in the ranker's TRAINING distribution as
well as its input, and −0.0051 is "TTA bolted onto a ranker fitted without
it", not "TTA". Tested claim that survives: *bolting D4 feature averaging onto
the shipped, non-TTA-trained ranker loses adj_J*. Tested claim that does NOT
survive: *D4 feature averaging is worth nothing to this pipeline.* The matched
version needs tta4 in the training rows -- `ttacache` is caching them.
Everything else below (the view-1 identity anchor, the row-set check, the
per-movie table) is unaffected.

Three hours earlier I wrote that the head's AUC dropping meant "the ~20 GPU
hours are refused". The user: *"边概率头变了和它不值之间的关联是？"* -- and
there was none. `ttascore` measured the edge head's own probability. The
ranker had not run, the ILP had not run, nothing had been scored. Then, told
that a 12-movie score test would probably come back "cannot tell":
*"只看12部的分数会死？"* and *"你都知道测不出来了，还在这里诚实的测不出来"*.

Both landed, and the second one was right in a way I did not expect: **the
test was not underpowered.**

**Where the ±0.005 I was quoting came from, and why it did not apply.** It
was hparabench round 1's interval, and that comparison changed the
COORDINATES -- detections moved, the gate re-formed, the candidate pair set
changed, so the arms shared almost nothing. Here `ttaprobe` held
`det_logits` at view 0 on purpose: the detections are byte-identical, the
`(t, i, j)` rows are identical, and the only difference in the entire
pipeline is the value in two columns. Borrowing an interval from a much
looser comparison is not measuring this one.

**THE ANCHOR, and it was free.** `ttaprobe`'s `noTTA` arm is the same
wrapper with the averaging switched off, so it must reproduce
`edgeprob_F4_` bit for bit. **12 of 12 movies, worst |dp| = 0.000e+00.** The
wrapper is transparent at view 1, so `hparabench_ref`'s CSV is a legitimate
control and the tta4 delta carries no offset of the wrapper's own. That
check ran in the first seconds of the kernel, before three hours of CPU.

#### The result, paired over the 12 movies that carry a tta4 cache

`tools/ttapair.py`, 4,000 draws, the same resample for both arms. 116 of the
128 movies are byte-identical between the runs by construction, and the
harness confirmed it: exactly 12 rows differ.

| | noTTA | tta4 | delta | 90% interval | P(>0) |
|---|---|---|---|---|---|
| `edge_J` | 0.9235 | 0.9159 | **−0.00741** | **[−0.01309, −0.00241]** | **0.003** |
| `adj_J` | 0.9265 | 0.9213 | **−0.00507** | **[−0.01051, −0.00023]** | **0.040** |
| `div_J` | 0.0769 | 0.0976 | +0.01933 | [−0.07094, +0.11475] | 0.632 |
| score | 0.9342 | 0.9311 | −0.00313 | [−0.01370, +0.00932] | 0.312 |

**The two terms that could be measured are both negative and both exclude
zero.** The composite score does not, and the reason is entirely `div_J`:
twelve movies contain twelve GT divisions, so its interval is 0.19 wide and
it swamps a term that enters at one tenth the weight. adj_J is the term
carrying weight 1.0, and it is down.

Per movie, 5 of 12 up on adj_J -- not unanimous. The aggregate is driven by
`283bf9f1` (−0.027) and `7af54fde` (−0.010), both heavy.

Diluted over all 128 the same numbers read adj_J −0.0005, exactly a tenth of
−0.0052, which is the 12-of-128 arithmetic and a useful check that nothing
else moved.

#### The chain, now measured end to end

    D4-average the U-Net features
      -> the edge head's own AUC falls 0.00417        (ttascore, 12/12 movies)
      -> 10.6% of gated pairs swap order              (Spearman 0.85-0.97)
      -> through ranker + ILP + patches + scorer:
         adj_J  −0.0051  90% [−0.0105, −0.0002]
         edge_J −0.0074  90% [−0.0131, −0.0024]

⚠ **I asserted the last step before measuring it and the measurement happened
to agree. That is luck, not method** -- and working-agreement.md records that
offline edge AUC anti-predicted this pipeline three times running, so the
prior was against me. The verdict stands; the way I reached it the first time
does not.

⚠ Two limits kept in view. This is the FIRST edge head only -- `ttaprobe`
cached `edgeprob_` and not `edgeprob2_`, so the second head stays
single-view on the 12 movies that have one. And 12 movies is 12 movies:
`ttacache` (140 movies, tta4, ~4.1 GPU h) is running to redo this on the
full scored set, which is affordable and was refused earlier for a cost
estimate belonging to a much bigger scope.


### hparabench ROUND 2 + zhpara PUSHED (2026-09-09, three CPU kernels). Full coverage at last: 128 of 128 scored movies now differ between the arms instead of 12, the AUC discriminator widens with it, and zebrahub gets its own harness. THE READING IS FIXED HERE, BEFORE THE NUMBERS

Round 1 could not answer its own question. It scored 128 6bba movies and
only TWELVE carried hpara coordinates, so every aggregate was diluted about
tenfold and had to be re-read on the changed subset by hand; its AUC
discriminator never ran at all (the version on Kaggle was still round 1's).
Three cache kernels have since landed -- `hparacache` v3 at 140 movies,
`ephpcache` and `ep2hpcache` behind it -- so the arm can finally be what it
was supposed to be.

**What is running.**

| kernel | arm | scope |
|---|---|---|
| `biohub-hparabench` | `DET_SRC=F4HP` | 128 6bba + the 12 44b6 D187 rows, hpara coordinates |
| `biohub-hparabench-ref` | `DET_SRC=F4` | the same, refine coordinates, hard anchors asserted |
| `biohub-zhpara` | REF vs HP | zebrahub, both arms in one run, dcmerge's solver |

**One real bug was caught by its own new assert, and it is worth recording
because it would have produced a NUMBER rather than a crash.** v6 died in
60 seconds on

> `edgeprob2: only 12 of the 128 scored 6bba movies have an F4hp cache`

`edgeprob2` is the OPTIONAL second head -- divbench9 reads it with
`.get(name)` -- and only 24 movies have an `edgeprob2_F4_` cache at all, so
116 of the 128 scored movies are solved without it **in both arms**.
`ep2hpcache` cached 140. Merging all 140 in would have handed the hpara arm
a second head on 116 movies where the control has none: the arms would then
have differed in the PIPELINE, not in the placement, and the delta would
have measured the second head. **The override now REPLACES and never ADDS**
-- the hpara dict is cut down to the keys the refine arm already has -- and
a movie with a refine cache and no hpara counterpart is a hard error.

**PRE-REGISTERED READING, round 2.** Same two-by-two as round 1, now with
the power to use it. `AUC` is the fitted ranker's out-of-fold AUC on
`poolsize`'s labels, same pairs, same labels, same model, the ONE difference
being where the points were placed.

| AUC | 6bba score | reading |
|---|---|---|
| down | down | the ranker MIS-READS hpara. It was fitted on refine coordinates and never saw ones this clean; refitting is the fix and the ~6 GPU hours are justified |
| flat | down | the ranker is fine and something downstream ate it -- the ILP's thresholds and the patch constants were all tuned on refine coordinates |
| flat / up | flat | it genuinely does not convert. Placement is better (`refinecmp`, jitter −31%) and the score cannot see it. Park the direction with that as the finding |
| up | up | it converts, and round 1's −0.0018 was the twelve-movie dilution talking |

**And the domain rule applies, so the 6bba column alone decides nothing.**
Three domains this time: 6bba (128 movies, the scored set), 44b6 (the 12
D187 rows, printed by both arms), zebrahub (`zhpara`, every crop carrying
both caches). A change counts when the sign agrees; a win in one domain is a
property of that domain. ⚠ `zhpara` runs a DIFFERENT pipeline -- dcmerge's
solver has no edge head and no fork gate -- so it answers "does better
placement help the LINKER", and `hparabench` answers "does it help the
shipped submission". Those are two questions and the write-up must not merge
them.

**What `zhpara` anchors, all three cheap and all three catching a real
failure mode.** The REF arm must reproduce dcmerge v2's recorded zebrahub
BASE (edge_J 0.8036, adj_J 0.8093, div_J 0.0230) on dcmerge's own 8 crops;
the hpara cache must carry byte-identical ground truth, or the arms are
scored against different truths; and the detection count must match frame by
frame, because hpara re-places and never re-detects. It also prints
`|hpara − refine|` per crop, so a null cannot be confused with a no-op --
the distinction `ttascore` had to make tonight on the TTA arm.


### ⭐⭐⭐⭐ EDGE-FEATURE TTA VERDICT (2026-09-09, `ttaprobe` v3 69 min T4 + `ttascore` 2 min CPU, gate anchored 12/12): averaging the U-Net features over D4 views makes our edge probability **WORSE**, −0.0042 AUC, 12/12 movies, bootstrap P(>0) = 0.000. The last public-0.946 candidate is closed and the ~20 GPU hours are REFUSED

The sixth and last difference between our configuration and the public 0.946
notebook was the only one that was never a tuned constant: they average the
U-Net output over D4 views before reading the edge features, and credit it
with +0.005 (their 0.941 -> 0.946). This measures the mechanism directly, on
the one thing it can change: is the edge probability more informative?

**The probe. Patch `model.encode`, not `predict_video`.** The U-Net output
is averaged over `n` D4 views (forward transform, encode, inverse transform,
mean); the detection logits are left alone, so the DETECTIONS, their order
and their identity are byte-identical across arms and the cached `(t, i, j)`
rows join exactly. 36 passes = 12 holdout 6bba movies x `noTTA/tta4/tta8`,
7,128 encode calls, 23,760 extra U-Net passes.

**The wrapper is not a no-op, and the log said so before any AUC existed.**
`fwd>0.5` falls on 12 of 12 movies; median `p_fwd` falls on 11 of 12
(0.827 -> 0.728 on `2312ac41`, 0.669 -> 0.399 on `283bf9f1`). Whatever
came back was going to be a real number.

| arm | views | AUC `p_fwd` | AUC `p_rev` | AUC `prod` | vs noTTA | \|dp\| p50 / p90 |
|---|---|---|---|---|---|---|
| noTTA | 1 | 0.99037 | 0.99073 | **0.99194** | +0.00000 | -- |
| tta4 | 4 | 0.98548 | 0.98674 | **0.98777** | **−0.00417** | 0.019 / 0.093 |
| tta8 | 8 | 0.98589 | 0.98672 | **0.98786** | **−0.00408** | 0.020 / 0.102 |

14,636 labelled pairs, 0.6625 positive, `poolsize`'s own labelling rule, the
same rows in every arm.

**It is not a wash, it is a loss, and it is not noise.** Per movie, tta4 is
worse on **12 of 12** and tta8 on 10 of 12. Movie-level paired bootstrap,
2,000 draws: tta4 −0.00415 90% CI [−0.00563, −0.00276], tta8 −0.00408
[−0.00599, −0.00232], **P(>0) = 0.000 for both.** And the view count does
not rescue it -- tta8 is indistinguishable from tta4, so this is not "not
enough views".

**THE TESTED CLAIM, narrowly.** D4-averaging the U-Net features before the
edge head makes the head's own forward, reverse and product probabilities
LESS able to separate true children from gated non-children on 6bba. **That
is the whole of what was measured.** The ranker never ran in this
measurement; the ILP never ran; no graph was scored.

**And "the probabilities got pushed down" is the wrong description of what
happened.** AUC is invariant to any monotone transform, so a uniform squash
would leave it EXACTLY unchanged. It moved, which means the ORDER moved:
Spearman between the noTTA and tta4 columns runs 0.85-0.97 per movie, and
**10.6% of randomly drawn pairs swap order**. The fall in median `p_fwd` is a
side effect; the reordering is the event.

⚠ **WHAT DOES NOT FOLLOW, and the first draft of this entry asserted it
anyway** -- *"the ~20 GPU hours are refused on that"*. The user:
*"边概率头变了和它不值之间的关联是？"*. There is no measured link. The chain
would have to be `head AUC down -> the ranker's three probability columns get
worse -> its edge ordering gets worse -> the ILP picks worse edges -> score
down`, and **not one of those steps was run.** This project's own
working-agreement says why that matters: *"An offline AUC is not a score. On
2026-08-24 edge AUC anti-predicted the pipeline three times running. Measure
adj_J."*

**What the decision actually rests on**, stated as what it is:

1. Their claimed MECHANISM is "the edge features get better". Here they get
   worse -- not flat, worse, on 12 of 12 movies with P(>0) = 0.000. So the
   reason they give for the +0.005 does not reproduce on our features. That
   part is measured, not inferred.
2. Everything downstream is a COST argument: ~20 GPU hours to re-cache three
   edge-probability datasets, against a mechanism whose one measurable
   precondition already failed.

That is a cost judgement, not a demonstration that TTA would lose score.
**The honest test is cheap and CPU-only** -- the 36 TTA probability caches
already exist, and the ranker, ILP and scorer all run on CPU -- but it would
score only the 12 movies that carry them, and hparabench round 1 showed a
12-movie score comparison has an interval of about ±0.005. It would very
likely return "cannot tell". That would be an honest "cannot tell" rather
than this paragraph's inference, which is the difference worth keeping in
mind if this line is ever reopened.

**Why it can lose rather than merely fail to help.** TTA is only free when
the network is equivariant under the transforms being averaged. Averaging
the outputs of a network that is NOT is a blur, and a blurred heat map costs
exactly the local contrast the edge head reads. Nothing in our training
recipe makes the U-Net D4-equivariant, and the size of the shift here
(median |dp| 0.02, p90 0.10) says it is not close.

**Residue -- what this does NOT close.**

- **It measures TTA on the EDGE features only, with the DETECTIONS held
  fixed.** That is deliberate -- it keeps the `(t, i, j)` rows identical
  across arms so the AUC compares the same pairs -- but their step averages
  the detection logits and the features together in one pass. Ours isolates
  the feature half.
- 6bba only, 12 movies. No 44b6, no zebrahub.
- Our U-Net is not theirs. A D4-augmented model would be a different
  experiment, not a re-run of this one.

**A correction to the first draft of this entry, which said their +0.005
should be looked for in DETECTION TTA "which we do not run". We do run it.**
`fuse6bba.branch_maps4` and `detfuse3` build every one of the four F4
heatmaps as *"the sigmoid of its 4-flip logit mean"* -- identity, flip y,
flip x, flip both -- and our detector has worked that way since F4 landed
(LB 0.925, 2026-08-31). What the public notebook has that we do not is
**eight** views there instead of four, on the OFFICIAL detector rather than
our ensemble. Two things bear on whether that gap is worth anything, and
both point the same way: `tta8` is indistinguishable from `tta4` here, so
doubling the view count bought nothing on this network; and our detector's
four flips are *the same four transforms* `tta4` uses (`_d4_fwd` views 0-3
are exactly `_flip`'s), so this is not an exotic augmentation that our
pipeline has never seen.

**Which sharpens the mechanism, and makes it testable rather than a story.**
The same four-flip average that our detector has used successfully for
weeks makes the edge probability worse. The difference is not the transform,
it is what reads the averaged map. **Peak-finding is robust to a blur that
leaves the maximum where it was; a head that reads the feature VALUE at a
point is not.** Our detector averages LOGITS and takes one sigmoid at the
end, which is the numerically right way to do it; `model.encode`'s
`unet_out` is averaged as it comes.

**With this, six of the SEVEN differences from the public 0.946 notebook are
closed:** three from the existing record (disappearance cost, ILP division
weight, density-adaptive gap), two by `gapcross` (gap radius, short-track
floor + rescue), and edge-feature TTA here. ⚠ The first draft of this
paragraph said "all six" and dropped one -- the comparison table's own
verdict column marks `division parent / sister um` **open**, and it had never
been measured. `divradius`, the entry above, closes it (their 9.0 / 14.0
costs −0.004 score). **Nothing in that notebook is left to
port.**

#### The bug this ran into, and the check that now exists

`ttaprobe` v3 printed its own AUC table and it was **wrong**, by one
subscript:

```python
prim = load_primaries(name, "F4")          # [ (N_t, 3) ] per frame
mp = match(np.asarray(prim[t_][0], ...))   # <- row 0 OF THE FRAME
```

`load_primaries` returns a list of arrays, not a list of `(coords, scores)`
tuples, so `prim[t][0]` is the first DETECTION of frame `t`. Only detection
index 0 could ever match, as source or as target. It printed a table on
**565 rows where the correct rule gives 14,636** -- and reported tta4 at
−0.052, an order of magnitude off the truth. The bug is in the labeller,
which runs after every cache is written, so the GPU work was intact and
`ttascore` redid the readout on CPU for 2 minutes.

**What caught it was not the code, it was a known population size.** 565
labelled rows across 12 movies is a quarter of one per frame; `accdiag` had
recorded ~14,817 for the same rule on the same holdout. `ttascore` now
carries both checks the probe lacked:

- **the gate anchor** -- rebuild `gated_pairs` from its own primaries and
  demand the cache agrees on the per-frame pair COUNT and on every cached
  pair being inside `GATE_UM`. Passed 12/12, worst distance 10.000 um.
  A wrong primaries index fails both.
- **a row-count floor** -- refuse to print a table on under 5,000 rows.


### ⭐⭐⭐ gapcross RESULT (2026-09-09, 238 min CPU, anchor exact, CONTROL PASSED): the interaction does NOT bite, and both live post-link candidates from the public notebook are worth nothing here. Only edge-feature TTA is left

Anchor `ANCHOR_D187 F4-6bba(12) adj_J 0.9216 -> OK (+0.0000)`. Control
`g9c3` -- the pre-cap20 baseline -- **loses, −0.0006**, so the harness is
genuinely exercising the gap stage and the other five rows are readable.
That check was written into the script's own output before the run, not
argued afterwards.

| arm | gate | cap | short | rescue | adj_J | div_J | score | d score |
|---|---|---|---|---|---|---|---|---|
| REF | 9.0 | 0.02 | 4 | off | 0.9133 | 0.1337 | 0.9266 | +0.0000 |
| g5 | **5.0** | 0.02 | 4 | off | 0.9132 | 0.1340 | 0.9266 | **−0.0001** |
| g5c3 | **5.0** | **0.003** | 4 | off | 0.9127 | 0.1344 | 0.9261 | −0.0005 |
| g9c3 | 9.0 | **0.003** | 4 | off | 0.9124 | 0.1360 | 0.9260 | −0.0006 (CONTROL) |
| s6 | 9.0 | 0.02 | **6** | off | 0.9133 | 0.1343 | 0.9267 | +0.0001 |
| s6r | 9.0 | 0.02 | **6** | **on** | 0.9134 | 0.1343 | 0.9268 | +0.0002 |

**THE INTERACTION THE 2x2 EXISTED FOR: it does not bite.** Read the
diagonals. At the loose budget, tightening the radius costs −0.0001
(REF -> g5). At the tight budget, tightening the radius costs −0.0001 more
(g9c3 −0.0006 -> g5c3 −0.0005, i.e. nothing, and slightly the other way).
**The radius does nothing at EITHER budget.** Our +0.002 leaderboard win
came from the budget and cannot be un-bound by their radius, which is the
question the user asked before going to bed. It is also the answer that a
one-at-a-time test could not have given: g5 alone reads "harmless", g5c3
alone reads "−0.0005 = the radius is bad", and only the square shows the
budget carrying both.

**And it costs the candidate.** Their `GAP_CLOSE_UM` 5.0 buys nothing here.
Ours is still the untouched public baseline 9.0 and can stay there.

**The short-track pair, also nothing.** `SHORT_MIN` 4 -> 6 is +0.0001,
consistent with the leaderboard record for the reverse move (6 -> 4 scored
0.000). Adding the rescue on top is a further +0.0001. The mechanism is NOT
broken -- the log shows it proposing 236 components and rescuing 114 of
them, 503 nodes -- it simply does not pay.

**HOW TO READ SIX ROWS INSIDE ±0.0006.** ⚠ CORRECTED 2026-09-09, and the
original text got this backwards. It said "accdiag put this holdout's own
noise at 0.002-0.003" and used that to call every row here "no detectable
effect". **`accdiag` never measured any such thing** -- it measured how often
the acceleration column is computable -- and there IS no noise floor of that
size, because **this pipeline is deterministic**: eight `random_state` values
through the gate give the identical 46 / 39 / 79, and the local
reproduction of divbench3 and divbench8 hits both recorded rows exactly.

Repeat-run noise is **zero**. So `s6r`'s +0.0002 is not "inside the noise",
it is an exact property of these 128 movies. What a bootstrap interval
measures is something else entirely -- **whether the SIGN would survive a
different draw of movies** -- and that is a transfer question, not a
measurement-error one. Conflating them dismisses real, reproducible
differences, and it collides with this project's own submission rule: local
positive + LB positive is a real gain, however small. What IS established is
comparative and
survives the noise floor: whatever the radius does, it does the same thing
at both budgets.

**What is left.** Of the six differences between our configuration and the
public 0.946 notebook's, five are now closed -- three from the existing
record (disappearance, ILP division weight, density-adaptive gap) and two
measured here. The remaining one is the only candidate that was never a
tuned number: **edge-feature TTA**, `ttaprobe`, queued behind Kaggle's
two-concurrent-GPU-session limit.

### ⭐⭐⭐⭐ THE PUBLIC 0.946 NOTEBOOK, READ AGAINST OUR RECORD (2026-09-08/09). Six of its knobs differ from ours; THREE we already measured and rejected with reasons, ONE is architectural, and only TWO are live. Plus the three-domain hpara caches, which now exist for 6bba, 44b6 AND zebrahub

The user, on being told public notebooks were appearing faster than our own
score moves: *"真是伤脑筋，一堆人上赶着开源。我那分的速度还没人家开源的速度
快"*. Then: *"除了这一点以外没有别的有用的信息了吗"*, and the systematic
read below is the answer to that.

**Three notebooks, ONE lineage.** `biohub-cell-tracking-0-946-lb` (Reyhan
Ksatria, 2026-09-08), `biohub-harmonic-fusion` (120 votes) and
`biohub-942tta` (96 votes) are the same code; harmonic-fusion is 942tta with
the comments stripped and a new title. It is not three teams passing us, it
is one.

**Their whole pipeline is driven by 57 `BIOHUB_*` environment variables, and
the official `predict_unet_transformer.py` reads NONE of them** (checked: 0
occurrences). The knobs are theirs, added by `_patch_text` rewriting the same
official script we use at runtime. So nothing is a switch we can flip -- but
the tuned VALUES are readable, and so are the patches.

**The comparison that matters, done by MEANING not by name.** Of eleven
constants that exist on both sides, **five are byte-identical** -- 2.25,
10.0, 0.0076, 0.00375, 0.48. Numbers like 0.0076 do not coincide: both
pipelines descend from the same public baseline and most of the
configuration IS shared. The user's objection to an earlier draft of this
entry -- *"你的意思是让我放弃我们调了这么久的参数直接用别人的吗"* -- was
right to reject the framing, because there is no rival SET to adopt. There
are six numbers.

| | theirs | ours | verdict |
|---|---|---|---|
| gap radius `GAP1_GATE_UM` | 5.0 | **9.0** | LIVE. Ours is the untouched public baseline; THEY moved it |
| short filter | 6 | 4 | live only paired with their rescue, below |
| division parent / sister um | 9.0 / 14.0 | 12.0 / 15.0 | **REJECTED, measured** (`divradius` 2026-09-09: their parent radius −0.0039 score, their sister radius an exact zero) |
| ILP disappearance | 2 | 1.5 | **REJECTED, measured** |
| ILP division weight | 1.2 | 0.55 | **ARCHITECTURAL, not a knob** |
| gap density adaptive | on | absent | **not worth it, measured from their code** |

**Rejected on our own evidence, not on taste:**

* **disappearance 2.** `ilpsweep` (2026-09-05) already ran the notebook's ILP
  settings on our nodes: **+0.0002 / −0.0115 / −0.0010**, deleting only 6-15%
  of our nodes where theirs deletes 39%, and the chains it deletes are
  **62% / 91% / 58% TRUE edges** where the scorer looks. REF is the local
  optimum of that knob. Their 2 is tuned for a node set far smaller than ours
  (18,618 against our 23,371 on the same movie).
* **ILP division weight 1.2.** The user's point, and the constant's own
  comment confirms it: ours is *"set from the fine sweep's argmax"*, swept
  WITH the fork gate present. The public baseline is 1.0; they went UP to
  make the ILP emit divisions natively and filter them with geometry +
  DeepCenter, we went DOWN to 0.55 and let the learned fork gate decide.
  Copying 1.2 would flood the graph with divisions the gate was never fitted
  to arbitrate. Different stage doing the work, not a better number.
* **gap density adaptive.** Read from their source rather than their prose:
  the radius moves by `clip(0.040 * (local_spacing - 6.5), ±0.125)` um, i.e.
  **at most ±0.125 on a 5 um radius -- 2.5%** -- and `GAIN 0.04` needs a 3 um
  departure from the reference to reach even that. It also appears nowhere in
  their own 0.934 -> 0.939 -> 0.941 -> 0.946 progression, so no gain is
  attributed to it.

**The one thing that is not a tuned number at all, and the only candidate
with a stated delta: EDGE-FEATURE TTA (+0.005, their 0.941 -> 0.946).** Their
own summary: the eight D4 views were already averaged into the DETECTION
logits, but the UNet feature map handed to the edge predictor came from the
original orientation only. The code difference is one character:

```python
_,       det_flip = model.encode(imgs_flip)   # baseline: features discarded
_u_flip, det_flip = model.encode(imgs_flip)   # theirs: kept, inverted, averaged
```

`model.encode` returns BOTH; the baseline throws the features away. With
detection TTA already on it costs them nothing. **Our `edgeprob_cache.py`
comment is correct and is exactly why this gap exists here:** *"det_tta only
affects the detection logits, which are discarded here... Off saves three
U-Net passes per window."* Right about the official code -- which is why they
had to ADD feature TTA -- and it leaves our edge features single-view.

`ttaprobe` (pushed 2026-09-09) decides whether it is worth ~20 GPU hours,
for about one. It wraps `model.encode` so the D4 average happens inside it,
leaving `predict_video` byte-identical and `det_tta` off; detections cannot
move because `_detect_cells_pooled` is already injected. 12 holdout movies x
3 arms (1, 4, 8 views), labelling its own pairs, reporting single-column AUC
of `p_fwd` -- no bench, no ILP, no thresholds. It also prints how far the
probabilities moved and the encode-call counts, so a flat AUC can be told
apart from a wrapper that silently did nothing.

**`gapcross` (pushed): the two live candidates, CROSSED.** The user, going to
bed: *"注意不同候选之间会不会互相影响，比如说一个放开限制一个收紧限制之类的"*.
That is this situation exactly:

* our only pre-fork-gate LB win was **loosening** the gap BUDGET
  (0.003 -> 0.02, +0.002), and it worked because the budget was provably
  binding -- 30 of 83 datasets at the cap, 64% of demand unrepaired;
* their step **tightened** the gap RADIUS. A tighter radius proposes fewer
  repairs, so it can un-bind the budget and delete our +0.002 by itself.

Six arms off ONE ILP solve per movie (`patch_final` rebuilds from a fresh
copy): REF, g5, g5c3, **g9c3 as a CONTROL that must lose**, s6, s6r.

And the interaction that makes a candidate a provable no-op alone: their
rescue only considers components of length `[4, OUTPUT_MIN_TRACK_LEN)`. **We
run SHORT_MIN = 4, so that window is EMPTY here.** Porting the rescue alone
would change nothing and read as a null result about the rescue. The unit
under test is (SHORT_MIN 6 + rescue).

**Separately, the three-domain hpara caches all landed, both anchors exact:**

| kernel | scope | cost | anchor |
|---|---|---|---|
| `hparacache` v3 | **140** movies (128 6bba + 12 44b6) | 203 min T4 | every frame of every movie reproduces `fdet_F4_` exactly |
| `zhhpcache` | **41 zebrahub crops** | 44 min T4 | 4,100 of 4,100 frames reproduce `zhdet_` exactly |

|hpara − refine|: 6bba/44b6 p50 median **0.460** um (p90 0.789, max 1.479),
zebrahub p50 **0.673** (p90 1.059, max 4.674, 1.28M detections). The
zebrahub displacement being larger is expected from a different instrument
and voxel size, and it is the first hpara number that exists for a third
domain at all -- which is what the user's *"比赛的验证集两个都不是"* makes the
point of.

### ⭐⭐⭐ hparabench ROUND 1 (2026-09-08, biohub-hparabench v4 + hparabench-ref, ~170 min CPU each, control anchor EXACT): THE BETTER PLACEMENT DOES NOT CONVERT. Edges get slightly WORSE (adj_J −0.0018 on the 12 movies it touches) and the apparent score gain is TWO division events. Round 2 with the AUC discriminator is pushed, because "not worth it" and "the ranker mis-reads it" are still not separated

`refinecmp` established that `hpara` PLACES the detections better -- jitter
p50 1.29 -> 0.89 um on 6bba, a 31% cut, at no node cost. This asks the next
question: does the ranker and the ILP CONVERT that into score.

**Control first.** `hparabench_ref` (DET_SRC=F4) reproduces divbench9 on all
four columns over its 128 rows -- edge_J 0.9104, adj_J 0.9133, div_J 0.1337,
score 0.9266 -- so the harness did not move and everything below is about
hpara.

**Twelve movies changed, not 128.** The bench scores `SIX` = all 128 6bba
movies; the hpara caches cover the 24 EVALUATION movies, of which 12 are
6bba. The other 12 are 44b6 and **this bench never scores 44b6 at all** (the
71 44b6 movies are training data only, on BASE coordinates) -- so those 12
caches produced nothing and this round says NOTHING about 44b6. The 116
untouched movies are identical in both arms, which dilutes the aggregate by
about 10x; every table below is the 12 that moved.

| | refine (ships) | hpara | delta |
|---|---|---|---|
| edge_J | 0.9235 | 0.9217 | **−0.0018** |
| adj_J | 0.9265 | 0.9248 | **−0.0018** |
| div_J | 0.0769 | 0.1250 | +0.0481 |
| **score** | 0.9342 | 0.9373 | **+0.0031** |

**The +0.0031 is two division events, and that is the whole of it.** Counts,
before writing any mechanism about them:

| | div TP | div FP | div FN | edge TP | edge FP | edge FN |
|---|---|---|---|---|---|---|
| refine | 3 | 27 | 9 | 9,836 | 338 | 477 |
| hpara | 5 | 28 | 7 | 9,828 | 350 | 485 |
| delta | **+2** | +1 | −2 | **−8** | **+12** | **+8** |

Twelve GT divisions across twelve movies; 3 caught became 5. div_J carries
0.1 weight but the base is small enough that two events move it 0.048. That
is noise wearing the shape of a result.

**What IS stable is the edge side, and it is NEGATIVE.** −8 TP, +12 FP, +8 FN
out of ~10,650, every component the same direction. Per movie 6 up 6 down.
The paired bootstrap spans zero everywhere: CHANGED-only +0.0026
[−0.0023, +0.0077], all-128 +0.0003 [−0.0002, +0.0009].

**The tested claim, narrowly.** With the ranker FIXED and fitted on BASE
coordinates, swapping refine for hpara on 12 movies does not improve the
6bba score, and moves the edges slightly the wrong way. It does NOT touch
`refinecmp`: that measured placement, and placement is still better.

**Residue -- what this direction still owns.**

- The placement result is untouched and independently measured (recall UP,
  offset down, jitter down 31%, on BOTH domains).
- The edge regression is small but coherent, and coherent is what a
  DISTRIBUTION SHIFT looks like: the ranker never saw coordinates this
  clean. That is a live alternative to "not worth it" and this round cannot
  rule it out.
- 44b6 is completely unmeasured here.
- n = 12 movies, 12 divisions. Nothing about this round has the power to
  settle anything.

**Round 2, pushed 2026-09-08 (v5 both arms, ~170 min CPU each, no GPU).**
The discriminator: label the F4/F4HP pair rows with `label_rows` -- which
reads the GT itself, so it costs only CPU -- and score them with the SAME
fitted model, out of fold, on the same 12 movies. Same pairs, same labels,
same model; the ONE difference is where the points were placed.

| AUC | score | reading |
|---|---|---|
| down | down | the ranker MIS-READS hpara; refitting is the fix and the 6 GPU h is justified |
| flat | down | the ranker is fine, the ILP's refine-tuned thresholds ate it |
| flat | flat | it genuinely does not convert -- park the direction |

**Cost of the decisive experiment, measured not guessed**, from the three
cache kernels' own runtimes: hparacache 24 movies / 35.9 min, ephpcache
15.2, ep2hpcache 15.1 -> about **6 GPU hours** to extend all three to 128
movies. Do not spend it until the AUC says which world we are in.

**Two process failures this round, both worth naming.**

1. **`hparapair` aggregated the score wrong** and was caught only because
   the control was run through the anchor: adj_edge_jaccard is NOT
   `edge_J * (1 - 0.1 * ratio)` on the aggregate. The scorer applies the
   node penalty PER SAMPLE and weight-averages the per-sample adjusted
   values by `tp+fp+fn`. The wrong form read 0.9126 against a recorded
   0.9133 -- **0.0007, four times the effect being measured**. Fixed from
   `tracking_cellmot/metrics.py` directly, and the anchor is now asserted in
   the output rather than printed as a number to eyeball.

2. **The F4HP arm was marked ERROR on a completely successful run.** It
   scored 128/128, wrote its CSV, then died on
   `ANCHOR_D187 0.9204 want 0.9216` and `REF_FORKS`. The maker DID demote
   the anchors -- it edited `ANCHOR_HARD`/`ANCHOR_SOFT`, and those two
   anchors never consult those dicts; both compare against a hard-coded
   number and append straight to `ANCHOR_FAIL`. Two anchor mechanisms, and
   the demotion touched the one that was not in the way. Both anchors sit on
   HOLDOUT_6BBA -- the very 12 movies this arm changes -- so a miss is the
   measurement. Now printed as `[expected]`; any other anchor still aborts.
   The CSV is written before the assert, which is why v4's data was usable
   and no re-run was needed for this alone.

### ⭐⭐⭐⭐⭐ refinecmp (2026-09-08, biohub-refinecmp v3, 36.5 min T4, both anchors exact): PUTTING THE DETECTION AT THE PARABOLIC VERTEX OF THE HEATMAP CUTS THE INVENTED MOTION BY 31% ON 6bba AND 7% ON 44b6, AT NO NODE COST. `hpara` beats the shipped `refine` on both domains, in both recall and jitter

The user, 2026-09-08: "refine跑吧". gtpeak had just measured that F4's
detections jitter 1.29 / 1.51 um p50 between consecutive frames against a
real displacement of 2.72 / 2.10 um -- i.e. **half the motion the ranker
sees on 6bba is manufactured by where the detection was placed**, and the
ranker's distance and velocity features read that as signal.

Five placement rules, the SAME peaks, so nothing is added or removed and
recall can only move by a point crossing the 7 um match radius:

| rule | what it does |
|---|---|
| `nore` | the bare pooled grid, no sub-voxel work at all -- the floor |
| `refine` | **what ships**: intensity centre of mass on the RAW volume |
| `refsm` | the same, smoothed |
| `hcent` | centre of mass of the HEATMAP in a 3x3x3 window |
| `hpara` | parabolic vertex of the HEATMAP, one axis at a time |

**6bba** (1,047 GT anchor points, recall 0.9761 reproduced exactly):

| rule | recall | \|offset\| p50 | jitter p50 | jitter >1um | >2um |
|---|---|---|---|---|---|
| `nore` | 0.9772 | 1.65 | 1.46 | 62.9% | 21.8% |
| `refine` (ships) | 0.9780 | 1.41 | 1.29 | 60.0% | 17.1% |
| `refsm` | 0.9778 | 1.46 | 1.37 | 62.8% | 19.8% |
| `hcent` | 0.9777 | 1.50 | 1.24 | 56.9% | 13.8% |
| **`hpara`** | **0.9789** | **1.31** | **0.89** | **47.3%** | **11.9%** |

**44b6** (297 points, recall 0.9596 reproduced exactly):

| rule | recall | \|offset\| p50 | jitter p50 | jitter >1um | >2um |
|---|---|---|---|---|---|
| `nore` | 0.9620 | 1.75 | 1.72 | 71.7% | 33.9% |
| `refine` (ships) | 0.9647 | 1.71 | 1.51 | 66.9% | 26.8% |
| `refsm` | 0.9637 | 1.77 | 1.54 | 70.8% | 29.7% |
| `hcent` | 0.9634 | 1.73 | 1.56 | 65.3% | 25.3% |
| **`hpara`** | **0.9647** | **1.62** | **1.40** | **58.6%** | **23.0%** |

Against the shipped rule: 6bba **+0.0008 recall, -0.10 um offset, -0.40 um
jitter**; 44b6 **+0.0000 recall, -0.08 um offset, -0.11 um jitter**. Same
sign on every column in both domains, and `hpara` is the best arm in both.
The jitter share above one micron falls 60.0% -> 47.3% and 66.9% -> 58.6%.

**Why this is worth more than the numbers look.** It adds no nodes, so none
of gtsplit's node-cost arithmetic applies -- the penalty term cannot move.
It is a strictly better answer to the question "where is this cell", fed to
a ranker whose distance and velocity features are computed from exactly
these coordinates. The 0.40 um is removed from a 2.72 um real displacement,
so on 6bba the manufactured share of apparent motion drops from 47% to 33%.

**AND IT CORRECTS AN ERROR OF MINE ON THE RECORD.** Earlier the same day I
used dcpool's null result as a prior that "picking points on the heatmap
does not work", and the user pushed back: "求你了，deepcenter本身是一个发散
的权重，或者说它的价值就是没有收敛，它的结果不能当真理更不能说明到热图上找
点是错的". They were right, and this is the measurement: the two rules that
read the HEATMAP are the two best rules for placement, and the best of them
beats the intensity-based rule that ships. `hcent` also beats `refine` on
6bba jitter. The heatmap does know where the centre is.

**NOT YET TESTED**: whether the reduced jitter converts into score. The
ranker was trained on `refine` coordinates, so swapping the placement
changes its input distribution; the honest next step is a pipeline arm with
`hpara`, retraining or at least re-checking the ranker, not a direct swap.
Recording the placement result separately from the score claim on purpose.

### ⭐⭐⭐ shoulderviz (2026-09-08, biohub-shoulderviz v1 1.9 min CPU, v2 with a rendering fix): the 313 shoulder cells LOOK like cells pressed against a brighter neighbour. 12 crops, not a measurement -- and it is NOT the density claim that was already refuted

The user, after the offline feature hunt returned only "dimmer" and
"movie-level": "把那三部烘成影片让我看看？". So: render them and look.

**What was rendered.** CPU only, no detector, ~2 minutes. For each of
the four movies carrying 194 of the 313 (`6bba_3a1849c2` 77,
`44b6_a2bb48bb` 44, `6bba_3abfe10a` 40, `6bba_268e1230` 33): one
`_overview.gif` (100 frames, thin Z-slab MIP, GT circled thin blue,
shoulder misses thick red) and six `_zoom<k>.gif` (a ~65 um crop
around one miss, +-5 frames, the slab centred on **that cell's own z**,
ordered dimmest-first). 28 files. The zooms are the instrument; the
overview is context.

**A RENDERING BUG IN v1, found by looking at the output before
believing it.** The overview draws every GT node of the frame onto a
slab fixed at `Z // 2`, so a cell at any other depth appears as a
circle on black. On `6bba_3a1849c2` t=50 that is 8 of 12 circles, and
it reads exactly like "the detector missed a cell that is plainly
there" -- the opposite of the truth. v2 circles only in-slab cells and
puts `GT in slab k/n` in the title; the zooms got the same filter for
their neighbour circles. **Nothing was concluded from v1's overviews.**
Recording it because the failure is generic: a visualisation that
silently drops a dimension will manufacture whichever finding the
viewer is looking for.

**What the zooms show.** In every one of the 12 miss frames inspected,
the red circle sits immediately against a brighter neighbour --
`6bba_3a1849c2` has its misses on the flanks of a bright chain of
cells packed nose-to-tail; `6bba_268e1230`'s sit in the dim notch
beside a bright blob; `44b6_a2bb48bb`'s sit in the dark seam between
two elongated nuclei in dense tissue. Isolated cells in the same crops,
circled blue, are found normally. This is the "one hill, two cells"
picture that gtpeak's numbers implied (nearest real peak ~6 um away, on
a neighbour) -- now visible.

**WHAT THIS IS NOT.** It is 12 crops chosen dimmest-first out of 313,
looked at by one pair of eyes, with no control group of MATCHED cells
rendered the same way. It is a hypothesis with a picture attached, not
a measurement, and it must not be written down later as if gtpeak had
proved it.

**And it does not contradict the density refutation.** That test
(2026-09-08, offline on gtsplit's CSVs) asked whether a MOVIE's
shoulder rate is explained by that MOVIE's detections per frame, and
answered no: corr +0.54 / +0.29 over 12 movies each, with
`6bba_268e1230` at the second-worst rate and the LOWEST density. What
the crops suggest is LOCAL crowding around the individual cell, which
that test never measured and cannot rule out. A movie can be sparse
overall and still pack its cells into chains -- `6bba_3a1849c2` is
exactly that.

**One number that came out of the CSVs while the render ran -- and the
WRONG reading of it, corrected the same day.** `h_gt` is NaN in gtpeak
exactly when the rounded GT index falls outside the heatmap array: **48
of 13,646 GT cells (0.35%)**, of which 30 are still matched and **17 are
in the 313 shoulder bucket**. It is almost entirely 44b6, 34 of the 48
in one movie, `44b6_c96cfa10`, along what looks like a single lineage
over t=64..94.

**SUPERSEDED, 2026-09-08:** this entry first read that as "the annotated
position is outside the imaged field" and counted those 17 as
information that was never recorded. The user killed it in one line:
"可是不是人工标注的吗，怎么可能飘出高度之外" -- **a human cannot annotate
a cell that is not in the picture.** And the refutation was already
inside the same paragraph: **30 of the 48 were MATCHED BY THE
DETECTOR**, which only ever runs inside the volume. They are in the
image.

The mechanism that actually fits is our own index arithmetic. gtpeak
reads `yc = int(round(gt_vox[j, 1] / POOL))` with `POOL = 4`; the pooled
axis has `Y/4` columns, so a cell in the last ~2 raw voxels of y (or x)
rounds to index `Y/4`, one past the end, and reads NaN. z is not pooled
and `round(z)` cannot leave `[0, Z-1]` for an in-volume cell, so z is
not the culprit. **These 17 are cells against the y/x border of the
frame that OUR MEASUREMENT could not read, not cells outside the
volume**, and nothing follows from them about recoverability. The
"edge + out-of-volume ~= 14% of the 313" figure derived from this is
withdrawn; the z-edge attribution alone stands at ~27/313 ~= 9%.
**MEASURED, `nanwhere` (2026-09-08, biohub-nanwhere, <1 min CPU)**, because
the user stopped the theorising: "我不行了，到底是什么情况，你去量一下吧我求
求你了". Every one of the 24 movies is **Z=64, Y=256, X=256**, pooled to
64 x 64, and `Y % 4 == X % 4 == 0` so `pool_xy` discards nothing (**0 GT
cells in the truncated strip**). Of the 48 NaN cells:

| axis out of range | count |
|---|---|
| y | 26 |
| x | 22 |
| **z** | **0** |

**Not one is out of range in z.** Their y or x is **254 or 255** -- the last
one or two pixels of the frame -- while their z is ordinary (1, 5, 12, 17,
35, 51, 59, 60...). `round(254 / 4) = round(255 / 4) = 64`, and the pooled
axis has indices 0..63. So it is exactly the off-by-one: **ordinary cells
against the border of the frame, read one index past the end by the
diagnostic**.

**Blast radius: none.** `h_gt` is a diagnostic column only; the pipeline
never indexes the heatmap at GT positions, so no score, bench number or
submission is affected. There is also no coverage hole -- 64 pooled columns
cover all 256 rows exactly.

**The process failure, recorded because it is the point.** I invented a
mechanism ("a lineage drifting out of frame over t=64..94"), wrote it into
this log as a finding, and defended it through two rounds of questions,
while the refutation sat in my own paragraph: 30 of the 48 were MATCHED BY
THE DETECTOR. The measurement that settled it cost under a minute and could
have been run the moment the NaNs appeared.

**v3 (2026-09-08, 2.8 min): MP4, and a Z SWEEP.** The user, on v1: "这个渲染
出来我也完全分辨不出来细胞在哪里呢，可能在下面被z轴叠掉了？" -- correct, and
the same slab bug described above. Then "可以渲染成mp4？因为单帧的话可能隐掉了".
v3 writes `.mp4` via libx264 (with a GIF fallback that PRINTS, so a silent
downgrade cannot hand back the rejected format), takes contrast from the CROP
instead of the whole frame -- the frame percentiles are set by the bright core
of the embryo and these cells are by definition the dim ones -- and adds
`<name>_zsweep<k>.mp4`: the crop at the miss frame, ONE z plane at a time from
the top of the stack to the bottom, 64 planes. That clip exists because of
`shoulderfeat` (entry above): the cells sit at a z EDGE in 17 of 18 movies, and
a sweep is the only view that shows whether a cell is CUT OFF by the end of the
stack -- which is the open question there, since "physically half outside the
volume" and "the UNet degrades at its own z border" produce identical tables
and only the second is fixable. On `44b6_a2bb48bb` the first sweep puts the
missed cell at z=52 of 63.

**v5 (2026-09-08): v4's fade was the drop-out bug a FOURTH time.** v4 faded
to alpha 0.16 and reached that floor 8 planes out, so on a 64-plane sweep
every distant marker was invisible over bright tissue -- markers hidden with
opacity instead of an `if`. Measured on the output: blue circles reached only
19 of 64 planes on `44b6_a2bb48bb` and 34 of 64 on `6bba_3a1849c2`, the red
miss marker only 9. v5 floors alpha at 0.50, fades over 40 voxels instead of
10, carries the depth cue mainly in RADIUS (down to 30% at distance) and dash,
and draws a dark backing ring so a thin cyan line survives a white nucleus.
**64 of 64 planes now carry every marker, blue and red.** The lesson is in
`working-agreement.md` under "Anything you omit will be read as a negative" --
and note that the fix for it was itself the same bug twice more, which is why
the rule says PREFER ENCODING TO DROPPING rather than "filter carefully".

**v4 (2026-09-08): STOP DROPPING MARKERS -- the same bug a third time.** The
user, on v3: "我是说你那个视频上的蓝圈… 这个标注是隔几帧有一个？". v3 drew a
circle only while the cell was within one voxel of the displayed plane, so on
a 64-plane sweep each cell was visible for about three planes and the circles
blinked: 11 of 64 planes carried any blue at all on `44b6_a2bb48bb`, 17 of 64
on `6bba_3a1849c2`. That is v1's bug in a different coat -- **a marker that
vanishes cannot tell you whether the CELL left or the RENDERER did**, and both
times the viewer read it as "the data is sparse".

v4 draws EVERY marker and encodes distance from the displayed plane instead:
alpha `clip(1 - |dz|/10, 0.16, 1)`, line width in three steps, solid within
1.5 voxels and dashed beyond, marker size shrinking slightly with depth. The
red miss marker follows the same rule, so it can be tracked across the whole
sweep rather than appearing for three planes. Offered the alternative of
simply widening the tolerance to +-2-3 planes; the user chose the encoding:
"第二个吧，连贯一点其实我也能看出来细胞下去了又或者暗掉了".

`tools/test_shoulderviz_depth.py` renders one synthetic frame through the
kernel's own drawing code and asserts the thing an eye cannot check quickly:
that all five markers reach the canvas (v3 would draw two), that alpha is
monotone in |dz| and floored above zero, and that matplotlib honours per-point
`linewidths` and RGBA `edgecolors` alongside a per-collection `linestyle` --
which is not obvious and would have failed silently into a uniform style.

**The other half of the answer is not a rendering choice at all**: GT annotates
a median 9.71% of 6bba's cells and 0.77% of 44b6's, so a 44b6 crop showing
fifty nuclei may legitimately carry one circle. In TIME the annotation is
dense -- every frame, in 23 of the 24 measured movies.

**Follow-up already pushed**: `shoulderfeat` (CPU, geffs only, no
images) puts numbers on what the eyes suggested -- depth in z, time,
lineage run-length, position relative to divisions/track ends, and
nearest ANNOTATED neighbour (labelled weak: GT covers 9.71% of 6bba and
0.77% of 44b6, so it measures the annotation, not the tissue). The
run-length column is the one with an immediate use: 313 spread over 313
cells for one frame each is a `gap_close` problem; 313 concentrated on
40 cells for 8 frames each is beyond any one-frame bridge. Its test,
`tools/test_shoulderfeat.py`, plants a known lineage and a known depth
effect, because there is no anchor for this run and a walk that failed
to back up would report "all single frames, bridgeable" while the
in-kernel `sum(runs) == len(sh_ids)` assertion still passed.

### ⭐⭐⭐⭐ gtsplit (2026-09-08, biohub-gtsplit v1, both anchors exact): SPLITTING EVERY BLOB RECOVERS AT MOST 44% / 55% OF THE MISSES AND COSTS 3.5-5.6x THE DETECTIONS. At a detection count the pipeline could actually afford it recovers 30 and 24 cells -- and on 6bba it LOSES 36 it already had, for a net of -6. The line closes on measurement

The user, after being told the fork gate is a model and this is not:
"算了你先试试全劈能捞回多少吧". Price the ceiling before anyone designs a
splitter. Shape split -- threshold the heatmap, distance-transform the
mask with the real voxel size, seed on maxima of the DISTANCE map,
watershed -- because gtpeak showed these cells have no intensity
maximum, so nothing seeded on intensity can separate them.

| arm | mask | NMS | seed | 6bba dets/fr | recovered of 235 | 44b6 dets/fr | recovered of 104 |
|---|---|---|---|---|---|---|---|
| BASE | -- | 4.0 | -- | **137.9** | -- | **310.1** | -- |
| s05 | 0.05 | -- | 1 | 485.0 (**3.5x**) | **103 (43.8%)** | 971.2 (**3.1x**) | 53 (51.0%) |
| s15 | 0.15 | -- | 1 | 539.6 (**3.9x**) | 91 (38.7%) | 1,721.3 (**5.6x**) | **57 (54.8%)** |
| s15d | 0.15 | -- | 2 | 178.3 (1.29x) | 43 (18.3%) | 464.3 (1.50x) | 37 (35.6%) |
| s15n | 0.15 | 4.0 | 2 | **142.8 (1.04x)** | 30 (12.8%) | **321.4 (1.04x)** | 24 (23.1%) |

**(1) The ceiling is not 100%, it is 44% / 55%.** Even splitting
everything with the most permissive seeding leaves more than half the
misses unrecovered. The local test predicted this: on smooth merged
Gaussians a shape split only separates the dim lobe in a narrow band of
separation and amplitude ratio, and 4 of 20 synthetic configurations
fell in it. The real data agrees.

**(2) The bucket breakdown says the split does exactly what it should,
and nothing more.** The `cold` bucket is never recovered (0/4, 0/1) --
correct, there is no signal there. The `1:1 gave it away` bucket is
recovered COMPLETELY on 44b6 (11/11), which is the cleanest possible
confirmation of the merge mechanism: two annotated cells sharing one
detection, separated by a shape cut. `nms-killed` comes back whenever
NMS is off, trivially.

**(3) NMS re-merges a third of what the split separates**, as
predicted before the push: s15d -> s15n costs 43 -> 30 (6bba) and
37 -> 24 (44b6) at the same seeding. **No splitter can ship without
changing `NMS_UM` too.**

**(4) THE NUMBER THAT CLOSES IT.** At `s15n` -- the only arm whose
detection count the pipeline could absorb, +3.5% -- 6bba's recall goes
0.9780 -> **0.9775**. It recovered 30 of BASE's misses and **lost 36
cells BASE already had**, for a net of **-6**. 44b6 nets +17
(recovers 24, loses 7). **6bba is 85% of the score weight, so the
affordable operating point is a wash at best.** Splitting is not
additive: cutting blobs also cuts blobs that were fine.

**(5) And the metric makes the expensive arms worse than they look.**
`reference.md`: an edge counts toward the denominator only if at least
one endpoint matched a GT node, and "a fork lying entirely among
unannotated cells enters none of the FP sets". So an edge between two
unannotated detections is invisible to `adj_edge_jaccard`, while
`N_pred` counts every node. **Recovering an unannotated cell is pure
cost.** At 44b6's s15 that trade is +1,411 detections per frame for 57
recovered annotated cells over 964 frames -- about **24,000 extra
detections per annotated cell recovered.**

⚠ `dets/fr` is the DETECTION stage, not `N_pred`. The ILP candidate
floor and `_short_filter` discard most surplus downstream, so a 3.5x
detection increase is not a 3.5x node increase and this run cannot say
what it would be. That translation needs a pipeline run, and (4) says
not to spend one.

**WHY NO GATE FOR SHOULDER POINTS IS POSSIBLE, and it is structural
rather than a missing idea** (user, 2026-09-08). Two independent walls:

1. **A shoulder is DEFINED as "not a local maximum".** So any purely
   LOCAL criterion -- value, shape, gradient, curvature -- that fires on
   "a shoulder with a cell under it" must also fire on "the side of a
   cell", because at local scale those are the same object. No feature
   set separates them.
2. **Neither class is labelable, and the negative side is the worse
   half.** Positives are visible only for the annotated fraction (6bba
   9.71%, 44b6 0.77%), which gtpeak counted as 313. Negatives can never
   be confirmed at all: "no annotation here" does not mean "no cell
   here". **A classifier needs a negative class and this problem cannot
   produce one.** This is exactly what the fork gate had and this does
   not -- every fork gets a TP/FP verdict from the official scorer.

So the only admissible evidence that a shoulder is a cell is EXTERNAL
and topological: the graph asserting that a cell must exist at that
place and time. `_gap_close` is precisely that assertion -- a track ends
at t and another starts at t+2, so something was at t+1 -- and it needs
neither annotation nor a classifier to make it. The question then stops
being "is there a cell here" and becomes "given that we are placing a
node, where", **and that is where the heatmap is entitled to speak.**

⚠ **CORRECTED within the hour, on the user's objection. An earlier
version of this paragraph called that route "largely already answered
null" by `dcpool` and used its result as a prior on reading the
heatmap for placement. That transfer is invalid for two independent
reasons, and making it was the exact failure the 2026-09-08 rule two
sections up was written against -- applying a run's conclusion past
what it could measure.**

1. **dcpool's own stated mechanism is about the MATCH RADIUS, not about
   position quality.** Its words: a bridge insert's position "affects
   the metric only through whether the inserted node matches GT within
   7.0 um", and a one-frame midpoint errs by about half a displacement
   (p50 1.82 um), so it "already lands inside the match radius
   essentially always". **That reasoning is correct and it is specific
   to BRIDGE nodes, which `_gap_close` inserts AFTER the solve.** Their
   position feeds nothing but the match. An ordinary detection's
   position is decided BEFORE the solve and feeds the ranker's distance
   and velocity features, i.e. which edges the ILP creates at all.
   Different channel; dcpool never touched it.
2. **DeepCenter ep05 is a deliberately NON-CONVERGED checkpoint**
   (`dccache3`: epoch 5 beats 7 beats 12, monotonically, in all three
   domains). Its value is the diversity of not having converged, not
   positional precision, so "its peaks did not improve placement" is not
   evidence that placement cannot be improved. **And its peaks are still
   PEAKS**: a shoulder in F4 is a shoulder in DeepCenter too, because
   merged-blob geometry is a property of the IMAGE, not of the model.
   dcpool's 20% snap rate is consistent with DeepCenter's peaks sitting
   where F4's already do -- it could never have reached a shoulder
   point.

**So dcpool carries no information about reading the heatmap for
placement.** What remains true from it is narrow and worth keeping: at
span 1, post-link bridge position is metric-equivalent to the midpoint
because 7 um swallows the error.

**The lever that needs no external evidence at all is `_refine`**: it
moves nodes that ALREADY EXIST, so the existence question never arises,
no labels are needed, and it touches 100% of cells. `refinecmp` is that
run.

Tested claim: that splitting merged blobs recovers the 313 cells gtpeak
found responding without a peak. **Closed: it recovers at most 44% /
55% of the misses at 3.5-5.6x the detections, and at an affordable
detection count it is net NEGATIVE on 6bba.** Residue, and it is
narrow but real: **44b6's 11/11 recovery of the "two annotated cells,
one detection" bucket.** That sub-case is provable without a classifier
(two GT cells inside one detection's radius), it is where 44b6's
density hurts, and it is 11 cells rather than 313. If anything from
this line is worth a second look it is that, targeted, not a global
split.

**The other lever gtpeak found is untouched by all of this**: fixing
`_refine` adds zero detections, so none of (4) or (5) applies to it,
and it moves 100% of cells rather than 2.3%.

### ⭐⭐⭐⭐⭐ gtpeak (2026-09-08, biohub-gtpeak v1, 29.7 min, both anchors exact): ON COMPETITION DATA THE UNDETECTABLE CLASS IS 5 CELLS OUT OF 13,646. 95% / 86% of F4's misses have a heatmap response with NO LOCAL MAXIMUM -- their nearest actual peak is ~6 um away, on another cell. And the detections that ARE found jitter 1.3-1.5 um between consecutive frames, which is HALF the real cell displacement

The user asked three things after ruling the official detector
inadmissible: is there a peak at all, is there still one after NMS, and
measure localisation because "off a bit here, off a bit there" could
bend the links. Every annotated GT point in the 12 6bba holdout + 12
excluded 44b6, **at every frame** (detfuse3 used every 10th, which is
why its competition columns had n = 8 and n = 4 and "FAILED to
inform"). Anchors exact on the `t % 10 == 0` subset: **6bba 1,047 GT /
recall 0.9761, 44b6 297 / 0.9596.** Full set: 10,699 and 2,947 points,
recall 0.9780 and 0.9647.

### 1 + 2. Where the misses go

| bucket | 6bba (235 misses) | 44b6 (104) | the fix it implies |
|---|---|---|---|
| no peak, and the map is **COLD** (h_win < 0.05) | **4 (1.7%)** | **1 (1.0%)** | a better detector |
| no peak, but the map **RESPONDS** | **224 (95.3%)** | **89 (85.6%)** | **peak extraction** |
| had a raw peak, **NMS removed it** | 6 (2.6%) | 3 (2.9%) | `NMS_UM` 4.0 |
| a detection survived, the 1:1 gave it away | 1 (0.4%) | **11 (10.6%)** | one detection, two cells |

**Five cells in 13,646 are genuinely invisible.** The class the
2026-08-25 entry described as "the detector produces no response for
them at all" is, on competition data, 0.04% of annotated GT.

What the heatmap actually reads at a missed cell: **h_win p50 = 0.15
and h at the GT voxel p50 = 0.10 / 0.11** in both batches -- above the
0.05 peak-finding floor, at or below the 0.15 gate. 98.3% / 99.0% are
>= 0.05 within 3.25 um, against detfuse3's 99.6% on Zebrahub, so the
Zebrahub finding transfers.

**The mechanism is now identified, not hypothesised.** The nearest RAW
`peak_local_max` output at ANY score is **p50 5.84 um (6bba) / 6.91 um
(44b6)** from a missed cell. So there is a response of 0.10-0.15 at the
cell, and the nearest actual local maximum is six or seven microns
away -- on a different cell. The response is a **shoulder absorbed into
a neighbour's blob**, which is exactly the failure a floor sweep cannot
reach: `peak_local_max` at threshold 0.05 would have emitted these if
they were maxima, and they are not. It is the same mechanism div-mass
saw as 44.7% of divisions with both daughters in one watershed label,
and 44b6's 10.6% "one detection, two cells" bucket -- 25x 6bba's -- is
the density signature of it.

**NMS is nearly innocent: 2.6% and 2.9%.** `NMS_UM` 4.0 has never been
swept and this says not to bother.

### 3. Localisation, and it is bigger than expected

Over the 10,464 / 2,843 cells F4 DOES find:

| | 6bba | 44b6 |
|---|---|---|
| \|offset\| p50 / p90 / p99 um | 1.41 / 3.47 / 6.67 | 1.71 / 5.91 / 6.90 |
| mean SIGNED offset (z, y, x) um | −0.384, −0.018, +0.056 | −0.142, +0.210, +0.145 |
| std per axis um | 1.533, 1.062, 1.074 | 2.205, 1.497, 1.517 |

**Bias is negligible; jitter is everything.** The only systematic term
is 6bba's −0.384 um in z against a std of 1.533. A bias mostly cancels
inside a link; jitter does not.

**The table the question was really about** -- the CHANGE in offset
between a cell and its own next frame, along 10,009 / 2,729 GT track
steps, which is motion no cell made:

| | 6bba | 44b6 |
|---|---|---|
| \|offset(t+1) − offset(t)\| p50 / p90 / p99 um | **1.29** / 2.41 / 8.16 | **1.51** / 5.76 / 10.49 |
| steps above 1 um | 60.0% | 66.9% |
| steps above 2 um | 17.1% | 26.8% |
| real link length `d_pc1` p50 / p90 um (fork table) | 2.72 / 5.05 | 2.10 / 3.81 |
| **invented motion as a share of real, at p50** | **47%** | **72%** |

**At the median, the detector's own frame-to-frame jitter is half of
6bba's real cell displacement and nearly three quarters of 44b6's. At
p90 in 44b6 it EXCEEDS it** (5.76 against 3.81). The ranker's distance
and velocity features see that jitter as motion.

**Where it comes from, and why that is good news.** The pooled grid is
1.625 um isotropic. Independent quantisation at t and t+1 gives a
difference with per-axis std 1.625/sqrt(12)·sqrt(2) = 0.66 um, about
1.15 um in magnitude -- **against 6bba's observed 1.29**. So most of
6bba's injected motion is the GRID, and `_refine` (centre of mass of the
raw volume over +-2 z, +-5 yx) exists to remove exactly that and barely
does: p50 1.65 -> 1.39 um, while p75 goes 1.93 -> 2.00 and 44b6's p90
goes 4.19 -> 4.25. **`_refine` helps the median slightly and hurts the
tail, and has never been validated until now.** 44b6's 1.51 p50 and
5.76 p90 exceed quantisation, so a second term is present there --
consistent with the merged-blob mechanism above.

Tested claim: that F4's missed cells are cells the detector cannot see.
**Refuted on competition data, with n = 13,646 rather than 12.** The
detector responds at 95%+ of them; what fails is turning a response
into a local maximum when a brighter neighbour owns the blob. Two
levers follow, neither of which is retraining anything:

1. **Peak extraction** -- splitting merged blobs (watershed on the
   heatmap, or a lower-threshold second pass constrained to non-maximal
   ridges). Target: 224 + 89 = 313 cells, 2.3% of annotated GT.
2. **Sub-voxel localisation** -- a better `_refine`. Target: about half
   of 6bba's frame-to-frame jitter, which is 47% of real displacement.

⚠ Caveats, all of them the standing ones. **Annotated cells only**
(6bba labels 9.71% of its cells, 44b6 0.77%), so this is recall against
a hand-picked sample. `d_pc1` is measured at FORK sites and is itself a
DETECTED distance, so it already contains the jitter -- the true
displacement is smaller and the 47% / 72% are therefore
UNDERSTATEMENTS. And the offset is to whichever detection the greedy
1:1 assigned; at 7 um that could occasionally be a neighbour, though a
p50 of 1.41 um says most are the right cell.

### ⭐⭐⭐⭐ divbench9 (2026-09-08, biohub-divbench9 v1, all THREE anchors exact): THE JOINT GATE ABSORBS DIV_W. Given the better gate, DIV_W 0.35 buys +0.0002, 90% [-0.0032, +0.0036] -- and divbench6's own +0.0015 does not survive its first paired bootstrap either. The argmax rule fires a second time, one day later, on the constant this project had just finished reopening

Two solves, four cells, three of them already measured -- so the run
carried three independent anchors and produced one new number.

| | gate6 (6bba only) | joint (+ 44b6's 67) |
|---|---|---|
| **REF (0.55)** | 42/47/83, 0.2442, **0.9387** *(anchor, divbench3)* | 46/39/79, 0.2805, **0.9424** *(anchor, divbench8)* |
| **dw35 (0.35)** | 50/68/75, 0.2591, **0.9402** *(anchor, divbench6)* | 50/52/75, 0.2825, **0.9426** |

All three anchors reproduce to four decimals, adj_J is 0.9143 in every
cell, so the table is readable and the whole difference is the division
term.

**Additivity, the question the run was for:**

    DIV_W alone  +0.0015     joint alone  +0.0037
    sum if independent  +0.0052     larger alone  +0.0037
    MEASURED TOGETHER   +0.0039   =  75% of the independent sum

75% sounds like most of it adds. **It is not, and the bootstrap says
why** -- paired over the 128 movies, 2,000 draws, same resample for
every cell:

| cell | mean d vs REF+gate6 | 90% | distinguishable |
|---|---|---|---|
| REF + joint | +0.0038 | [+0.0016, +0.0061] | **YES** |
| dw35 + gate6 | +0.0016 | [−0.0022, +0.0054] | **no** |
| dw35 + joint | +0.0039 | [+0.0009, +0.0069] | **YES** |
| **dw35+joint vs REF+joint** | **+0.0002** | **[−0.0032, +0.0036]** | **no** |

**Two things fall out, and the second is the important one.**

**(1) divbench6's +0.0015 does not survive its first interval.**
[−0.0022, +0.0054] straddles zero. That entry flagged the risk in its
own words -- "the effect size is at the resolution floor and has no
interval around it", because divbench6 was cut from divbench4 before
the bootstrap machinery existed -- and the flag was right. Passing the
both-folds rule is necessary, not sufficient; 128 movies with 125 GT
divisions cannot resolve +0.0015.

**(2) Given the joint gate, DIV_W 0.35 is worth +0.0002.** The direct
comparison, which needs no arithmetic about sums: dw35+joint against
REF+joint, [−0.0032, +0.0036]. Nothing. What DIV_W was doing -- pushing
fork production toward the better-calibrated source so a gate could
keep more true forks -- **the better gate already does by itself.**
dw35 still finds 4 more true divisions under the joint gate (50 against
46) and pays 13 more false ones (52 against 39), and those cancel.

**This is the 2026-09-08 argmax rule firing a second time, one day
after it was written, on the constant it was written about.**
`ranker-fine` chose 0.55 against a pipeline with no gate; divbench6
reopened it and chose 0.35 against the 6bba-only gate; divbench9 swaps
in a better gate and 0.35 is worth nothing again. The rule said a
constant is provisional whenever a stage changes the cost of the
failure it was chosen against. **Both re-openings were correct to run
and neither survives the next stage change** -- which is itself the
strongest evidence for the rule, and a warning that `DIV_W` should not
be re-swept every time the gate improves.

Tested claim: that `DIV_W` 0.35 and the joint gate fix different forks
and their gains add. **Refuted: they fix the same ones.** Residue:
`DIV_W` 0.35 does recover 4 true divisions the joint gate misses, and
the joint gate rejects 16 false forks 0.35 produces. Something that
kept the 4 without the 16 would be worth +0.001 or so -- below what
this bench can resolve, so it is not a measurement this project can
make on 125 GT divisions.

**What ships is unchanged: the joint gate alone, DIV_W stays 0.55.**
`submit_joint` (ref 56086066) is already the right submission and this
run says not to follow it with a DIV_W one.

### ⭐⭐⭐⭐ divbench6 (2026-09-08, biohub-divbench6 v1, 240.5 min, 128/128, anchor exact): SOURCE QUALITY GOVERNS FORK PRODUCTION, NOT FORK COUNT -- the prediction divbench7 forced this run to make, confirmed from the opposite direction. 0.55 was the argmax of a curve measured on 12 GT divisions with NO GATE, and with the 6bba-only gate 0.35 reads +0.0015 in both folds -- but see divbench9 immediately above: that +0.0015 does NOT survive a paired bootstrap ([-0.0022, +0.0054]) and the joint gate absorbs it entirely. The MECHANISM finding stands; the constant change does not ship

`DIV_W` is read by `solve_one`, so every arm is its own full ILP solve
over all 128 movies -- 4 solves, 240 minutes. REF (0.55) reproduces
50/249/75, div_J 0.1337, adj_J 0.9133, score 0.9266 exactly.

**(1) The mechanism check first, per the pre-registration: did lowering
`DIV_W` actually move fork production from the geometric patch to the
probabilistic solver?** It did, and by more than double:

| arm | forks | **ILP forks** | patch forks | div tp/fp/fn | div_J | adj_J | score |
|---|---|---|---|---|---|---|---|
| REF (0.55) | 7,004 | **2,490** | 4,514 | 50 / 249 / 75 | 0.1337 | 0.9133 | 0.9266 |
| dw45 | 9,955 | **5,692** | 4,263 | 61 / 369 / 64 | 0.1235 | 0.9130 | 0.9254 |
| dw35 | 10,027 | **5,762** | 4,265 | 62 / 371 / 63 | 0.1250 | 0.9130 | 0.9255 |
| dw20 | 10,169 | **5,903** | 4,266 | 62 / 375 / 63 | 0.1240 | 0.9129 | 0.9253 |

Ungated, this is the August shape reproduced exactly -- more forks,
div_J falls because the false ones outnumber the true, every arm loses
about -0.0012 of score. **That is the comparison that closed this line
in August, and it is not the decision.** Note what it hides: the arms
find **12 more true divisions** (50 -> 62).

**(2) The decision -- after the fork gate, each arm gated by a gate
trained out of fold on ITS OWN forks:**

| arm | div tp/fp/fn | div_J | adj_J | score | vs REF+gate | fold0 | fold1 | that gate's AUC |
|---|---|---|---|---|---|---|---|---|
| REF + gate | 42 / 47 / 83 | 0.2442 | 0.9143 | 0.9387 | -- | -- | -- | 0.9283 / 0.8326 |
| dw45 + gate | **52** / 71 / 73 | **0.2653** | 0.9143 | **0.9408** | **+0.0021** | +0.0060 | **-0.0002** | 0.9176 / 0.8656 |
| **dw35 + gate** | 50 / 68 / 75 | 0.2591 | 0.9143 | 0.9402 | **+0.0015** | **+0.0029** | **+0.0012** | 0.9166 / **0.8862** |
| **dw20 + gate** | 50 / 71 / 75 | 0.2551 | 0.9142 | 0.9397 | **+0.0010** | **+0.0029** | **+0.0004** | 0.9198 / 0.8792 |

The pre-registered rule was: a candidate iff it beats REF+gate in BOTH
folds. **dw35 and dw20 pass. dw45, which has the largest gain and the
most true divisions of any arm, FAILS on -0.0002 in fold 1** -- and the
rule was fixed in advance precisely so that the biggest number does not
get to argue its own case afterwards.

**(3) The prediction divbench7 forced this run to make, and its
answer.** divbench7 raised the PATCH's budget -- production on the
source whose counted forks are LESS precise (0.150 against the ILP's
0.196, re-confirmed on REF here) -- and the per-arm gate's AUC averaged
over folds FELL, 0.8805 -> 0.8656. The pre-registration said: if source
quality governs, moving production the other way should hold or raise
it; if it falls here too, fork COUNT is what governs and both levers
close together.

**It rises: 0.8805 -> 0.9014 at dw35.** And the movement is almost all
in the weaker fold, 0.8326 -> **0.8862**, while fold 0 gives up only
0.0117. So the two benches agree from opposite directions:

| run | production moved to | forks added | mean out-of-fold AUC | gated result |
|---|---|---|---|---|
| divbench7 | the patch (precision 0.150) | +1,862 | 0.8805 -> **0.8656** | -0.0005, folds disagree |
| divbench6 | the solver (precision 0.196) | +3,023 | 0.8805 -> **0.9014** | **+0.0015, both folds up** |

divbench6 adds MORE forks than divbench7 did and comes out ahead, so
**the governing variable is not how many forks are made, it is which
stage makes them.** That is a mechanism, in a unit that is not the
score, and it now rests on two runs pointing opposite ways rather than
on one.

**(4) The honest size of it -- and divbench9 (below, same day) settled
it against this entry: the paired bootstrap reads [-0.0022, +0.0054],
so +0.0015 is NOT distinguishable from zero, and given the joint gate
DIV_W 0.35 is worth +0.0002. The concern this paragraph raised was the
right one. DIV_W 0.35 does not ship.** dw35's div_J gain is +0.0149,
which sits INSIDE the +-0.03 bar the bench resolves, and unlike
divbench8 this run carries no paired bootstrap -- it was cut from divbench4, before that
machinery existed. adj_J is unchanged (0.9143 against 0.9143), and both
folds are positive, but **the effect size is at the resolution floor
and has no interval around it.** It is a candidate by the rule it was
given; it is not the same quality of evidence as divbench8's
+0.0037 [+0.0016, +0.0060].

**Why August got it wrong, stated as the general lesson.** `ranker-fine`
measured `DIV_W` on 12 GT divisions with no gate in the pipeline and
picked the argmax, 0.55, while writing in its own entry that
"0.45-0.65 are all defensible". What made 0.20 lose there was
over-forking -- 1,024 forks, 57 counted false ones -- **which is the one
failure the fork gate exists to remove**. The verdict was correct for
the pipeline that measured it and wrong for the pipeline that inherited
it. This is the third time the same shape has appeared here (gap2's
span behind a binding budget, divgate2's condition walk behind the
patch cap): **a constant chosen as the argmax of a curve is only valid
while every stage that shaped the curve is unchanged.**

Tested claim: that `DIV_W = 0.55` is still the right price for a fork
now that a gate cleans up behind the solver. **Refuted: 0.35 beats it
by +0.0015 in both folds, and the reason is that the extra forks come
from the better-calibrated source.** Residue, and the next run rather
than the next submission: `DIV_W` 0.35 and the joint gate act on the
SAME object -- the fork population -- so their gains may add
(+0.0015 + 0.0037) or may almost entirely overlap, and nothing here
says which. That cross is `divbench9`, and it needs divbench8's
bootstrap machinery carried over so the answer comes with an interval.

### ⭐⭐⭐ divbench7 (2026-09-08, biohub-divbench7 v1, 122.6 min, anchors exact): the patch's per-frame cap IS binding -- 27% of eligible proposals -- and lifting it does NOT pay, before or after the gate. The reason it differs from gap4 is that this budget is handed out in QUALITY order, not frame order

divbench4 found the cap refusing 1,672 of the 6,194 proposals that pass
every geometry gate. This lifts it. `DIV_FRAME_CAP` / `DIV_GLOBAL_CAP`
are read inside `_add_safe_divisions`, which runs after the solve, so
one solve per movie and one patch per arm. Anchors exact: D187
F4-6bba(12) 0.9216 / div_J 0.0606, +BIDIR 0.9261, REF 50/249/75,
div_J 0.1337, adj_J 0.9133, score 0.9266.

**The constraint really was lifted** -- the first thing to check, because
an arm that does not move the counter has not tested anything:

| arm | frame cap | forks added | refused by the frame cap | div tp/fp/fn | div_J | adj_J | score |
|---|---|---|---|---|---|---|---|
| REF | 0.0076 | 4,514 | **1,672** | 50 / 249 / 75 | 0.1337 | 0.9133 | 0.9266 |
| cap3x | 0.0228 | 6,072 | 323 | **54** / 283 / 71 | 0.1324 | 0.9129 | 0.9261 |
| cap8x | 0.0608 | 6,374 | 2 | 54 / 289 / 71 | 0.1304 | 0.9128 | 0.9258 |
| nocap | 1.0 | 6,376 | **0** | 54 / 289 / 71 | 0.1304 | 0.9128 | 0.9258 |

and after the fork gate at p > 0.03, each arm gated by a gate trained
out of fold on its OWN forks -- which is the decision:

| arm | div tp/fp/fn | div_J | adj_J | score | vs REF+gate | fold0 | fold1 | the gate's own AUC |
|---|---|---|---|---|---|---|---|---|
| **REF + gate** | 42 / 47 / 83 | 0.2442 | 0.9143 | **0.9387** | -- | -- | -- | 0.9283 / 0.8326 |
| cap3x + gate | 44 / 56 / 81 | 0.2431 | 0.9141 | 0.9384 | −0.0002 | −0.0021 | +0.0021 | 0.8936 / 0.8409 |
| cap8x + gate | 43 / 54 / 82 | 0.2402 | 0.9142 | 0.9382 | −0.0005 | −0.0038 | +0.0032 | 0.8841 / 0.8471 |
| nocap + gate | 43 / 54 / 82 | 0.2402 | 0.9142 | 0.9382 | −0.0005 | −0.0038 | +0.0032 | 0.8841 / 0.8471 |

**Lifting the cap recovers 4 true divisions (50 -> 54) and drags in 40
counted false forks; after the gate that is 2 more true and 9 more
false, the score is a wash, and the two folds disagree in sign in every
arm.** No candidate.

**Why this is NOT gap4, stated as a mechanism rather than a
disappointment.** On 2026-08-27 the gap-repair budget bound in 24 of 32
datasets and raising it gained +0.002 at LB, and the entry's own
explanation was that the gates being swept "were not selecting quality;
they were rationing scarcity more sensibly than frame order did" --
`_gap_close` handed its allowance out **in frame order**. This budget
does not: `_add_safe_divisions` sorts every proposal by
`d_pq + DIV_W_SISTER * d_s` and walks that list, so the per-frame
counter keeps the geometrically BEST proposals in each frame and refuses
the tail. It is already a quality-ordered budget. What the cap was
refusing is therefore the worst 27%, and the arms confirm it: the extra
forks are hard ones, and **the gate itself degrades when they arrive**
(out-of-fold AUC 0.9283 / 0.8326 falls to 0.8841 / 0.8471).

**The rule this refines**: "before sweeping a knob, ask what limits the
thing the knob feeds" still stands and was worth the 122 minutes -- a
binding budget had never been counted here. But a binding budget is not
automatically a lost gain. **Ask HOW the budget is rationed: in arrival
order, raising it is likely to pay; in quality order, it is likely not
to.** Two measurements now, one on each side.

Tested claim: that the division patch's per-frame cap is rationing away
recoverable divisions. **Closed: it binds on 27% of proposals, lifting
it recovers 4 of the 75 lost divisions, and the cost exceeds the gain
both raw and gated.** Residue: those 4 divisions ARE recoverable and
something better than "first 1 per frame by geometry" might keep them
without the other 1,862 forks -- but that is a selection problem on the
same population the fork gate already scores, and the gate got WORSE
when it was handed them, which is evidence against.

### ⭐⭐⭐⭐ divbench4 (2026-09-07, biohub-divbench4 v2, 214.8 min, all anchors exact): the 5a steal is CLOSED -- the solver's choice is right and the mother's own edge is 16x weaker; the eight new fork features do NOT clear the pre-registered bar; and the run found a constraint nobody had ever counted -- the patch's PER-FRAME CAP refuses 27% of the division proposals that pass every gate

Anchors, all exact: D187 F4-6bba(12) adj_J 0.9216 / div_J 0.0606 HARD;
+BIDIR at floor 0.48 on the same 12 = 0.9261; REF over the 128 =
50 / 249 / 75, div_J 0.1337, adj_J 0.9133, score 0.9266; the 20-feature
2-fold gate = AUC 0.9283 / 0.8326 and 42 / 47 / 83 at p > 0.03. The
harness is divbench3's.

**(1) THE 5a STEAL IS CLOSED, and the reason is not the one divsteal's
Zebrahub numbers suggested.** All 23 cases on the competition data, on
production primaries:

| | value |
|---|---|
| cases | 23 |
| competing edge annotated at BOTH ends | 6 |
| ... of those, a TRUE GT edge | **6 of 6** (Zebrahub: 8 of 27) |
| both ranker probabilities available | 13 of 23 |
| p_mother median (p25 / p75) | **0.054** (0.016 / 0.660) |
| p_other median (p25 / p75) | **0.872** (0.737 / 0.941) |
| p_mother beats p_other at all | **1 of 13** |
| d_mother median | **9.39 um** |
| d_other median | **1.95 um** |
| competitor already forks (the steal refuses) | 2 |

**Every competing edge that could be checked is a real edge.** So a
steal does not correct a mistake, it deletes a correct link. And the
geometry says why: the daughter the ground truth assigns to this mother
is 9.39 um away from her, while the node that took her sits 1.95 um
away. (I first guessed the competitor might be a duplicate detection of
the mother herself; d_other 1.95 um against d_mother 9.39 um says it is
not, and the guess is recorded only because checking it was one column.)

The arms, official scorer, all 128 movies:

| arm | steals made | div tp/fp/fn | div_J | adj_J | score | d score |
|---|---|---|---|---|---|---|
| REF | 0 | 50 / 249 / 75 | 0.1337 | 0.9133 | 0.9266 | -- |
| **steal_all** (unconditional) | **2,770** | 51 / 403 / 74 | 0.0966 | 0.9107 | 0.9204 | **−0.0063** |
| steal_m0 (mother's p wins) | **3** | 50 / 249 / 75 | 0.1337 | 0.9133 | 0.9266 | 0.0000 |
| steal_m20 (wins by 0.20) | **0** | 50 / 249 / 75 | 0.1337 | 0.9133 | 0.9266 | 0.0000 |

and after the gate at p > 0.03, which is the shipped composition:
REF + gate 42/47/83, div_J 0.2442, score 0.9387 (**divbench3 exactly**);
steal_all + gate 42/71/83, 0.2143, 0.9346, **−0.0041 against REF+gate,
both folds negative**; the two margin arms are REF to the digit.

Reading. The margin rule fires on 1 case in 13 and buys nothing. The
unconditional arm takes 2,770 daughters, gains ONE true fork and 154
counted false ones, and loses 0.0063 of score -- and it is a control,
not a candidate, so the right reading of a near-zero would have been to
look for the cancellation. There is no cancellation here: it is
one-sided loss.

**Tested claim: that a patch-side rule can take back the second
daughter the ILP has already placed. Closed -- the solver is right, and
the residue is elsewhere.** What the direction still owns: 23 divisions
are still lost at 5a, and the walk now says the cause is neither the
patch nor the solver but the RANKER -- the true mother-daughter edge
scores 0.054 while a wrong one scores 0.872, over a 9.39 um
displacement that sits at the edge of `GATE_UM` 10.0. Ten of the 23
have no ranker probability for the mother's edge at all. That is the
same 10-vs-12 um blind spot gatelab found on the fork gate, showing up
on the other side of the problem. Caveat on the whole walk: it matches
graph nodes to GT at `MATCH_UM` 5.0 while the official scorer matches
at 7.0 -- inherited from divgate2 so the divbench2/3/4 comparison is
clean, but the bucket assignment is stricter than the metric.

**(2) THE EIGHT NEW FORK FEATURES DO NOT CLEAR THE BAR** (pre-registered:
nothing under +0.03 div_J counts, whatever its sign). Single-feature
AUC on the 299 counted forks, new ones against the incumbents:

| feature | AUC | |
|---|---|---|
| sc_q (incumbent, raw detection score of the orphan) | 0.842 | |
| **dv2** (divergence two frames out) | **0.798** | the only new one with real signal |
| dv (incumbent, one frame) | 0.787 | |
| sc_p (incumbent) | 0.756 | |
| **p_q_marg** (mother's p minus the best other parent's) | 0.682 | on the 179 forks that have it |
| **sc_q_r** (sc_q as a per-movie quantile) | **0.630** | far WEAKER than the raw 0.842 |
| **sc_p_r** | 0.553 | against raw sc_p's 0.756 |
| **ang** (angle between the daughters at the mother) | 0.545 | nothing |
| **q_ends** | 0.529 | nothing |
| **dens5** | 0.498 | nothing |

Gates at p > 0.03, same forks, paired: 20 features / 2 folds
**0.2442** (the anchor); 28 / 2 folds 0.2367; 20 / 3 folds 0.2416;
28 / 3 folds 0.2457. Spread 0.009 against a bench that resolves ±0.03.
**Nothing here is distinguishable from the shipped gate**, exactly as
gatelab predicted offline, and the pre-registered rule says that is a
null result, not a small win. Both gates are written
(`fork_gate_f4v4_20.pkl`, `_28.pkl`) and `forks_f4_v4.csv` carries all
28 columns for offline work.

The one surprise worth keeping: **the per-movie RANK versions of the
detection scores are much weaker than the raw scores at home** (0.630
against 0.842). The reason to build them was transfer to 44b6, where
the raw scale moves; that is still open and divbench5 measures it. But
they cost a lot within 6bba, so a gate built on ranks would have to win
the transfer by more than it loses at home.

**gatelab6 (offline on `forks_f4_v4.csv`, same day) closes the feature
line properly**, because the kernel's four single rows cannot separate
"the ranking got worse" from "the probability scale moved". Paired
bootstrap over movies, 2,000 draws, same forks, cut p > 0.03:

| feature set | pooled AUC | div_J | paired d | 90% interval | distinguishable |
|---|---|---|---|---|---|
| 20 (SHIPPED) | 0.8798 | 0.2442 | -- | -- | -- |
| 20 + dv2 | 0.8892 | 0.2326 | −0.0118 | [−0.0372, +0.0132] | no |
| 20 + dv2 + p_q_marg | 0.8945 | 0.2241 | **−0.0199** | [−0.0419, −0.0007] | **YES, worse** |
| 28 (all) | 0.8945 | 0.2367 | −0.0077 | [−0.0308, +0.0143] | no |
| 20 with dv2 replacing dv | 0.8885 | 0.2179 | **−0.0264** | [−0.0536, −0.0001] | **YES, worse** |
| 6 strong + dv2 | 0.9018 | 0.2373 | −0.0067 | [−0.0308, +0.0165] | no |

and at each model's OWN best threshold, which is the fair comparison of
RANKINGS (upper bounds, chosen on the labels): shipped **0.2531**;
20 + dv2 0.2437; 20 + dv2 + p_q_marg 0.2400; 28 (all) 0.2384; dv2
replacing dv 0.2374; 6 strong + dv2 0.2541. **Every 28-feature variant
ranks WORSE, by 0.009 to 0.016, and it is not a calibration artefact.**

Why `dv2` fails despite the better single-feature AUC: it correlates
**0.899** with `dv` on the counted forks and is missing on 496 of them
(a daughter chain that ends or branches within two frames). It is the
same measurement one frame later, and it costs coverage.

⚠ **AUC went UP while div_J went DOWN** in four of the five variants --
0.8798 to 0.8945 while 0.2442 fell to 0.2241. That is this project's
own standing rule ("an offline AUC is not a score", 2026-08-24, where
edge AUC anti-predicted the pipeline three times running) firing again,
now inside the division stage. **Do not select a fork gate on AUC.**

Tested claim: that fork features the table lacks -- multi-frame
divergence, the angle between the daughters, per-movie score ranks, the
mother's margin over competing parents, density at a second radius,
whether the daughter's chain dies -- can improve the gate's ranking.
**Closed on 125 events: none does, and two combinations are measurably
worse.** Residue: `sc_q` (AUC 0.842 alone) still carries the gate, and
nothing built here explains WHY the 47 surviving false forks look like
true ones on every column (gatelab5). The features that would separate
them are not geometric or probabilistic -- everything of that kind has
now been tried -- so the next candidate class is IMAGE evidence at the
fork (intensity, shape, a nuclear-envelope signal), which needs a GPU
pass and has never been attempted for divisions.

**(3) ⭐ THE NEW FINDING: the division patch's PER-FRAME CAP is a
binding constraint, and nobody had ever counted it.** The budget table,
summed over 128 movies:

| | proposals |
|---|---|
| forks the patch ADDED | 4,514 |
| refused by the PER-FRAME cap (`DIV_FRAME_CAP` 0.0076 of the frame's nodes) | **1,672** |
| refused by the global cap (`DIV_GLOBAL_CAP` 0.00375 of the movie's nodes) | 8 |

**27% of the proposals that pass every geometry gate are refused because
their frame is full**, not because anything is wrong with them. divbench5
counts 861 more on 44b6's 71 movies, so it is not a 6bba artefact
(13.1 per movie against 12.1). This is the exact shape of the gap-closing
finding of 2026-08-27 -- `gap2` and `gap3` swept the bridge SPAN and four
GATES behind a budget that bound in 24 of 32 datasets, and raising the
budget beat every gate they had selected. Under that rule **every verdict
this project has issued about the division patch's geometry gates is
provisional on this cap**, including divgate2's condition walk and
divbench's tighter-geometry arms.

Two things make it worth spending a run on now rather than in August:
`DIV_FRAME_CAP` and `DIV_GLOBAL_CAP` are read inside `_add_safe_divisions`,
which runs in `patch_final` AFTER the solve, so they are POST-LINK
constants and a sweep costs one solve and N patches -- cheap. And
over-production is now safe in a way it was not: the fork gate removes
80% of forks while keeping 84% of the true ones, and it scored +0.010 at
LB. Raise the cap, let the gate filter. `divbench7` is that run.

### divbench4's pre-registration (kept for the record)

`divbench4.py` = divbench3 + three changes (make_divbench4.py):

1. **Eight new fork features**, all from the solved graph, chosen
   against gatelab's finding that the ranking is the whole remaining
   room: `dv2` (divergence two frames out -- `dv` is the second-best
   feature at one frame, so this asks whether the horizon is the
   limit), `ang` (the angle at the mother between the two daughters --
   a real division pushes them apart, a false fork picks up a
   neighbour travelling its own way; no current feature asks this),
   `sc_q_r` / `sc_p_r` (the two detection scores as PER-MOVIE
   quantiles, which is the transferable form of the strongest
   feature), `dens5`, `p_q_marg` (the mother's ranker p for the orphan
   edge minus the best OTHER parent's -- "is the mother even the best
   explanation for this node"), `q_ends`, `steal`. The 20-feature gate
   is still trained and written, so divbench3's rows are a hard anchor
   and the 28-vs-20 comparison is paired on the same forks.
2. **The 5a steal, measured**: every GT division that stops at 5a gets
   a row -- the ranker's p for the edge the patch WOULD make against
   the p for the edge the ILP DID make, both distances, the divergence,
   whether the competing edge is a true GT edge, and whether the
   competitor already forks. divsteal did this on Zebrahub (n=54) and
   on ONE case in each competition domain; this is 23 cases in the
   competition population on production primaries.
3. **The 5a steal, run**: a second pass in `_add_safe_divisions` that
   takes a daughter the ILP already placed (removes r->q, adds p->q)
   under the same geometry gates as the orphan pass. Three arms:
   `steal_all` (unconditional), `steal_m0` and `steal_m20` (the
   mother's ranker p must beat the current parent's by more than 0 /
   0.20). The orphan pass runs first and to completion, so REF is
   untouched by construction. Every arm is patched, written and scored
   by the official scorer, then pruned by the SAME out-of-fold
   20-feature gate at p > 0.03 -- production is patch -> gate and a
   stolen fork is a fork, so the composition is what has to be read.

Instrumented: the patch's own budget per arm (orphan added / global
cap hit / frame cap hit; steal proposals / added / no-probability /
margin-refused / caps). The rule is that a proposal refused by a CAP
is not a proposal refused on QUALITY, and nobody has ever counted this
patch's caps.

**Pre-registered reading.** (a) The new features are read from their
single-feature AUCs and from the 28-vs-20 gate rows at the same cut on
the same forks; gatelab's paired bootstrap says the bench's 90%
interval is about ±0.03 div_J, so **nothing under +0.03 div_J counts,
whatever its sign** -- a 28-feature gate that reads +0.01 is a null
result, not a small win. (b) A steal arm is a submission candidate iff
its `d score` against REF+gate at the same cut is positive in BOTH
folds. (c) The unconditional arm is a CONTROL, not a candidate: it
measures the sum of its good and bad cases, and if it comes back near
zero that is a reason to look for the cancellation, not to close the
line (2026-08-26). (d) If the 5a table shows the mother's edge almost
never beats the ILP's -- which is what divsteal found on Zebrahub,
p_mother 0.182 against p_other 0.765 -- then the margin rule fires on
nothing and the lever is the LINKER, not the patch. Anchors: D187
F4-6bba(12) adj_J 0.9216 / div_J 0.0606 HARD; +BIDIR at floor 0.48 =
0.9261 soft; REF over the 128 = 50/249/75, div_J 0.1337, adj_J 0.9133,
score 0.9266; the 20-feature 2-fold gate = AUC 0.9283 / 0.8326 and
42/47/83 at p > 0.03. Expected ~3 h CPU.

**v1 died at the first movie of stage B on `KeyError: 'orphan_frame'`,
which is the bug `tools/test_divbench4_steal.py` had already found
locally; v2 carries both fixes.** Two things came out of it for free:

- the traceback is proof that **the patch's PER-FRAME cap binds on the
  very first movie** -- `orphan_frame` is only touched when a proposal
  is refused because frame `t` already holds
  `max(1, int(0.0076 * n_frame[t]))` forks. Nobody had ever counted
  this stage's caps; v2's per-arm budget table now does. Under the
  2026-08-27 rule, every verdict about the patch's GATES is provisional
  on that cap until the table is read.
- the cost of the two bugs was the 35-minute features stage, once. The
  silent one (the steal could never fire) would have cost three hours
  and returned the control three times, and no anchor could have caught
  it, because the anchor IS the control.

**A fourth lever, named while designing the steal: `DIV_W = 0.55`, the
ILP's own price on forking, was chosen on TWELVE division events and
its known failure mode is exactly the one the fork gate now fixes.**

First, a correction to the obvious version of this argument, which is
wrong. `DIV_W` does NOT govern the 5a case. Write the two options for a
daughter `q` the solver must place:

- **`q` is an ORPHAN.** Either she appears (cost `appearance_weight` =
  0.0) or the mother forks to her (gain `p_Mq`, cost `DIV_W`). The fork
  is taken iff **`p_Mq > 0.55`**. This is `reference.md`'s rule and it
  is the population `_add_safe_divisions` patches by geometry.
- **`q` already has a parent `R` (the 5a case).** Taking her means `R`
  loses its only child and must disappear, at `disappearance_weight`
  **1.5**. So the fork is taken iff `p_Mq - p_Rq > DIV_W + 1.5 = 2.05`,
  and probabilities live in [0, 1]: **it can never be taken.** 5a is
  structural, not a threshold. The binding constant there is the
  DISAPPEARANCE weight, not the division weight, and `ilpsweep`
  (2026-09-05) already measured loosening it -- disappearance 1.0 lost
  12 of 12 on 6bba.

So the patch-side steal is the right instrument for 5a, which is what
divbench4 runs. What `DIV_W` governs is the ORPHAN fork, and there the
line was closed under a constraint that no longer exists:

| div_w | ILP forks | div TP/FP/FN | div_J | adj_J | score |
|---|---|---|---|---|---|
| 0.20 | 1024 | **6**/57/6 | 0.0870 | 0.9193 | 0.9280 |
| 0.35 | 394 | 5/39/7 | 0.0980 | 0.9211 | 0.9309 |
| 0.45 | 190 | 5/27/7 | 0.1282 | 0.9216 | 0.9344 |
| **0.55 (shipped)** | 111 | 5/23/7 | 0.1429 | 0.9220 | 0.9363 |
| 0.65 | 49 | 4/19/8 | 0.1290 | 0.9222 | 0.9351 |
| 1.50 | 0 | 3/16/9 | 0.1071 | 0.9224 | 0.9331 |

(`biohub-ranker-fine`, 2026-08-23, entry far below.) **0.55 was the
argmax of a curve measured on 12 GT divisions**, and that entry says so
in its own words: "0.45-0.65 are all defensible and 0.55 is simply the
best measured". The reason 0.20 lost is that it made 1,024 forks and
57 counted false ones -- **over-forking, which is precisely and only
what the fork gate removes**: divbench3 cuts 80% of forks while keeping
84% of the true ones, and it was worth +0.010 at LB on 2026-09-07.
There was no gate in August. This is the same shape as gap2's span
verdict, closed behind a budget that later turned out to bind.

Two further reasons to reopen it, both from divbench3's own table:
**ILP forks are MORE precise than patch forks** (0.196 against 0.150 on
counted forks) and the ILP makes far fewer of them (2,490 against
4,514). Lowering `DIV_W` moves fork production from the geometric patch
to the probabilistic solver, which is the better-calibrated source, and
the gate cleans up behind both.

Why it was never re-swept: `DIV_W` is read by `solve_one` when the
ILPSolver is constructed, so it is a SOLVE-TIME constant, and the
2026-08-28 rule says such a constant cannot go in a post-link sweep --
it needs one full solve per arm, roughly 3x divbench3's stage B.
Affordable; nobody has spent it.

Order of work, deliberately: divbench4's 5a table first, because it
says whether the patch-side steal fires at all, and its answer changes
nothing about the `DIV_W` question. Measure first, then spend the
solves.

**`divbench6.py` is BUILT and waiting for a slot** (make_divbench6.py):
`solve_one` takes `divw`, arms at 0.45 / 0.35 / 0.20 each re-solve all
128 movies, every arm is patched, scored by the official scorer, and
then gated by a fork gate trained out of fold **on that arm's own
forks** -- the arm changes the fork population, so a REF-trained gate
would be measuring the wrong thing. REF is div_w 0.55 and must
reproduce 50/249/75. Pre-registered reading: (a) the decision is the
GATED table, not the ungated one, and an arm is a candidate only if it
beats REF+gate in both folds; (b) adj_J is measured far more precisely
than div_J, so an arm trading −0.002 adj_J for +0.03 div_J is a real
loss -- read both columns, never the sum alone; (c) check the ilp /
patch fork counts first: if lowering `DIV_W` does not move fork
production from the patch to the solver, the arm did not test what it
claims. ~4 solves x 128 movies, expect 5-6 h CPU.

### ⏳ fuse44b6 / ep44b6 (pushed 2026-09-07): F4 primaries and F4 head probabilities for the 59 44b6 train movies that do not have them -- the missing piece for a 44b6 division bench and a gate trained on both domains

The user's first open item. It is **59 movies, not 71**: the 12
manifest-excluded 44b6 (fusecache's `EX_44B6` = dctrain's held-out 12 =
edgeprob_cache's seeded `SET_44B6`) already have `fdet_F4_` in
biohub-fusecache and head caches in biohub-edgeprob, and must not be
recomputed or every anchor that reads them moves.

`fuse44b6.py` = fuse6bba with the 44b6 list (make_fuse44b6.py);
`ep44b6.py` = edgeprob_cache with no base pass and the F4 pass over
those 59 (make_ep44b6.py). After both, all 199 train movies have F4
primaries and F4 head probabilities.

Structure is symmetric with the 6bba side and carries the same caveat:
these 59 are INSIDE dctrain's 175-movie pool, so F4's fourth heatmap
(ep05 DeepCenter) is in-sample on them, exactly as it is on fuse6bba's
116. **No detection claim may be made from these caches** -- they exist
to give the division bench its 44b6 half.

Sizing, from the logs of the runs they are cut from: fuse6bba ran
82 s per 6bba movie (162 min for 116); edgeprob's F4 pass ran 34-54 s
per 44b6 movie (13,800 to 55,290 nodes each), so ep44b6 should be under
an hour. GPU quota at the push: 17,460 s used of 162,000 s, refreshing
2026-09-12.

**fuse44b6 LANDED: 59 of 59 in 90.1 min, no errors** (91 s per movie,
close to fuse6bba's 82 s). Both name lists printed and checked: the 59
built and the 12 held back (`44b6_0b24845f 44b6_0db75fae 44b6_33b596bf
44b6_5740d24b 44b6_87bba6c4 44b6_8f5ab931 44b6_9bfa6a0a 44b6_a2bb48bb
44b6_c15fded2 44b6_c96cfa10 44b6_d2f34f90` + one more). Peaks at 0.15
per movie: min 4,768, **median 32,149**, max 67,265 -- against 6bba's
range of 3,366 to 80,832 with a median near 18,000, so 44b6's F4 node
sets are about 1.8x the size. That is the number to expect the 44b6
bench's solves to be slower by. **ep44b6 LANDED: 59 of 59, EDGEPROB OK,
45.3 min, 79.0 MB, no errors.** Its optional cross-check ran this time
(biohub-dctrain-175-weights was attached, which ep6bba did not do) and
printed "44b6 draw cross-checked against split_manifest.json: disjoint"
-- so the 12 held back are provably dctrain's held-out 12 and nothing an
anchor reads was recomputed. Per movie: 20,993 to 67,265 nodes, 41-64 s.

**All 199 train movies now have F4 primaries and F4 head probabilities**
(6bba: fusecache 12 + fuse6bba 116; 44b6: fusecache 12 + fuse44b6 59;
heads: edgeprob 24 + ep6bba 116 + ep44b6 59). That was the missing piece.

**divbench5 pushed** against the PRODUCTION gate rather than divbench4's
candidates: `fork_gate_f4.pkl` from biohub-divbench3 is the gate that
shipped in `submit_gate.py` and scored LB 0.943, so it is the one whose
cost on the other batch matters. divbench4's 28-feature candidate does
not need a kernel to compare -- both runs dump their fork tables with
all 28 columns, so that comparison is a gatelab job, offline, with the
paired bootstrap.

**divbench5 v1 (80.4 min for stage B) reached the report and then died
on `np.quantile` of an empty array, but its measured numbers are real
and both hard anchors held.** THE FIRST DIVISION BENCH ON 44b6:

| | 44b6 (71 movies) | 6bba (128, divbench3) |
|---|---|---|
| GT divisions | **26** | 125 |
| REF div tp / fp / fn | **6 / 61 / 20** | 50 / 249 / 75 |
| div_J | **0.0690** | 0.1337 |
| adj_J | **0.8868** | 0.9133 |
| score | **0.8937** | 0.9266 |
| forks in the final graphs | **5,490** | 7,004 |
| patch forks: TP / FP / uncounted (precision) | 2 / 37 / 4,062 (**0.051**) | 28 / 159 / 4,327 (0.150) |
| ILP forks: TP / FP / uncounted (precision) | 4 / 24 / 1,361 (**0.143**) | 22 / 90 / 2,378 (0.196) |
| lost divisions that stop at 5a | 8 of 20 | 23 of 75 |

Anchors, both exact: D187 at floor 0.10 on the 12 manifest-excluded 44b6
adj_J **0.8353** (want 0.8353, divsteal) and the production +BIDIR pkl
at floor 0.48 on the same 12 **0.8384** (want 0.8384, divbench3's 44b6
REF row). Per fold: 4/37/13 and 2/24/7, div_J 0.0741 / 0.0606.

Three things worth carrying out of it before v2 even lands:

- **The orphan patch is three times less precise on 44b6 than on 6bba**
  (0.051 against 0.150 on counted forks). Its geometry gates were tuned
  on the sparse batch; on the dense one they admit twenty false forks
  per true one. The solver's own forks degrade far less (0.143 against
  0.196), which is the second independent reason to move fork
  production toward the solver (the first is in the divbench6 note
  above).
- **The patch's per-frame cap refused 861 proposals on 44b6** against
  7 refusals by the global cap. Nobody had ever counted this stage's
  caps; on the dense batch the FRAME cap is the binding one. Every
  verdict about the patch's geometry gates on 44b6 is provisional on
  it (2026-08-27 rule).
- **44b6 cannot train a fork gate and that is now measured, not
  assumed**: 67 counted forks, **six** of them true. A 2-fold split
  leaves under 5 true forks per training fold, which is what crashed
  v1. A 44b6 gate has to be a JOINT gate over 125 + 26 events, and
  that join is an offline gatelab job on the two fork tables, not a
  kernel.

v2 turns all three of those into printed findings instead of a
traceback and goes on to the transfer test and the real pruning, which
are what the run is for.

**⭐⭐⭐ divbench5 v2 LANDED (117.9 min, 71 of 71, both hard anchors
exact): THE SHIPPED FORK GATE TRANSFERS TO THE OTHER IMAGING BATCH AND
GAINS THERE.** `fork_gate_f4.pkl` -- divbench3's, trained on 299
counted 6bba forks, shipped in `submit_gate.py`, LB 0.943 -- has never
seen a 44b6 movie:

| | 44b6, all 71 movies |
|---|---|
| its AUC on 44b6's 67 counted forks (6 TP / 61 FP) | **0.8607** |
| for comparison, its out-of-fold AUC at home | 0.9283 / 0.8326 |
| REF | 6 / 61 / 20, div_J 0.0690, adj_J 0.8868, score 0.8937 |
| **+ gate at p > 0.03, real pruning, official scorer** | **4 / 7 / 22, div_J 0.1212, adj_J 0.8881, score 0.9002** |
| delta | adj_J **+0.0014**, score **+0.0066**, **BOTH folds up** (+0.0015 / +0.0012 adj_J) |
| share of 44b6's 5,490 forks removed | 89.3% (6bba 80%, the submit-gate visible 4 85%) |

**The 44b6 AUC sits INSIDE the gate's own fold-to-fold spread at home**
(0.8607 against 0.9283 and 0.8326), so the ranking is not 6bba-specific.
It sheds 54 of 61 counted false forks for 2 of 6 true ones. And the
answer to the question the bench was built for -- does the gate cost the
other batch anything -- is no: it gains, on 71 movies and roughly seven
million edges, in both folds.

Consistency check against LB: the local three-domain weighting (6bba
85%, 44b6 15%) over +0.0121 and +0.0066 predicts +0.0113; the
leaderboard moved +0.010. divbench3's 12-movie draw had said +0.0020
adj_J on 2 GT divisions; the full 71 says +0.0014 adj_J on 26.

Two smaller things: on 44b6 a STRICTER cut reads better (p > 0.1 gives
4 / 3 / 22, div_J 0.1379 against 0.03's 0.1212) -- but that is 4 true
forks either way and the difference is three false ones, which 26
events cannot resolve. And divbench4's candidate gates were not mounted
(they did not exist when this was pushed); the run says so in a warning
rather than silently substituting, and gatelab6 has since shown they
are all worse anyway.

**Tested claim: that the fork gate is a 6bba artefact and the LB +0.010
came from the 85% alone. Refuted -- it transfers, and on the other 15%
it is worth +0.0066 of local score.** Residue: 44b6's own 26 events
cannot select anything, so the JOINT gate (125 + 26) is still open and
is an offline job on `forks_44b6.csv` + `forks_f4_v4.csv`; and 44b6's
20 lost divisions have their own walk, with 8 of them at 5a.

### ⭐⭐⭐⭐⭐ gatelab7-10 (2026-09-08, OFFLINE on both fork tables, no kernel): TRAINING THE FORK GATE ON BOTH BATCHES BEATS THE SHIPPED 6bba-ONLY GATE ON 6bba ITSELF, by +0.036 of div_J -- the first gate variant in this whole line to clear the bench's own noise bar, and four controls say it is the 44b6 LABELS that carry it

The shipped gate learns from 299 counted 6bba forks (50 true). divbench5
produced 67 counted 44b6 forks (6 true). Adding them to the training set
and evaluating OUT OF FOLD ON 6bba, cut p > 0.03:

| gate | pooled AUC on 6bba counted | TP / FP | div_J |
|---|---|---|---|
| 6bba only (SHIPPED) | 0.8798 | 42 / 47 | 0.2442 |
| **+ all 67 counted 44b6 forks** | **0.9016** | **46 / 39** | **0.2805** |

Paired bootstrap over 6bba movies, 2,000 draws, exact per-movie GT
counts: **+0.0360, 90% [+0.0163, +0.0597], distinguishable.** Against
the ±0.03 bar that gatelab set and divbench4's pre-registration used,
this is the first thing that clears it -- every feature-set, model and
fold-count variant tried before sat inside the interval.

At each gate's OWN best threshold, which compares RANKINGS rather than
calibrations: shipped 0.2531 (41 TP / 37 FP), joint **0.2866**
(45 / 32). On 44b6 out of fold the joint gate is also ahead, 4 / 4
against 4 / 6.

**Four controls, because a local number with no mechanism is worth
nothing here** (2026-08-28):

| control | paired d on 6bba | reading |
|---|---|---|
| joint | **+0.0360** [+0.0163, +0.0597] | the effect |
| (A) pad with a BOOTSTRAP of the same 6bba fold | **−0.0280** [−0.0572, −0.0008] | not row count -- more of the same is WORSE |
| (B) pad with 6bba negatives only | −0.0077 | not label balance |
| (B2) pad with 44b6 negatives only | −0.0111 | not the negatives |
| (E) the same 44b6 rows, LABELS SHUFFLED (20 shuffles) | **−0.0186** | not regularisation |
| (F) noise rows with 44b6's label mix (20 shuffles) | −0.0075 | not extra distinct rows |

**(E) is the one that settles it: the same feature rows with their
labels destroyed make the gate WORSE, so what 44b6 contributes is
label information, not extra data volume.** Robustness: +0.0360
(2-fold), +0.0196 (3-fold), +0.0235 (4-fold), +0.0191 (a shallower
model) all with intervals excluding zero; one setting (200 iterations,
min_samples_leaf 5) reads −0.0077 and is a wash.

**The mechanism in feature terms is NOT known, and one candidate is
refuted.** The obvious story was density: `dens10` and `n_orph` read
inverse (AUC 0.316 / 0.245) and 44b6 has 327 cells per frame against
6bba's 97, so a one-batch gate could use "few neighbours" as a proxy
for "true division". `n_orph` does behave differently across batches
(6bba TP median 3.5 against FP 10.0; 44b6 TP **18.5** against FP 14.0 --
the sign REVERSES). But dropping both density features makes the joint
advantage BIGGER, not smaller (+0.0547 against +0.0360), so density is
not what 44b6 is correcting. The entry says unknown rather than
inventing one.

Weighted division term at p > 0.03, 6bba 85% + 44b6 15%: shipped
+0.0102, joint **+0.0134**.

**NOT yet a submission candidate**, and two things stand in the way,
both stated before anyone spends a slot:

1. **div_J here is simulated from labels; adj_J is not measured.** The
   real pruning has to run (divbench3/4 both found adj_J RISES about
   +0.0010 under the cut, so the sign is likely, but "likely" is not a
   measurement).
2. **The production pkl must be trained ON KAGGLE.** Local
   scikit-learn 1.9.0 fits a model whose RANKING matches the recorded
   one exactly -- the 2-fold anchors reproduce to four decimals -- but
   whose probability CALIBRATION does not: the production gate's
   quantiles over all 7,004 forks read 60% 0.0041 / 70% 0.0074 / 80%
   0.0238 locally against Kaggle's 0.0044 / 0.0083 / 0.0275. A fixed
   cut of 0.03 sits at a different place in the two. **Any gate that
   ships must be fitted in the same image that will load it**, and the
   cut re-read from that fit's quantiles.

`divbench8` is that run: the 6bba bench again, with 44b6's fork table
mounted as a CSV, both gates trained out of fold, both applied with
real pruning and the official scorer, and `fork_gate_joint.pkl`
written.

### ⭐⭐⭐ divbench3 (2026-09-06, F4 = production primaries, anchors exact): the fork gate, pruning applied for real and scored by the official scorer -- +0.0121 of local score at p > 0.03 (div_J 0.134 -> 0.244, adj_J +0.0010), BOTH 6bba folds up (+0.0132 / +0.0096), and the 12-movie 44b6 draw +0.0020 with its 11 counted false forks all gone. Submission candidate; fork_gate_f4.pkl written

Built and pushed in sequence; each is the previous one's source.
`fuse6bba.py` = fusecache.py with the F4 arm only, no Zebrahub pass,
the 116 6bba train movies OUTSIDE the 12-movie holdout (the holdout's
fdet_F4_ files stay the ones in biohub-fusecache, so the F4 anchors
keep their nodes). `ep6bba.py` = edgeprob_cache.py with no base pass
and the F4 pass over those 116 (fdet_F4_ from biohub-fuse6bba).
`divbench3.py` = divbench2 on F4: ranker training rows stay BASE
(that is how the production +BIDIR ranker was trained, D175 pool on
BASE primaries, applied to F4 nodes), the solve features come from
F4 detections + the head's F4 caches (features_ext now tolerates a
missing head-2 cache: NaN columns, unused by +BIDIR); hard anchor
D187 on the holdout at floor 0.10 = F4-6bba 0.9216 / div_J 0.0606;
soft: the production +BIDIR pkl at floor 0.48 on the same 12 =
ilpsweep's F48 row 0.9261. New against divbench2: the fork table is
written (`forks_f4.csv`), the one-feature rule on the orphan's
detection score is swept, the 2-fold gate's curve is read by quantile
AND by fixed probability, the production gate (all counted F4 forks)
is written to `fork_gate_f4.pkl` with its p-quantiles over all
forks, **the pruning is applied for real** (the mother->orphan edge
removed for every fork under the cut, geff rewritten, official
scorer, adj_J included, per fold) at oof-q70 / oof-q80 / p > 0.01 /
p > 0.03, and the 12-movie 44b6 draw is solved under the production
pkl and pruned by the production gate at its own quantiles (2 GT
divisions there: read adj_J and false forks, not div_J).

Pre-registered reading: a cut is a submission candidate iff its "d
score" (adj_J + 0.1 div_J, official scorer) is positive in BOTH
6bba folds after the real pruning AND 44b6's adj_J does not fall;
the BASE-bench figure to beat is +0.011 at the 70% cut. If the F4
gate's out-of-fold AUC drops well below 0.90, the BASE result was
partly the BASE detector's score distribution and the gate is
detector-specific (still usable, since production IS F4). Expected:
fuse6bba ~3 h GPU, ep6bba ~1.5 h GPU, divbench3 ~3 h CPU.

**Landed: fuse6bba 162 min (116 movies, F4 peaks per movie printed),
ep6bba 72 min (116 F4 caches, EDGEPROB OK), divbench3 127 min
(stage B 92 min, 128 of 128).** Anchors: D187 F4-6bba(12) 0.9216 /
div_J 0.0606 exact (hard); the production +BIDIR pkl at floor 0.48
on the same 12 = 0.9261 exact (soft, ilpsweep's F48 row). The fold
rankers (BASE rows, as production) on F4 nodes: REF 50 / 249 / 75,
div_J 0.1337, adj_J 0.9133, score 0.9266 (BASE bench 54 / 258 / 71,
0.1410, 0.9136, 0.9277 -- the same picture, four fewer true forks).
Lost-division walk on F4: detection 31 (mother 9, daughter 22), 5a
steal 23, 8b divergence 11, 9 capped-or-taken 3, others 7 -- the
BASE numbers within a few events each.

Forks: 7,004 (patch 4,514: 28 TP / 159 FP / 4,327 uncounted,
precision 0.150; ILP 2,490: 22 / 90 / 2,378, 0.196). Feature AUCs on
the 299 counted forks: sc_q 0.842 (TP median 0.415 vs FP 0.313),
dv 0.787, sc_p 0.756, pf_q 0.677, sc_c1 0.657, d_s 0.654, p_q 0.637;
dens10 0.316 and n_orph 0.245 (inverse: true forks sit in sparser
frames with fewer orphans). `forks_f4.csv` written (7,004 rows).

Gate (HistGB 100 it, trained on the other fold's counted forks):
out-of-fold AUC fold0 0.9283 (122 counted forks), fold1 0.8326 (177).
Simulated curve (labels only): quantile 70% (p > 0.0126) 44 / 79 /
81, div_J 0.2157; quantile 80% (p > 0.0306) 42 / 46 / 83, 0.2456;
fixed p > 0.03: 42 / 47 / 83, 0.2442; fixed p > 0.02: 43 / 60 / 82,
0.2324; oracle 0.4000. The one-feature rule sc_q >= 0.345 (70% of
forks) reaches 0.2045 (+0.0071 of score); the gate beats it.

**Pruning applied for real** (the mother->orphan edge removed for
every fork under the cut, geff rewritten, official scorer, 128
movies, out-of-fold p):

| cut | div tp / fp / fn | div_J | adj_J | node_r | score | d adj_J | d div_J | d score | fold0 score | fold1 score |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 50 / 249 / 75 | 0.1337 | 0.9133 | −0.0238 | 0.9266 | -- | -- | -- | 0.9435 | 0.9116 |
| oof q70 (p > 0.0126) | 44 / 79 / 81 | 0.2157 | 0.9141 | −0.0239 | 0.9357 | +0.0009 | +0.0820 | +0.0091 | 0.9524 | 0.9195 |
| oof q80 (p > 0.0306) | 42 / 46 / 83 | 0.2456 | 0.9143 | −0.0239 | 0.9388 | +0.0010 | +0.1119 | +0.0122 | 0.9571 | 0.9212 |
| fixed p > 0.01 | 44 / 93 / 81 | 0.2018 | 0.9140 | −0.0239 | 0.9342 | +0.0008 | +0.0681 | +0.0076 | 0.9514 | 0.9178 |
| **fixed p > 0.03** | **42 / 47 / 83** | **0.2442** | **0.9143** | −0.0239 | **0.9387** | **+0.0010** | **+0.1105** | **+0.0121** | **0.9567 (+0.0132)** | **0.9212 (+0.0096)** |

adj_J RISES by 0.0010 under every cut (the removed edges were mostly
false edges too); the node term is untouched (nodes stay).

The 44b6 draw (12 movies, production +BIDIR pkl, floor 0.48, the
production gate = all 299 counted 6bba forks, applied at its own
quantiles / fixed cuts): REF 0 / 11 / 2, adj_J 0.8384 (= ilpsweep's
F48 row, exact); every cut removes all 11 counted false forks, loses
nothing (the 2 GT divisions were already FN), adj_J 0.8404 =
**+0.0020**; 957 forks in all across the 12, 11 counted. Per movie
the gate raised adj_J on 6, left 5 unchanged, and lowered one
(d2f34f90 0.9598 -> 0.9563, a fork whose orphan edge was a true
edge though the fork was not a counted division).

Production gate: trained on all 299 counted F4 forks (50 TP); its p
over all 7,004 bench forks has quantiles 50% 0.0028 / 60% 0.0044 /
70% 0.0083 / 80% 0.0275 / 90% 0.2558. Written to `fork_gate_f4.pkl`
(model, feature names, version, quantiles, thresholds) in the
biohub-divbench3 output.

Reading. (1) **Both pre-registered conditions hold: d score is
positive in both 6bba folds after the real pruning (+0.0132 /
+0.0096 at p > 0.03) and 44b6's adj_J rises (+0.0020).** The F4 gate
is a little weaker than the BASE one out-of-fold (fold1 AUC 0.83 vs
0.90) and the gain is the same size (+0.012 vs +0.011), because
adj_J also moves up. (2) The cut for production is a FIXED
probability, p > 0.03, not a quantile: the hidden set's fork
population is unknown, and 0.03 sits at the bench's 80% quantile
(the production gate's own 80% is 0.0275); the simulated curve is
flat between 0.02 and 0.05 (+0.0099 to +0.0110), so the choice is
not sharp. (3) The local three-domain weighting (6bba 85%, 44b6
15%) puts the expected LB move near +0.010 if the hidden movies
behave like the train movies of the same embryos; the caveat is the
usual one -- every fork the gate learned from is 6bba, annotated by
the same people. (4) zh was not run (its divisions are a different
population and it is not scored on LB).

Next: `submit_gate.py` = submit_f48.py + the gate (make_submit_gate.py):
the ranker's pair probabilities and the head's forward / reverse p
kept per movie, the orphan patch logging what it adds, the 20 bench
features computed for every fork of the final graph, fork_gate_f4.pkl
loaded from the biohub-divbench3 output, edges of forks with p <= 0.03
removed, nothing else changed. Kernel biohub-submit-gate, then LB.

**submit-gate v2 landed (28 min; v1 died at import on an `os`
reference pyflakes had flagged -- fixed before it ran a movie).** The
gate loaded from biohub-divbench3's output (version
2026-09-06a-divbench3, threshold p > 0.03). Visible 4, forks judged /
removed: 44b6_0113de3b 64 / 41, 44b6_0b24845f 75 / 75, 6bba_05b6850b
23 / 14, 6bba_05db0fb1 135 / 123 -- 297 forks, 253 removed (85%; the
bench's 80% quantile on 6bba, and on the dense 44b6 movie every fork
goes, as it did on the bench's 44b6 draw where those forks carried
no counted division and adj_J did not move). Output: nodes 120,411
(= F48, nodes untouched), edges 114,342 (F48 114,595, −253), rows
234,753. **Submitted as ref 56055405 (2026-09-06)**, its own control
ref 56030644 = 0.933 / 266,263,495 bytes. Pre-registered reading:
bytes should read about −0.2% (253 of 114,595 edges on the visible
4); LB ≥ +0.005 = the gate transfers and the division line is open
at LB for the first time; +0.001 to +0.004 = it transfers weakly (the
hidden 6bba forks are less separable, or the hidden set has fewer
counted divisions than the train movies); ≤ 0 = the bench's folds
(same embryo, same annotators) do not stand for the hidden movies,
and the gate is recorded, not shipped.

**LB 0.943, total_bytes 266,152,167 (scored 2026-09-07). +0.010 over
its control 0.933, bytes −0.04% (the bench's visible-4 edge removal
was −0.2%; the hidden set carries proportionally fewer forks or the
gate cuts fewer of them there). NEW BEST, the largest single LB gain
on record (edgeprob +0.007, F4 +0.003), and the first division
mechanism ever to score at LB.** Reading against the pre-registered
scale: ≥ +0.005 = the gate transfers; it did, at the size the local
three-domain weighting predicted (+0.010 from 6bba +0.012 / 44b6
+0.002). Two facts for next week: (1) the division term on LB was
worth at least 0.010 to us at 42 of 125 true forks recovered locally
-- the oracle (every false fork gone) is 0.40 div_J against 0.244
here, so the same lever has room left, and the FN side (5a steals,
23 of 71) has not been touched; (2) the record's byte pattern now
reads seven for seven: every submission that removed output gained.

Tested claim: false forks separable at fork time on a 125-event
out-of-fold bench transfer to the hidden set. Confirmed at LB:
+0.010. Residue: the gate was trained on 6bba forks only and applied
to every fork; 44b6's 71 train movies have no F4 caches yet, so the
gate has never been trained on a 44b6 fork; the cut 0.03 was chosen
on a flat part of the curve (0.02-0.05) and not swept at LB.

### ⭐⭐ divbench2 (2026-09-06, biohub-divbench2 v1, 97 min, anchor exact): the false forks CAN be told apart at fork time -- a 2-fold gate on the forks the pipeline emits reads AUC 0.92 / 0.90 out-of-fold, and cutting 70% of forks keeps 52 of 54 true ones, sheds 173 of 258 false ones, div_J 0.141 -> 0.248 (+0.011 of score), BOTH folds up; the orphan daughter's detection score alone is AUC 0.885

Same bench as divbench (128 6bba movies, two out-of-fold +BIDIR
rankers, BASE primaries, production ILP at disappearance 1.5 / floor
0.48), REF patch only, instrumented (make_divbench2.py): (1) every
fork the final graph carries, with the official scorer's verdict on
it from `score_divisions` (tp_forks / fp_forks / neither = uncounted)
and its features at fork time -- source (patch / ILP), the three
distances, divergence, the ranker's p for mother->child and
mother->orphan, the head's forward / reverse p for the orphan edge,
detection scores of the three nodes, the mother's backward chain
length, each daughter's forward chain length, primaries within 10 um
of the mother, orphans in the frame, time fraction, the p gap between
the two edges; (2) divgate2's condition walk for every GT division
the scorer did not pair, on the pre-patch graph, with the ILP's own
failures split (mother has no child / mother forked elsewhere); (3) a
2-fold fork gate (HistGB, 100 iterations, trained on the OTHER fold's
counted forks, TP = 1, FP = 0) applied to every fork, with the
pruning curve at 0-90% cuts: tp / fp / fn and div_J pooled and per
fold, and the oracle line (every FP gone, every TP kept, 0.43). Hard
anchor 0.9293 as before.

Pre-registered reading: (a) the walk's largest bucket names the lever
for the lost 71 -- 5a (the ILP already gave the second daughter a
parent) = a steal rule, priced on this bench next; 1a/1b = the ILP
ends or mis-forks the mother, a linker question; 8b = the divergence
constant, already shown to cost more than it refuses; (b) the fork
gate is read as div-gate was read: a cut is a candidate only if
fold0 AND fold1 div_J both rise at the same cut, with the feature
table saying which feature carries it; if the out-of-fold AUC on
counted forks is below ~0.65 or the curve is flat / downhill, the
false forks are not separable at fork time and the FP lever closes
with div-gate's 2026-08-24 lesson intact at 10x the sample; (c) the
patch's own forks vs the ILP's: precision by source says which of
the two to gate. Runtime ~2.5 h CPU.

**Landed (97 min). Anchor 0.9293 exact; REF reproduces divbench v1
exactly (54 / 258 / 71, 0.1410 / 0.9136 / 0.9277).**

Where the 71 lost divisions stop (first condition, pre-patch graph):

| condition | all | fold0 | fold1 |
|---|---|---|---|
| mother not detected (no node within 5 um) | 7 | 3 | 4 |
| a daughter not detected | 26 | 7 | 19 |
| 1a mother has no child at all | 2 | 2 | 0 |
| 2 mother has no single predecessor | 2 | 1 | 1 |
| **5a the ILP already gave the second daughter a parent** | **19** | 11 | 8 |
| 5c second daughter > 12 um from the mother | 3 | 2 | 1 |
| 7 not the sister's nearest orphan | 1 | 0 | 1 |
| **8b divergence < 2.25 um** | **10** | 4 | 6 |
| 9 passed every gate, then capped or taken | 1 | 0 | 1 |
| (recovered) | 54 | 31 | 23 |

So on competition data too, a third of the losses (33 of 71) are
DETECTION -- a mother or daughter with no BASE node within 5 um --
contrary to "competition GT divisions never lose a daughter at
detection", which was true of the 14 local ones and is not true of
125 (14 was too few to see it). Of the 38 linker-side losses, 19 are
the steal case (5a) and 10 the divergence constant (8b); the geometry
gates (5c, 7) stop 4.

Every fork the 128 final graphs carry: 7,063 (patch 4,532: 29 TP /
157 FP / 4,346 uncounted, precision 0.156 on counted; ILP 2,531: 25 /
101 / 2,405, 0.198). 96% of forks touch no annotated cell.

Fork features by verdict (median TP / FP / uncounted; AUC TP-vs-FP on
the 312 counted forks):

| feature | TP | FP | unc | AUC |
|---|---|---|---|---|
| **sc_q, detection score of the orphan daughter** | 0.431 | 0.316 | 0.296 | **0.885** |
| sc_p, mother's detection score | 0.463 | 0.400 | 0.353 | 0.786 |
| dv, divergence (um) | 3.17 | 2.49 | 2.57 | 0.761 |
| pf_q, the head's forward p for mother->orphan | 0.674 | 0.547 | 0.524 | 0.682 |
| sc_c1, first daughter's detection score | 0.427 | 0.389 | 0.340 | 0.672 |
| d_s, sister separation (um) | 8.73 | 6.96 | 7.52 | 0.647 |
| p_q, ranker p for mother->orphan | 0.545 | 0.459 | 0.436 | 0.629 |
| d_pc1 | 3.28 | 2.60 | 2.72 | 0.628 |
| len_q / len_c1 / len_back (chain lengths) | 12 / 12 / 12 | 11.5 / 11 / 12 | 11 / 10 / 8 | 0.615 / 0.573 / 0.589 |
| dens10 (primaries within 10 um) | 0 | 1 | 1 | 0.370 (inverse) |
| d_pq, p_c1, pf_c1, pr_q, t_frac, rank_gap | -- | -- | -- | 0.45-0.56 (nothing) |

The gate (HistGB 100 iterations, trained on the OTHER fold's counted
forks, applied to every fork of this fold): **out-of-fold AUC 0.9231
on fold0's 146 counted forks, 0.9006 on fold1's 166.** Pruning curve
(cut = fraction of ALL 7,063 forks removed, by the gate's p):

| cut | TP | FP | FN | div_J | d div_J | ~d score | fold0 J | fold1 J |
|---|---|---|---|---|---|---|---|---|
| 0 | 54 | 258 | 71 | 0.1410 | -- | -- | 0.1761 | 0.1111 |
| 30% | 53 | 196 | 72 | 0.1651 | +0.0241 | +0.0024 | 0.2026 | 0.1310 |
| 50% | 53 | 155 | 72 | 0.1893 | +0.0483 | +0.0048 | 0.2123 | 0.1642 |
| 60% | 53 | 118 | 72 | 0.2181 | +0.0771 | +0.0077 | 0.2348 | 0.1982 |
| **70%** | **52** | **85** | 73 | **0.2476** | **+0.1066** | **+0.0107** | **0.2696** | **0.2211** |
| 80% | 47 | 51 | 78 | 0.2670 | +0.1261 | +0.0126 | 0.3011 | 0.2289 |
| 90% | 36 | 15 | 89 | 0.2571 | +0.1162 | +0.0116 | 0.3188 | 0.1972 |
| oracle | 54 | 0 | 71 | 0.4320 | +0.2910 | +0.0291 | | |

Both folds rise monotonically to the 80% cut; at 70% two true forks
are lost for 173 false ones. adj_J is not re-simulated (a pruned fork
removes one edge to an orphan; div-gate 2026-08-24 measured +0.001
adj_J over a 90% cut, the removed edges being mostly false).

Reading. (1) **This is the first division mechanism in the record
that is positive in both folds at bench scale, and it is large:
+0.011 of local score at a 70% cut, +0.013 at 80%, against +0.007 for
edgeprob, the largest LB gain on record.** The false forks are
separable at the moment they are made, by things the patch never
looked at: the orphan daughter's detection score (0.316 vs 0.431
median, AUC 0.885 alone), the mother's score, the divergence
magnitude, the head's forward probability. The patch's geometry
gates admit forks onto weak detections -- which is exactly where
unlabelled-or-spurious nodes live -- and the ILP's own forks have
the same disease (precision 0.198). (2) **Why div-gate 2026-08-24
found the opposite**: its model was the kinematics model trained on
Zebrahub candidate triples and applied to 553 holdout forks with 5
true ones; here the gate is trained on the forks this pipeline
emits, labelled by the scorer, and evaluated on 125 events. Same
lesson, opposite outcome once the training population is the
intervention's population. (3) The lost-division walk moves the FN
lever down the list: 33 of 71 are detection (mother or daughter has
no BASE node), 19 are steals from a parent the ILP already chose
(divsteal on zh: the mother's p is 0.18 vs 0.77 for the competitor,
and 8 of 27 competitors were true edges -- a steal rule is not
obviously safe), 10 the divergence constant (relaxing it costs more
than it refuses, divbench). The FP lever is worth 3-4x more than
anything on the FN side and is the one to build.

Caveats before it ships, in order of danger: (a) **the bench is
BASE primaries; production is F4 (xl + ep05 map), whose detection
scores are a different distribution**, and sc_q / sc_p carry the
gate -- the gate must be trained on F4 forks, which needs F4
detections and head probabilities for all 128 6bba movies (GPU);
(b) 44b6 is not on the bench (26 divisions, local div_J 0.0000, 15%
of weight): the gate must at least not lose there; (c) the two folds
are the same embryo and the same annotators as the hidden 6bba
movies are presumed to be; (d) the fork table was not written to
disk, so the single-feature rule on sc_q that would make the gate
detector-transferable was not simulated -- next run dumps it.

Next: (1) build the F4 caches for the 128 6bba train movies (and the
71 44b6 if the GPU budget allows): detections via the fusecache path,
head probabilities via the edgeprob path; (2) divbench3 = divbench2
on F4 with the fork table dumped, the gate trained out-of-fold on F4
forks, plus the single-feature sc_q rule and the BASE-trained gate
applied to F4 forks (does it transfer?); (3) if both folds hold on
F4, wire the gate into submit_f48's division stage and the ILP
forks (prune by gate p at the bench's cut), 44b6 checked on its 12,
and submit.

### divbench (2026-09-05, biohub-divbench v1, 128 min, anchor exact 0.9293): the division bench on 125 GT divisions -- REF recovers 54 with 258 counted false forks (div_J 0.141 = 0.014 of score); the patch is worth +0.002; tighter geometry and looser divergence BOTH lose; the n = 12 signs were right

The user, after F48 landed: "去搞分裂吧，看看分裂". What the record
says before anything new is run: the competition's GT divisions are
few (151, 26 in 44b6 / 125 in 6bba) and wide (sister separation
median ~10-12 um on the 6bba holdout, vs 5-8 um on Zebrahub); none
of the 14 local ones loses a daughter at detection (a division enters
the competition GT only if a human saw mother and both daughters), so
on the domain LB scores, divisions are a LINKER + FORK-RULE question,
and Zebrahub (65% daughter-missed, 468k machine divisions, sisters
5-8 um) is the wrong population for it. The linker's ILP forks
correctly on 0 of 12 (divwhy); everything we score comes from the
orphan-fork patch (`_add_safe_divisions`, 456 forks on the holdout
for 2 true); divgate2 measured that 0% of real divisions pass the
divergence test at 2.25 um, and divsweep found that lowering it
COSTS 6bba div_J (0.0606 -> 0.0435 at 1.0) on n = 12, where one true
positive is worth 0.08 of div_J. Every division verdict in this
record rests on 12 events. That is the problem this run addresses,
not a new rule.

`divbench.py` (make_divbench.py in the scratchpad, from
edgeprob_rank2.py, which already computes features for all 199 train
movies to train the D175 rankers): BASE (xl) primaries + BASE
edgeprob caches, the only detector with caches for all 199; every
6bba movie keeps its full features; two +BIDIR rankers (18 + p_fwd /
p_rev / h_fused, poolsize hyperparameters), fold k trained on the 71
44b6 movies + the other 64 6bba movies, so every 6bba movie is solved
by a ranker that never saw it; one production solve per movie
(disappearance 1.5, floor 0.48 = ref 56030644); then five post-link
variants on that solve, scored by the official scorer: REF (shipped
rules: parent 12 / sister 15 / divergence 2.25), NB (the 0.948
notebook's geometry, parent 7 / sister 12, divergence 2.25), DIVG1
and DIVG0 (divergence 1.0 / 0), NODIV (no patch, the ILP's forks
only = what the patch buys in total). Hard anchor: D187 on the
12-movie holdout at floor 0.10 must read BASE-6bba 0.9293 (rank2
v1). Printed: pooled division tp / fp / fn and div_J per variant and
per fold, adj_J and score, every movie with a GT division and its
tp / fp / fn per variant, the count of division-free movies carrying
counted false forks, paired per-movie score deltas.

Pre-registered reading: (a) REF's pooled div_J on 125 is the first
real local estimate of what LB's 0.1 x div_J term is worth to us;
(b) a variant that raises div_J AND does not lower adj_J, in both
folds, is a submission candidate (44b6 is absent here, 15% of the
weight, 26 divisions, and its own local div_J is 0.0000); (c) NODIV
vs REF prices the whole orphan-fork patch on 125 events -- if it is
near zero or negative, the fork line is closed at the patch, not at
the rules; (d) the divergence rows decide whether divsweep's n = 12
loss at 1.0 um was noise. Expected runtime ~3-4 h CPU (features for
199 ~60 min, two fits, 128 solves at one arm).

**Landed (128 min: features for 199 in 24 min, two fits 4 s each,
stage B 102 min; the run then died in the paired-delta print block
after every table had printed -- the tables are complete, 128 of 128
movies x 5 variants).** Hard anchor BASE-6bba(12) D187 0.9293 exact
(div_J 0.1316, the poolsize figure). Fold rankers: val AUC 0.9910 /
0.9929, held-out-fold AUC 0.9933 / 0.9911 -- no in-sample gap.

| variant | div tp / fp / fn (125 GT) | div_J | adj_J | score | d div_J | d adj_J | d score |
|---|---|---|---|---|---|---|---|
| REF (parent 12 / sister 15 / diverge 2.25) | 54 / 258 / 71 | **0.1410** | 0.9136 | 0.9277 | — | — | — |
| NB (the notebook's 7 / 12 / 2.25) | 35 / 163 / 90 | 0.1215 | 0.9142 | 0.9264 | −0.0195 | +0.0006 | −0.0013 |
| DIVG1 (diverge 1.0) | 46 / 454 / 79 | 0.0794 | 0.9126 | 0.9205 | −0.0615 | −0.0010 | −0.0072 |
| DIVG0 (diverge 0) | 37 / 471 / 88 | 0.0621 | 0.9128 | 0.9190 | −0.0789 | −0.0009 | −0.0087 |
| NODIV (no patch; the ILP's forks) | 25 / 101 / 100 | 0.1106 | 0.9145 | 0.9256 | −0.0304 | +0.0009 | −0.0021 |

Per fold (REF): fold0 31 / 115 / 30, div_J 0.176, adj_J 0.9267;
fold1 23 / 143 / 41, div_J 0.111, adj_J 0.9013 -- the two halves of
6bba differ by 0.065 of div_J with 64 movies each, so even at this
size a division number carries about ±0.03. 45 of the 128 movies have
no GT division and still carry counted false forks. Per movie
(gt: REF tp/fp/fn), the ones with 3+ GT: 05db0fb1 3: 0/8/3; 09961292
4: 0/7/4; 12665c0e 3: 1/2/2; 20852818 3: 0/2/3; 48816121 5: 2/1/3;
4ffd3da3 3: 1/2/2; 57b7cc1e 3: 0/9/3; 5c039895 3: 3/1/0; 786893ac 3:
0/7/3; 7f87b3d8 3: 2/0/1; afb141ff 4: 1/0/3; bb9f20c3 3: 2/9/1;
cdcfe533 4: 4/3/0; debd7bfa 4: 3/5/1; df673a83 4: 2/1/2 (full table
in the kernel log).

Reading. (1) **The division term is worth 0.014 of local score at
REF** (0.1 x 0.141), with recall 43% and precision 17%: five counted
false forks per true one. If every false fork vanished at constant
TP, div_J would be 0.43 (+0.029 of score); if every lost division
were found at constant FP, 0.33 (+0.019). The FP side is the larger
lever and it is spread thin: 45 division-free movies carry counted
false forks. (2) **The orphan-fork patch buys +0.0030 of div term and
costs −0.0009 of adj_J, net +0.0021**: it adds 29 true forks and 157
false ones (16% precision), on top of the ILP's own 25 / 101. The
user's remark that most of our divisions come from the orphan patch
holds: 29 of 54. (3) **The geometry is at a local optimum in both
directions.** The notebook's tighter rule loses 19 TP to shed 95 FP
(−0.0195); the divergence test relaxed to 1.0 / 0 um adds ~200 FP and
LOSES 8-17 TP, because the extra proposals compete for the same
orphans and the per-mother / per-orphan / cap slots, so a nearer false
mother now claims the true second daughter first. divsweep's n = 12
loss at 1.0 um was the real sign. divgate2's "0% of real divisions
pass 2.25 um" was measured on the divisions that REACHED that test
(33 zh, 1, 1); on 125 competition divisions the test is doing
precision work that outweighs what it refuses. (4) The two folds
disagree by 0.065 of div_J: a division rule needs BOTH folds up to be
believed, and a sub-0.02 div_J change is inside the bench's own
noise.

Tested claims: the notebook's tighter division geometry transplants
(killed, −0.0195 div_J on 125); the divergence threshold is what
refuses real divisions (killed as a lever: relaxing it loses TP and
adds 200 FP); the patch is worth nothing (killed: +0.002 score, 29
of 54 TP). Residue: (a) the counted false forks -- 258 for 54 TP --
are the lever, and div-gate's 2026-08-24 lesson applies (a gate must
be measured on the forks the pipeline actually emits; that run had
23 removable FPs and found the kinematics model anti-informative on
them); this bench has 312 counted forks, out-of-fold, with the fork's
own features available at fork time (ranker p for mother->orphan and
mother->child, the head's forward probability, the distances,
divergence, detection scores, density); (b) the 71 lost divisions
have never been walked through the patch's conditions on competition
data at this size (divgate2 did it on 2 + 2 + 96 zh); (c) 44b6 (26
divisions, 15% of weight, local div_J 0.0000) is not on the bench.

### edgecmp (2026-09-05, landed) + submit-f48 (ref 56030644 = LB 0.933, NEW BEST, −1.43% bytes): edge by edge, the notebook is 0.026 BELOW us on the visible 4 and its 24% fewer rows sit where no annotation is -- its margin is the signed, unclamped node term; no edge-level merge helps

The user, after ilpsweep and the "why dense helps, sparse hurts"
answer: "都去做吧，我脑子有点乱". Three things were proposed; two are
pushed now, the third (a consensus submission) waits on the first.

**edgecmp (`biohub-edgecmp` v1, CPU, `edgecmp.py` = the gap2 harness
preamble + a comparison body, scratchpad edgecmp_body.py).** Ref
55974245's submission.csv (biohub-submit-ep v1, LB 0.932) against the
public notebook fork's (biohub-nb948-asis v1, LB 0.935) on the 4
visible movies. Their nodes matched to ours per frame (mutual nearest
within 5 um), their edges mapped through the match, and six graphs
scored by the official scorer per movie and weighted as the scorer
weights: ours, theirs, both (our nodes, edges present in both),
ours_only, theirs_only (their nodes), union (our nodes + their
unmatched nodes, all edges). Read: precision tp/(tp+fp) of ours_only
vs theirs_only vs both; and, for the nodes only one side has, the
share sitting within 5 um of an annotated GT cell. Caveat stated in
the script: both pipelines are in-sample on these 4 (train copies; 3
of the 4 are in our D175 pool), so absolutes are optimistic and only
bucket-vs-bucket is read. Pre-registered reading: ours_only precision
below ~0.48 = the intersection beats ours and a consensus submission
is built (both pipelines in one kernel, 1.4 h + 6 h); theirs_only
precision well above 0.48 = the union is the candidate; both buckets
near or above 0.9 = the two disagree on right edges of different
cells and no edge-level merge helps.

**edgecmp landed (4.2 min).** Six graphs, official scorer, the 4
movies weighted as the scorer weights (tp+fp+fn):

| graph | nodes | edges | tp | fp | fn | precision | adj_J | vs ours |
|---|---|---|---|---|---|---|---|---|
| ours (ref 55974245) | 126,488 | 121,593 | 2,044 | 97 | 83 | 0.955 | 0.9214 | — |
| theirs (the notebook fork) | 119,517 | 115,354 | 2,028 | 146 | 99 | 0.933 | 0.8952 | −0.0262 |
| both (our nodes, edges in both) | 126,488 | 98,925 | 1,976 | 74 | 151 | 0.964 | 0.9000 | −0.0213 |
| ours_only | 126,488 | 22,668 | 68 | 23 | — | 0.747 | — | — |
| theirs_only | 119,517 | 16,429 | 61 | 76 | — | 0.445 | — | — |
| union | 139,572 | 138,022 | 2,065 | 171 | 62 | 0.924 | 0.8940 | −0.0274 |

Per movie (ours / theirs): 44b6_0113de3b 24,416 / 25,484 nodes, 94%
of their nodes match ours, GT edges ~50, ours adj_J 0.9461 (node_r
−0.052) vs theirs 0.8688 (−0.011); **44b6_0b24845f 27,424 / 18,618,
only 13,101 matched (48% of ours); our 14,323 unmatched nodes: 0 sit
on an annotated cell; their 5,517 unmatched: 0**; GT edges ~50; ours
0.9167 (edge_J 0.902, node_r −0.164) vs theirs **0.9830 (edge_J 0.942,
node_r −0.432: 43% BELOW the estimate, and the signed, unclamped node
term multiplies its J by 1.043)**; 6bba_05b6850b 6,212 / 6,056, our 586
unmatched nodes 1.9% on annotated cells, theirs 3.5%; ours 0.9918 vs
0.9607; 6bba_05db0fb1 68,436 / 69,359, unmatched 0.5% / 0.9% on
annotated cells; ours 0.8735 vs 0.8492.

Reading, three parts.

1. **No edge-level merge helps; the consensus submission is not
   built (its pre-registered condition fails both ways).** ours_only
   precision 0.747 (above the 0.48 break-even: dropping those edges
   loses more than it sheds -- both −0.0213); theirs_only 0.445 (at
   break-even, and with 76 false edges: union −0.0274). Their in-
   sample advantage on these 4 (their detector and head trained on
   them) makes their buckets look BETTER here than they would on the
   hidden set, so the union would be worse there, not better. Closed.
2. **The scorer sees 1.8% of our edges on these movies, 0.2% on the
   44b6 pair.** 2,140 counted edges out of 121,593 (44b6_0b24845f: 51
   of 25,514). The 24% of rows the notebook does not carry are rows
   the scorer never looks at: on 44b6_0b24845f every one of the
   14,323 nodes only we have, and every one of the 5,517 only they
   have, sits away from any annotated cell. Deleting them changes no
   counted edge -- only the node term.
3. **Where the notebook's 24% pays: the node term, and only there.**
   The penalty is J × (1 − 0.1 × (N_pred − N_true)/N_true), signed,
   clamped at 0 below and NOT above (metrics.py line 362, reference.md
   "no upper clamp"): every 10% of nodes below the estimate is +1% of
   J. On 44b6_0b24845f the notebook is 43% below the estimate and
   collects ×1.043 = +0.04 on that movie, ours ×1.016. On the visible
   4 as a whole the notebook is 0.026 BELOW us (0.8952 vs 0.9214) while
   being in-sample; on the hidden set it is 0.003 above us with 24%
   fewer rows. If the hidden set is out-of-sample for its head the
   way 44b6_0b24845f behaves for it (the funnel: its ILP deletes 39%
   there, and its hidden bytes say it deletes >27% of the official
   nodes everywhere), then its hidden node_r is roughly −0.25 to
   −0.30 against ours near −0.05, worth about +0.02 to +0.025 of
   adj_J from the node term alone -- which is the size of its edge
   deficit against us here. The arithmetic closes: worse edges by
   ~0.02, node term +0.025, net +0.003. This is inference from four
   in-sample movies and one byte count, not a measurement of the
   hidden set.

What this says about "our rejection has room": the room is in the
node term, and a deletion pays only if the deleted nodes carry
counted edges at less than a tenth of the average rate (random
deletion of x of the nodes costs x of J and returns 0.1x). The
notebook's ILP finds such nodes on the hidden set because its head
is unsure there; ours is in-sample locally, so ilpsweep could not
see it (D20F48 on 44b6: 15% of nodes deleted, 2% of TP lost = a
depletion of 7.7x, just short of paying). The one venue that can
price it is LB, and that is what submit-f48 is (bytes, then score).
The user's reading of the node term (2026-09-05), which corrects my
"loophole" framing: it is unclamped below by design -- deleting nodes
already carries the risk of deleting tracked cells (their edges
become FN), so no clamp was needed, and the term's purpose is to
punish node spam and wild linking, which the edge term cannot see
when the spam sits off the annotation. What the numbers add to that:
the self-penalty the design relies on exists only inside the
annotation (every deletion costs on zh; on 44b6 it costs for 0.2% of
nodes), so on sparse movies a deletion outside the annotated tracks
is riskless and paid 0.1 per unit, and that is what the notebook
collects, by accident. Residue: (a) the organisers may still clamp
it (the hub-node exploit was patched 2026-07-17); anything built on
it is exposed to that; (b) a deletion rule aimed at unannotated
regions (dense regions, weak chains, low detection scores) is
testable locally only through the node term, never through edges,
since local GT is the same sparse annotation.

**submit-f48 (`biohub-submit-f48` v1, GPU T4, `submit_f48.py` =
`submit_ep.py` with ILP_FLOOR 0.10 -> 0.48, SCRIPT_VERSION
2026-09-05a-submit-f48, everything else byte-identical to ref
55974245).** This is NOT a three-domain winner (ilpsweep F48: 6bba
+0.0000, 44b6 −0.0131, zh +0.0025, 7 of 8 up); I said so and the user
said do all three. What it prices: the hidden set is out-of-sample
for the official head whose probabilities feed our ranker, the
visible / local groups are not, and zh (the one out-of-sample venue)
is the one place the floor helped. Two readings, in order: bytes
(expected to fall by 5-10% of 270,137,616: the floor removed 4.8% /
12.4% / 3.3% of our nodes locally before patches); then LB vs 0.932.
Bytes down and LB ≥ 0.932 = the rejection line is open at LB and
local cannot see it (then D20F48 is the next probe); LB below 0.932
= closed at every venue, and "our rejection has room" is not the ILP.

Kernel v1 landed (19 min). Visible-4: nodes 120,411 / edges 114,595 /
rows 235,006 against ref 55974245's 126,488 / 121,593 / 248,081, i.e.
−4.8% nodes, −5.8% edges, on the visible movies. Per movie nodes
24,390 / 22,115 / 6,146 / 67,760 (ref: 24,416 / 27,424 / 6,212 /
68,436; the floor bites on 44b6_0b24845f, −19%, the movie where the
scorer sees 51 edges). **Submitted as ref 56030644 (2026-09-05)**,
its own control ref 55974245 = 0.932 / 270,137,616 bytes. Reading as
pre-registered above: bytes first, then LB.

**LB 0.933, total_bytes 266,263,495 (scored 2026-09-05, ~18 h after
submission). +0.001 over its control, −1.43% output. NEW BEST.**

Reading. (1) Bytes: the floor removed 1.43% of our hidden output
against 4.8% of the visible-4 nodes (and 19% on 44b6_0b24845f). So
the hidden set is NOT a 44b6_0b24845f for our linker: our ranker's
probabilities are as polarised there as locally, and the chains
under 0.48 are few. The pre-registered hypothesis "our hidden
output carries weak chains the scorer does not need" is killed at
the scale that mattered: there is ~1.4% of it, not 24%. The
notebook's 24% comes from ITS head being unsure out-of-sample; our
ranker is not, and the deletion that made its file small does not
exist in ours. (2) Score: +0.001 is inside the LB's step, in the
direction the local zh sign (+0.0025) and the byte pattern
predicted, and against the local 44b6 sign (−0.0131 at 15% weight
= −0.002 expected). Read with the record's eleven byte-LB pairs it
is the sixth point on the same line: −0.22% +0.007, −1.43% +0.001,
and every addition of bytes ≤ 0. (3) It is the new reference. The
pipeline is ref 55974245 with ILP_FLOOR 0.48 (`submit_f48.py`,
SCRIPT_VERSION 2026-09-05a-submit-f48); every anchor and local table
in this record is still at floor 0.10 unless it says otherwise (the
ilpsweep F48 row IS this pipeline locally: 0.9261 / 0.8384 / 0.8143).

Tested claim: the byte gap to the notebook is weak chains in our
output that a stricter ILP would shed. Killed: 1.4% of the output
was. Residue: (a) the remaining 22% of rows between us and the
notebook are rows our ranker is confident about; if they are
"free" nodes off the annotation, the node term would pay for
deleting them, but no confidence-based rule finds them, and no
local venue can tell a confident right edge off the annotation
from a confident right edge on it; (b) the D20 arm (disappearance
2.0, +0.0013 6bba / −0.0029 44b6 / −0.0033 zh locally) is the one
neighbouring knob not priced at LB; on this evidence (44b6's local
sign did not hold on LB for F48) it is a plausible second +0.001,
not a line; (c) the byte pattern is now six for six and is the
strongest regularity in the record: on the hidden set, output we
remove has been worth more than output we add, every time.

### ilpsweep (2026-09-05, biohub-ilpsweep v1, 229 min, anchors exact, 160 of 160): the 0.948 notebook's ILP rejection on OUR nodes and linker -- CLOSED, signs disagree (+0.0002 / −0.0115 / −0.0010); the chains it deletes are 62% / 91% / 58% CORRECT where the scorer looks, and on our nodes it deletes only 6-15%, not 39%

The user, after the funnel and the "why does deleting that much not
delete right ones" answer (annotation 0.77% / 9.71%, edges among
unannotated cells enter no count, node term signed and uncapped,
whole-chain deletion costs FN only): "去吧". `ilpsweep.py` =
`gap2_rank.py` with the ILP's disappearance cost and candidate floor
as the swept variable (make_ilpsweep.py in the scratchpad; solve_one
takes disapp / floor; one solve per arm per dataset; the 3-frame
bridge code is inert, patch_final runs as REF). Everything else is the
gap2strict harness: F4 nodes, +BIDIR linker on 6bba / 44b6, D187 on
zh, hard anchor ANCHOR_D187 F4-6bba 0.9216, soft REF 0.9261 / 0.8515 /
0.8118, official scorer weighted as adj_J is, paired per-dataset
deltas, three-domain rule. Arms (disappearance, floor): REF (1.5,
0.10) = production; D10 (1.0, 0.10) the loosening direction; D20 (2.0,
0.10); F48 (1.5, 0.48); D20F48 (2.0, 0.48) = the notebook's ILP. New
table: per arm and domain, F4 detections in, nodes and edges out of
the ILP (its rejection, before patches; the notebook's is −15% of raw
on the visible 4), and the scorer's edge TP / FP / FN with deltas vs
REF, so the verdict says whether a deletion shed wrong edges (d_fp)
or lost right ones (d_fn) and what the node term did.

Pre-registered reading: D20F48 positive in 6bba AND 44b6 = the
notebook's rejection transplants, submit it (F4 + D20F48, everything
else as ref 55974245); positive in one, negative in the other = the
usual disagreement, record; negative in both with d_fn >> −d_fp = our
chains of 2-3 nodes are mostly right where the scorer looks, and the
hidden-embryo room is in nodes the scorer does not look at (only the
node term can pay there, at 0.1 × the surplus fraction); D10 positive
= the local optimum is looser than production, the opposite of the
byte story. Expected runtime ~3.5 h CPU (5 solves + anchor per
dataset; rank2's 7 solves took 198 min).

**Landed (229 min; ANCHOR_D187 0.9216 exact, REF 0.9261 / 0.8515 /
0.8118 exact).** adj_J vs REF, group / per-dataset up-down:

| arm (disapp, floor) | F4-6bba | F4-44b6 | F4-zh | verdict |
|---|---|---|---|---|
| D10 (1.0, 0.10) | −0.0017 (0 up 12 down) | −0.0057 (2 up 10 down) | +0.0020 (7 up 1 down) | 6bba+44b6 NEGATIVE |
| D20 (2.0, 0.10) | +0.0013 (9 up 2 down) | −0.0029 (8 up 4 down) | −0.0033 (2 up 6 down) | signs DISAGREE |
| F48 (1.5, 0.48) | +0.0000 (6 up 6 down) | −0.0131 (4 up 8 down) | +0.0025 (7 up 1 down) | signs DISAGREE |
| D20F48 (2.0, 0.48) = the notebook's ILP | +0.0002 (7 up 5 down) | −0.0115 (6 up 6 down) | −0.0010 (4 up 4 down) | signs DISAGREE |

**The ILP's rejection on our nodes, and what the deleted chains were
(edge TP / FP / FN of the finished graph vs REF):**

| arm | nodes out of the ILP vs F4 detections in, 6bba / 44b6 / zh | d_tp / d_fp / d_fn, 6bba | 44b6 | zh |
|---|---|---|---|---|
| REF | −3.7% / −7.9% / −3.2% (of 171,450 / 354,086 / 216,274) | 0 / 0 / 0 (9,861 / 357 / 452) | 0 / 0 / 0 (2,630 / 233 / 260) | 0 / 0 / 0 (182,582 / 5,440 / 38,658) |
| D10 | −3.2% / −6.5% / −2.9% | −6 / +10 / +6 | −7 / +10 / +7 | +467 / −51 / −467 |
| D20 | −4.9% / −9.9% / −4.1% | +1 / −7 / −1 | −12 / 0 / +12 | −868 / −49 / +868 |
| F48 | −4.8% / −12.4% / −3.3% | −26 / −16 / +26 | −45 / +5 / +45 | +111 / −554 / −111 |
| D20F48 | −6.4% / −15.4% / −4.3% | −33 / −20 / +33 | −52 / −5 / +52 | −844 / −618 / +844 |

node_r under D20F48: 6bba −0.020 -> −0.037, 44b6 −0.111 -> −0.162, zh
−0.082 -> −0.089 (all further BELOW the estimate; the signed node
term pays +0.0017 / +0.004 / +0.0006 of adj_J for it, and that bonus
is the whole of 6bba's +0.0002).

Reading. (1) **The notebook's ILP settings on our nodes do not
transplant**: −0.0115 on 44b6 with the 6bba side flat and the zh
side −0.0010. (2) **The chains it deletes are mostly right where the
scorer looks**: of the counted edges D20F48 removes, 33 of 53 on 6bba
(62%), 52 of 57 on 44b6 (91%), 844 of 1,462 on zh (58%) were true
edges; the break-even for a deletion is a precision below J/(1+J) ≈
0.48, and none of the three is near it. Even the candidate floor
alone (F48) loses 45 true edges for 5 false ones on 44b6. (3) **On
our nodes their ILP deletes 6% / 15% / 4%, not 39%**: the 39% on
44b6_0b24845f was a property of the official detector + official
head (weak, unpolarised probabilities, solution median 0.84); our
ranker scores true edges ≥ 0.9 for 92% / 79% of them (STAGE A), so a
Σp > 2 rule finds few weak chains to drop. The deletion that made
the notebook's file a quarter smaller is not available on our stack
by turning the same knob. (4) **REF is the local optimum of this
knob in both clean groups**: looser (D10) loses 12 of 12 on 6bba and
10 of 12 on 44b6 (+10 false edges for −6 / −7 true ones); tighter
loses 44b6. The one arm that gains anywhere clean is D20 on 6bba
(+0.0013, 9 of 12 up, the deletion there sheds 7 false edges for 1
true one), and it costs 44b6 −0.0029 and zh −0.0033. (5) zh's sign
is the opposite of 44b6's on the floor (F48 +0.0025, 7 of 8 up: 554
false edges shed AND 111 true edges GAINED there -- the weak wrong
edges were occupying nodes whose right edge the ILP could then place;
zh is the dense, machine-labelled venue where nearly every cell
counts). The share of TRUE edges among the counted edges a deletion
removes is monotone in the annotation fraction: 44b6 (0.77%) 91%,
6bba (9.7%) 62%, zh (97.5%) ~0% net. Two readings, not separable in
this run: sparse annotation is not random (it anticorrelates with
density, −0.20), so the counted cells in 44b6 / 6bba are the easy
places where a weak edge has no competitor and is usually right,
while the crowded places where weak edges are wrong are unannotated
and invisible, and zh shows them; or the D187 ranker is out-of-sample
on zh and its weak edges are simply worse there. The separating test
is F48 on zh under the zh-trained ranker (zhrank's pkls): removed
edges turning mostly true = the ranker; staying mostly false = the
annotation.

Tested claim: the 0.948 notebook's byte-saving rejection (its ILP's
disappearance 2 / floor 0.48) transplants to our nodes. Killed in
44b6 and zh, flat in 6bba. Residue: (a) "our rejection still has
room" is not the ILP knob, in either direction; (b) locally we are
below the node estimate in every domain, so the node term rewards
deletion here, and it still did not pay in edges -- on the hidden
embryo the node term's sign is the one unknown that could change
the reading, and only bytes-vs-LB can price it; (c) D20 on 6bba
alone (+0.0013, 9 of 12) is the only clean positive in the sweep and
is not a submission by the three-domain rule; (d) where our extra
24% of hidden rows sit -- nodes the scorer never looks at, or right
edges -- is still not measurable locally, and the local proxy (node_r
−0.02 / −0.11) says we are not over-producing where GT exists.

### ⏳ submit_off v2 / SHORT_MIN 6 (pushed 2026-09-04, pre-registered, submission ON HOLD): the over-detect-then-delete interaction on the HIDDEN embryo, the one venue where the official detector over-detects

The user, 2026-09-04, after zhoffrank: "太奇怪了。。。我完全不知道该怎么办呢，你提交一下
这个试试吧，看看最终文件大小". `submit_off.py` SHORT_MIN 4 -> 6
(SCRIPT_VERSION 2026-09-04b-submit-off-s6), everything else byte-
identical to v1 = ref 56000581 = LB 0.895 (280,141,150 bytes, +3.70%
over the F4 pipeline's 270,137,616 while the visible-4 had 3% FEWER
nodes). Kernel `biohub-submit-off` v2. What it tests: on the hidden
embryo the official detector produces MORE output than F4 does, i.e.
there it over-detects (on zebrahub it under-detects by 27-41%, so the
interaction could not be tested there); the 0.948 notebook deletes
with min track 6 + prune isolated; our v1 deleted with 4. Two readings
in order: (1) bytes -- the deletion should show as −3% to −6% of
output before any score is known; if the bytes do not fall, the
surplus is not in short fragments and the mechanism is dead on
arrival; (2) LB vs 0.895 -- a rise of ≥ +0.02 says the deletion rule
is a large part of what the notebook's post-processing buys on the
official detector's nodes and reopens "official nodes + the notebook's
whole deletion recipe"; +0.005 to +0.02 says the interaction exists but
is not the 0.053; ≤ 0 says the hidden-embryo surplus is not short
fragments and the official-detector line is closed at every venue,
with the residue that 0.948's margin is not decomposable into any part
we can run on our stack.

Kernel v2 landed (21 min, 0 fallbacks). Visible-4 under SHORT_MIN 6:
nodes 121,524 / edges 117,506 / rows 239,030 against v1's 124,334 /
119,703 / 244,037 -- the deletion removes 2.3% of the official nodes on
the visible movies (F4's own visible-4 was 128,157 nodes at ref
55889500). Submission ON HOLD; if it ever goes, the hidden bytes should
read about −2% of 280,141,150.

### zhoff / zhoffrank (2026-09-04): the official detector's nodes through OUR linker on Zebrahub -- CLOSED: −0.32 to −0.42 adj_J on all 8 crops under both deletion rules; at its production gate the official detector UNDER-detects zebrahub by 27-41%, so the over-detect-then-delete interaction cannot even be tested there

The user, 2026-09-04: "难道它的检测真的是神？实在想不到别的了" -> "去吧". The
ledger that motivates it: official nodes + official head + OUR ILP + our
patches = LB 0.895; the same nodes + the notebook's linker + its patches
= 0.948; yet on OUR nodes its linker loses 0.010 / 0.033 (motionlink),
its 3-frame bridge is zero (gap2strict), its division rule is weaker
(motionlink div_J 0.054 vs 0.152) and its edge probabilities are LESS
polarised than our ranker's. Every part alone is worse or zero; together
on its own nodes they are worth +0.053. So the 0.05 lives in the
detector × post-processing interaction, and the candidate mechanism
is: an over-detecting detector (zh, detfuse3: the dual-seed blend at
its best gate +37% detections, precision 0.59, F1 0.696 vs F4 0.790)
paired with post-processing built to delete (one-to-one linking, prune
isolated, min track 6 with rescue). Detection F1 is measured before
anything is deleted; the pipeline-level comparison on official nodes
has only ever been run where the detector is in-sample (offdet) or at
LB with OUR deletion rule (SHORT_MIN 4: 0.895, +3.7% bytes).

`zhoff_cache.py` -> `biohub-zhoff` v1 (GPU T4, no internet; sources:
pack, seed-314159 pack, biohub-zebrahub-extract): for fusescore's 8
zebrahub eval crops (sorted zhdet stems [:8], asserted), predict_video's
detection path on the crop's [::4] strided volume: crop-level 0.001 /
0.999 quantiles (detfuse3 used per-frame; the official path uses the
dataset's), the 2-frame window in which the frame is first seen,
4-view flip TTA, `_detect_cells_pooled` with the 3 um kernel (3, 3, 3),
sigmoid at the peak, coords × (1, 4, 4). Arms OFFP (primary) and OFF28
(the notebook's blend: secondary logits re-centred to the primary's
per-frame mean, std ratio clamped 0.5-2, 0.2 / 0.8, retention guard
0.90 -> primary alone on frames where the blend keeps < 90% of the
primary's peaks), gates 0.965 / 0.99. Output zhoff_<ARM><gate>_<crop>.npz
in the detcache layout.

`zhoffrank.py` -> `biohub-zhoffrank` (CPU; generated from gap2_rank.py
by make_zhoffrank.py; pushed when zhoff lands): per crop, detection
arms F4 / OFFP965 / OFFP990 / OFF28965 / OFF28990 through the D187
linker (18 columns; no head cache on zh), one ILP solve each, then
the patches at SHORT_MIN 4 (ships) and 6 (the notebook's) on copies;
scored on the crop's tracks as fusescore scored zh. Hard anchor: F4 ×
4 = 0.8118. Printed per variant: adj_J, edge_J, node_r, div_J, raw
nodes per frame, final nodes and edges, paired per-crop deltas vs F4 × 4.

Pre-registered reading: official nodes at SHORT_MIN 6 within 0.01 of
F4 at 4, or above, on 6 of 8 crops = the over-detect-then-delete
interaction is real and the detector is where the notebook's margin
comes from (then a submission with official nodes + SHORT_MIN 6 is the
LB test); official nodes ≥ 0.03 below F4 on most crops under both
rules = the detector is not better on unseen data under any deletion
rule we have, and the remaining explanation of 0.948 is that the
hidden embryo resembles the training embryos (where the official
detector recalls 99-100% of annotated cells, in-sample) more than
Zebrahub does -- an explanation no local test can reach. Caveat
carried: the D187 linker here is not the notebook's linker; the
interaction under test is detector × deletion rule, not detector × its
whole pipeline.

GPU half landed (biohub-zhoff v1, 14.3 min, ZHOFF OK 32 of 32). Official
nodes per frame (OFFP 0.965 / 0.99, OFF28 0.965 / 0.99) per crop:
tail_z2y0x3 410 / 356 / 423 / 389; tail_z2y1x2 328 / 270 / 316 / 273;
tail_z2y2x3 130 / 106 / 129 / 105; tail_z2y3x0 210 / 164 / 209 / 163;
tail_z2y3x2 221 / 154 / 220 / 154; z2y1x4 93 / 76 / 101 / 93; z2y1x5
108 / 87 / 132 / 120; z2y1x6 77 / 68 / 82 / 76. The retention guard
fell back to the primary on 0-3 frames of the four bright crops and on
62-99 of 100 frames of the three dim tail crops (q_high 117-236), so
on those the "dual-seed blend" IS the primary. Crop-level q_low is 0
everywhere (zero padding at the crop edge).

### zhoffrank results (v1, 67.6 min CPU, 2026-09-04): ZHOFFRANK OK, 80 of 80, anchor exact

F4 × SHORT_MIN 4 = 0.8118 / div_J 0.0215 (hard, exact). Est nodes = GT
nodes; GT nodes per frame per crop 475 / 392 / 192 / 459 / 562 / 51 /
96 / 53 (mean 285); F4 puts down 270 raw per frame, the official
detector 197 (0.965) / 160 (0.99), the blend 202 / 172.

| det | SHORT_MIN | adj_J | edge_J | node_r | div_J | raw/fr | final nodes | vs F4 × 4 | per-crop |
|---|---|---|---|---|---|---|---|---|---|
| **F4** | **4** | **0.8118** | 0.8055 | −0.082 | 0.0215 | 270 | 209,271 | — | — |
| F4 | 6 | 0.8090 | 0.8020 | −0.090 | 0.0209 | 270 | 207,465 | −0.0028 | 2 up / 6 down |
| OFFP 0.965 | 4 | 0.4936 | 0.4794 | −0.392 | 0.0054 | 197 | 138,774 | −0.3182 | 0 / 8 |
| OFFP 0.965 | 6 | 0.4852 | 0.4704 | −0.411 | 0.0048 | 197 | 134,355 | −0.3266 | 0 / 8 |
| OFFP 0.99 | 4 | 0.4011 | 0.3854 | −0.513 | 0.0054 | 160 | 110,978 | −0.4107 | 0 / 8 |
| OFFP 0.99 | 6 | 0.3911 | 0.3751 | −0.533 | 0.0048 | 160 | 106,503 | −0.4207 | 0 / 8 |
| OFF28 0.965 | 4 | 0.4821 | 0.4695 | −0.375 | 0.0094 | 202 | 142,251 | −0.3297 | 0 / 8 |
| OFF28 0.965 | 6 | 0.4737 | 0.4606 | −0.396 | 0.0088 | 202 | 137,639 | −0.3381 | 0 / 8 |
| OFF28 0.99 | 4 | 0.4033 | 0.3903 | −0.475 | 0.0088 | 172 | 119,462 | −0.4085 | 0 / 8 |
| OFF28 0.99 | 6 | 0.3944 | 0.3812 | −0.495 | 0.0082 | 172 | 115,030 | −0.4174 | 0 / 8 |

Per crop (F4/4 vs OFFP965/4): the bright crops lose 0.07-0.19
(z2y1x4 0.8139 -> 0.7405; z2y1x6 0.7616 -> 0.6964; tail_z2y0x3 0.8436
-> 0.6916), the dim tail crops lose 0.24-0.51 (tail_z2y3x0 0.8419 ->
0.3528; tail_z2y3x2 0.7772 -> 0.2695). The blend does not help (it IS
the primary on the dim crops, retention guard), and SHORT_MIN 6 costs
0.003 on F4 and 0.008-0.010 on official nodes: deleting more never
paid, on any node set.

**Tested claim, killed: "the 0.948 notebook's margin is an
over-detecting detector paired with post-processing that deletes."**
On zebrahub the official detector at its production gates does not
over-detect at all: it puts down 27% (0.965) to 41% (0.99) FEWER nodes
than F4 and node_r sits at −0.39 to −0.53 against GT. detfuse3's "+37%
detections" was at gate 0.15 on the saturated sigmoid; at 0.965 the
same maps are starved, worst on dim data (q_high 117-236 vs 219-276
for the bright crops). There is nothing for a deletion rule to delete.

**What this does and does not say about 0.948.** It says the official
detector, under any deletion rule we have, is far worse than F4 on the
one venue no detector trained on -- the fourth clean loss for that
detector (zh F1, LB 0.898, LB 0.895, now zh adj_J −0.32). It does NOT
test the interaction on the hidden embryo, where the same detector
does the opposite of what it does on zebrahub: at LB (submit_off) it
produced +3.7% more output than F4 while the visible-4 had 3% fewer
nodes, i.e. on the hidden embryo it over-detects. So the venue that
can show over-detection (hidden) is the one we cannot score, and the
venue we can score (zebrahub) shows under-detection. Residue: (a) the
over-detect-then-delete interaction on the HIDDEN embryo remains
testable only at LB: submit_off with SHORT_MIN 6 (and prune isolated),
one submission; the byte count would say whether the deletion bit
(expect −3..−5% bytes) before the score does; (b) zebrahub is a
brightness regime the official detector's saturated sigmoid cannot
handle at 0.965 (the dim tail crops: q_high 117-159, adj_J 0.27-0.35),
while the competition's embryos and, per the bytes, the hidden one
are within its regime -- so for DETECTOR questions zebrahub
overstates the domain shift relative to the hidden test; for linking
and post-processing questions, which are geometric, it remains the
honest denominator; (c) the notebook's dual-seed blend degenerates to
the primary on dim data through its own retention guard, so whatever
the second seed buys, it buys only on bright data.

### gap2strict (2026-09-04, biohub-gap2 v3): the 0.948 notebook's 3-frame bridge under its own conditions, on our linker -- CLOSED, zero: +0.0005 / −0.0017 / +0.0014, signs disagree, 149 / 560 / 204 bridges in all; and the head's probabilities are LESS polarised than our ranker's

The user, 2026-09-04: "可是我们的补桥其实也允许直接补，只是认为候选点的方式准确率会
更高一点。再去试一下三帧补桥吧，看看它们的笔记怎么操作". What the record
has: span2 (2026-08-28) = the same t -> t+3 bridge with two
interpolated nodes and no condition but distance, 35,711 bridges:
6bba −0.0014 (edge_J unchanged, nodes +1.4%, div_J −0.0173, wrong
bridges eating second daughters), 44b6 +0.0066, zh +0.0068 -- signs
disagreed, discarded. The notebook's gap2 (its source, read this
session) adds four conditions the flood never had: total ≤ 10.2 um and
step ≤ 4.4 um; direction must agree with the track's previous OR next
step (cos > −0.25, |Δstep| ≤ 6 um), else a penalty; proposals ranked
by distance + 2·penalty; caps 0.45% of edges / 180 per movie and 0.6%
of a frame's ends. It runs after the one-frame gap close and before
the short-track filter.

`gap2_rank.py` -> `biohub-gap2` v3 (CPU; generated from
edgeprob_rank2.py by make_gap2.py; sources as zhrank). ONE solve per
dataset (linker +BIDIR loaded on F4-6bba / F4-44b6, ANCHOR_D187 on
F4-zh where no head columns exist; ANCHOR_D187 solved again on 6bba as
the hard anchor 0.9216), four post-link variants on copies: REF (as
shipped; soft 0.9261 / 0.8515 / zh 0.8118), GAP2_NB (the rule
verbatim, interpolated nodes; no intensity refinement, no image in
the harness), GAP2_POOL (each interpolated node replaced by the
nearest sub-gate candidate within 3.2 um when one exists, our
_gap_close's mechanism), GAP2_NOCTX (without the direction condition).
Bookkeeping per variant: candidates, context-rejected, selected,
cap-skipped, pool-used, nodes added. Also printed, for the user's
"好的地方更好，差的地方更差" question: on the labelled eval rows, the share
of true edges ≥ 0.9 and of false edges ≤ 0.1 for h_fused, p_fwd, the
+BIDIR ranker and D187.

Pre-registered reading: a GAP2 variant positive in 6bba AND 44b6 (zh
the third vote) with 6bba's div_J not below REF's is a transplant
worth a submission; NOCTX vs NB prices the direction condition; POOL
vs NB prices measured over interpolated positions. Negative or
disagreeing = span2's verdict stands under the strict rule too.
Expected ~1.5 h CPU.

### gap2strict results (v3, 103.2 min CPU, 2026-09-04): GAP2 OK, 128 of 128, anchors exact

ANCHOR_D187 F4-6bba 0.9216 / 0.0606 hard exact; REF (+BIDIR) 0.9261 /
0.8515 with div_J 0.1515 / 0.0000 soft exact; REF on zh (D187) 0.8118
= divwhy. One solve per dataset, four patch variants on copies.

| variant | F4-6bba | Δ | up/down | div_J 6bba | F4-44b6 | Δ | up/down | F4-zh | Δ | up/down |
|---|---|---|---|---|---|---|---|---|---|---|
| **REF** (ships) | **0.9261** | — | — | 0.1515 | **0.8515** | — | — | **0.8118** | — | — |
| GAP2_NB (the rule verbatim) | 0.9266 | +0.0005 | 3/8 (1 same) | 0.1562 | 0.8499 | −0.0017 | 1/11 | 0.8132 | +0.0014 | 6/2 |
| GAP2_POOL (sub-gate candidates as the nodes) | 0.9266 | +0.0005 | 3/8 | 0.1562 | 0.8499 | −0.0017 | 1/11 | 0.8132 | +0.0014 | 6/2 |
| GAP2_NOCTX (no direction condition) | 0.9266 | +0.0005 | 3/8 | 0.1562 | 0.8498 | −0.0017 | 1/11 | 0.8133 | +0.0015 | 6/2 |

Bookkeeping (candidates / context-rejected / selected / nodes added):
6bba 381 / 50 / 149 / 298; 44b6 1,462 / 270 / 560 / 1,120; zh 342 /
37 / 204 / 408. No variant hit the cap. span2 made 35,711 bridges on
the same 24 competition sets; the strict rule makes 709. The 44b6 loss
is 44b6_9bfa6a0a −0.0198 (0.8887 -> 0.8689) and 44b6_d2f34f90 −0.0038
against 44b6_a2bb48bb +0.0055; 6bba's +0.0005 is 6bba_7af54fde
+0.0032 and 6bba_283bf9f1 +0.0022 against nine movies at −0.0001 to
−0.0004. POOL replaced 52 / 381 / 65 nodes and changed no score to
four places (a 3.2 um move stays inside the 7 um match); NOCTX added
24 / 54 / 16 bridges and changed nothing either.

**Tested claim, killed: "the notebook's 3-frame bridge, under its own
conditions, recovers what span2's flood could not."** Under those
conditions it fires ~30 times per movie and moves the score by
±0.002 with the signs disagreeing; it is neither span2's loss nor a
gain, and it cannot be where 0.016 lives. Residue: (a) 6bba's div_J
goes UP (+0.0047) under the strict rule, so the user's 2026-08-27
point ("补桥消灭的是分裂错误的候选") extends to a strict 3-frame bridge
where it did not to the flood; (b) 44b6_9bfa6a0a loses 0.02 from a
handful of bridges (that movie sits at node_r −0.13; a bridge there
joins fragments the scorer wanted apart), the one place the rule does
damage; (c) measured vs interpolated bridge positions are
indistinguishable to the 7 um scorer, which the dcpool entry already
said for the one-frame bridge.

### Probability polarisation (the user's "好的地方更好，差的地方更差"), labelled F4 eval rows

| scorer | domain | n_true | n_false | AUC | true ≥ 0.9 | true < 0.5 | false ≤ 0.1 | false ≥ 0.5 |
|---|---|---|---|---|---|---|---|---|
| head h_fused | 6bba | 9,696 | 4,940 | 0.9919 | 0.724 | 0.019 | 0.821 | 0.053 |
| head p_fwd | 6bba | | | 0.9904 | 0.720 | 0.020 | 0.781 | 0.060 |
| **+BIDIR ranker** | 6bba | | | 0.9929 | **0.922** | 0.017 | **0.871** | 0.057 |
| D187 ranker | 6bba | | | 0.9894 | 0.909 | 0.020 | 0.846 | 0.062 |
| head h_fused | 44b6 | 2,544 | 5,586 | 0.9848 | 0.542 | 0.072 | 0.885 | 0.037 |
| head p_fwd | 44b6 | | | 0.9818 | 0.558 | 0.070 | 0.858 | 0.040 |
| **+BIDIR ranker** | 44b6 | | | 0.9871 | **0.787** | 0.061 | **0.906** | 0.040 |
| D187 ranker | 44b6 | | | 0.9816 | 0.645 | 0.071 | 0.880 | 0.042 |

**Killed: "the official head's probabilities are more polarised than
our ranker's and that is what its post-processing feeds on."** The
opposite: our ranker puts 92% / 79% of true edges above 0.9 where the
head puts 72% / 54%, and more false edges below 0.1. The head's
softmax over sources spreads a true parent's mass across neighbours
(the p1head "none" bucket). Both scorers leave the same 2% / 6-7% of
true edges below 0.5 -- the hard cases are the same cases.

### zhrank (2026-09-04): the edge ranker trained on Zebrahub, and two ranker mechanisms -- CLOSED: the Zebrahub ranker loses 0.010 / 0.012 on the competition embryos and wins only 0.0014 at home; re-normalising the shipped ranker's probabilities loses 0.002 / 0.006

The user, 2026-09-04: "还是很怕zebrahub训练出来的模型水土不服呢，不过退一万步来说，
去试试吧，顺便看看排序器本身有没有什么可以改变的机制". `zhrank.py` ->
`biohub-zhrank` v1 (CPU; generated from edgeprob_rank2.py by
make_zhrank.py; sources add biohub-zh-detect and biohub-edgeprob-rank2).

Question 1, training data. The competition ranker's rows exist only
where the source primary matched an ANNOTATED GT node with children
(exclusion negatives; 44b6 annotates 0.77% of cells). Zebrahub's 41
crops (biohub-zh-detect: xl detections + machine tracks on the
competition voxel grid, every frame, ~2.5% cells unlabelled) label
every in-gate non-child as a true negative. Prior: pool 32 -> 187
competition movies moved LB 0.921 -> 0.920 (more of the same, nothing);
div-train's Zebrahub-only classifier scored AUC 0.81 on held-out 6bba
(this kind of transfer exists). Arms, 18 columns (no head cache on
zebrahub): BASE18_D175 (soft anchors 0.9232 / 0.8346, rank v2),
BASE18_ZH (the 33 crops outside fusescore's 8), BASE18_MIX (both
pools). Judged on F4-6bba, F4-44b6 (transfer) AND F4-zh (the 8 crops,
in-domain; scored as fusescore scored zh, GT = the crop's own tracks).
No brightness filter on the machine labels (zhdet carries no image);
dim GT points our detector misses simply produce no row.

Question 2, mechanism. The ranker scores each pair alone (only
rank_of_j_for_i / n_cand_i / n_cand_j describe the competition) and the
ILP takes raw −p; the official head softmaxes over the sources of each
target, which is where h_fused's structure comes from. Arms on the
+BIDIR ranker that ships (loaded pkl): NORM = harmonic mean of
p / Σ over the source's candidates and p / Σ over the target's
candidates (pure competition; a lone candidate becomes 1.0), MULT =
sqrt(p × NORM). Reference +BIDIR (soft 0.9261 / 0.8515), ANCHOR_D187
hard 0.9216.

Pre-registered reading: (1) BASE18_ZH / MIX vs BASE18_D175, same sign
in 6bba AND 44b6; negative in both = the user's fear measured, the
line closes; ≥ +0.003 in both = dense negatives buy what 187 movies did
not, and the head columns on zebrahub (a GPU pass over the crops) are
the next step; the zh column must show the zebrahub ranker winning at
home, else it is a bug not a transfer result. (2) NORM / MULT vs
+BIDIR ≥ +0.003 in both groups = a mechanism worth a submission (one
line in link_ilp, no new model). Expected ~4.5 h CPU.

### zhrank results (v1, 184.1 min CPU, 2026-09-04): ZHRANK OK, 200 of 200, anchors exact

ANCHOR_D187 F4-6bba 0.9216 / 0.0606 hard exact; +BIDIR 0.9261 / 0.8515,
BASE18_D175 0.9232 / 0.8346, ANCHOR_D187 F4-44b6 0.8353 soft exact;
F4-zh ANCHOR_D187 0.8118 = divwhy's zh anchor, so the zebrahub scoring
path (GT = the crop's tracks, est nodes = GT count) is the recorded one.
Pools: D175 217,688 rows (0.497 positive); zebrahub 33 crops 1,817,824
rows (0.292 positive; the crops label 18k-139k pairs each); F4-zh eval
771,463 labelled rows. Training took 11 / 31 / 37 s.

Stage A, edge AUC (val / F4-6bba / F4-44b6 / F4-zh): BASE18_D175
0.9875 / 0.9898 / 0.9800 / 0.9960; BASE18_ZH 0.9804 / 0.9806 / 0.9779 /
0.9960; BASE18_MIX 0.9811 / 0.9887 / 0.9797 / 0.9964. On zebrahub
itself the D175 ranker and the zebrahub ranker rank pairs identically
(0.9960 both) -- 1.8 M dense rows taught nothing the 218 k sparse rows
had not.

Stage B, adj_J:

| arm | F4-6bba | vs +BIDIR | vs B18_D175 | F4-44b6 | vs +BIDIR | vs B18_D175 | F4-zh | vs B18_D175 | div_J 6bba |
|---|---|---|---|---|---|---|---|---|---|
| ANCHOR_D187 | 0.9216 | −0.0044 | −0.0016 | 0.8353 | −0.0163 | +0.0006 | 0.8118 | −0.0002 | 0.0606 |
| **+BIDIR** (ships) | **0.9261** | — | +0.0028 | **0.8515** | — | +0.0169 | — | — | 0.1515 |
| BASE18_D175 | 0.9232 | −0.0028 | — | 0.8346 | −0.0169 | — | 0.8120 | — | 0.0645 |
| BASE18_ZH (33 zebrahub crops) | 0.9131 | −0.0129 | **−0.0101** (1/11) | 0.8226 | −0.0290 | **−0.0120** (4/8) | 0.8133 | **+0.0014** (7/1) | 0.0833 |
| BASE18_MIX (D175 + zebrahub) | 0.9207 | −0.0054 | −0.0026 (3/9) | 0.8292 | −0.0223 | −0.0054 (6/6) | 0.8138 | +0.0018 (8/0) | 0.0690 |
| NORM (+BIDIR, competition-normalised) | 0.9243 | −0.0018 (2/10) | +0.0011 | 0.8449 | −0.0066 (5/7) | +0.0103 | — | — | 0.1163 |
| MULT (sqrt(p · NORM)) | 0.9246 | −0.0014 (2/10) | +0.0014 | 0.8456 | −0.0059 (4/8) | +0.0110 | — | — | 0.1389 |

(up/down = per-dataset counts against the named reference.)

**Question 1, killed: "a ranker trained on Zebrahub's dense labels
transfers to the competition embryos."** BASE18_ZH is −0.0101 /
−0.0120 against the same 18 columns trained on D175, 11 of 12 6bba
movies down, and at home it wins by 0.0014 (7 of 8 up) -- so it is not
a bug, it is a ranker that is only marginally better even on its own
domain and clearly worse off it. Mixing the pools (MIX) halves the loss
(−0.0026 / −0.0054) and keeps the tiny home gain; still negative in
both clean groups. The user's fear ("水土不服") is measured. Residue: (a)
the zebrahub rows are 0.29 positive vs D175's 0.50 -- the dense
negatives shift the ranker's calibration toward "no link", which is
what node_r shows (BASE18_ZH −0.0291 on 6bba vs −0.0207: more nodes
lost to the short filter) and what the ILP's disappearance cost 1.5
was not tuned for; a re-tuned floor / disappearance for the ZH ranker
was not run and would be the only way to separate calibration from
transfer. (b) The head columns on zebrahub (GPU pass) are not worth
building on this evidence: the 18-column transfer is the prerequisite
and it failed. (c) On zebrahub, D175's 218 k sparse rows already rank
as well as 1.8 M dense rows (AUC 0.9960 both), i.e. the competition's
exclusion negatives are not the bottleneck the pre-registration
supposed.

**Question 2, killed: "re-normalising the ranker's probability over
each source's and each target's competitors (the head's structure)
helps the ILP."** NORM −0.0018 / −0.0066, MULT −0.0014 / −0.0059,
both negative in both groups. The per-dataset picture is a
redistribution, not a shift: on 44b6 the per-dataset mean is +0.0005 /
+0.0006 with 44b6_0b24845f +0.0713 / +0.0739 (0.9167 -> 0.9881 /
0.9906) against 44b6_9bfa6a0a −0.0542 (0.8887 -> 0.8345) and
44b6_e29f0176 −0.0542 / −0.0162. Normalisation makes a lone candidate
1.0, so the ILP links more (node_r −0.0162 / −0.0985 vs −0.0199 /
−0.1113) and div_J falls (0.116 / 0.139 vs 0.152: the second daughter
loses its share of the parent's mass). Residue: 44b6_0b24845f has now
jumped +0.07 to +0.08 under two unrelated changes (p1head's
+BIDIR_P1H 0.9964, here 0.9881 / 0.9906) while every other movie
barely moves or falls -- that movie sits on a knife-edge of the ILP
that nobody has looked at; a per-movie diff of its +BIDIR vs NORM edge
sets would say what flips. Pkls edge_ranker_zh_b18_{d175,zh,mix}.pkl
are on the kernel and unused.

### motionlink (2026-09-04): the 0.948 notebook's motion relink as OUR final linker -- CLOSED: it loses 0.010 / 0.033 to the ranker + ILP, 12/12 and 11/12 datasets down; the linker is not where that notebook's margin comes from

The user, 2026-09-04, after reading the notebook's post-processing:
"真的要怀疑边排序器和ILP是不是在摸鱼". What the record has: ILP over the
official plain greedy +0.03 on 12 sets (2026-08-18, half of it the
node penalty), ranker over distance-Hungarian +0.005 (2026-08-23),
head columns +0.007 LB (edgeprob). What it does not have: a
VELOCITY-PREDICTED Hungarian as the final linker. The notebook's is
per frame pair, gates 6 then 10 um, cost = |target − (source + 0.5 v)|
+ 0.05 |target − source| − 1.0 p, p = the ILP edge's probability (0
where the ILP chose otherwise), so the ILP is a ≤ 1 um tie-breaker.
Our own pre-pass `_link_sim` is the same shape (pred = P + 0.5 v, gates
6 / 10, plus a score-similarity term) and links the right parent 97.4%
/ 91.8% of the time where the parent is detected (p1head, layer 1).

`motionlink_rank.py` -> `biohub-motionlink` v1 (CPU; generated from
edgeprob_rank2.py by make_motionlink.py). F4 primaries, the usual
patches after the linker. No pool, no training: +BIDIR loaded from
biohub-edgeprob-rank2's `edge_ranker_ep175v3_bidir.pkl`. Arms:
ANCHOR_D187 (hard 0.9216), +BIDIR (soft 0.9261 / 0.8515), HEAD1 (soft
0.9245 / 0.8453), MOTION0 (pure geometry, bonus 0), MOTION_H1 (bonus
1.0 × h_fused on every candidate pair), MOTION_H5 (bonus 5.0 ×
h_fused), MOTION_ILP (the notebook's exact structure: bonus 1.0 × the
+BIDIR ranker's probability on the pairs +BIDIR's ILP kept, 0
elsewhere). Also printed: how many of each motion arm's raw edges
coincide with +BIDIR's ILP edges. Paired per dataset vs +BIDIR, both
groups. Pre-registered reading: a MOTION arm within 0.003 of +BIDIR in
both groups = the ranker + ILP are not earning their place; ≥ 0.01
below in both groups = the linker is not where the notebook's margin
over our 0.895 official-detector run comes from (then detection blend /
post-processing / DeepCenter veto remain). The user's caveat, same
day: the 0.895 arm was official nodes + official head + OUR ILP, i.e.
the head may not transplant onto our stack, and the notebook's
combination may be a local optimum of its own parts; this test runs
their linker on OUR nodes, which is the other half of that question.

### motionlink results (v1, 106.2 min CPU, 2026-09-04): MOTIONLINK OK, 168 of 168, anchors exact

ANCHOR_D187 0.9216 / 0.0606 hard exact; +BIDIR (loaded pkl) 0.9261 /
0.8515 and HEAD1 0.9245 / 0.8453 soft exact. Features 3.1 min (no
pool); stage B 103 min (6bba_3abfe10a 1434 s, 44b6_a2bb48bb 1138 s for
seven arms).

| arm | F4-6bba adj_J | Δ | up/down | F4-44b6 adj_J | Δ | up/down | edge_J 6bba / 44b6 | node_r 6bba / 44b6 | div_J 6bba |
|---|---|---|---|---|---|---|---|---|---|
| **+BIDIR** (ships) | **0.9261** | — | — | **0.8515** | — | — | 0.9242 / 0.8421 | −0.0199 / −0.1113 | 0.1515 |
| HEAD1 | 0.9245 | −0.0016 | 6/6 | 0.8453 | −0.0062 | 6/6 | 0.9223 / 0.8339 | −0.0228 / −0.1371 | 0.2069 |
| MOTION0 (pure geometry) | 0.9143 | −0.0118 | 0/12 | 0.8174 | −0.0341 | 1/11 | 0.9138 / 0.8114 | −0.0040 / −0.0728 | 0.0541 |
| MOTION_H1 (+1.0·h_fused) | 0.9145 | −0.0116 | 0/12 | 0.8188 | −0.0327 | 1/11 | 0.9140 / 0.8128 | −0.0039 / −0.0730 | 0.0541 |
| MOTION_H5 (+5.0·h_fused) | 0.9161 | −0.0100 | 0/12 | 0.8217 | −0.0298 | 1/11 | 0.9156 / 0.8156 | −0.0042 / −0.0733 | 0.0556 |
| MOTION_ILP (the notebook's structure, our ILP as tie-breaker) | 0.9155 | −0.0105 | 0/12 | 0.8180 | −0.0335 | 1/11 | 0.9150 / 0.8119 | −0.0046 / −0.0735 | 0.0541 |

Per dataset the motion arms lose on every 6bba movie (min −0.0229,
max −0.0008) and on 11 of 12 44b6 (min −0.1091 on 44b6_e29f0176 0.8690
-> 0.7599; the one gain is 44b6_5740d24b +0.0022, the movie whose node
ratio is above 1). Raw edge sets before the patches: the motion arms
put down MORE edges than the ILP (6bba 159,545 vs 157,141; 44b6
321,063 vs 307,707) and 97.3% / 96.3% of the ILP's edges are also
motion edges. So the two linkers agree on 96-97% of links, and the
3-4% they disagree on -- links the ILP's disappearance cost 1.5
refused and the Hungarian made -- cost 0.010 on 6bba and 0.033 on
44b6. Those extra links also keep more nodes alive through the short
filter (node_r −0.004 vs −0.020 on 6bba), which the scorer does not
reward. The tie-breaker barely matters: the head at 1 um changes
0.0002, at 5 um 0.0018 / 0.0043, our ILP's probability 0.0013 /
−0.0006 -- consistent with p1head (the head as the pre-pass cost did
not help either). Divisions: one-to-one linking leaves the safe-division
rule to find a third of what the ILP finds (div_J 0.054 vs 0.152).

**Tested claim, killed: "the ranker + ILP are not earning their place
(摸鱼); a velocity-predicted Hungarian with the learned probability as
a tie-breaker would do as well."** It does 0.010 / 0.033 worse, on 23
of 24 movies. Under the pre-registration this is the second branch:
the linker is NOT where the 0.948 notebook's margin over our 0.895
official-detector run comes from -- on our nodes its linker is the
worse one. Residue: (a) what remains untested of that notebook is its
detection (dual-seed logit blend 0.20 / 0.80 at gate 0.965, minimum
candidate retention 0.90) and its post-processing (min track 6 with
rescue, 3-frame gap2, the stricter safe-division rule, the DeepCenter
veto -- on checkpoint_last.pt, which dcprobe found dead, so that veto
may be a no-op); (b) the user's point stands: 0.895 was official nodes
+ official head through OUR ILP, whose costs (floor 0.10,
disappearance 1.5, division 0.55) were tuned against the ranker's
calibration, not h_fused's; HEAD1 on F4 nodes loses only 0.0016 /
0.0062 locally, so the mis-calibration is not large on our nodes, but
on the official detector's denser node set it was never measured;
(c) the ILP's refusal of low-probability links is the mechanism that
wins here (disappearance cost vs a Hungarian that must assign), which
is the same lever the 2026-08-18 greedy-vs-ILP result found.

### p1head (2026-09-04): the greedy pre-pass with the edge head as its cost -- CLOSED, zero: the head-cost pre-pass is NOT more accurate (layer 1) and the pipeline moves −0.001 / +0.003 with the signs disagreeing (layer 2)

The user, 2026-09-04: "ILP是靠排序器的结果分配的，它本身是瞎的。边头不是瞎的
... 我更在意贪心乱连的部分". nopred's control: the pre-pass gives a
readable cell the WRONG predecessor in 2.9% / 6.1% / 0.8% of cases
(6bba / 44b6 / zh), plus parent_taken. `p1head_rank.py` ->
`biohub-p1head` v1 (CPU; generated from edgeprob_rank2.py by
make_p1head.py). The pre-pass exists only to give the ranker its
velocity columns (md, d-md, |v|); here its cost becomes 1 - h_fused
(gate 10 um, one Hungarian stage, assignment kept at h >= 0.10) instead
of distance + score similarity with two gates. Two layers:

1. Pre-pass parent accuracy, measured directly on the 24 F4 sets and
   the pool: for every primary at t+1 matched (5 um) to a GT node whose
   GT parent is matched to a primary at t, is prev_of that primary?
   right / wrong / none per mode (dist vs head) and domain.
2. Pipeline: +BIDIR (distance pre-pass, pool and eval; soft anchors
   0.9261 / 0.8515), +BIDIR_P1H (head pre-pass in pool AND eval, ranker
   retrained on the new velocity columns), +BIDIR_XP1H (the
   distance-trained ranker on head-pre-pass features, twopass's eval-only
   swap). ANCHOR_D187 hard 0.9216. Paired per dataset vs +BIDIR, both
   groups. Pre-registered expectation (mine): layer 1 improves, layer 2
   near zero -- twopass and flowvel were zero and rank2 showed the
   velocity columns matter little beside h_fused; the user's point is
   that those tests never gave the pre-pass a non-blind cost, which this
   does.

### p1head results (v1, 134.0 min CPU, 2026-09-04): P1HEAD OK, 96 of 96 arm-datasets, anchors exact

ANCHOR_D187 F4-6bba 0.9216 / div_J 0.0606 (hard, exact); F4-44b6 0.8353
/ 0.0000; +BIDIR 0.9261 / 0.8515 (soft, both to four places). Join 398
passes (199 x two pre-pass modes), 22,549,808 pairs, 100% matched, 0
NaN p_rev. D175 pool 217,688 rows (0.497 positive); eval rows 14,636 /
8,130.

**Layer 1, pre-pass parent accuracy** (24 F4 sets; a primary at t+1
matched 5 um to a GT node whose GT parent is matched to a primary at t;
is prev_of that primary?):

| mode | domain | n | right | wrong | none | right% | wrong% | none% | wrong / assigned |
|---|---|---|---|---|---|---|---|---|---|
| dist (ships) | F4-6bba | 9,749 | 9,499 | 183 | 67 | 97.44 | 1.88 | 0.69 | 1.89% |
| head (1 − h_fused) | F4-6bba | 9,749 | 9,363 | 188 | 198 | 96.04 | 1.93 | 2.03 | 1.97% |
| dist (ships) | F4-44b6 | 2,559 | 2,349 | 156 | 54 | 91.79 | 6.10 | 2.11 | 6.23% |
| head (1 − h_fused) | F4-44b6 | 2,559 | 2,099 | 146 | 314 | 82.02 | 5.71 | 12.27 | 6.50% |

The head cost does not cut the wrong links: 188 vs 183 on 6bba, 146 vs
156 on 44b6, and conditional on making an assignment it is slightly
WORSE in both domains. What it changes is "none": 3x more cells left
without a predecessor on 6bba, 6x on 44b6 (the pool's no-prev counts
run 2-4x the distance pass's on the big movies: 6bba_05db0fb1 3,036 ->
12,014). The head-cost pass drops assignments whose h_fused < 0.10 and
has one Hungarian stage; the distance pass's two gates keep them. So
the pre-pass's 2% / 6% wrong-parent floor is not a blind-cost problem:
an edge scorer with AUC 0.99 picks the same wrong parents. The
measurement does not separate division daughters (one-to-one
assignment must leave the second daughter wrong or none in BOTH modes),
so part of that floor is structural and identical across modes.

**Layer 2, pipeline** (adj_J, official scorer; paired per dataset vs
+BIDIR = what ships):

| arm | F4-6bba | Δ | up/down | F4-44b6 | Δ | up/down | verdict |
|---|---|---|---|---|---|---|---|
| ANCHOR_D187 | 0.9216 | −0.0044 | 1/11 | 0.8353 | −0.0163 | 1/11 | both negative (v2's gain re-measured) |
| **+BIDIR** (ships) | **0.9261** | — | — | **0.8515** | — | — | — |
| +BIDIR_P1H (head pre-pass, pool + eval, retrained) | 0.9251 | −0.0010 | 7/4 | 0.8549 | +0.0033 | 8/4 | signs DISAGREE |
| +BIDIR_XP1H (dist ranker on head-pre-pass features) | 0.9259 | −0.0002 | 2/4 (6 same) | 0.8529 | +0.0014 | 4/8 | signs DISAGREE |

div_J 0.1515 / 0.0000 in every D175 arm, unchanged. The 44b6 +0.0033
of +BIDIR_P1H is one dataset: 44b6_0b24845f 0.9167 -> 0.9964 (+0.0797);
the per-dataset mean without it is about −0.0006, and 44b6_e29f0176
loses −0.0158, 44b6_87bba6c4 −0.0074. Edge AUC: +BIDIR 0.9929 / 0.9871,
+BIDIR_P1H 0.9928 / 0.9868, the same to three places; permutation
importance of h_fused rises from +0.1765 to +0.2134 when the velocity
columns come from the head pre-pass (they carry less that the head does
not already say). Pre-registered expectation was "layer 1 improves,
layer 2 near zero"; layer 2 held, layer 1 did not.

**Tested claim, killed: "the pre-pass links wrongly because its cost is
blind; the head's probability as the cost would cut the wrong links
and the ranker's velocity columns would improve."** With the head as
cost the wrong-parent count is unchanged (188 vs 183, 146 vs 156) and
the pipeline is −0.001 / +0.003 with the signs disagreeing and the 44b6
side carried by one movie. Residue: (a) the pre-pass's wrong links
occur where the true parent IS detected and BOTH costs still prefer
another primary, so they are either division daughters (structural
under one-to-one assignment) or genuinely ambiguous neighbours; a
per-division split of layer 1 was not run. (b) 44b6_0b24845f's +0.08
from the head pre-pass alone is unexplained; a single-movie diff of the
two arms' edge sets would say whether it is a gap-bridge / short-filter
cascade or the ranker. (c) The head-cost pass leaves 6x more 44b6 cells
without a predecessor at h >= 0.10; nopred already showed the
velocity-less cells are mostly undetected parents, and this adds the
cells whose parent the head does not recognise at 0.10 either. Nothing
here changes what ships. Pkls edge_ranker_ep175v5_bidir.pkl /
_p1h.pkl are on the kernel and unused.

### offdet (2026-09-04): official detector nodes through OUR linker, against F4 -- local WIN (+0.004 / +0.057), in-sample for the detector, went to LB and LOST: ref 56000581 = 0.895 (−0.037), CLOSED

`offdet_cache.py` -> `biohub-offdet` v1 (GPU T4): for the 24 evaluation
datasets and gates 0.965 / 0.99, the official detector's own peaks
(predict_video's `_detect_cells_pooled` wrapped, det_tta on, sigmoid
score kept) and the official head's forward + reverse softmax over those
nodes, one npz readable as a detcache AND an edgeprob cache.
`offdet_rank.py` -> `biohub-offdet-rank` (CPU, pushed when the GPU half
lands; generated from edgeprob_rank2.py by make_offdet_rank.py):
detection arms F4 / OFF965 / OFF99, linker arms +BIDIR (retrained, soft
anchors 0.9261 / 0.8515 on F4; off-distribution score columns on the
official nodes) and HEAD1 (no ranker, soft anchors 0.9245 / 0.8453 on
F4), ANCHOR_D187 hard 0.9216 on F4. Reading: CROSS-DET, the same linker
on official vs F4 nodes, paired per dataset, both groups; a win at >=
+0.003 in both groups buys a submission built on the official detector
(the linker is HEAD1, no ranker retraining needed). Motivation: the gate
curves above; the record's 0.898-vs-0.917 LB verdict changed detector
and linker together.

GPU half landed (biohub-offdet v1, 60.9 min, OFFDET OK 48 of 48, every
round-trip assert held). Nodes per frame on the FULL movies, official
0.965 / 0.99 vs F4: 6bba_3abfe10a 667 / 628 vs 573; 6bba_268e1230 69 /
66 vs 53; 44b6_a2bb48bb 642 / 553 vs 553; 44b6_9bfa6a0a 433 / 346 vs
493; 44b6_0b24845f 363 / 297 vs 339. So at 0.99 the official detector
sits within about -30% .. +20% of F4's node count per movie; the
stride-10 curve's "fewer on 44b6" was three movies' worth. The CPU
half (biohub-offdet-rank v1) is running; results below when it lands.

⚠ **The user's catch (2026-09-04 morning), before the CPU half landed:
"有没有可能是训练集差异呢".** It can. The seed-314159 pack's manifest lists
all 199 train movies as its training set, and the primary official
weights are the same author's recipe (holdout-split.md: `fold: 0`, no
file list). The official detector's training TARGET is the sparse GT
itself -- on 44b6 the 0.77% of cells that are annotated -- so a 100%
recall of annotated 44b6 cells on movies it trained on is what
memorisation looks like, not evidence. Zebrahub is out of sample for
every detector, and there the official one loses at equal detections.
The xl weights (xiaoleilian's public checkpoints) carry no split either;
only ep05 DeepCenter (dctrain, D175) is provably clean on the 24. So the
pre-registered reading is amended: a LOSS for the official nodes on
6bba / 44b6 closes the question (in-sample and still behind); a WIN is
NOT evidence and would need LB, the only venue neither detector has
seen. The kernel was already running and costs nothing more.

### offdet-rank results (v1, 212.6 min CPU, 2026-09-04): the official detector's nodes through the bidirectional head score 0.9297 / 0.9089 -- +0.004 / +0.057 over what ships -- and the pre-registered caveat says LB is the only reading that counts

Anchors exact: ANCHOR_D187 F4-6bba 0.9216 / 0.0606 (hard); +BIDIR F4
0.9261 / 0.8515 and HEAD1 F4 0.9245 / 0.8453 (soft, rank2's numbers
reproduced to four places). Join: 247 passes, 14,120,013 pairs, 100%
matched, 0 NaN p_rev. Labelled eval rows: F4 14,636 / 8,130; OFF965
17,461 / 10,390; OFF99 17,279 / 10,215 (6bba / 44b6).

Stage A, edge AUC of h_fused on its own nodes: F4 0.9919 / 0.9848,
OFF965 0.9817 / 0.9795, OFF99 0.9814 / 0.9797 -- the head ranks
candidate pairs slightly WORSE on the official nodes (more of them, and
more that are real). +BIDIR (D175, distance-pass features) on the
official nodes: 0.9799 / 0.9783 (its score columns are off-distribution
there; sigmoid >= 0.965).

Stage B, adj_J (official scorer), the same linker on each node set:

| linker | nodes | F4-6bba | F4-44b6 | | OFF965-6bba | OFF965-44b6 | | OFF99-6bba | OFF99-44b6 |
|---|---|---|---|---|---|---|---|---|---|
| +BIDIR (ranker + head; ships on F4) | | **0.9261** | **0.8515** | | 0.9258 | 0.9019 | | 0.9261 | 0.9098 |
| HEAD1 (head only, no ranker) | | 0.9245 | 0.8453 | | 0.9263 | 0.9045 | | **0.9297** | **0.9089** |

CROSS-DET, paired per dataset, official minus F4 for the same linker:

| linker | gate | 6bba group | up/down | 44b6 group | up/down | verdict |
|---|---|---|---|---|---|---|
| +BIDIR | 0.965 | −0.0003 | 5/7 | +0.0504 | 9/3 | signs disagree |
| +BIDIR | 0.99 | +0.0001 | 5/7 | +0.0583 | 10/2 | both positive |
| HEAD1 | 0.965 | +0.0018 | 5/7 | +0.0592 | 7/5 | both positive |
| **HEAD1** | **0.99** | **+0.0052** | 5/7 | **+0.0635** | 8/4 | **both positive** |

Against what ships (F4, +BIDIR, ref 55974245): HEAD1 on OFF99 is
+0.0036 on 6bba and +0.0574 on 44b6. div_J: OFF99 HEAD1 0.1379 / 0.0000
vs F4 +BIDIR 0.1515 / 0.0000. node_r (weighted total_node_ratio): OFF99
+0.0628 on 6bba (MORE nodes than estimated, where F4 sits at −0.0228)
and −0.1857 on 44b6 (fewer, F4 −0.1371). Per dataset the official nodes
lose on 5-7 of 12 6bba movies (min −0.044, max +0.060: 3abfe10a
0.7759 -> 0.8246, 61dd1e0d 0.9627 -> 0.9232) and win 8 of 12 44b6 (min
−0.028, max +0.196: 87bba6c4 0.7114 -> 0.7594, a2bb48bb 0.6545 ->
0.8506, 0b24845f 0.8148 -> 1.0077).

### Reading, under the amended pre-registration

The official detector was trained on these movies with these sparse
annotations as its target (the seed-314159 manifest lists all 199;
the primary carries no list). The 44b6 gain -- +0.06 on a domain where
0.77% of cells are annotated and adj_J is edge Jaccard over exactly
those cells -- is what memorisation of the training targets would
produce, and so is the 44b6 shape of the gate curve. The 6bba gain is
small (+0.004 / +0.005) and split 5 / 7 per dataset. **Nothing here is
evidence that the official detector is better; it is evidence that the
question cannot be settled locally.** The pre-registered rule was: a
loss closes the question, a win needs LB. It won, so it goes to LB.

What LB tests: on the hidden movies, which no detector has seen, does
"official detector at 0.99 with flip TTA -> official head, both
directions -> harmonic mean -> our ILP (floor 0.10, disappearance 1.5,
division 0.55) -> gap bridge / short filter / linefit / safe divisions"
beat 0.932? The record's prior against it is the 2026-08-22 family
comparison (official family 0.898 vs xl 0.917), which changed the linker
too. The public 0.948 notebook is exactly this recipe with a second
seed in detection and its own post-processing, so the LB reading also
says whether the detector is where its 0.016 over us comes from.
Submission kernel: `submit_off.py` (built from submit_ep2.py: the xl /
DC detection replaced by offdet_cache's DetectRun, HEAD_ONLY_COL
h_fused, no ranker). Runtime per hidden movie ~ 1.5 min for detection +
both head passes on T4 (offdet: 68-105 s per dataset).

**Built and pushed 2026-09-04: `make_submit_ep2.py` (RANKER_NAME None,
HEAD_ONLY_COL h_fused) -> `submit_ep2.py` -> `make_submit_off.py` ->
`submit_off.py` -> kernel `biohub-submit-off` v1 (GPU T4, no ranker
pkl, sources: pack + xiaoleilian weights x2 + cell-tracking +
dctrain-175-weights for the xl / DC fallback).** Per movie: official
detection at OFF_GATE 0.99 with flip TTA (predict_video's own
`_detect_cells_pooled`, wrapped to keep peaks + sigmoid scores and the
forward softmax matrices), reverse pass with those nodes injected,
h_fused straight into the ILP (floor 0.10, disappearance 1.5, division
0.55), then gap bridge (midpoints only: no sub-gate pool exists) /
short filter / linefit / safe divisions. A movie whose official path
fails falls back to submit_ep2's xl / DC detection + head pass, printed
and counted (OFF_FAILED). **Kernel v1 OK (20.9 min on T4 for the 4
visible movies; 426 s of that is weight loading): 0 of 4 movies fell
back, 0 head passes failed; per movie 82-493 s, the long one
6bba_05db0fb1 with 72,311 official nodes and 269,236 candidate pairs
(the ILP alone 375 s). Visible-4 output: nodes 124,334, edges 119,703,
244,037 rows (F4's visible-4 was 128,157 nodes at ref 55889500; the
official detector at 0.99 puts down 3% fewer).
SUBMITTED as ref 56000581 (2026-09-04 01:00 UTC, description carries
the recipe), control ref 55974245 = 0.932.** Pre-registered reading
(from the amended pre-registration above): > 0.932 = the official
detector's nodes transfer to an unseen embryo and the 0.948 notebook's
margin is at least partly the detector; <= 0.932 = the local +0.004 /
+0.057 was the training-set overlap talking, the detector line closes
at LB and the residue is the 44b6 recall shape (offdet's gate curves).

**LB 0.895 (ref 56000581, scored 2026-09-04): −0.037 against 0.932, the
worst submission on record (M15 was −0.008).** Hidden output
280,141,150 bytes, +3.70% over 55974245's 270,137,616, although on the
four visible movies the official detector put down 3% FEWER nodes than
F4. So on the unseen embryo the official detector at 0.99 detects MORE
than F4 does, and those extra nodes plus whatever the head-only ILP
does with them cost 0.037. This is the second LB reading of the
official detector: 0.898 on 2026-08-22 (official family, its own
linker) and 0.895 now (official nodes, our ILP + patches). The two
differ in everything after detection and land 0.003 apart, so the
detector is the common factor.

**Tested claim, killed: "the official detector's nodes are better than
F4's, and its local +0.004 / +0.057 through our linker would show on
the hidden embryo."** It lost by 0.037. What the record now knows: the
official detector's local numbers on 6bba / 44b6 are in-sample (all
199 train movies, sparse GT as target) and mislead by at least 0.04 in
the direction of the detector; the 0.948 notebook's margin over 0.932
is NOT its detector on its own, since the same detector under our
linker scores 0.895 (its own post-processing, second-seed detection
blend at 0.20 / 0.80, det_threshold 0.965 and DeepCenter veto are what
remain untested). Residue: (a) the hidden embryo's node count under
the official detector is only known through bytes (+3.7%); the
zebrahub gate curves (detfuse3 v4: at equal detections the official
detector is 8-12% behind F4 out of sample) are the closest local
proxy, and they predicted this sign. (b) The head-only linker (HEAD1,
no ranker) has never been scored at LB on F4 nodes; the local gap to
+BIDIR is 0.0016 / 0.0062, so it is not a candidate on its own. (c)
The visible-4 vs hidden node-count reversal (fewer visible, more
hidden) means visible-4 node counts are not a proxy for the hidden
embryo when the detector changes; they were when only the linker did.
No further detector submission without an out-of-sample local test
(zebrahub F1, not 6bba / 44b6 adj_J) that the official detector wins.

### The 0.948 notebook's post-processing, read from its source (scratchpad nb948.txt, cell "bidirectional-production-postprocess", preset harmonic_v3_division_wide), 2026-09-04

Order per movie, after the official inference (dual-seed detection at
0.965, ILP appearance 0 / disappearance 2 / division 1.2, edge
candidate threshold 0.48, bidirectional harmonic 0.15):

1. **Edge filter**: drop edges not to the next frame or longer than 14 um.
2. **MOTION RELINK (`OUTPUT_MOTION_RELINK` default on, not overridden):
   the ILP's edge set is THROWN AWAY and rebuilt greedily.** Per frame
   pair, two Hungarian passes (gate 6 um, then 10 um on the leftovers),
   cost = |target − (source + 0.5·velocity)| + 0.05·|target − source| −
   1.0·p_ILP, where velocity comes from the relink's own previous
   assignment and p_ILP is the ILP edge's probability (0 for pairs the
   ILP did not pick; the bonus is worth at most 1 um). One-to-one, so
   every ILP division is dropped here. Skipped (raw ILP edges kept) only
   if some frame has > 2,600 nodes. Then single-parent repair (best edge
   per target).
3. **Gap close, 1 frame** (max_gap is clamped to 1 in code): track end
   at t to track start at t+2 within 2 × 5.8 um, adjusted ±0.125 um per
   step by local spacing (median of 3 nearest neighbours vs 6.5 um
   reference), Hungarian on distance. Middle node = an isolated node
   within 3.2 um of the midpoint if one exists, else a synthetic
   midpoint pulled to the intensity centroid (window z±1, yx±3, shift ≤
   3.2 um); synthetic middles spanning ≥ 8.5 um must score ≥ 0.25 on
   DeepCenter (pilkwang's center-prior checkpoint, not ours). Cap 5% of
   nodes / 2,000.
4. **Gap2**: end at t to start at t+3, total ≤ 10.2 um, step ≤ 4.4 um,
   direction must agree with the previous or next step (cos > −0.25 and
   |Δstep| ≤ 6 um), two synthetic refined nodes; cap 0.45% of edges /
   180, 0.6% of that frame's ends.
5. **Safe divisions** (the only source of divisions after step 2):
   parent with exactly one child ≤ 10 um away, orphan candidate ≤ 7 um
   from parent and ≤ 12 um from the sister, AND the candidate must be
   the sister's nearest orphan (mutual NN), AND both daughters need a
   single successor at t+2 with the grandchildren ≥ 2.25 um further
   apart than the sisters (divergence), AND DeepCenter ≥ 0.12 at the
   candidate. Score parent_dist + 0.15·sister_dist; caps 0.76% of
   sources per frame, 0.375% of edges global.
6. Division geometry filter: off. **Prune isolated nodes.**
7. **Short-track filter**: components < 6 nodes removed unless they
   contain a division; if that would remove ≥ 10% of nodes, rescue
   4-5-node components whose mean p_ILP ≥ 0.88 and mean edge length ≤
   3.0 um, budget 1.2% of nodes / 120.
8. **Linefit**: window ±2 along single-successor chains, position moved
   80% of the way to the fitted line.

What differs from ours: our patches (gap bridge, SHORT_MIN 4, linefit,
safe divisions) are the same family with looser division rules and no
DeepCenter veto; the big structural difference is step 2, where the
notebook's final links are a velocity-predicted greedy assignment with
the ILP as a ≤ 1 um tie-breaker, and its divisions are all geometric.
With the same official detector, "official nodes → our ILP on h_fused"
scored 0.895 and this recipe scores 0.948, but the detection differs
too (dual-seed blend, gate 0.965 vs 0.99), so the linker's share is not
isolated.


### edgeprob2 results (v1, 2.8 h T4, 2026-09-03): 223 of 223 cached, EDGEPROB2 OK -- and the second seed was trained on ALL 199 train movies

Every round-trip assert held on 199 BASE + 24 F4 datasets, both
directions (same injection as biohub-edgeprob v3). Weight sha256
9bac2fa0... = the checksum the 0.948 notebook pins. Per-dataset numbers
look like head 1's (fwd>0.5 counts within a few percent of edgeprob v3's
for the same datasets, e.g. 6bba_3abfe10a 51,302 vs the BASE pass).

**Leakage, printed by the kernel from the seed pack's own
split_manifest.json: `train` = 199 names, overlapping our 6bba holdout
on 12 of 12 and our 44b6 draw on 12 of 12; `test` = 40 names (the hidden
set's), overlapping the 44b6 draw on 6.** The second head has seen every
local evaluation movie in training. The pack's split_0 weights ship no
manifest, so the same is presumed of head 1 (its entry never had a
number); head 1's local gain nonetheless transferred to LB (+0.007).
Read every local number from the rank2 kernel with this in mind: a
head-2 gain that is in-sample for the head may still be real on LB, or
may not; LB is the only clean reading for anything built on either head.

### edgeprob-rank2 results (v1, 197.9 min CPU, 2026-09-04): NEGATIVE in both groups for every arm -- the second head adds nothing, and the head WITHOUT a ranker sits 0.002 / 0.006 under the shipped ranker+head

Anchors exact: ANCHOR_D187 F4-6bba 0.9216 / 0.0606 (hard), F4-44b6
0.8353 / 0.0000; **+BIDIR retrained here reads exactly v2's 0.9261 /
0.1515 and 0.8515 / 0.0000** (soft anchors OK), so the retraining is
deterministic and the reference arm is the shipped ranker. Join: 199
passes, 11,274,904 pairs, 100% matched between the two heads' caches,
0 NaN p_rev. D175 pool 217,688 labelled rows, 0.497 positive.

Stage A, edge AUC (D175 val / F4-6bba rows / F4-44b6 rows):

| column or arm | val | F4-6bba | F4-44b6 |
|---|---|---|---|
| h_fused (head 1 alone) | — | 0.9919 | 0.9848 |
| h2_fused (head 2 alone) | — | 0.9917 | 0.9846 |
| h_avg (both) | — | 0.9926 | 0.9871 |
| ANCHOR_D187 (18 cols) | — | 0.9894 | 0.9816 |
| +BIDIR (ships) | 0.9926 | 0.9929 | 0.9871 |
| +BIDIR2 | 0.9925 | 0.9928 | 0.9872 |
| +BOTH | 0.9930 | 0.9932 | 0.9877 |
| +AVG | 0.9929 | 0.9933 | 0.9877 |

The two heads are near-duplicates as rankers: in +BOTH the permutation
drop splits between h_fused +0.0099 and h2_fused +0.0069 (each alone
carries +0.18 / +0.13), i.e. either one stands in for the other.

Stage B, the pipeline (official scorer, adj_J; d = paired delta vs +BIDIR):

| arm | F4-6bba | d | up/down | F4-44b6 | d | up/down |
|---|---|---|---|---|---|---|
| ANCHOR_D187 (shipped ranker, no head) | 0.9216 | −0.0044 | 1/11 | 0.8353 | −0.0163 | 1/11 |
| **+BIDIR (ships, ref 55974245)** | **0.9261** | — | — | **0.8515** | — | — |
| +BIDIR2 (second head instead) | 0.9237 | −0.0024 | 1/11 | 0.8425 | −0.0090 | 2/10 |
| +BOTH (both heads' 3 cols each) | 0.9239 | −0.0022 | 5/7 | 0.8414 | −0.0102 | 4/8 |
| +AVG (heads averaged) | 0.9241 | −0.0020 | 4/7 | 0.8464 | −0.0052 | 4/8 |
| HEAD1 (h_fused straight into the ILP, NO ranker) | 0.9245 | −0.0016 | 6/6 | 0.8453 | −0.0062 | 6/6 |
| HEAD12 (h_avg, NO ranker) | 0.9238 | −0.0023 | 5/7 | 0.8438 | −0.0077 | 4/8 |

div_J: +BIDIR 0.1515 / 0, HEAD1 0.2069 / 0, +BOTH 0.1818 / 0 (6bba /
44b6); node_r HEAD1 −0.0228 / −0.1371 against +BIDIR's −0.0199 /
−0.1113 (the head alone keeps fewer nodes).

Per dataset, 6bba is tight (every arm within ±0.02 of +BIDIR on every
movie; 3abfe10a, the big one, is where +BOTH / +AVG / HEAD lose 0.015-
0.02). 44b6 is wild as usual: 0b24845f reads 0.9167 (+BIDIR) / 0.9937
(+BIDIR2) / 0.9979 (+BOTH) / 0.8148 (HEAD1); 87bba6c4 0.6577 / 0.6982 /
0.6721 / 0.7114. The group means and the up/down counts, not those
single movies, are the reading.

### Reading

1. **The second seed adds nothing.** Alone it is a slightly worse
   feature than the first (−0.0024 / −0.0090); beside the first, or
   averaged with it, the ranker cannot turn two near-identical opinions
   into a better edge order (−0.002 / −0.005 to −0.010, both groups
   negative, per-dataset a coin flip on 6bba). The two heads agree too
   much to ensemble. Killed claim: "a second independently trained
   official head, fused as the 0.948 notebook fuses it, adds to the
   first". No submission.
2. **The head can nearly replace the ranker, but not quite.** Bidirectional
   h_fused fed straight to our ILP, with no coordinate column at all,
   scores 0.9245 / 0.8453: above the shipped D187 coordinate ranker
   (0.9216 / 0.8353) and 0.0016 / 0.0062 under the ranker-plus-head that
   ships. The 2026-08-22 chimera's −0.011 was the single-direction head on
   BASE primaries; bidirectional it is a linker in its own right. What the
   18 coordinate columns still buy on top of the head is that 0.0016 /
   0.0062, half of it on 6bba inside per-dataset noise (6 up / 6 down).
   Killed claim: "the head alone beats ranker+head". Residue: the ranker
   is now mostly a calibrator of the head (perm-drop h_fused +0.18, every
   coordinate column small), so the linking side is close to saturated on
   these detections; further gains from the head would have to come from
   the head itself (its own TTA, a fine-tune), not from how it is combined.
3. Both heads trained on all 199 train movies (edgeprob2's manifest, above),
   so every number here is in-sample for the heads. That cuts against the
   heads, not for the ranker: an in-sample head that still cannot beat
   ranker+head will not do so out of sample either.

Cost of the seed2 line: 2.8 h T4 + 0.5 h T4 + 3.3 h CPU. Nothing shipped;
`edge_ranker_ep175v3_*.pkl` (biohub-edgeprob-rank2 v1) and
`submit_ep2.py` / `make_submit_ep2.py` (two heads inline, or head-only,
selectable) exist should a later line want them.

## 2026-09-03 -- nopred: where the no-predecessor cells come from -- the detector and spurious nodes, not the linker

The user, 2026-09-03, on the zero-velocity cells: "没有前驱这个事儿，
可能性也太多了吧" -> "可以去跑一下". `nopred.py` -> `biohub-nopred` v1,
CPU, 59.1 min, 32 of 32 datasets, **hard anchors exact in all three
domains** (F4-6bba 0.9216 / 0.0606, F4-44b6 0.8353 / 0.0000, F4-zh
0.8118 / 0.0215; rankwhy's / twopass's harness, D187 ranker). Local
`tools/test_nopred_buckets.py` passed before the push.

Design: every primary at t >= 1 whose pass-1 velocity is exactly zero
is put in ONE bucket, decision order: `other/linked_zero_disp` (it HAS
a pass-1 predecessor, at identical coordinates) -> `unmatched` (no GT
node within MATCH_R at t) -> `gt_start` (GT track starts here; sub-split
border < 10 um vs interior) -> `division_daughter` (sister took the
parent / parent went elsewhere / parent undetected) -> `parent_undetected`
(GT parent at t-1 has no matched primary; `subthresh` = a detection
below 0.15 within MATCH_R of it, `cold` = nothing at all) ->
`parent_taken` (pass-1 gave the parent's primary to another cell;
competitor unmatched vs another GT cell) -> `parent_out_of_gate` ->
`parent_free_but_unlinked`. `recovered` = the FINAL graph has an in-edge
into this primary from a node at t-1 within 7 um of the GT parent (ILP
edge, gap bridge or division patch; only buckets with a GT parent).
Control = a seeded random sample of NON-zero-velocity primaries, same
size per dataset, same buckets, plus whether pass-1's predecessor IS the
GT parent's primary.

Population: zero-velocity primaries at t >= 1 = 21,524 of 170,079
(12.7%) F4-6bba, 29,448 of 350,570 (8.4%) F4-44b6, 9,058 of 214,186
(4.2%) F4-zh -- flowvel's numbers.

### The buckets (count and share of the zero group · in final graph · recovered)

| bucket | F4-6bba | F4-44b6 | F4-zh |
|---|---|---|---|
| unmatched | 10,353 (48.1%) · 61.4% · - | 29,345 (99.7%) · 50.3% · - | 2,804 (31.0%) · 40.8% · - |
| gt_start | 59 (0.3%) · 89.8% · - | 8 · 87.5% · - | 594 (6.6%) · 56.2% · - |
| division_daughter | 6 · 83.3% · 40.0% | 2 · 100% · 50.0% | 181 (2.0%) · 53.6% · 40.2% |
| parent_undetected | 38 (0.2%) · 60.5% · 17.4% | 26 (0.1%) · 53.8% · 42.9% | **4,897 (54.1%)** · 50.2% · 31.3% |
| parent_taken | 72 (0.3%) · 69.4% · 38.0% | 66 (0.2%) · 59.1% · 41.0% | 577 (6.4%) · 51.0% · 57.1% |
| parent_out_of_gate | 5 · 100% · 0.0% | 1 · 0% · - | 5 · 100% · 0.0% |
| parent_free_but_unlinked | **0** | **0** | **0** |
| other = linked_zero_disp | **10,991 (51.1%)** · 96.8% · - | 0 | 0 |

`unmatched` is UNVERIFIABLE on 6bba (9.7% of cells annotated) and 44b6
(0.77%): it means "not annotated", not "spurious". On zh (~97.5%
annotated) unmatched ~ spurious detection. So zh is the only domain
whose column can be read as causes; 44b6's column says nothing at all.

### Sub-splits and distances (medians, um)

- **gt_start**: border < 10 um in 43 of 59 (6bba), 8 of 8 (44b6), 576 of
  594 (zh, 97.0%); median distance to the nearest face 3.13 / 1.40 /
  1.98; z-face share 25.6% / 12.5% / 18.9%; no non-adjacent GT parent
  anywhere (0 non-adjacent GT edges in these files). These cells enter
  the field; a zero velocity is correct for them.
- **division_daughter** (zh 181): sister took the parent 158 (87.3%, rec
  41.4%), parent went to another cell 6 (rec 0%), parent undetected 17
  (rec 33.3%); this -> GT parent 4.09, sister -> GT parent 2.76. The
  farther daughter loses the greedy pass, as expected.
- **parent_undetected** (zh 4,897): `cold` 4,208 (85.9%, rec 32.7%),
  `subthresh` 689 (14.1%, rec 22.3%); this -> GT parent 1.88 um. The
  parent is right there and the detector produced nothing (86%) or
  something under the threshold (14%). 6bba 19/19, 44b6 11/15 split the
  same way at n ~ 30.
- **parent_taken** (zh 577): competitor unmatched (spurious) 269 (46.6%,
  rec 44.4%), competitor another GT cell 308 (53.4%, rec 64.5%).
  competitor -> parent primary 2.18 vs this -> parent primary 4.88;
  competitor -> GT child 6.12 vs this -> GT child 1.52. The competitor
  is closer to the parent in 465 of 577; closer to this cell's own GT
  node in 16 of 577 (2.8%). Same shape on 6bba (68 of 72) and 44b6 (61
  of 66), where every competitor is unmatched (unverifiable). This is
  rankwhy's case again: the nearer cell wins the greedy pass.
- **parent_out_of_gate**: 11 cells in all, raw distance to the parent
  primary 10.4-10.8 um, damped prediction 10.6-13.3.
- **linked_zero_disp** (6bba only, 10,991): a predecessor exists and its
  coordinates equal this primary's, two frames running. Concentrated in
  the small 6bba movies: 268e1230 339 of 945 zeros, 3a1849c2 667 of
  1,104, 3abfe10a 6,510 of 10,837, 61dd1e0d 763 of 1,267, 7af54fde 1,029
  of 1,679, 7b5d3b2c 392 of 581, afb141ff 611 of 1,091, aeee7805 541 of
  761, c27cba08 139 of 501; 2312ac41, 283bf9f1, 5c824876 have 0. 96.8%
  of them are in the final graph. NOT diagnosed: whether these cells are
  immobile or the detection coordinates are quantised so that a
  sub-voxel move reads as zero. They are 6.5% of every 6bba primary at
  t >= 1.

### Fate in the final graph, zero group vs control

| group | domain | in final | with parent | as start | not in final |
|---|---|---|---|---|---|
| zero | F4-6bba | 79.6% | 57.9% | 21.7% | 20.4% |
| control | F4-6bba | 97.4% | 96.6% | 0.8% | 2.6% |
| zero | F4-44b6 | 50.4% | 18.3% | 32.1% | 49.6% |
| control | F4-44b6 | 95.4% | 94.8% | 0.6% | 4.6% |
| zero | F4-zh | 47.8% | 15.5% | 32.3% | 52.2% |
| control | F4-zh | 98.5% | 98.3% | 0.2% | 1.5% |

Side by side, the rates that differ (zero vs control): zh unmatched
31.0% vs 9.1%, gt_start 6.6% vs 1.2%, parent_undetected 54.1% vs 2.1%,
recovered 34.2% vs 99.1%. The control's pass-1 predecessor is the GT
parent's primary in 1,176 of 1,211 (6bba), 216 of 230 (44b6), 7,811 of
7,875 (zh) matched cases; `linked_elsewhere` sits at 6.4-8.4 um from the
true parent and 2.0-3.3 um from the one it took.

Degradation counters over the 32 datasets: acceleration undefined (no
parent) 48,077 rows, one backward step only 37,001 rows.

### Reading

The zero-velocity cell has no pass-1 predecessor for one of three
reasons, and in the only domain where the reasons can be read (zh) the
LINKER is the smallest of them:

1. **The parent was never detected** -- 54% on zh, and 86% of those are
   cold, nothing within MATCH_R of a GT node 1.9 um away. The gap bridge
   repairs a third. This is the detector line (fatescore / M15 tried
   adding missed points and lost 0.008 on LB), not a linker feature.
2. **There is nothing to link to** -- spurious detection 31% (zh), field
   entry 6.6%, division daughter 2%. A zero is the right answer.
3. **The greedy pass gave the parent to a nearer cell** -- 6.4% on zh,
   half of the competitors spurious. The ILP + patches already recover
   57%. This is the case the edge head was added for (edgeprob).

`parent_free_but_unlinked` is 0 in every domain: the pre-pass never
leaves an available parent unlinked inside the gate. So a "better
velocity" had nothing to act on, which is why twopass and flowvel were
zero: the velocity is missing because the parent node is missing, not
the other way round. Killed claim: "the zero-velocity cells are linker
failures a motion feature can repair". Residue: (a) the 6bba
`linked_zero_disp` population, 6.5% of the domain's primaries, identical
coordinates two frames running, undiagnosed; (b) zh `parent_undetected
cold`, 4,208 cells, is a detector recall question at 1.9 um from a
detected cell, the same place M15 lost; (c) the zero group is dropped
from the final graph at 20% / 50% / 52% against 3% / 5% / 2% for the
control, so `_short_filter` and the ILP are the stages that price these
cells, consistent with most of them being spurious or unverifiable.
44b6 cannot be read at all (99.7% unmatched); whatever the hidden set's
44b6-like movies do with these cells is not measurable here.

## 2026-09-03 -- flowvel: the flow is coherent, the gain is zero -- motion is not what those cells lack

`flowvel.py` -> `biohub-flowvel` v1, CPU, pushed 2026-09-03 (twopass's
harness, same anchors, same D187 ranker). Flow for a primary at t = the
component-wise median of the pass-1 velocity vectors of its K=8 nearest
primaries within 25 um that have a non-zero pass-1 velocity (K_MIN 3,
else undefined). Arms: FLOWFILL (flow only where pass-1 velocity is
zero, 12.7% / 8.4% / 4.2% of primaries), FLOWALL (flow everywhere it is
defined). Prints the flow-vs-own-velocity cosine per domain, a
coherence sanity number. Results to be filed here.

### Results (157.2 min CPU, 96 of 96 arm-datasets, anchors exact in all three domains)

| arm | domain | adj_J | d adj_J vs ANCHOR | per-dataset up / down / same |
|---|---|---|---|---|
| FLOWFILL | F4-6bba | 0.9218 | +0.0002 | 5 / 6 / 1 |
| FLOWFILL | F4-44b6 | 0.8359 | +0.0006 | 6 / 6 / 0 |
| FLOWFILL | F4-zh | 0.8121 | +0.0002 | 7 / 0 / 1 |
| FLOWALL | F4-6bba | 0.9221 | +0.0004 | 5 / 5 / 2 |
| FLOWALL | F4-44b6 | 0.8328 | −0.0025 | 3 / 8 / 1 |
| FLOWALL | F4-zh | 0.8124 | +0.0006 | 6 / 2 / 0 |

FLOWFILL passes the three-domain sign rule, but every magnitude is
under 0.001, a fifth of the +0.003 floor; per-dataset it is a coin flip
in the two competition domains. FLOWALL's signs disagree.

Flow statistics (K 8, R 25 um, K_MIN 3), primaries at t >= 1:

| domain | defined | pass-1 zero | filled (zero AND defined) | med \|flow\| | med cos(flow, own v) |
|---|---|---|---|---|---|
| F4-6bba | 89.4% | 21,524 | 10,142 (47.1%) | 1.90 | 0.787 |
| F4-44b6 | 99.7% | 29,448 | 29,184 (99.1%) | 1.48 | 0.693 |
| F4-zh | 99.7% | 9,058 | 9,037 (99.8%) | 0.64 | 0.711 |

FLOWFILL touched 6.1% / 7.5% / 3.8% of ranker rows; FLOWALL 90-99%.
Refeature self-check 0.00e+00 in all three domains.

Reading: the tissue does move coherently (cosine 0.69-0.79 between a
cell's own link and its neighbours' median), so the filled cells got a
sensible motion prediction -- and the ranker still could not turn it
into correct edges. The 6bba zeros are half isolated cells (fewer than
3 linked neighbours within 25 um), which no flow can serve. Together
with twopass: the no-predecessor cells are not missing MOTION
information; whatever they lack is not in the coordinate stream.
Killed claim: "a velocity for the zero-velocity cells recovers their
edges". Residue: none on the velocity line; the pixel line (edgeprob)
is where the ranker's remaining signal was found.

## 2026-09-03 -- twopass: ZERO, and the premise is gone -- the ILP's velocities equal the greedy pre-pass's

`twopass.py` -> `biohub-twopass` v1, CPU, pushed 2026-09-03 while ref
55974245 rescores. The 2026-08-25 "version nobody has tried": solve
once (ANCHOR, F4 + D187), rebuild velocity from the ILP's own edges
(predecessor = the solved parent; no parent -> zero, as pass-1 does),
recompute the three velocity-dependent ranker columns (4 `md`, 5
`d − md`, 16 `|v|`; the other 15 are velocity-free), re-score with the
SAME D187 ranker, re-solve, patch, score. Arms: TWOPASS_SAME (pred =
P + v), TWOPASS_HALF (pred = P + 0.5 v, the pre-pass's own damping).
Three domains on the F4 arm, anchors 0.9216 / 0.8353 / 0.8118 hard.
Distribution shift stated up front: D187 was trained on pass-1
velocities, so a negative here closes only "the shipped ranker profits
without retraining". Results to be filed here.

### Results (199.5 min CPU, 96 of 96 arm-datasets, anchors exact in all three domains)

| arm | domain | adj_J | d adj_J vs ANCHOR | per-dataset up / down / same |
|---|---|---|---|---|
| ANCHOR | F4-6bba / 44b6 / zh | 0.9216 / 0.8353 / 0.8118 | — | — |
| TWOPASS_SAME | F4-6bba | 0.9217 | +0.0001 | 3 / 5 / 4 |
| TWOPASS_SAME | F4-44b6 | 0.8352 | −0.0000 | 2 / 10 / 0 |
| TWOPASS_SAME | F4-zh | 0.8119 | +0.0001 | 5 / 3 / 0 |
| TWOPASS_HALF | F4-6bba | 0.9218 | +0.0001 | 5 / 3 / 4 |
| TWOPASS_HALF | F4-44b6 | 0.8342 | −0.0010 | 4 / 8 / 0 |
| TWOPASS_HALF | F4-zh | 0.8122 | +0.0004 | 7 / 1 / 0 |

Both arms: signs disagree across domains, every magnitude inside noise.
Node/edge counts move by tens to a couple of hundred per domain.

### The velocity table is the finding

| source | domain | primaries (t ≥ 1) | velocity zero | zero % | med \|v\| (all / non-zero) |
|---|---|---|---|---|---|
| pass-1 (greedy) | F4-6bba | 170,079 | 21,524 | 12.7 | 2.14 / 2.31 |
| pass-1 | F4-44b6 | 350,570 | 29,448 | 8.4 | 1.89 / 2.05 |
| pass-1 | F4-zh | 214,186 | 9,058 | 4.2 | 1.37 / 1.39 |
| **solved tracks** | F4-6bba | 170,079 | **24,050** | **14.1** | 2.11 / 2.30 |
| solved tracks | F4-44b6 | 350,570 | 38,921 | 11.1 | 1.81 / 2.04 |
| solved tracks | F4-zh | 214,186 | 11,339 | 5.3 | 1.36 / 1.39 |

Of the solved-track zeros, 6bba: 6,404 primaries the ILP kept no node
for + 6,930 track starts; 44b6: 23,106 + 15,815; zh: 6,765 + 4,574.
Refeature self-check 0.00e+00 in all three domains (columns 4/5/16 are
the only velocity-dependent ones; verified at runtime).

Reading: the ILP's velocities are the greedy pre-pass's velocities.
Where pass-1 finds a predecessor the ILP picks the same edge; where it
does not, the ILP mostly has no node or no parent there either, so the
zero rate goes UP, not down. The 2026-08-25 premise ("the greedy chain
is exactly what fails; the ILP's tracks are far better trajectories")
is not true for velocity. That also reads back onto velwhy: the 24% of
6bba rank>=1 misses with velocity exactly zero are cells with no
predecessor in ANY linker, not a pre-pass artefact. The retrained-ranker
follow-up is not worth running: its inputs would be the same numbers.
Killed claim: "a better velocity source exists inside the pipeline".
Residue: a velocity source OUTSIDE the linker (neighbourhood flow) was
never tested.

## 2026-09-02 -- ⭐⭐ edgeprob: the official edge head's probability as a ranker feature -- POSITIVE in both clean groups (+0.0046 / +0.0106 adj_J), BASE arm, D32 pool

> Run with the user in the session that read the seven "待分析" entries.
> Results at the bottom of this entry. NOT yet on F4 primaries, NOT yet
> with the D187 pool, NOT submitted.

Why: the seven diagnostics say the ranker's misses are "true target 6-7
um away, a nearer cell 2-3 um away" (rankwhy: the preferred target is
closer in 97-100% of rank>=1 cases) and the 18 columns are all
coordinates. The one pixel-derived, mask-free signal never fed to the
ranker is the official edge head's opinion (`edge_prob`, dropped by the
transplant -- 2026-08-24 entry). The mask-based pixel line (2026-08-25
morph_edge, six ratio columns, AUC 0.9866 vs 0.9864) tested a different
claim; see the provenance note under that entry.

Design, two kernels:

1. `edgeprob_cache.py` -> `biohub-edgeprob` (GPU, T4). Official weights,
   `predict_video` with `_detect_cells_pooled` monkeypatched to return
   OUR BASE xl primaries (`seeds_of`, score >= 0.15, order kept; full-res
   voxel coords / downsample (1,4,4) as float grid coords), `threshold`
   2.0 so its own candidate loop keeps nothing, `model.predict_edges`
   wrapped to capture the raw logit matrix per consecutive pair; softmax
   over sources (the official activation) gated to the ranker's 10 um
   pairs. A reverse pass (frames remapped f -> T-1-f) scores t+1 -> t
   with the same model (the public 0.948 notebook's bidirectional
   fusion, harmonic mean at weight 0.15). Output per dataset:
   (t, i, j, p_fwd, p_rev, l_fwd, l_rev, n_per_frame), indices identical
   to `pair_features`' rows. Datasets: HOLDOUT_6BBA 12 + 12 44b6 drawn
   with dctrain's seed 20260827 from the 55 outside the D32 pool + the
   D32 pool 32 = 56. det_tta off (TTA touches only det_logits).
   **v1 SMOKE (2 datasets) passed all round-trip asserts: 36 s and 44 s
   per dataset, both directions.** v2 = all 56, ~40 min, running.
2. `edgeprob_rank.py` -> `biohub-edgeprob-rank` (CPU). rankwhy's
   harness; labels verbatim from poolsize.py; retrains D32-pool rankers
   for arms BASE18 / +PFWD / +BIDIR (p_fwd, p_rev, h_fused) / +ALL, plus
   the shipped D187 pkl as ANCHOR_D187 (loaded, not judged). Stage A:
   AUCs + permutation importance; Stage B: full solve + patches +
   official scorer on the 12 6bba and the 12 44b6 (which are OUT of the
   D32 pool, so both are clean holdouts in every judged arm). Hard
   anchors: ANCHOR_D187 6bba 0.9293, BASE18 6bba 0.9251 (fusescore
   2026-08-30). No zebrahub.

Expectation stated before the result: the chimera (official head ALONE
on xl nodes) lost 0.011 locally; as an added feature the ceiling is the
rank>=1 population, 1.4% of 6bba edges. A three-domain-sized gain here
would still be a local number.

### Results

`biohub-edgeprob` v2: 56 of 56 cached, 40.1 min T4, every round-trip
assert held (36-67 s per dataset, both directions). `biohub-edgeprob-rank`
v1: 116.5 min CPU, 120 of 120 arm-datasets scored, join 100% matched
(3,219,747 pairs, 0 NaN p_rev). BASE18 retrained on the D32 pool
reproduces `edge_ranker_d32.pkl` to max |dp| 0.00e+00.

ANCHORS, all four exact: ANCHOR_D187 6bba 0.9293 / 44b6 0.8185; BASE18
6bba 0.9251 / 44b6 0.8250 (fusescore 2026-08-30 to the digit).

Stage A, edge AUC (val split of the D32 pool / 6bba rows / 44b6 rows):

| | pool | 6bba | 44b6 |
|---|---|---|---|
| **the head ALONE, p_fwd** | 0.9898 | 0.9903 | 0.9819 |
| the head ALONE, p_rev | 0.9912 | 0.9914 | 0.9843 |
| the head ALONE, h_fused | **0.9917** | **0.9922** | **0.9849** |
| BASE18 (18 cols) | 0.9884 | 0.9886 | 0.9790 |
| +PFWD (19) | 0.9928 | 0.9921 | 0.9853 |
| +BIDIR (21) | 0.9932 | 0.9928 | 0.9861 |
| +ALL (23) | 0.9935 | 0.9927 | 0.9864 |

The official head by itself, on OUR node set, out-ranks the 18-column
ranker on edge AUC in all three populations. Permutation drop: +PFWD
p_fwd 0.3588; +BIDIR h_fused 0.0926 (p_fwd 0.0023, p_rev 0.0065).

Stage B, official scorer, per arm and group:

| arm | group | score | adj_J | div_J | d adj_J vs BASE18 | d div_J |
|---|---|---|---|---|---|---|
| ANCHOR_D187 | 6bba | 0.9424 | 0.9293 | 0.1316 | +0.0042 | +0.0348 |
| ANCHOR_D187 | 44b6 | 0.8185 | 0.8185 | 0.0000 | −0.0065 | 0 |
| BASE18 | 6bba | 0.9348 | 0.9251 | 0.0968 | — | — |
| BASE18 | 44b6 | 0.8250 | 0.8250 | 0.0000 | — | — |
| +PFWD | 6bba | 0.9385 | 0.9280 | 0.1042 | +0.0029 | +0.0074 |
| +PFWD | 44b6 | 0.8366 | 0.8329 | 0.0370 | +0.0080 | +0.0370 |
| **+BIDIR** | 6bba | 0.9455 | **0.9297** | 0.1579 | **+0.0046** | +0.0611 |
| **+BIDIR** | 44b6 | 0.8356 | **0.8356** | 0.0000 | **+0.0106** | 0 |
| +ALL | 6bba | 0.9437 | 0.9291 | 0.1463 | +0.0040 | +0.0496 |
| +ALL | 44b6 | 0.8295 | 0.8295 | 0.0000 | +0.0045 | 0 |

Paired per-dataset adj_J delta vs BASE18:

| arm | group | group d | per-dataset mean | min | max | up / down |
|---|---|---|---|---|---|---|
| +PFWD | 6bba | +0.0029 | +0.0015 | −0.0150 | +0.0175 | 7 / 5 |
| +PFWD | 44b6 | +0.0080 | +0.0064 | −0.0179 | +0.0639 | 5 / 7 |
| +BIDIR | 6bba | +0.0046 | +0.0038 | −0.0035 | +0.0138 | 8 / 4 |
| +BIDIR | 44b6 | +0.0106 | +0.0113 | −0.0104 | +0.0643 | 7 / 5 |
| +ALL | 6bba | +0.0040 | +0.0028 | −0.0110 | +0.0163 | 8 / 4 |
| +ALL | 44b6 | +0.0045 | +0.0011 | −0.0882 | +0.0542 | 7 / 5 |

Per-dataset adj_J, 6bba (ANCHOR_D187 / BASE18 / +PFWD / +BIDIR / +ALL):
268e1230 0.8914/0.8930/0.8972/0.8986/0.9004 · 2312ac41 0.9940/0.9940/
0.9939/0.9940/0.9939 · 283bf9f1 0.9441/0.9360/0.9535/0.9498/0.9490 ·
3a1849c2 0.8950/0.8952/0.8942/0.8917/0.8854 · 3abfe10a 0.8027/0.7900/
0.7940/0.8022/0.8053 · 5c824876 0.9321/0.9290/0.9319/0.9311/0.9303 ·
61dd1e0d 0.9529/0.9529/0.9531/0.9532/0.9531 · 7af54fde 0.9141/0.9025/
0.9094/0.9129/0.9188 · 7b5d3b2c 0.9955/0.9955/0.9937/0.9954/0.9936 ·
afb141ff 0.9304/0.9200/0.9218/0.9277/0.9232 · aeee7805 0.9975/0.9963/
0.9950/0.9964/0.9964 · c27cba08 0.9561/0.9637/0.9487/0.9606/0.9527.
44b6: 0b24845f 0.9660/0.9668/0.9585/0.9615/**0.8787** · 0db75fae 0.9233/
0.9113/0.9177/0.9367/0.9364 · 33b596bf 0.9053/0.9056/0.8877/0.9054/
0.9054 · 5740d24b 0.9939/1.0115/1.0073/1.0021/1.0016 · 87bba6c4 0.6339/
0.6469/0.7108/0.7111/0.7011 · 8f5ab931 0.8717/0.8816/0.8950/0.8964/
0.8947 · 9bfa6a0a 0.8358/0.8642/0.8601/0.8740/0.8669 · a2bb48bb 0.5268/
0.5237/0.5452/0.5461/0.5325 · c15fded2 0.8217/0.8454/0.8393/0.8631/
0.8555 · c96cfa10 0.8709/0.8709/0.8672/0.8605/0.8602 · d2f34f90 0.9736/
0.9735/0.9630/0.9676/0.9558 · e29f0176 0.7252/0.7512/0.7776/0.7633/0.7774.

### Reading (this session, with the user)

- **The tested claim survives: the head's probability carries
  information the 18 coordinate columns do not.** Head-alone edge AUC
  beats BASE18 in all three populations, and the paired pipeline delta
  is positive in both clean groups for every arm. +BIDIR is the best
  arm in both groups and the only one where 6bba has no dataset losing
  more than 0.0035.
- **+BIDIR trained on 32 datasets reaches 0.9297 on 6bba, above the
  shipped D187 ranker's 0.9293 trained on 187.** On 44b6 it is 0.8356
  vs D187's 0.8185, but D187's 44b6 is a different population (D187
  saw all 71 44b6 in training and still scores lower here because
  fusescore's BASE x D32 vs D187 gap is 44b6-specific; do not read that
  0.017 as the feature).
- **The reverse pass adds to the forward one** (+BIDIR > +PFWD in both
  groups); the raw logits do not (+ALL < +BIDIR, and +ALL collapses one
  44b6 dataset by 0.088), so ship probabilities, not logits.
- **Caveats, in order.** (1) BASE arm, D32 pool: the shipped pipeline is
  F4 primaries + D187 ranker, neither measured here. (2) 44b6 per-dataset
  spread is ±0.06 and up/down is 7/5; the group number is carried by the
  big movies. (3) The record's own rule: local +0.002 to +0.007 has not
  moved the LB 4 times in 5; this has a measured mechanism (a new
  information source with higher standalone AUC), which the one success
  also had, and the 44b6 delta is above that range. (4) Hidden-test cost:
  the head must run inline, ~40-50 s/dataset on a T4 for both directions
  = +2.2-2.8 h on the ~6 h pipeline; fits.
- Next, if the user wants an LB reading: edgeprob over the remaining 143
  pool datasets (~1.6 h GPU) and over the F4 primaries of the 24 eval
  sets (~20 min), retrain +BIDIR on the D187 pool, evaluate on the F4
  arm against 0.9216 / 0.8353, then a submission kernel with the head
  inline (the smoke script's injection is exactly that code).

### 2026-09-03 -- F4 arm, D175 pool: +BIDIR holds, 44b6 12 of 12 up

The user's morning instruction: run it. `biohub-edgeprob` v3 (GPU,
158 min): all 199 train movies on BASE primaries + the 24 eval sets on
F4 primaries (`edgeprob_F4_<name>.npz`), 223 of 223, every assert held.
`biohub-edgeprob-rank` v2 (CPU, 108 min): pool = **D175** (199 minus
HOLDOUT_6BBA minus the seeded 44b6 12, i.e. dctrain's split), so BOTH
eval groups are clean holdouts for the trained arms. Evaluation on the
**F4 arm**, what ships. Join 100% (11,274,904 pairs), D175 rows 217,688
(0.497 positive).

ANCHORS exact: ANCHOR_D187 F4-6bba adj_J 0.9216 / div_J 0.0606, F4-44b6
0.8353 / 0.0000.

Stage A, edge AUC (D175 val split / F4-6bba rows / F4-44b6 rows):
head alone p_fwd 0.9890 / 0.9904 / 0.9818, h_fused 0.9909 / 0.9919 /
0.9848; ANCHOR_D187 (loaded) — / 0.9894 / 0.9816; BASE18 0.9875 /
0.9898 / 0.9800; **+BIDIR 0.9926 / 0.9929 / 0.9871**. Permutation drop
in +BIDIR: h_fused 0.1765, p_fwd 0.0019, p_rev 0.0020.

Stage B, official scorer:

| arm | domain | score | adj_J | div_J | d adj_J vs BASE18 |
|---|---|---|---|---|---|
| ANCHOR_D187 (shipped) | F4-6bba | 0.9277 | 0.9216 | 0.0606 | −0.0016 |
| ANCHOR_D187 (shipped) | F4-44b6 | 0.8353 | 0.8353 | 0.0000 | +0.0006 (in-sample) |
| BASE18 (D175) | F4-6bba | 0.9297 | 0.9232 | 0.0645 | — |
| BASE18 (D175) | F4-44b6 | 0.8346 | 0.8346 | 0.0000 | — |
| **+BIDIR (D175)** | F4-6bba | 0.9412 | **0.9261** | 0.1515 | **+0.0028** |
| **+BIDIR (D175)** | F4-44b6 | 0.8515 | **0.8515** | 0.0000 | **+0.0169** |

Paired per-dataset adj_J delta, +BIDIR vs BASE18: F4-6bba mean +0.0026,
min −0.0045, max +0.0107, **up 8 down 4**; F4-44b6 mean +0.0194, min
+0.0001, max +0.0772, **up 12 down 0**.

Per dataset (ANCHOR_D187 / BASE18 / +BIDIR), F4-6bba: 268e1230 0.8887/
0.8974/0.8929 · 2312ac41 0.9950/0.9959/0.9957 · 283bf9f1 0.9378/0.9407/
0.9507 · 3a1849c2 0.8601/0.8617/0.8630 · 3abfe10a 0.7853/0.7870/0.7914 ·
5c824876 0.9265/0.9265/0.9304 · 61dd1e0d 0.9517/0.9517/0.9624 · 7af54fde
0.9038/0.9045/0.9015 · 7b5d3b2c 0.9956/0.9956/0.9956 · afb141ff 0.9302/
0.9303/0.9351 · aeee7805 0.9934/0.9935/0.9974 · c27cba08 0.9782/0.9822/
0.9822. F4-44b6: 0b24845f 0.9112/0.9112/0.9167 · 0db75fae 0.9234/0.9416/
0.9418 · 33b596bf 0.9023/0.9023/0.9024 · 5740d24b 1.0271/1.0151/1.0163 ·
87bba6c4 0.5887/0.5805/0.6577 · 8f5ab931 0.8544/0.8501/0.8640 · 9bfa6a0a
0.8884/0.8848/0.8887 · a2bb48bb 0.6450/0.6472/0.6718 · c15fded2 0.8211/
0.8211/0.8460 · c96cfa10 0.8728/0.8757/0.8827 · d2f34f90 0.9535/0.9535/
0.9661 · e29f0176 0.8071/0.8071/0.8690.

Against the SHIPPED ranker (D187) on the same F4 primaries: +BIDIR is
**+0.0045 F4-6bba / +0.0162 F4-44b6** adj_J, and +0.0135 / +0.0162 in
score (the 6bba score gain is mostly div_J on 12 divisions, not
resolvable).

Reading: the strongest local candidate on record -- the weakest domain
is +0.0028 (below the +0.003 floor the record uses), but with a
measured mechanism (a new information source whose standalone AUC beats
the ranker's), and the dense domain moves 12 of 12 datasets the same
way. No zebrahub arm (the head was not run on zh crops). Submission:
`submit_ep.py` (generated by `make_submit_ep.py` from `submit_f4.py`;
single variable = the head's p_fwd/p_rev/h_fused columns + the D175
+BIDIR ranker `edge_ranker_ep175_bidir.pkl`; the head runs inline
forward + reverse, ~40-50 s/movie; a failed head pass degrades that
movie to NaN columns and is counted). Pushed as `biohub-submit-ep` v1
for the visible-test run; competition submission follows if it runs
clean.

**Visible-4 run (v1): clean.** Head pass ok on all 4 movies (29 / 36 /
36 / 58 s), 0 movies degraded. Output nodes 126,488 / edges 121,593
(F4's visible-4 was 128,157 nodes, −1.3%). Tracking wall time 1,041 s
vs F4's 914 s (+14%, the head's ~40 s/movie) -> hidden ~8.2 h, under
12. The visible-set self-score block died on `ModuleNotFoundError:
tracking_cellmot` (internet off; the scorer is not in the pack) AFTER
submission.csv was written and validated -- same code path as
submit_f4, harmless to the submission, but the local score is absent.

**SUBMITTED as ref 55974245** (2026-09-03, description carries the
pre-registered reading). Control: 55889500 = 0.925. Pre-registered:
>0.925 = the head carries information the coordinates lack (the
mechanism the local run measured); 0.925 = uninformative; <0.925 =
the local +0.0045 / +0.0162 did not transfer (the record's 4-in-5).

**LB: 0.932 (+0.007). Pre-registered branch: the head carries
information the coordinates lack.** Hidden output 270,137,616 bytes vs
F4's 270,740,009 (−0.22%): the gain is not bought with more edges. The
holdout is now 2 for 6, and both successes had a measured mechanism
(F4: the repair budget bound; edgeprob: a new information source whose
standalone AUC beat the ranker's). Local-to-LB: F4-6bba +0.0045 and
F4-44b6 +0.0162 produced +0.007 on a hidden third embryo.

## 2026-09-02 -- ⏳ UNREVIEWED: velwhy, the velocity is mostly present and re-ordering by it recovers 12-19%

> ⏳ **待分析.** Same marker as the entries below; tables only.

`velwhy.py` -> `biohub-velwhy`, CPU, 35.2 min. rankwhy's rank >= 1
population, with the source's velocity magnitude and the true target's
position under two alternative orderings of the SAME candidate list:
the ranker's column 0 (raw distance) and column 4 (distance from the
velocity-predicted position). Case counts reproduce rankwhy exactly
(173 / 152 / 1,510). Anchors exact.

| | F4-6bba (173) | F4-44b6 (152) | F4-zh (1,510) |
|---|---|---|---|
| velocity EXACTLY ZERO | **42 (24.3%)** | 8 (5.3%) | 121 (8.0%) |
| where non-zero: median / p90 (um) | 2.17 / 4.60 | 2.16 / 4.79 | 1.61 / 3.59 |
| true target first by RAW distance | 5 (2.9%) | 0 (0.0%) | 9 (0.6%) |
| **true target first by PREDICTED distance** | **27 (15.6%)** | **29 (19.1%)** | **187 (12.4%)** |
| predicted beats raw | 36 (20.8%) | 36 (23.7%) | 246 (16.3%) |
| predicted loses to raw | 13 (7.5%) | 20 (13.2%) | 199 (13.2%) |
| median d_true / md_true (um) | 6.90 / 6.94 | 6.59 / 6.68 | 5.89 / 6.41 |
| median d_top / md_top (um) | 3.32 / 3.56 | 2.29 / 3.14 | 2.13 / 2.68 |
| median candidates | 2.0 | 3.0 | 4.0 |

`md_true` exceeds `d_true` in all three domains and `md_top` exceeds
`d_top`, so the velocity prediction moves the source away from both
candidates; the re-ordering gain comes from the relative change, not
from the prediction landing nearer the true target.

This is a re-ordering measurement on a fixed candidate list, not a
scored pipeline change.

Cost: 35.2 min CPU, no GPU quota, no submission slot.

## 2026-09-02 -- ⏳ UNREVIEWED: rankwhy, what the ranker prefers when the correct target is not first

> ⏳ **待分析.** Same marker as the entries below; tables only.

`rankwhy.py` -> `biohub-rankwhy`, CPU, 68.5 min. edgewhy with two extra
blocks: what the ranker put first when the correct pair sat at rank >= 1,
and what the `node_absent` sources looked like.

ANCHORS exact: F4-6bba adj_J 0.9216, F4-44b6 0.8353, F4-zh 0.8118.

### What the ranker preferred instead

| | F4-6bba (173) | F4-44b6 (152) | F4-zh (1,510) |
|---|---|---|---|
| preferred target IS an annotated cell | 9 (5.2%) | 5 (3.3%) | **929 (61.5%)** |
| ... and is another cell's GT successor | 9 (5.2%) | 4 (2.6%) | **864 (57.2%)** |
| the lost edge survived anyway | 24 (13.9%) | 20 (13.2%) | 150 (9.9%) |
| median src -> preferred (um) | 3.32 | 2.29 | 2.13 |
| median src -> true (um) | 6.90 | 6.59 | 5.89 |
| median preferred -> true (um) | 8.64 | 7.26 | 7.06 |
| detection score, preferred / true | 0.342 / 0.321 | 0.312 / 0.300 | 0.275 / 0.257 |
| sources bidding for the preferred target (median) | 2.0 | 3.0 | 4.0 |
| **preferred is CLOSER than the true target** | **168 (97.1%)** | **152 (100.0%)** | **1,500 (99.3%)** |

Distance from the preferred target to the true one:

| | 6bba | 44b6 | zh |
|---|---|---|---|
| 0-2 um | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 2-4 um | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 4-6 um | 20 (11.6%) | 41 (27.0%) | 417 (27.6%) |
| 6+ um | 153 (88.4%) | 111 (73.0%) | 1,093 (72.4%) |

6bba and 44b6 annotate 9.71% and 0.77% of their cells, so `preferred is
an annotated cell` is a floor there, not a rate; zebrahub's ~97.5%
annotation is the only column that reads as a rate.

### The nodes the solver dropped (node_absent)

Both endpoints detected, the correct pair the ranker's own TOP choice,
and the source never reached the solved graph.

| | F4-6bba (73) | F4-44b6 (66) | F4-zh (2,107) |
|---|---|---|---|
| best outgoing probability (median) | **0.727** | **0.815** | **0.809** |
| best incoming probability (median) | 0.678 | 0.522 | 0.315 |
| candidates it had (median) | 2.0 | 3.0 | 4.0 |
| sources bidding for it (median) | 2.0 | 3.0 | 3.0 |
| detection score (median) | 0.276 | 0.297 | 0.246 |
| had NO incoming candidate at all | 3 (4.1%) | 2 (3.0%) | 71 (3.4%) |
| best_out 0.1-0.3 | 8 (11.0%) | 7 (10.6%) | 37 (1.8%) |
| best_out 0.3-0.6 | 16 (21.9%) | 9 (13.6%) | 125 (5.9%) |
| **best_out 0.6-1.0** | **49 (67.1%)** | **50 (75.8%)** | **1,945 (92.3%)** |

Cost: 68.5 min CPU, no GPU quota, no submission slot.

## 2026-09-02 -- ⏳ UNREVIEWED: divsweep, DIV_DIVERGE_UM 2.25 down to 0.00, scored

> ⏳ **待分析.** Same marker as the entries below; tables only, no
> verdict filed.

`divsweep.py` -> `biohub-divsweep`, CPU, 108.7 min. Post-link sweep: one
ILP solve per dataset replayed five times with only `DIV_DIVERGE_UM`
changed, so every arm shares one detection set, one ranker and one
solution and the 44b6 in-sample optimism is identical across arms.

ANCHORS exact, all three: F4-6bba adj_J 0.9216 / div_J 0.0606, F4-44b6
0.8353 / 0.0000, F4-zh 0.8118 / 0.0215.

### F4-6bba (12 datasets)

| DIV_DIVERGE_UM | forks | edges | adj_J | div_J | score | d_adj_J | d_div_J | d_score |
|---|---|---|---|---|---|---|---|---|
| **2.25 (shipped)** | 460 | 157,581 | 0.9216 | 0.0606 | 0.9277 | — | — | — |
| 1.50 | 595 | 157,716 | 0.9213 | 0.0500 | 0.9263 | −0.0003 | −0.0106 | −0.0014 |
| 1.00 | 611 | 157,732 | 0.9216 | 0.0435 | 0.9259 | −0.0000 | −0.0171 | −0.0018 |
| 0.50 | 611 | 157,732 | 0.9220 | 0.0455 | 0.9266 | +0.0004 | −0.0152 | −0.0011 |
| 0.00 | 611 | 157,732 | 0.9226 | 0.0455 | 0.9271 | +0.0009 | −0.0152 | −0.0006 |

### F4-44b6 (12 datasets)

| DIV_DIVERGE_UM | forks | edges | adj_J | div_J | score | d_score |
|---|---|---|---|---|---|---|
| **2.25 (shipped)** | 869 | 315,582 | 0.8353 | 0.0000 | 0.8353 | — |
| 1.50 | 1,184 | 315,897 | 0.8343 | 0.0000 | 0.8343 | −0.0010 |
| 1.00 | 1,231 | 315,944 | 0.8367 | 0.0000 | 0.8367 | +0.0014 |
| 0.50 | 1,231 | 315,944 | 0.8360 | 0.0000 | 0.8360 | +0.0008 |
| 0.00 | 1,231 | 315,944 | 0.8347 | 0.0000 | 0.8347 | −0.0006 |

### F4-zh (8 crops)

| DIV_DIVERGE_UM | forks | edges | adj_J | div_J | score | d_adj_J | d_div_J | d_score |
|---|---|---|---|---|---|---|---|---|
| **2.25 (shipped)** | 129 | 204,198 | 0.8118 | 0.0215 | 0.8140 | — | — | — |
| 1.50 | 376 | 204,445 | 0.8112 | 0.0352 | 0.8147 | −0.0006 | +0.0137 | +0.0007 |
| 1.00 | 638 | 204,707 | 0.8105 | 0.0355 | 0.8141 | −0.0013 | +0.0140 | +0.0001 |
| 0.50 | 780 | 204,849 | 0.8102 | 0.0276 | 0.8129 | −0.0016 | +0.0061 | −0.0010 |
| 0.00 | 780 | 204,849 | 0.8101 | 0.0208 | 0.8122 | −0.0017 | −0.0007 | −0.0017 |

### The three domains side by side (d_score against the shipped arm)

| DIV_DIVERGE_UM | F4-6bba | F4-44b6 | F4-zh |
|---|---|---|---|
| 1.50 | −0.0014 | −0.0010 | +0.0007 |
| 1.00 | −0.0018 | +0.0014 | +0.0001 |
| 0.50 | −0.0011 | +0.0008 | −0.0010 |
| 0.00 | −0.0006 | −0.0006 | −0.0017 |

Node counts are identical in every arm (the patch adds edges, not
nodes): 164,287 / 329,636 / 209,271.

Fork counts saturate at 611 / 1,231 / 780 by gate 1.00-0.50, below the
`DIV_GLOBAL_CAP` allowance of 616 / 1,236 / 784, so the cap is not what
stops them. At 0.50 and 0.00 on 6bba the fork COUNT and edge count are
identical while adj_J differs (0.9220 vs 0.9226) -- proposals are
ordered by `d_pq + DIV_W_SISTER * d_s`, not by divergence, and
`used_p` / `used_q` are first-come-first-served, so a larger proposal
pool can displace which forks are taken at the same total.

Cost: 108.7 min CPU, no GPU quota, no submission slot.

## 2026-09-02 -- ⏳ UNREVIEWED: divsteal, what a steal rule would break

> ⏳ **待分析.** Same marker as the three entries below; recorded while
> the user was away, tables only.

`divsteal.py` -> `biohub-divsteal`, CPU, 71.1 min. divgate2 with its 5a
branch instrumented: for every GT division rejected because the second
daughter already had a parent, the ranker's probability for the edge
`_add_safe_divisions` WOULD make against the edge the ILP DID make, the
two distances, and whether the competing edge is a real GT edge.

ANCHORS exact: F4-6bba adj_J 0.9216 / div_J 0.0606, F4-44b6 0.8353,
F4-zh 0.8118.

### F4-zh, 54 divisions rejected at 5a

| | |
|---|---|
| competing edge annotated at both ends | 27 (50.0%) |
| ... of those, a TRUE GT edge | **8 (29.6%)** |
| daughter had more than one parent | 0 |
| p_mother median (p25 / p75) | **0.182** (0.122 / 0.284) |
| p_other median (p25 / p75) | **0.765** (0.356 / 0.826) |
| p_mother > p_other | 10 of 48 scored (20.8%) |
| margin > 0.1 / 0.2 / 0.3 | 7 (14.6%) / 5 (10.4%) / 2 (4.2%) |
| d_mother median | 6.91 um |
| d_other median | 1.13 um |

### F4-6bba and F4-44b6, one case each

Neither competing edge has both endpoints annotated, so `true_other` is
not measurable in either domain.

| | 6bba (n=1) | 44b6 (n=1) |
|---|---|---|
| p_mother | 0.447 | 0.797 |
| p_other | 0.108 | 0.280 |
| d_mother | 10.49 um | 5.78 um |
| d_other | 1.78 um | 2.64 um |

Cost: 71.1 min CPU, no GPU quota, no submission slot.

## 2026-09-02 -- ⏳ UNREVIEWED: divgate2, which of the orphan-fork patch's eight conditions rejects a real division

> ⏳ **待分析. The user has not read this entry, divwhy or edgewhy.**
> Recorded at their request on 2026-09-02 while they were away. Tables
> as printed; no conclusion is filed against them.

`divgate2.py` -> `biohub-divgate2`, CPU, 58.2 min. divwhy plus a replay:
every GT division the ILP half-linked or did not link is walked through
`_add_safe_divisions`' conditions IN ORDER on the graph the patch
actually saw, and the FIRST condition that rejects it is counted.

The pre-patch graph is recovered without touching the pipeline: the
patch only appends, and its edges are the final edges absent from the
ILP solution whose endpoints are both primaries (a `_gap_close` bridge
always has a non-primary endpoint).

ANCHORS, all three exact: F4-6bba adj_J 0.9216 / div_J 0.0606, F4-44b6
0.8353 / 0.0000, F4-zh 0.8118 / 0.0215.

### First condition to reject

| condition | F4-zh (96) | F4-6bba (2) | F4-44b6 (2) |
|---|---|---|---|
| 5a sister is not an orphan | 54 (56.25%) | 1 | 1 |
| 8b divergence < 2.25 um | 33 (34.38%) | 1 | 1 |
| 0 sister deleted post-link | 6 (6.25%) | 0 | 0 |
| 2 mother has no single predecessor | 3 (3.12%) | 0 | 0 |

Conditions 1, 3, 4, 5b, 5c, 6, 7 rejected nothing in any domain.

`pending` counts match divwhy's buckets exactly: zh 95 `div_ilp_one_only`
+ 1 `div_ilp_neither` = 96; 6bba 2; 44b6 2.

### The divergence test, measured

For the divisions that survived every earlier condition: how much
further apart the two daughters' own children are than the daughters
themselves. `DIV_DIVERGE_UM` demands >= 2.25 um in one frame.

| domain | n | p10 | p25 | median | p75 | p90 | share >= 2.25 |
|---|---|---|---|---|---|---|---|
| F4-zh | 33 | 0.00 | 0.44 | 0.80 | 1.20 | 1.94 | **0.0%** |
| F4-6bba | 1 | — | — | -0.09 | — | — | 0.0% |
| F4-44b6 | 1 | — | — | 0.71 | — | — | 0.0% |

For comparison, on record: `div_kinematics` measured real daughters
separating at `sep_slope` 1.155 um/frame against 0.082 for random pairs.

Constants at the time: `DIV_CHILD_UM` 10.0, `DIV_PARENT_UM` 12.0,
`DIV_SISTER_UM` 15.0, `DIV_DIVERGE_UM` 2.25.

Cost: 58.2 min CPU, no GPU quota, no submission slot.

## 2026-09-02 -- ⏳ UNREVIEWED: divwhy, where every GT division is lost, and every emitted fork

> ⏳ **待分析.** See the marker on the divgate2 entry above.

`divwhy.py` -> `biohub-divwhy`, CPU, 56.8 min (v1) and again for v2.
F4 detections, three domains, one arm. Anchors exact in both versions:
F4-6bba adj_J 0.9216 / div_J 0.0606, F4-44b6 0.8353 / 0.0000, F4-zh
0.8118 / 0.0215 (recorded zh anchor was 0.8115; this run reads 0.8118).

**v1 had a classification flaw and v2 is the version to read.** v1 asked
the ILP solution first, so a division rescued by `_add_safe_divisions`
after linking was filed under `div_ilp_one_only`; v1 reported
`div_recovered` 1 on zh while its own fork table counted 20 true forks.
v2 asks the FINAL graph first and splits `div_recovered_ilp` from
`div_recovered_patch`. Every table below is v2.

### Where every GT division goes (first losing stage)

| bucket | F4-6bba (12) | F4-44b6 (2) | F4-zh (1,081) |
|---|---|---|---|
| div_mother_missed | 0 | 0 | 60 (5.55%) |
| div_daughter_missed | 0 | 0 | 708 (65.49%) |
| div_no_candidate (>10 um) | 6 (50.00%) | 0 | 33 (3.05%) |
| div_below_floor (<0.10) | 1 (8.33%) | 0 | 94 (8.70%) |
| div_node_absent | 1 (8.33%) | 0 | 71 (6.57%) |
| div_ilp_neither | 0 | 0 | 1 (0.09%) |
| div_ilp_one_only | 2 (16.67%) | 2 (100%) | 95 (8.79%) |
| div_killed_post | 0 | 0 | 0 |
| **div_recovered_ilp** | **0** | **0** | **1 (0.09%)** |
| **div_recovered_patch** | **2 (16.67%)** | **0** | **18 (1.67%)** |

1,095 GT divisions across the three domains; the ILP forked correctly on
one of them.

### Conditions per bucket (medians)

| domain / bucket | n | sister | md_max | md_min | dens10 | p_min | p_max | sc_mum | n_cand |
|---|---|---|---|---|---|---|---|---|---|
| 6bba div_no_candidate | 6 | 11.87 | 7.72 | 4.11 | 1.00 | — | 0.944 | 0.397 | 1.00 |
| 6bba div_below_floor | 1 | 11.98 | 6.92 | 5.15 | 3.00 | 0.009 | 0.972 | 0.406 | 3.00 |
| 6bba div_node_absent | 1 | 11.55 | 8.26 | 4.91 | 1.00 | 0.247 | 0.331 | 0.464 | 2.00 |
| 6bba div_ilp_one_only | 2 | 9.72 | 7.17 | 3.57 | 1.00 | 0.433 | 0.936 | 0.462 | 2.00 |
| 6bba div_recovered_patch | 2 | 6.34 | 5.18 | 2.14 | 1.00 | 0.311 | 0.677 | 0.456 | 2.00 |
| 44b6 div_ilp_one_only | 2 | 7.84 | 6.18 | 2.09 | 3.00 | 0.251 | 0.849 | 0.397 | 3.50 |
| zh div_mother_missed | 60 | 6.86 | 4.98 | 2.52 | 3.00 | — | — | — | — |
| zh div_daughter_missed | 708 | 6.36 | 4.32 | 2.52 | 3.00 | — | — | 0.320 | 3.00 |
| zh div_no_candidate | 33 | 10.31 | 7.95 | 2.84 | 2.00 | — | 0.927 | 0.301 | 2.00 |
| zh div_below_floor | 94 | 8.22 | 5.61 | 2.79 | 3.00 | 0.040 | 0.871 | 0.303 | 4.00 |
| zh div_node_absent | 71 | 7.49 | 5.11 | 2.84 | 3.00 | 0.239 | 0.810 | 0.303 | 4.00 |
| zh div_ilp_one_only | 95 | 7.94 | 5.11 | 3.10 | 4.00 | 0.175 | 0.835 | 0.309 | 4.00 |
| zh div_recovered_ilp | 1 | 7.69 | 4.21 | 4.11 | 2.00 | 0.583 | 0.842 | 0.313 | 5.00 |
| zh div_recovered_patch | 18 | 6.37 | 4.00 | 3.01 | 3.00 | 0.222 | 0.755 | 0.429 | 4.00 |

`sister` = daughter separation in the frame after the division. `md_max`
/ `md_min` = the longer / shorter mother-to-daughter step. `dens10` =
primaries within 10 um of the mother. `p_min` / `p_max` = the two
mother-daughter ranker probabilities. `n_cand` = candidates the mother
had at all.

A `merge_um` column (missed daughter to the primary that matched its
sister, testing whether one peak served both) was computed and **is
missing from the printout** -- the header stops at `n_cand`. Not
measured.

### The fork price: p_min against DIV_W 0.55

| p_min | F4-zh n | recovered | one_only | other |
|---|---|---|---|---|
| 0.00-0.10 | 94 | 0.0% | 0.0% | 100.0% |
| 0.10-0.30 | 150 | 10.0% | 54.7% | 35.3% |
| 0.30-0.55 | 32 | 9.4% | 37.5% | 53.1% |
| 0.55-0.80 | 4 | 25.0% | 25.0% | 50.0% |

6bba, 6 divisions with both pairs scored: 0.10-0.30 n=2 (50% recovered),
0.30-0.55 n=3 (33.3% recovered, 66.7% one_only), 0.00-0.10 n=1.

### Recovery by sister separation (v1 numbers; v2 shifts recovery only)

| sister um | F4-zh n | recovered | detect-lost | link-lost |
|---|---|---|---|---|
| 0-3 | 2 | 0.0% | 100.0% | 0.0% |
| 3-5 | 88 | 0.0% | 96.6% | 3.4% |
| 5-8 | 680 | 0.1% | 77.4% | 22.5% |
| 8-12 | 293 | 0.0% | 50.5% | 49.5% |
| 12+ | 18 | 0.0% | 38.9% | 61.1% |

6bba v2: 5-8 um n=4 50% recovered; 8-12 n=5 and 12+ n=3 both 0%.

### Every fork the pipeline emits

| domain | forks | true | false | ILP | BRIDGE | DIVPATCH |
|---|---|---|---|---|---|---|
| F4-6bba | 574 | 2 | 572 | 114 (0.0% true) | 4 | 456 (0.4% true) |
| F4-44b6 | 1,029 | 0 | 1,029 | 160 (0.0%) | 8 | 861 (0.0%) |
| F4-zh | 173 | 20 | 153 | 44 (2.3%) | 1 | 128 (14.8%) |

Cap arithmetic, not binding in any domain: `DIV_GLOBAL_CAP` 0.00375 x
final nodes = 616 (6bba, 164,287 nodes) / 1,236 (44b6, 329,636) / 784
(zh, 209,271), against 456 / 861 / 128 forks actually added.

Cost: 56.8 min CPU per version, no GPU quota, no submission slot.

### ⚠ Denominators, the user's correction on reading these tables

The user, 2026-09-02: on Zebrahub every cell is annotated and the
detector's recall there is about 0.80, so **any** question asked of that
domain returns "detection" -- the 65.5% `div_daughter_missed` is a
statement about zh's detector recall, not about divisions, and does not
transfer. The competition domains carry the opposite bias: a division
enters their GT only if a human saw the mother and both daughters, so
their rows can never show a detection loss (0 of 14 here). `div_J`'s
denominator is annotated divisions, which is the competition-like
population.

## 2026-09-02 -- ⏳ UNREVIEWED: edgewhy, the conditions under which an edge is lost or wrongly linked

> ⏳ **待分析.** See the marker on the divgate2 entry above.

`edgewhy.py` -> `biohub-edgewhy`, CPU, 68.4 min. f4diag's edge buckets
plus per-edge covariates, F4 arm, three domains. Anchors exact: F4-6bba
adj_J 0.9216, F4-44b6 0.8353, F4-zh 0.8118.

### Missed edges: medians per bucket

`disp` = the GT displacement. `dens10` = primaries within 10 um of the
source. `gap2` = distance from the true target to the NEAREST OTHER
primary. `p_true` / `rank` = the ranker's probability for the correct
pair and its position in the source's candidate list. `p_top` = the
probability of whatever the ranker liked best.

| F4-6bba bucket | n | disp | p90 disp | dens10 | gap2 | z_um | sc_src | p_true | rank | p_top | %div |
|---|---|---|---|---|---|---|---|---|---|---|---|
| det_miss_subthresh | 146 | 1.72 | 4.20 | 1.00 | 12.23 | 39.81 | 0.19 | — | — | 0.53 | 0.0% |
| det_miss_cold | 146 | 1.72 | 4.24 | 2.00 | 9.54 | 68.25 | 0.31 | — | — | 0.82 | 0.0% |
| no_candidate | 78 | 3.36 | 9.98 | 2.00 | 7.25 | 53.62 | 0.37 | — | — | 0.94 | 7.7% |
| below_floor | 77 | 2.37 | 8.25 | 3.00 | 6.89 | 69.88 | 0.35 | 0.03 | 1.0 | 0.86 | 1.3% |
| node_absent | 73 | 2.03 | 4.82 | 3.00 | 7.52 | 73.12 | 0.28 | 0.56 | 0.0 | 0.73 | 1.4% |
| ilp_rejected | 74 | 2.07 | 6.08 | 2.00 | 7.43 | 66.62 | 0.34 | 0.31 | 1.0 | 0.79 | 5.4% |
| killed_post | 4 | 2.23 | 3.14 | 1.00 | 11.13 | 54.44 | 0.17 | 0.88 | 0.0 | 0.88 | 0.0% |
| **recovered** | 9,715 | 1.82 | 3.92 | 1.00 | 13.05 | 52.00 | 0.44 | 0.99 | 0.0 | 0.99 | 0.1% |

| F4-44b6 bucket | n | disp | p90 disp | dens10 | gap2 | sc_src | p_true | rank | p_top |
|---|---|---|---|---|---|---|---|---|---|
| det_miss_subthresh | 60 | 1.72 | 2.88 | 2.00 | 10.28 | 0.16 | — | — | 0.22 |
| det_miss_cold | 99 | 1.82 | 3.35 | 2.00 | 8.80 | 0.26 | — | — | 0.57 |
| no_candidate | 43 | 3.28 | 12.24 | 3.00 | 7.60 | 0.28 | — | — | 0.88 |
| below_floor | 71 | 2.19 | 7.93 | 3.00 | 6.45 | 0.30 | 0.04 | 1.0 | 0.83 |
| node_absent | 66 | 1.72 | 4.30 | 3.00 | 7.05 | 0.30 | 0.58 | 0.0 | 0.82 |
| ilp_rejected | 77 | 1.72 | 3.16 | 3.00 | 7.24 | 0.31 | 0.31 | 1.0 | 0.80 |
| killed_post | 2 | 2.12 | 2.79 | 2.00 | 9.59 | 0.17 | 0.87 | 0.0 | 0.87 |
| **recovered** | 2,472 | 1.72 | 3.17 | 3.00 | 8.06 | 0.36 | 0.94 | 0.0 | 0.94 |

| F4-zh bucket | n | disp | p90 disp | dens10 | gap2 | sc_src | p_true | rank | p_top | %div |
|---|---|---|---|---|---|---|---|---|---|---|
| det_miss_subthresh | 4,479 | 1.32 | 2.56 | 2.00 | 8.74 | 0.17 | — | — | 0.18 | 1.5% |
| det_miss_cold | 33,178 | 1.32 | 2.56 | 3.00 | 7.75 | 0.26 | — | — | 0.66 | 2.4% |
| no_candidate | 118 | 6.74 | 10.21 | 3.00 | 7.45 | 0.27 | — | — | 0.89 | 29.7% |
| below_floor | 583 | 2.91 | 7.18 | 4.00 | 6.43 | 0.27 | 0.04 | 1.0 | 0.84 | 17.3% |
| node_absent | 2,107 | 1.39 | 2.94 | 4.00 | 6.74 | 0.25 | 0.79 | 0.0 | 0.81 | 4.1% |
| ilp_rejected | 753 | 2.19 | 5.10 | 4.00 | 6.55 | 0.28 | 0.34 | 1.0 | 0.81 | 16.1% |
| killed_post | 433 | 1.24 | 2.01 | 4.00 | 6.70 | 0.23 | 0.83 | 0.0 | 0.83 | 0.0% |
| **recovered** | 179,589 | 1.32 | 2.01 | 4.00 | 7.62 | 0.32 | 0.93 | 0.0 | 0.93 | 0.5% |

Totals: 6bba 10,313 adjacent GT edges, 598 lost. 44b6 2,890, 418 lost.
zh 221,240, 41,651 lost.

### Loss rate by displacement

| disp um | 6bba n | 6bba lost | 44b6 n | 44b6 lost | zh n | zh lost |
|---|---|---|---|---|---|---|
| 0-1 | 2,917 | 5.04% | 690 | 13.33% | 91,120 | 16.37% |
| 1-2 | 2,966 | 5.06% | 1,080 | 12.31% | 103,195 | 18.07% |
| 2-3 | 2,098 | 5.15% | 757 | 13.74% | 20,576 | 24.14% |
| 3-4 | 1,289 | 6.44% | 226 | 15.49% | 3,880 | 37.99% |
| 4-6 | 804 | 5.35% | 93 | 18.28% | 1,837 | 58.85% |
| 6-8 | 159 | 16.98% | 16 | 56.25% | 429 | 84.62% |
| 8+ | 80 | 50.00% | 28 | 100.00% | 203 | 98.52% |

At 8+ um on 6bba, 22.50% of edges land in `no_candidate` and 12.50% in
`below_floor`; on 44b6, 50.00% and 25.00%.

### Loss rate by crowding (primaries within 10 um of the source)

| dens10 | 6bba n | 6bba lost | 44b6 n | 44b6 lost | zh n | zh lost |
|---|---|---|---|---|---|---|
| 0-2 | 7,336 | 3.49% | 358 | 22.35% | 19,407 | 27.17% |
| 2-5 | 2,794 | 11.02% | 2,213 | 13.42% | 119,795 | 21.05% |
| 5-10 | 183 | 18.58% | 319 | 12.85% | 81,948 | 13.60% |
| 10-20 | — | — | — | — | 90 | 10.00% |

### Was the correct pair ranked first?

Over GT edges whose both endpoints were detected and whose correct pair
was a candidate.

| rank | 6bba n | share | lost | med p_true | med p_top |
|---|---|---|---|---|---|
| 0 | 9,770 | 98.26% | 0.81% | 0.988 | 0.988 |
| 1 | 146 | 1.47% | 84.93% | 0.148 | 0.793 |
| 2 | 20 | 0.20% | 90.00% | 0.040 | 0.798 |
| 3-4 | 7 | 0.07% | 100.00% | 0.007 | 0.814 |

44b6: rank 0 n=2,536 (94.35%) lost 3.31%; rank 1 n=119 lost 84.03%;
rank 2 n=26 lost 96.15%; rank 3-4 n=7 lost 100%.
zh: rank 0 n=181,955 (99.18%) lost 1.38%; rank 1 n=1,255 lost 88.21%;
rank 2 n=190 lost 98.95%; rank 3-4 n=61 lost 100%; rank 5+ n=4 lost 100%.

### Final edges classified, with provenance

`prov`: ILP chose it from ranked candidates, BRIDGE is a `_gap_close`
insert, DIVPATCH is `_add_safe_divisions`' second daughter.

| domain / class | n | len_um | p90 len | p | dens10 | err_um | ILP | BRIDGE | DIVPATCH |
|---|---|---|---|---|---|---|---|---|---|
| 6bba tp_like | 9,839 | 1.38 | 3.23 | 0.988 | 1.00 | — | 99.76% | 0.17% | 0.07% |
| 6bba fp_confusion | 10 | 2.49 | 4.87 | 0.947 | 1.00 | — | 100.00% | 0.00% | 0.00% |
| 6bba fp_half | 843 | 1.64 | 3.41 | 0.943 | 2.00 | — | 97.03% | 0.71% | 2.25% |
| 6bba fp_invisible | 146,889 | 1.73 | 3.51 | 0.956 | 2.00 | — | 98.92% | 0.79% | 0.30% |
| 44b6 tp_like | 2,596 | 1.25 | 2.38 | 0.942 | 3.00 | — | 98.96% | 1.00% | 0.04% |
| 44b6 fp_confusion | 9 | 2.17 | 3.03 | 0.576 | 2.00 | 5.28 | 100.00% | 0.00% | 0.00% |
| 44b6 fp_half | 293 | 1.55 | 2.91 | 0.863 | 3.00 | — | 97.27% | 1.02% | 1.71% |
| 44b6 fp_invisible | 312,684 | 1.33 | 2.83 | 0.920 | 3.00 | — | 97.99% | 1.73% | 0.28% |
| zh tp_like | 182,777 | 0.62 | 1.06 | 0.926 | 4.00 | — | 99.24% | 0.74% | 0.01% |
| zh fp_confusion | 3,831 | 0.93 | 2.35 | 0.854 | 3.00 | 4.53 | 95.35% | 2.87% | 1.77% |
| zh fp_half | 4,877 | 0.64 | 1.38 | 0.921 | 3.00 | — | 97.19% | 2.30% | 0.51% |
| zh fp_invisible | 12,713 | 0.62 | 1.21 | 0.926 | 2.00 | — | 97.61% | 2.30% | 0.09% |

Final edge totals: 6bba 157,581; 44b6 315,582; zh 204,198.

### Wrong-link error distance (fp_confusion, distance to the correct target)

| distance | F4-zh (3,193) | F4-44b6 (9) |
|---|---|---|
| 0-3 um | 231 (7.23%) | 0 |
| 3-5 um | 1,748 (54.74%) | 4 |
| 5-8 um | 1,039 (32.54%) | 3 |
| 8-12 um | 160 (5.01%) | 1 |
| 12+ um | 15 (0.47%) | 1 |

6bba's 10 wrong links produced no error-distance rows (no correct target
resolved).

Cost: 68.4 min CPU, no GPU quota, no submission slot.

## 2026-09-01 -- f4diag: F4 traced against BASE, nodes and edges, every stage

`f4diag.py` -> `biohub-f4diag`, CPU, 59.1 min. edgediag with the
detection source swapped to F4's caches; BASE and F4 side by side, D187
both (what ref 55889500 ships), 6bba + 44b6, no zebrahub.

ANCHORS. BASE reproduces edgediag bucket by bucket: 6bba
1.06/1.33/0.70/0.78/0.71/0.69/0.04/94.71, 44b6
4.01/2.98/1.11/2.32/2.39/2.39/0.21/84.60; adj_J 0.9293 / 0.8185;
fp_invisible 93.3% / 99.0%. Primary counts match fatescore's F4 rows
(171,450 / 354,086).

### Official scorer

| | adj_J | div_J | score |
|---|---|---|---|
| BASE-6bba | 0.9293 | 0.1316 | 0.9424 |
| F4-6bba | 0.9216 | 0.0606 | 0.9277 |
| BASE-44b6 | 0.8185 | 0.0000 | 0.8185 |
| F4-44b6 | 0.8353 | 0.0000 | 0.8353 |

### Nodes, stage by stage

| arm-domain | detections | primaries >=0.15 | band [0.05,0.15) | in graph | ILP-linked | killed post | final | +repair |
|---|---|---|---|---|---|---|---|---|
| BASE-6bba | 188,998 | 173,269 | 15,729 | 96.2% | 96.2% | 0.7% | 166,081 | 622 |
| F4-6bba | 188,100 | 171,450 | 16,650 | 96.2% | 96.2% | 0.7% | 164,287 | 589 |
| BASE-44b6 | 441,478 | 336,948 | 104,530 | 92.9% | 92.9% | 1.2% | 311,865 | 2,781 |
| F4-44b6 | 437,884 | 354,086 | 83,798 | 93.3% | 93.3% | 1.0% | 329,636 | 2,725 |

`in graph` equals `ILP-linked` in every row: the solver drops nodes it
does not link, so they never reach the geff.

Final-output degree -- BASE-6bba deg0 0.0% / deg1 8.6% / deg2 91.1% /
deg3+ 0.3%; F4-6bba 0.0 / 8.5 / 91.1 / 0.3; BASE-44b6 0.0 / 9.2 / 90.5 /
0.3; F4-44b6 0.0 / 8.8 / 90.8 / 0.3.

### GT-node ledger

matched = a primary within the match radius; subthresh = no primary but
a 0.05-0.15 candidate is there; cold = nothing there.

| arm-domain | GT nodes | matched | subthresh | cold |
|---|---|---|---|---|
| BASE-6bba | 10,699 | 10,516 (98.3%) | 88 (0.8%) | 95 (0.9%) |
| F4-6bba | 10,699 | 10,464 (97.8%) | 121 (1.1%) | 114 (1.1%) |
| BASE-44b6 | 2,947 | 2,802 (95.1%) | 81 (2.7%) | 64 (2.2%) |
| F4-44b6 | 2,947 | 2,841 (96.4%) | 40 (1.4%) | 66 (2.2%) |

### Every GT edge, charged to the first stage that loses it

6bba 10,313 adjacent GT edges, 0 non-adjacent; 44b6 2,890, 0.

| bucket | BASE-6bba | F4-6bba | BASE-44b6 | F4-44b6 |
|---|---|---|---|---|
| det_miss_subthresh | 109 (1.06%) | 146 (1.42%) | 116 (4.01%) | 60 (2.08%) |
| det_miss_cold | 137 (1.33%) | 146 (1.42%) | 86 (2.98%) | 99 (3.43%) |
| no_candidate (>10um) | 72 (0.70%) | 78 (0.76%) | 32 (1.11%) | 43 (1.49%) |
| below_floor (<0.10) | 80 (0.78%) | 77 (0.75%) | 67 (2.32%)* | 71 (2.46%)* |
| node_absent | 73 (0.71%) | 73 (0.71%) | 69 (2.39%) | 66 (2.28%) |
| ilp_rejected | 71 (0.69%) | 74 (0.72%) | 69 (2.39%) | 77 (2.66%) |
| killed_post | 4 (0.04%) | 4 (0.04%) | 6 (0.21%) | 2 (0.07%) |
| recovered | 9,767 (94.71%) | 9,715 (94.20%) | 2,445 (84.60%) | 2,472 (85.54%) |

*44b6 below_floor is undercounted: D187 is in-sample there.

Cross-checks vs official edge_fn: BASE-6bba 546/411, F4-6bba 598/474,
BASE-44b6 445/349, F4-44b6 418/290.

### Final edges classified against GT

| | BASE-6bba | F4-6bba | BASE-44b6 | F4-44b6 |
|---|---|---|---|---|
| final edges | 159,262 | 157,581 | 298,049 | 315,582 |
| tp_like | 9,902 | 9,839 | 2,542 | 2,596 |
| fp_confusion | 13 | 10 | 7 | 9 |
| fp_half | 826 | 843 | 291 | 293 |
| fp_invisible | 148,521 (93.26%) | 146,889 (93.21%) | 295,209 (99.05%) | 312,684 (99.08%) |
| official edge_fp | 363 | 385 | 258 | 256 |
| official edge_tp | 9,902 | 9,839 | 2,541 | 2,600 |

Cost: 59.1 min CPU, no GPU quota, no submission slot.

## 2026-09-01 -- fatescore: M15's added points traced through every stage

`fatescore.py` -> `biohub-fatescore`, CPU, 117.4 min. mergescore with a
per-node trace; two arms (F4 control, M15), three domains. Built on
edgediag's (t, i) -> graph-id map by rounded coordinates and on
snapshots between the post-link stages.

ANCHORS, all six exact: F4 0.9198 / 0.8303 / 0.8115, M15 0.9134 /
0.8363 / 0.8219 (mergescore's rows).

Groups: base = F4's own detections; merged = the ep05 points M15 adds;
repair = nodes `_gap_close` created from the sub-threshold pool.
Percentages are of each group's own primary count.

| arm / domain | group | primary | in graph | in_pair | past floor | ILP | gap+ | short- | div+ | final | isolated | mean deg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F4 / 6bba | base | 171,450 | 96.0% | 99.8% | 99.6% | 96.0% | 0.7% | 0.6% | 0.5% | 95.3% | 0.0% | 1.92 |
| F4 / 6bba | repair | 589 | 100.0% | - | - | - | - | - | - | 100.0% | 0.0% | 2.00 |
| M15 / 6bba | base | 171,450 | 96.4% | 99.8% | 99.7% | 96.4% | 0.6% | 0.6% | 0.5% | 95.8% | 0.0% | 1.92 |
| M15 / 6bba | merged | 7,653 | 59.1% | 98.1% | 97.1% | 59.1% | 1.7% | 2.8% | 1.3% | 56.3% | 0.0% | 1.74 |
| M15 / 6bba | repair | 555 | 100.0% | - | - | - | - | - | - | 100.0% | 0.0% | 2.01 |
| F4 / 44b6 | base | 354,086 | 93.0% | 99.7% | 99.4% | 93.0% | 1.5% | 1.0% | 0.5% | 92.1% | 0.0% | 1.91 |
| F4 / 44b6 | repair | 2,680 | 100.0% | - | - | - | - | - | - | 100.0% | 0.0% | 2.00 |
| M15 / 44b6 | base | 354,086 | 93.7% | 99.7% | 99.4% | 93.7% | 1.4% | 0.8% | 0.5% | 92.9% | 0.0% | 1.92 |
| M15 / 44b6 | merged | 17,263 | 59.2% | 98.3% | 97.4% | 59.2% | 3.1% | 1.8% | 1.0% | 57.4% | 0.0% | 1.74 |
| M15 / 44b6 | repair | 2,706 | 100.0% | - | - | - | - | - | - | 100.0% | 0.0% | 2.00 |
| F4 / zh | base | 216,274 | 96.7% | 99.9% | 99.8% | 96.7% | 0.9% | 0.4% | 0.1% | 96.3% | 0.0% | 1.95 |
| F4 / zh | repair | 936 | 100.0% | - | - | - | - | - | - | 100.0% | 0.0% | 2.00 |
| M15 / zh | base | 216,274 | 97.2% | 99.9% | 99.8% | 97.2% | 0.8% | 0.3% | 0.1% | 96.9% | 0.0% | 1.95 |
| M15 / zh | merged | 4,642 | 67.9% | 100.0% | 99.8% | 67.9% | 3.4% | 1.9% | 0.6% | 66.1% | 0.0% | 1.79 |
| M15 / zh | repair | 902 | 100.0% | - | - | - | - | - | - | 100.0% | 0.0% | 2.00 |

`in graph` equals `ILP` in all nine rows: the solver drops nodes it does
not link, so `isolated` is 0 by construction. Seeds that never reached
the solved graph, summed over arms and domains: 84,762.

### Final-output degree histogram (3 = 3 or more)

| arm / domain / group | n | deg0 | deg1 | deg2 | deg3 |
|---|---|---|---|---|---|
| F4 / 6bba / base | 163,477 | 0.0% | 8.6% | 91.1% | 0.3% |
| M15 / 6bba / base | 164,279 | 0.0% | 8.1% | 91.5% | 0.3% |
| M15 / 6bba / merged | 4,305 | 0.0% | 26.0% | 73.8% | 0.3% |
| F4 / 44b6 / base | 326,068 | 0.0% | 9.0% | 90.7% | 0.3% |
| M15 / 44b6 / base | 329,056 | 0.0% | 8.3% | 91.4% | 0.3% |
| M15 / 44b6 / merged | 9,913 | 0.0% | 26.0% | 73.7% | 0.2% |
| F4 / zh / base | 208,224 | 0.0% | 5.0% | 95.0% | 0.1% |
| M15 / zh / base | 209,487 | 0.0% | 4.6% | 95.3% | 0.1% |
| M15 / zh / merged | 3,068 | 0.0% | 21.5% | 78.3% | 0.1% |

### Base points linked by the ILP, F4 vs M15

Same points, same (t, i) indices in both arms.

| domain | F4 | M15 | lost | gained | net |
|---|---|---|---|---|---|
| 44b6 | 329,467 | 331,941 | 2,619 | 5,093 | +2,474 |
| 6bba | 164,572 | 165,251 | 913 | 1,592 | +679 |
| zh | 209,088 | 210,199 | 444 | 1,555 | +1,111 |

Largest per-dataset movers: 44b6_a2bb48bb lost 1,134 / gained 1,877
(+743), 6bba_3abfe10a lost 490 / gained 903 (+413), 44b6_0b24845f lost
378 / gained 565 (+187). 6bba_283bf9f1 is the only negative net (-3).

Cost: 117.4 min CPU, no GPU quota, no submission slot.

## 2026-08-31 -- M15 submitted as ref 55911058: LB 0.917

Local (mergescore, D32, adj_J delta vs F4): 6bba -0.0064, 44b6 +0.0060,
zh +0.0104. Visible-4 nodes 130,732 vs F4's 128,157 (+2.01%); 3,380 ep05
points added on 164,014 F4 detections (2.06%). Hidden output
280,776,965 bytes vs F4's 270,740,009 (+3.71%).

**LB 0.917**, against F4's 0.925. Lowest score on record.

Per-dataset merge counts in the visible four: 44b6_0113de3b +85 on
25,816 (0.33%), 44b6_0b24845f +1,607 on 56,067 (2.87%), 6bba_05b6850b
+139 on 7,310 (1.90%), 6bba_05db0fb1 +1,549 on 74,821 (2.07%).

`mergecheck` (CPU) verified the submitted arm against the offline one
before submitting: 44b6_0b24845f reads +1,607 on 56,067 = 2.87% on both
sides, equal to the integer. Its per-dataset add rate over the 24 cached
competition datasets ranges 1.17% to 9.31%, median 3.56%, and reproduced
both domain totals exactly (44b6 3.94%, 6bba 4.07%).

Cost: one submission slot, ~25 min GPU for the instrumented rerun, one
CPU kernel.

## 2026-08-31 -- W1101 submitted as ref 55904403: LB 0.924

Single variable vs F4: the four-map average becomes
(h0+h1+0.5*h2+h3)/3.5, branch 3 demoted to half weight.

| | F4 (55889500) | W1101 (55904403) |
|---|---|---|
| LB | 0.925 | 0.924 |
| local 6bba vs BASE | -0.0053 | -0.0108 |
| local 44b6 | +0.0054 | +0.0054 |
| local zh | +0.0027 | +0.0024 |
| visible-4 nodes | 128,157 | 132,646 |
| hidden output bytes | 270,740,009 | 273,340,932 |

The user's pre-registration, written before the result: LOWER than 0.925
= the 6bba negativity has a real component beyond memorization; EQUAL =
the pure-artifact reading holds; HIGHER = uninformative.

Cost: one submission slot, no GPU quota.

## 2026-08-31 -- mergescore: dcmerge reopened with our own ep05

`mergescore.py` -> `biohub-mergescore`, CPU, 327.6 min. F4 detections as
the base; an ep05 point is admitted iff its own score >= tau AND it is
> 4.0 um from every F4 point at that frame, entering at score
UNET_THRESH as a primary. Five arms x three domains, one ILP solve per
arm per dataset. D32 ranker for every judged arm; BASE_D187 as the
harness anchor.

The rerun uses OUR ep05 checkpoint. dcmerge (2026-08-28) used pilkwang's
published DeepCenter, which left 44b6 -- dctrain's excluded set -- nan.

ANCHORS, both exact: BASE_D187 6bba adj_J 0.9293 (dcpool's), F4 x D32
0.9198 / 0.8303 / 0.8115 (fusescore's).

### adj_J delta vs F4, both on D32

| arm | 6bba | 44b6 | zh |
|---|---|---|---|
| M50 (tau 0.50) | +0.0000 | +0.0000 | +0.0000 |
| M30 (tau 0.30) | -0.0021 | -0.0010 | +0.0057 |
| M15 (tau 0.15) | -0.0064 | +0.0060 | +0.0104 |

div_J delta vs F4: M50 0/0/0, M30 +0.0023/+0.0000/+0.0009, M15
-0.0022/+0.0000/+0.0041.

### Decomposition

| domain | arm | edge_J | node_ratio | adj_J | div_J |
|---|---|---|---|---|---|
| 6bba | BASE_D187 | 0.9275 | -0.0179 | 0.9293 | 0.1316 |
| 6bba | F4 | 0.9178 | -0.0213 | 0.9198 | 0.0667 |
| 6bba | M50 | 0.9178 | -0.0213 | 0.9198 | 0.0667 |
| 6bba | M30 | 0.9166 | -0.0122 | 0.9178 | 0.0690 |
| 6bba | M15 | 0.9141 | +0.0092 | 0.9134 | 0.0645 |
| 44b6 | BASE_D187 | 0.8072 | -0.1463 | 0.8185 | 0.0000 |
| 44b6 | F4 | 0.8214 | -0.1066 | 0.8303 | 0.0000 |
| 44b6 | M50 | 0.8214 | -0.1066 | 0.8303 | 0.0000 |
| 44b6 | M30 | 0.8210 | -0.1007 | 0.8293 | 0.0000 |
| 44b6 | M15 | 0.8299 | -0.0738 | 0.8363 | 0.0000 |
| zh | BASE_D187 | 0.8036 | nan | 0.8093 | 0.0230 |
| zh | F4 | 0.8052 | nan | 0.8115 | 0.0186 |
| zh | M50 | 0.8052 | nan | 0.8115 | 0.0186 |
| zh | M30 | 0.8115 | nan | 0.8173 | 0.0196 |
| zh | M15 | 0.8169 | nan | 0.8219 | 0.0227 |

Splitting M15's adj_J delta into its two factors: node term -0.0028 and
edge_J +0.0085 on 44b6; -0.0014 and +0.0118 on zh; -0.0028 and -0.0037
on 6bba. The node term costs the two competition domains the same.

### Points added

| tau | 44b6 | 6bba | zh |
|---|---|---|---|
| 0.50 | +0 on 437,884 (0.00%) | +0 on 188,100 (0.00%) | +1 on 228,665 (0.00%) |
| 0.30 | +2,810 (0.64%) | +2,053 (1.09%) | +2,196 (0.96%) |
| 0.15 | +17,263 (3.94%) | +7,653 (4.07%) | +4,642 (2.03%) |

`mergecheck` (CPU, seconds) later reproduced both tau 0.15 domain totals
exactly and gave the per-dataset spread: 1.17% to 9.31%, p25 2.83%,
median 3.56%, p75 4.64%. 37% of 44b6's added points come from
44b6_a2bb48bb alone (6,456 of 17,263) and 37% of 6bba's from
6bba_3abfe10a (2,815 of 7,653). Add rate does not track density:
44b6_a2bb48bb 740 pts/frame adds 8.73% while 44b6_c96cfa10 at 481 adds
1.25%; 6bba_7af54fde at 113 adds 9.31% while 6bba_afb141ff at 68 adds
1.17%.

### Per-dataset, F4 vs M30 (both D32)

Recorded for the M30 arm only; M15 has no per-dataset detail.

6bba: 2312ac41 0.9950/0.9895 (-0.0054), 268e1230 0.8888/0.8846
(-0.0042), 283bf9f1 0.9363/0.9336 (-0.0028), 3a1849c2 0.8609/0.8560
(-0.0049), 3abfe10a 0.7761/0.7742 (-0.0019), 5c824876 0.9275/0.9248
(-0.0027), 61dd1e0d 0.9483/0.9544 (+0.0061), 7af54fde 0.8956/0.9014
(+0.0058), 7b5d3b2c 0.9956/0.9948 (-0.0007), aeee7805 0.9935/0.9842
(-0.0093), afb141ff 0.9303/0.9300 (-0.0003), c27cba08 0.9821/0.9804
(-0.0017).

44b6: 0b24845f 0.9120/0.9120 (-0.0001), 0db75fae 0.9115/0.9109
(-0.0006), 33b596bf 0.9028/0.9408 (+0.0380), 5740d24b 1.0151/1.0026
(-0.0126), 87bba6c4 0.5587/0.5740 (+0.0153), 8f5ab931 0.8584/0.8604
(+0.0020), 9bfa6a0a 0.8843/0.8680 (-0.0163), a2bb48bb 0.6286/0.6271
(-0.0015), c15fded2 0.8285/0.8271 (-0.0014), c96cfa10 0.8800/0.8769
(-0.0031), d2f34f90 0.9433/0.9429 (-0.0004), e29f0176 0.8071/0.8064
(-0.0007).

### Faithfulness note for anyone rebuilding this

The dcdet_ep05 caches peak at CAND_THR 0.05, refine, run the 4 um NMS,
and only THEN filter score >= tau. Thresholding before the NMS is a
different arm: a 0.16 peak would survive a 0.20 neighbour that
suppressed it in the cache.

Cost: 327.6 min CPU, no GPU quota.

## 2026-08-31 -- F4 submitted as ref 55889500: LB 0.925, new best

F4 = detfuse2's arm: the ep05 epoch-5 DeepCenter heatmap added as a
FOURTH equal-weight map before peak finding, full 4-flip TTA. Single
variable vs ref 55817630 (LB 0.922); everything else byte-identical
(D187 ranker, global solve, GAP1_CAP_FRAC 0.02 nocand, SHORT_MIN 4,
UNET_THRESH 0.15).

| | local 6bba | local 44b6 | local zh | LB |
|---|---|---|---|---|
| F4 vs BASE (D32) | -0.0053 | +0.0054 | +0.0027 | +0.003 |

**0.922 -> 0.925.** Largest single gain on record.

Visible-4 nodes 128,157 vs 130,210 (-1.6%); tracking 914s vs 947s.
Hidden output 268,154,477 -> 270,740,009 bytes (+0.96%).

The submission was made to arbitrate the user's observation (2026-08-30)
that the xl weights' training split is unknown (holdout-split.md:
`fold: 0`, no file list) and plausibly contains the 6bba eval sets,
while the hidden test is a third embryo (host-confirmed, no embryo_id
overlap). Pre-registered reading, written before the result: >0.922 =
the 6bba veto was memorization; 0.920-0.922 = uninformative to mildly
real; <=0.918 = the 6bba negativity is real.

Costs: one daily submission slot, 12.3 min GPU (dcsample), ~56 min
visible run.

## 2026-08-30 -- ⭐⭐⭐ dc_gate + vetoscore: THE PEAK-LIST PROXY FOR "DC IS SILENT" IS BROKEN; the heatmap veto the user designed is still untested and dcsample is building its cache.

The user's split of the fusion idea: xl leads, DeepCenter speaks only
at extremes. `dc_gate.py` (local, judge_rows.npz, zero GPU) split it
further: RESCUE is dead (xl-dropped candidates with ep05 >= 0.15 are
only 24.3% real, 29.8% at 0.30, and ep05 >= 0.9 comp positives are
0.0% -- domain shift flattens DC's confidence on comp), but VETO
looked strong: among xl-kept candidates (mean3 >= 0.15), those with
ep05 window-max hd < 0.02 are 90.2% fake at 0.31% collateral (6bba
comp positives killed: 0.19%). Gated AUC (0.742) loses to DCheavy
(0.763) -- but ranking was never the veto's niche; not moving/adding
points is.

`biohub-vetoscore` v1 (219.2 min CPU, anchor exact in all 3 domains)
then tested a CHEAP PROXY of the veto -- drop xl points with no dcdet
ep05 PEAK (score >= tau) within 3 um -- and the proxy failed hard and
uniformly:

| arm | 6bba | 44b6 | zh | verdict |
|---|---|---|---|---|
| V05 | -0.0619 | -0.0479 | -0.0558 | same sign NEGATIVE |
| V10 | -0.0642 | -0.0473 | -0.0597 | same sign NEGATIVE |
| V15 | -0.0649 | -0.0575 | -0.0676 | same sign NEGATIVE |

**The autopsy says the proxy, not the rule, was on trial**: it vetoed
12.24% (zh) / 15.66% (6bba) / 18.19% (44b6) of all cached points --
4-6x dc_gate's prediction (~3%) -- and the excess is ordered by cell
density, because dcdet's 4 um NMS erases the peak of any real cell in
a stronger neighbour's shadow, so dense regions read as "DC silent"
exactly where they are busiest. 44b6_0b24845f alone: -0.4788. 6bba
edge_J 0.9233 -> 0.8519 with node_ratio -0.019 -> -0.135 (real nodes
deleted wholesale, edges collapse).

Settled claim: peak-list-within-3um is NOT a usable silence oracle.
NOT settled: the heatmap-response veto itself. `biohub-dcsample` (T4)
is caching ep05's raw window-max response at every xl detection
(hdval_*.npz, 12+12+8 sets); `vetoscore2` (generated, ready) reruns
the harness with the faithful rule at tau 0.02/0.05/0.10. The user's
niche-overlap caveat is pre-registered with it: the pipeline already
deletes points post-link (_short_filter), so if the faithful veto is
also negative, the reading is that pre-ranker deletion at these taus
does not pay -- and the veto signal's remaining unused form is a
ranker FEATURE, not a filter.

## 2026-08-30 -- ⭐⭐⭐⭐ fuserows + judge audit: THE EVIDENCE JUDGE LEARNS +0.0018 OVER A FIXED WEIGHTED AVERAGE. The user's design, executed and measured to its edge; no pipeline spend.

`biohub-fuserows` (28.3 min T4) collected 381,405 evidence rows -- union
candidates of the four heatmaps, 7 response-pattern features (brightness
deliberately excluded from features), 6,710 competition-GT positives
from 36 pool datasets (18+18, seed 20260830), 374,695 zh rows from 33
non-eval crops -- and trained two negative variants. A local CPU audit
(judge_audit.py on judge_rows.npz) then put every variant and every
no-learning baseline on the SAME held-out test set (every second zh
crop, all rows, 223,746 of which 90,161 positive):

| scorer | held-out AUC |
|---|---|
| mean3 (shipped) | 0.7380 |
| hd alone (ep05) | 0.7466 |
| F4 / W1101 / DCheavy fixed blends | 0.7557 / 0.7632 / **0.7633** |
| **judge_all (trained)** | **0.7651** |
| judge_dim_only | 0.7536 |
| judge_dim+silent | 0.7002, degenerate (comp-pos q10 = 1.0) |

Three findings:

1. **fuserows' judge_filtered AUC 0.9956 was CIRCULAR** -- its test
   negatives were selected by the model's own input features
   (ep05-silent + dim). On the honest test it reads 0.7002 and its
   score distribution is a point mass (uncalibratable). Filtering
   negatives by features makes the problem trivial and the scorer
   useless -- a trap now on record.
2. **The trained judge clears the best FIXED blend by +0.0018** --
   noise. The joint pattern of the four responses is exhausted, to
   GBM-visible precision, by a weighted average. Since the fixed
   blends themselves just lost the pipeline test (fusescore), no
   variant meets the pre-registered bar for spending fusecache2 +
   fusescore2. Stopped here, ~0.5 h GPU total for the whole question.
3. What the direction still owns: only POINT responses were asked --
   per-map local shape, inter-map peak offsets, and raw-intensity
   context were never features; the negative-label problem is now
   precisely priced (zh gap 2.5-7%, each filter's cost measured);
   judge_rows.npz allows zero-GPU retests of any future feature idea.

## 2026-08-30 -- ⭐⭐⭐⭐ zhcomplete: ZEBRAHUB IS NOT FULLY LABELED. Gap ~2.5% (bounded above by ~7%), and it is whole-track misses, not flicker. The user's challenge, answered with numbers.

`biohub-zhcomplete` v1, CPU, 0.7 min. Two label-free measurements:

1. **Their own per-frame dropout is ~zero**: 927,424 mid-flight
   track-frames across 41 crops, 9 one-frame holes, 18 any-length
   (boundary-guarded at 5 um). Whatever their tracker finds, it keeps.
2. **Unmatched-peak forensics** (8 crops, 21,721 xl peaks >= 0.15,
   9.6% unmatched): of 2,090 unmatched peaks, **591 (28.3%) are bright
   (above the matched peaks' own q25) AND independently seconded by
   ep05 within 3 um** -- most plausibly unlabeled cells. 351 (16.8%)
   are dim with no consensus -- clean junk. 1,041 (49.8%) dim-with-
   consensus are ambiguous (the bright bar excludes 25% of real
   matched cells too, so some are dim real cells).

**Completeness gap: 591/(22,675+591) = 2.54% as the defensible
estimate; ~6.7% if every ambiguous peak is a cell.** Consequences:

- zh "precision" for every detector is UNDERSTATED by roughly this
  amount; arm orderings are largely robust (all arms inflated the same
  way) but absolute zh precision/F1 carries a footnote from now on.
- zh keeps the honest-DENOMINATOR role for recall (a 2.5% incomplete
  denominator is still 10-100x more complete than the competition's
  2.8% annotation), with the footnote.
- **The fusion judge's negative mining is sound ONLY in the filtered
  variant** (negatives = unmatched AND dim AND ep05-silent); the
  all-negatives variant would train on several hundred unlabeled cells
  as counter-examples. fuserows trains both; the filtered one is the
  pre-registered primary.

## 2026-08-30 -- ⭐⭐⭐⭐⭐ fusescore: THE BLENDS FAIL THE PIPELINE. Detection F1 anti-predicts adj_J for the NINTH time, and the whole DeepCenter arc closes at every level.

`biohub-fusescore` v1, CPU, 256.9 min. Five arms x three domains, one
solve per arm per dataset, D32 ranker for every judged arm (leak rule),
BASE x D187 as the harness anchor. **Anchor exact in all three domains**
(6bba 0.9293 / 44b6 0.8185 / zh 0.8093 = dcpool to the digit).

| arm | 6bba adj_J | 44b6 adj_J | zh adj_J |
|---|---|---|---|
| BASE x D187 (anchor) | 0.9293 | 0.8185 | 0.8093 |
| BASE x D32 (reference) | 0.9251 | 0.8250 | 0.8088 |

adj_J deltas vs BASE x D32:

| blend | 6bba | 44b6 | zh | verdict |
|---|---|---|---|---|
| F4 | **-0.0053** | +0.0054 | +0.0027 | signs disagree, discard |
| W1101 | **-0.0108** | +0.0054 | +0.0024 | signs disagree, discard |
| DCheavy | **-0.0145** | -0.0020 | -0.0028 | same sign NEGATIVE |

div_J deltas: F4 -0.0301/0/-0.0022, W1101 -0.0278/0/-0.0019, DCheavy
+0.0571/0/-0.0016 (a curious 6bba div gain on the arm whose edges are
worst -- not actionable while adj_J bleeds).

### What this settles

1. **detfuse2's +0.0155 detection F1 -- recall AND precision up, dets/fr
   DOWN, dense GT, anchored -- did not survive the pipeline.** zh
   converted at ~1:6 (+0.0027) and 6bba went NEGATIVE. That is the
   ninth anti-prediction by an offline detection number, and this one
   was the strongest detection evidence this project ever produced.
2. **The dcmerge pattern is the model's domain property, not the
   usage's**: helps where xl is weak (zh, mildly 44b6), hurts where xl
   is strong (6bba), under ALL THREE usages now tried -- points as
   primaries (dcmerge), points in the repair pool (dcpool, null),
   heatmap in the blend (fusescore).
3. **CORRECTED PHRASING (user, 2026-08-30: an experiment kills a
   claim, not a direction).** What is measured and settled: global
   blending of ep05 into the average, at these weights, on these 32
   datasets, is 6bba-negative. What the DIRECTION still owns, unused:
   (a) an UNEXPLAINED regularity -- three usages, one pattern, helps
   where xl is weak and hurts where it is strong; explain it and a
   conditional usage may exist; (b) the 44b6 node deficit HALVED
   (-14.8% -> -7.3%) -- the dense-domain under-detection fault is
   half-repaired by this tool, at a 6bba price; (c) F4/W1101 were
   positive in TWO domains; (d) ep05 solo beats the xl ensemble on zh;
   (e) the clean-eval machinery, the checkpoint, the undertraining
   dial. No submission spent.
4. Side-reading: BASE x D32 vs BASE x D187 -- D32 is -0.0042 adj_J on
   6bba (and -0.035 div_J), +0.0065 on 44b6, -0.0005 zh. The D187
   submission configuration stands.

## 2026-08-29 -- ⭐⭐⭐⭐⭐ detfuse2: THE FOURTH HEATMAP WORKS ON THE HONEST DENOMINATOR. +0.0155 F1 with recall AND precision up AND fewer detections -- the first detection change to satisfy the no-surplus criterion.

`biohub-detfuse2` v1, 13.8 min T4. ep05 DeepCenter blended as a fourth
heatmap BEFORE peak finding (dcmerge/dcpool used points AFTER; both
closed). Anchors digit-exact (zh FULL3 0.7774@7um / 0.7539@5um, W110h
0.7816).

### zh, dense GT, 101,615 points, 7 um, own gates

| arm (b0,b1,b2,dc) | dets/fr | recall | prec | F1* |
|---|---|---|---|---|
| FULL3 (1,1,1,0) shipped | 260.9 | 0.7979 | 0.7579 | 0.7774 |
| W110h (1,1,0.5,0) | 273.3 | 0.8217 | 0.7453 | 0.7816 |
| F4 (1,1,1,1) | 257.7 | 0.8060 | 0.7752 | 0.7903 |
| W1101 (1,1,0.5,1) | 263.2 | 0.8159 | 0.7682 | 0.7913 |
| **DCheavy (1,1,0.5,2)** | **257.7** | 0.8087 | **0.7778** | **0.7929** |
| DC solo (ep05) | 246.4 | 0.7856 | 0.7902 | 0.7879 |

- **DCheavy: +0.0155 F1 over the shipped detector with FEWER
  detections per frame** -- the pre-registered "no surplus" criterion
  is met; this is not dcmerge by another road. (It was written in as
  the wrong-direction control. It won.)
- **ep05 SOLO beats the whole three-branch xl ensemble on dense GT**
  (0.7879 vs 0.7774).
- Cold-miss rescue (edgediag's fat bucket): of FULL3's 15,774 zh
  misses, ep05 responds >= 0.15 near **89.9%** (mean3 82.0%, either
  93.5%).
- (W110hh and W2211 printed identical rows -- they are the same
  normalised weights; a harmless arm-design slip that doubles as a
  determinism check.)

### The competition-side section FAILED to inform, and why

Sampled frames left 8 (6bba) / 4 (44b6) cold misses -- no n. Sparse-GT
"own best gate" lands at 0.40 because the F1 is precision-dominated by
the annotated-only denominator; those recall columns are not readable.
Competition-domain judgment passes to the pipeline test, as it should.

### Next (pre-registered): the pipeline test

Cache blended detections (F4 / W1101 / DCheavy computed from one
shared 4-heatmap pass) for the 32 eval datasets (~1.5-2 h GPU), then a
dcpool-style three-domain adj_J run (CPU), BASE anchored on the
existing caches at 0.9293. Ranker FROZEN per thr5/dcmerge precedent --
the submission would not retrain it either.

## 2026-08-29 -- ⭐⭐⭐⭐⭐ edgediag: THE LOSS IS DETECTION COLD-MISSES, 8x EVERYTHING THE LINKER LOSES. The first full attribution since the classical pipeline.

`biohub-edgediag` v1, CPU, 49.7 min. Shipped config (cap20, SHORT_MIN
4, D187), every GT edge charged to the FIRST losing stage, every final
edge classified, official scorer on the same geffs. **Anchor exact:
6bba adj_J 0.9293.** First 44b6 baseline row: adj_J 0.8185.

### FN side (% of adjacent GT edges; zh is the only all-cells denominator)

| category | 6bba (10,313) | 44b6 (2,890) | zh (221,240) |
|---|---|---|---|
| det_miss_subthresh | 1.06 | 4.01 | 2.60 |
| **det_miss_cold** | 1.33 | 2.98 | **14.49** |
| no_candidate (>10um) | 0.70 | 1.11 | 0.06 |
| below_floor (<0.10) | 0.78 | 2.32* | 0.26 |
| node_absent | 0.71 | 2.39 | 0.95 |
| ilp_rejected | 0.69 | 2.39 | 0.35 |
| killed_post | 0.04 | 0.21 | 0.21 |
| recovered | 94.71 | 84.60 | 81.08 |

*44b6 below_floor is UNDERCOUNTED (D187 in-sample there). Cross-checks
vs official edge_fn: 546/411, 445/349, 41,849/38,936 -- category totals
run ~10-30% above official FN because some "recovered" edges still
score FN through the scorer's own matching; the ordering of buckets is
unaffected.

### FP side (final edges; official edge_fp in brackets)

| | 6bba | 44b6 | zh |
|---|---|---|---|
| fp_confusion (wrong link, both real) | 13 | 7 | 3,718 (1.81%) |
| fp_half (edge into a spurious det) | 826 | 291 | 5,298 (2.57%) |
| fp_invisible (no annotated endpoint) | 93.3% | 99.0% | 6.88% |
| official edge_fp | (363) | (258) | (5,615) |

### The verdict

**On the honest denominator, detection cold-misses (14.5%) are ~8x the
ENTIRE linking-side loss (no_cand + floor + absent + rejected + post
= 1.8%).** The ranker, the ILP and the post stages each lose under 1%;
the short filter barely misfires; wrong links are 1.8% of final edges.
The sparse domains understate the detection bucket by construction
(annotators label visible cells) and it still leads on 44b6. This
converges with gap2 (79.8% of missed frames have no CANDIDATE within 5 um
-- ⚠ this line originally said "NO heatmap response", which gap2 never
measured; see the correction at its entry, and `gtpeak`, which found the
response present at 95%+ of misses), the dim-quartile recall 0.52, and zh true recall 0.865: **the fat
target is detector recall on dim/vanishing cells. The linking stack is
not "done", but everything it loses combined is smaller than one
detection bucket.**

What attacks it, by existing evidence: training OUR OWN detector (never
done; dctrain built the machinery -- clean 175 split, PU loss, snapshot
discipline, 32.2 min/epoch measured -- and brightness jitter / dim-cell
weighting are one flag each); detfuse's W110h weighting (+2.8 recall at
equal F1); the 2.5% single-branch-visible miss headroom.

## 2026-08-29 -- ⭐⭐⭐⭐ divcurve: CROP COUNT IS NOT THE CONSTRAINT. The curve saturates by ~30 crops and the full set sits below the 30-crop draws; the extraction line closes for the price of one CPU kernel.

`biohub-divcurve` v1, CPU. div-train v3's protocol verbatim (same
triples, 25 features, MATCH_UM 5.0, same GBM), the mixture sweep
replaced by a crop-count learning curve. Eval set: 33,587 competition
6bba triples, **83 positives** -- AUC noise at this n is ~±0.02.

| crops | seeds' AUC on held-out 6bba | mean |
|---|---|---|
| 10 | 0.8029 / 0.7578 / 0.8215 | 0.794 |
| 20 | 0.8075 / 0.8100 / 0.8249 | 0.814 |
| 30 | 0.8326 / 0.8470 / 0.8292 | **0.836** |
| 39 (all with divisions) | 0.8201 | 0.820 |

10 -> 30 rises +0.042 (ranges barely overlap), but **the full 39-crop
set lands BELOW all three 30-crop draws**. The marginal crop at the top
end is worth zero or less; with Zebrahub labels machine-generated (~10%
on near-background), late-added crops plausibly carry label noise.
Extrapolating to 3x the crops is NOT supported. **Do not spend GPU on
extraction; the division model's constraint is the model / features /
label quality, which is the pixel-route territory that was PARKED.**

### ⚠ The anchor did not reproduce, and CANNOT: div_train's thinning is hash-salted

Full-set AUC read 0.8201 against div-train v3's recorded 0.8083. The
cause is in div_train.py itself: negative thinning seeds
`np.random.default_rng(hash(name) % 2**31)`, and Python salts str
hashes PER PROCESS. **v3's 0.8083 was one draw of a stochastic
thinning and was never exactly reproducible.** Within one run every
arm shares one triple set, so the curve above is internally clean;
across runs, zh-side AUCs carry ~±0.01-0.02 of thinning noise on top
of eval noise. Any future assembly from div_train.py must replace
`hash(name)` with a stable hash (e.g. zlib.crc32 of the encoded name)
before its numbers can anchor anything.

## 2026-08-29 -- ⭐⭐⭐⭐⭐ dcpool: THE MECHANISM FIRED AND THE SCORE DID NOT MOVE. Bridge positions never mattered, and the whole DeepCenter line closes on evidence at every branch.

`biohub-dcpool` v1, CPU. Epoch-5 DeepCenter points fed ONLY into the
repair pool (score 0.12, visible to `_gap_close` alone); one solve per
dataset, arms byte-identical upstream. **Every anchor held exactly**:
6bba BASE adj_J 0.9293 / div_J 0.1316, zh BASE 0.8093, and the first
clean 44b6 arm (dctrain's excluded 12) baselined at adj_J 0.8185.

### The result: a true null with its mechanism measured

| DC_TAU | bridge inserts: midpoint | xl snap | **DC snap** | adj_J delta, all 3 domains |
|---|---|---|---|---|
| BASE | 3,017 | 1,348 | 0 | — |
| 0.30 | 2,635 | 1,347 | 383 | ±0.0000 |
| 0.15 | 2,179 | 1,345 | 841 | ±0.0000 |
| **0.05** | 2,138 | 1,345 | **882** | **±0.0000** |

117,691 DC points entered the pool (+29.2%); at tau 0.05, **20% of all
bridge inserts moved from guessed midpoints onto measured positions --
the mechanism fired substantially -- and adj_J did not move at the
fourth decimal in any domain.** The cap never bound (17,460 made vs
55,188 allowed).

### Why zero, exactly

A bridge insert's position affects the metric only through whether the
inserted node matches GT within 7.0 um. A 1-frame gap's straight-line
midpoint errs by roughly half a displacement -- p50 displacement is
1.82 um -- so interpolation already lands inside the match radius
essentially always. Measured positions are metric-equivalent to guessed
ones AT SPAN 1. (This also retro-explains the cap20 win's anatomy: the
budget -- how MANY bridges -- was the entire effect; position quality
never entered.)

### The DeepCenter ledger, now closed at every branch

| use | run | verdict |
|---|---|---|
| primary add (published model) | dcmerge v2 | closed: breaks links, opposite signs |
| primary add would need clean eval | dctrain/dccache2/3 | built it; epoch-5 is the right artifact |
| repair-pool add (conditional, ep05) | **dcpool** | **closed: true null, mechanism measured** |
| span-2 bridges (guessed positions) | span4 | closed: eats daughters |

What survives from the whole arc: (1) the epoch-5 detector itself --
clean three-domain eval, beats xl's recall on 44b6, cached for all 40
eval datasets -- available if a future line needs a second detector for
something other than bridge positions (e.g. division daughters); (2)
the undertraining dial; (3) the detfuse finding that fusion signal
exists at the peak-ranking level (+0.0215 AUC), which is a DETECTION
question, untouched by this null.

## 2026-08-29 -- ⭐⭐⭐⭐⭐ dccache3: UNDERTRAINING IS THE MECHANISM, MONOTONE IN ALL THREE DOMAINS. Epoch 5 beats xl's recall on 44b6 outright and covers half to two-thirds of xl's misses on competition data.

`biohub-dccache3` v1, 23.4 min T4. Three checkpoints of the SAME
retrained model (epochs 5, 7, 12) through one harness, greedy 1:1 at
7.0 um, xl caches complete this time (199 + 41; edge-ranker attached):

| domain | ckpt | DC@.05 | DC@.15 | xl@.15 | DC covers xl-miss |
|---|---|---|---|---|---|
| 6bba | **ep05** | **0.9735** | 0.9656 | 0.9827 | **0.4865** (n=185) |
| 6bba | ep07 | 0.9649 | 0.9265 | 0.9827 | 0.4000 |
| 6bba | ep12 | 0.9420 | 0.8984 | 0.9827 | 0.3514 |
| 44b6 | **ep05** | **0.9678** | 0.9522 | 0.9508 | **0.6759** (n=145) |
| 44b6 | ep07 | 0.9566 | 0.8673 | 0.9508 | 0.6552 |
| 44b6 | ep12 | 0.8751 | 0.7686 | 0.9508 | 0.6069 |
| zh | **ep05** | **0.8555** | 0.8357 | 0.8650 | **0.3308** (n=30,768) |
| zh | ep07 | 0.6524 | 0.4552 | 0.8650 | 0.1861 |
| zh | ep12 | 0.5293 | 0.3499 | 0.8650 | 0.1875 |

(6bba/44b6 denominators are annotated cells; zh is all cells. ep12 rows
reproduce dccache2 exactly — the anchor held.)

**Every column is monotone in training length, in every domain.** Less
training = more recall and more coverage. The published model's value
was never the architecture or the 44b6-only data: it was being nearly
untrained. That is now a DIAL — and our ep05 comes with clean eval
domains, which the published epoch-2 never had.

Three readings, and a limit:

1. **ep05 is the artifact this line was hoping for on competition
   data**: on 44b6 its 0.9678@0.05 is ABOVE xl's 0.9508@0.15 and it
   covers 67.6% of xl's misses; on 6bba it covers 48.7%. The hidden
   test is competition-like (a third embryo, same instrument), so these
   are the operative numbers.
2. **zh transfer is restored only partly** (0.8555 vs the published
   0.9784; coverage 33.1% vs 69.5%). The published epoch-2 saw ~2 x
   7,100 frames, our ep05 ~5 x 16,700 — it is still five times more
   trained. Checkpoints below epoch 5 do not exist (snapshot interval
   was 5); an even-earlier stage would need a short re-run with
   --snapshot-interval 1, ~3 h GPU, ONLY worth it if the adj_J test
   says coverage converts to score.
3. **Between ep05 and ep07 sits a cliff on zh** (0.8555 -> 0.6524 at
   0.05). Whatever generic feature the early model uses, it is
   abandoned quickly — consistent with the published run's best-at-2.

dcdet_ep05_*/dcdet_ep07_*.npz for all 40 datasets are in the kernel
output; the ep05 set is the input for the adj_J test, which is the
actual decision point. Detection metrics have anti-predicted the
pipeline eight times — nothing here is a score until the three-domain
solve says so.

## 2026-08-29 -- ⭐⭐⭐⭐ dccache2: the retrain FIXED THE EVALUATION AND LOST THE VALUE. The epoch-12 model is weaker than xl in all three domains and its zh transfer collapsed.

`biohub-dccache2` v1, 14.5 min T4. Retrained DeepCenter (epoch 12),
greedy 1:1 at 7.0 um, our post-processing throughout:

| domain | GT n | DC@.05 | DC@.15 | xl@.15 | DC covers xl-miss | xl covers DC-miss |
|---|---|---|---|---|---|---|
| 6bba | 10,699 | 0.9420 | 0.8984 | — (xl caches not attached, see below) | — | — |
| 44b6 | 2,947 | 0.8751 | 0.7686 | **0.9508** | 0.607 (n=145) | 0.952 (n=682) |
| zh | 227,966 | **0.5293** | 0.3499 | 0.8650 | **0.1875** (n=30,768) | 0.840 |

⚠ denominators: 6bba/44b6 over ANNOTATED cells, zh over all cells.

**Against the published DeepCenter's zh numbers (dccache/detcmp: recall
0.9784@0.05, covering 69.5% of xl misses), the retrained optimum is a
collapse: 0.5293 and 18.8%.** In its own training domains it is merely
mediocre — below xl on 44b6 (0.875 vs 0.951) and below xl's recorded
6bba recall (0.942 vs 0.9813). Converging on competition data produced
a detector that is worse than xl everywhere and lost the one property
the line existed for.

**The surviving hypothesis is UNDERTRAINING**: the published model's
best checkpoint is epoch 2 of a diverging run — close to a generic
bright-blob detector — and that genericity, not the architecture or the
data, may be what transferred to zh and covered xl's misses.
`biohub-dccache3` PUSHED to test exactly that: epoch 5 and epoch 7
snapshots (already in hand) against epoch 12, all three domains, same
harness — plus the 6bba xl coverage column dccache2 lacked (its xl
caches live in biohub-edge-ranker's output, which was not attached; the
error was visible as WARN lines and a 0.0000 column, not silent).

If ep5/ep7 restore the coverage, undertraining becomes a dial we
control with clean eval domains. If not, the published model's transfer
is a property of that artifact and the retrain line closes.

## 2026-08-29 -- DCTRAIN v2: killed at the 12 h cap at epoch 22, AND IT DID NOT MATTER -- the output published anyway and the optimum was epoch 12

**The training itself is a clean success.** 167 movies, author's recipe
verbatim: train_loss fell smoothly 0.0305 -> 0.0223 over 22 epochs, NO
divergence (the published model's 500-epoch failure did not recur at
this LR on this data), and **val_loss bottomed at epoch 12 (0.011988)**
with zero improvement in the following ten. best.pt therefore carries
the converged optimum. Gate read on the 8 val movies (sparse GT, 7 um):
recall 0.875@0.10 / 0.848@0.15 at ~174 peaks/frame -- the same order as
the true median density (179 cells/frame). Split confirmed in the log:
167 train + 8 val + 24 excluded, all names printed.

**The cost accounting, so the next training run is sized right:**

- **32.2 min/epoch, dead constant** (my pre-launch estimate said 5-8;
  wrong by 4x -- the bottleneck is per-frame blosc decompress + pool +
  percentile on 4 workers, not the GPU). 22 epochs + snapshots = the
  whole 12 h session; epoch 30 was never reachable.
- The kill arrived as `CANCEL_ACKNOWLEDGED` at ~11.8 h. **A
  platform-cancelled kernel CAN still publish its output** -- all 45
  files (main best.pt/checkpoint_last.pt/history + four ep-snapshots)
  came out with signed URLs. Do not assume a timed-out run lost its
  work; LIST ITS OUTPUT first.
- Because a cancelled VERSION may not be attachable as a kernel source,
  the six files that matter (epoch-12 best.pt, config, history,
  split_manifest, ep0020 gate summary+metrics) were downloaded and
  re-uploaded as the DATASET **faustine649/biohub-dctrain-175-weights**.
  Consumers attach that, not the kernel.
- A future run of this kind: `--epochs 15` fits in ~8.5 h, or halve the
  epoch with `--frames-per-movie 50`. Epoch 12 being the optimum says 15
  is enough.

`biohub-dccache2` PUSHED against the new dataset: caches dcdet_*.npz
for the 24 excluded + 8 zh crops and prints per-domain recall at 7 um
plus mutual miss coverage against the xl caches. Waiter armed.

## 2026-08-29 -- ⭐⭐⭐ detfuse: the learned-fusion idea has real but modest signal, and every road points at down-weighting B2

`biohub-detfuse` v1, 10 min T4, 41 Zebrahub crops, 101,615 GT points.
**Anchors digit-exact against det_ablate3** (FULL@5um 0.7738/0.7350/
0.7539 at gate 0.15; B01 0.7585 at 0.20).

### 1. Scalar weights: +0.004 F1, a PLATEAU, all of it "suppress B2"

At the scorer's 7 um radius, each arm at its own best gate:

| arm (w0,w1,w2) | recall | prec | F1* |
|---|---|---|---|
| FULL (1,1,1) | 0.7979 | 0.7579 | 0.7774 |
| **B01 (1,1,0)** | 0.7935 | 0.7701 | **0.7816** |
| **W110h (1,1,0.5)** | **0.8217** | 0.7453 | **0.7816** |
| W210h (2,1,0.5) | 0.8179 | 0.7482 | 0.7815 |
| W311 (3,1,1) | 0.8079 | 0.7563 | 0.7813 |
| W121 (1,2,1) | 0.8232 | 0.7362 | 0.7772 |
| W112 (1,1,2) control | 0.8247 | 0.7222 | 0.7700 |

Four different weightings tie at 0.7811-0.7816: the surface is FLAT
once B2 is suppressed, so finer scalar tuning is exhausted. Note
**W110h ties B01's F1 while keeping +2.8 points of recall** -- if this
line ever gets a pipeline test, that is the arm to prefer (the node
penalty is weak, recall feeds the repair pool).

### 2. The 11% churn is NOT in the sub-threshold band (det_ablate3's open question, closed)

Peaks where branches disagree by >= 0.30 (7.9% of 128k): score
q25/50/75 = 0.204/0.230/0.261, only 2.0% sub-threshold against 16.5%
for all peaks. **The disagreement sits AT the operating band, not below
it** -- branch disagreement is live exactly where the gate decides.

### 3. Per-peak fusion ceiling: +0.0215 AUC, and B2's learned weight is ZERO

Logistic on the three branch logits, fit even crops / scored odd:
AUC 0.7939 (mean score) -> **0.8153** (+0.0215). Learned weights
**(0.94, 1.68, 0.07)** -- the third branch contributes nothing to
ranking real-vs-spurious. Signal for a learned combiner exists and its
per-peak ceiling is ~2 AUC points. `fuse_rows.npz` (128k peaks) and
`fuse_miss.npz` in the kernel output allow offline fusion training with
no new inference.

### 4. Miss headroom: ~2.5% of all cells are seen by ONE branch and suppressed by the mean

Of 15,774 GT cells unmatched at 7 um (floor 0.05): some single branch
has window-max >= 0.15 near **98.3%** of them, the branch MEAN near
82.0%. ⚠ The 82% column is NOT "mean sees it and still misses" pure:
unmatched includes peaks lost to the 1:1 greedy competition and to the
4-um NMS, so only the **16.3-point gap** (~2,570 cells, ~2.5% of GT) is
the fusion-specific recall a per-voxel combiner could recover. Those
are single-branch-visible cells -- the flicker population.

### Verdict

Precondition confirmed, ceiling priced, and it is modest: scalar fusion
is worth at most ~+0.004 detection F1 (all from B2), per-peak learned
fusion ~+0.02 AUC. All DETECTION numbers, one domain -- offline numbers
have anti-predicted adj_J eight times, so nothing here earns a pipeline
run on its own yet. The line holds until the retrained DeepCenter
lands: it is the strongest candidate fourth input, and the eval
machinery (re-cache + ranker retrain + three-domain solve) is the same
spend either way. Decide then whether to test W110h / a learned
combiner / DC-as-input together, not three times separately.

## 2026-08-28 -- DCTRAIN LAUNCHED: retrain DeepCenter on a split we control (user's call)

`biohub-dctrain` v1 pushed, T4, RUNNING. Design settled with the user:

**Why.** DeepCenter's shipped trainer contains the split bug in plain
sight: `split_samples` groups by name prefix, the prefix has two values
(44b6/6bba), and "hold out 10% of groups" held out one whole batch. So
44b6 is its training set (eval forever nan) and 6bba its selection set
(tainted), and dcmerge's opposite signs could never be settled. The
user: repair only helps when both endpoints exist, a union always loses
on the node surplus -- so retrain it on our split and re-measure, or
drop it.

**The split (199 = 167 train + 8 val + 24 excluded), all names printed
in the log:**

- excluded: the 12 6bba holdout (verbatim) + 12 44b6 drawn
  `random.Random(20260827).sample` from the 44b6 OUTSIDE D32's pool
  (random per the user's standing rule; outside-D32 because a detection
  change needs the D32 ranker and the leak rule). All three eval
  domains are clean -- the first detector here that can face the full
  three-domain rule.
- val: 4+4 from the remaining 175, seed 20260828, excluded from
  training; best.pt selects on it, not on any eval domain.

**Recipe: the author's defaults verbatim** (checked against their run
wrapper line by line), except `--snapshot-interval 5 --snapshot-final`
(their 500-epoch run diverged with best at epoch 2, so every 5-epoch
state is kept) and epochs 30 not 50 (167 movies/epoch = 2.35x their
per-epoch exposure; `--resume` continues if history says still
improving). Labels read via `viz_cells.read_geff` (zarr-free); frames
via the trainer's own per-chunk blosc2 path, which matches the
competition layout exactly. Smoke-read asserted at startup.

**Cost, an ESTIMATE not a measurement:** ~5-7 h GPU. 8.8 h of quota
remained at launch and the week refreshes 2026-08-29 00:00 UTC
mid-run. Waiter at 300 s.

**Next after it lands:** cache detections for the 24 excluded + 8 zh,
detcmp-style coverage + recall, then dcmerge-style arms with the
retrained weights -- now readable in all three domains.

## 2026-08-28 -- forum + leaderboard recon (second pass, evening session)

Read via the MCP connector; everything below is what OTHER PEOPLE claim,
not our measurements, except the leaderboard which is read directly.

### The leaderboard moved since 2026-08-20

#1 is now **0.962 held by TWO teams** (Soheil Ayati, z7777); #3 Tang
0.955; **#20 is ~0.943** (was 0.935). Our 0.922 is rank ~409, and the
~77-team cluster at exactly 0.927 (see the 0.922 entry) still sits
between us and the clean field.

### Two frontier teams named their bottleneck order, and it is DETECTION

- **The #1 (Soheil Ayati), /737101:** "many 'linking' issues actually
  originated earlier during node selection, so improving the edge model
  won't necessarily help. Break missed edges into missing-endpoint vs
  incorrect-association and focus exclusively on the dominant one."
- **The #3 (Tang, /737543):** "detection -> linking -> division.
  detection should come first."

Both consistent with this project's own record: `logbase` priced the
detector at +0.0566, and the 0.922 entry already flagged the public
temporal-UNet3D + 300-epoch ranker as "the one legitimate thing worth
stealing". Nobody at the frontier says the win is in post-processing.

⚠ **CORRECTED same day, user's catch: the temporal UNet3D was already
tried, and it LOST.** The official pipeline's detector IS the
TemporalUNet3D (competition-brief.md), i.e. the whole 0.876→0.898 line
ran it. The head-to-head is recorded at the 2026-08-22 Branch C entry:
on Zebrahub, both models naive, xl 0.766/0.736 recall/precision vs
temporal 0.650/0.637, and **dim-quartile recall 0.73 vs 0.52** — worst
exactly where a temporal model was supposed to help. LB agreed (xl
family 0.917, official family 0.898; the chimera buried). What was
never inspected is only the `seed314159` SECOND SEED + the 300-epoch
ranker + two-seed blending; given the 50ep result and Togashi's 0.915
end-to-end, not worth a run. "We never tried a better detector" in the
0.922 entry means never tried one better than xl — the temporal one was
tried and is worse. The detector idea still standing is DeepCenter as a
repair-pool source (measured coverage, different bias, no temporal
claim).

### ⭐ A rank-58 competitor reports LB div_J ~0.28, and names a mechanism

**Lime1123, /737543** (team-up intro claims div_J ~0.28 on LB with
adj_J ~0.89, i.e. the division term alone worth ~+0.028): "divisions
seemed much more sensitive to SPATIAL ACCURACY, roughly within 3 um,
even though the metric uses 7 um." The thread author (k3v1npark,
div_J 0.1176 — our level) then measured his own predictions: **39% of
his division nodes sit more than 3 um from GT while all pass 7 um.**

Plausible mechanism (mine, unverified): sister separation at t+1 has
p50 ~7-9 um, so around a division there are 2-3 GT nodes inside one
7-um match ball; a node a few um off can match the WRONG GT node and
the directed-topology check then fails the whole division. If true,
div FN/FP are partly a LOCALISATION problem, not a proposal problem —
a channel none of our division work (gates, rankers, mass, kinematics)
ever touched. **Nothing in this project has ever measured the
localisation error of division-adjacent nodes.** Cheap to measure from
caches + GT on CPU. Related transplant never run: the public tier's
linefit smoothing and full-resolution centroid refinement
(`public-solutions-recon.md` addenda) both exist to improve exactly
this quantity.

### Small corroborations

- /737896: a GT track visibly at background intensity ("very dim
  nodes") — ⚠ originally cited as "consistent with gap2's
  79.8%-vanished-response finding". That number is a PEAK count, not a
  response measurement, and `gtpeak` (2026-09-08) found the response
  present at 95%+ of missed cells. The forum observation stands on its
  own; it is not corroboration of a claim nobody measured.
- /737103 + Theo Viel: hand-labeling own training data is probably
  allowed (test data prohibited); no host ruling yet.
- /737577: yet another team stuck at div_J ~0 via ILP division-weight
  tuning — the field-wide pattern our own 8:1 measurement predicted.

## 2026-08-28 -- DCMERGE v2: DeepCenter as primary nodes WINS +0.0379 on Zebrahub and LOSES -0.0413 on 6bba, monotonically, in both directions

**The result WAS read in session; what it never got was an entry here.**
`bbd386f` committed, the conclusion carried forward as "DeepCenter merge:
closed" — and none of the numbers written down. A later session
(2026-08-28) searched `docs/` for them, found nothing, and re-derived
the whole thing from the Kaggle log. **Reading a result and recording it
are two separate acts, and only the second one survives the session.**
Recorded now, in full.

`biohub-dcmerge` v2, 108 min, 4 arms x 2 usable domains. DeepCenter
peaks added as PRIMARY nodes, strictly add-only: our peaks untouched, a
DeepCenter peak admitted only when further than `DC_MERGE_UM` from every
one of ours, scored at exactly `UNET_THRESH` so it lands primary without
entering the repair pool.

**The anchor held.** BASE reproduces `span4`'s `cap20` row exactly ---
`edge_J` 0.9275, `node_ratio` -0.0179, `adj_J` 0.9293 against span4's
0.9279 + 0.0013 = 0.9292. (The header's "BASE must give adj_J 0.9279" is
stale text describing the PRE-`cap20` base; this harness's BASE is the
shipped configuration. v1's broken BASE gave 0.8897 with `node_ratio`
+0.1336, so the difference is not subtle.)

### The result

| arm | 6bba `adj_J` | zh `adj_J` | verdict |
|---|---|---|---|
| `dc0.30` | **-0.0110** | **+0.0221** | signs disagree |
| `dc0.15` | **-0.0315** | **+0.0373** | signs disagree |
| `dc0.05` | **-0.0413** | **+0.0379** | signs disagree |

**Monotone in the threshold in BOTH domains, in OPPOSITE directions.**
Six points, perfectly ordered. This is not noise the three-domain rule
is catching --- it is a real effect with opposite sign in two domains.

⚠ **44b6 is nan and always will be.** All 71 are DeepCenter's training
set, so **this change can never satisfy the three-domain rule.** Two
domains is the ceiling for anything built on this model.

### The decomposition, which says WHY

| 6bba | `edge_J` | `node_ratio` | `adj_J` | `div_J` |
|---|---|---|---|---|
| BASE | 0.9275 | -0.0179 | 0.9293 | 0.1316 |
| `dc0.30` | 0.9197 | +0.0169 | 0.9183 | 0.1282 |
| `dc0.15` | 0.9048 | +0.0773 | 0.8978 | 0.1026 |
| `dc0.05` | 0.9011 | +0.1471 | 0.8880 | 0.1176 |

| zh | `edge_J` | `node_ratio` (derived) | `adj_J` | `div_J` |
|---|---|---|---|---|
| BASE | 0.8036 | -0.0709 | 0.8093 | 0.0230 |
| `dc0.30` | 0.8279 | -0.0423 | 0.8314 | 0.0333 |
| `dc0.15` | 0.8463 | -0.0047 | 0.8467 | 0.0369 |
| `dc0.05` | 0.8491 | +0.0224 | 0.8472 | 0.0381 |

**In 6bba `edge_J` itself falls 2.6 points.** This is not the node-count
term punishing over-detection --- the EDGES get worse, so the added
points are actively breaking correct links, presumably by competing in
the capacity-2 assignment. **In Zebrahub `edge_J` RISES 4.6 points**,
and `div_J` rises +0.0151 on top.

⚠ The zh `node_ratio` column prints `nan` --- that domain's rows carry no
`estimated_number_of_nodes`, the instrumentation gap `span4` also hit.
The values above are DERIVED from `adj_J / edge_J = 1 - 0.1 * ratio`,
so the official metric does apply the node penalty there; only our own
printed column is missing. Between BASE and `dc0.05` the surplus grows
**+0.165 in 6bba against +0.093 in Zebrahub** --- DeepCenter adds nearly
twice the surplus in 6bba, which is what "our detector already found
them there" looks like.

### ⭐⭐⭐⭐⭐ The mechanism, settled by `N_true` --- and it refutes the sparse-GT reading

The obvious explanation, and the one this session first reached for, is
**sparse annotation**: 6bba labels only 9.71% of its cells, so most of
what DeepCenter adds there would be *unannotated real cells* --- absent
from the numerator, present in the node-count denominator, and stealing
links from annotated cells. Zebrahub is densely annotated, so every
recovered cell pays.

**That reading is wrong, and `N_true` is what kills it.** The denominator
is `estimated_number_of_nodes`, which `competition-brief.md` and
`reference.md` both record as **the true TOTAL cell count, not the
annotated count**. Unannotated real cells are already in it. So
recovering one moves `node_ratio` toward 0 from below; it cannot push it
above 0.

Run the arithmetic on the printed rows (treating `N_true` as exact):

| domain | `node_ratio` BASE | `node_ratio` `dc0.05` | added, as a share of `N_true` | of that, SURPLUS past the true count |
|---|---|---|---|---|
| 6bba | −0.0179 | **+0.1471** | 16.50% | **14.71% — 89% of everything it added** |
| Zebrahub | −0.0709 | **+0.0224** | 9.33% | **2.24% — 24% of what it added** |

**On 6bba at most 11% of DeepCenter's additions can be real cells; the
other 89% overshoot the estimated total number of cells that exist.
On Zebrahub at least 76% of them close a genuine 7.1% detection
deficit.** Same model, same thresholds, opposite calibration --- and it
is calibration against the IMAGES, not against the annotation density.

That also explains the edge half. `edge_J` is measured only where
annotation exists, so an FP in an unannotated region is invisible to it
while still counting in `node_ratio`. 6bba's `edge_J` fell **2.6
points** anyway, which means enough of the surplus landed close enough
to annotated cells to steal their links under the capacity-2
assignment --- and by the metric's FP rule, one matched endpoint is
enough to make the resulting edge count against us.

⚠ **Not leakage, and the direction proves it.** 6bba is DeepCenter's
own validation split; Zebrahub it has never seen. A leak would make it
look BETTER on 6bba. It looks worse. **The 6bba loss is real.**

⚠ This treats `estimated_number_of_nodes` as accurate. It is an
estimate. But 89% against 24% is not a margin an estimate's error
plausibly closes.

### ⚠ 6bba is the domain with standing, and that is stronger than the rule here

The three-domain rule discards this for disagreeing signs, and 44b6 can
never be recruited to break the tie --- all 71 are DeepCenter's training
set. But the rule is not even the binding argument. **The hidden test is
competition data** (`test/` holds 4 sample datasets and the real set is
swapped in at rescore time), **not Zebrahub.** 6bba is the domain that
predicts the leaderboard, and 6bba says −0.0110 to −0.0413,
monotonically. That is not uncertainty; it is a clear negative.

**So the line closes --- but not as "the model is useless."** Its
Zebrahub `edge_J` gain of **+0.0455** is the largest single edge
improvement anything has produced in this project. What is wrong is its
operating point on competition images, and nothing short of retraining
or recalibrating it changes that.

### ⭐⭐⭐⭐⭐ The two routes fail together, and that is the finding

Put both routes on the SAME base --- the shipped configuration, `cap20`
--- and the pattern is not about guessing versus measuring at all:

| | 6bba | Zebrahub |
|---|---|---|
| `span2`, GUESS the position | **−0.0027** | **+0.0012** |
| `dc0.05`, MEASURE the position | **−0.0413** | **+0.0379** |

(span4 reported against the pre-`cap20` base: span2 −0.0014 / +0.0068
and cap20 +0.0013 / +0.0056, so the differences above are span2 minus
cap20. dcmerge's base already IS cap20.)

⚠ **2026-09-08: the headroom argument below was only ever checked on
6bba.** `offdet-rank` later measured the weighted node ratio for F4 at
**−0.0228 on 6bba** (consistent with the 1.8% here) but **−0.1371 on
44b6** — i.e. 44b6 is 13.7% SHORT of its estimated cell count, MORE
headroom than the Zebrahub 7.1% where both techniques won. Nobody has
counted 44b6's multi-frame dropouts at all. This does not reopen the
verdict; it records that the argument that closed it does not cover
15% of the score.

**Both routes lose on 6bba and win on Zebrahub. The sign is decided by
the domain, not by the method** --- and the two methods have nothing in
common except that they add nodes. The `N_true` arithmetic above says
why: on 6bba our node count is already within **1.8%** of the estimated
true total, so there is almost nothing left to find and additions
overshoot; on Zebrahub we are **7.1% short**, so additions land on cells
that exist.

**So the recovery of multi-frame dropouts is not a technique problem.**
Two unrelated techniques, one crude and one a trained network 30x
stronger, produce the same sign pattern at 30x the magnitude. What
decides the outcome is whether the detector had headroom in that domain,
which is the same wall `thr5` and `wide44` hit from the other side.

### The pre-registered check that could NOT be read

The run asked: if DeepCenter fills the holes where tracks break,
`_gap_close` has less to do and the bridge count FALLS. **The bridge
table printed one row --- BASE's 6,587 --- and no DC arm.** The check is
unanswerable from this log. If the question is reopened, that table has
to print per-arm.

### What this closes, and what it does not

**Closed: DeepCenter as an unconditional add.** It cannot pass the rule
by construction, and it loses on the one independent domain where our
detector is strong.

**NOT closed, and this is the honest residual: a CONDITIONAL add.** The
effect is monotone and enormous where our detector is weak. A rule that
admits foreign detections only where our own detector is weak would take
the Zebrahub gain and skip the 6bba loss --- but it needs a
weakness signal computable at test time, without labels, and no such
signal has been proposed, let alone measured. **Nothing about this is
submittable today.**

## 2026-08-28 -- ACCELERATION IS COMPUTABLE 87.3% OF THE TIME, AND IT STILL DOES NOT HELP. The line closes on measurement, and progress.md's own account of it was wrong twice.

`biohub-accdiag`, 9 min, CPU, 12 6bba holdout datasets, 173,269 seeds.
The user asked two things about `biohub-accel`: **does the acceleration
column subtract camera shake, and is the exact-zero even firing for a
real reason.** Neither had an answer in this file, and looking for one
turned up two errors in the entry that recorded the failure. Both are
marked in place below.

### 1. How often is acceleration computable -- with the right denominator

| `acc_hist` | meaning | seeds | share |
|---|---|---|---|
| 0 | no backward link at all | 13,384 | 7.7% |
| 1 | one backward step, needs two | 8,703 | 5.0% |
| 2 | **full, three positions** | **151,182** | **87.3%** |

**"For most candidates it cannot produce three" was wrong by a wide
margin.** It is computable for seven cells in eight. The old paragraph
compared a per-seed-per-frame counter (316,658) against a count of
candidate PAIRS (225,861) -- two different things, so the fraction had
never been measured at all. It is measured now.

⚠ The run's own degradation counters, 31,594 and 25,885, are STILL
per-seed-per-frame and still accumulate across the sweep, stage B and
the diagnostic. **They are not comparable to 173,269 either.** The table
above is the measurement; the counters are not.

### 2. Stage motion against cell motion -- the user's first question

Nothing anywhere in the pipeline subtracts global stage motion. A
constant drift is absorbed by the velocity column; a CHANGING drift
appears as the same spurious acceleration on every cell at once, which
is exactly what the per-frame median of the acceleration vectors
estimates.

| | median | p90 |
|---|---|---|
| \|median acceleration\| per frame (THE STAGE) | 0.4845 | 1.9795 um/frame^2 |
| per-cell \|acc - median\| (THE CELLS) | 2.0187 | 2.8265 |
| **ratio stage/cell** | **0.240** | |

**The stage is real but small -- a quarter of the per-cell residual.**
It is not what the column is mostly measuring. 1,176 frames had the
>=5 usable seeds needed to estimate a median at all.

### 3. Does de-meaning fix anything -- the one-line fix, measured instead of assumed

Single-column AUC against `poolsize`'s own labels, 14,817 labelled
pairs, 0.6588 positive. **Below 0.5 means informative with the sign
flipped**; 0.5 means nothing.

| col | raw | de-meaned | flipped-equivalent (raw) |
|---|---|---|---|
| `acc_len` | 0.4968 | 0.4919 | -- |
| `acc_cos` | 0.5136 | 0.5068 | -- |
| `md_acc` | **0.0624** | 0.0582 | 0.9376 |
| `md-md_acc` | 0.3013 | 0.3097 | 0.6987 |
| `acc_hist` | 0.5217 | 0.5217 | -- |
| for scale: `d` | 0.0319 | | 0.9681 |
| for scale: `md` | 0.0364 | | 0.9636 |
| for scale: `score_c` | 0.7955 | | -- |

**Every de-meaned column moves by at most 0.0084, and they do not even
agree on direction** -- `md_acc` improves, `md-md_acc` and `acc_cos` get
worse. **The one-line fix does nothing. The line closes here, on its own
evidence rather than on the wrong note this file used to carry.**

### ⭐⭐⭐⭐ Why it never helped, which is neither missing data nor stage drift

Read the scale row. `md` -- distance from the CONSTANT-VELOCITY
prediction -- is 0.9636 flipped. `md_acc`, the same distance from the
acceleration-corrected prediction, is **0.9376**. **Adding acceleration
moves the prediction AWAY from the true child.** It is a strictly worse
version of a column the ranker already has.

The arithmetic says the same thing, and it is arithmetic, not a
measurement. With `vels[t+1][ci] = (C[ci] - P[pi]) * SCALE`, the two
predictors are

```
pred     = 2.0*x_t - 1.0*x_{t-1}                        var  5.00 * sigma^2
pred_acc = 2.5*x_t - 2.0*x_{t-1} + 0.5*x_{t-2}          var 10.50 * sigma^2
```

under independent per-detection position noise `sigma`. **The
acceleration term slightly more than DOUBLES the variance of the
predictor** (2.1x, 1.45x in standard deviation). Real frame-to-frame
acceleration would have to beat that to pay for itself, and at 2.0
um/frame^2 median against a 10 um gate it does not. **A second-order
extrapolator needs a smoother position estimate than single-frame
centroids provide.**

That is a general result, not a fact about this implementation: it also
explains why `biohub-accel`'s permutation importance was +0.0000 to
+0.0017 with the indicator column present and working.

## 2026-08-28 -- ⭐⭐⭐⭐ span4: the reopening was justified and the answer is still NO. The gap-closing line is now closed on evidence.

`biohub-span4`, 167 min, 5 arms × 3 domains, all 71 44b6, D187.
**Both anchors held**: 6bba BASE adj_J 0.9279, and `cap20` reproduced
+0.0013 / +0.0085 / +0.0056 exactly, matching gap4 and wide44.

### The result: what we already shipped is the best of the five

| arm | 6bba | 44b6 | zh | verdict |
|---|---|---|---|---|
| **`cap20`** (ref 55817630, LB 0.922) | **+0.0013** | **+0.0085** | **+0.0056** | passes |
| `span2` | **−0.0014** | +0.0066 | +0.0068 | **signs disagree, discard** |
| `gate12` (9 → 12 µm) | +0.0012 | +0.0069 | +0.0053 | passes, but below `cap20` in all three |
| `span2_gate12` | −0.0024 | +0.0026 | +0.0013 | signs disagree, discard |

**Neither a wider span nor a wider distance gate pays.** `gate12` minus
`cap20` is −0.0001 / −0.0016 / −0.0003: widening the gate admits 28%
more bridges (23,322 against 18,248) and gets slightly worse everywhere.
**9.0 µm is right.**

### Why span 2 loses, from the decomposition

| 6bba | `edge_J` | node_ratio | `div_J` |
|---|---|---|---|
| BASE | 0.9258 | −0.0215 | 0.1316 |
| `cap20` | **0.9275** | −0.0179 | 0.1316 |
| `span2` | **0.9257** | −0.0074 | **0.1143** |

**span2's edges are BASE's** (0.9257 against 0.9258) **while its node
count is 1.4% higher.** In 6bba the two-frame bridges add nodes and no
correct edges — they are simply wrong. Zebrahub disagrees: its `edge_J`
is HIGHEST at span2 (0.8055 against `cap20`'s 0.8036). Two domains want
opposite things, which is what the three-domain rule exists to catch.

### ⚠ The user's division correction does NOT extend to span 2

Recorded 2026-08-27: *"你不能认为 gap_close 会消灭分裂候选…它反而在消灭
分裂错误的候选"*, and `wide44` confirmed it for span 1 — 44b6's `div_J`
rose +0.0148.

**At span 2 the sign flips: `div_J` −0.0173 in 6bba and −0.0039 in
Zebrahub** (and +0.0178 in 44b6). A longer bridge is likelier to be
wrong, and a wrong bridge consumes an orphan that may have been a real
second daughter. The structural protection that made span 1 safe —
`ends` requires `g not in succ`, so a mother is never a bridge source —
still holds, but it does not stop a 3-frame-distant WRONG pairing from
eating a genuine daughter.

**So: span 1 removes false division candidates; span 2 destroys real
divisions.** Two different claims, and only the first one is true.

### The reopening was correct even though the answer did not change

`gap2` closed this line with "the span never mattered", and that verdict
WAS measured behind a binding budget — `wide44` counted the cap binding
in 30 of 83 datasets. Reopening it was right. **The answer is the same
and it is now known for a reason instead of by accident.**

⚠ One residual: span2 makes 35,711 bridges against an allowance of
48,690 and **still caps in 23 datasets** (`span2_gate12` in 38). So the
span arms are partly budget-bound again. Not worth re-running at a
higher cap: span2's 6bba `edge_J` is already BELOW `cap20`'s, so more
two-frame bridges would push the wrong way.

### The bridge counts

| need_cand | span | gate µm | cap | made | allowed | demand | capped |
|---|---|---|---|---|---|---|---|
| False | 1 | 9.0 | 0.02 | 18,248 | 48,690 | 18,248 | 0 |
| False | 1 | 12.0 | 0.02 | 23,322 | 48,690 | 23,322 | 0 |
| False | 2 | 9.0 | 0.02 | 35,711 | 48,690 | 35,711 | **23** |
| False | 2 | 12.0 | 0.02 | 40,270 | 48,690 | 40,270 | **38** |
| True | 1 | 9.0 | 0.003 | 4,883 | 7,266 | 14,108 | **30** |

### A gap in the instrumentation

**Zebrahub's `node_ratio` column comes back `nan`** — that domain's rows
carry no `estimated_number_of_nodes`, so `_with_ratio` finds nothing to
average. Its `edge_J` is still printed and readable; only the
decomposition into edges-versus-node-count is unavailable there. Worth
knowing before designing a run that depends on it.

### Where the gap-closing line now stands

**Closed, with the budget as the only thing in it that ever mattered.**

| knob | verdict |
|---|---|
| **`GAP1_CAP_FRAC` 0.003 → 0.02** | **the win. +0.002 on the leaderboard** |
| the candidate requirement | drop it — part of the same win |
| `GAP_SPAN` 1 → 2 | fails the rule, and destroys divisions in 6bba |
| `GAP1_GATE_UM` 9 → 12 | passes the rule, loses to `cap20` in all three |
| the four gates of gap3 (candidate, velocity, length, clearance) | all were rationing a too-small budget, none selected quality |

## 2026-08-28 -- ⭐⭐⭐⭐⭐ FIVE GATES, INCLUDING A 0.97-AUC LEARNED ONE, ALL LOSE TO NO GATE. The gap-closing line is closed.

`biohub-rank5` v2, 255 min CPU. Lookup healthy this time: **94.64% of
seeds mapped, 0.90% of bridge pairs missing** (v1 was 100% missing).
Anchors held — `cap20` reproduced +0.0013 / +0.0085 / +0.0056.

| arm | 6bba | 44b6 | zh |
|---|---|---|---|
| **`cap20`** (ref 55817630, LB 0.922) | **+0.0013** | **+0.0085** | **+0.0056** |
| `tau0.2` | +0.0011 | +0.0080 | +0.0055 |
| `tau0.5` | +0.0012 | +0.0076 | +0.0052 |
| `tau0.8` | +0.0007 | +0.0030 | +0.0016 |

**Every threshold is below `cap20` in every domain, monotonically in how
much it rejects.** The more the ranker refuses, the worse the result.

### The complete answer to the user's question

The ranker is **not** marginal — 0.9693 AUC against distance-alone's
0.8829 at exactly the decision gap closing makes. **But using it to
REJECT bridges loses.** Both halves are true and they are not in
tension: the model ranks pairs well, and rejecting the low-ranked ones
still costs.

### Five gates, five losses, and now a mechanism

| gate | run | verdict |
|---|---|---|
| candidate required | `gap4` | loses |
| velocity cosine | `gap3` | loses |
| fragment length | `gap3`/`gap4` | loses |
| clearance radius | `gap3` | loses, fails the rule |
| **learned, 0.97 AUC** | `rank5` | **loses, monotonically** |

**A wrong bridge and no bridge are not symmetric, and `_short_filter` is
why.** With no bridge the fragment falls below `SHORT_MIN` and is
deleted **whole** — every correct edge inside it disappears with it.
With a wrong bridge the fragment survives and **keeps all of its correct
edges**; only the two bridge edges are false. So even a wrong bridge
nets positive, and any gate that trades away bridges trades away more
than it saves.

**This also explains `span4`.** `span2` adds MORE bridges and still
loses, so "more is better" is not the rule. A 2-frame bridge inserts
**two** nodes, spans three frames, and is likelier to be wrong — its
cost doubles while the fragment it rescues is the same. **The boundary
is: every one-frame bridge is worth making; two-frame bridges are not.**

### Instrumentation flaw, harmless here

`_GAP_LOG`'s key is `(need_cand, span, gate_um, cap_frac)` and does NOT
include `GAP_RANK_TAU`, so all four arms sharing those four values were
summed into one row — 53,748 bridges is `cap20`'s 18,248 plus the three
tau arms. Scores are unaffected. **Rule: every constant that
distinguishes two arms belongs in the accounting key**, or the counts
silently merge.

### Where gap closing stands, finally

| knob | verdict |
|---|---|
| **`GAP1_CAP_FRAC` 0.003 → 0.02** | **the win, +0.002 on the leaderboard** |
| the candidate requirement | drop it, part of the same win |
| `GAP_SPAN` 1 → 2 | fails the rule, destroys divisions in 6bba |
| `GAP1_GATE_UM` 9 → 12 | passes the rule, loses to `cap20` everywhere |
| four hand-made gates | all rationing, none selecting |
| **a learned gate at 0.97 AUC** | **loses monotonically** |

**Nothing further to try here.** The shipped `cap20` is the end of this
line.

## 2026-08-28 -- ⭐⭐⭐⭐ THE EDGE RANKER IS NOT MARGINAL: 0.9693 AUC against the ruler's 0.8829. (The pipeline arms were void and the instrumentation said so.)

`biohub-rank5`, 233 min CPU. Anchors held — `cap20` reproduced +0.0013 /
+0.0085 / +0.0056 exactly.

### The user's question, answered

> 这个边排序器看起来有点鸡肋，你先去试试吧，如果不行，我反而觉得这个边
> 排序器可能有点问题

A ranker was trained on **two-frame** pairs (`poolsize`'s label rule with
one extra hop through `child_of`: positive when j matches a GT
GRANDCHILD of i's GT node), on 116 6bba non-holdout datasets only.

```
138,734 two-frame pairs, 60.65% positive
18-feature ranker AUC   0.9693
distance alone   AUC    0.8829
the ranker buys         +0.0863
```

**Column 0 is the physical distance — exactly what `_gap_close`'s
Hungarian already sorts on.** The ranker beats it by 8.6 AUC points at
the decision gap closing actually makes. **The fallback hypothesis is
not triggered: the ranker is sound.**

Single-column AUCs (below 0.5 means informative with the sign flipped):

| col | AUC | flipped |
|---|---|---|
| `d` distance | 0.117 | **0.883** |
| `md` distance from the velocity prediction | 0.129 | **0.871** |
| `rank` among the parent's candidates | 0.132 | **0.868** |
| `n_cand_i` | 0.160 | 0.840 |
| `dxy` | 0.170 | 0.830 |
| **`score_c` child detection score** | **0.757** | — |
| `score_p` parent detection score | 0.697 | — |
| `d-md` | 0.612 | — |
| `t/T`, `\|v\|` | 0.508, 0.491 | ~nothing |

Three geometric columns sit at ~0.87 individually and the combination
reaches 0.969. **`score_c` at 0.757 is the strongest NON-geometric
feature**, which is a second reason the detector line matters: a better
detection score feeds straight into this.

Probability quantiles (1/10/25/50/75/90/99%): **0.007 0.020 0.076 0.863
0.985 0.996 0.999** — strongly bimodal, median 0.863.

### The pipeline arms were VOID, and the run said so itself

```
bridge pairs looked up 54,744, of which 54,744 had NO probability (100.00%)
```

Every bridge was rejected, so `tau0.2`, `tau0.5` and `tau0.8` measured
"reject everything", not the ranker — and they returned **identical**
deltas (+0.0001 / −0.0004 / −0.0006) at all three thresholds, which is
the signature of a threshold that never fires. The decomposition agrees:
their 6bba `node_ratio` is −0.0223 against BASE's −0.0215, i.e. slightly
FEWER nodes than the shipped configuration, which is what "no gap
closing at all" looks like.

**THE CAUSE.** `solve_one` builds the ILP's node coordinates with

```python
coords = np.asarray(np.round(coords_rows), np.int16)
```

so a graph node's xyz is the seed position **rounded to an integer**,
while `seeds_of` returns the sub-voxel refined float. `_seed_to_node`
matched at a 1e-6 tolerance and therefore never matched anything.

**Fixed at `SEED_MATCH_VOX = 1.0`, which is safe rather than merely
convenient**: rounding moves a coordinate by at most 0.5 per axis, so at
most √3/2 = 0.87 voxels, while `_physical_nms` has already forced 4.0 µm
between any two kept seeds — 2.46 voxels in z and 9.8 in xy. The nearest
wrong candidate is more than twice the tolerance away.

### The instrumentation is the finding

`_BRIDGE_MISS` was added while writing the script, on the reasoning that
*"a high miss rate would make the tau arms strict rather than selective
and the comparison would be meaningless"*. It was 100%. Without it, three
identical rows would have been read as "the threshold does not matter"
and the ranker would have been wrongly convicted — **which is exactly
the conclusion the user pre-registered as the interesting one, so the
error would have been believed.**

**Generalise it: when an arm can silently degenerate into a different
arm, print the count that distinguishes them.** A gate that rejects
everything, a budget that binds, a lookup that misses — each looks like
a result and reads like one.

## 2026-08-28 -- ⭐⭐⭐⭐⭐ DeepCenter covers 69.5% of our detector's misses against our own repair pool's 20.2%, and flickers 43% less

`biohub-detcmp` v3, 21 min GPU, 8 Zebrahub crops, `best.pt`.

**The venue was the user's call** — *"在zebrahub上面比就可以了，主打一个
谁都没看过"* — and it was right for a reason we only learned afterwards:
DeepCenter finds 0–1 cells on 6bba with the wrong checkpoint, so had we
tested only there, "it does not work" and "6bba is unfair to it" would
have been inseparable. A neutral venue kept those apart.

### Coverage — the number this run existed for

| DeepCenter | covers OUR misses | we cover ITS misses |
|---|---|---|
| **@ 0.05** | **69.5%** | 49.4% |
| @ 0.10 | 64.3% | 51.2% |
| @ 0.15 | 58.1% | 57.9% |
| @ 0.30 | 28.6% | 84.2% |

**Our own sub-threshold band [0.05, 0.15) covers 20.2%** (`flicker.py`,
same denominator). DeepCenter covers **69.5%** — 3.4×, and the
pre-registered bar was "much above 20.2%".

**And it is ASYMMETRIC**: it covers 69.5% of our misses while we cover
only 49.4% of its. The two detectors fail on genuinely different cells,
which is the condition this run pre-registered as the case for a union.

### Recall and flicker

| | recall | kept/frame |
|---|---|---|
| OURS @ 0.15 | 0.9642 | 272.7 |
| DeepCenter @ 0.15 | 0.9644 | 281.3 |
| **DeepCenter @ 0.05** | **0.9784** | 296.4 |

At the same threshold they are level; at 0.05 DeepCenter takes **+1.42
points of recall** for **+8.7% output**.

| | always | intermittent | never | gap=1 | gap≥2 |
|---|---|---|---|---|---|
| OURS @ 0.15 | 67.0% | **32.4%** | 0.6% | 2,660 | **1,465** |
| DeepCenter @ 0.05 | **76.5%** | **23.2%** | 0.3% | 1,889 | **835** |

**43% fewer long dropouts.** That is the population `_gap_close` exists
to patch and cannot reach beyond one frame.

### ⚠ CORRECTION: DeepCenter is NOT a temporal model

I told the user a temporal model would be the tool for flicker and that
this was DeepCenter's likely mechanism. **Wrong** — I conflated it with
the separate `biohub-temporal-unet3d-seed314159-v1` dataset.
`_DCDeepCenterUNet3D` takes a single volume, exactly as ours does. So
the flicker advantage has **no temporal explanation**; it is simply a
differently-biased per-frame detector. The measurement stands, the
mechanism I offered for it does not.

### What must be carried into any conclusion

1. **One domain, 8 crops. This does not satisfy the three-domain rule.**
2. Zebrahub is the only fair venue for DeepCenter (trained on all 71
   44b6, model-selected on all 128 6bba) and also the domain furthest
   from the hidden test.
3. The public solution has this mechanism and **disables** it — as a
   VETO, the opposite direction from supplying candidates.
4. **Recall is not the metric.** More detections can cost adj_J through
   the node-count term and through wrong links. Nothing here measures
   adj_J.

### The three runs it took to get here, and why

| run | result | cause |
|---|---|---|
| `detcmp` v1 | DeepCenter: 0.4 peaks/frame vs our 272.7, logit overflow | wrong checkpoint |
| `detcmp` v2 | four normalisations, **byte-identical logits** | normalisation was never the variable |
| `dcprobe` | 0–1 peaks on **6bba**, DeepCenter's own val split | wrong checkpoint, isolated |
| `dcprobe` v2 | `best.pt` 114–654 peaks, `checkpoint_last.pt` 0–1 | **answered** |

Three things worth keeping:

- **`GroupNorm` after a bias-free convolution erases any affine input
  transform.** v2's four normalisations returned identical logits to
  three decimals. That one row disproved the hypothesis faster than any
  reasoning could, and it was only visible because the run printed input
  and logit RANGES rather than only a recall number.
- **"Flip averaging made it worse" is a broken-wiring signal, not a
  weak-model signal.** v1 went 0.0017 → 0.0000 under 4-view averaging. A
  working model gets steadier; one whose views disagree gets cancelled.
  That was visible in v1 and should have stopped the read there.
- **The DeepCenter arm never had an anchor.** Our arm reproduced
  `flicker3`'s Zebrahub figures exactly (32.4% intermittent, 0.6%
  never), which is the only reason the fault was localisable at all.
  `working-agreement.md`'s "Always anchor" was applied to the inherited
  arms and not to the new one.

**And the checkpoint error came out of our own notes.** `reference.md`
said *"Do not read the 7x val_loss as 'got worse' — the loss is
positive-unlabeled BCE that punishes finding unannotated cells."* That
is a warning **not to conclude** "worse". It was used as a positive
argument that the epoch-500 checkpoint therefore finds MORE unannotated
cells — which is what a repair pool wants — and that picked a checkpoint
whose weights have grown 2.5× and whose largest logit anywhere in a 64³
volume is −0.34. **A note saying "do not infer X from this" is not
evidence for not-X.** `reference.md` now carries the measurement.

### The next step, and its price

Pipe DeepCenter into `_gap_close`'s repair pool as a second candidate
source — ADD-ONLY, primary detections untouched, so "worse in other
respects" cannot hurt. That needs DeepCenter detections cached for the
91 evaluation datasets (12 6bba + 71 44b6 + 8 Zebrahub), roughly **1.8 h
of GPU** at the measured ~0.7 s/frame, then the usual CPU sweep.

## 2026-08-28 -- ⭐⭐⭐⭐⭐ 0.922. FIRST SUBMISSION EVER TO BEAT 0.921, and the leaderboard turns out not to mean what we thought

**Ref 55817630 = 0.922.** Control ref 55772253 = 0.920, so **+0.002**.
Previous all-time best was 55707206 = 0.921, held since 2026-08-23.

| | |
|---|---|
| change | `GAP1_CAP_FRAC` 0.003 → 0.02 **and** the candidate requirement dropped |
| local, three domains | +0.0013 / +0.0085 / +0.0056 |
| **pre-registered prediction** | **0.920 (no change), 0.921 the upside** |
| **actual** | **0.922 — above my own upside** |
| hidden-test bytes | 267,030,255 → **268,154,477 (+0.42%)** |
| local node growth | +0.62% |

The output grew on the hidden test by roughly what it grew locally, so
the change behaved there the way it behaved here. **I under-predicted,
and that is the first time in this project that a local gain arrived
LARGER than forecast.** Every previous prediction erred optimistic.

### The holdout is now 1 for 4, and the one that worked is a different KIND of change

| change | local | LB |
|---|---|---|
| windowing | +0.0019 | −0.001 |
| ranker pool 32 → 187 | +0.0073 | −0.001 (indistinguishable) |
| `SHORT_MIN` 6 → 4 | +0.0025 / +0.0054 / +0.0045 | 0.000 |
| gate 0.15 → 0.12 | +0.0055 / +0.0045 / +0.0121 | −0.002 |
| **budget 0.003 → 0.02, no gate** | **+0.0013 / +0.0085 / +0.0056** | **+0.002** |

**The four failures were all CONSTANT TUNING, where the local number was
the entire argument. The success was a CONSTRAINT REMOVAL with a
measured mechanism** — `wide44` showed the budget bound in 30 of 83
datasets and repaired 36% of demand, so work that should have happened
provably was not happening. And note it had the SMALLEST weakest-domain
delta of the five (+0.0013). **The size of the local delta predicted
nothing; the presence of a mechanism did.**

n = 1, so this is a hypothesis, not a law. But it is the first thing
that has ever distinguished a leaderboard success from a leaderboard
failure in this project, and it is cheap to apply: before submitting,
ask what is PROVABLY going wrong, not how big the local number is.

### ⚠ CORRECTION: the daily submission limit is FIVE, not two

`max_daily_submissions: 5`. I have been rationing submissions at two a
day for days and saying so out loud. **The "two" is the FINAL SELECTION
limit and I conflated the two things.** The submission history already
contradicted me — 3 submissions on 08-23 and 3 on 08-25.

### The leaderboard is not shaped the way rank suggested

Rank is **409 of 2,788**, which is WORSE than the 334 of 2,717 recorded
a few days ago, despite the score going up. The distribution says why:

| score | teams (top 200 sampled) |
|---|---|
| 0.962 | 2 (the lead) |
| 0.930–0.955 | sparse, a few dozen |
| **0.927** | **~77 teams at exactly this value** |
| 0.926 and below | the field |
| **0.922** | **us, rank 409** |

**Seventy-seven teams share one value to three decimals.** That is not a
distribution, that is a shared artefact.

### And the clean public baseline is 0.915, not 0.927 — we are ABOVE it

`yusuketogashi/no-hack-biohub-cell-another-approch-3rd` (149 votes, last
run 2026-08-24) states in its own header:

> Fixed baseline: Biohub 159B, clean leaderboard `0.915`
> Immediate target: `0.916+`

**So the public "no hack" line sits at 0.915 and ref 55817630 is seven
ticks above it.** The competition's own notebook titles use "No Hack" as
a badge — that line, `🧬Clean Approach + Lightweight Local CV | No Hack`,
`No-Hack | Biohub Cell Another Approch 3rd` — and there is a separate
`kirneo/metric-hack-last-call-update` at 133 votes, last run the same
day.

**NOT investigated further, on purpose.** `reference.md` already carries
this project's position — *"Do not exploit: unmatched forks add no TP,
and fork-spam is the genre of the hub-node exploit patched
2026-07-17"* — and one exploit in this competition has already been
patched by the hosts mid-contest. Whether the 0.927 cluster is that
genre is **unconfirmed**; what is confirmed is the clean line at 0.915
and our 0.922 above it. Reading rank 409 as "we are behind the field"
is therefore wrong: it is "we are behind the field plus whatever 77
teams are all doing identically".

### One legitimate thing worth stealing from the public line

That notebook's inputs are `pilkwang/biohub-temporal-unet3d-seed314159-v1`
and `pilkwang/biohub-local-association-ranker-unet300-v1` — a **temporal
UNet3D** and a **300-epoch** local-association ranker — plus a two-model
detection ensemble blending independently seeded temporal models at
α ≈ 1/2. **We run the 50-epoch support pack and a single detector
stack.** Their end-to-end result is lower than ours (0.915 vs 0.922),
which says their LINKING is behind, not necessarily their DETECTION.
`logbase` already measured that our detector is worth +0.0566 over the
best classical filter, so detection is the expensive half and we have
never tried a better one.

## 2026-08-27 -- ⭐⭐⭐⭐⭐ wide44: the 12-dataset 44b6 sample OVERSTATED by a third, that arm DOES have divisions, and gap closing RAISES div_J

`biohub-wide44`, 145 min, BASE vs `nocand_cap20`, **all 71** 44b6 plus
the 12 6bba holdout. Zebrahub dropped on purpose; not a three-domain run.
Anchors held: 6bba BASE adj_J **0.9279**, `nocand_cap20` **+0.0013**.

### 1. The answer to the user's question

| | 12 datasets (`sorted(...)[:12]`) | **all 71** |
|---|---|---|
| 44b6 adj_J delta | +0.0113 | **+0.0085** |

**The prefix sample overstated by a third.** +0.0085 is now measured on
the whole population, so the sampling question is closed rather than
re-drawn.

**And the effect is not carried by one dataset:**

| | 6bba (12) | 44b6 (71) |
|---|---|---|
| largest single contribution | **42.9%** | **8.5%** |
| improved / worsened | 4 / 8 | 37 / 34 |

⚠ **The fragile arm turns out to be 6bba, not 44b6.** Read the
magnitudes before the counts, though: **7 of the 8 "worsened" 6bba
datasets moved by ≤0.0003**, which is the change doing nothing at all;
only `6bba_61dd1e0d` at −0.0021 is a real loss, against four gains of
+0.0028…+0.0054. So the honest description is **"inert in 7 of 12,
clearly helps in 4, hurts in 1"**, not "helps less than half".

44b6's extremes are wider: `44b6_0c582fdc` **+0.1002** (0.8357 →
0.9359, but weight 77, so 2nd by contribution), against
`44b6_9bfa6a0a` −0.0223 and `44b6_144b256d` −0.0168. Everything else
sits inside ±0.008.

### 2. RETRACTION: 44b6 is not division-free. Those twelve were.

| | 12 datasets | all 71 |
|---|---|---|
| 44b6 `div_J` at BASE | **0.0000** | **0.0455** |

The entry above says *"44b6's div_J is also exactly 0.0000 in every arm
of every run — that arm has essentially no annotated divisions either"*
and calls it a sign of a degenerate measurement. **That was a property
of the twelve-dataset prefix, not of 44b6.** Withdrawn.

### 3. The user's correction, confirmed directly

**44b6's `div_J` RISES by +0.0148** (0.0455 → 0.0602) under
`nocand_cap20`.

I had written that a bridge removes `gs` from the orphan pool and so
"destroys a division candidate", framed as a cost. The user corrected
the sign: `gs` is an orphan because its predecessor was MISSED, so
making it a second daughter manufactures a FALSE division, and bridging
it removes a false candidate. **A +0.0148 rise in div_J is that
statement measured.**

It read +0.0000 before only because the twelve datasets in use had
`div_J` identically zero — **a quantity fixed at zero cannot show a
change**, which is a second, separate way the prefix sample was hiding
things.

### 4. The metric decomposed, measured rather than derived

| | `edge_J` BASE → new | delta | node_ratio |
|---|---|---|---|
| 6bba | 0.9258 → 0.9275 | **+0.0017** | −0.0215 → −0.0179 |
| 44b6 | 0.8490 → 0.8579 | **+0.0089** | −0.1545 → −0.1480 |

The derived estimate given to the user for 6bba was **+0.0017** —
exact. The 44b6 estimate was +0.0116, and the whole discrepancy is the
+0.0113 → +0.0085 shrink, not the derivation.

**44b6's node_ratio over all 71 is −0.1545**, far more negative than the
prefix sample's −0.0875: the pipeline emits **15.5% fewer nodes than the
estimated cell count** there. Several datasets therefore score `adj_J`
above 1.0 (`44b6_808952d6` 1.0093, `44b6_8f9ecab4` 1.0144), which is
`reference.md`'s "no upper clamp" showing up in real output.

### 5. NEW: the shipped configuration is itself budget-starved

| arm | made | allowed | demand | capped |
|---|---|---|---|---|
| BASE (candidate required, 0.003) | 4,735 | 6,637 | **13,146** | **30 / 83** |
| `nocand_cap20` | 17,286 | 44,473 | 17,286 | **0** |

**BASE repairs 36% of demand and is stopped by the BUDGET, not by the
candidate requirement, in 30 of 83 datasets.** That is worse than gap4's
pooled figure suggested (4 of 32). The budget was never a safety margin;
it has been the binding constraint on the shipped pipeline all along.

### Where this leaves ref 55817630

Submitted before this run, on exactly `nocand_cap20`. Still supported in
all three domains — **6bba +0.0013, 44b6 +0.0085 (71 datasets),
Zebrahub +0.0056 (8 crops, from gap4)** — with 44b6 a third smaller than
the number in the submission description, and div_J now known to rise
rather than stay flat.

## 2026-08-27 -- ⭐⭐⭐⭐⭐ TWO THINGS THE USER CAUGHT: the 44b6 arm is `sorted(...)[:12]` of 71, and every submission since 08-23 has been built on the WORSE base

Both from one exchange, neither measured before, both cheap to fix.

### 1. The 44b6 arm is the first 12 of 71, alphabetically, never varied

```python
SET_44B6 = sorted(n for n in CACHES if n.startswith("44b6_"))[:12]
```

**There are 71 `44b6_` datasets** (`reference.md`: DeepCenter's split
manifest is exactly 71 `44b6_` + 128 `6bba_`), all 199 detection caches
are mounted, and every sweep this project has run used the same twelve
lowest hex ids: `0113de3b 0b24845f 0c582fdc 0db75fae 12dfb391 144b256d
1574802b 18ced818 1d530831 24264f12` and two more, all inside
0x01–0x2f. **Whether those ids encode anything — acquisition order,
embryo, session — has never been checked.** If they are content-free
hashes the subset is arbitrary; if not, it is systematically skewed.

The user's trigger was `gap4`'s **+0.0113** on that arm: *"这个波动不太
正常"*. It is out of line with every other number this project has
produced (typically 0.001–0.005), and **the log already carried the
reason and nobody connected it**:

| domain | datasets | GT edges | per dataset |
|---|---|---|---|
| `6bba` | 12 | 10,313 | 859 |
| `44b6` | 12 | **2,526** | **210** |

44b6 carries **24% of 6bba's ground truth**, so ~2x the standard
error; its thinnest dataset has **49 edges**; annotation coverage is
0.77% against 9.71%. **A +0.0113 swing on 2,526 annotated edges is
about thirty edges changing hands**, and `gap4` printed no per-dataset
deltas, so one or two datasets could be carrying all of it invisibly.
~~`44b6`'s `div_J` is also exactly 0.0000 in every arm of every
run — that arm has essentially no annotated divisions either.~~
**RETRACTED 2026-08-27 by `wide44`: over all 71, 44b6's `div_J` at
BASE is 0.0455, and `nocand_cap20` RAISES it by +0.0148. The zero
was a property of the twelve-dataset prefix, and a quantity pinned
at zero cannot show a change — a second way that sample was
hiding things.**

**The fix is cheap and there is no longer a reason not to.** The
12-dataset limit existed for runtime, and the ranker-leak rule does not
bind here: gap closing is POST-LINK, so all arms share one ranker, one
detection set and one ILP solution and the leak cancels in the delta.
At gap4's measured 28 s per dataset-config, BASE vs `nocand_cap20` over
**all 71** 44b6 plus 12 6bba plus 8 Zebrahub is **~1.8 h**. It answers
three questions at once: is +0.0113 a property of these twelve; is it
one dataset or spread; and what is the `edge_J` gain MEASURED rather
than derived (carry thr5's `_with_ratio()`, which gap4 should have had
and did not).

### 2. Every submission since 2026-08-23 has been built on the 0.920 branch

The user: *"我们为什么不以 0.921 分数的那个为基底"*.

Their stated mechanism does not hold — D32 and D187 emitted **262,623,879
vs 262,803,696 bytes on the hidden test, +0.07%**, so D187 does not
select meaningfully more points, and `thr5` showed the node-count term is
linear with a constant slope, so there is no boundary to hit at all.

**But the conclusion is right, for a stronger reason, and it is a cost
nobody has been accounting for.**

| | 55707206 (D32) | 55757064 (D187) |
|---|---|---|
| LB | **0.921** | 0.920 |
| hidden-test bytes | 262,623,879 | +0.07% |
| local 12-dataset 6bba | 0.9313 | 0.9386 (**+0.0073**) |
| **edge AUC** | **0.9884** | **0.9864 (LOWER)** |

**Two independent signals say D187 is not better** — the leaderboard
tick and its own lower edge AUC, which `biohub-poolsize` explained away
as "a bigger pool means a harder validation split" and which this log
later retracted. The only thing supporting D187 is the local
12-dataset 6bba number, which is the class of measurement now 0 for 3.

**And final selection allows TWO submissions, one of which is already
committed to 55707206 = 0.921.** So everything built on the D187 branch
must first repay the tick spent switching rankers before it buys
anything. Ref 55817630, submitted today, would have to fully deliver its
local +0.001 just to TIE a submission we already hold — and it would
consume the second final slot doing it.

**Proposal: put `nocand_cap20` on a D32 base.** That is a clean
single-variable test against the 0.921 anchor, and only 0.922 counts as
progress.

⚠ `submit_ranker.py` actively refuses D32:

```python
MIN_POOL = 100
assert _pool is not None and len(_pool) >= MIN_POOL, ...
```

added deliberately so a stale 32-dataset pkl could not be shipped by
accident. Going back is a conscious change to that guard, not a
constant tweak.

### The general lesson

**Improvements compound on the base you SHIP, not the base you measure
on.** Four submissions of work have been accumulating on a branch whose
own leaderboard point is one tick below a submission already banked.
Nothing measured on that branch was wrong; it was just being added to
the wrong starting number.


### 3. Why D187 "lost": more than half of its local gain was TWO division events

The user: *"为什么 D187 反而比 D32 差呢,实在是太奇怪了"*. It is not
strange, and "worse" is not established either — one tick is the LB's
reporting resolution and the two emitted output within 0.07%. What IS
established is that the +0.0073 did not appear. Decomposed:

| | D32 | D187 | delta |
|---|---|---|---|
| local score | 0.9313 | 0.9386 | **+0.0073** |
| `adj_J` | 0.9222 | 0.9254 | +0.0032 |
| `0.1 x div_J` | 0.0091 | 0.0132 | **+0.0041 (56%)** |

| | div TP | div FP | div FN |
|---|---|---|---|
| D32 | **3** | 21 | 9 |
| D187 | **5** | 26 | 7 |

`div_J` = TP/(TP+FP+FN): 3/33 = 0.0909 against 5/38 = 0.1316.

**The 12-dataset holdout contains TWELVE annotated divisions in total**
(3+9 = 5+7 = 12). D187 caught five, D32 caught three. **The entire
+0.0041 — 56% of the headline gain — is two events.**

And proposing more forks is nearly free on sparse ground truth: the
user's own verified observation is that **a fork lying entirely among
unannotated cells enters no FP set, so ~97% of extra forks are
INVISIBLE** (913 extra forks cost only -0.0027 adj_J, ~30 visible edge
FPs). D187 proposes ~30% more matched forks (31 against 24) and two
more happened to land.

**So the genuinely measured part of D187's advantage is +0.0032 of
adj_J** — inside the range this holdout has already inverted twice.
Nothing anomalous needs explaining; the anomaly was reading +0.0073 as
an improvement in the first place.

⚠ **CORRECTION to this entry's own section 2**: it called the lower edge
AUC (0.9864 vs 0.9884) an independent signal against D187. That is too
strong. The two AUCs were measured on validation splits drawn from
different pools — different populations, the denominator trap again —
and the log only ever marked the "bigger pool, harder split"
rationalisation as *may have been wrong*. Correct statement: **the AUC
is not evidence FOR D187**, not that it is evidence against.

**Which leaves the slot arithmetic as the whole argument**, and it does
not depend on either ranker being better: the two are indistinguishable,
55707206 = 0.921 already occupies a final-selection slot, so anything
built on the D187 branch must climb a tick merely to TIE something
already banked while consuming the second slot.

### 4. STANDING INSTRUCTION 2026-08-27: sample 44b6 at random

> 你后面,所有测试集的 44b6 都要随机抽

Implemented in `gap4.py`, which is the template every later assembly is
cut from:

```python
N_44B6 = 12
SEED_44B6 = 20260827
_ALL_44B6 = sorted(n for n in CACHES if n.startswith("44b6_"))
SET_44B6 = sorted(random.Random(SEED_44B6).sample(
    _ALL_44B6, min(N_44B6, len(_ALL_44B6))))
```

**Seeded, not freshly random**, so a recorded anchor still reproduces —
a sample that changes every run makes every anchor meaningless. **The
chosen names print every run**: a sample nobody can see is a sample
nobody can check. To test whether a result depends on the sample,
change `SEED_44B6`; that is a different question from changing
`N_44B6`, which changes the precision.

`SET_ZH = sorted(ZH)[:8]` is left alone — 8 is every Zebrahub crop
there is, so the prefix is the whole population. `SET_6BBA` is the
recorded 12-dataset holdout and must stay exactly those twelve, because
they are the only 6bba datasets outside D187's training pool.
## 2026-08-27 -- ⭐⭐⭐⭐ how an EDGE ranker moves DIVISIONS, and the coupling gap closing has to it

The user: *"为啥边检测器会抓分裂"*. The name misleads — the ranker does
not score edges in isolation, it decides the shape of the whole graph,
and **a division IS a graph shape: one node with two outgoing edges.**

### The split, measured on the 4 visible datasets (ref 55817630's run)

| dataset | ILP forks | final forks |
|---|---|---|
| 44b6_0113de3b | 11 | 54 |
| 44b6_0b24845f | 15 | 130 |
| 6bba_05b6850b | 6 | 18 |
| 6bba_05db0fb1 | 15 | 143 |
| **total** | **47** | **345** |

**The ILP makes 14% of the forks. `_add_safe_divisions` adds 86%.**

### Path 1 — the ILP, 14%

The solve is **capacity 2**: a node may take two outgoing edges, and
`div_w = 0.55` is what opening a fork costs in the objective. Whether it
is worth paying depends on the ranker's probabilities for both child
edges. So the ranker buys these forks directly.

### Path 2 — the orphan pool, 86%, and it is the main road

`_add_safe_divisions` opens with

```python
orph = [g for g in ids if not pred.get(g)]
```

**An orphan is a node with no predecessor, and which nodes have a
predecessor is decided entirely by what the ILP linked** — that is,
entirely by the ranker. The patch then takes each node with exactly one
child and looks for a nearby orphan to serve as the second daughter,
subject to `DIV_CHILD_UM` 10, `DIV_PARENT_UM` 12, `DIV_SISTER_UM` 15
and a requirement that the two daughters actually diverge
(`DIV_DIVERGE_UM` 2.25).

**So: the harder the linker links, the fewer orphans survive, and the
less raw material the division patch has.** This is exactly the worry
`ranker_fine`'s header recorded — a more confident 1:1 linker "steals
second daughters into wrong continuations, draining the orphan pool the
division patch mines".

For D187 that worry appeared not to materialise: div_J 0.0909 → 0.1316.
⚠ **Discount that reading.** As decomposed in the entry above, the whole
of that move is **two events out of the holdout's twelve annotated
divisions**. The explanation ("a ranker that is more CORRECT rather than
merely more confident feeds the mechanism instead of starving it") may
be true but it was never measured.

### The coupling to gap closing — I had the SIGN backwards

I first wrote: `_gap_close` executes `pred[gs] = ng`, so every bridge
removes `gs` from the orphan pool and **destroys a division candidate**,
framed as a cost to watch. **The user corrected it and they are right:**

> 你不能认为 gap_close 会消灭分裂候选，它本来就是连边的，它反而在消灭
> 分裂错误的候选…说白了这个点本来也不太会成为分裂候选

**`gs` is an orphan because its predecessor was MISSED, not because the
cell was born there.** Turning it into a second daughter would
manufacture a FALSE division — grafting a continuing cell onto a
neighbouring track. So bridging it removes a **false** division
candidate. The sign is a benefit, not a cost.

**And structurally gap closing cannot even reach a TRUE division
candidate:**

```python
ends = [g for g in by_t.get(t, []) if g not in succ and g in pred]
```

A bridge starts only from a node with **no successor at all**. A real
mother has already linked at least one daughter, so she is in `succ` and
can never be an `ends` node. The "gap closing steals a missed second
daughter from her mother" case is off the normal path by construction.

**Why it is invisible in the numbers, quantified.** 6bba at BASE is
5 TP / 26 FP / 7 FN, `div_J` = 5/38 = 0.1316. Removing just **three** of
those 26 false divisions would give 5/35 = 0.1429, **+0.0113 of div_J
and +0.0011 of score**. gap4 measured exactly **+0.0000** for every arm
in 6bba and 44b6. So not even three false divisions were touched: **the
bridged orphans and the division proposals barely intersect**, which is
what the user said — those nodes were never likely to become division
candidates in the first place.

`_add_safe_divisions`'s gates are why: the parent must have exactly one
child, the orphan within `DIV_PARENT_UM` 12 µm of the parent, the sister
within `DIV_SISTER_UM` 15 µm, both daughters must have exactly one
successor each, and they must diverge by more than `DIV_DIVERGE_UM`
2.25 µm.

## 2026-08-27 -- ⭐⭐⭐⭐⭐ gap4: THE BUDGET WAS THE WHOLE PROBLEM. Raise it, drop every gate, and 44b6 gains +0.0113 (REVISED to +0.0085 by `wide44` above — that arm was 12 of 71)

`biohub-gap4`, 177 min, 8 arms × 3 domains, D187, `SHORT_MIN` 4.

**All three anchors reproduced to the digit** — BASE 0.9279 / 0.8553 /
0.8037, `nocand` +0.0009 / +0.0012 / +0.0034, `len8` +0.0013 / +0.0043
/ +0.0019 — so the gate REORDERING is confirmed behaviour-preserving on
real data, not just on the 400 synthetic movies.

### The result

| arm | 6bba | 44b6 | zh | bridges made |
|---|---|---|---|---|
| `nocand` | +0.0009 | +0.0012 | +0.0034 | 2,195 |
| `len8` | +0.0013 | +0.0043 | +0.0019 | 1,689 |
| `or4` (the user's, ≥4) | +0.0010 | +0.0034 | +0.0028 | 2,143 |
| `or8` (the user's, ≥8) | +0.0012 | +0.0020 | +0.0023 | 1,962 |
| **`nocand_cap20`** | **+0.0013** | **+0.0113** | **+0.0056** | **4,772** |
| `len8_cap20` | +0.0013 | +0.0043 | +0.0019 | 1,816 |
| `or8_cap20` | +0.0012 | +0.0040 | +0.0024 | 2,858 |

`div_J` is +0.0000 in 6bba and 44b6 for every arm; Zebrahub moves
+0.0001…+0.0010.

### The bridge counts say why, and they close the argument

| need_cand | min_len | mode | cap_frac | made | allowed | demand | capped |
|---|---|---|---|---|---|---|---|
| False | 0 | always | 0.003 | 2,195 | 2,297 | 2,195 | **24 / 32** |
| False | 0 | always | **0.02** | **4,772** | 15,420 | 4,772 | **0** |
| False | 4 | if_nocand | 0.003 | 2,143 | 2,297 | 2,503 | 22 |
| False | 8 | always | 0.003 | 1,689 | 2,297 | 4,339 | 5 |
| False | 8 | always | 0.02 | 1,816 | 15,420 | 4,772 | 0 |
| False | 8 | if_nocand | 0.003 | 1,962 | 2,297 | 3,406 | 12 |
| False | 8 | if_nocand | 0.02 | 2,858 | 15,420 | 4,772 | 0 |
| True | 0 | always | 0.003 | 1,107 | 2,297 | 4,214 | 4 |

- **True demand is 4,772 bridges. `nocand` at the shipped budget made
  2,195** — more than half of every repairable gap went unrepaired
  purely because the budget ran out, in 24 of 32 datasets.
- `len8` at a budget that cannot bind still makes only 1,816. **The
  length gate discards 62% of what is repairable.**
- **More bridges is monotonically better in the two domains that move.**
  4,772 → +0.0113 / +0.0056; 2,858 → +0.0040 / +0.0024; 1,816 →
  +0.0043 / +0.0019.

**Every gate this project invented — the candidate requirement, the
velocity cosine, the fragment length, the clearance radius — was
rationing a budget that was too small, not selecting good bridges.**
Give the repair enough budget and taking everything beats all of them.

### What this costs in nodes

4,772 bridges against the 765,667 nodes implied by the 0.003 allowance
is **+0.62% of nodes**, a node-count multiplier cost of **0.00062** —
already inside the deltas above, which are net. Compare the two failed
submissions at +1.61% and +10.3%.

### The user called the mechanism before the run

> 其实预算绑死的情况下,最好是打分,算更优

Right about the diagnosis; the fix turned out to be simpler than the
remedy. Under a binding budget the allocator hands out `max_gaps` in
frame order, so what gets repaired is whatever sits at a low frame
number. **But the budget does not have to bind at all.** At
`GAP1_CAP_FRAC = 0.02` the allowance is 15,420 against 4,772 demand and
there is nothing left to ration. The scoring allocator recorded in the
entry below is therefore NOT the next step — it only matters if a
budget is reintroduced, or if demand is grown past 15,420.

### ⚠ THIS REOPENS A CLOSED LINE

`progress.md`'s summary table carries:

> gap closing (span, interpolation) — **closed.** all arms fail the
> rule; the span never mattered

**That verdict came from `gap2`, which ran entirely at
`GAP1_CAP_FRAC = 0.003` with the cap binding.** Widening the span
creates more demand, and under a binding budget more demand only
reshuffles who gets the fixed allowance — it cannot show a gain. The
same applies to `GAP1_GATE_UM = 9.0`, which is what holds demand at
4,772 in the first place. **Both were measured behind a constraint that
has now been shown to dominate the result, and both should be re-run at
the raised budget.**

### Reading the (C, L) decomposition — with a caveat that voids it

`or8 − BASE = d`, `nocand − BASE = d + e`, `len8 − BASE = d − b`, so
b = +0.0012−0.0013 / +0.0020−0.0043 / +0.0023−0.0019 = −0.0001 /
**−0.0023** / +0.0004 and e = −0.0003 / −0.0008 / +0.0011.

**Do not use these.** Every arm in that system ran at the shipped
budget with the cap binding in 5 to 24 of 32 datasets, so each group's
apparent "value" includes how much of the allowance it displaced from
other groups. It is not a measurement of bridge quality. Redo it at
cap 0.02 if it is ever needed.

## 2026-08-27 -- the DESIGN of gap4 (FINISHED; result in the entry above) (the user's call)

`biohub-gap4` pushed 2026-08-27, 8 arms × 3 domains, D187, `SHORT_MIN` 4.

### The user's policy

> 如果有候选点,可以直接补,没有候选点,就必须要 len 8

The candidate is **evidence** — a real sub-threshold peak sitting at the
interpolated position says a cell is there. The fragment-length test is
a **proxy**. `len8` demands the proxy even when the evidence is present,
so it deletes bridges a detected peak was backing. This policy demands
the proxy only when the evidence is missing.

With C = a qualifying candidate exists and L = both fragments ≥ N:

| policy | bridges | |
|---|---|---|
| BASE | C | what ships |
| `nocand` | everything | gap3 |
| `len8` | L | gap3 — deletes C-and-short |
| **the user's** | **C or L** | superset of BASE at the eligibility test |

Partition by (C, L) into a = C∧L, b = C∧¬L, d = ¬C∧L, e = ¬C∧¬L:

```
nocand − BASE = d + e   = +0.0009 / +0.0012 / +0.0034
len8   − BASE = d − b   = +0.0013 / +0.0043 / +0.0019
the user's    = d       ← three unknowns, two equations
```

Not derivable from gap3. `len8 > nocand` in 6bba and 44b6 gives
b + e < 0; Zebrahub gives the opposite; neither isolates `b`, which is
exactly the difference.

### THE BUDGET BINDS, AND IT IS HANDED OUT IN FRAME ORDER

`_gap_close` walks `for t in range(T - step)` and `break`s the moment
`n_added >= max_gaps`. progress.md already had the numbers and nobody
connected them:

| | |
|---|---|
| cap, `0.003 × nodes` | 476 bridges |
| eligible WITH the candidate rule | ~238 — never binds |
| eligible WITHOUT it | ~1,570 — **binds at 30% of demand** |

So under any loosened policy the budget binds, and **what gets the
budget is whatever sits at a low frame number, not whatever is a better
bridge.** Demonstrated on synthetic data:

| | superset held | BASE bridges LOST to `or8` |
|---|---|---|
| shipped budget 0.003 | 246 / 300 | **58** (both arms made exactly 600 — fully bound) |
| budget past demand | **300 / 300** | **0** |

**The user's read, and it is right**: raising the budget dodges the
problem; ranking the bridges solves it.

> 其实预算绑死的情况下,最好是打分,算更优

### The scoring allocator — DESIGNED, NOT BUILT

Two passes instead of one greedy sweep:

1. **Collect.** For every frame, run the same per-frame
   `linear_sum_assignment` and the same gates, but record
   `(score, t, ge, gs, picks)` and **mutate nothing**. With no bridges
   made, `ends`/`starts` are the original graph's endpoints, which is
   what should be ranked.
2. **Allocate.** Sort by score descending; accept greedily while the
   budget remains and `ge ∉ succ`, `gs ∉ pred`, and every candidate the
   bridge wants is still unused.

Keeping the per-frame Hungarian assignment in pass 1 preserves the
matching quality that already exists; only the *priority* across frames
changes. Candidate scores:

| score | available for | note |
|---|---|---|
| `−D`, the bridge distance | every bridge | already computed, free |
| `cand_sc[tt][idx]` | candidate-backed only | needs a floor for midpoint bridges |
| `min(frag_len_a, frag_len_b)` | every bridge | `_frag_len` caps at 12 |
| **the edge ranker** | every bridge | the strongest, and the most work |

**The edge ranker is the right answer and the expensive one.** The
pipeline has a trained 18-feature pair scorer and gap closing uses raw
Euclidean distance instead. Recorded in gap3 as left out on purpose:
its features are computed over the primary seed set (local density,
rank among candidates, per-frame counts) and rebuilding them for a
repair-pool point is not a small change. Still the obvious next thing
after a cheap score proves the allocator matters.

### The anchors, and why there are three

This assembly **reverses the gate order** — the candidate lookup now
runs before the length test, because "only when there is no candidate"
cannot be expressed otherwise. BASE, `nocand` and `len8` must all
reproduce gap3 to the digit.

Proved locally first, per the rule gap2 wrote in blood: **400 synthetic
movies × 7 arms, byte-identical to gap3**, and every arm added
1,580–12,284 edges so no arm was comparing two no-ops. The first version
of that test was nearly vacuous (tracks too short for a length-8 gate to
fire, candidates scattered so BASE made 4 bridges in 400 movies) and had
to be rebuilt.

⚠ One test artefact worth remembering: **do not compare raw edge tuples
across arms.** Inserted nodes are numbered from `next_id`, so an arm
making a different number of bridges renumbers all of them and every
edge touching an insertion compares unequal. That reported the superset
property failing 278 of 300 times — an artefact. Map back to the
original `(ge, gs)` endpoints.

### On len5/6/7, which the user asked first

Reasoned, not run. Across gap3's gate 0 → 4 → 8 the **binding domain is
6bba at every point** (+0.0009, +0.0010, +0.0013) and rises
monotonically, while Zebrahub at 8 is still +0.0019, above it. So 6bba
binds through the whole interval and 8 dominates 5, 6 and 7 unless
something non-monotone happens between. Crude linear extrapolation puts
the weakest-domain peak near gate 10 at +0.00145 — **+0.0002 over
len8, which is noise.** ⚠ `_frag_len` caps at 12, so `GAP_MIN_LEN > 12`
is unsatisfiable and silently disables gap closing entirely.

## 2026-08-26 -- ⭐⭐⭐⭐⭐ thr5 CORRECTS THE ENTRY BELOW. The node-count term is LINEAR with a constant slope, so there is no local/hidden asymmetry and it explains nothing.

`biohub-thr5`, 190 min, gates 0.15 / 0.18 / 0.22 / 0.26, `SHORT_MIN` 4,
D187. Anchor held: g0.15 reproduced thr4's `g0.15_s4` exactly, 0.9279 /
0.8553 / 0.8037.

### 1. The correction, which is the important half

The entry below (`THE NODE-COUNT PENALTY IS THE STORY`) claimed:

> Locally, adding nodes helps twice — better edges **and** a move toward
> the target. On the hidden test, if N_pred is at or over target, it
> helps once and hurts once.

**Both halves are wrong.**

`reference.md`: the penalty is `max(0, J * (1 - 0.1 * (N_pred -
N_true)/N_true))`. The `max(0, ...)` floors the whole product; it does
**not** clamp the ratio. Verified against thr5's own columns — 6bba
g0.15, `edge_J` 0.9258 × (1 − 0.1 × −0.0215) = 0.92779 against the
reported 0.9279.

**(a) A move toward the target HURTS, not helps.** Under target means a
negative ratio means a multiplier above 1. Adding nodes moves the ratio
up toward zero, which SHRINKS the bonus. Locally, adding nodes has
always been costing us.

**(b) There is no asymmetry, because the slope is constant.** Adding
ΔN nodes changes the multiplier by exactly `−0.1 × ΔN/N_true`,
regardless of where the ratio currently sits:

| starting ratio | add 1.61% of nodes | multiplier | adj_J change |
|---|---|---|---|
| −0.0215 (6bba, actual) | → −0.0054 | 1.00215 → 1.00054 | **−0.0015** |
| 0.0000 (hypothetical) | → +0.0161 | 1.00000 → 0.99839 | **−0.0015** |

Identical. The only way the cost could differ between local and hidden
is if the hidden `N_true` were a very different MAGNITUDE from ours —
about half, to double the cost — which is implausible if the hidden
split resembles the training split.

**(c) So both failed submissions' local gains were ALREADY net of the
node cost.** `SHORT_MIN` 4's +0.0025 / +0.0054 / +0.0045 and gate
0.12's +0.0055 / +0.0045 / +0.0121 are post-penalty numbers. The
leaderboard's 0.000 and −0.002 need a different explanation, and the
only one left standing is the boring one: **the local gain is not
present in the hidden data**, exactly as with windowing (+0.0019 →
−0.001) and the ranker pool (+0.0073 → −0.001).

The three-domain rule exists to fix precisely that, and its first
verdict is 0.920 → 0.920.

**One hole that cannot be closed from here**: `N_true` is the train
geff's `estimated_number_of_nodes`, and `reference.md` records that the
**test split has no such field**. What the official scorer uses for
N_true on the hidden test is unknown and unverifiable from our side.

### 2. Tightening the gate is dead in both independent domains

⚠ **This run's 44b6 arm is IN-SAMPLE.** All 12 datasets sit inside
D187's training pool, so it reports fit, not transfer. The script said
so at startup. **The independent domains here are 6bba and Zebrahub —
two.** The user chose D187 deliberately (it is the model the submission
runs) and on D187 no clean 44b6 subset exists.

| gate | 6bba | 44b6 (in-sample) | zh |
|---|---|---|---|
| 0.15 (shipped) | 0.9279 | 0.8553 | 0.8037 |
| 0.18 | −0.0042 | +0.0058 | −0.0236 |
| 0.22 | −0.0095 | −0.0095 | −0.0962 |
| 0.26 | −0.0166 | −0.0453 | **−0.2204** |

Both independent domains fall monotonically.

### 3. The decomposition, which is what the run was FOR

| | 6bba | | 44b6 | | zh | |
|---|---|---|---|---|---|---|
| gate | edge_J | ratio | edge_J | ratio | edge_J | ratio |
| 0.15 | 0.9258 | −0.0215 | 0.8473 | −0.0875 | 0.7976 | −0.0808 |
| 0.18 | 0.9193 | −0.0471 | 0.8465 | −0.1681 | 0.7709 | −0.1252 |
| 0.22 | 0.9102 | −0.0911 | 0.8214 | −0.2970 | 0.6924 | −0.2285 |
| 0.26 | 0.8975 | −0.1564 | 0.7768 | −0.4371 | 0.5631 | −0.3817 |

Three readings, none of which adj_J alone could have given:

- **`edge_J` is monotone DECREASING in the gate in every domain.**
  Loosening improves the edges everywhere; the node term is the only
  thing pushing back.
- **Tightening deletes real cells, not marginal false positives.**
  Zebrahub 0.15 → 0.26 loses 29% of `edge_J` relative. At 0.26 it emits
  38% fewer nodes than there are cells, and a missed detection costs
  twice (the GT edge becomes an FN, the neighbour's predicted edge
  becomes an FP).
- **The node-count bonus cannot pay for it.** Zebrahub's bonus grows
  from +0.008 to +0.038 across that range — 0.03 against a 0.23 loss.

**The threshold line is CLOSED at both ends.** Loosening: local gain,
leaderboard loss (ref 55780773, 0.918 against its control's 0.920).
Tightening: local loss in both independent domains. 0.15 stands.

`SHORT_MIN` 4 sits at ratio −0.0215 on 6bba, so the shipped pipeline is
already 2–9% BELOW the estimated cell count in every domain. That is a
standing bonus of 1.002 to 1.009 and it is bigger than this log ever
recorded — the 0.977 figure quoted below was **6bba's alone**; 44b6 is
0.913 and Zebrahub 0.919.

## 2026-08-26 -- ⭐⭐⭐⭐⭐ gap3: a GATED bridge beats the ungated one, six of seven arms agree in all three domains, and this line does NOT move the node count

`biohub-gap3`, 166 min, 8 arms × 3 domains, D187 ranker, `SHORT_MIN` 4.

**BASE reproduced all three anchors to the digit** — 0.9279 / 0.8553 /
0.8037, exactly short_sweep's `short_4` row. The `span+1` normalisation
bug that broke gap2's BASE is fixed and stayed fixed.

### The result

| arm | 6bba | 44b6 | zh | weakest domain |
|---|---|---|---|---|
| `nocand` (no gate at all) | +0.0009 | +0.0012 | +0.0034 | +0.0009 |
| `vel0.0` | +0.0004 | +0.0026 | +0.0033 | +0.0004 |
| `vel0.5` | +0.0005 | +0.0036 | +0.0029 | +0.0005 |
| `len4` | +0.0010 | +0.0035 | +0.0028 | +0.0010 |
| **`len8`** | **+0.0013** | **+0.0043** | **+0.0019** | **+0.0013** |
| `clear4` | +0.0009 | **−0.0002** | +0.0034 | **fails the rule** |
| `all` | +0.0009 | +0.0020 | +0.0025 | +0.0009 |

`div_J` is +0.0000 in 6bba and 44b6 for every arm and +0.0001…+0.0010 in
Zebrahub. Gap closing does not touch divisions in the competition
domains.

### The user was right: `nocand` is not an upper bound

I had called gap2's `nocand1` the ceiling of this line and concluded a
selector could only land below it. The user corrected that — it is the
**unconditional** case, not the best case, and its measured value is
what survives after good bridges and bad bridges cancel.

**`len8` is above `nocand` in two of three domains, and in 44b6 by
3.6×** (+0.0043 against +0.0012). That is the cancellation being
undone: requiring both fragments to be at least 8 nodes long removes
bridges that were subtracting.

**And 44b6 is monotone in the length gate**: nocand +0.0012 → len4
+0.0035 → len8 +0.0043. Zebrahub is monotone the OTHER way: +0.0034 →
+0.0028 → +0.0019. The two domains want different values and len8 is
the compromise that still clears every domain. `_frag_len` is capped at
12, so len12 is the last testable point on that curve.

### The user's objection to the velocity gate was half right

Recorded before the run: *cells in an embryo turn, so a velocity gate
may reject good bridges.* In **6bba that is exactly what happens** —
`vel0.0` is +0.0004, BELOW the completely ungated +0.0009 and the worst
of the surviving arms. In **44b6 the same gate more than doubles the
gain** (+0.0026 against +0.0012). The gate's sign is domain-dependent;
the objection holds where the user aimed it and not elsewhere.

### `clear4` is the one rejection, and it has a physical reading

Refusing a bridge when a PRIMARY node already sits within 4 µm is
identical to `nocand` in 6bba and Zebrahub (+0.0009, +0.0034) and
−0.0002 in 44b6. Reading, labelled as inference — the gate essentially
never fires in the sparse domains, and in 44b6 at **327 cells/frame**
the nearest neighbour is genuinely inside 4 µm often enough that the
gate blocks real repairs. The bridge counts were not printed, so this
is unconfirmed.

### Why this line is not contaminated by the node-count mechanism

`max_gaps = max(1, int(GAP1_CAP_FRAC * len(nodes)))` with
`GAP1_CAP_FRAC = 0.003`, and at span 1 each bridge inserts exactly one
node. **Node count can grow by at most 0.3%.** The two leaderboard
failures grew output by **+1.61%** and **+10.3%**. The adjustment term
`(1 − 0.1 × node_ratio)` is therefore worth ≤0.0003 here — two orders
of magnitude below the submission cases.

**This is the first candidate in days whose local gain is not
mechanically entangled with the thing that plausibly ate both previous
submissions.** Whatever `len8` measures, it measures in the edges.

> **DOWNGRADED the same day by `thr5`.** The paragraph above is
> arithmetically fine — len8 costs at most 0.0003 of the multiplier, and
> gap3's +0.0013 is already net of it. But its *significance* is gone:
> thr5 showed the node-count term is linear with a constant slope, so it
> never explained the two leaderboard results in the first place, and
> being clear of it buys nothing. The remaining case for len8 is only
> that it is the cleanest three-domain result on hand and it is small.
> If it reads 0.000 that now means the same thing every other 0.000 has
> meant.

### The cost, stated plainly

The weakest-domain gain is **+0.0013** against holdout noise of roughly
0.002–0.003, after six comparisons. It can read 0.000 on the
leaderboard without meaning anything. The three-domain rule is
satisfied, and the rule itself is still 0 for 2.

**Proposed submission**, one variable from ref 55772253 (LB 0.920):
`UNET_THRESH` back to **0.15** (0.12 cost three ticks and is the only
clear LB loss of the three), `SHORT_MIN` stays **4** (0.920 vs 0.921 is
one tick, and every gap3 arm was measured at 4), plus **`len8`**. That
matches the measured configuration exactly.

## 2026-08-26 -- ⭐⭐⭐⭐⭐ THE RULER WAS WRONG ALL ALONG. MATCH_UM 5.0 against the scorer's 7.0.

Every diagnostic in this project matched detections to GT at **5.0 µm**.
The official scorer matches at **7.0**
(`tracking_cellmot/metrics.py: evaluate(..., max_distance=7.0)`), so
every recall and miss figure in this log is **stricter than the thing
being optimised**.

| | at 5.0 (what was reported) | **at 7.0 (what counts)** |
|---|---|---|
| 6bba detector recall | 0.9561 | **0.9813** |
| 6bba node-level miss rate | 4.45% | **1.74%** |
| 6bba tracks intermittent | 28.8% | **14.2%** |
| 6bba dropout runs | 196 | **95** |
| length 2+, unrepaired | 94 | **34** |

**The detection line is much closer to done than this log has been
claiming.** 1.87% of annotated cells are missed, not 4.4%.

## 2026-08-26 -- ⭐⭐⭐ logbase: the UNet is worth +0.0566 over the best classical filter, and the tophat buys almost nothing on its own

`biohub-logbase`, 185 s CPU. Every arm scored at **matched detection
count** — top-N peaks with N set to what the UNet keeps on that frame —
so no threshold enters the comparison.

| arm | recall@7.0 | vs UNet |
|---|---|---|
| **UNet (shipped)** | **0.9813** | — |
| raw brightness | 0.8434 | −0.1379 |
| **tophat only** | **0.8462** | −0.1351 |
| LoG σ=1.3 | 0.8925 | −0.0888 |
| **LoG σ=1.6** | **0.9247** | **−0.0566** |
| LoG σ=2.0 | 0.9032 | −0.0781 |
| tophat + LoG σ=1.6 | 0.9163 | −0.0650 |

**The user's prior was right**: classical does not match the UNet, and
+0.0566 at the 0.92 level is a real gap, not a rounding difference.

**But the tophat hypothesis is dead.** raw 0.8434 → tophat 0.8462 is
**+0.0028**. I had inferred from the recorded `val_recall` (0.573
no-tophat against 0.659 tophat) that the preprocessing was load-bearing.
That 8.6-point gap is between **trained networks**, not between
preprocessings: the tophat changes the input distribution and the
network learns better on it, but **used directly as a detector the
tophat is worth nothing**. Same error class as the rest of this session —
*"a system that uses A is better"* is not *"A is useful"*.

The LoG scale prediction held: matched σ for a blob of radius r is
r/√3, cells are 7–9 µm across at a 1.625 µm isotropic heatmap voxel, so
r = 2.2–2.8 voxels and σ = 1.25–1.6. The measured optimum is σ=1.6.

## 2026-08-26 -- ⭐⭐⭐⭐ flicker3: three domains, and the only arm with a correct denominator says twice the flicker

`biohub-flicker3`, 385 s CPU. `flicker.py` ran on 6bba alone — **the
user caught it, the same criticism they have made repeatedly** — and at
the wrong radius. Both fixed.

At **MATCH_UM 7.0**:

| | 6bba | 44b6 | **zebrahub (DENSE)** |
|---|---|---|---|
| tracks | 379 | **49** | **6,298** |
| node-level miss rate | 1.74% | 1.39% | **3.51%** |
| always detected | 85.8% | 73.5% | **67.0%** |
| **never detected** | 0.0% | 0.0% | **0.6%** |
| intermittent | 14.2% | 26.5% | **32.4%** |
| dropout runs | 95 | 23 | **4,125** |
| length 2+, unrepaired | 34 | 5 | **1,465** |

Three things:

1. **Zebrahub's flicker rate is more than double 6bba's** — 32.4%
   against 14.2% — and Zebrahub is the only arm where every cell is
   annotated. **Annotation favours cells that are easy to see**, which
   this log has suspected all day and now has a number for.
2. **The 44b6 arm has 49 tracks.** Twelve datasets at 327 cells/frame,
   0.77% annotated, tracks of length ≥3 — forty-nine. Its 26.5% rests
   on **thirteen** intermittent tracks. **Every "three domains agree"
   verdict in this log has been carrying an arm with this little
   statistical weight, and I never said so.**
3. **"Never detected" is 0.6% in Zebrahub, not 0.** The class was empty
   in the sparse arms because those cells were never annotated. So
   "no cell is permanently invisible" becomes "0.6% are".

## 2026-08-26 -- ⭐⭐⭐ how `_gap_close` actually works, and two corrections to my own account

### The mechanism

The candidate pool is a band, cut from the detector's own peaks:

```
CAND_THR 0.05  ─────────────  UNET_THRESH 0.15  ─────────
   repair pool: 0.05 ≤ s < 0.15      primary nodes: s ≥ 0.15
```

and a bridge is built only if all of these hold:

| test | value | swept? |
|---|---|---|
| endpoints within the gate | `GAP1_GATE_UM` 9.0 | no |
| the **single nearest** candidate to the midpoint | — | never |
| within the snap radius | `GAP1_SNAP_UM` 3.2 | yes → noise |
| candidate score at least | `GAP1_MIN_CAND_SCORE` 0.10 | yes → noise |
| under the cap | `GAP1_CAP_FRAC` 0.003 | yes → noise |

`GAP1_MIN_CAND_SCORE` **cuts the usable pool in half** — `[0.10, 0.15)`
of a `[0.05, 0.15)` band. All four constants were inherited from the
public solution and none was derived.

**Why every sweep of them was noise, finally explained**: the mechanism
rarely fires at all — only ~20% of missed frames have any candidate in
the band — so adjusting sub-constants inside a gate that seldom opens
cannot move anything.

### Correction 1: the midpoint is NOT careless

I called the plain midpoint careless for ignoring velocity. **Wrong.**
`ge` is at t, `gs` at t+2, the missing node at t+1, so **the midpoint IS
the constant-velocity prediction**. Nothing to add for a one-frame gap.
The main linker is the cruder of the two — `pred = P + 0.5 * prev_vel`
uses a damped half-velocity, not a physical extrapolation.

### Correction 2: the node-count penalty does NOT apply to this line

I carried "the node-count penalty is the binding constraint" over from
the submissions, where output grew +1.61% and +10.3%. **Gap closing is
capped at 0.3% of nodes by construction.** Across the 12 6bba datasets:

| | |
|---|---|
| cap, `0.003 × nodes` | **476** bridges |
| eligible WITH the candidate requirement | ~238 — **the cap never binds** |
| eligible WITHOUT it | ~1,570 — **binds at 30% of demand** |
| nodes `nocand` adds | +238 = **+0.15%** |
| its node-count penalty | 0.00014 of J — **unreadable** |

**Two orders of magnitude from the submission case.** What decides this
line is whether the bridges are RIGHT: 476 bridges add 952 edges, 0.77%
of the total, a swing of about ±0.0077 — which is exactly the magnitude
gap2 measured (+0.0018 / −0.0016 / +0.0013), with signs disagreeing
across domains.

### The user's correction, which inverted my conclusion

I called gap2's `nocand1` the **upper bound** of the line and concluded
a better selector could only land below it. **Wrong.** `nocand1` is the
**unconditional** case — every gap gets a node whether or not there is
any reason — and its value is what remains after good and bad bridges
cancel. A **gated** version can be above *both*: it accepts bridges BASE
refuses and refuses bridges `nocand1` accepts.

> 理想情况下应该是直接插入一个点在一定的情况下可以触发,而不是直接插入一个
> 没有检测出来的点这种条件可以随时触发

gap2 measured "always" and "only with a candidate". **The entire space
between them was untested** — `biohub-gap3` tests it.

## 2026-08-26 -- the DESIGNS of gap3 and thr5 (both FINISHED; results in the two entries above)

**`biohub-gap3`** — FINISHED 2026-08-26, result in the entry above. Pure interpolation, GATED. 8 arms × 3 domains, ~2.7 h.
Each condition is its own arm because the user doubted the first:

| arm | candidate? | velocity cosine | fragment length | primary-node clearance |
|---|---|---|---|---|
| BASE | **yes** | — | — | — |
| nocand | no | — | — | — |
| vel0.0 / vel0.5 | no | ≥0 / ≥0.5 | — | — |
| len4 / len8 | no | — | ≥4 / ≥8 | — |
| clear4 | no | — | — | 4 µm |
| all | no | ≥0 | ≥4 | 4 µm |

The velocity gate **passes** a stationary endpoint rather than rejecting
it — a zero vector has no direction, and rejecting on it would kill
every still cell. BASE anchors on short_sweep's short_4 row (0.9279 /
0.8553 / 0.8037); gap2's BASE cannot be reused because its gate was
divided by `span+1`, doubling `GAP1_GATE_UM` at span 1. Fixed.

**`biohub-thr5`** — FINISHED 2026-08-26, result in the entry above. The gate TIGHTENED for the first time: 0.15, 0.18,
0.22, 0.26, `SHORT_MIN` 4, ~4.5 h. And it prints the metric decomposed:

```
adj_J = edge_J × (1 − 0.1 × node_ratio)
```

`summarise()` does not aggregate `total_node_ratio`, so `_with_ratio()`
adds it as a weighted mean using the same weights `adj_edge_jaccard`
uses — **purely additive**, unit-tested, summarise's own keys untouched.
Anchor: gate 0.15 must reproduce thr4's `g0.15_s4` exactly.

**What to read**: not how far adj_J falls, but **how far `edge_J` falls
against how much `node_ratio` improves**. If the edges barely suffer
while the node count moves a lot, tightening could win on the hidden
test even while losing locally — and that judgement is impossible from
adj_J alone.

## 2026-08-26 -- ⭐⭐ SUPERSEDED by thr5 (above): the node-count penalty is NOT the story

> **SUPERSEDED 2026-08-26 by `thr5`.** The arithmetic below is right
> and the interpretation is wrong in two places. (1) Being UNDER
> target does not make adding nodes free -- moving toward the target
> shrinks the bonus, so adding nodes has always cost us locally.
> (2) The term is linear with slope `-0.1 * dN/N_true`, independent
> of the current ratio, so local and hidden pay the SAME rate and
> the asymmetry this entry is built on does not exist. Both
> leaderboard results are therefore unexplained by it, and the local
> deltas quoted below were already net of the node cost. Also, the
> 0.977 figure is 6bba ALONE; 44b6 is 0.913 and Zebrahub 0.919.
> Kept in full because its measurements are sound.

### ORIGINAL HEADING: THE NODE-COUNT PENALTY IS THE STORY. Local is UNDER target, so it rewards what the leaderboard punishes.

### The two leaderboard results, and the number that explains both

| submission | change | local, three domains | output | **LB** |
|---|---|---|---|---|
| 55772253 | `SHORT_MIN` 6→4 | +0.0025 / +0.0054 / +0.0045 | +1.61% | **0.920** (0.000) |
| 55780773 | + gate 0.15→0.12 | **+0.0055 / +0.0045 / +0.0121** | +10.3% | **0.918** (−0.002) |

The second was submitted with a **pre-registered prediction of 0.919**,
derived from the output growth alone. It landed at 0.918 — direction
right, magnitude slightly understated.

**The three-domain rule is now 0 for 2**, exactly like the single-domain
holdout it was built to replace. And the second result is the strongest
local number this project has ever produced.

### Why: `N_pred / N_true` on the holdout is 0.977, so the penalty is a BONUS

`J_adj = max(0, J · (1 − 0.1 · (N_pred − N_true)/N_true))`, and on the
12-dataset holdout **11 of 12 datasets sit UNDER the target node count**:

| dataset | true/fr | N_true | N_pred | ratio | penalty factor |
|---|---|---|---|---|---|
| 7b5d3b2c | 58 | 5,780 | 4,990 | **0.863** | **1.0137** |
| 3a1849c2 | 70 | 7,003 | 6,223 | 0.889 | 1.0111 |
| 2312ac41 | 120 | 11,950 | 11,523 | 0.964 | 1.0036 |
| 283bf9f1 | 209 | 20,861 | 20,811 | 0.998 | 1.0002 |
| **3abfe10a** | 521 | 52,065 | 53,504 | **1.028** | 0.9972 |
| **TOTAL** | | 164,698 | 160,857 | **0.977** | **1.0023** |

**Locally, adding nodes helps TWICE** — more correct edges, and a move
toward the target that makes the adjustment term better. On the hidden
test, if `N_pred` is already at or over target, adding nodes helps once
and hurts once. That is a mechanism, and it needs no assumption about
density.

`corr(true cells/frame, N_pred/N_true) = +0.7498`, n=12, p≈0.005 — the
DENSE dataset is the one over target, not the sparse ones.

### ⚠ The user's correction: this is a WARNING, not a conclusion

Two limits, both theirs:

1. **The public split's composition is unknown and need not be
   adversarial to be skewed.** Taking the first or last N datasets is
   enough to give the public set a narrow density band while the hidden
   set spans wider. So LB 0.918 says "adding nodes loses on *that*
   subset" and nothing about the private set, which is what ranks.
2. **The penalty CONTAMINATES every judgement about edges and
   divisions**, because node count and edge quality are multiplied
   together inside `adj_edge_jaccard`. A change that improved the edges
   and a change that merely improved the node count are indistinguishable
   in the only column this project has been reading.

**And they are separable — the metric already reports them apart.**
`METRIC_COLUMNS` carries `edge_jaccard` (raw J), `total_node_ratio`
((N_pred − N_true)/N_true) and `adj_edge_jaccard` as three columns.
**Every sweep in this log read only the third.** All future sweeps must
report all three.

### The direction this opens: TIGHTEN the gate, which nobody has tried

Every threshold sweep so far went DOWN (0.15 → 0.12 → 0.10 → 0.08).
**0.18, 0.22, 0.26 have never been run.** And the asymmetry favours it:

| | local (ratio 0.977) | hidden (inferred: at/over target) |
|---|---|---|
| loosen | edges better **+** penalty better → **helped twice** | edges better **−** penalty worse → cancels |
| **tighten** | edges worse **−** penalty worse → **hurt twice** | edges worse **+** penalty better → cancels |

**Tightening is the only direction where looking bad locally is
compatible with scoring well on the leaderboard** — the mirror image of
the two failures already measured.

## 2026-08-26 -- ⭐⭐⭐ thr4: on D187 the gate SURVIVES the rule, 44b6's dissent was a D32 artefact, and the two knobs are ADDITIVE

`biohub-thr4`, 22,911 s = 6.4 h, 6 arms × 3 domains. **Both anchors held
exactly**: `g0.15_s6` reproduced post_sweep's BASE (0.9254 / 0.8499 /
0.7992) and `g0.15_s4` reproduced short_sweep's short_4 (0.9279 / 0.8553
/ 0.8037).

| config | 6bba | 44b6 | zebrahub |
|---|---|---|---|
| gate 0.12, `SHORT_MIN` 6 | +0.0033 | **+0.0006** | +0.0080 |
| gate 0.10, `SHORT_MIN` 6 | +0.0024 | +0.0027 | +0.0086 |
| `SHORT_MIN` 4, gate 0.15 | +0.0025 | +0.0054 | +0.0045 |
| **both** | **+0.0055** | **+0.0045** | **+0.0121** |

**thr3's 44b6 dissent (−0.0006) was a property of the D32 ranker, not
the data.** On D187 it is +0.0006. The ranker changed and the sign
flipped, which is why measuring on a model you do not ship is expensive.

**The two knobs are ADDITIVE**, unlike `gap2`'s, which were substitutes:

| | predicted if additive | measured |
|---|---|---|
| 6bba | 0.0033 + 0.0025 = 0.0058 | 0.0055 |
| zebrahub | 0.0080 + 0.0045 = 0.0125 | 0.0121 |

Two crossed experiments, two opposite answers. That is exactly why
independence cannot be assumed — recorded in `working-agreement.md`.

⚠ **This run has TWO independent domains, not three.** Every 44b6
dataset is inside D187's training pool, so that arm reports fit rather
than transfer. The log says so at the top.

## 2026-08-26 -- ⭐⭐⭐⭐ gap2: the CROSS was right, the gap line is dead, and the BASE anchor caught my own bug.

`biohub-gap2`, 204 min CPU, 10 arms × 3 domains. Built from the user's
"红色有很多是上一帧还检测到了,下一帧就检测不到了" and **crossed with
`SHORT_MIN` at their instruction**, against my original one-at-a-time
design.

### ⚠ First: the BASE anchor did NOT hold, and the cause was mine

| | recorded | gap2's BASE |
|---|---|---|
| adj_J | 0.9254 | **0.9253** |
| div_J | 0.1316 | **0.1282** |

The rewritten `_gap_close` normalised the gate by `step = span + 1`:

```python
D = D / float(step)      # span 1 -> step 2 -> gate 9 um becomes 18
```

**At span 1 — the shipped configuration — it divided by 2 and silently
doubled `GAP1_GATE_UM`.** BASE was not BASE. Fixed to divide by `span`,
which is 1 at span 1 and therefore byte-identical to the shipped code.

The docstring had claimed *"with GAP_SPAN=1 and GAP_NEED_CAND=True this
must be byte-identical … that is the BASE arm's job to prove"*, and the
anchor did its job. **What the anchor could not do is stop the run**, so
read the table below with the flaw in mind:

- **the `SHORT_MIN` deltas are unaffected** — the same loosened gate sat
  under both halves and cancelled in the difference, which is why
  `sm4_base` reproduced `short_sweep`'s `+0.0025 / +0.0054 / +0.0045` to
  four decimals;
- **the GAP arms are contaminated**, because they compare different
  spans and therefore different gates (×2, ×3, ×4 instead of ×1, ×2, ×3).

### THE CROSS WAS RIGHT: every gap arm's gain halves when `SHORT_MIN` goes 6 → 4

Each half is a delta against **its own** baseline.

| gap arm | 6bba sm6 | 44b6 sm6 | zh sm6 | | 6bba sm4 | 44b6 sm4 | zh sm4 |
|---|---|---|---|---|---|---|---|
| nocand1 | +0.0018 | −0.0016 | +0.0013 | → | **+0.0010** | **−0.0008** | **+0.0005** |
| span2 | +0.0002 | −0.0001 | +0.0003 | → | +0.0003 | −0.0001 | +0.0003 |
| span2i | +0.0019 | −0.0016 | +0.0013 | → | +0.0011 | −0.0008 | +0.0005 |
| span3i | +0.0019 | −0.0016 | +0.0013 | → | +0.0011 | −0.0008 | +0.0005 |

**Five arms, three domains, fifteen cells — every one shrinks by roughly
half.** The gap repair and the short filter are substitutes, exactly as
the flicker mechanism predicts: a wider bridge leaves fewer fragments,
so there is less for `SHORT_MIN` to delete.

**A one-at-a-time sweep would have overstated the gap arms by about 2x**,
and the user called it before the run. Record this as the standing
reason to cross knobs whose mechanisms touch: *the assumption of
independence is itself a claim, and here it was false.*

### The line is dead anyway

**Every gap arm fails the three-domain rule**: 44b6 negative in all of
them, 6bba and Zebrahub positive in all of them. And three structural
findings make it dead rather than merely unproven:

1. **`span2i` and `span3i` are identical in every column.** Bridging
   three frames finds nothing that bridging two does not.
2. **`span2` — the honest widening, candidate still required — is
   +0.0002 to +0.0003, i.e. nothing.** Exactly what `flicker.py`
   predicted: 79.8% of missed frames have no candidate in the repair
   band, so requiring one at *both* bridged frames fires on about
   0.2 × 0.2 = 4% of dropouts.
3. **`nocand1` ≈ `span2i` ≈ `span3i`.** All of the movement comes from
   dropping the candidate requirement at span 1. **The span never
   mattered at all.**

And pure interpolation costs divisions: div_J −0.0032 in `6bba` and
−0.0025 in Zebrahub. Inserting nodes the detector never saw manufactures
exactly the kind of node the division patch then mines for forks.

**Not re-running.** The fixed gate would loosen every arm equally, and
44b6's dissent is present at both `SHORT_MIN` values and in all five
arms — a looser gate does not flip a sign that consistent. The only
thing that survived is `sm4_base`, which is already submitted.

### Colab, for the record

The acceptance test did not finish again, and this time the session is
genuinely gone: **zero assignments on the server and no keep-alive
process**, unlike the three earlier cases where the VM was provably
alive. No theory offered — the evidence does not support one. One
fact worth keeping: **`hold` does not actually hold.** Its background
task reported "completed" while the keep-alive lingered for a further
three hours and then died with it.

## 2026-08-25 -- ⭐⭐⭐⭐ thr3: the detection gate FAILS the three-domain rule — but the dissenting domain is 44b6, by 0.0008, against Zebrahub's +0.0080.

`biohub-thr3`, 179 min CPU. Correctly loaded **D32** (`edge_ranker_d32.pkl`,
pool 32) and excluded the 16 `44b6` datasets inside that pool, because
this sweep changes DETECTIONS and the ranker-leak rule in
`working-agreement.md` requires it. **Its baseline is therefore
adj_J 0.9222, which lands 0.0002 from `biohub-div-gate`'s recorded D32
row — the anchor holds.**

⚠ **These absolute numbers are NOT comparable to `post_sweep`,
`short_sweep` or the submission**, all of which ran on D187. Only the
deltas within this run mean anything.

### Absolute

| gate | 6bba | 44b6 | zebrahub |
|---|---|---|---|
| **0.15** (shipped) | 0.9222 | **0.8603** | 0.7988 |
| 0.12 | **0.9243** | 0.8598 | 0.8068 |
| 0.10 | 0.9236 | 0.8596 | 0.8074 |
| 0.08 | 0.9236 | 0.8594 | **0.8075** |

### Delta vs the shipped 0.15

| gate | 6bba | 44b6 | zebrahub |
|---|---|---|---|
| 0.12 | +0.0021 | **−0.0006** | **+0.0080** |
| 0.10 | +0.0014 | −0.0008 | **+0.0085** |
| 0.08 | +0.0014 | −0.0009 | **+0.0087** |

**The rule says discard, and the rule was pre-registered, so discard is
the verdict.** But the shape must be recorded accurately, because the
script's own pre-written verdict string says *"6bba artefact"* and
**that is wrong about which domain dissents.**

- `6bba` gains and `zebrahub` gains — and Zebrahub gains **four times
  more**, +0.0080 against +0.0021. Zebrahub is the arm with dense
  ground truth, so its adj_J is over all cells, not over annotated ones.
- **`44b6` is the lone dissenter, at −0.0006 to −0.0009** — at or under
  the resolution floor.

So this is not one domain carrying a result the others refuse. It is two
domains agreeing, one of them strongly, against a third that moves by
less than the noise. That is a materially different situation from
`poolsize` (6bba alone, +0.0073, LB −0.001), and the record should not
file it as the same thing.

**Not acting on it.** Relaxing a pre-registered rule the first time a
result sits just outside it is how the rule stops meaning anything. The
honest state is: contested, with the dissent smaller than the floor, and
the decision belongs to the user.

### The column that explains the mechanism, and a new interaction

| gate | nodes | repair pool |
|---|---|---|
| 0.15 | 548,794 | **111,937** |
| 0.12 | 596,529 | 64,202 |
| 0.10 | 621,292 | 39,439 |
| 0.08 | 639,878 | **20,853** |

**Exactly conserved**: 0.15 → 0.08 adds 91,084 nodes and removes 91,084
from the repair pool. Lowering the gate does not create detections, it
**moves them from `_gap_close`'s repair pool into primary nodes** — the
script was written to print both precisely so a fall could not be
misattributed.

**And that collides with the flicker finding.** `_gap_close` repairs a
dropout by snapping to a candidate in `[CAND_THR 0.05, gate)`. At gate
0.08 that band holds 19% of what it held at 0.15. So **lowering the gate
progressively destroys the repair mechanism**, and `biohub-gap2`'s arms
— which depend on that pool — would behave differently under a lowered
gate. The two lines are not independent and a future sweep should cross
them, the same way `gap2` now crosses span with `SHORT_MIN`.

## 2026-08-25 -- ⭐⭐⭐⭐⭐ THE MISSES FLICKER. No annotated cell is permanently invisible, and 48% of dropouts are longer than the repair can reach.

**The user's observation, from watching the rendered movie**: in
`6bba_3abfe10a` many red circles are cells detected in one frame,
missing in the next, some flickering back and forth. `flicker.py`
measured it over 379 annotated GT tracks in the 12 `6bba` holdout
datasets, following the GT's own edges so a track is a real cell's
history.

| | tracks | |
|---|---|---|
| detected in EVERY frame | 270 | 71.2% |
| **detected in NO frame** | **0** | **0.0%** |
| INTERMITTENT | 109 | 28.8% |

**The "permanently invisible" class is empty.** So the ceiling
argument — dim is dim, nothing to do — does not apply to these losses.
Every annotated cell is found in some frame.

### The dropouts, by length

| length | runs | | |
|---|---|---|---|
| **1 frame** | 102 | **52.0%** | the only length `_gap_close` bridges |
| **2 frames** | 44 | 22.4% | unrepaired |
| 3 frames | 23 | 11.7% | unrepaired |
| 4+ frames | 27 | 13.8% | unrepaired |

**48% of dropouts are beyond the reach of the existing repair.**

### This is the same phenomenon as `SHORT_MIN`, seen from the other end

A two-frame dropout splits a track in two. If either piece falls under
6 nodes, the shipped `SHORT_MIN = 6` deleted it. That is why lowering it
to 4 gained adj_J in all three domains today — the filter was deleting
the fragments of real cells that flicker had shattered.

### What kills the obvious fix, and why the follow-up has TWO variables

`_gap_close` requires the inserted node to land on a real candidate from
the repair band `[CAND_THR 0.05, UNET_THRESH 0.15)`. But at the 465
missed frames:

| | |
|---|---|
| **no candidate within 5 µm at all** | **79.8%** |
| there is one | 20.2%, median score 0.1226, p75 0.1352 |

⚠ **CORRECTED 2026-09-08. The sentence that stood here -- "four times in
five the heatmap response does not dip below threshold, it vanishes" --
was an INTERPRETATION laid on a measurement that cannot support it.**
What the 79.8% counts is "no CANDIDATE within 5 um", and a candidate is
a `peak_local_max` output; it says nothing about the heatmap VALUE
there. `gtpeak` (2026-09-08) read the value directly on the same
domains: at F4's missed cells the 3.25 um window max is **p50 0.15** and
**98.3% / 99.0% are >= 0.05**, while the nearest raw peak is p50 5.84 /
6.91 um away -- on a neighbouring cell. **The two measurements AGREE;
only the sentence was wrong. The response does not vanish, it stops
being a local maximum.** So the existing repair can fire on at most a
fifth of dropouts, and a two-frame bridge needing a candidate at *both* missing
frames would fire on roughly 0.2 × 0.2 = 4%.

The remaining fifth sits just under 0.15, which is where
`biohub-thr-sweep`'s winning gate of 0.12 lands — those two results are
looking at the same cells.

### `biohub-gap2`, queued

`_gap_close` rewritten with two parameters, **crossed with `SHORT_MIN`
at the user's instruction** because the two are the same phenomenon and
cannot be assumed independent — if a wider bridge repairs the dropouts,
there are fewer fragments and `SHORT_MIN`'s optimum moves.

- `GAP_SPAN` ∈ {1, 2, 3} — how many missing frames to bridge. The gate
  is divided by the span so a longer bridge gets a proportional travel
  budget; without that it would test the gate, not the span.
- `GAP_NEED_CAND` ∈ {True, False} — whether an inserted node may be
  purely interpolated, with no detection evidence. The case for it: the
  official metric matches nodes by centroid distance at `max_distance`
  7.0 µm, so an interpolated node near a real GT cell counts as a match
  and brings two correct edges. The case against: interpolating across a
  genuinely ended track manufactures false nodes, and the metric carries
  a node-count penalty.

10 arms × 3 domains, with **two** must-reproduce anchors: `BASE`
(span 1, candidate required, `SHORT_MIN` 6) must give `biohub-poolsize`'s
D187 row, and `sm4_base` must give `biohub-short-sweep`'s short_4
deltas. `post_sweep` swept this function's four constants and all four
were noise — but the span is not a constant, it is the shape of the
loop, and nobody had ever moved it.

## 2026-08-25 -- ⭐⭐ depth3: the depth hypothesis does NOT survive. Signs disagree and nothing is significant.

Pre-registered before the run, and this is the branch that fired: *"recall
flat means the front/back difference on screen is not depth, and this
line closes with a negative."*

| domain | corr(recall, z) | p |
|---|---|---|
| 6bba | −0.6837 | 0.062 |
| 44b6 | −0.6800 | 0.064 |
| **zebrahub** | **+0.5011** | 0.206 |

Two domains fall with depth, **Zebrahub rises**, and at 8 bins the 5%
threshold is |r| ≥ 0.707 so **none of the three is significant**.
Zebrahub is the arm whose GT is dense, i.e. the only one whose `recall`
is recall over all cells rather than over annotated cells.

### The brightness link is broken in the opposite direction

| | r | p | |
|---|---|---|---|
| 6bba: GT brightness ~ z | **+0.8048** | 0.016 | **significant** |
| 6bba: recall ~ GT brightness | **−0.7788** | 0.023 | **significant** |
| 44b6: GT brightness ~ z | −0.0059 | 0.989 | nothing |

In `6bba`, **GT-point brightness RISES with depth** — the opposite of
attenuation — and brighter bins have *worse* recall.

### ⚠ Do not read that against the recorded 0.52/0.74

The recorded finding (recall 0.52 for the dimmest GT-brightness quartile
against 0.74 for the brightest) is **per cell**. The −0.7788 here is a
correlation over **8 bin means**. Those are different populations, and a
bin-level correlation can invert the individual-level one. **They are
not in contradiction; they are not the same quantity.**

That is the day's recurring error appearing inside my own analysis
design this time — the eleventh instance. A per-cell version of this
measurement would be the honest comparison and has not been run.

### A gap in the design, stated rather than glossed

Zebrahub's brightness column is `nan` throughout: the sampling only runs
where a raw volume is mounted, which is the two competition arms. **The
load-bearing arm has no brightness data**, so the depth → dim link is
tested only where the GT is sparse.

### What it buys

The depth line closes with a negative, and that protects the flicker
result above: the intermittent misses are **not** explained by cells
being too deep to see, which is the reading that would have made them a
ceiling rather than something to repair.

## 2026-08-25 -- ⭐⭐⭐⭐⭐ short_sweep: a CURVE, not a spike. `SHORT_MIN` 6 → 4 SUBMITTED as the first test of the three-domain rule.

`biohub-short-sweep`, 168 min CPU, 7 values × 3 domains. **Both anchors
held**, and the second one exactly: `short_4`'s deltas reproduced
`biohub-post-sweep` to four decimals in all three domains
(+0.0025 / +0.0054 / +0.0045). **The pipeline is deterministic and
post_sweep's table was real, not noise.**

### The curve

| `SHORT_MIN` | 6bba | 44b6 | zebrahub |
|---|---|---|---|
| 1 | 0.9258 | 0.8552 | **0.8060** |
| 2 | 0.9258 | 0.8552 | **0.8060** |
| 3 | 0.9258 | 0.8552 | **0.8060** |
| **4** | **0.9279** | **0.8553** | 0.8037 |
| 5 | 0.9277 | 0.8533 | 0.8015 |
| 6 (shipped) | 0.9254 | 0.8499 | 0.7992 |
| 7 | 0.9263 | 0.8515 | 0.7968 |

**1, 2 and 3 are byte-identical in every domain**, so the solved graph
contains no components of length 1 or 2. The filter only ever acts on
lengths 3, 4 and 5.

### Restated as "which track lengths should be deleted"

That is the form the three-domain rule can actually judge:

| delete tracks of length | 6bba | 44b6 | zebrahub | verdict |
|---|---|---|---|---|
| 5 | −0.0025 | −0.0034 | −0.0023 | same sign — **harmful everywhere** |
| 4 | −0.0002 | −0.0020 | −0.0022 | same sign — **harmful everywhere** |
| 3 | +0.0021 | +0.0001 | −0.0023 | **signs disagree — contested** |

The shipped `SHORT_MIN = 6` deletes all three lengths, and two of those
three deletions lose in every domain. `SHORT_MIN = 4` stops deleting 4s
and 5s and keeps deleting 3s; whether to stop deleting 3s as well is
exactly what the rule cannot decide, so the value stops at 4.

4 beats 5 in all three (+0.0002 / +0.0020 / +0.0022) and beats the
shipped 6 in all three (+0.0025 / +0.0054 / +0.0045).

### SUBMITTED, and what is actually being tested

`biohub-submit-ranker` v5. **Exactly one non-comment line differs from
the deployed version**, verified by diffing against the source pulled
back from Kaggle (which was byte-identical to the local copy):
`SHORT_MIN = 6` → `4`.

**The comparison is against ref 55757064, LB 0.920 — NOT 0.921.** The
deployed kernel prefers `edge_ranker_d187.pkl` and refuses the older
one, so it is the D187 pipeline, and D187 is also what `post_sweep` and
`short_sweep` measured on. Comparing the result to 0.921 would move two
things at once, which is the error this log already records twice.

**What is being measured is the rule, not the constant.** The weakest
domain's +0.0025 is the same scale as the two local gains that failed to
appear on the leaderboard (windowing +0.0019 → −0.001; ranker pool
+0.0073 → −0.001). Both of those were measured on `6bba` alone. This is
the first candidate the three-domain rule has ever produced, so:

- **it appears on the LB** → three-domain agreement buys something a
  single-domain holdout does not, and the rule is worth its cost;
- **it does not** → the rule is not sufficient either, and no local
  measurement at this scale should drive a submission again.

Either answer is worth more than the constant.

### A correction the user made, recorded because I repeated the error

I priced this submission at "about 11.8 h of GPU quota" three times.
**A submission does not cost GPU quota.** The 11.8 h is the hidden-test
runtime against the competition's 12 h cap — a timeout risk, not a
budget charge. Treating it as a charge inflated the apparent price of
the single most informative experiment available. Recorded in
`kaggle-toolchain.md` section 9.

## 2026-08-25 -- ⭐⭐⭐ nms4, the density test in all three domains: 2 of 3, none significant. The line closes, and the interesting number is not about NMS.

`biohub-nms4`, 13 min CPU. True cells per frame in every domain —
counted per frame on Zebrahub, and from the organisers'
`estimated_number_of_nodes / T` on the competition arms. All three
density ranges are about 10x wide, so all three are measurable and the
`NO RANGE` guard did not fire.

### The table

| domain | sets | density range | **corr(lost to NMS, true density)** | corr(reachable BEFORE NMS, true density) |
|---|---|---|---|---|
| 6bba | 12 | 50–521 (10.3x) | **−0.0633** | **−0.5580** |
| 44b6 | 12 | 74–786 (10.6x) | **+0.4856** | +0.0311 |
| zebrahub | 8 | 51–562 (11.1x) | **+0.4067** | −0.0266 |

### Read the sample size first

**Not one of the six correlations reaches conventional significance.**
At n=12 the two-tailed 5% threshold is |r| ≥ 0.576; at n=8 it is 0.707.

| | r | p |
|---|---|---|
| 6bba before~density | −0.5580 | 0.059 |
| 44b6 lost~density | +0.4856 | 0.110 |
| zebrahub lost~density | +0.4067 | 0.317 |
| the other three | ≤ 0.064 | ≥ 0.84 |

Twelve datasets is twelve points. A correlation of +0.49 on twelve
points is a shape, not a fact.

### The verdict the pre-registered rule gives

**The density hypothesis does NOT survive.** Two domains show it at
about +0.45, `6bba` shows nothing at all (−0.06, p=0.85). Signs
disagree, so by the rule written down before the run it is a property of
some domains and not of the method.

This was the user's line — they pushed three times to have it tested
where cells are actually crowded, against my closing it on `6bba` alone.
Testing it properly was right; the answer is that it does not replicate.

### The one number worth keeping is not about NMS

`6bba`'s **−0.5580 (p=0.059)** says that in `6bba`, datasets with more
true cells per frame have systematically worse reachability **before any
suppression happens**. The detector is failing upstream. The other two
domains show nothing there (+0.031, −0.027).

That is a candidate explanation for why `6bba` shows no NMS density
effect — the crowding damage is already done before suppression runs, so
suppression then takes an even share. **It is a hypothesis, not a
result**, and it is one point short of significance. Carried as
unresolved, per the standing rule about not explaining away a
disagreeing measurement.

If it holds it matters more than NMS ever did, because **the 12-dataset
holdout this project measures everything on is entirely `6bba`** — the
one domain with a density-dependent detection problem.

Caveat on it: `6bba` reachability is against annotated cells (9.71%).
The correlation is across 12 datasets within one domain, so the
annotation convention is roughly constant — but "roughly" is doing work
and has not been checked.

### The radius grid, refined, and why it closes the line

nms3's grid put everything between two adjacent points. Refined to
(2.5, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5), the curve is **smooth and
monotone in all three domains** — no plateau, no kink, no natural
optimum. Ceiling gain from the shipped 4.0 down to 3.0:

| domain | 4.0 | 3.0 | gain | extra points kept |
|---|---|---|---|---|
| 6bba | 0.9561 | 0.9564 | +0.0003 | +645 |
| 44b6 | 0.9527 | 0.9535 | +0.0008 | +1720 |
| zebrahub | 0.8947 | 0.8972 | +0.0025 | +1246 |

**Reachability is monotone in the radius and therefore structurally
incapable of choosing one** — it always votes smaller, all the way to
keeping every duplicate peak. Only the official score, which pays for
the extra points through precision and the node-count penalty, can pick.

And the ceiling gain is +0.0003 and +0.0008 in two of three domains,
at or under the resolution floor, **before** paying for the extra
points. A several-hour full-pipeline radius sweep would be spending
that time to chase a number the ceiling already says is too small.
**The NMS radius line closes here**, with the measurement the user asked
for rather than the one-domain version I tried to close it on.

## 2026-08-25 -- nms3 PARTIAL: two sparse arms landed, the dense arm crashed, and the two sparse arms CANNOT be compared to each other

`biohub-nms3` v1 ran `6bba`, ran `44b6`, then died entering Zebrahub.
**Both sparse arms landed and the 6bba anchor held** (0.0092 here against
`biohub-nms-loss`'s recorded 0.0091).

### What suppression costs, per domain

| domain | cells/frame | annotated | before | after | **cost** |
|---|---|---|---|---|---|
| 6bba | 97 | 9.71% | 0.9652 | 0.9561 | **0.0092** |
| 44b6 | 327 | **0.77%** | 0.9581 | 0.9527 | **0.0054** |

### ⚠ The cross-domain reading is NOT available, and the first version of this entry made it anyway

The first version of this entry said *"this is the opposite of the
density hypothesis — the denser domain loses less."* **Withdrawn.** The
user caught it within minutes, on the grounds that recall and annotation
are not the same thing.

`reachable()` counts **GT nodes with a detection point within 5.0 µm**.
Its denominator is not the cells that exist, it is **the cells somebody
chose to label**. So 0.9581 does not read "95.81% of 44b6's cells", it
reads *"95.81% of the two-or-three cells per frame that were annotated
in 44b6"*. That is a recall against a hand-picked sample, and it equals
recall only if annotation is a random draw from the true cells — which
nobody does, because annotators label what they can see.

The two rows therefore have **differently-selected denominators**:
`6bba` keeps 9.71% of its cells, `44b6` keeps 0.77%, a 12x stricter
sieve. If annotation favours bright isolated cells — and Zebrahub's
recall-by-brightness-quartile of **0.52 dimmest vs 0.74 brightest** says
brightness dominates detectability — then `44b6`'s sample is the more
purified of the two, and *isolated* is exactly the property that makes a
cell safe from suppression. **0.0054 < 0.0092 is consistent with 44b6
suppressing more gently AND with 44b6 being annotated more selectively.
These two rows cannot separate them.**

What survives: the *within-domain* differences are clean, because before
and after are measured on the same annotated set. "Suppression costs
0.92% of 6bba's annotated cells" stands. "0.0092 vs 0.0054 across
domains" does not.

Same error class as the six others today — **numbers from different
populations compared to each other** — and this one appeared in the same
entry that warned both arms were sparse.

### Why this makes the Zebrahub arm the whole experiment

Zebrahub labels **every** cell. Only there is the denominator the true
cell population, only there is the word "recall" earned, and only there
can the number be compared to anything else. It is the arm that did not
run. v2 is queued.

The radius sweep has the same limit and is recorded for later use, not
read now: on `44b6`, 4.0 → 2.0 moves reachable-annotated from 0.9527 to
0.9535, flat below 3.0.

### The crash

```
T = max(int(k[1:]) for k in z.files if k.startswith("p")) + 1
ValueError: invalid literal for int() with base 10: 'arent_track_id'
```

`prenms2` writes the peaks as `p0…pN` / `q0…qN` **and** puts the ground
truth in the same file, including `parent_track_id` — which also starts
with `"p"`. `startswith` alone matched it. The prefixes `p`/`q` were
chosen deliberately to avoid colliding with the detection cache's
`c`/`s`, and then collided with a GT array instead.

Fixed in v2 by a shared `_frame_keys(files, prefix)` that requires the
remainder to be digits, applied to **all three** frame-key scans rather
than only the one that broke, and unit-tested against the real key set.

Same family as `arm.lower()` on a float and `prev_of` as a dict: the
name existed, the arity matched, nothing static could see it.

## 2026-08-25 -- ⭐⭐⭐⭐ THE POST-LINK SWEEP: one of fourteen survived. `SHORT_MIN` 6 → 4, positive in all three domains.

`biohub-post-sweep`, 245 min, CPU. Seven post-link constants inherited
from the public solution, fourteen one-at-a-time changes, scored in
`6bba` (12 holdout), `44b6` (12 outside D32's pool) and Zebrahub (8
crops). Both anchors held: BASE `6bba` reproduced `biohub-poolsize`'s
D187 exactly at score 0.9386 / adj_J 0.9254 / div_J 0.1316.

### BASE, the shipped configuration

| domain | score | adj_J | div_J |
|---|---|---|---|
| 6bba | 0.9386 | 0.9254 | 0.1316 |
| 44b6 | 0.8499 | 0.8499 | **0.0000** |
| zebrahub | 0.8014 | 0.7992 | 0.0221 |

### The result

**`short_4` is the only change that survived the pre-registered
three-domain rule with a delta big enough to see.**

| config | 6bba | 44b6 | zebrahub | verdict |
|---|---|---|---|---|
| **short_4** (`SHORT_MIN` 6→4) | **+0.0025** | **+0.0054** | **+0.0045** | **same sign, positive** |
| short_8 (`SHORT_MIN` 6→8) | +0.0007 | +0.0044 | **−0.0056** | signs disagree |
| lf_w_0.5 | −0.0015 | −0.0027 | −0.0032 | same sign, negative |
| lf_win_1 | −0.0002 | −0.0007 | −0.0020 | same sign, negative |
| gap_minsc_0.15 | +0.0002 | −0.0003 | −0.0008 | signs disagree |
| lf_w_1.0 | +0.0014 | −0.0019 | +0.0007 | signs disagree |
| lf_win_3 | +0.0010 | −0.0027 | +0.0004 | signs disagree |
| the four `gap_*` gate/snap/cap rows | ±0.0000 | ±0.0006 | ±0.0001 | noise, see below |

### What `SHORT_MIN` is, in plain terms

`_short_filter` deletes a whole predicted track when it has fewer than
`SHORT_MIN` nodes, unless that track contains a division. The shipped
value is 6. Lowering it to 4 means **keeping the 4- and 5-node tracks we
were throwing away**, and doing that gains adj_J in every domain.

The filter is a bet that short predicted tracks are mostly wrong. At
length 4–5 the bet loses: those fragments are mostly right, and deleting
them removes correct edges from the numerator. Going the other way
(`short_8`, delete even more) helps `44b6` and hurts Zebrahub — that is
the signature of a domain-specific effect, and the rule discards it.

### A correction to the script's own verdict column, made here so it is not repeated

`gap_gate_7` printed **"SAME SIGN, POSITIVE"** with three deltas of
`+0.0000`. That is not a win. All three were positive but under `1e-4`,
and the test as written had no lower band, so it promoted a number a
thousand times smaller than the holdout's own noise. **Agreeing in sign
is not the same as being large enough to see.** `short_sweep.py` adds an
explicit `TINY = 5e-4` floor; anything whose weakest domain is under it
prints `TOO SMALL TO MATTER` instead. Read `short_4` as the only real
survivor of the fourteen.

### The one column that carries no information

`44b6`'s BASE div_J is **0.0000** — the 12 `44b6` datasets in this arm
produce no correct divisions at all, so every `44b6` entry in the div_J
delta table is `+0.0000` by construction. It is not evidence that these
changes leave divisions alone; it is evidence that there is nothing
there to move.

### What happens next

1. **`biohub-short-sweep` is queued** (CPU, 7 configs × 3 domains):
   `SHORT_MIN` ∈ {1, 2, 3, 4, 5, 6, 7}, where 1 means the filter is
   switched off entirely. **Is `short_4` a curve or a spike?** A real
   effect varies smoothly with the constant; a lone value that wins
   while both neighbours lose is what noise looks like. It carries two
   must-reproduce anchors, the second being `short_4`'s three deltas to
   four decimals — same code, same inputs, so any drift means the
   pipeline is not deterministic and this whole table was noise.
2. **Then, and only then, the leaderboard.** `SHORT_MIN` is line 127 of
   `submit_ranker.py`, the script behind ref 55707206 (LB 0.921), so
   `6 → 4` is a genuine single-variable submission. That is the
   experiment that tells us whether **the three-domain rule is worth
   anything** — the local holdout is 0 for 2 against the leaderboard,
   but both of those failures were `6bba`-only. This is the first
   candidate the rule has ever produced. Costs ~11.8 h GPU and one
   submission slot; **needs the user's go-ahead** because it is
   outward-facing.

## ⭐ PRIORITY as of 2026-08-25 (user's call): DIVISIONS FIRST, and the reason is structural

**This overrides the 2026-08-24 ordering below.** It is not a change of
mind about the loss budget -- that table is still correct. It is that
morphology arriving changed what the division machinery IS.

The user's argument, in their words: divisions were being caught by the
ORPHAN FORK, but morphology features are EDGE-correlated, so the simple
orphan fork can no longer serve as the division criterion.

Unpacked, and it is sharper than the "orphan starvation" worry recorded
in `morph_edge.py`:

**The orphan patch was never a division detector. It is a LINKER-FAILURE
detector.** Its raw material is the ILP refusing the second daughter and
leaving her unlinked; `_add_safe_divisions` then requires a parent with
EXACTLY ONE child (`div_train.py:418`, `if len(ch) != 1: continue`).
What it catches is therefore a function of how the linker fails. Change
the linker -- which is exactly what putting morphology in the ranker
does -- and what it catches changes, in a direction the division side
cannot predict.

Two regimes, opposite results, and the patch is silent in both:

| ranker becomes | what the ILP does | orphan patch | div_J |
|---|---|---|---|
| more FASTIDIOUS (daughters "look different", edges pushed down) | refuses both daughters, `len(ch)==0` | never fires | falls |
| more CORRECT | takes both daughters, `len(ch)==2` | never fires | rises |

### The third regime, and it is the one already running (user, 2026-08-25)

The two rows above are the cases where the patch goes SILENT. The user
named the dangerous one: the patch FIRES, on things that are not mother
and daughter at all. It is not hypothetical -- it is the status quo,
measured on the 12-dataset 6bba holdout by `biohub-poolsize` and carried
in ref 55757064's description:

| ranker | div TP | div FP | div FN |
|---|---|---|---|
| D32 | 3 | **21** | 9 |
| D187 (submitted) | 5 | **26** | 7 |

**26 false divisions bought 5 real ones.** Two things keep that from
being fatal, and both are already on record:

- a fork lying entirely among unannotated cells enters no FP set
  (user's observation, verified): 913 extra forks cost -0.0027 adj_J,
  about 30 visible edge FPs, so ~97% of extra forks are INVISIBLE.
- but div_J is not blind to them, and div_J is the 0.1 term.

And the exchange rate cannot be improved by forking more: across all 12
`gate_sweep` points (forks 0 to 1847) **division FN never drops below
6** while division FP explodes. The divisions we miss are not in the
candidate set. So the only move with value left is KILLING FALSE FORKS,
not proposing more.

That is what mass conservation is for: a real division sums to one
mother, a mother-plus-passing-neighbour sums to two.

**Bound on `div_mass`:** its fake forks are built from GT (mother + true
successor + any GT cell within GATE_UM), which is an EASIER population
than the patch's actual false positives -- those already survived sister
distance, mutual-nearest, both-children-continue and divergence. A good
AUC there says the route is worth walking; it does not say the 26 fall.
Testing that needs the patch's real survivors.


`poolsize` already ran the second regime: a better ranker moved div_J
0.0909 -> 0.1316. That is on record and it is why "a more confident
linker drains the orphan pool" was too simple -- CORRECT and CONFIDENT
are different axes.

**The consequence: divisions need a criterion that does not depend on
the linker failing first.** Mass conservation is a candidate for exactly
that -- "these two cells sum to one mother" is answerable without the
ILP having made a mistake. `div_mass.py` -> `biohub-div-mass` is that
measurement. If it lands, divisions decouple from linking and the
2026-08-24 ordering can resume.


SUPERSEDED 2026-08-25 by the block above -- ## ⭐ PRIORITY as of 2026-08-24 (user's call): detection and linking, not divisions

Divisions are parked (item 1). The loss budget says why -- decomposed
from the det-threshold table below:

| where the score leaks | size |
|---|---|
| detection recall: a GT edge's endpoint was never detected | ~1.75% |
| linking: a reachable edge linked wrong | ~1.6% |
| false edges: predicted where GT has none | ~3.3% |

**The last two together are about 3x the first**, and adj_J carries 10x
div_J's weight. Concrete leads, cheapest first:

- **A. Which of the xl detector's 12 forward passes earn their place.**
  `biohub-det-ablate` RUNNING. Three branches x four flip views, never
  ablated. Branches 1 and 2 get PIXEL-IDENTICAL input (both top-hat), so
  their errors should correlate; branch 0 is the only different view of
  the data and has the worst solo `val_recall` -- which is not what earns
  a place in an ensemble. One inference pass covers 11 configurations.
- **B. The ranker never sees `edge_prob`.** The transplant dropped
  yusuketogashi's first-listed feature because `detcache_*.npz` carries
  only coords and detection scores (full entry below). NOT free: the
  cached probs index the OFFICIAL detector's nodes, the pipeline runs xl
  nodes, so it needs the official edge head re-run over the xl node set.
- **C. Multi-frame trajectory in LINKING.** The official edge head is
  `window_size = 2` and "structurally cannot use velocity"; our ranker
  has one speed scalar greedily estimated from a single transition. The
  same shape of gap the division work just proved real.

  Two public references, both already on the table and neither pursued:

  - **Togashi's item 4, acceleration lookahead in motion prediction, was
    never transplanted** (`public-solutions-recon.md:313`). Pure CPU,
    existing caches, and it is a 0.915 solution's own component. Cheapest
    thing on this list.
  - **HOCT -- higher-order cell tracking**, forum /726521, posted by
    **Jordão Bragantini as HOST**: he is a co-author and it is the Royer
    lab, i.e. the lab that made this data. Paper arxiv 2607.11754, code
    and PRETRAINED MODEL at `github.com/royerlab/hoct`. "Higher-order" is
    exactly the multi-frame motion constraint this item is about, and he
    says plainly "we haven't tried it on the competition dataset yet".
    **Training code is withheld (under peer review), so it is
    inference-only.**

    The one recorded attempt is NEGATIVE and its reason is instructive: a
    rank-24 competitor fed it **uniform-ball masks** and it over-linked
    into spurious forks (/728551, via `strategy.md`). Bragantini's own
    caveat is "results depend on the quality of the segmentation mask" --
    and our pipeline emits CENTROIDS, not masks. Uniform balls are the
    worst possible mask. **Untested idea: watershed the xl detection
    heatmap seeded at the detected peaks, which respects real cell shape
    and is strictly better input than spheres.** Submission is offline,
    so the weights would need packaging as a dataset.
- **D. The ASSIGNMENT step, not the scoring.** `ranker_scale` measured
  per-dataset ranker AUC FLAT against density while adj_J slides hard --
  so crowding damages the pick, not the score. Untouched.

Discipline, from two failures today: an offline AUC/AP win is not an
adj_J. Everything here gets re-measured through the full pipeline before
it is believed.

### Why this ordering is structural, not a preference (user, 2026-08-24)

Divisions sit DOWNSTREAM of both. The forensics funnel says it exactly:
of the 12 holdout GT divisions, **1 parent undetected + 1 daughter
undetected + 3 outside `GATE_UM` + 2 below `ILP_FLOOR` = 7 are lost
before the division mechanism gets a say**, and it already takes all 5
that reach it. No division model can recover what detection and linking
threw away first.

The orphan patch is the sharper case: it feeds on the LINKER'S FAILURES.
A second daughter becomes adoptable only if it was detected AND the 1:1
assignment left it unclaimed. That is why improving the ranker backfired
once already -- "1:1 Hungarian steals second daughters into wrong
continuations, **draining the orphan pool the division patch mines**"
(`ranker_fine.py` header). A linker that gets more confident without
getting more correct starves the mechanism downstream of it.

So detection and linking are not merely higher-value targets. They are
the precondition for the division term to be worth anything at all.

### Process notes, each from something that went wrong today

- **A static undefined-name pass catches real bugs. Do not push when it
  is dirty.** It flagged `gc` as undefined; the push went out anyway and
  wasted a run. It had earlier caught `class UNet3D` and `_dist_um`
  missing from an assembled script -- both would have died at runtime.
- **Holdout deltas below ~0.003 are not resolvable on the LB.** Measured:
  two submissions whose holdout differed by +0.0014 scored -0.001 apart.
  Much of this session's fine tuning (the div_w peak at 0.55 was 0.0018
  above its neighbours) sits under that floor and should be treated as
  noise.
- **The family-offset rule predicts to about +-0.003, not +-0.001.** Two
  points now: -0.0158 and -0.0182. The first prediction landing on the
  nose was luck.
- Kernel assembly by slicing one script's head off another accumulates
  dead code -- `div_train.py` carries `GATE_SWEEP`, `train_at_gate`,
  `funnel` and two copies of `GATE_XY/GATE_Z` that nothing calls.
  Rebuild it from minimal parts before extending it further.



- **LB best 0.917** (ref 55690436, verbatim xl reproduction). Holdout
  baseline for the xl family: 0.9318 (adj 0.9185 / div 0.1333);
  family-specific local-to-LB offset ~-0.015.
- **Edge re-ranker line (2026-08-23):** v1 failed (sparse-GT labeling
  gave 97% positives); v2 fixed labels by EXCLUSION negatives (the
  candidate hitting the GT child is positive, all other in-gate
  candidates of that source are negative) -> 39.8k pairs, 46.8% pos,
  AUC 0.9884, linker-only adj_J 0.9239 vs Hungarian 0.9185 (+0.0054);
  v3 full-pipeline composite only 0.9320 vs 0.9318 — the edge gain is
  eaten by div_J 0.1333->0.0800. **Diagnosis (user's): 1:1 assignment
  steals division second-daughters into wrong continuations, draining
  the orphan pool the division patch mines.** v4 (in flight) tests the
  reconciliation: the ranker's labels already score parent->second-
  daughter edges HIGH (both GT children were positives), so a
  capacity-2 ILP fed ranker probs (division surcharge 0.75/0.85/0.95)
  can take both daughters natively; plus Hungarian p_min 0.30-0.40.
  Detection caches for all 44 datasets + the trained ranker pkl live in
  the `biohub-edge-ranker` kernel output (iterations cost ~minutes).
- **Toolchain notes for the next session:** CLI needs
  `KAGGLE_API_TOKEN` minted per call (kaggle-toolchain.md 1b); pack
  wheels + `pip -e repo --no-deps` is the proven env for ILP+scoring
  kernels; ERRORed kernels' outputs are downloadable but NOT attachable
  as kernel sources; official metric needs a GPU machine even to score.
- Uncommitted: two days of docs + scripts (progress/reference/recon
  restructure, zebrahub_extract rewrite, finetune_detect v3). Commit
  when the user says so.

## 2026-08-25 -- NMS costs 0.9% ON 6bba. NOT closed: the density hypothesis was tested on the LEAST dense domain.

> ⚠ **CORRECTED by the user before the ink was dry.** The hypothesis was
> that a FIXED 4.0 um radius hurts where cells are CROWDED. This run
> measured it on `6bba` at **97 cells per frame** -- the least crowded
> domain available. `44b6` runs **327** and Zebrahub's crops span
> **45 to 809**. A null result on the sparsest data says almost nothing
> about a density hypothesis, so **"the radius is already right" is
> supported for 6bba and nowhere else.** `nms3.py` runs all three.
>
> Why it happened, since the mechanism matters more than the instance:
> `biohub-prenms-cache` was built BEFORE the three-domain standard and
> covered only the 12 `6bba` holdout, and I ran the experiment the data
> allowed instead of waiting for the data the question needed. **Running
> what is available rather than what answers the question is its own
> failure mode**, and it produced a confident "closed" that was not
> earned.
>
> The user also pointed at where the test actually belongs: **Zebrahub
> is the only domain whose GT is dense, so it is the only place TRUE
> cell density per crop is known** -- on competition data
> `corr(annotation density, true density) = -0.20`. With crops spanning
> 18x in density, the right measurement is NMS loss correlated against
> true density across them, not three isolated numbers.

`nms_loss.py` -> `biohub-nms-loss` v1, **9 min on CPU**, on
`biohub-prenms-cache`'s pre-suppression peaks for the 12 `6bba` holdout
datasets.

| | |
|---|---|
| GT reachable BEFORE suppression | **0.9652** |
| GT reachable AFTER suppression | **0.9561** |
| **cost of NMS** | **0.0091** |

Per dataset the loss runs 0.0% to 3.8%, mostly under 1%.

### The radius sweep has nothing in it

| rule | recall | detections kept |
|---|---|---|
| fixed 2.0 um | 0.9564 | 168,240 |
| fixed 3.0 um | 0.9564 | 168,220 |
| **fixed 4.0 um (shipped)** | **0.9561** | 167,575 |
| fixed 5.0 um | 0.9554 | 166,609 |
| fixed 6.0 um | 0.9519 | 163,411 |
| adaptive, beta 0.30 | 0.9564 | 168,205 |
| adaptive, beta 0.45 | 0.9561 | 167,657 |
| adaptive, beta 0.80 | 0.9320 | 155,454 |

Halving the radius buys **+0.0003 recall for 665 more detections**, and
the adaptive rule -- radius = beta x the frame's median
nearest-neighbour distance, which was meant to remove the tuned
constant -- does not beat the fixed one anywhere.

**On `6bba`, at 97 cells per frame.** See the correction at the top:
this is the domain where the density hypothesis predicts the smallest
effect, so the table above cannot close the question. The adaptive rule
in particular was designed to matter WHERE DENSITY VARIES, and 6bba
alone barely varies.

### ⚠ Two errors of mine this run exposes

**1. The density correlation is computed against the wrong quantity.**
The `cells/fr` column reads 5-13, which is ANNOTATION density, not true
cell density -- `reference.md` records `corr(annotation density, true
density) = -0.20`, i.e. **negative**, and "N_true cannot be inferred
from the GT". So `corr(fraction lost to NMS, cells per frame) = +0.0041`
says nothing about true density and must not be quoted as if it did.

The conclusion survives anyway, by MAGNITUDE rather than by
correlation: NMS removes 0.91% of reachable GT in total, so even if
every bit of it landed on the densest datasets it could not produce
`corr(adj_J, density) = -0.733`. **The recorded explanation -- that the
density penalty sits downstream of edge scoring, in near-ties among many
candidates -- still stands, and the NMS hypothesis is refuted.**

**2. `det_floor`'s "13.1% of cells have no peak at any threshold" is a
ZEBRAHUB number and I generalised it.** I had called the 9.5%
"has a peak but is discarded" the largest recoverable loss on the board
and pointed NMS at it. That reframing applied a domain-gap number to the
competition, and the lead it motivated is now measured and empty.

**3. And the correction in (2) was ITSELF a bad comparison** -- the
user asked which dataset the NMS recall came from, which exposed it:

| | value | what the percentage MEANS |
|---|---|---|
| Zebrahub | 0.8688 | **dense GT: 86.9% of ALL cells have a peak** |
| 6bba (competition) | 0.9652 | **sparse GT: 96.5% of the 9.71% that are ANNOTATED** |

**These are not the same measurement.** The 6bba figure says nothing
about the ~90% of cells nobody labelled, and there is reason to suspect
the labelled ones are easier: the Zebrahub bench measured recall by GT
brightness quartile at **0.52 for the dimmest quartile against 0.74 for
the brightest**. If competition annotation skews bright, 96.5% is
optimistic by an unknown amount.

So the honest statement is: **among annotated cells on 6bba, 96.5% have
a peak.** Whether detection is "not where the loss is" on the full cell
population is UNMEASURED, and cannot be measured where the GT is sparse.

### Which dataset each 2026-08-25 experiment actually ran on

Written down because this session has repeatedly compared numbers from
different populations, and every instance was caught by the user rather
than by me.

| experiment | data | GT |
|---|---|---|
| `nms_loss`, `thr_sweep`, `poolsize`, `node_filter`, `div_gate` | 12 `6bba` holdout | **sparse** |
| `post_sweep` | 12 `6bba` + 12 `44b6` + 8 Zebrahub | mixed |
| `det_floor`, `det_ablate3`, `tta_probe`, `node_bright`, `morph_*`, `div_mass` | Zebrahub crops | **dense** |

**A percentage measured against sparse GT is not comparable to one
measured against dense GT**, even when both are called "recall".

### What this leaves

Detection on competition data is not where the loss is: 96.5% reachable
before NMS, 95.6% after, and the shipped gate and radius are both at or
near their best. The remaining gap between 95.6% reachable and the
pipeline's actual performance is downstream -- in which candidate the
linker picks, which is where `corr(adj_J, density) = -0.733` already
pointed.

## 2026-08-25 -- the GT is 212 KB, and the 44b6 arm is 2x noisier than the 6bba one

`gt_export.py` -> `biohub-gt-export` v1, **1 min on CPU**. Writes what
`read_geff` returns as one npz per dataset, for the 12 `6bba` holdout
and 12 `44b6` outside D32's training pool.

**24 datasets, 13,279 nodes, 12,839 edges, 212 KB total.** The
competition data is 87.6 GB of zarr and its `.geff` directories cannot
be fetched file-by-file; the GT inside them is a fifth of a megabyte.
That unblocks any machine without the competition mount -- see
`kaggle-toolchain.md` section 11 for the Colab worker it was blocking.

(Degradation noted in the log and worth keeping: `zarr` was not
installed, so `read_geff` fell back to its filesystem reader. A `.geff`
IS a zarr store, so that path matters even though the `geff` library is
optional.)

### ⚠ The measurement this exposes, which bears on `biohub-post-sweep`

| domain | datasets | GT edges | per dataset |
|---|---|---|---|
| `6bba` | 12 | **10,313** | 859 |
| `44b6` | 12 | **2,526** | **210** |

**The 44b6 arm carries 24% of the 6bba arm's ground truth**, so its
adj_J estimate is about **2x noisier** (standard error goes as
1/sqrt(n)). Its thinnest dataset has **49 edges**.

**44b6 is the worst of both worlds for a sweep arm:** 327 cells per
frame against 6bba's 97, so the ILP costs the most there, while the
annotated fraction is 0.77% against 9.71%, so the measurement is the
weakest. That is also why `biohub-post-sweep` ran past three hours
against an estimate of 100 minutes -- the estimate used `poolsize`'s
67 s per dataset, **measured on 6bba only.**

### How to read the three-domain rule given this

The rule -- an effect counts only if its adj_J delta has the same sign
in `6bba`, `44b6` and Zebrahub -- controls false positives well and has
**low power**. If each domain independently shows the correct sign for a
marginal effect about 65% of the time, all three agreeing happens ~27%
of the time. Most real small effects will fail it.

That is defensible: the holdout is 0 for 2 against the leaderboard at
this scale, so an effect the rule cannot detect is one that would not
have been actionable. **But do not read "nothing survived the rule" as
"these constants are already optimal."** Those are different claims, and
only the first one will have been measured.

If power is wanted later, enlarge the 44b6 arm rather than loosening the
rule: 71 `44b6` datasets exist and 12 are being used. Using every one
outside D32's pool would roughly quadruple that arm's ground truth, at
the cost of runtime and no quota.

## 2026-08-25 -- ⭐⭐⭐⭐ THE LOCAL HOLDOUT IS 0 FOR 2 AGAINST THE LEADERBOARD

**Submission 55757064 (D187 ranker, pool 32 -> 187) scored LB 0.920
against the D32 baseline's 0.921.** The local holdout had it at +0.0073
score and +0.0032 adj_J, reproduced bit for bit by two independent runs,
with both anchors holding.

### State it precisely: not "worse", INDISTINGUISHABLE

| | 55707206 (D32) | 55757064 (D187) |
|---|---|---|
| LB public | **0.921** | **0.920** |
| `total_bytes` on the hidden test | 262,623,879 | 262,803,696 (**+0.07%**) |
| local holdout score | 0.9313 | 0.9386 (**+0.0073**) |
| local holdout adj_J | 0.9222 | 0.9254 (+0.0032) |

The LB reports three decimals, so 0.001 IS its reporting resolution, and
the two pipelines emitted output within 0.07% of each other on the
hidden test. **The honest reading is that they are the same submission
as far as the leaderboard can see** -- which is worse news than losing,
because the +0.0073 the holdout measured simply is not there.

### Both paired local-vs-LB comparisons we have ever run

| change | local delta | LB delta | hidden-test output |
|---|---|---|---|
| windowed vs global | **+0.0019** | **-0.001** | +10% |
| ranker pool 32 -> 187 | **+0.0073** | **-0.001** | +0.07% |

**Two for two.** Every time a positive local delta has been taken to the
leaderboard it has failed to appear. The second one is the larger and
better-anchored of the two.

### The signal we explained away

`biohub-poolsize`'s own entry recorded that **D187's edge AUC was LOWER**
(0.9864 against 0.9884) while its adj_J was higher, and filed it as "a
bigger pool means a bigger, harder validation split, so edge AUC cannot
be compared across configurations". That rationalisation may have been
wrong: the simpler reading is that the larger pool made the ranker
slightly worse and the local adj_J gain was specific to `6bba`. **When a
second measurement disagrees with the one you like, do not explain it
away; carry it as unresolved.**

### What this changes

- **The best submission is still ref 55707206, LB 0.921.** Final
  selection allows two; that one holds a slot.
- **A local gain under ~0.01 on the 12-dataset holdout is not evidence.**
  All twelve are `6bba`, one imaging batch, and many knobs have now been
  tuned against them.
- The user identified both failure modes before this result landed:
  researcher overfitting to a 12-dataset holdout, and "only the LB is
  truth, but that means a lot of GPU or CPU may be wasted". Both are now
  measured rather than argued.
- `biohub-post-sweep`'s pre-registered three-domain rule -- an effect
  counts only if its sign agrees across `6bba`, `44b6` and Zebrahub --
  was designed as a precaution and now has evidence behind it.

### What it does NOT license

It does not say the pipeline cannot be improved, and it does not say
local measurement is useless. Local work still reliably rejects things:
`node_filter`'s -0.03 and `morph_edge`'s -0.0015 are far outside any
noise and would not have been worth an LB slot. **The holdout screens
out; it does not select in.**

## 2026-08-25 -- multi-frame linking, all three attempts, and the one that was never recorded

The user asked what happened to multi-frame tracking. Three attempts
exist; **`biohub-accel`'s result was never written down** and survived
only in a session scratchpad, which is how a run gets repeated by
accident. Recorded here.

### 1. Acceleration lookahead (Togashi's item 4) -- `biohub-accel`

Five columns on top of the ranker's 18: `acc_len`, `acc_cos`, `md_acc`,
`md-md_acc`, `acc_hist`.

| arm | cols | score | adj_J | div_J | TP/FP/FN |
|---|---|---|---|---|---|
| BASE | 18 | **0.9386** | **0.9254** | 0.1316 | 5/26/7 |
| ACC | 23 | 0.9319 | 0.9235 | 0.0833 | 3/24/9 |
| delta | | -0.0067 | **-0.0019** | -0.0483 | |

Stage A agreed in advance: edge AUC 0.9867 against 0.9864, permutation
importance +0.0000 to +0.0017 across all five.

**⭐ The cause is in the degradation counters, not in the model.**

```
accel_no_parent: 316,658
accel_one_step:  246,780
```

against **225,861 candidate pairs**. Acceleration needs THREE
consecutive positions; the backward chain is built by `pass1_tracks`, a
greedy linker, and for most candidates it cannot produce three. ~~All
five columns are then filled with 0 -- which the model reads as "no
acceleration", not as "unknown".~~

**So this is not evidence that multi-frame information does not help. It
is evidence that this implementation could not compute it.**

> ⚠ **TWO ERRORS IN THE PARAGRAPH ABOVE, both found 2026-08-28 after the
> user asked whether the zero fill was even firing correctly.**
>
> **1. The zeros ARE labelled.** `accel_of`'s own docstring: *"return
> zero and say how many steps there were, so the model can split on the
> count instead of reading 0 as 'not accelerating'"*. The fifth
> acceleration column is `acc_hist` — 0 = no backward link, 1 = one step,
> 2 = full — and `biohub-accel`'s ACC arm used all 23 columns, so the
> indicator was present and the model still lost by −0.0019. The
> "reads 0 as no acceleration" diagnosis is wrong, and a later session
> built on it, proposing a NaN fill that would have changed nothing.
>
> **2. The denominators do not match.** `accel_of` is called once per
> SEED per FRAME; 225,861 is a count of candidate PAIRS. Comparing
> 316,658 to it says nothing, so **"for most candidates it cannot produce
> three" is unsupported** — the real fraction has never been measured.
>
> What is actually unknown, and what `accdiag` measures: the share of
> seeds with `acc_hist == 2`; the per-frame MEDIAN acceleration, which is
> global stage drift and is not subtracted anywhere; and whether the
> backward chain from a greedy linker is right often enough for the
> second difference to mean anything.
>
> **ANSWERED 2026-08-28 by `biohub-accdiag`, top entry.** Computable
> **87.3%** of the time, so the "cannot produce three" story is dead.
> Stage drift is real but only **0.240** of the per-cell residual, and
> subtracting it changes every column by at most 0.0084 with no
> consistent sign. **The actual reason: `md_acc` (0.9376 flipped AUC) is
> a strictly WORSE version of `md` (0.9636) -- the acceleration term
> doubles the predictor's noise variance. Closed, negative, on
> measurement.**

Note also, by 2026-08-25's standard, that -0.0019 sits inside the range
where the holdout has inverted against the leaderboard twice. "It hurt"
is not supportable either. What is supportable: **no measurable effect,
and the feature is undefined for most rows.**

### 2. HOCT, higher-order cell tracking -- `biohub-hoct-score`

`similarity` AUC **0.7012** with images, against **0.9091** for plain
negative distance on the same 364 edges. Lost by 0.21. Training code is
withheld so there is no fine-tuning path. Closed; full entry below.

### 3. The division model's forward features -- `biohub-div-kinematics`

`sep_slope` separates a true division from a random pair 1.155 against
0.082, a 14x gap and the cleanest signal this project has found. It is
genuinely multi-frame. `biohub-div-gate` then wired it into the pipeline
and the prune sweep was monotone DOWNHILL. Closed as a post-hoc filter;
the measurement stands.

### ⭐ The version nobody has tried: TWO PASSES

Every attempt above computes multi-frame features from a GREEDY backward
chain, and that is exactly what the degradation counters say fails. The
ILP already produces a globally optimal capacity-2 solution whose tracks
are connected -- far better trajectories than the greedy pass.

**Solve once, build the multi-frame features from the SOLVED tracks,
re-solve.** That is not circular in the way it first sounds: the
features are not being used to build the trajectories they come from,
they are being used for a second solve. And it directly removes the
diagnosed failure -- three consecutive positions would be available for
most cells rather than a minority.

Cost: two ILP solves per dataset instead of one, so roughly double
`biohub-thr-sweep`'s ~10 min per configuration. Pure CPU on cached
detections. Untried.

## 2026-08-25 -- ⭐⭐⭐ the detection ceiling: 13.1% of cells have NO peak at any threshold, and 9.5% are thrown away after having one

`det_floor.py` -> `biohub-det-floor` v1, **16 min on a T4**, 41 crops,
every 20th frame, 50,897 GT points. Two questions in one pass.

**Anchor holds:** FULL at floor 0.05, gate 0.15 reads F1 0.7554 against
`det_ablate3` v3's 0.7539 (that run used every 10th frame).

### 1. The peak-finding floor is NOT a lever. Closed.

"Recall ceiling" = the fraction of GT points with ANY peak within 5 um,
before NMS and before any gate. Nothing downstream can exceed it.

| config | floor | peaks | ceiling |
|---|---|---|---|
| FULL | 0.05 | 64,584 | **0.8688** |
| FULL | 0.01 | 68,026 | **0.8690** (+0.0002) |
| B0 | 0.05 | 61,144 | 0.8610 |
| B0 | 0.01 | 63,405 | 0.8627 (+0.0017) |
| B2 | 0.05 | 51,282 | 0.7005 |
| B2 | 0.01 | 61,604 | 0.7712 (**+0.0707**) |

**Dropping the floor from 0.05 to 0.01 buys FULL +0.0002.** The F1
curves are IDENTICAL for every gate at or above 0.05 and strictly WORSE
below it -- the extra peaks are junk.

**The cells under 0.05 are not there.** They are not "below the
threshold", the detector produces no response for them at all. Do not
re-open this by lowering the floor further; `cache_train.py`'s 0.05 is
already the right choice and the caches need no rebuild.

### 2. ⭐ The number that matters: where the missing recall actually is

| | share of GT | recoverable by parameters? |
|---|---|---|
| detected today (FULL, gate 0.15) | **0.7738** | -- |
| **has a peak but is discarded** | **0.0950** | **YES** |
| no peak at any threshold | **0.1312** | **NO -- needs a better detector** |

**9.5 percentage points of recall sit in peaks that EXIST and are then
thrown away**, by `_physical_nms` (`NMS_UM` 4.0) or by the gate
(`UNET_THRESH` 0.15). That is five times the size of what
`biohub-thr-sweep` moved by touching the gate, and **`NMS_UM` has never
been swept.**

It points the same way as an unexplained number already on record:
`biohub-div-mass` found **44.7% of divisions had BOTH daughters inside
one watershed label** -- one peak serving two cells. NMS merging nearby
cells would produce exactly that.

⚠ **The ceiling is OPTIMISTIC and by the same mechanism.**
`ceiling_hits` counts a GT point as reachable if any peak lies within
5 um, with **no 1:1 constraint**, so two adjacent cells sharing one peak
are both counted as reachable. The true 1:1 ceiling is lower than
0.8688, and the gap between the two is itself a measure of how much NMS
merges. Worth computing directly.

### 3. B2 settled: genuinely the weakest, now measured rather than assumed

The user objected to `det_ablate3` v2's "B2 is nearly useless" and was
right that it was censored -- v3's sweep bottomed out at the
peak-finding floor with B2's curve still rising. At floor 0.01 the curve
**turns over inside the range**: 0.6712, **0.6744 (gate 0.02)**, 0.6741,
0.6693 ...

**B2's true best F1 is 0.6744**, against B0's 0.7536 and FULL's 0.7554 --
still last by about 0.08. Its recall ceiling also rises +0.0707 when the
floor drops, confirming the calibration diagnosis: B2's probabilities
really are systematically lower. The objection was correct, the
correction is now complete, and the ordering survives it.

## 2026-08-25 -- ⭐⭐⭐ flip-and-average: same F1, an 11% DIFFERENT detection set. The reverse proof falsified my claim.

`tta_probe.py` -> `biohub-tta-probe` v1, **17 min on a T4**. The user
asked for the reverse proof of "the four flip views buy nothing" -- the
claim I made from `biohub-det-ablate3` v2's F1 0.7546 (3 passes) against
0.7539 (12 passes). It is FALSE in the sense that matters.

### A. The four views are correlated but NOT the same volume

| | |
|---|---|
| Pearson r, view 0 vs views 1-3, per branch per frame (n=1845) | median **0.9781** [p10 0.9402, p90 0.9926] |
| spread ACROSS the 4 views / spread across voxels | median **0.1085** [p10 0.061, p90 0.180] |
| mean per-voxel abs difference, heatmap(4) vs heatmap(1) | 0.00198 |

A convolution is translation-equivariant by construction but **nothing
in the architecture makes it flip-equivariant** -- a kernel that responds
to "bright left, dark right" cannot respond identically to the mirror.
Flip invariance is LEARNED from augmentation, so it is approximate and
never exact. The 11% across-view spread is that residual, and it is
exactly what flip-averaging averages away.

This was predicted by `det_ablate3` v3 before the run: flip-averaging
buys about +0.003 F1 for a SINGLE branch (B0 +0.0028, B1 +0.0040, B2
+0.0029), which is impossible if the four views were identical volumes.

### B. The F1 difference is inside its own error bar

| | |
|---|---|
| paired over 41 crops, one view minus four | **+0.00109, SE 0.00150** |
| in units of its own SE | **+0.73** |
| crops where ONE view wins | **19 of 41** |

19 of 41 is a coin flip. **Neither "buys nothing" nor "slightly better"
is supported.** v2 pooled two F1 numbers and reported a 0.0007 gap with
no error estimate; the honest statement was always "indistinguishable at
this sample size".

### ⭐ C. And yet they find DIFFERENT CELLS -- 11% of them

| | |
|---|---|
| paired within 2 um | 50,560 |
| only in the FOUR-view set | 2,786 |
| only in the ONE-view set | 3,512 |
| **symmetric difference** | **6,298 of 56,858 = 11.08%** |

The COUNT gap was 1.3%. The SET gap is **11%**. About one detection in
nine differs. **These are two different detectors that happen to score
the same.** Count agreement is not set agreement, and F1 agreement is
not detector agreement.

### What this changes

**RETRACTED:** "three forward passes tie twelve, so switch and spend the
saved runtime on the lower gate." Switching changes 11% of the
detections, and `biohub-node-filter` measured that detection quality and
adj_J do not move together -- removing detections costs edges faster
than it removes noise. An 11% churn in the node set propagates into
linking, the ILP, and the division patch, and F1 says nothing about
where it lands.

**STANDS:** three passes are not WORSE than twelve on detection F1, and
that is a real 4x saving *if* a pipeline run confirms adj_J holds. It
has to be scored, not assumed. The same caution applies to `B0(4)` and
`B01(8)`, which the ablation listed as ties.

### Still untested

The prediction that the disagreement concentrates on LOW-SCORING,
near-threshold detections -- confident cells should be found by every
view. `set_overlap` counts the symmetric difference but does not record
the scores of the 6,298 detections in it. Cheap to add, and it would say
whether an 11% churn is 11% of the borderline cases or spread across the
whole score range. Those are very different risks.

## 2026-08-25 -- ⭐⭐⭐ the detector ablation, finally run: the ensemble buys DIVERSITY, and 3 passes tie 12

`det_ablate3.py` -> `biohub-det-ablate3` v3, **17 min on a T4**. The
question is the user's from 2026-08-24 and had never been answered; v1
and v2 of the old `det_ablate` never produced a number. Zebrahub dense
GT, greedy 1:1 within 5 um, **every configuration at ITS OWN best gate**.

**Anchor holds:** FULL reads recall 0.7738 / precision 0.7350 against
`biohub-node-bright` v2's 0.7659 / 0.7362 on the same 41 crops (that run
used every 5th frame, this every 10th).

| config | passes | gate* | recall | prec | **F1\*** | F1@0.15 |
|---|---|---|---|---|---|---|
| B0 (`bright`, NO tophat) | 4 | 0.15 | 0.7812 | 0.7235 | 0.7512 | 0.7512 |
| B1 (`traintophat`) | 4 | 0.20 | 0.7936 | 0.6915 | 0.7391 | 0.7251 |
| B2 (`v2_tophat_b32`) | 4 | 0.05 | 0.6699 | 0.6686 | 0.6693 | 0.6239 |
| **B01** | 8 | 0.20 | 0.7700 | 0.7474 | **0.7585** | 0.7522 |
| B02 | 8 | 0.10 | 0.7787 | 0.7177 | 0.7470 | 0.7217 |
| B12 | 8 | 0.10 | 0.8068 | 0.6651 | 0.7291 | 0.7289 |
| FULL | 12 | 0.15 | 0.7738 | 0.7350 | 0.7539 | 0.7539 |
| **FULL_f1** | **3** | 0.15 | 0.7796 | 0.7312 | **0.7546** | 0.7546 |
| FULL_f2 | 6 | 0.15 | 0.7762 | 0.7327 | 0.7539 | 0.7539 |
| **B0_f1** | **1** | 0.15 | 0.7817 | 0.7178 | 0.7484 | 0.7484 |
| B1_f1 | 1 | 0.20 | 0.7958 | 0.6830 | 0.7351 | 0.7193 |
| B2_f1 | 1 | 0.05 | 0.6841 | 0.6495 | 0.6663 | 0.6369 |

### ⭐ The ensemble buys DIVERSITY, not member quality

| pairing | v2 (all at 0.15) | **v3 (own gates)** |
|---|---|---|
| B01 - B12 | +0.0233 | **+0.0294** |
| B02 - B12 | -0.0072 | **+0.0179** |

**At fair gates, every pairing containing B0 beats the pairing without
it** -- and B0 has the WORST reported `val_recall` of the three (0.5731
against 0.6589 and 0.5789). B0 is also the only branch that does not get
top-hat input; branches 1 and 2 are pixel-identical in what they see.
The one genuinely different view of the data is the one the ensemble
cannot do without, and its solo metric said it was the weakest.

**`val_recall` anti-predicts here**, which is now the eighth time this
session an offline number has run opposite to the measured outcome.

### The user caught a real error, and the correction is partial

v2 scored every configuration at 0.15 and read B2's F1 of 0.6239 as
"nearly useless". The user objected that this needed more care. Correct:
**0.15 is calibrated for the THREE-BRANCH AVERAGE, and a single branch
lives on a different probability scale.** At its own gate B2 gains
**+0.045** (0.6239 -> 0.6693).

Two things the correction does NOT overturn, and one it leaves open:

- B2 is still last by ~0.07, far outside noise.
- **B2 does not help the ensemble either: FULL - B01 = -0.0046**, i.e.
  removing B2 from the three is slightly BETTER, and B01 at gate 0.20 is
  the best F1 of all twelve configurations.
- ⚠ **B2's optimum is CENSORED.** Its F1-vs-gate curve falls
  monotonically from the lowest gate swept (0.6693, 0.6585, 0.6499,
  0.6239, 0.5895 ...), so its peak is at or below 0.05 -- and 0.05 is
  `DETECT_THRESH`, the peak-finding floor, where no peaks exist below.
  Measuring B2's true best needs a fresh detection pass with a lower
  floor. Same for `B2_f1`.

### Flip-and-average: helps ONE branch, does nothing on top of three

| | 4 views minus 1 view |
|---|---|
| B0 alone | **+0.0028** |
| B1 alone | **+0.0040** |
| B2 alone | **+0.0029** |
| **FULL (three branches)** | **-0.0007** |

**SUPERSEDES what v2 was read as saying.** Reporting FULL_f1 0.7546
against FULL 0.7539 as "the four flip views buy nothing" was wrong twice
over: the 0.0007 had no error bar, and the flips DO buy about +0.003 --
for a single branch. Averaging across branches already performs the same
variance reduction, so the two are redundant with each other and the
second one is free to drop.

This also predicts what `biohub-tta-probe` must find: **the four views
cannot be identical volumes**, or averaging them could not have moved a
single branch by +0.003.

### ⭐ What is cheap enough to matter

Configurations within 0.003 F1 of FULL, cheapest first:
**`FULL_f1(3)`, `B0(4)`, `FULL_f2(6)`, `B01(8)`, `FULL(12)`.**

**Three forward passes tie twelve.** That is a 4x cut in detection cost
on every future GPU run, and it bears directly on the submission: the
global solve is projected at ~11.8 h against a 12 h cap (ref 55707206's
description), which is what currently makes the +0.0033 threshold change
from `biohub-thr-sweep` too risky to submit on its own.

> **SUPERSEDED 2026-08-25 — that verdict rests on a number nobody
> measured.** 11.8 h is `210 s/dataset × ~200 hidden volumes`, and both
> factors are assumptions: the 210 s was timed on the 4 VISIBLE movies,
> and the ~200 comes from the train split having 199, with nothing
> establishing the hidden split is the same size. Nothing on record
> measures the hidden rescore's duration at all. The user's standing
> observation — no submission has ever still been running the next
> morning — is the only direct evidence and puts it under 8 h. **So
> "too risky to submit on its own" is not established**, and the
> threshold change should be reconsidered on its merits. See
> `kaggle-toolchain.md` section 9 for the measurement that would settle
> it.

> WARNING **QUALIFIED the same day by `biohub-tta-probe` (entry above).**
> "Tie" means tie ON F1. The two configurations disagree on **11.08% of
> their detections** (symmetric difference 6,298 of 56,858, matched
> within 2 um), so switching is not a free saving -- it is a DIFFERENT
> detector with the same summary score, and adj_J does not follow
> detection F1. Score it through the pipeline before spending the
> saving. The same caveat applies to `B0(4)` and `B01(8)`, which this
> entry also lists as ties.

⚠ **This is DETECTION F1, not adj_J.** `biohub-node-filter` measured
that the two do not move together -- removing detections costs edges
faster than it removes noise. A configuration change has to be scored
through the pipeline before it can be believed, and the gate that
maximises F1 is not necessarily the gate that maximises adj_J.

## 2026-08-25 -- ⭐⭐⭐ morphology in the EDGE RANKER: adj_J -0.0015. The whole morphology line is closed.

`morph_edge.py` -> `biohub-morph-edge` v2, **106 min on CPU**, trained on
187 datasets. Six SIGNED morphology columns appended to the ranker's 18:
four child/parent ratios (mean, max, std, MASS) and the two absolute
brightness levels.

**BASE reproduced the anchor exactly** -- score 0.9386, adj_J 0.9254,
div_J 0.1316, TP/FP/FN 5/26/7 -- so the comparison is valid.

| arm | cols | score | adj_J | div_J | TP/FP/FN |
|---|---|---|---|---|---|
| BASE | 18 | **0.9386** | **0.9254** | 0.1316 | 5/26/7 |
| MORPH | 24 | 0.9326 | 0.9240 | 0.0857 | 3/23/9 |
| delta | | -0.0060 | **-0.0015** | -0.0459 | |

**adj_J -0.0015 is below the ~0.003 holdout floor. Morphology does
nothing for the edge ranker.** Stage A agreed in advance: edge AUC
0.9866 against 0.9864, and permutation importance for all six columns
between +0.0003 and +0.0007.

### The div_J drop is NOT attributable, and saying so matters

div_J fell 0.0459, which unpacks to **TP 5 -> 3: two divisions**. The
standing warning applies -- "⚠ Five true divisions. Every row rests on
TP counts of 1-5, far under anything resolvable alone."

More importantly the pre-registered mechanism is ruled out by this run's
own Stage A. The worry written into the script was that a
morphology-aware ranker would push mother->daughter edges DOWN, refuse
both daughters, and starve the orphan patch (whose proposals need a
parent with exactly one child). That requires the model to SEE the size
change. It cannot: **`MASS_ratio` over 225,861 rows is median 0.994,
p10 0.722, p90 1.270** -- a daughter should sit near 0.5 and the tenth
percentile does not approach it. The script's own pre-registered reading
of that distribution was "if the p10 does not reach anywhere near 0.5
the division signature is not in this feature and the orphan-starvation
worry is moot."

So: **no effect on edges, and a two-event move in divisions that this
sample cannot resolve.** Not "morphology starves divisions".

### The morphology line, closed end to end (2026-08-25)

| route | measurement | verdict |
|---|---|---|
| mass conservation, voxel count | per-fork AUC 0.605 | wrong quantity measured |
| fringe vs tessellation | 76.8% background-bounded, density slope -0.028 | **user right: fringe** |
| mass conservation, brightness | per-fork AUC 0.621, +0.0157 | fringe real, nothing under it |
| box brightness as a node filter | F10 - S10 adj_J **-0.0297** | loses to raising the threshold |
| morphology in the edge ranker | adj_J **-0.0015** | no effect |

Underneath all five: **no dataset here annotates cell size**, so size is
unverifiable rather than merely weak (see the ruler entry). Brightness
escapes that -- its label exists -- and brightness genuinely separates
real detections from hallucinations at AUC 0.866 on Zebrahub. It just
does not survive the transfer to competition data, and removing
detections is expensive under an edge-counting metric regardless.

**Do not reopen this with a better mask algorithm or another feature
built on masks.** Reopen it only if a size or segmentation annotation
appears, or if the filter can be trained on competition data, which
needs a spurious-detection label that sparse GT cannot give.

> ⚠ 2026-09-02, provenance: the "do not reopen" sentence above is the
> 2026-08-25 session's own verdict, NOT the user's instruction. The
> user, 2026-09-02: a better algorithm can certainly be tried. What the
> table above actually killed is the tested claim -- six mask-derived
> size/brightness ratio columns from this project's own masks add
> nothing to the edge ranker (AUC 0.9866 vs 0.9864). Untested residue:
> orientation/shape (inertia tensor, the HOCT/Trackastra feature set),
> texture, and learned pixel embeddings (the official edge head's
> U-Net features at node positions -- see the edgeprob line).

## 2026-08-25 -- ⭐⭐⭐ the node filter LOSES to raising the threshold. Cutting detections is expensive.

`node_filter.py` -> `biohub-node-filter` v1, **66 min on CPU**. Assembled
from `poolsize.py` so environment, caches, holdout and ILP are identical
to what produced the anchor; the D187 ranker is LOADED from
`biohub-poolsize`'s pkl rather than retrained, so the linker is
submission 55757064's byte for byte.

**BASE reproduced the anchor exactly** -- score 0.9386, adj_J 0.9254,
div_J 0.1316, TP/FP/FN 5/26/7 -- so every comparison below is valid.

| arm | cuts | score | adj_J | div_J | TP/FP/FN |
|---|---|---|---|---|---|
| **BASE** | nothing | **0.9386** | **0.9254** | 0.1316 | 5/26/7 |
| F10 | bottom 10% by node filter | 0.8879 | 0.8736 | **0.1429** | 5/23/7 |
| S10 | bottom 10% by det score | 0.9168 | 0.9033 | 0.1351 | 5/25/7 |
| F20 | bottom 20% by node filter | 0.8150 | 0.8029 | 0.1212 | 4/21/8 |
| S20 | bottom 20% by det score | 0.8878 | 0.8753 | 0.1250 | 4/20/8 |

**F10 - S10 = -0.0297 adj_J. F20 - S20 = -0.0724.** At a matched cut
count the learned filter is WORSE than simply raising the detection
threshold, despite leading it by +0.1418 AUC offline on Zebrahub.

### Two findings, and the second is the more useful one

**1. The Zebrahub -> competition transfer failed.** The run was designed
against exactly this risk -- cutting by PERCENTILE within each dataset so
only the ORDERING had to survive, never the probability scale. The
ordering did not survive either. `morph_cheap`'s rho 0.965 licenses box
statistics as a stand-in for mask statistics WITHIN a domain; it says
nothing about ranking detections across domains, and this run is the
evidence that it does not carry.

**This is the FIFTH time an offline gain reversed in the pipeline** --
after scale-invariant edge features, the division model's four arms,
D187's own edge AUC, and div-gate's prune.

**2. Removing detections is expensive under adj_J, by a mechanism that
generalises.** Cutting 10% by the BEST EXISTING SIGNAL, the detection
score itself, already costs **-0.0221 adj_J**; 20% costs -0.0501. The
metric counts EDGES: a real cell carries roughly two of them and loses
both when it is cut, while a false detection links to little and so
returns little when removed. Any node filter must therefore be far more
precise than the detection score merely to break even, and none of these
was.

The one signal pointing the right way: **F10 raised div_J to 0.1429 and
cut false divisions 26 -> 23** with TP and FN unchanged, i.e. it removed
hallucinations that were feeding spurious forks. div_J carries a tenth
of adj_J's weight, so -0.0518 adj_J buries +0.0113 div_J.

### What would have to change for this line to live

Train the filter on COMPETITION data. That needs a "this detection is
spurious" label, which sparse GT cannot give -- an unmatched detection
there means unannotated, not false. That absence is why Zebrahub was
used, and the transfer failure is why that substitution did not work.
**Both ends of this line are blocked by the same missing label.**

### ⚠ An open question this run RAISES, with the evidence conflicting

If removing detections is expensive because missing cells cost edges,
the asymmetry may run the other way: `UNET_THRESH` is 0.15, and LOWERING
it would add detections, buying recall at the price of precision.

Conflicting evidence, both already on record, neither decisive:

- FOR: this run measures a controlled intervention and it says removing
  real cells is what hurts.
- AGAINST: `corr(edge J, N/N_true) = -0.64` says over-detection tracks
  worse edge J -- but that is a correlation ACROSS datasets, confounded
  by density, not an intervention.

A threshold sweep would settle it and is the same shape of run as this
one. Nobody has run it downward.

## 2026-08-25 -- ⭐⭐⭐ THE RULER IS THE PROBLEM. Mask-based mass is unmeasurable here, and the line closes.

**User's objection, and it ends the line:** the detector does not even
achieve full recall on Zebrahub, so expecting its masks to be accurate is
asking too much. Checked against the record rather than argued:
**recall 0.65 / precision 0.64** at the pipeline's det gate
(`progress.md` "the clean ruler's first reading: recall 0.65, not 0.99").

**35% of cells have no detection at all and 36% of detections are
spurious.** The watershed's seeds ARE those peaks, so:

- a MISSED cell leaves its volume unclaimed and a neighbouring basin
  swallows it -- that neighbour's voxel count and brightness now contain
  a whole extra cell;
- a SPURIOUS peak splits one real cell into two basins.

And the bias runs in exactly the direction that manufactures 1.713: at t
there is one seed, at t+1 there are two, and the pair can claim not only
the mother's old territory but any of the 35% unclaimed ground nearby.
**The 40% merged-label rate is direct evidence of the same failure** --
the detector found ONE peak where there were TWO cells. An instrument
that cannot count cells cannot measure their size.

### The validation that was not one

`morph_probe` established our `regionprops` reproduces HOCT's
`create_graph` at **0.000e+00** and this was read as "the extractor is
correct". It is not: **it proves two implementations compute the SAME
quantity, not that the quantity is a cell.** Two correct implementations
of the wrong thing agree perfectly.

Same defect as the diameter anchor earlier today (7.08 / 8.83 um also
came from watershed-on-heatmap masks). **A check drawn from the same
construction as the thing under test validates nothing.** Both instances
were caught by the user, both within one day, and this is the general
form: an anchor must come from OUTSIDE the construction, or it is only
measuring self-consistency.

### Why a clean subsample does not rescue it either

Zebrahub's GT is dense, so one could keep only divisions whose whole
local neighbourhood is perfectly detected -- every GT node matched, no
unmatched detection nearby -- roughly 0.65^5 ~ 12% of events, a few
hundred. That could answer the SCIENCE (is mass conserved) but not the
ENGINEERING: **at inference there is no GT, so the pipeline cannot tell
a clean neighbourhood from a dirty one**, and in the dirty majority the
quantity is noise. A feature that is only trustworthy when you already
know the answer is not a feature.

### Status

CLOSED: mass conservation as a division test, on masks built from this
detector. Not because the physics is wrong -- matter does not appear
from nowhere and the user is right about that -- but because **no ruler
here can measure it.** Do not retry with another threshold, another
weighting, or another mask floor; all of them are downstream of the same
seeds.

**This raises the value of detector work.** Poor detection was costed at
~1.75% of the score through missed edge endpoints. It is now also the
thing that makes EVERY mask-based morphology feature untrustworthy --
size, shape, brightness, all of them.


## 2026-08-25 -- ⭐⭐ brightness strips the fringe and the signal is still not there. Size is DONE for divisions.

`div_intmass.py` -> `biohub-div-intmass` v2, **4 min on CPU**, 12 crops
carrying 5,699 division events. Sum BRIGHTNESS inside each mask instead
of counting voxels, so a dim fringe is discounted continuously with no
new threshold -- the retry `mask_anatomy` called for.

**Answer: per-fork AUC 0.6210 by brightness against 0.6054 by voxel
count. +0.0157. Mass conservation does not come back.**

### The gate passed -- brightness really does discount the fringe

| the outer one-voxel shell holds | |
|---|---|
| **volume** | 47.7% |
| **brightness** | **33.3%** |

A 14.4-point discount. The fringe is real and it is dim, exactly as
`mask_anatomy` said. So step 2's numbers are attributable.

### The anchor held to 0.0003

| | div_mass, 41 crops | here, 12 crops |
|---|---|---|
| voxel-count DIV median | 1.772 | **1.784** |
| voxel-count per-fork AUC | 0.6051 | **0.6054** |

The subset reproduces the whole, so the brightness-minus-voxel
difference is the only thing that varied.

### The measurement

| | voxel count | **brightness** | conserved predicts |
|---|---|---|---|
| DIV `(Ia+Ib)/Ip` | 1.784 | **1.713** | 1.0 |
| CONT | 1.000 | **1.000** | 1.0 |
| FAKE FORK | 2.009 | 2.027 | 2.0 |
| **per-fork AUC** | 0.6054 | **0.6210** | |

Stripping the fringe moved the division ratio 4% of the way toward 1.0
when it needed to move about 80%. **The fringe was real but it was not
what was hiding the signal; there was no signal under it.**

### ⭐ The user's photobleaching worry, answered

The user raised that mother-at-t and daughters-at-t+1 are two separate
exposures, so bleaching, focus drift or illumination drift would bias
I(t+1)/I(t) for reasons that are not biology. **`CONT Inext/Inow` is
exactly that control and it reads 1.000 over 45,657 non-dividing
cells.** No drift. (`_norm` also normalises each frame by its own 50th
and 99.5th percentiles, so global drift is removed before this is even
measured -- and the CONT row confirms the removal worked.)

### Two live explanations for 1.713, neither of which changes the call

1. the two daughters genuinely carry ~1.7x the mother's fluorescence --
   either not conserved, or the mother's last annotated frame is not the
   frame where her material is all present;
2. the mother is mid-mitosis at t, the detector responds weakly to
   condensed chromatin, her mask is small and her brightness undercounted.

Also a hard ceiling regardless: **2,289 of 5,699 divisions (40%) had both
daughters in one label** and were dropped, so the 2,870 usable ones are
the already-well-separated ones -- a biased, easier sample, and the
answer is still negative on it.

### ⚠ Where this leaves divisions, checked against the record rather than recalled

I recommended wiring div-kinematics' FULL model (1.8 FP per TP vs BASE's
6.9) into the pipeline. **That was already spent and it failed:**
`biohub-div-gate` did exactly that on 2026-08-24 and the prune sweep is
monotone DOWNHILL -- doing nothing wins, because the bottom 25% of forks
by `p_div` holds 20% of the TPs against only 4% of the FPs. The 1.8 was
measured over every candidate triple `build()` emits; the pipeline emits
553 forks that already passed the ILP or the orphan patch, and those are
the hard cases. Full entry below.

**The correction itself is the reusable part: I produced 1.8 from recall
and did not check whether it had been spent. `working-agreement.md` says
look numbers up at the moment of use -- that rule covers not just the
VALUE but whether the lead is still open.**

The division ledger as it now stands:

| route | status |
|---|---|
| morphology / size as a division signal | measured today, AUC 0.621, too weak |
| `p_div` as a post-hoc fork filter | dead 2026-08-24, monotone downhill |
| forking harder to buy true divisions | dead, division FN never below 6 |
| `div_forks`' three uses | all null 2026-08-24 |
| **morphology's effect on the ORPHAN SUPPLY** | **open -- `morph_edge`'s div_J** |


## 2026-08-25 -- ⭐⭐⭐ the masks are cells plus a FRINGE, not a tessellation. The user was right.

`mask_anatomy.py` -> `biohub-mask-anatomy` v1, **19 s on CPU**, labels
only. Two tests that share no arithmetic; both land on FRINGE.

**This overturns my reading of `div_mass` the same day I wrote it.** I
said a voxel count measures the TERRITORY between peaks and filed the
line closed. The user said it sounds like the edges are being
over-counted, then proposed the method that settled it: assume the
fringe hypothesis, derive what it forces, and check that.

### Test 1 -- when a mask stops, why does it stop?

A fringe ends where the heatmap falls below `MASK_FLOOR`, i.e. at
BACKGROUND. A territory boundary is a watershed line drawn halfway to a
neighbouring peak: cell on both sides, no background anywhere near. So
count boundary faces and ask what is on the other side.

| boundary faces against | count | share |
|---|---|---|
| **BACKGROUND** | 14,911,652 | **76.8%** |
| another mask | 4,508,503 | 23.2% |

Three quarters of every mask surface is cut by the threshold, not by a
neighbour.

### Test 2 -- does a mask shrink when the field is crowded?

The 41 crops span **45 to 809 cells per frame, an 18x range**.

| crop | cells/frame | median vol | bg face % |
|---|---|---|---|
| ZSNS004_z2y5x5 | 45 | 523 um^3 | 95.7% |
| ZSNS005_z1y1x5 | 55 | 451 um^3 | 97.5% |
| ZSNS001_z2y2x7 | 450 | 541 um^3 | 74.3% |
| ZSNS004_z2y1x3 | 606 | 571 um^3 | 81.4% |
| ZSNS005_z1y6x3 | 809 | 534 um^3 | 72.1% |

**log(volume) on log(cells per frame): slope -0.028, r -0.127.**
Territory requires -1 (a cell's share is the volume divided by the
cells). Volume spans 2.4x against density's 18x. A tessellation cannot
do this; a cell plus a fringe can, because neither knows how crowded it
is. Even the densest crop is still 72% background-bounded -- territory
exists but never dominates.

### What this means

`div_mass`'s 1.772 is not evidence that mass is not conserved. It is
evidence that **we weighed the wrong thing**: mask = core M + fringe H
with H = 3.39 M, so ~77% of the volume is a shell about 1.3 voxels thick
on a 6.62-voxel mask. Mass conservation has still never been tested.

**The retry is `div_intmass.py` -> `biohub-div-intmass`: sum BRIGHTNESS
inside the mask instead of counting voxels.** A fringe voxel sits at
heatmap 0.05-0.1 and reads dim; a core voxel is near 0.9 and reads
bright, so brightness discounts the fringe continuously with no new
threshold -- which matters because at 6.6 voxels across there is no room
to erode. It is gated on measuring that discount rather than assuming
it: the outer shell's share of VOLUME against its share of LIGHT, with
step 2 suppressed if they are equal.

### Method note worth keeping

The user's proof-by-contradiction beat the correlation study I was about
to write: **19 seconds against a multi-crop regression, because assuming
a hypothesis and hunting its forced consequence finds a one-number test.**
Also on record: my diameter anchor (7.08 / 8.83 um) could not arbitrate
this at all, because those numbers came from watershed-on-heatmap masks
too and carry the same fringe. An anchor drawn from the same
construction as the thing under test proves nothing about it.


## 2026-08-25 -- ⭐⭐ mass conservation: a voxel count cannot test it. WHY is open.

`div_mass.py` -> `biohub-div-mass` v1, **3 min on CPU**. Reads only
`biohub-zh-masks`; no detector, no images, no competition mount.

**Answer: (Va+Vb)/Vp is 1.772 where mass conservation predicts 1.0, and
a fake fork is 1.984 where it predicts 2.0. The two are nearly the same
number, so a voxel count on these masks cannot tell a real division from
a mother plus a passing neighbour. Per-fork AUC 0.605.**

> ⚠ **CORRECTED same day, and the correction matters.** This entry
> first said the cause was settled -- that the watershed measures
> TERRITORY between peaks rather than cell mass -- and filed the line
> CLOSED. That was over-claimed. The user proposed the competing
> reading: the masks are inflated by a fuzzy FRINGE and the real signal
> is buried under it. Writing mask = core M + fringe H, a division reads
> (M+2H)/(M+H), which equals the measured 1.772 at **H = 3.39 M** -- the
> cell would be 22.8% of the mask, a shell only ~1.3 voxels thick on a
> 6.62-voxel mask, since volume goes as the cube. A fake fork reads 2.0
> and a continuation 1.0 under EITHER reading. **Every number below fits
> both, and they prescribe opposite next moves:** territory means no
> threshold helps, fringe means strip it and the signal returns.
> `mask_anatomy.py` -> `biohub-mask-anatomy` decides it. Until that
> lands, the measured failure stands and the CAUSE does not.
>
> Note also that the diameter anchor cannot arbitrate: 7.08 / 8.83 um on
> the competition data came from watershed-on-heatmap masks too, so both
> could be inflated by the same fringe.

### Both gates and both anchors passed, so the negative is attributable

- 974,613 of 1,021,008 GT nodes (**95.5%**) land inside a label.
- median equivalent diameter **10.76 um** [9.00, 13.54]. Anchors: 7.08 um
  over all competition detections, 8.83 um over GT-matched ones. Zebrahub
  cells slightly larger; same order, so the pooled indexing is right.
- mining: **7,894 division events, 0 crops empty** (15,788 before the
  daughter-order dedupe). The broken-mining signature -- 68 positives,
  26 of 41 crops empty -- did not appear. 945,261 continuations.

### The measurement

| quantity | mass conservation predicts | measured |
|---|---|---|
| DIV `(Va+Vb)/Vp` | **1.0** | **1.772** [1.03, 2.93] |
| DIV `Vchild/Vmother` | 0.5 | 0.878 [0.43, 1.53] |
| DIV diameter ratio | 0.794 | 0.958 [0.76, 1.15] |
| CONT `Vnext/Vnow` | 1.0 | **1.000** [0.73, 1.37] |
| FAKE FORK `(V1+Vq)/V0` | 2.0 | **1.984** [1.31, 3.18] |

per-FORK AUC (true division vs fake fork) **0.6051**; per-EDGE AUC
(daughter vs continuation) **0.6247**.

### Why this is a clean negative and not a broken run

**The two control rows hit their predictions exactly.** Continuations
read 1.000 and fake forks read 1.984. The machinery reproduces the
physics wherever the physics is about COUNTING CELLS. It fails only on
the row where the answer depends on cell SIZE.

One explanation covers every number: the mother is ONE peak and so ONE
basin; the two daughters are TWO peaks and so TWO basins; basins are
sized by neighbour spacing, not by the cell. The ratio is then ~2
regardless of whether mass is conserved -- which is why true divisions
(1.772) and fake forks (1.984) are nearly the same number, and why the
AUC is 0.60 rather than 0.5 or 0.9.

Two corroborations:

- **CONT diameter ratio is 1.000 with p10-p90 of [0.900, 1.111].** Real
  cell volume does not hold to +-10% frame to frame. That tightness is a
  tessellation, not a measurement of matter.
- **44.7% of divisions had BOTH daughters inside ONE label** (3,532 of
  7,894; a further 678 lost a member to non-detection, leaving 3,684
  usable). Freshly separated sisters touch, and a watershed seeded on a
  detection heatmap does not split them. That fraction is a hard ceiling
  on what ANY size feature could see here.

### What this does and does not close

MEASURED and not in dispute: a VOXEL COUNT on these masks separates true
divisions from fake forks at AUC 0.605. As a feature, that is nothing.

OPEN: **why**. See the correction at the top. `mask_anatomy.py` asks the
one question that splits the two readings using nothing but the label
array -- what does a mask's surface actually touch? A fringe ends where
the heatmap falls below `MASK_FLOOR`, i.e. at BACKGROUND; a territory
boundary is a watershed line halfway to a neighbour, with cell on both
sides. It cross-checks with density: across the 41 crops, 44 to 815
cells per frame, territory predicts median mask volume falling as 1/N
(log-log slope near -1) and fringe predicts flat (near 0).

NEVER MEASURED: whether cell mass is actually conserved at division. An
intensity-weighted mass (background-subtracted brightness sum inside the
mask, which `morph_edge.py`'s `crop_stats_all` already computes as its
4th column) is a property of the cell rather than of the region, and
discounts a dim fringe for free. It needs the Zebrahub images mounted
alongside the masks, still CPU. **Weigh it against the 44.7% ceiling
before spending a slot** -- that fraction is a detector failure (one
peak served two daughters) and neither reading above touches it.


## 2026-08-25 -- masks for all 41 Zebrahub crops: the morphology line is now free

`zh_masks.py` -> `biohub-zh-masks` v1, **54 min GPU, 41/41 written, 0
skipped, no degradation warning fired.** Watershed on the detector
heatmap, pooled to an isotropic 1.625 um grid, one `.npz` per crop
carrying `labels` plus the GT (`nodes`, `track_id`, `parent_track_id`,
`voxel_um`, `t0`), so a consumer needs this one kernel and not the 7.5 GB
crops.

T=100 for every crop. Peaks per crop 4,438 (`ZSNS004_z2y5x5`) to 81,493
(`ZSNS005_z1y6x3`); 0.5-5.7 MB each.

**Why this run existed:** `regionprops` on these masks IS HOCT's
`create_graph`, verified at 0.000e+00 (entry below). One GPU pass buys
unlimited CPU iteration on size and shape -- which is what the four
wasted HOCT GPU runs were paying for.

## 2026-08-25 -- morphology change is a REAL edge signal, not detection quality leaking

`morph_confound.py` -> `biohub-morph-confound`, CPU. The worry was that
"morphology separates true from false links" is really "morphology
separates real cells from spurious detections", since a false link's far
endpoint is usually a spurious detection. Test: split the negatives by
whether the far endpoint is itself a GT-matched cell, on Zebrahub's
DENSE ground truth so real-endpoint negatives actually exist.

| negative class | n | AUC of \|d intensity\| |
|---|---|---|
| **N-real** (both endpoints real cells, simply not linked) | 2,022,702 | **0.8298** |
| N-fake (far endpoint spurious) | 88,941 | 0.8640 |

**0.8298 on two million negatives whose both ends are real cells.** The
detection component exists and is worth 0.034. The signal is not it.

This is what the competition data could not answer: its GT is sparse, so
"a matched cell near another matched cell" is nearly always its own
successor -- the same construction that produced `morph_probe` step 2b's
3 negatives and every HOCT gate's "109 of 109".

### `morph_cheap`: which of these can be had without masks

`morph_cheap.py` -> `biohub-morph-cheap`, CPU, Spearman against the
verified mask-based `regionprops` (ranking is all a tree needs):

| mask feature | best box proxy | rho | usable |
|---|---|---|---|
| `intensity_mean` | crop mean | **0.965** | yes |
| `intensity_max` | crop max | **0.968** | yes |
| `intensity_std` | crop std | **0.946** | yes |
| `equivalent_diameter_area` | `n_half` | 0.713 | **NO** |

**A box does not know where the cell ends.** Every size proxy failed.
So intensity is free from the raw volume; SIZE needs the masks, which is
why `zh_masks` was worth 54 minutes of GPU.


## 2026-08-25 -- ⭐⭐ morphology is OURS, and real cells are 28% bigger than spurious ones

`morph_probe.py` -> `biohub-morph-probe` v2, 6 min on CPU. The user
objected to testing morphology the fast way -- computing it a NEW way
while asking a NEW question means a negative result cannot be attributed
-- so the run is gated: step 2 is SUPPRESSED unless step 1 proves our
extractor reproduces HOCT's node for node.

### Step 1: MATCH, and v1's mismatch was worth having

v2: all five columns agree at **0.000e+00** over 770/770 nodes paired by
centroid. **`skimage.regionprops` on the cached masks IS `create_graph`.**
hoct can be dropped entirely -- no 25 MB weights, no offline packaging.

v1 MISMATCHED, and the shape of it named the cause in one step:
`equivalent_diameter_area` agreed to 0.000e+00 while all four intensity
columns differed (worst 9.4e-01). Same masks, same 770 nodes, so it had
to be the intensity IMAGE, not the regions. It was: `create_graph`
defaults to `normalize_images=True` and we passed images `_norm` had
already normalised, so HOCT normalised twice. Ours read i_mean median
0.6807 against its 0.4948.

**That is the fourth inherited default this session that nobody had
checked** -- after "the metric needs a GPU" (really `load_image=True`),
the ranker's 18 transplanted features, and its 32-dataset pool. Without
the gate we would have carried ~40% wrong intensities into the
hypothesis and been unable to attribute the answer.

### Step 2a: the over-detection problem finally has a signal

127 of 770 detections matched GT; the other 643 are what the node-count
penalty pays for. Medians, with [p10, p90]:

| | matched | unmatched | ratio |
|---|---|---|---|
| `diam` | **8.83** [7.08, 10.22] | **6.87** [5.35, 8.02] | **1.28** |
| `i_mean` | **0.856** [0.413, 1.117] | **0.654** [0.305, 0.979] | **1.31** |
| `i_max` | 1.405 | 1.266 | 1.11 |
| `i_std` | 0.216 | 0.233 | 0.92 |

**Real cells are 28% bigger and 31% brighter than spurious detections.**
The distributions overlap -- this is not a clean cut -- but `N/N_true`
running to 1.30 has been on record for days with no tool able to touch
it, because every feature this project owns is a coordinate or a
detection score. Size and brightness are a new channel.

### Step 2b: UNUSABLE, my design error, same one as the HOCT gates

114 candidate pairs, 111 true -- **3 negatives**. The printed ratios (0.58
and 0.38) are computed on three samples and mean nothing.

Cause: the pairing required BOTH endpoints to be GT-matched. GT is
sparsely annotated, so within 15 um of a matched cell the only other
matched cell is nearly always its own successor. The negative class is
filtered out by construction -- **exactly what produced the "109 of 109"
in every HOCT gate**. Fix: draw negatives as "matched parent + any
candidate that is not its GT successor", without requiring the candidate
to be matched.

⚠ **1 division in this dataset's 30 frames.** "Does the mother shrink
before dividing" needs orders of magnitude more; size that run against
this number.

## 2026-08-25 -- HOCT: gate CLOSED, negative. Its FEATURE EXTRACTOR is the keeper.

`hoct_score.py` -> `biohub-hoct-score` v4, 7 min on CPU. Candidate graph
with scores, 364 edges whose endpoints both matched GT, 111 real
(30.5% positive) -- so unlike the four earlier attempts there are
negatives and the AUC means something.

| | AUC |
|---|---|
| HOCT `similarity`, blind (v2, no images) | 0.6673 |
| HOCT `similarity`, with images (v4) | **0.7012** |
| **negative distance, same edges** | **0.9091** |

**It loses to "closer is better" by 0.21.** Images helped 0.034 and did
not come close to closing it. There is no fine-tuning path -- the
training code is withheld -- so the line is closed.

The domain gap was the stated risk and it landed: `general_v0` was
trained on Cell Tracking Challenge plus a bacteria benchmark, not
light-sheet embryos. Bounds worth keeping: one dataset, 30 frames, and
our masks are WATERSHED ON A DETECTION HEATMAP, not real segmentation --
Bragantini's own caveat is that results depend on mask quality. Better
than the rank-24 competitor's uniform balls, still synthetic.

### ⭐ What to keep: create_graph is a morphology extractor

`create_graph` computes per-node features from the masks, on CPU, in
under a second per frame. All live and non-degenerate on our data:

| column | median | p10 | p90 |
|---|---|---|---|
| `equivalent_diameter_area` | 7.08 | 5.41 | 8.95 |
| `intensity_mean` | 0.4948 | 0.2347 | 0.7242 |
| `border_dist` | 0.4812 | 0.0000 | 0.6102 |
| `orphan_prob` | 0.0775 | 0.0455 | 0.2609 |

plus `inertia_tensor` and `intensity_min/max/std`.

**Every feature this project owns is a coordinate or a detection score.**
Cell SIZE, SHAPE and BRIGHTNESS have never been available to any model
here. `division_features.py` was written to ask whether they separate a
dividing cell and has no recorded result anywhere.

**This reopens what the division entry above parked.** That entry says
the remaining division routes "need pixels" and are therefore expensive,
because reading morphology meant re-reading raw zarr per candidate. It is
a CPU function call over masks we already know how to build -- the same
watershed, on the pooled heatmap, which is isotropic for free. The
"too expensive" filing was wrong and should be revisited before the
division line is called finished.

⚠ The masks cost one GPU pass and `biohub-hoct-gate` v4 has them cached
(123 KB npz). Everything downstream is free.

### ⚠ SUPERSEDED 2026-08-25 by the morphology line's close (user's question)

**The table above is not independent evidence that morphology is
measurable here.** `create_graph` computes its features FROM THE MASKS,
and the masks are ours -- watershed on a detection heatmap, as this same
entry says two paragraphs up. The 7.08 um came from the same
construction `biohub-mask-anatomy` later showed is 76.8%
background-bounded, i.e. mostly fringe. HOCT ran its code over our
numbers; it did not supply an outside measurement.

The reason the pretrained model can rely on morphology at all is that
`general_v0` was trained on Cell Tracking Challenge data, **which has
real segmentation ground truth**. We have none: Zebrahub's GT is points
and the competition's is tracks. That asymmetry is the whole story of
the morphology line, and it was knowable here.

**The warning was already in this entry and went unacted on:**
"Bragantini's own caveat is that results depend on mask quality."

The split it implies was also derivable in advance, and matches what
five runs on 2026-08-25 measured:

| `create_graph` column family | depends on the mask boundary | outcome |
|---|---|---|
| `equivalent_diameter_area`, `inertia_tensor` | entirely | DEAD -- unverifiable, no size annotation exists |
| `intensity_mean/min/max/std` | weakly | ALIVE -- AUC 0.866 separating hallucinated detections, but does not transfer Zebrahub -> competition |

**Rule this yields: a feature is only as trustworthy as the most
derived thing it depends on.** Before adopting a feature, ask what
annotation would falsify it. If none exists, it cannot be validated no
matter how good it looks.

## 2026-08-25 -- THE RANKER TRAINS ON 32 DATASETS. 187 ARE CACHED. +0.0073 LOCALLY, NOTHING ON THE LB.

> SUPERSEDED 2026-08-25 by the leaderboard. Submission 55757064
> scored **0.920** against the D32 baseline's 0.921, with hidden-test
> output within 0.07%. The +0.0073 below is real on the 12 `6bba`
> holdout datasets and absent on the hidden test. The entry stands as
> a record of what was measured; it is not a description of an
> improvement. See the LB entry at the top.

`poolsize.py` -> `biohub-poolsize`, 39 min on **CPU**. Both arms use the
same 18 columns, the same code, the same protocol and the same holdout.
The ONLY difference is how many datasets the edge ranker saw. Rows are
built once over all 187 and each arm selects its subset by which mount
the cache came from, so D32 is `ranker_fine`'s pool exactly, not a random
32.

| arm | datasets | pairs | edge AUC | score | adj_J | div_J | div TP/FP/FN |
|---|---|---|---|---|---|---|---|
| D32 | 32 | 39,792 | **0.9884** | 0.9313 | 0.9222 | 0.0909 | 3/21/9 |
| **D187** | **187** | **225,861** | 0.9864 | **0.9386** | **0.9254** | **0.1316** | **5/26/7** |

**adj_J +0.0032, score +0.0073.** Both anchors hold: D32's edge AUC is
**0.9884**, the exact figure `ranker_fine` recorded, and D32's adj_J
0.9222 sits 0.0002 from div-gate's 0.9220 with the saved pkl. So the
delta is the pool and nothing else.

### Where the unused data came from

`ranker_fine` learned on `biohub-edge-ranker`'s 44 caches minus the 12
holdout = 32. `biohub-cache-train` cached the other 155 on 2026-08-23 for
the DIVISION census, and its own header lists "the edge ranker's training
pool goes from 32 datasets to 187" as payoff #2. **Nobody ever spent it.**
Every ranker since has loaded the 32-dataset pkl.

**This is free. No new idea, no new dependency, no pixels.** It needs the
submission kernel to train the ranker on all 187 (or to load a pkl that
was), and biohub-cpu-probe2 already proved that whole line runs on CPU.

### Edge AUC anti-predicted the pipeline for the THIRD time today

D187's edge AUC is LOWER (0.9864 vs 0.9884) while its adj_J is higher.
Same inversion as scale-invariant features and as the division model's
four arms. **Stop reading edge AUC.** (Its own drop is partly mechanical
-- a bigger pool means a bigger, harder validation split -- but that is
exactly why it cannot be compared across configurations.)

### A worry that did not materialise

div_J ALSO improved, 0.0909 -> 0.1316, divTP 3 -> 5. The `ranker_fine`
header warned that a more confident 1:1 linker "steals second daughters
into wrong continuations, draining the orphan pool the division patch
mines". A ranker that is more CORRECT, rather than merely more
confident, feeds that mechanism instead of starving it.

⚠ The run ERRORed at the very end on two lines that still said
`res["ACC"]`, arm names left over from `accel.py` -- **after** printing
every number. Fixed. A summary line is not worth a 39-minute run.

## 2026-08-24 -- ⭐⭐⭐ div-forks: the division line is CLOSED. Three uses, all null.

`div_forks.py` -> `biohub-div-forks`, 38 min. div-gate showed the model
could not AUDIT forks; the objection was that pruning was the wrong shape
of intervention, since the model was measured on SELECTION over the full
candidate space. This kernel gave it selection. It still fails.

All three arms start from the same pre-patch graph and the same quotas.

| arm | forks added | score | adj_J | div_J | TP/FP/FN |
|---|---|---|---|---|---|
| BASELINE nearest-first | 442 | 0.9363 | 0.9220 | 0.1429 | 5/23/7 |
| RERANK by `p_div` | 442 | 0.9369 | 0.9222 | 0.1471 | 5/**22**/7 |
| GENERATE, filters off | 595 | 0.9342 | 0.9226 | 0.1163 | 5/**31**/7 |

BASELINE reproduces div-gate's prune-0% row exactly. RERANK buys **one**
fewer false division, +0.0006, an order of magnitude under the resolution
floor. GENERATE is worse. **TP is 5 in all three** -- no arm found a
division another missed.

Per fork added the FP rate is identical: 23/442, 22/442, 31/595 = 5.2%,
5.0%, 5.2%. The model is not separating anything.

### Why -- and it closes the whole thread

The filter stack cuts **14,159 loose proposals to 769**, an 18x
reduction. The quota then takes **442 of those 769**. So the RANKING step
chooses 442 from 769: a 1.7x selection with almost no leverage, and it
cannot decline to fill the quota no matter how bad the remaining
candidates are.

And the filters already contain the model's best feature in crude form:
`DIV_DIVERGE_UM >= 2.25 um` is one-frame divergence, the hand-set ancestor
of `sep_slope` (true 1.155 vs random 0.082 um/frame). **The 18x cut is
done by a rough version of exactly what the model learned, so by the time
the model is consulted the discriminating work is already finished.**

### CORRECTION, same day, after the user pushed back

The first draft of this entry said the division line was exhausted and
that the ranking step had "almost no leverage". **That was wrong, and the
arithmetic says so.** RERANK chooses 442 of 769, so it may EXCLUDE 327.
Only 23 forks are counted FPs. If those 23 sit inside the 769 and can go
without costing a TP, div_J runs 5/35 = 0.143 -> 5/12 = 0.417, worth
**+0.027**. The model removed **one of the 23**.

That is not an absent lever. That is a lever the model failed to pull.
(Discount it for replacements drawn from the remaining 327 being counted
FPs themselves -- the true ceiling is lower than +0.027, but nowhere near
zero.)

So the honest status is **the division MODEL is not good enough**, not
"divisions are unreachable". Two candidate causes, neither tested:

1. **Features.** What the model has that the filter stack does not is
   thin: the filters already encode one-frame divergence, which is the
   crude ancestor of `sep_slope`.
2. **Training data -- and this one is quantitative.** `biohub-zh-divisions`
   validated **468,617** Zebrahub divisions. div-train and div-kinematics
   train on **2001**, because only 41 crops were ever extracted
   (`zebrahub_extract.py`, 7.5 GB) and ~51 of each crop's divisions land
   on detections. `PER_CROP = 400` is NOT binding; the crop COUNT is.
   **We are training on 0.43% of the divisions we have already
   validated.**

Extracting more crops is a known, already-written path. Nobody has
measured whether the division model's AP scales with it.

What IS still established: the funnel's FN 7 (2 detection, 3 gate, 2
floor) and divTP never leaving 5 across every configuration tried.

⚠ Same caveat as every division row: five true divisions on the holdout.

## 2026-08-24 -- ⭐⭐⭐ the edge ranker never sees `edge_prob`, and "window" means two opposite things

Found while answering the user's question "doesn't linking already use
multiple frames -- didn't the edge detector use a window?".

### Two `window_size`s, opposite meanings. Do not confuse them again.

| where | what it is | status |
|---|---|---|
| `window_size = 2`, the official edge head | how many frames the MODEL SEES = two, the minimum. `competition-brief.md:179`: it "structurally **cannot use velocity**" | architectural, not a knob |
| `window_size = 25`, `solve_edges` | ILP time-chunking for SPEED | REVERTED; the global solve is what runs |

Multi-frame evidence has never entered linking. The reverted windowing was
about how the ILP is chunked, nothing more.

### ⭐ The transplant dropped the first-listed feature

`public-solutions-recon.md:300` describes yusuketogashi's ranker as "~25
engineered features -- **primary edge_prob**, raw vs motion-extrapolated
distance and their difference...". Ours has 18 and **`edge_prob` is not
among them**. Its `ps[i]` / `cs[j]` are DETECTION scores from `seeds_of`,
not the edge head's opinion of the edge.

The cause is which cache the ranker reads:

| cache | keys | read by |
|---|---|---|
| `cache_candidates.py` -> `{name}.npz` | coords, src, tgt, **prob**, dist | the old replay line |
| `cache_train.py` -> `detcache_*.npz` | `c{t}`, `s{t}` only | **ranker_fine, div_train, everything current** |

So the current best pipeline feeds the ILP a gradient-boosted tree over 18
hand-made geometric numbers and **discards the end-to-end-trained edge
head entirely** (`edge_weight = -EdgeAttr("edge_prob")` where edge_prob is
the ranker's output). That replacement was a net win -- 0.876 -> 0.921 --
but "geometry alone beats the edge head alone" is not "the edge head has
nothing to add".

**NOT free.** The cached probs in `{name}.npz` are indexed to the OFFICIAL
detector's nodes; the pipeline now runs xl detections, which are different
nodes. Using `edge_prob` as a ranker feature needs the official edge head
re-run over the xl node set -- a GPU inference pass, and the head needs
image features at node locations, so it is not a pure cache join. Size the
pass before committing to it.

## 2026-08-24 -- ⭐⭐ div-gate: the offline AP gain does NOT survive contact with the pipeline

`div_gate.py` -> `biohub-div-gate`, 53 min on a T4. Solve each holdout
dataset exactly as the recorded pipeline does (div_w 0.55, global), score
every resulting fork with the FULL division model from
biohub-div-kinematics, sweep a prune threshold, rescore.

**Anchor reproduces.** prune 0% gives TP/FP/FN 5/23/7, div_J 0.1429,
adj_J 0.9220, score 0.9363, against the recorded 5/26/7 / 0.1316 / 0.9214
/ 0.9345 at div_w 0.50 -- identical TP and FN, three fewer FP, which is
the direction a higher div_w should move. The wiring is correct.

| prune | forks cut | TP | FP | FN | div_J | adj_J | score |
|---|---|---|---|---|---|---|---|
| **0%** | 0 | **5** | 23 | 7 | **0.1429** | 0.9220 | **0.9363** |
| 25% | 138 | 4 | 22 | 8 | 0.1176 | 0.9222 | 0.9340 |
| 50% | 276 | 2 | 10 | 10 | 0.0909 | 0.9226 | 0.9317 |
| 75% | 414 | 1 | 4 | 11 | 0.0625 | 0.9230 | 0.9292 |
| 90% | 497 | 1 | 3 | 11 | 0.0667 | 0.9229 | 0.9296 |

**Monotone downhill. Doing nothing wins.** Pruning removes true divisions
at least as fast as false ones: the bottom 25% of forks by `p_div` holds
20% of the TPs but only 4% of the FPs -- *anti*-informative. By the 50%
cut it is 60% of TPs against 57% of FPs, i.e. indistinguishable from
cutting at random.

### Why the offline number did not transfer -- the lesson

div-kinematics measured AP 0.0812 and 1.8 FP per TP **over every candidate
triple `build()` emits** -- every neighbour pair of every parent, which is
overwhelmingly obvious non-divisions. This kernel scores **only the 553
forks the pipeline already chose**, which have passed the ILP's division
weight or the orphan patch's strict geometric filters. Those are the hard
cases; the easy negatives were gone before the model ever saw them.

**The offline gain was the model re-separating negatives the pipeline had
already excluded.** Measure a classifier on the population the
intervention will actually act on, not on the population it was trained
on. This is now the second time an offline win evaporated in the pipeline
(the first: scale-invariant edge features).

### What survives and what does not

- STILL TRUE: forward features carry real signal on the full candidate
  set. `sep_slope` 1.155 vs 0.082 is a measurement, not a model artefact.
- DEAD: `p_div` as a post-hoc filter on selected forks.
- DISCOURAGED, not disproven: re-ranking the orphan patch's *proposals*
  by `p_div` instead of by `d_pq + 0.15*d_s`. That acts on a larger pool
  than the accepted forks, so it is not strictly the same population --
  but the model showing zero separation on the accepted subset is weak
  evidence it would separate the pool either.
- Ceiling unchanged: FN 7 is 2 lost to detection, 3 outside `GATE_UM`, 2
  below `ILP_FLOOR`. Even perfect FP removal caps div_J at 5/12 = 0.417,
  worth +0.027. **That ceiling needs a model that can separate; this one
  cannot.**

⚠ **Five true divisions.** Every row rests on TP counts of 1-5, far under
anything resolvable alone. The monotone trend across five points is the
evidence here, not any single delta.

**This is more weight behind the conclusion already on record below: the
divisions are lost upstream, and the division line is close to exhausted.**

## 2026-08-24 -- ⭐⭐ division kinematics: the DAUGHTERS' future works, the mother's past does not

`div_kinematics.py` -> `biohub-div-kinematics` v2, ~23 min on a T4. One
feature build, four column views. Extends the diagnosis already on record
in the gate-round-3 entry below: "the model has no way to recognise the
rare configuration where a long edge IS a second daughter, because it
scores each edge independently and a division is a property of the
triple". The triple model was the first half; **time is the second.**

Trained on Zebrahub only (2001 pos), validated on held-out 6bba (83 pos),
same protocol as div-train v3.

| arm | cols | AUC | AP | FP per TP @ 10% recall |
|---|---|---|---|---|
| BASE (div_train's 25) | 25 | 0.8275 | 0.0486 | **6.9** |
| + mother's PAST | 35 | 0.8289 | 0.0375 | 8.8 |
| + daughters' FUTURE | 35 | 0.8365 | **0.0812** | **2.5** |
| FULL | 49 | 0.8198 | 0.0747 | **1.8** |

**The user's call, and it was right against mine.** The session had
front-loaded the backward half because it is cheap (before t the mother
is ONE cell, no chicken-and-egg). The user pushed back -- "what if the
daughters' behaviour is the more suspicious one" -- and the data agrees:
backward does NOTHING (AP gets *worse*, 0.0486 -> 0.0375), forward nearly
doubles AP and cuts the false-division price 2.8x.

**Read AP and FP/TP, never AUC, on this problem.** Positives are 0.25%.
AUC moves 0.83 -> 0.84 across every arm while AP doubles, and FULL has
the WORST AUC of the four yet the BEST FP/TP. An AUC-only report would
have picked the wrong arm.

### The physics, checked with no model in the way (medians, 6bba)

| column | true div | other | ratio |
|---|---|---|---|
| **fwd `sep_slope`** (um/frame) | 1.155 | 0.082 | **14.1** |
| fwd `sep_end/0` | 1.431 | 1.031 | 1.39 |
| fwd `score_min` | 0.396 | 0.289 | 1.37 |
| **coup `cent_resid`** (um) | 2.872 | 5.053 | **0.57** |
| back `decel` | 0.905 | 0.934 | 0.97 |
| back `v_last` | 2.254 | 1.861 | 1.21 |

- **Real daughters separate at 1.16 um/frame; random pairs at 0.08.** A
  14x gap, and the single cleanest division signal this project has
  found.
- **The centroid-momentum claim holds**: at a true division the pair's
  centroid lands 2.87 um off the mother's extrapolated position, against
  5.05 um for a spurious pair. `cent_resid` is also the top permutation
  column (+0.0058, 4x the next).
- **The mitotic slow-down does not exist in this data.** `decel` is 0.905
  vs 0.934 -- indistinguishable, and `v_last` runs the WRONG way (dividing
  mothers are slightly *faster*). Most likely mitotic rounding is short
  against the frame interval and simply is not sampled. That hypothesis
  is DISPROVEN, not untested.
- Two predictions that failed: `sep_min/0` ("two cells merely passing each
  other converge first") ratio 1.05 and NEGATIVE permutation importance;
  `surv_both`/`surv_gap` ("one member vanishes") are 5.0/0.0 in both
  classes at FWD_K=5 -- nothing vanishes, so survival discriminates
  nothing.

Low permutation importance here means REDUNDANT, not useless: `sep_slope`
scores +0.0002 despite the 14x raw separation, because `sep_end/0` carries
the same information inside the FULL model.

### Where this cashes out

The metric's marginal budget is ~7 FP per TP. **BASE sits at 6.9 -- right
on the edge, which is why divisions have never been worth taking.** FWD
buys them at 2.5 and FULL at 1.8, roughly 4x cheaper. Against the
forensics funnel below (12 GT divisions: 2 lost to detection, 3 outside
the gate, 2 below the floor, 5 taken), the 2 below-floor ones are the
direct target, and the 3 gate ones become reachable if the gate widens
AND the classifier can now hold them above the floor -- which is exactly
the combination round 3 lacked.

**Not yet done: this is offline AUC/AP, not div_J.** Wiring the FULL model
into the pipeline and scoring div_J on the holdout is the real test.

### ⚠ `hash(name)` makes the negative sampling irreproducible across runs

`build()` seeds its negative thinning with
`np.random.default_rng(hash(name) % 2**31)`. Python randomises `hash()`
per process unless `PYTHONHASHSEED` is set, so every run keeps a
different negative sample: this run got 49586 zebrahub / 46770 competition
triples where div-train v3 got 49735 / 46631, and BASE scored 0.8275 here
against v3's recorded 0.8083. **Arm-to-arm comparisons inside one run are
valid (all arms see identical rows); any cross-run AUC comparison in this
project carries ~0.02 of pure sampling noise.** Fix with a stable seed
(`zlib.crc32(name.encode())`).

## 2026-08-24 -- scale-invariant edge features: NO EFFECT, and the -0.73 is not in the ranker

`ranker_scale.py` -> `biohub-ranker-scale`, 8.6 min on a T4. One feature
build, three column views, so the arms can differ ONLY in which columns
the model is allowed to see. The 18 base columns are a transplant of
yusuketogashi's public 0.915 ranker (`public-solutions-recon.md`,
addendum 2026-08-22b) -- nobody here chose them.

| arm | cols | what |
|---|---|---|
| BASE | 18 | the transplant, unchanged |
| EXT | 28 | + local spacing, `d/sp`, `md/sp`, `gain/sp`, `xy/sp`, `z/sp`, `vel/sp`, and density at an adaptive 2x-spacing radius instead of the hardcoded 7 um |
| PURE | 17 | scale-free only: no absolute um, no fixed radius, not even the spacing itself |

Local scale = median distance to the 5 nearest neighbours. Verified
before the push: a 6 um lattice gives 6.0, the same lattice at half size
gives 3.0, coincident detections do not collapse it to zero.

**Every measurement says the feature set does not matter.**

| | BASE | EXT | PURE |
|---|---|---|---|
| AUC, `ranker_fine`'s own split | **0.9884** | 0.9886 | 0.9875 |
| corr(per-dataset AUC, density) | +0.053 | +0.021 | -0.083 |
| sparse-60% -> dense-40% AUC drop (bigger = worse) | 0.0213 | 0.0229 | 0.0258 |
| holdout linker-only adj_J | **0.9293** | 0.9286 | 0.9269 |
| corr(adj_J, density) | **-0.733** | -0.736 | -0.739 |

BASE reproduced the recorded 0.9884 **exactly**, so the reproduction is
faithful and the comparison is trustworthy. The adj_J spread across arms
is 0.0024, below the ~0.003 resolvable floor. Scale-invariant features
are if anything very slightly WORSE, and PURE is worst on every single
line -- dropping absolute micrometres costs a little and buys nothing.

### The useful part: where the -0.73 is NOT

`corr(adj_J, density) = -0.733` reproduces the recorded -0.68 closely, so
the penalty is real and it survives every reparametrisation tried here.
But `corr(AUC, density)` is **+0.05, i.e. flat**. The ranker scores edges
just as well when crowded.

The clearest single case is `6bba_3abfe10a`, 692 cells/frame, the densest
holdout dataset by 3x:

- AUC 0.9773 -- mid-pack, indistinguishable from the sparse datasets
- adj_J 0.7788 -- worst of the twelve by 0.15

**So the density penalty is downstream of edge scoring.** AUC ranks all
pairs; adj_J depends on the top-1 choice per cell. As candidates per cell
grow, a good ranking still leaves many near-ties for 1:1 Hungarian at a
fixed `p_min` to break. Detection is the other suspect and the more
likely one -- `corr(edge J, N/N_true) = -0.64` is already on record.

**Do not re-try this by adding more geometry to the ranker.** Two things
that follow instead:

- the assignment step rather than the score: candidate count per cell, or
  a density-dependent `p_min`
- detection under crowding, which adj_J is bounded by

### Why the reparametrisation was nearly a no-op

Density spans 15x across this train set (46 -> 692 cells/frame) but local
spacing spans only 2.1x (10.03 -> 21.23 um) -- as it must, since spacing
goes as density^(-1/3) and 15^(1/3) = 2.47. Dividing by a quantity that
moves 2.1x cannot undo a 15x change in the thing that actually matters.
The premise "every distance feature is absolute micrometres" was true;
the inference that this is what costs us was not.

Artefacts: `edge_ranker_{base,ext,pure}.pkl` in the kernel output. BASE is
equivalent to the shipped `edge_ranker.pkl`; the other two are not worth
keeping.

## 2026-08-24 -- windowing reverted; the seam hypothesis is disproven

Three LB points on the same pipeline, one variable:

| ref | solve | LB | total_bytes (hidden test) |
|---|---|---|---|
| 55707206 | global | **0.921** | 262.6 MB |
| 55709347 | 25-frame windows | 0.920 | 289.6 MB |
| 55716291 | windows + seam constraint | 0.920 | 289.3 MB |

The windowed runs emit **10% more output on the hidden test** while
emitting 2% LESS on the visible 4 -- the visible 4 do not reproduce the
behaviour, so they could not have caught this.

The seam fix propagated the previous window's REFUSALS across the shared
frame (the half of ultrack's scheme the first version omitted, since
sharing a frame stops edge sets colliding but does not stop window k+1
treating a node window k rejected as a fresh track start). It fired --
25 to 170 candidate edges cut per movie -- and changed **nothing**:
bytes 289.6 -> 289.3, score unchanged. **The hypothesis is disproven.**

Untested alternative worth trying if anyone revisits this: the ILP
charges `disappearance_weight` 1.5 at every track end, and under
windowing every track appears to END at each seam, so the penalty is
paid W times instead of once. That biases the solver toward linking
more aggressively, which fits +10% output.

**The decision, though, is simpler than the diagnosis.** Windowing
existed to buy margin against the 12 h cap. The global solve
COMPLETED on the hidden test at 0.921. The risk never materialised, so
the 0.001 windowing costs buys nothing. Reverted.

## 2026-08-24 -- div-train v1/v2 INVALID: a time compared against a row index

Two runs produced numbers that must not be used. The Zebrahub division
extraction was broken by:

```python
if kk not in first or tt < first[kk]:     # tt is a TIME
    first[kk] = r                          # first[kk] is a ROW INDEX
```

so "each track's earliest node" resolved to arbitrary rows and
parent-child pairing collapsed. Result: **26 of 41 crops reported no
division at all, and Zebrahub yielded 68 positives** where hundreds of
thousands were expected. Every AUC printed by those runs (0.2255,
0.2161, 0.4332, 0.3417, 0.3145) is meaningless -- and AUC below 0.5 was
itself the tell.

Two further design faults found at the same time:

- **4,632,084 competition triples for 101 positives.** A 0.002% positive
  rate; the model sees nothing. v3 keeps every positive and thins
  negatives to 1%.
- **The split was unusable.** Training on 44b6 alone gives **18
  positives**. v3 asks the question that actually matters -- can
  Zebrahub carry training by itself? -- and validates on 6bba (83
  positives), which enters no training set.

The one solid number from the run: **124 of the 151 GT divisions land on
detections** once all 199 datasets are cached, confirming the earlier
projection of ~127.

## 2026-08-24 -- the division data question, measured end to end

The linker is finished (forensics entry below): the solver already takes
every division the candidate set contains. Two rounds then tried to widen
the candidate set and both failed -- but not for the reason they looked
like they failed for.

### `GATE_UM`: two rounds, and the first one was confounded

Round 1 (`biohub-gate-sweep`) held div_w at 0.55 while widening the gate,
and divTP COLLAPSED 5 -> 2. That looked like the gate failing. It was
not: a wider gate floods training with distant negatives (positives
46.8% -> 21.2% -> 11.4%), which drags the whole probability scale down --
median probability fell **90x**, 0.1196 -> 0.0013. div_w and ILP_FLOOR
are absolute thresholds ON that scale, so holding them fixed made them
far stricter. **The gate was never tested.** Recorded as untested, not
disproven, precisely so nobody closes the line on 0.9301.

Round 2 (`biohub-gate2`) trained once per gate and swept div_w on the
cached probabilities. Now it is a fair test, and the gate still loses:
best at gate 14 is 0.9357 (floor 0.03, div_w 0.25) against 0.9382 at
gate 10. The funnel says why: gate 14 closes stage 2 (gate 3 -> 0) but
those divisions land in stage 3 (floor 2 -> 4) and REACHABLE only moves
5 -> 6. At floor 0.03 divTP does reach **6** -- one more than gate 10
ever managed -- but it costs **+31 division FPs for that one TP**,
against a budget of ~7 (the marginal rate the metric allows).

Round 3 (`biohub-zgate`) loosened Z ONLY, an ellipsoid instead of a
sphere, because the missed divisions were z-dominated: volume grows as
GZ/GYX = 1.4 instead of (14/10)^3 = 2.74. The isotropic control
reproduced 0.9382 exactly, so the implementation is sound. It still
fails, and the funnel is decisive: gate 3 -> 2 -> 1 as GZ rises, floor
2 -> 3 -> 4, **REACHABLE never leaves 5**. Widening the gate moves a
division from "outside gate" to "below floor". The ranker scores those
long-z edges low -- and it is RIGHT to: hengck23 verified independently
on the host baseline that "long links are almost wrong". The binding
constraint is not the gate, it is that the model has no way to recognise
the rare configuration where a long edge IS a second daughter, because
it scores each edge independently and a division is a property of the
TRIPLE (parent, d1, d2).

### Why it cannot learn that: 151 divisions, from 2 embryos

`biohub-div-census`: **151 GT divisions in the whole 199-dataset train
split**, 87 datasets have any, median 1, max 5. Only 31 sit in the 44
cached datasets; 26 of those are fully detected. Caching the rest yields
~127. A forum comment pins the structure: the training set is **2
embryos**, split **26 (44b6) + 125 (6bba)** -- which matches our census
exactly. So the data is not merely scarce, it has no independent
replication.

For scale: the same forum thread offers a free CC0 synthetic set with
**165,267 divisions**, and Zebrahub has far more.

### Zebrahub divisions are real: 468,617 of them

`biohub-zh-divisions` validated every candidate (daughters born on the
same frame, both surviving >= 3 frames, non-zero separation):

| movie | raw | validated | sisterSep | z share | angle |
|---|---|---|---|---|---|
| ZSNS001 | 635,041 | 358,314 (56%) | 7.37 | 0.84 | 138 |
| ZSNS001_tail | 36,878 | 36,246 (98%) | 6.87 | 0.78 | 149 |
| ZSNS003 | 15,868 | 12,719 (80%) | 6.03 | 0.67 | 148 |
| ZSNS004 | 30,278 | 24,866 (82%) | 7.78 | 0.65 | 150 |
| ZSNS005 | 46,026 | 36,472 (79%) | 7.60 | 0.82 | 154 |

**Fragmentation is ruled out**: across 764k candidates, every single
parent has exactly 2 children -- a re-numbered broken track would give
3 and 4 as well. ZSNS001's 2.9% rate was the original suspicion; it is
simply a longer, denser movie.

### Transfer, measured feature by feature (not argued)

`biohub-zshare` computed competition geometry from GT alone -- no
detector, so no survivorship -- over all 151 divisions, split by embryo:

| | n | sisterSep | z share | angle |
|---|---|---|---|---|
| 44b6 | 26 | 8.98 | 0.43 | 135 |
| 6bba | 125 | 11.47 | 0.55 | 140 |
| ALL | 151 | **10.57** | **0.53** | 138 |

**CORRECTION.** An earlier entry called sister separation a match
(Zebrahub 7.37 vs "competition 7.24"). That 7.24 was the synthetic
dataset author's calibration, not our measurement. Ours is **10.57 um**.
Zebrahub is ~30% SMALLER, not matching.

`biohub-density` then explained that gap, testing the user's hypothesis
that Zebrahub is simply denser. Normalising each separation by the local
cell spacing (median distance to the 5 nearest detections/nodes):

**sisterSep / local spacing: competition 0.68-0.85, Zebrahub
0.62-0.79 -- the distributions overlap heavily.**

`SUPERSEDED 2026-08-24`: an earlier version of this entry claimed the
two agree "to 2%" (0.72 vs 0.71). That number came from pairing the
competition's COUNT-weighted value against Zebrahub's
EQUAL-PER-MOVIE value -- the Zebrahub side was capped at 4000 divisions
per movie, so ZSNS001's 358k of the 469k total counted the same as
ZSNS003's 13k. Weighted consistently the two sides cross over:

| | count-weighted | equal per unit |
|---|---|---|
| competition (0.85 / 0.68) | 0.71 | 0.77 |
| Zebrahub (0.79/0.72/0.62/0.72/0.69) | 0.77 | 0.71 |

Either way the gap is 7-8%, not 2%. **The defensible claim is overlap,
not equality** -- and 7-8% still sits inside the competition's own
between-embryo spread (0.68 to 0.85 is 20%), so the conclusion that
Zebrahub is another embryo domain rather than another problem survives;
only its precision was overstated.

Note the per-movie cap is not itself a flaw: **how much of each movie
enters training is a free parameter**, so equal sampling is a legitimate
design. The error was using a training-design weighting to make a
DESCRIPTIVE claim about the domains. Consequence for the build: do not
try to derive the right mixture analytically -- **sweep it** (per-movie
sampling, and the Zebrahub:competition weight ratio, currently 3700:1)
and let AUC on the 127 in-domain divisions choose.

"Daughters are born roughly 0.7 nearest-neighbour distances apart" holds
as an order-of-magnitude invariant on both sides. The 30% raw gap was
scale, not geometry. Caveat: the WITHIN-dataset correlation of spacing
against separation is ~0 (+0.11, +0.04), so this is a between-dataset
scaling, not a per-division crowding force.

**A second CORRECTION**: an isotropic direction gives median
|dz|/||d|| = **0.5** exactly (|cos theta| is uniform on [0,1]); an
earlier entry said 0.58, which is the RMS. So competition at 0.53 is
essentially ISOTROPIC and Zebrahub at 0.65-0.84 is genuinely z-biased.

Per-axis normalisation was tried next and **did not** reconcile them,
against expectation:

| | packing sz/sxy | \|dz\|/sz | \|dxy\|/sxy |
|---|---|---|---|
| 44b6 | 0.55 | 0.32 | 0.61 |
| 6bba | 0.70 | 0.54 | 0.56 |
| competition | 0.64 | 0.51 | 0.56 |
| Zebrahub | 0.52 | **0.75** | **0.33** |

Packing anisotropy is comparable (0.64 is exactly the isotropic
expectation, since E|dz| = r/2 and E||dxy|| = pi*r/4). But the division
AXIS differs and survives normalisation. Note the competition's own two
embryos disagree -- 44b6 is xy-dominated, 6bba balanced, Zebrahub
z-dominated: **a feature that is unstable between the two training
embryos has no claim to stability on unseen ones**, and train/test are
embryo-disjoint by the rules.

### The variance decomposition that answers "is Zebrahub another world"

| quantity | between competition embryos | between Zebrahub embryos | across datasets |
|---|---|---|---|
| z share | 0.12 | 0.19 | 0.22 |
| sister separation | 2.5 um | 1.75 um | ~3 um (scale; normalises away) |
| daughter angle | 5 deg | 16 deg | ~5 deg |

Every cross-dataset gap is the size of an embryo-to-embryo gap. Zebrahub
is one more embryo domain, not a different problem -- and we are required
to generalise across embryos regardless.

**On the local-to-LB gap**, the user's reading was that we score better
on sparse data (true: `corr(edge J, density) = -0.68`, 0.968 at 50
cells/frame vs 0.771 at 519) so the hidden test must be denser. Partly:
the two families show -0.042 and -0.015 against the SAME hidden test, so
most of the larger gap is holdout memorisation by the official weights,
not density. The xl family's -0.015 is the part density can explain, and
it just predicted 0.921 correctly.

## 2026-08-23 -- SUBMITTED ref 55709347: windowed ILP. Better on both rulers, and it fits the clock.

`biohub-submit-w25` = ref 55707206 with a single change, the ILP solved
in 25-frame windows overlapping by one frame.

**Runtime, measured on the visible 4 (this was the open question):**

| dataset | pairs | feats | solve | total | was |
|---|---|---|---|---|---|
| 44b6_0113de3b | 55 581 | 3 s | 19 s | 111 s | 120 s |
| 44b6_0b24845f | 73 589 | 5 s | 26 s | 126 s | 145 s |
| 6bba_05b6850b | 8 711 | 1 s | 4 s | 88 s | 90 s |
| 6bba_05db0fb1 | 272 577 | 16 s | **100 s** | **222 s** | **485 s** |

The densest movie's solve fell **390 s -> 100 s (3.9x)**: windowing pays
most exactly where the ~n^1.5 blow-up hurts. Average 137 s/dataset
against 210 s, so **~7.7 h for ~200 volumes**, a 36% margin under the
12 h cap where the global solve projected 11.8 h. **The runtime line is
closed.**

**Three-way paired score on the visible 4** (`biohub-score-subs`, all
three finished CSVs against train GT -- movies that are not the
holdout, two of them from the `44b6` family the holdout lacks):

| | score | adj_J | div_J | div TP/FP/FN |
|---|---|---|---|---|
| xl 0.917 pipeline | 0.8678 | 0.8678 | 0.0000 | 0/5/3 |
| ranker, global ILP | 0.8855 | 0.8855 | 0.0000 | 0/11/3 |
| **ranker, window 25** | **0.8932** | **0.8932** | 0.0000 | 0/12/3 |

w25 - xl0917 = **+0.0254**; w25 - global = **+0.0077**. Windowing gains
MORE here than on the holdout (+0.0014), so it is not a holdout artefact.

Still carried: div_J is 0.0000 for all three -- those 4 movies hold 3 GT
divisions and nobody catches any. **Every visible-4 delta is pure
adj_J**; the division half of the holdout result remains single-source.

## 2026-08-23 -- rule 1 audit: four silent degradations found and given a voice

The user reaffirmed engineering rule 1. Audited this session's scripts;
four paths dropped data without saying so, all now counted and warned:

| where | what was silent |
|---|---|
| `solve_edges` | a window producing no edges left those frames UNLINKED |
| geff edge filter | `if a in rm and b in rm` dropped edges whose endpoint lost its node |
| `_emit` | self-loops, missing nodes and duplicates all `continue`d together |
| `score_subs` | CSV edges naming an absent node id -- **in the SCORER**, so a config could be judged on a silently low score |

The last is the dangerous one: a silent drop there biases a measurement
rather than an output, and could get a better configuration rejected.

`_emit` was given a COUNT ONLY -- its three drops are xl-repro's verbatim
behaviour that produced 0.917, so it now speaks without changing what it
does. Note the running `gate_sweep` and `score_subs` jobs were launched
BEFORE this patch; if their numbers look odd, suspect this first.

## 2026-08-23 -- ⭐⭐⭐ FORENSICS: the linker is finished. Every remaining division is lost upstream.

`biohub-div-forensics` walked all 12 holdout GT divisions down the funnel.

| stage | what stops it | count |
|---|---|---|
| 0 | parent undetected | 1 |
| 1 | a daughter undetected | 1 |
| 2 | **pair outside `GATE_UM` (10 um)** | **3** |
| 3 | probability below `ILP_FLOOR` (0.10) | 2 |
| 4 | reachable -- the solver's call | 5 |

**divTP at div_w 0.55 is 5, and stage 4 is 5. The solver takes ALL of
them.** FN 7 equals the seven unreachable ones exactly. That closes the
question the twelve-point sweep raised: division FN never fell below 6
because the linker had already extracted 100% of what the candidate set
contained. **No linker-side knob -- div_w, windowing, calibration --
can move div_J further.** Stop tuning it.

Per-division detail (parent match distance, and for stage 1 the nearest
detection of any score to the missed daughter):

```
6bba_7af54fde t=55 stage=0  pd=5.16
6bba_afb141ff t=48 stage=1  pd=1.15  daughter miss 5.47 um, nearest det
                                     5.47 um scoring 0.412
6bba_268e1230 t=24 stage=2  pd=2.12
6bba_2312ac41 t=10 stage=2  pd=3.52
6bba_7af54fde t=52 stage=2  pd=1.55
6bba_3abfe10a t=81 stage=3  pd=3.16  min pair prob 0.016
6bba_5c824876 t=2  stage=3  pd=0.33  min pair prob 0.080
6bba_2312ac41 t=19 stage=4  prob 0.393    6bba_afb141ff t=47  0.711
6bba_afb141ff t=32 stage=4  prob 0.215    6bba_afb141ff t=28  0.605
6bba_aeee7805 t=13 stage=4  prob 0.797
```

Note stage 1 is not really a detection failure: a detection scoring
0.412 (well over `UNET_THRESH` 0.15) sits 5.47 um away, just outside
`MATCH_UM` 5.0. It is a matching-radius near-miss.

### The lever: `GATE_UM`, worth roughly twice the whole ranker line

Stage 2 is the biggest actionable bucket -- 3 divisions where parent AND
both daughters are detected and the pair simply never enters the
candidate set. Sister separations read off the strips:

```
268e1230 t=24  11.55 um     2312ac41 t=10  16.05 um
3abfe10a t=81  11.98 um     5c824876 t=2   11.55 um
7af54fde t=52  12.90 um     7af54fde t=55  12.19 um
afb141ff t=48   7.85 um
```

Against a 10 um gate, and the strips repeatedly print `OUT OF SLAB` with
z jumping 8-10 planes: **divisions here separate along Z**, the coarse
axis at 1.625 um/plane (4x the xy pitch), so daughters gain physical
distance fast. The gate was sized for ordinary motion (the 6bba median
displacement p99 is 7.81 um) and divisions are not ordinary motion.

Arithmetic if raising the gate recovers all three: TP 5 -> 8, FN 7 -> 4,
div_J 5/35 = 0.1429 -> 8/35 = 0.2286, worth **+0.0086** on the composite
-- nearly twice everything the ranker line has bought (+0.0045).

**Do not just raise the constant.** `pair_features` includes `rank`
within the source's candidate list and both candidate counts; widening
the gate shifts those feature distributions, so **the ranker must be
retrained at each gate value** or its probabilities are read off a
distribution it never saw. Retraining costs 170 s.

Design for the next kernel: sweep `GATE_UM` 10/14/18 with a retrain at
each, and report the funnel, the composite, and the solve time (a wider
gate means more candidate pairs, so it buys divisions and pays in
runtime -- exactly the trade the 12 h cap cares about).

### Timing split: the solve is 88% of linking, features are 12%

| dataset | pairs | vel | feats | predict | solve |
|---|---|---|---|---|---|
| 6bba_268e1230 | 6 938 | 0.1 s | 0.4 s | 0.3 s | 4.3 s |
| 6bba_2312ac41 | 18 484 | 0.2 s | 1.0 s | 0.3 s | 11.2 s |
| 6bba_283bf9f1 | 39 027 | 0.5 s | 2.0 s | 0.4 s | 22.6 s |
| 6bba_3a1849c2 | 10 968 | 0.1 s | 0.6 s | 0.2 s | 5.9 s |

Feature side 6 s against 44 s of solve. **Windowing attacks the right
88%.** Projection for the hidden test: the submission run's 210 s/dataset
splits as detect 86 s + features ~15 s + solve ~109 s; at the measured
2.08x the solve drops to ~52 s, giving ~153 s/dataset and **~8.5 h for
~200 volumes** against the 12 h cap. Comfortable-ish, not generous, and
`biohub-submit-w25` will replace this estimate with a measurement.

## 2026-08-23 -- ⭐⭐ windowing the ILP is free speed AND a better score; calibration is dead

`biohub-ranker-wincal`, two independent lines in one kernel.

### W line -- window the ILP in time (uncalibrated, div_w 0.55)

| window | score | adj_J | div_J | ILP forks | link | speedup |
|---|---|---|---|---|---|---|
| global | 0.9368 | 0.9233 | 0.1351 | 111 | 326 s | 1.00x |
| 50 | 0.9368 | 0.9230 | 0.1389 | 111 | 213 s | 1.53x |
| **25** | **0.9382** | 0.9235 | **0.1471** | 108 | **157 s** | **2.08x** |
| 10 | 0.9343 | 0.9214 | 0.1290 | 88 | 126 s | 2.59x |

**Window 25 is both faster and better** (+0.0014 over the global solve at
2.08x). W=10 cuts too deep and loses real context. Likely mechanism for
the gain: with `disappearance_weight` 1.5 the global solve is pushed to
build very long tracks; a window bounds how far that pressure travels --
and it is div_J that improves (0.1351 -> 0.1471, division FP 25 -> 22),
i.e. fewer wrong long-range commitments.

Scheme (copy it, do not reinvent): window k covers frames [a, b] and
k+1 covers [b, c] -- they SHARE ONE FRAME but never share an EDGE, since
k owns a->a+1 .. (b-1)->b and k+1 owns b->b+1 .. So merged edge sets
cannot conflict and no reconciliation pass is needed. Same as ultrack's
`window_size` / `overlap_size` (their default overlap is 1).

**Trap paid for in advance:** mapping solved nodes back to global indices
must be done BY COORDINATE, never by `node_id` -- node_id is not a stable
handle across the solve (progress.md 02f7c8b).

Caveat on the control: `wglobal` reads 0.9368 where the fine sweep's
div_w 0.55 read 0.9363. Not noise -- `run_config` assembles nodes from
ALL seeds and lets `_short_filter`/`_emit` prune, where the B line built
them from the ILP's kept nodes. The window comparison is paired inside
this kernel, so it stands; do not mix these two numbers across kernels.

### C line -- probability calibration: HYPOTHESIS WRONG, do not retry

The theory was that the ranker's labels (BOTH GT children positive)
inflate second-daughter probabilities, and that div_w 0.55 is paying for
that bias. Isotonic regression fitted on the held-out 10%:

```
raw mean 0.4191 -> cal 0.4168;  frac>0.55  0.4214 -> 0.4118
```

**The probabilities were already calibrated.** Isotonic moved the mean by
0.0023. And the calibrated sweep is monotone increasing in div_w and
never beats uncalibrated -- 0.9258 / 0.9286 / 0.9311 / 0.9343 at div_w
0.05 / 0.15 / 0.30 / 0.55, against 0.9368 uncalibrated at 0.55. Isotonic
flattens probabilities into steps, creating ties the ILP then breaks
arbitrarily; that is the likely cost. **Line closed.** The ultrack
contrast (their `division_weight` default is -0.001) therefore needs a
different explanation -- most likely their candidates are
segmentation-hypothesis based and mutually exclusive by construction.

### ⭐ The structural finding, worth more than either line

Across ALL twelve sweep points -- 8 uncalibrated, 4 calibrated, forks
ranging 0 to 1847 -- **division FN never drops below 6**. divTP moves
only between 4 and 6 while division FP explodes.

The divisions we miss are **not in the candidate set at all**; the solver
is not refusing them. That kills the "7 FPs per TP is a bargain" line of
attack (user's marginal analysis, which is arithmetically right): the
exchange rate is fine, there is simply nothing to buy. Anything that
moves div_J has to act UPSTREAM -- the candidate gate `GATE_UM`, the
detection threshold (a just-divided second daughter is small and dim and
plausibly sits under `UNET_THRESH` 0.15), or a purpose-built division
detector. `biohub-div-forensics` walks every GT division down the funnel
to find which stage loses them.

### Metric note that makes over-forking cheap (user's observation, verified)

`reference.md#the-metric`: a fork lying entirely among unannotated cells
enters no FP set. Measured from the div_w 0.55 -> 0.20 step: 913 extra
forks cost only -0.0027 adj_J, which at ~10.7k holdout edges is roughly
**30 visible edge FPs -- about 97% of the extra forks are invisible to
the metric**. But the div_J denominator is the real bill: the same step
cost -0.0056 there, twice the edge damage. So forks are nearly free on
edges and not free on divisions. (adj_J is a weighted per-dataset average
with a node-count penalty, so treat the 30 as an order of magnitude.)

## 2026-08-23 -- SUBMITTED ref 55707206: ranker + capacity-2 ILP linker

Kernel `biohub-submit-ranker` v2. Single variable against LB 0.917
(ref 55690436): the linker. Detection side and post-link patches are
unchanged; the Hungarian is replaced by ranker probabilities feeding a
capacity-2 tracksdata ILP at division weight 0.55.

**Independent confirmation before submitting.** `biohub-score-subs`
scored BOTH finished CSVs against train GT on the same visible 4 -- a
paired comparison on movies that are not the holdout, two of them from
the `44b6` family the holdout does not contain:

| | score | adj_J | div_J | div TP/FP/FN |
|---|---|---|---|---|
| xl 0.917 pipeline | 0.8678 | 0.8678 | 0.0000 | 0/5/3 |
| **ranker + cap-2 ILP** | **0.8855** | **0.8855** | 0.0000 | 0/11/3 |
| delta | **+0.0178** | +0.0178 | 0 | |

Four times the holdout delta, on different movies -- different data, so
the magnitudes are not comparable, but the sign is confirmed away from
the holdout, which is the thing the family-calibration rule says to
worry about.

**Limit on that evidence, carried deliberately:** div_J is 0.0000 for
BOTH pipelines here. Those 4 movies hold only 3 GT divisions and neither
pipeline caught any (ranker 11 division FPs vs xl's 5, but with 0 TP the
division term is 0 either way). So **the +0.0178 is entirely adj_J**;
the division half of the holdout result (div_J 0.1429 vs 0.1333) has NOT
been reproduced outside the holdout and remains single-source evidence.

The scoring kernel then died on a cosmetic per-dataset print
(`per_sample_metrics` returns no "score" key -- that appears only after
`summarise`); every number above had already been emitted. Not worth a
re-run, but fix `score_subs.py` before reusing it.

## 2026-08-23 -- the submission kernel runs, and the ILP does not fit the 12 h cap

`biohub-submit-ranker` v2 COMPLETE: offline env works (pack wheels,
tracksdata, `predict_unet_transformer` from the pack, ranker pkl via
kernel source), all three xl models load, `submission.csv` written with
the right schema. v1 had died on `edge_ranker.pkl` holding
`{"model": clf, "version": ...}` rather than a bare model -- the load
now unwraps and asserts `predict_proba` at load time, so that class of
error surfaces in 40 s instead of after a 10 min model load.

Counts against the 0.917 pipeline on the same visible 4 (a real paired
check that the graft is not producing well-formed garbage):

| dataset | xl0917 nodes/edges | ranker nodes/edges | ILP forks | runtime xl -> ranker |
|---|---|---|---|---|
| 44b6_0113de3b | 23878 / 23152 | 23794 / 23048 | 5 | 82 s -> 120 s |
| 44b6_0b24845f | 28147 / 26186 | 27599 / 25862 | 20 | 87 s -> 145 s |
| 6bba_05b6850b | 6135 / 5906 | 6090 / 5894 | 7 | 74 s -> 90 s |
| 6bba_05db0fb1 | 68259 / 65947 | 66813 / 64421 | 8 | 102 s -> **485 s** |

Nodes -1.7%, edges -1.6% -- the small drop the ILP's node-dropping
predicts. Forks per dataset match the holdout's 111/12.

### ⚠ THE RUNTIME PROBLEM (read before touching the submission kernel)

The hidden test is **~199-200 volumes** (`competition-brief.md:189`) and
the cap is 12 h. Measured here: **210 s/dataset average** -> 200 x 210 s
= **11.7 h + 10.6 min setup**, i.e. flush against the cap. The 0.917
pipeline runs 86 s/dataset (~5 h), so the whole excess is the linker.

Worse, the linking cost is **superlinear in candidate pairs, ~n^1.5**:

| pairs | features + ILP |
|---|---|
| 8.7k | ~16 s |
| 55.6k | ~38 s |
| 73.6k | ~58 s |
| **272.6k** | **~383 s** |

3.7x the pairs costs 6.6x the time. `6bba_05db0fb1` is not even the
worst case the train set contains -- the densest 6bba movie runs 519
cells/frame. A handful of denser hidden volumes blows the budget.

Fixes, in the order they should be tried:

1. **Window the ILP in time.** One global solve costs ~n^1.5; W windows
   (overlapping by a frame so tracks are not cut) cost n^1.5 / sqrt(W),
   ~3.2x faster at W=10, and turn the superlinearity into near-linear so
   dense movies stop exploding. Needs a holdout re-validation (~30 min).
2. **Raise `ILP_FLOOR`** (0.10). At AUC 0.9884 true edges score high, so
   0.25-0.30 should cost nothing and cuts solver edges. Note it does NOT
   cut feature cost -- features are built for every in-gate pair and only
   then filtered by probability.
3. **A wall-clock guard, needed regardless.** Track elapsed time and
   remaining datasets; when the projection exceeds the cap, fall the rest
   back to the Hungarian linker that is already in the file. That floors
   the result at baseline behaviour and guarantees a valid CSV. Rule 1:
   print how many datasets fell back and from which one.

**User's call (2026-08-23), overriding the recommendation to fix first:**
submit anyway. Their reasoning, which is sound: a timeout does not burn a
submission slot, and slots are not the scarce resource -- producing one
real improvement takes days, so the account rarely spends even one slot a
day. 11.8 h against a 12 h cap may simply pass. The optimisation work
above stays queued either way.

## 2026-08-23 -- ⭐⭐⭐ div_w 0.55: 0.9363, and it beats the baseline on BOTH terms

`biohub-ranker-fine` filled the 0.20-0.75 gap the diag left. Eight points
now, one clean unimodal curve, argmax 0.55.

| div_w | ILP forks | score | adj_J | div_J | div TP/FP/FN |
|---|---|---|---|---|---|
| 0.20 | 1024 | 0.9280 | 0.9193 | 0.0870 | 6/57/6 |
| 0.35 | 394 | 0.9309 | 0.9211 | 0.0980 | 5/39/7 |
| 0.45 | 190 | 0.9344 | 0.9216 | 0.1282 | 5/27/7 |
| 0.50 | 148 | 0.9345 | 0.9214 | 0.1316 | 5/26/7 |
| **0.55** | 111 | **0.9363** | **0.9220** | **0.1429** | 5/23/7 |
| 0.65 | 49 | 0.9351 | 0.9222 | 0.1290 | 4/19/8 |
| 0.75 | 12 | 0.9331 | 0.9224 | 0.1071 | 3/16/9 |
| 1.50 | 0 | 0.9331 | 0.9224 | 0.1071 | 3/16/9 |

**The win is not a trade.** At 0.55 both terms beat the xl baseline
(adj 0.9185 / div 0.1333): adj_J +0.0035 AND div_J +0.0096. Divisions
end up BETTER than the pipeline whose divisions the ranker was accused
of draining -- the capacity-2 solve more than repays what 1:1 stole.

Caveat to carry: the peak sits 0.0018 above its neighbours on SMALL
COUNTS -- div_J moves on FP 27/23/19 and TP 5/5/4 against 12 GT
divisions. The computation is deterministic (identical AUC 0.9884 in
all three training runs), so this is not run-to-run noise, but the
argmax is not sharply resolved: 0.45-0.65 are all defensible and 0.55
is simply the best measured. Predicted LB at the -0.015 family offset:
~0.921 vs the standing 0.917.

**Submission kernel built: `biohub-submit-ranker`** (`submit_ranker.py`).
Detection side is xl-repro verbatim; the linker is ranker probs -> the
capacity-2 ILP at div_w 0.55; every post-link patch is ranker-diag's,
because those -- not xl-repro's -- are the code path that measured
0.9363 (they differ: xl's gap-close sorts proposals by distance and
takes the closest under the cap, ranker's takes them in Hungarian
order). Verified before pushing:

- **Detector parity, the one that could have killed it.** The ranker
  trained on `biohub-edge-ranker`'s caches, which mount xiaoleilian's
  two weight datasets -- the same pair xl-repro uses -- with identical
  branches, flip views and POOL/CAND_THR/UNET_THRESH/NMS_UM/SCALE. The
  ranker learned on the xl detector's own coordinates, so the
  distribution shift that buried the chimera does not apply.
- **Offline.** Internet is off at submission; tracksdata/pyscipopt come
  from the support-pack wheels and `predict_unet_transformer` from the
  pack's repo copy. No clone anywhere.
- **An AST pass for undefined names caught two real graft bugs** that
  compile fine and die at runtime: `class UNet3D` and `_dist_um` were
  missing, because the block extractor took `def` and not `class`. Run
  that check on any assembled script.
- `submission.csv` is written BEFORE scoring, and scoring is wrapped
  whole: a scoring failure must never cost a submission. On the hidden
  rescore the train-GT lookup misses and scoring says so and skips.

## 2026-08-23 -- ⭐⭐ diag: the knob was live, the window was dead, and the reconciliation works

`biohub-ranker-diag` counted out-degree-2 nodes in the SOLVED graph
before any patch touched it, over a widened div_w sweep. This is the
measurement v4 could not produce.

| div_w | ILP forks | final forks | patch added | score | adj_J | div_J | div TP/FP/FN |
|---|---|---|---|---|---|---|---|
| 0.20 | **1024** | 1423 | 399 | 0.9280 | 0.9193 | 0.0870 | 6/57/6 |
| **0.50** | **148** | 590 | 442 | **0.9345** | 0.9214 | **0.1316** | 5/26/7 |
| 0.75 | 12 | 461 | 449 | 0.9331 | 0.9224 | 0.1071 | 3/16/9 |
| 1.50 | **0** | 451 | 451 | 0.9331 | 0.9224 | 0.1071 | 3/16/9 |

Fork count runs 1024 -> 148 -> 12 -> 0, monotone across a huge range:
`division_weight` was always wired correctly. **v4's 0.75-0.95 window
sat entirely in the dead zone** (12 forks at 0.75, zero above), so its
three "identical" rows were one solved graph three times. The
byte-identity was a badly placed window, not a broken knob -- worth
recording, because the pessimistic reading (mechanism never engaged)
was wrong and would have shelved a live line.

**The reconciliation is confirmed.** At div_w 0.50 the ILP natively
takes 148 forks, div_J returns to near-baseline (0.1316 vs 0.1333) and
finds one MORE true division than baseline (5 TP vs 4), while adj_J
holds at 0.9214 (+0.0029 over baseline). Taking both daughters
natively is exactly what stops the stealing, as diagnosed.

Clean decomposition of the +0.0027 over baseline, two separate sources:

- **+0.0013 is the ranker's edges alone** -- at div_w 1.50 the ILP forks
  zero times, divisions come purely from `_add_safe_divisions`, and the
  pipeline still scores 0.9331 vs baseline 0.9318.
- **+0.0014 more is native forking**, 0.9331 -> 0.9345 at div_w 0.50.

0.20 over-forks: most TPs (6) but division FP 57 and adj_J falls to
0.9193. The curve is peaked between 0.20 and 0.75 with only those two
brackets measured, hence `biohub-ranker-fine` (0.35/0.45/0.55/0.65)
before the submission slot is spent.

Kernels this round, all in the repo: `ranker_v4.py`, `ranker_diag.py`,
`ranker_fine.py`. The diag/fine pair skips the A line via
`P_MIN_SWEEP = ()` (measured twice, identical).

## 2026-08-23 -- ranker v4: the gate is cleared by 0.0001, and the knob never moved

Two runs. The first ERRORed at 8.5 min: `P` was bound to the pipeline
module at import, then clobbered inside the module-level `p_min` loop by
`pairs, rows, P, C = feats[t]`, so the B line's `P.build_graph` hit a
numpy array. Pure name shadowing; the import is now `as PIPE`. Also
hardened: the B line mapped ILP endpoints with `remap[int(a)]`, which
KeyErrors on the `edge_endpoint_missing` case `link_replay` has warned
about since 2026-08-20 -- now dropped loudly as a lower bound (rule 1).
The script lived only in a dead session's scratchpad and on Kaggle; it
is now `ranker_v4.py` in the repo.

| config | score | adj_J | div_J | div TP/FP/FN |
|---|---|---|---|---|
| xl baseline | 0.9318 | 0.9185 | **0.1333** | 4/18/8 |
| A line, p_min 0.30 | 0.9305 | 0.9222 | 0.0833 | -- |
| A line, p_min 0.35 | 0.9284 | 0.9200 | 0.0833 | -- |
| A line, p_min 0.40 | 0.9278 | 0.9195 | 0.0833 | -- |
| **B line, div_w 0.75** | **0.9331** | 0.9224 | 0.1071 | 3/16/9 |
| **B line, div_w 0.85** | **0.9331** | 0.9224 | 0.1071 | 3/16/9 |
| **B line, div_w 0.95** | **0.9331** | 0.9224 | 0.1071 | 3/16/9 |

Composite is exactly `adj_J + 0.1 * div_J` (verified on all seven rows).
That decomposes the A line's failure precisely: the ranker buys
**+0.0037 adj_J** and pays **-0.050 div_J = -0.0050**, net -0.0013. The
daughter-stealing diagnosis is confirmed as a mechanism -- and div_J
sits at 0.0833 for every p_min, so no threshold reaches it.

**The B line's result does not yet mean what it looks like.** Three
surcharges spanning 0.75-0.95 returned identical score, adj_J, div_J
AND TP/FP/FN. Identical to every digit means one solved graph, i.e.
`division_weight` had no effect. With `edge_weight = -1.0 *
edge_prob`, a fork is taken iff the second daughter's prob exceeds
div_w, so either every accepted fork scored > 0.95, or **no fork was
accepted at all** and all 3 TPs still come from the post-hoc
`_add_safe_divisions` patch. Supporting the latter: B is still BELOW
baseline on divisions (0.1071 vs 0.1333, 3 TP vs 4). It recovered part
of what the A line lost, not all. So the +0.0013 over baseline reads as
"the edge gain finally is not fully cancelled", not as "capacity-2
takes both daughters natively" -- which was the entire hypothesis.

Predicted LB at the -0.015 family offset: ~0.918 vs the standing 0.917.
Inside the noise. **The pre-registered rule is technically cleared and
substantively unsettled**; a submission slot is the scarce resource,
GPU quota is not, so `biohub-ranker-diag` counts forks in the solved
graph before any patch, over a div_w sweep widened to 0.2-1.5.

## Current state (2026-08-21)

### 2026-08-22 VERDICT: from-scratch retraining is closed; main line = post-processing + model swap/ensemble

Session 1 (4 full epochs, clean split, official recipe, 10.3 h) delivered
the learning curve: eval score (acc x recall on the 12-holdout) =
0.8980 / 0.9000 / 0.9157 / 0.9121 — one epoch buys 0.90, then the curve
crawls (+0.016 over three epochs, last epoch negative). Target: the
public weights' 0.9835 at epoch 402. Real epoch cost: **2.53 h** (not
2.1; smoke extrapolation was optimistic). 50 epochs = 127 h. The
400-epoch tail is real; parity from scratch is unaffordable. Appeal
path (not scheduled): a 3-epoch LR-schedule probe (3e-4 + cosine) —
constant lr 1e-4 may explain some of the crawl. Warm-start surgery
stays rejected (user: twisted melons; 0.764 as evidence). **Main line
is now B+C: the learned edge re-ranker + luxury inference (TTA,
ensemble, sub-voxel refinement) + adopting/ensembling the public
xiaoleilian detector.** Artifacts kept: session-1 checkpoint (clean
split, eval 0.9157) ferried to dataset `faustine649/biohub-train-ckpt`
— an ensemble-diversity candidate; the resumable trainer harness works
(4 epochs, per-epoch full checkpoints, zero incidents).

### 2026-08-22 post-processing round (details in entries below when logged)

- **Orphan patch v2 (0.917-derived strict filters: mutual-nearest sister,
  both children continue, divergence >= 2.25 um): local +0.0132** —
  v2 cap15 0.9401 vs v1 cap15 0.9348 (reproduced) vs cap0 0.9269, div
  TP/FP 2/13 -> 3/10. Higher caps degrade. Submission v2 in validation
  (`biohub-submit-orphan` v2); v1's local delta transferred 1:1 to LB.
- **Velocity-into-cost: line closed.** First-ever sweep of link_replay's
  velocity_adjust (24-cell grid, baseline 0.9387 reproduced): velocity
  only gains where disappearance is low (max +0.0041 at 0.35/2.5/0.25)
  and no cell beats the incumbent 0.40/3.5/0.0 — it substitutes for high
  disappearance cost rather than adding to it, and costs TP. Velocity
  survives only as a FEATURE for the learned edge re-ranker (the 0.915
  anatomy, recon addendum 2026-08-22b), which is the next post-processing
  build.
- Kernel-engineering trap for the toolchain file: importing numpy BEFORE
  a pip block that resolves dependencies can half-upgrade numpy on disk
  ("cannot import name '_center'") — do heavy imports after pip.

### 2026-08-22 Branch C (dual track) status

- **XL detector beats the official on the clean ruler**: recall 0.766 /
  prec 0.736 vs 0.650 / 0.637; dim-quartile 0.73 vs 0.52 (both models
  ZH-naive — a pure generalization duel). Its score distribution is
  compressed; 0.15 is its calibrated operating point.
- **Path 2 (verbatim 0.917 reproduction)**: validated on the visible 4
  (patches active, ~86 s/dataset) and SUBMITTED as ref 55690436 —
  expected ~0.917; also tests badge reproducibility.
- **Path 1 (chimera: XL detect -> official edge head -> our ILP + v2
  patch)**: v1 died of a double-downsample clamp (open_dataset returns
  already-downsampled shapes); v2 ran clean but LOST locally — adj_J
  0.9118 vs 0.9337, composite 0.9204 vs 0.9401. Caveat: the holdout is
  the official detector's HOME turf (possibly its training data), i.e.
  this filter is structurally biased against foreign detectors, plus the
  edge head faces a coordinate-distribution shift. Decision rule: if
  55690436 lands ~0.917 on the LB (an xl-family pipeline winning away
  from home), the chimera earns an LB slot despite the local reading;
  if the reproduction also flops, the xl family is suspect and the
  chimera is buried.

**Measurement doctrine (user's, adopted):** the only ruler is the LB —
strictly the private 71%, of which the public LB is the closest readable
proxy (+-0.03, 29%). Every local score (Zebrahub bench, holdout, val
losses) is a FILTER, not a ruler: it may only decide what earns a
submission slot, never declare improvement. "Better" is a word the LB
says.

**LB history (from the submissions API):** 0.681 (08-17, classical) ->
0.876 (08-19, ILP+A) -> 0.881 (08-20, thr 0.35 / disapp 2.5) -> 0.764
(08-21, fine-tune — REJECTED) -> 0.889 -> 0.898 (08-22, orphan patch
v1 -> v2) -> **0.917 (08-23, ref 55690436: verbatim reproduction of the
kunaldesale/xiaoleilian pipeline — the current best, badge reproduced
digit-for-digit)**. Gap to LB top-20 (0.935): 0.018.

**FAMILY-CALIBRATION DISCOVERY (2026-08-23):** local-to-LB offsets are
family-specific — official-weights family: local 0.9401 -> LB 0.898
(-0.042); xl family: local 0.9318 -> LB 0.917 (-0.015). Local rankings
INVERT on the LB. Hardest evidence yet that the holdout inflates the
official weights (memorization). Rule: absolute local scores must never
be compared across detector families; only within-family paired deltas
are valid. **Chimera buried** by this rule: within-family it trails the
repro by 0.011 (0.9204 vs 0.9318), predicted LB ~0.906 < 0.917 — the
official edge head is the weak link for foreign coordinates, no slot
spent. Main line: evolve the 0.917 xl pipeline with post-processing
upgrades, each a within-family paired delta against holdout baseline
0.9318 (its div line: div_J 0.1333, TP/FP/FN 4/18/8).

- **LB 0.876** (submitted 2026-08-19: official pipeline + pilkwang weights +
  `--use-ilp --ilp-appearance-weight 0.0 --ilp-disappearance-weight 1.5`,
  `det-threshold 0.99`). Public 0.916 notebook is the bar; its badge is
  pre-patch, so the honest gap is unknown but of order +0.04.
- **Local best on the 12-dataset `6bba` holdout: 0.9387** (cached replay,
  edge threshold 0.40, disappearance 3.5) vs 0.9269 for the submitted
  config. Local is optimistic vs LB by ~0.05; use it for paired deltas only.
- **Converged levers:** edge threshold x disappearance (done), det-threshold
  (done — not a lever; the detection ceiling is the model). **The open line
  is a better detector, trained on Zebrahub.**
- **Zebrahub crops extracted: 41 crops, 7.5 GB**, all five movies, in
  `biohub-zebrahub-extract` v4 output under `crops/` (entry below).
- **Clean detection ruler built and read**: public detector = recall 0.65 /
  prec 0.64 on Zebrahub dense GT (entry below). All training reads there.
- **Closed today:** sigmoid edge activation (best 0.9314 vs softmax's
  0.9387 on the holdout — no reason to pursue); NMS-merge division
  hypothesis (sisters are born ~10.6 um apart; entry below).
- **Fine-tune v3 trained** (augmented, two-domain): ZH val loss 1.307 ->
  0.194, but competition-domain val loss 0.018 -> 0.047 — the drift guard
  fired. Weights saved; verdict pends the clean-ruler re-bench.
- **Orphan forks v3: div_J moved for the first time** — cap 15 -> 0.080
  (2 TP / 13 FP). adj_J came out NaN (n_total read at the wrong nesting;
  it lives at gt["extra"]), so the net edge-cost verdict pends the
  rescore.
- **Fine-tuned weights passed the filter**: ZH bench recall@0.99
  0.650 -> 0.726, precision 0.637 -> 0.729 (FP -20.7k), dim-quartile
  recall 0.52 -> 0.71. Caveat: the fine-tune TRAINED on these crops, so
  the ZH ruler is no longer clean for this model — it only certifies
  "learned, not broken". The real verdict is the LB.
- **Fine-tune v3 REJECTED by the LB: 0.764** (ref 55670839, -0.112 vs
  0.876, single-variable). The comp-domain drift warning was real and
  decisive; every local gain (ZH recall +0.075) was paid for out of the
  test domain. Next fine-tune iteration needs a stronger anchor: higher
  comp fraction / lower LR / fewer steps, or post-hoc weight
  interpolation toward the original — and the LB stays the judge.
- **Orphan forks: NET +0.0080 at cap 15** (official metric, holdout):
  cap0 reproduces 0.9269 digit-for-digit (harness validated); cap 15 ->
  0.9348, div_J 0 -> 0.080, and adj_J pays only 0.0001 — the added edges
  are effectively free, as constructed. cap 40 is worse (0.9332).
- **Campaign session 1 LAUNCHED** (`biohub-train-campaign` v1): official
  recipe, clean split, 4 epochs (~8.5 h), resumable (full checkpoint per
  epoch; ferry `last_full.pth` via a ckpt dataset between sessions).
  Measured cost: **~2.1 h/epoch on T4 x2 at any batch size** (GPU-bound,
  ~2.4 windows/s; bs 8 hit a NaN anomaly — avoided; expandable_segments
  cures the bs-16 OOM). Budget: 30 h/week, full tank, ~180 h to deadline;
  no Kaggle paid tier (the one upgrade path: Colab Pro unlocks extra
  Kaggle GPU hours). CLI trap logged in kaggle-toolchain.md: access
  tokens expire and the CLI won't auto-refresh — inject
  `KAGGLE_API_TOKEN` from `kaggle auth print-access-token` per call.
- **De-risking doctrine for expensive training (user's concern: no
  one-shot 100 h bets):** (1) session 1 buys the LEARNING CURVE — how
  many epochs matter; (2) design variants go through a SCREENING round
  first (max_iters-sampled epochs, ~3 h/variant, ranked on ZH bench +
  holdout); (3) loss/data/augment changes branch WARM-STARTED from a
  shared trunk checkpoint — only architecture changes force from-scratch;
  (4) exactly one full-scale run, of the screened composite. AMP (absent
  from the official script) to be validated before screening — likely
  halves every number above. (An earlier draft here cited "pilkwang's
  50ep pack proves a 50-epoch Kaggle run" — struck; see the epoch-402
  correction below.)
- **STRATEGIC PIVOT (user's call, 2026-08-22): the training campaign is
  the main line.** Knob-tuning is provably exhausted (three days bought
  +0.013; the 0.952 author's sweep log is flat), the +0.046 gap to the
  top 20 lives in the edge term, which is model-bound. Plan: (0) smoke-run
  the official recipe for real epoch timing + first clean-split checkpoint
  — `biohub-train-smoke` in flight; (1) short reproduce on competition
  data, our split; (2) two-domain FROM-SCRATCH train (no catastrophic-
  forgetting frame): package Zebrahub crops as competition-format
  zarr+geff fakes (must carry `image_statistics.quantiles` in attrs — the
  loader raises without them) and add them to a custom splits file — zero
  changes to official training code; (3) train the edge head on
  Zebrahub's 20M edges. **CORRECTION (user's, 2026-08-22): the working
  public weights carry `epoch: 402` in their checkpoint — they are a
  ~400-epoch product, NOT 50; the pack's "50ep" name proves nothing about
  the edge predictor's training length.** At our measured 2.1 h/epoch,
  400 epochs ≈ 840 h — from-scratch parity may be unaffordable. Budget
  consequences: (a) the session-1 learning curve decides whether
  from-scratch is live at all; (b) the default trunk for improvements
  becomes CONTINUED training from the public weights (original loss,
  original data plus additions, low LR — unlike the rejected fine-tune,
  which changed loss weights and fed 70% foreign data); (c) AMP and
  max_iters sampling graduate from nice-to-have to required. Stage-2 ingredient (user's diagnosis, 2026-08-22):
  **the loss never supervises cell COUNT** — sparse dots + neg weight 0.01
  means counting is a training blind spot; that is why recall collapses
  with density (corr -0.68, ZH 0.65), why N/N_true drifts 1.01-1.30
  uncalibrated, and why ILP disappearance (a crude post-hoc counter) was
  the biggest post-processing lever. Fix: a count-consistency loss —
  heatmap probability mass pulled toward the known count (N_true exists
  for all 199 train sets; Zebrahub crops have exact per-frame counts),
  plus raised neg weight on dense crops; `cell_count.py` per-dataset
  estimates remain the inference-side fallback (+0.155 evidence in the
  classical era). Engineering gap: the official trainer has NO
  resume (best-weights only, no optimizer state) — 12 h kernel cap means
  a checkpoint-chain harness. Facts: trainer is Kaggle-native (T4 x2
  DataParallel hint in-code; pilkwang's "50ep" pack proves a 50-epoch run
  happened on Kaggle). Filters stay filters; every stage's verdict is
  the LB.
- **LB 0.889 — NEW BEST** (ref 55674219): the 0.881 config + the cap-15
  orphan patch, ported to a no-GT in-memory pass (`dr._candidates_one` on
  the solved graph, image feature via the zarr-free reader). The
  division term finally earns on the LB. Follow-ups on this line: tune
  the cap (15 was picked from a 4-point sweep), add the source-side
  mass-balance feature (the salvaged softmax idea), and re-measure the
  patch on top of any future linking config.
- Uncommitted: this docs restructure, `zebrahub_extract.py` rewrite.

### To do

- [x] ~~Measure the public detector on the 41 Zebrahub crops~~ — done,
      **recall 0.65 / precision 0.64** at the pipeline's det gate
      (entry below). The ruler works, has huge dynamic range, and costs
      6 min per full pass (`biohub-zh-detect-bench`).
- [ ] Fine-tune on the crops, re-run the bench, and if the gain is large,
      package weights as a dataset and SUBMIT — the LB (resolution ~0.03)
      is the final arbiter, per the user's call.
- [ ] Wire the 41 crops into `finetune_detect.py` (per-crop quantiles;
      dense labels need a different neg-weight scheme than the sparse-GT
      0.1; treat `parent_track_id < 0` OR the int64-cast NaN garbage as
      "no parent" — see the harvest entry).
- [ ] **Explain the 0.051 local-vs-LB offset.** Live probe: a `44b6`-only
      holdout — every number we own is `6bba`, which has a 7.47%
      duplicate-frame rate vs `44b6`'s zero. One inference pass.
- [ ] **Field-of-view drift** [CPU, GT only]: per frame pair, median
      displacement VECTOR of GT cells; coherent drift vs cancelling motion.
      Decides whether population growth can estimate division counts, and
      feeds appearance-based division recovery.
- [ ] `--ilp-division-weight` between 1.0 and 0.3 (0.85 would fork only on
      very confident second edges). Untested.
- [ ] `6bba_7af54fde`: 97 cells/frame yet J=0.828, worst over-detection
      (1.28). Motion suspected; one `run_gif` + displacement percentiles.
- [ ] Deferred: `cell_count.py` on real data (only if N_true estimation
      helps the node-count penalty).

Done/dead, for the record: ILP+A confirmed 12/12 (2026-08-19, 0.9269);
submitted (0.876); runtime budget resolved (5.2 h of 12, see reference);
det-threshold sweep (2026-08-21, dead as a lever); re-running the 0.916
notebook (dropped — pre-patch badge measures the closed exploit).

---

## 2026-08-21 — sigmoid edge activation: no win; sisters born far apart

**Sigmoid recache** (`biohub-cache-sigmoid`: edge_activation=sigmoid at
dump threshold 0.05, then a 7x3 replay grid). Mechanism confirmed: with
sigmoid, candidates above 0.5 EXCEED the target count (impossible under
softmax) — the mass-splitting starvation is real and softmax-specific.
But the score does not follow: grid best 0.9314 at threshold 0.70 /
disapp 3.5 vs softmax's 0.9387 at 0.40/3.5; edge_J lower everywhere
(0.9316 max vs 0.9363). The freed candidates are mostly the spurious pool
the stranded audit predicted; the ILP buys FPs with them. At threshold
0.9 the sigmoid graph collapses outright (adj_J 0.05 — probabilities
rarely reach 0.9). Verdict: line closed. (Holdout-measured, so the
negative carries the usual trust discount — but nothing here motivates
spending more on it.)

**A salvageable idea from the sigmoid post-mortem** (user's): the official
softmax normalises per TARGET (candidate parents compete), so a division
does NOT produce a 0.5/0.5 signature — each daughter's column gives the
shared parent ~1.0. But the transposed statistic can be built from the
cache: renormalise each SOURCE's outgoing candidate probs; a parent whose
top-2 outgoing shares are balanced AND whose edge lengths sit in the
division band (3.5-11 um) is a synthetic fork signature. Pure CPU, cache
only — a third feature for the orphan-fork scorer (existing: edge length
AUC 0.86, parent brightness AUC 0.71) if the base pass proves out.

**Sister-separation curve** (`biohub-sister-curve`, all 151 GT
divisions): at k=0, the daughters' first frame, separation p10/p50/p90 =
6.4 / 10.6 / 14.4 um; only 0.7% closer than the 3 um NMS kernel. The
2026-08-19 "NMS merges newborn daughters" hypothesis is dead: daughters
are annotated far apart from birth. Late-reporting within the +-1-frame
window solves nothing because nothing is merged. Division losses must
come from elsewhere (detection/linking of the second daughter — exactly
what the orphan-fork pass targets).

**Two kernel-engineering lessons** (cost: ~30 wasted GPU-min): (1) the
pack renames `tracking_cellmot` -> `biohub_tracking` AND its
`evaluate.py` is a different version without `evaluate_pairs`;
`division_recovery.sweep`'s import of it had never been executed. Score
against the official metrics API directly. (2) An ERRORed kernel's output
files are downloadable but CANNOT be attached as a kernel_source — a
run that dies at the last step strands its artefacts. (3) Loading all 41
crops decompressed needs ~33 GB RAM — OOM-killed finetune v2; use
/kaggle/tmp memmaps.

## 2026-08-21 — the clean ruler's first reading: recall 0.65, not 0.99

`biohub-zh-detect-bench` (v2): public weights, untouched, on all 41
Zebrahub crops — ~206k dense GT points, every 5th frame, greedy 1:1 match
within 5 um, NMS pool 3 um, no TTA. 6 minutes of T4.

| det gate | TP | FP | FN | recall | precision |
|---|---|---|---|---|---|
| 0.5 | 148,656 | 99,760 | 57,304 | 0.722 | 0.598 |
| 0.9 | 143,621 | 89,911 | 62,339 | 0.697 | 0.615 |
| **0.99 (pipeline)** | 133,787 | 76,243 | 72,173 | **0.650** | 0.637 |

Recall@0.99 by GT-point brightness quartile: Q1 (dimmest) 0.52 — partly
Zebrahub label noise; **Q4 (brightest) 0.74** — those are unambiguous real
cells, and the detector misses a quarter of them. Label noise cannot
explain the gap; the holdout's node_recall 0.9974 cannot survive it
either. Per-crop spread 0.32–0.92 (`zh_detect_bench.csv` in the kernel
output); ZSNS004 (a dim movie) is worst, dense ZSNS003/005 crops are fine.

Reading it honestly, three causes stack and this run cannot separate them:
(1) the holdout number is inflated (memorization + duplicate frames);
(2) genuine domain shift — Zebrahub is a different instrument setup, and
the hidden test (a third embryo on the COMPETITION instrument) sits closer
to the holdout than to Zebrahub, so 0.65 is a lower bound story, not the
hidden-test number; (3) method deltas (no TTA, approximate NMS) — worth a
few points at most.

What it establishes regardless: the ruler has dynamic range (0.65 baseline,
0.35 of visible headroom), a real fine-tune effect will be unmissable on
it, and "detection is solved" is dead as a general claim — it was only
ever measured where the model may have been grading its own homework.

**Correction (user's, accepted): "the hidden test is on the competition
instrument" was asserted without evidence — and (second push) so was
"same format".** Nobody has seen the hidden data; the visible test/ files
are train copies. And (third push, also the user's) "the pipeline read it
fine" proves NOTHING about similarity — Zebrahub itself, a different
instrument outright, goes through the same pipeline after resampling, as
this very bench demonstrates. Format compatibility and runtime only pin
format family and data volume. **The single load-bearing fact about
hidden-test similarity: train-tuned pipelines can reach 0.93+ on it —
LB #1 0.957 / #20 0.935, read via API 2026-08-20, post-rescore; our own
bare config 0.876.** (NOT "the whole field scores 0.9+" — an earlier
draft of this sentence overclaimed; most of the 2560 teams score below
0.9.) That top-end existence proof bounds the domain shift to an
eroding-percentage-points factor, not a new world; the public LB is 29%
of the test, so the bound carries that caveat too. Everything beyond
this is conjecture.
Photometrics are free to differ per batch — the competition's own two
batches differ 8x in brightness. Evidence the detector is photometrically brittle WITHIN one
instrument: ZSNS004 (dim) is the worst movie on this bench (crops down to
0.32 recall). So part of the local-vs-LB offset may be domain shift we
have never mitigated. And (user's addition) **brightness is only the
observed axis — sharpness/PSF width, noise character, scattering with
depth, and axes we have not thought to look at can all differ per batch
and are so far unmeasured.** Cheap probe if wanted: per-dataset sharpness
proxy (e.g. Laplacian energy / gradient p95 on a mid frame) across 44b6 /
6bba / Zebrahub, same style as the brightness survey. Consequences for
the fine-tune: (1) augmentation must cover more than gain — random
gain/offset/gamma/noise AND Gaussian blur of varying width (a blur-robust
detector is insurance against a softer-imaging test set); (2) mix
competition sparse-GT frames with Zebrahub dense crops — features that
hold across two domains stand a better chance on a third. LB submission
remains the only direct read on the hidden domain.

## 2026-08-21 — validation trust: null results on the holdout are void

User's argument, accepted: the 12-dataset `6bba` holdout is duplicate-
frame-inflated AND possibly inside the public weights' training split, so
a measurement of "no improvement" there is uninterpretable — a memorized
baseline leaves no visible headroom even for real gains. Corollary: if
0.982 detect_ceiling is memorization, the hidden-test detection is worse
than the books say and detector training is worth MORE, not less.

Trust tiers, applied from now on:

- **Still valid:** LB 0.876 (real); the scorer cross-check (same-graph
  comparison); the softmax-gate mechanism (read from code, not from the
  holdout); all Zebrahub facts (independently verified).
- **Discounted:** every absolute number on the holdout (0.9269, 0.9387,
  detect_ceiling 0.9823, node_recall 0.9974).
- **Void:** every NULL result measured on the holdout — including
  "det-threshold is not the lever" (the direction may still be dead, but
  the evidence is weak) and any "no effect" from the in-flight sigmoid /
  orphan-fork runs. Positive paired deltas keep some value; zeros do not.

Second, independent problem (also the user's): **no statistical power at
saturation.** The holdout has ~10.6k GT edges and ~180 detection-missed
endpoints; fixing 10% of the miss pool moves edge_J by ~0.0017 — the same
order as run-to-run noise (seeds, ILP tie-breaking, TTA), whose floor has
never even been measured. In the 0.98 regime, signal and noise are the
same size; small deltas in either direction prove nothing.

The fix for both: the 41 Zebrahub crops double as the first
contamination-proof detection benchmark (dense GT, ~900k nodes = ~100x
the holdout's effective sample; weights trained only on competition data
per the official training scripts). Measure the current detector there
before spending GPU on training; read all training experiments on that
ruler, where a real effect has room to be visibly larger than noise.

## 2026-08-21 — extraction v2 harvested: 41 crops, 7.5 GB, zero empty

`biohub-zebrahub-extract` v4 (SCRIPT_VERSION 2026-08-21d) ran 94 min and
completed cleanly. Per movie (units all detected correctly; crops with
cells/frame at T0):

| movie | units | T0 | crops | cells/frame span | divisions in full CSV |
|---|---|---|---|---|---|
| ZSNS001 | UM | 237 | 7 | 51-541 | 635,042 |
| ZSNS001_tail | UM | 237 | 5 | 63-566 | 36,878 |
| ZSNS003 | VOXEL | 154 | 9 | 43-384 | 15,869 |
| ZSNS004 | VOXEL | 180 | 10 | 52-578 | 30,279 |
| ZSNS005 | VOXEL | 180 | 10 | 36-672 | 46,027 |

- Density now spans 36-672 cells/frame, bracketing the competition's
  50-519, with sparse crops included — the v1 failure (3 empty, 2
  over-dense) is fixed. _tail's slab lacked the 80-250 bins (loudly
  degraded, accepted).
- Displacements p50 1.32-1.58 um everywhere; ZSNS004 p99 10.13 um is
  tail-heavy vs the competition's 6.71 — fine for detection, watch it if
  training a linker on 004.
- **Defect:** `ZSNS001_tail` uses NaN (not -1) for "no parent";
  `astype(np.int64)` cast it to garbage (RuntimeWarning in the log). All
  five tail crops carry it. Consumers must treat `parent_track_id < 0` as
  no-parent; the next extract revision should `fillna(-1)` before the
  cast. Division counting via "parent with >=2 children" must also drop
  the garbage id or every root pair looks like one giant fake division.
- Total division supervision now on disk: ~764k events across five movies
  vs the competition's 151.

## 2026-08-21 — extraction v2 (all five movies, density-bin placement)

`zebrahub_extract.py` rewritten (SCRIPT_VERSION 2026-08-21d), pushed as
`biohub-zebrahub-extract` v4. Changes, each answering a measured failure:

1. All five movies in one run, per-movie try/except, 14 GB output budget.
2. Units DETECTED per movie (coordinate max vs physical extent), columns
   normalised — the two traps in `reference.md#zebrahub-external-data`.
3. Placement: up to 2 crops per density bin {30-80, 80-150, 150-250,
   250-400, 400-650 cells/frame} inside the best z-slab, replacing "densest
   3x3 tile" which produced 3 empty crops and 2 over-dense ones (v1
   harvest, below). Small-chunk movies (001, _tail) confine candidates to a
   4x4-bin window so the per-frame read stays bounded; whole-plane-chunk
   movies (003/004/005) pay for the plane anyway and are unconstrained.
4. Fixed a silent whole-frame read: zarr 2 materialises `arr[t, 0]` (the
   full 4.7 GB frame) before a second `[slices]` applies. v1 downloaded
   every full frame without knowing. One-call indexing now (trap 22).

Placement logic dry-run locally on sampled real data before pushing:
4 crops covering 3 bins on the 7%-sample (bins fill at full density), bbox
108 MB raw/frame, within budget.

## 2026-08-21 — competition brightness is not uniform either

Question raised by the user: is per-dataset normalisation throwing away a
real feature, and is the competition's own brightness uniform? Measured
(`faustine649/competition-brightness-survey`): q0.999 spans 8.8x across the
199 train datasets — same regime as Zebrahub's 10x. The organizers store
per-dataset quantiles in the zarr attrs and the official `io.py` normalises
with them, so per-dataset normalisation is the designed-in policy, not a
choice we are making. Numbers and the frame-vs-movie caveat:
`reference.md#train-set-survey-199-datasets`.

Absolute brightness is instrument+expression+bleaching, not biology;
a linear per-dataset rescale preserves all within-image contrast and
ordering. `cell_count.py` had already excluded absolute-intensity features
for exactly this reason (kept only as a LOGO control).

## 2026-08-21 — first extraction run harvested (v1, ZSNS001 only)

`biohub-zebrahub-extract` SCRIPT_VERSION 2026-08-21a completed ~11:35 and
sat unread until evening (lesson: COMPLETE is not harvested). Results:

- Full CSV: 21,697,591 nodes, 635,042 divisions; displacement p50 1.24 /
  p99 5.19 um at t=200 — compatible with the competition's 1.82 / 6.71.
- 9 crops, 1278 MB, `crops/*.npz` in the kernel output. Six labelled
  (159 / 523 / 531 / 552 / 1041 / 1211 nodes per frame), **three empty**
  (densest tile hangs off the embryo edge — predicted by the pre-read).
- Two crops (1041, 1211/frame) are denser than anything in the competition
  (true 50-519/frame). "Maximise density" was the wrong placement
  objective -> replaced in v2 (entry above).

## 2026-08-21 — Zebrahub verified as training data: all 5 movies PASS

Re-evaluated the source bucket from the data alone, trusting no prior
assertion (user's instruction). Local probe on ZSNS001 (60 MB CSV head +
one zarr chunk), then `faustine649/zebrahub-align-spotcheck` ran four
checks per movie: units, alignment (the test the `kkunizaw` mirror failed),
displacement, density. All five PASS; full fact table and the traps now
live in `reference.md#zebrahub-external-data`.

What the check caught that assertions missed:

1. ZSNS003/004/005 coordinates are **voxel indices, not um** — the naive um
   reading fails alignment outright (v1 of the spot-check FLAGged both and
   crashed on 005; unit detection turned them into PASSes).
2. `ZSNS001_tail` renames columns (`ParentTrackID` etc.).
3. First displacement measurement said p50 = 0.00 — a self-inflicted bug
   (diffing across sampling gaps instead of dt==1 only). Corrected: 2.9%
   exact zeros, p50 1.52.

Verdict: suitable for training. Caveats that survive: machine-generated
labels (~10% near background — weight by local brightness), per-crop
normalisation mandatory, ZSNS002 unusable.

## 2026-08-21 — det-threshold is not the lever; the ceiling is the model

Scored on the cache at the converged linking optimum (threshold 0.40,
disappearance 3.5), so only `det_threshold` varies:

| det | adj_J | edge_J | detect_ceiling | link_rate | nodes |
|---|---|---|---|---|---|
| **0.99** | **0.9387** | 0.9363 | 0.9825 | 0.9839 | 161,649 |
| 0.98 | 0.9375 | 0.9362 | 0.9827 | 0.9836 | 164,185 |
| 0.95 | 0.9367 | 0.9365 | 0.9821 | 0.9842 | 166,209 |

`detect_ceiling` does not move and neither does `edge_J`; the adj_J fall is
entirely the node-count penalty on 2.8% more nodes. The extra detections
admitted by a lower gate are NOT the missing endpoints of the ~180
unreachable GT edges — those cells are not peaks in the heatmap at any
threshold down to 0.95. **The detection ceiling is set by the model's
output, not by our threshold.**

Pattern worth keeping (three wrong directions argued on this one knob, two
from stories about model internals, one sound-but-insufficient from the
metric alone): **score the thing.** The deflation story was wrong on
arithmetic: the negative loss term weighs ~2.4e-8 per unannotated voxel.

What it leaves open: a better detector, which means training — Zebrahub
(entries above).

---

## 2026-08-20 — the candidate cache, and what it measured

GPU inference now runs once (`cache_candidates.py` dumps coords/edges/probs
at threshold 0.05); every linking experiment replays on CPU
(`link_replay.py`, Kaggle entry `run_sweep.py`). One 12-dataset pass:
154 s CPU vs 11 min GPU. Cache verified faithful: replaying the submitted
config reproduces TP/FP/FN 9863/318/450 and adj_J 0.9269 digit for digit.

### Correction: 0.9953/0.9530 was the GREEDY decomposition

| | detect_ceiling | link_rate |
|---|---|---|
| as previously recorded | 0.9953 | 0.9530 |
| ILP+A, measured | **0.9823** | **0.9736** |

ILP wins by giving up detection reach (solver discards nodes) to buy
linking. Detection's share of the loss is 1.8%, not 0.5% — still
linking-dominated, but 3.5x what the log claimed.

### 11.8% of targets have no candidate parent at all

Pooled over the holdout at dump threshold 0.05: candidates above 0.5 =
167,398 (everything the submission ever saw); above 0.05 = 338,781 (2.02x);
targets = 189,872, of which **22,474 (11.8%) have no candidate above 0.5**
— structural false negatives. The softmax normalises over sources, so at
most one parent per target can exceed 0.5.

Stranded fraction correlates with over-detection (`N/N_true` r=+0.79), NOT
density (r=-0.07). Earlier "density is the primary failure axis" holds for
the score, not for this mechanism: a spurious detection has no true parent,
its column mass spreads, nothing clears 0.5.

### Counted, not estimated (`stranded_audit.py`)

1. Stranded targets are disproportionately spurious: GT-match rate 1.5% vs
   6.1% for confident targets (ratio 0.24, holds in all 12 datasets). Most
   of the 22,474 should stay unlinked.
2. Yet **278 of the 450 edge FNs (62%) are real edges stuck behind the 0.5
   gate** (27 more have an undetected target). Recovering all 278 at F new
   FPs: +0.026 at F=0, break-even near F=300, and the pool holds 22,196
   spurious targets — the real edges are 1.2% of it. Needle in haystack;
   whether it pays is entirely ILP selectivity.
3. Prediction for the 2-D sweep: the winning cell pairs a lower threshold
   with a LOWER disappearance weight (the current bias towards linking
   becomes a liability at a lower gate).

Over-detection became the higher-leverage target here — it pays the
node-count penalty AND buries the 278 recoverable edges.
`SUPERSEDED 2026-08-21`: the det-threshold knob was then swept and is dead
(entry above); the over-detection problem still stands, but its fix is a
better detector, not the threshold.

## 2026-08-20 — ⭐⭐ the edge threshold is the binding constraint

Read verbatim from `scripts/predict_unet_transformer.py` (defaults now in
`reference.md#running-the-official-pipeline`): the edge-candidate gate
`threshold=0.5` is the only limit on what the linker sees, and the softmax
normalises over sources. A target whose column mass splits between two
plausible parents (~0.5/0.5) gets NO candidate edge at all. This predicted,
before measurement: the 60% `DROPPED` failure bucket, `corr(edge J, true
density) = -0.68`, the densest movie being worst, and detect_ceiling >>
link_rate.

Arithmetic headroom estimated then: up to +0.041 if every dropped edge
recovers and retires an FP. `SUPERSEDED 2026-08-20` (same day, audit above):
the stranded pool is mostly spurious, so +0.041 was optimistic; the counted
recoverable number is 278 edges, +0.026 max at zero cost.

Consequences that stand:

- Sweep `threshold` x `disappearance_weight` TOGETHER (parameters
  interact; appearance 0.0 / disappearance 1.5 currently biases towards
  linking).
- Forks stay impossible throughout (needs edge_prob > 1.0), so the sweep is
  safe for the division term.
- `sweep_edge_threshold.py`'s greedy-only monotonicity argument does not
  transfer to ILP — hence the candidate cache instead of geff replays.
- The sweep landed at threshold 0.40 / disappearance 3.5 -> local 0.9387
  (state block). [Recorded 2026-08-21 alongside the det sweep.]

---

## 2026-08-19 — submitted: LB 0.876, local 0.9269

Official repo + pilkwang's public weights + two flags (`--use-ilp`,
appearance 0.0 / disappearance 1.5), `det-threshold 0.99`. No DeepCenter,
no second seed, none of the ~40 `BIOHUB_*` knobs. Up from 0.681 (classical
DoG pipeline).

| | LB | local (12 held-out `6bba`) |
|---|---|---|
| classical DoG | 0.681 | ~0.45 |
| **official + ILP+A** | **0.876** | 0.9269 |
| public 0.916 notebook (full stack) | 0.916 | never measured post-patch |

### The 0.051 offset: local is optimistic

Candidate causes, none separated yet: (1) the primary U-Net's split is
unknown — if it trained on the holdout, 0.9269 is inflated; (2) the hidden
test may be denser/differently distributed (density drives the score,
r=-0.68); (3) the node-count penalty depends on the density mix.
Operationally: **use local scores for paired deltas only** — "local 0.93 so
we are near 0.93" is false.

### Local best detail (12 datasets, official `evaluate.py`)

```
score              0.9269   (greedy default: 0.8847, improved 12 of 12)
edge_jaccard       0.9278   (TP/FP/FN 9863 / 318 / 450)
division_jaccard   0.0000   (0 forks emitted; 12 GT divisions all FN)
node_recall        0.9882
node-count penalty 0.0009   (was 0.0140)
```

Gain decomposition greedy -> ILP+A: edge_J +0.0318, penalty +0.0131,
division term -0.0027. TP +80, FP -288 (-48%), n_pred -12%.

Per-dataset (sorted by true density; `N/N_true` from geff
`estimated_number_of_nodes`):

| dataset | edge J | true cells/frame | N/N_true | div FP (greedy) |
|---|---|---|---|---|
| 6bba_3abfe10a | **0.7705** | **519** | 1.21 | 42 |
| 6bba_5c824876 | 0.9033 | 246 | 1.13 | 29 |
| 6bba_283bf9f1 | 0.9171 | 209 | 1.12 | 28 |
| 6bba_2312ac41 | 0.9611 | 119 | 1.14 | 11 |
| 6bba_61dd1e0d | 0.9164 | 102 | 1.09 | 14 |
| 6bba_7af54fde | **0.8280** | 97 | **1.28** | 29 |
| 6bba_3a1849c2 | 0.8783 | 70 | 1.17 | 20 |
| 6bba_c27cba08 | 0.9107 | 64 | 1.12 | 5 |
| 6bba_afb141ff | 0.9070 | 60 | 1.17 | 5 |
| 6bba_7b5d3b2c | 0.9596 | 58 | 1.01 | 8 |
| 6bba_268e1230 | 0.9016 | 51 | 1.30 | 15 |
| 6bba_aeee7805 | 0.9680 | 50 | 1.09 | 8 |

`corr(edge J, true density) = -0.68`, `corr(edge J, N/N_true) = -0.64`.
Two failure modes, separately attackable: crowding, and fast movement
(`7af54fde` is slow-density but fast-motion; the `6bba` batch's median
displacement p99 of 7.81 um exceeds the 7 um match radius).

### Division accounting at this config

Greedy emitted division TP/FP/FN 6/214/6 (div_J 0.0265); ILP+A emits no
forks: 0/0/12, ceiling 1.0 worth +0.1, realistic target the orphan-fork
pass's measured +0.0129. `SUPERSEDED-CORRECTION 2026-08-20`: an earlier
"18x fork over-prediction" claim was circular (220 = 6+214, never counted
from the graph); the metric only sees matched forks, so the true
over-prediction factor is unknown. Fork-leniency details:
`reference.md#the-metric`.

Headroom table (greedy config): div FP 214 -> 0 buys +0.047; past that
needs recovering the 6 FNs.

### Local scorer verified against official

Same 12 predicted graphs: edge TP/FP/FN identical (9783/606/530), adj_J
agrees to 5 decimals; division term differs slightly (ours 231 FP vs
official 214 — lenient approximation, quote the official). Licence to use
the fast local scorer for paired sweeps. `edge_fp_lenient = 5`: the old
scorer saw five of the 606 FPs.

### Divisions: negative feedback outweighs positive at the tested setting

`--ilp-division-weight 0.3` (scope: one knob, one notch, 3 datasets, 4 GT
divisions): div_J 0 -> 0.0308 (+0.0031) but adj_J 0.9152 -> 0.8907
(-0.0245). 8:1 against; default stays 1.0 (zero forks). Also measured:
missed divisions corrupt the edge term (GT edges leaving a dividing cell
fail at 29.2% vs 5.1% baseline) — worth roughly +0.001, real but small.
Obstacles: 151 divisions total (sample size); NMS `pool_kernel_um` may
merge daughters for ~3 frames — UNTESTED (`division_gif_cell.py` settles it
in a minute). Untried: weights between 0.3 and 1.0, smaller pool kernel,
deliberately-late splits within the ±1-frame window, official
`division_metrics.py` locally.

---

## 2026-08-18 — ⭐ ILP linking: +0.0295 over greedy (3-dataset pilot)

Paired, same weights, det 0.99, on `3abfe10a`/`2312ac41`/`aeee7805`:

| config | edge J | adj_J | penalty | score |
|---|---|---|---|---|
| greedy | 0.8956 | 0.8826 | 0.0130 | 0.8857 |
| `--use-ilp` | 0.9150 | 0.9073 | 0.0077 | 0.9073 |
| **+ app 0.0, disapp 1.5** | 0.9183 | **0.9152** | 0.0031 | **0.9152** |
| + division-weight 0.3 | 0.8999 | 0.8907 | 0.0092 | 0.8937 |
| + disapp 3.0 | 0.9097 | 0.9120 | -0.0023 | 0.9120 |

Most of the gain is the node-count penalty, not edge J: high disappearance
cost makes the solver prefer long tracks and discard scattered nodes
(n_pred -12%). At disapp 3.0 the penalty goes NEGATIVE (n_pred < n_true;
the formula has no upper clamp) but node_recall pays more. Optimum near
1.5, curve flat.

## 2026-08-18 — ⭐ the scorer bug

Real rule (`tracking_cellmot/metrics.py::_evaluate_matched_graph`): an edge
counts toward the denominator iff **either** endpoint matched a GT node of
the right degree. Ours required BOTH. Measured on 24 classical-pipeline
datasets: FP 2 -> 289; 99.3% of false positives were invisible. Also
implemented from the real scorer: t+1-only edges, merge collapse,
out-degree cap 2. Full metric rules: `reference.md#the-metric`.

Consequence: a missed detection costs twice (FN + induced FP), so detection
recall is worth far more than pre-fix measurements implied — exactly why
the official U-Net (recall 0.997) beats DoG (0.70) by more than recall
alone suggests.

**Every local number from before 2026-08-18 is void** (produced by the
broken scorer). In particular the old "0.916 pipeline scores 0.9614
locally" is dead.

Re-measured on the classical pipeline with the corrected scorer:
over-detection still wins (adj_J rising through ratio 2.8 at N_pred/N_true
0.94 — the DoG detector runs out of candidates; `nms_um` is its real
lever).

## 2026-08-18 — published artefacts read

DeepCenter's split, its two checkpoints, and the primary model's unknown
split — moved to `reference.md#published-artefact-facts-read-2026-08-18`.
Actionable consequence: the local holdout is `6bba`-only because DeepCenter
provably never saw `6bba`.
