# Working agreement — how to run a session on this project

*Migrated 2026-08-25 out of the local assistant memory directory, which
does not travel. Everything a fresh session needs now lives in the repo,
so a cloud session that clones this remote starts with full context.*

---

## How to talk to the user

**Plain language, result first.** Their words: "说的很厉害的样子但是我听不懂"
— it sounds impressive and I can't follow it. And sharper, later: long
reasoning chains do not merely bore them, **the chain drifts somewhere
wrong and the length hides it.**

- Lead with the result and what it changes, in two or three sentences.
- Explain a mechanism only when it changes a decision.
- Never describe a measurement before running it.
- They ask for detail when they want it.
- **No metaphors.** Said directly on 2026-08-25: "求你不要说话用比喻了".
  Say the literal thing — "the detector's recall is 0.65", not "the ruler
  is wrong"; "brightness weights dim voxels less", not "it strips the
  fringe". This governs what is written TO the user; entries in `docs/`
  already use figurative shorthand and are not being rewritten.
- **No unexplained abbreviations.** 2026-08-25: "TTA是什么，神奇的缩写增加了".
  Say the thing — "flip-and-average" / 翻转平均, not "TTA" — or expand it
  the first time. Project-local names coined WITH the user (adj_J, div_J,
  the orphan patch, the repair pool) are fine; field jargon imported from
  outside is not, and switching to an abbreviation for something already
  being discussed in plain words is the worst version of it.

**They will interrogate the provenance of any claim, and they are right
to.** Recalled "facts" drift — quantifiers inflate, inferences harden into
facts. So:

- look numbers up in `reference.md` / `progress.md` **at the moment of
  use**, never answer from recall;
- label anything unverified as unverified;
- keep facts and inferences visibly separate.

This has repeatedly paid: on 2026-08-24 they caught "the same person wrote
both" (it was a transplant from a public solution), "morphology change is
an edge signal" (unresolved confound, which then had to be tested), and
"aren't you changing too much at once" (which turned into the step-1 gate
that caught a real 40% intensity discrepancy).

## These docs are the master copy

`docs/` in this repository is authoritative. A Claude Project of the same
name holds an older transcription and is **not**. Several sessions have
edited this folder directly.

Cold start: the top of `progress.md` (current state and to-do), then
`strategy.md`, then `kaggle-toolchain.md` before running anything on
Kaggle. Grep `reference.md` as needed.

**The reader is an AI session, not a human.** Newest first, dated `##`
grep anchors, one fact in one place with pointers elsewhere, superseded
claims marked in place with `SUPERSEDED <date>:`, no narrative that
requires reading older entries first.

## Running things

Full detail in `kaggle-toolchain.md`. The three things a session gets
wrong if it does not read that file first:

1. **Push with `py kag.py`, never `py -m kaggle`** — the OAuth token has a
   30-minute dead window the SDK does not handle (trap 1b).
2. **Anything starting from cached detections belongs on a CPU kernel.**
   `enable_gpu: false` costs no quota and does not queue; the ILP and the
   official scorer reproduce bit for bit (section 9).
3. **Start one background waiter per run**, and grep it for
   `complete|error|cancel` — silence is indistinguishable from
   still-running (trap 6).
4. **When you assemble a script from a predecessor, take the SOURCE LIST
   from that predecessor's real metadata, not from its docstring**:

   ```bash
   py kag.py kernels pull faustine649/biohub-<pred> -p <tmp> -m
   ```

   Docstrings record what the sources were when the docstring was
   written. On 2026-09-08 `hparabench_ref` burned three pushes on this:
   the divbench9 docstring predates `biohub-edgeprob2`, and it never
   mentions that the bench RETRAINS the ranker on the 175-movie pool and
   therefore needs the pool's caches too. The real list also turned out
   to include two sources nobody would guess (`biohub-zh-detect`,
   `biohub-edgeprob-rank2`) and to exclude two that seemed obviously
   required (`fuse44b6`, `ep44b6`).

   Two things about `kernel_sources`, the second learned by getting the
   first wrong: there is **no cap at 10** (12 pushed fine; divbench9
   simply happened to use 10), but **a kernel can only be a source once
   it has completed a run and has output**. Pushing a chain of kernels
   means pushing and WAITING for each one before the next can name it —
   the push does not fail, it prints

   ```
   The following are not valid kernel sources and could not be added
   to the kernel: ['faustine649/biohub-<not-yet-run>']
   ```

   and then succeeds *without that source*, so the run starts and dies
   later on a missing file. Read that line; it scrolls past easily.
5. **Reading a bench kernel's log is slow**, because `kernels output`
   downloads the whole working directory and these kernels leave a
   cloned repo and a wheelhouse in it; the `.log` arrives LAST. Budget
   minutes, not seconds, and start it in the background.

### The assembly pattern

Most experiment scripts here are ASSEMBLED, not written from scratch: a
proven script's preamble and plumbing are cut out by LINE CONTENT and a
new feature block and measurement block are appended. This keeps the
environment, the caches, the holdout and the labels byte-identical to
whatever produced the number being compared against, so an A/B differs
only in what is intended. The assemblers live in the session scratchpad;
the assembled scripts are what gets committed.

Two rules learned the hard way:

- **Write every patch script to a FILE and run the file. Never pipe a
  patch through a heredoc or `py -c`.** The old rule here said "use the
  file tools for any patch containing an escape", and that condition is
  the bug: it asks me to notice, while writing a string, that the string
  contains an escape. I get that wrong. Four times on 2026-08-24 and
  **three more on 2026-08-27**.
  - The worst was silent. `py -c "…"` with DOUBLE quotes let the shell
    run command substitution on the backticks inside a `progress.md`
    entry, and it wrote **"SUPERSEDED by ."** and **"slope , independent"**
    — no error, no exit code, just a corrupted log that only turned up on
    a re-read. **A silently corrupted note is worse than a crash.**
  - Writing the script with the file tool and running the file skips the
    shell entirely: there is no quoting layer left to get wrong. It costs
    one extra call.
- **Locate cuts by line content, never by a search string containing an
  escape** (toolchain trap 8). And match on what is actually in the file:
  on 2026-08-27 a patch failed because the text used an em dash and the
  search string used `--`. `grep -n … | cat -A` shows the difference.
- **Run a static undefined-name pass before every push, and do not push
  when it is dirty.** It caught `root`, `GATE_XY`/`GATE_Z` and `POOL`
  missing from assemblies on 2026-08-24, each of which would have died
  at startup and wasted the queue wait. It cannot catch a missing MODULE
  or a data source the inherited preamble asserts on — check those by
  reading the preamble's imports and asserts.
- **When an assembly adds a NEW CODE PATH rather than re-parameterising
  an existing one, test that path on synthetic data before pushing**
  (2026-09-07). `divbench4` added a steal pass to the division patch and
  eight fork features. Both were clean under pyflakes and both were
  broken:
  - stage B calls `PATCH_CAPS.clear()` per arm and the patch then does
    `PATCH_CAPS["orphan_added"] += n` on a **plain dict** — `KeyError` on
    the first movie of stage B, 40 minutes in.
  - the steal required `q` to be the nearest STEALABLE node to `c1`,
    copied from the orphan pass's "nearest orphan" test — but `c1` is
    itself stealable and is its own nearest neighbour, so **every steal
    proposal was rejected**. That one is silent: three arms would have
    run for three hours and returned REF three times.
  A 200-line synthetic test (`tools/test_divbench4_steal.py`, which
  `exec`s the functions straight out of the shipped file so it tests the
  real text) found both in ten minutes. The rule is not "write tests";
  it is that **a re-parameterisation inherits its correctness from the
  run it was cut from and a new path does not**, and the silent failure
  mode — an arm that runs and returns the control — is exactly the one
  an anchor cannot catch, because the anchor is the control.

## Anything you omit will be read as a negative (2026-09-08)

Three separate failures in one afternoon, all the same shape:

- `shoulderviz` v1 drew a circle for every annotated cell onto a Z slab
  pinned at `Z//2`. Cells at other depths became rings on empty space.
  The user: "我也完全分辨不出来细胞在哪里呢". It read as *the detector
  missed a cell that is plainly there* -- the opposite of the truth.
- `shoulderviz` v3 "fixed" that by HIDING every marker more than one voxel
  off-plane. On a 64-plane sweep a cell was then visible for three planes
  and the circles blinked. It read as *the annotation is sparse in time*.
  It is not: 23 of 24 movies annotate every frame.
- `shoulderfeat` v1 iterated all 199 train movies when only 24 had ever
  been measured. The other 175 contributed ~120,000 cells labelled "not a
  shoulder cell" because nobody had looked, diluting every rate tenfold.

A viewer -- human or the next session -- cannot distinguish "absent from
the data" from "dropped by the tool". They will always read the first.

So: **when something is filtered out of a display or a denominator, say so
in the artifact itself, or do not filter.** Concretely --

1. Prefer ENCODING to DROPPING. A marker that fades, thins and goes dashed
   with distance still tells you the cell is there and where it went; a
   marker that disappears tells you nothing and implies something false.
2. Put the count of what was filtered in the frame: `GT 40 (12 in slab)`
   costs nothing and makes the filter visible.
3. A partial label set makes every unlabelled row a silent negative.
   Restrict to the labelled subset and print how many that is, or state
   the dilution.
4. This is the display twin of the DENOMINATOR rule below. That rule is
   about comparing two rates; this one is about a single picture or table
   that quietly answers a different question than the one asked.

The tell that you have done it again: the viewer proposes a mechanism for
what they are seeing ("被z轴叠掉了", "标注是隔几帧一个") and the mechanism
is about your tool rather than about the biology. Check the tool first.

## When you CUT a kernel, re-read its completion block first (2026-09-09)

Three false `INCOMPLETE`s in two days, all from the same cause and all on
runs whose outputs were correct:

| kernel | printed | actually |
|---|---|---|
| `ephpcache` (2026-09-08) | `EPHPCACHE INCOMPLETE ... NOT OK` | 24/24 cached, n_per_frame 24/24 exact |
| `ep2hpcache` v1 | would have, same bug | — |
| `ephpcache` (2026-09-09) | `EPHPCACHE INCOMPLETE ... NOT OK` | 140/140 cached, 0 real mismatches |

The cause every time: **the completion and verification block was written
for the ORIGINAL script's scope, and the cut changed the scope.** The three
concrete forms:

1. **A hard-coded label prefix.** `n_f4` counted files starting
   `edgeprob_F4_`, but the cut renamed the label to `F4hp_`, so it counted
   zero and called every output "base".
2. **An arm that no longer exists.** `complete` required
   `n_base == len(names)` -- 199 outputs a version with the base pass
   removed never sets out to write.
3. **A baseline that only covers the old scope.** The n_per_frame check
   compares against shipped caches that exist for 24 movies; widening the
   run to 140 made 116 movies uncheckable, and the no-baseline branch was
   appending to the MISMATCH list.

**The rule: a cut kernel's tail is not inherited, it is rewritten.** Before
pushing a script cut from another, read its completion block and answer
three questions in order:

- what does it COUNT, and does the count's prefix/label still exist here?
- what does it require to be COMPLETE, and does every arm in that
  requirement still run?
- what does it CHECK against, and does that baseline cover the new scope --
  or is "no baseline" now an expected state that must be reported
  separately from a failure?

**Why this is worth a rule rather than three fixes.** The failure is
asymmetric and expensive in the wrong direction: it never reports success on
a broken run, only failure on a working one. That is the safe direction, but
it costs a full re-read of the log every time to establish that the run was
fine, and twice it nearly cost a re-run of an hour or more of GPU. And it
scales with the practice this project relies on most -- almost every kernel
here is cut from another.

The corollary is already in this file under *Anything you omit will be read
as a negative*: **a check that cannot run must say so as its own category,
never as a failure.**

## An experiment kills a CLAIM, not a direction (2026-08-30)

The user's standing correction, in their words: "从来没有真的死了的路，
只是暂时还不知道怎么利用罢了". The record already proved them right
before they said it: gap2 "closed" the span line while the budget bound
in 24 of 32 datasets, and the only leaderboard gain this project ever
had (cap20) was buried inside that "closed" line. "DeepCenter does not
matter" (the public tier disables it) preceded detcmp's 69.5% miss
coverage. "The linker is finished" preceded rank5's 8.6-AUC-point gap.

**The rule: a verdict entry must state (1) the narrow tested claim --
usage, conditions, domains, n; (2) what the direction still OWNS --
unexplained regularities, assets, partial wins; never a funeral.**
"Closed" is reserved for the specific configuration actually measured.
The costume of finality is how "detection is solved" and "the metric
needs a GPU" survived for days; an unexplained residue line is how
cap20 was eventually found.

This does not license re-running measured configurations -- the
don't-re-run discipline stands. It changes what gets WRITTEN, so the
next session inherits the residue instead of the funeral.

## Always anchor

Every A/B must contain an arm that reproduces a recorded number. When
`poolsize`'s D32 arm returned edge AUC 0.9884 — the exact figure
`ranker_fine` recorded — and adj_J within 0.0002 of `div-gate`, the
+0.0073 delta was attributable to the one thing that varied. When
`morph_probe`'s step 1 did NOT reproduce, the shape of the mismatch
(geometry exact, all four intensity columns off) named the cause in one
step.

**An offline AUC is not a score.** On 2026-08-24 edge AUC anti-predicted
the pipeline three times running. Measure adj_J.

### Anchor the POPULATION, not only the number (2026-09-09)

`ttaprobe` v3 ran 36 GPU passes, wrote every cache, and then printed an
AUC table that was wrong by one subscript:

```python
prim = load_primaries(name, "F4")          # [ (N_t, 3) ] per frame
mp = match(np.asarray(prim[t_][0], ...))   # <- row 0 OF THE FRAME
```

`load_primaries` returns a list of arrays, not a list of
`(coords, scores)` tuples, so `prim[t][0]` is the FIRST DETECTION of
frame `t`. Only detection index 0 could ever be matched, as source or as
target. Nothing raised. Nothing was NaN. The table looked ordinary --
three arms, a positive rate near 0.5, an AUC near 0.7 -- and reported
tta4 at −0.052 where the truth is −0.004.

**The only thing that gave it away was a row count.** 565 labelled rows
across twelve 100-frame movies is a quarter of one per frame, and
`accdiag` had recorded ~14,817 for the same rule on the same holdout. The
corrected readout gives 14,636.

So the anchor rule has a second half. "Always anchor" means an arm must
reproduce a recorded VALUE; this says an arm must also reproduce a
recorded SIZE:

- **Print the population, every time.** Rows, movies, positives. A
  readout that cannot say how many rows it scored cannot be checked.
- **Assert a floor when the population is known.** `ttascore` refuses to
  print on fewer than 5,000 rows. That is not a statistical threshold,
  it is a tripwire for exactly this bug.
- **Reconstruct the join instead of trusting it.** The real check was
  cheap: rebuild the gate from the same primaries and demand the cache
  agrees on the per-frame pair COUNT and on every cached pair sitting
  inside `GATE_UM`. Passed 12/12, worst distance 10.000 um. A wrong
  primaries index fails both, at the first movie, before any GPU time.

**Why an index bug is the dangerous kind.** A wrong FILE raises. A wrong
COLUMN usually shifts a number until it looks silly. A wrong INDEX
silently selects a subset, and a subset of a real measurement produces a
plausible table. The failure mode is not a crash, it is a smaller `n` you
never printed.

**And it was recoverable, which is the other half of the lesson.** The
bug was in the labeller, which runs after every cache is written, so the
69 minutes of T4 were intact and `ttascore` redid the readout on CPU in
two minutes. Cache the expensive stage to disk and keep the readout
separate: the readout is where the bugs are, and it is the half you want
to be able to re-run for free.


## Before claiming a term behaves differently in two regimes, differentiate it (2026-08-26)

On 2026-08-26 I wrote a five-star entry arguing that the metric's
node-count term explained both leaderboard failures. The argument:

> the local holdout is UNDER the estimated cell count, so the term is a
> BONUS; adding nodes moves toward the target and is rewarded twice.
> On the hidden test, if we are at or over target, it is rewarded once
> and punished once.

The term is `1 - 0.1 * (N_pred - N_true)/N_true`. **Its derivative with
respect to added nodes is `-0.1/N_true`, a constant.** Where the ratio
currently sits does not enter. Local and hidden pay the same rate,
adding nodes always costs, and the entire asymmetry was imaginary.

Two things went wrong and both are cheap to prevent:

1. **I reasoned about the term's SIGN when the question was about its
   SLOPE.** "It is a bonus here and a penalty there" is a statement
   about the value; "adding nodes helps here and hurts there" is a
   statement about the change. A one-line derivative separates them.
2. **I read "under target" as "adding nodes is free".** That would be
   true of `1 - 0.1 * max(0, ratio)`. `reference.md` records the actual
   form, `max(0, J * (1 - 0.1 * ratio))`, where the floor is on the
   whole product. **I had the formula written down correctly and still
   reasoned from a clamp that is not in it.**

**The rule: when an argument turns on "X behaves differently in regime
A than regime B", write down the derivative before writing anything
else. If it is constant, there is no regime.** And when a formula has a
`max` or a `min` in it, say out loud what it is wrapping — the
difference between clamping a factor and clamping a product is the
difference between an effect and no effect.

Cost of not doing this: a five-star entry, a submission argued for on
its basis, and a run (`thr5`) designed to exploit a regime that does
not exist. `thr5` was not wasted — its decomposition closed the
tightening line on its own evidence — but that was luck, not design.

## Use the scorer's own constants (2026-08-26)

**`MATCH_UM` was 5.0 in every diagnostic in this project. The official
scorer matches at 7.0.** Every recall and miss figure ever reported here
was therefore stricter than the thing being optimised, and by a lot:
6bba's detector recall is 0.9813 at 7.0, not the 0.9561 this log
repeated for two days. Miss rate 1.74%, not 4.45%.

Before measuring anything against ground truth, read the constant out of
`tracking_cellmot/metrics.py` rather than inheriting it from a sibling
script. `evaluate(graph, gt_graph, scale=None, max_distance=7.0)`.

## Say how many datasets an arm actually has (2026-08-26)

`flicker3` reported 44b6 with **49 tracks**. Twelve datasets at 327
cells per frame, annotated at 0.77%, tracks of length ≥3 — forty-nine,
of which thirteen were intermittent.

**Every "three domains agree" verdict in this log has been carrying an
arm of that weight without saying so.** The rule counts domains; it does
not weigh them. Print the n for each arm, every time, so a reader can
see when "three domains" is really two and a rumour.

## An arbitrary sort key is not a random sample (2026-08-27)

Every sweep in this project selected its second-batch arm as

```python
SET_44B6 = sorted(n for n in CACHES if n.startswith("44b6_"))[:12]
```

**the twelve LOWEST hex ids out of 71**, identical in every run, and
nobody checked whether those ids encode acquisition order, embryo or
session. A hash-looking key FEELS random; a fixed prefix of a sort on it
is not a sample, and if the ids carry any structure then every
three-domain verdict this project has issued shares one skew.

The user's standing instruction, 2026-08-27: **"所有测试集的 44b6 都要
随机抽"**. Implemented in `gap4.py`, the template later assemblies are
cut from:

```python
N_44B6 = 12
SEED_44B6 = 20260827
_ALL_44B6 = sorted(n for n in CACHES if n.startswith("44b6_"))
SET_44B6 = sorted(random.Random(SEED_44B6).sample(
    _ALL_44B6, min(N_44B6, len(_ALL_44B6))))
```

Three properties, each deliberate:

- **Seeded, not freshly random.** A sample that changes every run makes
  every must-reproduce anchor meaningless, and anchors are what caught
  the gap2 normalisation bug and proved gap4's reorder safe.
- **The chosen names PRINT every run.** A sample nobody can see is a
  sample nobody can check.
- **Changing `SEED_44B6` asks "does this result depend on the sample";
  changing `N_44B6` asks "how precise is it".** Different questions, do
  not conflate them.

`SET_ZH = sorted(ZH)[:8]` is left alone because 8 is every Zebrahub crop
that exists, so the prefix is the whole population. `SET_6BBA` must stay
the recorded twelve: they are the only 6bba datasets outside D187's
training pool, so there is no sample to draw.

**Generally: whenever a subset is taken, say in the output HOW it was
drawn and WHAT was drawn.** "12 of 71" is not a description a reader can
audit; a seed and a name list is.

## A constant belongs to the stage that READS it (2026-08-28)

Every sweep script in this project descends from `post_sweep`, and they
all share one loop:

```python
for name in names:
    dets = load_dets(name)                  # ONCE per dataset
    nodes0, edges0, pool = solve_one(...)   # ONCE per dataset
    for cfg in SWEEP:
        apply_config(key, val)              # constants take effect HERE
        patch_and_save(...)
```

**`apply_config` runs AFTER the detections and the ILP.** That is the
whole point — the solve is expensive and every arm shares it, which is
also why the ranker leak cancels in the delta for a post-link sweep.

**So a constant read by `load_dets` or `solve_one` is fixed before the
sweep touches it, and is silently not swept.** `dcmerge` v1 put
`DC_TAU` — a detection-time constant — into that sweep. Every arm shared
one detection set, and because the inner loop leaves the constant at the
last arm's value, datasets 2 onward were all merged at the LAST
threshold. Four arms, identical numbers, and BASE contaminated.

**Every run in this family prints "POST-LINK SWEEP" in its own header,
including that one.** Knowing the name of the harness did not prevent
putting the wrong kind of constant in it.

**The rule: before adding a constant to a sweep, find the line that
READS it and check whether that line runs inside or outside the arm
loop.** If outside, the loop has to be inverted — solve per arm, at
roughly Nx the compute — or the constant cannot be swept at all.

Two supports worth carrying:

- **Name the detection-time constants in the file.** `dcmerge` now
  carries `DETECTION_TIME_CONSTANTS = {"DC_TAU", "DC_MERGE_UM",
  "DC_ADD_SCORE", "UNET_THRESH", "CAND_THR", "NMS_UM"}` next to
  `apply_config`, so the next assembly sees the list before it sees the
  loop.
- **The anchor is what caught it**, again: BASE returned adj_J 0.8897
  against the required 0.9292, and `node_ratio` flipped from −0.0215 to
  **+0.1336**. Without the must-reproduce arm, four identical rows would
  have read as "the threshold does not matter".

## Submit a MECHANISM, not a local number (2026-08-28)

Five paired local-vs-leaderboard comparisons now exist, and the pattern
in them is not the one this project spent two weeks assuming.

| change | local | LB |
|---|---|---|
| windowing | +0.0019 | −0.001 |
| ranker pool 32 → 187 | **+0.0073** | −0.001 |
| `SHORT_MIN` 6 → 4 | +0.0025 / +0.0054 / +0.0045 | 0.000 |
| detection gate 0.15 → 0.12 | +0.0055 / +0.0045 / **+0.0121** | −0.002 |
| **gap budget 0.003 → 0.02** | **+0.0013** / +0.0085 / +0.0056 | **+0.002** |

**The one that worked had the SMALLEST weakest-domain delta of the
five.** Local magnitude predicted nothing. What separates the last row
from the other four is not its size, it is that something was PROVABLY
broken: `wide44` counted the repair budget binding in **30 of 83
datasets** and leaving **64% of demand unrepaired**. The other four were
constant tuning, where the local number was the entire argument for the
change.

**The rule: before spending a submission, be able to state what is
provably going wrong, in a unit that is not the score.** A count of
work not done, a resource exhausted, a stage rejecting inputs it should
accept. If the only argument is "this value tests better on the
holdout", the record says that is worth nothing.

Two corollaries:

- **Instrument the stage before sweeping it.** `gap4` was the first
  script here to print bridges made / allowed / demand, and that one
  line is what turned a knob-sweep into a diagnosis. Any stage with a
  budget, a cap, a `break`, or a fixed-size pool should report its
  utilisation on every run. (This is the operational half of *Find the
  binding constraint before sweeping anything behind it*.)
- **A small delta with a mechanism outranks a large delta without one**
  when choosing what to submit — which is the opposite of how the
  first four were chosen.

n = 1 on the positive side, so this is a hypothesis. It is the first
thing that has ever distinguished a leaderboard success from a
leaderboard failure here, and it costs nothing to apply.

### Read the public score as a DIRECTION -- sign only (2026-09-08)

Two more paired rows exist now, both from the division line and both
positive: the fork gate (local +0.012, **LB +0.010**) and submit-joint
(local +0.0037, **LB +0.001**). Writing up the second one, this file's
own author twice turned the shortfall into a rule -- first "a +0.003
local delta is not worth a submission slot", then "read a bench delta as
an upper bound" -- and the user struck both down:

> 不用太因为LB分数而消极。因为不同的影片哪怕是同一组的都有很大的差别，更不要
> 说验证集完全是盲的，我们无从推断算出public score的那部分验证集是取的哪一部
> 分。所以理想情况下，验证集的分布应该和训练集是一样的，那么public score就当
> 一个方向就可以。如果本地全正，LB负了才是要警惕的。

- **ALARM: local positive across the board, LB NEGATIVE.** The bench and
  the hidden set disagree in KIND. Four of the five rows in the table
  above are exactly this, and that is what the table is for.
- **NOT alarm: LB positive but smaller than local.** No diagnosis, no
  paragraph, no downgrade of the bench, and above all not an argument
  against submitting the next small gain.

**The magnitude cannot be read, and this is measured, not assumed.**
Movies inside one domain are not interchangeable. Across the 128 6bba
movies in divbench9's own run log: candidate pairs 3,581 to 272,577
(76x, p90/p10 23.6x); **labelled GT edges 295 to 6,716 (22.8x, p90/p10
4.3x)**. The score sums edge TP/FP/FN across movies and divides once, so
a movie's WEIGHT in the aggregate is its edge count. Two subsets of the
same pool can therefore weight one change by 4x against each other
before any modelling difference enters, and the public split is one
unknown, small subset. If its distribution matches the training pool's
-- the ideal case -- the SIGN is what transfers; the size need not.

This does not soften *Submit a MECHANISM, not a local number* above:
that rule governs what is worth a submission BEFORE it is spent. This
one governs how to read the number that comes back.

## Write down what was COUNTED, not what it means (2026-09-08)

Three entries on this project stated, as fact, that the detector has no
response at the cells it misses. None of them measured a response.

| entry | what it actually counted | the sentence it grew |
|---|---|---|
| `det_floor` 2026-08-25 | lowering the peak FLOOR 0.05 -> 0.01 buys +0.0002 of ceiling | "the detector produces no response for them at all" |
| `gap2` 2026-08-26 | 79.8% of missed frames have no CANDIDATE within 5 um | "the heatmap response does not dip below threshold, it vanishes" |
| two later entries | nothing -- they cited `gap2` | "79.8% of missed frames have NO heatmap response" |

`gtpeak` finally read the heatmap: at 95.3% / 85.6% of missed cells the
response is present (window max p50 **0.15**), and the nearest actual
peak is ~6 um away on a neighbour. **Every one of the three
measurements was correct. Every sentence attached to them was wrong**,
and the wrong sentence is what got cited, for two weeks, as the reason
a whole line was closed.

The mechanism of the error is worth naming because it is not
carelessness: a peak-based measurement and a value-based measurement
answer different questions, and the peak-based one is the only one the
pipeline can cheaply make. So the interpretation quietly substitutes
for the measurement nobody ran.

**The rule: an entry states what was literally counted, in the units it
was counted in, and any interpretation stands in its own sentence with
the word that marks it as one.** "79.8% of missed frames have no
candidate within 5 um" is a fact. "So the response has vanished" is a
hypothesis and has to look like one, or the next reader -- who will be
a different session with no memory of the run -- will cite it as data.

Corollary for reading the log: **when an entry explains WHY a number is
what it is, check whether that run could have measured the why.** If it
could not, the explanation is a hypothesis carried forward, and it is
fair game to test rather than a settled result to build on.

## An argmax is only valid while the stages that shaped the curve are unchanged (2026-09-08)

Three constants on this project were closed by picking the best point
of a curve, and all three curves were measured in a pipeline that no
longer exists.

| constant | closed by | what changed underneath | reopened |
|---|---|---|---|
| `GAP1_CAP_FRAC` span/gates | `gap2`, `gap3` | the repair budget bound in 24 of 32 datasets | 2026-08-27, `gap4` gained +0.002 at LB |
| the patch's condition walk | `divgate2` | the per-frame cap refused 27% of proposals | 2026-09-08, `divbench7` -- and it did NOT pay |
| **`DIV_W = 0.55`** | `ranker-fine`, 12 GT divisions, no gate | **the fork gate now removes 80% of forks** | **2026-09-08, `divbench6`: 0.35 wins, both folds** |

`ranker-fine` was not careless. It measured the curve correctly, said
in its own words that "0.45-0.65 are all defensible and 0.55 is simply
the best measured", and picked the argmax. What made the low arms lose
was over-forking -- which is the one failure mode the fork gate was
later built to remove. **The verdict was right for the pipeline that
measured it and wrong for the pipeline that inherited it.**

The rule: **when a stage is ADDED that changes the cost of the failure
a constant was chosen to avoid, every constant chosen against that
failure is provisional again.** Write the dependency down at the time
of the choice -- "0.55 beats 0.20 because 0.20 over-forks" -- so that a
later reader can grep for what would invalidate it. An argmax with no
recorded reason is an argmax that cannot be reopened on purpose, only
by accident.

Cheap and worth doing: when closing a constant, name the mechanism that
makes the losing arms lose. It costs one sentence and it is what turns
a table into something a future run can falsify.

## Find the binding constraint before sweeping anything behind it (2026-08-27)

`_gap_close` caps repairs at `GAP1_CAP_FRAC = 0.003` of node count and
hands the allowance out in frame order, breaking the moment it runs
out. Across 32 datasets the true demand is **4,772** bridges and the
allowance is **2,297**. The cap bound in **24 of 32 datasets**.

Three separate sweeps ran behind that constraint without noticing:

| run | swept | what it actually measured |
|---|---|---|
| `gap2` | the bridge SPAN | who gets a fixed allowance |
| `gap3` | four bridge GATES | who gets a fixed allowance |
| `gap4` arms 1-4 | the user's policy | who gets a fixed allowance |

`gap2` concluded "the span never mattered" and the line was closed.
`gap3` concluded four gates were selecting good bridges. **Raising the
cap to 0.02 and removing every gate beat all of them: +0.0013 / +0.0113
/ +0.0056, roughly ten times `nocand`'s 44b6 delta.** The gates were
not selecting quality; they were rationing scarcity more sensibly than
frame order did, and once the scarcity was gone they were pure loss —
`len8` discards 62% of the repairable gaps.

**The rule: before sweeping a knob, ask what limits the thing the knob
feeds. If a fixed budget, a cap, a `max`, a `break`, or a fixed-size
pool sits downstream, the sweep measures allocation, not quality, and
every verdict it produces is provisional on that limit.**

Two cheap habits that would have caught this on day one:

1. **Count what the stage actually did, every run.** `gap4` was the
   first script in this project to print bridges made / allowed /
   demand. One line of output, and it made the whole thing obvious.
   Any stage with a budget should report its utilisation.
2. **Include one arm with the constraint removed.** Not as a candidate
   to ship — as a control that says whether the constraint is binding.
   If it is, every other arm in the run needs re-reading.

And when a limit turns out to bind, **the lines closed behind it are
not closed.** `gap2`'s span verdict was reopened by this.

## An unconditional arm is not a ceiling (2026-08-26)

`gap2` had an arm that dropped the candidate requirement entirely and
bridged **every** gap. It measured +0.0018 / −0.0016 / +0.0013, and I
wrote that down as the upper bound of the gap-closing line: a more
careful selector accepts a subset of those bridges, so it can only land
below.

That is wrong, and the user said so:

> 理想情况下应该是直接插入一个点在一定的情况下可以触发,
> 而不是直接插入一个没有检测出来的点这种条件可以随时触发

**An unconditional arm measures the SUM of its good and bad cases, which
have opposite signs. What it reports is what survives the cancellation,
not the best available.** A gated version sits above it whenever the
gate removes more harm than help — and it can sit above the *original*
too, because it is not a subset of the original's behaviour either.

`gap3` measured it: requiring both joined fragments to be at least 8
nodes long beat the unconditional arm in 6bba (+0.0013 vs +0.0009) and
in 44b6 by 3.6× (+0.0043 vs +0.0012), while both stayed above the
shipped BASE.

**So "drop the condition entirely" answers a different question from
"what is the best condition", and only the first one is cheap.** When
an ungated arm comes back near zero, that is a reason to look for the
cancellation, not a reason to close the line.

## Cross the knobs whose mechanisms touch (2026-08-26)

**Sweeping one thing at a time is not neutral. It is a claim that the
knobs are independent, and that claim needs a reason.**

`gap2` was built to sweep `_gap_close`'s span and candidate requirement
with `SHORT_MIN` fixed at 6. The user stopped it — *"交叉测呢"* — on the
grounds that the two are the same phenomenon from opposite ends: a
dropout longer than one frame shatters a track, the gap repair reaches
only one frame, and the short filter then deletes the fragments. If a
wider bridge repairs the dropouts, there is less for the filter to
delete.

Crossed, that is exactly what happened. **Five gap arms × three domains,
and all fifteen gains shrink by roughly half when `SHORT_MIN` goes
6 → 4:**

| gap arm | 6bba sm6 → sm4 | 44b6 sm6 → sm4 | zh sm6 → sm4 |
|---|---|---|---|
| nocand1 | +0.0018 → **+0.0010** | −0.0016 → **−0.0008** | +0.0013 → **+0.0005** |
| span2i | +0.0019 → +0.0011 | −0.0016 → −0.0008 | +0.0013 → +0.0005 |

**One at a time would have overstated the gap line by about 2x**, and
against a `SHORT_MIN` we had already decided to abandon.

The test to apply before choosing a design: *can I name a mechanism by
which these two knobs act on the same failure?* If yes, cross them. Two
knobs at two values is four arms, not two — usually affordable, and the
interaction is the finding.

Also, from the same run: **the shipped configuration must be a literal
arm, not an assumption.** `gap2`'s rewritten `_gap_close` normalised its
gate by `span + 1`, which at span 1 doubled `GAP1_GATE_UM` from 9 to 18
and quietly moved BASE off the shipped behaviour. The must-reproduce
anchor caught it — adj_J 0.9253 against the recorded 0.9254, div_J
0.1282 against 0.1316 — but only *after* 204 minutes of compute. A
refactor that claims to preserve behaviour should be tested on one
dataset before the sweep is launched.

## Three domains, or it does not count (2026-08-25)

**The 12-dataset holdout is 0 for 2 against the leaderboard.** Windowing
measured +0.0019 locally and scored -0.001; the ranker pool measured
**+0.0073**, reproduced bit for bit by two runs with both anchors
holding, and scored -0.001 with hidden-test output within 0.07%. Both
positive local deltas simply were not there.

The boundary is a MAGNITUDE, not a direction. It is tempting to say the
holdout "rejects reliably but selects unreliably" -- **there is no
evidence for that asymmetry** (the user's correction). If it inverts on
a +0.007 it has no claim on a -0.002 either. What it can still do is
reject an effect far outside the ~0.007 inversion scale: `node_filter`'s
-0.03 and `div_gate`'s monotone prune curve are safe rejections.

**So every pipeline evaluation runs in three domains and a change counts
only if its adj_J delta has the SAME SIGN in all of them:**

| domain | why it is there |
|---|---|
| `6bba` (12 holdout) | the recorded anchor lives here |
| `44b6` (12, outside the ranker's pool) | the OTHER imaging batch -- 327 cells/frame against 97, displacement p99 4.90 um against 7.81 |
| Zebrahub (6-8 crops) | a different instrument, and **no public model has seen it** -- which `6bba` cannot claim, since DeepCenter validated on 6bba and the xl detector's split is unknown |

Zebrahub is a TEST domain here, not a training one. Those are different
requirements and conflating them is a mistake already made once: a
Zebrahub-TRAINED node filter lost 0.03 adj_J on competition data
(`biohub-node-filter`), which says nothing about whether a difference
MEASURED there reproduces. Training transfer needs learned parameters to
stay valid across domains; test-domain use needs only the ORDERING to
survive.

Machinery: `post_sweep.py` and `thr3.py` carry all three, including
`zh_load` (from `zh_pipeline.py`) which builds a GT graph from Zebrahub's
own arrays and scores it with the OFFICIAL metric. Reuse them rather
than rebuilding.

**Watch the ranker leak when choosing which model to load.** D187 trained
on 187 of 199, so every `44b6` dataset is in its pool and its edge
probabilities there favour whatever configuration produced its training
candidates. Sweeps that change the DETECTIONS must use D32 and exclude
its 32 datasets from the 44b6 arm. Sweeps that only touch post-link
constants may use either, because the same ranker, detections and ILP
solution appear in both arms and the bias cancels in the delta.

## Say the DENOMINATOR out loud before comparing two rates (2026-08-25)

Seven times in one session I compared numbers whose denominators were
different populations. It is the single most common error this project
has produced, it survives code review because the arithmetic is always
correct, and it has twice reached a written conclusion before the user
caught it.

**The rule: a rate may only be compared to another rate when both
denominators are the same population, or when both are random samples of
it. Write the denominator into the sentence and the bad comparisons
become audible.**

The two live examples, both about the same quantity:

| written as | actually measures |
|---|---|
| "NMS costs 0.91% of ground truth in 6bba" | 0.91% of the **9.71% that were annotated** |
| "NMS costs 13.1% in Zebrahub" | 13.1% of **every cell, all labelled** |

and then, having already recorded that trap, the same entry compared
6bba's 0.0092 to 44b6's 0.0054 — 9.71% annotation coverage against
**0.77%**, a 12x stricter sieve. Both are "sparse", which felt like
enough. It is not: *equally named* is not *equally selected*.

**Sparse ground truth is a hand-picked sample, never a random one.**
Annotators label what they can see, and Zebrahub's recall by brightness
quartile (**0.52 dimmest, 0.74 brightest**) says visibility dominates.
So every sparse-GT rate is optimistic by an unknown amount, and the
amount differs per domain with the coverage.

Three practical consequences:

1. **Within one domain, differences are still clean.** Before-and-after
   on the same annotated set measures that set correctly. Report it that
   way — "0.92% of 6bba's *annotated* cells" — and it stays true.
2. **Across domains, only dense GT can be compared.** Zebrahub is the
   only arm whose denominator is the true cell population. In any
   experiment about what fraction of cells something loses, Zebrahub is
   not one of three equal arms; it is the measurement, and the sparse
   arms are context.
3. This is a different rule from *three domains, or it does not count*
   above, and neither implies the other. That rule is about whether an
   effect generalises. This one is about whether two numbers were ever
   the same kind of number.


## Cache the whole population, not the arm you are about to run (2026-09-09)

The user, twice in one night: *"思考，把44b6全跑一遍不行吗，你这分裂全挂0"*,
then *"你去把重训落地，这种只提供24部的资料的事情请不要再发生了"*.

**What went wrong, three times, the same way.** A cache kernel was scoped to
the movies the *next* experiment needed, and every later experiment then
inherited that scope as a hard limit:

| cache | scoped to | what it silently capped |
|---|---|---|
| `edgeprob` F4 pass | **24** eval movies | `edgeprob_rank2` could only train the shipped ranker on BASE rows — the F4 pool did not exist. The mismatch is still in production. |
| `hparacache` v1 | **24** bench movies | hparabench round 1 read hpara on 12 of 128 scored movies and concluded the opposite of round 2's 128-movie answer. |
| `hparacache` v3 | **140** (12 of 71 44b6) | hpara on 44b6 could only be scored on a draw holding **2 GT divisions**, so div_J was 0.0000 in both arms and the domain read "null" for a reason that was the draw, not the change. |

Each time the scoping decision was defensible on the day and wrong within a
week, because the cache outlives the experiment that motivated it and the
next experiment does not get to renegotiate it. `ep6bba` (09-06), `ep44b6`
(09-07) and `hpara44` (09-09) are all retrofits of the same omission.

**The rule.** When a cache kernel is about to run a model over a population,
the default is **the whole population that any bench in this repo scores** —
199 train movies for the competition sets, all 41 crops for zebrahub — not
the subset the current arm reads. Deviate only when the cost is genuinely
prohibitive, and when you do:

1. say in the kernel's docstring what the full population would have cost,
   in the same units as the partial run;
2. name, in the same paragraph, which comparison the partial cache forecloses;
3. treat any later number computed on the partial set as **provisional until
   the cache is complete** — the record now holds three cases where the
   partial answer had the wrong sign or the wrong magnitude
   (round 1's hpara verdict; `divbench3`'s +0.0020 against the full 71's
   +0.0014; ttabench's −0.0051 through a ranker that saw no tta4 rows).

The cost asymmetry is the whole argument: a cache pass is GPU-minutes per
movie and runs once, while a partial cache silently taxes every experiment
downstream of it and is discovered only by a contradiction weeks later.


## A log that will not DOWNLOAD looks exactly like a log that is empty (2026-09-09)

`kagglesdk`'s `kernels_output` writes the fetched log with

    out = open(os.path.join(target_dir, slug + ".log"), "w")

— no `encoding`, so it takes the platform default, which on this machine is
**gbk**. Every script in this project prints `⚠` (U+26A0), and cp936 cannot
encode it, so the download dies with

    UnicodeEncodeError: 'gbk' codec can't encode character '\u26a0'

**and no log file is written at all.** The failure is indistinguishable at a
glance from the zero-byte log a SIGKILLed session leaves, which is the one
diagnosis this project most needs to read correctly — `zhpara4` was nearly
recorded as "killed, no evidence" when in fact its log was 21 KB and ended in
`Killed`.

`tools/klog.py` re-execs itself under `-X utf8` when UTF-8 mode is off.
`PYTHONIOENCODING` does **not** fix this: it changes stdio, not `open()`'s
default. And `os.execv` cannot be used for the re-exec on Windows — it joins
the argv list with spaces and no quoting, so `D:\programming\cell tracking`
arrives as two arguments; use `subprocess`.

## Before pushing a run to CONFIRM a mechanism, try to falsify it from data you already have (2026-09-09)

`zhpara4` was pushed to confirm that a dense `N_pred × N_gt` cost matrix
kills the big zebrahub crops. Its probe fired and printed 17.7 GB for the
crop that dies — confirmation, read on its own. But `zhpara` v3's own output
CSV carried `n_gt` and `num_pred_nodes` for the 38 crops that **succeeded**,
and one of them needs **20.95 GB** by the same formula. The hypothesis was
falsifiable, for free, from a file already downloaded, before the kernel was
written.

The check costs one script and no compute:

- write down the number the mechanism predicts;
- find the rows already on disk where the mechanism predicts the OTHER
  outcome;
- if those rows exist, the mechanism is dead and the run is not needed.

This is the same failure as `measure-before-defending-a-mechanism`, one step
earlier: there the number was surprising and went unchecked, here the number
was *expected* and the counterexamples went unlooked-for. A confirming
measurement is the weakest kind, and a run designed only to confirm will
usually succeed.

## A killed kernel loses stdout. Diagnostics go to a FILE (2026-09-09)

Zebrahub has now OOM-killed four runs — `biohub-zh-pipeline` at 10 of 41
crops (progress.md:425), `zhpara` v1 at 9, `zhpara` v3 at 39, `zhpara2` at 9
— and **not one of them left a readable error.** Kaggle returns a
**zero-byte log** for a SIGKILLed session: stdout is not truncated, it is
gone. `zhpara2` printed `VmRSS` / `VmHWM` at every stage of every crop
specifically to catch this, and every one of those lines was lost with the
process.

`zhpara` v3 already encoded the right pattern for RESULTS — `_checkpoint()`
rewrote the per-crop CSV after every crop, which is why 38 crops of paired
data survived a run that died. The mistake was not applying the same rule to
DIAGNOSTICS.

**The rule.** Anything you would want to read after a kill must be written to
a file under `/kaggle/working`, flushed and `fsync`ed at the point of
writing. That includes:

- per-unit results (already standard here: `_checkpoint()`);
- memory/timing traces, and any "where was I" breadcrumb;
- the last successful step's name, so a kill points at a stage rather than a
  range.

`print(..., flush=True)` is for a run that FINISHES. It is worth nothing for
one that gets killed, which on the zebrahub domain is the common case.

**And artefacts on disk are evidence.** Two of this night's diagnoses came
from output files rather than logs: 39 GT geffs against 38 CSV rows put the
death inside crop 39, and the single leftover prediction geff named the crop
and proved the REF arm had already saved. Design the file layout so that a
half-finished run says where it stopped — write per-unit artefacts, delete
them when the unit completes, and the leftovers are the answer.
