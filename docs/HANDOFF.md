# HANDOFF — current state. Read this every session; `progress.md` is the log.

Updated 2026-09-11. Facts only. Every number here is copied from a dated
entry in `progress.md`; grep the ref or kernel name there for the source.

## Best submission

| | |
|---|---|
| LB | **0.956** |
| ref | 56487556 |
| kernel | `faustine649/biohub-submit-nc35` v1 (T4) |
| script | `submit_nc35.py` = `submit_nc.py` with the second-solve node cost 0.20 -> 0.35 (fork daughters and their chains exempt, hypx NC35X); `submit_nc.py` = `submit_prop.py` + a second pyscipopt solve with a node cost; `submit_prop.py` = `submit_img.py` + the uncapped, contrast-ranked orphan patch (DIV_DIVERGE_UM 1.5, caps x100) under `fork_gate_prop.pkl` (the image gate fitted on the patched fork population, 6bba + 44b6, p > 0.03) |
| its control | ref 56426751 = 0.955, `submit_nc.py` (node cost 0.20), kernel `biohub-submit-nc` v1 |

Final selection: leave on Kaggle's default (it takes the two highest
scorers). Daily submission cap: 5. Deadline: 2026-09-29.

Public leaderboard on 2026-09-19 08:30 UTC (after ref 56344592): top 0.974 / 0.973 / 0.968 / 0.967 / 0.966; 0.958 is
ranks 17-22, 0.955 ranks 31-37, our 0.954 ties ranks 38-44 (seven teams), 0.951 about rank 60. The public notebooks'
0.947 is no longer the bar; the top two moved up 0.004 in the last two days.

## Submission history

| ref | change | its own control | LB |
|---|---|---|---|
| 55707206 | D32 ranker, global solve | — | 0.921 |
| 55757064 | ranker pool 32 -> 187 | 55707206 = 0.921 | 0.920 |
| 55772253 | `SHORT_MIN` 6 -> 4 | 55757064 = 0.920 | 0.920 |
| 55780773 | + gate 0.15 -> 0.12 | 55772253 = 0.920 | 0.918 |
| 55817630 | gap budget 0.003 -> 0.02, candidate requirement dropped | 55772253 = 0.920 | 0.922 |
| 55889500 | F4: ep05 DeepCenter as 4th equal-weight heatmap | 55817630 = 0.922 | 0.925 |
| 55904403 | W1101: branch 3 to half weight | 55889500 = 0.925 | 0.924 |
| 55911058 | M15: ep05's missed points added as primaries | 55889500 = 0.925 | 0.917 |
| 55974245 | edgeprob: official edge head p_fwd/p_rev/h_fused as ranker columns, D175 +BIDIR ranker | 55889500 = 0.925 | 0.932 |
| 56000581 | submit_off: official detector nodes + head-only h_fused, no ranker | 55974245 = 0.932 | 0.895 |
| 56014734 | nb948-asis: the public "0.948" notebook forked unchanged | — | 0.935 |
| 56030644 | submit-f48: ILP candidate floor 0.10 -> 0.48 | 55974245 = 0.932 | 0.933 |
| 56055405 | submit-gate: fork gate `fork_gate_f4.pkl`, p > 0.03 | 56030644 = 0.933 | 0.943 |
| 56086066 | submit-joint: fork gate retrained on 6bba + 44b6 forks (`fork_gate_joint.pkl`) | 56055405 = 0.943 | 0.944 |
| **56135014** | **submit-hp: hpara placement** | 56086066 = 0.944 | **0.945** |
| 56135603 | submit-rk: F4-retrained ranker `edge_ranker_prod_f4_bidir.pkl` + `ILP_DIV_W` 0.35 | 56086066 = 0.944 | 0.944 |
| **56291359** | **submit-img: joint IMAGE fork gate (`fork_gate_jointimg.pkl`, 20 + 11 raw-image features at every fork)** | 56135014 = 0.945 | **0.951** |
| **56344592** | **submit-prop: uncapped contrast-ranked orphan patch (DIV_DIVERGE_UM 1.5) under `fork_gate_prop.pkl` (6bba + 44b6 image gate fitted on the patched fork population)** | 56291359 = 0.951 | **0.954** |
| 56399794 | submit-vt: vote candidates as attach-only primaries (>= 2 of 4 single maps agree, none within 7 um of a shipped detection, f4 >= 0.10; no candidate-candidate ILP pair; dangling dropped) | 56344592 = 0.954 | 0.953 |
| 56426751 | submit-nc: node cost 0.20 in the ILP, fork daughters and their chains exempt (hypx NC20XT), second pyscipopt solve | 56344592 = 0.954 | **0.955** |
| 56487556 | submit-nc35: the same with node cost 0.35 (nodes -3.1% vs submit-nc; 44b6_0b24845f -12%) | 56426751 = 0.955 | **0.956** |

Hidden-test output bytes (`total_bytes` in the submission metadata):
55707206 262,623,879 · 55757064 262,803,696 · 55772253 267,030,255 ·
55817630 268,154,477 · 55889500 270,740,009 · 55904403 273,340,932 ·
55780773 277,555,127 · 55911058 280,776,965 · 55974245 270,137,616 ·
56000581 280,141,150 · 56014734 204,966,451 · 56030644 266,263,495 ·
56055405 266,152,167. 56135014 and 56135603: hidden bytes not yet copied
here; on the visible four, 56135014 wrote 235,206 rows against the control's
234,761 (+0.19%) and 56135603 wrote 0.55% fewer than the control.

## Running now
- Nothing.
- Finished 2026-09-28: `biohub-flowrank` v1 (CPU, 406 min; entry "2026-09-28 flowrank launched"): six flow columns in the
  ranker add nothing (held-out AUC +0.0002 / +0.0003; FLCR vs R6CR adj_J -0.0009 [-0.0020, +0.0001]) and cost divisions
  (60 -> 54 true, a second daughter is off the flow); vs the shipped ranker score -0.0060 [-0.0110, -0.0013]. Not submitted.
- Final selection (proposed to the user 2026-09-28, the user's call): pick nc35 (56487556, 0.956) and the no-fallback
  Z + size gate (56611839, 0.955) by hand instead of Kaggle's default; Z is +0.0035 / +0.0071 locally over the same
  composition without z on 6bba / 44b6.
- Finished 2026-09-28: `biohub-ncsweep` v1 (CPU, 588 min; entry "2026-09-28 ncsweep launched"): the node cost above 0.35
  loses locally on the Z composition -- 0.50 score -0.0037 [-0.0072, -0.0004], 0.65 -0.0148 [-0.0201, -0.0095] -- so
  neither is submitted and no gate is refitted. The easiest quartile still gains (+0.0012 / +0.0040), the hardest loses:
  a per-movie cost is the residue, unbenched.
- Finished 2026-09-28: `biohub-flowgate` v1 (CPU, 495 min; entry "2026-09-27 flowgate launched"): the public 0.953
  notebook's flow gate as a pruning of our ILP candidates loses (FG7 adj_J -0.0013 [-0.0017, -0.0009], score -0.0063; FG9
  -0.0002): above the floor it removes 40 false pairs for 134 true ones and it cuts division edges (true divisions 59 -> 52).
  Residue: the flow prediction's distance is a stronger single column than the ranker's md_um (AUC 0.9795 vs 0.9608).
- Finished 2026-09-28: `biohub-coordhead` v1 (T4, 86 min; entry "2026-09-27 coordhead launched"): the public 0.953
  notebook's coordinate head gains 17% / 1% (44b6 / 6bba) from its own lattice start but ends at 1.43 / 1.84 um where our
  hpara placement is at 1.37 / 1.27; a head of its design trained on our pairs gains inside an embryo (1.27 -> 1.06) and
  not across (6bba head on 44b6: within 2 um 0.722 -> 0.704, carries +0.55 um of z). hrefine's reading again; no cache.
- Finished 2026-09-28: submission 56611839 (`biohub-submit-zsznf` v1, every fallback removed) LB 0.955 = submit_zsz's:
  the hidden rescore never took a fallback path. Later z submissions use `--no-fallback`.
- Finished 2026-09-27: `biohub-submit-zsznf` v1 (T4, 0.51 h; entry "2026-09-27 submit_zsznf pushed"): submit_zsz with every
  fallback removed; the visible run took no fallback path and wrote a submission.csv byte-identical to submit_zsz's.
  Submitted as 56611839.
- Finished 2026-09-27: `biohub-smoothchk` v1 (CPU, 0.4 min; entry "2026-09-27 smoothchk"): on the same annotated cells the
  shipped chain and the truth are the same edge in 98.8% of steps and have the same jitter (median 0.52 um per frame);
  forkvel's 'three times smoother' was a coordinate artefact (annotation voxels vs detector coordinates), corrected in
  place. The residue is a tail: at the 2% of steps with a jump >= 3 um per frame the linker is wrong half the time,
  two thirds swaps to a nearer smoother neighbour, one third breaks; 21 errors in 1,911 steps on 4 movies.
- Finished 2026-09-27: `biohub-twopass` v2 (CPU, 544 min; entry "2026-09-26 twopass launched"): seven multi-frame columns
  read off pass 1's solved tracks add nothing -- fold rankers held out 0.9932 / 0.9915 with or without them, TPCR vs R6CR
  adj_J +0.0000 [-0.0007, +0.0009]. The solved chain's velocity prediction equals the greedy chain's (md_tr AUC 0.964 vs
  md_um 0.961) and the solved acceleration is again worse (0.935). Not a submission.
- Finished 2026-09-25: `biohub-zhfork` v2 (CPU, 25 min; entry "2026-09-25 zhfork launched"): on Zebrahub's dense GT the
  pipeline forks correctly on 81 of 7,894 divisions (the detector loses the daughters) against 2,844 false forks, and no
  image column set separates better than the 20-column gate without images (AUC 0.73-0.79 on 81 true forks). Zebrahub
  gives the size gate no sign; the raw-image columns do not transfer as a family there.
- Finished 2026-09-26: submissions 56559928 (`biohub-submit-zsz` v1: Z + the 36-column size gate) LB 0.955 and 56559925
  (`biohub-submit-zxsz` v1: ZX + the same gate) LB 0.952, against submit_sz 0.954 and submit_nc35 0.956 (entry
  "2026-09-25 z chain launched"). Z's direction confirmed, ZX's not; best LB stays 0.956.
- Finished 2026-09-26: submission 56560340 (`biohub-submit-z` v1: Z + the 31-column gate, the control) LB 0.955 -- the
  same as Z + the size gate; the LB cannot see z (-0.001 vs nc35, +0.001 vs sz) or the size columns. Nothing running.
- Finished 2026-09-26: `biohub-fk44prop-z` v1 (CPU, 532 min): on 44b6 Z 10/6/16 0.9197 (+0.0096 [+0.0029, +0.0174] vs REF +
  jointimg), ZX 12/6/14 0.9259 (+0.0159 [+0.0060, +0.0280]); the five gate pickles written; the size columns are a wash on
  the gate's AUC (36 vs 31: 0.945 / 0.947 Z, 0.958 / 0.956 ZX).
  The user's timeout rule (2026-09-25): the triple stage has no wall-clock skip (a timeout fails the run), and the gate-
  failure fallback (forks kept with a WARN) is removed -- a gate that throws fails the run. The two older
  fallbacks (NC_DEADLINE_S 10 h keeps the shipped solution; a failed / non-optimal SCIP solve degrades to it) stay for THIS
  submission; if it finishes without a timeout, the next step the user asked for is a second submission with every
  fallback removed.
- Finished 2026-09-25: `biohub-hypz4` v1 (CPU, 234 min; entry "2026-09-25 hypz4 launched"): the two-pass z shape (first
  solve at node cost 0, its z daughters exempt in the costed solve) reads 58/56/67 score 0.9500 after the joint gate against
  Z10's 55/55/70 0.9485 and REF + joint's 0.9433 (+0.0067 [+0.0030, +0.0108] paired). ZX is the pick for the submission
  unless fk44prop_z's 44b6 arms reverse it.
- Finished 2026-09-25: `biohub-divscore` v1 (CPU, 12 min; entry "2026-09-25 divscore"): the two triple scorers trained on
  all 199 movies, out-of-fold AUC 0.86 / 0.88, S0 rule -9.00; `divscore.pkl` in bench_csv/divscore and on the kernel.
- Finished 2026-09-25: submission 56530848 (`biohub-submit-sz` v1; entry "2026-09-24 forkprop_sz launched"): the size gate
  LB 0.954 against submit_nc35's 0.956. Best LB stays 0.956 (submit_nc35, 31-column gate).
- Finished 2026-09-25: `biohub-hypz3` v1 (CPU, 425 min; entry "2026-09-25 hypz3 launched"): three levers on the division-
  variable ILP -- ranker probabilities in the scorer (worse: AUC 0.79 vs 0.88, 53/58/72 0.9470), a geometry scorer on all 121
  positives (53/55/72 0.9475), a 60-per-frame pool (52/58/73 0.9464), the z daughters exempt by a second solve (a no-op: the
  first pass already carried the node cost). Z10 stays the pick: 55/55/70 0.9485, +0.0052 vs REF + joint. Not shipped; the
  submission build (scorer pickles, in-kernel triples, gate refit chain) is the user's call.
- Finished 2026-09-24: `biohub-hypz` v2 (CPU, 372 min; entry "2026-09-24 hypz launched"): a division variable per
  divtrip triple in the shipped ILP. After the joint gate, Z10 (cost 0.55 - 0.10 per logit above -9, floor 0.10) reads
  56/55/69 div_J 0.311 score 0.9491 against the control's 50/49/75 0.287 0.9467: +0.0024 local, all division; the
  patches removed no z fork, the gate cut 2 true ones. Residue: 28-32 of 64 true triples with a variable still declined
  by the ILP. To ship: scorer A/B pickles, submit_z (triples computed in-kernel), the image-gate refit chain on the z
  population (~13 h CPU), then submit. The user decides.
- Finished 2026-09-24: `biohub-divtrip` v2 (CPU, 91 min; entry "2026-09-24 divtrip launched"): a division triple scored
  without the linker reaches AUC 0.86 (geometry) / 0.88 (+ image) out of fold by movie, and 35% / 51% of the reachable GT
  divisions are the top-1 / top-5 triple of their frame; but at one candidate per frame the pool holds 38% of the divisions
  at 0.3% precision (the shipped composition: 42% at 51%), so a per-triple division variable can only be a prior on the
  edge costs. The top-20 pool holds 55% of the GT divisions (the pipeline delivers 42%). The ILP bench with z variables
  (top-K pool, weight from scorer B, replacing the 0.55 toll + orphan patch) is designed in the entry, not built.
- Finished 2026-09-17: `biohub-lnbench` v1 (CPU, entry "2026-09-17 lnbench"): the rescue peaks admitted at 0.20 /
  0.25 move nothing (edge_J +0.0003 [-0.0001, +0.0008] / -0.0000; score -0.0011 / +0.0003, intervals contain
  zero). The local-normalisation line is closed.
- Finished 2026-09-17: `biohub-eplncache` v1 and `biohub-ep2lncache` v1 (T4, both OK 128/128: the two edge
  heads on the rescue set fdet_F4ln_; ep2lncache ran on the week's last 1.7 h of GPU).
- PREPARED (entry "2026-09-17 gatefit + submit_img"): `gatefit.py` (push after fk44img lands; metadata
  scratchpad/kernel-metadata.gatefit.json) then `submit_img.py` (metadata kernel-metadata.submit_img.json; control =
  submit_hp ref 56135014 LB 0.945). The hidden rescore costs no GPU quota (toolchain section 9); only the push's
  visible run does (~17 min of quota left on 2026-09-17; if the visible run is refused, re-push after the
  2026-09-19 00:00 UTC refresh). Read
  gatefit's 44b6 curve before pushing the submission: the 44b6 gate must not lose true forks at p > 0.03.
- Finished 2026-09-17: `biohub-lntrace` v1 (CPU, entry "2026-09-17 lntrace"): of the rescue peaks' 256 linkable
  edges, 128 die at the ranker (p median 0.07, nine in ten below 0.31: dim-to-dim pairs 5.7 um off the cell),
  24 are not candidate pairs, 19 lose their node, 2 to ILP competition, 9 to a nearer duplicate, 74 are true
  edges (placed within 2.8 um). Closed: the ranker, not the ILP, refuses the rescued cells.
- Finished 2026-09-18: `biohub-fk44prop` v1 (CPU, entry "2026-09-17 fk44prop"): on the 71 44b6 movies the uncapped,
  diverge-1.5 patch under its own image gate is flat against REF + jointimg (8/7/18 vs 8/8/18; score +0.0004
  [-0.0078, +0.0086], adj_J -0.0000 [-0.0002, +0.0001]); ungated it loses -0.0089. fork_gate_prop.pkl: n_train
  6bba 725 / 44b6 207 / tp 77. The shipping condition holds: no 44b6 loss, +0.0050 on 6bba.
- Finished 2026-09-18: `biohub-lnplace2` v1 (CPU, entry "2026-09-18 lnplace2"): of the 80 refused rescue peaks, 26 stand
  next to an annotated cell a shipped primary ALREADY holds (within 3.6 um) -- the ranker was right and lntrace's 128
  "lost" edges overcount; the other 54 are real annotated cells with no detection within 3 um at t, not interpolated
  (deviation from the neighbours' midpoint = control), persistently dim (raw contrast 0.17-0.19 in t-1 / t / t+1 against
  0.64 for detected cells), their own track detected at t+-1 only half the time. The detector line's target population
  is dim cells at a third of normal contrast; the raw image carries no position for them.
- Finished 2026-09-18: `biohub-lnplace` v3 (CPU, entry "2026-09-18 lnplace"): the raw image has NO placement signal for
  the rescue peaks -- intensity at the annotated centroid 0.1-0.2 over the local median (detected cells 0.6), a raw
  maximum within 3 um of it for 18% of the refused peaks and 4% of the linked ones; re-placing on the raw maximum is a
  5 um random move (refused 11% -> 31% within 3 um, linked 55% -> 18%, real detections 81% -> 45%). Closed: raw-image
  re-placement. The linked rescue peaks owe their 2.9 um to the normalised heatmap, so any placement residue is
  heatmap-side (vertex fit / learned offset on the normalised map), unpriced.
- Finished 2026-09-18: `biohub-forkprop2` v1 (CPU 655.6 min, entry "2026-09-18 forkprop2"; 119 of 128 movies replayed, the
  four-arm image-feature replay hit its budget): the prepared submit_prop composition (NOCAP_DV150 + the image gate with
  fk44prop's 44b6 forks, cut 0.03) beats what ships by +0.0039 [+0.0002, +0.0075] on those 119 movies; no knob beats it
  (cut 0.01 / 0.02 lose, 0.05 flat; divergence 1.0 / 0.0 flat to worse; wider geometry -0.0035 off zero). submit_prop's
  constants stand; nothing changes before Saturday's push.
- PREPARED 2026-09-17: `submit_prop.py` (entry "2026-09-17 submit_prop"; metadata scratchpad/kernel-metadata.submit_prop.json
  = submit_img's + kernel source biohub-fk44prop, T4) = submit_img.py + DIV_DIVERGE_UM 1.5 + caps x100 + contrast-ranked
  proposals + fork_gate_prop.pkl; the harness-vs-submission patch test passed, checkgen ok. fk44prop cleared it
  (no 44b6 loss). What remains is the GPU quota (spent: 162,258 s of 162,000 on 2026-09-17; refresh 2026-09-19
  00:00 UTC; a push on 2026-09-18 was refused with "Maximum weekly GPU quota of 45.00 hours reached", no version
  created). A background job of the 2026-09-17 session pushes it at 2026-09-19 00:05 UTC; if no `biohub-submit-prop`
  kernel exists after that, push by hand (`cp scratchpad/kernel-metadata.submit_prop.json kernel-metadata.json`, push).
  Then read the visible run (gate provenance line, patch line per movie, row count against submit_img's 235,230)
  and submit through the connector; control ref 56291359 = 0.951.
  DONE 2026-09-19 00:11 UTC: pushed, v1 COMPLETE, visible run healthy (rows 235,225 / nodes 120,651 / edges 114,574 against
  submit_img's 235,230 / 120,651 / 114,579; gate provenance line as expected; log bench_csv/submit_prop/submit_prop_v1.log).
  SUBMITTED 2026-09-19 01:01 UTC on the go-ahead: ref 56344592, scored ~08:30 UTC: LB 0.954, NEW BEST (+0.003 on its control
  56291359 = 0.951; the 6bba bench said +0.0039 [+0.0002, +0.0075], 44b6 flat). The hidden rescore took about 7.5 h.
- Finished 2026-09-19: `biohub-mapvote` v1 (T4 211 min, entry "2026-09-19 mapvote"): the agreed single-map maxima are
  annotated cells at 0.15x / 0.09x / 0.06x the shipped share at 2 / 3 / 4 votes (share falls with agreement: the maps agree
  on texture); 2 votes recover 31% of the missed annotated cells for +10% nodes, whose node term is ten times the edge ceiling.
  Closed as a scoring lever. The chain continues only for the ranker question (does it link the recovered cells):
- Finished 2026-09-22 (offline + `biohub-swapz` 4 min): the swap sites are NOT pre-division cells (an annotated division within
  3 frames in 1.2% vs 0.4% control; entry "2026-09-22 swapdiv"); by z profile they are a mixture, 44% one object on one plane,
  38% two objects >= 3 planes apart (entry "2026-09-22 swapz"). No mutual crossings among the swaps (entry "crossing").
  Everything on the swap side points at the detector (target shape / z placement), not the pipeline.
- Finished 2026-09-23: `biohub-swapvid` v1 (0.6 min, entry "2026-09-23 swapvid"): 24 xy-separated swap sites (dz <= 1 plane,
  a-b >= 8 um) + 8 baseline two-cell sites rendered as 7-frame sheets and GIFs for the eye, shuffled, key in
  `bench_csv/swapvid/swapvid_key.csv`. Read by eye: of the 24 far swap sites 7 are one nucleus (shading / mottling / a smooth
  big nucleus), 8 two nuclei, 9 undecidable; 2 of the 8 'two-cell' baseline sites read as one nucleus (that baseline is not clean).
- Finished 2026-09-23: `biohub-swapdip` v1 (4 min, entry "2026-09-23 swapdip"): the raw a -> b dip separates swap pairs from the clean
  annotated two-cell pairs (78% vs 34% flat) but cannot be a merge trigger: same-plane near pairs number 439k against ~1,600 swaps,
  the tightest rule fires 41 times per swap caught (break-even ~2). One-nucleus-two-detections is a detector-side class.
- Finished 2026-09-23: `biohub-mergegate` v1 (T4 20 min, entry "2026-09-23 mergegate"): a CNN over raw frames t-2..t+2 + graph features,
  FIX vs CASE2 out of fold by movie, AUC 0.790 against 0.793 for the graph features alone; under one case-2 break per fix only below 30%
  recall (160-240 fixes, +0.002-0.003 edge_J ceiling). The two-pass merge bench was not run (stop rule). Weights + oof table in `bench_csv/mergegate/`.
- Finished 2026-09-23: `biohub-swapms` v1 (9 min, entry "2026-09-23 swapms"): mean shift on the raw image from each detection of a pair; swap pairs
  walk together no more often than annotated two-cell pairs or random near pairs (g / f 1.5-2.2 at bandwidths 1.5-3 um, break-even ~7); the
  swap detections never reach A (3.0-3.5 um after the walk). No raw-image placement device is left untested on the swap side.
- Finished 2026-09-23: `biohub-focus3d-probe` v4/v5 (T4; weights from the public dataset `qiweiyin/focus3d-nuclei-runtime`; entry "2026-09-23 focus3d_probe"):
  FOCUS-3D on 8 frames of 6bba_ebff6e76 fragments the nuclei (865 instances per frame against 416 estimated, ~235 um^3 pieces), places
  centroids 3.5 um from the annotation (shipped detector 1.2), and puts a swap pair's a and b in one instance no more often than random near
  pairs (14-24% vs 12-25%); larger-radius settings lose instances rather than merge them. Not a label source, not a placement source, for 6bba.
- Finished 2026-09-24: `biohub-forkvel2` v1 (CPU 112 min, 128/128, entry "2026-09-24 forkvel2 launched"): forkimg's harness with
  velocity continuity (cont, cont2) and a raw-image half-max size (r_hm at mother / daughters + ratios). Velocity continuity KILLED
  in the gate: the pipeline's own true forks continue the mother at 1.66 um/frame (the ranker picks the smooth daughter, the patch adds
  the other), AUC 0.66, adding it -0.005 div_J, cutting on it -0.06. Size REAL as a feature (true daughters 3.1 um vs false 4.9-5.1,
  AUC 0.83) but on top of the shipped 31 columns +0.004..+0.043 div_J by fold split (2 of 5 splits distinguishable; the harness's
  2-fold cross: jointimg 0.3000 vs forkimg's 0.3029) -- the trees already reach size through contrast. Residue: a size-gate
  submission variant (submit_img's fork block + _fi_hm, joint gatefit), expected +0.000..+0.004 local; the user's call.
- Measured 2026-09-24: `biohub-forkvel` v2 (entry "2026-09-24 forkvel"; the user's idea): velocity continuity of the best branch,
  |v_branch - v_mother|, on detector coordinates: GT divisions median 4.05 um/frame (1.6% under 1.0, 8% under 1.5), the shipped
  graph's forks (4 visible movies, 76, unjudged) median 1.36 (35% under 1.0, 56% under 1.5), AUC 0.86; the linker's own chains
  jitter 0.63 against 1.95 for annotated cells. Not in the gate (its motion terms are angle and separation growth). Tested in forkvel2 (above): does not carry into the gate.
- Offline 2026-09-24 (entry "2026-09-24 fork gate: size and motion, offline"): on the 292 counted forks the daughter / mother radius
  ratio alone reads AUC 0.86 (real daughters smaller), the shipped trees already hold the raw radii, an explicit ratio moves the
  false forks kept at 50 true from 38 to 30 (inside the noise of 55 events); no velocity-continuity feature exists; a mask-based
  size or a velocity feature needs the forkimg loop (3.9 h CPU) + gatefit + divbench.
- Rendered 2026-09-24: `biohub-divvid` v3 (red / orange circles = the detections the matcher took as 2nd / 1st daughter and their
  back-walk on detections), same viewer. v1 (0.7 min, entry "2026-09-24 divvid"): the 22 timing divisions + 4 on-time controls as 12-frame
  GIFs of the raw image with the annotated lineage and the F4HP detections drawn on; viewer artifact
  https://claude.ai/artifact/CyWUniPfKnPhZn1E5e7qjr. The user's reading: of the 12 an eye can judge, 7 show a real second nucleus
  2-16 frames before the annotated division, 5 show no split; 10 undecidable (deep, noisy, crowded). Geometry does not separate the
  two classes. So the timing bucket is mostly the annotator recording the division later than the image shows it; the one unmeasured
  cue for the annotator's moment would have been the daughters' own brightness / size. Control #23 carries a GT lineage error (the
  annotated mother chain follows another cell). Then `biohub-divtwice` v1 (entry "2026-09-24 divtwice", GT + detections, no linker):
  no earlier division on any annotated ancestry, but the annotated mother is 0-5 frames long in 13 of 22 (lineages annotated from the
  division on); the linker's long early chains (-16 to -55) have no detection persisting at the spot (0-4 frames): edges through other
  cells; the detection matched as the second daughter in #9 / #11 / #3 walks back to an object 12-15 um from the mother, but that
  detection is up to 5 um off the GT daughter and the user's eye sees a real early split there (the walk likely followed a
  neighbour's point: the eye is the authority on the image); the GT is asymmetric (d1 on the mother, d2 9-13 um away) in six cases; d2's detection sits 3-6 um off in 11 of 22 (the true daughter
  likely undetected, the match a neighbour's detection = the 5a steal from the other side). Pipeline side: detection-limited daughters, linker chains through other cells and annotation picks. Image side (the eye):
  ~7 of 12 judgeable cases are real splits recorded 2-16 frames late by the annotator. Open: did the detector / matcher follow the
  real second nucleus in those 7 (render the matched detection + back-walk on them)? Left besides: 15 orphans + 3 one-frame-early
  cases, each under the LB's resolution.
- Finished 2026-09-22: `biohub-valley` v1 (entry "2026-09-22 valley"): the raw-image valley between the two nuclei of an early
  division is already as deep 5-6 frames before the annotated frame as between on-time daughters after it; no threshold
  fires at t0 +- 1 (3 of 16). The timing bucket (22 of 125) has no trigger in geometry or intensity; closed as measured.
- Finished 2026-09-22: `biohub-divtime` v1 (entry "2026-09-22 divtime"): of 125 GT divisions the shipped graph has 63 on time; the
  64 misses are 22 detection-limited, 22 timing (second daughter tracked >= 2 frames before the annotation, ~10 um from the
  mother from the start), 18 near misses (15 orphans not adopted / pruned, 3 one frame early). No distance trigger for a
  delay rule; the timing bucket needs an image cue. Cheap residue: adopt orphans starting one frame early (3 cases).
- Finished 2026-09-22: `biohub-nclost` v1 (entry "2026-09-22 nclost"): NC20XT loses 47 of 18,457 true edges on 44b6 (0.25%), all in
  2-4-node components and track ends -- the same population the cost is meant to delete; no exemption separates them. Per-domain
  cost keyed on the dataset prefix is meaningless for the LB: the LB, public and private alike, is scored on a separate set of
  movies that has nothing to do with the 4 sample movies a kernel run shows (their names, prefixes and per-movie node cuts say
  nothing about the scored movies); a setting is only ever read as one number on the LB (2026-09-23). One global cost per
  submission: 0.20 shipped (LB 0.955), 0.35 = ref 56487556 LB 0.956, NEW BEST.
- Finished 2026-09-21: `biohub-nc44` v1 (entry "2026-09-21 nc44"; ERROR in the gate cross after stage B, results rebuilt from the
  log): on the 71 44b6 movies NC20XT is neutral (adj_J -0.0005 edge-weighted, +0.0003 [-0.0016, +0.0024] per movie; edge_J
  -0.0017 for nodes -2.1%), so do not raise the cost globally. Floors 0.30 / 0.10 are flat in aggregate, +-0.04..0.07 per
  movie; the selected-share key separates them weakly. Next: the division line.
- Finished 2026-09-23: ref 56487556 = `biohub-submit-nc35` v1 (entry "2026-09-23 submit_nc35"): LB 0.956, NEW BEST (+0.001 on
  56426751 = 0.955). The node cost 0.20 -> 0.35, nothing else; visible nodes -3.1% vs submit_nc, -4.7% vs submit_prop. Three points
  on one line now (submit_vt +0.76% nodes -0.001; nc20 -1.7% +0.001; nc35 -3.1% +0.001): the hidden scorer pays for fewer nodes.
  Not yet benched or submitted: a higher cost (0.50); hypx benched NC35X as the top of its sweep, so a 0.50 arm needs a local
  number first (nc44 said 44b6 is neutral to the cost; the edge losses are at track ends). Submitted only with the user's say.
- Finished 2026-09-21: ref 56426751 = `biohub-submit-nc` v1 (entry "2026-09-21 submit_nc"): LB 0.955, NEW BEST (+0.001 on 56344592 =
  0.954). Node cost 0.20 in the ILP with fork daughters + chains exempt; visible nodes -1.7%. With submit_vt (-0.001 for
  +0.76% nodes) the pair says the hidden scorer pays for fewer nodes. submit_nc is now the control for what follows.
- Finished 2026-09-21: `biohub-hypx` v1 (CPU 420 min, 128/128; entry "2026-09-21 hypx"): the node cost with REF's fork
  daughters exempt keeps the node-term gain and gives the divisions back: NC20X adj_J +0.0011 [+0.0006, +0.0017] with the
  gated divisions identical to REF (49/51/76); NC20XT (daughters + their chains exempt) adj_J +0.0013 [+0.0008, +0.0018],
  score +0.0025 [+0.0011, +0.0040], ILP divisions = REF's 2,371, edge_J flat, final nodes -1.0%, 113 movies up / 14 down.
  Decision pending: port NC20XT into submit_prop (pyscipopt is in the pack; two solves per movie) as the LB test of the
  node term with the rewarded sign (submit_vt's mirror). Expected +0.001 on the LB.
- Finished 2026-09-20: ref 56399794 = `faustine649/biohub-submit-vt` v1 (entry "2026-09-20 submit_vt"): submit_prop + the vote
  candidates as attach-only primaries (f4 >= 0.10), +0.76% nodes on the visible set. LB 0.953 against the control 56344592 =
  0.954: -0.001, the alarm case (local +0.0006 in sample, +0.0023 out of sample; LB negative). Under the reading the user set
  before the result, the negative says the node term is charged for these nodes; the pure node-deletion test (hypilp NC20:
  -1.7% nodes, adj_J +0.0013 [+0.0008, +0.0019] locally, never submitted) is the mirror with the rewarded sign. submit_prop
  (56344592, 0.954) stays the best and the selected submission.
- Finished: zhattach (entry "2026-09-20 zhattach"): out of sample the attach-only vote arm is positive on the metric's sparse view,
  XB adj_J +0.0021 [+0.0011, +0.0032], XBF (f4 >= 0.10) +0.0023 [+0.0013, +0.0034] over 41 Zebrahub crops (in sample +0.0006
  [-0.0001, +0.0013]); the gain tracks the detector's miss rate (-0.0002 at 2.6%, +0.0016 at 6%, +0.0049 at 12%) and goes
  negative only where the detector over-predicts by 18%+. Free linking (VT) gains 4x the edges and pays it all in nodes. The
  f4 gate is free: ship XBF's admission if the arm ships. Expected LB effect +0.0005 to +0.0015.
- Finished: zhvote (entry "2026-09-20 zhvote"): mapvote's vote on the 41 Zebrahub crops (out of sample for all four weights,
  dense GT). The shipped detector misses 7.1% of dense GT (6bba sparse in-sample 1.8%). Candidates k >= 2: +14% nodes, GT share
  0.32x the shipped primaries' (6bba 0.13x; above 0.13x in 36 of 41 crops, median 0.29x), 17% stand on a MISSED cell (6bba ~5%
  after the sparse-to-dense conversion), recover 40% of the missed cells (6bba 31%). Zebrahub's domain shift exceeds the
  hidden set's, so the 2.5x is an upper bound; the direction is what the in-sample bench cannot show. An f4 >= 0.10 gate on the
  candidates keeps 83% of the on-missed ones for 62% of the count. zhattach (2026-09-20) then scored the arm on Zebrahub's
  sparse-thinned GT: XBF +0.0023 [+0.0013, +0.0034], the gain tracking the detector's miss rate. Ported and submitted as
  submit_vt (ref 56399794): LB 0.953 against 0.954. The line's LB test is done; see the submit_vt entry for the reading.
- Finished: vtfill (entry "2026-09-20 vtfill"): the vote candidates as patches on REF's graph, no re-solve. Two-sided holes
  (end at t-1, start at t+1, both p >= 0.48): 1,983 in 128 movies, 4 true edges, adj_J +0.0001. One-sided extensions at
  p >= 0.85: +154 true edges for 16k nodes, adj_J +0.0004 [+0.0000, +0.0007], score +0.0000 (133 new patch forks eat it);
  at 0.95: adj_J +0.0002. Candidates in the gap-close pool: nothing. The vote-candidate line ends here: every device on it
  (mapvote, vttrace, vtattach, vtlost, vtstatic, vtfill) sits within +/-0.001 of adj_J and no score interval excludes 0.
  2/3-frame bridges on vote candidates were not built: 55 + 33 annotated interior gaps exist (bridgerank), the candidates
  cover ~1/3 of missed cells, and the 1-frame two-sided version found 4 edges.
- Finished: vtlost (entry "2026-09-20 vtlost"): where the true edges go when the vote candidates attach. VT17XB: lost 289 /
  gained 356. Direct candidate effects: 19 thefts (on weak true edges, p med 0.3-0.5; thieves are annotated cells 8-9 um from the
  displaced node), 118 gains, net +96. Re-solve churn (shipped-shipped swaps/relinks + nodes leaving/entering the final graph):
  lost 267, gained 238, net -29, sign varies by movie; the worst movies are churn-only. Swapped-out true edges are strong (p 0.81),
  replacements weaker (0/55 better). The funnel's 182 = 118 new + takeovers of matches REF already had. A no-re-solve hole-filling
  rule (candidate between a track end and a track start, both edges above the floor) would keep the +96 and drop the churn;
  ceiling ~+100 true edges, adj_J of order +0.001.
- Finished: vtstatic (entry "2026-09-19 vtstatic"): do the vote candidates repeat in place? 30% have a repeat within 1 um in an
  adjacent frame, 9% run 3+ frames, ~1k run 6+ frames (annotated 0). Annotated share 0.0041-0.0104 across displacement bins (top
  = 0.19x shipped); best rule ALONE keeps 34% of candidates for 44% of the recovered cells. Candidates with ANY adjacent-frame
  candidate within 7 um = 152k = vttrace's 144k survivors: texture survives by candidate-candidate links at cell-like steps,
  which is what vtattach withholds. No rule goes to a bench arm.
- Finished: vtattach v2 (entry "2026-09-19 vtattach"): candidates may only ATTACH to existing tracks. VT17X: 44.6k candidates
  in the final graph (vttrace 144k), G_tp 387 (408), edge_J +0.0021 [+0.0011, +0.0031] but adj_J -0.0004, score -0.0001 vs
  REF + joint: the node term (node_r -0.0231 -> +0.0043) eats the identity gain. VT18X flat (score +0.0012, interval covers 0).
  The under-count-bonus question (does the server pay it?) is NOT tested by submission: the LB record already answers it in
  practice: every node-adding device lost on the LB except the placement move (hpara), and bridging gained less on the LB
  than locally. A clamped bonus would have made them look BETTER on the LB than locally, not worse.
- Finished: vtattach v3 (entry "2026-09-19 vtattach"): the hole-filling arms. VT17XB keeps 8,000 of 224k candidates (those
  attached on BOTH sides), 182 true edges, node_r -0.0198: edge_J +0.0009 [+0.0002, +0.0016], adj_J +0.0006 [-0.0001, +0.0013],
  score +0.0023 [-0.0007, +0.0056] vs REF + joint (two thirds of it the retrained division gate). VT18XB (3,935 kept) moves nothing.
  Per movie: up 58 / down 70, median -0.0001; net edge_tp only +70 (attachments displace ~110 true shipped edges).
  Shipping it means mapvote (4 maps, 211 min on 128 train movies) + the two vote edge heads inside the submission kernel.
  v2's XB arm was invalid (patch_final's gap-close ids from len(nodes)+1 collided after the drop); v3 uses max(nodes)+1, REF identical.
- Finished 2026-09-19: `biohub-epvtcache` + `biohub-ep2vtcache` (T4, OK 128/128) and `biohub-vttrace` v1 (CPU 230 min, entry
  "2026-09-19 vttrace"): the ranker links 408 of the vote candidates' 2,132 linkable true edges at p 0.89 (edge_J +0.0035
  [+0.0022, +0.0050], the largest identity gain of any detection-side device), 63% die at the 0.48 floor (p median 0.047);
  the 224k admitted candidates cost adj_J -0.0025 through the node term (144k survive as texture-to-texture chains), score
  -0.0031. Residue: withhold candidate-to-candidate pairs so a candidate can only attach to existing tracks; unpriced.
  Was: the vote
  detection cache fdet_F4vt_ = F4HP + the points where >= 2 of the four single maps have a floor-0.05 maximum within 2.5 um
  of each other and no shipped primary within 7 um; reads their annotated share by vote count against the shipped 5.5%.
  Chained behind it by a background job of this session: `biohub-epvtcache` + `biohub-ep2vtcache` (T4, the two edge heads on the
  vote set), then `biohub-vttrace` (CPU ~5 h: lnbench's harness with arms VT17 = 2 votes / VT18 = 3 votes plus lntrace's
  per-arm funnel; metadata scratchpad/kernel-metadata.{epvtcache,ep2vtcache,vttrace}.json; logs to bench_csv/<kernel>/).
  If the chain died, push them by hand in that order; each needs the previous kernel's output.
- Finished 2026-09-19: `biohub-hmshoulder` v1 (T4 31 min, entry "2026-09-18 hmshoulder"): fitting the host's heatmap blob and
  reading the residual finds a second blob at 86% of the missed cells' hosts and at 72-82% of ordinary cells, 10 um away in a
  random direction; within 3 um of the missed cell at 18%, AUC 0.47-0.58, no price at any recall. Closed: blind heatmap-side
  placement of the missed cell. Residue: the missed cells are NOT dark-background cells (94% above the frame's p40, 24% dimmer
  than their hosts, response 3x smaller), and at 45% of them a single map (b2 29%) has a floor-0.05 maximum within 3 um where
  the mean map has none (12%) -- unpriced base rate.
- Finished 2026-09-17: `biohub-lnrank` v1 (CPU, entry "2026-09-17 lnrank"): rankers retrained WITH the rescue peaks'
  rows are the shipped rankers (104 extra positives in 150k rows, AUC equal to the 4th decimal); the 128 rescue
  edges at the floor stay (129, p median 0.07), 3 more true edges taken, score -0.0004 [-0.0038, +0.0032] vs
  REF + joint. The refused rescue edges sit 5.7 um from the annotated cell, the linked ones 2.8 um: PLACEMENT,
  not exposure, decides. A detector line pays only if it places recovered cells within ~3 um (residue: re-place
  the rescue peaks on the raw image before the ranker; a placement-supervised head).
- Finished 2026-09-17: `biohub-forkprop` v1 (CPU, entry "2026-09-17 forkprop"): the orphan-fork patch with NO cap
  and diverge 1.5, pruned by the image gate retrained on its own forks, beats REF + jointimg by +0.0050
  [+0.0012, +0.0089] on the 128 6bba movies (61/36/64 against 55/42/70, adj_J unchanged); every capped arm is
  within its interval of zero, the contrast ranking itself does nothing, ungated the arm loses -0.0103. The 44b6
  half is unmeasured (fk44prop, running).
- LANDED 2026-09-17: ref 56291359 = LB 0.951 (NEW BEST, +0.006 over its control 0.945; the public notebooks' 0.947 is passed): `faustine649/biohub-submit-img` v1 COMPLETE (T4 visible run; submission.csv 235,230 rows, nodes
  120,651, edges 114,579 against submit_hp's 235,206 rows; forks 69/69/22/129 -> removed 23/69/13/104; image
  features 1-4 s per movie): submit_hp + the joint image fork gate
  (fork_gate_jointimg.pkl from `biohub-gatefit` v2, trained on the dataset `biohub-forktables`); entry
  "2026-09-17 gatefit + submit_img"; control ref 56135014 = 0.945.
- Finished 2026-09-17: `biohub-gatefit` v2 (CPU): out of fold at p > 0.03, 6bba 48/50/77 -> 52/44/73, 44b6
  6/10/20 -> 8/4/18 with the image columns.
- Finished 2026-09-17: `biohub-fk44img` v1 (CPU 127 min, entry "2026-09-17 fk44img"): 44b6's 88 counted forks
  carry the same contrast signal (AUC 0.68-0.71, 9 TP); forks_44b6img.csv is the joint gate's 44b6 half.
- Finished 2026-09-17: `biohub-forkimg` v2 (CPU 3.9 h, measurement complete, died in the last pickle block;
  entry "2026-09-17 forkimg", `bench_csv/forkimg/`): raw-image CONTRAST at the mother and both daughters
  separates true from false forks (AUC 0.83-0.84 each; true forks 3x over background, false 0.9x). Joint
  gate + 11 image columns: 53/50/72 div_J 0.3029 score 0.9457 vs REF + joint 49/51/76 0.9433, paired
  bootstrap +0.0025 [+0.0004, +0.0049], edge term untouched. NEXT: fk44img (44b6 forks with image columns)
  -> a gatefit kernel writes fork_gate_jointimg.pkl on Kaggle's sklearn from forks_dw55.csv + forks_44b6img.csv
  -> submit variant = submit_hp.py + the 11 features in _gate_prune (crop reads, ~30 min per 128 movies).
- Finished 2026-09-17: `biohub-lnpersist` v1/v2 (CPU, 0.4 min each, entry "2026-09-17 lnpersist"): prices the rescue peaks' admission rules
  (persistence in adjacent frames / hole in a shipped track) per score bin before any GPU is spent;
  v1 read: persistence selects AGAINST the cells (annotated share 0.0053 vs 0.0088 unfiltered; a hole in a
  shipped track 0.0115; shipped detections 0.0548), so lnbench is regenerated score-only, arms LN20 + LN25
  (`make_lnbench.py 0.20,0.25`; ceiling +0.001 to +0.002 edge_J). v2 division view: 1 of the 403 annotated
  rescue peaks is a missing daughter, 0 are mothers, so the rescue does nothing for the division term.
- Finished 2026-09-17: `biohub-lnrescue` v1 (200 min T4, `LNRESCUE OK`, entry "2026-09-17 lnrescue"):
  45,803 rescue peaks (+2.0% of primaries), 403 annotated; cells as often as shipped detections only
  above score 0.30 (160 peaks), half at 0.25, a fifth at 0.20. Ceiling +0.001 to +0.003 edge_J; the
  bench prices the noise, with temporal persistence of new peaks as the threshold-free filter.
  PREPARED, not pushed: `tools/bench/make_eplncache.py` -> eplncache.py and `make_ep2lncache.py` ->
  ep2lncache.py (heads on fdet_F4ln_, metadata scratchpad/kernel-metadata.{eplncache,ep2lncache}.json,
  T4, source biohub-lnrescue; push both after lnrescue lands, ~1.5-2 h each); `make_lnbench.py <taus>` ->
  lnbench.py (dupsup + REF on F4HP + arms LN<tt> on the rescue set with features built once per movie,
  keep mask by score; toy test passed; metadata kernel-metadata.lnbench.json; CPU ~7.5 h with 3 arms).
  Choose the taus from lnrescue's score-bin table before generating the bench.
- Finished 2026-09-17: `biohub-locnorm` v1 (62 min T4, `LOCNORM OK`, entry "2026-09-17 locnorm",
  `bench_csv/locnorm/`): a LOCAL input normalisation (background subtraction sigma 15 um) lets the same
  four networks find 19-22% of the cells the shipped pass misses (the first instrument to reach that
  bucket), but as a replacement it drifts 0.7-1.5% of clean singles beyond 7 um and half its new peaks
  are noise. Next: a RESCUE cache (F4HP + BGSUB15 peaks >= 7 um from every shipped detection, score
  recorded), heads for the pairs touching new nodes, a dupsup bench with score-threshold arms.
- Finished 2026-09-17: `biohub-blobsplit` v1 (7.6 min, `BLOBSPLIT OK`, entry "2026-09-17 blobsplit",
  `bench_csv/blobsplit/`): the cells the detector misses (1,817 annotated, 1.6%, host 8.1 um away) are
  NOT visible in the raw image as separate objects: a raw local maximum within 2.5 um of the missed cell
  in 7% of sites, 25% dimmer than the host, no dip on the way. Blob shape separates host from single
  nuclei at AUC 0.70-0.77 (it reads the crowded surroundings), 16 wrong splits per cell found at 30%
  recall (break-even ~2). Mask / raw-image routes are closed for both the swaps and the missed cells.
- Finished 2026-09-16: `biohub-nodeledger` v1 (130 min, 128/128, `NODELEDGER OK`, entry "2026-09-16
  nodeledger", `bench_csv/nodeledger/`): the node-count lever is EMPTY on 6bba. No GT-free bucket of
  final nodes (score decile, component size, track position, density, degree, inserted) sits on
  annotated cells below the 0.24% deletion break-even; the lowest-score tenth is at 0.97%. Every
  deletion rule priced from the ledger loses edge_J faster than the node term pays (best +0.0002
  adj_J). Annotation density (0.9-20% across 6bba movies, ~fixed budget per movie) CANCELS in the
  trade: gain/loss ~ 0.045 J / k, k = the bucket's cell-likeness relative to average; a bucket pays
  only at k < 0.04, ours bottom out at 0.18, and the rules lose most on the sparsest movies. Not
  domain-dependent; no 44b6 run needed. Ledger note: official FP = edges whose SOURCE is matched.
- Finished 2026-09-16: `biohub-mgpass` v1 (218 min, 128/128, `MGPASS OK`, entry "2026-09-16 mgpass",
  `bench_csv/mgpass/`): the two-pass merge loses in both arms. MG0 (78.7% of parent pairs merged)
  edge_J −0.0430; MGB (graph veto: both children alive, not both full tracks; 19.3% merged) edge_J
  −0.0153 [−0.0184, −0.0123], 110 of 124 movies worse, 2,016 true edges lost for 335 false removed.
  The merge line (hmerge, hypilp, mergefeat, mgpass) is closed on the bench in every form tried;
  the perfect-selection ceiling was ~ +0.009 edge_J and nothing selects it.
- Finished 2026-09-16: `biohub-mergejudge` v1 (106 min, 128/128, `MERGEJUDGE OK`, entry "2026-09-16
  mergejudge", `bench_csv/mergejudge/`): of the 9,724 one-end pairs 47.6% have a mate carrying a persistent
  track of its own (an unannotated neighbour OR a duplicate peak of the same cell: one-cell pairs show
  the same in 48.6%; hmerge's loss channel either way), 35.9% have a mate with <= 1 edge, 5.5% would put the
  midpoint beyond 7 um, 3.7% are fixable swaps. Real fixable one-cell swaps <= 177 (+271 parallel
  duplicates); oracle ceiling ~ +0.009 edge_J. "Both ends carry full tracks" separates case 2 (79%)
  from one-cell pairs (27%) on REF's own graph, no classifier. A bench of a graph-vetoed merge needs a
  two-pass kernel (solve -> decide -> merge -> re-solve); not built, sign not promised.
- Finished 2026-09-16: `biohub-mergefeat` v1 (141 min, 128/128, `MERGEFEAT OK`, entry "2026-09-15
  mergefeat", `bench_csv/mergefeat/`): among hypcache's 233,819 parents at d <= 8.5 um the
  both-annotated pairs are 1,226 one-cell vs 74 two-cell (94% one cell): the two-cell class is
  not the loss channel and cannot be learned (out-of-fold AUC 0.61, p ~ 1 everywhere, every
  threshold fires on 96-99% of the 9,598 one-end-annotated pairs = hmerge's rule again). The
  merge line's judge has to be edge-level on the one-end-annotated pairs from REF's solution
  (annotated end's GT edge missed + partner carries the GT neighbour = a fixable swap; annotated
  end's edges already correct = a merge only displaces). mergecache.py / mgbench_f4mg90.py /
  mergeveto.py are written and toy-tested but NOT pushed: a tau on mergefeat's p would merge
  ~98% of parents. Metadata in scratchpad/kernel-metadata.{mergecache,mgbench90,mergeveto}.json.
- Finished 2026-09-16: `biohub-bridgegate` v1 (257 min, 128/128, `BRIDGEGATE OK`, entry "2026-09-15
  bridgegate", `bench_csv/bridgegate/`): the span-2 selector veto on the shipped span-1 bridges
  moves scored edges by nothing (15,056 of 15,237 bridges have an unannotated endpoint; 181
  labelled, tau 0.2 removes 29 negatives / 9 positives). Its score delta (+0.0017 at tau 0.2)
  is div_J through the refit joint gate, not monotone in tau, and the size of the gate's own
  refit spread (`tools/gatenoise.py`: 5-95% width 0.0015 of score). Not shippable on this
  evidence. The DeepCenter-response veto of the public 0.947 notebook is a different signal
  and is not built (~5 h GPU; quota ~8 h until 2026-09-19).
- Finished 2026-09-15: `biohub-hypdiag` v2 (106 min, 16 movies, `HYPDIAG OK`;
  entry "2026-09-15 hypdiag", `bench_csv/hypdiag/`): of the 247 true edges the parents cost,
  185 join two children that are both still present (the parents pull the solve away from
  true edges AROUND them), 45 lose an endpoint, 17 break at the parent; divisions account
  for 2. The parents gain 43 true edges on the 16 movies (0.37% of those carried): the
  ceiling of the hypothesis-node line is ~+0.003 edge_J even with every loss removed. The
  one constraint the ledger supports (a parent may only take edges its children could have
  taken) is priced at that ceiling and not built; the user decides whether it is worth a
  10 h kernel. Constraint 1 (no parent at a fresh division) is refuted as a mechanism;
  a 'weak children' gate at 0.5 keeps 89% of what the solver already selects.
- Finished 2026-09-15: `biohub-hypilp` v3 (663 min, budget guard: 100 of
  128 movies; entry "2026-09-14 hypilp", `bench_csv/hypilp/`): C0 reproduces REF exactly on
  100 of 100 (the pyscipopt re-statement of the shipped ILP is verified); the parent
  hypotheses (HY10/20/35) are taken at 9.5% of offered pairs whatever the node cost, 96% on
  unannotated cells, and cost 601-1,228 true edges for 279-465 false ones removed, halving
  the ILP's divisions: edge_J -0.0040 [-0.0061, -0.0023], score -0.0074 at the mildest
  setting. The node cost alone (NC20/NC35) gains adj_J +0.0013 / +0.0027 through the node
  term only, no score gain once gated. Hypothesis-node line: the machinery works, the
  parents lose under the shipped probabilities; a narrower parent rule (only where neither
  child has a confident continuation; never at fresh divisions) is unpriced. The HY solves
  are 8x slower than REF's (exclusivity constraints).
- Finished 2026-09-15: `biohub-skipilp` v1 (660 min, budget guard: 114 of 128 movies; entry
  "2026-09-14 skipilp", `bench_csv/skipilp/`): span-2 edges inside the ILP at full weight
  wreck tracks (adj_J -0.044 / -0.022 at tau 0.5 / 0.8); at half weight they act as bridges,
  edge_J +0.0022 [+0.0011, +0.0034] but 35,769 inserted nodes and the fork gate eat it
  (adj_J +0.0002, score -0.0012). Only 1,070 labelled span-2 pairs are real holes, so the
  ceiling of any node-inserting span-2 device is ~+0.003 edge_J. The post-NMS repair pool
  has nothing within 20 um (median) of a hole's midpoint; whether a suppressed peak sits
  there is unmeasured (the pool cannot hold one) and needs a read of the map at the holes.
  Bridging at the linker: the untested variant is now tested and does not pay.
- Finished 2026-09-14: `biohub-ultrackbench` v3 (5.6 h, died on movie 2;
  entry "2026-09-13 ultrackbench", `bench_csv/ultrackbench/ultrackbench_v3.log`): the
  real Ultrack pipeline (pip ultrack 0.8, CBC, unet-daxi, sparse_zebrafish config) on
  6bba_05b6850b picks 47,116 nodes for 6,136 cells, places them 3.66 um from the
  annotated centroids (ours 1.49) and 3.35 um at swap sites (ours 2.75), edge_J 0.36
  (ours 0.99), 53 false divisions; on the dense 6bba_05db0fb1 the network calls 84% of
  the volume foreground and the first 25-frame ILP window (1.77 M binaries) found no
  solution in the hour. Reading: the public weights fail before the hierarchy gets a
  chance, so "public Ultrack + hierarchy reproduces the annotation" is dead; "the
  hierarchy on OUR heatmaps" is untested and would have to be built with a pruned
  hypothesis set (our detections + merged parents), never the full area hierarchy.
  Before it: `biohub-simviewnorm` v1 + v2 (entry "2026-09-14 simviewnorm"): official
  competition scaling misses 45% of annotated cells; the DeepCenter scaling finds them
  (3-9% missed) but inflates the foreground (22% of the volume, ~750 segments/frame);
  under no scaling does simview place swap-site cells better than our maps.
  `biohub-ultrackprobe` v1 (entry "2026-09-13 ultrackprobe"): Ultrack's UNet at ONE
  watershed level places swap-site cells at 2.5-2.6 um (ours 2.69), sees two blobs at
  2/3 of swap sites.
- Ultrack lead (2026-09-13, competition discussion + the Ultrack paper, Nature Methods
  2025): the GT is the paper's sparse-labelling experiment (dense H2B-mNeonGreen =
  our images; sparse H2B-mCherry nuclei tracked by Ultrack and curated = the
  annotations). Public recipe: torchscript UNet (foreground + contour), sparse_zebrafish
  config.toml (segmentation threshold 0.4, min_area 300, max_area 25000, area
  hierarchy; linking max_distance 10, max_neighbors 5; appear/disappear/division -0.1,
  power 4, window 25). Rules allow public external data and pretrained models. Status
  after ultrackprobe / simviewnorm / ultrackbench: the public weights do not fit these
  images (foreground 17-84% of the volume, 7.7x node count); the machinery is not
  refuted but has no usable network here. Dataset `faustine649/ultrack-unet-weights`.
- Ultrack lead (2026-09-13, from the competition discussion + the Ultrack paper,
  Nature Methods 2025): the competition GT is the paper's sparse-labelling
  experiment (dense H2B-mNeonGreen channel = our images; sparse H2B-mCherry
  nuclei tracked by Ultrack and curated = the annotations). The recipe is public:
  torchscript UNet (foreground + contour), `ultrack_supplementary/configuration/
  sparse_zebrafish/config.toml` (segmentation threshold 0.4, min_area 300,
  max_area 25000, area hierarchy; linking max_distance 10, max_neighbors 5;
  tracking appear/disappear/division -0.1, power 4, window 25). The public
  `zebrafish_embryo.ome.zarr` (522 frames, 505x2217x2170, scale 1.625/0.40625)
  is that experiment's dense channel. Rules allow public external data and
  pretrained models. `ultrackprobe.py` measures, on hplace's site frames (first
  6 movies), the distance from the annotated centroid to the nearest Ultrack-style
  segment centroid (single-level watershed approximation), against hplace's 2.69
  (f4, swap) / 1.27 (ctrl). If it lands near 1 um at swap sites, the next step is
  Ultrack's segmentation as the detection source for the whole bench.
- Finished 2026-09-13: `biohub-hplace` v1 (21 min, `HPLACE OK`, anchor exact;
  `bench_csv/hplace/`, entry "2026-09-13 hplace"): per-map placement against
  the annotated centroid. At the 1,643 swap sites the four F4 maps and their
  mean all sit at 2.7-3.0 um median (control cells 1.3-1.6), within 0.14 um of
  each other, nearest map per site spread evenly; an oracle best-of-four gains
  0.65 um. All four see two maxima within 7 um at 68-88% of swap sites (17-22%
  at ordinary cells). v2 (peak coordinates, `tools/hplace_cluster.py`): at 51% of
  swap sites the four peaks form one cluster (still 2.1 um from A), at 49% they
  split between two places ~4 um apart (A 2.5 um from the nearer); the mean's
  peak coincides with one map's peak (0.33 um), lands between them at 7%.
  Choosing among existing heatmaps is not a placement lever, and averaging is
  not what displaces the swap-site peak.
- Finished 2026-09-13: `biohub-hmbench` v1 (`HMBENCH OK`, `bench_csv/hmbench/`,
  entry "2026-09-13 hmbench"): saddle-keyed pair merging (HM90) LOSES edge_J
  -0.0082 [-0.0107, -0.0058], adj_J -0.0049, score -0.0097 against the shipped
  hpara placement; REF + joint 0.9334 vs 0.9433. HM80 not run. `biohub-ephmcache`
  v1 / `biohub-ep2hmcache` v1 (heads on the merged points, 140 caches each) and
  `biohub-hmerge` v1 (203.5 min, anchor exact; HM90 merged 5.35% of primaries,
  entry "2026-09-12 hmerge") are the inputs to it.
- Last finished 2026-09-12: `biohub-hsaddle` v3 (`bench_csv/hsaddle/`, entry "2026-09-12
  hsaddle"), `biohub-dupsup` v1 (`bench_csv/dupsup/`, entry "2026-09-12
  dupsup"), `biohub-swapimg` v2 (`bench_csv/swapimg/`, entry
  "2026-09-12 swapimg"), `biohub-divadmit` v1 (`bench_csv/divadmit/`, entry
  "2026-09-12 divadmit").
- Last finished 2026-09-11: `biohub-bridgerank` v2 (`bench_csv/bridgerank/v2/`,
  entry "2026-09-11 bridgerank"), v1 (`bench_csv/bridgerank/`), `biohub-zhbridge`
  v3 (`bench_csv/zhbridge/`), `biohub-divmiss` (`bench_csv/divmiss/`), `biohub-ttabench2`
  (`bench_csv/ttabench2_F4_tta4.csv`); 2026-09-10: `biohub-rkedge` (`bench_csv/rkedge/`).

## Local bench, 128 6bba train movies, ungated REF (DIV_W 0.55, disappearance 1.5, floor 0.48)

| cell | ranker trained on | points | div tp/fp/fn | div_J | adj_J | score | CSV |
|---|---|---|---|---|---|---|---|
| A ships | BASE rows | refine | 50/249/75 | 0.1337 | 0.9133 | 0.9266 | `bench_csv/rkpool_BASE.csv` |
| B retrain | F4 rows | refine | 52/225/73 | 0.1486 | 0.9141 | 0.9290 | `bench_csv/rkpool_F4.csv` |
| C hpara | BASE rows | hpara | 55/237/70 | 0.1519 | 0.9146 | 0.9298 | `bench_csv/hparabench_F4HP.csv` |
| D both | hpara rows | hpara | 50/222/75 | 0.1441 | 0.9138 | 0.9282 | `bench_csv/rkhp_F4_F4HP.csv` |

Harness anchors on the 12-movie 6bba holdout: D187 pkl adj_J 0.9216 on
refine points / 0.9204 on hpara; +BIDIR pkl 0.9261 / 0.9252. Aggregate
per-movie CSVs with `tools/benchagg.py` (the scorer's weighting), never a
plain mean.

## Assets

| what | where |
|---|---|
| production scripts | `submit_hp.py` (best), `submit_joint.py`, `submit_rk.py`, `submit_gate.py`, `submit_f48.py` |
| shipped ranker | `edge_ranker_ep175_bidir.pkl` (`biohub-edgeprob-rank` v2); `edge_ranker_ep175v3_bidir.pkl` (`biohub-edgeprob-rank2`) is the same model |
| F4-retrained production ranker + its BASE control | `edge_ranker_prod_f4_bidir.pkl`, `edge_ranker_prod_base_bidir.pkl` in `biohub-rkprod` output |
| fork gates | `fork_gate_f4.pkl` (`biohub-divbench3`), `fork_gate_joint.pkl` (`biohub-divbench8`) |
| F4 detections, all 199 train movies | `biohub-fusecache` (24), `biohub-fuse6bba` (116), `biohub-fuse44b6` (59) |
| head caches on F4 points | `biohub-edgeprob` (24), `biohub-ep6bba` (116), `biohub-ep44b6` (59); second head `biohub-edgeprob2` |
| hpara detections and head caches, all 199 | `biohub-hparacache`, `biohub-hpara44`, `biohub-ephpcache`, `biohub-ep2hpcache`, `biohub-ephp44` |
| tta4 first-head caches, all 199 | `biohub-ttacache` (140), `biohub-ttacache2` (59); second head never cached at tta4 |
| the full source list a bench needs | pull `biohub-rkhp`'s metadata (16 kernel sources); do not type it from a docstring |
| bench scripts | `divbench*.py`, `rkpool.py`, `rkhp.py`, `rkedge.py`, `ttabench2.py` in the repo root; generators `tools/bench/make_*.py`; pairing readers `tools/*pair*.py` |
| per-movie results and fork tables | `bench_csv/` |
| holdout lists | `HOLDOUT_6BBA` (12) and the seeded 44b6 draw (12) in `rkhp.py`; the D175 pool is everything else; `holdout-split.md` for the public weights' own splits |

## Open
- hrefine (2026-09-23, entries "2026-09-23 hrefine launched"): the placement head -- a CNN on the raw crop round each F4HP detection predicting the
  offset to the annotated centroid -- is KILLED for the hidden test. Out of fold within an embryo it helps (6bba 1.27 -> 1.08 um, 122 of 128 movies;
  44b6 1.37 -> 1.25 with 51% of cells placed worse), across embryos it hurts in both directions (a 44b6-only head on 6bba 1.27 -> 1.44, 22 of 128
  movies gain; a 6bba-only head on 44b6 1.37 -> 1.40, 21 of 71). Mechanism measured: the annotation sits at a whole-embryo z offset from the image's
  nucleus centre (+0.44 um on 6bba, ~0 on 44b6, per movie -0.6 to +0.8 inside each), a property of the acquisition (the annotation comes from the
  other fluorescence channel) no crop of the imaged channel can reveal for a new embryo; the remaining per-cell corrections do not transfer either.
  Even the embryo's own mean offset, if known, is worth 0.04 um on 6bba and nothing on 44b6. De-biasing checked against the literature (v2 rows):
  the per-movie constant is 5-7% of the label-offset variance (fixed-effect centring = v3 can move at most that), the offset is flat across brightness
  (importance weighting for the selection bias would re-weight cells whose offsets do not differ), and the image explains at most 5-14% of the offset
  even in-embryo (the rest is label noise ~ one z plane); the cross-embryo failure is a change of p(offset | image), which no re-weighting repairs
  without target-embryo labels. No v4. Zebrahub check DONE (hrefinezh v1, 41 crops, 809k primaries against dense labels): joint head 1.391 -> 1.540
  um p50 (64% placed worse, 0 of 41 crops gain), 6bba-only head 1.670, 44b6-only head 1.352 (47% worse; its whole gain is its small negative z
  constant matching Zebrahub's -0.43 um label offset, xy part neutral); three label sources, three z constants (+0.44 / +0.03 / -0.43). The plain
  intensity centroid beats every head on Zebrahub (1.319) and loses on 6bba: placement 'truth' belongs to the labeler. Line closed on three
  embryos. v3 (targets centred per movie before training) DONE: the constants are gone, the joint head keeps most of its in-embryo gain
  (6bba 1.27 -> 1.16, 44b6 1.37 -> 1.22), the 44b6-only head still hurts 6bba (1.27 -> 1.42, 66% worse), the 6bba-only head moves 44b6's median
  0.10 um down while placing 52% of cells worse and fewer within 2 um: a redistribution. hrefinezh v2 (v3 weights on Zebrahub): joint
  1.391 -> 1.439, 6bba head 1.488, 44b6 head 1.333 (45% worse; its gain is a -0.25 um z it emits on Zebrahub's look, matching the
  label offset's sign by chance), every xy-only variant within 0.03 of before; the plain centroid (1.319) beats all. CLOSED on three
  embryos: killed claim = a CNN placement head over F4HP detections improves placement on an unseen embryo. Next for the LB: the
  node-cost line (see "Finished 2026-09-23" below). Unpushed, gated: ephrcache.py / ep2hrcache.py / hrbench.py (the head-cache + bench chain); fdet_F4hr_ caches for all 199 in biohub-hrefine v2.
- Forum, read 2026-09-23 evening, detector TRAINING specifically (entry "2026-09-23 forum read: training a detector"; topics 741749, 732103,
  738217, 740145, 738773 + hengck23's public e2e notebook source): no self-trained or re-trained detector is reported lifting a 0.94+ score.
  Sparse-point training peaks at epoch 5-10 for everyone (nusrati: adj 0.40 -> 0.07 by epoch 120, peaks/frame 1,800 -> 2,100+; zhincez: fine-tuning
  the public detector on sparse labels -0.0161), which is dccache3's ep05 > ep12 seen from outside. Synthetic pretraining (hikarukai) is +0.012
  on his hold-out at a 0.939 pipeline and inflates nodes. Dense FOCUS-3D pseudo-labels give recall 1.0 detectors (hengck23, Satwik) that
  over-predict nodes (1.05-2.36x the estimate) and place at the FOCUS centre, not the annotation (hengck23: Kaggle points sit on Ultrack-segment
  "cell corners"; his refine head 1.91 -> 1.57 um); Satwik's LB with such a detector 0.83; hengck23 posts no LB. The one detector-side item
  with a measured precedent that matches this record's residue: a refine / dzyx head trained on the annotated centroids over the existing maps.
- Forum, read 2026-09-23 (27 detection / node-count / plateau topics, Opus subagent, numbers as posted): nobody above 0.94 reports a detector other
  than a 3D U-Net heatmap (no Cellpose / StarDist / FOCUS-3D / proximity targets / duplicate suppression / z placement anywhere). The one detector
  change that moved a score: synthetic pretraining (hikarukai, 741749: 80 epochs on 497 synthetic sequences then 60 on 175 real, hold-out 0.9146 ->
  0.9269, pipeline 0.939; it inflates the node count); wider / deeper / longer-trained nets, heatmap-loss variants, 9-frame windows: neutral. Node count
  is the term everyone sees on the LB (hikarukai: 'node-count calibration predicted my LB movement far better than missed detections'; justinch123,
  742266: ten single-knob changes all lost, the loss tracking the predicted node count; mige551: -1.7% nodes -> +0.002). CV vs LB: r ~ -0.2 in the
  0.93-0.95 band (hikarukai); +0.02-0.04 per fold -> +0.000 LB (pavelsavchenkov). Division heads: +-0.001-0.002 LB; divisions off costs 0.022.
  External data named by rank 1 (sersasj, 741386): the Janelia/linajea embryo, same spacing. GT: ~5% of disputed edges wrong (timkrige), an 8.9 um GT
  jump on byte-identical frames (729082); no fix reported. Reading: the record's node-term and alarm findings are the forum's consensus; the swap-side
  analysis here goes beyond anything posted.
- Public notebook `anvithpothula/biohub-0-953-lb-original` (215 votes), read 2026-09-27 by a code diff against the 0.947 stack
  (sjlee101 lf-dctta020-sectta1): the same models and settings plus two changed values (DeepCenter safe-division threshold
  0.20 -> 0.25; secondary edge-feature TTA weight 1.0 -> 0.75) and four added stages: (1) a frozen-feature coordinate head
  (`v1284_head.pt`: Linear 224 -> 32 -> 3 on the U-Net feature map, shift capped at 2 um, trained on 4,136 detection-GT pairs
  from 20 movies, comment claims -10% centroid error held out by movie; hrefine tested this class and it does not transfer
  across embryos); (2) a flow-guided relink (median displacement of the 12 nearest matched neighbours within 40 um predicts
  each cell, gate 7 um; flowvel read < 0.001); (3) readmit of free peaks >= 0.965 within 4 um of an open track end
  (lnrescue / lnpersist: such peaks are cells five times less often than shipped detections); (4) a 1-3 frame gap filler
  through peaks >= 0.5 on a straight line, capped at 3% nodes (gap2strict: +-0.002, signs disagree). No ablation, the local
  validator switched off, the only 0.953 in the title; the author's own team sits at 0.956 on 2026-09-27. Leaderboard that
  day: #1 0.978, #50 0.963, #100 0.959, our 0.956 about #190 in a 50-team band; the 0.96+ wave is dated 09-26/27, not a
  public notebook. Extracted cells: scratchpad nb953.py / nb947.py / cd.txt (session-local).
- Forum, read 2026-09-20 (topics 742064, 739685, 738217): (1) Eric (rank 72): the public 0.947 stack's secondary weights
  (pilkwang seed314159) were trained on ALL 199 train videos per their split_manifest; DeepCenter on 71; the 50ep pack ships no
  manifest but is reported the same. Any hold-out drawn from train is IN-SAMPLE for those detectors; his sub-voxel refinement
  gained +0.0078 on 12 validation videos and -0.002 on the LB twice. Our 128-movie bench inherits this for the pilkwang and
  xiaoleilian weights (and dctrain-175 if its 175 include the bench movies): detection-side fixes measured locally are measured
  where the detector has memorised the videos. (2) 'Problems with edge connection' (Mark rank 25, Yubo): every smarter motion
  linker is exactly +0.001 on the LB whatever the CV says (+0.012 CV -> +0.001 LB); hengck23: ask whether the repaired error
  will occur in the hidden test at all. (3) hengck23 in the focus3d thread (09-12..09-20): e2e node detector + link transformer
  on dense FOCUS3D labels, val edge_J 0.902 without ILP; Ultrack-style multi-hypothesis over whole trajectories as the missing
  third stage; the node count is what he is now fighting ('a model that doesn't over-estimate num of nodes'); Ultrack's
  foreground/boundary probabilities estimate N_est well; another user: detector on dense labels recall 1.0 but linker 0.83 LB.

1. rkedge read: D vs C is 254 true edges lost against 224 gained, all dim-target (score_j < 0.50) near-tie edges; 40% under the floor, 60% lost to ILP competition. No ranker-side fix identified.
2. Loss map of the shipped arm on 128 6bba movies (rkedge + divmiss, 2026-09-11): edge FN 5,698 = ~2,164 endpoint undetected + 2,438 below the 0.48 floor (fast, dim, crowded) + 1,805 swaps (56% two detections inside one annotated child's match radius, 44% to an unannotated detection) + 927 ILP-refused above floor; edge FP 4,305, 3,286 from the ILP at median p 0.84 (the earlier "69% to dead-end targets" was an artefact of the edge table, which only carries labelled sources: 67% of FP targets are unannotated and invisible there, and 94.5% of the annotated ones continue). Swaps have no dissenting signal: ranker p 0.80 vs 0.15, image head prefers the true child in 4-6%, track smoothness prefers it in 9% (swapfeat, 2026-09-11). Divisions 125 GT: 51 TP ungated; lost 70 = ~19 detector-blind, ~10 primaries dropped by the ILP, 20 5a, 11 8b (diverge < 2.25 um), 10 other. 4.5% of annotated cells have no peak at any score.
3. Bridges beyond one frame (bridgerank v1/v2, 2026-09-11): the distance rule at span 2 loses adj_J -0.0022 [-0.0028, -0.0016]; a selector trained on Zebrahub's dense labels (20,944 pairs) ranks 6bba's labelled pairs at AUC 0.93-0.94 with no wrong pick above p 0.5 but accepts only 31% of true bridges at 0.5 and 6% at 0.8 (6bba gaps are 2x longer per frame). Score flat at every threshold: at 0.5 the node penalty of 2,388 bridges with an unannotated endpoint equals the 40 edges gained (adj_J +0.0001 [-0.0002, +0.0004]); at 0.8, 11 edges, 0 FP (+0.0001 [+0.0000, +0.0002]). Ceiling of any selector on these proposals: 297 edges = +0.0026 adj_J before node cost. Residue: 511 annotated 2/3-frame gaps exist but only 88 are (end, start) pairs; the rest continued through a wrong link (a linker question).
4. Tested 2026-09-12 (divadmit): re-admitting dropped primaries as childless second daughters recovers 0 of 70 lost divisions (7,159 admits, cap-bound: 147k frame-cap refusals) and costs adj_J -0.0021 [-0.0025, -0.0017] through the nodes it adds; lowering DIV_DIVERGE_UM loses true divisions ungated (55 -> 54/46/40 at 1.5/0.75/0) because extra proposals displace the true ones; DV150 + joint -0.0008 [-0.0036, +0.0018]. Residue: an admit pass ranked by the ranker's p(mother -> q) with the pool cut before the cap; unpriced. Drawing daughters from the sub-threshold pool: 2 divisions, not worth a run.
5. Edge-feature TTA closed on the matched test: tta4-trained, tta4-deployed ranker loses adj_J -0.0042 [-0.0055, -0.0030] to the noTTA one on 128 movies (ttabench2). Second head never cached at tta4 (untested).
6. The retrain read LB 0.944 = its control; the leaderboard's three decimals cannot resolve its local +0.0023.
7. hpara + retrain together: local 0.9282, below either alone (0.9298 / 0.9290). Not submitted.
8. 5a steal case, PRICED by dupsup's 5a table (2026-09-12, F4HP, 128 6bba; `bench_csv/dupsup/steal5a.csv`): 20 lost divisions have their second daughter q already linked from another node r. Ranker p(mother -> q) median 0.07 (p25 0.01 / p75 0.71) against p(r -> q) 0.93 (0.85 / 0.95); of the 10 cases with both probabilities, the mother beats the competitor in 0. Distance mother -> q median 9.9 um, r -> q 2.45 um. 6 competing edges are annotated at both ends and all 6 are TRUE GT edges; 3 cases have sister distance 0 (q duplicates the linked daughter). No margin rule takes any of these without taking far more true edges. Closed.
9. Edge ledger (edgeledger, 2026-09-12): no post-link rule on the final graph clears the Jaccard break-even (~0.48 precision): below-floor free-end-to-free-start joins 0.24-0.45, one-frame extensions 0.41, short fragments already swept. The remaining misses are upstream: 1,805 swaps between two detections around one annotated cell, separated 5.4 um in xy and 3.7 um in z (swapgeom 2026-09-13; all signals agree on the wrong target), 2,438 below-floor edges with false-pair geometry, ~2,000 with an undetected endpoint. The swap bucket (~+0.03 adj_J ceiling) is detector PLACEMENT in dense regions (swapplace, 2026-09-12): swap detections sit 3.6 um from the annotated centroid against 1.2 um normally, two detections ~5 um apart around one annotated cell, identity decided by a fraction of a z plane. swapimg (2026-09-12) read the raw image there: one centroid flanked by two detections ~7 um apart, intensity flat across the pair, no raw peak at the centroid; annotated nuclei are almost never within 7 um, so such a pair is one scored cell. dupsup (2026-09-12) suppressed the pair before the ILP at 5.5 / 6.5 / 7.5 um and lost edge_J -0.0025 / -0.0115 / -0.0337 (all intervals exclude zero): the pairs are not removable duplicates; the 'annotated nuclei are never within 7 um' inference was a sparse-annotation artefact. hsaddle (2026-09-12): the HEATMAP does separate them: saddle between the two peaks 0.88 at swap pairs vs 0.45 between two annotated cells; a saddle-keyed merge (>= 0.9, <= 9.5 um) takes 43% of swap pairs, 4% of two-cell pairs, lands 1.65 um from the centroid. hmerge + hmbench (2026-09-12/13) applied that merge blind to every primary pair of the 128 movies: 118,820 merges against 1,805 swap edges in total, so at most 1.5% of merges could be swap fixes; the bench lost 708 true edges and gained 236 false ones, edge_J -0.0082 [-0.0107, -0.0058], adj_J -0.0049, REF + joint 0.9334 vs 0.9433. The swap bucket stays open; what is closed is any blind pre-ILP merge or suppression of near pairs (dupsup by distance, hmerge by saddle). hplace (2026-09-13): no single F4 map places swap-site cells better than the mean (all at 2.7-3.0 um, oracle best-of-four 2.0), and all four show two maxima around the one annotated centroid at 68-88% of swap sites: the placement error is upstream of the fusion. Untested: the saddle as a feature restricted to the linker's own uncertain sites; whether the second maxima at swap sites are unannotated real nuclei or split responses. A placement-aware retrained detector is NOT a ready route: dctrain's converged retrain (epoch 12) was below xl in every domain (dccache2/3, 2026-08-29) and only the undertrained ep05 snapshot helped, as one of four averaged maps; an offset head would need dense centroid supervision and the annotation is 9.7% of 6bba cells / 0.77% of 44b6, sparsest in the crowded regions where the swaps are. lnrank (2026-09-17): rankers retrained with the rescue peaks' rows do not link them either; the refused rescue edges are 5.7 um off the annotated cell against 2.8 um for the linked ones, so both large pots (swaps, undetected) are placement.
10. Local changes are uncommitted; the user has not asked for a commit.

## Rules

`working-agreement.md` (progress entries are six fields; reuse a kernel's
pulled metadata; every GPU kernel pins `NvidiaTeslaT4`), `kaggle-toolchain.md`,
`engineering-rules.md`, `reference.md` (metric, code map), `holdout-split.md`.
