# Biohub – Cell Tracking During Development (Kaggle, 2026)

My entry to the Kaggle competition
[Biohub – Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
(29 June – 29 September 2026). The task: find every cell nucleus in 3D
light-sheet movies of a developing zebrafish embryo, and link the detections
across time into a tracking graph, cell divisions included.

## Result

- Final standing: **1552nd of 3,947 teams**, private score **0.9143**
  (the selected submission, `submit_zsznf.py`).
- Best public-leaderboard score: **0.9561** (`submit_nc35.py`).
- Best private score among all my submissions: **0.9197** (`submit_vt.py`),
  which was not selected.

Why the public and private scores diverge is covered under
"Post-competition analysis" below.

## Start here

- [`src/submit_zsznf.py`](src/submit_zsznf.py): the final submission, the
  whole pipeline in one Kaggle kernel.
- [`src/edgeprob_rank.py`](src/edgeprob_rank.py): trains the edge ranker that
  the final submission loads (`edge_ranker_ep175_bidir.pkl`): gradient-boosted
  trees on 18 geometric pair features plus three columns from the official
  edge model, scored by AUC.
- [`src/ranker_fine.py`](src/ranker_fine.py): an earlier, shorter version of
  the same training, with the 18 geometric features only. Easier to read
  first; `pair_features()` builds the features.
- [`src/viz_cells.py`](src/viz_cells.py) and
  [`src/division_recovery.py`](src/division_recovery.py): the shared modules
  that most scripts import.

## How the pipeline evolved

The starting point was a verbatim reproduction of a public notebook (LB 0.917,
see Credits): a 3D U-Net detector and a frame-to-frame linker. From there, each
submission changed one thing against the previous best. The main steps:

| Script | What it changed | Public | Private |
|---|---|---|---|
| [`submit_ranker.py`](src/submit_ranker.py) | Replaced the linker with a learned edge ranker (gradient-boosted trees, 18 features) feeding an integer linear program (ILP) | 0.9213 | 0.8907 |
| [`submit_f4.py`](src/submit_f4.py) | Detection from four heatmaps: the U-Net ensemble plus a retrained DeepCenter model | 0.9251 | 0.9038 |
| [`submit_ep.py`](src/submit_ep.py) | The official edge model's forward and reverse probabilities as extra ranker features | 0.9328 | 0.9066 |
| [`submit_gate.py`](src/submit_gate.py) | A classifier (the "fork gate") that decides which candidate cell divisions to keep | 0.9433 | 0.9103 |
| [`submit_img.py`](src/submit_img.py) | The fork gate retrained on two embryos, with raw-image features at every fork | 0.9519 | 0.9163 |
| [`submit_prop.py`](src/submit_prop.py) | Orphaned track starts re-attached as division daughters, filtered by a gate fitted on them | 0.9545 | 0.9168 |
| [`submit_vt.py`](src/submit_vt.py) | Extra detections where at least two of the four heatmaps agree | 0.9534 | **0.9197** |
| [`submit_nc35.py`](src/submit_nc35.py) | A per-node cost in a second ILP solve, which drops weakly supported detections | **0.9561** | 0.9137 |
| [`submit_zsznf.py`](src/submit_zsznf.py) | Final selection: cell-size features in the fork gate, division variables in the ILP | 0.9550 | 0.9143 |

[`docs/HANDOFF.md`](docs/HANDOFF.md) lists every submission, including the
changes that did not help.

## Repository layout

```
src/                     all the code, in one flat folder
  submit_*.py            one Kaggle kernel per submission
  make_*.py,
  tools/bench/make_*.py  build a new script from its control by exact, asserted
                         string replacements, so the two differ in one place
  viz_cells.py,
  division_recovery.py   shared modules, imported by most scripts
  other *.py             experiments and diagnostics, one entry each in docs/progress.md
  tools/                 local analysis scripts, tests and Colab helpers
  kag.py                 wrapper around the Kaggle CLI, used for every push
  kernel-metadata.json   Kaggle kernel settings for the next push
docs/                    the project log (see "How this was built")
docs/media/              two 6bba datasets rendered as video by src/render6.py
third_party/             copies of two public notebooks (see Credits)
```

`src/` stays flat on purpose: the builders and tools find each other by
relative paths, so run every command from inside `src/`. The documents in
`docs/` were written while this code still sat at the top of the repository,
so a script they name is now under `src/`.

## Running it

The competition data is not in this repository (it is released under CC0;
download it from Kaggle). Every script runs as a Kaggle kernel, and
`kernel-metadata.json` picks the script, its inputs and the accelerator. On
Kaggle, the kernels import the shared modules from a code dataset uploaded
from this folder, and load trained models from the outputs of earlier
kernels.

```bash
cd src
# set "code_file" in kernel-metadata.json to the script you want, then
py kag.py kernels push -p .
```

`docs/kaggle-toolchain.md` covers the full workflow and its traps.

A local smoke test needs no competition data, but it does need zarr 3 and
blosc2. The synthetic data only proves the plumbing works; the numbers it
prints are meaningless.

<!-- TODO (me): run these commands in a clean environment and pin the
versions that work. -->

```bash
pip install "zarr>=3" blosc2
cd src
python make_fake_data.py
python viz_cells.py inspect --root ./fake_data
python submit_baseline.py score --root ./fake_data --limit 8 --per-frame 30
```

## How this was built

I built this with Claude (Anthropic's AI assistant) as the coding agent. Claude
wrote the code and the documents in `docs/`. The documents are addressed to the
next AI session, which is why they, and many commit messages, refer to "the
user". My part was directing the work: deciding what to try next, contributing
ideas, setting up the working process (`docs/working-agreement.md`), checking
results, and choosing what to submit. Nearly every commit carries a
`Co-Authored-By: Claude` line.

## Post-competition analysis

Cell tracking rewards continuity: a cell should stay one unbroken track from
frame to frame. Late in the competition I followed the public leaderboard into
more and more extreme settings, and two of the directions I pushed did not
carry over to the private test set.

**Removing cells.** The node-cost steps (`submit_nc`, `submit_nc35`) deleted
1.7% and then 3.1% of the detections. Each gained about 0.001 on the public
leaderboard and lost on the private one (0.9168 → 0.9155 → 0.9137).
`submit_vt` went the other way and added detections: it was 0.001 lower on
public, and the best of all my submissions on private (0.9197). The public
split rewarded reporting fewer cells; the private split did not.

**Division features.** Early on I assumed that a daughter cell is always one
of the mother's high-scoring link candidates. The orphan-fork patch was built
on that assumption: it picks an extra cell as the second daughter of a
division. When I later rendered clips of real divisions, many daughters were
not high-scoring candidates of the mother at all, because a division pushes the
daughters apart. Up to that point the reasoning held. The mistake was the next
step. Watching rendered divisions, I saw that true daughters were smaller than
their mother and false ones about as big, so I added cell-size features to the
fork gate (`submit_sz`, then the final `submit_zsznf`). The observation was
right; the measurement behind the features was not.

The competition provides no masks, so the size came from the raw intensity:
the region around a nucleus that is brighter than halfway between the local
background and the peak, converted to a radius. Nothing could confirm that this
region is the nucleus, and the readings are hard to believe. Of the 66 true
divisions among my pipeline's forks on the 6bba training embryo, the
measurement made both daughters larger than their mother in 11, and the two
daughters together more than twice her volume in 19; only 22 came out between
0.7 and 1.3 times her volume. A nucleus does pack its chromatin differently
around a division, so the volumes need not add up exactly, but a spread like
this is hard to explain by biology alone, and without masks there was no way to
tell how much of it came from the cell and how much from the measurement.

On average the direction was right: the two daughters together came to a
median 1.13 times the mother for true divisions and 2.10 for false forks, which
usually join the mother's own track to a neighbour. So the features carried a
weak signal under a large error for any single division. Their local gain
was +0.0012 (90% interval −0.0010 to +0.0032), inside the noise, and five
other ways of measuring size did no better. I shipped them anyway, and they
did not move the private score. The image-based gate before them
(`submit_img`), whose useful signal was contrast, did carry over (private
0.9113 → 0.9163), so the problem was the size measurement, not image features
as such.

**Comparison with the top solutions.** Most of the top solutions retrained
the detector instead of using the public weights, so they are hard to compare
with mine. One top competitor did use the public weights, and the clearest
difference is in validation. The small models in their decision pipeline were
always trained on one training embryo and validated on the other. I worried
that there was not enough data, so I mixed the two embryos and split each of
them into training and validation parts. A mixed split cannot show whether a
feature transfers to a new embryo. It is not what hid the size problem,
though: when I later trained the size gate on one embryo and tested it on the
other, it passed, because the measurement behaves the same in both training
embryos. The flaw was in the measurement itself, and no split of those two
embryos could have shown it.

**What I would do differently.**

- Validate every small model across embryos: train on one, validate on the
  other, even if that leaves less data. A mixed split cannot show whether a
  feature transfers.
- Prefer features that stay the same across embryos (motion, geometry,
  consistency over time) over features that describe how one batch of cells
  looks.
- Check a measurement I built myself against something it must satisfy before
  using it. A size measure should put two daughters together near their
  mother; mine often did not, and that was visible on the training data.
- Not let the public leaderboard push the pipeline toward extremes that break
  continuity, such as deleting cells for a 0.001 gain.
- Hedge the final selection: next to the public-leaderboard pick
  (`submit_zsznf`), select a submission that went in a different direction,
  such as `submit_vt`.

## Credits

- The organisers' official baseline and its edge model:
  [royerlab/kaggle-cell-tracking-competition](https://github.com/royerlab/kaggle-cell-tracking-competition).
- The starting pipeline: the public notebook
  [kunaldesale2408/biohub-cell-tracking](https://www.kaggle.com/code/kunaldesale2408/biohub-cell-tracking)
  (LB 0.917), with 3D U-Net weights from
  [xiaoleilian/biohub-unet3d-weights](https://www.kaggle.com/datasets/xiaoleilian/biohub-unet3d-weights).
- The edge ranker: the idea of re-scoring candidate edges with a trained
  model, the 18 pair features, and the official edge model's probability
  as an extra input, adapted from the public notebook
  [yusuketogashi/no-hack-biohub-cell-another-approch-3rd](https://www.kaggle.com/code/yusuketogashi/no-hack-biohub-cell-another-approch-3rd)
  (LB 0.915).
- Fusing that probability from the forward and the time-reversed movie by
  their harmonic mean (`h_fused`): from the public notebook
  [rishabhr0y/biohub-948-sew20](https://www.kaggle.com/code/rishabhr0y/biohub-948-sew20)
  (LB 0.948).
- The publicly released DeepCenter detector, retrained here on a different
  split (`src/dctrain.py`).
- pilkwang's public weights and offline support pack of Python wheels.
- `third_party/cell-tracking-getting-started-w-nearest-neighbor.ipynb`: a copy
  of the competition's starter notebook by
  [inversion](https://www.kaggle.com/code/inversion/cell-tracking-getting-started-w-nearest-neighbor)
  (Apache 2.0).
- `third_party/biohub-medal-v1.backup.ipynb`: a copy of the public notebook
  "Biohub medal v1" by Le Trong Hieu,
  [backtracking/biohub-medal-v1](https://www.kaggle.com/code/backtracking/biohub-medal-v1)
  (Apache 2.0).
- Other public notebooks and forum posts read during the competition are listed
  in `docs/public-solutions-recon.md`.
- Data: Chan Zuckerberg Biohub and the Royer lab, released under CC0.

## License

MIT for my own code; see `LICENSE`. The two notebooks in `third_party/` stay
under their authors' Apache 2.0 terms.
