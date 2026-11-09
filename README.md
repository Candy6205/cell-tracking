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

<!-- TODO (me): why the public and private scores diverge, and why the best
private submission was not the one selected. -->

## How the pipeline evolved

The starting point was a verbatim reproduction of a public notebook (LB 0.917,
see Credits): a 3D U-Net detector and a frame-to-frame linker. From there, each
submission changed one thing against the previous best. The main steps:

| Script | What it changed | Public | Private |
|---|---|---|---|
| `submit_ranker.py` | Replaced the linker with a learned edge ranker (gradient-boosted trees, 18 features) feeding an integer linear program (ILP) | 0.9213 | 0.8907 |
| `submit_f4.py` | Detection from four heatmaps: the U-Net ensemble plus a retrained DeepCenter model | 0.9251 | 0.9038 |
| `submit_ep.py` | The official edge model's forward and reverse probabilities as extra ranker features | 0.9328 | 0.9066 |
| `submit_gate.py` | A classifier (the "fork gate") that decides which candidate cell divisions to keep | 0.9433 | 0.9103 |
| `submit_img.py` | The fork gate retrained on two embryos, with raw-image features at every fork | 0.9519 | 0.9163 |
| `submit_prop.py` | Orphaned track starts re-attached as division daughters, filtered by a gate fitted on them | 0.9545 | 0.9168 |
| `submit_vt.py` | Extra detections where at least two of the four heatmaps agree | 0.9534 | **0.9197** |
| `submit_nc35.py` | A per-node cost in a second ILP solve, which drops weakly supported detections | **0.9561** | 0.9137 |
| `submit_zsznf.py` | Final selection: cell-size features in the fork gate, division variables in the ILP | 0.9550 | 0.9143 |

`docs/HANDOFF.md` lists every submission, including the changes that did not
help.

## Repository layout

```
submit_*.py            one self-contained Kaggle kernel per submission
make_*.py,
tools/bench/make_*.py  build a new script from its control by exact, asserted
                       string replacements, so the two differ in one place
other *.py             experiments and diagnostics, one entry each in docs/progress.md
kag.py                 wrapper around the Kaggle CLI, used for every push
kernel-metadata.json   Kaggle kernel settings for the next push
docs/                  the project log (see "How this was built")
```

The folder is flat on purpose: the Kaggle pushes and the imports between
scripts depend on the paths.

## Running it

The competition data is not in this repository (it is released under CC0;
download it from Kaggle). Every script runs as a Kaggle kernel on a T4 GPU:

```bash
# set "code_file" in kernel-metadata.json to the script you want, then
py kag.py kernels push -p .
```

`docs/kaggle-toolchain.md` covers the full workflow and its traps.

A local smoke test needs no competition data, but it does need zarr 3 and
blosc2. The synthetic data only proves the plumbing works; the numbers it
prints are meaningless.

<!-- TODO (me): run these three commands in a clean environment and pin the
versions that work. -->

```bash
pip install "zarr>=3" blosc2
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

<!-- TODO (me): what the top solutions did differently, where this pipeline
lost, and what I would change. -->

## Credits

- The organisers' official baseline and its edge model:
  [royerlab/kaggle-cell-tracking-competition](https://github.com/royerlab/kaggle-cell-tracking-competition).
- The starting pipeline: the public notebook
  [kunaldesale2408/biohub-cell-tracking](https://www.kaggle.com/code/kunaldesale2408/biohub-cell-tracking)
  (LB 0.917), with 3D U-Net weights from
  [xiaoleilian/biohub-unet3d-weights](https://www.kaggle.com/datasets/xiaoleilian/biohub-unet3d-weights).
- The publicly released DeepCenter detector, retrained here on a different
  split (`dctrain.py`).
- pilkwang's public weights and offline support pack of Python wheels.
- `cell-tracking-getting-started-w-nearest-neighbor.ipynb`: a copy of the
  competition's starter notebook by
  [inversion](https://www.kaggle.com/code/inversion/cell-tracking-getting-started-w-nearest-neighbor)
  (Apache 2.0).
- `biohub-medal-v1.backup.ipynb`: a copy of the public notebook
  "Biohub medal v1" by Le Trong Hieu,
  [backtracking/biohub-medal-v1](https://www.kaggle.com/code/backtracking/biohub-medal-v1)
  (Apache 2.0).
- Other public notebooks and forum posts read during the competition are listed
  in `docs/public-solutions-recon.md`.
- Data: Chan Zuckerberg Biohub and the Royer lab, released under CC0.

## License

<!-- TODO (me): pick a license for my own code, for example MIT. The two
notebooks above stay under their authors' Apache 2.0 terms. -->
