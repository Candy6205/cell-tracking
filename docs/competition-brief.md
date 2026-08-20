# Competition brief — Biohub Cell Tracking During Development

> Task definition, data format, and the metric. Confirmed numbers and current
> status live in `progress.md`; coding conventions in `engineering-rules.md`.

---

## What the task is

Given 3D light-sheet microscopy movies of zebrafish embryo development, detect
every cell and connect it across time into a **tracking graph**, including
divisions.

- Host: Chan Zuckerberg Biohub, data from the Loic Royer lab (ultrack,
  inTRACKtive)
- Started 2026-06-29

The output is a graph, not a label:

- **node** = one cell at one time, coordinates `(t, z, y, x)`
- **edge** = the same cell from t to t+1
- **division** = a node with two outgoing edges

```
3D+time image -> (1) detection: find cell centres per frame
              -> (2) linking: match across frames
              -> tracking graph -> CSV
```

This is a different world from tabular competitions: there is no feature table,
you construct the samples from voxels.

---

## Data format

### Images
- **OME-Zarr**, dimensions `(T, Z, Y, X)`, uint16
- Voxel size Z=1.625 um, Y=X=0.40625 um — **Z is 4x coarser**, so distances
  must be scaled physically. Read it per dataset from the zarr metadata
- Read slices on demand; never materialize a whole movie
- The zarr attrs also carry `image_statistics.quantiles`, precomputed by the
  organizers — use them instead of scanning the volume for normalization

### Annotations
- **GEFF** format (the `tracksdata` graph format): one `.geff` beside each
  `.zarr`
- **Annotation is sparse** — only a fraction of cells are labelled
- The geff metadata carries `estimated_number_of_nodes` = the true total cell
  count. Train only
- Tracks were generated with **Ultrack** (segmentation + tracking), so a node
  coordinate is the **centroid of a segmented nucleus**, quantised to integer
  voxels — not a hand-clicked point

### Directory layout on Kaggle
```
/kaggle/input/competitions/biohub-cell-tracking-during-development/
├── train/   {name}.zarr + {name}.geff    <- has ground truth
└── test/    {name}.zarr                   <- 4 sample datasets; the real test
                                              set is swapped in at rescore time
```

### Submission CSV

| Column | Meaning |
|---|---|
| `id` | row index |
| `dataset` | dataset name (matches the zarr filename) |
| `row_type` | `"node"` or `"edge"` |
| `node_id` | node id (`-1` on edge rows) |
| `t, z, y, x` | integer coordinates (`-1` on edge rows) |
| `source_id, target_id` | edge endpoints (`-1` on node rows) |

Node rows first, then edge rows. `node_id` only has to be unique within a
dataset. The official `geffs_to_csv.py` / `csv_to_geffs.py` do the conversion
both ways.

---

## The metric

```
score = adjusted_edge_jaccard + 0.1 * division_jaccard
```

Edges carry over 90% of the weight.

### Edge Jaccard

```
Edge Jaccard = TP / (TP + FP + FN)
```

1. **Node matching**: optimal bipartite matching by centre distance within each
   frame, capped at **7 um physical distance**
2. **Edge cleaning, in this order** — all three verified against
   `tracking_cellmot/metrics.py`:
   - duplicate `(source, target)` pairs are dropped
   - **any edge that is not exactly one frame forward is DISCARDED**, not
     penalised. Gap-closing links across 2+ frames earn nothing unless the
     intermediate nodes are emitted too
   - merges collapse: one predicted edge per matched GT edge
   - out-degree is capped at 2
3. **Classification**:
   - **TP**: both endpoints match GT nodes that are themselves connected
   - **FP**: an edge counts toward the denominator as soon as **ONE** endpoint
     matches a GT node of the right degree —
     `out_valid(source) OR in_valid(target)`. One matched endpoint is enough.
   - **FN**: a GT edge that is not a TP

**A missed detection therefore costs twice**: the GT edge becomes an FN, *and*
the predicted edge leaving its matched predecessor becomes an FP.

Sparse annotation still means an unmatched predicted **node** is not an FP —
the organizers know they did not label everything. But the node total is
penalized separately.

### The node-count adjustment

```
adjusted = max(0, jaccard * (1 - 0.1 * (N_pred - N_true) / N_true))
```

There is **no upper clamp**, so under-detection is mildly rewarded.

### Aggregation

`edge_jaccard` is micro-averaged, but the scored quantity `adj_edge_jaccard` is
a **per-dataset** adjusted Jaccard averaged with weights
`w_i = TP_i + FP_i + FN_i`. Each dataset must be calibrated on its own.

### Division Jaccard

The exact frame of a division is subjective, so a +-1 frame tolerance applies,
with strict directed local topology requirements
(`grandparent -> parent -> children -> grandchildren`). Weight 0.1.

The metric was **gamed and patched on 2026-07-17 (commit `aa65e90`)** and all
submissions rescored on 2026-07-23. Public notebook badges dated on or before
that day may be pre-patch numbers. See `public-solutions-recon.md`.

---

## Domain background: how cell tracking is normally done

### Classical (tracking-by-detection)
1. **Detect**: Difference-of-Gaussians / LoG blob detection, watershed, local
   maxima suppression
2. **Link**: build a bipartite graph between consecutive frames and solve it
   optimally with the Hungarian algorithm
   (`scipy.optimize.linear_sum_assignment`), cost = physical distance
3. **Post-process**: gap closing, division detection

No GPU needed, runs in a day, fully interpretable.

A classical detector is just a CNN whose weights are hand-written:
DoG = a fixed convolution kernel, thresholding = sigmoid at 0.5, non-maximum
suppression = max pooling. The Hungarian step is the same matching used in
DETR's loss.

### Learned
- **Detection**: 3D U-Net producing a per-voxel heatmap, then local maxima
- **Linking**: GNN or transformer scoring edges, then global optimization
  (ILP / min-cost flow)
- Reference work: **ultrack** (Nature Methods 2025, the same lab, ILP-based),
  **Trackastra** (transformer linking), Cellpose, TrackMate

The official baseline is `TemporalUNet3D` (3D U-Net with temporal attention)
for detection plus `SimpleNodeTransformer` (cross-attention) scoring each
(t, t+1) node pair, trained end to end with sparse supervision. The organizers
state it **has not been trained to convergence**.

Two structural properties of that model worth remembering:

- the edge head applies `softmax(logits, dim=0)`, normalising over **sources**.
  Each target picks one parent, so merges are forbidden and divisions are
  allowed. Nothing normalises over targets, which is why spurious forks are
  cheap to create.
- `window_size = 2`: the model only ever sees the two frames it is linking. It
  structurally **cannot use velocity**.

---

## Constraints

- No local GPU; 30 h/week of Kaggle GPU quota that does not accumulate
- Code competition: notebook submissions only, **12 h limit**, **internet
  disabled**, output must be named exactly `submission.csv`
- The hidden test is **~199-200 volumes**; a full learned pipeline takes 9-12 h.
  Do NOT frame-stride to fit: GT edges only ever span `t -> t+1`, so strided
  edges are structurally unmatchable and score exactly 0.0 with no error raised

---

## Links

- Competition: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development
- Official baseline: https://github.com/royerlab/kaggle-cell-tracking-competition
- Metric definition: https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md
- ultrack paper: https://www.nature.com/articles/s41592-025-02778-0
- tracksdata: https://github.com/royerlab/tracksdata
- Zebrahub imaging: https://zebrahub.sf.czbiohub.org/imaging
