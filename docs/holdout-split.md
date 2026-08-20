# Held-out split for the public U-Net weights — PARTIALLY RESOLVED

*Updated 2026-08-18 after reading the published artefacts with
`check_weights.py`.*

## Status, in one table

| component | its split | known how? |
|---|---|---|
| DeepCenter gate (`full_frame_center`) | train = **all 71 `44b6_`**, val = **all 128 `6bba_`** | **FACT** — `split_manifest.json` |
| primary U-Net+transformer (`unet_transformer/split_0`) | unknown; checkpoint says only `fold: 0` | still inferred |
| second seed (`seed314159`) | unknown, artefact not inspected | unknown |

## What `split_manifest.json` actually said

```
all:   199 datasets, 19/19 of our derived holdout in it
train:  71 datasets,  5/19 — exactly the five 44b6_ names
val:   128 datasets, 14/19 — exactly the fourteen 6bba_ names
seed: 2026, val_fraction: 0.1
```

71 is precisely the number of `44b6_` datasets in train (names are sorted, and
the first 71 are all `44b6_`). Every `44b6_` member of our 19 landed in `train`
and every `6bba_` member landed in `val`, with no exceptions. That is a
**group split by imaging batch**, not the random 90/10 the config's
`val_fraction: 0.1` and `seed: 2026` describe — those fields were overridden by
whatever `run_full_frame_center_training.sh` passed.

## Consequence 1 — two of our eight "held-out" datasets are contaminated

`HOLDOUT_8` contained `44b6_1574802b` and `44b6_706092f0`. Both are in
DeepCenter's **training** set. The other six are `6bba_` and are in its val set.

Mitigation: **any dataset used for held-out validation must be `6bba_`.**
The `6bba_` batch is unseen by DeepCenter *by construction*, independent of
whether the 19-dataset derivation is right. That is a free, permanent guarantee
and should replace the derived holdout as the primary validation pool wherever
the DeepCenter gate is enabled.

Severity is probably small: the gate is described as **add-only** — it can add
detections, never remove them — so contamination inflates recall on those two
datasets and cannot affect the other six.

## Consequence 2 — the derivation for the primary model is now *less* likely

The derivation assumed the repo's default `random.Random(0).shuffle`. We now
know this author does not use the repo defaults for their own training runs;
DeepCenter used a hand-built batch split with a manifest written out. The
primary checkpoint carries `fold: 0` and **no file list at all**, so the
derivation is neither confirmed nor refuted — but the prior on "they just ran
the default" has dropped.

Assumption 1 from the old version of this file still stands as the weakest
link: the shuffle runs over `sorted(stems)`, so one extra or missing `.zarr`
reshuffles everything.

## Consequence 3 — the DeepCenter checkpoint in use is the overfit one

| checkpoint | epoch | val_loss |
|---|---|---|
| `best.pt` | 2 | 0.04500 |
| `checkpoint_last.pt` | 500 | 0.31674 |

Val loss never improved after epoch 2 across 500 epochs and ended **7x worse**.
Since val = the whole `6bba_` batch, this is a clean cross-batch generalization
test, and the detector fails it: it memorised `44b6_` and did not transfer.

The 0.916 notebook hardcodes `checkpoint_last.pt` and even asserts
`epoch == 500` (`BIOHUB_DEEPCENTER_EXPECTED_EPOCH=500`).

**Caveat before acting on this:** val_loss on a center-heatmap regression is not
the competition metric, and an add-only prior can be useful even when its loss
is poor. This is a hypothesis with a cheap test, not a conclusion.

## The three experiments this unlocks

All are one env-var change to the 0.916 notebook, scored on a `6bba_`-only
holdout.

1. **Ablate the gate** — `BIOHUB_USE_DEEPCENTER_VETO=0`. Establishes whether
   DeepCenter contributes anything. If it does not, drop it and the
   contamination question with it.
2. **Swap to `best.pt`** — `BIOHUB_DEEPCENTER_RELATIVE=weights/full_frame_center/best.pt`
   plus `BIOHUB_DEEPCENTER_EXPECTED_EPOCH=2` (the assert fires otherwise).
3. **Re-score the old holdout without the two 44b6 datasets** — free, uses
   numbers we can already produce, and tells us how much of local 0.961 was
   contamination.

## What is still open

- the primary model's real split
- the `seed314159` artefact's split (its `ARTIFACT_MANIFEST.json` has not been
  read; it is not currently attached)
- therefore: the absolute local score remains untrustworthy. **Use local scores
  for paired relative comparisons; the leaderboard is the only fully trustworthy
  absolute number.**

## Why not real k-fold CV

Retraining the U-Net k times does not fit a 30 h/week GPU quota. Evaluating
existing weights on data they never saw answers the same question for free —
and for the DeepCenter component we now know exactly which data that is.

---

## NOTE (2026-08-20): largely superseded

`docs/public-solutions-recon.md` reports two findings from the competition forum
that outrank most of this file:

- the host confirmed **no overlap in embryo_ids between train and test** — the
  hidden test is a third embryo, neither `44b6` nor `6bba`. So the local-vs-LB
  offset is at least partly a distribution gap, not leakage.
- **DeepCenter turns out not to matter** — the public 0.915-0.917 tier carries
  its env vars but disables every one. That retires the DeepCenter
  contamination question entirely, and with it consequences 1 and 3 above.

The primary model's split remains unknown.

---

*Last updated: 2026-08-18, annotated 2026-08-20*
