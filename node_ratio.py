"""Is the detector ALREADY over the target node count, and does that
depend on density?

The user's hypothesis, after the leaderboard refused two
recall-increasing changes:

  1. the pipeline already emits far more nodes than N_true, so any
     threshold-loosening pays a node-count penalty it cannot win back;
  2. the public leaderboard subset may be skewed SPARSE, where the model
     already lands near N_true without loosening -- so loosening there
     is pure cost.

The metric's penalty is

    J_adj = max(0, J * (1 - 0.1 * (N_pred - N_true) / N_true))

so the ratio N_pred / N_true is the whole story, and it is measurable
per dataset: N_true is the organisers' `estimated_number_of_nodes`,
carried in the exported GT since the fix, and N_pred is the pipeline's
final node count.

⚠ The node counts below come from biohub-colab-check, which ran the
D187 pipeline at gate 0.15. They are FINAL counts, after gap-close and
short-filter. That is the right quantity -- it is what the submission
writes -- but it is one configuration, not the whole sweep.
"""
import pathlib

import numpy as np

GT = pathlib.Path(r"C:\Users\48688\AppData\Local\Temp\gtx2")

# final node counts, from biohub-colab-check's per-dataset lines
NPRED = {
    "6bba_2312ac41": 11523, "6bba_268e1230": 4609,
    "6bba_283bf9f1": 20811, "6bba_3a1849c2": 6223,
    "6bba_3abfe10a": 53504, "6bba_5c824876": 24189,
    "6bba_61dd1e0d": 9221, "6bba_7af54fde": 9145,
    "6bba_7b5d3b2c": 4990, "6bba_aeee7805": 4809,
    "6bba_afb141ff": 5759, "6bba_c27cba08": 6074,
}
T_FRAMES = 100

rows = []
for name, npred in NPRED.items():
    f = GT / f"gt_{name}.npz"
    if not f.exists():
        continue
    z = np.load(f)
    if "n_total" not in z.files:
        continue
    ntrue = float(z["n_total"])
    ratio = npred / ntrue
    pen = 1.0 - 0.1 * (npred - ntrue) / ntrue
    rows.append((name, ntrue / T_FRAMES, ntrue, npred, ratio, pen))

rows.sort(key=lambda r: r[1])
print(f"  {'dataset':<18} {'true/fr':>8} {'N_true':>9} {'N_pred':>9} "
      f"{'ratio':>7} {'penalty':>8}")
for r in rows:
    flag = "  <-- UNDER target" if r[4] < 1.0 else ""
    print(f"  {r[0]:<18} {r[1]:>8.0f} {r[2]:>9,.0f} {r[3]:>9,} "
          f"{r[4]:>7.3f} {r[5]:>8.4f}{flag}")

tt = sum(r[2] for r in rows)
tp = sum(r[3] for r in rows)
print(f"\n  TOTAL              {'':>8} {tt:>9,.0f} {tp:>9,} "
      f"{tp / tt:>7.3f} {1 - 0.1 * (tp - tt) / tt:>8.4f}")

d = np.array([r[1] for r in rows])
q = np.array([r[4] for r in rows])
print(f"\n  corr(true cells per frame, N_pred/N_true) = "
      f"{float(np.corrcoef(d, q)[0, 1]):+.4f}")

print("\n  READ IT:")
print("  ratio > 1  the pipeline already emits MORE nodes than there are")
print("             cells, and every extra node is charged at 0.1x. A")
print("             change that adds nodes starts from behind.")
print("  ratio < 1  there is HEADROOM: adding nodes moves toward the")
print("             target and the penalty shrinks rather than grows.")
print("  If the correlation with density is NEGATIVE, sparse datasets")
print("  sit closer to (or over) the target and dense ones have room --")
print("  which would mean a sparse-skewed public split punishes exactly")
print("  the loosening that helps on dense data. That is the user's")
print("  hypothesis and this is the number that tests it.")
