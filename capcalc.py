"""Is GAP1_CAP_FRAC = 0.003 actually binding? Arithmetic on measured numbers.

Every input below was measured today; nothing is assumed except where
marked. The question the user asked: put a concrete number on the 0.3%.

  cap      = 0.003 * len(nodes) at the moment _gap_close runs
  demand   = how many one-frame dropouts exist to be bridged
  eligible = how many of those pass the candidate requirement

⚠ THE ONE ASSUMPTION, stated because this is the error this session kept
making: flicker.py counted dropouts among ANNOTATED cells only. Scaling
to all cells assumes unannotated cells drop out at the same rate. They
may not -- annotation favours cells that are easy to see, and those are
plausibly the ones that flicker LESS. If so the true demand is HIGHER
than computed here, which makes the cap bind harder, not softer. The
direction of the error is known even though its size is not.
"""

# ---- final node counts, from biohub-colab-check's per-dataset lines ----
# These are FINAL counts, after gap-close and short-filter. The cap is
# computed on the count at gap-close time, which is slightly higher
# (short_filter deletes afterwards). So the caps below are slight
# UNDER-estimates.
NODES = {
    "6bba_2312ac41": 11523, "6bba_268e1230": 4609,
    "6bba_283bf9f1": 20811, "6bba_3a1849c2": 6223,
    "6bba_3abfe10a": 53504, "6bba_5c824876": 24189,
    "6bba_61dd1e0d": 9221, "6bba_7af54fde": 9145,
    "6bba_7b5d3b2c": 4990, "6bba_aeee7805": 4809,
    "6bba_afb141ff": 5759, "6bba_c27cba08": 6074,
}
# true cells per frame, from nms4 (estimated_number_of_nodes / T)
TRUE_PER_FRAME = {
    "6bba_268e1230": 51, "6bba_2312ac41": 120, "6bba_283bf9f1": 209,
    "6bba_3a1849c2": 70, "6bba_3abfe10a": 521, "6bba_5c824876": 246,
    "6bba_61dd1e0d": 101, "6bba_7af54fde": 96, "6bba_7b5d3b2c": 58,
    "6bba_afb141ff": 61, "6bba_aeee7805": 50, "6bba_c27cba08": 64,
}
T_FRAMES = 100
GAP1_CAP_FRAC = 0.003

# ---- from flicker.py, over the same 12 datasets -----------------------
ANNOT_GT_NODES = 10699      # depth3: "TOTAL recall 0.9561 over 10699 GT nodes"
DROPOUT_1F = 102            # runs of exactly 1 missing frame
DROPOUT_ALL = 196           # all dropout runs
# of the missed frames, the fraction with a usable repair candidate:
#   20.2% have any candidate in [0.05, 0.15) within 5 um
#   ~75% of those score >= GAP1_MIN_CAND_SCORE 0.10   (p25 = 0.1003)
ELIGIBLE_FRAC = 0.202 * 0.75

total_nodes = sum(NODES.values())
total_cap = sum(max(1, int(GAP1_CAP_FRAC * n)) for n in NODES.values())
true_cells = sum(v * T_FRAMES for v in TRUE_PER_FRAME.values())
scale = true_cells / ANNOT_GT_NODES

print("=== the 12 6bba holdout datasets ===")
print(f"  solved nodes (final)        {total_nodes:>10,}")
print(f"  TRUE cells (organisers)     {true_cells:>10,}")
print(f"  annotated GT nodes          {ANNOT_GT_NODES:>10,}"
      f"   = {100.0 * ANNOT_GT_NODES / true_cells:.1f}% of true")
print(f"  scale annotated -> all      {scale:>10.1f}x")

print("\n=== the cap ===")
print(f"  0.003 x nodes, summed       {total_cap:>10,}  bridges allowed")

print("\n=== the demand ===")
d1 = DROPOUT_1F * scale
dall = DROPOUT_ALL * scale
print(f"  1-frame dropouts, annotated {DROPOUT_1F:>10,}")
print(f"  1-frame dropouts, ALL cells {d1:>10,.0f}   <- the only kind bridgeable")
print(f"  all dropouts, ALL cells     {dall:>10,.0f}")

print("\n=== does it bind? ===")
elig = d1 * ELIGIBLE_FRAC
print(f"  WITH the candidate requirement:")
print(f"    eligible ({100 * ELIGIBLE_FRAC:.1f}% of 1-frame)  {elig:>8,.0f}"
      f"  vs cap {total_cap:,}"
      f"   -> {'BINDS' if elig > total_cap else 'does NOT bind'}")
print(f"  WITHOUT it (nocand1):")
print(f"    eligible (all 1-frame)          {d1:>8,.0f}"
      f"  vs cap {total_cap:,}"
      f"   -> {'BINDS' if d1 > total_cap else 'does NOT bind'}"
      f", at {100.0 * total_cap / d1:.0f}% of demand")

print("\n=== per dataset, the two extremes ===")
print(f"  {'dataset':<18} {'nodes':>8} {'cap':>6} {'1f drops':>9} {'cap/demand':>11}")
for name in ("6bba_3abfe10a", "6bba_aeee7805"):
    n = NODES[name]
    cap = max(1, int(GAP1_CAP_FRAC * n))
    share = TRUE_PER_FRAME[name] * T_FRAMES / true_cells
    dd = d1 * share
    print(f"  {name:<18} {n:>8,} {cap:>6} {dd:>9,.0f} "
          f"{100.0 * cap / max(1e-9, dd):>10.0f}%")

print("\n  READ IT: with the candidate requirement ON the cap is not")
print("  reached, which is why post_sweep's sweep of GAP1_CAP_FRAC")
print("  (0.0015 and 0.006) moved nothing in any domain -- it was never")
print("  the binding constraint. Drop the requirement and the cap becomes")
print("  the constraint for the first time, which is what gap2's nocand1")
print("  arm was actually measuring.")
