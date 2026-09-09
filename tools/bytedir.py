"""bytedir -- which way does each candidate move the OUTPUT SIZE?

This project has one predictor of leaderboard movement that is independent of
the local metric: **11 recorded (delta hidden-output bytes, delta LB) pairs in
which every byte-REMOVING change gained and every byte-ADDING change lost.**
It has been right where the local bench was wrong (submit-f48 read +0.0000 /
-0.0131 locally, removed 1.43% of the bytes, and gained +0.001 on LB).

The per-movie CSVs carry `num_pred_nodes` and the matched edge counts, so the
direction can be read for every candidate before anything is submitted. This
is a PROXY: it counts nodes and GT-matched edges on 128 train movies, not
bytes of the test submission. Direction only, and it is stated as such.

    py tools/bytedir.py
"""
import csv
import sys

BASE = "bench_csv/rkpool_BASE.csv"          # BASE ranker, refine -- what ships
ARMS = [
    ("retrain (F4 ranker, refine)", "bench_csv/rkpool_F4.csv"),
    ("hpara (BASE ranker, hpara)", "bench_csv/hparabench_F4HP.csv"),
]
if len(sys.argv) > 1:                        # extra arms as label=path
    ARMS += [tuple(a.split("=", 1)) for a in sys.argv[1:]]


def load(p):
    return {r["dataset"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}


def tot(rows, names):
    n = sum(float(rows[k]["num_pred_nodes"]) for k in names)
    e = sum(float(rows[k]["edge_tp"]) + float(rows[k]["edge_fp"])
            for k in names)
    return n, e


A = load(BASE)
print(f"control: {BASE}  ({len(A)} movies)\n")
print(f"  {'arm':<32}{'nodes':>12}{'d nodes':>11}{'d %':>9}"
      f"{'d edges %':>11}  byte pattern says")
nA, eA = tot(A, sorted(A))
print(f"  {'BASE ranker, refine (ships)':<32}{nA:>12,.0f}{'':>11}{'':>9}{'':>11}")
for label, path in ARMS:
    try:
        H = load(path)
    except OSError:
        print(f"  {label:<32}  (no CSV at {path})")
        continue
    names = sorted(set(A) & set(H))
    if len(names) != len(A):
        print(f"  {label:<32}  ⚠ only {len(names)} of {len(A)} movies pair")
    n0, e0 = tot(A, names)
    n1, e1 = tot(H, names)
    dn = (n1 - n0) / n0 * 100
    de = (e1 - e0) / max(1e-9, e0) * 100
    says = ("GAIN (smaller output)" if dn < 0 else
            "LOSE (larger output)" if dn > 0 else "no call")
    print(f"  {label:<32}{n1:>12,.0f}{n1 - n0:>+11,.0f}{dn:>+9.2f}"
          f"{de:>+11.2f}  {says}")

print("""
  ⚠ Read this as DIRECTION and nothing else. It counts nodes on 128 training
  movies; the pattern was measured on the bytes of the hidden test output.
  The two agree in direction whenever the change is a linker change, because
  a linker that keeps fewer detections writes fewer rows either way -- but
  the magnitudes are not comparable and no percentage here predicts a score.""")
