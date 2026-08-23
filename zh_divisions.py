"""zh_divisions.py -- are Zebrahub's 764k "divisions" real?

The counts read straight off the extract log are:

    ZSNS001      635,042 divisions in 21.7M rows   (2.9%)
    ZSNS001_tail  36,878 in  7.5M                  (0.5%)
    ZSNS003       15,869 in  4.1M                  (0.4%)
    ZSNS004       30,279 in  7.2M                  (0.4%)
    ZSNS005       46,027 in  6.5M                  (0.7%)

ZSNS001 is six times the others. That is not a plausible biological
difference; the obvious suspect is track-id fragmentation -- a track that
breaks and is re-numbered looks exactly like a parent with two children.

This decides, on CPU, before any GPU is spent on detection, whether
Zebrahub can supply division supervision. It also measures the geometry we
would actually train on, so the transfer question ("does a Zebrahub
division look like a competition division?") gets a number rather than an
argument. Competition reference, from our own census over 26 fully
detected divisions: worst parent->daughter distance median 7.81 um, z
share 0.42. The synthetic-dataset author independently calibrated sister
separation at 7.24 um.
"""
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd

SCRIPT_VERSION = "2026-08-23w-zh-divisions2"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

BUCKET = ("https://public.czbiohub.org/royerlab/zebrahub/imaging/"
          "single-objective")
MOVIES = ["ZSNS001_tail", "ZSNS003", "ZSNS004", "ZSNS005"]
# ZSNS001 done in v1: 358,314 validated of 635,041, sisterSep
# p50 7.37 um, z share 0.84, 96% inside a 10 um gate.
VOXEL = (1.24, 0.439, 0.439)          # Zebrahub native, z y x
EXTENT = {                            # um, for the unit sniff
    "ZSNS001": (556, 954, 1064), "ZSNS001_tail": (521, 534, 479),
    "ZSNS003": (320, 847, 895), "ZSNS004": (365, 846, 1133),
    "ZSNS005": (361, 838, 1207),
}
MIN_PERSIST = 3        # frames both daughters must survive
OUT = pathlib.Path("/kaggle/working")

summary = []
for movie in MOVIES:
    print(f"\n{'='*66}\n{movie}\n{'='*66}", flush=True)
    csv = OUT / f"{movie}_tracks.csv"
    t0 = time.time()
    r = subprocess.run(["curl", "-sS", "-o", str(csv),
                        f"{BUCKET}/{movie}_tracks.csv"],
                       capture_output=True, text=True)
    if r.returncode != 0 or not csv.exists():
        print(f"  WARN download failed ({r.stderr[:80]}) -- movie SKIPPED",
              flush=True)
        continue
    print(f"  downloaded {csv.stat().st_size/2**20:.0f} MB in "
          f"{time.time()-t0:.0f}s", flush=True)
    tr = pd.read_csv(csv)
    tr.columns = [c.strip().lower() for c in tr.columns]
    # ZSNS001_tail names the parent column `parenttrackid`; some movies
    # carry BOTH `id` and `track_id`, so renaming blindly produced two
    # columns of the same name and pandas handed back a DataFrame.
    ren = {"time": "t", "frame": "t",
           "parent": "parent_track_id", "parentid": "parent_track_id",
           "parenttrackid": "parent_track_id",
           "parent_id": "parent_track_id",
           "id": "track_id", "trackid": "track_id",
           "nodeid": "node_id"}
    ren = {k: v for k, v in ren.items()
           if k in tr.columns and v not in tr.columns}
    tr = tr.rename(columns=ren)
    dup = tr.columns[tr.columns.duplicated()].tolist()
    if dup:
        print(f"  WARN duplicate columns {dup} -- keeping the first",
              flush=True)
        tr = tr.loc[:, ~tr.columns.duplicated()]
    need = {"t", "track_id", "parent_track_id", "z", "y", "x"}
    if not need.issubset(tr.columns):
        print(f"  WARN columns {sorted(tr.columns)[:10]} lack {sorted(need)}"
              f" -- movie SKIPPED", flush=True)
        csv.unlink(missing_ok=True)
        continue

    ext = EXTENT[movie]
    over = sum(tr[c].max() > ext[a] * 1.05 for c, a in zip("zyx", range(3)))
    units = "voxel" if over else "um"
    for c, a in zip("zyx", range(3)):
        tr[c + "_um"] = tr[c] * VOXEL[a] if units == "voxel" else tr[c]
    print(f"  units: {units.upper()} | {len(tr):,} rows", flush=True)

    tr = tr.drop_duplicates(subset=["track_id", "t"])
    first = tr.groupby("track_id").t.min()
    last = tr.groupby("track_id").t.max()
    life = (last - first + 1)

    # a division: a parent id with >= 2 distinct children starting after it
    ch = (tr[["track_id", "parent_track_id"]].drop_duplicates()
          .dropna(subset=["parent_track_id"]))
    ch = ch[ch.parent_track_id > 0]
    kids = ch.groupby("parent_track_id").track_id.apply(list)
    kids = kids[kids.map(len) >= 2]
    n_raw = len(kids)
    deg = pd.Series([len(v) for v in kids]).value_counts().sort_index()
    print(f"  raw divisions (parent with >=2 children): {n_raw:,}", flush=True)
    print(f"  children per parent: "
          f"{dict(deg.head(6).items())}", flush=True)

    pos = tr.set_index(["track_id", "t"])[["z_um", "y_um", "x_um"]]
    pos = pos[~pos.index.duplicated()]

    kept, sep, dpar, zsh, ang = 0, [], [], [], []
    rej_persist = rej_nopos = rej_zero = 0
    for par, cs in kids.items():
        if len(cs) != 2:
            continue
        c1, c2 = cs[0], cs[1]
        if c1 not in life.index or c2 not in life.index:
            rej_nopos += 1
            continue
        if life.get(c1, 0) < MIN_PERSIST or life.get(c2, 0) < MIN_PERSIST:
            rej_persist += 1
            continue
        t1, t2 = first.get(c1), first.get(c2)
        if t1 != t2:                      # daughters must appear together
            rej_persist += 1
            continue
        tp = t1 - 1
        try:
            p = pos.loc[(par, tp)].to_numpy(float)
            a = pos.loc[(c1, t1)].to_numpy(float)
            b = pos.loc[(c2, t2)].to_numpy(float)
        except KeyError:
            rej_nopos += 1
            continue
        s = float(np.linalg.norm(a - b))
        if s < 1e-6:
            rej_zero += 1
            continue
        va, vb = a - p, b - p
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        kept += 1
        sep.append(s)
        far = max(na, nb)
        dpar.append(far)
        zsh.append(abs((va if na >= nb else vb)[0]) / max(far, 1e-9))
        if na > 1e-6 and nb > 1e-6:
            ang.append(np.degrees(np.arccos(
                np.clip(float(va @ vb) / (na * nb), -1, 1))))

    print(f"  VALIDATED: {kept:,} / {n_raw:,} "
          f"({100.0*kept/max(n_raw,1):.0f}%)", flush=True)
    print(f"    rejected: {rej_persist:,} short-lived or daughters not "
          f"born together, {rej_nopos:,} missing position, {rej_zero:,} "
          f"zero separation", flush=True)
    if kept:
        sep = np.array(sep); dpar = np.array(dpar)
        zsh = np.array(zsh); ang = np.array(ang)
        print(f"    sister separation um: p10 {np.quantile(sep,.1):.2f} "
              f"p50 {np.median(sep):.2f} p90 {np.quantile(sep,.9):.2f}"
              f"   (competition ref 7.24)", flush=True)
        print(f"    worst parent->daughter um: p50 {np.median(dpar):.2f} "
              f"p90 {np.quantile(dpar,.9):.2f}   (our census 7.81)",
              flush=True)
        print(f"    z share of it: p50 {np.median(zsh):.2f}   "
              f"(our census 0.42)", flush=True)
        if len(ang):
            print(f"    angle between the two daughters, deg: p50 "
                  f"{np.median(ang):.0f} p90 {np.quantile(ang,.9):.0f} "
                  f"(180 = opposed)", flush=True)
        print(f"    a 10 um gate would admit "
              f"{100.0*(dpar<=10).mean():.0f}% of them", flush=True)
        summary.append((movie, n_raw, kept, float(np.median(sep)),
                        float(np.median(dpar)), float(np.median(zsh))))
    csv.unlink(missing_ok=True)      # do not ship 200-850 MB

print(f"\n{'='*66}\nSUMMARY\n{'='*66}", flush=True)
print("  movie          raw      validated   sisterSep  parent->d   zShare",
      flush=True)
tot_raw = tot_kept = 0
for m, nr, k, s, d, z in summary:
    tot_raw += nr; tot_kept += k
    print(f"  {m:<13} {nr:>9,} {k:>9,}   {s:8.2f} {d:9.2f} {z:8.2f}",
          flush=True)
print(f"  {'TOTAL':<13} {tot_raw:>9,} {tot_kept:>9,}  "
      f"({100.0*tot_kept/max(tot_raw,1):.0f}% survive validation)", flush=True)
print("\n  competition, for comparison: 151 divisions TOTAL across 199 "
      "datasets, from only 2 embryos (26 + 125)", flush=True)
print("ZH DIVISIONS DONE", flush=True)
