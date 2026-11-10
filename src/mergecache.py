"""mergecache.py -- hard-merged detection caches at a classifier threshold, with the
union heads REMAPPED to the merged index sets (no GPU).

SCRIPT_VERSION 2026-09-15a-mergecache

INPUTS (attached kernels): biohub-hypcache (fdet_F4hyp_<name>.npz: c/s of every
detection with the parents appended after all of them, p{t} = the two children of
each parent as indices among primaries, w{t} = saddles), biohub-ephypcache and
biohub-ep2hypcache (edgeprob_F4hyp_ / edgeprob2_F4hyp_: the two heads over every
pair of union primaries within GATE_UM), biohub-mergefeat (mergefeat_<name>.npz:
t, q = parent index, p = out-of-fold merge probability, lab).

FOR EACH TAU in TAUS and each 6bba movie: parents with p >= tau are selected
greedily by p, each child in at most one selected parent; the selected parents'
children are removed, the unselected parents are removed, everything else keeps
its order. The det cache is written as fdet_F4mg<TT>_<name>.npz (c{t}/s{t}), and
each head cache as edgeprob_F4mg<TT>_<name>.npz / edgeprob2_F4mg<TT>_<name>.npz
with (t, i, j) remapped to the merged primary indices and n_per_frame recomputed:
the pairs of the surviving nodes are exactly the union's pairs between them, so
the harness's join (every pair within the gate must find its row) holds by
construction. Counts per movie and per tau are printed; a self-check re-derives
n_per_frame from the merged det cache with the harness's seeds_of rule.

Attach: biohub-hypcache, biohub-ephypcache, biohub-ep2hypcache, biohub-mergefeat.
CPU, minutes.
"""
import os
import pathlib
import time
from collections import defaultdict

import numpy as np

SCRIPT_VERSION = "2026-09-15a-mergecache"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

TAUS = [float(x) for x in os.environ.get("MG_TAUS", "0.90,0.95").split(",")]
MG_SRC = os.environ.get("MG_SRC", "mergefeat_")     # which per-movie score file: mergefeat_ or mergeveto_
MG_PKEY = os.environ.get("MG_PKEY", "p")            # which probability inside it: p (mergefeat) or p2 (the refit)
MG_VETO = os.environ.get("MG_VETO", "")             # optional veto, mergeveto files only: "div_sig" (n_m1 == 1 and age_det == 0)
print(f"TAUS {TAUS} source {MG_SRC}{MG_PKEY} veto {MG_VETO!r}", flush=True)
UNET_THRESH = 0.15


def tag(tau):
    return f"F4mg{int(round(tau * 100)):02d}"   # the det-source token; the score file / veto are in the log


def walk(prefix):
    out = {}
    for d in pathlib.Path("/kaggle/input").glob(f"*/**/{prefix}*.npz"):
        out[d.name[len(prefix):-len(".npz")]] = d
    return out


HYP = walk("fdet_F4hyp_")
EP1 = walk("edgeprob_F4hyp_")
EP2 = walk("edgeprob2_F4hyp_")
MF = walk(MG_SRC)
SIX = sorted(n for n in MF if n.startswith("6bba_"))
print(f"caches: hyp {len(HYP)}, head1 {len(EP1)}, head2 {len(EP2)}, mergefeat {len(MF)}; movies {len(SIX)}", flush=True)
assert SIX and all(n in HYP and n in EP1 and n in EP2 for n in SIX), "a movie lacks a cache"
OUT = pathlib.Path("/kaggle/working")


def merged_index_map(s_all, p_par, q_sel):
    """s_all: scores of the union array (sub-threshold + children + parents, in file order);
    p_par: (n_par, 2) child indices among primaries; q_sel: selected parent indices.
    Returns keep (bool over the union array) and new_of (old primary index -> new primary
    index or -1)."""
    prim_pos = np.where(s_all >= UNET_THRESH)[0]          # positions of primaries in the array
    n_prim = len(prim_pos)
    n_par = len(p_par)
    n_ch = n_prim - n_par
    keep_prim = np.ones(n_prim, bool)
    for q in range(n_par):                                # unselected parents go
        keep_prim[n_ch + q] = False
    for q in q_sel:
        keep_prim[n_ch + q] = True
        keep_prim[p_par[q, 0]] = False
        keep_prim[p_par[q, 1]] = False
    keep = np.ones(len(s_all), bool)
    keep[prim_pos[~keep_prim]] = False
    new_of = np.full(n_prim, -1, np.int64)
    new_of[keep_prim] = np.arange(int(keep_prim.sum()))
    return keep, new_of


def select_parents(p_par, p_prob, tau):
    """Greedy by probability, each child at most once."""
    order = np.argsort(-p_prob)
    used = set()
    sel = []
    for q in order:
        if p_prob[q] < tau:
            break
        a, b = int(p_par[q, 0]), int(p_par[q, 1])
        if a in used or b in used:
            continue
        used.add(a)
        used.add(b)
        sel.append(int(q))
    return sel


def remap_head(z, new_of_t, T):
    npf_old = np.asarray(z["n_per_frame"], np.int64)
    assert len(npf_old) == T
    et = np.asarray(z["t"], np.int64)
    ei = np.asarray(z["i"], np.int64)
    ej = np.asarray(z["j"], np.int64)
    ni = np.array([new_of_t[t][i] for t, i in zip(et, ei)], np.int64) if len(et) else np.zeros(0, np.int64)
    nj = np.array([new_of_t[t + 1][j] for t, j in zip(et, ej)], np.int64) if len(et) else np.zeros(0, np.int64)
    ok = (ni >= 0) & (nj >= 0)
    out = {"n_per_frame": np.array([int((new_of_t[t] >= 0).sum()) for t in range(T)], np.int64),
           "t": et[ok], "i": ni[ok], "j": nj[ok]}
    for k in z.files:
        if k in ("n_per_frame", "t", "i", "j"):
            continue
        v = z[k]
        out[k] = v[ok] if (hasattr(v, "shape") and v.shape[:1] == et.shape) else v
    return out, int(ok.sum()), int((~ok).sum())


t0 = time.time()
TOT = {tau: defaultdict(int) for tau in TAUS}
for k_, name in enumerate(SIX):
    zh = np.load(HYP[name])
    zm = np.load(MF[name])
    T = max(int(k[1:]) for k in zh.files if k.startswith("c")) + 1
    par = [np.asarray(zh[f"p{t}"], np.int64).reshape(-1, 2) if f"p{t}" in zh.files else np.zeros((0, 2), np.int64)
           for t in range(T)]
    prob = [np.full(len(par[t]), -1.0, np.float32) for t in range(T)]
    p_src = np.asarray(zm[MG_PKEY], np.float32)
    if MG_VETO == "div_sig":
        p_src = np.where((np.asarray(zm["n_m1"]) == 1) & (np.asarray(zm["age_det"]) == 0), -1.0, p_src)
    elif MG_VETO:
        raise SystemExit(f"unknown MG_VETO {MG_VETO!r}")
    for t_, q_, p_ in zip(zm["t"], zm["q"], p_src):
        prob[int(t_)][int(q_)] = float(p_)                  # parents mergefeat did not score (d > 8.5) stay -1
    z1 = np.load(EP1[name])
    z2 = np.load(EP2[name]) if name in EP2 else None
    for tau in TAUS:
        det = {}
        new_of_t = []
        n_sel = n_drop = 0
        for t in range(T):
            c = np.asarray(zh[f"c{t}"], np.float32)
            s = np.asarray(zh[f"s{t}"], np.float32)
            sel = select_parents(par[t], prob[t], tau)
            keep, new_of = merged_index_map(s, par[t], sel)
            det[f"c{t}"] = c[keep]
            det[f"s{t}"] = s[keep]
            new_of_t.append(new_of)
            n_sel += len(sel)
            n_drop += 2 * len(sel)
        tg = tag(tau)
        np.savez_compressed(OUT / f"fdet_{tg}_{name}.npz", **det)
        h1, kept1, dropped1 = remap_head(z1, new_of_t, T)
        np.savez_compressed(OUT / f"edgeprob_{tg}_{name}.npz", **h1)
        if z2 is not None:
            h2, kept2, dropped2 = remap_head(z2, new_of_t, T)
            np.savez_compressed(OUT / f"edgeprob2_{tg}_{name}.npz", **h2)
        # self-check: n_per_frame from the merged det cache by the harness's rule
        npf_det = np.array([int((det[f"s{t}"] >= UNET_THRESH).sum()) for t in range(T)])
        assert np.array_equal(npf_det, h1["n_per_frame"]), (name, tau, "n_per_frame mismatch")
        assert np.all(h1["i"] < h1["n_per_frame"][h1["t"]]) and np.all(h1["j"] < h1["n_per_frame"][h1["t"] + 1])
        d_ = TOT[tau]
        d_["parents"] += sum(len(p) for p in par)
        d_["selected"] += n_sel
        d_["children_dropped"] += n_drop
        d_["primaries_union"] += int(sum((np.asarray(zh[f"s{t}"]) >= UNET_THRESH).sum() for t in range(T)))
        d_["primaries_merged"] += int(npf_det.sum())
        d_["head1_rows_kept"] += kept1
        d_["head1_rows_dropped"] += dropped1
    print(f"[{k_ + 1}/{len(SIX)}] {name}: T={T} parents {sum(len(p) for p in par):,}; "
          + "; ".join(f"tau {tau:.2f} selected {TOT[tau]['selected']:,} (cumulative)" for tau in TAUS)
          + f"  {time.time() - t0:.0f}s", flush=True)
print("\n=== MERGECACHE TOTALS over the 6bba movies ===", flush=True)
for tau in TAUS:
    d_ = TOT[tau]
    print(f"  tau {tau:.2f} ({tag(tau)}): parents {d_['parents']:,}, selected {d_['selected']:,} "
          f"({d_['selected'] / max(1, d_['parents']):.2%}), children dropped {d_['children_dropped']:,}; primaries "
          f"union {d_['primaries_union']:,} -> merged {d_['primaries_merged']:,}; head rows kept {d_['head1_rows_kept']:,}, "
          f"dropped {d_['head1_rows_dropped']:,}", flush=True)
print(f"\nMERGECACHE DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
print("MERGECACHE OK", flush=True)
