"""divtime.py -> valley.py: the raw-image valley between the two predicted tracks of a GT division, frame by frame.

WHY (2026-09-22). divtime: 22 of the 125 GT divisions have their second daughter tracked as its own cell two or
more frames before the annotated division, 7-13 um from the mother's track from the start; geometry cannot say
when the annotator calls the split. The annotator looks at the image: two heatmap peaks can sit in one nucleus
before the intensity valley between them forms. This run reads, for every GT division whose two daughters are
matched, the smoothed raw intensity along the segment between the two predicted tracks at every frame from 6
before the division to 2 after: valley = min along the segment, peaks = the two ends, ratio = (valley - bg) /
(lower peak - bg). If the ratio drops at t0 +- 1 in the early cases, a late re-attachment has its trigger.
Output valley_rows.csv (movie, t_div, cls, offset, sep_um, ratio, peak_a, peak_b, valley, bg).
"""
src = open("divtime.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-22a-divtime"', 'SCRIPT_VERSION = "2026-09-22b-valley"')
rep('DT_FK_COLS = ["movie", "t_fork", "off", "on_lineage"]\n',
    'DT_FK_COLS = ["movie", "t_fork", "off", "on_lineage"]\n'
    'VL_ROWS = []\n'
    'VL_COLS = ["movie", "t_div", "cls", "off", "sep_um", "ratio", "peak_a", "peak_b", "valley", "bg"]\n'
    'VL_BEFORE, VL_AFTER, VL_N = 6, 2, 25\n'
    '\n'
    '\n'
    'def vl_profile(img, t, xa, xb):\n'
    '    """(sep_um, ratio, peak_a, peak_b, valley, bg) of the smoothed raw image along the segment xa -> xb (voxels) at frame t."""\n'
    '    xa, xb = np.asarray(xa, float), np.asarray(xb, float)\n'
    '    mid = 0.5 * (xa + xb)\n'
    '    sm, off = _fi_crop(img, t, mid)\n'
    '    if sm is None:\n'
    '        return (float("nan"),) * 6\n'
    '    sep = float(np.linalg.norm((xa - xb) * SCALE))\n'
    '    pts = np.linspace(0.0, 1.0, VL_N)[:, None] * (xb - xa)[None, :] + xa[None, :] - off[None, :]\n'
    '    ijk = np.round(pts).astype(int)\n'
    '    ok = np.all(ijk >= 0, axis=1) & np.all(ijk < np.asarray(sm.shape)[None, :], axis=1)\n'
    '    if ok.sum() < 5:\n'
    '        return (sep,) + (float("nan"),) * 5\n'
    '    v = sm[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]].astype(float)\n'
    '    k_end = max(2, int(round(2.0 / max(sep / VL_N, 1e-6))))          # samples within ~2 um of either end\n'
    '    pa, pb = float(v[:k_end].max()), float(v[-k_end:].max())\n'
    '    core = v[k_end:-k_end] if len(v) > 2 * k_end else v\n'
    '    val = float(core.min())\n'
    '    bg = float(np.median(sm))\n'
    '    lo = min(pa, pb) - bg\n'
    '    ratio = (val - bg) / lo if lo > 1e-6 else float("nan")\n'
    '    return sep, ratio, pa, pb, val, bg\n'
    '\n'
    '\n')

# inside dt_ledger: open the image once, and record the profile for every division with both daughters matched
rep("    forks = [f for f, ss in ps.items() if len(ss) >= 2]\n"
    "    divs = [m for m, ss in gs.items() if len(ss) >= 2]\n",
    "    forks = [f for f, ss in ps.items() if len(ss) >= 2]\n"
    "    divs = [m for m, ss in gs.items() if len(ss) >= 2]\n"
    "    try:\n"
    "        IMG_CTX.clear()\n"
    "        IMG_CTX.update(name=name, img=_open_image(pathlib.Path(root) / \"train\" / f\"{name}.zarr\"))\n"
    "        vl_img = IMG_CTX[\"img\"]\n"
    "    except Exception as _e:                                          # noqa: BLE001\n"
    "        warn_once(\"valley_open\", f\"{name}: image not opened ({_e})\")\n"
    "        vl_img = None\n")
rep("        DT_GT.append((name, t0, int(pm is not None), int(p1 is not None), int(p2 is not None), len(offs), best, cls,\n"
    "                      so, ln, sep0, sepm, same))\n",
    "        DT_GT.append((name, t0, int(pm is not None), int(p1 is not None), int(p2 is not None), len(offs), best, cls,\n"
    "                      so, ln, sep0, sepm, same))\n"
    "        if vl_img is not None and p1 is not None and p2 is not None:\n"
    "            pair = {t0 + 1: (p1, p2)}\n"
    "            for j_ in range(min(len(c1), len(c2), VL_BEFORE + 1)):\n"
    "                pair[t0 - j_] = (c1[j_], c2[j_])\n"
    "            a_, b_ = p1, p2\n"
    "            for j_ in range(1, VL_AFTER):\n"
    "                if len(ps.get(a_, [])) != 1 or len(ps.get(b_, [])) != 1:\n"
    "                    break\n"
    "                a_, b_ = ps[a_][0], ps[b_][0]\n"
    "                pair[t0 + 1 + j_] = (a_, b_)\n"
    "            for f_, (u_, v_) in sorted(pair.items()):\n"
    "                if p_t[u_] != f_ or p_t[v_] != f_:\n"
    "                    continue\n"
    "                prof = vl_profile(vl_img, f_, arr[u_, 1:], arr[v_, 1:])\n"
    "                VL_ROWS.append((name, t0, cls, f_ - t0, *prof))\n"
    "    IMG_CTX.clear()\n")

rep('print("DIVTIME OK", flush=True)\n',
    'print("DIVTIME OK", flush=True)\n'
    '_v = _pd.DataFrame(VL_ROWS, columns=VL_COLS)\n'
    '_v.to_csv("/kaggle/working/valley_rows.csv", index=False)\n'
    '_v["grp"] = np.where(_v.cls == "on_time", "on_time", np.where(_v.cls.isin(["no_fork", "early"]), "no_fork", "other"))\n'
    'print(f"\\n=== VALLEY: {len(_v)} (division, frame) rows over {_v.groupby([\'movie\', \'t_div\']).ngroups} divisions ===", flush=True)\n'
    'print("  median valley ratio by frame offset (0 = the annotated division frame; the daughters are at +1), n in brackets:", flush=True)\n'
    'for grp_ in ("no_fork", "on_time"):\n'
    '    g_ = _v[_v.grp == grp_]\n'
    '    line = []\n'
    '    for o_ in range(-VL_BEFORE, VL_AFTER + 1):\n'
    '        r_ = g_.loc[g_.off == o_, "ratio"].dropna()\n'
    '        line.append(f"{o_:+d}: {r_.median():.2f} [{len(r_)}]" if len(r_) else f"{o_:+d}: - [0]")\n'
    '    print(f"    {grp_:<8} " + "  ".join(line), flush=True)\n'
    'print("  median separation um by offset:", flush=True)\n'
    'for grp_ in ("no_fork", "on_time"):\n'
    '    g_ = _v[_v.grp == grp_]\n'
    '    print(f"    {grp_:<8} " + "  ".join(f"{o_:+d}: {g_.loc[g_.off == o_, \'sep_um\'].median():.1f}" for o_ in range(-VL_BEFORE, VL_AFTER + 1) if (g_.off == o_).any()), flush=True)\n'
    '_e = _v[(_v.grp == "no_fork")]\n'
    'print("  early cases with >= 3 frames before the division: the frame where the ratio first drops below a threshold (offset), histogram:", flush=True)\n'
    'for thr_ in (0.5, 0.6, 0.7, 0.8):\n'
    '    firsts = []\n'
    '    for (m_, t_), d_ in _e.groupby(["movie", "t_div"]):\n'
    '        d_ = d_.sort_values("off")\n'
    '        if (d_.off <= -3).sum() < 1 or d_.ratio.isna().all():\n'
    '            continue\n'
    '        below = d_[d_.ratio < thr_]\n'
    '        firsts.append(int(below.off.iloc[0]) if len(below) else 9)\n'
    '    from collections import Counter as _C\n'
    '    c_ = _C(firsts)\n'
    '    print(f"    thr {thr_}: " + ", ".join(f"{k_:+d}: {v_}" if k_ != 9 else f"never: {v_}" for k_, v_ in sorted(c_.items())) + f"   (n={len(firsts)}; within -1..+1: {sum(v_ for k_, v_ in c_.items() if -1 <= k_ <= 1)})", flush=True)\n'
    'print("VALLEY OK", flush=True)\n')

open("valley.py", "w", encoding="utf-8", newline="\n").write(src)
print("written valley.py", src.count("\n"), "lines")
