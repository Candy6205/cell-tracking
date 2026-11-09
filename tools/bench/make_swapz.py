"""swapimg.py -> swapz.py: what the swap-site nucleus IS -- one big nucleus or two stacked in z.

WHY (2026-09-22). swapdiv refuted the mitosis reading of the swap sites (an annotated division within 3 frames
after a swap in 1.2% against 0.4% at random true edges; none within 30 frames in 95%). swapimg read the raw
image along the a -> b segment only: flat in 82%, 1.6x the baseline brightness, 7 um apart. Two readings
remain: a large interphase nucleus that the fixed-size detector target splits into two shoulder peaks (the
half-max region around the raw maximum is long along a -> b), or two nuclei stacked diagonally in z (swapgeom:
5.4 um xy, 3.7 um z) whose intensities add along the segment (the z profile through a and through b shows two
maxima at different planes; the half-max region is tall in z). Same sites, same crops, four measures added:
  ext_ab / ext_z / ext_perp   half-max extent of the connected region around the raw maximum nearest A, um,
                              along the a -> b axis, along z, and along the in-plane perpendicular
  vol                         that region's volume, um^3
  nz_a / nz_b                 maxima of the smoothed z profile through a / through b (prominence 10% of the
                              crop range), and dz_ab = the plane distance between the z-maximum at a and at b
Baseline = the same measures at the random TP sites with their nearest other primary as b.
"""
src = open("swapimg.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-12a-swapimg"', 'SCRIPT_VERSION = "2026-09-22a-swapz"')

rep('    out["I_A"] = _val(Ac)\n'
    '    out["I_a"] = _val(ac)\n',
    '    out["I_A"] = _val(Ac)\n'
    '    out["I_a"] = _val(ac)\n'
    '    # ---- swapz: the half-max region around the raw maximum nearest A, its extents and volume ----\n'
    '    for k_ in ("ext_ab", "ext_z", "ext_perp", "vol", "nz_a", "nz_b", "dz_ab"):\n'
    '        out[k_] = np.nan\n'
    '    if len(pk):\n'
    '        m_ = pk[int(np.argmin(np.linalg.norm((pk - Ac) * SCALE, axis=1)))].astype(int)\n'
    '        if np.linalg.norm((m_ - Ac) * SCALE) <= R_MAX_UM:\n'
    '            half = lo_ + 0.5 * (float(sm[tuple(m_)]) - lo_)\n'
    '            lab, _n = ndi.label(sm > half)\n'
    '            reg = np.argwhere(lab == lab[tuple(m_)]).astype(float)\n'
    '            if len(reg):\n'
    '                ru = reg * SCALE[None, :]\n'
    '                if b is not None and np.all(np.isfinite(b)):\n'
    '                    u_ = (np.asarray(b, float) - np.asarray(a, float)) * SCALE\n'
    '                else:\n'
    '                    u_ = np.array([0.0, 1.0, 0.0])\n'
    '                u_ = u_ / max(np.linalg.norm(u_), 1e-9)\n'
    '                pz = np.array([1.0, 0.0, 0.0])\n'
    '                uxy = u_ * np.array([0.0, 1.0, 1.0])\n'
    '                uxy = uxy / max(np.linalg.norm(uxy), 1e-9)\n'
    '                perp = np.array([0.0, -uxy[2], uxy[1]])\n'
    '                pr_ = ru @ u_\n'
    '                out["ext_ab"] = float(pr_.max() - pr_.min())\n'
    '                out["ext_z"] = float((ru @ pz).max() - (ru @ pz).min())\n'
    '                out["ext_perp"] = float((ru @ perp).max() - (ru @ perp).min())\n'
    '                out["vol"] = float(len(reg) * SCALE[0] * SCALE[1] * SCALE[2])\n'
    '    ref_ = float(hi_ - lo_)\n'
    '\n'
    '    def _zprof(p):\n'
    '        y_, x_ = int(np.clip(round(p[1]), 0, sm.shape[1] - 1)), int(np.clip(round(p[2]), 0, sm.shape[2] - 1))\n'
    '        return sm[:, y_, x_]\n'
    '    za = _zprof(ac)\n'
    '    pa_ = profile_peaks(za, ref=ref_)\n'
    '    out["nz_a"] = len(pa_)\n'
    '    if b is not None and np.all(np.isfinite(b)):\n'
    '        zb = _zprof(np.asarray(b, float) - off)\n'
    '        pb_ = profile_peaks(zb, ref=ref_)\n'
    '        out["nz_b"] = len(pb_)\n'
    '        out["dz_ab"] = float(abs(pa_[int(np.argmax(za[pa_]))] - pb_[int(np.argmax(zb[pb_]))]))\n')

rep('KEYS = ["n_max6", "d_max_A", "d_cent_A", "d_a_A", "d_b_A", "d_max_a", "d_max_b", "n_prof", "dip",\n'
    '        "d_ab", "I_A", "I_a", "I_b"]\n',
    'KEYS = ["n_max6", "d_max_A", "d_cent_A", "d_a_A", "d_b_A", "d_max_a", "d_max_b", "n_prof", "dip",\n'
    '        "d_ab", "I_A", "I_a", "I_b", "ext_ab", "ext_z", "ext_perp", "vol", "nz_a", "nz_b", "dz_ab"]\n')

rep('    print(f"\\nSWAPIMG DONE in {(time.time() - t_all) / 60:.1f} min", flush=True)\n',
    '    print("\\n=== SWAPZ: the nucleus at the site (half-max region around the raw maximum nearest A) ===", flush=True)\n'
    '    print(f"  {\'measure\':<34} {\'swap med (p25/p75)\':>26} {\'base med (p25/p75)\':>26}", flush=True)\n'
    '    for k_, lab_ in (("ext_ab", "extent along a->b, um"), ("ext_perp", "extent in-plane perpendicular, um"),\n'
    '                     ("ext_z", "extent in z, um"), ("vol", "volume above half-max, um^3"),\n'
    '                     ("dz_ab", "planes between z-max at a and at b"), ("d_ab", "a-b distance, um")):\n'
    '        s_, b_ = RES["swap"], RES["base"]\n'
    '        print(f"  {lab_:<34} {q([o[k_] for o in s_], .5):>8.2f} ({q([o[k_] for o in s_], .25):.2f}/{q([o[k_] for o in s_], .75):.2f})"\n'
    '              f"   {q([o[k_] for o in b_], .5):>8.2f} ({q([o[k_] for o in b_], .25):.2f}/{q([o[k_] for o in b_], .75):.2f})", flush=True)\n'
    '    for k_, lab_ in (("nz_a", "z profile through a: 2+ maxima"), ("nz_b", "z profile through b: 2+ maxima")):\n'
    '        print(f"  {lab_:<34} {share([o[k_] for o in RES[\'swap\']], lambda v: v >= 2):>8.2f}"\n'
    '              f"                     {share([o[k_] for o in RES[\'base\']], lambda v: v >= 2):>8.2f}", flush=True)\n'
    '    for k_, lab_ in (("ext_ab", "extent along a->b >= 6 um"), ("ext_z", "extent in z >= 4 um"), ("dz_ab", "z-max planes differ by >= 2")):\n'
    '        thr_ = 6.0 if k_ == "ext_ab" else 4.0 if k_ == "ext_z" else 2.0\n'
    '        print(f"  {lab_:<34} {share([o[k_] for o in RES[\'swap\']], lambda v, t_=thr_: v >= t_):>8.2f}"\n'
    '              f"                     {share([o[k_] for o in RES[\'base\']], lambda v, t_=thr_: v >= t_):>8.2f}", flush=True)\n'
    '    sw1 = [o for o in RES["swap"] if o["n_max6"] == 1]\n'
    '    sw2 = [o for o in RES["swap"] if o["n_max6"] >= 2]\n'
    '    print(f"  swap sites with ONE raw maximum ({len(sw1)}): ext_ab med {q([o[\'ext_ab\'] for o in sw1], .5):.2f}, ext_z {q([o[\'ext_z\'] for o in sw1], .5):.2f}, "\n'
    '          f"vol {q([o[\'vol\'] for o in sw1], .5):.0f}; with TWO+ ({len(sw2)}): ext_ab {q([o[\'ext_ab\'] for o in sw2], .5):.2f}, ext_z {q([o[\'ext_z\'] for o in sw2], .5):.2f}, "\n'
    '          f"dz_ab {q([o[\'dz_ab\'] for o in sw2], .5):.1f}", flush=True)\n'
    '    print(f"\\nSWAPIMG DONE in {(time.time() - t_all) / 60:.1f} min", flush=True)\n')
rep('    print("SWAPIMG OK", flush=True)', '    print("SWAPZ OK", flush=True)')
rep('fh = open(OUT / "swapimg_sites.csv"', 'fh = open(OUT / "swapz_sites.csv"')

open("swapz.py", "w", encoding="utf-8", newline="\n").write(src)
print("written swapz.py", src.count("\n"), "lines")
