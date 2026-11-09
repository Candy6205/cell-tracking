"""hp44full.py -> fk44img.py: the 44b6 forks WITH the image features, for the joint image gate.

WHY (2026-09-17). forkimg (128 6bba movies) showed that raw-image CONTRAST at the mother and
both daughters separates true from false forks (per-feature AUC 0.83-0.84; gate AUC 0.88 ->
0.92-0.95 out of fold) and that the joint gate with the 11 image columns scores +0.0025
[+0.0004, +0.0049] over what ships. But the joint gate's 44b6 half (67 counted forks from
divbench5) has no image columns: on the hidden 44b6 movies the gate would apply contrast
splits it learned on 6bba only. This run produces the 44b6 forks with the same 11 columns:
hp44full's F4HP arm (all 71 44b6 movies, out-of-fold rankers, the production ILP and patches,
official scorer) + forkimg's block (FEAT_IMG, fork_img_feats, the movie opened around the
judge). forks_44b6img.csv is the deliverable; the gate tables print as before (the 44b6 folds
alone carry 26 GT divisions, so their curves are a sanity read, not the measurement).

Attach: hp44full's sources. CPU, ~2.5 h (hp44full 2 h + crops).
"""
import os

src = open("hp44full.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-09a-hp44full"', 'SCRIPT_VERSION = "2026-09-17a-fk44img"')
rep("FEAT_NAMES = FEAT20 + FEAT_NEW      # the first 20 are divbench3's, in order",
    'FEAT_IMG = ["i_p", "c_p", "r_p", "br_p", "c_c1", "c_q", "r_c1", "r_q", "sym_i", "sym_r", "dau_ratio"]\n'
    "FEAT_NAMES = FEAT20 + FEAT_NEW + FEAT_IMG      # the first 20 are divbench3's, in order; the last 11 read the image")

fi = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_forkimg.py"), encoding="utf-8").read()
BLOCK = fi[fi.index("BLOCK = r'''") + len("BLOCK = r'''"):fi.index("'''\nrep('\\ndef fork_rows(")]
assert "# ---- forkimg block ----" in BLOCK and "def fork_img_feats" in BLOCK
rep('\ndef fork_rows(name, k, seeds, aux, tp_forks, fp_forks):\n', BLOCK + '\n\ndef fork_rows(name, k, seeds, aux, tp_forks, fp_forks):\n')
rep('''                   steal=float(lg.get("steal", 0)))
        label = 2 if p in tp_n else (1 if p in fp_n else 0)     # 2 TP, 1 FP, 0 uncounted
''', '''                   steal=float(lg.get("steal", 0)))
        row.update(fork_img_feats(name, nodes, pred, p, c1, q))
        label = 2 if p in tp_n else (1 if p in fp_n else 0)     # 2 TP, 1 FP, 0 uncounted
''')
rep('''    row, ds, fr = _judge(name, k, outp, seeds, aux)
''', '''    _tfi = time.time()
    try:
        IMG_CTX.clear()
        IMG_CTX.update(name=name, img=_open_image(pathlib.Path(root) / "train" / f"{name}.zarr"))
    except Exception as _e:                                       # noqa: BLE001
        warn_once("forkimg_open", f"{name}: image not opened ({_e}); image features NaN")
        IMG_CTX.clear()
    row, ds, fr = _judge(name, k, outp, seeds, aux)
    IMG_CTX.clear()
    FI_STAT["seconds"] += int(time.time() - _tfi)
''')
rep('''                         ("28 feats, 2 folds", list(range(len(FEAT_NAMES))), FOLDS2),
                         ("28 feats, 3 folds", list(range(len(FEAT_NAMES))), FOLDS3)):''',
    '''                         ("28 feats, 2 folds", list(range(len(FEAT20) + len(FEAT_NEW))), FOLDS2),
                         ("28 feats, 3 folds", list(range(len(FEAT20) + len(FEAT_NEW))), FOLDS3),
                         ("20 feats + 11 image, 2 folds", J20 + JIMG, FOLDS2),
                         ("11 image alone, 2 folds", JIMG, FOLDS2)):''')
rep('J20 = list(range(len(FEAT20)))\n',
    'J20 = list(range(len(FEAT20)))\n'
    'JIMG = [FEAT_NAMES.index(f_) for f_ in FEAT_IMG]\n'
    'print(f"\\n  fk44img: image features on {FI_STAT[\'forks\']:,} forks, {FI_STAT[\'nan_rows\']:,} unreadable, "\n'
    '      f"{FI_STAT[\'no_img\']:,} without an image handle, {FI_STAT[\'seconds\'] / 60:.0f} min of crops", flush=True)\n')
rep('"/kaggle/working/forks_44b6.csv"', '"/kaggle/working/forks_44b6img.csv"')
rep('print("HP44FULL OK", flush=True)', 'print("FK44IMG OK", flush=True)')
rep('print(f"[degraded] HP44FULL INCOMPLETE: {len(ROWS)}/{len(BENCH)}", flush=True)',
    'print(f"[degraded] FK44IMG INCOMPLETE: {len(ROWS)}/{len(BENCH)}", flush=True)')
rep('print(f"\\nHP44FULL DONE in', 'print(f"\\nFK44IMG DONE in')
rep('f"hp44full_{DET_SRC}.csv"', 'f"fk44img_{DET_SRC}.csv"')

assert src.count("fork_img_feats(") == 2 and src.count("forks_44b6img.csv") >= 1
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "fk44img.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written fk44img.py", len(src))
