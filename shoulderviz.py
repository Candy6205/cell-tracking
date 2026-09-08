"""shoulderviz -- see the module docstring of make_shoulderviz.py."""
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
import imageio.v2 as imageio                                 # noqa: E402

SCRIPT_VERSION = "2026-09-08e-shoulderviz"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

CODE_ROOTS = [
    "/kaggle/input/datasets/faustine649/cell-tracking/cell tracking",
    "/kaggle/input/datasets/faustine649/cell-tracking",
    "/kaggle/input/cell-tracking/cell tracking",
    "/kaggle/input/cell-tracking",
]
code = next((c for c in CODE_ROOTS
             if pathlib.Path(c, "viz_cells.py").exists()), None)
assert code is not None, "attach faustine649/cell-tracking"
sys.path.insert(0, code)
from viz_cells import (find_root, geff_path, read_geff, open_image,   # noqa: E402
                       zarr_path, read_scale)

ROOT = find_root(None)
OUT = pathlib.Path("/kaggle/working")
SLAB = 9              # z voxels in the MIP, viz_cells' default
ZOOM_UM = 65.0        # crop half-width around a shoulder cell
ZOOM_DT = 5           # +- frames in a zoom clip
N_ZOOM = 4            # zoom clips per movie (each also gets a z sweep)
FPS = 6

SHOULDER = [
    ('6bba_268e1230',11,12000069,0.1972,0.2746),
    ('6bba_268e1230',42,43000356,0.1311,0.1431),
    ('6bba_268e1230',57,58000491,0.0793,0.1381),
    ('6bba_268e1230',58,59000499,0.0793,0.1381),
    ('6bba_268e1230',59,60000510,0.0862,0.1166),
    ('6bba_268e1230',60,61000520,0.0862,0.1166),
    ('6bba_268e1230',62,63000542,0.0798,0.1282),
    ('6bba_268e1230',63,64000552,0.0414,0.1134),
    ('6bba_268e1230',64,65000563,0.0612,0.097),
    ('6bba_268e1230',65,66000572,0.0436,0.1403),
    ('6bba_268e1230',66,67000580,0.0621,0.1163),
    ('6bba_268e1230',67,68000589,0.0621,0.1163),
    ('6bba_268e1230',71,72000632,0.0471,0.1434),
    ('6bba_268e1230',72,73000642,0.0784,0.1747),
    ('6bba_268e1230',73,74000652,0.0674,0.1544),
    ('6bba_268e1230',74,75000661,0.055,0.1562),
    ('6bba_268e1230',81,82000722,0.0327,0.1458),
    ('6bba_268e1230',83,84000741,0.0433,0.1408),
    ('6bba_268e1230',87,88000773,0.0302,0.1275),
    ('6bba_268e1230',88,89000781,0.0878,0.1448),
    ('6bba_268e1230',89,90000788,0.0858,0.1329),
    ('6bba_268e1230',90,91000797,0.0747,0.1064),
    ('6bba_268e1230',91,92000807,0.0757,0.1036),
    ('6bba_268e1230',92,93000816,0.0795,0.0969),
    ('6bba_268e1230',92,93000818,0.0405,0.1366),
    ('6bba_268e1230',93,94000824,0.0184,0.0577),
    ('6bba_268e1230',93,94000826,0.0444,0.1324),
    ('6bba_268e1230',94,95000834,0.0572,0.1374),
    ('6bba_268e1230',95,96000841,0.0633,0.1255),
    ('6bba_268e1230',96,97000848,0.0087,0.0786),
    ('6bba_268e1230',97,98000856,0.0509,0.067),
    ('6bba_268e1230',98,99000863,0.0398,0.0582),
    ('6bba_268e1230',99,100000869,0.0444,0.0722),
    ('6bba_3a1849c2',0,1000002,0.0333,0.0955),
    ('6bba_3a1849c2',0,1000004,0.0521,0.1436),
    ('6bba_3a1849c2',8,9000080,0.1071,0.135),
    ('6bba_3a1849c2',9,10000090,0.1198,0.1391),
    ('6bba_3a1849c2',10,11000100,0.1242,0.1448),
    ('6bba_3a1849c2',11,12000110,0.069,0.1184),
    ('6bba_3a1849c2',12,13000120,0.053,0.0711),
    ('6bba_3a1849c2',12,13000127,0.1082,0.1268),
    ('6bba_3a1849c2',13,14000131,0.0754,0.0979),
    ('6bba_3a1849c2',13,14000140,0.0926,0.1169),
    ('6bba_3a1849c2',14,15000143,0.1038,0.1314),
    ('6bba_3a1849c2',14,15000151,0.1008,0.1323),
    ('6bba_3a1849c2',15,16000155,0.0715,0.1291),
    ('6bba_3a1849c2',15,16000164,0.1114,0.1352),
    ('6bba_3a1849c2',16,17000174,0.1097,0.1374),
    ('6bba_3a1849c2',17,18000187,0.1097,0.1374),
    ('6bba_3a1849c2',18,19000197,0.1066,0.1279),
    ('6bba_3a1849c2',19,20000208,0.1046,0.1196),
    ('6bba_3a1849c2',20,21000219,0.1,0.1224),
    ('6bba_3a1849c2',20,21000223,0.1966,0.2922),
    ('6bba_3a1849c2',21,22000230,0.0893,0.1028),
    ('6bba_3a1849c2',22,23000241,0.0935,0.1031),
    ('6bba_3a1849c2',23,24000251,0.0956,0.1068),
    ('6bba_3a1849c2',24,25000262,0.1012,0.1156),
    ('6bba_3a1849c2',25,26000268,0.0553,0.1613),
    ('6bba_3a1849c2',25,26000272,0.102,0.1114),
    ('6bba_3a1849c2',26,27000277,0.0634,0.154),
    ('6bba_3a1849c2',26,27000282,0.0926,0.1128),
    ('6bba_3a1849c2',27,28000286,0.0634,0.154),
    ('6bba_3a1849c2',27,28000291,0.0926,0.1128),
    ('6bba_3a1849c2',28,29000295,0.0914,0.2497),
    ('6bba_3a1849c2',28,29000299,0.0901,0.1167),
    ('6bba_3a1849c2',29,30000304,0.0887,0.2337),
    ('6bba_3a1849c2',29,30000308,0.0991,0.1249),
    ('6bba_3a1849c2',30,31000313,0.089,0.2676),
    ('6bba_3a1849c2',30,31000317,0.097,0.1245),
    ('6bba_3a1849c2',31,32000322,0.0792,0.1471),
    ('6bba_3a1849c2',31,32000326,0.0768,0.1232),
    ('6bba_3a1849c2',32,33000332,0.0792,0.1471),
    ('6bba_3a1849c2',32,33000336,0.0768,0.1232),
    ('6bba_3a1849c2',35,36000362,0.1103,0.1938),
    ('6bba_3a1849c2',35,36000366,0.0805,0.0987),
    ('6bba_3a1849c2',36,37000377,0.0866,0.1158),
    ('6bba_3a1849c2',37,38000388,0.0866,0.1158),
    ('6bba_3a1849c2',38,39000395,0.1025,0.217),
    ('6bba_3a1849c2',38,39000399,0.0599,0.0946),
    ('6bba_3a1849c2',39,40000405,0.1391,0.2467),
    ('6bba_3a1849c2',39,40000410,0.0609,0.0969),
    ('6bba_3a1849c2',40,41000419,0.0852,0.1222),
    ('6bba_3a1849c2',41,42000428,0.0852,0.1222),
    ('6bba_3a1849c2',57,58000622,0.1613,0.1876),
    ('6bba_3a1849c2',58,59000634,0.1465,0.1927),
    ('6bba_3a1849c2',60,61000659,0.165,0.2312),
    ('6bba_3a1849c2',62,63000686,0.1202,0.222),
    ('6bba_3a1849c2',63,64000702,0.0986,0.2396),
    ('6bba_3a1849c2',64,65000716,0.1095,0.2275),
    ('6bba_3a1849c2',66,67000750,0.1173,0.2189),
    ('6bba_3a1849c2',69,70000798,0.1192,0.1916),
    ('6bba_3a1849c2',70,71000817,0.1096,0.1985),
    ('6bba_3a1849c2',75,76000895,0.13,0.1687),
    ('6bba_3a1849c2',75,76000898,0.1062,0.3071),
    ('6bba_3a1849c2',76,77000911,0.1393,0.1655),
    ('6bba_3a1849c2',76,77000913,0.0976,0.2605),
    ('6bba_3a1849c2',79,80000960,0.1376,0.1857),
    ('6bba_3a1849c2',83,84001015,0.2214,0.2842),
    ('6bba_3a1849c2',85,86001053,0.0935,0.1708),
    ('6bba_3a1849c2',86,87001071,0.1549,0.1875),
    ('6bba_3a1849c2',86,87001081,0.1742,0.2971),
    ('6bba_3a1849c2',91,92001158,0.0818,0.3325),
    ('6bba_3a1849c2',92,93001175,0.0901,0.2679),
    ('6bba_3a1849c2',93,94001191,0.0837,0.1653),
    ('6bba_3a1849c2',94,95001206,0.0904,0.1511),
    ('6bba_3a1849c2',95,96001223,0.1451,0.3346),
    ('6bba_3a1849c2',95,96001227,0.0966,0.162),
    ('6bba_3a1849c2',96,97001242,0.1451,0.3346),
    ('6bba_3a1849c2',96,97001246,0.0966,0.162),
    ('6bba_3a1849c2',97,98001261,0.1986,0.2058),
    ('6bba_3abfe10a',42,43000111,0.1025,0.1274),
    ('6bba_3abfe10a',43,44000127,0.0794,0.1373),
    ('6bba_3abfe10a',45,46000168,0.0925,0.11),
    ('6bba_3abfe10a',46,47000191,0.0776,0.0942),
    ('6bba_3abfe10a',47,48000207,0.0813,0.1416),
    ('6bba_3abfe10a',48,49000229,0.0813,0.1416),
    ('6bba_3abfe10a',49,50000251,0.0813,0.1416),
    ('6bba_3abfe10a',50,51000273,0.0813,0.1416),
    ('6bba_3abfe10a',51,52000295,0.1417,0.1417),
    ('6bba_3abfe10a',51,52000298,0.1066,0.2915),
    ('6bba_3abfe10a',52,53000324,0.1246,0.1246),
    ('6bba_3abfe10a',53,54000351,0.1111,0.1474),
    ('6bba_3abfe10a',54,55000380,0.1111,0.1474),
    ('6bba_3abfe10a',55,56000407,0.0406,0.1272),
    ('6bba_3abfe10a',56,57000441,0.0056,0.0649),
    ('6bba_3abfe10a',60,61000573,0.1064,0.1203),
    ('6bba_3abfe10a',60,61000596,0.2447,0.2857),
    ('6bba_3abfe10a',61,62000606,0.0882,0.0992),
    ('6bba_3abfe10a',62,63000640,0.0882,0.0992),
    ('6bba_3abfe10a',67,68000809,0.1494,0.1587),
    ('6bba_3abfe10a',68,69000856,0.0958,0.1626),
    ('6bba_3abfe10a',69,70000886,0.0894,0.1013),
    ('6bba_3abfe10a',70,71000919,0.0941,0.0941),
    ('6bba_3abfe10a',71,72000954,0.0941,0.0941),
    ('6bba_3abfe10a',80,81001251,0.1364,0.1573),
    ('6bba_3abfe10a',82,83001319,0.052,0.0938),
    ('6bba_3abfe10a',83,84001357,0.0686,0.1306),
    ('6bba_3abfe10a',85,86001433,0.1328,0.2115),
    ('6bba_3abfe10a',85,86001442,0.0885,0.2815),
    ('6bba_3abfe10a',86,87001468,0.1133,0.153),
    ('6bba_3abfe10a',86,87001477,0.1156,0.2391),
    ('6bba_3abfe10a',89,90001592,0.1618,0.2491),
    ('6bba_3abfe10a',89,90001607,0.092,0.2737),
    ('6bba_3abfe10a',91,92001681,0.1288,0.2188),
    ('6bba_3abfe10a',93,94001753,0.1107,0.2607),
    ('6bba_3abfe10a',94,95001759,0.1677,0.3103),
    ('6bba_3abfe10a',95,96001790,0.1879,0.2878),
    ('6bba_3abfe10a',97,98001859,0.1978,0.2851),
    ('6bba_3abfe10a',97,98001876,0.0715,0.2982),
    ('6bba_3abfe10a',99,100001928,0.2508,0.3145),
    ('44b6_a2bb48bb',5,247000000031,0.1977,0.2949),
    ('44b6_a2bb48bb',11,253000000011,0.1154,0.249),
    ('44b6_a2bb48bb',16,258000000011,0.0838,0.1816),
    ('44b6_a2bb48bb',30,272000000015,0.1486,0.2063),
    ('44b6_a2bb48bb',56,298000000018,0.0992,0.1632),
    ('44b6_a2bb48bb',57,299000000013,0.1607,0.2424),
    ('44b6_a2bb48bb',57,299000000018,0.092,0.1126),
    ('44b6_a2bb48bb',58,300000000018,0.0897,0.108),
    ('44b6_a2bb48bb',59,301000000017,0.1243,0.1487),
    ('44b6_a2bb48bb',59,301000000018,0.0727,0.0982),
    ('44b6_a2bb48bb',60,302000000018,0.0594,0.1027),
    ('44b6_a2bb48bb',61,303000000012,0.1161,0.2054),
    ('44b6_a2bb48bb',61,303000000018,0.061,0.0909),
    ('44b6_a2bb48bb',62,304000000018,0.0741,0.1366),
    ('44b6_a2bb48bb',63,305000000017,0.1176,0.1285),
    ('44b6_a2bb48bb',64,306000000017,0.1078,0.1327),
    ('44b6_a2bb48bb',65,307000000013,0.1785,0.2088),
    ('44b6_a2bb48bb',65,307000000017,0.1053,0.1291),
    ('44b6_a2bb48bb',66,308000000017,0.1189,0.1346),
    ('44b6_a2bb48bb',71,313000000013,0.1452,0.15),
    ('44b6_a2bb48bb',73,315000000017,0.1122,0.1411),
    ('44b6_a2bb48bb',74,316000000012,0.1055,0.1623),
    ('44b6_a2bb48bb',74,316000000017,0.1219,0.1441),
    ('44b6_a2bb48bb',76,318000000017,0.1401,0.1458),
    ('44b6_a2bb48bb',77,319000000012,0.1097,0.1626),
    ('44b6_a2bb48bb',77,319000000017,0.1361,0.1444),
    ('44b6_a2bb48bb',78,320000000012,0.1164,0.1272),
    ('44b6_a2bb48bb',79,321000000017,0.1051,0.1307),
    ('44b6_a2bb48bb',79,321000000018,0.0703,0.1382),
    ('44b6_a2bb48bb',80,322000000017,0.1155,0.135),
    ('44b6_a2bb48bb',81,323000000018,0.1144,0.1686),
    ('44b6_a2bb48bb',84,326000000013,0.0977,0.1499),
    ('44b6_a2bb48bb',85,327000000018,0.0996,0.1461),
    ('44b6_a2bb48bb',86,328000000016,0.0899,0.1818),
    ('44b6_a2bb48bb',87,329000000013,0.0991,0.1533),
    ('44b6_a2bb48bb',88,330000000015,0.0918,0.1321),
    ('44b6_a2bb48bb',89,331000000013,0.0986,0.137),
    ('44b6_a2bb48bb',89,331000000015,0.1146,0.12),
    ('44b6_a2bb48bb',90,332000000015,0.1378,0.1485),
    ('44b6_a2bb48bb',90,332000000016,0.1042,0.146),
    ('44b6_a2bb48bb',92,334000000012,0.1315,0.1439),
    ('44b6_a2bb48bb',94,336000000012,0.0736,0.1552),
    ('44b6_a2bb48bb',98,340000000012,0.1131,0.1598),
    ('44b6_a2bb48bb',99,341000000016,0.0582,0.111),
]


BY_MOVIE = defaultdict(list)
for _m, _t, _n, _hg, _hw in SHOULDER:
    BY_MOVIE[_m].append((_t, _n, _hg, _hw))
print(f"{len(SHOULDER)} shoulder points over {len(BY_MOVIE)} movies: "
      + ", ".join(f"{k} {len(v)}" for k, v in sorted(BY_MOVIE.items())),
      flush=True)

t0 = time.time()


def _slab(Z, zc, slab):
    z0 = max(0, int(zc) - slab // 2)
    z1 = min(Z, z0 + slab)
    return max(0, z1 - slab), z1


def _mip(img, t, zc, slab=SLAB, roi=None):
    z0, z1 = _slab(img.shape[1], zc, slab)
    sub = np.asarray(img[t, z0:z1])
    if roi is not None:
        y0, y1, x0, x1 = roi
        sub = sub[:, y0:y1, x0:x1]
    return sub.max(axis=0)


def _save(path_noext, frames, fps):
    """MP4 if ffmpeg is here, GIF if it is not -- and SAY WHICH.

    The user asked for mp4 because a GIF's palette and a single still both
    lose the dim cells this run is about. A silent fall back to GIF would
    hand back exactly the file that was rejected, so the fallback prints.
    """
    try:
        imageio.mimsave(f"{path_noext}.mp4", frames, fps=fps, codec="libx264",
                        quality=9, macro_block_size=1)
        return f"{path_noext}.mp4"
    except Exception as e:                                   # noqa: BLE001
        print(f"  [degraded] mp4 failed ({type(e).__name__}: {e}) -> GIF. "
              f"The frames are identical; only the container changed.",
              flush=True)
        imageio.mimsave(f"{path_noext}.gif", frames, fps=fps, loop=0)
        return f"{path_noext}.gif"


def _lohi(a):
    """Contrast from the CROP, not the whole frame.

    The whole-frame percentiles are set by the bright core of the embryo, so
    a dim cell at the edge of the stack renders as near-black. These cells
    ARE the dim ones; normalising on what is actually in view is the only way
    to see them, at the price that brightness is no longer comparable between
    clips. That price is worth it here and is stated in the title.
    """
    v = np.asarray(a, np.float32)
    return float(np.percentile(v, 2.0)), float(np.percentile(v, 99.7))


def _depth_style(dz, rgb, s_in, lw_in):
    """(size, linewidth, rgba, is_near) for markers |dz| voxels off-plane.

    v3 DROPPED every marker more than one voxel off the displayed plane, so
    circles blinked in and out and the user could not follow a cell at all --
    which is the same complaint as v1's, in a different disguise: a marker
    that vanishes tells you nothing about whether the CELL went away or the
    RENDERER did. So nothing is dropped any more. Distance from the plane is
    drawn instead: bright, thick and solid in-plane; fading, thinner and
    dashed as it goes away, with a floor so it never disappears entirely.
    The user, 2026-09-08: "连贯一点其实我也能看出来细胞下去了又或者暗掉了".

    v4a NOTE: v4's first cut faded to alpha 0.16 and hit that floor 8 planes
    out, which on a 64-plane stack meant every distant marker was invisible
    over bright tissue -- the drop-out bug AGAIN, this time achieved with
    opacity instead of an if-statement. The fade is now shallow and floored
    at 0.5, and the depth cue is carried mainly by RADIUS and dash density,
    which stay readable at any contrast. A dark backing ring (drawn by
    `_scat`) keeps the marker legible over a bright field.
    """
    d = np.abs(np.asarray(dz, float))
    a = np.clip(1.0 - d / 40.0, 0.50, 1.0)
    near = d <= 1.5
    size = s_in * np.clip(1.0 - d / 16.0, 0.30, 1.0)
    lw = np.where(near, lw_in, np.where(d <= 4.0, lw_in * 0.7, lw_in * 0.5))
    rgba = np.tile(np.asarray(rgb + (1.0,), float), (len(d), 1))
    rgba[:, 3] = a
    return size, lw, rgba, near


def _scat(ax, pts, dz, rgb, s_in, lw_in):
    """Two draws: solid for what is near this plane, dashed for what is not.

    matplotlib carries linestyle per collection, not per point, so the split
    is the only way to say 'this one is off-plane' with the line itself.
    """
    if not len(pts):
        return
    size, lw, rgba, near = _depth_style(dz, rgb, s_in, lw_in)
    # a dark backing ring first: half these crops are bright tissue, and a
    # thin cyan line on a white nucleus is as invisible as one that was never
    # drawn
    ax.scatter(pts[:, 1], pts[:, 0], s=size, facecolors="none",
               edgecolors=(0, 0, 0, 0.55), linewidths=lw * 2.6)
    for msk, ls in ((near, "solid"), (~near, (0, (2, 2)))):
        if msk.any():
            ax.scatter(pts[msk, 1], pts[msk, 0], s=size[msk],
                       facecolors="none", edgecolors=rgba[msk],
                       linewidths=lw[msk], linestyle=ls)


def _frame(ax, m, lo, hi, pts, pdz, marks, mdz, roi, title):
    """pts/marks: (n,2) yx in ROI pixels. pdz/mdz: voxels off the plane."""
    ax.clear()
    ax.imshow(m, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    _scat(ax, pts, pdz, (0.25, 0.75, 1.0), 90.0, 1.1)
    _scat(ax, marks, mdz, (1.0, 0.19, 0.19), 260.0, 2.2)
    ax.set_title(title, fontsize=9, color="w")
    ax.set_axis_off()


def render(name, hits):
    zp = zarr_path(ROOT, "train", name)
    img = open_image(zp)
    sc = read_scale(zp)                    # (z, y, x) um per voxel
    gt = read_geff(geff_path(ROOT, "train", name))
    T, Z = img.shape[0], img.shape[1]
    gbt = defaultdict(list)
    for tt, z_, y_, x_ in zip(gt["t"], gt["z"], gt["y"], gt["x"]):
        gbt[int(tt)].append((float(z_), float(y_), float(x_)))
    nid_xyz = {int(n): (int(tt), float(z_), float(y_), float(x_))
               for n, tt, z_, y_, x_ in zip(gt["node_ids"], gt["t"], gt["z"],
                                            gt["y"], gt["x"])}
    miss_bt = defaultdict(list)
    for tt, nn, hg, hw in hits:
        v = nid_xyz.get(nn)
        if v is None:
            print(f"  ! {name}: node {nn} not in the geff", flush=True)
            continue
        miss_bt[int(tt)].append((v[1], v[2], v[3], hg, hw))
    samp = np.asarray(img[0]).astype(np.float32)
    lo, hi = (float(np.percentile(samp, 1.0)),
              float(np.percentile(samp, 99.8)))
    zc = Z // 2

    # ---- overview -------------------------------------------------------
    # A GT cell OUTSIDE the slab would otherwise be drawn as a circle on
    # black, which reads as "the detector missed a cell that is not there".
    # It is not there IN THIS SLICE. So only in-slab cells are circled, and
    # the title says how many of the frame's GT that is.
    zs0, zs1 = _slab(Z, zc, SLAB)
    zmid = (zs0 + zs1 - 1) / 2.0
    fig, ax = plt.subplots(figsize=(6.4, 6.4), facecolor="k")
    fr = []
    for t in range(T):
        m = _mip(img, t, zc)
        allg = gbt.get(t, [])
        g = np.asarray([[y_, x_] for _z, y_, x_ in allg], float).reshape(-1, 2)
        gd = np.asarray([z_ - zmid for z_, _y, _x in allg], float)
        km = miss_bt.get(t, [])
        k = np.asarray([[y_, x_] for _z, y_, x_, _a, _b in km],
                       float).reshape(-1, 2)
        kd = np.asarray([z_ - zmid for z_, _y, _x, _a, _b in km], float)
        n_in = int((np.abs(gd) <= SLAB / 2.0).sum())
        _frame(ax, m, lo, hi, g, gd, k, kd, None,
               f"{name}  t={t}   slab z {zs0}-{zs1} of {Z}   GT {len(g)} "
               f"({n_in} in slab)   shoulder miss {len(k)}"
               f"\nsolid = on this slab, dashed+faint = that many z away")
        fig.canvas.draw()
        fr.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    _save(str(OUT / f"{name}_overview"), fr, FPS)
    plt.close(fig)
    print(f"  {name}_overview  {T} frames  ({time.time() - t0:.0f}s)",
          flush=True)

    # ---- zooms, dimmest first: those are the hardest cases --------------
    flat = sorted(((t, z_, y_, x_, hg, hw)
                   for t, lst in miss_bt.items()
                   for z_, y_, x_, hg, hw in lst), key=lambda r: r[4])
    step = max(1, len(flat) // N_ZOOM)
    for k, (t, z_, y_, x_, hg, hw) in enumerate(flat[::step][:N_ZOOM]):
        ry = int(round(ZOOM_UM / sc[1]))
        rx = int(round(ZOOM_UM / sc[2]))
        y0, y1 = max(0, int(y_) - ry), min(img.shape[2], int(y_) + ry)
        x0, x1 = max(0, int(x_) - rx), min(img.shape[3], int(x_) + rx)
        roi = (y0, y1, x0, x1)
        zz0, zz1 = _slab(Z, z_, 5)
        clo, chi = _lohi(np.asarray(img[t, :, y0:y1, x0:x1]))
        fig, ax = plt.subplots(figsize=(5.2, 5.2), facecolor="k")
        fr = []
        zmid2 = (zz0 + zz1 - 1) / 2.0
        for tt in range(max(0, t - ZOOM_DT), min(T, t + ZOOM_DT + 1)):
            m = _mip(img, tt, z_, slab=5, roi=roi)
            vis = [(zq, yy, xx) for zq, yy, xx in gbt.get(tt, [])
                   if y0 <= yy < y1 and x0 <= xx < x1]
            g = np.asarray([[yy - y0, xx - x0] for _z, yy, xx in vis],
                           float).reshape(-1, 2)
            gd = np.asarray([zq - zmid2 for zq, _y, _x in vis], float)
            vm = [(zq, yy, xx) for zq, yy, xx, _a, _b in miss_bt.get(tt, [])
                  if y0 <= yy < y1 and x0 <= xx < x1]
            mk = np.asarray([[yy - y0, xx - x0] for _z, yy, xx in vm],
                            float).reshape(-1, 2)
            md = np.asarray([zq - zmid2 for zq, _y, _x in vm], float)
            _frame(ax, m, clo, chi, g, gd, mk, md, roi,
                   f"{name} zoom{k}  t={tt}  slab z {zz0}-{zz1} of {Z}"
                   + (f"   <- the miss (h={hg:.3f}, win={hw:.3f})"
                      if tt == t else "")
                   + "\nsolid = on this slab, dashed+faint = that many z away")
            fig.canvas.draw()
            fr.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        _save(str(OUT / f"{name}_zoom{k}"), fr, 3)
        plt.close(fig)

        # ---- z sweep: ONE plane at a time through the whole stack, at the
        # miss frame. shoulderfeat found these cells sit at a z EDGE in 17 of
        # 18 movies, so the question a time clip cannot answer is whether the
        # cell is cut off by the end of the stack. This answers it directly:
        # watch where the marker's cell appears and whether it is still
        # growing when the planes run out.
        fig, ax = plt.subplots(figsize=(5.2, 5.2), facecolor="k")
        fr = []
        vis = [(zc2, yy, xx) for zc2, yy, xx in gbt.get(t, [])
               if y0 <= yy < y1 and x0 <= xx < x1]
        gxy = np.asarray([[yy - y0, xx - x0] for _z, yy, xx in vis],
                         float).reshape(-1, 2)
        gz = np.asarray([zc2 for zc2, _y, _x in vis], float)
        mxy = np.asarray([[y_ - y0, x_ - x0]], float)
        for zq in range(Z):
            m = np.asarray(img[t, zq, y0:y1, x0:x1])
            _frame(ax, m, clo, chi, gxy, gz - zq, mxy,
                   np.asarray([z_ - zq], float), roi,
                   f"{name} zsweep{k}  t={t}  z={zq}/{Z - 1}   "
                   f"the missed cell is at z={z_:.1f}"
                   + ("   <- HERE" if abs(z_ - zq) <= 1.5 else
                      f" ({z_ - zq:+.0f})")
                   + f"\n{len(gxy)} annotated cells here; solid = this "
                     f"plane, faint dashed = that many z off")
            fig.canvas.draw()
            fr.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        _save(str(OUT / f"{name}_zsweep{k}"), fr, 5)
        plt.close(fig)
    print(f"  {name}: {min(N_ZOOM, len(flat))} zooms, dimmest h_gt "
          f"{flat[0][4]:.3f} .. {flat[::step][:N_ZOOM][-1][4]:.3f}"
          f"  ({time.time() - t0:.0f}s)", flush=True)
    del img


for _name in sorted(BY_MOVIE):
    print(f"\n=== {_name} ({len(BY_MOVIE[_name])} shoulder misses) ===",
          flush=True)
    render(_name, BY_MOVIE[_name])

print(f"\nSHOULDERVIZ DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
print("SHOULDERVIZ OK", flush=True)
