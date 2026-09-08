"""shoulderfeat -- see the module docstring of make_shoulderfeat.py."""
import pathlib
import sys
import time
import collections
from collections import defaultdict

import numpy as np

SCRIPT_VERSION = "2026-09-08b-shoulderfeat"
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
from viz_cells import (find_root, geff_path, read_geff,        # noqa: E402
                       list_datasets)

ROOT = find_root(None)
SCALE = np.array([1.625, 0.40625, 0.40625])   # z, y, x um per voxel, fallback
NDEC = 10

# (dataset, t, node_id, h_gt, h_win) -- every GT cell with a heatmap
# response (win max >= 0.05) but no detection within 7 um, from
# gtsplit's CSVs. This is gtpeak's 'no-peak' bucket, both domains.
SHOULDER = [
    ('6bba_268e1230', 11, 12000069, 0.1972, 0.2746),
    ('6bba_268e1230', 42, 43000356, 0.1311, 0.1431),
    ('6bba_268e1230', 57, 58000491, 0.0793, 0.1381),
    ('6bba_268e1230', 58, 59000499, 0.0793, 0.1381),
    ('6bba_268e1230', 59, 60000510, 0.0862, 0.1166),
    ('6bba_268e1230', 60, 61000520, 0.0862, 0.1166),
    ('6bba_268e1230', 62, 63000542, 0.0798, 0.1282),
    ('6bba_268e1230', 63, 64000552, 0.0414, 0.1134),
    ('6bba_268e1230', 64, 65000563, 0.0612, 0.097),
    ('6bba_268e1230', 65, 66000572, 0.0436, 0.1403),
    ('6bba_268e1230', 66, 67000580, 0.0621, 0.1163),
    ('6bba_268e1230', 67, 68000589, 0.0621, 0.1163),
    ('6bba_268e1230', 71, 72000632, 0.0471, 0.1434),
    ('6bba_268e1230', 72, 73000642, 0.0784, 0.1747),
    ('6bba_268e1230', 73, 74000652, 0.0674, 0.1544),
    ('6bba_268e1230', 74, 75000661, 0.055, 0.1562),
    ('6bba_268e1230', 81, 82000722, 0.0327, 0.1458),
    ('6bba_268e1230', 83, 84000741, 0.0433, 0.1408),
    ('6bba_268e1230', 87, 88000773, 0.0302, 0.1275),
    ('6bba_268e1230', 88, 89000781, 0.0878, 0.1448),
    ('6bba_268e1230', 89, 90000788, 0.0858, 0.1329),
    ('6bba_268e1230', 90, 91000797, 0.0747, 0.1064),
    ('6bba_268e1230', 91, 92000807, 0.0757, 0.1036),
    ('6bba_268e1230', 92, 93000816, 0.0795, 0.0969),
    ('6bba_268e1230', 92, 93000818, 0.0405, 0.1366),
    ('6bba_268e1230', 93, 94000824, 0.0184, 0.0577),
    ('6bba_268e1230', 93, 94000826, 0.0444, 0.1324),
    ('6bba_268e1230', 94, 95000834, 0.0572, 0.1374),
    ('6bba_268e1230', 95, 96000841, 0.0633, 0.1255),
    ('6bba_268e1230', 96, 97000848, 0.0087, 0.0786),
    ('6bba_268e1230', 97, 98000856, 0.0509, 0.067),
    ('6bba_268e1230', 98, 99000863, 0.0398, 0.0582),
    ('6bba_268e1230', 99, 100000869, 0.0444, 0.0722),
    ('6bba_283bf9f1', 0, 1000022, 0.1149, 0.145),
    ('6bba_283bf9f1', 22, 23000770, 0.266, 0.3094),
    ('6bba_283bf9f1', 58, 59001753, 0.1931, 0.294),
    ('6bba_283bf9f1', 62, 63001861, 0.1409, 0.2369),
    ('6bba_283bf9f1', 63, 64001887, 0.1595, 0.27),
    ('6bba_283bf9f1', 98, 99002716, 0.088, 0.2124),
    ('6bba_3a1849c2', 0, 1000002, 0.0333, 0.0955),
    ('6bba_3a1849c2', 0, 1000004, 0.0521, 0.1436),
    ('6bba_3a1849c2', 8, 9000080, 0.1071, 0.135),
    ('6bba_3a1849c2', 9, 10000090, 0.1198, 0.1391),
    ('6bba_3a1849c2', 10, 11000100, 0.1242, 0.1448),
    ('6bba_3a1849c2', 11, 12000110, 0.069, 0.1184),
    ('6bba_3a1849c2', 12, 13000120, 0.053, 0.0711),
    ('6bba_3a1849c2', 12, 13000127, 0.1082, 0.1268),
    ('6bba_3a1849c2', 13, 14000131, 0.0754, 0.0979),
    ('6bba_3a1849c2', 13, 14000140, 0.0926, 0.1169),
    ('6bba_3a1849c2', 14, 15000143, 0.1038, 0.1314),
    ('6bba_3a1849c2', 14, 15000151, 0.1008, 0.1323),
    ('6bba_3a1849c2', 15, 16000155, 0.0715, 0.1291),
    ('6bba_3a1849c2', 15, 16000164, 0.1114, 0.1352),
    ('6bba_3a1849c2', 16, 17000174, 0.1097, 0.1374),
    ('6bba_3a1849c2', 17, 18000187, 0.1097, 0.1374),
    ('6bba_3a1849c2', 18, 19000197, 0.1066, 0.1279),
    ('6bba_3a1849c2', 19, 20000208, 0.1046, 0.1196),
    ('6bba_3a1849c2', 20, 21000219, 0.1, 0.1224),
    ('6bba_3a1849c2', 20, 21000223, 0.1966, 0.2922),
    ('6bba_3a1849c2', 21, 22000230, 0.0893, 0.1028),
    ('6bba_3a1849c2', 22, 23000241, 0.0935, 0.1031),
    ('6bba_3a1849c2', 23, 24000251, 0.0956, 0.1068),
    ('6bba_3a1849c2', 24, 25000262, 0.1012, 0.1156),
    ('6bba_3a1849c2', 25, 26000268, 0.0553, 0.1613),
    ('6bba_3a1849c2', 25, 26000272, 0.102, 0.1114),
    ('6bba_3a1849c2', 26, 27000277, 0.0634, 0.154),
    ('6bba_3a1849c2', 26, 27000282, 0.0926, 0.1128),
    ('6bba_3a1849c2', 27, 28000286, 0.0634, 0.154),
    ('6bba_3a1849c2', 27, 28000291, 0.0926, 0.1128),
    ('6bba_3a1849c2', 28, 29000295, 0.0914, 0.2497),
    ('6bba_3a1849c2', 28, 29000299, 0.0901, 0.1167),
    ('6bba_3a1849c2', 29, 30000304, 0.0887, 0.2337),
    ('6bba_3a1849c2', 29, 30000308, 0.0991, 0.1249),
    ('6bba_3a1849c2', 30, 31000313, 0.089, 0.2676),
    ('6bba_3a1849c2', 30, 31000317, 0.097, 0.1245),
    ('6bba_3a1849c2', 31, 32000322, 0.0792, 0.1471),
    ('6bba_3a1849c2', 31, 32000326, 0.0768, 0.1232),
    ('6bba_3a1849c2', 32, 33000332, 0.0792, 0.1471),
    ('6bba_3a1849c2', 32, 33000336, 0.0768, 0.1232),
    ('6bba_3a1849c2', 35, 36000362, 0.1103, 0.1938),
    ('6bba_3a1849c2', 35, 36000366, 0.0805, 0.0987),
    ('6bba_3a1849c2', 36, 37000377, 0.0866, 0.1158),
    ('6bba_3a1849c2', 37, 38000388, 0.0866, 0.1158),
    ('6bba_3a1849c2', 38, 39000395, 0.1025, 0.217),
    ('6bba_3a1849c2', 38, 39000399, 0.0599, 0.0946),
    ('6bba_3a1849c2', 39, 40000405, 0.1391, 0.2467),
    ('6bba_3a1849c2', 39, 40000410, 0.0609, 0.0969),
    ('6bba_3a1849c2', 40, 41000419, 0.0852, 0.1222),
    ('6bba_3a1849c2', 41, 42000428, 0.0852, 0.1222),
    ('6bba_3a1849c2', 57, 58000622, 0.1613, 0.1876),
    ('6bba_3a1849c2', 58, 59000634, 0.1465, 0.1927),
    ('6bba_3a1849c2', 60, 61000659, 0.165, 0.2312),
    ('6bba_3a1849c2', 62, 63000686, 0.1202, 0.222),
    ('6bba_3a1849c2', 63, 64000702, 0.0986, 0.2396),
    ('6bba_3a1849c2', 64, 65000716, 0.1095, 0.2275),
    ('6bba_3a1849c2', 66, 67000750, 0.1173, 0.2189),
    ('6bba_3a1849c2', 69, 70000798, 0.1192, 0.1916),
    ('6bba_3a1849c2', 70, 71000817, 0.1096, 0.1985),
    ('6bba_3a1849c2', 75, 76000895, 0.13, 0.1687),
    ('6bba_3a1849c2', 75, 76000898, 0.1062, 0.3071),
    ('6bba_3a1849c2', 76, 77000911, 0.1393, 0.1655),
    ('6bba_3a1849c2', 76, 77000913, 0.0976, 0.2605),
    ('6bba_3a1849c2', 79, 80000960, 0.1376, 0.1857),
    ('6bba_3a1849c2', 83, 84001015, 0.2214, 0.2842),
    ('6bba_3a1849c2', 85, 86001053, 0.0935, 0.1708),
    ('6bba_3a1849c2', 86, 87001071, 0.1549, 0.1875),
    ('6bba_3a1849c2', 86, 87001081, 0.1742, 0.2971),
    ('6bba_3a1849c2', 91, 92001158, 0.0818, 0.3325),
    ('6bba_3a1849c2', 92, 93001175, 0.0901, 0.2679),
    ('6bba_3a1849c2', 93, 94001191, 0.0837, 0.1653),
    ('6bba_3a1849c2', 94, 95001206, 0.0904, 0.1511),
    ('6bba_3a1849c2', 95, 96001223, 0.1451, 0.3346),
    ('6bba_3a1849c2', 95, 96001227, 0.0966, 0.162),
    ('6bba_3a1849c2', 96, 97001242, 0.1451, 0.3346),
    ('6bba_3a1849c2', 96, 97001246, 0.0966, 0.162),
    ('6bba_3a1849c2', 97, 98001261, 0.1986, 0.2058),
    ('6bba_3abfe10a', 42, 43000111, 0.1025, 0.1274),
    ('6bba_3abfe10a', 43, 44000127, 0.0794, 0.1373),
    ('6bba_3abfe10a', 45, 46000168, 0.0925, 0.11),
    ('6bba_3abfe10a', 46, 47000191, 0.0776, 0.0942),
    ('6bba_3abfe10a', 47, 48000207, 0.0813, 0.1416),
    ('6bba_3abfe10a', 48, 49000229, 0.0813, 0.1416),
    ('6bba_3abfe10a', 49, 50000251, 0.0813, 0.1416),
    ('6bba_3abfe10a', 50, 51000273, 0.0813, 0.1416),
    ('6bba_3abfe10a', 51, 52000295, 0.1417, 0.1417),
    ('6bba_3abfe10a', 51, 52000298, 0.1066, 0.2915),
    ('6bba_3abfe10a', 52, 53000324, 0.1246, 0.1246),
    ('6bba_3abfe10a', 53, 54000351, 0.1111, 0.1474),
    ('6bba_3abfe10a', 54, 55000380, 0.1111, 0.1474),
    ('6bba_3abfe10a', 55, 56000407, 0.0406, 0.1272),
    ('6bba_3abfe10a', 56, 57000441, 0.0056, 0.0649),
    ('6bba_3abfe10a', 60, 61000573, 0.1064, 0.1203),
    ('6bba_3abfe10a', 60, 61000596, 0.2447, 0.2857),
    ('6bba_3abfe10a', 61, 62000606, 0.0882, 0.0992),
    ('6bba_3abfe10a', 62, 63000640, 0.0882, 0.0992),
    ('6bba_3abfe10a', 67, 68000809, 0.1494, 0.1587),
    ('6bba_3abfe10a', 68, 69000856, 0.0958, 0.1626),
    ('6bba_3abfe10a', 69, 70000886, 0.0894, 0.1013),
    ('6bba_3abfe10a', 70, 71000919, 0.0941, 0.0941),
    ('6bba_3abfe10a', 71, 72000954, 0.0941, 0.0941),
    ('6bba_3abfe10a', 80, 81001251, 0.1364, 0.1573),
    ('6bba_3abfe10a', 82, 83001319, 0.052, 0.0938),
    ('6bba_3abfe10a', 83, 84001357, 0.0686, 0.1306),
    ('6bba_3abfe10a', 85, 86001433, 0.1328, 0.2115),
    ('6bba_3abfe10a', 85, 86001442, 0.0885, 0.2815),
    ('6bba_3abfe10a', 86, 87001468, 0.1133, 0.153),
    ('6bba_3abfe10a', 86, 87001477, 0.1156, 0.2391),
    ('6bba_3abfe10a', 89, 90001592, 0.1618, 0.2491),
    ('6bba_3abfe10a', 89, 90001607, 0.092, 0.2737),
    ('6bba_3abfe10a', 91, 92001681, 0.1288, 0.2188),
    ('6bba_3abfe10a', 93, 94001753, 0.1107, 0.2607),
    ('6bba_3abfe10a', 94, 95001759, 0.1677, 0.3103),
    ('6bba_3abfe10a', 95, 96001790, 0.1879, 0.2878),
    ('6bba_3abfe10a', 97, 98001859, 0.1978, 0.2851),
    ('6bba_3abfe10a', 97, 98001876, 0.0715, 0.2982),
    ('6bba_3abfe10a', 99, 100001928, 0.2508, 0.3145),
    ('6bba_5c824876', 15, 16000257, 0.1188, 0.1287),
    ('6bba_5c824876', 17, 18000299, 0.018, 0.1161),
    ('6bba_5c824876', 28, 29000523, 0.0073, 0.105),
    ('6bba_5c824876', 38, 39000735, 0.0655, 0.1363),
    ('6bba_5c824876', 42, 43000824, 0.1758, 0.2744),
    ('6bba_5c824876', 67, 68001208, 0.0529, 0.0997),
    ('6bba_5c824876', 68, 69001212, 0.0715, 0.1175),
    ('6bba_5c824876', 71, 72001254, 0.001, 0.3077),
    ('6bba_5c824876', 76, 77001300, 0.0637, 0.1243),
    ('6bba_5c824876', 93, 94001462, 0.0872, 0.1374),
    ('6bba_5c824876', 97, 98001502, 0.2252, 0.2588),
    ('6bba_5c824876', 98, 99001511, 0.0883, 0.1102),
    ('6bba_5c824876', 98, 99001512, 0.193, 0.244),
    ('6bba_5c824876', 99, 100001521, 0.1233, 0.1273),
    ('6bba_61dd1e0d', 59, 60000339, 0.0473, 0.1756),
    ('6bba_61dd1e0d', 65, 66000385, 0.061, 0.1494),
    ('6bba_61dd1e0d', 67, 68000397, 0.0684, 0.1633),
    ('6bba_61dd1e0d', 68, 69000404, 0.0735, 0.1246),
    ('6bba_61dd1e0d', 69, 70000411, 0.0735, 0.1246),
    ('6bba_61dd1e0d', 70, 71000417, 0.0691, 0.1292),
    ('6bba_61dd1e0d', 95, 96000617, 0.1272, 0.2975),
    ('6bba_61dd1e0d', 98, 99000648, 0.1048, 0.3323),
    ('6bba_61dd1e0d', 99, 100000658, 0.1048, 0.3323),
    ('6bba_7af54fde', 2, 3000041, 0.1297, 0.2816),
    ('6bba_7af54fde', 2, 3000046, 0.2033, 0.3001),
    ('6bba_7af54fde', 3, 4000057, 0.2213, 0.2724),
    ('6bba_7af54fde', 3, 4000060, 0.073, 0.1567),
    ('6bba_7af54fde', 3, 4000061, 0.175, 0.2342),
    ('6bba_7af54fde', 4, 5000073, 0.1637, 0.2262),
    ('6bba_7af54fde', 4, 5000078, 0.1971, 0.2293),
    ('6bba_7af54fde', 5, 6000093, 0.1101, 0.1826),
    ('6bba_7af54fde', 6, 7000108, 0.1727, 0.1936),
    ('6bba_7af54fde', 9, 10000166, 0.2012, 0.2681),
    ('6bba_7af54fde', 10, 11000178, 0.0953, 0.1354),
    ('6bba_7af54fde', 10, 11000183, 0.114, 0.2364),
    ('6bba_7af54fde', 23, 24000379, 0.2126, 0.2414),
    ('6bba_7af54fde', 25, 26000410, 0.1077, 0.1477),
    ('6bba_7af54fde', 26, 27000432, 0.0981, 0.1689),
    ('6bba_7af54fde', 27, 28000453, 0.0839, 0.1479),
    ('6bba_7af54fde', 28, 29000475, 0.0433, 0.159),
    ('6bba_7af54fde', 31, 32000538, 0.1615, 0.2563),
    ('6bba_7af54fde', 37, 38000657, 0.1137, 0.2825),
    ('6bba_7af54fde', 40, 41000703, 0.2287, 0.3583),
    ('6bba_7af54fde', 41, 42000718, 0.2287, 0.3583),
    ('6bba_7af54fde', 42, 43000733, 0.2287, 0.3583),
    ('6bba_7af54fde', 43, 44000748, 0.2287, 0.3583),
    ('6bba_7af54fde', 44, 45000764, 0.0586, 0.1474),
    ('6bba_7af54fde', 65, 66001048, 0.1945, 0.2398),
    ('6bba_7af54fde', 66, 67001060, 0.1872, 0.2466),
    ('6bba_7af54fde', 67, 68001072, 0.1784, 0.2776),
    ('6bba_7b5d3b2c', 0, 1000003, 0.0145, 0.1564),
    ('6bba_7b5d3b2c', 1, 2000010, 0.0404, 0.0974),
    ('6bba_7b5d3b2c', 2, 3000016, 0.0996, 0.1358),
    ('6bba_7b5d3b2c', 24, 25000153, 0.1168, 0.2884),
    ('6bba_7b5d3b2c', 25, 26000160, 0.114, 0.3),
    ('6bba_7b5d3b2c', 42, 43000282, 0.2261, 0.3384),
    ('6bba_afb141ff', 43, 44000394, 0.0753, 0.1759),
    ('6bba_afb141ff', 44, 45000411, 0.1352, 0.141),
    ('6bba_afb141ff', 45, 46000428, 0.118, 0.1421),
    ('6bba_afb141ff', 52, 53000509, 0.1865, 0.3104),
    ('6bba_afb141ff', 84, 85000822, 0.0546, 0.1962),
    ('6bba_afb141ff', 85, 86000832, 0.0835, 0.0946),
    ('6bba_afb141ff', 86, 87000843, 0.0816, 0.095),
    ('6bba_afb141ff', 87, 88000854, 0.111, 0.1278),
    ('6bba_afb141ff', 95, 96000912, 0.1107, 0.2123),
    ('6bba_c27cba08', 66, 67000629, 0.2448, 0.294),
    ('6bba_c27cba08', 69, 70000658, 0.2226, 0.2629),
    ('6bba_c27cba08', 70, 71000666, 0.1892, 0.2778),
    ('44b6_0b24845f', 19, 98000000036, 0.1432, 0.1711),
    ('44b6_0b24845f', 38, 117000000036, 0.0858, 0.2244),
    ('44b6_0b24845f', 39, 118000000036, 0.1231, 0.2356),
    ('44b6_0db75fae', 12, 53000000002, 0.039, 0.2262),
    ('44b6_87bba6c4', 38, 44000000021, 0.0922, 0.2008),
    ('44b6_87bba6c4', 50, 56000000022, 0.1451, 0.2207),
    ('44b6_87bba6c4', 51, 57000000022, 0.071, 0.2156),
    ('44b6_87bba6c4', 77, 83000000075, 0.0866, 0.1894),
    ('44b6_87bba6c4', 78, 84000000035, 0.0879, 0.227),
    ('44b6_8f5ab931', 1, 127000000046, 0.2451, 0.2815),
    ('44b6_8f5ab931', 3, 129000000062, 0.0737, 0.235),
    ('44b6_8f5ab931', 5, 131000000046, 0.1733, 0.2381),
    ('44b6_8f5ab931', 13, 139000000069, 0.0852, 0.1216),
    ('44b6_8f5ab931', 47, 173000000062, 0.1137, 0.2572),
    ('44b6_8f5ab931', 48, 174000000011, 0.1232, 0.2536),
    ('44b6_8f5ab931', 51, 177000000011, 0.0421, 0.2146),
    ('44b6_8f5ab931', 60, 186000000011, float("nan"), 0.1417),
    ('44b6_8f5ab931', 72, 198000000011, 0.0378, 0.2134),
    ('44b6_8f5ab931', 75, 201000000053, 0.1083, 0.2719),
    ('44b6_9bfa6a0a', 1, 181000000042, 0.282, 0.331),
    ('44b6_9bfa6a0a', 6, 186000000068, 0.1216, 0.1216),
    ('44b6_9bfa6a0a', 20, 200000000015, 0.2993, 0.3205),
    ('44b6_a2bb48bb', 5, 247000000031, 0.1977, 0.2949),
    ('44b6_a2bb48bb', 11, 253000000011, 0.1154, 0.249),
    ('44b6_a2bb48bb', 16, 258000000011, 0.0838, 0.1816),
    ('44b6_a2bb48bb', 30, 272000000015, 0.1486, 0.2063),
    ('44b6_a2bb48bb', 56, 298000000018, 0.0992, 0.1632),
    ('44b6_a2bb48bb', 57, 299000000013, 0.1607, 0.2424),
    ('44b6_a2bb48bb', 57, 299000000018, 0.092, 0.1126),
    ('44b6_a2bb48bb', 58, 300000000018, 0.0897, 0.108),
    ('44b6_a2bb48bb', 59, 301000000017, 0.1243, 0.1487),
    ('44b6_a2bb48bb', 59, 301000000018, 0.0727, 0.0982),
    ('44b6_a2bb48bb', 60, 302000000018, 0.0594, 0.1027),
    ('44b6_a2bb48bb', 61, 303000000012, 0.1161, 0.2054),
    ('44b6_a2bb48bb', 61, 303000000018, 0.061, 0.0909),
    ('44b6_a2bb48bb', 62, 304000000018, 0.0741, 0.1366),
    ('44b6_a2bb48bb', 63, 305000000017, 0.1176, 0.1285),
    ('44b6_a2bb48bb', 64, 306000000017, 0.1078, 0.1327),
    ('44b6_a2bb48bb', 65, 307000000013, 0.1785, 0.2088),
    ('44b6_a2bb48bb', 65, 307000000017, 0.1053, 0.1291),
    ('44b6_a2bb48bb', 66, 308000000017, 0.1189, 0.1346),
    ('44b6_a2bb48bb', 71, 313000000013, 0.1452, 0.15),
    ('44b6_a2bb48bb', 73, 315000000017, 0.1122, 0.1411),
    ('44b6_a2bb48bb', 74, 316000000012, 0.1055, 0.1623),
    ('44b6_a2bb48bb', 74, 316000000017, 0.1219, 0.1441),
    ('44b6_a2bb48bb', 76, 318000000017, 0.1401, 0.1458),
    ('44b6_a2bb48bb', 77, 319000000012, 0.1097, 0.1626),
    ('44b6_a2bb48bb', 77, 319000000017, 0.1361, 0.1444),
    ('44b6_a2bb48bb', 78, 320000000012, 0.1164, 0.1272),
    ('44b6_a2bb48bb', 79, 321000000017, 0.1051, 0.1307),
    ('44b6_a2bb48bb', 79, 321000000018, 0.0703, 0.1382),
    ('44b6_a2bb48bb', 80, 322000000017, 0.1155, 0.135),
    ('44b6_a2bb48bb', 81, 323000000018, 0.1144, 0.1686),
    ('44b6_a2bb48bb', 84, 326000000013, 0.0977, 0.1499),
    ('44b6_a2bb48bb', 85, 327000000018, 0.0996, 0.1461),
    ('44b6_a2bb48bb', 86, 328000000016, 0.0899, 0.1818),
    ('44b6_a2bb48bb', 87, 329000000013, 0.0991, 0.1533),
    ('44b6_a2bb48bb', 88, 330000000015, 0.0918, 0.1321),
    ('44b6_a2bb48bb', 89, 331000000013, 0.0986, 0.137),
    ('44b6_a2bb48bb', 89, 331000000015, 0.1146, 0.12),
    ('44b6_a2bb48bb', 90, 332000000015, 0.1378, 0.1485),
    ('44b6_a2bb48bb', 90, 332000000016, 0.1042, 0.146),
    ('44b6_a2bb48bb', 92, 334000000012, 0.1315, 0.1439),
    ('44b6_a2bb48bb', 94, 336000000012, 0.0736, 0.1552),
    ('44b6_a2bb48bb', 98, 340000000012, 0.1131, 0.1598),
    ('44b6_a2bb48bb', 99, 341000000016, 0.0582, 0.111),
    ('44b6_c96cfa10', 40, 123000000075, 0.091, 0.1212),
    ('44b6_c96cfa10', 61, 144000000034, 0.0526, 0.0904),
    ('44b6_c96cfa10', 63, 146000000023, 0.209, 0.2591),
    ('44b6_c96cfa10', 63, 146000000034, 0.0375, 0.1096),
    ('44b6_c96cfa10', 64, 147000000034, float("nan"), 0.0515),
    ('44b6_c96cfa10', 65, 148000000034, float("nan"), 0.0981),
    ('44b6_c96cfa10', 71, 154000000034, float("nan"), 0.1013),
    ('44b6_c96cfa10', 76, 159000000034, float("nan"), 0.0784),
    ('44b6_c96cfa10', 77, 160000000034, float("nan"), 0.0644),
    ('44b6_c96cfa10', 80, 163000000034, float("nan"), 0.0811),
    ('44b6_c96cfa10', 81, 164000000034, float("nan"), 0.0681),
    ('44b6_c96cfa10', 82, 165000000034, float("nan"), 0.0597),
    ('44b6_c96cfa10', 83, 166000000034, float("nan"), 0.0703),
    ('44b6_c96cfa10', 84, 167000000034, float("nan"), 0.0897),
    ('44b6_c96cfa10', 85, 168000000034, float("nan"), 0.0662),
    ('44b6_c96cfa10', 87, 170000000032, 0.0659, 0.1679),
    ('44b6_c96cfa10', 88, 171000000032, 0.0776, 0.1635),
    ('44b6_c96cfa10', 92, 175000000020, float("nan"), 0.0807),
    ('44b6_c96cfa10', 92, 175000000033, float("nan"), 0.0773),
    ('44b6_c96cfa10', 93, 176000000020, float("nan"), 0.0708),
    ('44b6_c96cfa10', 94, 177000000020, float("nan"), 0.052),
    ('44b6_c96cfa10', 94, 177000000033, float("nan"), 0.1094),
    ('44b6_e29f0176', 57, 78000000016, 0.2665, 0.2975),
]

# the movies gtpeak/gtsplit actually measured; every OTHER train
# movie has no shoulder labels at all, so including it would put
# unlabelled cells in the denominator of every rate below.
MEASURED = [
    '6bba_2312ac41',
    '6bba_268e1230',
    '6bba_283bf9f1',
    '6bba_3a1849c2',
    '6bba_3abfe10a',
    '6bba_5c824876',
    '6bba_61dd1e0d',
    '6bba_7af54fde',
    '6bba_7b5d3b2c',
    '6bba_aeee7805',
    '6bba_afb141ff',
    '6bba_c27cba08',
    '44b6_0b24845f',
    '44b6_0db75fae',
    '44b6_33b596bf',
    '44b6_5740d24b',
    '44b6_87bba6c4',
    '44b6_8f5ab931',
    '44b6_9bfa6a0a',
    '44b6_a2bb48bb',
    '44b6_c15fded2',
    '44b6_c96cfa10',
    '44b6_d2f34f90',
    '44b6_e29f0176',
]


BY_MOVIE = defaultdict(list)
for _m, _t, _n, _hg, _hw in SHOULDER:
    BY_MOVIE[_m].append((_t, _n, _hg, _hw))
print(f"{len(SHOULDER)} shoulder cells over {len(BY_MOVIE)} movies", flush=True)

t0 = time.time()

# v1 ran over all 199 train movies of the two domains. That was wrong: only 24
# of them were ever measured by gtpeak, so the other 175 contributed ~120,000
# cells labelled "not shoulder" because nobody looked, and every rate in every
# table was diluted by whatever those movies' z and t distributions happen to
# be. v1's decile tables are superseded. Its PER-MOVIE table was never
# affected -- each row uses only its own movie's cells.
MOVIES = [d for d in list_datasets(ROOT, "train") if d in set(MEASURED)]
assert len(MOVIES) == len(MEASURED), (len(MOVIES), len(MEASURED))
print(f"{len(MOVIES)} MEASURED movies -- the other train movies in these two "
      f"domains have no shoulder labels at all and would only dilute the "
      f"denominators", flush=True)

ROWS = []
PER_MOVIE = []


def _runs(sh_ids, kids, par):
    """Lengths of maximal chains of shoulder cells that are linked in the GT.

    A chain is walked back to its first shoulder node and then forward, so a
    run is counted once no matter which of its members the outer loop reaches
    first. Divisions make the lineage a tree, not a path: a run that hits a
    fork continues down ONE branch and the other branch, being unvisited, is
    picked up as a separate run. That is the conservative reading -- it can
    only make runs look SHORTER, never longer, so "n of these are a single
    frame" is never inflated by it.
    """
    runs, seen = [], set()
    for n in sorted(sh_ids):
        if n in seen:
            continue
        cur, guard = n, 0
        while guard < 100000:
            guard += 1
            ps = [p for p in par.get(cur, ()) if p in sh_ids and p not in seen]
            if not ps:
                break
            cur = ps[0]
        L = 0
        while cur in sh_ids and cur not in seen:
            seen.add(cur)
            L += 1
            nx = [c for c in kids.get(cur, ()) if c in sh_ids and c not in seen]
            if not nx:
                break
            cur = nx[0]
        if L:
            runs.append(L)
    return runs


def _links(node_ids, edges):
    """parent -> children and child -> parents, on the raw geff ids."""
    kids = defaultdict(list)
    par = defaultdict(list)
    known = set(int(n) for n in node_ids)
    for a, b in edges:
        a, b = int(a), int(b)
        if a in known and b in known:
            kids[a].append(b)
            par[b].append(a)
    return kids, par


for name in MOVIES:
    g = read_geff(geff_path(ROOT, "train", name))
    sc = np.asarray(g.get("scale") or SCALE, float)
    nid = np.asarray(g["node_ids"], np.int64)
    tt = np.asarray(g["t"], float)
    zz = np.asarray(g["z"], float) * sc[0]
    yy = np.asarray(g["y"], float) * sc[1]
    xx = np.asarray(g["x"], float) * sc[2]
    hits = {int(n): (hg, hw) for _t, n, hg, hw in BY_MOVIE.get(name, [])}
    is_sh = np.array([int(n) in hits for n in nid], bool)
    if int(is_sh.sum()) != len(hits):
        print(f"  ! {name}: {int(is_sh.sum())} of {len(hits)} shoulder ids "
              f"found in the geff -- IDS DO NOT MATCH, every table is suspect",
              flush=True)
    lo_z, hi_z = np.percentile(zz, [1.0, 99.0])
    zn = (zz - lo_z) / max(1e-9, hi_z - lo_z)
    lo_t, hi_t = float(tt.min()), float(tt.max())
    tn = (tt - lo_t) / max(1e-9, hi_t - lo_t)

    nn = np.full(len(nid), np.nan)
    byt = defaultdict(list)
    for i, t_ in enumerate(tt):
        byt[int(t_)].append(i)
    for t_, idx in byt.items():
        if len(idx) < 2:
            continue
        p = np.stack([zz[idx], yy[idx], xx[idx]], 1)
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        nn[idx] = d.min(axis=1)

    kids, par = _links(nid, g["edges"])
    sh_ids = set(int(n) for n in nid[is_sh])
    at_div = np.zeros(len(nid), bool)
    for i, n in enumerate(nid):
        n = int(n)
        if len(kids.get(n, ())) > 1:
            at_div[i] = True
            continue
        for p_ in par.get(n, ()):
            if len(kids.get(p_, ())) > 1:
                at_div[i] = True
        for c_ in kids.get(n, ()):
            if len(kids.get(c_, ())) > 1:
                at_div[i] = True

    runs = _runs(sh_ids, kids, par)
    assert sum(runs) == len(sh_ids), (sum(runs), len(sh_ids), name)

    for i, n in enumerate(nid):
        hg, _hw = hits.get(int(n), (np.nan, np.nan))
        ROWS.append((name, name[:4], bool(is_sh[i]), float(zn[i]),
                     float(zz[i]), float(tn[i]), float(nn[i]), bool(at_div[i]),
                     len(par.get(int(n), ())) == 0,
                     len(kids.get(int(n), ())) == 0, float(hg)))
    PER_MOVIE.append((name, len(nid), int(is_sh.sum()), runs,
                      float(np.median(zn[is_sh])) if is_sh.any() else np.nan,
                      float(np.median(zn)),
                      float(np.median(zz[is_sh])) if is_sh.any() else np.nan,
                      float(np.median(zz))))
    print(f"  {name:<16} N={len(nid):>5}  shoulder={int(is_sh.sum()):>3}  "
          f"z {zz.min():.0f}-{zz.max():.0f} um  "
          f"runs={sorted(runs, reverse=True)[:8]}  ({time.time() - t0:.0f}s)",
          flush=True)

DOM = np.array([r[1] for r in ROWS])
SH = np.array([r[2] for r in ROWS], bool)
ZN = np.array([r[3] for r in ROWS], float)
ZU = np.array([r[4] for r in ROWS], float)
TN = np.array([r[5] for r in ROWS], float)
NNB = np.array([r[6] for r in ROWS], float)
DIV = np.array([r[7] for r in ROWS], bool)
TSTART = np.array([r[8] for r in ROWS], bool)
TEND = np.array([r[9] for r in ROWS], bool)
HG = np.array([r[10] for r in ROWS], float)
print(f"\n{len(ROWS)} GT cells, {int(SH.sum())} of them shoulder "
      f"({100.0 * SH.mean():.2f}%)", flush=True)
assert int(SH.sum()) == len(SHOULDER), (int(SH.sum()), len(SHOULDER))


def decile_table(val, title, unit=""):
    print(f"\n=== {title} ===", flush=True)
    print(f"  {'bin':<16} {'6bba n':>8} {'6bba sh':>8} {'rate':>7}   "
          f"{'44b6 n':>8} {'44b6 sh':>8} {'rate':>7}", flush=True)
    ok = np.isfinite(val)
    edges = np.percentile(val[ok], np.linspace(0, 100, NDEC + 1))
    edges[-1] += 1e-9
    for k in range(NDEC):
        m = ok & (val >= edges[k]) & (val < edges[k + 1])
        cells = []
        for dm in ("6bba", "44b6"):
            q = m & (DOM == dm)
            n = int(q.sum())
            s = int((q & SH).sum())
            cells.append(f"{n:>8} {s:>8} {100.0 * s / max(1, n):>6.2f}%")
        lbl = f"{edges[k]:.2f}-{edges[k + 1]:.2f}{unit}"
        print(f"  {lbl:<16} " + "   ".join(cells), flush=True)


decile_table(ZN, "A. DEPTH: shoulder rate by normalised z (0 = one edge of "
                 "the stack)")
# v1's per-movie table showed the shoulder cells sit nearer a z EDGE than
# their own movie's cells in 17 of 18 movies (sign test p = 1.5e-5), at BOTH
# ends, which a monotone depth table cannot express. So fold the axis: 0 =
# hard against either edge, 0.5 = the middle of the stack.
decile_table(np.minimum(ZN, 1.0 - ZN),
             "A3. DISTANCE TO THE NEARER z EDGE (0 = at an edge, 0.5 = "
             "mid-stack) -- THE ONE THAT MATTERS")
decile_table(ZU, "A2. DEPTH in raw um (pools movies with different stacks)",
             "um")
decile_table(TN, "C. TIME: shoulder rate by normalised t")
decile_table(NNB, "E. CROWDING: nearest ANNOTATED neighbour in um -- WEAK")
print("  ^ GT annotates 9.71% of 6bba and 0.77% of 44b6, so this is mostly a "
      "fact\n    about the annotation, not the tissue. It cannot refute "
      "anything.", flush=True)

print("\n=== B. is the DIMNESS explained by DEPTH? ===", flush=True)
n_h = int(np.isfinite(HG).sum())
print(f"  h_gt is available for {n_h} of {len(ROWS)} cells"
      + ("" if n_h > len(SHOULDER) + 10 else
         " -- the CSVs carried it for the shoulder cells only, so this is a "
         "within-group\n  correlation: among cells that already failed, are "
         "the deeper ones the dimmer ones?"), flush=True)
for dm in ("6bba", "44b6"):
    q = (DOM == dm) & np.isfinite(HG)
    if int(q.sum()) < 10:
        continue
    r = float(np.corrcoef(ZN[q], HG[q])[0, 1])
    print(f"  {dm}: corr(normalised z, h at the GT voxel) = {r:+.3f} over "
          f"{int(q.sum())} cells", flush=True)
    for k in range(4):
        m = q & (ZN >= k / 4) & (ZN < (k + 1) / 4 + (1e-9 if k == 3 else 0))
        if m.any():
            print(f"    z {k / 4:.2f}-{(k + 1) / 4:.2f}: n={int(m.sum()):>3} "
                  f"h_gt p50 {float(np.median(HG[m])):.3f}", flush=True)

print("\n=== D. LINEAGE: where do the 313 sit in the tracks? ===", flush=True)
for dm in ("6bba", "44b6"):
    q = DOM == dm
    tot = int((q & SH).sum())
    if not tot:
        continue
    print(f"  {dm}: {tot} shoulder cells of {int(q.sum())} GT cells",
          flush=True)
    for lbl, msk in (("next to a division", DIV),
                     ("track START (no parent)", TSTART),
                     ("track END (no child)", TEND)):
        a = int((q & SH & msk).sum())
        b = int((q & msk).sum())
        print(f"    {lbl:<24} {a:>4} / {tot} = {100.0 * a / tot:>5.1f}%   "
              f"(all GT: {100.0 * b / max(1, int(q.sum())):>5.1f}%)", flush=True)

ALLRUNS = [L for _n, _N, _s, rr, *_ in PER_MOVIE for L in rr]
print("\n  run length of CONSECUTIVE shoulder frames on one lineage:",
      flush=True)
if ALLRUNS:
    c = collections.Counter(ALLRUNS)
    for L in sorted(c):
        print(f"    {L:>2} frame(s): {c[L]:>4} runs = {L * c[L]:>4} cells "
              f"({100.0 * L * c[L] / len(SHOULDER):>5.1f}% of the 313)",
              flush=True)
    print(f"    {len(ALLRUNS)} runs in total, longest {max(ALLRUNS)}",
          flush=True)
    iso = c.get(1, 0)
    print(f"\n  READ IT AS: {iso} of the {len(ALLRUNS)} runs are a SINGLE "
          f"frame. Those are the ones a\n  one-frame bridge can reach at all; "
          f"the rest are holes gap_close cannot span at\n  GAP_SPAN=1, and no "
          f"amount of better point-picking changes that.", flush=True)

print("\n=== per movie ===", flush=True)
print(f"  {'movie':<16} {'N':>6} {'sh':>4} {'rate':>7} {'z50 sh':>8} "
      f"{'z50 all':>8} {'um sh':>7} {'um all':>7}", flush=True)
for nm, N, s, rr, z1, z2, u1, u2 in sorted(PER_MOVIE, key=lambda r: -r[2]):
    print(f"  {nm:<16} {N:>6} {s:>4} {100.0 * s / max(1, N):>6.2f}% "
          f"{z1:>8.3f} {z2:>8.3f} {u1:>7.1f} {u2:>7.1f}", flush=True)

print(f"\nSHOULDERFEAT DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
print("SHOULDERFEAT OK", flush=True)
