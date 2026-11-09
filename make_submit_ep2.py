"""make_submit_ep2.py -- generate submit_ep2.py from submit_ep.py (ref 55974245,
LB 0.932): the same submission with TWO official edge heads run inline
(pack split_0 = head 1, pilkwang seed 314159 = head 2) and a ranker pkl
from biohub-edgeprob-rank2 whose `extra_cols` name any of the nine head
columns (p_fwd p_rev h_fused / p2_fwd p2_rev h2_fused / p_avg_fwd
p_avg_rev h_avg). Optionally NO ranker at all (HEAD_ONLY_COL set): the
named column is the edge probability the ILP sees, exactly as
edgeprob_rank2.py's HEAD1 / HEAD12 arms score it.

Marker-sliced, not line-matched: the two regions submit_ep.py's generator
inserted (the ranker block, the head block) are replaced whole; every
other byte of submit_ep.py is kept. `ast.parse` at the end.

    py make_submit_ep2.py            -> submit_ep2.py
"""
import ast
import pathlib

SRC = pathlib.Path(__file__).with_name("submit_ep.py")
DST = pathlib.Path(__file__).with_name("submit_ep2.py")
text = SRC.read_text(encoding="utf-8")

# ---- choose the arm here (from biohub-edgeprob-rank2's verdict) -----------
RANKER_NAME = None          # e.g. "edge_ranker_ep175v3_both.pkl"; None = head-only
HEAD_ONLY_COL = "h_fused"   # "h_fused" / "h_avg" when RANKER_NAME is None
# 2026-09-04: set to head-only (HEAD1) as the base of submit_off.py; the
# rank2 arms with a second head were all negative and are not submitted.


def rep(old, new, n=1):
    global text
    c = text.count(old)
    assert c == n, (c, old[:80])
    text = text.replace(old, new)


def cut(start_marker, end_marker, new, keep_end=True):
    """Replace [start_marker, end_marker) with `new`; with keep_end False
    the end marker itself is replaced too."""
    global text
    assert text.count(start_marker) == 1, start_marker
    a = text.index(start_marker)
    b = text.index(end_marker, a)
    if not keep_end:
        b += len(end_marker)
    text = text[:a] + new + text[b:]


rep('SCRIPT_VERSION = "2026-09-03a-submit-ep"',
    'SCRIPT_VERSION = "2026-09-03b-submit-ep2"')

# ---- 1. the ranker block ---------------------------------------------------
cut('# 2026-09-03 (submit_ep): the ranker is edge_ranker_ep175_bidir.pkl from',
    '''f"{' '.join(EXTRA_COLS)} | h_fused: {_blob.get('h_fused')}", flush=True)\n''',
    f'''# 2026-09-03 (submit_ep2): the ranker is a biohub-edgeprob-rank2 pkl --
# trained on the 18 pair_features columns PLUS a named subset of the nine
# head columns (two official heads, both directions, and their average).
# REQUIRED by name; its extra_cols decide which head columns are built.
# With RANKER_NAME None the ILP is fed HEAD_ONLY_COL directly (no ranker),
# the HEAD1 / HEAD12 arms of edgeprob_rank2.py.
RANKER_NAME = {RANKER_NAME!r}
HEAD_ONLY_COL = {HEAD_ONLY_COL!r}
MIN_POOL = 170
NFEAT_BASE = 18
HEAD_COL_NAMES = ["p_fwd", "p_rev", "h_fused", "p2_fwd", "p2_rev", "h2_fused",
                  "p_avg_fwd", "p_avg_rev", "h_avg"]
BASE_NAMES = ["d_um", "dz_um", "abs_dz_um", "dxy_um", "md_um", "d_minus_md",
              "score_i", "score_j", "abs_logit_diff", "dens_i", "dens_j",
              "n_prev", "n_curr", "rank_of_j_for_i", "n_cand_i", "n_cand_j",
              "speed_i", "t_frac"]           # pair_features row order
assert len(BASE_NAMES) == NFEAT_BASE
assert (RANKER_NAME is None) != (HEAD_ONLY_COL is None), \\
    "exactly one of RANKER_NAME / HEAD_ONLY_COL must be set"
if RANKER_NAME is not None:
    _pk = next(pathlib.Path("/kaggle/input").glob(f"*/**/{{RANKER_NAME}}"), None)
    assert _pk is not None, (
        f"{{RANKER_NAME}} not found under /kaggle/input -- attach "
        f"faustine649/biohub-edgeprob-rank2 as a kernel source. No other "
        f"ranker pkl is accepted by this submission.")
    with open(_pk, "rb") as _f:
        _blob = pickle.load(_f)
    assert isinstance(_blob, dict), f"{{_pk}} holds a bare {{type(_blob).__name__}}"
    _pool = _blob.get("pool")
    assert _pool is not None and len(_pool) >= MIN_POOL, (
        f"{{RANKER_NAME}} at {{_pk}} was trained on "
        f"{{len(_pool) if _pool else 'an unrecorded number of'}} datasets; this "
        f"submission requires >= {{MIN_POOL}} (the D175 pool).")
    EXTRA_COLS = [str(c) for c in _blob["extra_cols"]]
    assert EXTRA_COLS and all(c in HEAD_COL_NAMES for c in EXTRA_COLS), (
        f"{{RANKER_NAME}}: extra_cols {{EXTRA_COLS}} not all in {{HEAD_COL_NAMES}}")
    _cols = [int(c) for c in _blob["cols"]]
    assert len(_cols) == NFEAT_BASE + len(EXTRA_COLS), (
        f"{{RANKER_NAME}}: {{len(_cols)}} cols vs 18 + {{len(EXTRA_COLS)}} extra")
    assert _cols[:NFEAT_BASE] == list(range(NFEAT_BASE)), (
        f"{{RANKER_NAME}}: the first 18 cols are not 0..17: {{_cols[:NFEAT_BASE]}}")
    assert int(_blob["n_features"]) == len(_cols), _blob["n_features"]
    assert list(_blob["feature_names"]) == BASE_NAMES + EXTRA_COLS, (
        f"{{RANKER_NAME}}: feature_names {{_blob['feature_names']}}")
    CLF = _blob["model"]
    assert hasattr(CLF, "predict_proba"), (
        f"{{RANKER_NAME}} holds {{type(CLF).__name__}}, not a classifier")
    _nfi = getattr(CLF, "n_features_in_", None)
    assert _nfi is None or int(_nfi) == len(_cols), (
        f"{{RANKER_NAME}}: model.n_features_in_ = {{_nfi}}, expected {{len(_cols)}}")
    print(f"ranker {{RANKER_NAME}} from {{_pk}} | pool {{len(_pool)}} datasets | "
          f"version {{_blob.get('version')}} | {{len(_cols)}} features = 18 + "
          f"{{' '.join(EXTRA_COLS)}} | h_fused: {{_blob.get('h_fused')}}", flush=True)
else:
    assert HEAD_ONLY_COL in HEAD_COL_NAMES, HEAD_ONLY_COL
    CLF = None
    EXTRA_COLS = [HEAD_ONLY_COL]
    print(f"NO RANKER: the ILP is fed {{HEAD_ONLY_COL}} directly (edgeprob_rank2's "
          f"head-only arm)", flush=True)
NEED_H2 = any(c in ("p2_fwd", "p2_rev", "h2_fused", "p_avg_fwd", "p_avg_rev",
                    "h_avg") for c in EXTRA_COLS)
NEED_H1 = any(c in ("p_fwd", "p_rev", "h_fused", "p_avg_fwd", "p_avg_rev",
                    "h_avg") for c in EXTRA_COLS)
''', keep_end=False)

# ---- 2. the head block: two models -----------------------------------------
cut('# ---- the official edge head, run inline over OUR primaries',
    '# ---- the linker: ranker probs -> capacity-2 ILP',
    '''# ---- the official edge heads, run inline over OUR primaries ------------
# edgeprob_cache.py's injection (v2/v3 and biohub-edgeprob2: every
# round-trip assert held on 446 dataset passes). predict_video finds cells
# via the module-level _detect_cells_pooled and expects (N, 4) [t, z, y, x]
# "in downsampled space"; everything after it consumes that array.
# _load_frame reads frames STRIDED, so grid index g sits on full-res voxel
# g*d and full-res (z, y, x) -> grid (z/dz, y/dy, x/dx), floats injected.
# Reverse pass: remap the frame index f -> T-1-f in both patches.
SECONDARY = pathlib.Path("/kaggle/input/datasets/pilkwang/"
                         "biohub-temporal-unet3d-seed314159-v1")
if not SECONDARY.exists():
    SECONDARY = pathlib.Path("/kaggle/input/biohub-temporal-unet3d-seed314159-v1")
HEAD_WEIGHT_PATHS = {
    "h1": PACK / "weights/unet_transformer/split_0/edge_predictor_best.pth",
    "h2": SECONDARY / "weights/unet_transformer/split_0/edge_predictor_best.pth",
}
HEADS_USED = [h for h, need in (("h1", NEED_H1), ("h2", NEED_H2)) if need]
assert HEADS_USED, "no head column requested"
assert callable(getattr(PIPE, "_detect_cells_pooled", None)), (
    "predict_unet_transformer has no _detect_cells_pooled -- injection "
    "point gone")
assert callable(getattr(PIPE, "_load_frame", None)), (
    "predict_unet_transformer has no _load_frame -- reverse pass impossible")
HEAD_DEVICE = torch.device(DEVICE)
HEAD_NO_CANDIDATES = 2.0
HEAD_CFG = PIPE.PredictConfig(det_threshold=0.99, det_tta=False, use_ilp=True,
                              threshold=HEAD_NO_CANDIDATES)
assert HEAD_CFG.edge_activation == "softmax", HEAD_CFG.edge_activation
HEAD_MODELS = {}
HEAD_DS = None
HEAD_WINDOW = None
HEAD_DOWNSAMPLE = None
for _h in HEADS_USED:
    _wp = HEAD_WEIGHT_PATHS[_h]
    assert _wp.exists(), f"head {_h} weights missing: {_wp}"
    _m, _win, _dsm = PIPE.load_model(_wp, HEAD_DEVICE)
    assert _win == 2, (
        f"{_h}: window_size={_win}: the wrapper assumes one predict_edges "
        f"call per consecutive pair in ascending order, which holds for W=2 only")
    _ds = np.asarray(_dsm, dtype=np.float32)
    assert _ds.shape == (3,) and np.all(_ds >= 1), _dsm
    if HEAD_DS is None:
        HEAD_DS, HEAD_WINDOW, HEAD_DOWNSAMPLE = _ds, _win, tuple(_dsm)
    assert np.array_equal(_ds, HEAD_DS), (_h, _dsm, HEAD_DOWNSAMPLE)
    HEAD_MODELS[_h] = _m
    print(f"official head {_h} loaded from {_wp}: downsample={tuple(_dsm)} "
          f"window={_win}", flush=True)
print(f"heads used: {HEADS_USED}; columns built: {EXTRA_COLS}; "
      f"edge_activation={HEAD_CFG.edge_activation} threshold={HEAD_CFG.threshold} "
      f"det_tta={HEAD_CFG.det_tta}", flush=True)

_HEAD_ORIG_DETECT = PIPE._detect_cells_pooled
_HEAD_ORIG_LOAD = PIPE._load_frame
_HEAD_ORIG_EDGES = {h: m.predict_edges for h, m in HEAD_MODELS.items()}
HEAD_FAILED = []            # (dataset, head) scored with NaN head columns (rule 1)


class _HeadRun:
    """edgeprob_cache.Run for ONE head, ONE pass over ONE movie, capturing
    the FULL softmax matrix per consecutive pair.

    direction 'fwd': predict_video's frame index f is the movie's t.
    direction 'rev': predict_video's frame index f is the movie's T-1-f,
    so its pair (f, f+1) is the movie's (t+1 -> t) with t = T-2-f.
    """

    def __init__(self, name, prim, direction, head):
        self.name, self.prim, self.dir, self.head = name, prim, direction, head
        self.model = HEAD_MODELS[head]
        self.orig_edges = _HEAD_ORIG_EDGES[head]
        self.T = len(prim)
        self.n = np.array([len(p) for p in prim], np.int32)
        self.full = [prim[self.movie_t(f)].astype(np.float32)
                     for f in range(self.T)]
        self.expected = [self.movie_t(f) if direction == "fwd" else
                         self.movie_t(f) - 1
                         for f in range(self.T - 1)
                         if len(self.full[f]) and len(self.full[f + 1])]
        self.ptr = 0
        self.mat = {}       # movie t -> float32 (n_src, n_tgt), softmax dim 0

    def movie_t(self, f):
        return f if self.dir == "fwd" else self.T - 1 - f

    def detect(self, det_logits, f, det_threshold, pool_kernel):
        c = self.full[f]
        if len(c) == 0:
            return np.empty((0, 4), dtype=np.float32)
        grid = c / HEAD_DS[None, :]                  # full-res -> U-Net grid
        return np.concatenate(
            [np.full((len(c), 1), f, np.float32), grid.astype(np.float32)],
            axis=1)

    def load_frame(self, zarr_arr, f, target_shape, downsample_):
        return _HEAD_ORIG_LOAD(zarr_arr, self.movie_t(f), target_shape,
                               downsample_)

    def edges(self, feat_src, feat_tgt, coords_src, coords_tgt, *rest):
        out = self.orig_edges(feat_src, feat_tgt, coords_src, coords_tgt,
                              *rest)
        assert self.ptr < len(self.expected), (
            f"{self.name} {self.head} {self.dir}: predict_edges called more "
            f"often than there are non-empty consecutive pairs "
            f"({len(self.expected)})")
        t = self.expected[self.ptr]
        self.ptr += 1
        src_t, tgt_t = (t, t + 1) if self.dir == "fwd" else (t + 1, t)
        raw = out[0]
        assert tuple(raw.shape) == (self.n[src_t], self.n[tgt_t]), (
            f"{self.name} {self.head} {self.dir} t={t}: logits "
            f"{tuple(raw.shape)} vs injected ({self.n[src_t]}, {self.n[tgt_t]})")
        for got, exp_t in ((coords_src[0], src_t), (coords_tgt[0], tgt_t)):
            exp = torch.from_numpy(self.prim[exp_t].astype(np.float32)
                                   ).to(got.device)
            err = float((got - exp).abs().max()) if len(exp) else 0.0
            assert err <= 0.5, (
                f"{self.name} {self.head} {self.dir} t={exp_t}: coords handed "
                f"to predict_edges differ from injected by {err:.3f} voxel")
        self.mat[t] = torch.softmax(raw, dim=0).float().cpu().numpy()
        return out

    def install(self):
        PIPE._detect_cells_pooled = self.detect
        PIPE._load_frame = self.load_frame
        self.model.predict_edges = self.edges

    @staticmethod
    def uninstall():
        PIPE._detect_cells_pooled = _HEAD_ORIG_DETECT
        PIPE._load_frame = _HEAD_ORIG_LOAD
        for h, m in HEAD_MODELS.items():
            m.predict_edges = _HEAD_ORIG_EDGES[h]

    def check_return(self, coords, edges):
        assert len(edges) == 0, (
            f"{self.name} {self.head} {self.dir}: predict_video kept "
            f"{len(edges)} edges at threshold {HEAD_NO_CANDIDATES}")
        assert self.ptr == len(self.expected), (
            f"{self.name} {self.head} {self.dir}: {self.ptr} predict_edges "
            f"calls, expected {len(self.expected)}")
        coords = np.asarray(coords)
        assert len(coords) == int(self.n.sum()), (
            f"{self.name} {self.head} {self.dir}: predict_video reports "
            f"{len(coords)} nodes, injected {int(self.n.sum())}")
        off = 0
        for f in range(self.T):
            k = len(self.full[f])
            blk = coords[off:off + k]
            off += k
            if k == 0:
                continue
            assert np.all(blk[:, 0] == f), \\
                f"{self.name} {self.head} {self.dir}: frame column mismatch at f={f}"
            grid = (self.full[f] / HEAD_DS[None, :]).astype(np.float32)
            exp = np.trunc(grid * HEAD_DS[None, :])
            err = float(np.abs(blk[:, 1:].astype(np.float32) - exp).max())
            assert err <= 0.5, (
                f"{self.name} {self.head} {self.dir} f={f}: returned coords "
                f"differ from injected by {err:.2f} voxel")


def _one_head_pass(zp, ds, prim, T, head):
    """(P_f, P_r) for one head, or None on failure (printed, counted)."""
    try:
        runs = {}
        for direction in ("fwd", "rev"):
            run = _HeadRun(ds, prim, direction, head)
            run.install()
            try:
                coords, edges = PIPE.predict_video(
                    HEAD_MODELS[head], zp.parent / ds, HEAD_DEVICE, cfg=HEAD_CFG,
                    window_size=HEAD_WINDOW, max_frames=T,
                    unet_batch_size=4, downsample=HEAD_DOWNSAMPLE)
            finally:
                _HeadRun.uninstall()
            run.check_return(coords, edges)
            runs[direction] = run
        P_f, P_r = [], []
        for t in range(T - 1):
            n_s, n_t = len(prim[t]), len(prim[t + 1])
            if n_s == 0 or n_t == 0:
                P_f.append(None)
                P_r.append(None)
                continue
            Mf = runs["fwd"].mat[t]
            Mr = runs["rev"].mat[t]
            assert Mf.shape == (n_s, n_t) and Mr.shape == (n_t, n_s), (
                f"{ds} {head} t={t}: matrices {Mf.shape} / {Mr.shape} vs "
                f"primaries ({n_s}, {n_t})")
            assert np.isfinite(Mf).all() and np.isfinite(Mr).all(), (
                f"{ds} {head} t={t}: non-finite softmax probability")
            assert Mf.min() >= 0 and Mf.max() <= 1 and \\
                Mr.min() >= 0 and Mr.max() <= 1, f"{ds} {head} t={t}: prob outside [0,1]"
            P_f.append(Mf)
            P_r.append(Mr)
        return P_f, P_r
    except Exception as exc:                                  # noqa: BLE001
        _HeadRun.uninstall()
        HEAD_FAILED.append((ds, head))
        print(f"[degraded] {ds}: head {head} pass failed ({type(exc).__name__}: "
              f"{str(exc)[:300]}) -- this movie's {head} columns are NaN",
              flush=True)
        return None
    finally:
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


def head_pass(zp, ds, dets, T):
    """{head: (P_f, P_r) | None} for every head in HEADS_USED. Never raises."""
    prim = [seeds_of(dets[t])[0] for t in range(T)]     # link_ilp's order
    return {h: _one_head_pass(zp, ds, prim, T, h) for h in HEADS_USED}


def _three(hp, t, ii, jj):
    """[p_fwd, p_rev, h_fused] of one head at pairs (ii, jj); NaN x3 if None."""
    if hp is None:
        return np.full((len(ii), 3), np.nan, np.float32)
    P_f, P_r = hp
    p_fwd = P_f[t][ii, jj].astype(np.float32)
    p_rev = P_r[t][jj, ii].astype(np.float32)      # rows are t+1 nodes there
    ok = np.isfinite(p_rev)
    pr = np.where(ok, p_rev, 0.0).astype(np.float32)
    h_fused = np.where(ok, 2.0 * p_fwd * pr / np.maximum(p_fwd + pr, 1e-12),
                       p_fwd).astype(np.float32)
    return np.stack([p_fwd, p_rev, h_fused], axis=1).astype(np.float32)


def _head_columns(head, t, pairs):
    """The EXTRA_COLS, in order, float32 (len(pairs), len(EXTRA_COLS)).
    edgeprob_rank2.py's definitions: h_fused per head = harmonic mean of
    p_fwd and p_rev; p_avg_* = mean of the two heads' (reverse: the finite
    one when only one is); h_avg = mean of the two h_fused."""
    ii = np.fromiter((i for i, _ in pairs), np.int64, len(pairs))
    jj = np.fromiter((j for _, j in pairs), np.int64, len(pairs))
    c1 = _three(head.get("h1"), t, ii, jj)
    c2 = _three(head.get("h2"), t, ii, jj)
    r1, r2 = c1[:, 1], c2[:, 1]
    p_avg_rev = np.where(np.isfinite(r1) & np.isfinite(r2), 0.5 * (r1 + r2),
                         np.where(np.isfinite(r1), r1, r2)).astype(np.float32)
    allc = {"p_fwd": c1[:, 0], "p_rev": c1[:, 1], "h_fused": c1[:, 2],
            "p2_fwd": c2[:, 0], "p2_rev": c2[:, 1], "h2_fused": c2[:, 2],
            "p_avg_fwd": (0.5 * (c1[:, 0] + c2[:, 0])).astype(np.float32),
            "p_avg_rev": p_avg_rev,
            "h_avg": (0.5 * (c1[:, 2] + c2[:, 2])).astype(np.float32)}
    return np.stack([allc[c] for c in EXTRA_COLS], axis=1).astype(np.float32)


''')

# ---- 3. link_ilp: ranker or head-only ---------------------------------------
rep('''        assert rows.shape == (len(pairs), NFEAT_BASE), rows.shape
        extra = _head_columns(head, t, pairs)
        pr = CLF.predict_proba(np.concatenate([rows, extra], axis=1))[:, 1]
''',
    '''        assert rows.shape == (len(pairs), NFEAT_BASE), rows.shape
        extra = _head_columns(head, t, pairs)
        if CLF is None:                       # head-only: the column IS p
            pr = np.nan_to_num(extra[:, 0], nan=0.0).astype(float)
        else:
            pr = CLF.predict_proba(np.concatenate([rows, extra], axis=1))[:, 1]
''')

rep('''    print(f"    head: {'ok' if head is not None else 'FAILED -> NaN columns'}"
          f" ({time.time() - t_h:.0f}s)", flush=True)
''',
    '''    print("    heads: " + " ".join(f"{h}={'ok' if v is not None else 'FAILED'}"
                                     for h, v in head.items())
          + f" ({time.time() - t_h:.0f}s)", flush=True)
''')

rep('''print(f"HEAD PASS: {len(HEAD_FAILED)} of {len(KEEP)} movies scored with NaN "
      f"head columns" + (f": {HEAD_FAILED}" if HEAD_FAILED else " (none)"),
      flush=True)
''',
    '''print(f"HEAD PASS: {len(HEAD_FAILED)} (movie, head) passes failed of "
      f"{len(KEEP) * len(HEADS_USED)}" + (f": {HEAD_FAILED}" if HEAD_FAILED
                                          else " (none)"), flush=True)
''')
rep('''    print(f"[degraded] {len(HEAD_FAILED)} movie(s) ran WITHOUT the head's "
          f"columns: {HEAD_FAILED}", flush=True)
''',
    '''    print(f"[degraded] {len(HEAD_FAILED)} (movie, head) pass(es) ran with NaN "
          f"columns: {HEAD_FAILED}", flush=True)
''')

assert "HEAD_MODEL," not in text and "HEAD_MODEL.predict" not in text, "old single-head refs"
DST.write_text(text, encoding="utf-8")
ast.parse(text)
print(f"wrote {DST} ({len(text.splitlines())} lines), ast OK; "
      f"RANKER_NAME={RANKER_NAME} HEAD_ONLY_COL={HEAD_ONLY_COL}")
