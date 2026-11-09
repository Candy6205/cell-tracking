"""Generate submit_ep.py from submit_f4.py by asserted replacement.

Single variable vs ref 55889500 (LB 0.925, submit_f4.py): the edge ranker
gets three columns from the OFFICIAL edge head (p_fwd, p_rev, h_fused),
run inline per movie over our own primaries exactly as edgeprob_cache.py
does, and the ranker pkl is edge_ranker_ep175_bidir.pkl (retrained on the
D175 pool with those columns). Everything else -- detection (F4, four
heatmaps), refine, NMS, ILP constants, gap closing, SHORT_MIN 4,
_add_safe_divisions, output writing, the local score block -- byte-identical.
"""
import ast
import pathlib

SRC = pathlib.Path(__file__).parent / "submit_f4.py"
DST = pathlib.Path(__file__).parent / "submit_ep.py"
text = SRC.read_text(encoding="utf-8")


def rep(old, new, n=1):
    global text
    c = text.count(old)
    assert c == n, f"anchor x{c}, wanted x{n}: {old[:70]!r}"
    text = text.replace(old, new)


# ---- 1. docstring ------------------------------------------------------
rep('"""submit_f4.py -- ref 55817630 (LB 0.922) with ep05 DeepCenter as '
    'a 4th heatmap.\n',
    '"""submit_ep.py -- ref 55889500 (LB 0.925, submit_f4) + the official '
    "edge head's three columns in the ranker.\n"
    '\n'
    'SINGLE VARIABLE vs 55889500: the edge ranker. link_ilp\'s classifier '
    'is\nedge_ranker_ep175_bidir.pkl (kernel biohub-edgeprob-rank, D175 '
    'pool),\ntrained on the SAME 18 pair_features columns PLUS three from '
    'the official\nunet_transformer edge head run inline over OUR '
    'primaries:\n\n    p_fwd    softmax over sources, t -> t+1 (the '
    'official activation)\n    p_rev    the same head over the reversed '
    'movie, t+1 -> t\n    h_fused  2ab/(a+b), harmonic mean of the two '
    '(p_fwd where p_rev is NaN)\n\nEverything else -- detection (F4, four '
    'heatmaps), refine, NMS, ILP\nconstants, gap closing, SHORT_MIN 4, '
    '_add_safe_divisions, output writing,\nthe visible-test local score '
    'block -- is byte-identical to submit_f4.py.\n\nWHAT THIS TESTS: '
    'edgeprob_rank v1 (D32 pool, BASE primaries) measured the\n+BIDIR arm '
    'at +0.0046 / +0.0106 adj_J on the 6bba / 44b6 holdouts. The\nF4-arm, '
    'D175-pool deltas of the very pkl shipped here (biohub-edgeprob-rank\n'
    'v2, Stage B): [TO BE FILLED IN BY THE USER before submitting].\n'
    'PRE-REGISTERED READING: LB > 0.925 = the head carries information '
    'the 18\ncoordinate columns lack, and it transfers to a third embryo; '
    'LB <= 0.925 =\nthe local gain did not transfer (the record\'s 4-in-5 '
    'pattern of local\ngains that read 0.000 or below at the official '
    'judge).\n\nHOW THE HEAD RUNS (edgeprob_cache.py\'s injection, the '
    'v2/v3 code path that\nheld every round-trip assert on 223 dataset '
    'passes): per movie,\nPIPE._detect_cells_pooled is replaced by a '
    'function returning\nseeds_of(dets[t]) -- the same primaries, same '
    'order, that pass1_velocities\nand pair_features use -- converted '
    'full-res voxel -> head grid by\n/ downsample as float (no rounding, '
    'the training convention);\nmodel.predict_edges is wrapped to capture '
    'the raw (n_t, n_t+1) logit matrix\nof every consecutive pair; '
    'cfg.threshold = 2.0 so predict_video\'s own\ncandidate loop keeps '
    'nothing; det_tta off (TTA touches only det_logits).\nNO PRE-GATING: '
    'the FULL softmax matrices P_f[t] (n_t x n_t+1, softmax over\ndim 0 = '
    'sources) and P_r[t] (n_t+1 x n_t, same activation on the reversed\n'
    'movie) are kept on the CPU as float32 and link_ilp looks up\np_fwd = '
    'P_f[t][i, j], p_rev = P_r[t][j, i] for every pair pair_features\n'
    'yields, so a gate mismatch is impossible. The round-trip asserts are '
    'kept\n(coords handed to predict_edges == injected within 0.5 voxel; '
    'logit shape\n== injected counts; one predict_edges call per non-empty '
    'consecutive pair;\npredict_video\'s returned coords == injected), but '
    'on the hidden test a\nfailure must not cost the submission: it is '
    'caught PER MOVIE, printed as\n[degraded], and that movie is scored '
    'with p_fwd = p_rev = h_fused = NaN\n(HistGradientBoosting accepts '
    'NaN; this pkl has no 18-column path). The\nnumber of such movies is '
    'printed after submission.csv is written and again\nat the end; 0 is '
    'the expected value.\n\nRUNTIME: the head adds ~40-50 s per movie on a '
    'T4 (biohub-edgeprob v3\nmeasured 36-67 s for both directions), i.e. '
    '+2.2-2.8 h on the ~6 h\npipeline -- still under the 12 h cap. '
    'submit_f4.py has no wall-clock guard\nand none is added here (nothing '
    'else changes). The full matrices are\ntransient per movie: at 700 '
    'cells/frame and 100 frames they are ~400 MB.\n\n---- submit_f4.py\'s '
    'docstring follows, kept as provenance ----\n')

rep('SCRIPT_VERSION = "2026-08-30a-submit-f4"',
    'SCRIPT_VERSION = "2026-09-03a-submit-ep"')

# ---- 2. ranker loading: REQUIRE the +BIDIR pkl, assert its layout -------
rep('''# Prefer the D187 pkl and REFUSE to run on the old one. The filename
# differs, so a plain glob for edge_ranker.pkl would silently pick up
# ranker_fine's 32-dataset model and produce a submission that looks
# entirely normal -- the exact failure mode this session hit four times.
MIN_POOL = 100
_pk = None
for _pat in ("*/**/edge_ranker_d187.pkl", "*/**/edge_ranker.pkl"):
    for _d in pathlib.Path("/kaggle/input").glob(_pat):
        _pk = _d
        break
    if _pk is not None:
        break
assert _pk is not None, "attach a ranker kernel (edge_ranker*.pkl) as source"
with open(_pk, "rb") as _f:
    _blob = pickle.load(_f)
_pool = _blob.get("pool") if isinstance(_blob, dict) else None
assert _pool is not None and len(_pool) >= MIN_POOL, (
    f"ranker pkl at {_pk} was trained on "
    f"{len(_pool) if _pool else 'an unrecorded number of'} datasets; this "
    f"submission requires >= {MIN_POOL}. Attach faustine649/biohub-poolsize "
    f"and detach biohub-ranker-fine.")
print(f"ranker pool: {len(_pool)} datasets", flush=True)
# the ranker kernels dump {"model": clf, "version": ...}, not a bare model
CLF = _blob["model"] if isinstance(_blob, dict) else _blob
assert hasattr(CLF, "predict_proba"),     f"ranker pkl holds {type(CLF).__name__}, not a classifier"
print("ranker loaded from", _pk, "| trained by",
      _blob.get("version") if isinstance(_blob, dict) else "(bare)",
      flush=True)
''',
    '''# 2026-09-03 (submit_ep): the ranker is edge_ranker_ep175_bidir.pkl from
# biohub-edgeprob-rank -- trained on the 18 pair_features columns PLUS
# p_fwd, p_rev, h_fused from the official edge head. REQUIRE it by name
# and refuse any other pkl: the D187 pkl expects 18 columns and a plain
# edge_ranker*.pkl glob would silently feed it 21 (or feed this one 18).
# Layout, from edgeprob_rank.py's save block / "RANKER LAYOUT" print:
#   pkl["cols"]          21 indices into THAT kernel's 28-column joined
#                        layout: 0..17 (the 18 base) then 23, 24, 27
#                        (p_fwd, p_rev, h_fused) -- NOT 0..20
#   pkl["extra_cols"]    ["p_fwd", "p_rev", "h_fused"], appended after the 18
#   pkl["feature_names"] BASE_NAMES + extra_cols  (21)
#   pkl["n_features"]    21
#   pkl["pool"]          the 175 D175 names
# so here X = concat([the 18 pair_features columns, p_fwd, p_rev, h_fused])
# is fed directly; pkl["cols"] is asserted, not used for indexing.
RANKER_NAME = "edge_ranker_ep175_bidir.pkl"
MIN_POOL = 170
NFEAT_BASE = 18
EXTRA_COLS = ["p_fwd", "p_rev", "h_fused"]
BASE_NAMES = ["d_um", "dz_um", "abs_dz_um", "dxy_um", "md_um", "d_minus_md",
              "score_i", "score_j", "abs_logit_diff", "dens_i", "dens_j",
              "n_prev", "n_curr", "rank_of_j_for_i", "n_cand_i", "n_cand_j",
              "speed_i", "t_frac"]           # pair_features row order
assert len(BASE_NAMES) == NFEAT_BASE
_pk = next(pathlib.Path("/kaggle/input").glob(f"*/**/{RANKER_NAME}"), None)
assert _pk is not None, (
    f"{RANKER_NAME} not found under /kaggle/input -- attach "
    f"faustine649/biohub-edgeprob-rank as a kernel source. No other ranker "
    f"pkl is accepted by this submission.")
with open(_pk, "rb") as _f:
    _blob = pickle.load(_f)
assert isinstance(_blob, dict), f"{_pk} holds a bare {type(_blob).__name__}"
_pool = _blob.get("pool")
assert _pool is not None and len(_pool) >= MIN_POOL, (
    f"{RANKER_NAME} at {_pk} was trained on "
    f"{len(_pool) if _pool else 'an unrecorded number of'} datasets; this "
    f"submission requires >= {MIN_POOL} (the D175 pool).")
_cols = [int(c) for c in _blob["cols"]]
assert len(_cols) == NFEAT_BASE + len(EXTRA_COLS) == 21, (
    f"{RANKER_NAME}: {len(_cols)} columns, expected 21 (18 + p_fwd p_rev "
    f"h_fused): cols={_cols}")
assert _cols[:NFEAT_BASE] == list(range(NFEAT_BASE)), (
    f"{RANKER_NAME}: the first 18 cols are not 0..17: {_cols[:NFEAT_BASE]}")
assert list(_blob["extra_cols"]) == EXTRA_COLS, (
    f"{RANKER_NAME}: extra_cols {_blob['extra_cols']} != {EXTRA_COLS}")
assert int(_blob["n_features"]) == 21, _blob["n_features"]
assert list(_blob["feature_names"]) == BASE_NAMES + EXTRA_COLS, (
    f"{RANKER_NAME}: feature_names {_blob['feature_names']}")
CLF = _blob["model"]
assert hasattr(CLF, "predict_proba"), (
    f"{RANKER_NAME} holds {type(CLF).__name__}, not a classifier")
_nfi = getattr(CLF, "n_features_in_", None)
assert _nfi is None or int(_nfi) == 21, (
    f"{RANKER_NAME}: model.n_features_in_ = {_nfi}, expected 21")
print(f"ranker {RANKER_NAME} from {_pk} | pool {len(_pool)} datasets | "
      f"version {_blob.get('version')} | {len(_cols)} features = 18 + "
      f"{' '.join(EXTRA_COLS)} | h_fused: {_blob.get('h_fused')}", flush=True)
''')

# ---- 3. the head pass, inserted before link_ilp; link_ilp gains `head` --
rep('''# ---- the linker: ranker probs -> capacity-2 ILP ------------------------
def link_ilp(dets, T, div_w):
    """B line of biohub-ranker-diag: same behaviour, same constants."""
    vels = pass1_velocities(dets)
    feats = pair_features(dets, vels, T)
''',
    '''# ---- the official edge head, run inline over OUR primaries -------------
# edgeprob_cache.py's injection (v2/v3: every round-trip assert held on
# 223 dataset passes). Read from the pack's predict_unet_transformer.py:
# predict_video finds cells via the module-level _detect_cells_pooled and
# expects (N, 4) [t, z, y, x] "in downsampled space"; everything after it
# (_index_features, extract_pos_features, predict_edges with
# p_coords * ds) consumes that array. _load_frame reads frames STRIDED
# (zarr_arr[t, ::dz, ::dy, ::dx]), so grid index g sits on full-res voxel
# g*d and full-res (z, y, x) -> grid (z/dz, y/dy, x/dx), floats injected.
# Reverse pass: remap the frame index f -> T-1-f in both patches, and the
# identical model scores t+1 -> t with its softmax over the t+1 side.
HEAD_WEIGHTS = PACK / "weights/unet_transformer/split_0/edge_predictor_best.pth"
assert HEAD_WEIGHTS.exists(), f"official head weights missing: {HEAD_WEIGHTS}"
assert callable(getattr(PIPE, "_detect_cells_pooled", None)), (
    "predict_unet_transformer has no _detect_cells_pooled -- injection "
    "point gone")
assert callable(getattr(PIPE, "_load_frame", None)), (
    "predict_unet_transformer has no _load_frame -- reverse pass impossible")
HEAD_DEVICE = torch.device(DEVICE)
HEAD_MODEL, HEAD_WINDOW, HEAD_DOWNSAMPLE = PIPE.load_model(HEAD_WEIGHTS,
                                                          HEAD_DEVICE)
assert HEAD_WINDOW == 2, (
    f"window_size={HEAD_WINDOW}: the wrapper below assumes one predict_edges "
    f"call per consecutive pair in ascending order, which holds for W=2 only")
HEAD_DS = np.asarray(HEAD_DOWNSAMPLE, dtype=np.float32)      # (dz, dy, dx)
assert HEAD_DS.shape == (3,) and np.all(HEAD_DS >= 1), HEAD_DOWNSAMPLE
# predict_video's own candidate gate, set ABOVE 1 so its Python loop keeps
# nothing; the probabilities come from the captured matrices instead.
HEAD_NO_CANDIDATES = 2.0
HEAD_CFG = PIPE.PredictConfig(det_threshold=0.99, det_tta=False, use_ilp=True,
                              threshold=HEAD_NO_CANDIDATES)
assert HEAD_CFG.edge_activation == "softmax", HEAD_CFG.edge_activation
print(f"official head loaded: downsample={tuple(HEAD_DOWNSAMPLE)} "
      f"window={HEAD_WINDOW} edge_activation={HEAD_CFG.edge_activation} "
      f"threshold={HEAD_CFG.threshold} det_tta={HEAD_CFG.det_tta}",
      flush=True)

_HEAD_ORIG_DETECT = PIPE._detect_cells_pooled
_HEAD_ORIG_LOAD = PIPE._load_frame
_HEAD_ORIG_EDGES = HEAD_MODEL.predict_edges
HEAD_FAILED = []            # datasets scored with NaN head columns (rule 1)


class _HeadRun:
    """edgeprob_cache.Run: state shared between the three patched
    functions for ONE pass over ONE movie, capturing the FULL softmax
    matrix per consecutive pair instead of gated entries.

    direction 'fwd': predict_video's frame index f is the movie's t.
    direction 'rev': predict_video's frame index f is the movie's T-1-f,
    so its pair (f, f+1) is the movie's (t+1 -> t) with t = T-2-f.
    """

    def __init__(self, name, prim, direction):
        self.name, self.prim, self.dir = name, prim, direction
        self.T = len(prim)
        self.n = np.array([len(p) for p in prim], np.int32)
        # frames in predict_video order -> full-res coords
        self.full = [prim[self.movie_t(f)].astype(np.float32)
                     for f in range(self.T)]
        # the movie t of each predict_edges call, in call order; a pair is
        # skipped by predict_video when either frame is empty
        self.expected = [self.movie_t(f) if direction == "fwd" else
                         self.movie_t(f) - 1
                         for f in range(self.T - 1)
                         if len(self.full[f]) and len(self.full[f + 1])]
        self.ptr = 0
        self.mat = {}       # movie t -> float32 (n_src, n_tgt), softmax dim 0

    def movie_t(self, f):
        return f if self.dir == "fwd" else self.T - 1 - f

    # -- PIPE._detect_cells_pooled replacement -----------------------------
    def detect(self, det_logits, f, det_threshold, pool_kernel):
        c = self.full[f]
        if len(c) == 0:
            return np.empty((0, 4), dtype=np.float32)
        grid = c / HEAD_DS[None, :]                  # full-res -> U-Net grid
        return np.concatenate(
            [np.full((len(c), 1), f, np.float32), grid.astype(np.float32)],
            axis=1)

    # -- PIPE._load_frame replacement (identity in the forward pass) -------
    def load_frame(self, zarr_arr, f, target_shape, downsample_):
        return _HEAD_ORIG_LOAD(zarr_arr, self.movie_t(f), target_shape,
                               downsample_)

    # -- model.predict_edges wrapper ---------------------------------------
    def edges(self, feat_src, feat_tgt, coords_src, coords_tgt, *rest):
        out = _HEAD_ORIG_EDGES(feat_src, feat_tgt, coords_src, coords_tgt,
                               *rest)
        assert self.ptr < len(self.expected), (
            f"{self.name} {self.dir}: predict_edges called more often than "
            f"there are non-empty consecutive pairs ({len(self.expected)})")
        t = self.expected[self.ptr]
        self.ptr += 1
        src_t, tgt_t = (t, t + 1) if self.dir == "fwd" else (t + 1, t)
        raw = out[0]
        assert tuple(raw.shape) == (self.n[src_t], self.n[tgt_t]), (
            f"{self.name} {self.dir} t={t}: logits {tuple(raw.shape)} vs "
            f"injected ({self.n[src_t]}, {self.n[tgt_t]})")
        # Round trip at the consumption point: the head was handed
        # grid * downsample, which must be the full-res coords injected.
        for got, exp_t in ((coords_src[0], src_t), (coords_tgt[0], tgt_t)):
            exp = torch.from_numpy(self.prim[exp_t].astype(np.float32)
                                   ).to(got.device)
            err = float((got - exp).abs().max()) if len(exp) else 0.0
            assert err <= 0.5, (
                f"{self.name} {self.dir} t={exp_t}: coords handed to "
                f"predict_edges differ from injected by {err:.3f} voxel")
        # official activation: softmax over SOURCES (dim 0); the whole
        # matrix, so the CPU side can look up ANY (i, j) pair_features yields
        self.mat[t] = torch.softmax(raw, dim=0).float().cpu().numpy()
        return out

    def install(self):
        PIPE._detect_cells_pooled = self.detect
        PIPE._load_frame = self.load_frame
        HEAD_MODEL.predict_edges = self.edges

    @staticmethod
    def uninstall():
        PIPE._detect_cells_pooled = _HEAD_ORIG_DETECT
        PIPE._load_frame = _HEAD_ORIG_LOAD
        HEAD_MODEL.predict_edges = _HEAD_ORIG_EDGES

    def check_return(self, coords, edges):
        """predict_video's own return must round-trip our injection."""
        assert len(edges) == 0, (
            f"{self.name} {self.dir}: predict_video kept {len(edges)} edges "
            f"at threshold {HEAD_NO_CANDIDATES}; softmax probs above 1 are "
            f"impossible")
        assert self.ptr == len(self.expected), (
            f"{self.name} {self.dir}: {self.ptr} predict_edges calls, "
            f"expected {len(self.expected)}")
        coords = np.asarray(coords)
        assert len(coords) == int(self.n.sum()), (
            f"{self.name} {self.dir}: predict_video reports {len(coords)} "
            f"nodes, injected {int(self.n.sum())}")
        # predict_video returns int16(trunc(grid * ds)); compare per frame in
        # ITS frame order against trunc of what we injected.
        off = 0
        for f in range(self.T):
            k = len(self.full[f])
            blk = coords[off:off + k]
            off += k
            if k == 0:
                continue
            assert np.all(blk[:, 0] == f), \\
                f"{self.name} {self.dir}: frame column mismatch at f={f}"
            grid = (self.full[f] / HEAD_DS[None, :]).astype(np.float32)
            exp = np.trunc(grid * HEAD_DS[None, :])
            err = float(np.abs(blk[:, 1:].astype(np.float32) - exp).max())
            assert err <= 0.5, (
                f"{self.name} {self.dir} f={f}: returned coords differ from "
                f"injected by {err:.2f} voxel (truncation accounted for)")


def head_pass(zp, ds, dets, T):
    """The official head over seeds_of(dets[t]), both directions.

    Returns (P_f, P_r), two lists of length T-1: P_f[t] is the float32
    (n_t, n_t+1) forward softmax matrix (over dim 0 = sources) and P_r[t]
    the float32 (n_t+1, n_t) reverse one -- None at a t where either frame
    has no primaries (predict_video skips that pair; pair_features yields
    no pair there either). Returns None when the pass fails for this
    movie: printed as [degraded], counted in HEAD_FAILED, and link_ilp
    then scores the movie with p_fwd = p_rev = h_fused = NaN. Never
    raises -- a failed assert here must not cost the submission.
    """
    prim = [seeds_of(dets[t])[0] for t in range(T)]     # link_ilp's order
    try:
        runs = {}
        for direction in ("fwd", "rev"):
            run = _HeadRun(ds, prim, direction)
            run.install()
            try:
                coords, edges = PIPE.predict_video(
                    HEAD_MODEL, zp.parent / ds, HEAD_DEVICE, cfg=HEAD_CFG,
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
                f"{ds} t={t}: matrices {Mf.shape} / {Mr.shape} vs primaries "
                f"({n_s}, {n_t})")
            assert np.isfinite(Mf).all() and np.isfinite(Mr).all(), (
                f"{ds} t={t}: non-finite softmax probability")
            assert Mf.min() >= 0 and Mf.max() <= 1 and \\
                Mr.min() >= 0 and Mr.max() <= 1, f"{ds} t={t}: prob outside [0,1]"
            P_f.append(Mf)
            P_r.append(Mr)
        return P_f, P_r
    except Exception as exc:                                  # noqa: BLE001
        _HeadRun.uninstall()
        HEAD_FAILED.append(ds)
        print(f"[degraded] {ds}: head pass failed ({type(exc).__name__}: "
              f"{str(exc)[:300]}) -- falling back to the 18-column path is "
              f"NOT available with this pkl, so this movie is scored with "
              f"p_fwd=p_rev=h_fused=NaN", flush=True)
        return None
    finally:
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


def _head_columns(head, t, pairs):
    """[p_fwd, p_rev, h_fused] per pair, float32 (len(pairs), 3).

    The pkl's h_fused: 2*p_fwd*p_rev/(p_fwd+p_rev), p_fwd where p_rev is
    NaN. With the full matrices every lookup is defined; only the
    all-NaN fallback (head is None) leaves the columns missing.
    """
    if head is None:
        return np.full((len(pairs), 3), np.nan, np.float32)
    P_f, P_r = head
    ii = np.fromiter((i for i, _ in pairs), np.int64, len(pairs))
    jj = np.fromiter((j for _, j in pairs), np.int64, len(pairs))
    p_fwd = P_f[t][ii, jj].astype(np.float32)
    p_rev = P_r[t][jj, ii].astype(np.float32)      # rows are t+1 nodes there
    ok = np.isfinite(p_rev)
    pr = np.where(ok, p_rev, 0.0).astype(np.float32)
    h_fused = np.where(ok, 2.0 * p_fwd * pr / np.maximum(p_fwd + pr, 1e-12),
                       p_fwd).astype(np.float32)
    return np.stack([p_fwd, p_rev, h_fused], axis=1).astype(np.float32)


# ---- the linker: ranker probs -> capacity-2 ILP ------------------------
def link_ilp(dets, T, div_w, head):
    """B line of biohub-ranker-diag: same behaviour, same constants.
    `head` = head_pass()'s (P_f, P_r), or None -> NaN head columns."""
    vels = pass1_velocities(dets)
    feats = pair_features(dets, vels, T)
''')

rep('''        pairs, rows, Pp, Cc = feats[t]
        if not pairs:
            continue
        pr = CLF.predict_proba(rows)[:, 1]
''',
    '''        pairs, rows, Pp, Cc = feats[t]
        if not pairs:
            continue
        assert rows.shape == (len(pairs), NFEAT_BASE), rows.shape
        extra = _head_columns(head, t, pairs)
        pr = CLF.predict_proba(np.concatenate([rows, extra], axis=1))[:, 1]
''')

# ---- 4. track_movie: head pass between detection and the linker --------
rep('''    nodes, edges, cand, cand_sc, cand_trees = link_ilp(dets, T, ILP_DIV_W)
''',
    '''    if DEVICE == "cuda":
        torch.cuda.empty_cache()          # the xl/DC activations are done
    t_h = time.time()
    head = head_pass(zp, ds, dets, T)
    print(f"    head: {'ok' if head is not None else 'FAILED -> NaN columns'}"
          f" ({time.time() - t_h:.0f}s)", flush=True)
    nodes, edges, cand, cand_sc, cand_trees = link_ilp(dets, T, ILP_DIV_W,
                                                       head)
    del head
''')

# ---- 5. the degradation count, never silent (rule 1) --------------------
rep('''      (sub.row_type == "edge").sum(), flush=True)

# ---- local score, AFTER the CSV is safely on disk -----------------------
''',
    '''      (sub.row_type == "edge").sum(), flush=True)
print(f"HEAD PASS: {len(HEAD_FAILED)} of {len(KEEP)} movies scored with NaN "
      f"head columns" + (f": {HEAD_FAILED}" if HEAD_FAILED else " (none)"),
      flush=True)

# ---- local score, AFTER the CSV is safely on disk -----------------------
''')

rep('''print("SUBMIT RANKER DONE", flush=True)
''',
    '''if HEAD_FAILED:
    print(f"[degraded] {len(HEAD_FAILED)} movie(s) ran WITHOUT the head's "
          f"columns: {HEAD_FAILED}", flush=True)
print("SUBMIT RANKER DONE", flush=True)
''')

DST.write_text(text, encoding="utf-8")
ast.parse(text)
print(f"wrote {DST} ({len(text.splitlines())} lines), ast OK")
