"""make_submit_off.py -- generate submit_off.py from submit_ep2.py (head-only
build: RANKER_NAME None, HEAD_ONLY_COL "h_fused"): the OFFICIAL detector's
nodes instead of the xl / DC blend, linked by the official head's
bidirectional probability straight into our ILP, then our patches.

This is offdet_rank.py's HEAD1-on-OFF99 arm as a submission (local
0.9297 / 0.9089 vs the shipped 0.9261 / 0.8515; in-sample for the
detector, so LB decides -- progress.md 2026-09-04, offdet).

Per movie: forward predict_video with the official detection at OFF_GATE
(flip TTA), `_detect_cells_pooled` WRAPPED to keep every frame's peaks and
sigmoid scores, `predict_edges` wrapped to keep the forward softmax
matrices; then the reverse pass with those nodes injected (_HeadRun, rev);
then link_ilp with CLF None (h_fused is p). If the official path fails for
a movie, the xl / DC detection + head pass of submit_ep2 runs instead
(printed as [degraded], counted).

    py make_submit_ep2.py   (RANKER_NAME None, HEAD_ONLY_COL "h_fused")
    py make_submit_off.py   -> submit_off.py
"""
import ast
import pathlib

SRC = pathlib.Path(__file__).with_name("submit_ep2.py")
DST = pathlib.Path(__file__).with_name("submit_off.py")
text = SRC.read_text(encoding="utf-8")


def rep(old, new, n=1):
    global text
    c = text.count(old)
    assert c == n, (c, old[:80])
    text = text.replace(old, new)


assert "RANKER_NAME = None" in text and "HEAD_ONLY_COL = 'h_fused'" in text, \
    "regenerate submit_ep2.py with RANKER_NAME None / HEAD_ONLY_COL h_fused first"

rep('SCRIPT_VERSION = "2026-09-03b-submit-ep2"',
    'SCRIPT_VERSION = "2026-09-04a-submit-off"')

rep('''def track_movie(zp, ds, T):
    meta = read_array_meta(zp)
    dets = []
    for t in range(T):
        dets.append(detect(load_volume(zp, t, meta)))
        gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()          # the xl/DC activations are done
    t_h = time.time()
    head = head_pass(zp, ds, dets, T)
''',
    '''# ---- the OFFICIAL detector as the node source (offdet_cache.DetectRun) ----
OFF_GATE = 0.99             # offdet_rank: 0.99 > 0.965 on both groups
OFF_FAILED = []             # movies that fell back to xl / DC detection


class _DetectRun:
    """Forward pass with the OFFICIAL detection kept: records each frame's
    peaks (grid -> full-res voxel coords) and sigmoid scores, and the full
    forward softmax matrix of every consecutive pair, in call order."""

    def __init__(self, name, gate):
        self.name, self.gate = name, gate
        self.model = HEAD_MODELS["h1"]
        self.orig_edges = _HEAD_ORIG_EDGES["h1"]
        self.full, self.score = {}, {}
        self.T = 0
        self.next_f = 0
        self.mat = {}

    def detect(self, det_logits, f, det_threshold, pool_kernel):
        arr = _HEAD_ORIG_DETECT(det_logits, f, det_threshold, pool_kernel)
        assert det_threshold == self.gate, (det_threshold, self.gate)
        assert f not in self.full, f"{self.name}: frame {f} detected twice"
        self.T = max(self.T, f + 1)
        if len(arr) == 0:
            self.full[f] = np.zeros((0, 3), np.float32)
            self.score[f] = np.zeros(0, np.float32)
            return arr
        g = arr[:, 1:].astype(np.int64)
        lg = det_logits[0][g[:, 0], g[:, 1], g[:, 2]]
        self.score[f] = torch.sigmoid(lg).float().cpu().numpy().astype(np.float32)
        self.full[f] = (g.astype(np.float32) * HEAD_DS[None, :]).astype(np.float32)
        return arr

    def edges(self, feat_src, feat_tgt, coords_src, coords_tgt, *rest):
        out = self.orig_edges(feat_src, feat_tgt, coords_src, coords_tgt, *rest)
        f = self.next_f
        assert f in self.full and (f + 1) in self.full, (
            f"{self.name}: predict_edges for pair {f} before its detections")
        while len(self.full[f]) == 0 or len(self.full[f + 1]) == 0:
            f += 1
            assert f in self.full and (f + 1) in self.full, (
                f"{self.name}: predict_edges for pair {f} before its detections")
        raw = out[0]
        assert tuple(raw.shape) == (len(self.full[f]), len(self.full[f + 1])), (
            f"{self.name} t={f}: logits {tuple(raw.shape)} vs detected "
            f"({len(self.full[f])}, {len(self.full[f + 1])})")
        for got, ft in ((coords_src[0], f), (coords_tgt[0], f + 1)):
            exp = torch.from_numpy(self.full[ft]).to(got.device)
            err = float((got - exp).abs().max()) if len(exp) else 0.0
            assert err <= 0.5, f"{self.name} t={ft}: coords differ by {err:.3f}"
        self.mat[f] = torch.softmax(raw, dim=0).float().cpu().numpy()
        self.next_f = f + 1
        return out

    def install(self):
        PIPE._detect_cells_pooled = self.detect
        self.model.predict_edges = self.edges


def official_detect(zp, ds, T):
    """(dets, {"h1": (P_f, P_r)}) from the official detector at OFF_GATE with
    the head's forward and reverse matrices over those nodes, or None on
    any failure (printed, counted in OFF_FAILED). Never raises."""
    try:
        cfg_f = PIPE.PredictConfig(det_threshold=OFF_GATE, det_tta=True,
                                   use_ilp=True, threshold=HEAD_NO_CANDIDATES)
        fwd = _DetectRun(ds, OFF_GATE)
        fwd.install()
        try:
            coords, edges = PIPE.predict_video(
                HEAD_MODELS["h1"], zp.parent / ds, HEAD_DEVICE, cfg=cfg_f,
                window_size=HEAD_WINDOW, max_frames=T,
                unet_batch_size=4, downsample=HEAD_DOWNSAMPLE)
        finally:
            _HeadRun.uninstall()
        assert len(edges) == 0, f"{ds}: predict_video kept {len(edges)} edges"
        assert fwd.T == T and all(f in fwd.full for f in range(T)), (
            f"{ds}: detected {len(fwd.full)} frames of T={T}")
        full = [fwd.full[f] for f in range(T)]
        score = [fwd.score[f] for f in range(T)]
        n = np.array([len(x) for x in full], np.int64)
        assert len(coords) == int(n.sum()), (ds, len(coords), int(n.sum()))
        prim = [x.astype(np.float64) for x in full]
        rev = _HeadRun(ds, prim, "rev", "h1")
        rev.install()
        try:
            coords_r, edges_r = PIPE.predict_video(
                HEAD_MODELS["h1"], zp.parent / ds, HEAD_DEVICE, cfg=HEAD_CFG,
                window_size=HEAD_WINDOW, max_frames=T,
                unet_batch_size=4, downsample=HEAD_DOWNSAMPLE)
        finally:
            _HeadRun.uninstall()
        rev.check_return(coords_r, edges_r)
        P_f, P_r = [], []
        for t in range(T - 1):
            n_s, n_t = len(prim[t]), len(prim[t + 1])
            if n_s == 0 or n_t == 0:
                P_f.append(None)
                P_r.append(None)
                continue
            Mf, Mr = fwd.mat[t], rev.mat[t]
            assert Mf.shape == (n_s, n_t) and Mr.shape == (n_t, n_s), (
                f"{ds} t={t}: matrices {Mf.shape} / {Mr.shape} vs ({n_s}, {n_t})")
            assert np.isfinite(Mf).all() and np.isfinite(Mr).all(), (
                f"{ds} t={t}: non-finite softmax probability")
            P_f.append(Mf.astype(np.float32))
            P_r.append(Mr.astype(np.float32))
        dets = [(full[t].astype(np.float64), score[t].astype(np.float64))
                for t in range(T)]
        return dets, {"h1": (P_f, P_r)}
    except Exception as exc:                                  # noqa: BLE001
        _HeadRun.uninstall()
        OFF_FAILED.append(ds)
        print(f"[degraded] {ds}: OFFICIAL detection failed ({type(exc).__name__}: "
              f"{str(exc)[:300]}) -> falling back to xl / DC detection + head pass",
              flush=True)
        return None
    finally:
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


def track_movie(zp, ds, T):
    t_h = time.time()
    off = official_detect(zp, ds, T)
    if off is not None:
        dets, head = off
        print(f"    detection: OFFICIAL gate {OFF_GATE} flip-TTA, "
              f"nodes={sum(len(c) for c, _s in dets)} ({time.time() - t_h:.0f}s)",
              flush=True)
    else:
        meta = read_array_meta(zp)
        dets = []
        for t in range(T):
            dets.append(detect(load_volume(zp, t, meta)))
            gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()          # the xl/DC activations are done
        head = head_pass(zp, ds, dets, T)
''')

rep('''print(f"HEAD PASS: {len(HEAD_FAILED)} (movie, head) passes failed of "
      f"{len(KEEP) * len(HEADS_USED)}" + (f": {HEAD_FAILED}" if HEAD_FAILED
                                          else " (none)"), flush=True)
''',
    '''print(f"HEAD PASS: {len(HEAD_FAILED)} (movie, head) passes failed of "
      f"{len(KEEP) * len(HEADS_USED)}" + (f": {HEAD_FAILED}" if HEAD_FAILED
                                          else " (none)"), flush=True)
print(f"OFFICIAL DETECTION: {len(OFF_FAILED)} of {len(KEEP)} movies fell back "
      f"to xl / DC" + (f": {OFF_FAILED}" if OFF_FAILED else " (none)"), flush=True)
''')

DST.write_text(text, encoding="utf-8")
ast.parse(text)
print(f"wrote {DST} ({len(text.splitlines())} lines), ast OK")
