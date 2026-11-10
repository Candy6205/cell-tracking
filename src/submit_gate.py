"""submit_ep.py -- ref 55889500 (LB 0.925, submit_f4) + the official edge head's three columns in the ranker.

SINGLE VARIABLE vs 55889500: the edge ranker. link_ilp's classifier is
edge_ranker_ep175_bidir.pkl (kernel biohub-edgeprob-rank, D175 pool),
trained on the SAME 18 pair_features columns PLUS three from the official
unet_transformer edge head run inline over OUR primaries:

    p_fwd    softmax over sources, t -> t+1 (the official activation)
    p_rev    the same head over the reversed movie, t+1 -> t
    h_fused  2ab/(a+b), harmonic mean of the two (p_fwd where p_rev is NaN)

Everything else -- detection (F4, four heatmaps), refine, NMS, ILP
constants, gap closing, SHORT_MIN 4, _add_safe_divisions, output writing,
the visible-test local score block -- is byte-identical to submit_f4.py.

WHAT THIS TESTS: edgeprob_rank v1 (D32 pool, BASE primaries) measured the
+BIDIR arm at +0.0046 / +0.0106 adj_J on the 6bba / 44b6 holdouts. The
F4-arm, D175-pool deltas of the very pkl shipped here (biohub-edgeprob-rank
v2, Stage B): [TO BE FILLED IN BY THE USER before submitting].
PRE-REGISTERED READING: LB > 0.925 = the head carries information the 18
coordinate columns lack, and it transfers to a third embryo; LB <= 0.925 =
the local gain did not transfer (the record's 4-in-5 pattern of local
gains that read 0.000 or below at the official judge).

HOW THE HEAD RUNS (edgeprob_cache.py's injection, the v2/v3 code path that
held every round-trip assert on 223 dataset passes): per movie,
PIPE._detect_cells_pooled is replaced by a function returning
seeds_of(dets[t]) -- the same primaries, same order, that pass1_velocities
and pair_features use -- converted full-res voxel -> head grid by
/ downsample as float (no rounding, the training convention);
model.predict_edges is wrapped to capture the raw (n_t, n_t+1) logit matrix
of every consecutive pair; cfg.threshold = 2.0 so predict_video's own
candidate loop keeps nothing; det_tta off (TTA touches only det_logits).
NO PRE-GATING: the FULL softmax matrices P_f[t] (n_t x n_t+1, softmax over
dim 0 = sources) and P_r[t] (n_t+1 x n_t, same activation on the reversed
movie) are kept on the CPU as float32 and link_ilp looks up
p_fwd = P_f[t][i, j], p_rev = P_r[t][j, i] for every pair pair_features
yields, so a gate mismatch is impossible. The round-trip asserts are kept
(coords handed to predict_edges == injected within 0.5 voxel; logit shape
== injected counts; one predict_edges call per non-empty consecutive pair;
predict_video's returned coords == injected), but on the hidden test a
failure must not cost the submission: it is caught PER MOVIE, printed as
[degraded], and that movie is scored with p_fwd = p_rev = h_fused = NaN
(HistGradientBoosting accepts NaN; this pkl has no 18-column path). The
number of such movies is printed after submission.csv is written and again
at the end; 0 is the expected value.

RUNTIME: the head adds ~40-50 s per movie on a T4 (biohub-edgeprob v3
measured 36-67 s for both directions), i.e. +2.2-2.8 h on the ~6 h
pipeline -- still under the 12 h cap. submit_f4.py has no wall-clock guard
and none is added here (nothing else changes). The full matrices are
transient per movie: at 700 cells/frame and 100 frames they are ~400 MB.

---- submit_f4.py's docstring follows, kept as provenance ----

SINGLE VARIABLE vs 55817630: detect() averages FOUR heatmaps instead of
three -- the fourth is the retrained DeepCenter epoch-5 checkpoint
(pct50-99.5 normalisation, 4-flip TTA, same pooled grid), detfuse2's F4
arm. WHAT THIS TESTS: fusescore rejected F4 on the 6bba domain alone
(-0.0053 there, +0.0054 44b6, +0.0027 zh) -- but the xl weights' training
split is unknown and plausibly contains the 6bba eval sets, while the
hidden test is a THIRD embryo (host-confirmed, no embryo_id overlap), so
6bba's veto may be a memorization artifact. The leaderboard is the only
arbiter left. PRE-REGISTERED READING: >0.922 = the 6bba veto was
memorization, DC arc reopens; 0.920-0.922 = uninformative-to-mildly-real;
<=0.918 = the 6bba negativity is real and the DC-injection line closes at
the official judge. Full 4-flip TTA, no degraded variant -- if this hits
the 12 h cap, that is itself the measurement (user's instruction).


Detection side is biohub-xl-repro verbatim (3-branch ensemble + flip TTA);
the linker is the learned edge re-ranker feeding a capacity-2 tracksdata
ILP, and every post-link patch is taken from biohub-ranker-diag, which is
the code path that measured 0.9345 on the 12-dataset 6bba holdout against
the xl baseline's 0.9318 (+0.0027, same detector, same datasets).

Single variable vs LB 0.917 (ref 55690436): the linker.

Internet is OFF at submission time, so tracksdata/pyscipopt come from the
support-pack wheels and `predict_unet_transformer` from the pack's repo
copy -- no git clone anywhere.
"""
import gc
import glob
import json
import pathlib
import pickle
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

SCRIPT_VERSION = "2026-09-06a-submit-gate"
print("SCRIPT_VERSION", SCRIPT_VERSION, flush=True)

DEVICE = "cpu"
if torch.cuda.is_available():
    try:
        _p = nn.Conv3d(1, 1, 3).to("cuda")
        _ = _p(torch.zeros(1, 1, 4, 4, 4, device="cuda")).cpu()
        DEVICE = "cuda"
        del _p
    except Exception as e:                                     # noqa: BLE001
        print("GPU present but conv3d unusable -> CPU:", str(e)[:80])

# ---- offline env: wheels + the pack's repo copy (no clone) --------------
PACK = pathlib.Path("/kaggle/input/datasets/pilkwang/"
                    "biohub-tracking-support-pack-50ep-v1")
if not PACK.exists():
    PACK = pathlib.Path("/kaggle/input/biohub-tracking-support-pack-50ep-v1")
assert PACK.exists(), "support pack not attached"
KEEP_OUT = {"numpy", "scipy", "torch", "pandas", "pyarrow", "numba",
            "llvmlite", "matplotlib", "pillow", "networkx", "scikit-learn",
            "scikit-image", "setuptools", "wheel", "pip"}
WH = pathlib.Path("/kaggle/working/wheelhouse")
shutil.rmtree(WH, ignore_errors=True)
WH.mkdir(parents=True)
for w in sorted(PACK.rglob("*.whl")):
    if w.name.split("-")[0].replace("_", "-").lower() not in KEEP_OUT:
        shutil.copy(w, WH / w.name)
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "--no-index", "--no-deps"]
                   + [str(w) for w in sorted(WH.glob("*.whl"))],
                   capture_output=True, text=True)
print(f"pip wheels exit {r.returncode}", flush=True)
for p in (PACK / "repo" / "src", PACK / "repo" / "scripts"):
    sys.path.insert(0, str(p))

import tracksdata as td                                        # noqa: E402
import predict_unet_transformer as PIPE                        # noqa: E402
print("tracksdata + pipeline import OK", flush=True)

# ---- the trained ranker, from the attached kernel output ----------------
# 2026-08-25: biohub-poolsize showed the ranker had been learning from 32
# datasets while 187 sat cached -- ranker_fine drew on biohub-edge-ranker's
# 44 caches minus the 12 holdout, and biohub-cache-train added the other
# 155 on 08-23 for the division census without anyone retraining. Holding
# everything else fixed, 187 is worth adj_J +0.0032 and score +0.0073, and
# both anchors held (D32 reproduced ranker_fine's 0.9884 edge AUC exactly
# and landed 0.0002 from div-gate's adj_J).
#
# 2026-09-03 (submit_ep): the ranker is edge_ranker_ep175_bidir.pkl from
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

# ---- the fork gate (divbench3, 2026-09-06) -------------------------------
GATE_NAME = "fork_gate_f4.pkl"
GATE_THR = 0.03            # divbench3: p > 0.03 = +0.0121 local score, both folds up, 44b6 +0.0020
GATE_FEATS = ["src", "d_pc1", "d_pq", "d_s", "dv", "p_c1", "p_q", "pf_c1", "pf_q",
              "pr_q", "sc_p", "sc_c1", "sc_q", "len_back", "len_c1", "len_q",
              "dens10", "n_orph", "t_frac", "rank_gap"]
GATE = None
_gk = None
import os as _os                                   # noqa: E402
for _root, _dirs, _files in _os.walk("/kaggle/input"):
    if GATE_NAME in _files:
        _gk = _os.path.join(_root, GATE_NAME)
        break
assert _gk is not None, (f"{GATE_NAME} not found under /kaggle/input -- attach "
                         f"faustine649/biohub-divbench3 as a kernel source")
with open(_gk, "rb") as _f:
    _gb = pickle.load(_f)
assert list(_gb["feature_names"]) == GATE_FEATS, _gb["feature_names"]
assert _gb.get("detector") == "F4", _gb.get("detector")
GATE = _gb["model"]
assert hasattr(GATE, "predict_proba")
print(f"fork gate {GATE_NAME} from {_gk} | version {_gb.get('version')} | "
      f"trained on {_gb.get('trained_on')} | threshold p > {GATE_THR} | "
      f"its p-quantiles over the bench's forks {_gb.get('p_quantiles_all_forks')}", flush=True)
PMAP = {}              # (t, i, j) -> (ranker p, head p_fwd, head p_rev), per movie
SEED_OF_NODE = {}      # ILP node id -> (t, seed index), per movie
FORK_LOG = {}          # (mother, orphan) -> the patch's numbers, per movie

# ---- detect-side constants (xl-repro verbatim) ------------------------
SCALE = np.array([1.625, 0.40625, 0.40625])
POOL = 4
REPAIR = True
CAND_THR = 0.05
# ---- REVERTED 2026-08-26. The block below argued for 0.12; ref
# 55780773 shipped it and scored 0.918 against its own control's 0.920.
# Three ticks is the only clear LOSS of the three submissions on record
# (55757064 0.920 -> 55772253 0.920 was a true zero, not a loss), and
# thr3's own warning -- +8.7% nodes costing about -0.0080 of J -- is the
# most likely reason. The argument is kept in full because thr5 is
# measuring the OPPOSITE direction and will need it.
# ------------------------------------------------------------------------
# SUPERSEDED 2026-08-26: The detection gate. 0.15 was inherited from the public solution and
# swept downward for the first time by biohub-thr-sweep, then in three
# domains by biohub-thr3:
#
#   gate     6bba      44b6   zebrahub
#   0.15   0.9222    0.8603     0.7988   <- was shipped
#   0.12   0.9243    0.8598     0.8068
#   delta  +0.0021   -0.0006    +0.0080
#
# The three-domain rule DISCARDS this: 44b6 dissents. But it dissents by
# 0.0006, under the resolution floor, while Zebrahub -- the dense-GT arm,
# whose adj_J is over ALL cells rather than annotated ones -- gains
# +0.0080, four times 6bba's gain. Submitting it is the user's call
# against the rule, on the reasoning that a leaderboard that reports
# three decimals cannot reward a policy of shipping only >= 0.005, and
# that a score which does not FALL is a green light rather than a red
# one. Rank has gone 155 -> 285 -> 334 while we held still.
#
# THE MECHANISM, which is why this one has a reason and not just a
# number: flicker.py found that at the 20.2% of missed frames which do
# have a sub-threshold peak, the median score is 0.1226. Those cells sit
# between 0.12 and 0.15 and this change admits them.
#
# ⚠ TWO THINGS TO WATCH, both measured by thr3 and both adverse:
# 1. NODES. 548,794 -> 596,529, +8.7%. The metric's node-count penalty
#    is 0.1 * (N_pred - N_true)/N_true, so +8.7% costs about -0.0080 of
#    J -- the same size as Zebrahub's gain. Ref 55772253 showed exactly
#    this cancellation: +1.61% bytes, local +0.0025..+0.0054, LB 0.000.
#    Read total_bytes on this one before reading the score.
# 2. THE REPAIR POOL. _gap_close draws candidates from
#    [CAND_THR, UNET_THRESH) and requires >= GAP1_MIN_CAND_SCORE 0.10,
#    so the usable band goes from [0.10, 0.15) to [0.10, 0.12) -- cut by
#    60%. thr3 measured the whole pool 111,937 -> 64,202. Lowering the
#    gate does not create detections, it MOVES them out of the repair
#    pool into primary nodes, exactly conserved.
UNET_THRESH = 0.15      # reverted from 0.12, see the block above
NMS_UM = 4.0
DETECT_THRESH = min(UNET_THRESH, CAND_THR) if REPAIR else UNET_THRESH

# ---- pipeline constants (ranker-diag verbatim: the 0.9345 config) -----
GATE_UM = 10.0
MAX_LINK_UM = 10.0
TIGHT_UM = 6.0
LINK_SIM_W = 2.0
# ---- THE ONLY CHANGE vs ref 55757064 (LB 0.920) ------------------------
# SHORT_MIN was 6, inherited from the public solution and never swept.
# `_short_filter` deletes a whole predicted track with fewer than
# SHORT_MIN nodes unless it contains a division, so 6 deletes every
# component of length 3, 4 and 5. (Lengths 1 and 2 never occur --
# biohub-short-sweep found SHORT_MIN 1, 2 and 3 give byte-identical
# results in all three domains.)
#
# biohub-short-sweep, 7 values x 3 domains, both anchors held and
# short_4's deltas reproduced biohub-post-sweep EXACTLY:
#
#   SHORT_MIN      6bba      44b6        zh
#     3          0.9258    0.8552    0.8060
#     4          0.9279    0.8553    0.8037
#     5          0.9277    0.8533    0.8015
#     6          0.9254    0.8499    0.7992   <- was shipped
#
# Read as "which track lengths should be deleted":
#   length 5 -> hurts all three domains
#   length 4 -> hurts all three domains
#   length 3 -> +0.0021 6bba, +0.0001 44b6, -0.0023 zh -- CONTESTED
#
# So 4 is the value every domain agrees is at least as good as 6
# (+0.0025 / +0.0054 / +0.0045) and better than 5, while whether to go
# further to 3 is exactly what the three-domain rule cannot decide.
#
# ⚠ The weakest domain's +0.0025 is the same scale as two local gains
# that did NOT appear on the leaderboard (windowing +0.0019 -> -0.001,
# ranker pool +0.0073 -> -0.001). Both of those were measured on 6bba
# alone. This submission is the first test of whether agreeing across
# three domains changes that -- which is the thing being measured here,
# more than the constant itself.
SHORT_MIN = 4
LINEFIT_WEIGHT = 0.8
LINEFIT_WINDOW = 2
GAP1_GATE_UM = 9.0
GAP1_SNAP_UM = 3.2
GAP1_MIN_CAND_SCORE = 0.10
# ---- THE ONLY CHANGE vs ref 55772253 (LB 0.920) ------------------------
# biohub-gap4's `nocand_cap20` arm. TWO constants, no new gate:
#
#   1. GAP1_CAP_FRAC 0.003 -> 0.02, the repair BUDGET;
#   2. the bridge no longer REQUIRES a real sub-threshold candidate --
#      with none, it inserts the midpoint, which at a one-frame gap IS
#      the constant-velocity prediction.
#
#   arm                       6bba      44b6        zh   bridges
#   BASE (shipped)          0.9279    0.8553    0.8037     1,107
#   nocand, budget 0.003   +0.0009   +0.0012   +0.0034     2,195
#   len8,   budget 0.003   +0.0013   +0.0043   +0.0019     1,689
#   nocand, budget 0.02    +0.0013   +0.0113   +0.0056     4,772  <- this
#
# THE BUDGET WAS THE WHOLE PROBLEM. `max_gaps` is handed out in FRAME
# ORDER and the loop breaks the moment it runs out, so under any
# loosened policy what gets repaired is whatever sits at a low frame
# number. Across gap4's 32 datasets true demand is 4,772 bridges, the
# 0.003 allowance is 2,297, and the cap bound in 24 of them. More than
# half of every repairable gap went unrepaired for no reason but the
# budget.
#
# Every gate this project invented -- the candidate requirement, the
# velocity cosine, the fragment length, the clearance radius -- was
# rationing that scarcity rather than selecting good bridges. With the
# budget lifted, len8 still makes only 1,816 of 4,772 and LOSES: it
# discards 62% of what is repairable.
#
# Same sign in all three domains. It ties len8 on the weakest domain
# (+0.0013) and is 2.6x better in 44b6 and 3x better in ZEBRAHUB, the
# only arm whose ground truth is dense -- its rates describe all cells,
# not the annotated ones.
#
# NODE COST: 4,772 bridges against the 765,667 nodes implied by the
# 0.003 allowance is +0.62% of nodes, a node-count multiplier cost of
# 0.00062 -- already inside the deltas above, which are net. The two
# failed submissions grew output by +1.61% and +10.3%.
#
# WARNING: the weakest domain is +0.0013 against holdout noise of
# roughly 0.002-0.003. 6bba barely moves; the case rests on 44b6 and
# Zebrahub.
GAP1_CAP_FRAC = 0.02
DIV_PARENT_UM = 12.0
DIV_SISTER_UM = 15.0
DIV_CHILD_UM = 10.0
DIV_DIVERGE_UM = 2.25
DIV_W_SISTER = 0.15
DIV_FRAME_CAP = 0.0076
DIV_GLOBAL_CAP = 0.00375
ILP_FLOOR = 0.48                            # candidates fed to the ILP
ILP_DISAPP = 1.5
ILP_DIV_W = 0.55   # set from the fine sweep's argmax

CANDIROOT = ["/kaggle/input/biohub-cell-tracking-during-development",
             "/kaggle/input/competitions/biohub-cell-tracking-during-development"]
ROOT = next((p for p in CANDIROOT if Path(p, "test").exists()), None)
assert ROOT is not None, "competition not attached"
TEST_DIR = Path(ROOT) / "test"

# ---- xl detect stack (xl-repro verbatim) ------------------------------

def _find_weight(name):
    cands = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert cands, f"{name} not found"
    return cands[0]

def _block(ci, co):
    return nn.Sequential(nn.Conv3d(ci, co, 3, padding=1), nn.BatchNorm3d(co),
                         nn.ReLU(inplace=True),
                         nn.Conv3d(co, co, 3, padding=1), nn.BatchNorm3d(co),
                         nn.ReLU(inplace=True))

class UNet3D(nn.Module):
    def __init__(self, base=24):
        super().__init__()
        self.e1 = _block(1, base)
        self.e2 = _block(base, base * 2)
        self.e3 = _block(base * 2, base * 4)
        self.pool = nn.MaxPool3d(2)
        self.bott = _block(base * 4, base * 8)
        self.u3 = nn.ConvTranspose3d(base * 8, base * 4, 2, stride=2)
        self.d3 = _block(base * 8, base * 4)
        self.u2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.d2 = _block(base * 4, base * 2)
        self.u1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.d1 = _block(base * 2, base)
        self.out = nn.Conv3d(base, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.bott(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


def _load_model(name):
    ck = torch.load(_find_weight(name), map_location=DEVICE)
    m = UNet3D(base=ck.get("base", 24)).to(DEVICE)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    print("loaded", name, "| val_recall", ck.get("val_recall"), flush=True)
    return m

MODELS = [_load_model("unet3d_bright.pt"), _load_model("unet3d_traintophat.pt")]

def read_array_meta(zp):
    with open(Path(zp) / "0" / "zarr.json") as f:
        m = json.load(f)
    return dict(shape=tuple(m["shape"]), dtype=np.dtype(m["data_type"]))

_ZC = {}

def load_volume(zp, t, meta=None):
    try:
        import zarr
        k = str(zp)
        if k not in _ZC:
            _ZC[k] = zarr.open(k, mode="r")["0"]
        return np.asarray(_ZC[k][t])
    except Exception:
        import blosc2
        if meta is None:
            meta = read_array_meta(zp)
        buf = blosc2.decompress(
            open(Path(zp) / "0" / "c" / str(t) / "0" / "0" / "0", "rb").read())
        return np.frombuffer(buf, dtype=meta["dtype"]).reshape(meta["shape"][1:])

def pool_xy(vol, f=POOL):
    Z, Y, X = vol.shape
    Y2, X2 = (Y // f) * f, (X // f) * f
    v = vol[:, :Y2, :X2].astype(np.float32, copy=False)
    return v.reshape(Z, Y2 // f, f, X2 // f, f).mean(axis=(2, 4))

def pool_norm(vol, preproc=""):
    p = pool_xy(vol)
    if preproc == "tophat":
        from scipy.ndimage import grey_opening
        p = np.clip(p - grey_opening(p, size=(1, 7, 7)), 0.0, None)
    lo = float(np.percentile(p, 50))
    hi = float(np.percentile(p, 99.5))
    return np.clip((p - lo) / (hi - lo + 1e-6), -0.5, 6.0).astype(np.float32)

def _refine(vol, zyx, rz=2, ryx=5):
    Z, Y, X = vol.shape
    z, y, x = (int(round(v)) for v in zyx)
    z0, z1 = max(0, z - rz), min(Z, z + rz + 1)
    y0, y1 = max(0, y - ryx), min(Y, y + ryx + 1)
    x0, x1 = max(0, x - ryx), min(X, x + ryx + 1)
    crop = vol[z0:z1, y0:y1, x0:x1].astype(np.float32)
    bg = float(crop.min())
    w = np.clip(crop - bg, 0, None)
    s = float(w.sum())
    if s <= 0:
        return np.array([z, y, x], float), 0.0
    zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1]
    return (np.array([(zz * w).sum(), (yy * w).sum(), (xx * w).sum()]) / s,
            float(crop.max() - bg))

def _physical_nms(coords, scores, radius_um, scale=SCALE):
    if len(coords) <= 1:
        return coords, scores
    pts = coords * scale[None, :]
    order = np.argsort(-scores)
    tree = cKDTree(pts)
    killed = np.zeros(len(coords), bool)
    keep = []
    for i in order:
        if killed[i]:
            continue
        keep.append(int(i))
        killed[tree.query_ball_point(pts[i], r=radius_um)] = True
    keep = np.array(keep)
    return coords[keep], scores[keep]

# ---- F4: ep05 DeepCenter, the fourth heatmap ---------------------------
class _DCConvBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = min(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _DCDeepCenterUNet3D(nn.Module):
    def __init__(self, in_channels=1, base_channels=24):
        super().__init__()
        c = int(base_channels)
        self.enc1 = _DCConvBlock3d(in_channels, c)
        self.down1 = nn.MaxPool3d(2, 2)
        self.enc2 = _DCConvBlock3d(c, c * 2)
        self.down2 = nn.MaxPool3d(2, 2)
        self.enc3 = _DCConvBlock3d(c * 2, c * 4)
        self.down3 = nn.MaxPool3d(2, 2)
        self.bottleneck = _DCConvBlock3d(c * 4, c * 8)
        self.up3 = nn.ConvTranspose3d(c * 8, c * 4, 2, 2)
        self.dec3 = _DCConvBlock3d(c * 8, c * 4)
        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, 2, 2)
        self.dec2 = _DCConvBlock3d(c * 4, c * 2)
        self.up1 = nn.ConvTranspose3d(c * 2, c, 2, 2)
        self.dec1 = _DCConvBlock3d(c * 2, c)
        self.head = nn.Conv3d(c, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


_DC_PATHS = [
    "/kaggle/input/datasets/faustine649/biohub-dctrain-175-weights/"
    "best_ep05.pt",
    "/kaggle/input/biohub-dctrain-175-weights/best_ep05.pt",
]
_dcp = next((p for p in _DC_PATHS if pathlib.Path(p).exists()), None)
assert _dcp is not None, (
    "best_ep05.pt not found -- attach faustine649/biohub-dctrain-175-weights")
_dck = torch.load(_dcp, map_location=DEVICE, weights_only=False)
assert int(_dck.get("epoch", -1)) == 5, (
    f"best_ep05.pt is epoch {_dck.get('epoch')}, expected 5")
_dccfg = dict(_dck.get("config", {}))
DC_NORM = (float(_dccfg.get("norm_lo_pct", 50.0)),
           float(_dccfg.get("norm_hi_pct", 99.5)),
           float(_dccfg.get("norm_clip_lo", -0.5)),
           float(_dccfg.get("norm_clip_hi", 6.0)))
DC_MODEL = _DCDeepCenterUNet3D(
    base_channels=int(_dccfg.get("base_channels", 24)))
DC_MODEL.load_state_dict(_dck["model_state"])
DC_MODEL = DC_MODEL.to(DEVICE).eval()
print("loaded best_ep05.pt | epoch", _dck.get("epoch"),
      "| val best_score", _dck.get("best_score"), flush=True)
del _dck


def _dc_normalize(p):
    lo_pct, hi_pct, clip_lo, clip_hi = DC_NORM
    lo = float(np.percentile(p, lo_pct))
    hi = float(np.percentile(p, hi_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(p, dtype=np.float32)
    return np.clip((p - lo) / (hi - lo), clip_lo, clip_hi).astype(np.float32)


_BRANCHES = [([MODELS[0]], [1.0], ""), ([MODELS[1]], [1.0], "tophat"),
             ([_load_model("unet3d_v2_tophat_b32.pt")], [1.0], "tophat")]
_FLIP_VIEWS = (0, 1, 2, 3)

def _flip(x, i):
    if i & 1:
        x = np.flip(x, -2)
    if i & 2:
        x = np.flip(x, -1)
    return x

def detect(vol):
    hs = []
    for models, ws, pp in _BRANCHES:
        x = pool_norm(vol, pp)
        acc = None
        for i in _FLIP_VIEWS:
            xv = np.ascontiguousarray(_flip(x, i))
            with torch.no_grad():
                lg = None
                for m, w in zip(models, ws):
                    l_ = m(torch.from_numpy(xv)[None, None].to(DEVICE)
                           )[0, 0].float().cpu().numpy()
                    lg = w * l_ if lg is None else lg + w * l_
            lg = _flip(lg, i)
            acc = lg if acc is None else acc + lg
        hs.append(1.0 / (1.0 + np.exp(-(acc / len(_FLIP_VIEWS)))))
    xd = _dc_normalize(pool_xy(vol))
    accd = None
    for i in _FLIP_VIEWS:
        xv = np.ascontiguousarray(_flip(xd, i))
        with torch.no_grad():
            lg = DC_MODEL(torch.from_numpy(xv)[None, None].to(DEVICE)
                          )[0, 0].float().cpu().numpy()
        lg = _flip(lg, i)
        accd = lg if accd is None else accd + lg
    hs.append(1.0 / (1.0 + np.exp(-np.clip(accd / 4.0, -60.0, 60.0))))
    h = hs[0] if len(hs) == 1 else np.mean(hs, axis=0)
    pk = peak_local_max(h, min_distance=1, threshold_abs=DETECT_THRESH,
                        exclude_border=False)
    if len(pk) == 0:
        return np.zeros((0, 3)), np.zeros(0)
    sc = h[pk[:, 0], pk[:, 1], pk[:, 2]].astype(float)
    coords = pk.astype(float)
    coords[:, 1] = coords[:, 1] * POOL + (POOL - 1) / 2
    coords[:, 2] = coords[:, 2] * POOL + (POOL - 1) / 2
    ref = np.array([_refine(vol, c)[0] for c in coords])
    return _physical_nms(ref, sc, NMS_UM)


# ------------------------------------------------------------- graph ops

def avail_T(zp):
    meta = read_array_meta(zp)
    T = meta["shape"][0]
    present = [t for t in range(T)
               if (Path(zp) / "0" / "c" / str(t) / "0" / "0" / "0").exists()]
    return max(present) + 1 if present else 0

# ---- ranker stack (ranker-diag verbatim) ------------------------------

def _dist_um(a, b):
    d = (np.asarray(a, float) - np.asarray(b, float)) * SCALE
    return float(np.sqrt((d * d).sum()))


def _logit(s):
    s = np.clip(s, 1e-4, 1 - 1e-4)
    return np.log(s / (1 - s))

def _link_sim(prev_xyz, curr_xyz, prev_vel, prev_sc, curr_sc):
    if len(prev_xyz) == 0 or len(curr_xyz) == 0:
        return []
    P = prev_xyz * SCALE[None, :]
    C = curr_xyz * SCALE[None, :]
    pred = P + (0.5 * prev_vel if prev_vel is not None else 0.0)
    N, M = len(P), len(C)
    BIG = 1e9
    sim = LINK_SIM_W * np.abs(_logit(prev_sc)[:, None] - _logit(curr_sc)[None])

    def _hun(pi, ci, gate):
        if len(pi) == 0 or len(ci) == 0:
            return []
        Draw = np.sqrt(((P[pi][:, None] - C[ci][None]) ** 2).sum(2))
        D = np.sqrt(((pred[pi][:, None] - C[ci][None]) ** 2).sum(2))
        cost = np.where(Draw > gate, BIG, D) + \
            np.where(Draw > gate, 0.0, sim[np.ix_(pi, ci)])
        ri, rc = linear_sum_assignment(cost)
        return [(int(pi[r]), int(ci[c]))
                for r, c in zip(ri, rc) if cost[r, c] < BIG]

    links = _hun(np.arange(N), np.arange(M), min(TIGHT_UM, MAX_LINK_UM))
    up = {p for p, _ in links}
    uc = {c for _, c in links}
    fp = np.array([i for i in range(N) if i not in up], int)
    fc = np.array([j for j in range(M) if j not in uc], int)
    return links + _hun(fp, fc, MAX_LINK_UM)

def seeds_of(dets_t):
    coords, scores = dets_t
    coords = np.asarray(coords, float).reshape(-1, 3)
    scores = np.asarray(scores, float).reshape(-1)
    m = scores >= UNET_THRESH
    return coords[m], scores[m]

def pass1_velocities(dets):
    T = len(dets)
    vels = [np.zeros((len(seeds_of(dets[t])[0]), 3)) for t in range(T)]
    for t in range(T - 1):
        P, ps = seeds_of(dets[t])
        C, cs = seeds_of(dets[t + 1])
        if len(P) == 0 or len(C) == 0:
            continue
        for pi, ci in _link_sim(P, C, vels[t] if len(vels[t]) else None,
                                ps, cs):
            vels[t + 1][ci] = (C[ci] - P[pi]) * SCALE
    return vels

def pair_features(dets, vels, T):
    out = []
    for t in range(T - 1):
        P, ps = seeds_of(dets[t])
        C, cs = seeds_of(dets[t + 1])
        if len(P) == 0 or len(C) == 0:
            out.append(([], np.zeros((0, 18), np.float32), P, C))
            continue
        Pu = P * SCALE[None, :]
        Cu = C * SCALE[None, :]
        tree_p = cKDTree(Pu)
        tree_c = cKDTree(Cu)
        dens_p = np.array([len(tree_p.query_ball_point(x, 7.0)) - 1
                           for x in Pu])
        dens_c = np.array([len(tree_c.query_ball_point(x, 7.0)) - 1
                           for x in Cu])
        pred = Pu + vels[t]
        pairs = []
        rows = []
        cand_of_i = defaultdict(list)
        cand_of_j = defaultdict(list)
        raw = {}
        for i, x in enumerate(Pu):
            for j in tree_c.query_ball_point(x, GATE_UM):
                d = float(np.linalg.norm(Cu[j] - x))
                raw[(i, j)] = d
                cand_of_i[i].append((d, j))
                cand_of_j[j].append((d, i))
        for i in cand_of_i:
            cand_of_i[i].sort()
        for (i, j), d in raw.items():
            delta = Cu[j] - Pu[i]
            md = float(np.linalg.norm(Cu[j] - pred[i]))
            rank = next(k for k, (_, jj) in enumerate(cand_of_i[i])
                        if jj == j)
            rows.append([d, delta[0], abs(delta[0]),
                         float(np.linalg.norm(delta[1:])), md, d - md,
                         ps[i], cs[j],
                         abs(float(_logit(np.array([ps[i]]))[0]
                                   - _logit(np.array([cs[j]]))[0])),
                         dens_p[i], dens_c[j], len(P), len(C),
                         rank, len(cand_of_i[i]), len(cand_of_j[j]),
                         float(np.linalg.norm(vels[t][i])), t / max(1, T)])
            pairs.append((i, j))
        out.append((pairs, np.asarray(rows, np.float32), P, C))
    return out

# ---- post-link patches (ranker-diag verbatim: the 0.9345 path) --------

def _gap_close(nodes, edges, cand, cand_sc, cand_trees, next_id):
    T = len(cand)
    succ = {}
    pred = {}
    for a, b in edges:
        succ[a] = b
        pred[b] = a
    by_t = defaultdict(list)
    for nid, n in nodes.items():
        by_t[n["t"]].append(nid)
    cand_used = [np.zeros(len(c), bool) for c in cand]
    max_gaps = max(1, int(GAP1_CAP_FRAC * len(nodes)))
    n_added = 0
    BIG = 1e9
    for t in range(T - 2):
        if n_added >= max_gaps:
            break
        ends = [g for g in by_t.get(t, []) if g not in succ and g in pred]
        starts = [g for g in by_t.get(t + 2, [])
                  if g not in pred and g in succ]
        if not ends or not starts:
            continue
        P = np.asarray([nodes[g]["xyz"] for g in ends], float) * SCALE
        C = np.asarray([nodes[g]["xyz"] for g in starts], float) * SCALE
        D = np.sqrt(((P[:, None] - C[None]) ** 2).sum(2))
        cost = np.where(D > GAP1_GATE_UM, BIG, D)
        ri, ci = linear_sum_assignment(cost)
        for r_, c_ in zip(ri, ci):
            if n_added >= max_gaps or cost[r_, c_] >= BIG:
                continue
            ge, gs = ends[r_], starts[c_]
            if ge in succ or gs in pred:
                continue
            # ---- NO GATE. biohub-gap4 removed the len8 gate that stood
            # here: with the budget lifted it discards 62% of the
            # repairable gaps (1,816 bridges made of 4,772 available)
            # and loses 2.6x in 44b6 and 3x in Zebrahub.
            mid = 0.5 * (nodes[ge]["xyz"] + nodes[gs]["xyz"])
            # ---- gap3's `nocand`: a candidate is preferred, not required.
            # The old code did `continue` on every miss below; now a miss
            # falls through to the midpoint. At a ONE-frame gap the
            # midpoint is the constant-velocity prediction, not a guess.
            use = None
            tree = cand_trees[t + 1]
            if tree is not None:
                dist, idx = tree.query(mid * SCALE)
                if dist <= GAP1_SNAP_UM and not cand_used[t + 1][idx] \
                        and cand_sc[t + 1][idx] >= GAP1_MIN_CAND_SCORE:
                    use = np.asarray(cand[t + 1][idx], float)
                    cand_used[t + 1][idx] = True
            if use is None:
                use = np.asarray(mid, float)
            ng = next_id
            next_id += 1
            nodes[ng] = {"t": t + 1, "xyz": use}
            edges.append((ge, ng))
            edges.append((ng, gs))
            succ[ge] = ng
            pred[ng] = ge
            succ[ng] = gs
            pred[gs] = ng
            n_added += 1
    return next_id

def _short_filter(nodes, edges):
    parent = {nid: nid for nid in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    out_count = defaultdict(int)
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
        out_count[a] += 1
    comps = defaultdict(list)
    for nid in nodes:
        comps[find(nid)].append(nid)
    keep = set()
    for members in comps.values():
        if len(members) >= SHORT_MIN or \
                any(out_count[n] >= 2 for n in members):
            keep.update(members)
    nodes2 = {nid: n for nid, n in nodes.items() if nid in keep}
    edges2 = [(a, b) for a, b in edges if a in nodes2 and b in nodes2]
    return nodes2, edges2

def _linefit(nodes, edges):
    pred = defaultdict(list)
    succ = defaultdict(list)
    for a, b in edges:
        if int(nodes[b]["t"]) == int(nodes[a]["t"]) + 1:
            succ[a].append(b)
            pred[b].append(a)
    orig = {k: v["xyz"].copy() for k, v in nodes.items()}
    updates = {}
    for nid in nodes:
        neigh = [(0, nid)]
        cur = nid
        for step in range(1, LINEFIT_WINDOW + 1):
            ps = pred.get(cur, [])
            if len(ps) != 1:
                break
            cur = ps[0]
            neigh.append((-step, cur))
        cur = nid
        for step in range(1, LINEFIT_WINDOW + 1):
            ss = succ.get(cur, [])
            if len(ss) != 1:
                break
            cur = ss[0]
            neigh.append((step, cur))
        if len(neigh) < 3:
            continue
        dt = np.array([a for a, _ in neigh], float)
        xyz = np.stack([orig[n] for _, n in neigh])
        fit = np.array([np.polyval(np.polyfit(dt, xyz[:, ax], 1), 0.0)
                        for ax in range(3)])
        if np.isfinite(fit).all():
            updates[nid] = (1 - LINEFIT_WEIGHT) * orig[nid] + \
                LINEFIT_WEIGHT * fit
    for nid, xyz in updates.items():
        nodes[nid]["xyz"] = xyz

def _add_safe_divisions(nodes, edges):
    succ = defaultdict(list)
    pred = defaultdict(list)
    for a, b in edges:
        succ[a].append(b)
        pred[b].append(a)
    by_t = defaultdict(list)
    for nid, n in nodes.items():
        by_t[n["t"]].append(nid)
    orphan_tree = {}
    orphan_ids = {}
    for t, ids in by_t.items():
        orph = [g for g in ids if not pred.get(g)]
        orphan_ids[t] = orph
        if orph:
            orphan_tree[t] = cKDTree(
                np.asarray([nodes[g]["xyz"] for g in orph], float) * SCALE)
    proposals = []
    for p, n in nodes.items():
        ch = succ.get(p, [])
        if len(ch) != 1:
            continue
        c1 = ch[0]
        t = n["t"]
        if nodes[c1]["t"] != t + 1 or len(pred.get(p, [])) != 1:
            continue
        if _dist_um(n["xyz"], nodes[c1]["xyz"]) > DIV_CHILD_UM:
            continue
        tree = orphan_tree.get(t + 1)
        if tree is None:
            continue
        for idx in tree.query_ball_point(n["xyz"] * SCALE, r=DIV_PARENT_UM):
            q = orphan_ids[t + 1][idx]
            if q == c1:
                continue
            d_pq = _dist_um(n["xyz"], nodes[q]["xyz"])
            d_s = _dist_um(nodes[c1]["xyz"], nodes[q]["xyz"])
            if d_s > DIV_SISTER_UM:
                continue
            d_q, q_idx = tree.query(nodes[c1]["xyz"] * SCALE)
            if orphan_ids[t + 1][q_idx] != q or d_q > DIV_SISTER_UM:
                continue
            c1n, qn = succ.get(c1, []), succ.get(q, [])
            if len(c1n) != 1 or len(qn) != 1:
                continue
            dv_ = _dist_um(nodes[c1n[0]]["xyz"], nodes[qn[0]]["xyz"]) - d_s
            if dv_ < DIV_DIVERGE_UM:
                continue
            proposals.append((d_pq + DIV_W_SISTER * d_s, t, p, q, c1, d_pq, d_s, dv_))
    proposals.sort()
    used_p = set()
    used_q = set()
    per_frame = defaultdict(int)
    n_frame = {t: len(ids) for t, ids in by_t.items()}
    max_global = max(1, int(DIV_GLOBAL_CAP * len(nodes)))
    added = 0
    for score, t, p, q, c1, d_pq, d_s, dv_ in proposals:
        if added >= max_global:
            break
        if p in used_p or q in used_q:
            continue
        if per_frame[t] >= max(1, int(DIV_FRAME_CAP * n_frame[t])):
            continue
        FORK_LOG[(p, q)] = dict(c1=c1, d_pq=d_pq, d_s=d_s, dv=dv_,
                                n_orph=len(orphan_ids.get(t + 1, [])))
        edges.append((p, q))
        succ[p].append(q)
        pred[q].append(p)
        used_p.add(p)
        used_q.add(q)
        per_frame[t] += 1
        added += 1
    return added

COLS = ["dataset", "row_type", "node_id", "t", "z", "y", "x",
        "source_id", "target_id"]

def _chain_len(start, step_map, cap=12):
    n, cur = 0, start
    while n < cap:
        nxt = step_map.get(cur, [])
        if len(nxt) != 1:
            break
        cur = nxt[0]
        n += 1
    return n


def _gate_prune(nodes, edges, dets):
    """divbench3's fork_rows + the cut: every fork of the final graph gets the
    20 bench features, the gate's p, and loses its mother->orphan edge when
    p <= GATE_THR. Returns the new edge list and (forks, removed)."""
    succ, pred = defaultdict(list), defaultdict(list)
    for a, b in edges:
        succ[a].append(b)
        pred[b].append(a)
    T = max(v["t"] for v in nodes.values()) + 1
    seeds = [seeds_of(dets[t]) for t in range(T)]
    trees = {t: (cKDTree(np.asarray(seeds[t][0], float) * SCALE) if len(seeds[t][0]) else None)
             for t in range(T)}

    def _p(a, b):
        sa, sb = SEED_OF_NODE.get(a), SEED_OF_NODE.get(b)
        if sa is None or sb is None or sb[0] != sa[0] + 1:
            return (np.nan, np.nan, np.nan)
        return PMAP.get((sa[0], sa[1], sb[1]), (np.nan, np.nan, np.nan))

    def _sc(a):
        sa = SEED_OF_NODE.get(a)
        return float(seeds[sa[0]][1][sa[1]]) if sa is not None else np.nan

    forks = []
    X = []
    for p, ch in succ.items():
        if len(ch) < 2:
            continue
        t = nodes[p]["t"]
        patched = [q for q in ch if (p, q) in FORK_LOG]
        if patched:
            q = patched[0]
            src = 1
            c1 = FORK_LOG[(p, q)]["c1"]
            if c1 not in ch:
                c1 = [x for x in ch if x != q][0]
        else:
            src = 0
            pr_ = {x: _p(p, x)[0] for x in ch}
            ch_sorted = sorted(ch, key=lambda x: np.nan_to_num(pr_[x], nan=-1.0))
            q, c1 = ch_sorted[0], ch_sorted[-1]
        p_c1, pf_c1, _ = _p(p, c1)
        p_q, pf_q, pr_q = _p(p, q)
        d_pc1 = _dist_um(nodes[p]["xyz"], nodes[c1]["xyz"])
        d_pq = _dist_um(nodes[p]["xyz"], nodes[q]["xyz"])
        d_s = _dist_um(nodes[c1]["xyz"], nodes[q]["xyz"])
        c1n, qn = succ.get(c1, []), succ.get(q, [])
        dv = (_dist_um(nodes[c1n[0]]["xyz"], nodes[qn[0]]["xyz"]) - d_s
              if len(c1n) == 1 and len(qn) == 1 else np.nan)
        tr = trees.get(t)
        dens10 = (len(tr.query_ball_point(np.asarray(nodes[p]["xyz"], float) * SCALE, r=10.0)) - 1
                  if tr is not None else np.nan)
        lg = FORK_LOG.get((p, q), {})
        rank_gap = (p_c1 - p_q) if (p_c1 == p_c1 and p_q == p_q) else np.nan
        row = dict(src=src, d_pc1=d_pc1, d_pq=d_pq, d_s=d_s, dv=dv, p_c1=p_c1, p_q=p_q,
                   pf_c1=pf_c1, pf_q=pf_q, pr_q=pr_q, sc_p=_sc(p), sc_c1=_sc(c1), sc_q=_sc(q),
                   len_back=_chain_len(p, pred), len_c1=_chain_len(c1, succ),
                   len_q=_chain_len(q, succ), dens10=dens10,
                   n_orph=lg.get("n_orph", np.nan), t_frac=t / max(1, T - 1),
                   rank_gap=rank_gap)
        forks.append((p, q))
        X.append([row[f] for f in GATE_FEATS])
    if not forks:
        return edges, (0, 0)
    pg = GATE.predict_proba(np.asarray(X, float))[:, 1]
    drop = {pq for pq, pv in zip(forks, pg) if pv <= GATE_THR}
    kept = [e for e in edges if e not in drop]
    return kept, (len(forks), len(drop))


def _emit(ds, nodes, edges):
    edge_set = []
    seen = set()
    for a, b in edges:
        if a == b or a not in nodes or b not in nodes or (a, b) in seen:
            continue
        seen.add((a, b))
        edge_set.append((a, b))
    used = set()
    for a, b in edge_set:
        used.add(a)
        used.add(b)
    nrows = []
    erows = []
    for nid in sorted(used):
        n = nodes[nid]
        z, y, x = n["xyz"]
        nrows.append((ds, "node", int(nid), int(n["t"]), float(z), float(y),
                      float(x), -1, -1))
    for a, b in edge_set:
        if a in used and b in used:
            erows.append((ds, "edge", -1, -1, -1, -1, -1, int(a), int(b)))
    return (pd.DataFrame(nrows, columns=COLS),
            pd.DataFrame(erows, columns=COLS))


# ------------------------------------------- post-link patches (final)


# ---- the official edge head, run inline over OUR primaries -------------
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
            assert np.all(blk[:, 0] == f), \
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
            assert Mf.min() >= 0 and Mf.max() <= 1 and \
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
    PMAP.clear()
    SEED_OF_NODE.clear()
    vels = pass1_velocities(dets)
    feats = pair_features(dets, vels, T)
    coords_rows = []
    offset = {}
    n_glob = 0
    for t in range(T):
        c, s_ = seeds_of(dets[t])
        offset[t] = n_glob
        n_glob += len(c)
        for xyz in c:
            coords_rows.append([t, xyz[0], xyz[1], xyz[2]])
    coords = np.asarray(np.round(coords_rows), np.int16)
    cand_edges = []
    n_pairs = 0
    for t in range(T - 1):
        pairs, rows, Pp, Cc = feats[t]
        if not pairs:
            continue
        assert rows.shape == (len(pairs), NFEAT_BASE), rows.shape
        extra = _head_columns(head, t, pairs)
        pr = CLF.predict_proba(np.concatenate([rows, extra], axis=1))[:, 1]
        n_pairs += len(pairs)
        for (i, j), p, ex in zip(pairs, pr, extra):
            PMAP[(t, int(i), int(j))] = (float(p), float(ex[0]), float(ex[1]))
        for (i, j), p in zip(pairs, pr):
            if p >= ILP_FLOOR:
                cand_edges.append((offset[t] + i, offset[t + 1] + j,
                                   float(p), 0.0))
    graph = PIPE.build_graph(coords, cand_edges)
    if graph.num_edges() > 0:
        solver = td.solvers.ILPSolver(
            edge_weight=-1.0 * td.EdgeAttr("edge_prob"),
            appearance_weight=0.0,
            disappearance_weight=ILP_DISAPP,
            division_weight=div_w)
        with PIPE.suppress_output():
            graph = solver.solve(graph)
    na = graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
    nid_l = [int(v) for v in na["node_id"].to_list()]
    remap = {v: i for i, v in enumerate(nid_l)}
    nodes_arr = np.stack([np.asarray(na[c].to_list(), float)
                          for c in ("t", "z", "y", "x")], axis=1)
    if graph.num_edges() > 0:
        ea = graph.edge_attrs(attr_keys=["source_id", "target_id"])
        rows_e, dropped = [], 0
        for a, b in zip(ea["source_id"].to_list(),
                        ea["target_id"].to_list()):
            ra, rb = remap.get(int(a)), remap.get(int(b))
            if ra is None or rb is None:
                dropped += 1
                continue
            rows_e.append([ra, rb])
        if dropped:
            print(f"    WARN dropped {dropped} ILP edges with absent "
                  f"endpoints -- this movie is a LOWER BOUND", flush=True)
        e2 = np.array(rows_e, int) if rows_e else np.zeros((0, 2), int)
    else:
        e2 = np.zeros((0, 2), int)
    _oc = defaultdict(int)
    for _a, _b in e2:
        _oc[int(_a)] += 1
    print(f"    ilp: pairs={n_pairs} cand={len(cand_edges)} "
          f"edges={len(e2)} forks={sum(1 for v in _oc.values() if v >= 2)}",
          flush=True)
    nodes = {i + 1: {"t": int(nodes_arr[i, 0]),
                     "xyz": nodes_arr[i, 1:].astype(float)}
             for i in range(len(nodes_arr))}
    edges = [(int(a) + 1, int(b) + 1) for a, b in e2]
    _cd = {}
    for g_, (t_, z_, y_, x_) in enumerate(coords.tolist()):
        _cd.setdefault((int(t_), int(z_), int(y_), int(x_)), (int(t_), g_ - offset[int(t_)]))
    for i in range(len(nodes_arr)):
        SEED_OF_NODE[i + 1] = _cd.get(tuple(int(round(v)) for v in nodes_arr[i]))
    cand = []
    cand_sc = []
    cand_trees = []
    for t in range(T):
        c_all = np.asarray(dets[t][0], float)
        s_all = np.asarray(dets[t][1], float)
        cm = (s_all >= CAND_THR) & (s_all < UNET_THRESH)
        cand.append(c_all[cm])
        cand_sc.append(s_all[cm])
        cand_trees.append(cKDTree(cand[-1] * SCALE)
                          if len(cand[-1]) else None)
    return nodes, edges, cand, cand_sc, cand_trees


def track_movie(zp, ds, T):
    meta = read_array_meta(zp)
    dets = []
    for t in range(T):
        dets.append(detect(load_volume(zp, t, meta)))
        gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()          # the xl/DC activations are done
    t_h = time.time()
    head = head_pass(zp, ds, dets, T)
    print(f"    head: {'ok' if head is not None else 'FAILED -> NaN columns'}"
          f" ({time.time() - t_h:.0f}s)", flush=True)
    nodes, edges, cand, cand_sc, cand_trees = link_ilp(dets, T, ILP_DIV_W,
                                                       head)
    del head
    _gap_close(nodes, edges, cand, cand_sc, cand_trees, len(nodes) + 1)
    nodes, edges = _short_filter(nodes, edges)
    _linefit(nodes, edges)
    FORK_LOG.clear()
    _add_safe_divisions(nodes, edges)
    n_before = len(edges)
    try:
        edges, (n_forks, n_drop) = _gate_prune(nodes, edges, dets)
    except Exception as _e:                      # noqa: BLE001
        print(f"    WARN fork gate failed ({type(_e).__name__}: {_e}) -- forks kept", flush=True)
        n_forks, n_drop = -1, 0
    _oc = defaultdict(int)
    for a, b in edges:
        _oc[a] += 1
    print(f"    gate: forks={n_forks} removed={n_drop} (edges {n_before} -> {len(edges)})",
          flush=True)
    print(f"    final: nodes={len(nodes)} edges={len(edges)} "
          f"forks={sum(1 for v in _oc.values() if v >= 2)}", flush=True)
    return _emit(ds, nodes, edges), nodes, edges


parts = []
KEEP = {}                    # ds -> (nodes, edges), kept for the local score
for zp in sorted(TEST_DIR.glob("*.zarr")):
    ds = zp.name.replace(".zarr", "")
    T = avail_T(zp)
    if T == 0:
        print("skip", ds, flush=True)
        continue
    t0 = time.time()
    (nrows, erows), nodes, edges = track_movie(zp, ds, T)
    parts += [nrows, erows]
    KEEP[ds] = (nodes, edges)
    print(f"  {ds}: T={T} nodes={len(nrows)} edges={len(erows)} "
          f"({time.time()-t0:.1f}s)", flush=True)
assert parts, "no test movie produced output"
sub = pd.concat(parts, ignore_index=True)
sub.index.name = "id"
sub.to_csv("submission.csv")
exp = ["dataset", "row_type", "node_id", "t", "z", "y", "x",
       "source_id", "target_id"]
assert list(sub.columns) == exp
print("wrote submission.csv rows", len(sub), "| nodes",
      (sub.row_type == "node").sum(), "edges",
      (sub.row_type == "edge").sum(), flush=True)
print(f"HEAD PASS: {len(HEAD_FAILED)} of {len(KEEP)} movies scored with NaN "
      f"head columns" + (f": {HEAD_FAILED}" if HEAD_FAILED else " (none)"),
      flush=True)

# ---- local score, AFTER the CSV is safely on disk -----------------------
# The visible test/ holds 4 copies of train movies, so this run can be
# scored against real GT. At the hidden rescore those names are absent and
# scoring is skipped -- announced, never silent (rule 1). The whole block
# is wrapped: a scoring failure must never cost a submission.
try:
    CODE_ROOTS = [
        "/kaggle/input/datasets/faustine649/cell-tracking/cell tracking",
        "/kaggle/input/datasets/faustine649/cell-tracking",
        "/kaggle/input/cell-tracking/cell tracking",
        "/kaggle/input/cell-tracking",
    ]
    code = next((c for c in CODE_ROOTS
                 if pathlib.Path(c, "division_recovery.py").exists()), None)
    assert code is not None, "code dataset not attached"
    sys.path.insert(0, code)
    import division_recovery as dr                             # noqa: E402
    from viz_cells import find_root, geff_path, read_geff       # noqa: E402
    from tracking_cellmot.io import open_dataset                # noqa: E402
    from tracking_cellmot.metrics import (                      # noqa: E402
        evaluate as compute_metric, node_recall, per_sample_metrics,
        summarise)

    root = find_root(None)
    PRED = pathlib.Path("/kaggle/working/pred_score")
    PRED.mkdir(exist_ok=True)
    scored = []
    for ds, (nodes, edges) in KEEP.items():
        if not (root / "train" / f"{ds}.zarr").exists():
            print(f"SCORING: {ds} has no train GT -- hidden data, skipped",
                  flush=True)
            continue
        ids = sorted(nodes)
        rm = {g: i for i, g in enumerate(ids)}
        arr = np.array([[nodes[g]["t"], *nodes[g]["xyz"]] for g in ids],
                       float)
        e3 = np.array([[rm[a], rm[b]] for a, b in edges
                       if a in rm and b in rm], int)
        dr._save(str(PRED / f"{ds}.geff"), arr, e3)
        scored.append(ds)
    if not scored:
        print("SCORING SKIPPED: no emitted dataset has train GT (the "
              "expected path on the hidden rescore)", flush=True)
    else:
        rows = []
        for ds in scored:
            gp = PRED / f"{ds}.geff"
            dset = open_dataset(root / "train" / ds, require_tracks=True)
            res_ = td.graph.IndexedRXGraph.from_geff(str(gp))
            g = res_[0] if isinstance(res_, tuple) else res_
            er = compute_metric(g, dset.tracks, scale=dset.scale)
            rec = (node_recall(g, dset.tracks)
                   if g.num_edges() > 0 and g.num_nodes() > 0 else 0.0)
            gt = read_geff(geff_path(root, "train", ds))
            est = gt["extra"].get("estimated_number_of_nodes")
            rows.append(per_sample_metrics(er, float(est) if est
                                           else float("nan"), rec))
        sm = summarise(rows)
        print(f"LOCAL SCORE on {len(scored)} visible datasets: "
              f"score {sm['score']:.4f}  adj_J {sm['adj_edge_jaccard']:.4f}  "
              f"div_J {sm['division_jaccard']:.4f}  "
              f"(divTP {sm['division_tp']} FP {sm['division_fp']} "
              f"FN {sm['division_fn']})", flush=True)
        print("  datasets:", ", ".join(scored), flush=True)
except Exception as e:                                       # noqa: BLE001
    import traceback
    print("SCORING FAILED (submission.csv is already written and valid):",
          flush=True)
    traceback.print_exc()

if HEAD_FAILED:
    print(f"[degraded] {len(HEAD_FAILED)} movie(s) ran WITHOUT the head's "
          f"columns: {HEAD_FAILED}", flush=True)
print("SUBMIT RANKER DONE", flush=True)
