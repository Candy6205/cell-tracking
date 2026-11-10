"""Generate submit_f4.py from submit_ranker.py by asserted replacement.

Single variable vs ref 55817630 (LB 0.922): ep05 DeepCenter added as a
FOURTH equal-weight heatmap before peak finding (detfuse2's F4 arm).
Everything else -- thresholds, refine, NMS, D187 ranker, global solve,
gap budget 0.02 nocand, SHORT_MIN 4, all patches -- byte-identical.
"""
import ast
import pathlib

SRC = pathlib.Path(__file__).parent / "submit_ranker.py"
DST = pathlib.Path(__file__).parent / "submit_f4.py"
text = SRC.read_text(encoding="utf-8")


def rep(old, new, n=1):
    global text
    c = text.count(old)
    assert c == n, f"anchor x{c}, wanted x{n}: {old[:70]!r}"
    text = text.replace(old, new)


rep('"""submit_ranker.py -- the 0.917 xl pipeline with its linker '
    'replaced.',
    '"""submit_f4.py -- ref 55817630 (LB 0.922) with ep05 DeepCenter as '
    'a 4th heatmap.\n\nSINGLE VARIABLE vs 55817630: detect() averages '
    'FOUR heatmaps instead of\nthree -- the fourth is the retrained '
    'DeepCenter epoch-5 checkpoint\n(pct50-99.5 normalisation, 4-flip '
    'TTA, same pooled grid), detfuse2\'s F4\narm. WHAT THIS TESTS: '
    'fusescore rejected F4 on the 6bba domain alone\n(-0.0053 there, '
    '+0.0054 44b6, +0.0027 zh) -- but the xl weights\' training\nsplit '
    'is unknown and plausibly contains the 6bba eval sets, while the\n'
    'hidden test is a THIRD embryo (host-confirmed, no embryo_id '
    'overlap), so\n6bba\'s veto may be a memorization artifact. The '
    'leaderboard is the only\narbiter left. PRE-REGISTERED READING: '
    '>0.922 = the 6bba veto was\nmemorization, DC arc reopens; '
    '0.920-0.922 = uninformative-to-mildly-real;\n<=0.918 = the 6bba '
    'negativity is real and the DC-injection line closes at\nthe '
    'official judge. Full 4-flip TTA, no degraded variant -- if this '
    'hits\nthe 12 h cap, that is itself the measurement (user\'s '
    'instruction).\n')

rep('SCRIPT_VERSION = "2026-08-23i-submit-ranker"',
    'SCRIPT_VERSION = "2026-08-30a-submit-f4"')

# ---- DC model + loader, inserted just before the branch table -----------
rep('''_BRANCHES = [([MODELS[0]], [1.0], ""), ([MODELS[1]], [1.0], "tophat"),
             ([_load_model("unet3d_v2_tophat_b32.pt")], [1.0], "tophat")]''',
    '''# ---- F4: ep05 DeepCenter, the fourth heatmap ---------------------------
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
             ([_load_model("unet3d_v2_tophat_b32.pt")], [1.0], "tophat")]''')

# ---- detect(): append the fourth map before the average -----------------
rep('''    h = hs[0] if len(hs) == 1 else np.mean(hs, axis=0)''',
    '''    xd = _dc_normalize(pool_xy(vol))
    accd = None
    for i in _FLIP_VIEWS:
        xv = np.ascontiguousarray(_flip(xd, i))
        with torch.no_grad():
            lg = DC_MODEL(torch.from_numpy(xv)[None, None].to(DEVICE)
                          )[0, 0].float().cpu().numpy()
        lg = _flip(lg, i)
        accd = lg if accd is None else accd + lg
    hs.append(1.0 / (1.0 + np.exp(-np.clip(accd / 4.0, -60.0, 60.0))))
    h = hs[0] if len(hs) == 1 else np.mean(hs, axis=0)''')

DST.write_text(text, encoding="utf-8")
ast.parse(text)
print(f"wrote {DST} ({len(text.splitlines())} lines), ast OK")
