"""Generate submit_w1101.py from submit_f4.py by asserted replacement.

Single variable vs ref 55889500 (F4, LB 0.925): the blend weights.
F4 = (h0+h1+h2+hd)/4; W1101 = (h0+h1+0.5*h2+hd)/3.5 -- branch 3 (the
v2 tophat model, highly correlated with branch 2) demoted to half
weight, the xl-side configuration detfuse2 measured as better than
equal-3 on dense GT. Everything else byte-identical.
"""
import ast
import pathlib

SRC = pathlib.Path(__file__).parent / "submit_f4.py"
DST = pathlib.Path(__file__).parent / "submit_w1101.py"
text = SRC.read_text(encoding="utf-8")


def rep(old, new, n=1):
    global text
    c = text.count(old)
    assert c == n, f"anchor x{c}, wanted x{n}: {old[:70]!r}"
    text = text.replace(old, new)


rep('"""submit_f4.py -- ref 55817630 (LB 0.922) with ep05 DeepCenter as '
    'a 4th heatmap.',
    '"""submit_w1101.py -- ref 55889500 (F4, LB 0.925) with the W1101 '
    'blend weights.\n\nSINGLE VARIABLE vs 55889500: the four-map average '
    'becomes (h0+h1+0.5*h2+hd)/3.5\n-- branch 3 demoted to half weight '
    '(it is a variant of branch 2 and their\nerrors correlate; detfuse2 '
    'measured the demotion as better than equal-3 on\ndense GT). Local '
    'pipeline deltas vs BASE: -0.0108 6bba / +0.0054 44b6 /\n+0.0024 zh '
    '(F4: -0.0053 / +0.0054 / +0.0027) -- identical in the clean\n'
    'domains, twice the loss in the memorization-tainted one. '
    'PRE-REGISTERED\nREADING (user\'s, 2026-08-31): LOWER than F4\'s '
    '0.925 = the 6bba negativity\nhas a REAL component beyond '
    'memorization; EQUAL = pure-artifact reading\nholds; HIGHER = '
    'unpredicted by either hypothesis, uninformative noise.\n')

rep('SCRIPT_VERSION = "2026-08-30a-submit-f4"',
    'SCRIPT_VERSION = "2026-08-31a-submit-w1101"')

rep('    h = hs[0] if len(hs) == 1 else np.mean(hs, axis=0)',
    '    h = (hs[0] + hs[1] + 0.5 * hs[2] + hs[3]) / 3.5')

DST.write_text(text, encoding="utf-8")
ast.parse(text)
print(f"wrote {DST} ({len(text.splitlines())} lines), ast OK")
