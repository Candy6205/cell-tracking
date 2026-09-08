"""submit_gate.py -> submit_joint.py: the SAME pipeline and the SAME cut, with
the fork gate swapped for the one trained on both imaging batches.

Nothing else changes -- not the detector, not the ranker, not the ILP, not the
patch, not the 20 features, not the 0.03 threshold. The only difference between
this kernel and the one that scored 0.943 at LB is which pickle is loaded, and
the only difference between the two pickles is that this one saw 67 more
labelled forks, from 44b6's 71 movies.

WHY IT IS WORTH A SUBMISSION, in the unit the 2026-08-28 rule asks for -- a
mechanism, not a local number. The shipped gate learns to rank forks from 299
counted examples with 50 positives, all from one imaging batch. divbench8
measured, under real pruning and the official scorer:

  out-of-fold AUC        0.9283 / 0.8326  ->  0.9419 / 0.8506   (both folds)
  ranking at each gate's
  own best threshold      div_J 0.2531    ->  0.2848            (+0.0317)
  forks kept / cut        42 TP, 47 FP    ->  46 TP, 39 FP

that is four MORE true divisions kept while eight MORE false forks are cut --
both sides of the ledger at once, which almost nothing on this line has done.
And the two null controls come back negative: the same 67 rows with their
labels SHUFFLED score -0.0030 against the shipped gate, and 67 rows
bootstrapped from the training fold itself -0.0017. So the gain is label
information from the other batch, not extra rows and not regularisation.

Local, 128 6bba movies, official scorer: score 0.9387 -> 0.9424, +0.0037, both
folds up (+0.0028 / +0.0047), paired bootstrap over movies 90%
[+0.0016, +0.0060]. adj_J is UNCHANGED at 0.9143, so the whole move is the
division term and nothing is being traded away for it.

THE PKL IS THE ONE FITTED ON KAGGLE by divbench8, because local scikit-learn
1.9.0 reproduces the ranking but not the calibration and the cut is a fixed
0.03. Its quantiles over the bench's 7,004 forks are 80% 0.0321 against the
shipped gate's 0.0275; at p > 0.03 it keeps 20.4% of forks against 19.5%, so
0.03 is nearly the same operating point in both and the cut does not move.

ATTACH `faustine649/biohub-divbench8` AS A KERNEL SOURCE. divbench3 may stay
attached -- the two pickles have different filenames, so the /kaggle/input walk
cannot pick up the wrong one.
"""
import os

src = open("submit_gate.py", encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:80])
    src = src.replace(old, new)


rep('SCRIPT_VERSION = "2026-09-06a-submit-gate"',
    'SCRIPT_VERSION = "2026-09-08a-submit-joint"')

rep('GATE_NAME = "fork_gate_f4.pkl"', 'GATE_NAME = "fork_gate_joint.pkl"')

rep('GATE_THR = 0.03            # divbench3: p > 0.03 = +0.0121 local score, '
    'both folds up, 44b6 +0.0020',
    'GATE_THR = 0.03            # UNCHANGED from divbench3. divbench8: this gate at the '
    'same cut is\n'
    '                           # +0.0037 over the shipped one, both folds, 90% '
    '[+0.0016, +0.0060]')

rep('''assert _gk is not None, (f"{GATE_NAME} not found under /kaggle/input -- attach "
                         f"faustine649/biohub-divbench3 as a kernel source")''',
    '''assert _gk is not None, (f"{GATE_NAME} not found under /kaggle/input -- attach "
                         f"faustine649/biohub-divbench8 as a kernel source")''')

# The joint gate is the ONLY change in this kernel, so refuse to run on the
# wrong pickle rather than silently shipping the incumbent under a new name.
rep('''assert _gb.get("detector") == "F4", _gb.get("detector")
GATE = _gb["model"]''',
    '''assert _gb.get("detector") == "F4", _gb.get("detector")
_nt = _gb.get("n_train") or {}
assert _nt.get("6bba") == 299 and _nt.get("44b6") == 67 and _nt.get("tp") == 56, (
    f"fork_gate_joint.pkl was trained on {_nt}, not divbench8's 299 6bba + 67 44b6 "
    f"counted forks with 56 TP -- this is not the gate the +0.0037 was measured on")
GATE = _gb["model"]''')

rep('''print(f"fork gate {GATE_NAME} from {_gk} | version {_gb.get('version')} | "
      f"trained on {_gb.get('trained_on')} | threshold p > {GATE_THR} | "
      f"its p-quantiles over the bench's forks {_gb.get('p_quantiles_all_forks')}", flush=True)''',
    '''print(f"fork gate {GATE_NAME} from {_gk} | version {_gb.get('version')} | "
      f"trained on {_gb.get('trained_on')} | n_train {_nt} | threshold p > {GATE_THR} | "
      f"its p-quantiles over the bench's forks {_gb.get('p_quantiles_all_forks')}", flush=True)
print("  (divbench8, 2026-09-08: this gate against the shipped 6bba-only one, on the "
      "128 6bba train movies with real pruning and the official scorer -- 46/39/79 "
      "against 42/47/83, div_J 0.2805 against 0.2442, adj_J 0.9143 both, score 0.9424 "
      "against 0.9387, both folds up, paired 90% [+0.0016, +0.0060]. Controls: the "
      "same 67 rows label-shuffled -0.0030, 67 bootstrapped 6bba rows -0.0017.)",
      flush=True)''')

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "submit_joint.py")
open(os.path.abspath(out), "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
