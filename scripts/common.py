"""Shared helpers: folder paths, reading the toy data, and importing Finn's CoSTAR code."""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = Path(os.environ.get("TOY_DATA", ROOT / "data" / "toy_rnas.dbn"))  # override: TOY_DATA=...
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Where Finn's repository is. Default: <project>/external/CoSTAR (override with COSTAR_DIR).
COSTAR_DIR = Path(os.environ.get("COSTAR_DIR", ROOT / "external" / "CoSTAR"))

# Column order of CoSTAR's per-stem feature array (see CoSTAR/SEdict.py, _calc_features).
FEATURE_NAMES = ["energy", "pairs", "inner_distance"]

# CoSTAR's relation array R[a, b, k] is True when stem a sits relative to stem b as label k
# (see CoSTAR/SEdict.py, _calc_relation_array).
RELATION_NAMES = ["5' of", "crosses from 5'", "inside", "3' of", "crosses from 3'", "contains"]


def load_costar():
    """Make Finn's code importable and return the three things we use from it."""
    if not (COSTAR_DIR / "CoSTAR" / "align.py").exists():
        sys.exit(
            f"CoSTAR code not found in {COSTAR_DIR}\n"
            f"Clone it first:\n  git clone https://github.com/TheCOBRALab/CoSTAR {ROOT / 'external' / 'CoSTAR'}\n"
            "or set COSTAR_DIR to an existing copy."
        )
    sys.path.insert(0, str(COSTAR_DIR))
    from CoSTAR.SEdict import create_SEdict
    from CoSTAR.align import Aligner, clear_tree
    return create_SEdict, Aligner, clear_tree


def read_dbn(path=DATA_FILE):
    """Read a dot-bracket file: '>name' line, then sequence, then structure. '#' lines are comments."""
    lines = [l.strip() for l in open(path) if l.strip() and not l.lstrip().startswith("#")]
    rnas = {}
    for k in range(0, len(lines), 3):
        name = lines[k].lstrip(">").strip()
        seq, struct = lines[k + 1], lines[k + 2]
        if len(seq) != len(struct):
            sys.exit(f"{name}: sequence ({len(seq)}) and structure ({len(struct)}) differ in length")
        rnas[name] = (seq, struct)
    return rnas


def build_stem_lists(create_SEdict, rnas):
    """Run CoSTAR's preprocessing on every RNA. Returns {name: SE_dict}."""
    return {name: create_SEdict(seq, struct) for name, (seq, struct) in rnas.items()}


def stem_name(rna, k):
    """Stem k (0-based, in 5' order) of RNA 'A' is called a1, a2, ..."""
    return f"{rna.lower()}{k + 1}"


def stem_pairs(se_dict, k):
    """1-based base pairs (i, j) of stem k, read from CoSTAR's own per-stem layer."""
    rows, cols = np.nonzero(se_dict["layers"][k])
    return sorted((int(min(r, c)) + 1, int(max(r, c)) + 1) for r, c in zip(rows, cols))


def stem_arms(se_dict, k):
    """('1-4', '17-20'): the 5' arm and 3' arm positions of stem k."""
    pairs = stem_pairs(se_dict, k)
    left = [i for i, _ in pairs]
    right = [j for _, j in pairs]
    return f"{min(left)}-{max(left)}", f"{min(right)}-{max(right)}"


def relation(se_dict, a, b):
    """How stem a sits relative to stem b, as one of RELATION_NAMES."""
    hits = np.nonzero(se_dict["relations"][a, b])[0]
    return RELATION_NAMES[hits[0]] if len(hits) else "none"


def relation_sentence(name_a, name_b, label):
    return {
        "5' of": f"{name_a} lies entirely 5' of {name_b}",
        "crosses from 5'": f"{name_a} crosses {name_b} ({name_a} opens first)",
        "inside": f"{name_a} is inside {name_b}",
        "3' of": f"{name_a} lies entirely 3' of {name_b}",
        "crosses from 3'": f"{name_a} crosses {name_b} ({name_b} opens first)",
        "contains": f"{name_a} contains {name_b}",
    }.get(label, f"{name_a} ? {name_b}")


def match_cost(fx, fy):
    """Cost of matching two stems: sum of absolute feature differences (CoSTAR default weights = 1)."""
    return float(np.abs(np.asarray(fx) - np.asarray(fy)).sum())


def skip_cost(f):
    """Cost of skipping a stem: the sum of its own features (its distance to an 'empty' stem)."""
    return float(np.asarray(f).sum())


def fmt(x):
    """Print 4.0 as '4' and 6.48 as '6.48'."""
    return f"{x:.0f}" if float(x).is_integer() else f"{x:.2f}"
