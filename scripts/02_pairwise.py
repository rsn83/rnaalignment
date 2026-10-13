"""Step 2: align every pair of RNAs with Finn's Aligner (exact, two RNAs at a time).

For each pair it reports which stems were matched, which were skipped, and the cost:
  match cost = sum of |feature differences|      skip cost = sum of the stem's own features
It also re-adds the cost by hand and checks it equals the Aligner's reported cost.

Writes results/pairwise.json (matches, used by step 3) and results/pairwise_costs.tsv.
"""
import csv
import itertools
import json

import numpy as np

from common import (RESULTS, build_stem_lists, fmt, load_costar, match_cost, read_dbn,
                    skip_cost, stem_name)


def explain(x, y, fx, fy, a_alignment):
    """List every decision in the alignment with its cost and the arithmetic behind it."""
    rows = []
    matched_y = set()
    for i, j in enumerate(a_alignment):
        if j >= 0:
            matched_y.add(int(j))
            terms = " + ".join(f"|{fmt(a)}-{fmt(b)}|" for a, b in zip(fx[i], fy[j]))
            rows.append({"stem1": stem_name(x, i), "stem2": stem_name(y, int(j)), "i": i, "j": int(j),
                         "cost": match_cost(fx[i], fy[j]), "math": terms})
        else:
            rows.append({"stem1": stem_name(x, i), "stem2": "-", "i": i, "j": -1,
                         "cost": skip_cost(fx[i]), "math": " + ".join(fmt(v) for v in fx[i])})
    for j in range(len(fy)):
        if j not in matched_y:
            rows.append({"stem1": "-", "stem2": stem_name(y, j), "i": -1, "j": j,
                         "cost": skip_cost(fy[j]), "math": " + ".join(fmt(v) for v in fy[j])})
    return rows


def main():
    create_SEdict, Aligner, clear_tree = load_costar()
    rnas = read_dbn()
    se = build_stem_lists(create_SEdict, rnas)
    names = list(rnas)
    RESULTS.mkdir(exist_ok=True)

    out = {"pairs": []}
    cost = {x: {y: 0.0 for y in names} for x in names}
    for x, y in itertools.combinations(names, 2):
        aligner = Aligner(se[x], se[y])          # Finn's code, default weights [1, 1, 1]
        optimal = aligner.align()                # every minimum-cost alignment
        clear_tree(aligner)
        best = optimal[0]
        rows = explain(x, y, se[x]["features"], se[y]["features"], best["a_alignment"])
        total = sum(r["cost"] for r in rows)
        reported = float(best["alignment_cost"])
        assert np.isclose(total, reported), f"{x} vs {y}: hand total {total} != Aligner {reported}"

        print(f"\n{x} vs {y}: total cost {total:.2f}  ({len(optimal)} optimal alignment(s))")
        for r in rows:
            kind = "match" if r["i"] >= 0 and r["j"] >= 0 else "skip "
            print(f"  {kind}  {r['stem1']:>4} <-> {r['stem2']:<4} {r['cost']:6.2f}   = {r['math']}")
        cost[x][y] = cost[y][x] = total
        out["pairs"].append({"rna1": x, "rna2": y, "cost": round(total, 6),
                             "n_optimal": len(optimal), "decisions": rows})

    with open(RESULTS / "pairwise.json", "w") as f:
        json.dump(out, f, indent=2)
    with open(RESULTS / "pairwise_costs.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([""] + names)
        for x in names:
            w.writerow([x] + [f"{cost[x][y]:.2f}" for y in names])
    print(f"\nSaved {RESULTS / 'pairwise.json'} and {RESULTS / 'pairwise_costs.tsv'}")


if __name__ == "__main__":
    main()
