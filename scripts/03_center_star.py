"""Step 3: combine the pairwise results into one multiple alignment of stems (center-star).

Center-star (Gusfield 1993), applied to stems instead of letters:
  1. center = the RNA with the lowest total pairwise cost to all the others
  2. every other RNA was already aligned to the center in step 2;
     stems matched to the same center stem go into the same column
  3. a stem matched to nothing in the center gets its own new column, kept in 5'->3' order
Then it checks the topology rule (two columns must relate the same way in every RNA that has
both) and compares the total cost with a lower bound on the best possible total.

Writes results/msa_table.tsv and results/msa.json.
"""
import csv
import itertools
import json

from common import (RESULTS, build_stem_lists, load_costar, match_cost, read_dbn, relation,
                    skip_cost, stem_name)


def load_pairwise():
    data = json.load(open(RESULTS / "pairwise.json"))
    cost, maps = {}, {}
    for p in data["pairs"]:
        x, y = p["rna1"], p["rna2"]
        cost[(x, y)] = cost[(y, x)] = p["cost"]
        m = {d["i"]: d["j"] for d in p["decisions"] if d["i"] >= 0 and d["j"] >= 0}
        maps[(x, y)] = m                                  # stem index in x -> stem index in y
        maps[(y, x)] = {j: i for i, j in m.items()}
    return cost, maps


def center_star(names, n_stems, cost, maps):
    totals = {x: sum(cost[(x, y)] for y in names if y != x) for x in names}
    center = min(names, key=lambda x: totals[x])
    columns = [{center: k} for k in range(n_stems[center])]  # one column per center stem
    for x in names:
        if x == center:
            continue
        to_center = maps[(x, center)]
        pointer = -1                                       # last column used by RNA x
        for k in range(n_stems[x]):                        # x's stems in 5'->3' order
            if k in to_center:                             # matched: join the center stem's column
                pointer = next(i for i, col in enumerate(columns) if col.get(center) == to_center[k])
                columns[pointer][x] = k
            else:                                          # unmatched: new column right after
                columns.insert(pointer + 1, {x: k})
                pointer += 1
    return center, totals, columns


def topology_problems(names, se, columns):
    """Column pairs where RNAs disagree on how the two stems sit (should be none)."""
    problems = []
    for p, q in itertools.combinations(range(len(columns)), 2):
        labels = {x: relation(se[x], columns[p][x], columns[q][x])
                  for x in names if x in columns[p] and x in columns[q]}
        if len(set(labels.values())) > 1:
            problems.append((p + 1, q + 1, labels))
    return problems


def induced_cost(x, y, se, columns):
    """Cost of the x-vs-y alignment that the table implies (same match/skip costs as step 2)."""
    fx, fy = se[x]["features"], se[y]["features"]
    total = 0.0
    for col in columns:
        if x in col and y in col:
            total += match_cost(fx[col[x]], fy[col[y]])
        elif x in col:
            total += skip_cost(fx[col[x]])
        elif y in col:
            total += skip_cost(fy[col[y]])
    return total


def main():
    create_SEdict, _, _ = load_costar()
    rnas = read_dbn()
    se = build_stem_lists(create_SEdict, rnas)
    names = list(rnas)
    n_stems = {x: se[x]["features"].shape[0] for x in names}
    cost, maps = load_pairwise()

    center, totals, columns = center_star(names, n_stems, cost, maps)
    print("Total pairwise cost of each RNA to all others:")
    for x in names:
        print(f"  {x}: {totals[x]:.2f}" + ("   <- lowest, so this is the center" if x == center else ""))

    header = [f"col{c + 1}" for c in range(len(columns))]
    table = {x: [stem_name(x, col[x]) if x in col else "-" for col in columns] for x in names}
    print("\nMultiple alignment of stems (- = skipped in that RNA):")
    print("        " + "".join(f"{h:<7}" for h in header))
    for x in names:
        print(f"  RNA {x} " + "".join(f"{v:<7}" for v in table[x]))

    problems = topology_problems(names, se, columns)
    if problems:
        print("\nTopology check: FAILED")
        for p, q, labels in problems:
            print(f"  col{p} vs col{q}: {labels}")
    else:
        print("\nTopology check: OK (every pair of columns relates the same way in all RNAs)")

    sp = sum(induced_cost(x, y, se, columns) for x, y in itertools.combinations(names, 2))
    lower = sum(cost[(x, y)] for x, y in itertools.combinations(names, 2))
    n = len(names)
    print(f"\nTotal cost of the table over all RNA pairs: {sp:.2f}")
    print(f"No table can cost less than the sum of the pairwise optima: {lower:.2f}")
    print(f"Ratio {sp / lower:.3f}; center-star guarantees at most {2 - 2 / n:.3f} (Gusfield)")

    with open(RESULTS / "msa_table.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["rna"] + header)
        for x in names:
            w.writerow([x] + table[x])
    with open(RESULTS / "msa.json", "w") as f:
        json.dump({"center": center, "totals": totals, "columns": columns,
                   "total_cost": sp, "lower_bound": lower}, f, indent=2)
    print(f"\nSaved {RESULTS / 'msa_table.tsv'} and {RESULTS / 'msa.json'}")


if __name__ == "__main__":
    main()
