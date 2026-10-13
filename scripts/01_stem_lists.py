"""Step 1: turn each RNA's structure into CoSTAR's stem list.

For every RNA in data/toy_rnas.dbn this runs Finn's preprocessing (create_SEdict) and reports:
  - each stem: its two arms and its 3 numbers (energy, pairs, inner distance)
  - each pair of stems: how they sit (before / after / inside / contains / cross)

Writes results/stem_lists.tsv and results/stem_relations.tsv.
"""
import csv

from common import (FEATURE_NAMES, RESULTS, build_stem_lists, fmt, load_costar, read_dbn,
                    relation, relation_sentence, stem_arms, stem_name)


def main():
    create_SEdict, _, _ = load_costar()
    rnas = read_dbn()
    se = build_stem_lists(create_SEdict, rnas)
    RESULTS.mkdir(exist_ok=True)

    stem_rows, rel_rows = [], []
    for rna, d in se.items():
        seq, struct = rnas[rna]
        n = d["features"].shape[0]
        plural = "s" if n != 1 else ""
        print(f"\nRNA {rna}  ({len(seq)} letters -> {n} stem{plural})")
        print(f"  {seq}\n  {struct}")
        for k in range(n):
            five, three = stem_arms(d, k)
            e, p, dist = d["features"][k]
            name = stem_name(rna, k)
            print(f"  {name}: 5' arm {five} pairs with 3' arm {three} | "
                  f"energy {fmt(e)} | pairs {fmt(p)} | inner distance {fmt(dist)}")
            stem_rows.append([rna, name, five, three, fmt(e), fmt(p), fmt(dist)])
        for a in range(n):
            for b in range(a + 1, n):
                label = relation(d, a, b)
                print("  " + relation_sentence(stem_name(rna, a), stem_name(rna, b), label))
                rel_rows.append([rna, stem_name(rna, a), stem_name(rna, b), label])

    with open(RESULTS / "stem_lists.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["rna", "stem", "five_prime_arm", "three_prime_arm"] + FEATURE_NAMES)
        w.writerows(stem_rows)
    with open(RESULTS / "stem_relations.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["rna", "stem_a", "stem_b", "relation_of_a_to_b"])
        w.writerows(rel_rows)
    print(f"\nSaved {RESULTS / 'stem_lists.tsv'} and {RESULTS / 'stem_relations.tsv'}")


if __name__ == "__main__":
    main()
