"""Step 4: draw the four slide figures from the data and the results of steps 1-3.

  fig1_one_rna.png            RNA A: sequence + structure as an arc diagram
  fig2_stem_list.png          RNA A: its stems, their 3 numbers and their relation
  fig3_pairwise_A_vs_B.png    Finn's Aligner on A vs B: matches, skip, costs (from step 2)
  fig4_multiple_alignment.png all RNAs: the multiple alignment table (from step 3)

Every number drawn is read from CoSTAR's output or results/, not typed in by hand.
Usage: python 04_figures.py [RNA_FOR_FIG1_2] [RNA1_FOR_FIG3] [RNA2_FOR_FIG3]   (default: A A B)
"""
import csv
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from common import (FIGURES, RESULTS, build_stem_lists, fmt, load_costar, read_dbn, relation,
                    relation_sentence, stem_name, stem_pairs)

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRAY, INK, INK2, BG = "#a9a7a0", "#0b0b0b", "#52514e", "#ffffff"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15})


def color(k):
    return PALETTE[k % len(PALETTE)]


def draw_stem(ax, pairs, y0, col, up=True, scale=0.55, ls="-"):
    """One thin arc per base pair; returns the apex of the outermost arc (for the label)."""
    t = np.linspace(0, np.pi, 120)
    sgn = 1 if up else -1
    for i, j in pairs:
        c, r = (i + j) / 2, (j - i) / 2
        ax.plot(c + r * np.cos(t), y0 + sgn * scale * r * np.sin(t), color=col, lw=1.8, ls=ls,
                solid_capstyle="round")
    i, j = pairs[0]
    return (i + j) / 2, y0 + sgn * scale * (j - i) / 2


def baseline(ax, n, y, label):
    ax.plot([0.6, n + 0.4], [y, y], color=INK2, lw=2.2, solid_capstyle="round")
    for p in range(1, n + 1):
        ax.plot([p, p], [y - 0.12, y + 0.12], color=INK2, lw=0.8)
    ax.text(-0.6, y, label, ha="right", va="center", fontsize=17, fontweight="bold", color=INK)


def new_axes(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    ax.set_aspect("equal")
    return ax


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=200, facecolor=BG, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"Saved {FIGURES / name}")


def fig1(rna, seq, struct, d):
    stems = [stem_pairs(d, k) for k in range(d["features"].shape[0])]
    pos_color = {p: color(k) for k, st in enumerate(stems) for pair in st for p in pair}
    fig = plt.figure(figsize=(13, 5.2))
    ax = new_axes(fig, [0, 0, 1, 1])
    y_arc, scale = 0.55, 0.55
    top = 0
    for k, st in enumerate(stems):
        cx, cy = draw_stem(ax, st, y_arc, color(k), scale=scale)
        ax.text(cx, cy + 0.35, stem_name(rna, k), ha="center", va="bottom", fontsize=18,
                fontweight="bold", color=INK)
        top = max(top, cy)
    for p, (ch, b) in enumerate(zip(seq, struct), start=1):
        style = dict(ha="center", va="center", fontsize=17, family="DejaVu Sans Mono",
                     color=pos_color.get(p, INK2), fontweight="bold" if p in pos_color else "normal")
        ax.text(p, 0, ch, **style)
        ax.text(p, -1.0, b, **style)
    n = len(seq)
    for p in sorted({1, n} | set(range(5, n, 5))):
        ax.text(p, -1.9, str(p), ha="center", va="center", fontsize=11, color=INK2)
    for y, lab, fs in [(0, "sequence", 14), (-1.0, "structure", 14), (-1.9, "position", 11)]:
        ax.text(-0.6, y, lab, ha="right", va="center", fontsize=fs, color=INK2)
    # point at the first crossing, if the RNA has one
    for a in range(len(stems)):
        b = next((b for b in range(a + 1, len(stems)) if relation(d, a, b) == "crosses from 5'"), None)
        if b is None:
            continue
        (i1, j1), (i2, _) = stems[a][0], stems[b][0]
        x = (i2 + j1) / 2
        c, r = (i1 + j1) / 2, (j1 - i1) / 2
        y = y_arc + scale * np.sqrt(max(r * r - (x - c) ** 2, 0))
        ax.annotate("the two stems cross\n= pseudoknot", xy=(x, y), xytext=(n + 3.2, top * 0.9),
                    fontsize=14, color=INK, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=1.0))
        break
    ax.set_xlim(-6, n + 9)
    ax.set_ylim(-2.4, top + 1.7)
    save(fig, "fig1_one_rna.png")


def fig2(rna, seq, d):
    k_stems = d["features"].shape[0]
    stems = [stem_pairs(d, k) for k in range(k_stems)]
    n = len(seq)
    fig = plt.figure(figsize=(13, 4.6))
    ax = new_axes(fig, [0.0, 0.0, 0.5, 1.0])
    baseline(ax, n, 0, f"RNA {rna}")
    top = 0
    for k, st in enumerate(stems):
        cx, cy = draw_stem(ax, st, 0.15, color(k))
        ax.text(cx, cy + 0.3, stem_name(rna, k), ha="center", va="bottom", fontsize=18,
                fontweight="bold", color=INK)
        top = max(top, cy)
        for arm in ([i for i, _ in st], [j for _, j in st]):
            ax.add_patch(Rectangle((min(arm) - 0.45, -0.75), max(arm) - min(arm) + 0.9, 0.42,
                                   color=color(k), lw=0))
    ax.text(n / 2 + 0.5, -1.5, "colored blocks = the two arms of each stem", ha="center",
            va="center", fontsize=12, color=INK2)
    ax.set_xlim(-6, n + 2)
    ax.set_ylim(-2.2, top + 1.6)

    tx = fig.add_axes([0.53, 0.0, 0.47, 1.0])
    tx.set_axis_off(); tx.set_xlim(0, 10); tx.set_ylim(0, 10)
    xs = [0.6, 3.3, 5.6, 8.0]
    for x, h in zip(xs, ["stem", "energy", "pairs", "inner\ndistance"]):
        tx.text(x, 8.6, h, ha="center", va="center", fontsize=15, color=INK2, fontweight="bold")
    tx.plot([0.0, 9.6], [7.6, 7.6], color=INK2, lw=1.0)
    step = min(1.5, 4.5 / max(k_stems, 1))
    y = 7.6
    for k in range(k_stems):
        y = 6.6 - step * k
        energy, pairs, inner = d["features"][k]
        tx.add_patch(Rectangle((0.0, y - 0.35), 0.35, 0.7, color=color(k), lw=0))
        tx.text(0.9, y, stem_name(rna, k), ha="center", va="center", fontsize=17, fontweight="bold", color=INK)
        for x, v in zip(xs[1:], [f"{energy:.2f}", fmt(pairs), fmt(inner)]):
            tx.text(x, y, v, ha="center", va="center", fontsize=17, color=INK)
    y_line = y - 0.9
    tx.plot([0.0, 9.6], [y_line, y_line], color=INK2, lw=1.0)
    sentences = [relation_sentence(stem_name(rna, a), stem_name(rna, b), relation(d, a, b))
                 for a in range(k_stems) for b in range(a + 1, k_stems)]
    y = y_line - 1.0
    for s in sentences:
        tx.text(0.0, y, "relation:  " + s, ha="left", va="center", fontsize=15, color=INK)
        y -= 1.0
    tx.text(0.0, y - 0.4, f"{n} letters  →  {k_stems} stems × 3 numbers + "
            f"{len(sentences)} relation{'s' if len(sentences) != 1 else ''}",
            ha="left", va="center", fontsize=15, color=INK2)
    save(fig, "fig2_stem_list.png")


def fig3(x, y, rnas, se, record):
    decisions = record["decisions"]
    matches = [dd for dd in decisions if dd["i"] >= 0 and dd["j"] >= 0]
    col_x = {dd["i"]: color(m) for m, dd in enumerate(sorted(matches, key=lambda dd: dd["i"]))}
    col_y = {dd["j"]: col_x[dd["i"]] for dd in matches}
    fig = plt.figure(figsize=(13, 6.4))
    ax = new_axes(fig, [0.0, 0.0, 0.72, 1.0])
    yx, yy = 0.0, -2.6
    nx_, ny_ = len(rnas[x][0]), len(rnas[y][0])
    baseline(ax, nx_, yx, f"RNA {x}")
    baseline(ax, ny_, yy, f"RNA {y}")
    top, bottom = 0, yy
    for k in range(se[x]["features"].shape[0]):
        skipped = k not in col_x
        cx, cy = draw_stem(ax, stem_pairs(se[x], k), yx + 0.15, col_x.get(k, GRAY), up=True,
                           ls="--" if skipped else "-")
        ax.text(cx, cy + 0.3, stem_name(x, k) + ("  (skipped)" if skipped else ""), ha="center",
                va="bottom", fontsize=18, fontweight="bold", color=INK)
        top = max(top, cy)
    for k in range(se[y]["features"].shape[0]):
        skipped = k not in col_y
        cx, cy = draw_stem(ax, stem_pairs(se[y], k), yy - 0.15, col_y.get(k, GRAY), up=False,
                           ls="--" if skipped else "-")
        ax.text(cx, cy - 0.3, stem_name(y, k), ha="center", va="top", fontsize=18,
                fontweight="bold", color=INK)
        if skipped:
            ax.text(cx, cy - 2.0, f"no partner in {x}\n→ skipped", ha="center", va="top",
                    fontsize=13, color=INK2)
        bottom = min(bottom, cy)
    ax.set_xlim(-6, max(nx_, ny_) + 2)
    ax.set_ylim(bottom - 2.3, top + 1.7)

    bx = fig.add_axes([0.74, 0.12, 0.26, 0.76])
    bx.set_axis_off(); bx.set_xlim(0, 10); bx.set_ylim(0, 10)
    bx.text(0, 9.3, "matching cost", ha="left", va="center", fontsize=16, fontweight="bold", color=INK)
    step = min(1.5, 4.5 / max(len(decisions), 1))
    yrow = 7.6
    for m, dd in enumerate(decisions):
        yrow = 7.6 - step * m
        if dd["i"] >= 0 and dd["j"] >= 0:
            chip, label = col_x[dd["i"]], f"{dd['stem1']} ↔ {dd['stem2']}"
        else:
            chip, label = GRAY, f"{dd['stem1'] if dd['stem1'] != '-' else dd['stem2']} skipped"
        bx.add_patch(Rectangle((0, yrow - 0.3), 0.45, 0.6, color=chip, lw=0))
        bx.text(0.8, yrow, label, ha="left", va="center", fontsize=15, color=INK)
        bx.text(9.6, yrow, f"{dd['cost']:.2f}", ha="right", va="center", fontsize=15, color=INK)
    yl = yrow - 0.9
    bx.plot([0, 9.6], [yl, yl], color=INK2, lw=1.0)
    bx.text(0.8, yl - 0.8, "total", ha="left", va="center", fontsize=15, fontweight="bold", color=INK)
    bx.text(9.6, yl - 0.8, f"{record['cost']:.2f}", ha="right", va="center", fontsize=15,
            fontweight="bold", color=INK)
    bx.text(0, yl - 2.5, "lowest possible cost\nfor this pair (exact)", ha="left", va="center",
            fontsize=13, color=INK2)
    save(fig, f"fig3_pairwise_{x}_vs_{y}.png")


def fig4(names, rnas, se, header, table):
    col_of = {x: {} for x in names}               # stem index -> column index
    for x in names:
        for c, v in enumerate(table[x]):
            if v != "-":
                col_of[x][int(v[len(x):]) - 1] = c
    max_len = max(len(rnas[x][0]) for x in names)
    max_r = max((j - i) / 2 for x in names for k in range(se[x]["features"].shape[0])
                for i, j in stem_pairs(se[x], k)[:1])
    gap = 0.33 * max_r + 3.0
    ys = {x: gap * (len(names) - 1 - r) for r, x in enumerate(names)}
    fig = plt.figure(figsize=(13, 2.5 * len(names) + 0.1))
    ax = new_axes(fig, [0, 0, 1, 1])
    for x in names:
        baseline(ax, len(rnas[x][0]), ys[x], f"RNA {x}")
        for k in range(se[x]["features"].shape[0]):
            c = col_of[x].get(k)
            cx, cy = draw_stem(ax, stem_pairs(se[x], k), ys[x] + 0.15,
                               color(c) if c is not None else GRAY, scale=0.33)
            ax.text(cx, cy + 0.2, stem_name(x, k), ha="center", va="bottom", fontsize=15,
                    fontweight="bold", color=INK)
    x0 = max_len + 6.5
    gx = [x0 + 3.5 + 6.5 * c for c in range(len(header))]
    hy = ys[names[0]] + gap - 0.9
    for c, (x, h) in enumerate(zip(gx, header)):
        ax.add_patch(Rectangle((x - 2.6, hy - 0.45), 0.9, 0.9, color=color(c), lw=0))
        ax.text(x - 1.4, hy, h.replace("col", "col "), ha="left", va="center", fontsize=15,
                fontweight="bold", color=INK)
    ax.plot([x0, gx[-1] + 2.5], [hy - 1.4, hy - 1.4], color=INK2, lw=1.0)
    for x in names:
        ax.text(x0, ys[x], x, ha="center", va="center", fontsize=16, fontweight="bold", color=INK2)
        for gxx, v in zip(gx, table[x]):
            dash = v == "-"
            ax.text(gxx, ys[x], "–" if dash else v, ha="center", va="center", fontsize=17,
                    color=INK2 if dash else INK, fontweight="normal" if dash else "bold")
    ax.text((x0 + gx[-1]) / 2 + 1.5, -2.4, "– = skipped in that RNA", ha="center",
            va="center", fontsize=13, color=INK2)
    ax.set_xlim(-6, gx[-1] + 4)
    ax.set_ylim(-3.4, hy + 1.6)
    save(fig, "fig4_multiple_alignment.png")


def main():
    create_SEdict, _, _ = load_costar()
    rnas = read_dbn()
    se = build_stem_lists(create_SEdict, rnas)
    names = list(rnas)
    if len(sys.argv) >= 4:
        one, x, y = sys.argv[1:4]
    else:
        one, x, y = names[0], names[0], names[1]

    fig1(one, *rnas[one], se[one])
    fig2(one, rnas[one][0], se[one])

    pairs = json.load(open(RESULTS / "pairwise.json"))["pairs"]
    record = next(p for p in pairs if {p["rna1"], p["rna2"]} == {x, y})
    x, y = record["rna1"], record["rna2"]        # keep the order used in step 2
    fig3(x, y, rnas, se, record)

    with open(RESULTS / "msa_table.tsv") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    header, table = rows[0][1:], {r[0]: r[1:] for r in rows[1:]}
    fig4(names, rnas, se, header, table)


if __name__ == "__main__":
    main()
