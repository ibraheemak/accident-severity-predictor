#!/usr/bin/env python3
"""Figure for the baseline comparison.

  baselines_accuracy.png   The budget-knobbed policies -- best fixed subset
                           and static priority -- plus the no-acquisition and
                           full-acquisition anchors. You set a time budget and
                           get whatever accuracy it buys, so these are matched
                           to the agent's acquisition time and compared on
                           ACCURACY.

Lambdas where the agent is DOMINATED by another lambda on its own curve -- some
other setting gives both higher accuracy and lower time -- are drawn hollow.
Nobody would deploy at one, and the test is computed from the agent's curve
alone, before any baseline is consulted, so it is a deployment argument rather
than a filter for flattering rows.

  threshold_time.png       The confidence threshold. Its knob tau IS an
                           accuracy target, so holding its time fixed and
                           comparing accuracy would compare a number we chose
                           against one the agent earned. Instead tau is set so
                           it matches the agent's accuracy, and the two are
                           compared on TIME.

Caveat on static priority, which the accuracy figure cannot show: its value
there is a step function, because its prefixes cost 0.23 s, then 1.23 s, then
3.23 s and nothing in between. baselines.csv carries static_matched_seconds,
the accuracy-matched reading, which does not have that sensitivity.

Both use lambda as the x axis, so the two read against each other.

Reads docs/figures/baselines/baselines.csv, written by baselines.py.

Usage (from the repo root):
    python CODE/training/agent/analysis/plot_baselines.py
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#26241f", "#6b6862", "#898781", "#e8e6e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
# Reference rules. A fourth hue, well clear of the three policy colours (blue
# ~210, orange ~20, aqua ~160, this ~265), so the anchors read as their own kind
# of thing rather than as a fourth policy. Kept dotted and thin: they are
# context, and a reference line that competes with the data is a distraction.
ANCHOR = "#7a5bb5"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9.5,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
})

LAM_LABEL = "$\\lambda$  (accuracy points charged per second of acquisition)"


def setup(t, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_xticks(range(len(t)))
    ax.set_xticklabels([f"{v:g}" for v in t.lam])
    ax.set_xlabel(LAM_LABEL)
    return fig, ax


def line(ax, x, y, color, label, dashed=False):
    ax.plot(list(x), y, "--" if dashed else "-", marker="o", color=color,
            lw=1.7 if dashed else 2, dashes=(4, 2.5) if dashed else (None, None),
            ms=6, mew=2, mec=SURFACE, zorder=4 if dashed else 3, label=label)


def accuracy_figure(t, out, label=""):
    """Time-matched: same acquisition budget, whose accuracy is higher.

    Lambda on a SYMLOG axis rather than a categorical one. The nineteen values
    span three orders of magnitude and are not evenly spaced -- 0.01 to 0.02 is
    a tenth of the step from 0.1 to 0.2 -- so drawing them at equal intervals
    crowds the left half, compresses the right, and makes the slope between any
    two points meaningless. Symlog spaces them by their actual separation and,
    unlike a plain log scale, can place lambda=0: it is linear below the
    threshold and logarithmic above it.
    """
    t = t.sort_values("lam").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.grid(True, zorder=0); ax.set_axisbelow(True); ax.tick_params(length=0)
    x = t.lam

    # The two anchors are constants -- no knob at all, so they are rules rather
    # than series, labelled in place beside their own line. Each sits at the end
    # the curves have left free: they start near the ceiling at lambda=0 and end
    # near the floor at high lambda, so the ceiling is labelled on the right and
    # the floor on the left.
    full, floor = t.full_acquisition.iloc[0], t.no_acquisition.iloc[0]
    ax.axhline(full, color=ANCHOR, lw=1.6, ls=(0, (6, 3)), zorder=2)
    ax.axhline(floor, color=ANCHOR, lw=1.6, ls=(0, (6, 3)), zorder=2)
    ax.text(0.995, full + 0.003,
            f"full acquisition, {t.full_acquisition_seconds.iloc[0]:g} s",
            transform=ax.get_yaxis_transform(), fontsize=8, color=ANCHOR,
            ha="right")
    ax.text(0.005, floor + 0.003, "no acquisition, timestamp only",
            transform=ax.get_yaxis_transform(), fontsize=8, color=ANCHOR,
            ha="left")

    # Static priority belongs here: it is budget-knobbed like the best fixed
    # subset, so accuracy at a matched time is the comparison that fits it.
    # Read it knowing the value is a STEP function -- its prefixes cost 0.23 s,
    # then 1.23 s, then 3.23 s, and nothing in between, so where the agent's
    # mean time falls relative to a step sets which prefix it is measured
    # against. A 0.28 s change in the agent between runs moved this line by
    # 0.026 at lambda=0.02. The accuracy-matched view in the other figure does
    # not have that sensitivity; the two should be read together.
    line(ax, x, t.best_fixed, BLUE, "Best fixed subset")
    line(ax, x, t.static, ORANGE, "Static priority", dashed=True)
    # grey, because this one is the null model: the same budget spent without
    # choosing. It is a reference the others are read against rather than a
    # fourth competitor, and grey keeps it from claiming a hue of its own.
    ax.plot(list(x), t.random, "-", color=MUTED, lw=1.5, zorder=2,
            label="Random acquisition")
    line(ax, x, t.agent, AQUA, "RL agent")

    # linear below 0.01 so lambda=0 has a place, logarithmic above it
    ax.set_xscale("symlog", linthresh=0.01, linscale=0.6)
    # ticks and limit follow the data, so a 6-lambda run does not get a decade
    # of empty axis drawn for a 19-lambda one
    hi = float(t.lam.max())
    ticks = [v for v in (0, 0.01, 0.1, 1, 10) if v <= hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{v:g}" for v in ticks])   # plain, not 10^-2
    ax.set_xlim(-0.0015, hi * 1.5)
    ax.set_xlabel(LAM_LABEL)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy of each policy at the agent's acquisition time" + label,
                 fontsize=11.5, color=INK, pad=10, loc="left")
    # raised clear of the floor label, which sits on the bottom left
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.01, 0.09),
              fontsize=9, handlelength=2.6, labelspacing=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def regime_figure(t, out, cut=1.0, label=""):
    """All lambdas, with the operationally relevant ones shaded.

    The agent's standing against the baselines depends strongly on where on the
    curve you look, so the honest way to focus on one region is to SHOW the
    whole sweep and mark the region, rather than to plot a subset. Dropping the
    other lambdas would be indistinguishable from choosing them by outcome --
    on this data, a subset that removes six of the seven losses takes the agent
    from 11/19 ahead to 10/12, with no change to the agent at all.

    `cut` is an acquisition-time budget in seconds, applied as a property of the
    deployment rather than of the results: 1.0 s is the standard interface
    responsiveness figure (the limit for a user's flow of thought to stay
    uninterrupted). Anything stricter is defensible too but needs its own
    argument, and that argument has to be made before looking at which side of
    the line the agent does better on.
    """
    lams = list(t.lam)
    fig, ax = setup(t, (7.6, 4.6))
    x = range(len(t))

    inside = [i for i, s_ in enumerate(t.agent_seconds) if s_ < cut]
    if inside:
        ax.axvspan(min(inside) - 0.5, len(t) - 0.5, color=MUTED, alpha=0.07,
                   lw=0, zorder=1)
        # clear of the full-acquisition rule, which sits near the top
        ax.text(min(inside) - 0.35, 0.90,
                f"acquisition under {cut:g} s", transform=ax.get_xaxis_transform(),
                fontsize=8, color=MUTED, ha="left", va="top")

    ax.axhline(t.full_acquisition.iloc[0], color=ANCHOR, lw=1.6, ls=(0, (6, 3)), zorder=2)
    ax.axhline(t.no_acquisition.iloc[0], color=ANCHOR, lw=1.6, ls=(0, (6, 3)), zorder=2)
    line(ax, x, t.best_fixed, BLUE, "Best fixed subset")
    line(ax, x, t.static, ORANGE, "Static priority", dashed=True)
    # grey, because this one is the null model: the same budget spent without
    # choosing. It is a reference the others are read against rather than a
    # fourth competitor, and grey keeps it from claiming a hue of its own.
    ax.plot(list(x), t.random, "-", color=MUTED, lw=1.5, zorder=2,
            label="Random acquisition")
    line(ax, x, t.agent, AQUA, "RL agent")

    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy of each policy across the full range of cost settings"
                 + label, fontsize=11.5, color=INK, pad=10, loc="left")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.01, 0.06),
              fontsize=9, handlelength=3.0, labelspacing=0.55)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    print(f"wrote {out}")


def tradeoff_figure(t, out, label=""):
    """What the agent buys you: accuracy against acquisition time.

    The baseline figure asks "is the agent better than a fixed policy at the
    same budget", which is one question and not the only one. This asks what
    the agent DELIVERS -- the dial it gives an operator between the free
    timestamp-only floor and the full-acquisition ceiling. Both anchors are
    facts about the problem rather than competitors, so nothing is hidden here;
    the baseline comparison lives in the other figure and still stands.
    """
    floor = t.no_acquisition.iloc[0]
    full = t.full_acquisition.iloc[0]
    tfull = t.full_acquisition_seconds.iloc[0]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.grid(True, zorder=0); ax.set_axisbelow(True); ax.tick_params(length=0)

    ax.axhline(full, color=ANCHOR, lw=1.6, ls=(0, (6, 3)), zorder=2)
    ax.axhline(floor, color=ANCHOR, lw=1.6, ls=(0, (6, 3)), zorder=2)
    ax.text(0.0028, full + 0.004, f"full acquisition, {tfull:g} s",
            fontsize=8, color=ANCHOR)
    ax.text(6.0, floor + 0.004, "no acquisition, timestamp only",
            fontsize=8, color=MUTED, ha="right")

    x = t.agent_seconds.clip(lower=2.5e-3)
    ax.plot(x, t.agent, "-o", color=AQUA, lw=2, ms=6, mew=2, mec=SURFACE,
            zorder=3)

    # Label the quotable operating points. Positions are hand-placed into
    # empty regions with leader lines: the curve sweeps the middle of the
    # panel, so anything anchored near its own point lands on top of it.
    for lam, tx, ty, ha in [(0.01, 0.35, 0.7165, "left"),
                            (0.07, 2.20, 0.6000, "center"),
                            (1.00, 0.012, 0.6360, "left")]:
        r = t[t.lam == lam]
        if r.empty:
            continue
        r = r.iloc[0]
        share = 100 * (r.agent - floor) / (full - floor)
        ax.annotate(f"$\\lambda$={lam:g}\n{share:.0f}% of the gain\n"
                    f"for {100*r.agent_seconds/tfull:.1f}% of the time",
                    (r.agent_seconds, r.agent), xytext=(tx, ty),
                    fontsize=8, color=INK2, ha=ha, va="center", linespacing=1.45,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                                    shrinkA=2, shrinkB=4))

    ax.set_xscale("log")
    ax.set_xlabel("Mean acquisition time per collision (seconds, log scale)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy the agent reaches at each acquisition time" + label,
                 fontsize=11.5, color=INK, pad=10, loc="left")
    ax.set_xticks([0.003, 0.01, 0.1, 1, 6])
    ax.set_xticklabels(["0.003", "0.01", "0.1", "1", "6"])
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, default=Path("docs/figures/baselines"))
    ap.add_argument("--out-name", default="baselines_accuracy.png",
                    help="filename for the accuracy figure, so a filtered "
                         "version does not overwrite the complete one")
    ap.add_argument("--label", default="",
                    help="appended to both titles, so a figure says which cost "
                         "model produced it -- without this the per-model "
                         "figures are indistinguishable once separated from "
                         "their folder")
    args = ap.parse_args()
    label = f"\n{args.label}" if args.label else ""
    t = pd.read_csv(args.in_dir / "baselines.csv")
    accuracy_figure(t, args.in_dir / args.out_name, label)
    tradeoff_figure(t, args.in_dir / "agent_tradeoff.png", label)
    regime_figure(t, args.in_dir / "operational_regime.png", cut=1.0, label=label)


if __name__ == "__main__":
    main()
