"""Figure 2 - horizontal four-stage workflow schematic.

Stage 1 input -> stage 2 session fit -> stage 3 acceptance gates ->
stage 4 per-sample flagged release. Colors come from palette_standard.
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from _paths import OUTPUT_DIR  # noqa: E402  (also puts figures/ on sys.path)

HERE = Path(__file__).resolve().parent
try:
    from palette_standard import ROLE

    COLORS = {
        "ink": ROLE["ink"],
        "muted": ROLE["muted"],
        "rule": ROLE["rule"],
        "blue": ROLE["model_primary"],
        "green": ROLE["flag1_released"],
        "orange": ROLE["flag2_transition"],
        "purple": ROLE["flag4_complex"],
        "flag3": ROLE["flag3_weak"],
        "flag0": ROLE["flag0_or_raw"],
        "light_blue": ROLE["fit_ramp_band"],
        "paper": ROLE["paper"],
        "light_gray": ROLE["light_fill"],
    }
except Exception:
    COLORS = {
        "ink": "#2F343A",
        "muted": "#6B7280",
        "rule": "#D1D5DB",
        "blue": "#3A5F8A",
        "green": "#73c79e",
        "orange": "#E49B5D",
        "purple": "#a577ad",
        "flag3": "#5299cc",
        "flag0": "#B8BDC6",
        "light_blue": "#9fd7e9",
        "paper": "#F7F8FA",
        "light_gray": "#EEF2F5",
    }

OUT_DIR = OUTPUT_DIR

# Card geometry in figure coordinates -------------------------------------
CARDS = {
    1: (0.015, 0.175),
    2: (0.208, 0.448),
    3: (0.481, 0.671),
    4: (0.704, 0.985),
}
CARD_TOP = 0.958
CARD_BOTTOM = 0.030
HEAD_TOP = 0.946
HEAD_BOTTOM = 0.888
SUBTITLE_Y = 0.866
ARROW_Y = 0.50

TITLES = {
    1: "1   Input",
    2: "2   Session fit",
    3: "3   Gates",
    4: "4   Release",
}

SUBTITLES = {
    1: "catalog edges + waveform",
    2: "ramp-boxcar geometry",
    3: "acceptance, in order",
    4: "per-sample flags",
}


def ramp_boxcar(t, a, b, tau):
    box = np.zeros_like(t, dtype=float)
    rising = (t >= a) & (t < a + tau)
    plateau = (t >= a + tau) & (t < b)
    falling = (t >= b) & (t < b + tau)
    box[rising] = (t[rising] - a) / tau
    box[plateau] = 1.0
    box[falling] = 1.0 - (t[falling] - b) / tau
    return box


def draw_cards(fig):
    for stage, (x0, x1) in CARDS.items():
        fig.add_artist(
            FancyBboxPatch(
                (x0, CARD_BOTTOM),
                x1 - x0,
                CARD_TOP - CARD_BOTTOM,
                transform=fig.transFigure,
                boxstyle="round,pad=0,rounding_size=0.010",
                facecolor=COLORS["paper"],
                edgecolor=COLORS["rule"],
                linewidth=0.8,
                zorder=0,
            )
        )
        fig.add_artist(
            FancyBboxPatch(
                (x0 + 0.007, HEAD_BOTTOM),
                (x1 - x0) - 0.014,
                HEAD_TOP - HEAD_BOTTOM,
                transform=fig.transFigure,
                boxstyle="round,pad=0,rounding_size=0.008",
                facecolor=COLORS["blue"],
                edgecolor="none",
                zorder=1,
            )
        )
        fig.text(
            (x0 + x1) / 2,
            (HEAD_TOP + HEAD_BOTTOM) / 2,
            TITLES[stage],
            fontsize=7.4,
            weight="bold",
            color="white",
            ha="center",
            va="center",
            zorder=2,
        )
        fig.text(
            (x0 + x1) / 2,
            SUBTITLE_Y,
            SUBTITLES[stage],
            fontsize=5.6,
            color=COLORS["muted"],
            ha="center",
            va="center",
            zorder=2,
        )


def draw_stage_arrows(fig):
    for left, right in ((1, 2), (2, 3), (3, 4)):
        start = CARDS[left][1] + 0.006
        end = CARDS[right][0] - 0.006
        fig.add_artist(
            FancyArrowPatch(
                (start, ARROW_Y),
                (end, ARROW_Y),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.4,
                color=COLORS["blue"],
                shrinkA=0,
                shrinkB=0,
                zorder=3,
            )
        )


def rule(fig, x0, x1, y, color=None, lw=0.7):
    fig.add_artist(
        Line2D(
            [x0, x1],
            [y, y],
            transform=fig.transFigure,
            color=color or COLORS["rule"],
            linewidth=lw,
            zorder=2,
        )
    )


def draw_stage1(fig):
    x0, x1 = CARDS[1][0] + 0.012, CARDS[1][1] - 0.012

    fig.text(x0, 0.822, "Catalog edge records", fontsize=6.4, weight="bold")
    rule(fig, x0, x1, 0.808, COLORS["ink"], 0.8)
    fig.text(x0, 0.780, "edge", fontsize=5.4, color=COLORS["muted"])
    fig.text(x0 + 0.048, 0.780, "step", fontsize=5.4, color=COLORS["muted"])
    fig.text(x1, 0.780, "anchor", fontsize=5.4, color=COLORS["muted"], ha="right")
    rule(fig, x0, x1, 0.768)
    fig.text(x0, 0.740, "on-edge", fontsize=5.8)
    fig.text(x0 + 0.048, 0.740, r"$+A_{\rm ref}$", fontsize=5.8)
    fig.text(x1, 0.740, r"$t_{\rm on}^{cat}$", fontsize=5.8, ha="right")
    fig.text(x0, 0.704, "off-edge", fontsize=5.8)
    fig.text(x0 + 0.048, 0.704, r"$-A_{\rm ref}$", fontsize=5.8)
    fig.text(x1, 0.704, r"$t_{\rm off}^{cat}$", fontsize=5.8, ha="right")
    rule(fig, x0, x1, 0.688, COLORS["ink"], 0.8)

    fig.text(x0, 0.648, "Matched 1 Hz $Z$ record", fontsize=6.4, weight="bold")
    ax = fig.add_axes([x0, 0.512, x1 - x0, 0.116])
    ax.set_facecolor("none")
    t = np.linspace(0, 900, 901)
    trace = (
        0.15 * np.sin(2 * np.pi * t / 1100)
        + 2.6 * ramp_boxcar(t, 200.0, 700.0, 45.0)
        + 0.05 * np.sin(2 * np.pi * t / 17)
    )
    ax.plot(t / 60, trace, color=COLORS["ink"], linewidth=0.9)
    ax.set_xlim(0, 15)
    ax.set_ylim(-0.6, 3.2)
    ax.axis("off")

    fig.text(x0, 0.456, "Session assembly", fontsize=6.4, weight="bold")
    rule(fig, x0, x1, 0.442, COLORS["ink"], 0.8)
    fig.text(
        x0,
        0.418,
        "group rows by operation\nand station; order edges\nin time",
        fontsize=5.3,
        color=COLORS["muted"],
        linespacing=1.45,
        va="top",
    )
    fig.text(
        x0,
        0.300,
        "complete paired session:\none on-edge followed by\none oppositely signed\noff-edge",
        fontsize=5.3,
        color=COLORS["ink"],
        linespacing=1.45,
        va="top",
    )


def draw_stage2(fig):
    x0, x1 = CARDS[2][0] + 0.014, CARDS[2][1] - 0.014

    ax = fig.add_axes([x0 + 0.030, 0.540, (x1 - x0) - 0.034, 0.238])
    ax.set_facecolor("none")
    t = np.linspace(0, 1320, 1321)
    a, b, tau = 180.0, 1030.0, 130.0
    cat_on, cat_off = 120.0, 975.0
    background = 0.00055 * t + 0.18 * np.sin(2 * np.pi * t / 1500)
    fitted = background + 4.1 * ramp_boxcar(t, a, b, tau)
    raw = fitted + 0.035 * np.sin(2 * np.pi * t / 19)

    w0, w1 = (a + tau + 30) / 60, (a + tau + 210) / 60
    ax.axvspan(w0, w1, color=COLORS["light_blue"], alpha=0.75, zorder=0)
    ax.plot(t / 60, raw, color=COLORS["ink"], linewidth=1.15)
    ax.plot(t / 60, fitted, color=COLORS["blue"], linewidth=1.15, linestyle=(0, (4, 2)))
    ax.plot(t / 60, background, color=COLORS["muted"], linewidth=0.95, linestyle=":")
    for x in (cat_on, cat_off):
        ax.axvline(x / 60, color=COLORS["orange"], linewidth=0.9, linestyle=(0, (3, 2)))
    for x in (a, b):
        ax.axvline(x / 60, color=COLORS["blue"], linewidth=0.85, alpha=0.75)

    ax.text(a / 60, 5.42, r"$a$", ha="center", va="bottom", fontsize=6.0,
            color=COLORS["blue"])
    ax.text(b / 60, 5.42, r"$b$", ha="center", va="bottom", fontsize=6.0,
            color=COLORS["blue"])
    ax.annotate(
        "",
        xy=(a / 60, 4.70),
        xytext=((a + tau) / 60, 4.70),
        arrowprops={"arrowstyle": "<->", "color": COLORS["blue"], "lw": 0.7,
                    "shrinkA": 0, "shrinkB": 0},
    )
    ax.text((a + tau / 2) / 60, 4.80, r"$\tau$", ha="center", va="bottom",
            fontsize=6.0, color=COLORS["blue"])
    ax.text((w0 + w1) / 2, 4.80, r"$W$", ha="center", va="bottom", fontsize=6.0,
            color=COLORS["blue"])

    handles = [
        Line2D([0], [0], color=COLORS["ink"], lw=1.15, label="1 Hz $Z$"),
        Line2D([0], [0], color=COLORS["blue"], lw=1.15, ls=(0, (4, 2)),
               label="ramp-boxcar"),
        Line2D([0], [0], color=COLORS["muted"], lw=0.95, ls=":", label=r"$B(t)$"),
        Line2D([0], [0], color=COLORS["orange"], lw=0.9, ls=(0, (3, 2)),
               label="catalog anchor"),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=5.6,
        handlelength=1.5,
        columnspacing=0.9,
        handletextpad=0.5,
        labelspacing=0.35,
    )
    ax.set_xlim(0, 22)
    ax.set_ylim(-0.45, 5.90)
    ax.set_xlabel("Time (min)", fontsize=6.2, labelpad=1.5)
    ax.set_ylabel("Relative $Z$ (nT)", fontsize=6.2, labelpad=1.5)
    ax.tick_params(labelsize=5.6, width=0.7, length=2.4, pad=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    fig.text(x0, 0.456, "Estimated per session", fontsize=6.4, weight="bold")
    rule(fig, x0, x1, 0.442, COLORS["ink"], 0.8)
    rows = [
        ("transition geometry", r"$a,\ b,\ \tau$"),
        ("fitted amplitude", r"$A_{\rm fit}$"),
        ("edge-local amplitude", r"$A_{\rm edge}$"),
        ("stable-run offset", r"$A_j$"),
        ("local background", r"$B(t)$"),
    ]
    y = 0.412
    for label, symbol in rows:
        fig.text(x0, y, label, fontsize=5.9, color=COLORS["muted"])
        fig.text(x1, y, symbol, fontsize=6.2, ha="right")
        y -= 0.036

    fig.text(x0, 0.196, "Multi-edge group", fontsize=6.3, weight="bold")
    ax2 = fig.add_axes([x0, 0.108, 0.072, 0.062])
    ax2.set_facecolor("none")
    xx = np.array([0, 1, 1, 2, 2, 3, 3, 4], dtype=float)
    yy = np.array([0, 0, 1, 1, 0, 0, 0.65, 0.65])
    ax2.plot(xx, yy, color=COLORS["ink"], linewidth=1.0)
    ax2.set_xlim(-0.1, 4.1)
    ax2.set_ylim(-0.15, 1.2)
    ax2.axis("off")
    fig.text(
        x0 + 0.082,
        0.139,
        "candidate plateaus only;\nno global boxcar fit",
        fontsize=5.6,
        color=COLORS["muted"],
        linespacing=1.35,
        va="center",
    )


def draw_stage3(fig):
    x0, x1 = CARDS[3][0] + 0.012, CARDS[3][1] - 0.012

    chips = [
        ("identifiable response", [r"$|A_{\rm baseline}|\geq\theta_{\rm id}$"], False),
        ("stable run", [r"spread $\leq\theta_{\rm run}$"], False),
        ("local residual", [r"${\rm RMS}_{local}\leq\theta_{\rm res}$"], False),
        (
            "auxiliary guard",
            ["cross-station support", "conditional: low-amplitude", "sessions only"],
            True,
        ),
    ]

    top = 0.836
    for index, (name, lines, conditional) in enumerate(chips):
        height = 0.056 + 0.024 * len(lines)
        bottom = top - height
        fig.add_artist(
            FancyBboxPatch(
                (x0, bottom),
                x1 - x0,
                height,
                transform=fig.transFigure,
                boxstyle="round,pad=0,rounding_size=0.008",
                facecolor="white",
                edgecolor=COLORS["blue"] if not conditional else COLORS["muted"],
                linewidth=0.8,
                linestyle="solid" if not conditional else (0, (3, 2)),
                zorder=1,
            )
        )
        fig.text(x0 + 0.008, top - 0.022, name, fontsize=6.0, weight="bold",
                 va="center", zorder=2)
        ly = top - 0.046
        for line in lines:
            fig.text(x0 + 0.008, ly, line, fontsize=5.4, color=COLORS["muted"],
                     va="center", zorder=2)
            ly -= 0.024
        if index < len(chips) - 1:
            fig.add_artist(
                FancyArrowPatch(
                    ((x0 + x1) / 2, bottom - 0.003),
                    ((x0 + x1) / 2, bottom - 0.023),
                    transform=fig.transFigure,
                    arrowstyle="-|>",
                    mutation_scale=9,
                    linewidth=0.9,
                    color=COLORS["blue"],
                    shrinkA=0,
                    shrinkB=0,
                    zorder=2,
                )
            )
        top = bottom - 0.028

    rule(fig, x0, x1, 0.240, COLORS["ink"], 0.8)
    fig.text(x0, 0.216, "If any gate fails", fontsize=6.0, weight="bold",
             va="center", zorder=2)
    fig.text(
        x0,
        0.192,
        "no offset is removed and the\nsamples are returned unchanged\n"
        "as flag 3 (weak) or flag 4\n(complex or low confidence)",
        fontsize=5.4,
        color=COLORS["muted"],
        linespacing=1.45,
        va="top",
        zorder=2,
    )


def draw_stage4(fig):
    x0, x1 = CARDS[4][0] + 0.014, CARDS[4][1] - 0.014

    ax = fig.add_axes([x0 + 0.032, 0.632, (x1 - x0) - 0.036, 0.176])
    ax.set_facecolor("none")
    t = np.linspace(0, 600, 601)
    a, b, tau = 115.0, 460.0, 35.0
    background = 0.08 * np.sin(2 * np.pi * t / 480) + 0.0003 * t
    raw = background + 2.8 * ramp_boxcar(t, a, b, tau)

    ax.plot(t / 60, raw, color=COLORS["flag0"], linewidth=1.05)
    pre = t < a
    on = (t >= a) & (t < a + tau)
    plateau = (t >= a + tau) & (t < b)
    off = (t >= b) & (t < b + tau)
    post = t >= b + tau
    ax.plot(t[pre] / 60, background[pre], color=COLORS["ink"], linewidth=1.25)
    ax.plot(t[on] / 60, background[on], color=COLORS["orange"], linewidth=1.25,
            linestyle=(0, (3, 2)))
    ax.plot(t[plateau] / 60, background[plateau], color=COLORS["green"],
            linewidth=1.6)
    ax.plot(t[off] / 60, background[off], color=COLORS["orange"], linewidth=1.25,
            linestyle=(0, (3, 2)))
    ax.plot(t[post] / 60, background[post], color=COLORS["ink"], linewidth=1.25)
    ax.annotate(
        "",
        xy=(4.6, 2.82),
        xytext=(4.6, 0.13),
        arrowprops={"arrowstyle": "<->", "color": COLORS["green"], "lw": 0.85,
                    "shrinkA": 0, "shrinkB": 0},
    )
    ax.text(4.85, 1.45, r"subtract $A_j$", color=COLORS["green"], fontsize=6.0,
            va="center")
    ax.legend(
        handles=[
            Line2D([0], [0], color=COLORS["flag0"], lw=1.05, label="raw $Z$"),
            Line2D([0], [0], color=COLORS["ink"], lw=1.25, label="returned $Z$"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.00),
        ncol=2,
        frameon=False,
        fontsize=5.8,
        handlelength=1.5,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.45, 3.35)
    ax.set_ylabel("Relative $Z$ (nT)", fontsize=6.2, labelpad=1.5)
    ax.tick_params(labelsize=5.6, width=0.7, length=2.4, pad=1.5,
                   labelbottom=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    flag_ax = fig.add_axes([x0 + 0.032, 0.578, (x1 - x0) - 0.036, 0.032])
    spans = [
        (0.0, a / 60, COLORS["flag0"], "0"),
        (a / 60, tau / 60, COLORS["orange"], "2"),
        ((a + tau) / 60, (b - a - tau) / 60, COLORS["green"], "1"),
        (b / 60, tau / 60, COLORS["orange"], "2"),
        ((b + tau) / 60, (600 - b - tau) / 60, COLORS["flag0"], "0"),
    ]
    for start, width, color, label in spans:
        flag_ax.add_patch(
            Rectangle((start, 0), width, 1, facecolor=color, edgecolor="white",
                      linewidth=0.5)
        )
        if width >= 0.5:
            flag_ax.text(start + width / 2, 0.5, label, ha="center", va="center",
                         fontsize=5.6,
                         color="white" if label in {"1", "2"} else COLORS["ink"])
    flag_ax.set_xlim(0, 10)
    flag_ax.set_ylim(0, 1)
    flag_ax.set_yticks([])
    flag_ax.set_xticks(np.arange(0, 11, 2))
    flag_ax.tick_params(axis="x", labelsize=5.6, width=0.6, length=2, pad=1.5)
    for spine in ("top", "left", "right"):
        flag_ax.spines[spine].set_visible(False)
    flag_ax.spines["bottom"].set_linewidth(0.6)
    flag_ax.set_xlabel("Time (min)", fontsize=6.2, labelpad=1.5)
    fig.text(x0 + 0.026, 0.594, "flag", fontsize=5.9, ha="right", va="center")

    fig.text(x0, 0.456, "Per-sample outcome", fontsize=6.4, weight="bold")
    rule(fig, x0, x1, 0.442, COLORS["ink"], 0.8)
    fig.text(x0 + 0.018, 0.416, "flag", fontsize=5.4, color=COLORS["muted"])
    fig.text(x0 + 0.052, 0.416, "sample class", fontsize=5.4, color=COLORS["muted"])
    fig.text(x1, 0.416, "action", fontsize=5.4, color=COLORS["muted"], ha="right")
    rule(fig, x0, x1, 0.404)

    rows = [
        ("0", "outside session", "unchanged", COLORS["flag0"]),
        ("1", "stable plateau", r"subtract $A_j$", COLORS["green"]),
        ("2", "transition", "interpolate", COLORS["orange"]),
        ("3", "weak response", "unchanged", COLORS["flag3"]),
        ("4", "complex / low-conf.", "unchanged", COLORS["purple"]),
    ]
    y = 0.374
    for flag, label, action, color in rows:
        fig.add_artist(
            Rectangle(
                (x0, y - 0.007),
                0.010,
                0.015,
                transform=fig.transFigure,
                facecolor=color,
                edgecolor="none",
                zorder=2,
            )
        )
        fig.text(x0 + 0.021, y, flag, fontsize=6.0, va="center")
        fig.text(x0 + 0.052, y, label, fontsize=5.8, va="center")
        fig.text(x1, y, action, fontsize=5.8, va="center", ha="right")
        y -= 0.038
    rule(fig, x0, x1, 0.196, COLORS["ink"], 0.8)
    fig.text(
        x0,
        0.170,
        "The returned series keeps the full\n"
        "1 Hz grid; every sample carries one\n"
        "flag, so corrected, interpolated and\n"
        "unchanged samples stay separable.",
        fontsize=5.4,
        color=COLORS["muted"],
        linespacing=1.45,
        va="top",
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig = plt.figure(figsize=(7.2, 4.0), facecolor="white")
    draw_cards(fig)
    draw_stage_arrows(fig)
    draw_stage1(fig)
    draw_stage2(fig)
    draw_stage3(fig)
    draw_stage4(fig)

    png_path = OUT_DIR / "Figure_02.png"
    pdf_path = OUT_DIR / "Figure_02.pdf"
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)

    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
