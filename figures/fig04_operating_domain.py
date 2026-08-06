from pathlib import Path
import csv
import shutil
import sys

import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np


from _paths import DATA_DIR, OUTPUT_DIR  # noqa: E402  (puts figures/ on sys.path)

HERE = Path(__file__).resolve().parent

SOURCE = DATA_DIR / "injection_grid_metrics.csv"
OUT = OUTPUT_DIR

AMPLITUDES = [1.0, 3.0, 10.0, 30.0]
DURATIONS = [3.0, 15.0, 60.0, 240.0]

from palette_standard import ROLE, get_cmap

# Domain outlines are drawn in ink and separated by line style, so that they stay
# legible on both the cool (coverage) and warm (error) colour ramps.
SUMMARY_COLOR = ROLE["ink"]
STRESS_COLOR = ROLE["ink"]
CMAP_COVERAGE = get_cmap("sequential")
CMAP_ERROR = get_cmap("sequential_warm")


def pivot(rows: list[dict[str, float]], column: str) -> np.ndarray:
    lookup = {(row["amplitude_nT"], row["duration_min"]): row[column] for row in rows}
    return np.array(
        [[lookup[(amplitude, duration)] for duration in DURATIONS] for amplitude in AMPLITUDES],
        dtype=float,
    )


def add_domain_marks(ax: plt.Axes) -> None:
    """Draw domain outlines only; shared labels live at the figure top."""
    ax.add_patch(
        Rectangle(
            (-0.46, 0.54),
            2.92,
            2.92,
            fill=False,
            edgecolor=SUMMARY_COLOR,
            linewidth=1.6,
            zorder=5,
        )
    )
    ax.add_patch(
        Rectangle(
            (2.5, -0.5),
            1,
            4,
            fill=False,
            edgecolor=STRESS_COLOR,
            linewidth=1.4,
            linestyle=(0, (4, 2)),
            zorder=5,
        )
    )


def add_shared_domain_labels(fig: plt.Figure) -> None:
    """One shared label row above all panels to avoid crowding titles."""
    # Coordinates assume fig.subplots_adjust(top≈0.90) left a clear header band.
    fig.text(
        0.28,
        0.975,
        "summary region",
        ha="center",
        va="bottom",
        color=SUMMARY_COLOR,
        fontsize=9.2,
        fontweight="bold",
        transform=fig.transFigure,
    )
    fig.text(
        0.62,
        0.975,
        "multi-hour stress column",
        ha="center",
        va="bottom",
        color=STRESS_COLOR,
        fontsize=9.2,
        fontweight="bold",
        transform=fig.transFigure,
    )
    summary_bracket = FancyArrowPatch(
        (0.10, 0.962),
        (0.46, 0.962),
        transform=fig.transFigure,
        arrowstyle="|-|",
        mutation_scale=7,
        linewidth=1.6,
        color=SUMMARY_COLOR,
        shrinkA=0,
        shrinkB=0,
    )
    fig.add_artist(summary_bracket)
    stress_bracket = FancyArrowPatch(
        (0.52, 0.962),
        (0.72, 0.962),
        transform=fig.transFigure,
        arrowstyle="|-|",
        mutation_scale=7,
        linewidth=1.4,
        linestyle=(0, (4, 2)),
        color=STRESS_COLOR,
        shrinkA=0,
        shrinkB=0,
    )
    fig.add_artist(stress_bracket)
    fig.text(
        0.90,
        0.975,
        "n = 24 per cell",
        ha="right",
        va="bottom",
        color=ROLE["muted"],
        fontsize=9.0,
        transform=fig.transFigure,
    )


def main() -> None:
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    data = [{key: float(value) for key, value in row.items()} for row in raw_rows]
    assert len(data) == 16
    assert all(row["n_total"] == 24 for row in data)
    assert int(sum(row["n_total"] for row in data)) == 384
    assert int(sum(row["n_released"] for row in data)) == 320

    release = pivot(data, "release_rate") * 100
    amp_error = pivot(data, "primary_Aj_abs_error_median_nT")
    rms = pivot(data, "flag1_truth_RMS_median_nT")
    released_n = pivot(data, "n_released").astype(int)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.5))
    fig.subplots_adjust(left=0.10, right=0.90, top=0.90, bottom=0.06, hspace=0.38)
    specs = [
        (release, "(a) Release rate", CMAP_COVERAGE, Normalize(50, 100), "%"),
        (
            amp_error,
            "(b) Median amplitude error, first accepted stable run",
            CMAP_ERROR,
            Normalize(0.15, 0.36),
            "nT",
        ),
        (
            rms,
            "(c) Median truth RMS, all released samples",
            CMAP_ERROR,
            Normalize(0.15, 2.4),
            "nT",
        ),
    ]

    for panel_index, (ax, (values, title, cmap, norm, unit)) in enumerate(zip(axes, specs)):
        image = ax.imshow(values, cmap=cmap, norm=norm, origin="upper", aspect="auto")
        ax.set_title(title, loc="left", fontweight="bold", pad=6)
        ax.set_xticks(range(4), [str(int(v)) for v in DURATIONS])
        ax.set_yticks(range(4), [str(int(v)) for v in AMPLITUDES])
        ax.set_xlabel("Duration (min)")
        ax.set_ylabel("Injected offset (nT)")
        ax.tick_params(length=3, width=0.8)
        for spine in ax.spines.values():
            spine.set_color(ROLE["ink"])

        for row in range(4):
            for col in range(4):
                value = values[row, col]
                rgba = image.cmap(image.norm(value))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                color = ROLE["ink"] if luminance > 0.55 else "white"
                if panel_index == 0:
                    label = f"{released_n[row, col]}/24\n{value:.0f}%"
                else:
                    label = f"{value:.2f}"
                ax.text(col, row, label, ha="center", va="center", color=color, fontsize=10)

        add_domain_marks(ax)
        cbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03, shrink=0.9)
        cbar.set_label(unit, rotation=90, labelpad=6, fontsize=10)
        cbar.ax.tick_params(labelsize=9)
        cbar.outline.set_linewidth(0.7)

    add_shared_domain_labels(fig)

    OUT.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT / "Figure_04.pdf"
    png_path = OUT / "Figure_04.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
