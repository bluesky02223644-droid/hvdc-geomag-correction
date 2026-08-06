"""Rebuild Figure 5 from the canonical frozen source only."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter


from _paths import DATA_DIR, OUTPUT_DIR, REPO_ROOT  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = REPO_ROOT
SOURCE = DATA_DIR / "figure5_source.csv"

try:
    from palette_standard import ROLE

    YEAR_COLORS = {2024: ROLE["year_2024"], 2025: ROLE["year_2025"]}
    RELEASE_COLOR = ROLE["flag1_released"]
    NOT_RELEASED_COLOR = ROLE["rule"]
    TEXT_COLOR = ROLE["ink"]
    GRID_COLOR = ROLE["rule"]
    REFERENCE_COLOR = ROLE["muted"]
    NOTE_COLOR = ROLE["muted"]
except Exception:
    YEAR_COLORS = {2024: "#326A9B", 2025: "#D19A2A"}
    RELEASE_COLOR = "#2A7F62"
    NOT_RELEASED_COLOR = "#D9DDE1"
    TEXT_COLOR = "#202428"
    GRID_COLOR = "#D5D9DD"
    REFERENCE_COLOR = "#6B7280"
    NOTE_COLOR = "#6B7280"


def bool_values(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.7,
            "axes.linewidth": 0.8,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SOURCE, encoding="utf-8-sig")
    required = {
        "session_id",
        "year",
        "strong_event",
        "released",
        "final_route",
        "amp_ratio_abs",
        "A_edge_nT",
        "A_ref_on_nT",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Figure 5 source missing columns: {sorted(missing)}")

    strong = bool_values(data["strong_event"])
    released = bool_values(data["released"])
    assert len(data) == 1332
    assert strong.all()
    assert set(data["year"].unique()) == {2024, 2025}
    assert int(released.sum()) == 795
    assert data["session_id"].is_unique
    assert data["amp_ratio_abs"].notna().equals(released)
    expected_ratio = (
        data.loc[released, "A_edge_nT"].astype(float).abs()
        / data.loc[released, "A_ref_on_nT"].astype(float).abs()
    )
    assert np.allclose(
        data.loc[released, "amp_ratio_abs"].astype(float),
        expected_ratio,
        rtol=0,
        atol=1e-12,
    )
    assert (
        data.loc[released, "final_route"].eq("released").all()
        and data.loc[~released, "final_route"].ne("released").all()
    )

    years = [2024, 2025]
    annotations: dict[str, object] = {
        "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "ratio_definition": "abs(A_edge_nT) / abs(A_ref_on_nT)",
        "ratio_population": "released strong events only",
        "total_strong_n": len(data),
        "released_strong_n": int(released.sum()),
        "years": {},
    }
    ratio_groups: list[np.ndarray] = []
    for year in years:
        mask = data["year"].eq(year)
        year_released = released & mask
        values = data.loc[year_released, "amp_ratio_abs"].astype(float).to_numpy()
        ratio_groups.append(values)
        annotations["years"][str(year)] = {
            "strong_total_n": int(mask.sum()),
            "released_n": int(year_released.sum()),
            "released_fraction": float(year_released.sum() / mask.sum()),
            "ratio_n": len(values),
            "ratio_p10": float(np.quantile(values, 0.10)),
            "ratio_p25": float(np.quantile(values, 0.25)),
            "ratio_p50": float(np.quantile(values, 0.50)),
            "ratio_p75": float(np.quantile(values, 0.75)),
            "ratio_p90": float(np.quantile(values, 0.90)),
        }

    expected = {
        "2024": (462, 243),
        "2025": (870, 552),
    }
    for year, (total_n, released_n) in expected.items():
        observed = annotations["years"][year]
        assert observed["strong_total_n"] == total_n
        assert observed["released_n"] == released_n

    configure_style()
    figure, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(6.69, 3.15),
        gridspec_kw={"width_ratios": [0.92, 1.08]},
        constrained_layout=True,
    )

    x = np.arange(len(years))
    release_percent = np.array(
        [annotations["years"][str(year)]["released_fraction"] * 100 for year in years]
    )
    ax_a.bar(
        x,
        release_percent,
        width=0.58,
        color=RELEASE_COLOR,
        edgecolor="white",
        linewidth=0.7,
        label="Released",
        zorder=3,
    )
    ax_a.bar(
        x,
        100 - release_percent,
        bottom=release_percent,
        width=0.58,
        color=NOT_RELEASED_COLOR,
        edgecolor="white",
        linewidth=0.7,
        label="Not released",
        zorder=2,
    )
    for position, year, percent in zip(x, years, release_percent):
        stats = annotations["years"][str(year)]
        ax_a.text(
            position,
            max(percent / 2, 7),
            f"{percent:.1f}%",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=8.2,
        )
        ax_a.text(
            position,
            103.2,
            f"{stats['released_n']} / {stats['strong_total_n']}",
            ha="center",
            va="bottom",
            color=TEXT_COLOR,
            fontsize=8,
        )
    ax_a.set_xticks(x, [str(year) for year in years])
    ax_a.set_ylim(0, 112)
    ax_a.set_yticks([0, 25, 50, 75, 100])
    ax_a.set_ylabel("Share of complete strong events (%)")
    ax_a.set_title("(a) Strong-event release", loc="left", fontweight="bold")
    ax_a.grid(axis="y", color=GRID_COLOR, linewidth=0.6, zorder=0)
    ax_a.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.27),
        ncol=2,
        frameon=False,
        handlelength=1.2,
    )

    rng = np.random.default_rng(20260725)
    positions = [1, 2]
    box = ax_b.boxplot(
        ratio_groups,
        positions=positions,
        widths=0.42,
        whis=(10, 90),
        showfliers=False,
        patch_artist=True,
        medianprops={"color": TEXT_COLOR, "linewidth": 1.4},
        whiskerprops={"color": "#596168", "linewidth": 0.9},
        capprops={"color": "#596168", "linewidth": 0.9},
        boxprops={"edgecolor": "#596168", "linewidth": 0.9},
    )
    for patch, year in zip(box["boxes"], years):
        patch.set_facecolor(YEAR_COLORS[year])
        patch.set_alpha(0.58)
    # The informative band is P10-P90 = 0.84-1.15; plotting down to 0.01 spent
    # most of the panel height on a handful of points. Points below the display
    # floor are drawn as clamped markers with an explicit count so that no
    # session disappears. Box, whiskers, and every reported statistic are still
    # computed from the full released population.
    display_floor = 0.3
    clipped_below: dict[str, int] = {}
    for position, year, values in zip(positions, years, ratio_groups):
        jitter = rng.uniform(-0.13, 0.13, size=len(values))
        inside = values >= display_floor
        ax_b.scatter(
            position + jitter[inside],
            values[inside],
            s=6,
            color=YEAR_COLORS[year],
            alpha=0.20,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        n_below = int((~inside).sum())
        clipped_below[str(year)] = n_below
        if n_below:
            ax_b.scatter(
                position + jitter[~inside],
                np.full(n_below, display_floor * 1.045),
                s=17,
                marker="v",
                color=YEAR_COLORS[year],
                alpha=0.9,
                linewidths=0,
                zorder=3,
            )
            ax_b.text(
                position,
                display_floor * 1.14,
                f"{n_below} below {display_floor:g}",
                ha="center",
                va="bottom",
                fontsize=6.6,
                color=TEXT_COLOR,
            )
        stats = annotations["years"][str(year)]
        ax_b.text(
            position,
            2.56,
            f"n={stats['ratio_n']}\n$P_{{50}}$={stats['ratio_p50']:.3f}",
            ha="center",
            va="top",
            color=TEXT_COLOR,
            fontsize=7.8,
        )
    ax_b.axhline(1.0, color=REFERENCE_COLOR, linewidth=1.0, linestyle="--", zorder=0)
    annotations["display_floor"] = display_floor
    annotations["clipped_below_display_floor"] = clipped_below
    ax_b.set_yscale("log")
    ax_b.set_ylim(display_floor, 3.2)
    ticks = [0.3, 0.5, 1.0, 2.0, 3.0]
    labels = {tick: f"{tick:g}" for tick in ticks}
    ax_b.yaxis.set_major_locator(FixedLocator(ticks))
    ax_b.yaxis.set_major_formatter(FuncFormatter(lambda value, _: labels.get(value, "")))
    ax_b.set_xticks(positions, [str(year) for year in years])
    ax_b.set_ylabel(
        r"Edge-local ratio, $|A_{\mathrm{edge}}|/|A_{\mathrm{ref}}|$"
    )
    ax_b.set_title("(b) Conditional edge-local consistency", loc="left", fontweight="bold")
    ax_b.grid(axis="y", which="major", color=GRID_COLOR, linewidth=0.6, zorder=0)
    ax_b.text(
        2.43,
        1.04,
        "ratio = 1",
        color=REFERENCE_COLOR,
        fontsize=7.4,
        va="bottom",
        ha="right",
    )
    ax_b.text(
        0.02,
        0.98,
        "log scale",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color=NOTE_COLOR,
    )

    for axis in (ax_a, ax_b):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(length=3, color="#596168")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "Figure_05.pdf"
    png_path = OUTPUT_DIR / "Figure_05.png"
    metadata = {
        "Title": "Figure 5: Strong-event release and amplitude consistency",
        "Author": "canonical frozen-source build",
        "Creator": "scripts/make_figure5_v3.py",
    }
    figure.savefig(pdf_path, bbox_inches="tight", metadata=metadata)
    figure.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(figure)

    (OUTPUT_DIR / "figure5_canonical_annotations.json").write_text(
        json.dumps(annotations, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    caption = (
        "Strong-event release and conditional amplitude consistency in the "
        "2024-2025 development cohorts. (a) Released and non-released outcomes "
        "among all 1,332 complete strong events; labels give released/total "
        "counts. (b) Frozen edge-local amplitude ratios for the 795 released "
        "strong events, shown separately by year on a log scale. Boxes span the "
        "interquartile range, whiskers mark P10-P90, points denote individual "
        "sessions, and the dashed line marks ratio = 1. The vertical axis starts "
        "at 0.3; sessions below that value are drawn as downward triangles on the "
        "axis floor and counted in the panel, and all plotted statistics use the "
        "full released population. Amplitude ratios are defined only for released "
        "strong events.\n"
    )
    (OUTPUT_DIR / "figure5_canonical_caption.md").write_text(caption, encoding="utf-8")

    print(json.dumps(annotations, indent=2, ensure_ascii=False))
    print(f"PDF: {pdf_path}")
    print(f"PNG: {png_path}")
    print("Figure 5 assertions: PASS")


if __name__ == "__main__":
    main()
