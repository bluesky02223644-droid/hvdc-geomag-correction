"""Figure 8 - downstream impact of the released corrections on hourly means.

Panel (a) contrasts raw and returned hourly means for one released station-day.
Panels (b) and (c) quantify, over the whole 2024-2025 released population, the
hourly- and daily-mean bias that the released stable-run offsets remove.

The bias of a released run is exact for flag-1 samples: subtracting A_j from
n seconds inside a UTC hour shifts that hourly mean by A_j * n / 3600.
Interpolated flag-2 samples are excluded because they are not an offset removal.
"""

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
from matplotlib.lines import Line2D

from _paths import DATA_DIR, OUTPUT_DIR  # noqa: E402

from palette_standard import ROLE

HERE = Path(__file__).resolve().parent
RUNS_SOURCE = DATA_DIR / "released_stable_runs.csv"
POINTS_SOURCE = DATA_DIR / "demo_station_days.csv.gz"

EXAMPLE_PANEL = "c"
EXAMPLE_LABEL = "Liyang (32011), 2025-08-04"

THRESHOLDS = (0.5, 1.0, 2.0, 5.0, 10.0)


def hourly_bias_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    released = pd.read_csv(RUNS_SOURCE)

    rows = []
    for run in released.itertuples(index=False):
        offset = float(run.A_j_nT)
        start = max(0.0, float(run.run_start_second))
        end = min(86400.0, float(run.run_end_second) + 1.0)
        if not np.isfinite(offset) or end <= start:
            continue
        first_hour = int(start // 3600)
        last_hour = min(int((end - 1e-9) // 3600), 23)
        for hour in range(first_hour, last_hour + 1):
            low = max(start, hour * 3600.0)
            high = min(end, (hour + 1) * 3600.0)
            if high > low:
                rows.append((run.station_code, run.date, hour,
                             offset * (high - low) / 3600.0))

    frame = pd.DataFrame(rows, columns=["station_code", "date", "hour", "bias_nT"])
    hourly = frame.groupby(["station_code", "date", "hour"], as_index=False)["bias_nT"].sum()
    hourly["abs_bias"] = hourly["bias_nT"].abs()

    frame["day_bias_nT"] = frame["bias_nT"] / 24.0
    daily = frame.groupby(["station_code", "date"], as_index=False)["day_bias_nT"].sum()
    daily["abs_bias"] = daily["day_bias_nT"].abs()
    return hourly, daily


def example_hourly_means() -> pd.DataFrame:
    points = pd.read_csv(POINTS_SOURCE)
    day = points[points["panel"] == EXAMPLE_PANEL].copy()
    day["hour"] = day["second_of_day"] // 3600
    grouped = day.groupby("hour").agg(
        raw=("raw_Z_nT", "mean"),
        returned=("returned_Z_nT", "mean"),
        corrected_samples=("flag", lambda values: int((values == 1).sum())),
    )
    return grouped.reset_index()


def survival(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.array([(values >= level).mean() for level in grid])


def draw(hourly: pd.DataFrame, daily: pd.DataFrame, example: pd.DataFrame) -> dict:
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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure = plt.figure(figsize=(7.2, 5.7))
    grid = figure.add_gridspec(
        3, 2, height_ratios=[1.5, 0.60, 1.5], hspace=0.42, wspace=0.28,
        left=0.088, right=0.975, top=0.930, bottom=0.088,
    )
    ax_a = figure.add_subplot(grid[0, :])
    ax_diff = figure.add_subplot(grid[1, :], sharex=ax_a)
    ax_b = figure.add_subplot(grid[2, 0])
    ax_c = figure.add_subplot(grid[2, 1])

    # ---- (a) worked example -------------------------------------------------
    hours = example["hour"].to_numpy()
    raw = example["raw"].to_numpy()
    returned = example["returned"].to_numpy()
    ax_a.step(hours, raw, where="mid", color=ROLE["raw_trace"], linewidth=1.5,
              label="raw hourly mean")
    ax_a.step(hours, returned, where="mid", color=ROLE["released_trace"],
              linewidth=1.5, label="returned hourly mean")
    affected = example["corrected_samples"].to_numpy() > 0
    for hour in hours[affected]:
        ax_a.axvspan(hour - 0.5, hour + 0.5, color=ROLE["fit_ramp_band"],
                     alpha=0.45, zorder=0)
    ax_a.set_ylabel("Hourly mean $Z$ (nT)")
    ax_a.set_title(
        f"(a) Hourly means for one released station-day: {EXAMPLE_LABEL}",
        loc="left", fontweight="bold",
    )
    ax_a.legend(loc="lower right", frameon=False, ncol=2)
    ax_a.text(
        0.015, 0.96,
        "curves coincide outside the shaded hours",
        transform=ax_a.transAxes, fontsize=7.0, color=ROLE["muted"], va="top",
    )
    ax_a.grid(axis="y", color=ROLE["rule"], linewidth=0.6, zorder=0)
    ax_a.tick_params(labelbottom=False)

    difference = returned - raw
    ax_diff.bar(hours, difference, width=0.72, color=ROLE["flag1_released"],
                edgecolor="white", linewidth=0.5, zorder=2)
    ax_diff.axhline(0, color=ROLE["muted"], linewidth=0.8, zorder=1)
    peak = int(np.argmax(np.abs(difference)))
    ax_diff.annotate(
        f"{difference[peak]:+.2f} nT",
        xy=(hours[peak], difference[peak]),
        xytext=(hours[peak] + 1.4, difference[peak]),
        fontsize=7.4, color=ROLE["ink"], va="center",
        arrowprops={"arrowstyle": "-", "color": ROLE["muted"], "lw": 0.7},
    )
    ax_diff.set_ylabel("returned\n$-$ raw (nT)")
    ax_diff.set_xlabel("Hour of day (UTC)")
    ax_diff.set_xlim(-0.6, 23.6)
    ax_diff.set_xticks(range(0, 24, 3))
    ax_diff.grid(axis="y", color=ROLE["rule"], linewidth=0.6, zorder=0)

    # ---- (b) distribution ---------------------------------------------------
    values = hourly["abs_bias"].to_numpy()
    values = values[values > 0]
    bins = np.logspace(np.log10(max(values.min(), 1e-3)), np.log10(values.max()), 34)
    counts, _, _ = ax_b.hist(values, bins=bins, color=ROLE["model_primary"],
                             edgecolor="white", linewidth=0.4, zorder=2)
    median = float(np.median(values))
    ax_b.set_ylim(0, counts.max() * 1.30)
    ax_b.axvline(median, color=ROLE["flag2_transition"], linewidth=1.2,
                 linestyle="--", zorder=3)
    ax_b.text(median * 1.25, counts.max() * 1.09,
              f"median {median:.2f} nT", fontsize=7.2,
              color=ROLE["flag2_transition"], va="center")
    ax_b.set_xscale("log")
    ax_b.set_xlabel("| hourly-mean bias removed | (nT)")
    ax_b.set_ylabel("Station-hours")
    ax_b.set_title("(b) Removed hourly-mean bias", loc="left", fontweight="bold")
    ax_b.text(
        0.015, 0.975, f"n = {len(values):,} affected station-hours",
        transform=ax_b.transAxes, fontsize=7.0, color=ROLE["muted"], va="top",
    )
    ax_b.grid(axis="y", color=ROLE["rule"], linewidth=0.6, zorder=0)

    # ---- (c) exceedance -----------------------------------------------------
    grid_levels = np.logspace(-2, np.log10(50), 220)
    hourly_curve = survival(values, grid_levels) * 100
    daily_values = daily["abs_bias"].to_numpy()
    daily_values = daily_values[daily_values > 0]
    daily_curve = survival(daily_values, grid_levels) * 100

    ax_c.plot(grid_levels, hourly_curve, color=ROLE["model_primary"], linewidth=1.6,
              label=f"hourly means (n = {len(values):,})")
    ax_c.plot(grid_levels, daily_curve, color=ROLE["flag2_transition"], linewidth=1.6,
              label=f"daily means (n = {len(daily_values):,})")
    marked = []
    for level in (1.0, 5.0, 10.0):
        share = (values >= level).mean() * 100
        marked.append((level, share))
        ax_c.plot([level], [share], marker="o", markersize=4.2,
                  color=ROLE["model_primary"], markeredgecolor="white",
                  markeredgewidth=0.7, zorder=4)
    callout = "\n".join(
        f"$\\geq$ {level:>2g} nT   {share:4.1f}%" for level, share in marked
    )
    ax_c.text(
        0.975, 0.965, "hourly means\n" + callout, transform=ax_c.transAxes,
        ha="right", va="top", fontsize=7.2, color=ROLE["ink"], linespacing=1.5,
    )
    ax_c.set_xscale("log")
    ax_c.set_xlim(0.01, 60)
    ax_c.set_ylim(0, 108)
    ax_c.set_xlabel("Threshold (nT)")
    ax_c.set_ylabel("Units above threshold (%)")
    ax_c.set_title("(c) Exceedance of the removed bias", loc="left",
                   fontweight="bold")
    ax_c.grid(color=ROLE["rule"], linewidth=0.6, zorder=0)
    ax_c.legend(loc="lower left", frameon=False)

    for axis in (ax_a, ax_diff, ax_b, ax_c):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(length=3, color=ROLE["muted"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "Figure_08.pdf"
    png_path = OUTPUT_DIR / "Figure_08.png"
    figure.savefig(pdf_path)
    figure.savefig(png_path, dpi=300)
    plt.close(figure)

    summary = {
        "affected_station_hours": int(len(values)),
        "affected_station_days": int(len(daily_values)),
        "hourly_median_nT": round(float(np.median(values)), 4),
        "hourly_p90_nT": round(float(np.percentile(values, 90)), 4),
        "hourly_max_nT": round(float(values.max()), 4),
        "hourly_share_above": {
            f"{level:g}": round(float((values >= level).mean() * 100), 2)
            for level in THRESHOLDS
        },
        "daily_median_nT": round(float(np.median(daily_values)), 4),
        "daily_max_nT": round(float(daily_values.max()), 4),
        "daily_share_above_0.1_nT": round(float((daily_values >= 0.1).mean() * 100), 2),
        "example": {
            "label": EXAMPLE_LABEL,
            "max_abs_hourly_shift_nT": round(float(np.abs(difference).max()), 4),
            "affected_hours": int(affected.sum()),
        },
        "pdf": str(pdf_path),
        "png": str(png_path),
    }
    (OUTPUT_DIR / "figure8_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    hourly, daily = hourly_bias_table()
    example = example_hourly_means()
    summary = draw(hourly, daily, example)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
