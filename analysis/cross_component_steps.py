"""Cross-component feasibility audit: how much of the released Z step appears in H and D.

Scope: the released, catalog-strong paired sessions of the 2024-2025 cohorts.
For each session we re-run the identical baseline-window measurement used by the
Z workflow, but on the H and D channels of the same raw file. Because the D unit
convention is not verified across the instrument fleet, D is reported only as a
dimensionless signal-to-noise ratio against its own pre-window scatter, while H
is reported in nT.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"F:\高压直流地磁数据处理")
sys.path.insert(0, str(ROOT / "scripts"))

from run_protocol_v3_paired_sessions import mad, second_of_day, window_values  # noqa: E402

MANIFEST = ROOT / "outputs" / "canonical" / "complete_paired_sessions_2024_2025.csv"
SESSIONS = ROOT / "outputs" / "protocol_canonical_full" / "protocol_v3_session_results.csv"
OUT_DIR = ROOT / "outputs" / "cross_component_feasibility"

LENGTH = 120
BUFFER = 10
MIN_VALID_FRACTION = 0.8
ABS_FLOOR_NT = 0.5


def read_components(path_text: str) -> dict[str, np.ndarray]:
    """Read Z, H, and D onto an exact 86,400-s grid.

    Mirrors read_raw_z of the frozen runner, but keeps all three components.
    """
    path = Path(path_text)
    grids: dict[str, np.ndarray] = {}
    with path.open("rb") as handle:
        header = handle.readline()
        columns = header.split()
        if len(columns) < 3:
            raise ValueError("raw header has fewer than three columns")
        upper = [item.upper() for item in columns[2:]]
        index = {}
        for name in (b"Z", b"H", b"D"):
            if name not in upper:
                raise ValueError(f"raw file has no {name.decode()} column")
            index[name.decode()] = 2 + upper.index(name)
            grids[name.decode()] = np.full(86400, np.nan, dtype=np.float64)
        for raw_line in handle:
            parts = raw_line.split()
            if len(parts) < 2:
                continue
            clock = parts[1]
            if len(clock) < 8:
                continue
            try:
                second = int(clock[0:2]) * 3600 + int(clock[3:5]) * 60 + int(clock[6:8])
            except ValueError:
                continue
            if not 0 <= second < 86400:
                continue
            for name, position in index.items():
                if len(parts) <= position:
                    continue
                try:
                    value = float(parts[position])
                except ValueError:
                    continue
                if value != 999999 and math.isfinite(value) and not math.isfinite(grids[name][second]):
                    grids[name][second] = value
    return grids


def step_and_noise(series: np.ndarray, on: int, on_end: int) -> tuple[float, float, float]:
    """Measured local step, pre-window scatter, and pre-window valid fraction."""
    pre = window_values(series, on - BUFFER - LENGTH, on - BUFFER)
    post = window_values(series, on_end + BUFFER, on_end + BUFFER + LENGTH)
    if not len(pre) or not len(post):
        return math.nan, math.nan, 0.0
    pre_valid = np.isfinite(pre)
    post_valid = np.isfinite(post)
    fraction = min(pre_valid.mean(), post_valid.mean())
    if pre_valid.sum() < 5 or post_valid.sum() < 5:
        return math.nan, math.nan, fraction
    step = float(np.median(post[post_valid]) - np.median(pre[pre_valid]))
    differences = np.diff(pre[pre_valid])
    sigma = 1.4826 * mad(differences) / math.sqrt(2.0) if len(differences) else math.nan
    return step, sigma, float(fraction)


def process_file(task: dict) -> list[dict]:
    try:
        grids = read_components(task["raw_file"])
    except Exception as error:  # noqa: BLE001
        return [{"session_uid": row["session_uid"], "error": str(error)} for row in task["rows"]]

    results = []
    for row in task["rows"]:
        on = int(row["on"])
        on_end = int(row["on_end"])
        record = {
            "session_uid": row["session_uid"],
            "station_code": row["station_code"],
            "date": row["date"],
            "year": row["year"],
            "duration_min": row["duration_min"],
            "A_ref_nT": row["A_ref_nT"],
            "A_released_Z_nT": row["A_released_Z_nT"],
            "error": "",
        }
        for name in ("Z", "H", "D"):
            step, sigma, fraction = step_and_noise(grids[name], on, on_end)
            record[f"step_{name}"] = step
            record[f"sigma_{name}"] = sigma
            record[f"valid_fraction_{name}"] = fraction
            record[f"snr_{name}"] = (
                abs(step) / sigma if math.isfinite(step) and math.isfinite(sigma) and sigma > 0 else math.nan
            )
        results.append(record)
    return results


def main() -> None:
    manifest = pd.read_csv(MANIFEST, low_memory=False)
    sessions = pd.read_csv(SESSIONS, low_memory=False)

    released = sessions[
        (sessions["final_route"] == "released") & (sessions["catalog_strong"].astype(str).str.lower() == "true")
    ][["session_uid", "released_primary_A_j_nT"]]
    print(f"released catalog-strong sessions: {len(released)}")

    frame = manifest.merge(released, on="session_uid", how="inner")
    print(f"matched to manifest: {len(frame)} over {frame['raw_file'].nunique()} raw files")

    frame["on"] = pd.to_datetime(frame["on_time"]).map(second_of_day)
    frame["on_end"] = pd.to_datetime(frame["on_edge_end_time"]).map(second_of_day)

    tasks = []
    for raw_file, group in frame.groupby("raw_file"):
        rows = [
            {
                "session_uid": item.session_uid,
                "station_code": item.station_code,
                "date": item.date,
                "year": item.year,
                "duration_min": item.duration_min,
                "A_ref_nT": item.A_ref_nT,
                "A_released_Z_nT": item.released_primary_A_j_nT,
                "on": item.on,
                "on_end": item.on_end,
            }
            for item in group.itertuples(index=False)
        ]
        tasks.append({"raw_file": raw_file, "rows": rows})

    records = []
    done = 0
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(process_file, task) for task in tasks]
        for future in as_completed(futures):
            records.extend(future.result())
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(tasks)} files", flush=True)

    result = pd.DataFrame(records)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_DIR / "cross_component_steps.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'cross_component_steps.csv'}  ({len(result)} rows)")

    usable = result[
        (result["error"] == "")
        & result["step_Z"].notna()
        & result["step_H"].notna()
        & result["step_D"].notna()
        & (result[["valid_fraction_Z", "valid_fraction_H", "valid_fraction_D"]].min(axis=1) >= MIN_VALID_FRACTION)
    ].copy()
    print(f"usable sessions with all three components: {len(usable)}")

    usable["threshold_H"] = np.maximum(ABS_FLOOR_NT, 3.0 * usable["sigma_H"])
    usable["detect_H"] = usable["step_H"].abs() >= usable["threshold_H"]
    usable["detect_D"] = usable["snr_D"] >= 3.0
    usable["ratio_H_over_Z"] = usable["step_H"].abs() / usable["step_Z"].abs()

    summary = {
        "n_released_strong_sessions": int(len(released)),
        "n_usable": int(len(usable)),
        "median_abs_step_Z_nT": round(float(usable["step_Z"].abs().median()), 3),
        "median_abs_step_H_nT": round(float(usable["step_H"].abs().median()), 3),
        "H": {
            "detectable_n": int(usable["detect_H"].sum()),
            "detectable_pct": round(float(usable["detect_H"].mean() * 100), 1),
            "median_ratio_to_Z": round(float(usable["ratio_H_over_Z"].median()), 3),
            "p90_ratio_to_Z": round(float(usable["ratio_H_over_Z"].quantile(0.9)), 3),
            "median_snr": round(float(usable["snr_H"].median()), 2),
        },
        "D": {
            "detectable_n": int(usable["detect_D"].sum()),
            "detectable_pct": round(float(usable["detect_D"].mean() * 100), 1),
            "median_snr": round(float(usable["snr_D"].median()), 2),
        },
        "Z": {
            "median_snr": round(float(usable["snr_Z"].median()), 2),
        },
        "note": (
            "D is reported only as a dimensionless signal-to-noise ratio because "
            "its recorded unit convention is not verified across the instrument fleet."
        ),
    }
    (OUT_DIR / "cross_component_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
