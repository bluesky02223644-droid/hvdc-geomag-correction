"""Gap-matched control test.

Each released session contributes control measurements taken at quiet times on
the same station-day, using the same window geometry AND the same pre-to-post
gap as that session's own on-edge, so the only difference between the event and
control statistic is whether a cataloged HVDC edge sits in the gap.
"""

from __future__ import annotations

import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"F:\高压直流地磁数据处理")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(r"C:\Users\Lenovo\AppData\Local\Temp")))

from hd_feasibility import read_components, step_and_noise  # noqa: E402
from run_protocol_v3_paired_sessions import second_of_day  # noqa: E402

MANIFEST = ROOT / "outputs" / "canonical" / "complete_paired_sessions_2024_2025.csv"
STEPS = ROOT / "outputs" / "cross_component_feasibility" / "cross_component_steps.csv"
OUT_DIR = ROOT / "outputs" / "cross_component_feasibility"

GUARD_S = 1800
CONTROLS_PER_SESSION = 3
RNG_SEED = 20260725


def process_file(task: dict) -> list[dict]:
    try:
        grids = read_components(task["raw_file"])
    except Exception:  # noqa: BLE001
        return []
    out = []
    for item in task["controls"]:
        record = {
            "session_uid": item["session_uid"],
            "station_code": task["station_code"],
            "gap_s": item["gap"],
            "second": item["second"],
        }
        for name in ("Z", "H", "D"):
            step, sigma, fraction = step_and_noise(
                grids[name], item["second"], item["second"] + item["gap"]
            )
            record[f"step_{name}"] = step
            record[f"sigma_{name}"] = sigma
            record[f"valid_fraction_{name}"] = fraction
            record[f"snr_{name}"] = (
                abs(step) / sigma if math.isfinite(step) and math.isfinite(sigma) and sigma > 0 else math.nan
            )
        out.append(record)
    return out


def main() -> None:
    event = pd.read_csv(STEPS)
    manifest = pd.read_csv(MANIFEST, low_memory=False)
    joined = event[["session_uid"]].merge(manifest, on="session_uid", how="left")
    joined["on"] = pd.to_datetime(joined["on_time"]).map(second_of_day)
    joined["on_end"] = pd.to_datetime(joined["on_edge_end_time"]).map(second_of_day)
    joined["gap"] = joined["on_end"] - joined["on"]

    edges_by_file: dict[str, list[int]] = {}
    for item in manifest.itertuples(index=False):
        seconds = []
        for column in ("on_time", "on_edge_end_time", "off_time", "session_end_time"):
            value = getattr(item, column, None)
            if isinstance(value, str) and len(value) >= 19:
                stamp = pd.Timestamp(value)
                seconds.append(stamp.hour * 3600 + stamp.minute * 60 + stamp.second)
        edges_by_file.setdefault(item.raw_file, []).extend(seconds)

    rng = np.random.default_rng(RNG_SEED)
    by_file: dict[str, dict] = {}
    for item in joined.itertuples(index=False):
        gap = int(item.gap)
        if gap < 0 or gap > 3600:
            continue
        edges = np.array(sorted(set(edges_by_file.get(item.raw_file, []))), dtype=float)
        candidates = np.arange(3600, 86400 - 3600 - gap, 60, dtype=float)
        if len(edges):
            start_distance = np.abs(candidates[:, None] - edges[None, :]).min(axis=1)
            end_distance = np.abs((candidates + gap)[:, None] - edges[None, :]).min(axis=1)
            candidates = candidates[
                (start_distance >= GUARD_S) & (end_distance >= GUARD_S)
            ]
        if len(candidates) < CONTROLS_PER_SESSION:
            continue
        chosen = rng.choice(candidates, size=CONTROLS_PER_SESSION, replace=False)
        entry = by_file.setdefault(
            item.raw_file, {"raw_file": item.raw_file, "station_code": item.station_code, "controls": []}
        )
        for value in chosen:
            entry["controls"].append(
                {"session_uid": item.session_uid, "gap": gap, "second": int(value)}
            )

    tasks = list(by_file.values())
    print(f"gap-matched control tasks: {len(tasks)} files, "
          f"{sum(len(t['controls']) for t in tasks)} control windows")

    records = []
    done = 0
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(process_file, task) for task in tasks]
        for future in as_completed(futures):
            records.extend(future.result())
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(tasks)}", flush=True)

    control = pd.DataFrame(records)
    control.to_csv(OUT_DIR / "control_window_steps_gapmatched.csv", index=False)

    valid = ["valid_fraction_Z", "valid_fraction_H", "valid_fraction_D"]
    event = event[event[valid].min(axis=1) >= 0.8]
    control = control[control[valid].min(axis=1) >= 0.8]

    summary = {
        "n_event": int(len(event)),
        "n_control": int(len(control)),
        "controls_per_session": CONTROLS_PER_SESSION,
        "guard_s": GUARD_S,
        "median_gap_s": int(control["gap_s"].median()),
    }
    for name in ("Z", "H", "D"):
        e_step = event[f"step_{name}"].abs()
        c_step = control[f"step_{name}"].abs()
        e_hit = e_step >= np.maximum(0.5, 3.0 * event[f"sigma_{name}"])
        c_hit = c_step >= np.maximum(0.5, 3.0 * control[f"sigma_{name}"])
        summary[name] = {
            "event_median_abs_step_nT": round(float(e_step.median()), 4),
            "control_median_abs_step_nT": round(float(c_step.median()), 4),
            "contrast": round(float(e_step.median() / c_step.median()), 2),
            "event_fire_pct": round(float(e_hit.mean() * 100), 1),
            "control_fire_pct": round(float(c_hit.mean() * 100), 1),
            "event_median_snr": round(float(event[f"snr_{name}"].median()), 2),
            "control_median_snr": round(float(control[f"snr_{name}"].median()), 2),
        }
    (OUT_DIR / "control_comparison_gapmatched.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
