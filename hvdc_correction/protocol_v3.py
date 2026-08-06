"""Run the frozen protocol-v3 paired-session correction workflow.

The v3 cohort is defined from input information only. This runner applies one
implementation to every selected complete paired session and writes:

* one session-level result row per input session;
* one row per released stable plateau run;
* one raw-file inventory with SHA-256 hashes; and
* a machine-readable run lock containing code, configuration, and input hashes.

The historical A-lite products are never modified. Part files make the run
resumable at the raw station-day level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import percentile_filter, uniform_filter1d
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT
    / "outputs"
    / "canonical"
    / "complete_paired_sessions_2024_2025.csv"
)
DEFAULT_TRANSITION_INVENTORY = (
    ROOT / "outputs" / "canonical" / "transition_inventory_all_years.csv"
)
DEFAULT_CONFIG = ROOT / "configs" / "protocol_v3.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "protocol_v3"
NOISE_FLOOR_PATH = (
    ROOT / "outputs" / "tierB_shared" / "station_noise_floor_combined.csv"
)
CATALOG_2024 = ROOT / "outputs" / "hvdc" / "events_master_2024_v0.1.csv"
CATALOG_2025 = (
    ROOT / "outputs" / "rebuttal_ammo" / "events_2025_TRACEABLE_repaired.csv"
)

RESULT_COLUMNS = [
    "protocol_version",
    "session_uid",
    "operation_uid",
    "year",
    "station_code",
    "station_name",
    "line_id",
    "line_name",
    "event_id",
    "date",
    "raw_file",
    "raw_file_sha256",
    "raw_rows",
    "raw_duplicate_seconds",
    "raw_out_of_order_seconds",
    "on_time",
    "on_edge_end_time",
    "off_time",
    "session_end_time",
    "duration_min",
    "A_ref_nT",
    "A_off_ref_nT",
    "balance_abs_nT",
    "catalog_strong",
    "catalog_weak",
    "amplitude_regime",
    "seven_matched_station",
    "stress23_station",
    "z_noise_floor_nT",
    "costation_count",
    "resolved_costation_count",
    "max_resolved_costation_abs_A_ref_nT",
    "cross_station_weak_coupling",
    "cross_station_over_candidate",
    "cross_station_guard_withhold",
    "catalog_overlap_count",
    "on_edge_quality",
    "off_edge_quality",
    "pre_median_nT",
    "on_post_median_nT",
    "off_pre_median_nT",
    "post_median_nT",
    "pre_mad_nT",
    "on_post_mad_nT",
    "off_pre_mad_nT",
    "post_mad_nT",
    "pre_drift_nT_per_120s",
    "on_post_drift_nT_per_120s",
    "off_pre_drift_nT_per_120s",
    "post_drift_nT_per_120s",
    "A_baseline_nT",
    "A_baseline_off_nT",
    "identifiability_threshold_nT",
    "high_frequency_sigma_nT",
    "stable_spread_threshold_nT",
    "background_enabled",
    "background_slope_nT_per_s",
    "beta0_nT",
    "A_fit_nT",
    "fitted_on_time",
    "fitted_off_time",
    "fitted_on_shift_s",
    "fitted_off_shift_s",
    "fitted_tau_s",
    "fit_R2",
    "fit_success",
    "edge_window_start_time",
    "edge_window_end_time",
    "edge_window_valid_samples",
    "A_edge_nT",
    "amp_ratio_abs",
    "sign_agreement",
    "released_run_count",
    "released_primary_A_j_nT",
    "flag1_local_RMS_nT",
    "n_in_session",
    "n_flag1",
    "n_flag2",
    "n_flag3",
    "n_flag4",
    "flag1_sample_fraction",
    "final_route",
    "route_stage",
    "decision_reason",
    "code_sha256",
    "config_sha256",
    "manifest_sha256",
]

RUN_COLUMNS = [
    "protocol_version",
    "session_uid",
    "run_index",
    "run_start_time",
    "run_end_time",
    "run_start_second",
    "run_end_second",
    "run_samples",
    "valid_samples",
    "A_j_nT",
    "background_at_midpoint_nT",
    "local_residual_RMS_nT",
    "stable_spread_threshold_nT",
    "sign_matches_A_ref",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_part_id(
    raw_file: str,
    session_uids: list[str],
    context: dict[str, Any],
) -> str:
    payload = "|".join(
        [
            raw_file,
            context["code_sha256"],
            context["config_sha256"],
            context["manifest_sha256"],
            *sorted(session_uids),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def finite_number(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan
    center = np.median(values)
    return float(np.median(np.abs(values - center)))


def linear_slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if valid.sum() < 5:
        return math.nan
    x = np.arange(len(values), dtype=float)[valid]
    y = values[valid]
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.dot(x, x))
    return float(np.dot(x, y) / denominator) if denominator > 0 else math.nan


def r2_score(observed: np.ndarray, fitted: np.ndarray) -> float:
    valid = np.isfinite(observed) & np.isfinite(fitted)
    if valid.sum() < 3:
        return math.nan
    y = observed[valid]
    yhat = fitted[valid]
    total = float(np.sum((y - y.mean()) ** 2))
    if total <= 0:
        return math.nan
    return float(1.0 - np.sum((y - yhat) ** 2) / total)


def second_of_day(value: pd.Timestamp) -> int:
    return int(value.hour * 3600 + value.minute * 60 + value.second)


def iso_at(date: str, second: float) -> str:
    if not math.isfinite(second):
        return ""
    base = pd.to_datetime(str(date), format="%Y%m%d")
    return str(base + pd.to_timedelta(float(second), unit="s"))


def window_values(z: np.ndarray, start: int, end: int) -> np.ndarray:
    start = max(0, int(start))
    end = min(len(z), int(end))
    return z[start:end] if end > start else np.array([], dtype=float)


def read_raw_z(path_text: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Read Z onto an exact 86,400-s grid while hashing the source file."""
    path = Path(path_text)
    digest = hashlib.sha256()
    z = np.full(86400, np.nan, dtype=np.float64)
    rows = 0
    duplicate_seconds = 0
    out_of_order_seconds = 0
    previous_second = -1
    with path.open("rb") as handle:
        header = handle.readline()
        digest.update(header)
        columns = header.split()
        if len(columns) < 3:
            raise ValueError("raw header has fewer than three columns")
        component_columns = columns[2:]
        upper = [item.upper() for item in component_columns]
        if b"Z" not in upper:
            raise ValueError("raw file has no Z column")
        z_index = 2 + upper.index(b"Z")
        for raw_line in handle:
            digest.update(raw_line)
            parts = raw_line.split()
            if len(parts) <= z_index or len(parts) < 2:
                continue
            clock = parts[1]
            if len(clock) < 8:
                continue
            try:
                hour = int(clock[0:2])
                minute = int(clock[3:5])
                second = int(clock[6:8])
                index = hour * 3600 + minute * 60 + second
                value = float(parts[z_index])
            except (ValueError, IndexError):
                continue
            if not 0 <= index < 86400:
                continue
            rows += 1
            if index < previous_second:
                out_of_order_seconds += 1
            previous_second = index
            if math.isfinite(z[index]):
                duplicate_seconds += 1
                continue
            if value != 999999 and math.isfinite(value):
                z[index] = value
    metadata = {
        "raw_file_sha256": digest.hexdigest(),
        "raw_rows": rows,
        "raw_duplicate_seconds": duplicate_seconds,
        "raw_out_of_order_seconds": out_of_order_seconds,
        "raw_valid_z": int(np.isfinite(z).sum()),
    }
    return z, metadata


def baseline_window_stats(
    values: np.ndarray,
    station_floor: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    window = config["window_geometry"]
    qc = config["baseline_qc"]
    valid = np.isfinite(values)
    valid_n = int(valid.sum())
    fraction = valid_n / len(values) if len(values) else 0.0
    median = float(np.median(values[valid])) if valid_n >= 5 else math.nan
    spread = mad(values)
    slope = linear_slope(values)
    drift = (
        slope * float(window["baseline_length_s"])
        if math.isfinite(slope)
        else math.nan
    )
    truncated = fraction < float(window["minimum_valid_fraction"])
    noisy = (
        math.isfinite(spread)
        and spread > float(qc["noise_floor_multiple"]) * station_floor
    )
    drifting = (
        math.isfinite(drift)
        and abs(drift) > float(qc["drift_limit_nT_per_120s"])
    )
    if truncated:
        quality = "truncated"
    elif noisy and drifting:
        quality = "drift|noisy"
    elif noisy:
        quality = "noisy"
    elif drifting:
        quality = "drift"
    else:
        quality = "ok"
    return {
        "median": median,
        "mad": spread,
        "slope": slope,
        "drift": drift,
        "valid_n": valid_n,
        "valid_fraction": fraction,
        "quality": quality,
    }


def edge_quality(pre: dict[str, Any], post: dict[str, Any]) -> str:
    qualities = {pre["quality"], post["quality"]}
    if "truncated" in qualities:
        return "truncated"
    noisy = "noisy" in qualities or "drift|noisy" in qualities
    drifting = "drift" in qualities or "drift|noisy" in qualities
    if noisy and drifting:
        return "drift|noisy"
    if noisy:
        return "noisy"
    if drifting:
        return "drift"
    return "ok"


def ramp_boxcar_shape(
    t: np.ndarray,
    t_on: float,
    t_off: float,
    tau: float,
) -> np.ndarray:
    tau = max(float(tau), 1.0)
    rise = np.clip((t - t_on) / tau, 0.0, 1.0)
    fall = np.clip((t - t_off) / tau, 0.0, 1.0)
    return rise - fall


def fit_geometry(
    t: np.ndarray,
    y: np.ndarray,
    background: np.ndarray,
    b0: float,
    amplitude0: float,
    on0: float,
    off0: float,
    config: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    fit = config["geometry_fit"]
    lower_multiple = float(fit["amplitude_lower_multiple"])
    upper_multiple = float(fit["amplitude_upper_multiple"])
    amp_lo, amp_hi = sorted(
        [lower_multiple * amplitude0, upper_multiple * amplitude0]
    )
    if amp_hi - amp_lo < 1.0:
        amp_lo = amplitude0 - abs(amplitude0) - 1.0
        amp_hi = amplitude0 + abs(amplitude0) + 1.0
    onset_bound = float(fit["onset_bound_s"])
    beta_bound = float(fit["nuisance_intercept_bound_nT"])
    lower = np.array(
        [
            -beta_bound,
            amp_lo,
            on0 - onset_bound,
            off0 - onset_bound,
            float(fit["tau_min_s"]),
        ],
        dtype=float,
    )
    upper = np.array(
        [
            beta_bound,
            amp_hi,
            on0 + onset_bound,
            off0 + onset_bound,
            float(fit["tau_max_s"]),
        ],
        dtype=float,
    )
    initial = np.array(
        [
            0.0,
            amplitude0,
            on0,
            off0,
            float(fit["tau_initial_s"]),
        ],
        dtype=float,
    )
    initial = np.clip(initial, lower + 1e-6, upper - 1e-6)
    valid = np.isfinite(y) & np.isfinite(background)
    if valid.sum() < 40:
        return {}, False
    tv = t[valid]
    yv = y[valid]
    bv = background[valid]

    def residual(parameters: np.ndarray) -> np.ndarray:
        beta0, amplitude, on, off, tau = parameters
        predicted = (
            bv
            + beta0
            + amplitude * ramp_boxcar_shape(tv, on, off, tau)
        )
        return predicted - yv

    try:
        solved = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            method="trf",
            max_nfev=int(fit["maximum_function_evaluations"]),
        )
    except Exception:
        return {}, False
    beta0, amplitude, on, off, tau = solved.x
    predicted = (
        background
        + beta0
        + amplitude * ramp_boxcar_shape(t, on, off, tau)
    )
    result = {
        "beta0": float(beta0),
        "A_fit": float(amplitude),
        "fitted_on": float(on),
        "fitted_off": float(off),
        "tau": float(tau),
        "R2": r2_score(y, predicted),
        "optimizer_success": bool(solved.success),
        "optimizer_status": int(solved.status),
        "optimizer_cost": float(solved.cost),
    }
    return result, bool(solved.success)


def contiguous_runs(mask: np.ndarray, minimum_length: int) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[indices[0], indices[breaks + 1]]
    ends = np.r_[indices[breaks], indices[-1]]
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends)
        if end - start + 1 >= minimum_length
    ]


def stable_runs(
    t_event: np.ndarray,
    y_event: np.ndarray,
    background_event: np.ndarray,
    fitted_on: float,
    fitted_off: float,
    tau: float,
    threshold: float,
    a_ref: float,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    gate = config["stable_run_gate"]
    size = int(gate["centered_window_s"])
    minimum_run = int(gate["minimum_run_s"])
    valid = np.isfinite(y_event)
    if valid.sum() < minimum_run:
        return []
    valid_indices = np.flatnonzero(valid)
    filled = np.interp(
        np.arange(len(y_event), dtype=float),
        valid_indices.astype(float),
        y_event[valid],
    )
    low = percentile_filter(
        filled,
        percentile=float(gate["lower_percentile"]),
        size=size,
        mode="nearest",
    )
    high = percentile_filter(
        filled,
        percentile=float(gate["upper_percentile"]),
        size=size,
        mode="nearest",
    )
    valid_fraction = uniform_filter1d(
        valid.astype(float),
        size=size,
        mode="constant",
        cval=0.0,
    )
    candidate = (
        (t_event >= fitted_on + tau)
        & (t_event < fitted_off)
        & valid
        & (
            valid_fraction
            >= float(gate["minimum_valid_fraction"])
        )
        & ((high - low) <= threshold)
    )
    result: list[dict[str, Any]] = []
    for start, end in contiguous_runs(candidate, minimum_run):
        run_values = y_event[start : end + 1]
        run_background = background_event[start : end + 1]
        run_valid = np.isfinite(run_values) & np.isfinite(run_background)
        if run_valid.sum() < minimum_run:
            continue
        midpoint = 0.5 * (t_event[start] + t_event[end])
        background_midpoint = float(
            np.interp(midpoint, t_event, background_event)
        )
        amplitude = float(np.median(run_values[run_valid]) - background_midpoint)
        x = t_event[start : end + 1][run_valid]
        corrected = run_values[run_valid] - amplitude
        if len(x) >= 2:
            x_centered = x - x.mean()
            denominator = float(np.dot(x_centered, x_centered))
            slope = (
                float(
                    np.dot(
                        x_centered,
                        corrected - corrected.mean(),
                    )
                    / denominator
                )
                if denominator > 0
                else 0.0
            )
            detrended = corrected - (
                corrected.mean() + slope * x_centered
            )
            residual_rms = float(np.sqrt(np.mean(detrended**2)))
        else:
            residual_rms = math.nan
        if not math.isfinite(residual_rms) or residual_rms > threshold:
            continue
        result.append(
            {
                "start_index": start,
                "end_index": end,
                "start_second": float(t_event[start]),
                "end_second": float(t_event[end]),
                "run_samples": int(end - start + 1),
                "valid_samples": int(run_valid.sum()),
                "A_j": amplitude,
                "background_midpoint": background_midpoint,
                "local_residual_RMS": residual_rms,
                "sign_matches_A_ref": bool(
                    np.sign(amplitude) == np.sign(a_ref)
                ),
            }
        )
    return result


def blank_result(
    row: dict[str, Any],
    context: dict[str, Any],
    raw_metadata: dict[str, Any],
) -> dict[str, Any]:
    result = {column: "" for column in RESULT_COLUMNS}
    for column in (
        "session_uid",
        "operation_uid",
        "year",
        "station_code",
        "station_name",
        "line_id",
        "line_name",
        "event_id",
        "date",
        "raw_file",
        "on_time",
        "on_edge_end_time",
        "off_time",
        "session_end_time",
        "duration_min",
        "A_ref_nT",
        "A_off_ref_nT",
        "balance_abs_nT",
        "catalog_strong",
        "catalog_weak",
        "amplitude_regime",
        "seven_matched_station",
        "stress23_station",
        "z_noise_floor_nT",
        "costation_count",
        "resolved_costation_count",
        "max_resolved_costation_abs_A_ref_nT",
        "cross_station_weak_coupling",
        "cross_station_guard_withhold",
        "catalog_overlap_count",
    ):
        if column in row:
            result[column] = row[column]
    result.update(raw_metadata)
    result.update(
        {
            "protocol_version": context["protocol_version"],
            "code_sha256": context["code_sha256"],
            "config_sha256": context["config_sha256"],
            "manifest_sha256": context["manifest_sha256"],
        }
    )
    return result


def finalize_nonrelease(
    result: dict[str, Any],
    route: str,
    stage: str,
    reason: str,
    n_session: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if result.get("fit_success", "") in ("", None):
        result["fit_success"] = False
    result.update(
        {
            "released_run_count": 0,
            "n_in_session": n_session,
            "n_flag1": 0,
            "n_flag2": 0,
            "n_flag3": n_session if route == "weak_abstention" else 0,
            "n_flag4": 0 if route == "weak_abstention" else n_session,
            "flag1_sample_fraction": 0.0,
            "final_route": route,
            "route_stage": stage,
            "decision_reason": reason,
        }
    )
    return result, []


def process_session(
    row: dict[str, Any],
    z: np.ndarray,
    raw_metadata: dict[str, Any],
    config: dict[str, Any],
    context: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = blank_result(row, context, raw_metadata)
    window = config["window_geometry"]
    qc = config["baseline_qc"]
    fit_config = config["geometry_fit"]
    edge_config = config["edge_local_amplitude"]
    ident = config["identifiability_gate"]
    stable_config = config["stable_run_gate"]
    date = str(row["date"])
    on_ts = pd.Timestamp(row["on_time"])
    on_end_ts = pd.Timestamp(row["on_edge_end_time"])
    off_ts = pd.Timestamp(row["off_time"])
    off_end_ts = pd.Timestamp(row["session_end_time"])
    on = second_of_day(on_ts)
    on_end = second_of_day(on_end_ts)
    off = second_of_day(off_ts)
    off_end = second_of_day(off_end_ts)
    n_session = max(0, off_end - on + 1)
    length = int(window["baseline_length_s"])
    buffer = int(window["baseline_buffer_s"])
    station_floor = finite_number(row["z_noise_floor_nT"], 0.05)

    on_pre_values = window_values(z, on - buffer - length, on - buffer)
    on_post_values = window_values(
        z, on_end + buffer, on_end + buffer + length
    )
    off_pre_values = window_values(
        z, off - buffer - length, off - buffer
    )
    off_post_values = window_values(
        z, off_end + buffer, off_end + buffer + length
    )
    on_pre = baseline_window_stats(on_pre_values, station_floor, config)
    on_post = baseline_window_stats(on_post_values, station_floor, config)
    off_pre = baseline_window_stats(off_pre_values, station_floor, config)
    off_post = baseline_window_stats(off_post_values, station_floor, config)
    on_quality = edge_quality(on_pre, on_post)
    off_quality = edge_quality(off_pre, off_post)
    a_baseline = on_post["median"] - on_pre["median"]
    a_baseline_off = off_post["median"] - off_pre["median"]
    effective_noise = max(
        finite_number(on_pre["mad"], -math.inf),
        finite_number(on_post["mad"], -math.inf),
        finite_number(off_pre["mad"], -math.inf),
        finite_number(off_post["mad"], -math.inf),
    )
    threshold_id = max(
        float(ident["absolute_floor_nT"]),
        float(ident["station_noise_floor_multiple"]) * station_floor,
    )
    differences = np.diff(on_pre_values[np.isfinite(on_pre_values)])
    sigma_hf = (
        1.4826 * mad(differences) / math.sqrt(2.0)
        if len(differences)
        else math.nan
    )
    stable_threshold = max(
        float(stable_config["absolute_spread_floor_nT"]),
        float(stable_config["high_frequency_noise_multiple"])
        * finite_number(sigma_hf, 0.0),
    )
    result.update(
        {
            "on_edge_quality": on_quality,
            "off_edge_quality": off_quality,
            "pre_median_nT": on_pre["median"],
            "on_post_median_nT": on_post["median"],
            "off_pre_median_nT": off_pre["median"],
            "post_median_nT": off_post["median"],
            "pre_mad_nT": on_pre["mad"],
            "on_post_mad_nT": on_post["mad"],
            "off_pre_mad_nT": off_pre["mad"],
            "post_mad_nT": off_post["mad"],
            "pre_drift_nT_per_120s": on_pre["drift"],
            "on_post_drift_nT_per_120s": on_post["drift"],
            "off_pre_drift_nT_per_120s": off_pre["drift"],
            "post_drift_nT_per_120s": off_post["drift"],
            "A_baseline_nT": a_baseline,
            "A_baseline_off_nT": a_baseline_off,
            "identifiability_threshold_nT": threshold_id,
            "high_frequency_sigma_nT": sigma_hf,
            "stable_spread_threshold_nT": stable_threshold,
        }
    )
    a_ref = finite_number(row["A_ref_nT"])
    guard = config["cross_station_guard"]
    cross_station_over_candidate = bool(
        as_bool(row["catalog_weak"])
        and math.isfinite(a_baseline)
        and math.isfinite(a_ref)
        and abs(a_baseline)
        > abs(a_ref) + float(guard["local_step_excess_margin_nT"])
    )
    cross_station_withhold = bool(
        cross_station_over_candidate
        and as_bool(row["cross_station_weak_coupling"])
    )
    result.update(
        {
            "cross_station_over_candidate": cross_station_over_candidate,
            "cross_station_guard_withhold": cross_station_withhold,
        }
    )

    if on_quality == "truncated" or off_quality == "truncated":
        return finalize_nonrelease(
            result,
            "invalid_window",
            "invalid_window",
            "edge_baseline_window_truncated",
            n_session,
        )
    if on_end >= off:
        return finalize_nonrelease(
            result,
            "structural_misfit",
            "catalog_overlap",
            "catalog_on_edge_interval_reaches_or_overlaps_off_edge",
            n_session,
        )
    if effective_noise > float(qc["diagnostic_only_effective_noise_nT"]):
        return finalize_nonrelease(
            result,
            "invalid_window",
            "extreme_effective_noise_cutoff",
            "effective_noise_above_diagnostic_cutoff",
            n_session,
        )
    if int(row["catalog_overlap_count"]) > 0:
        return finalize_nonrelease(
            result,
            "structural_misfit",
            "catalog_overlap",
            "other_catalog_transition_overlaps_fit_window",
            n_session,
        )
    if cross_station_withhold:
        return finalize_nonrelease(
            result,
            "weak_abstention",
            "cross_station_abstention",
            "local_catalog_response_weak_relative_to_resolved_costation",
            n_session,
        )
    if not math.isfinite(a_baseline) or abs(a_baseline) < threshold_id:
        return finalize_nonrelease(
            result,
            "weak_abstention",
            "local_identifiability_abstention",
            "measured_local_step_below_identifiability_floor",
            n_session,
        )

    fit_start = max(0, on - int(window["fit_margin_s"]))
    fit_end = min(86400, off_end + int(window["fit_margin_s"]) + 1)
    t = np.arange(fit_start, fit_end, dtype=float)
    y = z[fit_start:fit_end]
    b0 = on_pre["median"]
    pre_center = on - buffer - 0.5 * length
    post_center = off_end + buffer + 0.5 * length
    background_enabled = (
        finite_number(row["duration_min"], 0.0)
        > float(fit_config["long_session_threshold_min"])
    )
    slope = 0.0
    if background_enabled and post_center > pre_center:
        slope = (off_post["median"] - on_pre["median"]) / (
            post_center - pre_center
        )
        clip = float(fit_config["background_slope_clip_nT_per_s"])
        slope = float(np.clip(slope, -clip, clip))
    background = b0 + slope * (t - pre_center)
    result.update(
        {
            "background_enabled": background_enabled,
            "background_slope_nT_per_s": slope,
        }
    )
    fitted, fit_ok = fit_geometry(
        t,
        y,
        background,
        b0,
        a_baseline,
        float(on),
        float(off),
        config,
    )
    if not fit_ok:
        return finalize_nonrelease(
            result,
            "structural_misfit",
            "geometry_fit",
            "bounded_geometry_fit_failed",
            n_session,
        )
    fitted_on = fitted["fitted_on"]
    fitted_off = fitted["fitted_off"]
    tau = fitted["tau"]
    result.update(
        {
            "beta0_nT": fitted["beta0"],
            "A_fit_nT": fitted["A_fit"],
            "fitted_on_time": iso_at(date, fitted_on),
            "fitted_off_time": iso_at(date, fitted_off),
            "fitted_on_shift_s": fitted_on - on,
            "fitted_off_shift_s": fitted_off - off,
            "fitted_tau_s": tau,
            "fit_R2": fitted["R2"],
            "fit_success": True,
        }
    )
    if fitted_off <= fitted_on + tau:
        return finalize_nonrelease(
            result,
            "structural_misfit",
            "geometry_fit",
            "fitted_geometry_has_no_positive_plateau",
            n_session,
        )

    edge_start = (
        fitted_on
        + tau
        + float(edge_config["post_ramp_buffer_s"])
    )
    edge_end = min(
        edge_start + float(edge_config["maximum_window_s"]),
        fitted_off - float(edge_config["off_edge_buffer_s"]),
    )
    edge_mask = (t >= edge_start) & (t <= edge_end) & np.isfinite(y)
    edge_valid_n = int(edge_mask.sum())
    y_background_removed = y - slope * (t - pre_center)
    a_edge = (
        float(np.median(y_background_removed[edge_mask]) - b0)
        if edge_valid_n >= int(edge_config["minimum_valid_samples"])
        else math.nan
    )
    result.update(
        {
            "edge_window_start_time": iso_at(date, edge_start),
            "edge_window_end_time": iso_at(date, edge_end),
            "edge_window_valid_samples": edge_valid_n,
            "A_edge_nT": a_edge,
            "amp_ratio_abs": (
                abs(a_edge) / abs(a_ref)
                if math.isfinite(a_edge)
                and math.isfinite(a_ref)
                and a_ref != 0
                else math.nan
            ),
            "sign_agreement": (
                bool(np.sign(a_edge) == np.sign(a_ref))
                if math.isfinite(a_edge) and math.isfinite(a_ref)
                else ""
            ),
        }
    )
    if not math.isfinite(a_edge):
        return finalize_nonrelease(
            result,
            "structural_misfit",
            "edge_local_window",
            "edge_local_window_has_fewer_than_five_valid_samples",
            n_session,
        )

    event_start = max(0, on)
    event_end = min(86399, off_end)
    t_event = np.arange(event_start, event_end + 1, dtype=float)
    y_event = z[event_start : event_end + 1]
    background_event = b0 + slope * (t_event - pre_center)
    accepted_runs = stable_runs(
        t_event,
        y_event,
        background_event,
        fitted_on,
        fitted_off,
        tau,
        stable_threshold,
        a_ref,
        config,
    )
    if not accepted_runs:
        return finalize_nonrelease(
            result,
            "structural_misfit",
            "stable_run_and_residual_gate",
            "no_stable_plateau_run_passed_local_residual_gate",
            n_session,
        )

    flag1 = np.zeros(len(t_event), dtype=bool)
    run_rows: list[dict[str, Any]] = []
    for index, run in enumerate(accepted_runs, start=1):
        flag1[run["start_index"] : run["end_index"] + 1] = True
        run_rows.append(
            {
                "protocol_version": context["protocol_version"],
                "session_uid": row["session_uid"],
                "run_index": index,
                "run_start_time": iso_at(date, run["start_second"]),
                "run_end_time": iso_at(date, run["end_second"]),
                "run_start_second": run["start_second"],
                "run_end_second": run["end_second"],
                "run_samples": run["run_samples"],
                "valid_samples": run["valid_samples"],
                "A_j_nT": run["A_j"],
                "background_at_midpoint_nT": run["background_midpoint"],
                "local_residual_RMS_nT": run["local_residual_RMS"],
                "stable_spread_threshold_nT": stable_threshold,
                "sign_matches_A_ref": run["sign_matches_A_ref"],
            }
        )
    flag2 = np.zeros(len(t_event), dtype=bool)
    first = accepted_runs[0]["start_index"]
    last = accepted_runs[-1]["end_index"]
    flag2[:first] = True
    flag2[last + 1 :] = True
    flag4 = ~(flag1 | flag2)
    n1 = int(flag1.sum())
    n2 = int(flag2.sum())
    n4 = int(flag4.sum())
    rms_values = [run["local_residual_RMS"] for run in accepted_runs]
    result.update(
        {
            "released_run_count": len(accepted_runs),
            "released_primary_A_j_nT": accepted_runs[0]["A_j"],
            "flag1_local_RMS_nT": float(np.median(rms_values)),
            "n_in_session": len(t_event),
            "n_flag1": n1,
            "n_flag2": n2,
            "n_flag3": 0,
            "n_flag4": n4,
            "flag1_sample_fraction": n1 / len(t_event),
            "final_route": "released",
            "route_stage": "released",
            "decision_reason": "one_or_more_stable_plateau_runs_released",
        }
    )
    return result, run_rows


def process_raw_group(task: dict[str, Any]) -> dict[str, Any]:
    raw_file = task["raw_file"]
    try:
        z, raw_metadata = read_raw_z(raw_file)
        raw_error = ""
    except Exception as exc:
        z = np.full(86400, np.nan, dtype=float)
        raw_metadata = {
            "raw_file_sha256": "",
            "raw_rows": 0,
            "raw_duplicate_seconds": 0,
            "raw_out_of_order_seconds": 0,
            "raw_valid_z": 0,
        }
        raw_error = f"{type(exc).__name__}: {exc}"
    results: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for row in task["rows"]:
        if raw_error:
            result = blank_result(row, task["context"], raw_metadata)
            n_session = max(
                0,
                second_of_day(pd.Timestamp(row["session_end_time"]))
                - second_of_day(pd.Timestamp(row["on_time"]))
                + 1,
            )
            result, session_runs = finalize_nonrelease(
                result,
                "invalid_window",
                "invalid_window",
                f"raw_file_read_failed:{raw_error}",
                n_session,
            )
        else:
            result, session_runs = process_session(
                row,
                z,
                raw_metadata,
                task["config"],
                task["context"],
            )
        results.append(result)
        runs.extend(session_runs)
    return {
        "part_id": task["part_id"],
        "raw_file": raw_file,
        "results": results,
        "runs": runs,
        "raw_metadata": raw_metadata,
        "raw_error": raw_error,
    }


def load_noise_floors(path: Path = NOISE_FLOOR_PATH) -> dict[int, float]:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["station_code"] = pd.to_numeric(
        frame["station_code"], errors="coerce"
    )
    frame["z_noise_floor_nT"] = pd.to_numeric(
        frame["z_noise_floor_nT"], errors="coerce"
    )
    return {
        int(row["station_code"]): float(row["z_noise_floor_nT"])
        for _, row in frame.dropna(
            subset=["station_code", "z_noise_floor_nT"]
        ).iterrows()
    }


def enrich_cross_station(
    manifest: pd.DataFrame,
    floors: dict[int, float],
    config: dict[str, Any],
) -> pd.DataFrame:
    result = manifest.copy()
    factor = float(
        config["cross_station_guard"]["corroboration_factor"]
    )
    result["z_noise_floor_nT"] = result["station_code"].map(floors)
    if result["z_noise_floor_nT"].isna().any():
        missing = sorted(
            result.loc[
                result["z_noise_floor_nT"].isna(), "station_code"
            ].unique()
        )
        raise RuntimeError(f"Missing Z noise floors for stations: {missing}")
    records: list[dict[str, Any]] = []
    for operation_uid, group in result.groupby("operation_uid", sort=False):
        group_rows = list(group.to_dict("records"))
        for local in group_rows:
            local_station = int(local["station_code"])
            others = [
                item
                for item in group_rows
                if int(item["station_code"]) != local_station
            ]
            resolved = [
                item
                for item in others
                if abs(float(item["A_ref_nT"]))
                >= floors[int(item["station_code"])]
            ]
            maximum = (
                max(abs(float(item["A_ref_nT"])) for item in resolved)
                if resolved
                else math.nan
            )
            local_scale = max(
                abs(float(local["A_ref_nT"])),
                floors[local_station],
            )
            records.append(
                {
                    "session_uid": local["session_uid"],
                    "costation_count": len(others),
                    "resolved_costation_count": len(resolved),
                    "max_resolved_costation_abs_A_ref_nT": maximum,
                    "cross_station_weak_coupling": bool(
                        resolved and maximum >= factor * local_scale
                    ),
                }
            )
    return result.merge(
        pd.DataFrame(records), on="session_uid", how="left", validate="1:1"
    )


def transition_inventory(
    inventory_path: Path | None = None,
) -> pd.DataFrame:
    if inventory_path is not None:
        inventory = pd.read_csv(inventory_path, encoding="utf-8-sig")
        required = {
            "year",
            "station_code",
            "operation_uid",
            "start",
            "end",
        }
        missing = required - set(inventory.columns)
        if missing:
            raise RuntimeError(
                "Transition inventory is missing columns: "
                + ", ".join(sorted(missing))
            )
        inventory["year"] = pd.to_numeric(
            inventory["year"], errors="coerce"
        )
        inventory["station_code"] = pd.to_numeric(
            inventory["station_code"], errors="coerce"
        )
        inventory["start"] = pd.to_datetime(
            inventory["start"], errors="coerce"
        )
        inventory["end"] = pd.to_datetime(
            inventory["end"], errors="coerce"
        )
        return inventory.dropna(
            subset=[
                "year",
                "station_code",
                "operation_uid",
                "start",
                "end",
            ]
        ).drop_duplicates()

    source24 = pd.read_csv(CATALOG_2024, encoding="utf-8-sig")
    source24 = source24.rename(
        columns={
            "\u6765\u6e90\u6587\u4ef6": "source_file",
            "\u7ebf\u8def\u7f16\u53f7": "line_id",
            "\u4e8b\u4ef6\u7f16\u53f7": "event_no",
            "\u53f0\u7ad9\u4ee3\u7801": "station_code",
            "\u5f00\u59cb\u65f6\u95f4": "start",
            "\u7ed3\u675f\u65f6\u95f4": "end",
        }
    )
    source24["start"] = pd.to_datetime(source24["start"], errors="coerce")
    source24["end"] = pd.to_datetime(source24["end"], errors="coerce")
    for column in ("line_id", "event_no", "station_code"):
        source24[column] = pd.to_numeric(source24[column], errors="coerce")
    source24 = source24.dropna(
        subset=[
            "source_file",
            "line_id",
            "event_no",
            "station_code",
            "start",
            "end",
        ]
    ).copy()
    source24["operation_uid"] = source24.apply(
        lambda row: (
            f"2024|source={row['source_file']}"
            f"|line={int(round(row['line_id']))}"
            f"|event={int(round(row['event_no']))}"
        ),
        axis=1,
    )
    source24["year"] = 2024

    source25 = pd.read_csv(CATALOG_2025, encoding="utf-8-sig")
    for column in ("line_id", "event_no", "station_code"):
        source25[column] = pd.to_numeric(source25[column], errors="coerce")
    source25["start"] = pd.to_datetime(
        source25["start_time"], errors="coerce"
    )
    source25["end"] = pd.to_datetime(source25["end_time"], errors="coerce")
    source25 = source25.dropna(
        subset=["line_id", "event_no", "station_code", "start", "end"]
    ).copy()
    source25["date"] = source25["start"].dt.strftime("%Y%m%d")
    source25["operation_uid"] = source25.apply(
        lambda row: (
            f"2025|line={int(round(row['line_id']))}"
            f"|event={int(round(row['event_no']))}|date={row['date']}"
        ),
        axis=1,
    )
    source25["year"] = 2025
    return pd.concat(
        [
            source24[
                [
                    "year",
                    "station_code",
                    "operation_uid",
                    "start",
                    "end",
                ]
            ],
            source25[
                [
                    "year",
                    "station_code",
                    "operation_uid",
                    "start",
                    "end",
                ]
            ],
        ],
        ignore_index=True,
    ).drop_duplicates()


def enrich_catalog_overlap(
    manifest: pd.DataFrame,
    config: dict[str, Any],
    inventory: pd.DataFrame | None = None,
) -> pd.DataFrame:
    result = manifest.copy()
    if inventory is None:
        inventory = transition_inventory()
    eligible = set(result["station_code"].astype(int))
    inventory = inventory[
        inventory["station_code"].astype(int).isin(eligible)
    ].copy()
    inventory["date"] = inventory["start"].dt.strftime("%Y%m%d")
    grouped = {
        (int(station), str(date)): group
        for (station, date), group in inventory.groupby(
            ["station_code", "date"], sort=False
        )
    }
    margin = pd.Timedelta(
        seconds=float(
            config["catalog_overlap_guard"]["fit_window_margin_s"]
        )
    )
    counts: list[int] = []
    for row in result.itertuples(index=False):
        group = grouped.get((int(row.station_code), str(row.date)))
        if group is None:
            counts.append(0)
            continue
        start = pd.Timestamp(row.on_time) - margin
        end = pd.Timestamp(row.session_end_time) + margin
        overlaps = group[
            group["operation_uid"].ne(row.operation_uid)
            & group["start"].le(end)
            & group["end"].ge(start)
        ]
        counts.append(len(overlaps))
    result["catalog_overlap_count"] = counts
    return result


def write_part(
    part_dir: Path,
    payload: dict[str, Any],
) -> None:
    part_id = payload["part_id"]
    pd.DataFrame(payload["results"], columns=RESULT_COLUMNS).to_csv(
        part_dir / f"{part_id}_sessions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(payload["runs"], columns=RUN_COLUMNS).to_csv(
        part_dir / f"{part_id}_runs.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metadata = {
        "part_id": part_id,
        "raw_file": payload["raw_file"],
        **payload["raw_metadata"],
        "raw_error": payload["raw_error"],
        "session_count": len(payload["results"]),
        "released_run_count": len(payload["runs"]),
    }
    (part_dir / f"{part_id}_raw.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def part_complete(part_dir: Path, part_id: str) -> bool:
    return all(
        (part_dir / f"{part_id}_{suffix}").exists()
        for suffix in ("sessions.csv", "runs.csv", "raw.json")
    )


def combine_parts(
    selected: pd.DataFrame,
    part_dir: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    part_ids = list(selected["_part_id"].drop_duplicates())
    results = pd.concat(
        [
            pd.read_csv(
                part_dir / f"{part_id}_sessions.csv",
                encoding="utf-8-sig",
                low_memory=False,
            )
            for part_id in part_ids
        ],
        ignore_index=True,
    )
    run_frames = []
    for part_id in part_ids:
        frame = pd.read_csv(
            part_dir / f"{part_id}_runs.csv",
            encoding="utf-8-sig",
            low_memory=False,
        )
        if len(frame):
            run_frames.append(frame)
    runs = (
        pd.concat(run_frames, ignore_index=True)
        if run_frames
        else pd.DataFrame(columns=RUN_COLUMNS)
    )
    raw_rows = [
        json.loads(
            (part_dir / f"{part_id}_raw.json").read_text(encoding="utf-8")
        )
        for part_id in part_ids
    ]
    raw_inventory = pd.DataFrame(raw_rows)
    order = {
        session_uid: index
        for index, session_uid in enumerate(selected["session_uid"])
    }
    results["_order"] = results["session_uid"].map(order)
    results = results.sort_values("_order").drop(columns="_order")
    if len(results) != len(selected):
        raise RuntimeError(
            f"Result row count {len(results)} != selected {len(selected)}"
        )
    if results["session_uid"].duplicated().any():
        raise RuntimeError("Duplicate session_uid in v3 results")
    if set(results["session_uid"]) != set(selected["session_uid"]):
        raise RuntimeError("v3 results do not match selected manifest sessions")
    results.to_csv(
        output_dir / "protocol_v3_session_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    runs.to_csv(
        output_dir / "protocol_v3_stable_runs.csv",
        index=False,
        encoding="utf-8-sig",
    )
    raw_inventory.to_csv(
        output_dir / "protocol_v3_raw_file_inventory.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return results, runs, raw_inventory


def build_report(
    results: pd.DataFrame,
    runs: pd.DataFrame,
    selected: pd.DataFrame,
    elapsed: float,
) -> str:
    route = (
        results.groupby(["year", "final_route"])
        .size()
        .unstack(fill_value=0)
    )
    subset = (
        selected.groupby(["year", "amplitude_regime"])
        .size()
        .unstack(fill_value=0)
    )
    return f"""# Protocol v3.0 paired-session run

## Scope

- Input sessions: **{len(selected):,}**
- Raw station-day files: **{selected['raw_file'].nunique():,}**
- Released stable runs: **{len(runs):,}**
- Elapsed wall time: **{elapsed / 60.0:.1f} min**

## Input subsets

```
{subset.to_string()}
```

## Final routes

```
{route.to_string()}
```

## Implementation lock

The nuisance intercept is fitted only during geometry optimization. It is not
subtracted from the released record and is not used in the edge-local
amplitude. The edge-local amplitude and released run offsets are recomputed
from the observed local background.

This run uses input-defined v3 sessions. No fitted quantity determines cohort
membership.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--noise-floor-file",
        type=Path,
        default=NOISE_FLOOR_PATH,
    )
    parser.add_argument(
        "--transition-inventory-file",
        type=Path,
        default=DEFAULT_TRANSITION_INVENTORY,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seven-only", action="store_true")
    parser.add_argument("--year", type=int, action="append")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    part_dir = output_dir / "parts"
    output_dir.mkdir(parents=True, exist_ok=True)
    part_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = pd.read_csv(
        args.manifest,
        encoding="utf-8-sig",
        low_memory=False,
    )
    for column in (
        "on_time",
        "on_edge_end_time",
        "off_time",
        "session_end_time",
    ):
        manifest[column] = pd.to_datetime(manifest[column], errors="raise")
    physical_key = [
        "year",
        "station_code",
        "line_id",
        "on_time",
        "off_time",
    ]
    missing_physical_key = set(physical_key) - set(manifest.columns)
    if missing_physical_key:
        raise RuntimeError(
            "Manifest is missing physical-session key columns: "
            + ", ".join(sorted(missing_physical_key))
        )
    duplicate_physical = manifest.duplicated(physical_key, keep=False)
    if duplicate_physical.any():
        examples = manifest.loc[
            duplicate_physical, physical_key
        ].head(10).to_dict("records")
        raise RuntimeError(
            "Manifest contains duplicate physical sessions; use the "
            f"canonical preprocessing output. Examples: {examples}"
        )
    for column in (
        "catalog_strong",
        "catalog_weak",
        "seven_matched_station",
        "stress23_station",
    ):
        manifest[column] = manifest[column].map(as_bool)
    floors = load_noise_floors(args.noise_floor_file)
    enriched = enrich_cross_station(manifest, floors, config)
    inventory = transition_inventory(args.transition_inventory_file)
    enriched = enrich_catalog_overlap(enriched, config, inventory)
    if args.year:
        selected = enriched[enriched["year"].isin(args.year)].copy()
    else:
        selected = enriched.copy()
    if args.seven_only:
        selected = selected[selected["seven_matched_station"]].copy()
    if args.limit is not None:
        selected = selected.head(args.limit).copy()
    selected = selected.reset_index(drop=True)
    if not len(selected):
        raise RuntimeError("No sessions selected")

    code_hash = sha256_file(Path(__file__))
    config_hash = sha256_file(args.config)
    manifest_hash = sha256_file(args.manifest)
    context = {
        "protocol_version": config["protocol_version"],
        "code_sha256": code_hash,
        "config_sha256": config_hash,
        "manifest_sha256": manifest_hash,
    }
    tasks: list[dict[str, Any]] = []
    for raw_file, group in selected.groupby("raw_file", sort=False):
        part_id = stable_part_id(
            str(raw_file),
            group["session_uid"].astype(str).tolist(),
            context,
        )
        selected.loc[group.index, "_part_id"] = part_id
        if part_complete(part_dir, part_id) and not args.overwrite:
            continue
        tasks.append(
            {
                "part_id": part_id,
                "raw_file": str(raw_file),
                "rows": group.to_dict("records"),
                "config": config,
                "context": context,
            }
        )
    selected.to_csv(
        output_dir / "protocol_v3_selected_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    start = time.time()
    total_files = selected["raw_file"].nunique()
    completed_before = total_files - len(tasks)
    if tasks:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            future_map = {
                executor.submit(process_raw_group, task): task["raw_file"]
                for task in tasks
            }
            completed = completed_before
            for future in as_completed(future_map):
                raw_file = future_map[future]
                payload = future.result()
                write_part(part_dir, payload)
                completed += 1
                if completed % 25 == 0 or completed == total_files:
                    print(
                        f"processed raw files: {completed}/{total_files}",
                        flush=True,
                    )
                if payload["raw_error"]:
                    print(
                        f"raw read failure: {raw_file}: "
                        f"{payload['raw_error']}",
                        flush=True,
                    )
    results, runs, raw_inventory = combine_parts(
        selected, part_dir, output_dir
    )
    elapsed = time.time() - start
    lock = {
        "protocol_version": config["protocol_version"],
        "created_at": datetime.now().astimezone().isoformat(),
        "selection": {
            "year": args.year,
            "seven_only": args.seven_only,
            "limit": args.limit,
            "session_count": len(selected),
            "raw_file_count": int(selected["raw_file"].nunique()),
        },
        "hashes": {
            "runner_sha256": code_hash,
            "config_sha256": config_hash,
            "manifest_sha256": manifest_hash,
            "noise_floor_sha256": sha256_file(args.noise_floor_file),
        },
        "outputs": {
            "session_results": "protocol_v3_session_results.csv",
            "stable_runs": "protocol_v3_stable_runs.csv",
            "raw_inventory": "protocol_v3_raw_file_inventory.csv",
        },
        "result_counts": {
            "sessions": len(results),
            "stable_runs": len(runs),
            "raw_files": len(raw_inventory),
        },
    }
    if args.transition_inventory_file is not None:
        lock["hashes"]["transition_inventory_sha256"] = sha256_file(
            args.transition_inventory_file
        )
    else:
        lock["hashes"]["catalog_2024_sha256"] = sha256_file(CATALOG_2024)
        lock["hashes"]["catalog_2025_sha256"] = sha256_file(CATALOG_2025)
    (output_dir / "protocol_v3_run_lock.json").write_text(
        json.dumps(lock, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = build_report(results, runs, selected, elapsed)
    (output_dir / "protocol_v3_run_summary.md").write_text(
        report, encoding="utf-8"
    )
    print(report)
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
