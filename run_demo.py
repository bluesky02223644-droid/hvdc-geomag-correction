"""End-to-end demonstration of the correction workflow on a synthetic day.

The demonstration builds one 86,400-sample station-day in the observatory's
own raw text format, injects a ramp-boxcar offset of known amplitude, and then
runs the frozen protocol-v3 functions over it exactly as the batch runner does:

    baseline windows -> local step -> background -> bounded geometry fit
    -> edge-local amplitude -> stable-run and residual gate -> flagged release

Because the injected offset is known, the demonstration can report the
amplitude error and the corrected-sample RMS error against ground truth. This
is the same construction as the known-offset injection audit of the paper, run
on synthetic host data so that it needs no restricted observatory records.

Usage
-----
    python run_demo.py
    python run_demo.py --amplitude -6.0 --duration-min 35 --tau 45
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from hvdc_correction import (
    baseline_window_stats,
    edge_quality,
    fit_geometry,
    mad,
    ramp_boxcar_shape,
    read_raw_z,
    stable_runs,
    window_values,
)

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "configs" / "protocol_v3.json"
OUTPUT_DIR = REPO_ROOT / "outputs" / "demo"

DEMO_DATE = "2025-08-04"
QUANTUM_NT = 0.01


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def synthesize_raw_file(
    path: Path,
    on_second: int,
    off_second: int,
    amplitude: float,
    tau: float,
    noise_nt: float,
    seed: int,
) -> np.ndarray:
    """Write a synthetic raw station-day and return the noise-free host field.

    The host field is a smooth diurnal variation plus a short-period term, which
    stands in for quiet-time geomagnetic activity. It is the ground truth the
    correction should recover. The HVDC contribution added on top is the same
    ramp-boxcar shape the workflow fits, so the injected offset is known exactly
    at every second.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(86400, dtype=float)

    diurnal = 12.0 * np.sin(2 * np.pi * (t - 21600) / 86400)
    semidiurnal = 3.0 * np.sin(2 * np.pi * (t - 3600) / 43200)
    slow = np.cumsum(rng.normal(0.0, 0.0016, size=86400))
    slow -= np.linspace(slow[0], slow[-1], 86400)
    host = 45000.0 + diurnal + semidiurnal + slow

    injected = amplitude * ramp_boxcar_shape(t, float(on_second), float(off_second), tau)
    observed = host + injected + rng.normal(0.0, noise_nt, size=86400)
    observed = np.round(observed / QUANTUM_NT) * QUANTUM_NT

    horizontal = 33000.0 + 0.35 * diurnal + rng.normal(0.0, noise_nt, size=86400)
    declination = -5.2 + 0.002 * diurnal + rng.normal(0.0, noise_nt, size=86400)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("DATE       TIME     H D Z\n")
        for second in range(86400):
            clock = "%02d:%02d:%02d" % (
                second // 3600,
                (second % 3600) // 60,
                second % 60,
            )
            handle.write(
                "%s %s %.2f %.4f %.2f\n"
                % (DEMO_DATE, clock, horizontal[second], declination[second], observed[second])
            )
    return host


def correct_session(
    z: np.ndarray,
    on: int,
    on_end: int,
    off: int,
    off_end: int,
    duration_min: float,
    a_ref: float,
    station_floor: float,
    config: dict,
) -> dict:
    """Apply the frozen release path to one session and return its decision."""
    window = config["window_geometry"]
    fit_config = config["geometry_fit"]
    edge_config = config["edge_local_amplitude"]
    ident = config["identifiability_gate"]
    stable_config = config["stable_run_gate"]
    length = int(window["baseline_length_s"])
    buffer = int(window["baseline_buffer_s"])

    on_pre_values = window_values(z, on - buffer - length, on - buffer)
    on_pre = baseline_window_stats(on_pre_values, station_floor, config)
    on_post = baseline_window_stats(
        window_values(z, on_end + buffer, on_end + buffer + length), station_floor, config
    )
    off_pre = baseline_window_stats(
        window_values(z, off - buffer - length, off - buffer), station_floor, config
    )
    off_post = baseline_window_stats(
        window_values(z, off_end + buffer, off_end + buffer + length), station_floor, config
    )

    if "truncated" in {edge_quality(on_pre, on_post), edge_quality(off_pre, off_post)}:
        return {"route": "invalid_window", "stage": "edge_baseline_window_truncated"}

    a_baseline = on_post["median"] - on_pre["median"]
    threshold_id = max(
        float(ident["absolute_floor_nT"]),
        float(ident["station_noise_floor_multiple"]) * station_floor,
    )
    if not math.isfinite(a_baseline) or abs(a_baseline) < threshold_id:
        return {
            "route": "weak_abstention",
            "stage": "local_identifiability_abstention",
            "A_baseline_nT": a_baseline,
            "identifiability_threshold_nT": threshold_id,
        }

    differences = np.diff(on_pre_values[np.isfinite(on_pre_values)])
    sigma_hf = 1.4826 * mad(differences) / math.sqrt(2.0) if len(differences) else math.nan
    stable_threshold = max(
        float(stable_config["absolute_spread_floor_nT"]),
        float(stable_config["high_frequency_noise_multiple"]) * (sigma_hf if math.isfinite(sigma_hf) else 0.0),
    )

    fit_start = max(0, on - int(window["fit_margin_s"]))
    fit_end = min(86400, off_end + int(window["fit_margin_s"]) + 1)
    t = np.arange(fit_start, fit_end, dtype=float)
    y = z[fit_start:fit_end]
    b0 = on_pre["median"]
    pre_center = on - buffer - 0.5 * length
    post_center = off_end + buffer + 0.5 * length

    background_enabled = duration_min > float(fit_config["long_session_threshold_min"])
    slope = 0.0
    if background_enabled and post_center > pre_center:
        slope = (off_post["median"] - on_pre["median"]) / (post_center - pre_center)
        clip = float(fit_config["background_slope_clip_nT_per_s"])
        slope = float(np.clip(slope, -clip, clip))
    background = b0 + slope * (t - pre_center)

    fitted, fit_ok = fit_geometry(t, y, background, b0, a_baseline, float(on), float(off), config)
    if not fit_ok:
        return {"route": "structural_misfit", "stage": "geometry_fit"}

    fitted_on = fitted["fitted_on"]
    fitted_off = fitted["fitted_off"]
    tau = fitted["tau"]
    if fitted_off <= fitted_on + tau:
        return {"route": "structural_misfit", "stage": "no_positive_plateau"}

    edge_start = fitted_on + tau + float(edge_config["post_ramp_buffer_s"])
    edge_end = min(
        edge_start + float(edge_config["maximum_window_s"]),
        fitted_off - float(edge_config["off_edge_buffer_s"]),
    )
    edge_mask = (t >= edge_start) & (t <= edge_end) & np.isfinite(y)
    if int(edge_mask.sum()) < int(edge_config["minimum_valid_samples"]):
        return {"route": "structural_misfit", "stage": "edge_local_window"}
    a_edge = float(np.median((y - slope * (t - pre_center))[edge_mask]) - b0)

    event_start = max(0, on)
    event_end = min(86399, off_end)
    t_event = np.arange(event_start, event_end + 1, dtype=float)
    y_event = z[event_start : event_end + 1]
    background_event = b0 + slope * (t_event - pre_center)

    runs = stable_runs(
        t_event, y_event, background_event, fitted_on, fitted_off, tau,
        stable_threshold, a_ref, config,
    )
    if not runs:
        return {"route": "structural_misfit", "stage": "stable_run_and_residual_gate"}

    return {
        "route": "released",
        "stage": "released",
        "A_baseline_nT": a_baseline,
        "A_fit_nT": fitted["A_fit"],
        "A_edge_nT": a_edge,
        "fit_R2": fitted["R2"],
        "fitted_on": fitted_on,
        "fitted_off": fitted_off,
        "tau_s": tau,
        "background_enabled": background_enabled,
        "background_slope_nT_per_s": slope,
        "stable_spread_threshold_nT": stable_threshold,
        "runs": runs,
        "event_start": event_start,
        "event_end": event_end,
    }


def apply_release(z: np.ndarray, decision: dict) -> tuple[np.ndarray, np.ndarray]:
    """Turn a released decision into a returned series and its quality flags.

    Flag 1 samples have their stable-run offset subtracted. Flag 2 samples,
    the switching transitions between the first and last released run, are
    linearly interpolated between the flanking trusted values. Everything else
    is returned unchanged.
    """
    returned = z.copy()
    flags = np.zeros(86400, dtype=np.int8)
    if decision["route"] != "released":
        return returned, flags

    start = decision["event_start"]
    end = decision["event_end"]
    n_event = end - start + 1
    flag1 = np.zeros(n_event, dtype=bool)

    for run in decision["runs"]:
        lo = start + run["start_index"]
        hi = start + run["end_index"]
        returned[lo : hi + 1] = z[lo : hi + 1] - run["A_j"]
        flag1[run["start_index"] : run["end_index"] + 1] = True

    first = decision["runs"][0]["start_index"]
    last = decision["runs"][-1]["end_index"]
    flag2 = np.zeros(n_event, dtype=bool)
    flag2[:first] = True
    flag2[last + 1 :] = True

    flags[start : end + 1] = np.where(flag1, 1, np.where(flag2, 2, 4))

    trusted = np.ones(86400, dtype=bool)
    trusted[start : end + 1] = flag1 | ~(flag1 | flag2)
    fill = np.flatnonzero(flags == 2)
    if len(fill):
        anchors = np.flatnonzero(trusted & np.isfinite(returned))
        returned[fill] = np.interp(fill.astype(float), anchors.astype(float), returned[anchors])
    return returned, flags


def plot(truth: np.ndarray, raw: np.ndarray, returned: np.ndarray, flags: np.ndarray,
         on: int, off_end: int, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import sys

    sys.path.insert(0, str(REPO_ROOT / "figures"))
    from palette_standard import ROLE

    plt.rcParams.update({"font.size": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42})

    lo = max(0, on - 900)
    hi = min(86400, off_end + 900)
    minutes = (np.arange(lo, hi) - on) / 60.0

    figure, (ax, ax_flag) = plt.subplots(
        2, 1, figsize=(7.2, 4.2), height_ratios=[5, 1], sharex=True,
        gridspec_kw={"hspace": 0.15},
    )
    ax.plot(minutes, raw[lo:hi], color=ROLE["raw_trace"], lw=0.9,
            label="raw (offset injected)")
    ax.plot(minutes, returned[lo:hi], color=ROLE["released_trace"], lw=1.6,
            label="returned by workflow")
    ax.plot(minutes, truth[lo:hi], color=ROLE["flag2_transition"], lw=1.0, ls="--",
            label="truth (offset-free host)")
    ax.set_ylabel("Z (nT)")
    ax.legend(frameon=False, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), handlelength=2.0)
    ax.set_title("Synthetic known-offset demonstration", loc="left",
                 fontweight="bold", pad=22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    labels = {
        0: ("flag 0 outside session", ROLE["flag0_or_raw"]),
        1: ("flag 1 offset removed", ROLE["flag1_released"]),
        2: ("flag 2 transition interpolated", ROLE["flag2_transition"]),
        4: ("flag 4 not released", ROLE["flag4_complex"]),
    }
    handles = []
    for value, (label, color) in labels.items():
        mask = flags[lo:hi] == value
        if not mask.any():
            continue
        ax_flag.fill_between(minutes, 0, 1, where=mask, color=color, step="mid", lw=0)
        handles.append(matplotlib.patches.Patch(facecolor=color, label=label))
    ax_flag.set_ylim(0, 1)
    ax_flag.set_yticks([])
    ax_flag.set_xlabel("Minutes from catalog on-time")
    ax_flag.legend(handles=handles, frameon=False, ncol=len(handles), fontsize=7,
                   loc="upper center", bbox_to_anchor=(0.5, -0.55))
    for spine in ("top", "right", "left"):
        ax_flag.spines[spine].set_visible(False)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amplitude", type=float, default=-6.0, help="injected offset in nT")
    parser.add_argument("--duration-min", type=float, default=35.0, help="session length in minutes")
    parser.add_argument("--tau", type=float, default=45.0, help="switching ramp time constant in s")
    parser.add_argument("--noise", type=float, default=0.06, help="high-frequency noise sigma in nT")
    parser.add_argument("--on-hour", type=float, default=10.0, help="catalog on-time, hours UTC")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    config = load_config()
    seed = args.seed if args.seed is not None else int(config["random_seed"])

    on = int(round(args.on_hour * 3600))
    off = on + int(round(args.duration_min * 60))
    on_end = on + 120
    off_end = off + 120

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT_DIR / "synthetic_station_day.Txt"
    truth = synthesize_raw_file(
        raw_path, on, off, args.amplitude, args.tau, args.noise, seed
    )  # truth is the offset-free host field

    z, metadata = read_raw_z(str(raw_path))

    decision = correct_session(
        z, on, on_end, off, off_end,
        duration_min=args.duration_min,
        a_ref=args.amplitude,
        station_floor=args.noise,
        config=config,
    )

    print("=" * 68)
    print("HVDC correction workflow - synthetic known-offset demonstration")
    print("=" * 68)
    print(f"raw file            : {raw_path}")
    print(f"raw SHA-256         : {metadata['raw_file_sha256'][:16]}...")
    print(f"injected amplitude  : {args.amplitude:+.3f} nT")
    print(f"injected duration   : {args.duration_min:.0f} min, tau = {args.tau:.0f} s")
    print(f"route               : {decision['route']} ({decision['stage']})")

    if decision["route"] != "released":
        print("\nThe session was withheld. This is a valid outcome: the workflow")
        print("abstains rather than emitting a correction it cannot support.")
        return

    returned, flags = apply_release(z, decision)
    flag1 = flags == 1
    amp_error = abs(decision["runs"][0]["A_j"] - args.amplitude)
    rms_error = float(np.sqrt(np.mean((returned[flag1] - truth[flag1]) ** 2)))
    raw_rms = float(np.sqrt(np.mean((z[flag1] - truth[flag1]) ** 2)))

    print(f"released runs       : {len(decision['runs'])}")
    print(f"fitted tau          : {decision['tau_s']:.1f} s")
    print(f"fit R^2             : {decision['fit_R2']:.5f}")
    print(f"A_edge              : {decision['A_edge_nT']:+.3f} nT")
    print(f"first-run A_j       : {decision['runs'][0]['A_j']:+.3f} nT")
    print("-" * 68)
    print(f"amplitude error     : {amp_error:.3f} nT")
    print(f"corrected-sample RMS: {rms_error:.3f} nT  (was {raw_rms:.3f} nT before correction)")
    print(f"flag 1 / 2 / 4      : {int(flag1.sum())} / {int((flags == 2).sum())} / {int((flags == 4).sum())} s")
    print("-" * 68)

    summary = {
        "injected_amplitude_nT": args.amplitude,
        "injected_duration_min": args.duration_min,
        "injected_tau_s": args.tau,
        "noise_sigma_nT": args.noise,
        "seed": seed,
        "route": decision["route"],
        "released_run_count": len(decision["runs"]),
        "A_fit_nT": round(decision["A_fit_nT"], 4),
        "A_edge_nT": round(decision["A_edge_nT"], 4),
        "first_run_A_j_nT": round(decision["runs"][0]["A_j"], 4),
        "fitted_tau_s": round(decision["tau_s"], 3),
        "fit_R2": round(decision["fit_R2"], 6),
        "amplitude_error_nT": round(amp_error, 4),
        "corrected_sample_rms_error_nT": round(rms_error, 4),
        "uncorrected_sample_rms_error_nT": round(raw_rms, 4),
        "n_flag1": int(flag1.sum()),
        "n_flag2": int((flags == 2).sum()),
        "n_flag4": int((flags == 4).sum()),
    }
    (OUTPUT_DIR / "demo_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {OUTPUT_DIR / 'demo_summary.json'}")

    if not args.no_plot:
        figure_path = OUTPUT_DIR / "demo_correction.png"
        plot(truth, z, returned, flags, on, off_end, figure_path)
        print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
