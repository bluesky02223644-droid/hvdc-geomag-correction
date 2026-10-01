> Legacy version. Current manuscript software: [Selective-HVDC-Offset-Correction](https://github.com/bluesky02223644-droid/Selective-HVDC-Offset-Correction).

# Event-guided correction of HVDC-induced geomagnetic disturbances

Reference implementation and reproduction materials for:

> Event-guided, morphology-aware correction of HVDC-induced geomagnetic
> disturbances in 1 Hz observatory records. Submitted to *Earth, Planets and
> Space*.

High-voltage direct-current (HVDC) transmission lines operating in monopolar
mode inject step-like offsets into nearby geomagnetic observatory records. This
repository contains the frozen workflow that detects, fits, and selectively
removes those offsets from one-second vertical-component (Z) data, together
with enough derived data to reproduce the paper's main quantitative figures and
to run the method end to end on a synthetic day.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_demo.py
```

`run_demo.py` builds one synthetic 86,400-sample station-day in the
observatory's own raw text format, injects a ramp-boxcar offset of known
amplitude, and runs the frozen protocol functions over it. Expected output with
the default settings:

```
route               : released (released)
injected amplitude  : -6.000 nT
first-run A_j       : -5.954 nT
amplitude error     : 0.046 nT
corrected-sample RMS: 0.077 nT  (was 6.002 nT before correction)
```

Because the injected offset is known at every second, the demonstration reports
error against ground truth rather than against a reference catalog. Try
`python run_demo.py --amplitude -1.2 --duration-min 240` to see the workflow
abstain instead of releasing: withholding a correction it cannot support is a
designed outcome, not a failure.

## What the workflow does

For each catalog-paired on–off session the workflow runs a fixed sequence and
records where a session leaves it:

1. **Baseline windows.** Two 120 s windows flank each switching edge. Their
   medians give the measured local step; their scatter and drift set the
   per-session noise and stability thresholds.
2. **Identifiability gate.** Sessions whose measured local step falls below the
   station noise floor are withheld (`weak_abstention`).
3. **Bounded geometry fit.** A ramp-boxcar shape with free onset, offset,
   amplitude, and ramp time constant τ is fitted over the session plus margin,
   on top of a background that is flat for short sessions and locally linear
   beyond 30 min.
4. **Edge-local amplitude.** An independent amplitude is measured in a short
   window just after the ramp completes, and compared with the catalog value.
5. **Stable-run and residual gate.** Inside the fitted plateau, the workflow
   searches for contiguous runs of at least 60 s whose 10–90 percentile spread
   and detrended residual RMS both stay below a noise-scaled threshold.
6. **Flagged release.** Each accepted run has its own offset `A_j` subtracted;
   switching transitions are interpolated; everything else is returned
   unchanged. A session counts as *released* when at least one of its stable
   runs passes the gates.

Every returned sample carries a quality flag:

| Flag | Meaning |
| --- | --- |
| 0 | outside the session; unchanged |
| 1 | stable plateau; estimated offset subtracted |
| 2 | switching transition; interpolated |
| 3 | weak response; unchanged |
| 4 | structural, noisy, invalid, or low-confidence interval; unchanged |

## Repository layout

```
configs/protocol_v3.json          frozen parameters; every threshold in the paper
hvdc_correction/protocol_v3.py    the implementation that produced the results
run_demo.py                       synthetic known-offset end-to-end demonstration
figures/                          scripts for the main quantitative figures
analysis/                         cross-component specificity check (needs raw data)
data/                             derived inputs those scripts read
outputs/                          created on first run (git-ignored)
```

`hvdc_correction/protocol_v3.py` is shipped verbatim, including the batch
runner that walks the restricted observatory archive, so that the released code
and the published numbers correspond exactly. Its `main()` entry point requires
inputs that cannot be redistributed (see below); its session-level functions,
which are what the method actually consists of, run on any one-second Z series
and are re-exported from `hvdc_correction/__init__.py`.

## Reproducing the figures

```bash
python figures/fig02_workflow.py             # Fig. 2  workflow schematic
python figures/fig04_operating_domain.py     # Fig. 4  known-offset operating domain
python figures/fig05_amplitude_consistency.py # Fig. 5 release and amplitude consistency
python figures/fig08_downstream_impact.py    # Fig. 8  effect on hourly and daily means
```

Outputs are written to `outputs/figures/` as PDF and 300 dpi PNG. All four run
from the data in this repository with no further setup. `figures/fig08` also
writes `figure8_summary.json`, which reproduces the downstream statistics quoted
in the paper, including the +9.61 nT maximum hourly shift of the worked example.

`figures/palette_standard.py` holds the colour system used across all figures in
the paper; import `ROLE` from it to keep new panels consistent.

Figures 1, 3, 6, and 7 are not included here because they depend on the raw
one-second archive, a terrain model, or the full HVDC event catalog, none of
which can be redistributed.

## Data

`data/` contains derived, aggregated products only. No raw observatory records
are included.

| File | Contents |
| --- | --- |
| `injection_grid_metrics.csv` | Per-cell release rate and error metrics for the 384-injection known-offset grid |
| `figure5_source.csv` | Per-session release outcome and edge-local amplitude ratio for the 2024–2025 strong-event cohort |
| `released_stable_runs.csv` | The 5,178 released stable runs across 2,824 released sessions: timing, duration, and estimated offset `A_j` |
| `session_routing_summary.csv` | Session counts by year and routing outcome |
| `demo_station_days.csv.gz` | Raw, returned, and flag series for four representative station-days used as figure cases |
| `cross_component_specificity.csv` | Event-versus-control acceptance of the release criterion in Z, H, and D |
| `control_comparison_gapmatched.json` | The same figures with the run parameters that produced them |

### Raw data availability

The one-second geomagnetic data underlying this study are provided by the
National Geomagnetic Network Center, Institute of Geophysics, China Earthquake
Administration. The raw observations are subject to institutional access
policies and cannot be redistributed here. Interested researchers may submit a
formal written request to the National Geomagnetic Network Center.

The HVDC operating catalog used to define sessions is likewise held by the
operating utilities and is not redistributable.

Running the full batch workflow therefore requires access to both. Everything
else in this repository, including the complete method implementation and the
synthetic demonstration, runs without them.

## Requirements

Python 3.10 or newer, with `numpy`, `scipy`, `pandas`, and `matplotlib`. See
`requirements.txt` for the tested versions.

## Citation

See `CITATION.cff`. Please cite the paper if you use this code.

## License

MIT for the code (`LICENSE`). The derived data files in `data/` are released
under CC BY 4.0.
