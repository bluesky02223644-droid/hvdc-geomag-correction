# Derived data

Everything here is a derived, aggregated product of the analysis. No raw
one-second observatory records and no HVDC operating-catalog entries are
included; both are held under institutional access policies and cannot be
redistributed. See the "Raw data availability" section of the top-level README.

All values are in nT unless stated otherwise, and all times are UTC.

## `injection_grid_metrics.csv`

One row per cell of the known-offset injection grid (amplitude × duration).
Offsets of known amplitude were injected into otherwise untouched host
waveforms, so the recovery error in these cells is measured against ground
truth rather than against a catalog value.

| Column | Meaning |
| --- | --- |
| `amplitude_nT`, `duration_min` | Cell coordinates of the grid |
| `n_total` | Injections attempted in the cell (24 per cell) |
| `n_released` | Injections with at least one released stable run |
| `release_rate` | `n_released / n_total` |
| `n_weak_abstention`, `n_structural_misfit`, `n_invalid_window` | Withheld injections by routing stage |
| `primary_Aj_abs_error_median_nT`, `..._P90_nT` | Absolute error of the first released run's offset |
| `flag1_truth_RMS_median_nT`, `..._P90_nT` | RMS error over all released samples |
| `flag1_sample_count_median` | Released samples per injection |

Read by `figures/fig04_operating_domain.py`. 16 cells × 24 injections = 384.

## `figure5_source.csv`

One row per catalog-strong paired session in the 2024–2025 development cohorts
(1,332 rows), carrying station and line identity, session duration, the routing
outcome, and the fitted, edge-local, and baseline amplitudes with their ratio
to the catalog value `A_ref_on_nT`. That catalog amplitude is a same-source
reference read from the same waveform family, not an independent ground truth.

Read by `figures/fig05_amplitude_consistency.py`.

## `released_stable_runs.csv`

One row per released stable run: 5,178 runs across 2,824 released sessions of
the 2024–2025 cohorts.

| Column | Meaning |
| --- | --- |
| `session_uid`, `station_code`, `date`, `year` | Session identity |
| `run_index` | Index of the run within its session, from 1 |
| `run_start_second`, `run_end_second` | Run bounds as seconds of day |
| `run_samples`, `valid_samples` | Run length and how much of it is finite |
| `A_j_nT` | Offset subtracted from this run |
| `local_residual_RMS_nT` | Detrended residual RMS inside the run |
| `duration_min`, `A_ref_nT`, `catalog_strong` | Session-level context |

Read by `figures/fig08_downstream_impact.py`. The hourly-mean bias removed by a
run is exact given these columns: subtracting `A_j_nT` from `n` seconds inside a
UTC hour shifts that hourly mean by `A_j_nT * n / 3600`.

## `session_routing_summary.csv`

Session counts by year and final routing outcome (`released`,
`weak_abstention`, `structural_misfit`, `invalid_window`).

## `demo_station_days.csv.gz`

Second-by-second raw, returned, background, and flag series for four
representative station-days used as figure cases (4 × 86,400 rows). Columns:
`panel`, `session_id`, `timestamp`, `second_of_day`, `raw_Z_nT`,
`returned_Z_nT`, `flag`, `background_Z_nT`.

These are corrected outputs of the workflow rather than raw archive files, and
are included so that the downstream-impact figure and the case waveforms can be
reproduced. Panel `c` is Liyang (32011) on 2025-08-04, the worked example of
Figure 8a.
