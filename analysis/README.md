# Cross-component specificity analysis

These two scripts produce the cross-component result reported in the paper: the
release criterion separates cataloged HVDC operations from quiet times in $Z$
with a 99.9% to 20.1% contrast, but only 76.0% to 57.5% in $H$, which is why no
horizontal-component product is released.

`cross_component_steps.py`
: Applies the workflow's baseline-window step measurement, unchanged, to the
  `Z`, `H`, and `D` channels of the raw files for every released strong session
  of the 2024–2025 cohorts.

`cross_component_control.py`
: Draws gap-matched control measurements from quiet times on the same
  station-days — same window pair, same pre-to-post gap, at least 1,800 s from
  every cataloged edge — and compares the two populations.

Unlike everything else in this repository, these two scripts read the raw
one-second archive and the session manifest, neither of which can be
redistributed (see the top-level README). They are included so that the
procedure is fully specified and auditable, not so that it can be re-run here.
Their outputs are shipped in `../data/`:

- `cross_component_specificity.csv` — the per-component event-versus-control table
- `control_comparison_gapmatched.json` — the same figures with run parameters

Path constants at the top of each script point at the original project layout
and would need to be repointed for any other archive.
