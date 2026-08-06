"""
HVDC geomagnetic papers — locked figure color standard (low-saturation).

All 7 user reference images were inventoried. They collapse to 4 distinct
schemes (several images reuse the same Long et al. 2025 Nat Commun sets on
different figure types). Do NOT invent new hex per figure.

Scheme inventory
----------------
S1  SCI low-sat blue–teal–orange (ref img 1)
    Charts: line / bar / box / algorithm comparison
    #3A5F8A #79A7C8 #63A69F #E49B5D #B8BDC6

S2  Nat Commun pastel categorical (ref img 2, 4, 7 — same hex row)
    Categories, pies, soft multi-series, schematics
    #f599a1 #9fd7e9 #95aeda #fcd590 #a577ad #73c79e #5299cc

S3  Nat Commun diverging map (ref img 3, 5 — same warm↔cool ramp)
    Signed continuous fields, spatial trends
    #e73618 #f6bf5c #def1e4 #7bcbf1 #1e4e9e

S4  Nat Commun deep blue–teal–gold (ref img 6)
    Multi-series maps/lines needing stronger mid-tones than S2
    #3c4397 #3c70b7 #2eb5a3 #dab532 #a577ad #73c79e #5299cc

Usage rule for THIS paper
-------------------------
- Default multi-series charts     -> S1 (CYCLE)
- Flags / soft categories / SI pie -> S2 roles
- Heat / signed continuous         -> S3 (get_cmap diverging)
- Stronger multi-line on maps      -> S4 (CYCLE_DEEP) when S1 is too soft
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# S1 — SCI low-sat chart cycle
# ---------------------------------------------------------------------------
S1 = {
    "navy": "#3A5F8A",
    "steel": "#79A7C8",
    "teal": "#63A69F",
    "apricot": "#E49B5D",
    "mist": "#B8BDC6",
}
CYCLE = [S1["navy"], S1["steel"], S1["teal"], S1["apricot"], S1["mist"]]

# ---------------------------------------------------------------------------
# S2 — Nat Comm pastel categorical
# ---------------------------------------------------------------------------
S2 = {
    "blush": "#f599a1",
    "ice": "#9fd7e9",
    "periwinkle": "#95aeda",
    "sand": "#fcd590",
    "mauve": "#a577ad",
    "sage": "#73c79e",
    "sky": "#5299cc",
}
CYCLE_PASTEL = [
    S2["sky"],
    S2["sage"],
    S2["blush"],
    S2["sand"],
    S2["mauve"],
    S2["periwinkle"],
    S2["ice"],
]

# ---------------------------------------------------------------------------
# S3 — diverging map / signed continuous
# ---------------------------------------------------------------------------
S3_DIVERGING = ["#e73618", "#f6bf5c", "#def1e4", "#7bcbf1", "#1e4e9e"]
S3_SEQUENTIAL = ["#def1e4", "#7bcbf1", "#5299cc", "#3A5F8A", "#1e4e9e"]
# Warm counterpart, reused wherever dark must read as "worse" rather than "more".
S3_SEQUENTIAL_WARM = ["#F7F8FA", "#fcd590", "#f6bf5c", "#E49B5D", "#e73618"]

# ---------------------------------------------------------------------------
# S4 — deeper blue–teal–gold
# ---------------------------------------------------------------------------
S4 = {
    "indigo": "#3c4397",
    "royal": "#3c70b7",
    "cyan": "#2eb5a3",
    "gold": "#dab532",
    "mauve": "#a577ad",
    "sage": "#73c79e",
    "sky": "#5299cc",
}
CYCLE_DEEP = [
    S4["indigo"],
    S4["royal"],
    S4["cyan"],
    S4["gold"],
    S4["mauve"],
    S4["sage"],
    S4["sky"],
]

# ---------------------------------------------------------------------------
# Shared ink / structure
# ---------------------------------------------------------------------------
INK = {
    "ink": "#2F343A",
    "muted": "#6B7280",
    "rule": "#D1D5DB",
    "light_fill": "#EEF2F5",
    "legend_fill": "#F0F4F8",
    "paper": "#F7F8FA",
}

# ---------------------------------------------------------------------------
# Paper-semantic roles (drawn from S1+S2; locked across Fig 1–7)
# ---------------------------------------------------------------------------
ROLE = {
    **INK,
    "neutral": S1["mist"],

    # stations / lines
    "strong_station": S2["sage"],
    "weak_station": S1["apricot"],
    "diagnostic_station": S1["navy"],
    "line_1100": S2["mauve"],
    "line_800": S2["blush"],
    "line_mid": S1["mist"],

    # years / models
    "year_2024": S1["navy"],
    "year_2025": S1["apricot"],
    "year_2026": S1["teal"],
    "model_primary": S1["navy"],
    "model_old": S2["blush"],
    "model_alt": S1["apricot"],
    "model_neutral": S1["mist"],

    # flags / anchors
    "flag0_or_raw": S1["mist"],
    "flag1_released": S2["sage"],
    "flag2_transition": S1["apricot"],
    "flag3_weak": S2["sky"],
    "flag4_complex": S2["mauve"],
    "catalog_anchor": S2["blush"],
    "fit_ramp_band": S2["ice"],
    "background_line": S1["navy"],
    "raw_trace": INK["muted"],
    "released_trace": S1["navy"],
}

# Back-compat aliases used by earlier draft
BASE = {**S1, **S2, **S4, **INK}
CONTINUOUS = {"sequential": "S3_seq", "diverging": "S3_div"}


def _listed(hex_list, name):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, hex_list, N=256)


def get_cmap(kind: str = "sequential"):
    if kind == "diverging":
        return _listed(S3_DIVERGING, "hvdc_s3_div")
    if kind == "sequential_warm":
        return _listed(S3_SEQUENTIAL_WARM, "hvdc_s3_seq_warm")
    return _listed(S3_SEQUENTIAL, "hvdc_s3_seq")


def role(name: str) -> str:
    return ROLE[name]
