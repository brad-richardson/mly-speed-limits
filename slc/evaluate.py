"""Evaluation: compare our speed estimates against Overture ground truth.

Overture transportation segments carry a ``speed_limits`` property that is
already normalized.  :func:`~slc.fetch.extract_overture_speed_limits` parses
this into a ``speed_limit_value`` column, which we use directly as ground
truth — no separate OSM fetch or spatial matching is needed.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd

# ---------------------------------------------------------------------------
# Overture-based comparison
# ---------------------------------------------------------------------------


def compare_to_overture(
    estimates: gpd.GeoDataFrame,
    overture_segments: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Compare our speed estimates against Overture's own speed limit values.

    Both datasets share the same ``overture_id`` / ``id`` key, so this is a
    simple join — no spatial matching required.

    Args:
        estimates: GeoDataFrame from :func:`slc.consensus.compute_consensus`.
                   Must have columns ``overture_id, speed_mph``.
        overture_segments: GeoDataFrame from
                           :func:`slc.fetch.fetch_overture_segments` **after**
                           calling :func:`slc.fetch.extract_overture_speed_limits`
                           so that the ``speed_limit_value`` column is present.

    Returns:
        GeoDataFrame with columns
        ``overture_id, our_speed_mph, overture_speed_mph``.
        Only segments that have *both* an estimate and an Overture speed limit
        are included.
    """
    if estimates.empty or overture_segments.empty:
        return gpd.GeoDataFrame(
            columns=["overture_id", "our_speed_mph", "overture_speed_mph"]
        )

    id_col = "id" if "id" in overture_segments.columns else overture_segments.columns[0]
    truth = overture_segments[[id_col, "speed_limit_value"]].rename(
        columns={id_col: "overture_id", "speed_limit_value": "overture_speed_mph"}
    )

    merged = estimates[["overture_id", "speed_mph"]].rename(
        columns={"speed_mph": "our_speed_mph"}
    ).merge(truth, on="overture_id", how="inner")

    # Keep only rows where Overture has a ground-truth value
    merged = merged.dropna(subset=["overture_speed_mph"])
    merged["overture_speed_mph"] = merged["overture_speed_mph"].astype(int)

    return gpd.GeoDataFrame(merged.reset_index(drop=True))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(comparison: gpd.GeoDataFrame) -> dict[str, Any]:
    """Compute evaluation metrics comparing our estimates to Overture speed limits.

    Args:
        comparison: Output of :func:`compare_to_overture`.

    Returns:
        Dictionary with keys:
        ``exact_match_rate, within_5mph_rate, within_10mph_rate,
        n_estimates, n_with_ground_truth, confusion_matrix``.
    """
    if comparison.empty:
        return {
            "exact_match_rate": None,
            "within_5mph_rate": None,
            "within_10mph_rate": None,
            "n_estimates": 0,
            "n_with_ground_truth": 0,
            "confusion_matrix": {},
        }

    valid = comparison.dropna(subset=["overture_speed_mph", "our_speed_mph"])

    n_total = len(comparison)
    n_valid = len(valid)

    if n_valid == 0:
        return {
            "exact_match_rate": None,
            "within_5mph_rate": None,
            "within_10mph_rate": None,
            "n_estimates": n_total,
            "n_with_ground_truth": 0,
            "confusion_matrix": {},
        }

    our = valid["our_speed_mph"].astype(int)
    truth = valid["overture_speed_mph"].astype(int)
    diff = (our - truth).abs()

    exact = int((diff == 0).sum())
    within5 = int((diff <= 5).sum())
    within10 = int((diff <= 10).sum())

    conf_matrix: dict[int, dict[int, int]] = {}
    for our_val, truth_val in zip(our, truth):
        conf_matrix.setdefault(int(our_val), {})
        conf_matrix[int(our_val)][int(truth_val)] = (
            conf_matrix[int(our_val)].get(int(truth_val), 0) + 1
        )

    return {
        "exact_match_rate": round(exact / n_valid, 4),
        "within_5mph_rate": round(within5 / n_valid, 4),
        "within_10mph_rate": round(within10 / n_valid, 4),
        "n_estimates": n_total,
        "n_with_ground_truth": n_valid,
        "confusion_matrix": conf_matrix,
    }


def generate_report(comparison: gpd.GeoDataFrame, metrics: dict[str, Any]) -> str:
    """Generate a Markdown summary of the evaluation results.

    Args:
        comparison: Output of :func:`compare_to_overture`.
        metrics: Output of :func:`compute_metrics`.

    Returns:
        Multi-line Markdown string.
    """
    lines = [
        "# Speed Limit Conflation — Evaluation Report",
        "",
        "## Coverage",
        f"- Total Overture segments with estimates: **{metrics.get('n_estimates', 0)}**",
        f"- Segments with Overture ground truth: **{metrics.get('n_with_ground_truth', 0)}**",
        "",
        "## Accuracy vs Overture speed limits",
    ]

    for key, label in [
        ("exact_match_rate", "Exact match"),
        ("within_5mph_rate", "Within 5 mph"),
        ("within_10mph_rate", "Within 10 mph"),
    ]:
        val = metrics.get(key)
        if val is not None:
            lines.append(f"- {label}: **{val * 100:.1f}%**")
        else:
            lines.append(f"- {label}: N/A")

    conf = metrics.get("confusion_matrix", {})
    if conf:
        truth_vals = sorted({v for row in conf.values() for v in row})
        lines += [
            "",
            "## Confusion Matrix (our estimate → Overture value → count)",
            "",
            "| Our \\ Overture | " + " | ".join(str(v) for v in truth_vals) + " |",
            "| --- | " + " | ".join(["---"] * len(truth_vals)) + " |",
        ]
        for our_val in sorted(conf):
            row_counts = conf[our_val]
            cells = " | ".join(str(row_counts.get(v, 0)) for v in truth_vals)
            lines.append(f"| {our_val} | {cells} |")

    return "\n".join(lines) + "\n"

