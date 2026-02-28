"""Multi-observation voting and linear reference assignment.

Multiple Mapillary sequences may cover the same Overture segment.
This module aggregates observations from :func:`slc.match.match_edges_to_overture`
into a single :class:`~slc.types.SpeedEstimate` per segment (or per LR sub-range
when a speed zone boundary is detected within the segment).
"""

from __future__ import annotations

import math
from typing import Any

import geopandas as gpd

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _modal_speed(speeds: list[int]) -> int:
    """Return the most common speed value; in a tie, pick the lowest speed."""
    counts: dict[int, int] = {}
    for s in speeds:
        counts[s] = counts.get(s, 0) + 1
    # max by (count, -v): highest count wins; on tie, -v is largest for
    # smallest v, so the lowest speed is chosen (conservative for safety).
    return max(counts, key=lambda v: (counts[v], -v))


def _agreement_ratio(speeds: list[int], modal: int) -> float:
    """Fraction of observations that match the modal speed."""
    if not speeds:
        return 0.0
    return sum(1 for s in speeds if s == modal) / len(speeds)


def _confidence(observation_count: int, agreement_ratio: float) -> float:
    """Combine observation count and agreement into a 0–1 confidence score.

    Uses a simple formula:
    ``confidence = agreement_ratio * (1 - exp(-observation_count / 3))``

    This gives full weight to 100 % agreement with many observations and
    discounts lonely single-observation estimates.
    """
    saturation = 1.0 - math.exp(-observation_count / 3.0)
    return round(agreement_ratio * saturation, 4)


# ---------------------------------------------------------------------------
# Speed zone boundary detection
# ---------------------------------------------------------------------------


def detect_speed_zone_boundaries(
    matches: gpd.GeoDataFrame,
    overture_id: str,
) -> list[float]:
    """Find LR positions where the speed limit changes within one segment.

    When observations on the same Overture segment disagree, this function
    tries to find a single *change point* along the LR axis that separates
    two consistent speed zones.

    The algorithm sorts observations by ``lr_start`` and looks for the first
    position where the majority speed shifts from one value to another.

    Args:
        matches: Full matches DataFrame (output of
                 :func:`slc.match.match_edges_to_overture`).
        overture_id: The Overture segment to inspect.

    Returns:
        List of estimated LR positions (0–1) of speed zone boundaries.
        Empty when no clear boundary is found.
    """
    seg_matches = matches[matches["overture_id"] == overture_id].copy()
    if seg_matches.empty or seg_matches["speed_mph"].nunique() <= 1:
        return []

    seg_matches = seg_matches.sort_values("lr_start")
    speeds = seg_matches["speed_mph"].tolist()
    lrs = seg_matches["lr_start"].tolist()

    # Simple change-point: find the first index where a run of one speed
    # transitions to a run of a different speed
    boundaries: list[float] = []
    for i in range(1, len(speeds)):
        if speeds[i] != speeds[i - 1]:
            boundary_lr = (lrs[i - 1] + lrs[i]) / 2.0
            # Only record if not too close to 0 or 1
            if 0.05 < boundary_lr < 0.95:
                boundaries.append(round(boundary_lr, 4))
                break  # Report first boundary only for simplicity

    return boundaries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_consensus(
    matches: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Aggregate speed observations per Overture segment into consensus estimates.

    For each ``overture_id`` group:

    * Computes the **modal speed** (majority vote).
    * Computes the **agreement ratio** and **observation count**.
    * Assigns a **confidence score** combining count and agreement.
    * Flags **conflicts** where two or more different speeds were observed.
    * Detects potential **speed zone boundaries** within the segment.
    * Reports the mean LR range covered by the observations.

    Args:
        matches: GeoDataFrame from :func:`slc.match.match_edges_to_overture`.
                 Must have columns
                 ``overture_id, speed_mph, lr_start, lr_end, edge_id, score``.

    Returns:
        GeoDataFrame with one row per Overture segment (or per LR sub-range
        when a zone boundary is detected) and columns:
        ``overture_id, speed_mph, lr_start, lr_end, confidence,
        observation_count, agreement_ratio, has_conflict,
        zone_boundary_lr, source_edges``.
    """
    if matches.empty:
        return gpd.GeoDataFrame(
            columns=[
                "overture_id",
                "speed_mph",
                "lr_start",
                "lr_end",
                "confidence",
                "observation_count",
                "agreement_ratio",
                "has_conflict",
                "zone_boundary_lr",
                "source_edges",
            ]
        )

    rows: list[dict[str, Any]] = []

    for overture_id, grp in matches.groupby("overture_id"):
        speeds = [int(s) for s in grp["speed_mph"].tolist()]
        modal = _modal_speed(speeds)
        agree = _agreement_ratio(speeds, modal)
        conf = _confidence(len(speeds), agree)

        has_conflict = grp["speed_mph"].nunique() > 1
        zone_boundaries = detect_speed_zone_boundaries(matches, str(overture_id))

        source_edges = grp["edge_id"].tolist()

        rows.append(
            {
                "overture_id": overture_id,
                "speed_mph": modal,
                "lr_start": float(grp["lr_start"].min()),
                "lr_end": float(grp["lr_end"].max()),
                "confidence": conf,
                "observation_count": len(speeds),
                "agreement_ratio": agree,
                "has_conflict": has_conflict,
                "zone_boundary_lr": zone_boundaries[0] if zone_boundaries else None,
                "source_edges": source_edges,
            }
        )

    return gpd.GeoDataFrame(rows)
