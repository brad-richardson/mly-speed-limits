"""Sign-to-sequence projection with heading validation.

Each Mapillary speed sign is projected (snapped) perpendicularly onto the
nearest sequence within a distance and heading tolerance.  Signs that are
likely on a cross street (heading disagreement) are rejected.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from slc.fetch import angle_diff, bearing, haversine_distance, linestring_length_m

# ---------------------------------------------------------------------------
# Heading / bearing utilities
# ---------------------------------------------------------------------------

_METRES_PER_DEGREE_LAT = 111_320.0


def _degrees_to_metres(degrees: float) -> float:
    """Very rough conversion: 1° latitude ≈ 111 320 m."""
    return degrees * _METRES_PER_DEGREE_LAT


def compute_bearing_at_position(
    line: LineString,
    distance_along: float,
    window_m: float = 10.0,
) -> float:
    """Return the bearing of *line* at *distance_along* metres from its start.

    Uses a small window around the target point to smooth GPS jitter.

    Args:
        line: WGS-84 LineString.
        distance_along: Distance in metres from the start of *line*.
        window_m: Half-window size in metres used to determine the two
                  reference points for bearing computation.

    Returns:
        Bearing in degrees (0 = north, clockwise).
    """
    total_len = linestring_length_m(line)
    if total_len == 0:
        return 0.0

    # Normalised fractions for the window
    half = min(window_m, total_len / 2.0)

    frac_before = max(0.0, (distance_along - half) / total_len)
    frac_after = min(1.0, (distance_along + half) / total_len)

    pt_before = line.interpolate(frac_before, normalized=True)
    pt_after = line.interpolate(frac_after, normalized=True)

    return bearing(pt_before.x, pt_before.y, pt_after.x, pt_after.y)


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------


def snap_signs_to_sequences(
    signs: gpd.GeoDataFrame,
    sequences: gpd.GeoDataFrame,
    max_distance_m: float = 30.0,
    max_heading_diff: float = 30.0,
) -> gpd.GeoDataFrame:
    """Project each sign onto the nearest qualifying sequence.

    The matching procedure for each sign is:

    1. Use an :class:`~shapely.strtree.STRtree` to find candidate sequences
       within *max_distance_m* of the sign.
    2. For each candidate, compute the perpendicular snap distance using
       Shapely's ``project`` / ``interpolate`` linear referencing.
    3. Optionally validate heading: if the sign carries a camera heading,
       reject candidates whose bearing at the snap point differs by more than
       *max_heading_diff* degrees.
    4. Keep **all** valid snaps (a sign may appear on multiple sequences
       from different drives).

    Args:
        signs: GeoDataFrame produced by :func:`slc.fetch.fetch_mapillary_signs`.
               Must have columns ``id, geometry, speed_mph, heading``.
        sequences: GeoDataFrame produced by :func:`slc.fetch.build_sequences`.
                   Must have columns ``sequence_id, geometry``.
        max_distance_m: Maximum perpendicular snap distance in metres.
        max_heading_diff: Maximum angular difference (degrees) between the
                          camera heading and the sequence bearing at the snap
                          point.  Only applied when ``heading`` is not ``None``.

    Returns:
        GeoDataFrame with one row per valid (sign, sequence) pair and columns:
        ``sign_id, sequence_id, snap_distance_m, distance_along_m,
        heading_agreement, projected_point, speed_mph``.
    """
    if signs.empty or sequences.empty:
        import pandas as pd
        crs = signs.crs if not signs.empty else "EPSG:4326"
        return gpd.GeoDataFrame(
            {
                "sign_id": pd.Series(dtype="object"),
                "sequence_id": pd.Series(dtype="object"),
                "snap_distance_m": pd.Series(dtype="float64"),
                "distance_along_m": pd.Series(dtype="float64"),
                "heading_agreement": pd.Series(dtype="bool"),
                "projected_point": gpd.GeoSeries(dtype="geometry"),
                "speed_mph": pd.Series(dtype="int64"),
            },
            geometry="projected_point",
            crs=crs,
        )

    # Build a spatial index on sequences
    seq_geoms = list(sequences.geometry)
    tree = STRtree(seq_geoms)

    # Rough degree buffer for initial candidate filtering (1° ≈ 111 km)
    buffer_deg = max_distance_m / _METRES_PER_DEGREE_LAT

    rows: list[dict[str, Any]] = []
    for _, sign_row in signs.iterrows():
        sign_pt: Point = sign_row.geometry
        heading_val = sign_row.get("heading")

        # Candidate sequences within buffer
        buffered = sign_pt.buffer(buffer_deg)
        candidate_indices = tree.query(buffered)

        for idx in candidate_indices:
            seq_row = sequences.iloc[idx]
            seq_line: LineString = seq_row.geometry

            # Linear referencing: fraction along sequence
            frac = seq_line.project(sign_pt, normalized=True)
            snap_pt = seq_line.interpolate(frac, normalized=True)

            # Snap distance in metres (haversine)
            dist_m = haversine_distance(sign_pt.x, sign_pt.y, snap_pt.x, snap_pt.y)
            if dist_m > max_distance_m:
                continue

            # Distance along sequence in metres
            total_len = linestring_length_m(seq_line)
            dist_along_m = frac * total_len

            # Heading validation
            heading_ok = True
            if heading_val is not None:
                seq_bearing = compute_bearing_at_position(seq_line, dist_along_m)
                diff = angle_diff(float(heading_val), seq_bearing)
                heading_ok = diff <= max_heading_diff

            if not heading_ok:
                continue

            rows.append(
                {
                    "sign_id": sign_row["id"],
                    "sequence_id": seq_row["sequence_id"],
                    "snap_distance_m": dist_m,
                    "distance_along_m": dist_along_m,
                    "heading_agreement": heading_val is None or heading_ok,
                    "projected_point": snap_pt,
                    "speed_mph": sign_row["speed_mph"],
                }
            )

    if not rows:
        import pandas as pd
        return gpd.GeoDataFrame(
            {
                "sign_id": pd.Series(dtype="object"),
                "sequence_id": pd.Series(dtype="object"),
                "snap_distance_m": pd.Series(dtype="float64"),
                "distance_along_m": pd.Series(dtype="float64"),
                "heading_agreement": pd.Series(dtype="bool"),
                "projected_point": gpd.GeoSeries(dtype="geometry"),
                "speed_mph": pd.Series(dtype="int64"),
            },
            geometry="projected_point",
            crs=signs.crs,
        )

    return gpd.GeoDataFrame(rows, geometry="projected_point", crs=signs.crs)
