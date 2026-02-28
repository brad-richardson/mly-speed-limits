"""Split edge → Overture segment matching.

Each labeled :class:`~slc.types.SplitEdge` is matched to one or more Overture
road segments using densified point projection.  The result includes linear
reference positions (LR 0–1) on the Overture segment and a match quality score.
"""

from __future__ import annotations

import math
from typing import Any

import geopandas as gpd
from shapely.geometry import LineString, MultiPoint, Point
from shapely.strtree import STRtree

from slc.fetch import angle_diff, haversine_distance, linestring_length_m

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_METRES_PER_DEGREE = 111_320.0


def _densify_line(line: LineString, step_m: float = 10.0) -> list[Point]:
    """Return a list of evenly spaced points sampled along *line*."""
    total = linestring_length_m(line)
    if total == 0:
        return [Point(line.coords[0])]
    n = max(2, int(total / step_m) + 1)
    fracs = [i / (n - 1) for i in range(n)]
    return [line.interpolate(f, normalized=True) for f in fracs]


def project_edge_onto_segment(
    edge: LineString,
    segment: LineString,
    sample_step_m: float = 10.0,
) -> tuple[float, float, float]:
    """Project a split edge onto an Overture segment.

    Densifies the split edge, projects each sample point onto the segment,
    and returns the min/max normalised linear reference positions together
    with the mean offset.

    Args:
        edge: The split edge geometry (WGS-84 LineString).
        segment: The Overture road segment geometry (WGS-84 LineString).
        sample_step_m: Sampling interval along the edge in metres.

    Returns:
        ``(lr_start, lr_end, mean_offset_m)`` where *lr_start* and *lr_end*
        are normalised positions (0–1) along *segment* and *mean_offset_m*
        is the average perpendicular distance in metres.
    """
    sample_pts = _densify_line(edge, step_m=sample_step_m)
    seg_len = linestring_length_m(segment)

    lrs: list[float] = []
    offsets: list[float] = []

    for pt in sample_pts:
        frac = segment.project(pt, normalized=True)
        snap_pt = segment.interpolate(frac, normalized=True)
        lrs.append(frac)
        offsets.append(haversine_distance(pt.x, pt.y, snap_pt.x, snap_pt.y))

    lr_start = min(lrs)
    lr_end = max(lrs)
    mean_offset = sum(offsets) / len(offsets) if offsets else 0.0
    return lr_start, lr_end, mean_offset


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def match_edges_to_overture(
    split_edges: gpd.GeoDataFrame,
    overture_segments: gpd.GeoDataFrame,
    max_distance_m: float = 25.0,
    min_overlap_fraction: float = 0.3,
) -> gpd.GeoDataFrame:
    """Match split edges to Overture segments.

    For each split edge that carries a speed label (``speed_mph`` is not
    ``None``):

    1. Query the spatial index for candidate Overture segments within
       *max_distance_m*.
    2. For each candidate compute the densified point projection
       (mean offset and LR range).
    3. Validate bearing agreement between the edge and the candidate.
    4. Compute a simple quality score: ``1 - mean_offset / max_distance_m``.
    5. Keep all candidates that pass the distance threshold; an edge can
       match multiple segments when it spans an intersection.

    Args:
        split_edges: GeoDataFrame from :func:`slc.split.split_all_sequences`.
        overture_segments: GeoDataFrame from :func:`slc.fetch.fetch_overture_segments`.
        max_distance_m: Maximum mean perpendicular offset allowed.
        min_overlap_fraction: Minimum fraction of the edge's length that must
                              overlap with the candidate segment's LR range.
                              Currently used as a soft filter.

    Returns:
        GeoDataFrame with columns
        ``edge_id, overture_id, lr_start, lr_end, score, speed_mph``.
    """
    # Only match labeled edges
    labeled = split_edges[split_edges["speed_mph"].notna()].copy()

    if labeled.empty or overture_segments.empty:
        return gpd.GeoDataFrame(
            columns=["edge_id", "overture_id", "lr_start", "lr_end", "score", "speed_mph"],
        )

    seg_geoms = list(overture_segments.geometry)
    tree = STRtree(seg_geoms)

    buffer_deg = max_distance_m / _METRES_PER_DEGREE

    rows: list[dict[str, Any]] = []
    for _, edge_row in labeled.iterrows():
        edge_geom: LineString = edge_row.geometry
        edge_id = edge_row["edge_id"]
        speed = int(edge_row["speed_mph"])
        edge_hdg = float(edge_row.get("avg_heading", 0.0))

        buffered = edge_geom.buffer(buffer_deg)
        candidate_idxs = tree.query(buffered)

        for idx in candidate_idxs:
            seg_row = overture_segments.iloc[idx]
            seg_geom: LineString = seg_row.geometry

            lr_start, lr_end, mean_offset = project_edge_onto_segment(edge_geom, seg_geom)

            if mean_offset > max_distance_m:
                continue

            # Bearing check
            seg_coords = list(seg_geom.coords)
            if len(seg_coords) >= 2:
                from slc.fetch import bearing as _bearing

                seg_hdg = _bearing(
                    seg_coords[0][0],
                    seg_coords[0][1],
                    seg_coords[-1][0],
                    seg_coords[-1][1],
                )
                hdg_diff = angle_diff(edge_hdg, seg_hdg)
                # Allow up to 45° for minor geometric deviations; reject clearly
                # perpendicular or opposite-direction matches
                if hdg_diff > 45.0 and hdg_diff < 135.0:
                    continue

            score = max(0.0, 1.0 - mean_offset / max_distance_m)
            seg_id = seg_row.get("id", str(idx))

            rows.append(
                {
                    "edge_id": edge_id,
                    "overture_id": seg_id,
                    "lr_start": lr_start,
                    "lr_end": lr_end,
                    "score": score,
                    "speed_mph": speed,
                }
            )

    if not rows:
        return gpd.GeoDataFrame(
            columns=["edge_id", "overture_id", "lr_start", "lr_end", "score", "speed_mph"],
        )

    return gpd.GeoDataFrame(rows)
