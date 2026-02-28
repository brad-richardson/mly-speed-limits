"""Sequence splitting into labeled sub-edges.

A Mapillary sequence is split at two kinds of points:

* **Sign splits** — where a speed limit sign was detected.  The downstream
  edge carries the sign's speed.
* **Turn splits** — where the bearing changes sharply.  Speed resets to
  ``None`` because we can't reliably propagate a limit across a turn.
"""

from __future__ import annotations

import math
from typing import Any

import geopandas as gpd
from shapely.ops import substring

from slc.fetch import bearing, haversine_distance, linestring_length_m
from slc.types import SplitEdge

# ---------------------------------------------------------------------------
# Bearing-change detection
# ---------------------------------------------------------------------------


def detect_turn_splits(
    sequence: Any,  # LineString
    bearing_threshold_deg: float = 60.0,
    window_m: float = 80.0,
) -> list[float]:
    """Find distances (metres) along *sequence* where bearing changes sharply.

    A sliding window of size *window_m* is moved along the line.  At each
    step the bearing entering the window is compared to the bearing leaving
    it; when the difference exceeds *bearing_threshold_deg* the midpoint of
    the window is recorded as a split distance.

    Args:
        sequence: WGS-84 LineString.
        bearing_threshold_deg: Minimum angular difference to trigger a split.
        window_m: Approximate look-ahead / look-behind window in metres.

    Returns:
        Sorted list of distances (metres from the start) where splits should
        be inserted.
    """
    coords = list(sequence.coords)
    if len(coords) < 3:
        return []

    total_len = linestring_length_m(sequence)
    if total_len == 0:
        return []

    # Accumulate cumulative distances for each vertex
    cum_dists: list[float] = [0.0]
    for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
        cum_dists.append(cum_dists[-1] + haversine_distance(x1, y1, x2, y2))

    half_win = window_m / 2.0
    splits: list[float] = []

    for i, d_mid in enumerate(cum_dists):
        d_before = d_mid - half_win
        d_after = d_mid + half_win
        if d_before < 0 or d_after > total_len:
            continue

        frac_before = d_before / total_len
        frac_after = d_after / total_len
        pt_before = sequence.interpolate(frac_before, normalized=True)
        pt_after = sequence.interpolate(frac_after, normalized=True)

        b_enter = bearing(pt_before.x, pt_before.y, coords[i][0], coords[i][1])
        b_exit = bearing(coords[i][0], coords[i][1], pt_after.x, pt_after.y)

        diff = abs(b_enter - b_exit) % 360
        diff = min(diff, 360 - diff)
        if diff >= bearing_threshold_deg:
            splits.append(d_mid)

    # De-duplicate nearby split points (within half_win metres)
    merged: list[float] = []
    for d in sorted(splits):
        if not merged or (d - merged[-1]) > half_win:
            merged.append(d)

    return merged


# ---------------------------------------------------------------------------
# Sub-edge construction helpers
# ---------------------------------------------------------------------------


def _avg_heading_of_line(line: Any) -> float:
    """Compute the mean bearing along a LineString from first to last coordinate."""
    coords = list(line.coords)
    if len(coords) < 2:
        return 0.0
    # Use overall start→end bearing as a lightweight proxy
    return bearing(coords[0][0], coords[0][1], coords[-1][0], coords[-1][1])


def _cut_line(line: Any, start_m: float, end_m: float) -> Any:
    """Return a sub-LineString from *start_m* to *end_m* metres along *line*."""
    total = linestring_length_m(line)
    if total == 0:
        return line
    start_norm = max(0.0, start_m / total)
    end_norm = min(1.0, end_m / total)
    if start_norm >= end_norm:
        return None
    return substring(line, start_norm, end_norm, normalized=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def split_sequence(
    sequence_id: str,
    sequence_geom: Any,  # LineString
    snapped_signs: list[dict[str, Any]],
    turn_distances: list[float],
) -> list[SplitEdge]:
    """Split one sequence into a list of :class:`~slc.types.SplitEdge` objects.

    Algorithm:

    1. Merge sign positions and turn positions into one sorted split list,
       tagging each with its reason.
    2. Walk the sequence, cutting at each split point.
    3. **Sign splits**: the downstream edge receives the sign's speed.
    4. **Turn splits**: the downstream edge's speed resets to ``None``.
    5. The very first edge (before any sign) always gets ``speed_mph=None``.

    Args:
        sequence_id: Parent sequence identifier.
        sequence_geom: WGS-84 LineString of the full sequence.
        snapped_signs: List of dicts with keys
                       ``distance_along_m`` (float) and ``speed_mph`` (int).
        turn_distances: List of distances (metres) from
                        :func:`detect_turn_splits`.

    Returns:
        List of :class:`~slc.types.SplitEdge` instances, one per sub-edge.
    """
    total_len = linestring_length_m(sequence_geom)
    if total_len == 0:
        return []

    # Build a unified split list: (distance_m, reason, speed_mph_or_None)
    splits: list[tuple[float, str, int | None]] = []
    for sign in snapped_signs:
        splits.append((float(sign["distance_along_m"]), "sign", int(sign["speed_mph"])))
    for d in turn_distances:
        splits.append((d, "turn", None))

    splits.sort(key=lambda t: t[0])

    # Add sentinel start and end
    boundaries = [(0.0, "start", None)] + splits + [(total_len, "end", None)]

    edges: list[SplitEdge] = []
    current_speed: int | None = None

    for i, (d_start, reason_start, speed_at_split) in enumerate(boundaries[:-1]):
        d_end = boundaries[i + 1][0]
        if d_end <= d_start:
            continue

        sub_line = _cut_line(sequence_geom, d_start, d_end)
        if sub_line is None:
            continue

        # Determine the speed for this edge:
        # - After a sign: downstream edge takes sign's speed
        # - After a turn: speed resets to None
        # - At start: None (before first sign)
        if reason_start == "sign" and speed_at_split is not None:
            current_speed = speed_at_split
        elif reason_start == "turn":
            current_speed = None
        # "start" and "end" don't change current_speed

        sub_len = linestring_length_m(sub_line)
        avg_hdg = _avg_heading_of_line(sub_line)

        edge_id = f"{sequence_id}_{i}"
        edges.append(
            SplitEdge(
                sequence_id=sequence_id,
                geometry=sub_line,
                speed_mph=current_speed,
                split_reason=reason_start,
                length_m=sub_len,
                avg_heading=avg_hdg,
                edge_id=edge_id,
            )
        )

    return edges


def split_all_sequences(
    sequences: gpd.GeoDataFrame,
    snapped_signs: gpd.GeoDataFrame,
    bearing_threshold_deg: float = 60.0,
    window_m: float = 80.0,
) -> gpd.GeoDataFrame:
    """Run the split pipeline on every sequence.

    Args:
        sequences: GeoDataFrame from :func:`slc.fetch.build_sequences`.
        snapped_signs: GeoDataFrame from :func:`slc.snap.snap_signs_to_sequences`.
        bearing_threshold_deg: Passed to :func:`detect_turn_splits`.
        window_m: Passed to :func:`detect_turn_splits`.

    Returns:
        GeoDataFrame of :class:`~slc.types.SplitEdge` rows with columns
        ``edge_id, sequence_id, geometry, speed_mph, split_reason,
        length_m, avg_heading``.
    """
    all_edges: list[dict[str, Any]] = []

    for _, seq_row in sequences.iterrows():
        seq_id = seq_row["sequence_id"]
        seq_geom = seq_row.geometry

        # Signs on this sequence
        if not snapped_signs.empty and "sequence_id" in snapped_signs.columns:
            seq_signs_df = snapped_signs[snapped_signs["sequence_id"] == seq_id]
            seq_signs = seq_signs_df[["distance_along_m", "speed_mph"]].to_dict("records")
        else:
            seq_signs = []

        turn_dists = detect_turn_splits(seq_geom, bearing_threshold_deg, window_m)
        edges = split_sequence(seq_id, seq_geom, seq_signs, turn_dists)

        for edge in edges:
            all_edges.append(
                {
                    "edge_id": edge.edge_id,
                    "sequence_id": edge.sequence_id,
                    "geometry": edge.geometry,
                    "speed_mph": edge.speed_mph,
                    "split_reason": edge.split_reason,
                    "length_m": edge.length_m,
                    "avg_heading": edge.avg_heading,
                }
            )

    if not all_edges:
        return gpd.GeoDataFrame(
            columns=[
                "edge_id",
                "sequence_id",
                "geometry",
                "speed_mph",
                "split_reason",
                "length_m",
                "avg_heading",
            ],
            crs=sequences.crs if not sequences.empty else "EPSG:4326",
        )

    return gpd.GeoDataFrame(all_edges, crs=sequences.crs)
