"""Tests for slc.split — sequence splitting."""

import geopandas as gpd
import pytest
from shapely.geometry import LineString

from slc.split import detect_turn_splits, split_all_sequences, split_sequence


def _straight_line():
    """East-west straight line, ~2 km."""
    return LineString([(-111.920, 40.880), (-111.900, 40.880), (-111.880, 40.880)])


def _turn_line():
    """Line that makes a 90-degree turn: goes east then north."""
    return LineString(
        [
            (-111.920, 40.880),
            (-111.900, 40.880),  # eastward leg
            (-111.900, 40.900),  # northward leg after 90° turn
        ]
    )


# ---------------------------------------------------------------------------
# detect_turn_splits
# ---------------------------------------------------------------------------


def test_no_turns_on_straight_line():
    line = _straight_line()
    turns = detect_turn_splits(line, bearing_threshold_deg=60.0, window_m=100.0)
    assert turns == []


def test_detects_90_degree_turn():
    line = _turn_line()
    turns = detect_turn_splits(line, bearing_threshold_deg=60.0, window_m=200.0)
    assert len(turns) >= 1


def test_turn_detection_short_line():
    # Two-point line — no room for a window → no turns
    line = LineString([(-111.900, 40.880), (-111.890, 40.880)])
    turns = detect_turn_splits(line)
    assert turns == []


# ---------------------------------------------------------------------------
# split_sequence
# ---------------------------------------------------------------------------


def test_split_no_signs_no_turns():
    line = _straight_line()
    edges = split_sequence("seq_1", line, snapped_signs=[], turn_distances=[])
    # Should produce a single edge spanning the whole sequence
    assert len(edges) == 1
    assert edges[0].speed_mph is None
    assert edges[0].split_reason == "start"


def test_split_single_sign():
    line = _straight_line()
    from slc.fetch import linestring_length_m

    total = linestring_length_m(line)
    mid = total / 2.0
    signs = [{"distance_along_m": mid, "speed_mph": 35}]
    edges = split_sequence("seq_1", line, snapped_signs=signs, turn_distances=[])
    # Should produce 2 edges: unlabeled before sign, labeled after
    assert len(edges) == 2
    assert edges[0].speed_mph is None
    assert edges[1].speed_mph == 35


def test_split_turn_resets_speed():
    line = _turn_line()
    from slc.fetch import linestring_length_m

    total = linestring_length_m(line)
    # Put sign before the turn and turn in the middle
    quarter = total * 0.25
    half = total * 0.5
    signs = [{"distance_along_m": quarter, "speed_mph": 45}]
    turns = [half]
    edges = split_sequence("seq_1", line, snapped_signs=signs, turn_distances=turns)
    # After the turn, speed should reset
    speeds = [e.speed_mph for e in edges]
    # Last edge (after turn) should have None
    assert edges[-1].speed_mph is None


def test_split_edge_ids_are_unique():
    line = _straight_line()
    from slc.fetch import linestring_length_m

    total = linestring_length_m(line)
    signs = [{"distance_along_m": total / 3, "speed_mph": 30}]
    edges = split_sequence("seq_X", line, snapped_signs=signs, turn_distances=[])
    ids = [e.edge_id for e in edges]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# split_all_sequences
# ---------------------------------------------------------------------------


def test_split_all_sequences_basic():
    seqs = gpd.GeoDataFrame(
        [
            {
                "sequence_id": "seq_1",
                "geometry": _straight_line(),
                "image_count": 5,
                "image_headings": [90.0] * 5,
            }
        ],
        crs="EPSG:4326",
    )
    import pandas as pd

    snapped = gpd.GeoDataFrame(
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
        crs="EPSG:4326",
    )
    result = split_all_sequences(seqs, snapped)
    assert not result.empty
    assert "edge_id" in result.columns
    assert "speed_mph" in result.columns


def test_split_all_sequences_empty():
    import pandas as pd

    empty_seqs = gpd.GeoDataFrame(
        {
            "sequence_id": pd.Series(dtype="object"),
            "geometry": gpd.GeoSeries(dtype="geometry"),
            "image_count": pd.Series(dtype="int64"),
            "image_headings": pd.Series(dtype="object"),
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    empty_signs = gpd.GeoDataFrame(
        {
            "sign_id": pd.Series(dtype="object"),
            "sequence_id": pd.Series(dtype="object"),
            "distance_along_m": pd.Series(dtype="float64"),
            "speed_mph": pd.Series(dtype="int64"),
            "projected_point": gpd.GeoSeries(dtype="geometry"),
        },
        geometry="projected_point",
        crs="EPSG:4326",
    )
    result = split_all_sequences(empty_seqs, empty_signs)
    assert result.empty
