"""Tests for slc.fetch utility functions (no network calls)."""

import pytest

from slc.fetch import (
    _parse_maxspeed_tag,
    _parse_speed_mph,
    angle_diff,
    bearing,
    build_sequences,
    haversine_distance,
    linestring_length_m,
)
import geopandas as gpd
from shapely.geometry import LineString, Point


# ---------------------------------------------------------------------------
# _parse_speed_mph
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("regulatory--maximum-speed-limit--35", 35),
        ("regulatory--maximum-speed-limit--25", 25),
        ("REGULATORY--MAXIMUM-SPEED-LIMIT--65", 65),
        ("unknown-sign-value", None),
        ("", None),
    ],
)
def test_parse_speed_mph(value, expected):
    assert _parse_speed_mph(value) == expected


# ---------------------------------------------------------------------------
# _parse_maxspeed_tag
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("35 mph", 35),
        ("35mph", 35),
        ("35", 35),
        ("56 km/h", 35),  # 56 km/h ≈ 34.8 → rounds to 35
        ("56 kmh", 35),
        ("", None),
        ("national", None),
    ],
)
def test_parse_maxspeed_tag(value, expected):
    result = _parse_maxspeed_tag(value)
    assert result == expected


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------


def test_haversine_distance_same_point():
    assert haversine_distance(0, 0, 0, 0) == pytest.approx(0.0)


def test_haversine_distance_known():
    # ~111 km between (0,0) and (0,1) (1 degree latitude)
    d = haversine_distance(0, 0, 0, 1)
    assert 110_000 < d < 112_000


def test_bearing_north():
    b = bearing(0, 0, 0, 1)  # Moving north
    assert b == pytest.approx(0.0, abs=1.0)


def test_bearing_east():
    b = bearing(0, 0, 1, 0)  # Moving east
    assert b == pytest.approx(90.0, abs=1.0)


def test_bearing_south():
    b = bearing(0, 1, 0, 0)  # Moving south
    assert b == pytest.approx(180.0, abs=1.0)


def test_angle_diff():
    assert angle_diff(0, 0) == pytest.approx(0.0)
    assert angle_diff(0, 180) == pytest.approx(180.0)
    assert angle_diff(350, 10) == pytest.approx(20.0)
    assert angle_diff(10, 350) == pytest.approx(20.0)


def test_linestring_length_m():
    line = LineString([(0, 0), (0, 1)])  # ~111 km
    length = linestring_length_m(line)
    assert 110_000 < length < 112_000


# ---------------------------------------------------------------------------
# build_sequences
# ---------------------------------------------------------------------------


def _make_images_gdf():
    rows = [
        {
            "id": "img_1",
            "sequence_id": "seq_A",
            "geometry": Point(-111.89, 40.88),
            "compass_angle": 90.0,
            "captured_at": "2023-01-01T10:00:00Z",
        },
        {
            "id": "img_2",
            "sequence_id": "seq_A",
            "geometry": Point(-111.88, 40.88),
            "compass_angle": 90.0,
            "captured_at": "2023-01-01T10:00:05Z",
        },
        {
            "id": "img_3",
            "sequence_id": "seq_B",
            "geometry": Point(-111.90, 40.89),
            "compass_angle": 180.0,
            "captured_at": "2023-01-01T11:00:00Z",
        },
    ]
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def test_build_sequences_basic():
    images = _make_images_gdf()
    seqs = build_sequences(images)

    assert "seq_A" in seqs["sequence_id"].values
    # seq_B has only 1 image → should be dropped
    assert "seq_B" not in seqs["sequence_id"].values


def test_build_sequences_linestring():
    images = _make_images_gdf()
    seqs = build_sequences(images)
    row = seqs[seqs["sequence_id"] == "seq_A"].iloc[0]
    assert isinstance(row.geometry, LineString)
    assert row["image_count"] == 2


def test_build_sequences_empty():
    empty = gpd.GeoDataFrame(
        columns=["id", "sequence_id", "geometry", "compass_angle", "captured_at"],
        crs="EPSG:4326",
    )
    result = build_sequences(empty)
    assert result.empty
