"""Tests for slc.fetch utility functions (no network calls)."""

import pytest

from slc.fetch import (
    _parse_speed_mph,
    angle_diff,
    bearing,
    build_sequences,
    extract_overture_speed_limits,
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
# extract_overture_speed_limits
# ---------------------------------------------------------------------------


def test_extract_overture_speed_limits_basic():
    segs = gpd.GeoDataFrame(
        [
            {
                "id": "seg_1",
                "geometry": LineString([(-111.9, 40.88), (-111.88, 40.88)]),
                "speed_limits": [{"max_speed": {"value": 35, "unit": "mph"}, "when": None}],
            },
            {
                "id": "seg_2",
                "geometry": LineString([(-111.9, 40.89), (-111.88, 40.89)]),
                "speed_limits": [{"max_speed": {"value": 25, "unit": "mph"}, "when": None}],
            },
        ],
        crs="EPSG:4326",
    )
    result = extract_overture_speed_limits(segs)
    assert result.loc[result["id"] == "seg_1", "speed_limit_value"].iloc[0] == 35
    assert result.loc[result["id"] == "seg_2", "speed_limit_value"].iloc[0] == 25


def test_extract_overture_speed_limits_no_column():
    segs = gpd.GeoDataFrame(
        [{"id": "seg_1", "geometry": LineString([(-111.9, 40.88), (-111.88, 40.88)])}],
        crs="EPSG:4326",
    )
    result = extract_overture_speed_limits(segs)
    assert "speed_limit_value" in result.columns
    assert result["speed_limit_value"].iloc[0] is None


def test_extract_overture_speed_limits_prefers_unconditional():
    """When both conditional and unconditional entries exist, pick unconditional."""
    segs = gpd.GeoDataFrame(
        [
            {
                "id": "seg_1",
                "geometry": LineString([(-111.9, 40.88), (-111.88, 40.88)]),
                "speed_limits": [
                    {"max_speed": {"value": 20, "unit": "mph"}, "when": {"time": "school"}},
                    {"max_speed": {"value": 35, "unit": "mph"}, "when": None},
                ],
            }
        ],
        crs="EPSG:4326",
    )
    result = extract_overture_speed_limits(segs)
    assert result["speed_limit_value"].iloc[0] == 35


def test_extract_overture_speed_limits_null_entry():
    segs = gpd.GeoDataFrame(
        [
            {
                "id": "seg_1",
                "geometry": LineString([(-111.9, 40.88), (-111.88, 40.88)]),
                "speed_limits": None,
            }
        ],
        crs="EPSG:4326",
    )
    result = extract_overture_speed_limits(segs)
    assert result["speed_limit_value"].iloc[0] is None


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
