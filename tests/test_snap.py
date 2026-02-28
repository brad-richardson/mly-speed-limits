"""Tests for slc.snap — sign-to-sequence projection."""

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from slc.snap import compute_bearing_at_position, snap_signs_to_sequences


def _sequences_gdf():
    """A simple east-west sequence."""
    return gpd.GeoDataFrame(
        [
            {
                "sequence_id": "seq_1",
                "geometry": LineString([(-111.900, 40.880), (-111.880, 40.880)]),
                "image_count": 10,
                "image_headings": [90.0] * 10,
            }
        ],
        crs="EPSG:4326",
    )


def _signs_gdf(heading=None):
    """A sign very close (< 20 m) to the east-west sequence."""
    return gpd.GeoDataFrame(
        [
            {
                "id": "sign_1",
                "geometry": Point(-111.890, 40.8802),  # ~22 m north of sequence
                "speed_mph": 35,
                "raw_value": "regulatory--maximum-speed-limit--35",
                "confidence": 0.9,
                "heading": heading,
            }
        ],
        crs="EPSG:4326",
    )


def test_snap_basic():
    signs = _signs_gdf()
    seqs = _sequences_gdf()
    result = snap_signs_to_sequences(signs, seqs, max_distance_m=50.0)
    assert len(result) == 1
    assert result.iloc[0]["sign_id"] == "sign_1"
    assert result.iloc[0]["speed_mph"] == 35


def test_snap_distance_filtering():
    """Signs too far away should not snap."""
    signs = _signs_gdf()
    seqs = _sequences_gdf()
    result = snap_signs_to_sequences(signs, seqs, max_distance_m=5.0)
    assert result.empty


def test_snap_heading_rejection():
    """A sign with a perpendicular camera heading should be rejected."""
    signs = _signs_gdf(heading=0.0)  # pointing north, seq goes east
    seqs = _sequences_gdf()
    result = snap_signs_to_sequences(signs, seqs, max_distance_m=50.0, max_heading_diff=20.0)
    assert result.empty


def test_snap_heading_acceptance():
    """A sign with a matching heading should be accepted."""
    signs = _signs_gdf(heading=90.0)  # pointing east, seq goes east
    seqs = _sequences_gdf()
    result = snap_signs_to_sequences(signs, seqs, max_distance_m=50.0, max_heading_diff=20.0)
    assert len(result) == 1


def test_snap_empty_inputs():
    empty_signs = gpd.GeoDataFrame(
        columns=["id", "geometry", "speed_mph", "raw_value", "confidence", "heading"],
        crs="EPSG:4326",
    )
    seqs = _sequences_gdf()
    result = snap_signs_to_sequences(empty_signs, seqs)
    assert result.empty

    signs = _signs_gdf()
    empty_seqs = gpd.GeoDataFrame(
        columns=["sequence_id", "geometry", "image_count", "image_headings"],
        crs="EPSG:4326",
    )
    result = snap_signs_to_sequences(signs, empty_seqs)
    assert result.empty


def test_compute_bearing_at_position():
    line = LineString([(-111.900, 40.880), (-111.880, 40.880)])  # goes east
    b = compute_bearing_at_position(line, distance_along=100.0)
    assert 80.0 < b < 100.0  # roughly east
