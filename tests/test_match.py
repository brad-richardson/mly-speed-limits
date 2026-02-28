"""Tests for slc.match — split edge to Overture segment matching."""

import geopandas as gpd
import pytest
from shapely.geometry import LineString

from slc.match import match_edges_to_overture, project_edge_onto_segment


def _parallel_line(offset_lat: float = 0.0) -> LineString:
    """East-west line, optionally offset north/south."""
    return LineString(
        [(-111.900, 40.880 + offset_lat), (-111.880, 40.880 + offset_lat)]
    )


def _split_edges_gdf(speed_mph=35, offset_lat=0.0001):
    """A labeled split edge slightly offset from the Overture segment."""
    return gpd.GeoDataFrame(
        [
            {
                "edge_id": "seq_1_1",
                "sequence_id": "seq_1",
                "geometry": _parallel_line(offset_lat),
                "speed_mph": speed_mph,
                "split_reason": "sign",
                "length_m": 1500.0,
                "avg_heading": 90.0,
            }
        ],
        crs="EPSG:4326",
    )


def _overture_gdf():
    return gpd.GeoDataFrame(
        [
            {
                "id": "overture_abc",
                "geometry": _parallel_line(0.0),
            }
        ],
        crs="EPSG:4326",
    )


# ---------------------------------------------------------------------------
# project_edge_onto_segment
# ---------------------------------------------------------------------------


def test_project_same_line():
    line = _parallel_line()
    lr_start, lr_end, offset = project_edge_onto_segment(line, line)
    assert lr_start == pytest.approx(0.0, abs=0.05)
    assert lr_end == pytest.approx(1.0, abs=0.05)
    assert offset == pytest.approx(0.0, abs=5.0)  # should be near 0 m


def test_project_parallel_offset():
    edge = _parallel_line(0.0001)  # ~11 m north
    seg = _parallel_line(0.0)
    lr_start, lr_end, offset = project_edge_onto_segment(edge, seg)
    assert 0.0 <= lr_start <= lr_end <= 1.0
    assert 5.0 < offset < 30.0  # expect ~11 m offset


# ---------------------------------------------------------------------------
# match_edges_to_overture
# ---------------------------------------------------------------------------


def test_match_close_edge():
    edges = _split_edges_gdf(speed_mph=35, offset_lat=0.0001)
    overture = _overture_gdf()
    result = match_edges_to_overture(edges, overture, max_distance_m=50.0)
    assert len(result) == 1
    assert result.iloc[0]["overture_id"] == "overture_abc"
    assert result.iloc[0]["speed_mph"] == 35


def test_match_too_far_edge():
    edges = _split_edges_gdf(speed_mph=35, offset_lat=0.01)  # ~1 km away
    overture = _overture_gdf()
    result = match_edges_to_overture(edges, overture, max_distance_m=50.0)
    assert result.empty


def test_match_unlabeled_edge_ignored():
    """Unlabeled (speed_mph=None) edges should not appear in matches."""
    edges = gpd.GeoDataFrame(
        [
            {
                "edge_id": "seq_1_0",
                "sequence_id": "seq_1",
                "geometry": _parallel_line(0.0001),
                "speed_mph": None,
                "split_reason": "start",
                "length_m": 1500.0,
                "avg_heading": 90.0,
            }
        ],
        crs="EPSG:4326",
    )
    overture = _overture_gdf()
    result = match_edges_to_overture(edges, overture, max_distance_m=50.0)
    assert result.empty


def test_match_empty_inputs():
    empty_edges = gpd.GeoDataFrame(
        columns=["edge_id", "sequence_id", "geometry", "speed_mph", "avg_heading"],
        crs="EPSG:4326",
    )
    overture = _overture_gdf()
    result = match_edges_to_overture(empty_edges, overture)
    assert result.empty

    edges = _split_edges_gdf()
    empty_overture = gpd.GeoDataFrame(
        columns=["id", "geometry"],
        crs="EPSG:4326",
    )
    result = match_edges_to_overture(edges, empty_overture)
    assert result.empty
