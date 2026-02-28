"""Tests for slc.consensus — multi-observation voting."""

import geopandas as gpd
import pytest

from slc.consensus import (
    _agreement_ratio,
    _confidence,
    _modal_speed,
    compute_consensus,
    detect_speed_zone_boundaries,
)


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------


def test_modal_speed_majority():
    assert _modal_speed([35, 35, 45]) == 35


def test_modal_speed_tie_broken_by_highest():
    # Tie: both appear once → max wins by our tiebreak
    assert _modal_speed([35, 45]) == 45


def test_agreement_ratio_full():
    assert _agreement_ratio([35, 35, 35], 35) == pytest.approx(1.0)


def test_agreement_ratio_partial():
    assert _agreement_ratio([35, 35, 45], 35) == pytest.approx(2 / 3)


def test_confidence_single_observation():
    """Single observation with perfect agreement should have < 1 confidence."""
    c = _confidence(1, 1.0)
    assert 0 < c < 1.0


def test_confidence_many_observations():
    """Many agreeing observations → high confidence."""
    c = _confidence(20, 1.0)
    assert c > 0.9


# ---------------------------------------------------------------------------
# compute_consensus
# ---------------------------------------------------------------------------


def _matches_gdf(rows):
    return gpd.GeoDataFrame(rows)


def test_compute_consensus_single_segment():
    matches = _matches_gdf(
        [
            {
                "edge_id": "e1",
                "overture_id": "seg_A",
                "lr_start": 0.0,
                "lr_end": 0.5,
                "score": 0.9,
                "speed_mph": 35,
            },
            {
                "edge_id": "e2",
                "overture_id": "seg_A",
                "lr_start": 0.2,
                "lr_end": 0.8,
                "score": 0.85,
                "speed_mph": 35,
            },
        ]
    )
    result = compute_consensus(matches)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["overture_id"] == "seg_A"
    assert row["speed_mph"] == 35
    assert row["observation_count"] == 2
    assert row["has_conflict"] is False


def test_compute_consensus_conflict():
    matches = _matches_gdf(
        [
            {
                "edge_id": "e1",
                "overture_id": "seg_B",
                "lr_start": 0.0,
                "lr_end": 0.4,
                "score": 0.9,
                "speed_mph": 35,
            },
            {
                "edge_id": "e2",
                "overture_id": "seg_B",
                "lr_start": 0.5,
                "lr_end": 1.0,
                "score": 0.8,
                "speed_mph": 45,
            },
        ]
    )
    result = compute_consensus(matches)
    row = result.iloc[0]
    assert row["has_conflict"] is True


def test_compute_consensus_empty():
    result = compute_consensus(gpd.GeoDataFrame())
    assert result.empty


# ---------------------------------------------------------------------------
# detect_speed_zone_boundaries
# ---------------------------------------------------------------------------


def test_detect_boundary_found():
    matches = _matches_gdf(
        [
            {
                "edge_id": "e1",
                "overture_id": "seg_C",
                "lr_start": 0.1,
                "lr_end": 0.4,
                "speed_mph": 35,
            },
            {
                "edge_id": "e2",
                "overture_id": "seg_C",
                "lr_start": 0.6,
                "lr_end": 0.9,
                "speed_mph": 45,
            },
        ]
    )
    boundaries = detect_speed_zone_boundaries(matches, "seg_C")
    assert len(boundaries) == 1
    assert 0.1 < boundaries[0] < 0.9


def test_detect_boundary_no_conflict():
    matches = _matches_gdf(
        [
            {
                "edge_id": "e1",
                "overture_id": "seg_D",
                "lr_start": 0.0,
                "lr_end": 1.0,
                "speed_mph": 35,
            }
        ]
    )
    boundaries = detect_speed_zone_boundaries(matches, "seg_D")
    assert boundaries == []
