"""Tests for slc.types dataclasses."""

import pytest
from shapely.geometry import LineString, Point

from slc.types import SpeedEstimate, SpeedSign, SplitEdge


def test_speed_sign_basic():
    sign = SpeedSign(
        id="sign_1",
        geometry=Point(-111.89, 40.88),
        speed_mph=35,
        raw_value="regulatory--maximum-speed-limit--35",
        sign_type="standard",
        confidence=0.95,
    )
    assert sign.speed_mph == 35
    assert sign.sign_type == "standard"
    assert sign.heading is None


def test_speed_sign_with_heading():
    sign = SpeedSign(
        id="sign_2",
        geometry=Point(-111.89, 40.88),
        speed_mph=25,
        raw_value="regulatory--maximum-speed-limit--25",
        sign_type="standard",
        confidence=0.80,
        heading=270.0,
    )
    assert sign.heading == 270.0


def test_split_edge_requires_edge_id():
    with pytest.raises(ValueError):
        SplitEdge(
            sequence_id="seq_1",
            geometry=LineString([(-111.89, 40.88), (-111.88, 40.89)]),
            speed_mph=35,
            split_reason="sign",
            length_m=500.0,
            avg_heading=90.0,
            edge_id="",  # empty → should raise
        )


def test_split_edge_valid():
    edge = SplitEdge(
        sequence_id="seq_1",
        geometry=LineString([(-111.89, 40.88), (-111.88, 40.89)]),
        speed_mph=35,
        split_reason="sign",
        length_m=500.0,
        avg_heading=90.0,
        edge_id="seq_1_0",
    )
    assert edge.speed_mph == 35
    assert edge.edge_id == "seq_1_0"


def test_split_edge_unlabeled():
    edge = SplitEdge(
        sequence_id="seq_1",
        geometry=LineString([(-111.89, 40.88), (-111.88, 40.89)]),
        speed_mph=None,
        split_reason="start",
        length_m=200.0,
        avg_heading=180.0,
        edge_id="seq_1_0",
    )
    assert edge.speed_mph is None


def test_speed_estimate_defaults():
    est = SpeedEstimate(
        overture_id="overture_abc",
        speed_mph=45,
        lr_start=0.0,
        lr_end=1.0,
        confidence=0.9,
        observation_count=5,
    )
    assert est.source_edges == []
    assert est.confidence == 0.9
