"""Tests for slc.evaluate — metrics and report generation."""

import geopandas as gpd
import pytest

from slc.evaluate import compute_metrics, generate_report, match_to_osm


def _comparison_gdf(rows):
    return gpd.GeoDataFrame(rows)


def test_compute_metrics_perfect():
    comp = _comparison_gdf(
        [
            {"overture_id": "a", "our_speed_mph": 35, "osm_maxspeed_mph": 35},
            {"overture_id": "b", "our_speed_mph": 45, "osm_maxspeed_mph": 45},
        ]
    )
    metrics = compute_metrics(comp)
    assert metrics["exact_match_rate"] == pytest.approx(1.0)
    assert metrics["within_5mph_rate"] == pytest.approx(1.0)
    assert metrics["n_estimates"] == 2
    assert metrics["n_with_osm"] == 2


def test_compute_metrics_partial():
    comp = _comparison_gdf(
        [
            {"overture_id": "a", "our_speed_mph": 35, "osm_maxspeed_mph": 35},
            {"overture_id": "b", "our_speed_mph": 45, "osm_maxspeed_mph": 55},
        ]
    )
    metrics = compute_metrics(comp)
    assert metrics["exact_match_rate"] == pytest.approx(0.5)
    assert metrics["within_10mph_rate"] == pytest.approx(1.0)


def test_compute_metrics_empty():
    metrics = compute_metrics(gpd.GeoDataFrame())
    assert metrics["exact_match_rate"] is None
    assert metrics["n_estimates"] == 0


def test_compute_metrics_no_osm():
    comp = _comparison_gdf(
        [{"overture_id": "a", "our_speed_mph": 35, "osm_maxspeed_mph": None}]
    )
    metrics = compute_metrics(comp)
    assert metrics["n_with_osm"] == 0
    assert metrics["exact_match_rate"] is None


def test_generate_report_contains_headings():
    comp = _comparison_gdf(
        [{"overture_id": "a", "our_speed_mph": 35, "osm_maxspeed_mph": 35}]
    )
    metrics = compute_metrics(comp)
    report = generate_report(comp, metrics)
    assert "# Speed Limit Conflation" in report
    assert "Exact match" in report
    assert "100.0%" in report


def test_generate_report_no_osm():
    metrics = compute_metrics(gpd.GeoDataFrame())
    report = generate_report(gpd.GeoDataFrame(), metrics)
    assert "N/A" in report


def test_match_to_osm_empty():
    result = match_to_osm(
        gpd.GeoDataFrame(),
        gpd.GeoDataFrame(),
        gpd.GeoDataFrame(),
    )
    assert result.empty
