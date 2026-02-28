"""Tests for slc.evaluate — metrics and report generation."""

import geopandas as gpd
import pytest

from slc.evaluate import compare_to_overture, compute_metrics, generate_report


def _comparison_gdf(rows):
    return gpd.GeoDataFrame(rows)


def test_compute_metrics_perfect():
    comp = _comparison_gdf(
        [
            {"overture_id": "a", "our_speed_mph": 35, "overture_speed_mph": 35},
            {"overture_id": "b", "our_speed_mph": 45, "overture_speed_mph": 45},
        ]
    )
    metrics = compute_metrics(comp)
    assert metrics["exact_match_rate"] == pytest.approx(1.0)
    assert metrics["within_5mph_rate"] == pytest.approx(1.0)
    assert metrics["n_estimates"] == 2
    assert metrics["n_with_ground_truth"] == 2


def test_compute_metrics_partial():
    comp = _comparison_gdf(
        [
            {"overture_id": "a", "our_speed_mph": 35, "overture_speed_mph": 35},
            {"overture_id": "b", "our_speed_mph": 45, "overture_speed_mph": 55},
        ]
    )
    metrics = compute_metrics(comp)
    assert metrics["exact_match_rate"] == pytest.approx(0.5)
    assert metrics["within_10mph_rate"] == pytest.approx(1.0)


def test_compute_metrics_empty():
    metrics = compute_metrics(gpd.GeoDataFrame())
    assert metrics["exact_match_rate"] is None
    assert metrics["n_estimates"] == 0


def test_compute_metrics_no_ground_truth():
    comp = _comparison_gdf(
        [{"overture_id": "a", "our_speed_mph": 35, "overture_speed_mph": None}]
    )
    metrics = compute_metrics(comp)
    assert metrics["n_with_ground_truth"] == 0
    assert metrics["exact_match_rate"] is None


def test_generate_report_contains_headings():
    comp = _comparison_gdf(
        [{"overture_id": "a", "our_speed_mph": 35, "overture_speed_mph": 35}]
    )
    metrics = compute_metrics(comp)
    report = generate_report(comp, metrics)
    assert "# Speed Limit Conflation" in report
    assert "Exact match" in report
    assert "100.0%" in report
    assert "Overture" in report


def test_generate_report_no_ground_truth():
    metrics = compute_metrics(gpd.GeoDataFrame())
    report = generate_report(gpd.GeoDataFrame(), metrics)
    assert "N/A" in report


def test_compare_to_overture_basic():
    from shapely.geometry import LineString

    estimates = gpd.GeoDataFrame(
        [
            {"overture_id": "seg_1", "speed_mph": 35},
            {"overture_id": "seg_2", "speed_mph": 45},
        ]
    )
    overture = gpd.GeoDataFrame(
        [
            {
                "id": "seg_1",
                "geometry": LineString([(-111.9, 40.88), (-111.88, 40.88)]),
                "speed_limit_value": 35,
            },
            {
                "id": "seg_2",
                "geometry": LineString([(-111.9, 40.89), (-111.88, 40.89)]),
                "speed_limit_value": 40,
            },
        ],
        crs="EPSG:4326",
    )
    result = compare_to_overture(estimates, overture)
    assert len(result) == 2
    assert set(result.columns) >= {"overture_id", "our_speed_mph", "overture_speed_mph"}
    row1 = result[result["overture_id"] == "seg_1"].iloc[0]
    assert row1["our_speed_mph"] == 35
    assert row1["overture_speed_mph"] == 35


def test_compare_to_overture_drops_no_ground_truth():
    estimates = gpd.GeoDataFrame([{"overture_id": "seg_1", "speed_mph": 35}])
    overture = gpd.GeoDataFrame(
        [
            {
                "id": "seg_1",
                "geometry": None,
                "speed_limit_value": None,  # no ground truth
            }
        ]
    )
    result = compare_to_overture(estimates, overture)
    assert result.empty


def test_compare_to_overture_empty():
    result = compare_to_overture(gpd.GeoDataFrame(), gpd.GeoDataFrame())
    assert result.empty

