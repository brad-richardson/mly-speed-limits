"""Evaluation: compare our speed estimates against OSM maxspeed ground truth."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd

# ---------------------------------------------------------------------------
# OSM matching
# ---------------------------------------------------------------------------


def match_to_osm(
    estimates: gpd.GeoDataFrame,
    osm_ways: gpd.GeoDataFrame,
    overture_segments: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Spatially match our speed estimates to OSM ways for comparison.

    Both Overture and OSM geometries are derived from similar sources so the
    geometries are close.  This function uses a simple buffer overlap approach:
    for each Overture segment that has an estimate, find the OSM way with the
    largest intersection area (buffered geometries).

    Args:
        estimates: GeoDataFrame from :func:`slc.consensus.compute_consensus`.
                   Must have columns ``overture_id, speed_mph``.
        osm_ways: GeoDataFrame from :func:`slc.fetch.fetch_osm_maxspeed`.
                  Must have columns ``osm_id, geometry, maxspeed_mph``.
        overture_segments: GeoDataFrame from
                           :func:`slc.fetch.fetch_overture_segments`.
                           Must have columns ``id, geometry``.

    Returns:
        GeoDataFrame with columns
        ``overture_id, our_speed_mph, osm_maxspeed_mph, osm_id, match_quality``.
    """
    if estimates.empty or osm_ways.empty or overture_segments.empty:
        return gpd.GeoDataFrame(
            columns=[
                "overture_id",
                "our_speed_mph",
                "osm_maxspeed_mph",
                "osm_id",
                "match_quality",
            ]
        )

    # Join estimates with Overture geometries
    segs = overture_segments[["id", "geometry"]].rename(columns={"id": "overture_id"})
    est_with_geom = estimates.merge(segs, on="overture_id", how="left")

    if "geometry" not in est_with_geom.columns:
        return gpd.GeoDataFrame(
            columns=[
                "overture_id",
                "our_speed_mph",
                "osm_maxspeed_mph",
                "osm_id",
                "match_quality",
            ]
        )

    est_gdf = gpd.GeoDataFrame(est_with_geom, geometry="geometry", crs=overture_segments.crs)

    # Project to a metres-based CRS for buffering
    try:
        est_proj = est_gdf.to_crs(epsg=3857)
        osm_proj = osm_ways.to_crs(epsg=3857)
    except Exception:
        est_proj = est_gdf
        osm_proj = osm_ways

    buffer_m = 20.0
    est_proj = est_proj.copy()
    est_proj["_buf"] = est_proj.geometry.buffer(buffer_m)

    osm_proj_idx = osm_proj.copy()
    osm_proj_idx = osm_proj_idx.set_index("osm_id")

    from shapely.strtree import STRtree

    osm_geoms = list(osm_proj_idx.geometry)
    tree = STRtree(osm_geoms)
    osm_ids_list = list(osm_proj_idx.index)

    rows: list[dict[str, Any]] = []
    for _, est_row in est_proj.iterrows():
        buf = est_row["_buf"]
        cands = tree.query(buf)
        best_osm_id = None
        best_quality = 0.0

        for idx in cands:
            osm_geom = osm_geoms[idx]
            try:
                inter = buf.intersection(osm_geom.buffer(buffer_m))
                quality = inter.area / buf.area if buf.area > 0 else 0.0
            except Exception:
                quality = 0.0

            if quality > best_quality:
                best_quality = quality
                best_osm_id = osm_ids_list[idx]

        if best_osm_id is None:
            continue

        osm_speed = osm_proj_idx.loc[best_osm_id, "maxspeed_mph"]
        rows.append(
            {
                "overture_id": est_row["overture_id"],
                "our_speed_mph": int(est_row["speed_mph"]),
                "osm_maxspeed_mph": osm_speed,
                "osm_id": best_osm_id,
                "match_quality": round(float(best_quality), 4),
            }
        )

    if not rows:
        return gpd.GeoDataFrame(
            columns=[
                "overture_id",
                "our_speed_mph",
                "osm_maxspeed_mph",
                "osm_id",
                "match_quality",
            ]
        )

    return gpd.GeoDataFrame(rows)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(comparison: gpd.GeoDataFrame) -> dict[str, Any]:
    """Compute evaluation metrics comparing our estimates to OSM maxspeed.

    Args:
        comparison: Output of :func:`match_to_osm`.

    Returns:
        Dictionary with keys:
        ``exact_match_rate, within_5mph_rate, within_10mph_rate,
        coverage, conflict_rate, n_estimates, n_with_osm, confusion_matrix``.
    """
    if comparison.empty:
        return {
            "exact_match_rate": None,
            "within_5mph_rate": None,
            "within_10mph_rate": None,
            "n_estimates": 0,
            "n_with_osm": 0,
            "confusion_matrix": {},
        }

    # Only rows where OSM speed is available and parseable
    valid = comparison.dropna(subset=["osm_maxspeed_mph", "our_speed_mph"])

    n_total = len(comparison)
    n_valid = len(valid)

    if n_valid == 0:
        return {
            "exact_match_rate": None,
            "within_5mph_rate": None,
            "within_10mph_rate": None,
            "n_estimates": n_total,
            "n_with_osm": 0,
            "confusion_matrix": {},
        }

    our = valid["our_speed_mph"].astype(int)
    osm = valid["osm_maxspeed_mph"].astype(int)
    diff = (our - osm).abs()

    exact = int((diff == 0).sum())
    within5 = int((diff <= 5).sum())
    within10 = int((diff <= 10).sum())

    # Confusion matrix: our_speed → osm_speed → count
    conf_matrix: dict[int, dict[int, int]] = {}
    for our_val, osm_val in zip(our, osm):
        conf_matrix.setdefault(int(our_val), {})
        conf_matrix[int(our_val)][int(osm_val)] = (
            conf_matrix[int(our_val)].get(int(osm_val), 0) + 1
        )

    return {
        "exact_match_rate": round(exact / n_valid, 4),
        "within_5mph_rate": round(within5 / n_valid, 4),
        "within_10mph_rate": round(within10 / n_valid, 4),
        "n_estimates": n_total,
        "n_with_osm": n_valid,
        "confusion_matrix": conf_matrix,
    }


def generate_report(comparison: gpd.GeoDataFrame, metrics: dict[str, Any]) -> str:
    """Generate a Markdown summary of the evaluation results.

    Args:
        comparison: Output of :func:`match_to_osm`.
        metrics: Output of :func:`compute_metrics`.

    Returns:
        Multi-line Markdown string.
    """
    lines = [
        "# Speed Limit Conflation — Evaluation Report",
        "",
        "## Coverage",
        f"- Total Overture segments with estimates: **{metrics.get('n_estimates', 0)}**",
        f"- Segments matched to OSM for comparison: **{metrics.get('n_with_osm', 0)}**",
        "",
        "## Accuracy vs OSM maxspeed",
    ]

    for key, label in [
        ("exact_match_rate", "Exact match"),
        ("within_5mph_rate", "Within 5 mph"),
        ("within_10mph_rate", "Within 10 mph"),
    ]:
        val = metrics.get(key)
        if val is not None:
            lines.append(f"- {label}: **{val * 100:.1f}%**")
        else:
            lines.append(f"- {label}: N/A")

    conf = metrics.get("confusion_matrix", {})
    if conf:
        lines += [
            "",
            "## Confusion Matrix (our estimate → OSM value → count)",
            "",
            "| Our \\ OSM | " + " | ".join(str(v) for v in sorted({
                osm_val
                for row in conf.values()
                for osm_val in row
            })) + " |",
        ]
        osm_vals = sorted({osm_val for row in conf.values() for osm_val in row})
        lines.append("| --- | " + " | ".join(["---"] * len(osm_vals)) + " |")
        for our_val in sorted(conf):
            row_counts = conf[our_val]
            cells = " | ".join(str(row_counts.get(v, 0)) for v in osm_vals)
            lines.append(f"| {our_val} | {cells} |")

    return "\n".join(lines) + "\n"
