"""Data acquisition helpers for the speed limit conflation pipeline.

Two usage modes:

* **Notebook / demo** — fetch small areas via the Mapillary REST API and the
  Overture CLI.  Suitable for city-sized bounding boxes.
* **Bulk pipeline** — swap to reading the full Mapillary dump (parquet /
  flatgeobuf) and Overture parquet from S3.  The signatures stay the same.
"""

from __future__ import annotations

import math
import re
from typing import Any

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import LineString, Point, shape

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MLY_GRAPH_URL = "https://graph.mapillary.com"
_MLY_SPEED_PATTERN = re.compile(
    r"regulatory--maximum-speed-limit--(\d+)", re.IGNORECASE
)
_DEFAULT_CRS = "EPSG:4326"


# ---------------------------------------------------------------------------
# Mapillary API helpers
# ---------------------------------------------------------------------------


def _mly_get(endpoint: str, token: str, params: dict[str, Any]) -> dict[str, Any]:
    """Perform a single Mapillary Graph API GET request."""
    params = dict(params)
    params["access_token"] = token
    resp = requests.get(f"{_MLY_GRAPH_URL}/{endpoint}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _parse_speed_mph(value: str) -> int | None:
    """Extract an integer mph value from a Mapillary object_value string.

    Mapillary encodes speed limits as
    ``regulatory--maximum-speed-limit--<speed>`` where *<speed>* is in
    the native unit of the country.  For the US demo area we assume mph.

    Returns ``None`` when the value cannot be parsed.
    """
    m = _MLY_SPEED_PATTERN.search(value)
    if m:
        return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_mapillary_signs(
    bbox: tuple[float, float, float, float],
    token: str,
    speed_values: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """Fetch speed limit sign detections from the Mapillary API.

    Handles cursor-based pagination automatically.  Only detections whose
    ``object_value`` can be parsed into a valid speed limit are returned.

    Args:
        bbox: ``(min_lon, min_lat, max_lon, max_lat)`` in WGS-84.
        token: Mapillary access token.
        speed_values: Optional allow-list of ``object_value`` strings.  When
                      ``None``, all parseable speed limit signs are returned.

    Returns:
        GeoDataFrame with columns
        ``id, geometry, speed_mph, raw_value, confidence, heading``.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    bbox_str = f"{min_lon},{min_lat},{max_lon},{max_lat}"

    fields = "id,geometry,object_value,value,first_seen_at,last_seen_at"
    params: dict[str, Any] = {
        "fields": fields,
        "bbox": bbox_str,
        "object_value": "regulatory--maximum-speed-limit",
        "limit": 2000,
    }

    rows: list[dict[str, Any]] = []
    while True:
        data = _mly_get("map_features", token, params)
        for feat in data.get("data", []):
            raw = feat.get("object_value", "")
            if speed_values and raw not in speed_values:
                continue
            mph = _parse_speed_mph(raw)
            if mph is None:
                continue
            geom_data = feat.get("geometry", {})
            if not geom_data:
                continue
            geom = shape(geom_data) if isinstance(geom_data, dict) else Point(geom_data)
            rows.append(
                {
                    "id": feat["id"],
                    "geometry": geom,
                    "speed_mph": mph,
                    "raw_value": raw,
                    "confidence": feat.get("value", 1.0),
                    "heading": None,  # Populated later via fetch_mapillary_images
                }
            )

        cursor = data.get("paging", {}).get("next")
        if not cursor:
            break
        params["after"] = cursor

    if not rows:
        return gpd.GeoDataFrame(
            columns=["id", "geometry", "speed_mph", "raw_value", "confidence", "heading"],
            crs=_DEFAULT_CRS,
        )
    return gpd.GeoDataFrame(rows, crs=_DEFAULT_CRS)


def fetch_mapillary_images(
    bbox: tuple[float, float, float, float],
    token: str,
) -> gpd.GeoDataFrame:
    """Fetch Mapillary images with sequence IDs and metadata.

    Returns one row per image, sorted by ``captured_at`` within each
    sequence, with columns
    ``id, sequence_id, geometry, compass_angle, captured_at``.

    Args:
        bbox: ``(min_lon, min_lat, max_lon, max_lat)`` in WGS-84.
        token: Mapillary access token.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    bbox_str = f"{min_lon},{min_lat},{max_lon},{max_lat}"

    fields = "id,sequence,geometry,compass_angle,captured_at"
    params: dict[str, Any] = {
        "fields": fields,
        "bbox": bbox_str,
        "limit": 2000,
    }

    rows: list[dict[str, Any]] = []
    while True:
        data = _mly_get("images", token, params)
        for img in data.get("data", []):
            geom_data = img.get("geometry", {})
            if not geom_data:
                continue
            geom = shape(geom_data) if isinstance(geom_data, dict) else Point(geom_data)
            rows.append(
                {
                    "id": img["id"],
                    "sequence_id": img.get("sequence", ""),
                    "geometry": geom,
                    "compass_angle": img.get("compass_angle"),
                    "captured_at": img.get("captured_at"),
                }
            )

        cursor = data.get("paging", {}).get("next")
        if not cursor:
            break
        params["after"] = cursor

    if not rows:
        return gpd.GeoDataFrame(
            columns=["id", "sequence_id", "geometry", "compass_angle", "captured_at"],
            crs=_DEFAULT_CRS,
        )

    gdf = gpd.GeoDataFrame(rows, crs=_DEFAULT_CRS)
    gdf = gdf.sort_values(["sequence_id", "captured_at"]).reset_index(drop=True)
    return gdf


def build_sequences(images_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Group individual images into per-sequence LineString geometries.

    Args:
        images_gdf: Output of :func:`fetch_mapillary_images`.

    Returns:
        GeoDataFrame with one row per sequence and columns
        ``sequence_id, geometry, image_count, image_headings``.
        Sequences with fewer than 2 images are dropped (cannot form a line).
    """
    rows: list[dict[str, Any]] = []
    for seq_id, grp in images_gdf.groupby("sequence_id"):
        grp = grp.sort_values("captured_at")
        coords = [(pt.x, pt.y) for pt in grp.geometry]
        if len(coords) < 2:
            continue
        headings = grp["compass_angle"].tolist()
        rows.append(
            {
                "sequence_id": seq_id,
                "geometry": LineString(coords),
                "image_count": len(coords),
                "image_headings": headings,
            }
        )

    if not rows:
        return gpd.GeoDataFrame(
            columns=["sequence_id", "geometry", "image_count", "image_headings"],
            crs=_DEFAULT_CRS,
        )
    return gpd.GeoDataFrame(rows, crs=_DEFAULT_CRS)


def fetch_overture_segments(
    bbox: tuple[float, float, float, float],
    release: str | None = None,
) -> gpd.GeoDataFrame:
    """Fetch Overture transportation segments for the given bounding box.

    Uses the ``overturemaps`` CLI to download segments as GeoParquet and
    reads them back into a GeoDataFrame.  Requires the ``overturemaps``
    package (``pip install overturemaps``).

    Args:
        bbox: ``(min_lon, min_lat, max_lon, max_lat)`` in WGS-84.
        release: Overture release string (e.g. ``"2024-04-16-beta.0"``).
                 When ``None``, the CLI default (latest stable) is used.

    Returns:
        GeoDataFrame with at minimum the columns
        ``id, geometry, class, subclass, names, speed_limits, connectors``.
    """
    import subprocess
    import tempfile

    min_lon, min_lat, max_lon, max_lat = bbox
    bbox_str = f"{min_lon},{min_lat},{max_lon},{max_lat}"

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name

    cmd = [
        "overturemaps",
        "download",
        f"--bbox={bbox_str}",
        "--type=segment",
        "-f",
        "geoparquet",
        "-o",
        tmp_path,
    ]
    if release:
        cmd += ["--release", release]

    subprocess.run(cmd, check=True)
    gdf = gpd.read_parquet(tmp_path)
    return gdf


def fetch_osm_maxspeed(
    bbox: tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    """Fetch OSM ways with ``maxspeed`` tags via the Overpass API.

    Used as a ground-truth proxy for evaluation.

    Args:
        bbox: ``(min_lon, min_lat, max_lon, max_lat)`` in WGS-84.

    Returns:
        GeoDataFrame with columns ``osm_id, geometry, maxspeed, name``.
        Only ways that carry a ``maxspeed`` tag are returned.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    # Overpass uses (south, west, north, east)
    overpass_bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"

    query = f"""
    [out:json][timeout:60];
    way["maxspeed"]({overpass_bbox});
    out geom;
    """

    resp = requests.post(
        "https://overpass-api.de/api/interpreter",
        data={"data": query},
        timeout=90,
    )
    resp.raise_for_status()
    result = resp.json()

    rows: list[dict[str, Any]] = []
    for element in result.get("elements", []):
        if element.get("type") != "way":
            continue
        nodes = element.get("geometry", [])
        if len(nodes) < 2:
            continue
        coords = [(n["lon"], n["lat"]) for n in nodes]
        tags = element.get("tags", {})
        maxspeed_raw = tags.get("maxspeed", "")
        mph = _parse_maxspeed_tag(maxspeed_raw)
        rows.append(
            {
                "osm_id": element["id"],
                "geometry": LineString(coords),
                "maxspeed": maxspeed_raw,
                "maxspeed_mph": mph,
                "name": tags.get("name", ""),
            }
        )

    if not rows:
        return gpd.GeoDataFrame(
            columns=["osm_id", "geometry", "maxspeed", "maxspeed_mph", "name"],
            crs=_DEFAULT_CRS,
        )
    return gpd.GeoDataFrame(rows, crs=_DEFAULT_CRS)


def _parse_maxspeed_tag(value: str) -> int | None:
    """Parse an OSM ``maxspeed`` tag value into mph.

    Handles plain integers (assumed mph in the US), ``XX mph``, and
    ``XX km/h`` formats.  Returns ``None`` for unparseable values.
    """
    if not value:
        return None
    value = value.strip()
    # "35 mph" or "35mph"
    m = re.match(r"^(\d+)\s*mph$", value, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # "56 km/h" or "56kmh"
    m = re.match(r"^(\d+)\s*km/?h$", value, re.IGNORECASE)
    if m:
        return int(round(int(m.group(1)) / 1.60934))
    # bare integer — assume mph for US
    m = re.match(r"^(\d+)$", value)
    if m:
        return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Internal geometry utilities (used here and in snap.py / split.py)
# ---------------------------------------------------------------------------


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Return the great-circle distance in metres between two WGS-84 points."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def linestring_length_m(line: LineString) -> float:
    """Approximate arc length of a WGS-84 LineString in metres."""
    coords = list(line.coords)
    total = 0.0
    for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
        total += haversine_distance(x1, y1, x2, y2)
    return total


def bearing(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Compute the initial bearing (degrees, 0 = north) from point 1 to point 2."""
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlon_r = math.radians(lon2 - lon1)
    x = math.sin(dlon_r) * math.cos(lat2_r)
    y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(
        dlon_r
    )
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def angle_diff(a: float, b: float) -> float:
    """Smallest absolute angular difference between two bearings (0–180°)."""
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)
