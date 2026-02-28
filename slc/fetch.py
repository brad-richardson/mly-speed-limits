"""Data acquisition helpers for the speed limit conflation pipeline.

Two usage modes:

* **Notebook / demo** — fetch small areas via the Mapillary REST API and the
  Overture CLI.  Suitable for city-sized bounding boxes.
* **Bulk pipeline** — swap to reading the full Mapillary dump (parquet /
  flatgeobuf) and Overture parquet from S3.  The signatures stay the same.

Ground truth for evaluation comes directly from the Overture segment
``speed_limits`` property — no separate OSM fetch is required.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import tempfile
import time
from typing import Any

import geopandas as gpd
import requests
from shapely.geometry import LineString, Point, shape

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MLY_GRAPH_URL = "https://graph.mapillary.com"
_MLY_SPEED_PATTERN = re.compile(
    r"(?:regulatory|complementary)--(?:maximum-speed-limit(?:-led)?|night-speed-limit)-(\d+)",
    re.IGNORECASE,
)
_MLY_SIGN_TYPE_PATTERN = re.compile(
    r"(regulatory|complementary)--(maximum-speed-limit(?:-led)?|night-speed-limit)",
    re.IGNORECASE,
)
_KMH_TO_MPH = 0.621371
_DEFAULT_CRS = "EPSG:4326"


# ---------------------------------------------------------------------------
# Mapillary API helpers
# ---------------------------------------------------------------------------


def _mly_get(
    endpoint: str, token: str, params: dict[str, Any], max_retries: int = 3
) -> dict[str, Any]:
    """Perform a single Mapillary Graph API GET request with retry."""
    params = dict(params)
    params["access_token"] = token
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                f"{_MLY_GRAPH_URL}/{endpoint}", params=params, timeout=60
            )
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.Timeout, requests.exceptions.HTTPError) as exc:
            last_exc = exc
            if isinstance(exc, requests.exceptions.HTTPError):
                if exc.response is not None and exc.response.status_code != 500:
                    raise
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise last_exc  # type: ignore[misc]  # unreachable, satisfies return type


def _split_bbox(
    bbox: tuple[float, float, float, float],
) -> list[tuple[float, float, float, float]]:
    """Split a bounding box into 4 quadrants."""
    min_lon, min_lat, max_lon, max_lat = bbox
    mid_lon = (min_lon + max_lon) / 2
    mid_lat = (min_lat + max_lat) / 2
    return [
        (min_lon, min_lat, mid_lon, mid_lat),  # SW
        (mid_lon, min_lat, max_lon, mid_lat),  # SE
        (min_lon, mid_lat, mid_lon, max_lat),  # NW
        (mid_lon, mid_lat, max_lon, max_lat),  # NE
    ]


def _mly_paginated_fetch(
    endpoint: str,
    token: str,
    bbox: tuple[float, float, float, float],
    fields: str,
    extra_params: dict[str, Any] | None = None,
    limit: int = 2000,
    _depth: int = 0,
    _max_depth: int = 6,
) -> list[dict[str, Any]]:
    """Fetch results from a Mapillary endpoint, subdividing on 500 errors."""
    min_lon, min_lat, max_lon, max_lat = bbox
    bbox_str = f"{min_lon},{min_lat},{max_lon},{max_lat}"

    params: dict[str, Any] = {"fields": fields, "bbox": bbox_str, "limit": limit}
    if extra_params:
        params.update(extra_params)

    try:
        data = _mly_get(endpoint, token, params)
        results = data.get("data", [])
        if len(results) >= limit:
            import warnings

            warnings.warn(
                f"Mapillary returned {len(results)} results (limit={limit}) for "
                f"bbox {bbox_str}; results may be truncated.",
                stacklevel=2,
            )
        return results
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 500:
            if _depth >= _max_depth:
                raise
            all_data: list[dict[str, Any]] = []
            for sub_bbox in _split_bbox(bbox):
                all_data.extend(
                    _mly_paginated_fetch(
                        endpoint, token, sub_bbox, fields,
                        extra_params, limit,
                        _depth=_depth + 1, _max_depth=_max_depth,
                    )
                )
            return all_data
        raise


def _parse_speed_mph(value: str, unit: str = "mph") -> int | None:
    """Extract a speed value from a Mapillary object_value string, in mph.

    Matches the following Mapillary sign families:

    * ``regulatory--maximum-speed-limit-<speed>``  (standard)
    * ``complementary--maximum-speed-limit-<speed>``  (supplementary)
    * ``regulatory--maximum-speed-limit-led-<speed>``  (LED/variable)
    * ``regulatory--night-speed-limit-<speed>``  (night-time)

    The numeric value on the sign is in the native unit of the country.
    Use *unit* to specify what unit the sign values are in:

    * ``"mph"`` — values are already miles-per-hour (US, UK).
    * ``"kmh"`` — values are kilometres-per-hour; converted to mph.

    Returns ``None`` when the value cannot be parsed.
    """
    if unit not in ("mph", "kmh"):
        raise ValueError(f"Unknown unit {unit!r}; expected 'mph' or 'kmh'")
    m = _MLY_SPEED_PATTERN.search(value)
    if not m:
        return None
    raw = int(m.group(1))
    if unit == "kmh":
        return round(raw * _KMH_TO_MPH)
    return raw


def _parse_sign_type(value: str) -> str:
    """Classify a Mapillary speed-limit sign into a type category.

    Returns one of ``"standard"``, ``"led"``, ``"night"``,
    ``"complementary"``, or ``"unknown"``.
    """
    m = _MLY_SIGN_TYPE_PATTERN.search(value)
    if not m:
        return "unknown"
    prefix, kind = m.group(1).lower(), m.group(2).lower()
    if prefix == "complementary":
        return "complementary"
    if "led" in kind:
        return "led"
    if "night" in kind:
        return "night"
    return "standard"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_mapillary_signs(
    bbox: tuple[float, float, float, float],
    token: str,
    speed_values: list[str] | None = None,
    unit: str = "mph",
) -> gpd.GeoDataFrame:
    """Fetch speed limit sign detections from the Mapillary API.

    Handles cursor-based pagination automatically.  Only detections whose
    ``object_value`` can be parsed into a valid speed limit are returned.

    Args:
        bbox: ``(min_lon, min_lat, max_lon, max_lat)`` in WGS-84.
        token: Mapillary access token.
        speed_values: Optional allow-list of ``object_value`` strings.  When
                      ``None``, all parseable speed limit signs are returned.
        unit: Unit the sign values are in — ``"mph"`` (US/UK) or ``"kmh"``
              (most other countries).  When ``"kmh"``, values are converted
              to mph.

    Returns:
        GeoDataFrame with columns
        ``id, geometry, speed_mph, raw_value, sign_type, confidence, heading``.
    """
    # Fetch each sign family separately to use server-side filtering
    raw_data: list[dict[str, Any]] = []
    for prefix in (
        "regulatory--maximum-speed-limit",
        "complementary--maximum-speed-limit",
        "regulatory--night-speed-limit",
    ):
        raw_data.extend(
            _mly_paginated_fetch(
                "map_features",
                token,
                bbox,
                fields="id,geometry,object_value,value",
                extra_params={"object_value": prefix},
            )
        )

    seen_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    for feat in raw_data:
        fid = feat.get("id")
        if fid in seen_ids:
            continue
        seen_ids.add(fid)
        raw = feat.get("object_value", "")
        if speed_values and raw not in speed_values:
            continue
        mph = _parse_speed_mph(raw, unit=unit)
        if mph is None:
            continue
        geom_data = feat.get("geometry", {})
        if not geom_data:
            continue
        geom = shape(geom_data) if isinstance(geom_data, dict) else Point(geom_data)
        rows.append(
            {
                "id": fid,
                "geometry": geom,
                "speed_mph": mph,
                "raw_value": raw,
                "sign_type": _parse_sign_type(raw),
                "confidence": feat.get("value", 1.0),
                "heading": None,
            }
        )

    cols = ["id", "geometry", "speed_mph", "raw_value", "sign_type", "confidence", "heading"]
    if not rows:
        return gpd.GeoDataFrame(columns=cols, crs=_DEFAULT_CRS)
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
    raw_data = _mly_paginated_fetch(
        "images", token, bbox, fields="id,sequence,geometry,compass_angle,captured_at"
    )

    seen_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    for img in raw_data:
        img_id = img.get("id")
        if img_id in seen_ids:
            continue
        seen_ids.add(img_id)
        geom_data = img.get("geometry", {})
        if not geom_data:
            continue
        geom = shape(geom_data) if isinstance(geom_data, dict) else Point(geom_data)
        sequence = img.get("sequence", "")
        if isinstance(sequence, dict):
            sequence = sequence.get("id", "")
        rows.append(
            {
                "id": img_id,
                "sequence_id": sequence,
                "geometry": geom,
                "compass_angle": img.get("compass_angle"),
                "captured_at": img.get("captured_at"),
            }
        )

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

    try:
        subprocess.run(cmd, check=True)
        gdf = gpd.read_parquet(tmp_path)
    finally:
        os.unlink(tmp_path)
    return gdf


def extract_overture_speed_limits(segments: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Parse the ``speed_limits`` column of an Overture segments GeoDataFrame.

    Overture transportation segments carry a ``speed_limits`` field that is
    already normalized — each entry is a struct with a ``max_speed`` sub-struct
    containing ``value`` (int) and ``unit`` (string, e.g. ``"mph"``).

    This function picks the primary unconditional max speed for each segment
    (first entry where ``when`` is absent or ``None``) and adds it as a new
    ``speed_limit_value`` column.  Segments without a parseable speed limit
    receive ``None``.

    Args:
        segments: GeoDataFrame returned by :func:`fetch_overture_segments`.

    Returns:
        The same GeoDataFrame with an additional ``speed_limit_value`` column
        (int or ``None``).
    """
    if "speed_limits" not in segments.columns:
        segments = segments.copy()
        segments["speed_limit_value"] = None
        return segments

    def _pick_primary(cell: Any) -> int | None:
        if cell is None:
            return None
        try:
            entries = list(cell)
        except (TypeError, ValueError):
            return None
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            # Prefer unconditional entries (when is None / absent)
            when = entry.get("when")
            if when is not None:
                continue
            ms = entry.get("max_speed")
            if isinstance(ms, dict):
                val = ms.get("value")
                if val is not None:
                    return int(val)
        # Fall back to first entry regardless of condition
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            ms = entry.get("max_speed")
            if isinstance(ms, dict):
                val = ms.get("value")
                if val is not None:
                    return int(val)
        return None

    result = segments.copy()
    result["speed_limit_value"] = result["speed_limits"].apply(_pick_primary)
    return result


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
