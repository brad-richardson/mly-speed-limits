"""Visualization helpers for the speed limit conflation pipeline.

Provides consistent color mapping and Folium map builders for each pipeline
stage.  All functions return a :class:`folium.Map` that can be displayed
directly in a Jupyter notebook or saved to HTML.
"""

from __future__ import annotations

from typing import Any

# Folium is an optional runtime dependency; guard imports so that the rest of
# the library can be imported without it.
try:
    import folium
    from folium import PolyLine, Popup

    _FOLIUM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FOLIUM_AVAILABLE = False

import geopandas as gpd

# ---------------------------------------------------------------------------
# Color mapping
# ---------------------------------------------------------------------------

_SPEED_COLORS: dict[int | None, str] = {
    None: "#aaaaaa",
    15: "#4575b4",
    20: "#74add1",
    25: "#abd9e9",
    30: "#e0f3f8",
    35: "#fee090",
    40: "#fdae61",
    45: "#f46d43",
    50: "#d73027",
    55: "#a50026",
    60: "#91001d",
    65: "#7b0014",
    70: "#4d0008",
}


def speed_color(mph: int | None) -> str:
    """Return a consistent hex color for a speed limit value.

    Known values map to a diverging palette from cool (low) to warm (high).
    Unknown speeds map to a neutral gray.

    Args:
        mph: Speed limit in mph, or ``None`` for unlabeled edges.

    Returns:
        Hex color string (e.g. ``"#fee090"``).
    """
    if mph in _SPEED_COLORS:
        return _SPEED_COLORS[mph]
    # Fallback: interpolate toward red for higher values
    if mph is not None and mph > 70:
        return "#2d0004"
    return _SPEED_COLORS[None]


# ---------------------------------------------------------------------------
# Map builders
# ---------------------------------------------------------------------------


def _require_folium() -> None:
    if not _FOLIUM_AVAILABLE:
        raise ImportError(
            "folium is required for visualization. "
            "Install it with: pip install folium"
        )


def _default_center(
    gdf: gpd.GeoDataFrame,
) -> tuple[float, float]:
    """Return (lat, lon) centroid of *gdf*."""
    total_bounds = gdf.total_bounds  # minx, miny, maxx, maxy
    lat = (total_bounds[1] + total_bounds[3]) / 2
    lon = (total_bounds[0] + total_bounds[2]) / 2
    return lat, lon


def map_signs_and_sequences(
    signs: gpd.GeoDataFrame,
    sequences: gpd.GeoDataFrame,
    center: tuple[float, float] | None = None,
) -> Any:
    """Quick overview map of raw Mapillary data.

    Args:
        signs: GeoDataFrame from :func:`slc.fetch.fetch_mapillary_signs`.
        sequences: GeoDataFrame from :func:`slc.fetch.build_sequences`.
        center: ``(lat, lon)`` map center.  Derived from data when ``None``.

    Returns:
        :class:`folium.Map` with signs as circle markers and sequences as
        blue polylines.
    """
    _require_folium()

    if center is None:
        if not sequences.empty:
            center = _default_center(sequences)
        elif not signs.empty:
            center = _default_center(signs)
        else:
            center = (0.0, 0.0)

    m = folium.Map(location=center, zoom_start=14)

    # Sequences
    for _, row in sequences.iterrows():
        coords = [(y, x) for x, y in row.geometry.coords]
        PolyLine(coords, color="blue", weight=2, opacity=0.6).add_to(m)

    # Signs
    for _, row in signs.iterrows():
        pt = row.geometry
        folium.CircleMarker(
            location=(pt.y, pt.x),
            radius=6,
            color="red",
            fill=True,
            fill_color="red",
            popup=Popup(f"Sign {row['id']}: {row['speed_mph']} mph"),
        ).add_to(m)

    return m


def map_split_edges(
    split_edges: gpd.GeoDataFrame,
    signs: gpd.GeoDataFrame | None = None,
    overture: gpd.GeoDataFrame | None = None,
) -> Any:
    """Map split edges colored by speed limit.

    Args:
        split_edges: GeoDataFrame from :func:`slc.split.split_all_sequences`.
        signs: Optional signs overlay.
        overture: Optional Overture segments overlay (gray polylines).

    Returns:
        :class:`folium.Map`.
    """
    _require_folium()

    center = _default_center(split_edges) if not split_edges.empty else (0.0, 0.0)
    m = folium.Map(location=center, zoom_start=14)

    # Overture background
    if overture is not None and not overture.empty:
        for _, row in overture.iterrows():
            coords = [(y, x) for x, y in row.geometry.coords]
            PolyLine(coords, color="#cccccc", weight=1.5, opacity=0.5).add_to(m)

    # Split edges
    for _, row in split_edges.iterrows():
        color = speed_color(row.get("speed_mph"))
        coords = [(y, x) for x, y in row.geometry.coords]
        PolyLine(
            coords,
            color=color,
            weight=4,
            opacity=0.8,
            popup=Popup(f"Edge {row['edge_id']}: {row.get('speed_mph')} mph"),
        ).add_to(m)

    # Signs overlay
    if signs is not None and not signs.empty:
        for _, row in signs.iterrows():
            pt = row.geometry
            folium.CircleMarker(
                location=(pt.y, pt.x),
                radius=5,
                color="black",
                fill=True,
                fill_color="yellow",
                popup=Popup(f"{row['speed_mph']} mph"),
            ).add_to(m)

    return m


def map_estimates(
    estimates: gpd.GeoDataFrame,
    overture_segments: gpd.GeoDataFrame,
) -> Any:
    """Final result map: Overture segments colored by estimated speed limit.

    Args:
        estimates: GeoDataFrame from :func:`slc.consensus.compute_consensus`.
        overture_segments: Full Overture segments GeoDataFrame.

    Returns:
        :class:`folium.Map`.
    """
    _require_folium()

    center = (
        _default_center(overture_segments) if not overture_segments.empty else (0.0, 0.0)
    )
    m = folium.Map(location=center, zoom_start=14)

    # Build lookup: overture_id → speed_mph
    speed_lookup = {}
    if not estimates.empty:
        for _, row in estimates.iterrows():
            speed_lookup[row["overture_id"]] = row.get("speed_mph")

    # Overture segments
    id_col = "id" if "id" in overture_segments.columns else overture_segments.columns[0]
    for _, row in overture_segments.iterrows():
        seg_id = row[id_col]
        mph = speed_lookup.get(seg_id)
        color = speed_color(mph)
        coords = [(y, x) for x, y in row.geometry.coords]
        PolyLine(
            coords,
            color=color,
            weight=4,
            opacity=0.85,
            popup=Popup(f"Segment {seg_id}: {mph} mph"),
        ).add_to(m)

    return m


def map_comparison(
    comparison: gpd.GeoDataFrame,
    overture_segments: gpd.GeoDataFrame,
) -> Any:
    """Highlight agreement / disagreement between our estimates and Overture speed limits.

    Color coding:
    * **Green** — exact match with Overture.
    * **Yellow** — within 5 mph.
    * **Orange** — within 10 mph.
    * **Red** — more than 10 mph off.
    * **Gray** — no Overture ground truth available.

    Args:
        comparison: Output of :func:`slc.evaluate.compare_to_overture`.
        overture_segments: Full Overture segments GeoDataFrame.

    Returns:
        :class:`folium.Map`.
    """
    _require_folium()

    center = (
        _default_center(overture_segments) if not overture_segments.empty else (0.0, 0.0)
    )
    m = folium.Map(location=center, zoom_start=14)

    # Build lookup: overture_id → comparison row
    comp_lookup: dict[str, dict] = {}
    if not comparison.empty:
        for _, row in comparison.iterrows():
            comp_lookup[row["overture_id"]] = row.to_dict()

    id_col = "id" if "id" in overture_segments.columns else overture_segments.columns[0]
    for _, row in overture_segments.iterrows():
        seg_id = row[id_col]
        comp = comp_lookup.get(seg_id)

        if comp is None:
            color = "#aaaaaa"
            popup_text = f"Segment {seg_id}: no ground truth"
        else:
            our = comp.get("our_speed_mph")
            truth = comp.get("overture_speed_mph")
            if our is None or truth is None:
                color = "#aaaaaa"
                popup_text = f"Segment {seg_id}: missing data"
            else:
                diff = abs(int(our) - int(truth))
                if diff == 0:
                    color = "#2ca25f"
                elif diff <= 5:
                    color = "#fed976"
                elif diff <= 10:
                    color = "#fd8d3c"
                else:
                    color = "#e31a1c"
                popup_text = (
                    f"Segment {seg_id}: ours={our} mph, "
                    f"Overture={truth} mph (Δ{diff})"
                )

        coords = [(y, x) for x, y in row.geometry.coords]
        PolyLine(
            coords,
            color=color,
            weight=4,
            opacity=0.9,
            popup=Popup(popup_text),
        ).add_to(m)

    return m
