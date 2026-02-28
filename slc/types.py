"""Data model dataclasses for the speed limit conflation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from shapely.geometry import LineString, Point


@dataclass
class SpeedSign:
    """A single speed limit sign detection from Mapillary.

    Attributes:
        id: Mapillary detection / object ID.
        geometry: Estimated sign location as a WGS-84 Point.
        speed_mph: Parsed speed limit in miles-per-hour.
        raw_value: The raw Mapillary ``object_value`` string (e.g. ``"regulatory--maximum-speed-limit--35"``)
        confidence: Mapillary detection confidence (0–1).
        heading: Camera compass heading at the moment of detection (degrees, 0 = north).
                 ``None`` when the image metadata is unavailable.
    """

    id: str
    geometry: Point
    speed_mph: int
    raw_value: str
    confidence: float
    heading: float | None = None


@dataclass
class SplitEdge:
    """A sub-segment of a Mapillary sequence between two split points.

    Split points are created either at sign locations (speed changes) or at
    bearing-change turns.

    Attributes:
        sequence_id: Parent Mapillary sequence ID.
        geometry: LineString sub-segment in WGS-84.
        speed_mph: Speed limit that applies to this edge, or ``None`` when the
                   edge precedes the first sign or immediately follows a turn.
        split_reason: Why the edge starts here —
                      ``"start"`` | ``"sign"`` | ``"turn"`` | ``"end"``.
        length_m: Approximate arc length of *geometry* in metres.
        avg_heading: Mean bearing (degrees, 0 = north) along this edge.
        edge_id: Stable identifier built from *sequence_id* and a 0-based index.
    """

    sequence_id: str
    geometry: LineString
    speed_mph: int | None
    split_reason: str
    length_m: float
    avg_heading: float
    edge_id: str = ""

    def __post_init__(self) -> None:
        if not self.edge_id:
            raise ValueError("edge_id must be set explicitly after construction")


@dataclass
class SpeedEstimate:
    """A speed limit estimate attached to an Overture road segment.

    Attributes:
        overture_id: The GERS (Global Entity Reference System) ID of the Overture segment.
        speed_mph: Estimated speed limit in miles-per-hour.
        lr_start: Linear reference start position along the Overture segment (0–1).
        lr_end: Linear reference end position along the Overture segment (0–1).
        confidence: Combined confidence score (0–1) based on observation count
                    and inter-observation agreement.
        observation_count: Number of independent Mapillary observations that
                           contributed to this estimate.
        source_edges: IDs of the :class:`SplitEdge` objects that contributed.
    """

    overture_id: str
    speed_mph: int
    lr_start: float
    lr_end: float
    confidence: float
    observation_count: int
    source_edges: list[str] = field(default_factory=list)
