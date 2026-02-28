"""Speed Limit Conflation (slc) — core library.

Derives speed limit estimates for Overture transportation segments by projecting
Mapillary sign detections onto Mapillary image sequences, splitting those sequences
into labeled edges, and matching the labeled edges to Overture road segments.
"""

from slc.types import SpeedEstimate, SpeedSign, SplitEdge

__all__ = ["SpeedSign", "SplitEdge", "SpeedEstimate"]
