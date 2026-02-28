# mly-speed-limits

Derive speed limit estimates for [Overture Maps](https://overturemaps.org/)
transportation segments by projecting Mapillary sign detections onto Mapillary
image sequences, splitting those sequences into labeled edges, and matching the
labeled edges to Overture road segments.

**Output**: An Overture segment-level speed limit dataset with linear references,
confidence scores, and a comparison against OSM `maxspeed` tags as a ground
truth proxy.

---

## Architecture

```
speed-limit-conflation/
├── slc/                          # Core library ("speed limit conflation")
│   ├── __init__.py
│   ├── fetch.py                  # Mapillary API + Overture data fetching
│   ├── snap.py                   # Point-to-line snapping, heading validation
│   ├── split.py                  # Sequence splitting (signs + bearing change)
│   ├── match.py                  # Split edge → Overture segment matching
│   ├── consensus.py              # Multi-observation voting / LR assignment
│   ├── evaluate.py               # OSM comparison, metrics, QA
│   ├── viz.py                    # Folium map helpers
│   └── types.py                  # Dataclasses: SpeedSign, SplitEdge, SpeedEstimate
├── notebooks/
│   ├── 01_fetch_and_explore.ipynb
│   ├── 02_split_sequences.ipynb
│   ├── 03_match_to_overture.ipynb
│   ├── 04_consensus_and_evaluate.ipynb
│   └── 05_full_pipeline.ipynb
├── tests/
├── pyproject.toml
└── README.md
```

---

## Quick Start

```bash
pip install -e ".[dev]"
export MAPILLARY_ACCESS_TOKEN='<your token>'
jupyter notebook notebooks/05_full_pipeline.ipynb
```

### Run the tests

```bash
pytest tests/
```

---

## Pipeline Overview

| Step | Module | Description |
|------|--------|-------------|
| 1 | `fetch` | Fetch Mapillary signs, images, Overture segments, OSM ways |
| 2 | `snap` | Project each sign onto the nearest qualifying sequence |
| 3 | `split` | Split sequences at sign locations and turn points |
| 4 | `match` | Match labeled split edges to Overture segments (with LR) |
| 5 | `consensus` | Aggregate multiple observations per segment |
| 6 | `evaluate` | Compare against OSM `maxspeed`, compute metrics |

---

## Dependencies

- Python ≥ 3.11
- geopandas ≥ 1.0, shapely ≥ 2.0
- requests, pyarrow, folium, matplotlib
- overturemaps ≥ 0.9 (for `fetch_overture_segments`)

Install everything:

```bash
pip install -e ".[dev]"
```
