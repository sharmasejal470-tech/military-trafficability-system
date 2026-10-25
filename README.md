# Automated Vehicle Trafficability Prediction & Mapping System
### Multi-Spectral Terrain, GIS & Autonomous Underwater Vehicle (AUV) Hydrographic Sonar Fusion

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Streamlit Cloud Ready](https://img.shields.io/badge/Streamlit-Ready-FF4B4B.svg)](https://streamlit.io/cloud)
[![Pure-Pip Compliant](https://img.shields.io/badge/Dependencies-Pure--Pip-brightgreen.svg)]()
[![Tests Passing](https://img.shields.io/badge/pytest-31%2F31%20passing-success.svg)]()

---

## 1. System Overview & Operational Purpose

Developed under defense geospatial specifications for **Sapta Shakti Command (Jaipur, Rajasthan)**, the **Automated Vehicle Trafficability Prediction & Mapping System** is an operational tactical intelligence decision-support platform. It fuses multi-spectral terrain layers (DEM elevation, slope dynamics, soil mechanics, surface moisture) with Autonomous Underwater Vehicle (AUV) hydrographic sonar bathymetry to produce real-time NATO-standard Go / Slow-Go / No-Go mobility surfaces for tracked and wheeled military platforms.

The system features:
- **Vectorized Terramechanics Classification**: Sub-second evaluation of $10,000$ contiguous $100\,\text{m} \times 100\,\text{m}$ tactical grid cells ($10\,\text{km} \times 10\,\text{km}$ theater of operations).
- **Hydrodynamic Fording Models**: Dynamically accounts for river current drag velocity and precipitation-induced substrate pore-pressure saturation.
- **Optimal A\* Tactical Routing**: 8-connected least-risk route planning with octile admissible heuristics, preventing corner-cutting and strictly prohibiting traversal of fatal No-Go terrain.
- **Interactive Geospatial Dashboard**: Dual-layer Folium Leaflet mapping (OpenStreetMap & Esri World Imagery) with fast in-memory RGBA raster overlays, longitudinal elevation/riverbed profiles, and tactical SITREP briefings.
- **Pure-Pip Streamlit Cloud Portability**: Zero dependence on C-extension geospatial binaries (`GDAL`, `geopandas`, `rasterio`), eliminating container build failures on Streamlit Community Cloud.

---

## 2. System Architecture

```text
+-----------------------------------------------------------------------------------+
|                            DEFENSE DATA INGESTION                                 |
|   +-----------------------+     +--------------------+     +------------------+   |
|   | Satellite DEM / Slope |     | Soil Classification|     | AUV Sonar Survey |   |
|   | (Elevation, Gradients)|     | (Marsh, Clay, Loam)|     | (Depth, Firmness)|   |
|   +-----------+-----------+     +---------+----------+     +--------+---------+   |
|               |                           |                         |             |
|               +--------------------+      |      +------------------+             |
|                                    v      v      v                                |
|                        [ core/io_adapters.py: cKDTree Ingest ]                    |
+--------------------------------------------+--------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------+
|                        CORE GEOSPATIAL SIMULATION PIPELINE                        |
|                                                                                   |
|    +--------------------+       +---------------------+       +---------------+   |
|    |   core/terrain.py  | ----> | core/trafficability | <---- |core/vehicles  |   |
|    | Multi-Octave DEM   |       | Vectorized NATO     |       |Tracked MBT /  |   |
|    | & Carved Riverbed  |       | Mobility Engine     |       |Tactical Trucks|   |
|    +--------------------+       +----------+----------+       +---------------+   |
|                                            |                                      |
|                                            v                                      |
|                                 +---------------------+                           |
|                                 | core/pathfinding.py |                           |
|                                 | Custom A* (heapq)   |                           |
|                                 | Octile Heuristic    |                           |
|                                 +----------+----------+                           |
+--------------------------------------------+--------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------+
|                           TACTICAL OPERATIONAL UI LAYER                           |
|                                                                                   |
|   +---------------------------------------------------------------------------+   |
|   |                                  app.py                                   |   |
|   |  - Sidebar: Vehicle presets, custom sliders, monsoon & current modifiers  |   |
|   |  - KPI Cards: Tactical Distance, Transit ETA, % Go / Slow-Go, Crossing    |   |
|   |  - Folium Map: Raster ImageOverlay, OSM/Esri Satellite, Route PolyLine    |   |
|   |  - Tabs: Elevation/Riverbed Cross-Section, Mobility Breakdown, SITREP TXT |   |
|   +---------------------------------------------------------------------------+   |
+-----------------------------------------------------------------------------------+
```

---

## 3. Directory Layout

```text
trafficability/
├── app.py                  # UI only, zero business logic
├── requirements.txt        # Pinned pure-pip dependencies
├── runtime.txt             # python-3.11
├── Dockerfile              # python:3.11-slim, port 8501
├── README.md               # Architecture, setup, and deployment manual
├── DECISIONS.md            # Technical decisions and rationale
├── .gitignore              # Git ignore rules
├── .streamlit/
│   └── config.toml         # Dark tactical theme & server parameters
├── core/
│   ├── __init__.py
│   ├── config.py           # Constants, enums, colors, coordinate transforms
│   ├── terrain.py          # Synthetic DEM, slope, soil, hydrography synthesis
│   ├── vehicles.py         # Vehicle profiles & NATO mobility presets
│   ├── trafficability.py   # Vectorized terramechanics & NATO classification
│   ├── pathfinding.py      # Custom A* with admissible octile heuristic
│   ├── analytics.py        # Tactical metrics, crossing risk & profile builder
│   ├── io_adapters.py      # CSV & KD-Tree AUV survey rasterizer
│   └── viz.py              # Folium Leaflet builder & Plotly analytics
├── sample_data/
│   └── auv_sample.csv      # Realistically distributed AUV survey dataset
└── tests/
    ├── test_terrain.py         # Gate 1 verification suite
    ├── test_trafficability.py  # Gate 2 verification suite
    ├── test_pathfinding.py     # Gate 3 pathfinding suite
    ├── test_analytics.py       # Gate 3 analytics & IO adapter suite
    └── test_ui_app.py          # Gate 4 headless AppTest integration suite
```

---

## 4. Local Setup & Quickstart

### Prerequisites
- Python 3.11 installed
- Git CLI

### Local Execution Steps
```bash
# 1. Clone repository
git clone https://github.com/<your-username>/trafficability.git
cd trafficability

# 2. Create isolated virtual environment
python -m venv venv

# On Linux/macOS:
source venv/bin/activate
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1

# 3. Install verified pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run test suite
pytest tests/ -v

# 5. Launch tactical Streamlit application
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 5. Docker Deployment

### Build and Run with Docker
```bash
# Build Docker image
docker build -t trafficability-system:v1.0 .

# Run container on port 8501
docker run -d --name trafficability_app -p 8501:8501 trafficability-system:v1.0

# Verify running container
docker ps
```
Access the application at `http://localhost:8501`.

---

## 6. Streamlit Community Cloud Deployment Guide

Follow these exact steps to launch the system live on Streamlit Community Cloud:

1. **Initialize Git Repository**:
   ```bash
   git init
   git add .
   git commit -m "feat: trafficability mapping system v1.0"
   ```
2. **Push to GitHub**:
   - Create a new public repository on GitHub named `trafficability-mapping-system`.
   - Push your local commits:
     ```bash
     git remote add origin https://github.com/<your-username>/trafficability-mapping-system.git
     git branch -M main
     git push -u origin main
     ```
3. **Deploy on Streamlit Community Cloud**:
   - Navigate to [share.streamlit.io](https://share.streamlit.io/).
   - Click **"New app"**.
   - Select your GitHub repository: `<your-username>/trafficability-mapping-system`.
   - Set **Branch** to `main`.
   - Set **Main file path** to `app.py`.
   - Click **"Deploy"**.
   - The app will automatically build using `runtime.txt` (Python 3.11) and `requirements.txt` with zero build errors.

---

## 7. Verification Test Suite Commands

Run individual or full test suites with verbose execution reporting:

```bash
# Run complete test suite (24 unit tests + 7 headless AppTest UI tests)
pytest tests/ -v

# Run Gate 1 (Terrain & DEM georeferencing)
pytest tests/test_terrain.py -v

# Run Gate 2 (Vehicles & NATO trafficability classification)
pytest tests/test_trafficability.py -v

# Run Gate 3 (A* Pathfinding & Mission Analytics)
pytest tests/test_pathfinding.py tests/test_analytics.py -v

# Run Gate 4 (Streamlit AppTest headless integration tests)
pytest tests/test_ui_app.py -v
```

---

## 8. Terramechanics & Defense-AI Methodology

1. **Vectorized Terramechanics Classification**:
   - **Slope Envelope**: Terrain gradients are derived via central finite differences: $\theta = \arctan(\sqrt{(\partial z / \partial x)^2 + (\partial z / \partial y)^2})$. Gradients exceeding platform design thresholds ($20^\circ$ for 4x4 Jeep, $25^\circ$ for 6x6 Truck, $30^\circ$ for Heavy MBT) cause track slip, loss of traction, or rollover risk, classifying the cell as **No-Go (Red)**.
   - **Fording & Current Drag Penalty**: River currents reduce allowable fording depth due to hydrodynamic drag forces:
     $$d_{\text{effective}} = d_{\text{max}} \times (1.0 - 0.40 \times C_f)$$
     Water depths exceeding $d_{\text{effective}}$ cause cylinder hydrolock and electrical flooding.
   - **Substrate Bearing Capacity**: Heavy armor platforms (e.g. 60-ton tanks) impose high ground contact pressures ($>100\,\text{kPa}$). Saturated wetlands (Marshes) exhibit shear failure under heavy armor, classifying them as **No-Go (Red)**.
   - **Monotonic Rainfall Degradation**: Rainfall scaling swells river depth by $15\%$ and increases substrate pore-water pressure by $30\%$ per unit, guaranteeing mathematically that higher precipitation never makes any cell safer.
2. **Optimal Tactical Pathfinding**:
   - Custom priority queue A\* with an admissible octile distance heuristic:
     $$h(n) = ((\sqrt{2} - 1) \cdot \min(\Delta x, \Delta y) + \max(\Delta x, \Delta y)) \times \text{cell\_size}$$
   - Penalizes Slow-Go terrain with a $5\times$ traversal cost multiplier while strictly barring No-Go cells and prohibiting diagonal corner-cutting through obstacle pinch points.

---

## 9. Limitations & Tactical Constraints

1. **Synthetic DEM Baseline**: The default digital elevation model is synthesized using multi-octave Gaussian filtering and Rajasthan foothill topography rather than a live satellite API feed. Real SRTM or Sentinel rasters must be ingested through the provided `TerrainGrid` and `core/io_adapters.py` pipeline.
2. **Static River Current Assumption**: The river current drag factor is currently modeled as a uniform scalar across the river reach rather than a 2D hydrodynamic Navier-Stokes hydraulic simulation with eddy circulations.
3. **Single-Vehicle Planning**: Routing computes the least-risk path for a single platform at a time rather than resolving convoy dispersion, spacing constraints, and turn-radius envelopes for tactical columns.

---

## 10. Screenshots & Interface Walkthrough

```text
+-----------------------------------------------------------------------------------------------+
| [🛡️ Automated Vehicle Trafficability Prediction & Mapping System]                              |
| Joint Defense-AI & Geospatial Intelligence • Sapta Shakti Command, Jaipur                     |
|                                                                                               |
| [Distance: 10.84 km] [ETA: 25.0 min] [GO: 82.3%] [SLOW-GO: 17.7%] [Safety: CAUTION]          |
|                                                                                               |
| +---------------------------------------------------------+ +-------------------------------+ |
| |                    FOLIUM MAP                           | |      TACTICAL CONTROLS        | |
| | [OpenStreetMap / Esri Satellite Imagery]                | | - Platform: Light Recon Jeep  | |
| |                                                         | | - Rainfall: 1.00x             | |
| |  (Start SW) [o]                                         | | - Current: 0.20               | |
| |         \                                               | | - Origin X,Y: [15, 85]        | |
| |          \== (Tactical Route Corridor) ==\              | | - Target X,Y: [85, 15]        | |
| |                                           \             | |                               | |
| |                                         [x] (Target NE) | | [✓] Snap to Nearest Passable  | |
| |                                                         | | [✓] River Overlay Toggle      | |
| +---------------------------------------------------------+ +-------------------------------+ |
|                                                                                               |
| [Tabs: Longitudinal Profile | Mobility Breakdown | Raw Rasters | Route CSV | Mission SITREP]  |
+-----------------------------------------------------------------------------------------------+
```
