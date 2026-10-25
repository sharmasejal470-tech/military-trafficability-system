"""Geospatial visualization module.

Builds interactive Folium Leaflet maps with raster ImageOverlays, Plotly longitudinal
cross-sections, tactical mobility breakdown analytics, and sensor heatmap dashboards.
"""

from __future__ import annotations
import base64
import io
import json
from typing import List, Tuple, Dict, Any, Optional

import folium
from folium import raster_layers
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

from core.config import (
    TrafficabilityClass,
    COLOR_GO,
    COLOR_SLOW_GO,
    COLOR_NO_GO,
    RGBA_GO,
    RGBA_SLOW_GO,
    RGBA_NO_GO,
    RGBA_TRANSPARENT,
    RGBA_RIVER,
    SOIL_LABELS,
    SoilType,
    get_grid_bounds,
    cell_to_latlon,
)
from core.terrain import TerrainGrid
from core.trafficability import TrafficabilityResult
from core.pathfinding import PathResult
from core.analytics import RouteMetrics, ProfileData, CrossingSafetyStatus


def create_trafficability_rgba_array(grid: np.ndarray) -> np.ndarray:
    """
    Constructs an RGBA uint8 array from a 2D trafficability grid.
    Indexing: row 0 is North edge, row (H-1) is South edge.
    """
    h, w = grid.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    go_mask = (grid == TrafficabilityClass.GO)
    slowgo_mask = (grid == TrafficabilityClass.SLOW_GO)
    nogo_mask = (grid == TrafficabilityClass.NO_GO)

    rgba[go_mask] = RGBA_GO
    rgba[slowgo_mask] = RGBA_SLOW_GO
    rgba[nogo_mask] = RGBA_NO_GO
    return rgba


def create_river_rgba_array(river_mask: np.ndarray) -> np.ndarray:
    """Constructs an RGBA uint8 array highlighting the hydrographic river channel."""
    h, w = river_mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[river_mask] = RGBA_RIVER
    rgba[~river_mask] = RGBA_TRANSPARENT
    return rgba


def rgba_to_png_data_url(rgba: np.ndarray) -> str:
    """Encodes an RGBA image array into an in-memory base64 PNG data URL."""
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def build_trafficability_folium_map(
    terrain: TerrainGrid,
    trafficability: TrafficabilityResult,
    path_result: Optional[PathResult] = None,
    start_cell: Optional[Tuple[int, int]] = None,
    goal_cell: Optional[Tuple[int, int]] = None,
    show_river_overlay: bool = True,
    zoom_start: int = 12,
) -> folium.Map:
    """
    Constructs a high-performance Leaflet Folium map with raster ImageOverlays,
    tactical waypoints, and interactive layer controls.
    """
    south, west, north, east = get_grid_bounds(
        base_lat=terrain.base_lat,
        base_lon=terrain.base_lon,
        rows=terrain.rows,
        cols=terrain.cols,
        cell_size_m=terrain.cell_size_m,
    )
    center_lat = (north + south) / 2.0
    center_lon = (east + west) / 2.0
    bounds = [[south, west], [north, east]]

    # 1. Initialize Folium Map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles=None,
        control_scale=True,
    )

    # Base layers
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap (Standard)",
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite Imagery (Esri)",
        control=True,
    ).add_to(m)

    # 2. Raster Trafficability ImageOverlay
    rgba_trafficability = create_trafficability_rgba_array(trafficability.grid)
    data_url_traff = rgba_to_png_data_url(rgba_trafficability)

    raster_layers.ImageOverlay(
        image=data_url_traff,
        bounds=bounds,
        opacity=0.65,
        name="Trafficability Surface (Go/Slow-Go/No-Go)",
        interactive=False,
    ).add_to(m)

    # 3. Optional River Channel Overlay
    if show_river_overlay:
        rgba_river = create_river_rgba_array(terrain.river_mask)
        data_url_river = rgba_to_png_data_url(rgba_river)
        raster_layers.ImageOverlay(
            image=data_url_river,
            bounds=bounds,
            opacity=0.55,
            name="Hydrographic River Channel",
            show=False,
            interactive=False,
        ).add_to(m)

    # 4. Route Polyline
    if path_result is not None and path_result.path_cells:
        latlon_points = []
        for r, c in path_result.path_cells:
            coords = cell_to_latlon(
                int(r), int(c),
                terrain.base_lat, terrain.base_lon,
                terrain.rows, terrain.cols, terrain.cell_size_m
            )
            if coords is not None:
                latlon_points.append(coords)

        if len(latlon_points) >= 2:
            folium.PolyLine(
                locations=latlon_points,
                color="#00d2ff",
                weight=5,
                opacity=0.95,
                tooltip="Tactical A* Mission Route",
            ).add_to(m)

    # 5. Start & Goal Markers
    if start_cell is not None:
        sr, sc = start_cell
        s_coords = cell_to_latlon(
            int(sr), int(sc),
            terrain.base_lat, terrain.base_lon,
            terrain.rows, terrain.cols, terrain.cell_size_m
        )
        if s_coords is not None:
            folium.Marker(
                location=s_coords,
                tooltip=f"Start Position: [{sr}, {sc}]",
                icon=folium.Icon(color="green", icon="play", prefix="fa"),
            ).add_to(m)

    if goal_cell is not None:
        gr, gc = goal_cell
        g_coords = cell_to_latlon(
            int(gr), int(gc),
            terrain.base_lat, terrain.base_lon,
            terrain.rows, terrain.cols, terrain.cell_size_m
        )
        if g_coords is not None:
            folium.Marker(
                location=g_coords,
                tooltip=f"Destination Objective: [{gr}, {gc}]",
                icon=folium.Icon(color="red", icon="flag", prefix="fa"),
            ).add_to(m)

    # Fit bounds
    m.fit_bounds(bounds)

    # Layer Control
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    # 6. HTML Floating Legend
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 25px;
        right: 25px;
        width: 195px;
        background-color: rgba(18, 24, 38, 0.88);
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 10px 14px;
        color: #f3f4f6;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 12px;
        z-index: 9999;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    ">
        <div style="font-weight: 700; margin-bottom: 8px; border-bottom: 1px solid #4b5563; padding-bottom: 4px; letter-spacing: 0.5px;">
            MOBILITY LEGEND
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <span style="background-color: {COLOR_GO}; width: 14px; height: 14px; display: inline-block; border-radius: 3px; margin-right: 8px;"></span>
            <span>Go (Unrestricted)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <span style="background-color: {COLOR_SLOW_GO}; width: 14px; height: 14px; display: inline-block; border-radius: 3px; margin-right: 8px;"></span>
            <span>Slow-Go (Degraded)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <span style="background-color: {COLOR_NO_GO}; width: 14px; height: 14px; display: inline-block; border-radius: 3px; margin-right: 8px;"></span>
            <span>No-Go (Impassable)</span>
        </div>
        <div style="display: flex; align-items: center; margin-top: 6px; border-top: 1px solid #374151; padding-top: 5px;">
            <span style="background-color: #00d2ff; width: 18px; height: 4px; display: inline-block; border-radius: 2px; margin-right: 6px;"></span>
            <span style="font-weight: 600;">Tactical Route</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def build_profile_chart(profile: ProfileData, metrics: RouteMetrics, vehicle_name: str) -> go.Figure:
    """Generates an interactive Plotly longitudinal profile of elevation, water level, and riverbed."""
    fig = go.Figure()

    # Ground surface elevation
    fig.add_trace(go.Scatter(
        x=profile.cumulative_distance_km,
        y=profile.elevation_m,
        mode="lines",
        name="Ground Surface (m)",
        line=dict(color="#27ae60", width=2.5),
        hovertemplate="Distance: %{x:.2f} km<br>Elevation: %{y:.1f} m<extra></extra>",
    ))

    # Riverbed elevation
    fig.add_trace(go.Scatter(
        x=profile.cumulative_distance_km,
        y=profile.riverbed_level_m,
        mode="lines",
        name="Riverbed Thalweg (m)",
        line=dict(color="#8d6e63", width=2, dash="dash"),
        hovertemplate="Distance: %{x:.2f} km<br>Riverbed: %{y:.1f} m<extra></extra>",
    ))

    # Water surface elevation where river is present
    if np.any(profile.is_river):
        fig.add_trace(go.Scatter(
            x=profile.cumulative_distance_km,
            y=profile.water_surface_m,
            mode="lines",
            name="Water Surface (m)",
            line=dict(color="#2980b9", width=2),
            connectgaps=False,
            hovertemplate="Water Level: %{y:.1f} m<extra></extra>",
        ))

    # Layout styling
    fig.update_layout(
        title=f"Tactical Longitudinal Terrain & Riverbed Profile ({vehicle_name})",
        xaxis_title="Cumulative Route Distance (km)",
        yaxis_title="Elevation Above Sea Level (m)",
        template="plotly_dark",
        height=420,
        margin=dict(l=50, r=40, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_breakdown_chart(
    trafficability: TrafficabilityResult,
    metrics: Optional[RouteMetrics] = None,
) -> go.Figure:
    """Generates a comparative bar chart comparing whole area mobility vs route mobility."""
    total_cells = trafficability.grid.size
    area_go_pct = float((trafficability.grid == TrafficabilityClass.GO).sum() / total_cells * 100.0)
    area_slowgo_pct = float((trafficability.grid == TrafficabilityClass.SLOW_GO).sum() / total_cells * 100.0)
    area_nogo_pct = float((trafficability.grid == TrafficabilityClass.NO_GO).sum() / total_cells * 100.0)

    categories = ["Go (Unrestricted)", "Slow-Go (Degraded)", "No-Go (Impassable)"]
    colors = [COLOR_GO, COLOR_SLOW_GO, COLOR_NO_GO]

    area_values = [area_go_pct, area_slowgo_pct, area_nogo_pct]
    route_values = [
        metrics.pct_go if metrics else 0.0,
        metrics.pct_slowgo if metrics else 0.0,
        0.0,  # A* strictly navigates 0% No-Go
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Entire Theater Area (%)",
        x=categories,
        y=area_values,
        marker_color=["#2ecc71", "#f1c40f", "#e74c3c"],
        text=[f"{v:.1f}%" for v in area_values],
        textposition="auto",
    ))

    if metrics is not None:
        fig.add_trace(go.Bar(
            name="Planned Mission Route (%)",
            x=categories,
            y=route_values,
            marker_color=["#27ae60", "#f39c12", "#c0392b"],
            text=[f"{v:.1f}%" for v in route_values],
            textposition="auto",
        ))

    fig.update_layout(
        barmode="group",
        title="Mobility Classification Breakdown (Theater vs Route)",
        yaxis_title="Percentage (%)",
        template="plotly_dark",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_raw_sensor_heatmaps(terrain: TerrainGrid) -> go.Figure:
    """Builds a 5-panel subplot showing raw DEM, slope, soil, moisture, and water depth."""
    fig = make_subplots(
        rows=1, cols=5,
        subplot_titles=[
            "Elevation (m)",
            "Slope (deg)",
            "Soil Category",
            "Moisture [0-1]",
            "Water Depth (m)",
        ],
    )

    # 1. Elevation
    fig.add_trace(
        go.Heatmap(z=terrain.elevation, colorscale="Viridis", showscale=False),
        row=1, col=1
    )
    # 2. Slope
    fig.add_trace(
        go.Heatmap(z=terrain.slope, colorscale="Hot", showscale=False),
        row=1, col=2
    )
    # 3. Soil
    fig.add_trace(
        go.Heatmap(z=terrain.soil_type, colorscale="Earth", showscale=False),
        row=1, col=3
    )
    # 4. Moisture
    fig.add_trace(
        go.Heatmap(z=terrain.moisture, colorscale="Blues", showscale=False),
        row=1, col=4
    )
    # 5. Water Depth
    fig.add_trace(
        go.Heatmap(z=terrain.water_depth, colorscale="YlGnBu", showscale=False),
        row=1, col=5
    )

    fig.update_layout(
        title="Raw GIS & Hydrographic Sensor Rasters (100x100 Grid Cells)",
        template="plotly_dark",
        height=280,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    # Invert y axis so row 0 is top
    for i in range(1, 6):
        fig.update_yaxes(autorange="reversed", row=1, col=i)
    return fig


def export_route_geojson(
    path_result: PathResult,
    terrain: TerrainGrid,
    metrics: RouteMetrics,
    vehicle_name: str,
) -> str:
    """Exports the tactical route as standard RFC 7946 GeoJSON LineString."""
    coordinates = []
    for r, c in path_result.path_cells:
        coords = cell_to_latlon(
            int(r), int(c),
            terrain.base_lat, terrain.base_lon,
            terrain.rows, terrain.cols, terrain.cell_size_m
        )
        if coords is not None:
            # GeoJSON coordinates are [lon, lat]
            coordinates.append([coords[1], coords[0]])

    feature = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": coordinates,
        },
        "properties": {
            "vehicle": vehicle_name,
            "total_distance_km": round(metrics.distance_km, 3),
            "eta_minutes": round(metrics.eta_minutes, 1),
            "pct_go": round(metrics.pct_go, 1),
            "pct_slowgo": round(metrics.pct_slowgo, 1),
            "river_cells_crossed": metrics.river_cells_crossed,
            "crossing_status": str(metrics.crossing_status.value),
            "crossing_reason": metrics.crossing_reason,
            "max_water_depth_m": round(metrics.max_water_depth_m, 2),
            "max_slope_deg": round(metrics.max_slope_deg, 2),
        },
    }

    geojson_doc = {
        "type": "FeatureCollection",
        "features": [feature],
    }
    return json.dumps(geojson_doc, indent=2)


def generate_mission_summary_text(
    vehicle_name: str,
    metrics: RouteMetrics,
    start_xy: Tuple[int, int],
    goal_xy: Tuple[int, int],
    rainfall_mult: float,
    current_factor: float,
) -> str:
    """Generates a structured tactical mission SITREP summary."""
    return f"""========================================================================
TACTICAL MOBILITY PREDICTION & ROUTING MISSION BRIEFING (SITREP)
SAPTA SHAKTI COMMAND - GEOSPATIAL INTELLIGENCE & DEFENSE-AI DIVISION
========================================================================

1. MISSION CONTEXT & OPERATIONAL ASSET
   - Vehicle Platform       : {vehicle_name}
   - Origin Waypoint (X,Y)  : {start_xy[0]}, {start_xy[1]}
   - Objective Target (X,Y) : {goal_xy[0]}, {goal_xy[1]}
   - Weather / Precipitation: {rainfall_mult:.2f}x Seasonal Multiplier
   - River Hydrodynamics    : {current_factor:.2f} Current Intensity Factor

2. TACTICAL ROUTE METRICS
   - Total Corridor Distance: {metrics.distance_km:.2f} km
   - Estimated Transit ETA  : {metrics.eta_minutes:.1f} minutes ({metrics.eta_hours:.2f} hours)
   - Route Traversal Profile: {metrics.pct_go:.1f}% GO (Unrestricted) / {metrics.pct_slowgo:.1f}% SLOW-GO (Caution)
   - No-Go Traversals       : 0.0% (Strictly prohibited by A* optimization)
   - Maximum Gradient       : {metrics.max_slope_deg:.1f}°
   - Waypoints Traversed    : {metrics.waypoint_count} cells

3. HYDROGRAPHIC FORDING & AUV RIVERBED ASSESSMENT
   - River Cells Traversed  : {metrics.river_cells_crossed} cells ({metrics.river_cells_crossed * 100} m channel width)
   - Max Water Depth        : {metrics.max_water_depth_m:.2f} meters
   - Min Substrate Firmness : {metrics.min_riverbed_firmness if metrics.min_riverbed_firmness is not None else 0.0:.2f}
   - Crossing Feasibility   : {metrics.crossing_status.value}
   - Intelligence Summary   : {metrics.crossing_reason}

4. ROUTING DIRECTIVES & ORDERS
   - Follow designated vector corridor via tactical sandbar / ford.
   - Maintain Slow-Go speed throttle across wet clay and riparian approach ramps.
   - Combat engineers on standby if precipitation exceeds threshold.

END OF TRANSMISSION.
========================================================================
"""
