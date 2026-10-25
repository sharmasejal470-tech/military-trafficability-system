"""Automated Vehicle Trafficability Prediction and Mapping System.

Streamlit Geospatial Intelligence Dashboard for multi-spectral terrain and AUV sonar
bathymetry fusion, NATO terramechanics classification, and tactical A* pathfinding.
"""

from __future__ import annotations
import io
import json
import logging
import traceback
from typing import Optional, Tuple

import streamlit as st
import numpy as np
import pandas as pd
from streamlit_folium import st_folium

from core.config import (
    DEFAULT_GRID_ROWS,
    DEFAULT_GRID_COLS,
    DEFAULT_CELL_SIZE_M,
    DEFAULT_BASE_LAT,
    DEFAULT_BASE_LON,
    DEFAULT_SEED,
    TrafficabilityClass,
    latlon_to_cell,
    cell_to_latlon,
)
from core.terrain import TerrainGrid, generate_terrain
from core.vehicles import VEHICLE_PRESETS, VehicleProfile, DEFAULT_VEHICLE_NAME
from core.trafficability import TrafficabilityResult, compute_trafficability
from core.pathfinding import (
    PathResult,
    PathStatus,
    find_path,
    nearest_valid_cell,
)
from core.analytics import (
    RouteMetrics,
    ProfileData,
    CrossingSafetyStatus,
    compute_route_analytics,
)
from core.io_adapters import load_auv_csv
from core.viz import (
    build_trafficability_folium_map,
    build_profile_chart,
    build_breakdown_chart,
    build_raw_sensor_heatmaps,
    export_route_geojson,
    generate_mission_summary_text,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trafficability_app")


# --- Caching Adapters ---

@st.cache_data
def get_cached_terrain(
    seed: int,
    base_lat: float,
    base_lon: float,
    file_bytes: Optional[bytes] = None,
) -> TerrainGrid:
    """Generates or loads cached TerrainGrid based on parameters and survey input."""
    terrain = generate_terrain(
        rows=DEFAULT_GRID_ROWS,
        cols=DEFAULT_GRID_COLS,
        cell_size_m=DEFAULT_CELL_SIZE_M,
        base_lat=base_lat,
        base_lon=base_lon,
        seed=seed,
    )
    if file_bytes is not None and len(file_bytes) > 0:
        success, msg, updated = load_auv_csv(io.BytesIO(file_bytes), terrain)
        if success and updated is not None:
            return updated
    return terrain


@st.cache_data
def get_cached_trafficability(
    _terrain: TerrainGrid,
    vehicle_name: str,
    max_slope_deg: float,
    max_fording_m: float,
    min_firmness: float,
    speed_go: float,
    speed_slowgo: float,
    weight_class: str,
    rainfall_multiplier: float,
    current_factor: float,
) -> TrafficabilityResult:
    """Evaluates and caches per-cell trafficability classifications."""
    vehicle = VehicleProfile(
        name=vehicle_name,
        weight_class=weight_class,
        max_slope_deg=max_slope_deg,
        max_fording_m=max_fording_m,
        min_firmness=min_firmness,
        speed_kmph_go=speed_go,
        speed_kmph_slowgo=speed_slowgo,
    )
    return compute_trafficability(
        _terrain,
        vehicle,
        rainfall_multiplier=rainfall_multiplier,
        current_factor=current_factor,
    )


# --- UI Main Entry Point ---

def main() -> None:
    """Main application lifecycle with global error safety wrap."""
    st.set_page_config(
        page_title="Automated Vehicle Trafficability Prediction System",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    try:
        run_app()
    except Exception as exc:
        logger.error(f"Application Runtime Exception: {traceback.format_exc()}")
        st.error(
            "An unexpected operational error occurred while evaluating terrain mobility. "
            "The mission parameters have been logged."
        )
        st.info(f"Diagnostic detail: {str(exc)}")


def run_app() -> None:
    """Renders the defense geospatial intelligence UI."""
    # Initialize session state coordinates
    if "start_x" not in st.session_state:
        st.session_state["start_x"] = 15
    if "start_y" not in st.session_state:
        st.session_state["start_y"] = 85
    if "goal_x" not in st.session_state:
        st.session_state["goal_x"] = 85
    if "goal_y" not in st.session_state:
        st.session_state["goal_y"] = 15
    if "last_processed_click" not in st.session_state:
        st.session_state["last_processed_click"] = None

    # Title header
    st.title("🛡️ Automated Vehicle Trafficability Prediction & Mapping System")
    st.caption(
        "Joint Defense-AI & Geospatial Intelligence • Satellite DEM, Soil & AUV Hydrographic Sonar Fusion • Sapta Shakti Command"
    )

    # --- Sidebar Controls ---
    st.sidebar.header("🎯 Mission Parameters")

    vehicle_preset_name = st.sidebar.selectbox(
        "Select Vehicle Platform",
        options=list(VEHICLE_PRESETS.keys()),
        index=list(VEHICLE_PRESETS.keys()).index(DEFAULT_VEHICLE_NAME),
        help="Selects NATO standardized military vehicle mobility and fording envelope.",
    )
    base_vehicle = VEHICLE_PRESETS[vehicle_preset_name]

    with st.sidebar.expander("⚙️ Vehicle Specification Overrides", expanded=False):
        custom_slope = st.slider(
            "Max Slope Limit (deg)",
            10.0, 45.0, float(base_vehicle.max_slope_deg), 1.0,
            help="Maximum traversable terrain gradient before track/wheel slippage or rollover."
        )
        custom_fording = st.slider(
            "Max Fording Depth (m)",
            0.2, 2.5, float(base_vehicle.max_fording_m), 0.05,
            help="Static water wading depth limit before air intake hydrolock."
        )
        custom_firmness = st.slider(
            "Min Substrate Firmness",
            0.1, 0.95, float(base_vehicle.min_firmness), 0.05,
            help="Minimum substrate bearing strength required to prevent sinkage and belly hang-up."
        )
        custom_speed_go = st.slider(
            "Go Dash Speed (km/h)",
            10.0, 100.0, float(base_vehicle.speed_kmph_go), 5.0
        )
        custom_speed_slowgo = st.slider(
            "Slow-Go Tactical Speed (km/h)",
            5.0, 50.0, float(base_vehicle.speed_kmph_slowgo), 1.0
        )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🌧️ Environmental & Hydrographic Conditions")

    rainfall_mult = st.sidebar.slider(
        "Seasonal Rainfall Multiplier",
        0.5, 2.0, 1.0, 0.05,
        help="Simulates monsoon saturation: degrades substrate bearing strength and swells water depth."
    )
    current_factor = st.sidebar.slider(
        "River Current Drag Intensity",
        0.0, 1.0, 0.20, 0.05,
        help="Hydrodynamic river current drag factor reduces allowable safe fording depth."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🌐 Geospatial Grid Reference")
    seed_input = st.sidebar.number_input("Deterministic Simulation Seed", value=DEFAULT_SEED, step=1)
    base_lat = st.sidebar.number_input("Base Latitude (WGS84)", value=DEFAULT_BASE_LAT, format="%.4f")
    base_lon = st.sidebar.number_input("Base Longitude (WGS84)", value=DEFAULT_BASE_LON, format="%.4f")

    st.sidebar.markdown("---")
    st.sidebar.subheader("📡 AUV Sonar Survey Ingestion")
    auv_file = st.sidebar.file_uploader(
        "Upload AUV Bathymetry CSV (Optional)",
        type=["csv"],
        help="Upload field AUV sonar data with lat, lon, water_depth_m, firmness columns.",
    )
    file_bytes = auv_file.getvalue() if auv_file is not None else None

    # Load terrain and trafficability
    terrain = get_cached_terrain(
        seed=int(seed_input),
        base_lat=float(base_lat),
        base_lon=float(base_lon),
        file_bytes=file_bytes,
    )

    trafficability = get_cached_trafficability(
        _terrain=terrain,
        vehicle_name=vehicle_preset_name,
        max_slope_deg=custom_slope,
        max_fording_m=custom_fording,
        min_firmness=custom_firmness,
        speed_go=custom_speed_go,
        speed_slowgo=custom_speed_slowgo,
        weight_class=base_vehicle.weight_class,
        rainfall_multiplier=rainfall_mult,
        current_factor=current_factor,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📍 Mission Routing Coordinates")
    st.sidebar.caption("Grid domain: [0, 99] for X (Col / West->East) and Y (Row / North->South)")

    c_col1, c_col2 = st.sidebar.columns(2)
    with c_col1:
        st.session_state["start_x"] = st.number_input(
            "Start X", 0, DEFAULT_GRID_COLS - 1, int(st.session_state["start_x"]), key="input_start_x"
        )
        st.session_state["start_y"] = st.number_input(
            "Start Y", 0, DEFAULT_GRID_ROWS - 1, int(st.session_state["start_y"]), key="input_start_y"
        )
    with c_col2:
        st.session_state["goal_x"] = st.number_input(
            "Goal X", 0, DEFAULT_GRID_COLS - 1, int(st.session_state["goal_x"]), key="input_goal_x"
        )
        st.session_state["goal_y"] = st.number_input(
            "Goal Y", 0, DEFAULT_GRID_ROWS - 1, int(st.session_state["goal_y"]), key="input_goal_y"
        )

    click_mode = st.sidebar.radio(
        "Map Interactive Click Action",
        options=["Inspect / Inactive", "Set Start (SW)", "Set Destination (NE)"],
        index=0,
        help="Click anywhere on the satellite Folium map to reposition Start or Destination waypoints.",
    )

    # Convert coordinates: UI uses (x=col, y=row) -> grid uses [row, col]
    start_row = int(st.session_state["start_y"])
    start_col = int(st.session_state["start_x"])
    goal_row = int(st.session_state["goal_y"])
    goal_col = int(st.session_state["goal_x"])

    # --- Compute Tactical Path & Analytics ---
    path_res = find_path(
        trafficability.grid,
        start=(start_row, start_col),
        goal=(goal_row, goal_col),
        is_xy=False,
        cell_size_m=terrain.cell_size_m,
    )

    metrics, profile = compute_route_analytics(
        terrain,
        trafficability,
        path_res,
        is_xy=False,
        cell_size_m=terrain.cell_size_m,
    )

    # --- Main Content: Top Metric Cards ---
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)

    if metrics is not None:
        m_col1.metric("Tactical Distance", f"{metrics.distance_km:.2f} km")
        m_col2.metric("Transit ETA", f"{metrics.eta_minutes:.1f} min", f"{metrics.eta_hours:.2f} hrs")
        m_col3.metric("GO Traversal", f"{metrics.pct_go:.1f}%")
        m_col4.metric("SLOW-GO Traversal", f"{metrics.pct_slowgo:.1f}%")

        # Color-coded crossing status
        cross_val = metrics.crossing_status.value
        m_col5.metric("Crossing Safety", cross_val, delta=metrics.crossing_reason[:24] + "...")
    else:
        m_col1.metric("Tactical Distance", "--")
        m_col2.metric("Transit ETA", "--")
        m_col3.metric("GO Traversal", "--")
        m_col4.metric("SLOW-GO Traversal", "--")
        m_col5.metric("Crossing Safety", "BLOCKED / NO PATH")

    # Endpoint Validation Alerts
    if path_res.status == PathStatus.START_BLOCKED:
        st.warning("⚠️ Tactical Origin (Start) is located on impassable No-Go terrain.")
        if st.button("📍 Snap Start to Nearest Passable Cell"):
            valid = nearest_valid_cell(trafficability.grid, (start_row, start_col), is_xy=False)
            if valid is not None:
                st.session_state["start_y"], st.session_state["start_x"] = valid[0], valid[1]
                st.rerun()

    elif path_res.status == PathStatus.GOAL_BLOCKED:
        st.warning("⚠️ Mission Objective (Destination) is located on impassable No-Go terrain.")
        if st.button("📍 Snap Destination to Nearest Passable Cell"):
            valid = nearest_valid_cell(trafficability.grid, (goal_row, goal_col), is_xy=False)
            if valid is not None:
                st.session_state["goal_y"], st.session_state["goal_x"] = valid[0], valid[1]
                st.rerun()

    elif path_res.status == PathStatus.NO_PATH:
        st.error(
            f"🚫 No viable corridor exists between Origin [{start_col},{start_row}] and Target [{goal_col},{goal_row}]. "
            f"The hydrographic river channel or surrounding wetlands exceed the mobility envelope of {vehicle_preset_name} "
            f"under rainfall {rainfall_mult:.2f}x and current factor {current_factor:.2f}. "
            "Actionable remedy: Select Heavy Tracked Tank, reduce seasonal rainfall multiplier, or relocate route endpoints."
        )

    # --- Interactive Folium Map ---
    st.subheader("🗺️ Tactical Mobility Overlay & Vector Routing")

    folium_map = build_trafficability_folium_map(
        terrain=terrain,
        trafficability=trafficability,
        path_result=path_res,
        start_cell=(start_row, start_col),
        goal_cell=(goal_row, goal_col),
        show_river_overlay=True,
    )

    map_state = st_folium(
        folium_map,
        width="100%",
        height=540,
        returned_objects=["last_clicked"],
        key="trafficability_folium_dashboard",
    )

    # Process interactive map click
    if map_state and "last_clicked" in map_state and map_state["last_clicked"] is not None:
        click_lat = map_state["last_clicked"]["lat"]
        click_lon = map_state["last_clicked"]["lng"]
        click_tuple = (round(click_lat, 6), round(click_lon, 6))

        if click_tuple != st.session_state["last_processed_click"]:
            st.session_state["last_processed_click"] = click_tuple
            cell = latlon_to_cell(
                click_lat, click_lon,
                base_lat=terrain.base_lat,
                base_lon=terrain.base_lon,
                rows=terrain.rows,
                cols=terrain.cols,
                cell_size_m=terrain.cell_size_m,
            )
            if cell is not None:
                cr, cc = cell
                if click_mode == "Set Start (SW)":
                    st.session_state["start_x"] = cc
                    st.session_state["start_y"] = cr
                    st.rerun()
                elif click_mode == "Set Destination (NE)":
                    st.session_state["goal_x"] = cc
                    st.session_state["goal_y"] = cr
                    st.rerun()

    # --- Analysis Tabs ---
    tab_profile, tab_breakdown, tab_rasters, tab_table, tab_export = st.tabs([
        "📈 Longitudinal Profile",
        "📊 Mobility Breakdown",
        "🛰️ Raw Sensor Layers",
        "📋 Route Table",
        "📦 Mission Export",
    ])

    with tab_profile:
        if profile is not None and metrics is not None:
            fig_prof = build_profile_chart(profile, metrics, vehicle_preset_name)
            st.plotly_chart(fig_prof, use_container_width=True)
            st.info(f"Hydrographic Clearance Assessment: {metrics.crossing_reason}")
        else:
            st.warning("No active route profile available to visualize.")

    with tab_breakdown:
        fig_bar = build_breakdown_chart(trafficability, metrics)
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab_rasters:
        fig_heat = build_raw_sensor_heatmaps(terrain)
        st.plotly_chart(fig_heat, use_container_width=True)

    with tab_table:
        if path_res.status == PathStatus.OK and profile is not None:
            df_route = profile.to_dataframe()
            st.dataframe(df_route, use_container_width=True, height=280)
            csv_route = df_route.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Route Waypoints CSV",
                data=csv_route,
                file_name=f"tactical_route_{vehicle_preset_name.replace(' ', '_').lower()}.csv",
                mime="text/csv",
            )
        else:
            st.info("No active path data to display.")

    with tab_export:
        st.subheader("Tactical Intelligence Mission Deliverables")
        if metrics is not None and path_res.status == PathStatus.OK:
            geojson_str = export_route_geojson(
                path_result=path_res,
                terrain=terrain,
                metrics=metrics,
                vehicle_name=vehicle_preset_name,
            )
            sitrep_str = generate_mission_summary_text(
                vehicle_name=vehicle_preset_name,
                metrics=metrics,
                start_xy=(start_col, start_row),
                goal_xy=(goal_col, goal_row),
                rainfall_mult=rainfall_mult,
                current_factor=current_factor,
            )

            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                st.download_button(
                    "📥 Export Route as RFC 7946 GeoJSON",
                    data=geojson_str,
                    file_name="tactical_route.geojson",
                    mime="application/geo+json",
                    use_container_width=True,
                )
            with exp_col2:
                st.download_button(
                    "📥 Export 1-Page SITREP Briefing (TXT)",
                    data=sitrep_str,
                    file_name="mission_sitrep.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            st.text_area("SITREP Preview", value=sitrep_str, height=260)
        else:
            st.info("Compute a viable tactical path to enable GeoJSON and SITREP exports.")

    # --- Methodology & Doctrine Expander ---
    with st.expander("📖 Tactical Doctrine, Terramechanics & Sensor Integration Methodology"):
        st.markdown(r"""
        ### Military Terrain Analysis & Terramechanics Methodology
        1. **Digital Elevation Model (DEM) & Slope Dynamics**:
           - Gradients exceeding vehicle critical pitch/roll thresholds ($20^\circ$ for 4x4, $25^\circ$ for 6x6, $30^\circ$ for MBT) cause ground track detachment, lateral rollover, or transmission stall.
           - Slopes are computed using central finite difference gradients: $\\theta = \\arctan(\\sqrt{(\\partial z / \\partial x)^2 + (\\partial z / \\partial y)^2})$.
        2. **Hydrographic Bathymetry & Hydrodynamic Drag**:
           - Static water depth exceeding air intake / snorkel levels causes immediate cylinder hydrolock.
           - River current exerts hydrodynamic stagnation pressure ($F_d = \\frac{1}{2} C_d \\rho v^2 A$), reducing effective safe wading depth via the penalty function:
             $$d_{\\text{effective}} = d_{\\text{max}} \\times (1 - k \\cdot C_f)$$ with $k = 0.40$.
        3. **Substrate Bearing Strength & Soil Saturation**:
           - Heavy armored assets require firm riverbed substrates ($>0.75$) to prevent belly grounding and track sinkage.
           - Saturated alluvial clays and wetlands (Marshes) lack internal friction angles ($\phi \approx 0$), inducing catastrophic bearing shear failure under heavy armor.
        4. **Real-world Satellite & AUV Data Ingestion**:
           - **Satellite DEMs**: Real SRTM (30m) or Copernicus DEMs rasterize directly into `TerrainGrid.elevation`.
           - **AUV Survey**: Sub-aquatic sonars deliver point bathymetry and acoustic backscatter firmness via `core.io_adapters.load_auv_csv`, seamlessly overriding synthetic defaults via KD-Tree spatial interpolation.
        """)


if __name__ == "__main__":
    main()
