"""Automated Vehicle Trafficability Prediction and Mapping System.

Streamlit Geospatial Intelligence Dashboard for multi-spectral terrain and AUV sonar
bathymetry fusion, NATO terramechanics classification, and tactical A* pathfinding.
"""

from __future__ import annotations
import os
import sys

# Ensure application root directory is always on Python import path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

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
    ConvoyPlan,
    plan_tactical_convoy,
    find_path,
    nearest_valid_cell,
)
from core.analytics import (
    RouteMetrics,
    ProfileData,
    CrossingSafetyStatus,
    compute_route_analytics,
)
from core.io_adapters import load_auv_csv, load_dem_raster
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
    dem_file_bytes: Optional[bytes] = None,
    dem_file_name: Optional[str] = None,
    auv_file_bytes: Optional[bytes] = None,
) -> TerrainGrid:
    """Generates or loads cached TerrainGrid based on parameters and satellite/AUV sensor input."""
    terrain = generate_terrain(
        rows=DEFAULT_GRID_ROWS,
        cols=DEFAULT_GRID_COLS,
        cell_size_m=DEFAULT_CELL_SIZE_M,
        base_lat=base_lat,
        base_lon=base_lon,
        seed=seed,
    )
    if dem_file_bytes is not None and len(dem_file_bytes) > 0:
        success, msg, updated = load_dem_raster(
            io.BytesIO(dem_file_bytes), terrain, filename=dem_file_name
        )
        if success and updated is not None:
            terrain = updated
    if auv_file_bytes is not None and len(auv_file_bytes) > 0:
        success, msg, updated = load_auv_csv(io.BytesIO(auv_file_bytes), terrain)
        if success and updated is not None:
            terrain = updated
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

    mission_mode = st.sidebar.radio(
        "Operational Command Mode",
        options=["Single Vehicle Asset", "Tactical Echelon Convoy (Joint Arms)"],
        index=0,
        help="Switch between single vehicle reconnaissance and multi-echelon military convoy logistics with tactical bridgehead deployment.",
    )
    is_convoy_mode = (mission_mode == "Tactical Echelon Convoy (Joint Arms)")

    if is_convoy_mode:
        st.sidebar.info(
            "🎖️ **Convoy Echelons Active**:\n"
            "• 🚙 **Recon**: Light 4x4 Jeep (Scout / Trailblazer)\n"
            "• 🛡️ **Spearhead**: Heavy Tracked MBT (Main Armor)\n"
            "• 🚚 **Logistics**: Medium Tactical 6x6 Truck (Supplies)"
        )
        deploy_avlb = st.sidebar.checkbox(
            "Deploy Combat Engineering Asset (AVLB Bridge)",
            value=True,
            help="Deploys an Armored Vehicle-Launched Bridge across impassable river gaps so logistics trucks can follow.",
        )
        vehicle_preset_name = "Heavy Tracked Tank"
        base_vehicle = VEHICLE_PRESETS[vehicle_preset_name]
        custom_slope = float(base_vehicle.max_slope_deg)
        custom_fording = float(base_vehicle.max_fording_m)
        custom_firmness = float(base_vehicle.min_firmness)
        custom_speed_go = float(base_vehicle.speed_kmph_go)
        custom_speed_slowgo = float(base_vehicle.speed_kmph_slowgo)
    else:
        deploy_avlb = False
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
    st.sidebar.subheader("📡 Satellite & AUV Sensor Ingestion")
    dem_file = st.sidebar.file_uploader(
        "Upload Satellite DEM GeoTIFF/TIFF/CSV (Optional)",
        type=["tif", "tiff", "csv"],
        help="Upload real Copernicus/SRTM DEM raster or elevation matrix (pure-pip ingestion).",
    )
    auv_file = st.sidebar.file_uploader(
        "Upload AUV Bathymetry CSV (Optional)",
        type=["csv"],
        help="Upload field AUV sonar data with lat, lon, water_depth_m, firmness columns.",
    )

    dem_bytes = dem_file.getvalue() if dem_file is not None else None
    dem_name = dem_file.name if dem_file is not None else None
    auv_bytes = auv_file.getvalue() if auv_file is not None else None

    # Load terrain and trafficability
    terrain = get_cached_terrain(
        seed=int(seed_input),
        base_lat=float(base_lat),
        base_lon=float(base_lon),
        dem_file_bytes=dem_bytes,
        dem_file_name=dem_name,
        auv_file_bytes=auv_bytes,
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
    if is_convoy_mode:
        convoy_plan = plan_tactical_convoy(
            terrain=terrain,
            start=(start_row, start_col),
            goal=(goal_row, goal_col),
            is_xy=False,
            rainfall_multiplier=rainfall_mult,
            current_factor=current_factor,
            deploy_avlb=deploy_avlb,
        )
        spearhead_element = convoy_plan.elements.get("Spearhead")
        path_res = spearhead_element.path_result if spearhead_element else PathResult([], 0.0, PathStatus.NO_PATH, "No Spearhead")
        metrics, profile = compute_route_analytics(
            terrain,
            trafficability,
            path_res,
            is_xy=False,
            cell_size_m=terrain.cell_size_m,
        )
    else:
        convoy_plan = None
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

    if is_convoy_mode:
        m_col1.metric("Convoy Mission", convoy_plan.status)
        m_col2.metric("Total Convoy Span", f"{convoy_plan.total_convoy_span_min:.1f} min")
        recon_el = convoy_plan.elements.get("Recon")
        spear_el = convoy_plan.elements.get("Spearhead")
        logis_el = convoy_plan.elements.get("Logistics")
        
        recon_str = f"+{recon_el.arrival_time_min:.1f} min" if (recon_el and recon_el.path_result.status == PathStatus.OK) else "BLOCKED"
        m_col3.metric("Recon Scout ETA", recon_str)

        spear_str = f"+{spear_el.arrival_time_min:.1f} min" if (spear_el and spear_el.path_result.status == PathStatus.OK) else "BLOCKED"
        m_col4.metric("Spearhead MBT ETA", spear_str)

        bridge_badge = "AVLB Deployed" if convoy_plan.bridge_deployed else "No Bridge Needed"
        logis_str = f"+{logis_el.arrival_time_min:.1f} min" if (logis_el and logis_el.path_result.status == PathStatus.OK) else "BLOCKED"
        m_col5.metric("Logistics Truck ETA", logis_str, delta=bridge_badge)

    else:
        if metrics is not None:
            m_col1.metric("Tactical Distance", f"{metrics.distance_km:.2f} km")
            m_col2.metric("Transit ETA", f"{metrics.eta_minutes:.1f} min", f"{metrics.eta_hours:.2f} hrs")
            m_col3.metric("GO Traversal", f"{metrics.pct_go:.1f}%")
            m_col4.metric("SLOW-GO Traversal", f"{metrics.pct_slowgo:.1f}%")
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
            "Actionable remedy: Switch to Convoy Mode with AVLB Combat Bridge, select Heavy Tracked Tank, or reduce rainfall."
        )

    # --- Interactive Folium Map ---
    st.subheader("🗺️ Tactical Mobility Overlay & Vector Routing")

    folium_map = build_trafficability_folium_map(
        terrain=terrain,
        trafficability=trafficability,
        path_result=path_res if not is_convoy_mode else None,
        convoy_plan=convoy_plan,
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
    if is_convoy_mode:
        tab_convoy, tab_profile, tab_breakdown, tab_rasters, tab_table, tab_export = st.tabs([
            "🎖️ Convoy Logistics Schedule",
            "📈 Longitudinal Profile",
            "📊 Mobility Breakdown",
            "🛰️ Raw Sensor Layers",
            "📋 Route Table",
            "📦 Mission Export",
        ])
        with tab_convoy:
            st.subheader("Tactical Echelon Convoy Movement Order & Bridging Logistics")
            st.info(convoy_plan.message)

            convoy_rows = []
            for name, elem in convoy_plan.elements.items():
                convoy_rows.append({
                    "Echelon": name,
                    "Vehicle Asset": elem.vehicle.name,
                    "Role": elem.vehicle.role.value,
                    "Departure Delay (min)": f"+{elem.departure_delay_min:.1f}",
                    "Transit Duration (min)": f"{elem.transit_duration_min:.1f}",
                    "Arrival Time (min)": f"+{elem.arrival_time_min:.1f}",
                    "Waypoints Traversed": len(elem.path_result.path_cells),
                    "Status": elem.path_result.status.value,
                })
            df_convoy = pd.DataFrame(convoy_rows)
            st.dataframe(df_convoy, use_container_width=True)

            if convoy_plan.bridge_cells:
                st.success(
                    f"🌉 Combat Engineering Asset deployed AVLB bridge across {len(convoy_plan.bridge_cells)} river cell(s). "
                    "Logistics support trucks cleared for crossing!"
                )
    else:
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
