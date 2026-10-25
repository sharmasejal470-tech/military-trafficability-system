"""Unit tests for Tactical Analytics, River Crossing Assessment, and I/O Adapters (Gate 3)."""

import io
import numpy as np
import pytest

from core.config import (
    TrafficabilityClass,
    DEFAULT_CELL_SIZE_M,
)
from core.terrain import generate_terrain
from core.vehicles import VEHICLE_PRESETS
from core.trafficability import compute_trafficability
from core.pathfinding import find_path, nearest_valid_cell, PathStatus
from core.analytics import (
    compute_route_analytics,
    assess_river_crossing,
    CrossingSafetyStatus,
)
from core.io_adapters import load_auv_csv, generate_sample_auv_csv


@pytest.fixture
def base_scenario():
    terrain = generate_terrain(seed=42)
    truck = VEHICLE_PRESETS["Medium Tactical 6x6 Truck"]
    traff = compute_trafficability(terrain, truck, rainfall_multiplier=1.0, current_factor=0.0)
    start = nearest_valid_cell(traff.grid, (85, 15))
    goal = nearest_valid_cell(traff.grid, (15, 85))
    path_res = find_path(traff.grid, start, goal)
    return terrain, traff, path_res


def test_route_analytics_calculation(base_scenario):
    """Verify route metrics and longitudinal profile computation."""
    terrain, traff, path_res = base_scenario
    metrics, profile = compute_route_analytics(terrain, traff, path_res)

    assert metrics is not None
    assert profile is not None

    # Distance must be positive and realistic (10km x 10km grid diagonal ~ 14km)
    assert 5.0 < metrics.distance_km < 25.0
    # ETA must be positive
    assert metrics.eta_minutes > 0.0
    # Percentage go + slow-go should sum to 100% since no Red cells are traversed
    assert np.isclose(metrics.pct_go + metrics.pct_slowgo, 100.0, atol=1e-3)
    # Waypoint counts match
    assert metrics.waypoint_count == len(path_res.path_cells)

    # Check profile arrays
    df_prof = profile.to_dataframe()
    assert len(df_prof) == metrics.waypoint_count
    assert np.all(df_prof["cumulative_distance_km"] >= 0.0)
    assert np.all(df_prof["elevation_m"] > 300.0)


def test_river_crossing_assessment_logic():
    """Verify SAFE, CAUTION, UNSAFE, and NO_CROSSING assessment branches."""
    effective_fording = 1.0  # 1.0m
    vehicle_min_firm = 0.60  # 0.60

    # No crossing
    status, reason = assess_river_crossing(0, 0.0, None, effective_fording, vehicle_min_firm)
    assert status == CrossingSafetyStatus.NO_CROSSING

    # UNSAFE due to water depth exceeding limit
    status, reason = assess_river_crossing(5, 1.2, 0.85, effective_fording, vehicle_min_firm)
    assert status == CrossingSafetyStatus.UNSAFE
    assert "UNSAFE: Water depth" in reason

    # UNSAFE due to weak firmness
    status, reason = assess_river_crossing(5, 0.5, 0.45, effective_fording, vehicle_min_firm)
    assert status == CrossingSafetyStatus.UNSAFE
    assert "UNSAFE: Substrate firmness" in reason

    # CAUTION due to narrow depth margin (< 0.25m clearance, e.g. depth 0.85m -> margin 0.15m)
    status, reason = assess_river_crossing(5, 0.85, 0.80, effective_fording, vehicle_min_firm)
    assert status == CrossingSafetyStatus.CAUTION
    assert "CAUTION" in reason

    # SAFE: generous depth clearance (> 0.25m) and firmness buffer (> 0.10)
    status, reason = assess_river_crossing(5, 0.45, 0.85, effective_fording, vehicle_min_firm)
    assert status == CrossingSafetyStatus.SAFE
    assert "SAFE" in reason


def test_auv_csv_loader_valid(base_scenario):
    """Verify loading and KDTree rasterization of valid AUV CSV."""
    terrain, _, _ = base_scenario
    df = generate_sample_auv_csv(terrain, num_samples=100)
    csv_bytes = io.BytesIO()
    df.to_csv(csv_bytes, index=False)
    csv_bytes.seek(0)

    success, msg, new_terrain = load_auv_csv(csv_bytes, terrain)
    assert success is True
    assert "Successfully rasterized" in msg
    assert new_terrain is not None
    assert new_terrain.water_depth.shape == terrain.water_depth.shape


def test_auv_csv_loader_invalid_columns(base_scenario):
    """Verify robust error reporting when required columns are missing."""
    terrain, _, _ = base_scenario
    csv_data = io.StringIO("latitude,longitude,depth\n26.9,75.7,1.2\n")

    success, msg, new_terrain = load_auv_csv(csv_data, terrain)
    assert success is False
    assert "Missing required columns" in msg
    assert new_terrain is None


def test_auv_csv_loader_empty_file(base_scenario):
    """Verify error reporting on empty CSV file."""
    terrain, _, _ = base_scenario
    csv_data = io.StringIO("")

    success, msg, new_terrain = load_auv_csv(csv_data, terrain)
    assert success is False
    assert "contains no data rows" in msg or "Failed to parse" in msg
