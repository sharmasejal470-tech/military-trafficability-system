"""Unit tests for Vehicle Trafficability Prediction Engine (Gate 2)."""

import numpy as np
import pytest
from core.config import (
    TrafficabilityClass,
    MobilityReason,
    SoilType,
)
from core.terrain import generate_terrain
from core.vehicles import VEHICLE_PRESETS, VehicleProfile
from core.trafficability import compute_trafficability


@pytest.fixture
def base_terrain():
    return generate_terrain(seed=42)


def test_heavy_vehicle_on_marsh_is_no_go(base_terrain):
    """Verify Heavy Tracked Tank on Marsh is strictly No-Go (Red)."""
    tank = VEHICLE_PRESETS["Heavy Tracked Tank"]
    result = compute_trafficability(base_terrain, tank, rainfall_multiplier=1.0, current_factor=0.0)

    marsh_mask = (base_terrain.soil_type == SoilType.MARSH_WETLAND)
    assert np.any(marsh_mask), "Terrain should contain marsh cells"

    # All marsh cells for heavy tank must be NO_GO (Red = 2)
    assert np.all(result.grid[marsh_mask] == TrafficabilityClass.NO_GO)
    # Check that the reason code bitmask recorded HEAVY_ON_MARSH
    assert np.all((result.reasons[marsh_mask] & MobilityReason.NO_GO_HEAVY_ON_MARSH) != 0)


def test_non_heavy_vehicle_on_marsh_is_slow_go_unless_slope_or_water(base_terrain):
    """Verify Light Recon Jeep on flat non-flooded marsh is Slow-Go (Yellow), not automatically Red."""
    jeep = VEHICLE_PRESETS["Light Recon 4x4 Jeep"]
    result = compute_trafficability(base_terrain, jeep, rainfall_multiplier=1.0, current_factor=0.0)

    marsh_mask = (
        (base_terrain.soil_type == SoilType.MARSH_WETLAND) &
        (base_terrain.slope <= jeep.max_slope_deg) &
        (base_terrain.water_depth <= jeep.max_fording_m) &
        (~base_terrain.river_mask)
    )
    assert np.any(marsh_mask), "Should have dry marsh cells"
    assert np.all(result.grid[marsh_mask] == TrafficabilityClass.SLOW_GO)


def test_differential_fording_depth(base_terrain):
    """Verify that a deep water cell (e.g. 1.1m) is Red for Jeep (max 0.6m) and safe/Yellow for Tank (max 1.4m)."""
    jeep = VEHICLE_PRESETS["Light Recon 4x4 Jeep"]
    tank = VEHICLE_PRESETS["Heavy Tracked Tank"]

    res_jeep = compute_trafficability(base_terrain, jeep, rainfall_multiplier=1.0, current_factor=0.0)
    res_tank = compute_trafficability(base_terrain, tank, rainfall_multiplier=1.0, current_factor=0.0)

    # Find river cells where depth is between 0.8m and 1.2m and slope is low, firmness adequate for tank
    candidate_mask = (
        base_terrain.river_mask &
        (base_terrain.water_depth > 0.8) &
        (base_terrain.water_depth < 1.2) &
        (base_terrain.slope < 10.0) &
        (base_terrain.riverbed_firmness >= tank.min_firmness)
    )

    assert np.any(candidate_mask), "Deep water cells should exist in synthetic river"

    # For Jeep, these cells must be NO_GO due to water depth
    assert np.all(res_jeep.grid[candidate_mask] == TrafficabilityClass.NO_GO)
    assert np.all((res_jeep.reasons[candidate_mask] & MobilityReason.NO_GO_WATER_DEPTH) != 0)

    # For Tank, water depth (0.8-1.2m) <= 1.4m and firmness >= 0.75, slope < 10.
    # Therefore Tank can traverse it (either Slow-Go or Go, but NOT No-Go)
    assert np.all(res_tank.grid[candidate_mask] != TrafficabilityClass.NO_GO)


def test_rainfall_hazard_monotonicity(base_terrain):
    """
    Verify that increasing rainfall from 1.0 to 2.0 NEVER makes any cell safer.
    Grid values: 0 = Go, 1 = Slow-Go, 2 = No-Go.
    Condition: grid(rain=2.0) >= grid(rain=1.0) everywhere.
    """
    truck = VEHICLE_PRESETS["Medium Tactical 6x6 Truck"]
    res_rain_1 = compute_trafficability(base_terrain, truck, rainfall_multiplier=1.0, current_factor=0.2)
    res_rain_2 = compute_trafficability(base_terrain, truck, rainfall_multiplier=2.0, current_factor=0.2)

    diff = res_rain_2.grid - res_rain_1.grid
    # diff must be >= 0 everywhere (can remain same, or increase danger, never decrease)
    assert np.all(diff >= 0), "Rainfall 2.0 must never make any cell safer than rainfall 1.0"


def test_red_overrides_yellow_priority(base_terrain):
    """
    Verify that whenever a Red condition is met, the cell is strictly classified as Red (NO_GO),
    even if Yellow criteria are also triggered.
    """
    truck = VEHICLE_PRESETS["Medium Tactical 6x6 Truck"]
    result = compute_trafficability(base_terrain, truck, rainfall_multiplier=1.2, current_factor=0.3)

    # Identify any cells with slope > max_slope
    excessive_slope = base_terrain.slope > truck.max_slope_deg
    assert np.all(result.grid[excessive_slope] == TrafficabilityClass.NO_GO)

    # Identify any cells with water > effective max fording
    excessive_water = (base_terrain.water_depth * (1.0 + 0.15 * 0.2)) > result.effective_max_fording
    assert np.all(result.grid[excessive_water] == TrafficabilityClass.NO_GO)


def test_current_factor_reduces_fording_limit(base_terrain):
    """Verify that current factor properly depresses the allowable safe fording limit."""
    truck = VEHICLE_PRESETS["Medium Tactical 6x6 Truck"]
    res_calm = compute_trafficability(base_terrain, truck, current_factor=0.0)
    res_torrent = compute_trafficability(base_terrain, truck, current_factor=1.0)

    assert res_calm.effective_max_fording == truck.max_fording_m
    assert res_torrent.effective_max_fording < res_calm.effective_max_fording
    # With k=0.40, effective depth at current=1.0 should be 60% of base
    assert np.isclose(res_torrent.effective_max_fording, 0.60 * truck.max_fording_m)
