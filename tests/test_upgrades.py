"""Unit tests for the advanced capabilities:
1. Pure-pip Satellite DEM / GeoTIFF ingestion (core/io_adapters.py)
2. 2D Hydrodynamic flow velocity & variable drag (core/terrain.py & core/trafficability.py)
3. Multi-vehicle tactical convoy echelon planning & AVLB combat bridging (core/pathfinding.py)
"""

import io
import numpy as np
import pytest
from PIL import Image

from core.terrain import generate_terrain
from core.trafficability import compute_trafficability
from core.vehicles import VEHICLE_PRESETS
from core.io_adapters import load_dem_raster
from core.pathfinding import plan_tactical_convoy, PathStatus
from core.config import (
    DEFAULT_SEED,
    DEFAULT_BASE_LAT,
    DEFAULT_BASE_LON,
    DEFAULT_GRID_ROWS,
    DEFAULT_GRID_COLS,
    DEFAULT_CELL_SIZE_M,
    TrafficabilityClass,
)


@pytest.fixture
def base_terrain():
    return generate_terrain(
        rows=DEFAULT_GRID_ROWS,
        cols=DEFAULT_GRID_COLS,
        cell_size_m=DEFAULT_CELL_SIZE_M,
        base_lat=DEFAULT_BASE_LAT,
        base_lon=DEFAULT_BASE_LON,
        seed=DEFAULT_SEED,
    )


def test_2d_hydrodynamic_flow_velocity(base_terrain):
    """Verifies that 2D hydrodynamic flow field is correctly modeled."""
    # 1. Existence and shapes of fields
    assert hasattr(base_terrain, "river_velocity")
    assert hasattr(base_terrain, "flow_vector_u")
    assert hasattr(base_terrain, "flow_vector_v")
    assert base_terrain.river_velocity.shape == (100, 100)

    # 2. Non-river cells must have 0.0 velocity
    dry_mask = ~base_terrain.river_mask
    assert np.all(base_terrain.river_velocity[dry_mask] == 0.0)

    # 3. River cells must have positive velocities
    wet_mask = base_terrain.river_mask
    assert np.all(base_terrain.river_velocity[wet_mask] > 0.0)
    assert np.max(base_terrain.river_velocity) <= 3.5

    # 4. Hydrodynamic flow effect on trafficability
    jeep = VEHICLE_PRESETS["Light Recon 4x4 Jeep"]
    traff_calm = compute_trafficability(base_terrain, jeep, rainfall_multiplier=1.0, current_factor=0.0)
    traff_fast = compute_trafficability(base_terrain, jeep, rainfall_multiplier=1.0, current_factor=1.0)

    # Under maximum river current, Go cells can degrade to Slow-Go or No-Go in the river
    assert (traff_fast.grid == TrafficabilityClass.NO_GO).sum() >= (traff_calm.grid == TrafficabilityClass.NO_GO).sum()


def test_pure_pip_satellite_dem_csv_ingestion(base_terrain):
    """Verifies pure-pip ingestion of CSV elevation matrices."""
    # Generate a synthetic 40x40 elevation matrix
    raw_elev = np.full((40, 40), 500.0, dtype=float)
    raw_elev[10:20, 10:20] = 750.0  # plateau

    bio = io.BytesIO()
    np.savetxt(bio, raw_elev, delimiter=",", fmt="%.2f")
    bio.seek(0)

    success, msg, updated = load_dem_raster(bio, base_terrain, filename="satellite_dem.csv")
    assert success is True
    assert updated is not None
    assert updated.elevation.shape == (100, 100)
    # The plateau should be smoothly resampled
    assert np.isclose(updated.elevation[0, 0], 500.0, atol=2.0)
    assert updated.elevation.max() >= 740.0
    # Slope should be computed automatically
    assert updated.slope.shape == (100, 100)
    assert updated.slope.max() > 0.0


def test_pure_pip_satellite_dem_tiff_ingestion(base_terrain):
    """Verifies pure-pip ingestion of TIFF / GeoTIFF elevation rasters via PIL."""
    # Create an in-memory 50x50 16-bit or 32-bit TIFF
    data = (np.random.default_rng(42).random((50, 50)) * 300.0 + 200.0).astype(np.float32)
    img = Image.fromarray(data)
    bio = io.BytesIO()
    img.save(bio, format="TIFF")
    bio.seek(0)

    success, msg, updated = load_dem_raster(bio, base_terrain, filename="srtm_copernicus.tif")
    assert success is True
    assert updated is not None
    assert updated.elevation.shape == (100, 100)
    assert np.all(updated.elevation >= 190.0)
    assert np.all(updated.elevation <= 510.0)


def test_tactical_convoy_logistics_and_avlb(base_terrain):
    """Verifies military convoy echelon coordination and AVLB bridge deployment."""
    start_cell = (15, 85)
    goal_cell = (85, 15)

    # 1. Run convoy planning with AVLB engineering asset enabled
    convoy = plan_tactical_convoy(
        terrain=base_terrain,
        start=start_cell,
        goal=goal_cell,
        is_xy=False,
        rainfall_multiplier=1.0,
        current_factor=0.20,
        deploy_avlb=True,
    )

    assert convoy.status in ("CONVOY_OPERATIONAL", "CONVOY_OBSTRUCTED")
    assert "Recon" in convoy.elements
    assert "Spearhead" in convoy.elements
    assert "Logistics" in convoy.elements

    recon = convoy.elements["Recon"]
    spearhead = convoy.elements["Spearhead"]
    logistics = convoy.elements["Logistics"]

    # Doctrine departure echelon delay ordering
    assert recon.departure_delay_min == 0.0
    assert spearhead.departure_delay_min == 10.0
    assert logistics.departure_delay_min == 25.0

    # Spearhead MBT should reach destination
    assert spearhead.path_result.status == PathStatus.OK
    assert spearhead.arrival_time_min > spearhead.departure_delay_min

    # Logistics truck: if bridge was deployed, it successfully crosses
    if convoy.bridge_deployed:
        assert len(convoy.bridge_cells) > 0
        assert logistics.path_result.status == PathStatus.OK
        assert logistics.arrival_time_min > logistics.departure_delay_min
