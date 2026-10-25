"""Unit tests for Terrain & Environmental Model (Gate 1)."""

import numpy as np
import pytest
from core.config import (
    DEFAULT_GRID_ROWS,
    DEFAULT_GRID_COLS,
    DEFAULT_CELL_SIZE_M,
    DEFAULT_BASE_LAT,
    DEFAULT_BASE_LON,
    cell_to_latlon,
    latlon_to_cell,
    get_grid_bounds,
)
from core.terrain import generate_terrain, compute_slope


def test_slope_exact_on_tilted_plane():
    """Verify that slope computation on a known synthetic tilted plane is mathematically exact."""
    rows, cols = 50, 50
    cell_size = 100.0
    # Create an elevation plane tilted along x (dz/dx = tan(30 degrees))
    # tan(30 deg) = 0.5773502691896257
    target_slope_deg = 30.0
    gradient_val = np.tan(np.radians(target_slope_deg))

    # z = gradient_val * (x * cell_size)
    _, x_coords = np.indices((rows, cols))
    plane = gradient_val * (x_coords * cell_size)

    slope = compute_slope(plane, cell_size_m=cell_size)

    # Interior cells (excluding boundary where one-sided difference occurs)
    interior_slope = slope[1:-1, 1:-1]
    assert np.allclose(interior_slope, target_slope_deg, atol=1e-5), (
        f"Slope should be {target_slope_deg} deg, got mean {np.mean(interior_slope)}"
    )

    # Test tilt along diagonal (45 deg azimuth tilt with 20 deg slope)
    target_slope_20 = 20.0
    grad_20 = np.tan(np.radians(target_slope_20))
    y_coords, _ = np.indices((rows, cols))
    # dz/dx = grad_20 / sqrt(2), dz/dy = grad_20 / sqrt(2)
    diag_plane = (grad_20 / np.sqrt(2)) * (x_coords * cell_size) + (grad_20 / np.sqrt(2)) * (y_coords * cell_size)
    diag_slope = compute_slope(diag_plane, cell_size_m=cell_size)
    assert np.allclose(diag_slope[1:-1, 1:-1], target_slope_20, atol=1e-5)


def test_terrain_reproducibility():
    """Verify that same seed generates byte-for-byte identical terrain grids."""
    grid_a = generate_terrain(seed=1234)
    grid_b = generate_terrain(seed=1234)

    assert np.array_equal(grid_a.elevation, grid_b.elevation)
    assert np.array_equal(grid_a.slope, grid_b.slope)
    assert np.array_equal(grid_a.soil_type, grid_b.soil_type)
    assert np.array_equal(grid_a.moisture, grid_b.moisture)
    assert np.array_equal(grid_a.river_mask, grid_b.river_mask)
    assert np.array_equal(grid_a.water_depth, grid_b.water_depth)
    # nan-safe check for riverbed firmness
    assert np.allclose(grid_a.riverbed_firmness, grid_b.riverbed_firmness, equal_nan=True)

    # Different seeds must produce different terrain
    grid_c = generate_terrain(seed=9999)
    assert not np.array_equal(grid_a.elevation, grid_c.elevation)


def test_river_depth_ranges_and_outside():
    """Verify river depths are strictly within [0.2, 3.5] m inside river, and 0.0 outside."""
    grid = generate_terrain(seed=42)

    # Outside river: depth is 0.0, firmness is NaN
    assert np.all(grid.water_depth[~grid.river_mask] == 0.0)
    assert np.all(np.isnan(grid.riverbed_firmness[~grid.river_mask]))

    # Inside river: depths in [0.2, 3.5] m, firmness in [0.0, 1.0]
    river_depths = grid.water_depth[grid.river_mask]
    assert np.all(river_depths >= 0.20)
    assert np.all(river_depths <= 3.50)

    river_firmness = grid.riverbed_firmness[grid.river_mask]
    assert np.all(~np.isnan(river_firmness))
    assert np.all(river_firmness >= 0.0)
    assert np.all(river_firmness <= 1.0)


def test_river_connects_north_and_south_edges():
    """Verify that the river continuously connects the North edge (row 0) to South edge (row N-1)."""
    grid = generate_terrain(seed=42)

    # Every row must have river cells
    for r in range(grid.rows):
        assert np.any(grid.river_mask[r, :]), f"River broken at row {r}"

    # Check 8-connectivity of river mask from north to south
    from scipy.ndimage import label
    # Structure for 8-connectivity
    struct = np.ones((3, 3), dtype=int)
    labeled, num_features = label(grid.river_mask, structure=struct)

    north_labels = set(labeled[0, grid.river_mask[0, :]])
    south_labels = set(labeled[grid.rows - 1, grid.river_mask[grid.rows - 1, :]])

    # There must be at least one common connected component connecting row 0 to row N-1
    assert len(north_labels.intersection(south_labels)) > 0, "River does not form a connected path from North to South"


def test_coordinate_transforms_and_bounds():
    """Verify coordinate roundtripping and bounds checking."""
    rows, cols = DEFAULT_GRID_ROWS, DEFAULT_GRID_COLS
    cell_size = DEFAULT_CELL_SIZE_M
    base_lat = DEFAULT_BASE_LAT
    base_lon = DEFAULT_BASE_LON

    # Check center cell roundtrip
    center_coords = cell_to_latlon(50, 50, base_lat, base_lon, rows, cols, cell_size)
    assert center_coords is not None
    # Center cell (50, 50) is centered at base_lat, base_lon
    assert np.isclose(center_coords[0], base_lat, atol=1e-3)
    assert np.isclose(center_coords[1], base_lon, atol=1e-3)

    # Roundtrip check across arbitrary cells
    test_cells = [(0, 0), (99, 99), (25, 75), (10, 80)]
    for r, c in test_cells:
        latlon = cell_to_latlon(r, c, base_lat, base_lon, rows, cols, cell_size)
        assert latlon is not None
        recon_r, recon_c = latlon_to_cell(latlon[0], latlon[1], base_lat, base_lon, rows, cols, cell_size)
        assert (recon_r, recon_c) == (r, c)

    # Out of bounds checks
    assert cell_to_latlon(-1, 0) is None
    assert cell_to_latlon(0, 100) is None
    assert latlon_to_cell(0.0, 0.0) is None  # Far away


def test_terrain_to_dataframe():
    """Verify DataFrame export schema and row count."""
    grid = generate_terrain(seed=42)
    df = grid.to_dataframe()
    assert len(df) == grid.rows * grid.cols
    expected_cols = {
        "x", "y", "lat", "lon", "elevation", "slope",
        "soil_type", "moisture", "water_depth", "riverbed_firmness"
    }
    assert expected_cols.issubset(set(df.columns))
