"""Input/Output and Data Ingestion Adapters.

Enables CSV/GeoTIFF-free ingestion of real-world AUV bathymetry surveys,
satellite elevation point clouds, and GIS field sensor measurements.
"""

from __future__ import annotations
import io
from dataclasses import replace
from typing import Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from core.config import (
    cell_to_latlon,
    latlon_to_cell,
    get_grid_bounds,
    DEFAULT_GRID_ROWS,
    DEFAULT_GRID_COLS,
    DEFAULT_CELL_SIZE_M,
    DEFAULT_BASE_LAT,
    DEFAULT_BASE_LON,
)
from core.terrain import TerrainGrid, compute_slope


REQUIRED_AUV_COLUMNS = {"lat", "lon", "water_depth_m", "firmness"}


def load_auv_csv(
    file_or_path: Union[str, io.BytesIO, io.StringIO, Any],
    terrain: TerrainGrid,
    max_search_dist_cells: float = 4.0,
) -> Tuple[bool, str, Optional[TerrainGrid]]:
    """
    Ingests and validates an AUV sonar bathymetry survey CSV, rasterizing the point
    measurements onto the terrain grid using spatial KD-Tree interpolation.
    
    Parameters:
        file_or_path: Filepath or file-like object containing tabular survey points.
        terrain: Baseline TerrainGrid to update with survey data.
        max_search_dist_cells: Maximum spatial radius in cells for survey point influence.
    
    Returns:
        (success: bool, message: str, updated_terrain: Optional[TerrainGrid])
    """
    # 1. Ingestion & Format Validation
    try:
        if isinstance(file_or_path, (str, io.StringIO, io.BytesIO)):
            df = pd.read_csv(file_or_path)
        elif hasattr(file_or_path, "read"):
            # Streamlit UploadedFile or file-like
            df = pd.read_csv(file_or_path)
        else:
            return False, "Unsupported file input type.", None
    except Exception as e:
        return False, f"Failed to parse CSV file: {str(e)}", None

    if df.empty:
        return False, "AUV survey CSV file contains no data rows.", None

    missing_cols = REQUIRED_AUV_COLUMNS - set(df.columns)
    if missing_cols:
        return False, f"Missing required columns in CSV: {sorted(list(missing_cols))}", None

    # Clean numeric fields
    for col in REQUIRED_AUV_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid_rows = df.dropna(subset=list(REQUIRED_AUV_COLUMNS))
    if len(valid_rows) < 3:
        return False, f"Insufficient valid hydrographic survey points (found {len(valid_rows)}, minimum 3 required).", None

    # 2. Project geographic lat/lon points to grid row/col space
    pts_lat = valid_rows["lat"].values
    pts_lon = valid_rows["lon"].values
    pts_depth = valid_rows["water_depth_m"].values
    pts_firmness = valid_rows["firmness"].values

    grid_pts = []
    depths = []
    firmness = []

    for lat, lon, d, f in zip(pts_lat, pts_lon, pts_depth, pts_firmness):
        cell = latlon_to_cell(
            lat, lon,
            base_lat=terrain.base_lat,
            base_lon=terrain.base_lon,
            rows=terrain.rows,
            cols=terrain.cols,
            cell_size_m=terrain.cell_size_m,
        )
        if cell is not None:
            r, c = cell
            grid_pts.append([float(r), float(c)])
            depths.append(float(d))
            firmness.append(float(f))

    if not grid_pts:
        return (
            False,
            "None of the survey coordinates overlap with the current tactical grid bounding box.",
            None,
        )

    grid_pts_arr = np.array(grid_pts)
    depths_arr = np.clip(np.array(depths), 0.0, 10.0)
    firmness_arr = np.clip(np.array(firmness), 0.0, 1.0)

    # 3. Spatial Interpolation with cKDTree
    tree = cKDTree(grid_pts_arr)

    # Create writable copies of hydrographic arrays
    new_water_depth = terrain.water_depth.copy()
    new_firmness = terrain.riverbed_firmness.copy()

    # Query all river cells
    river_r, river_c = np.where(terrain.river_mask)
    if len(river_r) == 0:
        return False, "Target grid contains no defined river channel.", None

    river_coords = np.column_stack([river_r, river_c])
    # Find nearest survey point within max search distance
    dists, indices = tree.query(river_coords, distance_upper_bound=max_search_dist_cells)

    updated_count = 0
    for idx, (dist, target_idx) in enumerate(zip(dists, indices)):
        if dist < max_search_dist_cells and target_idx < len(depths_arr):
            r = river_r[idx]
            c = river_c[idx]
            new_water_depth[r, c] = depths_arr[target_idx]
            new_firmness[r, c] = firmness_arr[target_idx]
            updated_count += 1

    if updated_count == 0:
        return (
            False,
            f"AUV survey points did not fall within {max_search_dist_cells} cells of the active river channel.",
            None,
        )

    # Return updated immutable TerrainGrid
    updated_terrain = replace(
        terrain,
        water_depth=new_water_depth,
        riverbed_firmness=new_firmness,
    )

    msg = f"Successfully rasterized {len(grid_pts_arr)} AUV sonar points ({updated_count} river cells updated)."
    return True, msg, updated_terrain


def generate_sample_auv_csv(
    terrain: TerrainGrid,
    num_samples: int = 150,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates synthetic hydrographic survey points along the river for demo and testing."""
    rng = np.random.default_rng(seed)
    river_rows, river_cols = np.where(terrain.river_mask)

    if len(river_rows) == 0:
        return pd.DataFrame(columns=list(REQUIRED_AUV_COLUMNS))

    # Sample random river cells
    sample_indices = rng.choice(len(river_rows), size=min(num_samples, len(river_rows)), replace=False)
    sample_rows = river_rows[sample_indices]
    sample_cols = river_cols[sample_indices]

    lats = []
    lons = []
    depths = []
    firmness = []

    for r, c in zip(sample_rows, sample_cols):
        # Slightly jitter position within the cell
        coords = cell_to_latlon(
            int(r), int(c),
            base_lat=terrain.base_lat,
            base_lon=terrain.base_lon,
            rows=terrain.rows,
            cols=terrain.cols,
            cell_size_m=terrain.cell_size_m,
        )
        if coords is not None:
            lats.append(coords[0] + rng.uniform(-0.0002, 0.0002))
            lons.append(coords[1] + rng.uniform(-0.0002, 0.0002))
            # Measured depth with sonar noise
            d = float(terrain.water_depth[r, c] + rng.uniform(-0.05, 0.05))
            f = float(terrain.riverbed_firmness[r, c] + rng.uniform(-0.03, 0.03))
            depths.append(np.clip(d, 0.20, 3.50))
            firmness.append(np.clip(f, 0.15, 0.98))

    return pd.DataFrame({
        "lat": lats,
        "lon": lons,
        "water_depth_m": depths,
        "firmness": firmness,
    })
