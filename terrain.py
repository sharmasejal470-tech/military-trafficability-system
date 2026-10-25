"""Terrain and hydrographic environment synthesis module.

Fuses synthetic digital elevation models (DEM), slope analysis, soil categorization,
surface moisture fields, AUV sonar bathymetry, and 2D hydrodynamic river flow velocities.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, distance_transform_edt

from core.config import (
    DEFAULT_GRID_ROWS,
    DEFAULT_GRID_COLS,
    DEFAULT_CELL_SIZE_M,
    DEFAULT_BASE_LAT,
    DEFAULT_BASE_LON,
    DEFAULT_SEED,
    SoilType,
    cell_to_latlon,
)


@dataclass(frozen=True)
class TerrainGrid:
    """
    Immutable representation of the 2D environmental and hydrographic grid.
    Grid indexing convention: [row, col] where row 0 = North, col 0 = West.
    """
    elevation: np.ndarray          # meters above sea level [rows, cols]
    slope: np.ndarray              # slope in degrees [rows, cols]
    soil_type: np.ndarray          # SoilType int code [rows, cols]
    moisture: np.ndarray           # normalized moisture [0.0, 1.0] [rows, cols]
    river_mask: np.ndarray         # boolean mask of river channel [rows, cols]
    water_depth: np.ndarray        # water depth in meters [rows, cols]
    riverbed_firmness: np.ndarray   # AUV measured firmness [0.0, 1.0] (NaN outside river)
    river_velocity: np.ndarray     # 2D hydrodynamic current speed (m/s) [rows, cols]
    flow_vector_u: np.ndarray      # East-West flow velocity component [rows, cols]
    flow_vector_v: np.ndarray      # North-South flow velocity component [rows, cols]
    rows: int
    cols: int
    cell_size_m: float
    base_lat: float
    base_lon: float
    seed: int

    def to_dataframe(self) -> pd.DataFrame:
        """
        Exports the terrain grid cells to a flattened pandas DataFrame.
        Coordinate convention: x = col (West->East), y = row (North->South).
        """
        rows_idx, cols_idx = np.indices((self.rows, self.cols))
        rows_flat = rows_idx.ravel()
        cols_flat = cols_idx.ravel()

        lats = np.empty(self.rows * self.cols, dtype=float)
        lons = np.empty(self.rows * self.cols, dtype=float)

        for i, (r, c) in enumerate(zip(rows_flat, cols_flat)):
            coords = cell_to_latlon(
                int(r), int(c),
                self.base_lat, self.base_lon,
                self.rows, self.cols, self.cell_size_m
            )
            if coords is not None:
                lats[i], lons[i] = coords
            else:
                lats[i], lons[i] = np.nan, np.nan

        return pd.DataFrame({
            "x": cols_flat,
            "y": rows_flat,
            "lat": lats,
            "lon": lons,
            "elevation": self.elevation.ravel(),
            "slope": self.slope.ravel(),
            "soil_type": self.soil_type.ravel(),
            "moisture": self.moisture.ravel(),
            "water_depth": self.water_depth.ravel(),
            "riverbed_firmness": self.riverbed_firmness.ravel(),
            "river_velocity": self.river_velocity.ravel(),
        })


def compute_slope(elevation: np.ndarray, cell_size_m: float = DEFAULT_CELL_SIZE_M) -> np.ndarray:
    """
    Calculates slope in degrees across a 2D elevation grid using central differences.
    Formula: degrees(arctan(hypot(dz/dx, dz/dy)))
    """
    dz_dy, dz_dx = np.gradient(elevation, cell_size_m)
    gradient_magnitude = np.hypot(dz_dx, dz_dy)
    slope_deg = np.degrees(np.arctan(gradient_magnitude))
    return slope_deg


def generate_terrain(
    rows: int = DEFAULT_GRID_ROWS,
    cols: int = DEFAULT_GRID_COLS,
    cell_size_m: float = DEFAULT_CELL_SIZE_M,
    base_lat: float = DEFAULT_BASE_LAT,
    base_lon: float = DEFAULT_BASE_LON,
    seed: int = DEFAULT_SEED,
) -> TerrainGrid:
    """
    Generates a deterministic synthetic terrain grid combining satellite DEM features,
    surface moisture, soil classifications, AUV river bathymetry, and 2D hydrodynamic currents.
    """
    rng = np.random.default_rng(seed)

    # 1. Multi-octave DEM synthesis with gaussian hills
    base_elevation = 420.0
    octave1 = gaussian_filter(rng.standard_normal((rows, cols)), sigma=18.0) * 45.0
    octave2 = gaussian_filter(rng.standard_normal((rows, cols)), sigma=8.0) * 18.0
    octave3 = gaussian_filter(rng.standard_normal((rows, cols)), sigma=3.0) * 5.0
    raw_elevation = base_elevation + octave1 + octave2 + octave3

    # Add 3 prominent gaussian topographic features (tactical vantage hills / ridges)
    y_coords, x_coords = np.indices((rows, cols))
    hill_centers = [(rows * 0.25, cols * 0.2), (rows * 0.75, cols * 0.8), (rows * 0.15, cols * 0.85)]
    for cy, cx in hill_centers:
        dist_sq = (x_coords - cx) ** 2 + (y_coords - cy) ** 2
        raw_elevation += 75.0 * np.exp(-dist_sq / (2.0 * (10.0 ** 2)))

    # 2. Continuous Meandering River from North (row 0) to South (row N-1)
    row_indices = np.arange(rows)
    freq = 1.3
    center_cols = (cols / 2.0) + 12.0 * np.sin(2.0 * np.pi * freq * row_indices / rows) + \
                  4.0 * np.cos(4.0 * np.pi * freq * row_indices / rows)

    half_widths = 3.2 + 1.2 * np.sin(np.pi * row_indices / rows)

    river_mask = np.zeros((rows, cols), dtype=bool)
    norm_dist_to_center = np.ones((rows, cols), dtype=float)

    for r in range(rows):
        cc = center_cols[r]
        hw = half_widths[r]
        d = np.abs(x_coords[r, :] - cc)
        mask_r = d <= hw
        river_mask[r, :] = mask_r
        norm_dist_to_center[r, mask_r] = d[mask_r] / hw

    # Carve river channel into DEM: sloping banks down by up to 12 meters
    carve_depth = 12.0 * (1.0 - norm_dist_to_center) * river_mask
    elevation = raw_elevation - carve_depth
    elevation = gaussian_filter(elevation, sigma=0.8)

    # Compute exact slope
    slope = compute_slope(elevation, cell_size_m)

    # 3. AUV Hydrographic Bathymetry & Riverbed Firmness
    water_depth = np.zeros((rows, cols), dtype=float)
    riverbed_firmness = np.full((rows, cols), np.nan, dtype=float)

    sonar_noise = rng.uniform(-0.05, 0.05, size=(rows, cols))

    is_ford_section = (row_indices >= 46) & (row_indices <= 52)
    is_gravel_reach = (row_indices >= 20) & (row_indices <= 30)

    for r in range(rows):
        if not np.any(river_mask[r]):
            continue
        c_mask = river_mask[r]
        d_norm = norm_dist_to_center[r, c_mask]

        if is_ford_section[r]:
            depth_vals = 0.35 + 0.12 * (1.0 - d_norm) + sonar_noise[r, c_mask]
            depth_vals = np.clip(depth_vals, 0.25, 0.52)
            firmness_vals = 0.86 + 0.08 * (1.0 - d_norm) + rng.uniform(-0.02, 0.02, size=len(depth_vals))
            firmness_vals = np.clip(firmness_vals, 0.80, 0.98)
        elif is_gravel_reach[r]:
            depth_vals = 0.85 + 0.30 * (1.0 - d_norm) + sonar_noise[r, c_mask]
            depth_vals = np.clip(depth_vals, 0.80, 1.20)
            firmness_vals = 0.82 + 0.06 * (1.0 - d_norm) + rng.uniform(-0.02, 0.02, size=len(depth_vals))
            firmness_vals = np.clip(firmness_vals, 0.77, 0.92)
        else:
            depth_vals = 0.6 + 2.8 * (1.0 - d_norm) + sonar_noise[r, c_mask]
            depth_vals = np.clip(depth_vals, 0.20, 3.50)
            firmness_vals = 0.65 - 0.35 * (1.0 - d_norm) + rng.uniform(-0.05, 0.05, size=len(depth_vals))
            firmness_vals = np.clip(firmness_vals, 0.15, 0.70)

        water_depth[r, c_mask] = depth_vals
        riverbed_firmness[r, c_mask] = firmness_vals

    # 4. 2D Hydrodynamic River Velocity & Flow Vectors
    # Based on open-channel hydraulics: velocity scales with water depth^(2/3) and meander curvature
    river_velocity = np.zeros((rows, cols), dtype=float)
    flow_u = np.zeros((rows, cols), dtype=float)
    flow_v = np.zeros((rows, cols), dtype=float)

    # Tangent vector along centerline (downstream North to South)
    dc_dr = np.gradient(center_cols)
    tangent_mag = np.hypot(dc_dr, 1.0)
    unit_u = dc_dr / tangent_mag
    unit_v = 1.0 / tangent_mag

    for r in range(rows):
        if not np.any(river_mask[r]):
            continue
        c_mask = river_mask[r]
        d_center = 1.0 - norm_dist_to_center[r, c_mask]
        depth_ratio = water_depth[r, c_mask] / 3.5

        # Parabolic cross-channel velocity + depth scaling (Manning-inspired)
        # Shallow banks: ~0.3 m/s, Deep center: up to 2.4 m/s
        local_speed = (0.3 + 2.1 * (depth_ratio ** 0.67) * (d_center ** 0.5))
        if is_ford_section[r]:
            local_speed *= 0.45  # Shallow wide sandbar decelerates flow

        river_velocity[r, c_mask] = local_speed
        flow_u[r, c_mask] = local_speed * unit_u[r]
        flow_v[r, c_mask] = local_speed * unit_v[r]

    # 5. Spatially Coherent Soil Classification
    dist_from_river = distance_transform_edt(~river_mask)

    norm_elev = (elevation - np.min(elevation)) / (np.max(elevation) - np.min(elevation) + 1e-6)
    soil_potential = (
        0.55 * norm_elev +
        0.35 * np.clip(dist_from_river / 30.0, 0.0, 1.0) +
        gaussian_filter(rng.standard_normal((rows, cols)), sigma=4.0) * 0.15
    )

    soil_type = np.full((rows, cols), SoilType.SANDY_LOAM, dtype=int)

    is_approach_ramp = np.zeros((rows, cols), dtype=bool)
    is_approach_ramp[(row_indices >= 20) & (row_indices <= 30), :] = True
    is_approach_ramp[(row_indices >= 45) & (row_indices <= 53), :] = True

    marsh_mask = (dist_from_river <= 4.0) & (norm_elev < 0.40) & (~river_mask) & (~is_approach_ramp)
    soil_type[marsh_mask] = SoilType.MARSH_WETLAND
    clay_mask = (~marsh_mask) & (~river_mask) & (soil_potential < 0.38)
    soil_type[clay_mask] = SoilType.SOFT_CLAY
    loam_mask = (~marsh_mask) & (~river_mask) & (soil_potential >= 0.38) & (soil_potential < 0.68)
    soil_type[loam_mask] = SoilType.SANDY_LOAM
    hard_mask = (~marsh_mask) & (~river_mask) & (soil_potential >= 0.68)
    soil_type[hard_mask] = SoilType.HARD_GROUND

    # 6. Moisture field [0.0, 1.0]
    proximity_factor = np.exp(-dist_from_river / 12.0)
    elevation_inversion = 1.0 - norm_elev
    soil_moisture_bias = np.zeros((rows, cols), dtype=float)
    soil_moisture_bias[soil_type == SoilType.MARSH_WETLAND] = 0.35
    soil_moisture_bias[soil_type == SoilType.SOFT_CLAY] = 0.20
    soil_moisture_bias[soil_type == SoilType.SANDY_LOAM] = 0.05
    soil_moisture_bias[soil_type == SoilType.HARD_GROUND] = -0.10

    raw_moisture = 0.45 * proximity_factor + 0.30 * elevation_inversion + soil_moisture_bias + 0.15
    raw_moisture = gaussian_filter(raw_moisture, sigma=2.0)
    raw_moisture[river_mask] = 1.0
    moisture = np.clip(raw_moisture, 0.0, 1.0)

    return TerrainGrid(
        elevation=elevation,
        slope=slope,
        soil_type=soil_type,
        moisture=moisture,
        river_mask=river_mask,
        water_depth=water_depth,
        riverbed_firmness=riverbed_firmness,
        river_velocity=river_velocity,
        flow_vector_u=flow_u,
        flow_vector_v=flow_v,
        rows=rows,
        cols=cols,
        cell_size_m=cell_size_m,
        base_lat=base_lat,
        base_lon=base_lon,
        seed=seed,
    )
