"""Configuration constants, coordinate transformations, and data models."""

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from typing import Optional, Tuple
import numpy as np


# --- Grid & Spatial Constants ---
DEFAULT_GRID_ROWS: int = 100
DEFAULT_GRID_COLS: int = 100
DEFAULT_CELL_SIZE_M: float = 100.0  # 100 meters per cell (10 km x 10 km grid)
DEFAULT_BASE_LAT: float = 26.9124   # Jaipur, Rajasthan (Sapta Shakti Command HQ)
DEFAULT_BASE_LON: float = 75.7873
DEFAULT_SEED: int = 42

# WGS84 Meridional reference
METERS_PER_DEGREE_LAT: float = 111132.954


# --- Soil Classification ---
class SoilType(IntEnum):
    """Integer soil category codes."""
    HARD_GROUND = 0
    SANDY_LOAM = 1
    SOFT_CLAY = 2
    MARSH_WETLAND = 3


SOIL_LABELS = {
    SoilType.HARD_GROUND: "Hard Ground",
    SoilType.SANDY_LOAM: "Sandy Loam",
    SoilType.SOFT_CLAY: "Soft Clay",
    SoilType.MARSH_WETLAND: "Marsh / Wetland",
}


# --- Trafficability Classification ---
class TrafficabilityClass(IntEnum):
    """NATO mobility classification."""
    GO = 0
    SLOW_GO = 1
    NO_GO = 2


# Color hex definitions (consistent across Folium & Plotly)
COLOR_GO = "#2ecc71"       # Emerald Green
COLOR_SLOW_GO = "#f1c40f"  # Amber Yellow
COLOR_NO_GO = "#e74c3c"    # Crimson Red

RGBA_GO = (46, 204, 113, 160)
RGBA_SLOW_GO = (241, 196, 15, 160)
RGBA_NO_GO = (231, 76, 60, 190)
RGBA_TRANSPARENT = (0, 0, 0, 0)
RGBA_RIVER = (52, 152, 219, 140)


# --- Mobility Reason Codes (Bitmask) ---
class MobilityReason(IntFlag):
    """Diagnostic reasons for Slow-Go / No-Go classifications."""
    NONE = 0
    NO_GO_SLOPE = 1 << 0
    NO_GO_WATER_DEPTH = 1 << 1
    NO_GO_FIRMNESS = 1 << 2
    NO_GO_HEAVY_ON_MARSH = 1 << 3
    SLOW_GO_SLOPE = 1 << 4
    SLOW_GO_WATER_DEPTH = 1 << 5
    SLOW_GO_SOIL = 1 << 6
    SLOW_GO_MARSH = 1 << 7


# --- Mobility & Environment Constants ---
DEFAULT_CURRENT_FACTOR_K: float = 0.40  # Fording limit reduction factor: max_ford * (1 - k * current)
RAINFALL_MIN: float = 0.5
RAINFALL_MAX: float = 2.0
CURRENT_MIN: float = 0.0
CURRENT_MAX: float = 1.0

# Trafficability rule thresholds
SLOW_GO_SLOPE_THRESHOLD_RATIO: float = 0.75       # 75% of max slope
SLOW_GO_WATER_DEPTH_THRESHOLD_RATIO: float = 0.60  # 60% of effective fording limit

# Pathfinding Edge Multipliers
PATH_COST_GO: float = 1.0
PATH_COST_SLOW_GO: float = 5.0
SQRT2: float = 1.4142135623730951

# River crossing safety thresholds
CROSSING_DEPTH_MARGIN_SAFE: float = 0.25      # meters clearance above vehicle limit
CROSSING_DEPTH_MARGIN_CAUTION: float = 0.05   # meters clearance above vehicle limit
CROSSING_FIRMNESS_MARGIN_SAFE: float = 0.10   # firmness buffer above vehicle min_firmness


# --- Coordinate Transformation Functions ---

def get_grid_bounds(
    base_lat: float = DEFAULT_BASE_LAT,
    base_lon: float = DEFAULT_BASE_LON,
    rows: int = DEFAULT_GRID_ROWS,
    cols: int = DEFAULT_GRID_COLS,
    cell_size_m: float = DEFAULT_CELL_SIZE_M,
) -> Tuple[float, float, float, float]:
    """
    Computes (south, west, north, east) bounding box in degrees lat/lon.
    The grid is centered on (base_lat, base_lon).
    Row 0 represents the NORTH edge.
    Col 0 represents the WEST edge.
    """
    half_h_m = (rows * cell_size_m) / 2.0
    half_w_m = (cols * cell_size_m) / 2.0

    dlat = half_h_m / METERS_PER_DEGREE_LAT
    meters_per_deg_lon = METERS_PER_DEGREE_LAT * np.cos(np.radians(base_lat))
    dlon = half_w_m / meters_per_deg_lon

    north = base_lat + dlat
    south = base_lat - dlat
    west = base_lon - dlon
    east = base_lon + dlon
    return south, west, north, east


def cell_to_latlon(
    row: int,
    col: int,
    base_lat: float = DEFAULT_BASE_LAT,
    base_lon: float = DEFAULT_BASE_LON,
    rows: int = DEFAULT_GRID_ROWS,
    cols: int = DEFAULT_GRID_COLS,
    cell_size_m: float = DEFAULT_CELL_SIZE_M,
) -> Optional[Tuple[float, float]]:
    """
    Converts grid cell indices [row, col] to center (lat, lon) coordinates.
    Returns None if cell is out of bounds.
    """
    if row < 0 or row >= rows or col < 0 or col >= cols:
        return None

    south, west, north, east = get_grid_bounds(base_lat, base_lon, rows, cols, cell_size_m)
    lat_step = (north - south) / rows
    lon_step = (east - west) / cols

    # Row 0 is at the North boundary, descending southwards
    lat = north - (row + 0.5) * lat_step
    # Col 0 is at the West boundary, ascending eastwards
    lon = west + (col + 0.5) * lon_step
    return float(lat), float(lon)


def latlon_to_cell(
    lat: float,
    lon: float,
    base_lat: float = DEFAULT_BASE_LAT,
    base_lon: float = DEFAULT_BASE_LON,
    rows: int = DEFAULT_GRID_ROWS,
    cols: int = DEFAULT_GRID_COLS,
    cell_size_m: float = DEFAULT_CELL_SIZE_M,
) -> Optional[Tuple[int, int]]:
    """
    Converts geographic coordinates (lat, lon) to grid cell indices [row, col].
    Returns None if the coordinates lie outside the grid bounding box.
    """
    south, west, north, east = get_grid_bounds(base_lat, base_lon, rows, cols, cell_size_m)
    if not (south <= lat <= north and west <= lon <= east):
        return None

    lat_step = (north - south) / rows
    lon_step = (east - west) / cols

    row = int((north - lat) / lat_step)
    col = int((lon - west) / lon_step)

    # Clamp boundary edge floating errors
    row = min(max(row, 0), rows - 1)
    col = min(max(col, 0), cols - 1)
    return row, col
