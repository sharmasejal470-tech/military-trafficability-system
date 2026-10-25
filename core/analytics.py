"""Tactical Mission Analytics and Hydrographic Crossing Assessment Engine.

Computes route distance, travel ETA, slope and water depth extremes, substrate
bearing safety margins, and longitudinal elevation/riverbed profile cross-sections.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

from core.config import (
    TrafficabilityClass,
    CROSSING_DEPTH_MARGIN_SAFE,
    CROSSING_DEPTH_MARGIN_CAUTION,
    CROSSING_FIRMNESS_MARGIN_SAFE,
    DEFAULT_CELL_SIZE_M,
    SQRT2,
)
from core.terrain import TerrainGrid
from core.vehicles import VehicleProfile
from core.trafficability import TrafficabilityResult
from core.pathfinding import PathResult


class CrossingSafetyStatus(str, Enum):
    """Hydrographic fording safety classification."""
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    UNSAFE = "UNSAFE"
    NO_CROSSING = "NO_CROSSING"


@dataclass(frozen=True)
class RouteMetrics:
    """Quantitative performance and risk metrics along a computed tactical corridor."""
    distance_km: float
    eta_minutes: float
    eta_hours: float
    pct_go: float
    pct_slowgo: float
    max_slope_deg: float
    river_cells_crossed: int
    max_water_depth_m: float
    min_riverbed_firmness: Optional[float]
    crossing_status: CrossingSafetyStatus
    crossing_reason: str
    waypoint_count: int


@dataclass(frozen=True)
class ProfileData:
    """Longitudinal terrain and hydrographic profile along the route."""
    cumulative_distance_km: np.ndarray
    elevation_m: np.ndarray
    water_surface_m: np.ndarray        # Surface elevation of water or NaN
    riverbed_level_m: np.ndarray       # Bed level: elevation - water_depth
    slope_deg: np.ndarray
    trafficability: np.ndarray         # Int array of Go/Slow-Go
    is_river: np.ndarray               # Boolean array of river cells

    def to_dataframe(self) -> pd.DataFrame:
        """Returns profile data as a pandas DataFrame."""
        return pd.DataFrame({
            "cumulative_distance_km": self.cumulative_distance_km,
            "elevation_m": self.elevation_m,
            "water_surface_m": self.water_surface_m,
            "riverbed_level_m": self.riverbed_level_m,
            "slope_deg": self.slope_deg,
            "trafficability": self.trafficability,
            "is_river": self.is_river,
        })


def assess_river_crossing(
    river_cells_crossed: int,
    max_water_depth: float,
    min_firmness: Optional[float],
    effective_max_fording: float,
    vehicle_min_firmness: float,
) -> Tuple[CrossingSafetyStatus, str]:
    """
    Evaluates hydrographic fording risk based on depth and firmness clearances.
    
    Thresholds:
    - SAFE: depth_margin >= 0.25m AND firmness_margin >= 0.10
    - CAUTION: depth_margin in [0.05m, 0.25m) OR firmness_margin in [0.0, 0.10)
    - UNSAFE: depth_margin < 0.05m OR firmness_margin < 0.0
    """
    if river_cells_crossed == 0:
        return CrossingSafetyStatus.NO_CROSSING, "No river crossing required along this tactical corridor."

    depth_margin = effective_max_fording - max_water_depth
    firmness_margin = (min_firmness - vehicle_min_firmness) if min_firmness is not None else 0.0

    if depth_margin < CROSSING_DEPTH_MARGIN_CAUTION:
        reason = f"UNSAFE: Water depth ({max_water_depth:.2f}m) exceeds or matches fording limit ({effective_max_fording:.2f}m)."
        return CrossingSafetyStatus.UNSAFE, reason

    if firmness_margin < 0.0:
        reason = f"UNSAFE: Substrate firmness ({min_firmness:.2f}) below vehicle minimum ({vehicle_min_firmness:.2f}). Track miring imminent."
        return CrossingSafetyStatus.UNSAFE, reason

    if depth_margin < CROSSING_DEPTH_MARGIN_SAFE or firmness_margin < CROSSING_FIRMNESS_MARGIN_SAFE:
        reason = (
            f"CAUTION: Marginal fording clearance. Depth margin: {depth_margin:.2f}m, "
            f"Firmness buffer: {firmness_margin:.2f} above minimum."
        )
        return CrossingSafetyStatus.CAUTION, reason

    reason = (
        f"SAFE: Fording clearance is {depth_margin:.2f}m and substrate firmness buffer is "
        f"{firmness_margin:.2f} above minimum requirement."
    )
    return CrossingSafetyStatus.SAFE, reason


def compute_route_analytics(
    terrain: TerrainGrid,
    trafficability: TrafficabilityResult,
    path_result: PathResult,
    is_xy: bool = False,
    cell_size_m: float = DEFAULT_CELL_SIZE_M,
) -> Tuple[Optional[RouteMetrics], Optional[ProfileData]]:
    """
    Computes all mission metrics and cross-sectional profile arrays for a planned path.
    Returns (None, None) if path_result is empty or failed.
    """
    if not path_result.path_cells or len(path_result.path_cells) < 1:
        return None, None

    # Standardize path coordinates to [row, col]
    if is_xy:
        path_rc = [(r, c) for c, r in path_result.path_cells]
    else:
        path_rc = path_result.path_cells

    vehicle = trafficability.vehicle
    n_points = len(path_rc)

    # Initialize accumulation arrays
    cum_dist_m = np.zeros(n_points, dtype=float)
    elevations = np.zeros(n_points, dtype=float)
    slopes = np.zeros(n_points, dtype=float)
    water_depths = np.zeros(n_points, dtype=float)
    firmness_vals = np.zeros(n_points, dtype=float)
    is_river = np.zeros(n_points, dtype=bool)
    cell_traff = np.zeros(n_points, dtype=int)

    total_time_hours = 0.0
    river_firmness_list: List[float] = []

    for i in range(n_points):
        r, c = path_rc[i]
        elevations[i] = terrain.elevation[r, c]
        slopes[i] = terrain.slope[r, c]
        water_depths[i] = terrain.water_depth[r, c]
        firmness_vals[i] = trafficability.effective_firmness[r, c]
        is_river[i] = terrain.river_mask[r, c]
        cell_traff[i] = trafficability.grid[r, c]

        if is_river[i]:
            if not np.isnan(trafficability.effective_firmness[r, c]):
                river_firmness_list.append(trafficability.effective_firmness[r, c])

        if i > 0:
            pr, pc = path_rc[i - 1]
            dr = abs(r - pr)
            dc = abs(c - pc)
            is_diag = (dr > 0 and dc > 0)
            step_m = (SQRT2 * cell_size_m) if is_diag else cell_size_m
            cum_dist_m[i] = cum_dist_m[i - 1] + step_m

            # Speed calculation for this step based on destination cell condition
            step_dist_km = step_m / 1000.0
            if cell_traff[i] == TrafficabilityClass.SLOW_GO:
                speed = vehicle.speed_kmph_slowgo
            else:
                speed = vehicle.speed_kmph_go
            total_time_hours += step_dist_km / speed

    total_dist_km = float(cum_dist_m[-1] / 1000.0)
    eta_minutes = float(total_time_hours * 60.0)

    # Percentage classifications
    pct_go = float((cell_traff == TrafficabilityClass.GO).mean() * 100.0)
    pct_slowgo = float((cell_traff == TrafficabilityClass.SLOW_GO).mean() * 100.0)
    max_slope = float(np.max(slopes)) if n_points > 0 else 0.0
    river_cells = int(is_river.sum())
    max_water = float(np.max(water_depths)) if n_points > 0 else 0.0
    min_firmness = float(min(river_firmness_list)) if river_firmness_list else None

    # Crossing evaluation
    crossing_status, crossing_reason = assess_river_crossing(
        river_cells_crossed=river_cells,
        max_water_depth=max_water,
        min_firmness=min_firmness,
        effective_max_fording=trafficability.effective_max_fording,
        vehicle_min_firmness=vehicle.min_firmness,
    )

    metrics = RouteMetrics(
        distance_km=total_dist_km,
        eta_minutes=eta_minutes,
        eta_hours=total_time_hours,
        pct_go=pct_go,
        pct_slowgo=pct_slowgo,
        max_slope_deg=max_slope,
        river_cells_crossed=river_cells,
        max_water_depth_m=max_water,
        min_riverbed_firmness=min_firmness,
        crossing_status=crossing_status,
        crossing_reason=crossing_reason,
        waypoint_count=n_points,
    )

    # Water surface level and riverbed profile calculations
    water_surface = np.full(n_points, np.nan, dtype=float)
    water_surface[is_river] = elevations[is_river]
    riverbed_level = elevations - water_depths

    profile = ProfileData(
        cumulative_distance_km=cum_dist_m / 1000.0,
        elevation_m=elevations,
        water_surface_m=water_surface,
        riverbed_level_m=riverbed_level,
        slope_deg=slopes,
        trafficability=cell_traff,
        is_river=is_river,
    )

    return metrics, profile
