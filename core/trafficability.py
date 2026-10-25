"""Automated Trafficability Prediction Engine.

Implements NATO-standard terramechanics and 2D hydrodynamic flow mobility classification
fusing multi-spectral GIS terrain layers with AUV sonar riverbed measurements.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Union
import numpy as np

from core.config import (
    TrafficabilityClass,
    MobilityReason,
    SoilType,
    DEFAULT_CURRENT_FACTOR_K,
    SLOW_GO_SLOPE_THRESHOLD_RATIO,
    SLOW_GO_WATER_DEPTH_THRESHOLD_RATIO,
)
from core.terrain import TerrainGrid
from core.vehicles import VehicleProfile


@dataclass(frozen=True)
class TrafficabilityResult:
    """
    Evaluated mobility surface for a specific vehicle platform and environmental state.
    """
    grid: np.ndarray                  # 2D int array: 0=Go, 1=Slow-Go, 2=No-Go [rows, cols]
    reasons: np.ndarray               # 2D int bitmask array of MobilityReason [rows, cols]
    vehicle: VehicleProfile
    rainfall_multiplier: float
    current_factor: float
    effective_max_fording: float      # Representative maximum safe fording threshold
    effective_firmness: np.ndarray    # 2D array of degraded substrate firmness
    effective_fording_field: np.ndarray # 2D array of spatially distributed hydrodynamic fording limits


def compute_trafficability(
    terrain: TerrainGrid,
    vehicle: VehicleProfile,
    rainfall_multiplier: float = 1.0,
    current_factor: float = 0.0,
    bridge_cells: Union[None, list[Tuple[int, int]]] = None,
) -> TrafficabilityResult:
    """
    Classifies every grid cell into Go (0), Slow-Go (1), or No-Go (2) with diagnostic reason codes.
    
    Incorporates:
    - 2D hydrodynamic river current velocity field
    - Terzaghi pore-water pressure substrate degradation
    - AVLB tactical bridgehead deployment across designated bridge cells
    """
    rainfall = float(np.clip(rainfall_multiplier, 0.5, 2.0))
    current = float(np.clip(current_factor, 0.0, 1.0))

    rows, cols = terrain.rows, terrain.cols

    # 1. Spatially distributed 2D Hydrodynamic Fording Limits
    # Representative baseline fording limit
    rep_effective_max_fording = float(vehicle.max_fording_m * (1.0 - DEFAULT_CURRENT_FACTOR_K * current))

    # Scales drag penalty with local river flow velocity (m/s)
    # Higher flow velocity in deep thalweg induces additional hydrodynamic stagnation drag
    if hasattr(terrain, "river_velocity") and terrain.river_velocity is not None:
        local_vel_ratio = np.clip(terrain.river_velocity / 2.0, 0.0, 1.5)
    else:
        local_vel_ratio = np.ones((rows, cols), dtype=float)

    # Effective fording limit per cell: fast currents depress the allowable wading depth
    effective_fording_field = rep_effective_max_fording / (1.0 + 0.35 * current * local_vel_ratio)

    # 2. Environmental degradation of substrate and hydrography
    firmness_degradation = 1.0 + 0.30 * (rainfall - 1.0)
    effective_firmness = np.clip(terrain.riverbed_firmness / firmness_degradation, 0.0, 1.0)

    # Precipitation runoff expands effective channel depth
    depth_swelling = 1.0 + 0.15 * (rainfall - 1.0)
    effective_water_depth = terrain.water_depth * depth_swelling

    grid = np.full((rows, cols), TrafficabilityClass.GO, dtype=int)
    reasons = np.zeros((rows, cols), dtype=int)

    is_heavy = bool(vehicle.weight_class.lower() == "heavy")
    is_medium = bool(vehicle.weight_class.lower() == "medium")

    # 3. Vectorized NO-GO (RED) Evaluation
    red_slope = terrain.slope > vehicle.max_slope_deg
    red_water = effective_water_depth > effective_fording_field
    red_firmness = terrain.river_mask & (effective_firmness < vehicle.min_firmness)

    # Precipitation runoff and soil liquefaction (Monsoon mud hazards)
    excess_rain = max(0.0, rainfall - 1.0)
    eff_moisture = np.clip(terrain.moisture + 0.35 * excess_rain, 0.0, 1.0)

    if is_heavy:
        red_marsh = terrain.soil_type == SoilType.MARSH_WETLAND
        yellow_marsh = np.zeros((rows, cols), dtype=bool)
        red_mud = (terrain.soil_type == SoilType.SOFT_CLAY) & (excess_rain > 0.15) & (eff_moisture >= 0.85)
    elif is_medium:
        red_marsh = (terrain.soil_type == SoilType.MARSH_WETLAND) & (excess_rain > 0.25) & (eff_moisture >= 0.88)
        yellow_marsh = (terrain.soil_type == SoilType.MARSH_WETLAND) & (~red_marsh)
        red_mud = (terrain.soil_type == SoilType.SOFT_CLAY) & (excess_rain > 0.35) & (eff_moisture >= 0.92)
    else:
        red_marsh = (terrain.soil_type == SoilType.MARSH_WETLAND) & (excess_rain > 0.25) & (eff_moisture >= 0.88)
        yellow_marsh = (terrain.soil_type == SoilType.MARSH_WETLAND) & (~red_marsh)
        red_mud = (terrain.soil_type == SoilType.SOFT_CLAY) & (excess_rain > 0.40) & (eff_moisture >= 0.93)

    red_mask = red_slope | red_water | red_firmness | red_marsh | red_mud

    # 4. Vectorized SLOW-GO (YELLOW) Evaluation
    yellow_slope_thresh = SLOW_GO_SLOPE_THRESHOLD_RATIO * vehicle.max_slope_deg
    yellow_slope = (terrain.slope >= yellow_slope_thresh) & (terrain.slope <= vehicle.max_slope_deg)

    yellow_water_thresh = SLOW_GO_WATER_DEPTH_THRESHOLD_RATIO * effective_fording_field
    yellow_water = (
        (effective_water_depth >= yellow_water_thresh) &
        (effective_water_depth <= effective_fording_field) &
        (effective_water_depth > 0.0)
    )

    yellow_soil = (
        (terrain.soil_type == SoilType.SOFT_CLAY) |
        (terrain.soil_type == SoilType.SANDY_LOAM)
    )

    yellow_mask = (yellow_slope | yellow_water | yellow_soil | yellow_marsh) & (~red_mask)

    # 5. Populate Result Grid & Bitmask Reasons
    grid[yellow_mask] = TrafficabilityClass.SLOW_GO
    grid[red_mask] = TrafficabilityClass.NO_GO

    reasons[red_slope] |= MobilityReason.NO_GO_SLOPE
    reasons[red_water] |= MobilityReason.NO_GO_WATER_DEPTH
    reasons[red_firmness] |= MobilityReason.NO_GO_FIRMNESS
    reasons[red_marsh] |= MobilityReason.NO_GO_HEAVY_ON_MARSH
    reasons[red_mud] |= MobilityReason.NO_GO_FIRMNESS

    reasons[yellow_slope & ~red_mask] |= MobilityReason.SLOW_GO_SLOPE
    reasons[yellow_water & ~red_mask] |= MobilityReason.SLOW_GO_WATER_DEPTH
    reasons[yellow_soil & ~red_mask] |= MobilityReason.SLOW_GO_SOIL
    reasons[yellow_marsh & ~red_mask] |= MobilityReason.SLOW_GO_MARSH

    # 6. Apply AVLB Tactical Bridgehead (if deployed)
    # Converts bridgehead crossing cells into stable GO corridor
    if bridge_cells is not None:
        for br, bc in bridge_cells:
            if 0 <= br < rows and 0 <= bc < cols:
                grid[br, bc] = TrafficabilityClass.GO
                reasons[br, bc] = MobilityReason.NONE

    return TrafficabilityResult(
        grid=grid,
        reasons=reasons,
        vehicle=vehicle,
        rainfall_multiplier=rainfall,
        current_factor=current,
        effective_max_fording=rep_effective_max_fording,
        effective_firmness=effective_firmness,
        effective_fording_field=effective_fording_field,
    )
