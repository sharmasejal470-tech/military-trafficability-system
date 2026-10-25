"""Automated Trafficability Prediction Engine.

Implements NATO-standard terramechanics and hydrographic mobility classification
fusing multi-spectral GIS terrain layers with AUV sonar riverbed measurements.

Defense & Mobility Context:
- Slope Limits: Exceeding maximum gradient causes catastrophic track slip, ground clearance loss,
  transmission overheating, or lateral vehicle rollover (pitch/roll stability envelopes).
- Fording Limits: Exceeding static fording depth causes engine air intake hydrolock, electronics
  flooding, and amphibious buoyancy loss. Hydrodynamic river currents exert hydrodynamic drag forces
  (Fd = 0.5 * Cd * rho * v^2 * A), reducing the allowable safe fording depth.
- Riverbed Firmness: Heavy combat vehicles (e.g., 60-ton MBTs) generate ground contact pressures
  exceeding 100 kPa. Inadequate sub-aquatic substrate bearing capacity leads to instantaneous track
  sinkage, belly hang-up, and riverbed miring.
- Saturated Soils / Marshes: Marshes exhibit near-zero cohesion and high plasticity indices.
  Heavy tracked armor experiences complete bearing capacity failure (shear failure) in wetlands,
  requiring combat engineering bridge-laying assets.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
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
    grid: np.ndarray             # 2D int array: 0=Go, 1=Slow-Go, 2=No-Go [rows, cols]
    reasons: np.ndarray          # 2D int bitmask array of MobilityReason [rows, cols]
    vehicle: VehicleProfile
    rainfall_multiplier: float
    current_factor: float
    effective_max_fording: float
    effective_firmness: np.ndarray


def compute_trafficability(
    terrain: TerrainGrid,
    vehicle: VehicleProfile,
    rainfall_multiplier: float = 1.0,
    current_factor: float = 0.0,
) -> TrafficabilityResult:
    """
    Classifies every grid cell into Go (0), Slow-Go (1), or No-Go (2) with diagnostic reason codes.
    
    All calculations are vectorized across 2D NumPy arrays without per-cell Python iteration.
    
    Environmental Modifiers:
    1. River Current:
       effective_max_fording = max_fording * (1.0 - k * current_factor)
       where k = DEFAULT_CURRENT_FACTOR_K (0.40).
    2. Rainfall Multiplier (0.5 to 2.0):
       - Degrades effective riverbed firmness via increased pore pressure:
         effective_firmness = riverbed_firmness / (1.0 + 0.30 * (rainfall - 1.0))
       - Swells effective water depth in river channels:
         effective_water_depth = water_depth * (1.0 + 0.15 * (rainfall - 1.0))
       - Monotonically preserves or worsens mobility hazards (never makes a cell safer at higher rain).
    
    Classification Logic (RED ALWAYS OVERRIDES YELLOW):
    - RED (No-Go):
      * slope > max_slope_deg
      * water_depth > effective_max_fording
      * river_mask AND (effective_firmness < min_firmness)
      * Marsh/Wetland AND weight_class == 'Heavy'
    - YELLOW (Slow-Go):
      * slope in [0.75 * max_slope_deg, max_slope_deg]
      * water_depth in [0.60 * effective_max_fording, effective_max_fording]
      * soil_type in (Soft Clay, Sandy Loam)
      * Marsh/Wetland AND weight_class != 'Heavy'
    - GREEN (Go):
      * All remaining unrestricted terrain cells.
    """
    # Clamp environmental modifiers within operational boundaries
    rainfall = float(np.clip(rainfall_multiplier, 0.5, 2.0))
    current = float(np.clip(current_factor, 0.0, 1.0))

    # 1. Effective hydrodynamic fording depth
    effective_max_fording = float(vehicle.max_fording_m * (1.0 - DEFAULT_CURRENT_FACTOR_K * current))

    # 2. Environmental degradation of substrate and hydrography
    firmness_degradation = 1.0 + 0.30 * (rainfall - 1.0)
    effective_firmness = np.clip(terrain.riverbed_firmness / firmness_degradation, 0.0, 1.0)

    # Precipitation runoff expands effective channel depth
    depth_swelling = 1.0 + 0.15 * (rainfall - 1.0)
    effective_water_depth = terrain.water_depth * depth_swelling

    rows, cols = terrain.rows, terrain.cols
    grid = np.full((rows, cols), TrafficabilityClass.GO, dtype=int)
    reasons = np.zeros((rows, cols), dtype=int)

    is_heavy = bool(vehicle.weight_class.lower() == "heavy")

    # 3. Vectorized NO-GO (RED) Evaluation
    red_slope = terrain.slope > vehicle.max_slope_deg
    red_water = effective_water_depth > effective_max_fording
    # Inside river: firmness must meet vehicle minimum bearing requirement
    red_firmness = terrain.river_mask & (effective_firmness < vehicle.min_firmness)
    # Heavy armored vehicles cannot traverse wetlands without catastrophic belly sinkage
    if is_heavy:
        red_marsh = terrain.soil_type == SoilType.MARSH_WETLAND
        yellow_marsh = np.zeros((rows, cols), dtype=bool)
    else:
        red_marsh = np.zeros((rows, cols), dtype=bool)
        yellow_marsh = terrain.soil_type == SoilType.MARSH_WETLAND

    red_mask = red_slope | red_water | red_firmness | red_marsh

    # 4. Vectorized SLOW-GO (YELLOW) Evaluation
    yellow_slope_thresh = SLOW_GO_SLOPE_THRESHOLD_RATIO * vehicle.max_slope_deg
    yellow_slope = (terrain.slope >= yellow_slope_thresh) & (terrain.slope <= vehicle.max_slope_deg)

    yellow_water_thresh = SLOW_GO_WATER_DEPTH_THRESHOLD_RATIO * effective_max_fording
    yellow_water = (
        (effective_water_depth >= yellow_water_thresh) &
        (effective_water_depth <= effective_max_fording) &
        (effective_water_depth > 0.0)
    )

    yellow_soil = (
        (terrain.soil_type == SoilType.SOFT_CLAY) |
        (terrain.soil_type == SoilType.SANDY_LOAM)
    )

    # Yellow condition is overridden whenever red_mask is true
    yellow_mask = (yellow_slope | yellow_water | yellow_soil | yellow_marsh) & (~red_mask)

    # 5. Populate Result Grid & Bitmask Reasons
    grid[yellow_mask] = TrafficabilityClass.SLOW_GO
    grid[red_mask] = TrafficabilityClass.NO_GO

    # Tag specific reason codes for tactical intelligence explanation
    reasons[red_slope] |= MobilityReason.NO_GO_SLOPE
    reasons[red_water] |= MobilityReason.NO_GO_WATER_DEPTH
    reasons[red_firmness] |= MobilityReason.NO_GO_FIRMNESS
    reasons[red_marsh] |= MobilityReason.NO_GO_HEAVY_ON_MARSH

    # Tag yellow reasons only on cells that are not red
    reasons[yellow_slope & ~red_mask] |= MobilityReason.SLOW_GO_SLOPE
    reasons[yellow_water & ~red_mask] |= MobilityReason.SLOW_GO_WATER_DEPTH
    reasons[yellow_soil & ~red_mask] |= MobilityReason.SLOW_GO_SOIL
    reasons[yellow_marsh & ~red_mask] |= MobilityReason.SLOW_GO_MARSH

    return TrafficabilityResult(
        grid=grid,
        reasons=reasons,
        vehicle=vehicle,
        rainfall_multiplier=rainfall,
        current_factor=current,
        effective_max_fording=effective_max_fording,
        effective_firmness=effective_firmness,
    )
