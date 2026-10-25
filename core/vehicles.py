"""Vehicle specifications and mobility profiles for military terrain assessment."""

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Dict, Any


@dataclass(frozen=True)
class VehicleProfile:
    """
    Tactical mobility specifications of a military vehicle platform.
    
    Attributes:
        name: Designation of the vehicle platform.
        weight_class: Operational gross vehicle weight tier ('Light', 'Medium', 'Heavy').
        max_slope_deg: Maximum climb/descent slope angle before track slip or rollover.
        max_fording_m: Nominal static water wading depth limit before engine hydrolock.
        min_firmness: Minimum required substrate bearing firmness [0.0, 1.0].
        speed_kmph_go: Nominal dash/cruising speed on GO terrain (km/h).
        speed_kmph_slowgo: Controlled tactical maneuvering speed on SLOW-GO terrain (km/h).
    """
    name: str
    weight_class: str
    max_slope_deg: float
    max_fording_m: float
    min_firmness: float
    speed_kmph_go: float
    speed_kmph_slowgo: float

    def with_overrides(self, **kwargs: Any) -> VehicleProfile:
        """Returns a modified immutable copy of the vehicle profile with user overrides."""
        return replace(self, **kwargs)


# NATO-standard vehicle mobility presets
VEHICLE_PRESETS: Dict[str, VehicleProfile] = {
    "Light Recon 4x4 Jeep": VehicleProfile(
        name="Light Recon 4x4 Jeep",
        weight_class="Light",
        max_slope_deg=20.0,
        max_fording_m=0.60,
        min_firmness=0.50,
        speed_kmph_go=60.0,
        speed_kmph_slowgo=25.0,
    ),
    "Medium Tactical 6x6 Truck": VehicleProfile(
        name="Medium Tactical 6x6 Truck",
        weight_class="Medium",
        max_slope_deg=25.0,
        max_fording_m=1.00,
        min_firmness=0.60,
        speed_kmph_go=45.0,
        speed_kmph_slowgo=18.0,
    ),
    "Heavy Tracked Tank": VehicleProfile(
        name="Heavy Tracked Tank",
        weight_class="Heavy",
        max_slope_deg=30.0,
        max_fording_m=1.40,
        min_firmness=0.75,
        speed_kmph_go=35.0,
        speed_kmph_slowgo=12.0,
    ),
}

DEFAULT_VEHICLE_NAME = "Medium Tactical 6x6 Truck"
