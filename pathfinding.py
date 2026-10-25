"""A* Tactical Pathfinding Engine & Convoy Logistics Planner.

Computes least-risk, minimum-cost operational routes across 8-connected
trafficability grids avoiding Red (No-Go) impassable terrain and planning
echeloned multi-vehicle military convoys with tactical bridge-laying (AVLB).
"""

from __future__ import annotations
import heapq
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

from core.config import (
    TrafficabilityClass,
    PATH_COST_GO,
    PATH_COST_SLOW_GO,
    SQRT2,
    DEFAULT_CELL_SIZE_M,
    COLOR_CONVOY_RECON,
    COLOR_CONVOY_SPEARHEAD,
    COLOR_CONVOY_LOGISTICS,
)
from core.terrain import TerrainGrid
from core.vehicles import VehicleProfile, VEHICLE_PRESETS


class PathStatus(str, Enum):
    """Execution status for pathfinding queries."""
    OK = "OK"
    START_BLOCKED = "START_BLOCKED"
    GOAL_BLOCKED = "GOAL_BLOCKED"
    NO_PATH = "NO_PATH"
    SAME_CELL = "SAME_CELL"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"


@dataclass(frozen=True)
class PathResult:
    """
    Immutable outcome of an A* pathfinding calculation.
    """
    path_cells: List[Tuple[int, int]]
    total_cost: float
    status: PathStatus
    message: str

    @property
    def path_xy(self) -> List[Tuple[int, int]]:
        """Returns path as public API (x=col, y=row) coordinates."""
        return [(c, r) for r, c in self.path_cells]


@dataclass(frozen=True)
class ConvoyElementPlan:
    """Tactical trajectory and timing for a single convoy echelon element."""
    vehicle: VehicleProfile
    path_result: PathResult
    departure_delay_min: float
    transit_duration_min: float
    arrival_time_min: float
    color: str


@dataclass(frozen=True)
class ConvoyPlan:
    """Multi-vehicle echeloned movement and engineering plan."""
    elements: Dict[str, ConvoyElementPlan]
    bridge_deployed: bool
    bridge_cells: List[Tuple[int, int]]
    total_convoy_span_min: float
    status: str
    message: str


# 8-Directional neighbor offsets: (dr, dc, is_diagonal)
NEIGHBOR_OFFSETS: List[Tuple[int, int, bool]] = [
    (-1, 0, False),  # North
    (1, 0, False),   # South
    (0, -1, False),  # West
    (0, 1, False),   # East
    (-1, -1, True),  # North-West
    (-1, 1, True),   # North-East
    (1, -1, True),   # South-West
    (1, 1, True),    # South-East
]


def octile_heuristic(r1: int, c1: int, r2: int, c2: int, cell_size_m: float) -> float:
    """Computes admissible and consistent Octile distance heuristic for 8-connected grids."""
    dr = abs(r1 - r2)
    dc = abs(c1 - c2)
    min_d = min(dr, dc)
    max_d = max(dr, dc)
    return ((SQRT2 - 1.0) * min_d + max_d) * cell_size_m * PATH_COST_GO


def nearest_valid_cell(
    grid: np.ndarray,
    cell: Tuple[int, int],
    is_xy: bool = False,
) -> Optional[Tuple[int, int]]:
    """Breadth-First Search (BFS) search for the nearest passable (Go or Slow-Go) cell."""
    rows, cols = grid.shape
    if is_xy:
        init_c, init_r = cell
    else:
        init_r, init_c = cell

    init_r = int(init_r)
    init_c = int(init_c)

    init_r = max(0, min(init_r, rows - 1))
    init_c = max(0, min(init_c, cols - 1))

    if grid[init_r, init_c] != TrafficabilityClass.NO_GO:
        return (init_c, init_r) if is_xy else (init_r, init_c)

    queue = [(init_r, init_c)]
    visited = {(init_r, init_c)}

    while queue:
        r, c = queue.pop(0)
        for dr, dc, _ in NEIGHBOR_OFFSETS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                if grid[nr, nc] != TrafficabilityClass.NO_GO:
                    return (nc, nr) if is_xy else (nr, nc)
                visited.add((nr, nc))
                queue.append((nr, nc))

    return None


def find_path(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    is_xy: bool = False,
    cell_size_m: float = DEFAULT_CELL_SIZE_M,
    use_dijkstra: bool = False,
) -> PathResult:
    """Finds the optimal path between start and goal avoiding Red cells."""
    rows, cols = grid.shape

    if is_xy:
        s_col, s_row = start
        g_col, g_row = goal
    else:
        s_row, s_col = start
        g_row, g_col = goal

    s_row, s_col = int(s_row), int(s_col)
    g_row, g_col = int(g_row), int(g_col)

    if not (0 <= s_row < rows and 0 <= s_col < cols) or not (0 <= g_row < rows and 0 <= g_col < cols):
        return PathResult(
            path_cells=[],
            total_cost=0.0,
            status=PathStatus.OUT_OF_BOUNDS,
            message=f"Coordinates out of bounds: start=({s_col},{s_row}), goal=({g_col},{g_row}) for grid ({cols}x{rows}).",
        )

    if grid[s_row, s_col] == TrafficabilityClass.NO_GO:
        return PathResult(
            path_cells=[],
            total_cost=0.0,
            status=PathStatus.START_BLOCKED,
            message="Start point is located on impassable No-Go terrain.",
        )

    if grid[g_row, g_col] == TrafficabilityClass.NO_GO:
        return PathResult(
            path_cells=[],
            total_cost=0.0,
            status=PathStatus.GOAL_BLOCKED,
            message="Destination is located on impassable No-Go terrain.",
        )

    if s_row == g_row and s_col == g_col:
        pt = (s_col, s_row) if is_xy else (s_row, s_col)
        return PathResult(
            path_cells=[pt],
            total_cost=0.0,
            status=PathStatus.SAME_CELL,
            message="Start and destination points are identical.",
        )

    cost_mult = {
        TrafficabilityClass.GO: PATH_COST_GO,
        TrafficabilityClass.SLOW_GO: PATH_COST_SLOW_GO,
    }

    g_scores = np.full((rows, cols), np.inf, dtype=np.float64)
    g_scores[s_row, s_col] = 0.0

    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    h_start = 0.0 if use_dijkstra else octile_heuristic(s_row, s_col, g_row, g_col, cell_size_m)
    open_heap = [(h_start, 0.0, s_row, s_col)]
    closed_set = set()

    step_ortho = cell_size_m
    step_diag = SQRT2 * cell_size_m

    while open_heap:
        f, current_g, r, c = heapq.heappop(open_heap)

        if (r, c) in closed_set:
            continue
        closed_set.add((r, c))

        if r == g_row and c == g_col:
            path: List[Tuple[int, int]] = []
            curr = (g_row, g_col)
            while curr in came_from:
                path.append(curr)
                curr = came_from[curr]
            path.append((s_row, s_col))
            path.reverse()

            output_path = [(col, row) for row, col in path] if is_xy else path
            return PathResult(
                path_cells=output_path,
                total_cost=float(current_g),
                status=PathStatus.OK,
                message=f"Optimal tactical path computed successfully ({len(path)} waypoints).",
            )

        curr_mult = cost_mult[grid[r, c]]

        for dr, dc, is_diagonal in NEIGHBOR_OFFSETS:
            nr, nc = r + dr, c + dc

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            if grid[nr, nc] == TrafficabilityClass.NO_GO:
                continue

            if is_diagonal:
                ortho1_red = grid[r + dr, c] == TrafficabilityClass.NO_GO
                ortho2_red = grid[r, c + dc] == TrafficabilityClass.NO_GO
                if ortho1_red and ortho2_red:
                    continue

            neighbor_mult = cost_mult[grid[nr, nc]]
            mean_mult = 0.5 * (curr_mult + neighbor_mult)
            step_len = step_diag if is_diagonal else step_ortho
            tentative_g = current_g + step_len * mean_mult

            if tentative_g < g_scores[nr, nc]:
                g_scores[nr, nc] = tentative_g
                came_from[(nr, nc)] = (r, c)
                h = 0.0 if use_dijkstra else octile_heuristic(nr, nc, g_row, g_col, cell_size_m)
                heapq.heappush(open_heap, (tentative_g + h, tentative_g, nr, nc))

    return PathResult(
        path_cells=[],
        total_cost=0.0,
        status=PathStatus.NO_PATH,
        message="No viable tactical path exists between start and goal due to continuous impassable terrain barriers.",
    )


def find_path_xy(
    grid: np.ndarray,
    start_xy: Tuple[int, int],
    goal_xy: Tuple[int, int],
    cell_size_m: float = DEFAULT_CELL_SIZE_M,
) -> PathResult:
    """Public API convenience wrapper using (x=col, y=row) coordinates."""
    return find_path(grid, start=start_xy, goal=goal_xy, is_xy=True, cell_size_m=cell_size_m)


def plan_tactical_convoy(
    terrain: TerrainGrid,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    is_xy: bool = False,
    rainfall_multiplier: float = 1.0,
    current_factor: float = 0.20,
    deploy_avlb: bool = True,
    departure_interval_min: float = 10.0,
) -> ConvoyPlan:
    """
    Plans multi-vehicle echeloned convoy movement (Recon -> Spearhead -> Logistics)
    incorporating combat engineering AVLB tactical bridge-laying across river barriers.
    """
    from core.trafficability import compute_trafficability

    if is_xy:
        start_rc = (int(start[1]), int(start[0]))
        goal_rc = (int(goal[1]), int(goal[0]))
    else:
        start_rc = (int(start[0]), int(start[1]))
        goal_rc = (int(goal[0]), int(goal[1]))

    jeep = VEHICLE_PRESETS["Light Recon 4x4 Jeep"]
    tank = VEHICLE_PRESETS["Heavy Tracked Tank"]
    truck = VEHICLE_PRESETS["Medium Tactical 6x6 Truck"]

    # 1. Reconnaissance Screen (Jeep scouts the terrain)
    traff_jeep = compute_trafficability(terrain, jeep, rainfall_multiplier, current_factor)
    s_jeep = nearest_valid_cell(traff_jeep.grid, start_rc) or start_rc
    g_jeep = nearest_valid_cell(traff_jeep.grid, goal_rc) or goal_rc
    res_jeep = find_path(traff_jeep.grid, s_jeep, g_jeep)

    # 2. Identify Bridgehead location if AVLB is deployed
    bridge_cells: List[Tuple[int, int]] = []
    if deploy_avlb:
        # Find where the river intersects the direct corridor (near rows 48-52 or mid-latitude)
        crossing_r = 50
        river_cols = np.where(terrain.river_mask[crossing_r])[0]
        if len(river_cols) > 0:
            for c in river_cols:
                bridge_cells.append((crossing_r, int(c)))

    # 3. Main Armored Spearhead (Tank)
    traff_tank = compute_trafficability(
        terrain, tank, rainfall_multiplier, current_factor, bridge_cells=bridge_cells if deploy_avlb else None
    )
    s_tank = nearest_valid_cell(traff_tank.grid, start_rc) or start_rc
    g_tank = nearest_valid_cell(traff_tank.grid, goal_rc) or goal_rc
    res_tank = find_path(traff_tank.grid, s_tank, g_tank)

    # 4. Logistics Echelon (Trucks utilize the secured bridgehead or ford corridor)
    traff_truck = compute_trafficability(
        terrain, truck, rainfall_multiplier, current_factor, bridge_cells=bridge_cells if deploy_avlb else None
    )
    s_truck = nearest_valid_cell(traff_truck.grid, start_rc) or start_rc
    g_truck = nearest_valid_cell(traff_truck.grid, goal_rc) or goal_rc
    res_truck = find_path(traff_truck.grid, s_truck, g_truck)

    # Calculate transit timings
    def calc_transit_time(path_res: PathResult, v: VehicleProfile) -> float:
        if not path_res.path_cells or len(path_res.path_cells) < 2:
            return 0.0
        dist_km = len(path_res.path_cells) * 0.1
        avg_speed = 0.5 * (v.speed_kmph_go + v.speed_kmph_slowgo)
        return float((dist_km / avg_speed) * 60.0)

    t_jeep = calc_transit_time(res_jeep, jeep)
    t_tank = calc_transit_time(res_tank, tank)
    t_truck = calc_transit_time(res_truck, truck)

    dep_jeep = 0.0
    arr_jeep = dep_jeep + t_jeep

    dep_tank = departure_interval_min
    arr_tank = dep_tank + t_tank

    dep_truck = departure_interval_min * 2.5
    arr_truck = dep_truck + t_truck

    plan_recon = ConvoyElementPlan(
        vehicle=jeep,
        path_result=res_jeep,
        departure_delay_min=dep_jeep,
        transit_duration_min=t_jeep,
        arrival_time_min=arr_jeep,
        color=COLOR_CONVOY_RECON,
    )
    plan_spearhead = ConvoyElementPlan(
        vehicle=tank,
        path_result=res_tank,
        departure_delay_min=dep_tank,
        transit_duration_min=t_tank,
        arrival_time_min=arr_tank,
        color=COLOR_CONVOY_SPEARHEAD,
    )
    plan_logistics = ConvoyElementPlan(
        vehicle=truck,
        path_result=res_truck,
        departure_delay_min=dep_truck,
        transit_duration_min=t_truck,
        arrival_time_min=arr_truck,
        color=COLOR_CONVOY_LOGISTICS,
    )

    elements = {
        "Recon": plan_recon,
        "Spearhead": plan_spearhead,
        "Logistics": plan_logistics,
        "Recon Screen (Jeep)": plan_recon,
        "Armored Spearhead (Tank)": plan_spearhead,
        "Logistics Column (Truck)": plan_logistics,
    }

    total_span = max(arr_jeep, arr_tank, arr_truck)
    all_ok = all(e.path_result.status == PathStatus.OK for e in [plan_recon, plan_spearhead, plan_logistics])

    return ConvoyPlan(
        elements=elements,
        bridge_deployed=bool(deploy_avlb and len(bridge_cells) > 0),
        bridge_cells=bridge_cells,
        total_convoy_span_min=round(total_span, 1),
        status="CONVOY_OPERATIONAL" if all_ok else "CONVOY_OBSTRUCTED",
        message="All convoy echelons successfully routed across theater." if all_ok else "One or more convoy echelons encountered blocking terrain.",
    )
