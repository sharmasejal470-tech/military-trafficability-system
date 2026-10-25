"""A* Tactical Pathfinding Engine.

Computes least-risk, minimum-cost operational routes across 8-connected
trafficability grids avoiding Red (No-Go) impassable terrain and minimizing
traversal across Yellow (Slow-Go) mobility corridors.
"""

from __future__ import annotations
import heapq
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional, Dict

import numpy as np

from core.config import (
    TrafficabilityClass,
    PATH_COST_GO,
    PATH_COST_SLOW_GO,
    SQRT2,
    DEFAULT_CELL_SIZE_M,
)


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
    
    Attributes:
        path_cells: Sequence of grid cells [row, col] from start to goal.
        total_cost: Weighted accumulated path cost (meters-equivalent).
        status: PathStatus execution code.
        message: Operational tactical explanation.
    """
    path_cells: List[Tuple[int, int]]
    total_cost: float
    status: PathStatus
    message: str

    @property
    def path_xy(self) -> List[Tuple[int, int]]:
        """Returns path as public API (x=col, y=row) coordinates."""
        return [(c, r) for r, c in self.path_cells]


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
    """
    Computes the admissible and consistent Octile distance heuristic for 8-connected grids.
    Formula: ((sqrt(2) - 1) * min(dx, dy) + max(dx, dy)) * cell_size * min_cost_multiplier.
    Since min_cost_multiplier = PATH_COST_GO (1.0), this heuristic never overestimates true cost.
    """
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
    """
    Breadth-First Search (BFS) search for the nearest passable (Go or Slow-Go) cell.
    Returns (row, col) if is_xy is False, or (x, y) if is_xy is True.
    """
    rows, cols = grid.shape
    if is_xy:
        init_c, init_r = cell
    else:
        init_r, init_c = cell

    init_r = int(init_r)
    init_c = int(init_c)

    # Clamp starting point to grid bounds for BFS initialization
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
    """
    Finds the optimal path between start and goal avoiding Red cells.
    
    Parameters:
        grid: 2D integer array (0=Go, 1=Slow-Go, 2=No-Go).
        start: Start coordinates. (row, col) if is_xy=False, or (x=col, y=row) if is_xy=True.
        goal: Goal coordinates. (row, col) if is_xy=False, or (x=col, y=row) if is_xy=True.
        is_xy: If True, inputs/outputs are formatted as (x=col, y=row).
        cell_size_m: Grid cell spatial resolution in meters.
        use_dijkstra: If True, sets heuristic to 0.0 (Dijkstra baseline for optimality testing).
    
    Returns:
        PathResult dataclass with path cells, total cost, status, and diagnostic message.
    """
    rows, cols = grid.shape

    if is_xy:
        s_col, s_row = start
        g_col, g_row = goal
    else:
        s_row, s_col = start
        g_row, g_col = goal

    s_row, s_col = int(s_row), int(s_col)
    g_row, g_col = int(g_row), int(g_col)

    # 1. Bounds Validation
    if not (0 <= s_row < rows and 0 <= s_col < cols) or not (0 <= g_row < rows and 0 <= g_col < cols):
        return PathResult(
            path_cells=[],
            total_cost=0.0,
            status=PathStatus.OUT_OF_BOUNDS,
            message=f"Coordinates out of bounds: start=({s_col},{s_row}), goal=({g_col},{g_row}) for grid ({cols}x{rows}).",
        )

    # 2. Blocked Endpoint Checks
    if grid[s_row, s_col] == TrafficabilityClass.NO_GO:
        return PathResult(
            path_cells=[],
            total_cost=0.0,
            status=PathStatus.START_BLOCKED,
            message="Start point is located on impassable No-Go terrain (steep slope, deep water, or weak substrate).",
        )

    if grid[g_row, g_col] == TrafficabilityClass.NO_GO:
        return PathResult(
            path_cells=[],
            total_cost=0.0,
            status=PathStatus.GOAL_BLOCKED,
            message="Destination is located on impassable No-Go terrain (steep slope, deep water, or weak substrate).",
        )

    # 3. Same Cell Check
    if s_row == g_row and s_col == g_col:
        pt = (s_col, s_row) if is_xy else (s_row, s_col)
        return PathResult(
            path_cells=[pt],
            total_cost=0.0,
            status=PathStatus.SAME_CELL,
            message="Start and destination points are identical.",
        )

    # Multiplier lookup table: Go -> 1.0, Slow-Go -> 5.0
    cost_mult = {
        TrafficabilityClass.GO: PATH_COST_GO,
        TrafficabilityClass.SLOW_GO: PATH_COST_SLOW_GO,
    }

    # 4. A* Search Setup
    # g_score array initialized to infinity
    g_scores = np.full((rows, cols), np.inf, dtype=np.float64)
    g_scores[s_row, s_col] = 0.0

    # Came_from map for backtracking
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

    # Priority queue: (f_score, g_score, row, col)
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

        # Goal reached
        if r == g_row and c == g_col:
            # Reconstruct path
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

        # Explore 8 neighbors
        for dr, dc, is_diagonal in NEIGHBOR_OFFSETS:
            nr, nc = r + dr, c + dc

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            # Neighbor must not be No-Go (Red)
            if grid[nr, nc] == TrafficabilityClass.NO_GO:
                continue

            # Corner-cutting prevention:
            # For diagonal movement, if BOTH adjacent orthogonal cells are RED,
            # corner cutting is strictly disallowed.
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
