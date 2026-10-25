"""Unit tests for A* Pathfinding Engine (Gate 3)."""

import time
import numpy as np
import pytest

from core.config import (
    TrafficabilityClass,
    DEFAULT_CELL_SIZE_M,
    PATH_COST_GO,
    PATH_COST_SLOW_GO,
)
from core.terrain import generate_terrain
from core.vehicles import VEHICLE_PRESETS
from core.trafficability import compute_trafficability
from core.pathfinding import (
    find_path,
    find_path_xy,
    nearest_valid_cell,
    PathStatus,
)


def test_no_red_cells_in_path():
    """Verify that computed paths never contain any No-Go (Red) cells."""
    terrain = generate_terrain(seed=42)
    truck = VEHICLE_PRESETS["Medium Tactical 6x6 Truck"]
    traff = compute_trafficability(terrain, truck)

    start = nearest_valid_cell(traff.grid, (85, 15))
    goal = nearest_valid_cell(traff.grid, (15, 85))
    assert start is not None and goal is not None

    res = find_path(traff.grid, start, goal, is_xy=False)
    assert res.status == PathStatus.OK, f"Expected path to be found, got {res.status}"
    assert len(res.path_cells) > 0

    # Ensure no cell in path is NO_GO
    for r, c in res.path_cells:
        assert traff.grid[r, c] != TrafficabilityClass.NO_GO, f"Red cell found in path at ({r}, {c})"


def test_path_is_8_neighbor_connected():
    """Verify every consecutive pair of cells along the path satisfies 8-neighbor connectivity."""
    terrain = generate_terrain(seed=101)
    tank = VEHICLE_PRESETS["Heavy Tracked Tank"]
    traff = compute_trafficability(terrain, tank)

    start = nearest_valid_cell(traff.grid, (80, 20))
    goal = nearest_valid_cell(traff.grid, (20, 80))
    assert start is not None and goal is not None

    res = find_path(traff.grid, start, goal, is_xy=False)
    assert res.status == PathStatus.OK, f"Expected path to be found, got {res.status}: {res.message}"

    path = res.path_cells
    for i in range(1, len(path)):
        r1, c1 = path[i - 1]
        r2, c2 = path[i]
        dr = abs(r2 - r1)
        dc = abs(c2 - c1)
        chebyshev_dist = max(dr, dc)
        assert chebyshev_dist == 1, f"Discontinuity at step {i}: from ({r1},{c1}) to ({r2},{c2})"


def test_no_path_returns_cleanly():
    """Verify that a fully blocked scenario returns NO_PATH status cleanly without exceptions."""
    grid = np.zeros((50, 50), dtype=int)
    # Build an impenetrable vertical wall of No-Go (Red = 2) cells down the center
    grid[:, 25] = TrafficabilityClass.NO_GO

    start = (25, 10)
    goal = (25, 40)

    res = find_path(grid, start, goal, is_xy=False)
    assert res.status == PathStatus.NO_PATH
    assert len(res.path_cells) == 0
    assert "No viable tactical path" in res.message


def test_start_and_goal_blocked_statuses():
    """Verify start and goal blocked detection and nearest valid cell recovery."""
    grid = np.zeros((30, 30), dtype=int)
    grid[5, 5] = TrafficabilityClass.NO_GO
    grid[20, 20] = TrafficabilityClass.NO_GO

    res_start_blocked = find_path(grid, (5, 5), (15, 15), is_xy=False)
    assert res_start_blocked.status == PathStatus.START_BLOCKED

    res_goal_blocked = find_path(grid, (15, 15), (20, 20), is_xy=False)
    assert res_goal_blocked.status == PathStatus.GOAL_BLOCKED

    # Test recovery with nearest_valid_cell
    valid_start = nearest_valid_cell(grid, (5, 5), is_xy=False)
    assert valid_start is not None
    assert grid[valid_start[0], valid_start[1]] != TrafficabilityClass.NO_GO

    # Out of bounds check
    res_oob = find_path(grid, (-1, 0), (10, 10), is_xy=False)
    assert res_oob.status == PathStatus.OUT_OF_BOUNDS


def test_corner_cutting_disallowed():
    """Verify that diagonal corner cutting between two diagonal red blocks is blocked."""
    grid = np.zeros((5, 5), dtype=int)
    # Create two Red obstacles forming a pinch point:
    # (1, 2) is Red, (2, 1) is Red
    # Trying to move from (1, 1) to (2, 2)
    grid[1, 2] = TrafficabilityClass.NO_GO
    grid[2, 1] = TrafficabilityClass.NO_GO

    res = find_path(grid, (1, 1), (2, 2), is_xy=False)
    # The path must detour around the pinch point, NOT cut diagonally from (1,1) to (2,2)
    assert res.status == PathStatus.OK
    assert (2, 2) in res.path_cells
    idx_start = res.path_cells.index((1, 1))
    idx_goal = res.path_cells.index((2, 2))
    assert idx_goal - idx_start > 1, "Direct diagonal corner cut should have been prohibited"


def test_astar_cost_equals_dijkstra_across_5_seeds():
    """
    Optimality Check: Verify that A* with octile heuristic finds the mathematically
    identical least-cost route as Dijkstra across 5 distinct random terrain seeds.
    """
    seeds = [11, 42, 77, 105, 999]
    truck = VEHICLE_PRESETS["Medium Tactical 6x6 Truck"]

    for seed in seeds:
        terrain = generate_terrain(seed=seed)
        traff = compute_trafficability(terrain, truck)

        # Use test coordinates with moderate separation to run swiftly while crossing varied terrain
        raw_start = (65, 30)
        raw_goal = (35, 70)
        start = nearest_valid_cell(traff.grid, raw_start)
        goal = nearest_valid_cell(traff.grid, raw_goal)

        assert start is not None and goal is not None

        res_astar = find_path(traff.grid, start, goal, use_dijkstra=False)
        res_dijkstra = find_path(traff.grid, start, goal, use_dijkstra=True)

        assert res_astar.status == res_dijkstra.status
        if res_astar.status == PathStatus.OK:
            # Total costs must be mathematically identical within float rounding
            assert np.isclose(res_astar.total_cost, res_dijkstra.total_cost, atol=1e-4), (
                f"Seed {seed}: A* cost {res_astar.total_cost} != Dijkstra cost {res_dijkstra.total_cost}"
            )


def test_pathfinding_performance_under_two_seconds():
    """Performance Benchmark: Verify A* pathfinding finishes in < 2.0 s on 100x100 grid."""
    terrain = generate_terrain(seed=42)
    truck = VEHICLE_PRESETS["Medium Tactical 6x6 Truck"]
    traff = compute_trafficability(terrain, truck)

    start = nearest_valid_cell(traff.grid, (90, 10))
    goal = nearest_valid_cell(traff.grid, (10, 90))

    t0 = time.perf_counter()
    res = find_path(traff.grid, start, goal)
    elapsed = time.perf_counter() - t0

    assert elapsed < 2.0, f"Pathfinding took {elapsed:.3f} s, expected < 2.0 s"
    assert res.status == PathStatus.OK
