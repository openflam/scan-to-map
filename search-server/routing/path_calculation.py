"""Path calculation module for routing between coordinates."""

from typing import List, Tuple, Optional
import numpy as np
import heapq
from pathlib import Path
import json


def world_to_grid(
    x: float, y: float, origin: List[float], cell_size: float
) -> Tuple[int, int]:
    """
    Convert world coordinates to grid indices.

    Args:
        x: World x coordinate
        y: World y coordinate
        origin: Grid origin [x, y]
        cell_size: Size of each grid cell

    Returns:
        Tuple of (row, col) grid indices
    """
    col = int((x - origin[0]) / cell_size)
    row = int((y - origin[1]) / cell_size)
    return row, col


def grid_to_world(
    row: int, col: int, origin: List[float], cell_size: float
) -> Tuple[float, float]:
    """
    Convert grid indices to world coordinates (cell center).

    Args:
        row: Grid row index
        col: Grid column index
        origin: Grid origin [x, y]
        cell_size: Size of each grid cell

    Returns:
        Tuple of (x, y) world coordinates
    """
    x = origin[0] + (col + 0.5) * cell_size
    y = origin[1] + (row + 0.5) * cell_size
    return x, y


def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """
    Calculate the heuristic (Euclidean distance) between two grid positions.

    Args:
        a: First position (row, col)
        b: Second position (row, col)

    Returns:
        Euclidean distance between the positions
    """
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def find_nearest_unoccupied(
    position: Tuple[int, int], occupancy_grid: np.ndarray, max_search_radius: int = 50
) -> Optional[Tuple[int, int]]:
    """
    Find the nearest unoccupied cell to a given position using BFS.

    Args:
        position: Starting position (row, col)
        occupancy_grid: Binary grid where 0 is floor (traversable) and 1 is occupied
        max_search_radius: Maximum distance to search for unoccupied cell

    Returns:
        Nearest unoccupied position (row, col), or None if not found within radius
    """
    rows, cols = occupancy_grid.shape
    start_row, start_col = position

    # If already unoccupied, return the position
    if occupancy_grid[start_row, start_col] == 0:
        return position

    # BFS to find nearest unoccupied cell
    queue = [(start_row, start_col, 0)]  # (row, col, distance)
    visited = {(start_row, start_col)}

    # 8-connected neighbors
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while queue:
        row, col, dist = queue.pop(0)

        # Check if we've exceeded max search radius
        if dist > max_search_radius:
            break

        for dr, dc in neighbors:
            new_row, new_col = row + dr, col + dc

            # Check bounds
            if not (0 <= new_row < rows and 0 <= new_col < cols):
                continue

            # Skip if already visited
            if (new_row, new_col) in visited:
                continue

            visited.add((new_row, new_col))

            # Check if unoccupied
            if occupancy_grid[new_row, new_col] == 0:
                return (new_row, new_col)

            # Add to queue for further exploration
            queue.append((new_row, new_col, dist + 1))

    # No unoccupied cell found within radius
    return None


def astar_search(
    start: Tuple[int, int], goal: Tuple[int, int], occupancy_grid: np.ndarray
) -> Optional[List[Tuple[int, int]]]:
    """
    Perform A* search on the occupancy grid.

    Args:
        start: Start position (row, col)
        goal: Goal position (row, col)
        occupancy_grid: Binary grid where 0 is floor (traversable) and 1 is occupied

    Returns:
        List of grid positions forming the path, or None if no path exists
    """
    rows, cols = occupancy_grid.shape

    # Check if start and goal are within bounds
    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        return None
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        return None

    # If start or goal are occupied, find nearest unoccupied cell
    if occupancy_grid[start[0], start[1]] == 1:
        start = find_nearest_unoccupied(start, occupancy_grid)
        if start is None:
            return None

    if occupancy_grid[goal[0], goal[1]] == 1:
        goal = find_nearest_unoccupied(goal, occupancy_grid)
        if goal is None:
            return None

    # Priority queue: (f_score, counter, position)
    counter = 0
    open_set = [(0, counter, start)]
    counter += 1

    # Track the path
    came_from = {}

    # Cost from start to each node
    g_score = {start: 0}

    # Estimated total cost from start to goal through each node
    f_score = {start: heuristic(start, goal)}

    # Set of positions in the open set for quick lookup
    open_set_positions = {start}

    # 8-connected neighbors (including diagonals)
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while open_set:
        _, _, current = heapq.heappop(open_set)
        open_set_positions.discard(current)

        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        current_g = g_score[current]

        for dr, dc in neighbors:
            neighbor = (current[0] + dr, current[1] + dc)

            # Check if neighbor is valid
            if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                continue

            # Check if neighbor is traversable
            if occupancy_grid[neighbor[0], neighbor[1]] == 1:
                continue

            # Calculate movement cost (diagonal vs straight)
            if dr != 0 and dc != 0:
                move_cost = np.sqrt(2)  # Diagonal movement
            else:
                move_cost = 1.0  # Straight movement

            tentative_g_score = current_g + move_cost

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f = tentative_g_score + heuristic(neighbor, goal)
                f_score[neighbor] = f

                if neighbor not in open_set_positions:
                    heapq.heappush(open_set, (f, counter, neighbor))
                    counter += 1
                    open_set_positions.add(neighbor)

    # No path found
    return None


def calculate_route(
    source: Tuple[float, float, float],
    destination: Tuple[float, float, float],
    occupancy_grid: np.ndarray,
    metadata: dict,
    floor_height_file: Optional[str] = None,
) -> List[Tuple[float, float, float]]:
    """
    Calculate a route from source to destination using the occupancy grid.

    Uses A* pathfinding algorithm to find the shortest path on the occupancy grid.

    Args:
        source: Source coordinates (x, y, z)
        destination: Destination coordinates (x, y, z)
        occupancy_grid: 2D numpy array representing the occupancy grid
                       (0 = floor/traversable, 1 = occupied)
        metadata: Metadata dictionary with grid information including:
                 - origin: [x, y] origin coordinates
                 - cell_size: size of each grid cell
                 - grid_shape: [rows, cols] shape of the grid
        floor_height_file: Optional path to floor_height.json file.
                          If provided, uses the floor height from the file for z-coordinates.

    Returns:
        List of coordinates representing the path from source to destination.
        Returns empty list if no path is found.
    """
    # Extract metadata
    origin = metadata["origin"]
    cell_size = metadata["cell_size"]

    # Load floor height if file is provided
    floor_height = None
    if floor_height_file:
        try:
            with open(floor_height_file, "r") as f:
                floor_height_data = json.load(f)
                floor_height = floor_height_data.get("floor_height")
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass  # Fall back to using source z-coordinate

    # Convert world coordinates to grid indices
    start_grid = world_to_grid(source[0], source[1], origin, cell_size)
    goal_grid = world_to_grid(destination[0], destination[1], origin, cell_size)

    # Find path using A* search
    grid_path = astar_search(start_grid, goal_grid, occupancy_grid)

    if grid_path is None:
        # No path found, return empty list
        return []

    # Convert grid path back to world coordinates
    world_path = []
    for row, col in grid_path:
        x, y = grid_to_world(row, col, origin, cell_size)
        # Use floor height if available, otherwise use source z-coordinate
        z = floor_height if floor_height is not None else source[2]
        world_path.append((x, y, z))

    return world_path
