from __future__ import annotations

import heapq
from collections import deque
from typing import List, Tuple, Optional


Grid = List[List[int]]  # 0 = boş, 1 = duvar
Coord = Tuple[int, int]


def neighbors(pos: Coord, grid: Grid) -> List[Coord]:
    x, y = pos
    h, w = len(grid), len(grid[0])
    result = []
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < h and 0 <= ny < w and grid[nx][ny] == 0:
            result.append((nx, ny))
    return result


def reconstruct_path(came_from: dict[Coord, Coord], start: Coord, goal: Coord) -> List[Coord]:
    cur = goal
    path = [cur]
    while cur != start:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path


def bfs(grid: Grid, start: Coord, goal: Coord) -> Tuple[Optional[List[Coord]], int]:
    """0-1 grid üzerinde BFS ile en kısa yol (adım sayısı) arar."""
    queue = deque([start])
    visited = {start}
    came_from: dict[Coord, Coord] = {}
    expanded = 0

    while queue:
        current = queue.popleft()
        expanded += 1
        if current == goal:
            return reconstruct_path(came_from, start, goal), expanded
        for nb in neighbors(current, grid):
            if nb not in visited:
                visited.add(nb)
                came_from[nb] = current
                queue.append(nb)

    return None, expanded


def dfs(grid: Grid, start: Coord, goal: Coord) -> Tuple[Optional[List[Coord]], int]:
    """DFS; en kısa yolu garanti etmez, ama arama düzenini gösterir."""
    stack = [start]
    visited = {start}
    came_from: dict[Coord, Coord] = {}
    expanded = 0

    while stack:
        current = stack.pop()
        expanded += 1
        if current == goal:
            return reconstruct_path(came_from, start, goal), expanded
        for nb in neighbors(current, grid):
            if nb not in visited:
                visited.add(nb)
                came_from[nb] = current
                stack.append(nb)

    return None, expanded


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid: Grid, start: Coord, goal: Coord) -> Tuple[Optional[List[Coord]], int]:
    """A* araması; maliyeti 1 olan grid’de genelde BFS’ten daha az düğüm genişletir."""
    open_set: List[Tuple[int, Coord]] = []
    heapq.heappush(open_set, (0, start))

    came_from: dict[Coord, Coord] = {}
    g_score = {start: 0}
    expanded = 0
    closed = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        if current in closed:
            continue
        closed.add(current)
        expanded += 1

        if current == goal:
            return reconstruct_path(came_from, start, goal), expanded

        for nb in neighbors(current, grid):
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(nb, 1e9):
                came_from[nb] = current
                g_score[nb] = tentative_g
                f_score = tentative_g + manhattan(nb, goal)
                heapq.heappush(open_set, (f_score, nb))

    return None, expanded


def print_grid_with_path(grid: Grid, path: Optional[List[Coord]], start: Coord, goal: Coord) -> None:
    h, w = len(grid), len(grid[0])
    path_set = set(path) if path else set()
    for i in range(h):
        row = ""
        for j in range(w):
            if (i, j) == start:
                row += "S "
            elif (i, j) == goal:
                row += "G "
            elif grid[i][j] == 1:
                row += "# "
            elif (i, j) in path_set:
                row += "* "
            else:
                row += ". "
        print(row)
    print()


def demo():
    # 0 = boş, 1 = duvar
    grid: Grid = [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
    ]

    start = (0, 0)
    goal = (4, 5)

    print("=== BFS ===")
    path_bfs, expanded_bfs = bfs(grid, start, goal)
    print(f"BFS: path length = {len(path_bfs) if path_bfs else None}, expanded = {expanded_bfs}")
    print_grid_with_path(grid, path_bfs, start, goal)

    print("=== DFS ===")
    path_dfs, expanded_dfs = dfs(grid, start, goal)
    print(f"DFS: path length = {len(path_dfs) if path_dfs else None}, expanded = {expanded_dfs}")
    print_grid_with_path(grid, path_dfs, start, goal)

    print("=== A* ===")
    path_astar, expanded_astar = astar(grid, start, goal)
    print(f"A*: path length = {len(path_astar) if path_astar else None}, expanded = {expanded_astar}")
    print_grid_with_path(grid, path_astar, start, goal)


def main():
    demo()


if __name__ == "__main__":
    main()
