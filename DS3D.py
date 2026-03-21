#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import heapq, time, math

# ------------------- Environment Loader -------------------
def load_environment(filename):
    """
    File format (whitespace‐separated):
      boundary xmin ymin zmin xmax ymax zmax
      obstacle xmin ymin zmin xmax ymax zmax   (multiple lines allowed)
      start x y z
      goal x y z
    Lines starting with '#' are ignored.
    """
    boundary = None
    obstacles = []
    start = None
    goal = None
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            key = parts[0].lower()
            if key == 'boundary':
                boundary = list(map(float, parts[1:7]))
            elif key == 'obstacle':
                obstacles.append(list(map(float, parts[1:7])))
            elif key == 'start':
                start = list(map(float, parts[1:4]))
            elif key == 'goal':
                goal = list(map(float, parts[1:4]))
    return boundary, obstacles, start, goal

# ------------------- Grid Conversion Functions -------------------
def world_to_grid(point, boundary, resolution):
    xmin, ymin, zmin, _, _, _ = boundary
    # Force point's z to be ground level.
    point = [point[0], point[1], zmin]
    i = int(round((point[0]-xmin)/resolution))
    j = int(round((point[1]-ymin)/resolution))
    # With ground-only planning, use a single layer.
    k = 0  
    return (i, j, k)

def grid_to_world(idx, boundary, resolution):
    xmin, ymin, zmin, _, _, _ = boundary
    x = xmin + idx[0]*resolution
    y = ymin + idx[1]*resolution
    # All points are on the ground.
    z = zmin
    return np.array([x, y, z])

def is_in_obstacle(point, obstacles):
    for obs in obstacles:
        oxmin, oymin, ozmin, oxmax, oymax, ozmax = obs
        if oxmin <= point[0] <= oxmax and oymin <= point[1] <= oymax and ozmin <= point[2] <= ozmax:
            return True
    return False

def get_neighbors(idx, grid_shape):
    neighbors = []
    # Only allow neighbors in the X and Y directions since we use a single layer (k=0).
    for di in [-1,0,1]:
        for dj in [-1,0,1]:
            # Skip vertical movement.
            dk = 0
            if di==0 and dj==0:
                continue
            ni, nj, nk = idx[0]+di, idx[1]+dj, idx[2]
            if 0 <= ni < grid_shape[0] and 0 <= nj < grid_shape[1]:
                neighbors.append((ni, nj, nk))
    return neighbors

def heuristic(a, b):
    return np.linalg.norm(np.array(a)-np.array(b))

def a_star_search(start_idx, goal_idx, grid_shape, boundary, resolution, obstacles):
    open_set = []
    heapq.heappush(open_set, (heuristic(start_idx, goal_idx), 0, start_idx))
    came_from = {}
    cost_so_far = {start_idx: 0}
    while open_set:
        _, cost, current = heapq.heappop(open_set)
        if current == goal_idx:
            break
        for neighbor in get_neighbors(current, grid_shape):
            world_pt = grid_to_world(neighbor, boundary, resolution)
            if is_in_obstacle(world_pt, obstacles):
                continue
            new_cost = cost_so_far[current] + heuristic(current, neighbor)
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal_idx)
                heapq.heappush(open_set, (priority, new_cost, neighbor))
                came_from[neighbor] = current
    if goal_idx not in came_from:
        return None
    path = []
    current = goal_idx
    while current != start_idx:
        path.append(current)
        current = came_from[current]
    path.append(start_idx)
    return path[::-1]

# ------------------- Plotting Functions -------------------
def draw_obstacle(ax, obs, color='black', alpha=0.5):
    xmin, ymin, zmin, xmax, ymax, zmax = obs
    vertices = [
        [(xmin, ymin, zmin), (xmax, ymin, zmin), (xmax, ymax, zmin), (xmin, ymax, zmin)],  # bottom
        [(xmin, ymin, zmax), (xmax, ymin, zmax), (xmax, ymax, zmax), (xmin, ymax, zmax)],  # top
        [(xmin, ymin, zmin), (xmax, ymin, zmin), (xmax, ymin, zmax), (xmin, ymin, zmax)],  # front
        [(xmin, ymax, zmin), (xmax, ymax, zmin), (xmax, ymax, zmax), (xmin, ymax, zmax)],  # back
        [(xmin, ymin, zmin), (xmin, ymax, zmin), (xmin, ymax, zmax), (xmin, ymin, zmax)],  # left
        [(xmax, ymin, zmin), (xmax, ymax, zmin), (xmax, ymax, zmax), (xmax, ymin, zmax)]   # right
    ]
    poly3d = Poly3DCollection(vertices, facecolors='black', edgecolors=None, linewidths=0.5, alpha=0.5)
    ax.add_collection3d(poly3d)

def draw_ground(ax, boundary):
    xmin, ymin, zmin, xmax, ymax, _ = boundary
    ground_corners = [(xmin, ymin, zmin),
                      (xmax, ymin, zmin),
                      (xmax, ymax, zmin),
                      (xmin, ymax, zmin)]
    poly = Poly3DCollection([ground_corners], facecolors='white', alpha=0, edgecolors=None)
    ax.add_collection3d(poly)

def plot_environment(boundary, obstacles, start, goal, path, resolution):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xmin, ymin, zmin, xmax, ymax, zmax = boundary

    # Draw ground surface in black.
    draw_ground(ax, boundary)
    
    # Draw obstacles as translucent white boxes.
    for obs in obstacles:
        draw_obstacle(ax, obs, color='white', alpha=0.5)
    
    # Plot start and goal (forced on ground).
    ax.scatter(start[0], start[1], zmin, color='green', s=100, label='Start')
    ax.scatter(goal[0], goal[1], zmin, color='red', s=100, label='Goal')
    
    # Plot the path (convert grid indices to world coordinates; force z to ground).
    if path is not None:
        world_path = [grid_to_world(idx, boundary, resolution) for idx in path]
        world_path = np.array(world_path)
        ax.plot(world_path[:,0], world_path[:,1], world_path[:,2], color='blue', linewidth=2, label='Path')
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('DStar3D (Grid-based A*) Path Planning (Ground-Constrained)')
    ax.legend()
    plt.show()

# ------------------- Main -------------------
def main():
    env_file = "BigEnv.txt"
    boundary, obstacles, start, goal = load_environment(env_file)
    print("Environment loaded:")
    print("Boundary:", boundary)
    print("Obstacles:", obstacles)
    print("Start:", start, "Goal:", goal)
    resolution = 0.5
    xmin, ymin, zmin, xmax, ymax, zmax = boundary
    # Force start and goal to ground.
    start = [start[0], start[1], zmin]
    goal = [goal[0], goal[1], zmin]
    # Use only one layer in z.
    grid_shape = (int(round((xmax - xmin)/resolution)) + 1,
                  int(round((ymax - ymin)/resolution)) + 1,
                  1)
    start_idx = world_to_grid(start, boundary, resolution)
    goal_idx = world_to_grid(goal, boundary, resolution)
    t0 = time.time()
    path = a_star_search(start_idx, goal_idx, grid_shape, boundary, resolution, obstacles)
    t_elapsed = time.time() - t0
    if path is None:
        print("No path found!")
        return
    world_path = [grid_to_world(idx, boundary, resolution) for idx in path]
    path_length = sum(np.linalg.norm(world_path[i+1]-world_path[i]) for i in range(len(world_path)-1))
    print(f"DStar3D: Time taken: {t_elapsed:.2f} sec, Path length: {path_length:.2f}")
    plot_environment(boundary, obstacles, start, goal, path, resolution)

if __name__ == '__main__':
    main()
