#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import time, math, random

# ------------------- Environment Loader -------------------
def load_environment(filename):
    """
    Loads the environment from a file.
    Expected file format:
      boundary xmin ymin zmin xmax ymax zmax
      obstacle xmin ymin zmin xmax ymax zmax
      start x y z
      goal x y z
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

# ------------------- Collision Checking -------------------
def point_in_obstacle(point, obstacles):
    for obs in obstacles:
        oxmin, oymin, ozmin, oxmax, oymax, ozmax = obs
        if oxmin <= point[0] <= oxmax and oymin <= point[1] <= oymax and ozmin <= point[2] <= ozmax:
            return True
    return False

def is_line_collision_free(p1, p2, obstacles, step_size=0.1):
    """
    Check if the straight line path between p1 and p2 is free of obstacles.
    The line is sampled at intervals of step_size.
    """
    dist = np.linalg.norm(np.array(p2) - np.array(p1))
    steps = max(int(dist / step_size), 1)
    for i in range(steps + 1):
        t = i / steps
        point = np.array(p1) * (1 - t) + np.array(p2) * t
        if point_in_obstacle(point, obstacles):
            return False
    return True

# ------------------- RRT Node Definition -------------------
class Node:
    def __init__(self, point, parent=None):
        self.point = np.array(point)
        self.parent = parent

# ------------------- RRT Algorithm -------------------
def rrt_planning(start, goal, boundary, obstacles, max_iters=5000, step_size=0.5, goal_sample_rate=0.1):
    xmin, ymin, zmin, xmax, ymax, zmax = boundary
    start_node = Node(start)
    goal_node = Node(goal)
    tree = [start_node]

    for i in range(max_iters):
        # Sample random point (with bias toward goal)
        if random.random() < goal_sample_rate:
            rnd = np.array(goal)
        else:
            rnd = np.array([random.uniform(xmin, xmax),
                            random.uniform(ymin, ymax),
                            random.uniform(zmin, zmax)])
        # Find nearest node in the tree
        dists = [np.linalg.norm(n.point - rnd) for n in tree]
        nearest = tree[np.argmin(dists)]
        direction = rnd - nearest.point
        norm = np.linalg.norm(direction)
        if norm == 0:
            continue
        direction = direction / norm
        new_point = nearest.point + step_size * direction

        # Check for collision on the path from nearest to new_point
        if not is_line_collision_free(nearest.point, new_point, obstacles):
            continue

        new_node = Node(new_point, nearest)
        tree.append(new_node)

        # Check if goal is reached
        if np.linalg.norm(new_node.point - np.array(goal)) <= step_size:
            # Connect directly to goal if collision free
            if is_line_collision_free(new_node.point, goal, obstacles):
                goal_node.parent = new_node
                tree.append(goal_node)
                return goal_node, tree

    return None, tree

def reconstruct_path(goal_node):
    path = []
    node = goal_node
    while node is not None:
        path.append(node.point)
        node = node.parent
    return path[::-1]

# ------------------- Plotting Functions -------------------
def draw_box(ax, box, color='black', alpha=0.5):
    xmin, ymin, zmin, xmax, ymax, zmax = box
    corners = np.array([
        [xmin, ymin, zmin],
        [xmax, ymin, zmin],
        [xmax, ymax, zmin],
        [xmin, ymax, zmin],
        [xmin, ymin, zmax],
        [xmax, ymin, zmax],
        [xmax, ymax, zmax],
        [xmin, ymax, zmax]
    ])
    edges = [(0,1), (1,2), (2,3), (3,0),
             (4,5), (5,6), (6,7), (7,4),
             (0,4), (1,5), (2,6), (3,7)]
    for e in edges:
        pts = corners[list(e), :]
        ax.plot(pts[:,0], pts[:,1], pts[:,2], color=color, alpha=alpha)

def plot_environment(boundary, obstacles, start, goal, path, tree):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xmin, ymin, zmin, xmax, ymax, zmax = boundary

    # Plot obstacles
    for obs in obstacles:
        draw_box(ax, obs, color='gray')
    
    # Plot RRT tree edges
    for node in tree:
        if node.parent is not None:
            p1 = node.point
            p2 = node.parent.point
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='orange', alpha=0.3)
    
    # Plot path if found
    if path is not None:
        path = np.array(path)
        ax.plot(path[:,0], path[:,1], path[:,2], color='blue', linewidth=3, label='Path')
    
    ax.scatter(start[0], start[1], start[2], color='green', s=100, label='Start')
    ax.scatter(goal[0], goal[1], goal[2], color='red', s=100, label='Goal')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('RRT3D Path Planning')
    ax.legend()
    plt.show()

# ------------------- Main -------------------
def main():
    env_file = "environment.txt"
    boundary, obstacles, start, goal = load_environment(env_file)
    print("Environment loaded:")
    print("Boundary:", boundary)
    print("Obstacles:", obstacles)
    print("Start:", start, "Goal:", goal)
    
    t0 = time.time()
    goal_node, tree = rrt_planning(start, goal, boundary, obstacles,
                                   max_iters=5000, step_size=0.5, goal_sample_rate=0.1)
    t_elapsed = time.time() - t0

    if goal_node is None:
        print("RRT3D: No path found after maximum iterations!")
        return

    path = reconstruct_path(goal_node)
    path_length = sum(np.linalg.norm(np.array(path[i+1])-np.array(path[i])) for i in range(len(path)-1))
    print(f"RRT3D: Time taken: {t_elapsed:.2f} sec, Path length: {path_length:.2f}")

    plot_environment(boundary, obstacles, start, goal, path, tree)

if __name__ == '__main__':
    main()
