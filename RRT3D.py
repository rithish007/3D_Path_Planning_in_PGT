#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import time, math, random
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
    Check if the straight-line path between p1 and p2 is free of obstacles.
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

# ------------------- RRT Algorithm Constrained to Ground -------------------
def rrt_planning(start, goal, boundary, obstacles, max_iters=5000, step_size=0.5, goal_sample_rate=0.1):
    xmin, ymin, zmin, xmax, ymax, zmax = boundary
    ground_z = zmin  # ground level

    # Constrain start and goal to the ground
    start = [start[0], start[1], ground_z]
    goal = [goal[0], goal[1], ground_z]

    start_node = Node(start)
    goal_node = Node(goal)
    tree = [start_node]

    for i in range(max_iters):
        # Sample random point on ground
        if random.random() < goal_sample_rate:
            rnd = np.array(goal)
        else:
            rnd = np.array([random.uniform(xmin, xmax),
                            random.uniform(ymin, ymax),
                            ground_z])
        # Find nearest node in the tree
        dists = [np.linalg.norm(n.point - rnd) for n in tree]
        nearest = tree[np.argmin(dists)]
        direction = rnd - nearest.point
        norm = np.linalg.norm(direction)
        if norm == 0:
            continue
        direction = direction / norm
        new_point = nearest.point + step_size * direction
        new_point[2] = ground_z  # ensure new point stays on the ground

        if not is_line_collision_free(nearest.point, new_point, obstacles):
            continue

        new_node = Node(new_point, nearest)
        tree.append(new_node)

        # Check if goal is reached
        if np.linalg.norm(new_node.point - np.array(goal)) <= step_size:
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
def draw_obstacle(ax, obs, color='black', alpha=0.5):
    """
    Draw a translucent 3D box for the obstacle.
    """
    xmin, ymin, zmin, xmax, ymax, zmax = obs
    # Define vertices for each face of the box
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
    """
    Draw the ground surface (at z=zmin) as a solid black surface.
    """
    xmin, ymin, zmin, xmax, ymax, _ = boundary
    zmin=-0.1
    # Define the four corners of the ground
    ground_corners = [(xmin, ymin, zmin),
                      (xmax, ymin, zmin),
                      (xmax, ymax, zmin),
                      (xmin, ymax, zmin)]
    poly = Poly3DCollection([ground_corners], facecolors='white', alpha=0, edgecolors=None)
    ax.add_collection3d(poly)


def plot_environment(boundary, obstacles, start, goal, path, tree):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xmin, ymin, zmin, xmax, ymax, zmax = boundary
    ground_z = zmin

    # Draw ground surface in black
    draw_ground(ax, boundary)
    
    # Plot obstacles with full height in translucent white
    for obs in obstacles:
        draw_obstacle(ax, obs, color='white', alpha=0.5)
    
    # Plot RRT tree edges
    for node in tree:
        if node.parent is not None:
            p1 = node.point
            p2 = node.parent.point
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='orange', alpha=0.3)
    
    # Plot the final path (on the ground)
    if path is not None:
        # Ensure all path points are forced to the ground
        path_ground = np.array([[p[0], p[1], ground_z] for p in path])
        ax.plot(path_ground[:,0], path_ground[:,1], path_ground[:,2], color='blue', linewidth=3, label='Path')
    
    # Plot start and goal on the ground
    ax.scatter(start[0], start[1], ground_z, color='green', s=100, label='Start')
    ax.scatter(goal[0], goal[1], ground_z, color='red', s=100, label='Goal')
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('RRT3D Path Planning (Ground-Constrained Path)')
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
    
    # Constrain start and goal to ground level (z = zmin)
    ground_z = boundary[2]
    start = [start[0], start[1], ground_z]
    goal = [goal[0], goal[1], ground_z]

    t0 = time.time()
    goal_node, tree = rrt_planning(start, goal, boundary, obstacles,
                                   max_iters=5000, step_size=0.5, goal_sample_rate=0.1)
    t_elapsed = time.time() - t0

    if goal_node is None:
        print("RRT3D (Ground-Constrained): No path found after maximum iterations!")
        return

    path = reconstruct_path(goal_node)
    path_length = sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i])) for i in range(len(path)-1))
    print(f"RRT3D (Ground-Constrained): Time taken: {t_elapsed:.2f} sec, Path length: {path_length:.2f}")

    plot_environment(boundary, obstacles, start, goal, path, tree)

if __name__ == '__main__':
    main()
