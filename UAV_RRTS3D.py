#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time, random, math

# ------------------- Environment Loader -------------------
def load_environment(filename):
    """
    Expected file format (whitespace‐separated):
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

# ------------------- RRT* Classes and Functions -------------------
class Node:
    def __init__(self, pos):
        self.pos = np.array(pos)
        self.parent = None
        self.cost = 0.0

class RRTStar3D:
    def __init__(self, boundary, obstacles, start, goal, step_size=0.5, max_iter=2000, goal_sample_rate=0.1, neighbor_radius=2.0):
        self.boundary = boundary
        self.obstacles = obstacles
        self.start = Node(start)
        self.goal = Node(goal)
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate
        self.neighbor_radius = neighbor_radius
        self.nodes = [self.start]

    def plan(self):
        start_time = time.time()
        for i in range(self.max_iter):
            rnd_point = self.sample()
            nearest_node = self.get_nearest_node(rnd_point)
            new_node = self.steer(nearest_node, rnd_point)
            if new_node is None:
                continue
            if self.collision_free(nearest_node.pos, new_node.pos):
                near_nodes = self.get_near_nodes(new_node)
                new_node = self.choose_parent(near_nodes, nearest_node, new_node)
                self.nodes.append(new_node)
                self.rewire(new_node, near_nodes)
                if np.linalg.norm(new_node.pos - self.goal.pos) < self.step_size:
                    if self.collision_free(new_node.pos, self.goal.pos):
                        self.goal.parent = new_node
                        self.goal.cost = new_node.cost + np.linalg.norm(new_node.pos - self.goal.pos)
                        self.nodes.append(self.goal)
                        elapsed = time.time() - start_time
                        path = self.extract_path()
                        return path, elapsed
        return None, time.time() - start_time

    def sample(self):
        if random.random() < self.goal_sample_rate:
            return self.goal.pos
        xmin, ymin, zmin, xmax, ymax, zmax = self.boundary
        return np.array([
            random.uniform(xmin, xmax),
            random.uniform(ymin, ymax),
            random.uniform(zmin, zmax)
        ])

    def get_nearest_node(self, point):
        return min(self.nodes, key=lambda node: np.linalg.norm(node.pos - point))

    def steer(self, from_node, to_point):
        direction = to_point - from_node.pos
        dist = np.linalg.norm(direction)
        if dist == 0:
            return None
        direction = direction / dist
        dist = min(self.step_size, dist)
        new_pos = from_node.pos + dist * direction
        new_node = Node(new_pos)
        new_node.parent = from_node
        new_node.cost = from_node.cost + dist
        return new_node

    def collision_free(self, p1, p2):
        n_checks = int(math.ceil(np.linalg.norm(p2-p1) / (self.step_size/10)))
        for i in range(n_checks+1):
            t = i / n_checks
            p = p1 + t*(p2-p1)
            if not self.point_in_boundaries(p) or self.in_obstacle(p):
                return False
        return True

    def point_in_boundaries(self, p):
        xmin, ymin, zmin, xmax, ymax, zmax = self.boundary
        return xmin <= p[0] <= xmax and ymin <= p[1] <= ymax and zmin <= p[2] <= zmax

    def in_obstacle(self, p):
        for obs in self.obstacles:
            oxmin, oymin, ozmin, oxmax, oymax, ozmax = obs
            if (oxmin <= p[0] <= oxmax and
                oymin <= p[1] <= oymax and
                ozmin <= p[2] <= ozmax):
                return True
        return False

    def get_near_nodes(self, new_node):
        n = len(self.nodes) + 1
        r = self.neighbor_radius * np.sqrt((np.log(n) / n))
        r = min(r, self.neighbor_radius)
        return [node for node in self.nodes if np.linalg.norm(node.pos - new_node.pos) <= r]

    def choose_parent(self, near_nodes, nearest_node, new_node):
        best_cost = nearest_node.cost + np.linalg.norm(nearest_node.pos - new_node.pos)
        best_node = nearest_node
        for node in near_nodes:
            if self.collision_free(node.pos, new_node.pos):
                cost = node.cost + np.linalg.norm(node.pos - new_node.pos)
                if cost < best_cost:
                    best_cost = cost
                    best_node = node
        new_node.parent = best_node
        new_node.cost = best_cost
        return new_node

    def rewire(self, new_node, near_nodes):
        for node in near_nodes:
            if self.collision_free(new_node.pos, node.pos):
                cost = new_node.cost + np.linalg.norm(new_node.pos - node.pos)
                if cost < node.cost:
                    node.parent = new_node
                    node.cost = cost

    def extract_path(self):
        path = []
        node = self.goal
        while node is not None:
            path.append(node.pos)
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

def plot_environment(boundary, obstacles, start, goal, path, tree_nodes):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xmin, ymin, zmin, xmax, ymax, zmax = boundary
    for obs in obstacles:
        draw_box(ax, obs, color='gray')
    for node in tree_nodes:
        if node.parent is not None:
            pts = np.vstack((node.pos, node.parent.pos))
            ax.plot(pts[:,0], pts[:,1], pts[:,2], color='blue', linewidth=0.5)
    ax.scatter(start[0], start[1], start[2], color='green', s=100, label='Start')
    ax.scatter(goal[0], goal[1], goal[2], color='red', s=100, label='Goal')
    if path is not None:
        path = np.array(path)
        ax.plot(path[:,0], path[:,1], path[:,2], color='magenta', linewidth=2, label='Path')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('RRT* 3D Path Planning')
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
    
    planner = RRTStar3D(boundary, obstacles, start, goal, step_size=0.5, max_iter=2000, goal_sample_rate=0.1, neighbor_radius=2.0)
    path, time_taken = planner.plan()
    if path is None:
        print("No path found!")
    else:
        path_length = sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i])) for i in range(len(path)-1))
        print(f"RRT* 3D: Time taken: {time_taken:.2f} sec, Path length: {path_length:.2f}")
    plot_environment(boundary, obstacles, start, goal, path, planner.nodes)

if __name__ == '__main__':
    main()
