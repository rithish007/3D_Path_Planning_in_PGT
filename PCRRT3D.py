import numpy as np
import matplotlib.pyplot as plt
import time, random, math

class Node:
    def __init__(self, point, parent=None):
        self.point = np.array(point)
        self.parent = parent

def load_environment(filename):
    boundary, obstacles, start, goal = None, [], None, None
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts or line.startswith('#'):
                continue
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

def point_in_obstacle(point, obstacles):
    for obs in obstacles:
        oxmin, oymin, _, oxmax, oymax, _ = obs
        if oxmin <= point[0] <= oxmax and oymin <= point[1] <= oymax:
            return True
    return False

def is_line_collision_free(p1, p2, obstacles, step_size=0.1):
    dist = np.linalg.norm(np.array(p2) - np.array(p1))
    steps = max(int(dist / step_size), 1)
    for i in range(steps + 1):
        t = i / steps
        point = np.array(p1) * (1 - t) + np.array(p2) * t
        if point_in_obstacle(point, obstacles):
            return False
    return True

def rrt_planning(start, goal, boundary, obstacles, max_iters=5000, step_size=0.5, goal_sample_rate=0.1):
    start_time = time.time()
    xmin, ymin, _, xmax, ymax, _ = boundary
    start, goal = [start[0], start[1]], [goal[0], goal[1]]
    start_node, goal_node = Node(start), Node(goal)
    tree = [start_node]

    for _ in range(max_iters):
        rnd = np.array(goal) if random.random() < goal_sample_rate else np.array([
            random.uniform(xmin, xmax), random.uniform(ymin, ymax)])
        nearest = min(tree, key=lambda n: np.linalg.norm(n.point - rnd))
        direction = (rnd - nearest.point) / np.linalg.norm(rnd - nearest.point)
        new_point = nearest.point + step_size * direction
        if not is_line_collision_free(nearest.point, new_point, obstacles):
            continue
        new_node = Node(new_point, nearest)
        tree.append(new_node)
        if np.linalg.norm(new_node.point - np.array(goal)) <= step_size:
            if is_line_collision_free(new_node.point, goal, obstacles):
                goal_node.parent = new_node
                tree.append(goal_node)
                elapsed_time = time.time() - start_time
                return goal_node, tree, elapsed_time
    return None, tree, time.time() - start_time

def reconstruct_path(goal_node):
    path, length = [], 0
    node = goal_node
    while node is not None:
        if node.parent:
            length += np.linalg.norm(node.point - node.parent.point)
        path.append(node.point)
        node = node.parent
    return path[::-1], length

def plot_environment(boundary, obstacles, start, goal, path, tree):
    fig, ax = plt.subplots()
    xmin, ymin, _, xmax, ymax, _ = boundary
    for obs in obstacles:
        oxmin, oymin, _, oxmax, oymax, _ = obs
        rect = plt.Rectangle((oxmin, oymin), oxmax - oxmin, oymax - oymin, color='black', alpha=0.5)
        ax.add_patch(rect)
    for node in tree:
        if node.parent is not None:
            p1, p2 = node.point, node.parent.point
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='orange', alpha=0.3)
    if path:
        path_array = np.array(path)
        ax.plot(path_array[:, 0], path_array[:, 1], color='blue', linewidth=2, label='Path')
    ax.scatter(start[0], start[1], color='green', s=100, label='Start')
    ax.scatter(goal[0], goal[1], color='red', s=100, label='Goal')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('RRT 3D - 2D Plot')
    ax.legend()
    plt.show()

def main():
    env_file = "BigEnv.txt"
    boundary, obstacles, start, goal = load_environment(env_file)
    goal_node, tree, time_taken = rrt_planning(start, goal, boundary, obstacles)
    if goal_node is None:
        print("No path found!")
        return
    path, path_length = reconstruct_path(goal_node)
    print(f"Path found! Time taken: {time_taken:.2f} seconds, Path length: {path_length:.2f}")
    plot_environment(boundary, obstacles, start, goal, path, tree)

if __name__ == '__main__':
    main()
