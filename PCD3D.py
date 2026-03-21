import numpy as np
import matplotlib.pyplot as plt
import heapq, time

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

def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def is_in_obstacle(point, obstacles):
    for obs in obstacles:
        if obs[0] <= point[0] <= obs[3] and obs[1] <= point[1] <= obs[4]:
            return True
    return False

def get_neighbors(node, grid_shape):
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = node[0] + dx, node[1] + dy
        if 0 <= nx < grid_shape[0] and 0 <= ny < grid_shape[1]:
            neighbors.append((nx, ny))
    return neighbors

def a_star_search(start, goal, grid_shape, boundary, resolution, obstacles):
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start))
    came_from = {}
    cost_so_far = {start: 0}
    while open_set:
        _, cost, current = heapq.heappop(open_set)
        if current == goal:
            break
        for neighbor in get_neighbors(current, grid_shape):
            if is_in_obstacle(neighbor, obstacles):
                continue
            new_cost = cost_so_far[current] + heuristic(current, neighbor)
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                heapq.heappush(open_set, (new_cost + heuristic(neighbor, goal), new_cost, neighbor))
                came_from[neighbor] = current
    if goal not in came_from:
        return None
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    return path[::-1]

def plot_results(boundary, obstacles, start, goal, path, resolution, time_taken, path_length):
    plt.figure(figsize=(8, 6))
    plt.xlim(boundary[0], boundary[3])
    plt.ylim(boundary[1], boundary[4])
    for obs in obstacles:
        plt.fill([obs[0], obs[3], obs[3], obs[0]], [obs[1], obs[1], obs[4], obs[4]], 'k', alpha=0.5)
    plt.scatter(start[0], start[1], c='g', marker='o', label='Start')
    plt.scatter(goal[0], goal[1], c='r', marker='o', label='Goal')
    if path:
        path_np = np.array(path)
        plt.plot(path_np[:, 0], path_np[:, 1], 'b-', label='Path')
    plt.title(f"DStar3D A* Path Planning\nTime: {time_taken:.2f}s | Path Length: {path_length:.2f}")
    plt.legend()
    plt.show()

def main():
    env_file = "BigEnv.txt"
    boundary, obstacles, start, goal = load_environment(env_file)
    resolution = 0.5
    grid_shape = (int((boundary[3] - boundary[0]) / resolution),
                  int((boundary[4] - boundary[1]) / resolution))
    start_idx = (int((start[0] - boundary[0]) / resolution),
                 int((start[1] - boundary[1]) / resolution))
    goal_idx = (int((goal[0] - boundary[0]) / resolution),
                int((goal[1] - boundary[1]) / resolution))
    t0 = time.time()
    path = a_star_search(start_idx, goal_idx, grid_shape, boundary, resolution, obstacles)
    time_taken = time.time() - t0
    if path is None:
        print("No path found!")
        return
    path_length = sum(heuristic(path[i], path[i+1]) for i in range(len(path)-1))
    plot_results(boundary, obstacles, start, goal, path, resolution, time_taken, path_length)

if __name__ == '__main__':
    main()
