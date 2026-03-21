import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

class Node:
    def __init__(self, position, parent=None):
        self.position = np.array(position)
        self.parent = parent
        self.cost = 0  # Cost from start to this node

class RRTStar3D:
    def __init__(self, boundary, obstacles, start, goal, step_size=2, max_iter=1000, search_radius=10):
        self.boundary = boundary
        self.obstacles = obstacles
        self.start = Node(start)
        self.goal = Node(goal)
        self.step_size = step_size
        self.max_iter = max_iter
        self.search_radius = search_radius
        self.nodes = [self.start]
        self.zmin = start[2]  # Constraining to ground level
    
    def sample_point(self):
        x = random.uniform(self.boundary[0][0], self.boundary[1][0])
        y = random.uniform(self.boundary[0][1], self.boundary[1][1])
        return np.array([x, y, self.zmin])  # Keeping z fixed

    def nearest_node(self, point):
        return min(self.nodes, key=lambda node: np.linalg.norm(node.position - point))
    
    def steer(self, from_node, to_point):
        direction = to_point - from_node.position
        distance = np.linalg.norm(direction)
        if distance < self.step_size:
            return to_point
        return from_node.position + (direction / distance) * self.step_size

    def is_collision_free(self, from_point, to_point):
        for obs in self.obstacles:
            if self.line_intersects_box(from_point, to_point, obs):
                return False
        return True
    
    def line_intersects_box(self, p1, p2, box):
        """ Checks if a line segment intersects an axis-aligned bounding box """
        min_bounds, max_bounds = np.array(box[0]), np.array(box[1])
        direction = p2 - p1
        t_min, t_max = 0, 1
        for i in range(3):
            if abs(direction[i]) < 1e-6:
                if p1[i] < min_bounds[i] or p1[i] > max_bounds[i]:
                    return True
            else:
                t1 = (min_bounds[i] - p1[i]) / direction[i]
                t2 = (max_bounds[i] - p1[i]) / direction[i]
                t_min, t_max = max(t_min, min(t1, t2)), min(t_max, max(t1, t2))
                if t_min > t_max:
                    return True
        return False
    
    def find_nearby_nodes(self, new_node):
        return [node for node in self.nodes if np.linalg.norm(node.position - new_node.position) < self.search_radius]
    
    def rewire(self, new_node, nearby_nodes):
        for node in nearby_nodes:
            if self.is_collision_free(node.position, new_node.position):
                new_cost = node.cost + np.linalg.norm(node.position - new_node.position)
                if new_cost < new_node.cost:
                    new_node.parent = node
                    new_node.cost = new_cost
    
    def plan(self):
        for _ in range(self.max_iter):
            rand_point = self.sample_point()
            nearest = self.nearest_node(rand_point)
            new_position = self.steer(nearest, rand_point)
            if self.is_collision_free(nearest.position, new_position):
                new_node = Node(new_position, nearest)
                new_node.cost = nearest.cost + np.linalg.norm(new_node.position - nearest.position)
                self.nodes.append(new_node)
                self.rewire(new_node, self.find_nearby_nodes(new_node))
                if np.linalg.norm(new_node.position - self.goal.position) < self.step_size:
                    self.goal.parent = new_node
                    return self.extract_path()
        return None
    
    def extract_path(self):
        path = []
        node = self.goal
        while node:
            path.append(node.position)
            node = node.parent
        return path[::-1]
    
    def visualize(self, path):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(self.boundary[0][0], self.boundary[1][0])
        ax.set_ylim(self.boundary[0][1], self.boundary[1][1])
        ax.set_zlim(self.boundary[0][2], self.boundary[1][2])
        
        for obs in self.obstacles:
            ax.bar3d(obs[0][0], obs[0][1], obs[0][2],
                     obs[1][0] - obs[0][0], obs[1][1] - obs[0][1], obs[1][2] - obs[0][2], color='gray', alpha=0.5)
        
        for node in self.nodes:
            if node.parent:
                ax.plot([node.position[0], node.parent.position[0]],
                        [node.position[1], node.parent.position[1]],
                        [node.position[2], node.parent.position[2]], 'b-')
        
        if path:
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], path[:, 2], 'm-', linewidth=3)
        
        ax.scatter(*self.start.position, color='g', s=100, label='Start')
        ax.scatter(*self.goal.position, color='r', s=100, label='Goal')
        ax.legend()
        plt.show()

# Example Usage
def load_environment(filename):
    boundary, obstacles, start, goal = None, [], None, None
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('#') or not line.strip():
                continue
            values = list(map(float, line.split(',')))
            if len(values) == 6 and boundary is None:  # Corrected condition for boundary
                boundary = [(values[0], values[1], values[2]), (values[3], values[4], values[5])]
            elif len(values) == 6:
                obstacles.append([(values[0], values[1], values[2]), (values[3], values[4], values[5])])
            elif len(values) == 3:
                if start is None:
                    start = values
                else:
                    goal = values
    return boundary, obstacles, start, goal

filename = "BigEnv.txt"
boundary, obstacles, start, goal = load_environment(filename)
rrt_star = RRTStar3D(boundary, obstacles, start, goal)
path = rrt_star.plan()
rrt_star.visualize(path)