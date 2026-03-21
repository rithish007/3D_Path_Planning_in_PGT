#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Voronoi
import networkx as nx
import time

# ------------------- Environment Loader -------------------
def load_environment(filename):
    boundary = None
    obstacles = []
    start = None
    goal = None
    with open(filename, 'r') as f:
        for line in f:
            line=line.strip()
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

def get_corners(boundary, obstacles):
    points = []
    xmin, ymin, zmin, xmax, ymax, zmax = boundary
    boundary_corners = [
        [xmin, ymin, zmin],
        [xmax, ymin, zmin],
        [xmax, ymax, zmin],
        [xmin, ymax, zmin],
        [xmin, ymin, zmax],
        [xmax, ymin, zmax],
        [xmax, ymax, zmax],
        [xmin, ymax, zmax]
    ]
    points.extend(boundary_corners)
    for obs in obstacles:
        oxmin, oymin, ozmin, oxmax, oymax, ozmax = obs
        obs_corners = [
            [oxmin, oymin, ozmin],
            [oxmax, oymin, ozmin],
            [oxmax, oymax, ozmin],
            [oxmin, oymax, ozmin],
            [oxmin, oymin, ozmax],
            [oxmax, oymin, ozmax],
            [oxmax, oymax, ozmax],
            [oxmin, oymax, ozmax]
        ]
        points.extend(obs_corners)
    return np.array(points)

def is_collision_free(p1, p2, obstacles, num_samples=10):
    for i in range(num_samples+1):
        t = i/num_samples
        pt = p1 + t*(p2-p1)
        for obs in obstacles:
            oxmin, oymin, ozmin, oxmax, oymax, ozmax = obs
            if oxmin <= pt[0] <= oxmax and oymin <= pt[1] <= oymax and ozmin <= pt[2] <= ozmax:
                return False
    return True

def build_voronoi_graph(vor, obstacles, boundary):
    G = nx.Graph()
    for i, v in enumerate(vor.vertices):
        xmin, ymin, zmin, xmax, ymax, zmax = boundary
        if xmin <= v[0] <= xmax and ymin <= v[1] <= ymax and zmin <= v[2] <= zmax:
            G.add_node(i, pos=v)
    for ridge in vor.ridge_vertices:
        if -1 in ridge:
            continue
        if ridge[0] in G.nodes and ridge[1] in G.nodes:
            p1 = vor.vertices[ridge[0]]
            p2 = vor.vertices[ridge[1]]
            if is_collision_free(p1, p2, obstacles):
                dist = np.linalg.norm(p1-p2)
                G.add_edge(ridge[0], ridge[1], weight=dist)
    return G

def add_point_to_graph(G, point, obstacles):
    node_id = len(G.nodes)
    G.add_node(node_id, pos=point)
    for i in list(G.nodes):
        if i == node_id:
            continue
        v = G.nodes[i]['pos']
        if is_collision_free(point, v, obstacles):
            dist = np.linalg.norm(point-v)
            G.add_edge(node_id, i, weight=dist)
    return node_id

def draw_box(ax, box, color='black', alpha=0.5):
    xmin, ymin, zmin, xmax, ymax, ozmax = box
    corners = np.array([
        [box[0], box[1], box[2]],
        [box[3], box[1], box[2]],
        [box[3], box[4], box[2]],
        [box[0], box[4], box[2]],
        [box[0], box[1], box[2]+(box[5]-box[2])],
        [box[3], box[1], box[2]+(box[5]-box[2])],
        [box[3], box[4], box[2]+(box[5]-box[2])],
        [box[0], box[4], box[2]+(box[5]-box[2])]
    ])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for e in edges:
        pts = corners[list(e), :]
        ax.plot(pts[:,0], pts[:,1], pts[:,2], color=color, alpha=alpha)

def plot_environment(boundary, obstacles, start, goal, path, vor, G):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xmin, ymin, zmin, xmax, ymax, zmax = boundary
    for obs in obstacles:
        draw_box(ax, obs, color='gray')
    ax.scatter(start[0], start[1], start[2], color='green', s=100, label='Start')
    ax.scatter(goal[0], goal[1], goal[2], color='red', s=100, label='Goal')
    ax.scatter(vor.vertices[:,0], vor.vertices[:,1], vor.vertices[:,2], color='blue', s=10)
    for edge in G.edges():
        p1 = G.nodes[edge[0]]['pos']
        p2 = G.nodes[edge[1]]['pos']
        ax.plot([p1[0], p2[0]],[p1[1], p2[1]],[p1[2], p2[2]], color='cyan', linewidth=0.5)
    if path is not None:
        path = np.array(path)
        ax.plot(path[:,0], path[:,1], path[:,2], color='magenta', linewidth=2, label='Path')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Voronoi Diagram Path Planning')
    ax.legend()
    plt.show()

def main():
    env_file = "environment.txt"
    boundary, obstacles, start, goal = load_environment(env_file)
    print("Environment loaded:")
    print("Boundary:", boundary)
    print("Obstacles:", obstacles)
    print("Start:", start, "Goal:", goal)
    points = get_corners(boundary, obstacles)
    t0 = time.time()
    vor = Voronoi(points)
    G = build_voronoi_graph(vor, obstacles, boundary)
    start_id = add_point_to_graph(G, np.array(start), obstacles)
    goal_id = add_point_to_graph(G, np.array(goal), obstacles)
    try:
        path_nodes = nx.shortest_path(G, source=start_id, target=goal_id, weight='weight')
        path = [G.nodes[n]['pos'] for n in path_nodes]
    except nx.NetworkXNoPath:
        path = None
    t_elapsed = time.time()-t0
    if path is None:
        print("No path found!")
    else:
        path_length = sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i])) for i in range(len(path)-1))
        print(f"Voronoi: Time taken: {t_elapsed:.2f} sec, Path length: {path_length:.2f}")
    plot_environment(boundary, obstacles, np.array(start), np.array(goal), path, vor, G)

if __name__ == '__main__':
    main()
