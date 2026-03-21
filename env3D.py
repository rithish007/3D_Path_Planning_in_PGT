import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from perlin_noise import PerlinNoise

def parse_environment(file_path):
    boundary = None
    blocks = []
    
    with open(file_path, 'r') as file:
        for line in file:
            tokens = line.strip().split()
            if not tokens:
                continue
            if tokens[0] == 'boundary':
                boundary = list(map(float, tokens[1:7]))
            elif tokens[0] == 'block':
                blocks.append(list(map(float, tokens[1:7])))
    
    return boundary, blocks

def generate_perlin_noise(width, depth, scale=10):
    noise = PerlinNoise(octaves=3, seed=1)
    noise_grid = np.zeros((width, depth))
    for i in range(width):
        for j in range(depth):
            noise_grid[i][j] = noise([i / scale, j / scale])
    return noise_grid

def plot_environment(boundary, blocks, noise_scale=7):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate terrain using Perlin noise
    width = int((boundary[3] - boundary[0]) * noise_scale)
    depth = int((boundary[4] - boundary[1]) * noise_scale)
    noise_grid = generate_perlin_noise(width, depth)
    x = np.linspace(boundary[0], boundary[3], width)
    y = np.linspace(boundary[1], boundary[4], depth)
    X, Y = np.meshgrid(x, y)
    Z = np.array(noise_grid).T * 0.5  # Scale noise height
    
    ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.6)
    
    # Plot boundary
    bx, by, bz, ex, ey, ez = boundary
    ax.plot([bx, ex, ex, bx, bx], [by, by, ey, ey, by], [bz, bz, bz, bz, bz], color='black', linewidth=2)
    
    # Plot blocks
    for block in blocks:
        bx, by, bz, ex, ey, ez = block
        vertices = [
            [(bx, by, bz), (ex, by, bz), (ex, ey, bz), (bx, ey, bz)],
            [(bx, by, ez), (ex, by, ez), (ex, ey, ez), (bx, ey, ez)],
            [(bx, by, bz), (bx, by, ez), (bx, ey, ez), (bx, ey, bz)],
            [(ex, by, bz), (ex, by, ez), (ex, ey, ez), (ex, ey, bz)],
            [(bx, by, bz), (ex, by, bz), (ex, by, ez), (bx, by, ez)],
            [(bx, ey, bz), (ex, ey, bz), (ex, ey, ez), (bx, ey, ez)],
        ]
        ax.add_collection3d(Poly3DCollection(vertices, color='gray', alpha=0.8))
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Environment Visualization with Perlin Noise')
    
    plt.show()

if __name__ == "__main__":
    boundary, blocks = parse_environment("env.txt")
    if boundary:
        plot_environment(boundary, blocks)