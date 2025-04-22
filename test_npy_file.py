import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def print_npy_values(npy_path):
    if not os.path.exists(npy_path):
        print(f"File not found: {npy_path}")
        return

    data = np.load(npy_path)
    print(f"Loaded .npy file: {npy_path}")
    print(f"Shape: {data.shape}, Dtype: {data.dtype}")
    # print("Data (truncated to 10x10):")
    # print(data[:10, :10])  # Print a small portion for readability
    print(data)
    plot_trajectory(data)

def plot_trajectory(points):
    if len(points) < 2:
        print("Not enough points to plot trajectory.")
        return

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Create segments for Line3DCollection
    segments = np.array([[points[i], points[i + 1]] for i in range(len(points) - 1)])

    # Normalize time steps for colormap
    t = np.linspace(0, 1, len(segments))
    colors = plt.cm.plasma(t)  # you can try 'viridis', 'cool', 'jet', etc.

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create the colored line segments
    lc = Line3DCollection(segments, colors=colors, linewidths=2)
    ax.add_collection3d(lc)

    # Equal scaling
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Trajectory Over Time (Blue → Red)')

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 print_npy.py path_to_npy_file")
    else:
        print_npy_values(sys.argv[1])
