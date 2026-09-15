import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_solid_angle():

    # Define theta and phi ranges
    theta = np.radians(6.3) * np.ones(1)  # Convert degrees to radians
    phi = np.radians(45) * np.ones(1)  # Full circle in phi

    # Define a constant radius r = 1
    r = np.linspace(139, 149, 20)  # Radius from 0 to 200 with 100 points

    # Create a meshgrid for theta and phi
    theta, phi = np.meshgrid(theta, phi)

    # Convert spherical coordinates to Cartesian coordinates
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    print(z[0, 1])
    # Define rotation matrix for 30-degree tilt in the YZ-plane
    angle = np.radians(90)  # Convert 30 degrees to radians
    rotation_matrix = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])

    # Apply rotation to the coordinates (ZY-plane rotation)
    coords = np.array([x.flatten(), y.flatten(), z.flatten()])
    rotated_coords = rotation_matrix @ coords

    # Reshape rotated coordinates back to original shape
    x_rotated = rotated_coords[0].reshape(x.shape)
    y_rotated = rotated_coords[1].reshape(y.shape)
    z_rotated = rotated_coords[2].reshape(z.shape)
    
    print(z_rotated[0, 1])
    # Remove axis swapping
    # x_rotated, y_rotated = y_rotated, x_rotated
    # for i in range(z_rotated.shape[0]):
    #     for j in range(z_rotated.shape[1]):
    #         if z_rotated[i, j] > 200: # Set z values greater than 200 to 200
    #             z_rotated[i, j] = 200

    # Plot the solid angle
    fig = plt.figure(figsize=(12, 6))

    # 3D plot
    ax3d = fig.add_subplot(121, projection='3d')
    ax3d.scatter(x_rotated, y_rotated, z_rotated, alpha=0.7, edgecolor='k', cmap='viridis')
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.set_title('3D Solid Angle')

    # Plot the original and rotated axes
    ax3d.quiver(0, 0, 0, 0, 0, max(z_rotated[0, :]), color='r', label='Original Z-axis')  # Original Z-axis
    ax3d.quiver(0, 0, 0, max(z_rotated[0, :]) * rotation_matrix[0, 2], max(z_rotated[0, :]) * rotation_matrix[1, 2], max(z_rotated[0, :]) * rotation_matrix[2, 2], color='b', label='Rotated Z-axis')  # Rotated Z-axis

    # Add legend to the 3D plot
    ax3d.legend()

    # Correct 2D projection on the original X and Y axes
    ax2d = fig.add_subplot(122)
    ax2d.scatter(x_rotated.flatten(), z_rotated.flatten(), lw=2, color='darkblue')
    ax2d.set_xlabel('X')
    ax2d.set_ylabel('Z')
    ax2d.set_title('2D Projection (XZ-plane)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    plot_solid_angle()