import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to parse input data from a text file
def parse_input_data_from_file(file_path: str):
    # Read the content of the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Initialize lists for storing positions and other data
    timestamps = []
    positions = []
    
    # Process each line
    for line in lines:
        # Split each line into values and convert to float
        values = list(map(float, line.split()))
        
        # Extract timestamp (first column) and position (next 3 columns)
        timestamp = values[0]
        position = values[1:4]  # x, y, z
        
        timestamps.append(timestamp)
        positions.append(position)
    
    # Convert lists to numpy arrays for easy manipulation
    timestamps = np.array(timestamps)
    positions = np.array(positions)
    
    return timestamps, positions

# Example file path (update with your file location)
file_path = "slam_output/f_dataset-faculty_Z_3.txt"  # Replace with the path to your text file

# Parse the input data
timestamps, positions = parse_input_data_from_file(file_path)

# Plotting the trajectory (3D scatter plot)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract x, y, z positions
x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

# Plot the trajectory
ax.plot(x, y, z, marker='o', color='b', label="Trajectory")

# Labels and title
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('SLAM Trajectory (Position)')

# Show the plot
plt.legend()
plt.show()
