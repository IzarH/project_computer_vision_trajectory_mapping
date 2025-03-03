import numpy as np
from scipy.spatial.transform import Rotation as R

def process_slam_output(file_path,frames_num,transformation_matrices):
    """
    Read and process SLAM output data from a text file to extract frame number, rotation matrix, and translation vector.
    
    Args:
        file_path (str): Path to the text file containing SLAM output data.
        
    Returns:
        None
    """
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into components
            parts = line.strip().split()
            
            # Extract frame number and convert to integer
            frame_number = int(float(parts[0]))
            
            # Extract quaternion and translation vector
            translation_vector= list(map(float, parts[1:4]))  # First 3 numbers are the translation vector
            quaternion  = list(map(float, parts[4:]))  # Last 4 numbers are the quaternion
            
            # Convert quaternion to rotation matrix
            rotation_matrix = R.from_quat(quaternion).as_matrix()
            
            # Construct the 4x4 transformation matrix
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = translation_vector
            
            # Print the results
            #print(f"Frame Number: {frame_number}")
            #print("Rotation Matrix:")
            #print(rotation_matrix)
            #print("Translation Vector:")
            #print(translation_vector)
            #print("Transformation Matrix:")
            #print(transformation_matrix)
            #print()  # Newline for separation
            frames_num.append(frame_number)
            transformation_matrices.append(transformation_matrix)

# Path to the text file containing the SLAM output
file_path = 'slam_output/f_dataset-faculty_Z_3.txt'  # path to file
frames_num_slam=[]
transformation_matrices_slam=[]
process_slam_output(file_path,frames_num_slam,transformation_matrices_slam)


with open('slam_output/kf_dataset-MH01_mono.txt', 'r') as file:
    # Read the first line
    first_line = file.readline().strip()
    # Split the line into parts and get the first number
    anchor = first_line.split()[0]
    # Convert to float
    float_number = float(anchor)
    # Convert to integer
    anchor = int(float_number)

print("Anchor frame:", anchor)

with open(file_path, 'r') as file:
    # Read the first line
    first_line = file.readline().strip()
    # Split the line into parts and get the first number
    start_frame_slam = first_line.split()[0]
    # Convert to float
    float_number = float(start_frame_slam)
    # Convert to integer
    start_frame_slam = int(float_number)

print("start frame slam:", start_frame_slam)



