import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Aruco.aruco_try import seen_ids , seen_frame , camera_translations , camera_rotation_vectors, create_transformation_matrix, camera_positions_world, markers_used,real_world_coords
from integration.preprocess_slam import frames_num_slam , transformation_matrices_slam , start_frame_slam
from integration.check_id import find_consecutive_id_index

def find_closest(sorted_list, target):
    if not sorted_list:
        return None
    
    low, high = 0, len(sorted_list) - 1
    
    # Binary search
    while low <= high:
        mid = (low + high) // 2
        if sorted_list[mid] < target:
            low = mid + 1
        elif sorted_list[mid] > target:
            high = mid - 1
        else:
            return sorted_list[mid]
    
    # After binary search, low is the index where the target would be inserted
    # Check the closest element among sorted_list[low] and sorted_list[high]
    if low >= len(sorted_list):
        return sorted_list[-1]
    if high < 0:
        return sorted_list[0]
    
    if abs(sorted_list[low] - target) < abs(sorted_list[high] - target):
        return sorted_list[low]
    else:
        return sorted_list[high]

def compute_total_transformation(matrices, start_frame, end_frame):
    """
    Computes the total transformation from start_frame to end_frame.

    Parameters:
        matrices (list of np.ndarray): List of transformation matrices.
        start_frame (int): The starting frame index (0-based).
        end_frame (int): The ending frame index (1-based).

    Returns:
        np.ndarray: The total transformation matrix.
    """
    # Ensure the indices are within bounds
    if start_frame < 0 or end_frame > len(matrices) or start_frame >= end_frame:
        raise ValueError("Invalid frame indices")

    # Initialize total transformation as identity matrix
    total_transformation = np.eye(matrices[0].shape[0])

    # Multiply the transformation matrices in sequence
    for i in range(start_frame, end_frame):
        total_transformation = np.dot(matrices[i],total_transformation)
    
    return total_transformation

def inverse_matrix(matrix):
    R = matrix[:3, :3]
    t = matrix[:3, 3]
    R_inv = np.transpose(R)
    t_inv = -np.dot(R_inv, t)
    return np.vstack((np.hstack((R_inv, t_inv.reshape(-1, 1))), [0, 0, 0, 1]))

def location_camera_from_trans(matrix):
    R = matrix[:3, :3]
    t = matrix[:3, 3]
    R_inv = np.transpose(R)
    t_inv = -np.dot(R_inv, t)
    return t_inv

# the orb slam doesnt work when i make pure rotation!!!!
# the orb slam doesnt work when i make pure rotation!!!!
target_ids=[4,19]
index0 = find_consecutive_id_index(seen_ids, target_ids[0],consecutive_count=30)
index1 = find_consecutive_id_index(seen_ids, target_ids[1],consecutive_count=30)
if index0 != -1:
    print(f"The last index where ID {target_ids[0]} appears consecutively 20 times is: {index0}")
    target_frame_slam_0=find_closest(frames_num_slam,seen_frame[index0])-start_frame_slam
else:
    print(f"ID {target_ids[0]} does not appear consecutively 20 times.")

if index1 != -1:
    print(f"The last index where ID {target_ids[1]} appears consecutively 20 times is: {index1}")
    target_frame_slam_1=find_closest(frames_num_slam,seen_frame[index1])-start_frame_slam
else:
    print(f"ID {target_ids[1]} does not appear consecutively 20 times.")

print("The last index where ID 4 appears consecutively 30 times in slam:")
print(target_frame_slam_0)
print("The last index where ID 19 appears consecutively 30 times in slam:")
print(target_frame_slam_1)
print("length of slam frames:")
print(len(transformation_matrices_slam))

last_location_marker_1= camera_positions_world[index0-1]
print("camera 0 location:")
print(last_location_marker_1)
last_location_marker_2= camera_positions_world[index1-1]
print("camera n location:")
print(last_location_marker_2)

aruco0_to_camera0=create_transformation_matrix(camera_rotation_vectors[index0], camera_translations[index0])
print("aruco0 to camera0:")
print(aruco0_to_camera0)

camera0_to_aruco0=inverse_matrix(aruco0_to_camera0)
print("camera0 to aruco0:")
print(camera0_to_aruco0)

anchor_to_camera0=transformation_matrices_slam[target_frame_slam_0]
print("anchor to camera 0:")
print(anchor_to_camera0)

camera0_to_anchor=inverse_matrix(anchor_to_camera0)
print("camera 0 to anchor:")
print(camera0_to_anchor)

transformation_matrices_slam_new=[]
# Loop through the matrices
for i in range(0,len(transformation_matrices_slam) - 1):
    product = np.dot(camera0_to_anchor,transformation_matrices_slam[i])
    transformation_matrices_slam_new.append(product)

camera0_to_cameran=transformation_matrices_slam_new[target_frame_slam_1]
print("camera0 to camera n:")
print(camera0_to_cameran)

cameran_to_camera0=inverse_matrix(camera0_to_cameran)
print("camera n to camera 0:")
print(cameran_to_camera0)

cameran_slam= location_camera_from_trans(cameran_to_camera0)     
print("camera n slam:")
print(cameran_slam)

cameran_slam_true= aruco0_to_camera0@np.append(last_location_marker_2, 1)
print("camera n slam true:")
print(cameran_slam_true)

scale_factor_alpha = np.linalg.norm(cameran_slam_true) / np.linalg.norm(cameran_slam)
print("Transformation scale factor alpha:")
#print(scale_factor_alpha)

#alpha_2=(cameran_slam_true[:3]+10)/(cameran_slam+10)
alpha_2=(cameran_slam_true[:3])/(cameran_slam)
print("Transformation scale factor alpha 2:")
print(alpha_2)

aruco1_to_cameran=create_transformation_matrix(camera_rotation_vectors[index1-1], camera_translations[index1-1])
print("aruco1 to cameran:")
print(aruco1_to_cameran)

position_marker_2=real_world_coords[19]

alpha_3=(-aruco0_to_camera0[:3,:3]@(((aruco1_to_cameran[:3,:3]).T)@aruco1_to_cameran[:3,3])+aruco0_to_camera0[:3,:3]@(position_marker_2).T+aruco0_to_camera0[:3,3])/camera0_to_cameran[:3,3]
print("alpha 3:")
print(alpha_3)

location_camera_n_world=((aruco0_to_camera0[:3,:3]).T)@(alpha_3*camera0_to_cameran[:3,3]-aruco0_to_camera0[:3,3])
print("location camera n world with alpha 3:")
print(location_camera_n_world)
print("camera n true location:")
print(last_location_marker_2)
markers_2_location_direct=-(aruco1_to_cameran[:3,:3].T)@(aruco1_to_cameran[:3,3])+position_marker_2
print("camera n true location direct calculation:")
print(markers_2_location_direct)

slam_locations=np.array([np.append(last_location_marker_1, 1)])
for i in range(target_frame_slam_0,target_frame_slam_1):
    #print(i)
    temp_trans=transformation_matrices_slam_new[i]
    #temp_trans[:3,3]=-10+(temp_trans[:3,3]+10)#*alpha_2        #fixing scale. choose best alpha scale/3 or alpha 2/4
    temp_trans[:3,3]=temp_trans[:3,3]*alpha_3
    #temp=(camera0_to_aruco0@((temp_trans@np.array([0,0,0,1]))))
    temp=((aruco0_to_camera0[:3,:3]).T)@(temp_trans[:3,3]-aruco0_to_camera0[:3,3])
    slam_locations=np.append(slam_locations,[np.append(temp, 1)], axis=0)   #problem with scale: plus to minus. 0.1 close to -0.1, but scale makes them further apart

# 1  1.5  2/3   ,, -1  1  (-1)  

slam_locations_final=slam_locations

# Plotting

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(slam_locations_final[:,0], slam_locations_final[:,1], slam_locations_final[:,2])
ax.scatter(camera_positions_world[:,0], camera_positions_world[:,1], camera_positions_world[:,2])
for i in markers_used:
    ax.scatter(real_world_coords[i][0],real_world_coords[i][1],real_world_coords[i][2],marker='x')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
# Tn@...T2@T1@T_aruco_to_camera@aruco_location

print("end")
# Tn@...T2@T1@(x,y,z,1)(camera location)

