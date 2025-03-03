import cv2
import numpy as np
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the dictionary and parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
parameters = aruco.DetectorParameters()

# Real-world coordinates of markers (example)
markers_used = [4,19]
real_world_coords = {   #red= x , green = y, blue = z
    4: np.array([0,0,0]),  # Marker ID 4 at (1.25, 0.05, 0)  in video4  , origin in mark2 
    19: np.array([29*0.3+4.5*1.7,-0.3*8-1.25*3-11*0.3, 0]),  # Marker ID 19 at origin     in video4 , (1.33,0, 0) in mark2  0.6*5, 0.6*7, 0 in 5-7y small_hall 0.606
    # Add more markers as needed
}

# Camera matrix and distortion coefficients 
camera_matrix = np.array([[1.68197223e+03, 0.00000000e+00, 5.34811763e+02],
                          [0.00000000e+00, 1.68805220e+03, 9.53559018e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],dtype=np.float64)

dist_coeffs = np.array([2.20714677e-01 ,-1.31468706e+00 ,-3.90773293e-03 ,-3.56503849e-04 ,2.85959236e+00])  

cap = cv2.VideoCapture('my_vid\Faculty_z_2.mp4')
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)
print("video captured good:")
print(cap.isOpened())
camera_translations =[]
camera_rotation_vectors=[]
seen_ids=[]
frame_num=0

seen_frame=[]
camera_positions_world = []
squareLength=0.2
object_locations = np.array([(-squareLength / 2, squareLength / 2, 0),
                          (squareLength / 2, squareLength / 2, 0),
                          (squareLength / 2, -squareLength / 2, 0),
                          (-squareLength / 2, -squareLength / 2, 0)],dtype=np.double)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    if ids is not None:
        # Estimate pose for each marker
        if ids in markers_used:
            for i, corner in zip(ids.flatten(), corners):
                rows_list = np.array([corner[0][0],
                            corner[0][1],
                            corner[0][2],
                            corner[0][3]],dtype=np.double)
                #returns the rotation and the translation vectors that transform a 3D point 
                # expressed in the object coordinate frame to the camera coordinate frame
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corner, 0.20, camera_matrix, dist_coeffs)
                #_, rvec, tvec = cv2.solvePnP(object_locations,rows_list, camera_matrix, dist_coeffs,flags = cv2.SOLVEPNP_ITERATIVE)
                # Draw markers and axes
                aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.drawFrameAxes( frame, camera_matrix, dist_coeffs, rvec, tvec, length=0.2 )
                # Print or store the pose
                camera_translations.append(np.array(tvec).flatten())
                camera_rotation_vectors.append(np.array(rvec).flatten())
                seen_frame.append(frame_num)
                R = cv2.Rodrigues(np.array(rvec).flatten())[0]
                p=-np.dot((R.T),np.array(tvec).flatten())
                """
                if seen_ids:
                    if(seen_ids[-1]==i):
                        if camera_positions_world:
                            last_one=camera_positions_world[-1]
                            if (last_one*(p+real_world_coords[int(i)])<0).any():
                                continue
                """
                camera_positions_world.append(p+real_world_coords[int(i)])
                seen_ids.append(np.array(i))
                print(f"Marker ID {i}: Rotation Vector = {np.array(rvec)}, Translation Vector = {np.array(tvec)}")
                print("position in world:")
                print(p+real_world_coords[int(i)])
                print("position to marker:")
                print(p)
                

    else:
        print("no marker found")
    img = cv2.resize(frame, (500, 890), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('Frame', img)
    frame_num=frame_num+1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#MAP
def create_transformation_matrix(rotation_vector, translation_vector):
    """
    Create a 4x4 transformation matrix from rotation vector and translation vector.
    
    Parameters:
        rotation_vector (np.array): Rotation vector (axis-angle) with shape (3,).
        translation_vector (np.array): Translation vector with shape (3,).
    
    Returns:
        np.array: 4x4 Transformation matrix.
    """
    # Compute the rotation matrix
    R = cv2.Rodrigues(rotation_vector)[0]
    # Create the 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation_vector
    
    return T
#camera_positions_world = []
#for T_c, rvec,current_id in zip(camera_translations, camera_rotation_vectors, seen_ids):
    #print(current_id)
    #transformation_frame=create_transformation_matrix(rvec,T_c)
    
    #vec=np.dot(transformation_frame,real_world_coords[int(current_id)])
    # Normalize each vec by its 4th column
    #camera_positions_world.append(vec / vec[3])
    #camera_positions_world.append(T_c)
    #R = cv2.Rodrigues( rvec)[0]
    #p=-np.dot((R.T),T_c)
    #camera_positions_world.append(p+real_world_coords[int(current_id)])


# Convert to NumPy array for plotting
camera_positions_world = np.array(camera_positions_world)


# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.yticks(np.arange(-5, 5, 0.5))
ax.scatter(camera_positions_world[:,0], camera_positions_world[:,1], camera_positions_world[:,2])
#for i in range(0,len(camera_translations[:])):
 #   ax.scatter(camera_translations[i][0][0], camera_translations[i][0][1], camera_translations[i][0][2])

#draw the centers of the markers
for i in markers_used:
    ax.scatter(real_world_coords[i][0],real_world_coords[i][1],real_world_coords[i][2],marker='x')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()



