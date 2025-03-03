import cv2
import numpy as np
import glob
import os

# Define the calibration pattern properties
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
checkerboard_size = (10, 7)
square_size = 2.5  # Square size in centimeters

# Prepare 3D points in real-world space
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Path to the folder containing calibration images
calibration_images_path = "calibration/Calibration_vid_frames/*.png"  # Adjust the path and extension as needed

# Load images from the folder
images = glob.glob(calibration_images_path)
for img_path in images:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

# Perform calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print the intrinsic parameters
print("Camera matrix:\n", camera_matrix)
print("Distortion coefficients:\n", dist_coeffs)
