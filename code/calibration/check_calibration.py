import cv2
import matplotlib.pyplot as plt
import numpy as np

camera_matrix = np.array([[1.68197223e+03, 0.00000000e+00, 5.34811763e+02],
                          [0.00000000e+00, 1.68805220e+03, 9.53559018e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],dtype=np.float64)

dist_coeffs = np.array([2.20714677e-01 ,-1.31468706e+00 ,-3.90773293e-03 ,-3.56503849e-04 ,2.85959236e+00])  
# Load a real image
real_image = cv2.imread('calibration/test_calib.png')
print(real_image)
# Undistort the image
undistorted_image = cv2.undistort(real_image, camera_matrix, dist_coeffs)

# Display undistorted image
plt.imshow(cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB))
plt.title('Undistorted Image')
plt.show()