import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
import statistics


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
square_size=0.025
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((10*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:10].T.reshape(-1,2) * square_size
print(objp)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
 
images = glob.glob('calibration/Calibration_vid_frames/*.png')
img = cv.imread(images[0])
imgsize=img.shape

for fname in images:
   img = cv.imread(fname)
   print(fname)
   gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
   # Find the chess board corners
   ret, corners = cv.findChessboardCorners(gray, (7,10), None)
   print(ret)
   # If found, add object points, image points (after refining them)
   if ret:
      objpoints.append(objp)
   
      corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
      imgpoints.append(corners2)
      
      # Draw and display the corners
      #cv.drawChessboardCorners(img, (7,6), corners2, ret)
      #cv.imshow('img', img)
      #cv.waitKey(500)
 
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Displaying required output 
print(" Camera matrix:") 
print(mtx) 
  
print("\n Distortion coefficient:") 
print(dist) 
  
print("\n Rotation Vectors:") 
#print(rvecs) 
  
print("\n Translation Vectors:") 
#print(tvecs) 

mean_error = 0
error_plot=[]
for i in range(len(objpoints)):
 imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
 error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
 error_plot.append(error)
 mean_error += error
 
print( "Mean error: {}".format(mean_error/len(objpoints)) )
print( "standard deviation: {}".format(statistics.stdev(error_plot)) )


xpoints = np.array([0, 8])
ypoints = np.array([0, 10])

print("\n img points:") 
print(len(imgpoints))
print(imgpoints[0].shape)
print(imgpoints[0][:,0,0])

x=[]
y=[]
for i in range(len(imgpoints)): #pic,point,0,x or y.
   x.append(imgpoints[i][:,0,0])
   y.append(imgpoints[i][:,0,1])


colors = np.random.rand(len(imgpoints))
# Create a customized scatter plot
for i in range(len(imgpoints)):
   plt.scatter(imgpoints[i][:,0,0], imgpoints[i][:,0,1], s=10, alpha=0.7, cmap='viridis')
 
# Add title and axis labels
plt.title("Corners in every image Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
 
# Display color intensity scale
plt.colorbar(label='Color Intensity')
 
# Show the plot
plt.show()





