# Abstract

Process a video captured while moving through an environment to output an accurate 3D trajectory of the movement path relative to a known map. By leveraging prior knowledge of the environment and using ArUco markers distributed throughout, we developed a robust system for reconstructing 3D trajectories from video data. The proposed solution employs a hybrid approach combining ArUco marker-based localization and ORB-SLAM3. ArUco markers provide reliable pose estimation, including orientation and location, whenever visible. When markers are not detected, ORB-SLAM3 seamlessly takes over to continue tracking and mapping. By integrating ArUco-derived pose information as ground truth for ORB-SLAM3, the system achieves enhanced accuracy and robustness across diverse scenarios.


docs – מסמך אפיון ודו”ח סופי .

poster – פוסטר .

code – קבצי קוד  

data – מוסיקה שמשומשת לאימון

results – תוצרי הפרויקט.


How to run:
1.	Calibrate camera
2.	Run video on ORBSLAM3 and save output
3.	Go to aruco_try and type the mp4 location file, you can run to see if the aruco part works well
4.	Go to slam preprocess, make sure the SLAM output is saved in the right place run it.
5.	Go to main_new_new_new ,  run it.
