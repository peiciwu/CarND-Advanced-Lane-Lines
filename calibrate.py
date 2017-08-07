import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import pickle

# chessboard gird size
chessboard_nx = 9
chessboard_ny = 6

# Arrays to store object points and image points from all the calibration images.
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image space

# Prepare object points like (0, 0, 0), (1, 0, 0), (2, 0, 0), ..., (chessboard_nx-1, chessboard_ny-1, 0)
objp = np.zeros([chessboard_nx*chessboard_ny, 3], np.float32);
objp[:,:2] = np.mgrid[0:chessboard_nx, 0:chessboard_ny].T.reshape(-1, 2)

# Make a list of calibration images
images = glob.glob('./camera_cal/calibration*.jpg')

plt.figure(figsize=(12, 10)) # to display images

for i, fname in enumerate(images):
    img = cv2.imread(fname);
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (chessboard_nx, chessboard_ny), None)
        
    # If corners are found, add object points, image points
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)

        # draw the corners
        img = cv2.drawChessboardCorners(img, (chessboard_nx, chessboard_ny), corners, ret)

    plt.subplot(5, 4, i+1);
    plt.imshow(img)
    plt.title(i+1);
    plt.axis('off')

plt.savefig('corners.png', bbox_inches='tight')
plt.show(block=True)

# Camera calibration using object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, cv2.imread(images[0]).shape[0:2], None, None)

# Test on one of the images
img = cv2.imread(images[0]);
undist = cv2.undistort(img, mtx, dist, None, mtx);
plt.figure(figsize=(12, 10))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(undist)
plt.title('Undistorted')
plt.axis('off')
plt.savefig('undistort.png', bbox_inches='tight')
plt.show(block=True)

# Dump to pickle for the future use
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open("calibration.p", "wb"))
