import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import pickle

# Helper function for displaying two images in parallel to compare
def displayTwo(orig_img, new_img, orig_title, new_title, save_name=''):
    plt.figure(figsize=(12, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    plt.title(orig_title)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
    plt.title(new_title)
    plt.axis('off')
    if save_name != '':
        plt.savefig(save_name, bbox_inches='tight')
    plt.show(block=True)

def undistort(img):
    return cv2.undistort(img, mtx, dist, None, mtx);

# Read in the saved camera calibration matrix and distortion cofficients
dist_pickle = pickle.load(open("calibration.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# List of all the test images
testFiles = glob.glob('./test_images/*.jpg')

# Original images
orig_images = []
for fname in testFiles:
    orig_images.append(cv2.imread(fname))
# Undistorted images
undistorted_images = []
for img in orig_images:
    undist = undistort(img)
    undistorted_images.append(undist)
    #displayTwo(img, undist, 'original', 'undistorted')

example = len(orig_images) - 2;
displayTwo(orig_images[example], undistorted_images[example], 'original', 'undistorted', 'undistorted.png');


