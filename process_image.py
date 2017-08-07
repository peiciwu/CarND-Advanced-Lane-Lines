import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

# Helper function for displaying two images in parallel to compare.
def plotTwo(image, title, colorMap=('',''), saveName=''):
    fig, axis = plt.subplots(1, 2, figsize=(12, 10))
    fig.tight_layout()
    for i in range(2):
        if (colorMap[i] == ''):
            axis[i].imshow(image[i])
        else:
            axis[i].imshow(image[i], cmap=colorMap[i])
        axis[i].set_title(title[i])
    if saveName != '':
        plt.savefig(saveName, bbox_inches='tight')
    plt.show(block=True)

def undistort(img):
    return cv2.undistort(img, mtx, dist, None, mtx);

# Applies Sobel x or y, then takes an absolute value and applies a threshold.
# NOTE the given image must be grayscale.
def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    sobel = cv2.Sobel(gray, cv2.CV_64F, orient=='x', orient=='y', ksize=sobel_kernel) # Take the dervivative either in x or y
    abs_sobel = np.absolute(sobel)
    # Scale to 8-bit and convert to type uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

# Computes the magnitude of the gradient, and applies a threshold.
# NOTE the given image must be grayscale.
def mag_thresh(gray, sobel_kernel=3, thresh=(0, 255)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag = np.sqrt(sobelx**2 + sobely**2) # Magnitude of the gradient
    # Scale to 8-bit and convert to type uint8
    scaled_mag = np.uint8(255 * mag / np.max(mag))
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(scaled_mag)
    binary_output[(scaled_mag >= thresh[0]) & (scaled_mag <= thresh[1])] = 1
    return binary_output

# Computes the direction of the graident, and applies a threshold.
# NOTE the given image must be grayscale.
def dir_thesh(gray, sobel_kerne=3, thresh=(0, np.pi/2)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take absolute of sobelx and sobely to get the gradient direction
    grad_dir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(grad_dir)
    binary_output[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    return binary_output

# Applies a threshold on the S channel of HLS color space.
# NOTE the given image must be on HLS color space.
def hls_s_thresh(hls, thresh=(0, 255)):
    s_channel = hls[:, :, 2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

# Read in the saved camera calibration matrix and distortion cofficients
dist_pickle = pickle.load(open("calibration.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# List of all the test images
testFiles = glob.glob('./test_images/*.jpg')

"""
# Original images
orig_images = []
for fname in testFiles:
    orig_images.append(mpimg.imread(fname))
# Undistorted images
undistorted_images = []
for img in orig_images:
    undist = undistort(img)
    undistorted_images.append(undist)
    #plotTwo(img, undist, 'original', 'undistorted')

example = len(orig_images) - 2;
plotTwo(orig_images[example], undistorted_images[example], 'original', 'undistorted', 'undistorted.png');
"""

# FIXME: pw test color and threshold
example = mpimg.imread('./test_images/test5.jpg');
img = undistort(example)
plotTwo((example, img), ('original', 'undistorted'))

hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

sobelx_binary = abs_sobel_thresh(gray, 'x', 3, (20, 100))
s_binary = hls_s_thresh(hls, (170, 255))

combined_binary = np.zeros_like(sobelx_binary)
combined_binary[(s_binary == 1) | (sobelx_binary == 1)] = 1

plotTwo((img, combined_binary), ('undistorted', 'combined_1'), ('', 'gray'), 'combined_1.png')
