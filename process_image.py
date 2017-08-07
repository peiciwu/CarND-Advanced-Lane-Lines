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

# Create a binary image of ones where threshold is met, zeros otherwise
def binary_thresh(channel, thresh=(0, 255)):
    binary_output = np.zeros_like(channel)
    binary_output[(channel >= thresh[0]) & (channel <= thresh[1])] = 1
    return binary_output

# Applies Sobel x or y, then takes an absolute value and applies a threshold.
# NOTE the given image must be grayscale.
def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    sobel = cv2.Sobel(gray, cv2.CV_64F, orient=='x', orient=='y', ksize=sobel_kernel) # Take the dervivative either in x or y
    abs_sobel = np.absolute(sobel)
    # Scale to 8-bit and convert to type uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    return binary_thresh(scaled_sobel, thresh)

# Computes the magnitude of the gradient, and applies a threshold.
# NOTE the given image must be grayscale.
def mag_thresh(gray, sobel_kernel=3, thresh=(0, 255)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag = np.sqrt(sobelx**2 + sobely**2) # Magnitude of the gradient
    # Scale to 8-bit and convert to type uint8
    scaled_mag = np.uint8(255 * mag / np.max(mag))
    return binary_thresh(scaled_mag, thresh)

# Computes the direction of the graident, and applies a threshold.
# NOTE the given image must be grayscale.
def dir_thesh(gray, sobel_kerne=3, thresh=(0, np.pi/2)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take absolute of sobelx and sobely to get the gradient direction
    grad_dir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    return binary_thresh(grad_dir, thresh)

# Main function to combine color and gradient thresholds
def thresholding(img, plot = False):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    s_channel = hls[:,:,2]
    r_channel = img[:,:,0]

    sobelx_binary = abs_sobel_thresh(gray, 'x', 3, (30, 150))
    s_binary = binary_thresh(s_channel, (175, 250)) # On s-channel
    r_binary = binary_thresh(r_channel, (230, 255)) # On r-channel

    combined_binary = np.zeros_like(sobelx_binary)
    combined_binary[(s_binary == 1) | (sobelx_binary == 1) | (r_binary == 1)] = 1

    if plot == True:
        plotTwo((gray, sobelx_binary), ('gray', 'sobelx'), ('gray', 'gray'))
        plotTwo((s_channel, s_binary), ('s-channel', 's-threshold'), ('gray', 'gray'))
        plotTwo((r_channel, r_binary), ('r-channel', 'r-threshold'), ('gray', 'gray'))
        plotTwo((img, combined_binary), ('undistorted', 'combined'), ('', 'gray'))

    return combined_binary

def perspective_tranform(img):
    img_size 


# Read in the saved camera calibration matrix and distortion cofficients
dist_pickle = pickle.load(open("calibration.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# List of all the test images
testFiles = glob.glob('./test_images/*.jpg')

"""
# Process all the images: distort, thresholding, and perspective transform
for fname in testFiles:
    img = mpimg.imread(fname)
    undist = undistort(img)
    plotTwo((img, undist), ('original', 'undistorted'))
    thresh = thresholding(undist, plot=True)
"""



# FIXME: pw test color and threshold
#example = mpimg.imread('./test_images/test5.jpg');
example = mpimg.imread('./test_images/straight_lines1.jpg');
undist = undistort(example)
plotTwo((example, undist), ('original', 'undistorted'))
thresh = thresholding(undist, plot=False)
plotTwo((undist, thresh), ('undistorted', 'thresholding'), ('', 'gray'))

h, w = thresh.shape[:2]
src = np.float32([(575,464), (707,464), (258,682), (1049,682)])
dst = np.float32([(300,0), (w-300,0), (300,h), (w-300,h)])
M = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(thresh, M, (w, h))
plotTwo((thresh, warped), ('thresholding', 'warped'), ('gray', 'gray'))

