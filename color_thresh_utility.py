import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob

class ImageProcessor:
    def __init__(self):
        # Read in the saved camera calibration matrix and distortion cofficients
        dist_pickle = pickle.load(open("calibration.p", "rb"))
        self.mtx = dist_pickle["mtx"]
        self.dist = dist_pickle["dist"]
        self.src, self.dst, self.M, self.invM = self.get_perspective_transform_parameters()

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx);

    # Main function to combine color and gradient thresholds
    def thresholding(self, img, plot = False):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        s_channel = hls[:,:,2]
        r_channel = img[:,:,0]

        #sobelx_binary = abs_sobel_thresh(gray, 'x', 3, (30, 150))
        sobelx_binary = abs_sobel_thresh(gray, 'x', 3, (20, 150))
        #mag_binary = mag_thresh(gray, 9, (50, 255))
        #dir_binary = dir_thresh(gray, 15, (0.7, 1.3))
        #s_binary = binary_thresh(s_channel, (170, 200)) # On s-channel
        s_binary = binary_thresh(s_channel, (230, 250)) # On s-channel
        r_binary = binary_thresh(r_channel, (215, 255)) # On r-channel

        combined_binary = np.zeros_like(sobelx_binary)
        combined_binary[(s_binary == 1) | (sobelx_binary == 1) | (r_binary == 1)] = 1
        #combined_binary[((mag_binary == 1) & (dir_binary == 1)) | (s_binary == 1) | (r_binary == 1)] = 1

        if plot == True:
            plotTwo((gray, sobelx_binary), ('gray', 'sobelx'))
            plotTwo((s_channel, s_binary), ('s-channel', 's-threshold'))
            plotTwo((r_channel, r_binary), ('r-channel', 'r-threshold'))
            plotTwo((img, combined_binary), ('undistorted', 'combined'))

        return combined_binary

    def perspective_transform(self, img, plot = False):
        warped = cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]))
        if plot == True:
            img_region = cv2.polylines(img, [np.int32(self.src)], True, (255, 0, 0), 3)
            warped_region = cv2.polylines(warped, [np.int32(self.dst)], True, (255, 0, 0), 3)
            plotTwo((img_region, warped_region), ('img_with_region', 'warped_with_region'))
        return warped

    # Return (src, dst, M, invM).
    # Start roughly from 4 points on the two lines at straight_lines1.jpg (two
    # points on one line), and then perform some minor adjust based on th warped results.
    def get_perspective_transform_parameters(self):
        example = mpimg.imread('./test_images/straight_lines1.jpg');
        undist = self.undistort(example)
        # Get perspective transform matrix
        h, w = undist.shape[:2]
        # Two lines on the "straight_lines1.jpg": (216, 705), (585, 454), (687, 450), (1073, 693)
        slope1 = (705-454)/(216-585);
        slope2 = (450-693)/(670-1073)
        y = 460
        #x0 = 200
        x0 = 195
        #x4 = 1122
        x4 = 1140
        #x1 =  (y-h)/slope1+x0 - 15
        x1 =  (y-h)/slope1+x0
        x2 = x4 - (h-y)/slope2
        #print ("x1: ", x1, ", x2: ", x2)

        src = np.float32([(x0,h), (x1,y), (x2,y), (x4,h)])
        dst = np.float32([(250,h), (250,0), (w-300,0), (w-300, h)])
        M = cv2.getPerspectiveTransform(src, dst)
        invM = cv2.getPerspectiveTransform(dst, src)

        return (src, dst, M, invM)

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
def dir_thresh(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take absolute of sobelx and sobely to get the gradient direction
    grad_dir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    return binary_thresh(grad_dir, thresh)

# Helper function for displaying two images in parallel to compare.
def plotTwo(image, title, saveName=''):
    fig, axis = plt.subplots(1, 2, figsize=(12, 10))
    fig.tight_layout()
    for i in range(2):
        if len(image[i].shape) == 2:
            axis[i].imshow(image[i], cmap='gray')
        else:
            axis[i].imshow(image[i])
        axis[i].set_title(title[i])
    if saveName != '':
        plt.savefig(saveName, bbox_inches='tight')
    plt.show(block=True)

"""
# Verify perspective_transform:
# - The lines on the image warped from straight_lines1.jpg should be vertical. 
# - The curvel lines should be parallel to each other.
image_processor = ImageProcessor()
#testFiles = glob.glob('./test_images/*.jpg')
testFiles = glob.glob('./my_test_images/*.jpg')
for fname in testFiles:
    img = mpimg.imread(fname)
    undist = image_processor.undistort(img)
    warped = image_processor.perspective_transform(undist, plot=True)
"""
