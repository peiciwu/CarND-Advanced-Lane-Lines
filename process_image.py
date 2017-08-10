import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
import os

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

# Apply perspective transform using the given matrix/source/destination points.
def perspective_transform(img, plot = False):
    warped = cv2.warpPerspective(img, p_M, (img.shape[1], img.shape[0]))
    if plot == True:
        img_region = cv2.polylines(img, [np.int32(p_src)], True, (255, 0, 0), 3)
        warped_region = cv2.polylines(warped, [np.int32(p_dst)], True, (255, 0, 0), 3)
        plotTwo((img_region, warped_region), ('img_with_region', 'warped_with_region'))
    return warped

def run_all_test_images():
    # List of all the test images
    testFiles = glob.glob('./test_images/*.jpg')

    # Process all the images: distort, thresholding, and perspective transform
    for fname in testFiles:
        img = mpimg.imread(fname)
        undist = undistort(img)
        name = '(' + os.path.splitext(os.path.basename(fname))[0] + ')'
        plotTwo((img, undist), ('Original'+name, 'Undistorted'+name))
        thresh = thresholding(undist)
        plotTwo((undist, thresh), ('Undistorted'+name, 'Thresholding'+name), ('', 'gray'))
        perspective_transform(undist, plot=True)
        warped = perspective_transform(thresh)
        plotTwo((thresh, warped), ('Thresholding'+name, 'Warped'+name), ('gray', 'gray'))
        left_fitx, right_fitx, ploty = find_lane_lines(warped, plot=True)
        result = draw_lane(img, warped, left_fitx, right_fitx, ploty, plot=True)

#FIXME: findling the line 
def find_lane_lines(img, plot=False):
    # Parameters setting
    num_windows = 9
    window_height = np.int(img.shape[0]/num_windows)
    margin = 100 # widnow width = 2 * margin
    min_pixels = 50 # minimum number of pixels found in one window

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((img, img, img))*255

    # Starting points: The peaks of the histogram at the left half and the right half.
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    leftx_current = np.argmax(histogram[:midpoint])
    rightx_current = np.argmax(histogram[midpoint:]) + midpoint

    # X and y positions where the pixels are nonzero
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # For pixel found within windows, record the indices to nonzero array
    left_lane_indices = []
    right_lane_indices = []

    for window in range(num_windows):
        # Window boundaries
        win_y = (img.shape[0] - (window+1)*window_height, img.shape[0] - window*window_height)
        win_left_x = (leftx_current - margin, leftx_current + margin)
        win_right_x = (rightx_current - margin, rightx_current + margin)

        if plot == True:
            # Draw the windows
            cv2.rectangle(out_img, (win_left_x[0], win_y[0]), (win_left_x[1], win_y[1]), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_right_x[0], win_y[0]), (win_right_x[1], win_y[1]), (0, 255, 0), 2)

        # Find nonzero pixels within the window
        nonzero_left_indices = ((nonzeroy >= win_y[0]) & (nonzeroy < win_y[1]) & (nonzerox >= win_left_x[0]) & (nonzerox < win_left_x[1])).nonzero()[0]
        nonzero_right_indices = ((nonzeroy >= win_y[0]) & (nonzeroy < win_y[1]) & (nonzerox >= win_right_x[0]) & (nonzerox < win_right_x[1])).nonzero()[0]
        # Record the indices of nonzero pixels
        left_lane_indices.append(nonzero_left_indices)
        right_lane_indices.append(nonzero_right_indices)
        # Move to the next widnow if > minimum pixels found
        if len(nonzero_left_indices) > min_pixels:
            leftx_current = np.int(np.mean(nonzerox[nonzero_left_indices]))
        if len(nonzero_right_indices) > min_pixels:
            rightx_current = np.int(np.mean(nonzerox[nonzero_right_indices]))
    
    # Concatenate the indices array
    left_lane_indices = np.concatenate(left_lane_indices)
    right_lane_indices = np.concatenate(right_lane_indices)
    # Fit a second order polynomial
    left_fit = np.polyfit(nonzeroy[left_lane_indices], nonzerox[left_lane_indices], 2)
    right_fit = np.polyfit(nonzeroy[right_lane_indices], nonzerox[right_lane_indices], 2)

    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    if plot == True:
        # Mark the pixels in the window: left with red, right with blue
        out_img[nonzeroy[left_lane_indices], nonzerox[left_lane_indices]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_indices], nonzerox[right_lane_indices]] = [0, 0, 255]
        plt.imshow(out_img)

        # Plot the fitted polynomial
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, img.shape[1])
        plt.ylim(img.shape[0], 0)

        plt.show(block=True)
    
    return left_fitx, right_fitx, ploty

def draw_lane(img, warped, left_fitx, right_fitx, ploty, plot=False):
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255, 0, 0), thickness=20)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0, 0, 255), thickness=20)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    #FIXME: should store
    Minv = cv2.getPerspectiveTransform(p_dst, p_src)
    new_warp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, new_warp, 0.5, 0)

    if plot == True:
        plt.imshow(result)
        plt.show(block=True)	

    return result

def process_image(img):
    undist = undistort(img)
    thresh = thresholding(undist)
    warped = perspective_transform(thresh)
    left_fitx, right_fitx, ploty = find_lane_lines(warped)
    result = draw_lane(img, warped, left_fitx, right_fitx, ploty)
    return result


# Read in the saved camera calibration matrix and distortion cofficients
dist_pickle = pickle.load(open("calibration.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# FIXME: pw test color, threshold, and perspective transform
#example = mpimg.imread('./test_images/test5.jpg');
#example = mpimg.imread('./test_images/straight_lines1.jpg');
example = mpimg.imread('./test_images/test4.jpg');
undist = undistort(example)
plotTwo((example, undist), ('original', 'undistorted'))
thresh = thresholding(undist, plot=False)
plotTwo((undist, thresh), ('undistorted', 'thresholding'), ('', 'gray'))

"""
h, w = thresh.shape[:2]
src = np.float32([(575,464), (707,464), (258,682), (1049,682)])
dst = np.float32([(300,0), (w-300,0), (300,h), (w-300,h)])
M = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(thresh, M, (w, h))
plotTwo((thresh, warped), ('thresholding', 'warped'), ('gray', 'gray'))
"""

# Get perspective transform matrix
h, w = undist.shape[:2]
# Two lines on the "straight_lines1.jpg": (216, 705), (585, 454), (687, 450), (1073, 693)
slope1 = (705-454)/(216-585);
slope2 = (450-693)/(687-1073)
y = 460
x0 = 198+5
x4 = 1122
x1 =  (y-h)/slope1+x0
x2 = x4 - (h-y)/slope2
print ("x1: ", x1, ", x2: ", x2)

p_src = np.float32([(x0,h), (x1-3,y), (x2-3,y), (x4,h)])
p_dst = np.float32([(300,h), (300,0), (w-300,0), (w-300, h)])

p_M = cv2.getPerspectiveTransform(p_src, p_dst)

perspective_transform(undist, plot = True)

warped = perspective_transform(thresh)
plotTwo((thresh, warped), ('thresholding', 'warped'), ('gray', 'gray'))

find_lane_lines(warped, plot = True)

# FIXME: test on all images
run_all_test_images()

# FIXME: on videos
from moviepy.editor import VideoFileClip
output1 = './project_video_processed.mp4'
clip1 = VideoFileClip("./project_video.mp4")
processed_clip1 = clip1.fl_image(process_image) #NOTE: this function expects color images!!
processed_clip1.write_videofile(output1, audio=False)
