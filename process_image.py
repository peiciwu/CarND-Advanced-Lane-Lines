import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import collections
from color_thresh_utility import ImageProcessor, plotTwo

ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
ploty = np.linspace(0, 719, 720) # y range of image
# FIXME: pw to delete
frame_counter = 0

def run_all_test_images():
    # List of all the test images
    # FIXME: pw
    #testFiles = glob.glob('./test_images/*.jpg')
    #testFiles = glob.glob('./my_test_images/*.jpg')
    testFiles = glob.glob('./my_test_images/other_types_test_images/*.jpg')

    for fname in testFiles:
        img = mpimg.imread(fname)
        # FIXME: pw
        print("shape: ", img.shape)
        # Process all the images: distort, thresholding, and perspective transform
        undist = image_processor.undistort(img)
        name = '(' + os.path.splitext(os.path.basename(fname))[0] + ')'
        plotTwo((img, undist), ('Original'+name, 'Undistorted'+name))
        thresh = image_processor.thresholding(undist, plot = True)
        plotTwo((undist, thresh), ('Undistorted'+name, 'Thresholding'+name), ('', 'gray'))
        warped = image_processor.perspective_transform(thresh)
        plotTwo((thresh, warped), ('Thresholding'+name, 'Warped'+name), ('gray', 'gray'))

        # Use window sliding to find the lanes
        left_fit, right_fit = find_lane_lines_sliding_window(warped, plot=True)
        print(name, ": ", left_fit, ", ", right_fit)
        result = draw_lane(img, warped, left_fit, right_fit)
        draw_radius_of_curvature(result, (left_line.radius_of_curvature + right_line.radius_of_curvature)/ 2)

        # FIXME: pw debug
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        diff = right_fitx - left_fitx
        min_width = np.min(diff)
        max_width = np.max(diff)
        cv2.putText(result, 'min_width = ' + str(round(min_width, 3))+' (pixel)',(50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
        cv2.putText(result, 'max_width = ' + str(round(max_width, 3))+' (pixel)',(50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
        print("min_width: ", min_width)
        print("max_width: ", max_width)

        # Reset, not using the previous fit and no smoothing.
        left_line.reset()
        right_line.reset()

        plt.imshow(result)
        plt.show(block=True)


#FIXME: findling the line 
def find_lane_lines_sliding_window(img, plot=False):
    # Parameters setting
    num_windows = 9
    window_height = np.int(img.shape[0]/num_windows)
    margin = 80 # widnow width = 2 * margin
    min_pixels = 50 # minimum number of pixels found in one window

    # Create an output image to draw on and  visualize the result
    if plot == True:
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
    leftx = nonzerox[left_lane_indices]
    lefty = nonzeroy[left_lane_indices]
    rightx = nonzerox[right_lane_indices]
    righty = nonzeroy[right_lane_indices]
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    left_line.add_fit(left_fit)
    right_line.add_fit(right_fit)

    # Plot the current fitted polynomial
    if plot == True:
        # Mark the pixels in the window: left with red, right with blue
        out_img[nonzeroy[left_lane_indices], nonzerox[left_lane_indices]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_indices], nonzerox[right_lane_indices]] = [0, 0, 255]
        plt.imshow(out_img)

        # Plot the fitted polynomial
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, img.shape[1])
        plt.ylim(img.shape[0], 0)

        plt.show(block=True)

    return left_fit, right_fit

def find_lane_lines_using_previous_fit(img, left_fit, right_fit, plot=False):
    # X and y positions where the pixels are nonzero
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100 # widnow width = 2 * margin
    # Indices within the margin of the polynomial
    left_lane_indices = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2]) - margin) & 
                         (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2]) + margin))
    right_lane_indices = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2]) - margin) & 
                          (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2]) + margin))
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_indices]
    lefty = nonzeroy[left_lane_indices]
    rightx = nonzerox[right_lane_indices]
    righty = nonzeroy[right_lane_indices]
    # Fit a second order polynomial
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    left_line.add_fit(left_fit)
    right_line.add_fit(right_fit)

    if plot == True:
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((img, img, img))*255
        # Mark the pixels in the window: left with red, right with blue
        out_img[nonzeroy[left_lane_indices], nonzerox[left_lane_indices]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_indices], nonzerox[right_lane_indices]] = [0, 0, 255]
        plt.imshow(out_img)

	# Generate a polygon to illustrate the search window area
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # window img
        window_img = np.zeros_like(out_img)
        # Draw the lane onto the window image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, img.shape[1])
        plt.ylim(img.shape[0], 0)

        plt.show(block=True)

    return left_fit, right_fit

def draw_lane(img, warped, left_fit, right_fit):
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    #FIXME: pw debug
    #color_warp = np.dstack((warp_zero, warp_zero, warp_zero, warp_zero))

    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255, 0, 0), thickness=20)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0, 0, 255), thickness=20)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    new_warp = cv2.warpPerspective(color_warp, image_processor.invM, (warped.shape[1], warped.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, new_warp, 0.3, 0)

    return result

# Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # polynomial coefficients for the most recent fit
        self.current_fit = None
        # x values of the last n fits of the line
        # FIXME: pw 
        #self.recent_fitted = collections.deque(maxlen=10)
        self.recent_xfitted = collections.deque(maxlen=10)
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        # radius of curvature of the best fit
        self.radius_of_curvature = None 
        # of consecutive failed frames 
        self.num_of_fails = 0
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        # difference in x values between last and new fits
        self.fitx_diffs = np.array([0,0,0], dtype='float') 
        """
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        """
    def add_fit(self, fit):
        self.current_fit = fit
        if self.best_fit != None:
            self.diffs = abs(fit - self.best_fit)

        fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
        if len(self.recent_xfitted) > 0:
            self.fitx_diffs = abs(fitx - self.recent_xfitted[-1])
        self.recent_xfitted.append(fitx)

        avg_fitx = np.average(self.recent_xfitted, axis=0)
        self.best_fit = np.polyfit(ploty, avg_fitx, 2)
        self.radius_of_curvature = self.cal_radius_of_curvature(self.best_fit)
        # FIXME: pw
        self.detected = True

        """
        # FIXME: pw debug version
        self.current_fit = fit
        self.best_fit = fit
        self.radius_of_curvature = self.cal_radius_of_curvature(self.best_fit)
        self.detected = True

        fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
        """

    def reset(self):
        self.detected = False  
        self.current_fit = None
        self.recent_xfitted.clear()
        self.best_fit = None
        self.radius_of_curvature = None
        self.num_of_faileds = 0

    def cal_radius_of_curvature(self, fit):
        fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
        # Fit new polynomials to x, y in world space
        fit_cr = np.polyfit(ploty * ym_per_pix, fitx * xm_per_pix, 2)
        y_eval = np.max(ploty)
        curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
        return curverad

    def check_curvature(self, fit):
        if self.radius_of_curvature == None:
            return True
        
        curverad = cal_radius_of_curvature(fit)
        change = abs(curverad - self.radius_of_curvature)/self.radius_of_curvature
        return change <= 0.3

    def sanity_check(self, curverad):
        if self.radius_of_curvature == None:
            return True

        change = abs(curverad - self.radius_of_curvature)/self.radius_of_curvature
        return change <= 0.3

# Draw radius of curvature on the result image
def draw_radius_of_curvature(img, curv):
    cv2.putText(img, 'Radius of curvature = ' + str(round(curv, 3))+' (m)',(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)

def sanity_check(left_l, right_l):
    if left_l.best_fit == None or right_l.best_fit == None:
        return True

    # Current frame has a bad fit. No need to do any further checking.
    if left_l.current_fit == None or right_l.current_fit == None:
        return False

    if abs(left_l.current_fit[0] - right_l.current_fit[0]) > 0.001:
        return False

    return True

def process_image(img, plot = False):
    global frame_counter 
    undist = image_processor.undistort(img)
    thresh = image_processor.thresholding(undist)
    warped = image_processor.perspective_transform(thresh)

    """
    if left_line.detected == False or right_line.detected == False:
        reset()
        find_lane_lines_sliding_window(warped, plot)
    else:
        find_lane_lines_using_previous_fit(warped, left_line.current_fit, right_line.current_fit, plot)

    # Sanity check
    if sanity_check(left_line, right_line):
        left_line.best_fit = left_line.current_fit
        right_line.best_fit = right_line.current_fit
        left_line.detected = True
        right_line.detected = True
    else:
        left_line.detected = False
        right_line.detected = False
    """

    left_fit, right_fit = find_lane_lines_sliding_window(warped, plot)
    result = draw_lane(img, warped, left_line.best_fit, right_line.best_fit)
    draw_radius_of_curvature(result,
            (left_line.radius_of_curvature+right_line.radius_of_curvature)/2)

    # FIXME: pw debug
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    diff = right_fitx - left_fitx
    min_width = np.min(diff)
    max_width = np.max(diff)
    cv2.putText(result, 'min_width = ' + str(round(min_width, 3))+' (pixel)',(50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    cv2.putText(result, 'max_width = ' + str(round(max_width, 3))+' (pixel)',(50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)

    # FIXME: pw debug
    #if check_lane(left_fit, right_fit) == False:
    if min_width < 550 or max_width > 850:
        filename = "./video_frames_images/" + str(frame_counter) + ".jpg"
        mpimg.imsave(filename, img)
        filename = "./video_frames_processed_images/" + str(frame_counter) + ".jpg"
        mpimg.imsave(filename, result)
    frame_counter += 1

    # Reset, not using the previous fit and no smoothing.
    left_line.reset()
    right_line.reset()

    if plot == True:
        plt.imshow(result)
        plt.show(block=True)	

    return result

def check_lane(left_fit, right_fit):
    """
    if sanity_check(left_fit, right_fit) == False:
        return False
    """

    if left_line.check_curvature(left_fit) == False:
        return False

    if right_line.check_curvature(right_fit) == False:
        return False
    
    return True

image_processor = ImageProcessor()
left_line = Line()
right_line = Line()
#img1 = process_image(mpimg.imread('./test_images/test4.jpg'), plot = True)
#img2 = process_image(mpimg.imread('./test_images/test5.jpg'), plot = True)
#quit()
 
# FIXME: test on all images
#run_all_test_images()
#quit()

# FIXME: on videos
from moviepy.editor import VideoFileClip
output1 = './project_video_processed.mp4'
clip1 = VideoFileClip("./project_video.mp4")
processed_clip1 = clip1.fl_image(process_image) #NOTE: this function expects color images!!
processed_clip1.write_videofile(output1, audio=False)

