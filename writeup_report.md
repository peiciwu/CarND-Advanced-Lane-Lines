## Writeup Template
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/corners.png "Finding Corners"
[image2]: ./output_images/chessboard_undistort.png "Chessboard Undistorted"
[image3]: ./output_images/undistorted.png "Undistort Example"
[image4]: ./output_images/thresholding.png "Thresholding Example"
[image4]: ./output_images/warp_straight_lines.png "Warp Example - Straight Lines"
[image5]: ./output_images/warp_curve.png "Warp Example - Curve lines"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is at `./calibrate.py`.

For each image in "./camera_cal", `cv2.findChessboardCorners()` is used to find all the corners on the chessboard in the image. For each corner, it would be mapped to a three-dimensional coordinate (x, y, z), where (x, y) represents the corner at which column and row (z is always 0 as assuming the chessboard is at fixed on a plane.)

Here is the result of corners found on the images:

![alt text][image1]

Note that the first image fails to find corners, so no corners drawn on the image.

Next, the found corner points and the respective coordinates are used in `cv2.calibrateCamera()` to compute the camera calibration and distortion coefficients.

An example image applied `cv2.undistort()` using the obtained coefficient is shown below:

![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image3]

This undistort function is called `ImageProcessor.undistort()` at `image_processor.py`. Note that the undistort function needs the camera calibration matrix and distortion coefficients, and those are load from a pre-stored pickle file (which are written in the above camera calibration step.) The related code is at `__init__()` of class `ImageProcessor`.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. `ImageProcessor.thresholding()` at `image_processor.py` is this combination function. Gradient threshold is applied with min-threshold = 20, max-threshold = 150, and use kernel of size 3. Two color thresholds are applied at s-space of HLS image and r-space of RGB image, respectively. I've set a slight larger min-threshold for the s-space to avoid unnecessary detections on shaded images. R-space is chosen because it's good at yellow lines while s-space not.

Here is an example for an image applies this thresholding function.

![alt text][image4]  

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I first worked on one straight line image, drew two line that are parallel to the lanes. Two endpoints for the left line are `(216, 705)` and `(585, 454)`, and two endpoints for the right line are `(693, 450)` and `(1073, 693)`. With these, I can have slopes for the two lines, so I just need to decide what are two bottom left and bottom right points I like, the other two endpoints can automatically be obtained using slopes. This procedure is called `ImageProcessor.get_perspective_transform_parameters()` at `image_processor.py`. With this procedure, I can easily get a set of source and destination points by quickly moving the left-bottom(right-bottom) point left and right. Same as the camera calibration parameters, these perspective transform parameters are calculated in `__init__()` of class `ImageProcessor`. The perspective transform function is `ImageProcessor.perspective_transform()` at `image_processor.py`.

Below are two example images. One is a straight line image where the warped lines are (almost) vertical. The other one is a curve image where the warped line are (almost) parallel. 

![alt text][image4]

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
