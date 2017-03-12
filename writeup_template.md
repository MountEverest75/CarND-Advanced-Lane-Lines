##Self Driving Car Nano Degree
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

[image1]: ./examples/01-calibration.png "Camera Calibration"
[image2]: ./examples/02-undistort_chessboard.png "Distorted Vs Undistorted Chessboard Images"
[image3]: ./examples/03-undistortedimages.png "Undistorted Images"
[image4]: ./examples/04-colorschemes.png "Color Schemes"
[image5]: ./examples/05-perspectivetransform.png "Perspective Transform"
[image6]: ./examples/06-SobelAbsoluteGradient.png "Sobel Absolute Gradient"
[image7]: ./examples/07-SobelMagnitudeThreshold.png "Sobel Magnitude Gradient"
[image8]: ./examples/08-SobelDirectionThreshold.png "Sobel Direction Gradient"
[image9]: ./examples/09-SColorChannel.png "S Color Channel"
[image10]: ./examples/10-BColorChannel.png "B Color Channel"
[image11]: ./examples/11-Pipeline.png "Final Pipeline Image"
[image12]: ./examples/12-PolynomialAndHistogram.png "Polynomial Fit"
[image13]: ./examples/13-PaintedLane.png "Painted Lane"
[image14]: ./examples/14-PaintedLaneAnnotated.png "Painted Lane Annotated"
[video1]: ./project_video_output.mp4 "Project Video Output"
[video2]: ./challenge_video_output.mp4 "Challenge Video Output"
[video3]: ./harder_challenge_video_output.mp4 "Harder Challenge Video Output"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step can be found in the second and third code cells of the IPython notebook "P4-Advanced-LaneLinesV1.ipynb" with steps detailed below. The following [Open CV documentation link](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html) provides reference to the techniques tried as part of this project.

#### Prepration Steps for Camera Calibration
1. Define data containers to capture "object points" in 3D real world space and "image points" in 2D image space.
2. The chessboard corner dimensions have been adjusted to 9x6 instead of 8x6 used in the lessons.
3. The termination criteria of algorithm used were epsilon and max iterations for the calibration to stop. e.g. criteria = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
4. All images in the camera_cal folder used iteratively to examine the accuracy of the model.
5. All images are converted from BGR to Gray before applying cv2.cornerSubPix(gray, corners,(11,11),(-1,-1),criteria) before capturing image points.
6. Once "Object Points" and "Image Points" are captured Camera was calibrated following steps below.

![alt text][image1]

## Camera Calibration
1. As part of this project I have tried capturing Camera Matrix and Distortion Coefficients for all the images. The rotation and translation vectors have been ignored.
2. The Calibration is was performed using cv2.cameraCalibrate using Object Points and Image Points captured in the previous steps. The matrix was captured under variable mtx and distortion coefficients as dist variable in third code cell of the IPython notebook "P4-Advanced-LaneLinesV1.ipynb".
3. The distortion coefficient has been adjusted using the following steps:
- Capture Optimal New Camera Matrix for each image using

```
h, w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
```

- Undistort image using Curved Path instead of Shortest path approach. The curved path approach requires rectifying the map using new optimal camera matrix, remapping and cropping the image using Region Of Image. The code snippet describes the approach.
```
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
x,y,w,h = roi #Capture attributes using the Region Of Image
dst = dst[y:y+h, x:x+w] #Adjust by cropping Undistorted Image.
```

![alt text][image2]

*NB: The findChessBoardCorners() method could not detect corners for all images*


###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
The images below show case the original test images and their appearance once distortion-correction is applied.

![alt text][image3]

The effect of distortion is very minute, but is often visible at the bottom of the image or by looking at the objects. My original approach was to use undistortion  procedure for curved path, but it was cropping the edges a little more than expected during subsequently perspective transform. So I have used shortest path undistortion by calling cv2.undistort(img, mtx, dist, None, mtx) method. The source code for these steps could be found in code cell#4 in IPython notebook "P4-Advanced-LaneLinesV1.ipynb".

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I have experimented with 3 color schemes RGB, HSV and LAB. The code cell#6 in Python notebook "P4-Advanced-LaneLinesV1.ipynb" has all the details. The suggestion to use LAB channel in forums was the additional approach chosen. The "S Channel" in HSV color scheme and "B Channel" in LAB channel were really useful in detecting lane lines. Especially B Channel with detects Blue-Yellow in LAB color scheme was accurate in finding yellow lines. The samples images with color scheme explorations given below.   

![alt text][image4]

I have explored all combinations of sobel threshold gradients with details of each gradient given below. The approach required a little bit of tuning and experimentation with minimum and maximum threshold values. The source code in cell#7 contains all the details.

Examining Absolute Sobel Threshold Gradient. Sample images below:

![alt text][image6]

Examining Magnitude Threshold Gradient below:

![alt text][image7]

Examining Direction Threshold Gradient below:

![alt text][image8]

Examining S Channel in HSV color scheme:

![alt text][image9]

Examining B Channel in LAB color scheme to trap yellow lines:

![alt text][image10]

Finally I chose to use all gradients for edge detection along with S Channel in HSV color scheme to detect most colors and B channel in LAB color space to isolate yellow lanes to create image processing pipeline. The final results are given below for all images:

![alt text][image11]


####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

To experiment perspective transform I tried two functions `warp_image()` and `unwarp()` in code cell#5. Both functions takes accept 3 inputs an image (`img`), source (`src`) and destination (`dst`) points. The unwarp function performed better for me. The warp_image() using image size added some inaccuracies to the pipeline. Also I have experimented with two types of src and dst points below. The first approach is the standard approach learnt in the lessons.

### First sample of src and dst points from lesson samples. This approach has been tried in IPython notebook P4-Advanced-LaneLinesV0.ipynb which is previous version
```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 460      | 320, 0        |
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

### Second sample of src and dst points learnt from forums with src hardcoded and dst dynamically set based on image width. The latest version P4-Advanced-LaneLinesV1.ipynb uses this approach
```
src = np.float32([(575,464),
                  (707,464),
                  (258,682),
                  (1049,682)])

dst = np.float32([(450,0),
                  (w-450,0),
                  (450,h),
                  (w-450,h)])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. Searching for betters ideas to make the process of setting `src` and `dst` values dynamic.

![alt text][image5]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The functions orchestrate_sliding_windows() in code cell#16 and orchestrate_image_stream() in code cell#25 describe the polynomial fit and second order of polynomial fit to detect lane lines and left/right lines. The first function depicts histogram and detects the starting x positions of left and right lines. The function determines the windows where lane pixels can be found. The image below demonstrates how this process works showcasing Original image, Pipeline image, polynomial fit and histogram side by side comparatively:

![alt text][image12]

The orchestrate_image_stream() performs the same functionality of control break processing from previous fit to build piple for the videos.

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature calculation has been performed based on the suggested website link in the lessons that can be accessed by [clicking here](http://www.intmath.com/applications-differentiation/8-radius-curvature.php). This logic can be found code cells#18 and 19 in the IPython notebook. The idea is the convert pixels to meters has been borrowed from the following [blog post](https://medium.com/@jeremyeshannon/udacity-self-driving-car-nanodegree-project-4-advanced-lane-finding-f4492136a19d#.bkk0d0vw3).

The distance from center has been calculated as mean of left and right fit vertices.

The steps to calculate radius of curvature has been detailed below:
1. Identify conversion factors for pixels in image space to meters in real space.
2. Identify x and y co-ordinates of non zero pixels in the image
3. Identify left and right pixels based on the lane indicators.
4. Fit polynomials using np.polyfit to x, y co-ordinates in real space
5. Calculate new radius of curvature using the following equation
```
curve_radius = ((1 + (2*fit[0]*y_0*y_meters_per_pixel + fit[1])**2)**1.5) / np.absolute(2*fit[0])
```

In this example, fit[0] is the first coefficient (the y-squared coefficient) of the second order polynomial fit, and fit[1] is the second (y) coefficient. y_0 is the y position within the image upon which the curvature calculation is based (the bottom-most y - the position of the car in the image - was chosen). y_meters_per_pixel is the factor used for converting from pixels to meters. This conversion was also used to generate a new fit with coefficients in terms of meters.

The position of the vehicle with respect to the center of the lane is calculated with the following lines of code:
```
lane_center_position = (r_fit_x_int + l_fit_x_int) /2
center_dist = (car_position - lane_center_position) * x_meters_per_pix
```

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Painting the lane area ahead as a polygon has been coded in code cell#20 as draw_lane_polygon() function. A polygon is generated based on the left and right fit co-ordinates. The polygon area is color painted using methods cv2.polyfill() method. The left and right lanes are color marked to identify boundaries using cv2.polylines(). Using inverse perspective matrix the image is warped back to original perspective from birds eye view to show lane ahead. The reverse unwarp is done simply by using this method cv2.warpPerspective(color_warp, Minv, (w, h)).

![alt text][image13]

The painted lane image is annotated by calling cv2.putText() method in put_data_on_image() method in code cell#22. The annotated images can be found below:

![alt text][image14]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's the [link to my video result](./project_video_output.mp4)
Here's the [link to my challenge video result](./challenge_video_output.mp4)
Here's the [link to my harder challenge video result](./harder_challenge_video_output.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The major problems I encountered were in the challenge videos:
1. In the Challenge video the painting of lanes has been found to be wobbly, however as the car approached closer to the curve it tried to rectify back into the lane. Tuning the radius of curvature would help reduce the wobbliness and risk of crossing into the adjacent lane.
2. In the harder challenge my pipeline process could not detect conditions like excessive sun light, reflections and transition from excess light to shadows due to trees. The wobbliness in curves is more prominent in this video. However I did not see the car veer out of the track. Fine tuning the left and right fits might reduce wobbliness.

Scope for improvement:
- Testing pipeline in snowy or cloudy conditions to account for hazy conditions.
- Presence of snow and sun might add addition reflective surfaces that could fail the pipeline process.
- Dynamic assignment of 'src' and 'dst' points used for Perspective transform could be useful to reduce some of the errors when images other than test sample provided. However since we are unlikely to change cameras often once mounted and calibrated, the limitations in this project to assign static values is accepted.

I am happy to get exposure to LAB color space as part of this exercise which helped detect yellow lines very well. However one challenge I faced with color channels and gradients was to identify the right threshold range to detect the lanes and edges. This could also reduce any noise or wobbliness in the pipeline process.

I have explored two undistort techniques to use shorted path or curved path. The curved path approach made the perspective transform look like an eagle sky view rather than birds eye view. Would like to explore this option further to see if I can readjust to achieve more accuracy in detecting lanes.
