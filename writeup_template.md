## Writeup
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car_noncar]: ./writeup_images/car_noncar.png
[car_hog]: ./writeup_images/car_hog.png
[noncar_hog]: ./writeup_images/noncar_hog.png
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][car_noncar]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][car_hog]
![alt text][noncar_hog]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and I found that increasing the values beyond the ones mentioned did not improve the performance of the detection by any resonable amount whereas it was detrimental to the speed of both training as well as prediction. Hence I chose to use `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` for the HOG parameters.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`). 

I converted  the image to  `YCrCb` Color space as I found that it gave the best results overall in accuracy.
I used spatial binning with `spatial_size=(16,16)` and color binning with 8 histogram bins and combined this along with the HOG features from all channels. I tried various values for spatial size and number of bins. While increasing these provided marginal improvements, the amount of increase in parameters caused the subsequent pipeline to be executed extremely slowly. while decreasing them subsequently decreased performance. I tried removing binning altogether, however this gave large number of false positives.
Therefore i settled for the aforementioned values to strike a balance between accuracy and performance.

I have also tried training multiple types of SVM's for classification. I initially trained using LinearSVC and got some good results with occasional false positives. The accuracy averaged at 99%. I tried to improve this by useing GridSearchCV to get a good set of parameters and test out linear and rbf kernels and multiple C values. This gave me a very good accuracy of 99.5% with almost no false positives and clean detection of cars. However the model was extremely slow to both train as well as predict with training taking almost an hour and a half while prediction was taking almost 75seconds on average on a single frame. This meant a time of over 13 Hours to completely detect objects in the video. I initially thought that machine hardware could have been the issue as LinearSVC was predicting at 2 to 4 seconds per frame. However on taking a dump of the classifier I found that the classifier was nearly 140MB and was practically unusable. So I settled on the better performing LinearSVC model.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

