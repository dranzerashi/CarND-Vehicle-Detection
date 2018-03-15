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
[initial_output]: ./output_images/initial_output/test1.png
[heatmap_output]: ./output_images/heatmap_output/test1.png
[final_output]: ./output_images/final_output/test1.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 17 through 109 of the file called `util_functions.py` and is being called in the  code cell ln 4 of `project.ipynb` Ipynb note book.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][car_noncar]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the HOG parameters of `orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:


![alt text][car_hog]
![alt text][noncar_hog]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and I found that increasing the values beyond the ones mentioned did not improve the performance of the detection by any resonable amount whereas it was detrimental to the speed of both training as well as prediction. Hence I chose to use `orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)` for the HOG parameters. This gave me good accuracy with higher performance. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in the code cells ln 5 and 6 of `project.ipynb` IPython notebook. 

I converted  the image to  `YCrCb` Color space as I found that it gave the best results overall in accuracy.
I used spatial binning with `spatial_size=(16,16)` and color binning with 8 histogram bins and combined this along with the HOG features from all channels. I tried various values for spatial size and number of bins. While increasing these provided marginal improvements, the amount of increase in parameters caused the subsequent pipeline to be executed extremely slowly. while decreasing them subsequently decreased performance. I tried removing binning altogether, however this gave large number of false positives.
Therefore i settled for the aforementioned values to strike a balance between accuracy and performance.

I have also tried training multiple types of SVM's for classification. I initially trained using LinearSVC and got some good results with occasional false positives. The accuracy averaged at 99%. I tried to improve this by using GridSearchCV to get a good set of parameters and test out linear and rbf kernels and multiple C values. This gave me a very good accuracy of 99.5% with almost no false positives and clean detection of cars. However the model was extremely slow to both train as well as predict with training taking almost an hour and a half while prediction was taking almost 75 seconds on average on a single frame. This meant a time of over 13 Hours to completely detect objects in the video. I initially thought that machine hardware could have been the issue as LinearSVC was predicting at 2 to 4 seconds per frame. However on taking a dump of the classifier I found that the classifier was nearly 140MB and was practically unusable. So I settled on the better performing LinearSVC model.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this step is contained in lines 143 through 224 of the file called `util_functions.py` and is being called in the  code cell ln 7 and 8 of `project.ipynb` Ipynb note book.  


I used HOG subsampling to extract hog features once, for each of a small set of predetermined window sizes defined by a scale argument, and then can be sub-sampled to get all of its overlaying windows. Each window is defined by a scaling factor that impacts the window size. I used a section of the image rather than the whole image from `400` to `656` pixels along the y-axis to remove the horizon and keep only the road. I kept the window size of 64x64. Then I resize the base image according to the scaling factor for each scale and run on the image. An overlap of each window is achieved by using `cells_per_step=2`. This gave an overlap of 75%. 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 5 scales `scales = [0.8,1.2,1.5,2.0,2.5]` using YCrCb 3-channel HOG features plus spatially binned (`spatial_size=(16,16)`) color and histograms of color (`8 bins`) in the feature vector, which provided a nice result. Then I used Hog sub sampling on each scale to extract features for the entire image subsection defined by `y_start_stop=[400,656]` to extract the HOG features for each window. Then I used the combined features to make a prediction of car or not car as well as get a confidence score to judge how sure the model is that it is a car. I calculate the  bounding boxes for all those windows which were predicted as car and had a confidence score greater than 0.8 and returned this list. This gave the following results on the sample test image.

![alt text][initial_output]

The complete list of intermediate outputs are available [here](https://github.com/dranzerashi/CarND-Vehicle-Detection/tree/master/output_images/initial_output)

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://github.com/dranzerashi/CarND-Vehicle-Detection/blob/master/res_vid.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for this step is contained in lines 112 through 141 and 209 to 224 of the file called `util_functions.py`  

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 
I also maintained a history of the recent 10 frames of heats that were detected and used a sum of those heats to smoothen out the overall detection of the car. I used a threshold of 2 as I was receiving minimal false positives. This allowed me to detect the car towards the horizon better.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is the output of the corresponding heatmap:

![alt text][heatmap_output]

The complete list of intermediate outputs are available [here](https://github.com/dranzerashi/CarND-Vehicle-Detection/tree/master/output_images/heatmap_output)

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][final_output]

The complete list of intermediate outputs are available [here](https://github.com/dranzerashi/CarND-Vehicle-Detection/tree/master/output_images/final_output)


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My Initial problem was the linear SVC was giving a couple of false positives. I tried runnnig GridSearchCV to find best parameters and kernel choices. This gave me an rbf kernel that gave very good results. However the model was extremely slow to both train as well as predict with training taking almost an hour and a half while prediction was taking almost 75seconds on average on a single frame. This meant a time of over 13 Hours to completely detect objects in the video. I initially thought that machine hardware could have been the issue as LinearSVC was predicting at 2 to 4 seconds per frame. However on taking a dump of the classifier I found that the classifier was nearly 140MB and was practically unusable. So I settled on the better performing LinearSVC model.
The output of these non linear models is available [here](https://github.com/dranzerashi/CarND-Vehicle-Detection/tree/master/output_images/nonlinear_classifier)

I also found it difficult to strike a balance between performance and accuracy. I tried tuning various parameters to do this. As videos take hours to complete trying out different parameters and thresholds also takes longer. I think this could have been because I was using a pentium processor with no gpu. But I think even with a good CPU without support for parallelism within the libraray the result could not have been any better.

I have also found that there a few false positives that have appeared due to the fact that I improvised the classifier for performance rather than accuracy, I could try to improve this by using ensemble methods to get better accuracy.

I could also try Convolutional Deep learning models like YOLO to predict the boxes directly. This would give real time performance results. 
