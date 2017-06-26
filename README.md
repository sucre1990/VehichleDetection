
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
[image1]: ./output_images/Vehicle_non-vehicle.png
[image2]: ./output_images/LUV_hog.png
[image3]: ./output_images/win_64.png
[image4]: ./output_images/win_128.png
[image5]: ./output_images/win_192.png
[image6]: ./output_images/win_256.png
[image7]: ./output_images/win_combined.png
[image8]: ./output_images/search_example1.png
[image9]: ./output_images/search_example2.png
[image10]: ./output_images/search_example3.png

[image11]: ./output_images/detected1.png
[image12]: ./output_images/heat1.png
[image13]: ./output_images/final1.png

[image14]: ./output_images/detected2.png
[image15]: ./output_images/heat2.png
[image16]: ./output_images/final2.png

[image17]: ./output_images/detected3.png
[image18]: ./output_images/heat3.png
[image19]: ./output_images/final3.png


[video]: ./project_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it! [link to my notebook](./detection.ipynb)

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the code cell 6 of the IPython notebook (currently it is using skimage.feature's hog() function, it works but it is also slow. I am planing to replace it with cv2.hog()).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from vehichle class and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `LUV` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and selected the best set by looking at the classifier error. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The features I used include bin spatial features, color hist features and hog features. After combining them together, I implemted a normalization by using `sklearn.preprocessing.StandardScaler.fit()` and `transform()` functions. (code cell 8, 9 and 10)

Then I use `sklearn.model_selectio`'s `GridSearchCV()` function to select the svm parameters (code cell 13). In the end, I chose `kernel = 'rbf'`, and `C = 1`. It gave me around 99.5% accuracy on test data set (code cell 14)(I didn't do anything to avoid sequential images problem. It may lead the model tend to be bias and create more false negtives).


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

First, I chose the biggest window (4 * 64 = 256)and smallest window scale (64) by drawing windows on testing pictures. Then I chose two intermediate windows (2 * 64 = 128 and 3 * 64 = 192). Every window size is a integer times of 64 and it is to make scaling in window searching easier.


For the position, I obeyed the rule of object will be smaller and nearer to the earth horizon: search small windows only within the small middle part of picture and gradually espand area as window size grows. For overlapping, I majorly considered (1) adequate searching samples and (2) efficient searching window quantities, in order to guarantee enough searching size under a reasonable time.  

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using LUV 1-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image8]
![alt text][image9]
![alt text][image10]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. 

The whole reason is since I intentionally created overlapping between different window scales and was using a very high accuracy classifier. If it is true positive, there would be a really high chance of having multiple windows reporting positive, otherwise it may be a false positive.   

Here's an example result showing the heatmap from test images, the result of `scipy.ndimage.measurements.label()` and the bounding boxes:


### Here are three test images and their corresponding heatmaps:

![alt text][image11]
![alt text][image14]
![alt text][image17]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image12]
![alt text][image15]
![alt text][image18]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image13]
![alt text][image16]
![alt text][image19]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I spent over 20 hours on this project majorly addressing issues related to deciding what scales of windows and how many of them I should use. When I was testing, after I fix the scales of windows, even a slight change of overlapping or start and stop postion can lead to a different result. I will devote more time on learning how to plan window searching. (Could you provide me some guidance?)

Another thing I realized is that my current algorithm is slow. On average, I spent like 3s to process a frame. It is unrealistic to implement it in the real world. After researching in the forum, I found some ways to improve it like using cv2's hog() function (I haven't implemented yet), optimizing searching area etc.. The next step for me would be working on those two directions. (Could you also provide me some guidances? Thank you!!!!)

Thank you for your review!!!


