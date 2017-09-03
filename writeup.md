**Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./rate_of_each_class.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Here is a link to my [project code]
(https://github.com/ngard/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)
and its [html]
(https://github.com/ngard/CarND-Traffic-Sign-Classifier-Project/blob/master/report.html).


###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is 32.
* The number of unique classes/labels in the data set is 43.

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how much is the rate of images in each class out of all the images in Training, Validation and Test datasets.

Eventhough each class has different number of images, the trends are very similar. Therefore I decided not to manipulate the number of images in each class.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

For preprocessing, I just executed normalization, which seems essential for deep learning, since I wanted to focus on composing neural network itself this time.

Also, I did not execute grayscaling since it just seems for me that we dispose the detail of the input images.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is based on VGG16 (but fewer layer because the input images are much smaller than original VGG16 neural network) and is consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x128 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x128	 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x256 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x256 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x256 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x256	 				|
| Fully connected		| 4096 to 1024 									|
| RELU					|												|
| Fully connected		| 1024 to 256 									|
| RELU					|												|
| Fully connected		| 256  to 43 									|
| Softmax				| 	           									|


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer.
Learning rate was 0.0001 which is low enough to gradually settle the model to the best parameters.

Epoch was 350 which is about 15 times more than LeNet because I used a deeper neural network which is consisted of much more parameters inside and it takes time to tune all the parameters.

Batch size was 256. I tried bigger batch sizes, however, it seems it does not contribute to achieve higher accuracy, therefore, I settled the batch size to the initial value.

It took more than one hour to train my model even on my GTX1080.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 94.5% 
* test set accuracy of 93.8%

First, I tried to fine tune LeNet model. For example, deepening each depth or adding one more layer.
However, I found that those small approaches does not contribute to higher the precision but usually results in much worse result.

Then I researched on the web and found that mimic famous model could realize better precision.

I decided to mimic VGG16 network which is much deeper model than LeNet but not so deep as current state-of-the-art networks which require much more computing power.

I designed a network which is a deeper but basic network without techniques such as dropout or multi-scale and trained it for several hundreds epochs.

Then I found that it realizes an accuracy high enough to pass this project.

I tried to implement gradual decreasing learning rate with tf.train.exponential_decay() to realize faster convergence and higher accuracy, however, with the function, the learning fails with bad accuracy.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

<img src="https://github.com/ngard/CarND-Traffic-Sign-Classifier-Project/blob/master/new_data/01.jpg?raw=true" width="200px">
<img src="https://github.com/ngard/CarND-Traffic-Sign-Classifier-Project/blob/master/new_data/02.jpg?raw=true" width="200px">
<img src="https://github.com/ngard/CarND-Traffic-Sign-Classifier-Project/blob/master/new_data/03.jpg?raw=true" width="200px">
<img src="https://github.com/ngard/CarND-Traffic-Sign-Classifier-Project/blob/master/new_data/04.jpg?raw=true" width="200px">
<img src="https://github.com/ngard/CarND-Traffic-Sign-Classifier-Project/blob/master/new_data/05.jpg?raw=true" width="200px">

The first image might be difficult to classify because there are lots of patterns in the background of the image.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image							        |     Prediction	        					| 
|:-------------------------------------:|:---------------------------------------------:| 
| Road Work      				    	| Road Work   									| 
| End of all speed and passing limits   | End of all speed and passing limits 			|
| No Entry	 	   	   					| No Entry										|
| Speed limit (30km/h)		     		| Speed limit (30km/h)							|
| Stop		  							| Stop		        							|


The model was able to correctly guess all of the traffic signs, which gives an accuracy of 100%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

As shown in the cell, my classifier classified all the new images with 100% sure.

That is probably because I chose too clean images for this problem.
