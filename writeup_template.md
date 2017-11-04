#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image9]: ./examples/my_visualization.png "Visualization"
[image10]: ./examples/count_plot.png "count in training set"
[image11]: ./examples/before_preprocess.png "before preprocessing"
[image12]: ./examples/after_preprocess.png "after preprocessing"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/amirkhango/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ? 34799
* The size of the validation set is ? 4410
* The size of test set is ? 12630
* The shape of a traffic sign image is ? (32,32,3)
* The number of unique classes/labels in the data set is ? 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histgram and density chart showing how the data distributed.
1. We can see that the data distribution are similar in train/validation/test, which satisfy the basic assumption in statistical machine learning.

2. Some classes like ID=1,2,12,13,38 which mean 'Speed limit (30km/h)','Speed limit (50km/h)','Priority road','Yield','Keep right' have a large of numbers.

![density_visualization][image9]

In particular, the count plot of training set is displayed as below:
![count plot][image10]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I must preprocess the input data so that all pixels range from [-1,1). I decided to normalize the one image by simply doing (image-128)/128 . It works because raw image pixels range from 0-255.
By doing (image-128)/128, they range from [-1,0.9922].

Here is an example of a traffic sign image before and after normalization.
Before they look like:
![before preprocessing][image11]

After preprocessing, the sampled images for showing look as below, which are more realistic.
![after preprocessing][image12]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x64 	|
| RELU					|												|
| Max pooling 2x2	    | 2x2 stride, outputs 14x14x64 					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x64.  |
| RELU               	|
|  Max pooling 2x2      | 2x2 stride, valid padding, outputs 5x5x64=1600
|	Fully connected		| outputs 512.        							|
| 	RELU                |
|	Fully connected		|	outputs 216
|	Drop out            |   keep_prob = 0.5
|   RELU                |
|	Fully connected		|	outputs 43 （i.e. logits)
|	Softmax				| 	        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an optimizer=AdamOptimizer, batch size=256, learning rate=0.001, epochs=100 and drop out=0.5.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ? 99.6%
* validation set accuracy of ? 94.3%
* test set accuracy of ? 92.7%

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?

For simplicity, the first architecture dose not clude Drop out layer.

* What were some problems with the initial architecture?

It seems overfitting, i.e. higher acc on train set, but lower acc on validation set.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I add Dropout layer after the last full connected layer.

* Which parameters were tuned? How were they adjusted and why?

I decrease epochs from 100 to 30. Because 30 epochs is enough for up 93% on validation set.

I set the dropout rate=0.5 which is an empirical setting.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

I design 2 onv layers to capture invariant shift features in images and use dropout layer to avoid overfitting.

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web and I made the original pictures to suit the CNN input size:

[//]: # (Image References)
[image4]: ./traffic-signs-data/online_pic/pre_img1.png "image4"
[image5]: ./traffic-signs-data/online_pic/pre_img2.jpg "mumbaistampede"
[image6]: ./traffic-signs-data/online_pic/pre_img3.png "mumbaistampede"
[image7]: ./traffic-signs-data/online_pic/pre_img4.jpg "mumbaistampede"
[image8]: ./traffic-signs-data/online_pic/pre_img5.jpg "mumbaistampede"

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because its logo seems small.
The second, third, and fourth seem easy to be predcited.
The last one is ont the up-right corner and maybe difficult to be predicted.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


