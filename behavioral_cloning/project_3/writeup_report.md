**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)


[image3]: ./report_image/center_2016_12_01_13_30_48_287.jpg "center Image"
[image4]: ./report_image/left_2016_12_01_13_30_48_287.jpg "left Image"
[image5]: ./report_image/right_2016_12_01_13_30_48_287.jpg "right Image"
[image6]: ./report_image/center_2016_12_01_13_33_00_336.jpg "center Image"
[image7]: ./report_image/center_flip_2016_12_01_13_33_00_336.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network base on Nvidia with some adjustment, http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf (model.py lines 23-39) 

The model includes RELU layers to introduce nonlinearity (code line 27-31), and the data is normalized in the model using a Keras lambda layer (code line 25). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 33,35,37). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 126). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 125).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. And add some recovering data for some section that the car may run out from the road.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to keep the car stay on the road without correct more data. 

My first step was to use a convolution neural network model similar to the Nvidia network provide from the "End to End Learning for Self-Driving Cars". I thought this model might be appropriate because recently I read a paprt about Nvidia's self driving car  driving through Lombard street in San Francisco.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that add three layer of dropout after flatten. 

Then I adjust the angles adjectment for the left and right steeing angle.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, the left turn before the bridge and the left trun follow by right turn next to the water. to improve the driving behavior in these cases, I increase the epoch to 10 and also cropping the top and bottome of the image. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 23-39) consisted of a convolution neural network with the following layers and layer sizes 15. I include the model summary and vildation in the ipython notebook.

####3. Creation of the Training Set & Training Process

I use the sample provide by the Udacity, and here is the left, center, right camera image. 

![alt text][image3]
![alt_text][image2]
![alt_text][image4]



To augment the data set, I also flipped images this would increase the data set and give right turn image to the model. before I added the flip images, most of the images are left turnning, which make the outcome only know about left turn and preform badly on right turn. And here is the flipped images.

![alt text][image6]
![alt text][image7]


After the collection process, I had six times of number of data points. I then preprocessed this data by cropping the top and the bottom of the image.


I finally randomly shuffled the data set and I use the center data set as the validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
