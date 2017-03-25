import os
import json
import numpy as np
import pandas as pd
import cv2
from sklearn.utils import shuffle
import tensorflow as tf

from keras.models import Model, Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

import progressbar as pb

## this is the Nvidia network base on
# http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
def steering_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,20), (0,0))))
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation = "relu"))
    model.add(Convolution2D(32,5,5,subsample=(2,2),activation = "relu"))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation = "relu"))
    model.add(Convolution2D(64,3,3,activation = "relu"))
    model.add(Convolution2D(64,3,3,activation = "relu"))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(100, activation = "relu"))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation = "relu"))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation = "relu"))
    model.add(Dense(1))
#     model.summary()
    return model


## I read the image and flip them 
def image_process(db):
    path = db["path"].tolist()
    images = []
    images_flipped= []

    progress = pb.ProgressBar(maxval = len(db)).start()
    progvar = 0
    
    for img in path:
        image =cv2.imread(img)
        images.append(image)
        image_flip = cv2.flip(image, 1)
        images_flipped.append(image_flip)

        progress.update(progvar)
        progvar += 1
    print("finish loading images to array")
    
    steering = db["steering"].tolist()
    db["steering_flip"] = -db['steering']
    steering_flipped = db['steering_flip'].tolist()
    #put together
    images_ready = images+images_flipped
    steering_ready = steering+steering_flipped
    
    images_set, steering_set = shuffle(images_ready, steering_ready)
    return images_set, steering_set

## I use generator to run on my macbook air so it will not use all the memory
def generator(x , y , batch_size):
    num_samples = len(x)
    while 1: # Loop forever so the generator never terminates
        shuffle(x, y)
        for offset in range(0, num_samples, batch_size):

            images = x[offset:offset+batch_size]
            angles = y[offset:offset+batch_size]

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


# reading the data from the csv file and take the path and steering angle out from the csv
#            
sample_data_file = "driving_log.csv"
sample_data = pd.read_csv(sample_data_file, sep = ",", skipinitialspace=True)

angle_adjust = 0.21

## I want to use all three camera and adjust the left and right steering angle
center = sample_data[["center","steering"]]
left = sample_data[["left","steering"]]
right = sample_data[["right","steering"]]

left['steering'] = left['steering'] + angle_adjust
right['steering'] = right['steering'] - angle_adjust

center.columns=["path","steering"]
left.columns=["path","steering"]
right.columns=["path","steering"]

## put toghter
data = left.append(center).append(right)




epoch = 10
batch = 64

x_train, y_train = image_process(data)
x_test, y_test = image_process(center)
train_generator = generator(x_train, y_train, batch)
test_generator = generator(x_test, y_test, batch)

## i use adam optimizer
## for the validation I use mse, but the best validation for this project is actually running the car simulator 

model = steering_model()
model.compile(optimizer="adam", loss = "mse")
history = model.fit_generator(train_generator, samples_per_epoch= len(x_train), validation_data=test_generator, 
                    nb_val_samples=len(x_test), nb_epoch=epoch)


model.save('model.h5')

print("model trained")


json_str = model.to_json()
with open('model.json','w') as f:
    f.write(json_str)
print('json file saved')


model.summary()