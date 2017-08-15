# coding: utf-8

# In[20]:

import csv
import matplotlib.pyplot as plt
from matplotlib.colors import ColorConverter
from PIL import Image
import numpy as np
import cv2
import gc
from sklearn.model_selection import train_test_split
import os
import csv
import sklearn
import random # for shuffle
import time

gc.collect()
#import h5py

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Lambda
from keras.layers import Conv2D, Dropout, MaxPooling2D
from keras.layers.normalization import BatchNormalization

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def generator(samples,batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            
            for batch_sample in batch_samples:
                center_angle = float(batch_sample[3])
                cam_position_steering_offsets = [0, 0.25, -0.25]
                
                for image_position in range(len(['center','left','right'])):
                    image_subpath = batch_sample[image_position].split('/')[-1]
                    image_path = '../data/IMG/'+image_subpath
                    image = cv2.imread(image_path)
                    image = image/127.5
                    #image = sample_images[image_subpath]
                    angle_for_image = center_angle + cam_position_steering_offsets[image_position]
                    images.append(image)
                    angles.append(angle_for_image)
                    
                    # also add flipped image
                    #flipped_image = np.fliplr(image)
                    #flipped_angle_for_image = -angle_for_image
                    #images.append(image)
                    #images.append(angle_for_image)
         
            # trim image to only see section with road
            X_train = np.array(images)
            #print(X_train.shape)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# In[23]:

samples = []
first = True
#with open('../data/driving_log.csv') as csvfile:
with open('../data/driving_log.csv') as csvfile: 
    reader = csv.reader(csvfile)
    for line in reader:
        if not first:
            samples.append(line)
        else:
            first = False
            
random.shuffle(samples)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

BATCH_SIZE = 32

train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

img_input_shape=(160,320,3)
batch_size=256
epochs=15

# Crop the image - top 60px and bottom 0px
# ((top_crop, bottom_crop), (left_crop, right_crop)) = 
crop_pattern = ((70,25),(0,0))
model = Sequential()
#model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=img_input_shape))
model.add(Cropping2D(cropping=crop_pattern, input_shape=img_input_shape))
model.add(Conv2D(24,(5,5),padding='valid', activation='relu', strides=(2,2)))
model.add(Conv2D(36,(5,5),padding='valid', activation='relu', strides=(2,2)))
model.add(Conv2D(48,(5,5),padding='valid', activation='relu', strides=(2,2)))
model.add(Conv2D(64,(3,3),padding='valid', activation='relu', strides=(1,1)))
model.add(Conv2D(64,(3,3),padding='valid', activation='relu', strides=(1,1)))
model.add(Flatten())
model.add(Dense(120)) #, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(84))# , activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1)) #, activation='tanh'))


# Adam(lr=0.0001)
model.compile(loss='mse', optimizer='adam')
model.summary()

#print("before fit")
#model.fit(X_train, y_train,
#          epochs=4,
#          batch_size=128, 
#          shuffle=True,
#          validation_data=(X_validation, y_validation), verbose=1)
#print("after fit")


model.fit_generator(train_generator, steps_per_epoch=int(3*len(train_samples)/BATCH_SIZE),
                    validation_data = validation_generator,
                    validation_steps = int(3*len(validation_samples)/BATCH_SIZE), epochs=5,
                    verbose=1)

model.save('model.h5')


