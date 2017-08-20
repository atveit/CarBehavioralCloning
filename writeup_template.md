#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model]: model.png "Model Visualization"

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* train_car_to_drive.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The train_car_to_drive.ipynb file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

Used the NVIDIA Autopilot Deep Learning model for self-driving as inspiration (ref: paper "End to End Learning for Self-Driving Cars" - https://arxiv.org/abs/1604.07316 and implementation of it: https://github.com/0bserver07/Nvidia-Autopilot-Keras), but did some changes to it:

1. Added normalization in the model itself (ref Lambda(lambda x: x/255.0 - 0.5, input_shape=img_input_shape)), since it is likely to be faster than doing it in pure Python.
2. Added Max Pooling after the first convolution layers, i.e. making the model a more "traditional" conv.net wrt being capable of
   detecting low level features such as edges (similar to classic networks such as LeNet). 
3. Added Batch Normalization in early layers to be more robust wrt different learning rates
4. Used he_normal normalization (truncated normal distribution) since this type of normalization with TensorFlow has earlier mattered a lot
5. Used L2 regularizer (ref: "rule of thumb" - https://www.quora.com/What-is-the-difference-between-L1-and-L2-regularization-How-does-it-solve-the-problem-of-overfitting-Which-regularizer-to-use-and-when )
6. Made the model (much) smaller by reducing the fully connected layers (got problems running larger model on 1070 card, but in retrospect it was not the model size but my misunderstandings of Keras 2 that caused this trouble)
7. Used selu (ref: paper "Self-Normalizing Neural Networks" https://arxiv.org/abs/1706.02515) instead of relu as rectifier functions in later layers (fully connected) - since previous experience have shown (with traffic sign classification and tensorflow) showed that using selu gave faster convergence rates (though not better final result). 
8. Used dropout in later layers to avoid overfitting
9. Used l1 regularization on the final layer, since I've seen that it is good for regression problems (better than l2)

Image of Model
![model Image][model]

Detailed model
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 85, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 41, 158, 24)       1824      
_________________________________________________________________
batch_normalization_1 (Batch (None, 41, 158, 24)       96        
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 21, 79, 24)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 9, 38, 36)         21636     
_________________________________________________________________
batch_normalization_2 (Batch (None, 9, 38, 36)         144       
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 3, 17, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 1, 15, 64)         27712     
_________________________________________________________________
flatten_1 (Flatten)          (None, 960)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 400)               384400    
_________________________________________________________________
dropout_1 (Dropout)          (None, 400)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 100)               40100     
_________________________________________________________________
dropout_2 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_4 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 11        
=================================================================
Total params: 524,731
Trainable params: 524,611
Non-trainable params: 120
```


####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (ref dropout_1 and dropout_2 in figure above and train_car_to_drive.ipynb). 

Partially related: Used also balancing of data sets in the generator, see sample_weight in generator function and snippet below
```
            sample_weight = sklearn.utils.class_weight.compute_sample_weight('balanced', y_train)
	    ...
            yield sklearn.utils.shuffle(X_train, y_train, sample_weight)

```

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. See mp4 file in this github repository.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually

####4. Appropriate training data

Used the training data that was provided as part of the project, and in addition added two runs of data to avoid problems (e.g. curve without lane line on the right side - until the bridge started and also a separate training set driving on the bridge). Data is available on https://amundtveit.com/DATA0.tgz). 
###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a conv.net, first tried the previous one I used for Traffic Sign detection based on LeNet, but it didn't work (probably too big images as input), and then started with the Nvidia model (see above for details about changes to it).

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Primary finding was that numerical performance of the models I tried was not a good predictor of how well it would it perform on actual driving in the simulator. Perhaps overfitting could be good for this task (i.e. memorize track), but I attempted to get a correctly trained model without overfitting (ref. dropout/selu and batch normalization). There were many failed runs before the car actually could drive around the first track.

####2. Final Model Architecture

See above for final architecture

####3. Creation of the Training Set & Training Process

I redrove and captured training data for the sections that were problematic (as mentioned the curve without lane lines on right and the bridge and part just before bridge). Regarding center-driving I didn't get much success adding data for that, but perhaps my rebalancing (ref. generator output above) actually was counter-productive?

For each example line in the training data I generated 6 variants (for data augmentetation), i.e. flipped image (along center vertical axis) +  also used the 3 different cameras (left, center and right) with adjustments for the angle. 

After the collection process, I had 10485 lines in driving_log.csv, i.e. number of data points = 62430 (6*10485). Preprocessing used to flip image, convert images to numpy arrays and also (as part of Keras model) to scale values. Also did cropping of the image as part of the model. I finally randomly shuffled the data set and put 20 of the data into a validation set, see generator for details

#### generator
```
# dealing with unbalanced data with class_weight in Keras
# https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras

import sklearn.utils.class_weight

def generator(samples,batch_size=32, image_prefix_path="../DATA0/IMG/"):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            angle_offsets = [0.0, 0.20, -0.20]
            
            for batch_sample in batch_samples:
                center_angle = float(batch_sample[3])

                for image_position in range(3):
                    image_subpath = batch_sample[image_position].split('/')[-1]
                    image_path = image_prefix_path + image_subpath
                    image = cv2.imread(image_path)
                    images.append(image)
                    angle = center_angle + angle_offsets[image_position]
                    angles.append(angle)

                    # also add flipped image and angle
                    flipped_image = np.fliplr(image)
                    flipped_angle_for_image = -angle
                    images.append(flipped_image)
                    angles.append(flipped_angle_for_image)

            X_train = np.array(images)
            y_train = np.array(angles)
            sample_weight = sklearn.utils.class_weight.compute_sample_weight('balanced', y_train)
            yield sklearn.utils.shuffle(X_train, y_train, sample_weight)
```

I used this training data for training the model. The validation helped determine if the model was over or under fitting. The ideal
number of epochs was 5 as evidenced by the quick flattening of loss
and validation loss (to around 0.03), in earlier runs validation loss
increased above training loss when having more epochs.  I used an adam
optimizer so that manually training the learning rate wasn't
necessary.

```
Epoch 1/5
178s - loss: 0.2916 - val_loss: 0.0603
Epoch 2/5
180s - loss: 0.0530 - val_loss: 0.0463
Epoch 3/5
181s - loss: 0.0398 - val_loss: 0.0330
Epoch 4/5
179s - loss: 0.0309 - val_loss: 0.0326
Epoch 5/5
178s - loss: 0.0302 - val_loss: 0.0312
```

####4. Challenges

Challenges along the way - found it to be a very hard task, since the
model loss and validation loss weren't good predictors for actual
driving performance, also had cases when adding more training data
with nice driving data (at the center and far from the edges) actually
gave worse results and made the car drive off the road. Other
challenges were Keras 2 related, the semantics of parameters in Keras
1 and Keras 2 fooled me a bit using Keras 2, ref the
steps_per_epoch. Also had issues with the progress bar not working in
Keras 2 in Jupyter notebook, so had to use 3rd party library
https://pypi.python.org/pypi/keras-tqdm/2.0.1

