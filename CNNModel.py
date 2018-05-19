#!/usr/bin/env python3

"""
We are defining a custom class CNNModel to encapsulate a CNN model
created through keras and manipulate it easily.
"""

import numpy as np

from helpers_loading import *
from helpers_submission import *
from helpers_visualization import *
from helpers_cnn import *

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import backend as K
from keras.regularizers import l2
from keras.layers import LeakyReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils.np_utils import to_categorical

class CNNModel:
       
    def __init__(self, window_size=32, model_constructor=construct_keras_model_4conv, classes=2):
        """
        Constructs a CNN classifier with the specified model constructor (see helpers_cnn).
        """
        
        self.patch_size = 16
        self.window_size = window_size
        self.padding = (self.window_size - self.patch_size) // 2
        self.classes = classes
        self.model = model_constructor(self.window_size, self.classes)
        
    
    def train(self, X, Y, epochs=200, batch_size=125, steps_per_epoch=100, image_augmentation=True, lr=0.001):
        """
        Train the model with the specified parameters. The training will stop early if there's no
        improvement after some time. The image augmentation automatically enhance the data set with
        rotations or symmetries.
        """

        print('Training set shape: ', X.shape)

        opt = Adam(lr=lr) # Adam optimizer with default initial learning rate
        self.model.compile(loss='binary_crossentropy',
                      optimizer=opt,
                      metrics=['acc'])
        
        X_new = np.empty((X.shape[0],
                         X.shape[1] + 2*self.padding, X.shape[2] + 2*self.padding,
                         X.shape[3]))
        Y_new = np.empty((Y.shape[0],
                         Y.shape[1] + 2*self.padding, Y.shape[2] + 2*self.padding))
        for i in range(X.shape[0]):
            X_new[i] = pad_image(X[i], self.padding)
            Y_new[i] = pad_image(Y[i], self.padding)
        X = X_new
        Y = Y_new
        
        def generate_minibatch():
            """
            Function called at each step of the training, returning a minibatch
            created randomly.
            """
            while 1:
                # Generate one minibatch
                X_batch = np.empty((batch_size, self.window_size, self.window_size, 3))
                Y_batch = np.empty((batch_size, 2))
                for i in range(batch_size):
                    # Select a random image
                    idx = np.random.choice(X.shape[0])
                    shape = X[idx].shape
                    
                    # Sample a random window from the image
                    center = np.random.randint(self.window_size//2, shape[0] - self.window_size//2, 2)
                    sub_image = X[idx][center[0]-self.window_size//2:center[0]+self.window_size//2,
                                       center[1]-self.window_size//2:center[1]+self.window_size//2]
                    gt_sub_image = Y[idx][center[0]-self.patch_size//2:center[0]+self.patch_size//2,
                                          center[1]-self.patch_size//2:center[1]+self.patch_size//2]
                    
                    
                    label = value_to_class(np.mean(gt_sub_image))
                    
                    # Image augmentation
                    if image_augmentation:
                        # Random flip
                        if np.random.choice(2) == 0:
                            # Flip vertically
                            sub_image = np.flipud(sub_image)
                        if np.random.choice(2) == 0:
                            # Flip horizontally
                            sub_image = np.fliplr(sub_image)

                        # Random rotation in steps of 90°
                        num_rot = np.random.choice(4)
                        sub_image = np.rot90(sub_image, num_rot)

                    label = to_categorical(label, self.classes)
                    X_batch[i] = sub_image
                    Y_batch[i] = label
                
                yield (X_batch, Y_batch)

        # This callback reduces the learning rate when the training accuracy does not improve any more
        lr_callback = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=5,
                                        verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
        
        # Stops the training process upon convergence
        stop_callback = EarlyStopping(monitor='acc', min_delta=0.0001, patience=11, verbose=1, mode='auto')
        
        try:
            self.model.fit_generator(generate_minibatch(),
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[lr_callback, stop_callback])
        except KeyboardInterrupt:
            # Do not throw away the model in case the user stops the training process
            pass

        print('\nTraining completed')
        
    def save(self, filename):
        """
        Saves the parameters of the model.
        """
        self.model.save_weights(filename)
        
    def load(self, filename):
        """
        Loads the parameters for the model.
        """
        self.model.load_weights(filename)
        
    def predict(self, X):
        """
        Returns an array of predictions for the given images.
        """
        # Subdivide the images into blocks
        img_patches = extract_patches(X, w=self.patch_size, h=self.patch_size, stride=self.patch_size, padding=self.padding)
        
        # Run prediction
        Y = self.model.predict(img_patches)
        Y = (Y[:,0] < Y[:,1]) * 1
        
        # Regroup patches into images
        return labels_to_images(Y, n=X.shape[0], w=self.patch_size, h=self.patch_size, imgwidth=X.shape[1], imgheight=X.shape[2])
        