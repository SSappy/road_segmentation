#!/usr/bin/env python3

"""
Helper functions to create different CNN models using
the framework keras (under tensorflow backend).
"""

import numpy as np

from keras.models import Sequential, Model
from keras.layers import Input, concatenate, UpSampling2D, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import backend as K
from keras.regularizers import l2
from keras.layers import LeakyReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils.np_utils import to_categorical

def construct_keras_model_basic(window_size, classes):
    """
    Basic model with 2 convolutions.
    """
    input_shape = (window_size, window_size, 3)
    
    pool_size = (2, 2)
    
    model = Sequential()

    model.add(Conv2D(32, (5, 5), # 32 5x5 filters
                            padding='same',
                            input_shape=input_shape,
                            activation='relu'))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))

    model.add(Conv2D(64, (5, 5), # 64 5x5 filters
                            padding='same',
                            activation='relu'
                               ))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))

    model.add(Flatten())
    
    model.add(Dense(1024, activation='relu'))
    #model.add(LeakyReLU(alpha=0.1))
              
    model.add(Dropout(0.4))
    
    model.add(Dense(classes, activation='softmax'))

    return model

def construct_keras_model_4conv(window_size, classes):
    """
    Model with 4 convolutions.
    """
    input_shape = (window_size, window_size, 3)
    
    # Size of pooling area for max pooling
    pool_size = (2, 2)

    reg = 1e-6 # L2 regularization factor (used on weights, but not biases)

    model = Sequential()

    model.add(Conv2D(64, (5, 5), # 64 5x5 filters
                            padding='same',
                            input_shape=input_shape,
                            activation='relu'))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), # 128 3x3 filters
                            padding='same',
                            activation='relu'
                               ))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), # 256 3x3 filters
                            padding='same',
                            activation='relu'
                            ))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), # 256 3x3 filters
                            padding='same',
                            activation='relu'
                           ))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128,
                    kernel_regularizer=l2(reg),
                    activation='relu'
                        )) # Fully connected layer (128 neurons)
    #model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))

    model.add(Dense(classes,
                    kernel_regularizer=l2(reg),
                    activation='softmax'
                        ))

    return model

def construct_keras_model_4conv_leaky(window_size, classes):
    """
    Model with 4 convolutions and leaky activation functions.
    """
    input_shape = (window_size, window_size, 3)
    
    # Size of pooling area for max pooling
    pool_size = (2, 2)

    reg = 1e-6 # L2 regularization factor (used on weights, but not biases)

    model = Sequential()

    model.add(Conv2D(64, (5, 5), # 64 5x5 filters
                            padding='same',
                            input_shape=input_shape,
                            #activation='relu'
                    ))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), # 128 3x3 filters
                            padding='same',
                            #activation='relu'
                               ))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), # 256 3x3 filters
                            padding='same',
                            #activation='relu'
                            ))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), # 256 3x3 filters
                            padding='same',
                            #activation='relu'
                           ))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128,
                    kernel_regularizer=l2(reg),
                    #activation='relu'
                        )) # Fully connected layer (128 neurons)
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))

    model.add(Dense(classes,
                    kernel_regularizer=l2(reg),
                    activation='softmax'
                        ))

    return model

def construct_keras_model_3conv_leaky(window_size, classes):
    """
    Model with 3 convolutions and leaky activation functions.
    """
    input_shape = (window_size, window_size, 3)
    
    # Size of pooling area for max pooling
    pool_size = (2, 2)

    reg = 1e-6 # L2 regularization factor (used on weights, but not biases)

    model = Sequential()

    model.add(Conv2D(64, (5, 5), # 64 5x5 filters
                            padding='same',
                            input_shape=input_shape,
                            #activation='relu'
                    ))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), # 128 3x3 filters
                            padding='same',
                            #activation='relu'
                               ))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), # 256 3x3 filters
                            padding='same',
                            #activation='relu'
                            ))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    model.add(Dropout(0.25))


    model.add(Flatten())
    model.add(Dense(128,
                    kernel_regularizer=l2(reg),
                    #activation='relu'
                        )) # Fully connected layer (128 neurons)
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))

    model.add(Dense(classes,
                    kernel_regularizer=l2(reg),
                    activation='softmax'
                        ))

    return model

def construct_keras_model_6conv_leaky(window_size, classes):
    """
    Model with 6 convolutions and leaky activation functions.
    """
    input_shape = (window_size, window_size, 3)
    
    # Size of pooling area for max pooling
    pool_size = (2, 2)

    reg = 1e-6 # L2 regularization factor (used on weights, but not biases)

    model = Sequential()

    model.add(Conv2D(64, (5, 5), # 64 5x5 filters
                            padding='same',
                            input_shape=input_shape,
                            #activation='relu'
                    ))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), # 128 3x3 filters
                            padding='same',
                            #activation='relu'
                               ))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), # 256 3x3 filters
                            padding='same',
                            #activation='relu'
                            ))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), # 256 3x3 filters
                            padding='same',
                            #activation='relu'
                           ))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(256, (3, 3), # 256 3x3 filters
                            padding='same',
                            #activation='relu'
                           ))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(256, (3, 3), # 256 3x3 filters
                            padding='same',
                            #activation='relu'
                           ))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128,
                    kernel_regularizer=l2(reg),
                    #activation='relu'
                        )) # Fully connected layer (128 neurons)
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))

    model.add(Dense(classes,
                    kernel_regularizer=l2(reg),
                    activation='softmax'
                        ))

    return model

def construct_keras_model_8conv_leaky(window_size, classes):
    """
    Model with 8 convolutions and leaky activation functions.
    """
    input_shape = (window_size, window_size, 3)
    
    # Size of pooling area for max pooling
    pool_size = (2, 2)

    reg = 1e-6 # L2 regularization factor (used on weights, but not biases)

    model = Sequential()

    model.add(Conv2D(64, (5, 5), # 64 5x5 filters
                            padding='same',
                            input_shape=input_shape,
                            #activation='relu'
                    ))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), # 128 3x3 filters
                            padding='same',
                            #activation='relu'
                               ))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), # 256 3x3 filters
                            padding='same',
                            #activation='relu'
                            ))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), # 256 3x3 filters
                            padding='same',
                            #activation='relu'
                           ))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(256, (3, 3), # 256 3x3 filters
                            padding='same',
                            #activation='relu'
                           ))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(256, (3, 3), # 256 3x3 filters
                            padding='same',
                            #activation='relu'
                           ))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(256, (3, 3), # 256 3x3 filters
                            padding='same',
                            #activation='relu'
                           ))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(256, (3, 3), # 256 3x3 filters
                            padding='same',
                            #activation='relu'
                           ))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128,
                    kernel_regularizer=l2(reg),
                    #activation='relu'
                        )) # Fully connected layer (128 neurons)
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))

    model.add(Dense(classes,
                    kernel_regularizer=l2(reg),
                    activation='softmax'
                        ))

    return model

def construct_keras_model_unet(window_size, classes):
    """
    Model unet.
    """
    input_shape = (window_size, window_size, 3)
    
    pool_size = (2, 2)
    
    inputs = Input(input_shape)
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=pool_size)(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=pool_size)(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=pool_size)(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=pool_size)(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=pool_size)(conv5), conv4])#, concat_axis=1)
    up6 = Dropout(0.5)(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=pool_size)(conv6), conv3])#, mode='concat', concat_axis=1)
    up7 = Dropout(0.5)(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=pool_size)(conv7), conv2])#, mode='concat', concat_axis=1)
    up8 = Dropout(0.5)(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=pool_size)(conv8), conv1])#, mode='concat', concat_axis=1)
    up9 = Dropout(0.5)(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    #conv10 = Conv2D(classes, (1, 1), activation='sigmoid')(conv9)
    flatten = Flatten()(conv9)
    dense = Dense(classes,
                    activation='softmax'
                        )(flatten)
    
    
    model = Model(inputs=inputs, outputs=dense)
    
    return model