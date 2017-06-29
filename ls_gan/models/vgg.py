import warnings

from keras.models import Sequential

from keras.layers import Flatten, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


def vgg(weights_fname=None, include_pred_layer=True):
    
    m = Sequential()
    
    # Block 1
    m.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1',
          input_shape=(32,32,3)))
    m.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    m.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    m.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    m.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    m.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    m.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    m.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    m.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    m.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
    
    # Fully Connected layers
    m.add(Flatten(name='f_new'))
    m.add(Dense(256, activation='relu', name='fc1_new'))
    
    if include_pred_layer:
        m.add(Dropout(0.5, name='do1_new'))
        m.add(Dense(10, activation='sigmoid', name='fc2_new'))
        
    if weights_fname is not None:
        m.load_weights(weights_fname, by_name=True)

    return m