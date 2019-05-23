"""
Simple CNN classifier model.
"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

def build_model(input_shape=(32, 32, 3), n_classes=10, dropout=0):
    """Construct the simple CNN model"""
    conv_args = dict(kernel_size=3, padding='same', activation='relu')
    model = Sequential()
    model.add(Conv2D(16, input_shape=input_shape, **conv_args))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(32, **conv_args))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(64, **conv_args))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(n_classes, activation='softmax'))
    return model
