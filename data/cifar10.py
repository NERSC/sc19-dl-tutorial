"""
CIFAR10 dataset specification.

https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
"""

# Externals
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

def get_datasets(batch_size, n_train=None, n_valid=None):
    """
    Load the CIFAR10 data and construct pipeline.
    """
    (x_train, y_train), (x_valid, y_valid) = cifar10.load_data()

    # Normalize pixel values
    x_train = x_train.astype('float32') / 255
    x_valid = x_valid.astype('float32') / 255

    # Select subset of data if specified
    if n_train is not None:
        x_train, y_train = x_train[:n_train], y_train[:n_train]
    if n_valid is not None:
        x_valid, y_valid = x_valid[:n_valid], y_valid[:n_valid]

    # Convert labels to class vectors
    n_classes = 10
    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_valid = keras.utils.to_categorical(y_valid, n_classes)

    # Prepare the generators with data augmentation
    train_gen = ImageDataGenerator(width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip=True)
    valid_gen = ImageDataGenerator()
    train_iter = train_gen.flow(x_train, y_train, batch_size=batch_size, shuffle=True)
    valid_iter = valid_gen.flow(x_valid, y_valid, batch_size=batch_size, shuffle=True)
    return train_iter, valid_iter
