"""
ImageNet dataset specification.

Adapted from
https://github.com/uber/horovod/blob/master/examples/keras_imagenet_resnet50.py
"""

# Externals
import keras
from keras.preprocessing.image import ImageDataGenerator

def get_datasets(batch_size, train_dir, valid_dir):
    train_gen = ImageDataGenerator(
        preprocessing_function=keras.applications.resnet50.preprocess_input,
        width_shift_range=0.33, height_shift_range=0.33, zoom_range=0.5,
        horizontal_flip=True)
    test_gen = ImageDataGenerator(
        preprocessing_function=keras.applications.resnet50.preprocess_input,
        zoom_range=(0.875, 0.875))
    train_iter = train_gen.flow_from_directory(train_dir, batch_size=batch_size,
                                               target_size=(224, 224), shuffle=True)
    test_iter = train_gen.flow_from_directory(valid_dir, batch_size=batch_size,
                                              target_size=(224, 224), shuffle=True)
    return train_iter, test_iter
