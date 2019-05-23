"""
ResNet models for Keras.
Implementations have been adapted from keras_applications/resnet50.py
"""

# Externals
import keras
from keras import backend, layers, models, regularizers

def identity_block(input_tensor, kernel_size, filters, stage, block,
		   l2_reg=5e-5, bn_mom=0.9):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        l2_reg: L2 weight regularization (weight decay)
        bn_mom: batch-norm momentum

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(l2_reg),
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a',
                                  momentum=bn_mom, epsilon=1e-5)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(l2_reg),
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b',
                                  momentum=bn_mom, epsilon=1e-5)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(l2_reg),
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c',
                                  momentum=bn_mom, epsilon=1e-5)(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), l2_reg=5e-5, bn_mom=0.9):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
        l2_reg: L2 weight regularization (weight decay)
        bn_mom: batch-norm momentum

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(l2_reg),
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(l2_reg),
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(l2_reg),
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             kernel_regularizer=regularizers.l2(l2_reg),
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def ResNet50(input_shape=(224, 224, 3), n_classes=1000,
             l2_reg=5e-5, bn_mom=0.9):
    """Instantiates the ResNet50 architecture.

    # Arguments
        input_shape: input shape tuple. It should have 3 input channels.
        n_classes: number of classes to classify images.
        l2_reg: L2 weight regularization (weight decay)
        bn_mom: batch-norm momentum

    # Returns
        A Keras model instance.
    """
    img_input = layers.Input(shape=input_shape)

    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='valid',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(l2_reg),
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1),
                   l2_reg=l2_reg, bn_mom=bn_mom)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b',
                       l2_reg=l2_reg, bn_mom=bn_mom)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c',
                       l2_reg=l2_reg, bn_mom=bn_mom)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a',
                   l2_reg=l2_reg, bn_mom=bn_mom)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b',
                       l2_reg=l2_reg, bn_mom=bn_mom)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c',
                       l2_reg=l2_reg, bn_mom=bn_mom)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d',
                       l2_reg=l2_reg, bn_mom=bn_mom)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a',
                   l2_reg=l2_reg, bn_mom=bn_mom)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b',
                       l2_reg=l2_reg, bn_mom=bn_mom)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c',
                       l2_reg=l2_reg, bn_mom=bn_mom)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d',
                       l2_reg=l2_reg, bn_mom=bn_mom)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e',
                       l2_reg=l2_reg, bn_mom=bn_mom)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f',
                       l2_reg=l2_reg, bn_mom=bn_mom)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a',
                   l2_reg=l2_reg, bn_mom=bn_mom)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b',
                       l2_reg=l2_reg, bn_mom=bn_mom)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c',
                       l2_reg=l2_reg, bn_mom=bn_mom)

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(n_classes, activation='softmax',
                     kernel_regularizer=regularizers.l2(l2_reg),
                     name='fc1000')(x)

    return models.Model(img_input, x, name='resnet50')


def ResNetSmall(input_shape=(32, 32, 3), n_classes=10,
                l2_reg=5e-5, bn_mom=0.9):
    """Instantiates the small ResNet architecture.

    # Arguments
        input_shape: input shape tuple. It should have 3 input channels.
        n_classes: number of classes to classify images.
        l2_reg: L2 weight regularization (weight decay)
        bn_mom: batch-norm momentum

    # Returns
        A Keras model instance.
    """
    img_input = layers.Input(shape=input_shape)

    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = img_input
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(l2_reg),
                      name='conv1')(img_input)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)

    x = conv_block(x, 3, [64, 64, 64], stage=2, block='a',
                   l2_reg=l2_reg, bn_mom=bn_mom)
    x = identity_block(x, 3, [64, 64, 64], stage=2, block='b',
                       l2_reg=l2_reg, bn_mom=bn_mom)

    x = conv_block(x, 3, [128, 128, 128], stage=3, block='a',
                   l2_reg=l2_reg, bn_mom=bn_mom)
    x = identity_block(x, 3, [128, 128, 128], stage=3, block='b',
                       l2_reg=l2_reg, bn_mom=bn_mom)

    x = conv_block(x, 3, [256, 256, 256], stage=4, block='a',
                   l2_reg=l2_reg, bn_mom=bn_mom)
    x = identity_block(x, 3, [256, 256, 256], stage=4, block='b',
                       l2_reg=l2_reg, bn_mom=bn_mom)

    x = conv_block(x, 3, [512, 512, 512], stage=5, block='a',
                   l2_reg=l2_reg, bn_mom=bn_mom)
    x = identity_block(x, 3, [512, 512, 512], stage=5, block='b',
                       l2_reg=l2_reg, bn_mom=bn_mom)

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(n_classes, activation='softmax',
                     kernel_regularizer=regularizers.l2(l2_reg),
                     name='fc1000')(x)

    return models.Model(img_input, x, name='resnet50')
