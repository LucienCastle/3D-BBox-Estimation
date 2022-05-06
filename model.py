from keras.layers import Input, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dropout, Reshape, Lambda
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50
import tensorflow as tf


def l2_normalize(x):
    return tf.nn.l2_normalize(x, dim=2)


def bbox_3D_net(input_shape=(224, 224, 3), weights=None, freeze_model=False, bin_num=6):
    # vgg16_model = VGG16(include_top=False, weights=vgg_weights, input_shape=input_shape)
    resnet50_model = ResNet50(include_top=False, weights=weights, input_shape=input_shape)

    if freeze_model:
        for layer in resnet50_model.layers:
            layer.trainable = False

    x = Flatten()(resnet50_model.output)
    
    # bbox_reg = Dense(128)(x)
    # bbox_reg = LeakyReLU(alpha=0.1)(bbox_reg)
    # bbox_reg = Dropout(0.5)(bbox_reg)
    # bbox_reg = Dense(units=4, name='bbox')(bbox_reg)

    dimension = Dense(512)(x)
    dimension = LeakyReLU(alpha=0.1)(dimension)
    dimension = Dropout(0.5)(dimension)
    dimension = Dense(3)(dimension)
    dimension = LeakyReLU(alpha=0.1, name='dimension')(dimension)

    orientation = Dense(256)(x)
    orientation = LeakyReLU(alpha=0.1)(orientation)
    orientation = Dropout(0.5)(orientation)
    orientation = Dense(bin_num * 2)(orientation)
    orientation = LeakyReLU(alpha=0.1)(orientation)
    orientation = Reshape((bin_num, -1))(orientation)
    orientation = Lambda(l2_normalize, name='orientation')(orientation)

    confidence = Dense(256)(x)
    confidence = LeakyReLU(alpha=0.1)(confidence)
    confidence = Dropout(0.5)(confidence)
    confidence = Dense(bin_num, activation='softmax', name='confidence')(confidence)

    model = Model(resnet50_model.input, outputs=[dimension, orientation, confidence])
    return model