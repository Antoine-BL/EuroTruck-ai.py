import os
from datetime import datetime

import numpy as np
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.backend import resize_images
from tensorflow.python.keras.layers import Lambda, Convolution2D, BatchNormalization, Flatten, Dense
from tensorflow.python.keras.models import load_model, Sequential
from tensorflow.python.keras.optimizers import Adam

from keras_tools.DataGenerator import DataGenerator
from keras_tools.Model import create_model

DATA_PATH = os.path.realpath('.\\data')
START_MODEL_NAME = ''
MODEL_NAME = 'PNet-better-aug-50e'
MODEL_FORMAT = 'models/{}.h5'


def resize(img):
    return resize_images(img, 66, 200, 'channels_first')


def normalize(img):
    return (img / 255.0) - 0.5


def crop(img):
    pos_x = 108
    pos_y = 128
    dim_y = 252
    dim_x = 126
    return img[pos_y:pos_y + dim_y, pos_x:pos_x + dim_x]


def nvidia_model():
    model = Sequential()

    model.add(Lambda(crop, input_shape=(270, 480, 1)))

    model.add(Lambda(normalize))
    model.add(Lambda(resize))

    model.add(Convolution2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())

    model.add(Convolution2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())

    model.add(Convolution2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())

    model.add(Convolution2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(BatchNormalization())

    model.add(Convolution2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(1164, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(200, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(50, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(10, activation='relu'))
    model.add(BatchNormalization())

    # Output layer
    model.add(Dense(1))

    model.compile(loss="MSE", optimizer=Adam(lr=0.001))
    return model


if __name__ == "__main__":
    nvidia_model()

