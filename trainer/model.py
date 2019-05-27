from tensorflow.python.keras.layers import Lambda, Convolution2D, BatchNormalization, Flatten, Dense, Cropping2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam

from trainer.util import normalize, resize


def model():
    m = Sequential()

    m.add(Convolution2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu', input_shape=(66, 200, 1)))
    m.add(BatchNormalization())

    m.add(Convolution2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    m.add(BatchNormalization())

    m.add(Convolution2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    m.add(BatchNormalization())

    m.add(Convolution2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    m.add(BatchNormalization())

    m.add(Convolution2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    m.add(BatchNormalization())

    m.add(Flatten())

    m.add(Dense(1164, activation='relu'))
    m.add(BatchNormalization())

    m.add(Dense(200, activation='relu'))
    m.add(BatchNormalization())

    m.add(Dense(50, activation='relu'))
    m.add(BatchNormalization())

    m.add(Dense(10, activation='relu'))
    m.add(BatchNormalization())

    # Output layer
    m.add(Dense(1))

    m.compile(loss="MSE", optimizer=Adam(lr=0.001))

    print(m.summary())
    return m

