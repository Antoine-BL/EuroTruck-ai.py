import tensorflow as tf

from tensorflow.python import Constant
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.initializers import TruncatedNormal
from tensorflow.python.keras.layers import BatchNormalization, Conv2D, Flatten, Dense, Activation, Dropout, Lambda
from tensorflow.python.keras.regularizers import l2


def atan_layer(x):
    return tf.multiply(tf.atan(x), 2)


def atan_layer_shape(input_shape):
    return input_shape


def create_model(dim_x, dim_y) -> Sequential:
    print('defining model...')
    model = Sequential()

    model.add(Conv2D(1, kernel_size=(5, 5),
                     strides=(2, 2),
                     activation='relu',
                     kernel_initializer=TruncatedNormal(stddev=0.1),
                     bias_initializer=Constant(0.1),
                     input_shape=(dim_y, dim_x, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(24, kernel_size=(5, 5),
                     strides=(2, 2), activation='relu',
                     kernel_initializer=TruncatedNormal(stddev=0.1),
                     bias_initializer=Constant(0.1),
                     activity_regularizer=l2(1e-3)))
    model.add(BatchNormalization())
    model.add(Conv2D(36, kernel_size=(5, 5),
                     strides=(2, 2), activation='relu',
                     kernel_initializer=TruncatedNormal(stddev=0.1),
                     bias_initializer=Constant(0.1),
                     activity_regularizer=l2(1e-3)))
    model.add(BatchNormalization())
    model.add(Conv2D(48, kernel_size=(3, 3),
                     strides=(1, 1),
                     activation='relu',
                     kernel_initializer=TruncatedNormal(stddev=0.1),
                     bias_initializer=Constant(0.1),
                     activity_regularizer=l2(1e-3)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3),
                     strides=(1, 1),
                     activation='relu',
                     kernel_initializer=TruncatedNormal(stddev=0.1),
                     bias_initializer=Constant(0.1),
                     activity_regularizer=l2(1e-3)))
    model.add(BatchNormalization())

    print('adding flatten layer...')
    model.add(Flatten())

    print('defining network layers...')
    model.add(Dense(1164, activation='linear',
                    kernel_initializer=TruncatedNormal(stddev=0.1),
                    bias_initializer=Constant(0.1),
                    activity_regularizer=l2(1e-3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(100, activation='linear',
                    kernel_initializer=TruncatedNormal(stddev=0.1),
                    bias_initializer=Constant(0.1),
                    activity_regularizer=l2(1e-3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(50, activation='linear',
                    kernel_initializer=TruncatedNormal(stddev=0.1),
                    bias_initializer=Constant(0.1),
                    activity_regularizer=l2(1e-3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10, activation='linear',
                    kernel_initializer=TruncatedNormal(stddev=0.1),
                    bias_initializer=Constant(0.1),
                    activity_regularizer=l2(1e-3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1,
                    kernel_initializer=TruncatedNormal(stddev=0.1),
                    bias_initializer=Constant(0.1),
                    activity_regularizer=l2(1e-3)))

    model.add(Lambda(atan_layer, output_shape=atan_layer_shape, name="atan_0"))

    print(model.summary())

    return model
