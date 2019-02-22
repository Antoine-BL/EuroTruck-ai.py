from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, BatchNormalization


def create_model() -> Sequential:
    dim_x = 200
    dim_y = 66

    print('defining model...')
    model = Sequential()

    print('defining conv layers...')
    model.add(Conv2D(1, kernel_size=(5,5), strides=(2, 2), activation='relu', input_shape=(dim_y, dim_x, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(24, kernel_size=(5,5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, kernel_size=(5,5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, kernel_size=(3,3), strides=(1, 1), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3,3), strides=(1, 1), activation='relu'))

    print('adding flatten layer...')
    model.add(Flatten())

    print('defining network layers...')
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='relu'))

    print(model.summary())

    return model
