import cv2

import numpy as np
from tensorflow.python import keras

IMG_INDEX = 0
INPUT_INDEX = 1
JOY_INDEX = 1


class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_ids, batch_size=32, dim=(32, 32, 32), shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_ids
        self.n_channels = 1
        self.n_classes = 1
        self.shuffle = shuffle
        self.indexes = None
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty(self.batch_size, dtype=int)

        (dim_y, dim_x) = self.dim
        # Generate data
        old_file = ''
        data_set = None
        print('Batch')
        pos_x = 128
        pos_y = 147
        for i, ID in enumerate(list_IDs_temp):
            (file_name, index) = self.__sep_ID(ID)

            if old_file != file_name:
                data_set = np.load('data/' + file_name + '.npy')

            # Normalize img
            sample = data_set[int(index)]
            img = sample[IMG_INDEX]
            img = img[pos_y:pos_y + dim_y, pos_x:pos_x + dim_x]
            img = img.reshape(dim_y, dim_x, 1)

            # Normalize label
            label = (sample[INPUT_INDEX][JOY_INDEX] + 1) / 2.0
            X[i, ] = img
            y[i] = label

        return X, y

    def __sep_ID(self, ID: str):
        ID_parts = ID.split('|')
        file_name = ID_parts[0]
        index = ID_parts[1]
        return file_name, index

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
