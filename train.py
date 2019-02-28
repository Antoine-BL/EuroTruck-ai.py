import os
import tensorflow as tf
from datetime import datetime

import numpy as np
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizers import Adam

from keras_tools.DataGenerator import DataGenerator
from keras_tools.Model import create_model

DATA_PATH = os.path.realpath('.\\data')
START_MODEL_NAME = 'PNet-atan-100e'
MODEL_NAME = 'PNet-atan-150e'
MODEL_FORMAT = 'models/{}.h5'


def main():
    print('generating partitions...')
    partition = generate_partition()
    dim_x = 280
    dim_y = 100

    params = {
        'dim': (dim_y, dim_x),
        'batch_size': 100,
        'shuffle': True
    }

    print('defining generators...')
    training_generator = DataGenerator(partition['train'], **params)
    testing_generator = DataGenerator(partition['test'], **params)

    if START_MODEL_NAME != '':
        print('Loading model...')
        model = load_model(MODEL_FORMAT.format(START_MODEL_NAME))
    else:
        print('compiling model...')
        model = create_model(dim_x, dim_y)
        optimizer = Adam(lr=0.00001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

    cpoint = callbacks.ModelCheckpoint('.\\models\\checkpoints\\weights{epoch:02d}-{val_loss:.4f}.hdf5',
                                       monitor='val_loss', verbose=0,
                                       save_best_only=True, save_weights_only=False)

    print('training model...')
    model.fit_generator(generator=training_generator,
                        validation_data=testing_generator,
                        use_multiprocessing=True,
                        steps_per_epoch=len(partition['train']) // params['batch_size'],
                        validation_steps=len(partition['test']) // params['batch_size'],
                        workers=4,
                        epochs=50,
                        callbacks=[cpoint])

    print('saving model...')
    modelpath = os.path.realpath(MODEL_FORMAT.format(MODEL_NAME if MODEL_NAME != '' else datetime.now()))
    model_dir = os.path.dirname(modelpath)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save(modelpath)
    print('model saved to {}...'.format(modelpath))

    print('exiting...')


def generate_partition():
    training_files = np.load(os.path.join(DATA_PATH, 'train-set.npy'))[:-1]  # The last file may not be of adequate size
    testing_files = np.load(os.path.join(DATA_PATH, 'test-set.npy'))[:-1]

    training_set_size = len(training_files)
    testing_set_size = len(testing_files)

    training_set = create_set(training_files, training_set_size, 'train')
    testing_set = create_set(testing_files, testing_set_size, 'test')

    return {
        'train': training_set,
        'test': testing_set
    }


def create_set(files: list, set_size: int, prefix: str) -> list:
    id_format = '{prefix}/session-{file_id}|{sample_num}'
    sample_set = []

    for file_id in files[:set_size]:
        for sample_num in range(500):
            sample_set.append(id_format.format(file_id=file_id, sample_num=sample_num, prefix=prefix))

    return sample_set


if __name__ == "__main__":
    main()

