import argparse
from random import random

from io import BytesIO

import psutil
from tensorflow.python.keras import callbacks
from tensorflow.python.lib.io import file_io

from tensorflow.python.keras import backend as K

from trainer.model import model
from trainer.util import *


def main(job_dir, **args):
    # Setting up the path for saving logs
    logs_path = job_dir + '/logs/'

    with tf.Session().as_default() as sess:
        K.set_session(sess)

        f = BytesIO(file_io.FileIO(os.path.join(job_dir, LABELS_PATH), mode='rb').read())
        labels = np.load(f)

        labels_train = []
        labels_val = []

        for i, label in enumerate(labels, start=0):
            rdm = random()

            if rdm <= PCT_TRAIN:
                add_to = labels_train
            else:
                add_to = labels_val

            add_to.append((i, label))
        labels = None

        b_size = 50

        cpoint = callbacks.ModelCheckpoint(job_dir + '/' + CPOINT_PATH,
                                           monitor='val_loss',
                                           verbose=0,
                                           save_best_only=True,
                                           save_weights_only=False)

        tensorboard = callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)

        m = model()
        gen_train = generate_images(ds=labels_train,
                                    sess=sess,
                                    directory=job_dir,
                                    target_dimensions=(66, 200, 1),
                                    batch_size=b_size)

        # Take 20% of the data for validation
        gen_val = generate_images(ds=labels_val,
                                  sess=sess,
                                  directory=job_dir,
                                  target_dimensions=(66, 200, 1),
                                  batch_size=b_size,
                                  data_aug_pct=0.0)

        # Train the model
        m.fit_generator(generator=gen_train,
                        validation_data=gen_val,
                        steps_per_epoch=len(labels_train) // b_size,
                        validation_steps=len(labels_val) // b_size,
                        workers=1,
                        epochs=50,
                        callbacks=[cpoint,tensorboard],
                        verbose=1)

        # Save model.h5 on to google storage
        m.save(MODELS_PATH)
        with file_io.FileIO(MODELS_PATH, mode='r') as input_f:
            with file_io.FileIO(os.path.join(job_dir, MODELS_PATH), mode='w+') as output_f:
                output_f.write(input_f.read())
        print("Successfully saved model")


def generate_images(ds,
                    target_dimensions,
                    directory,
                    sess,
                    batch_size=100,
                    data_aug_pct=0.8,
                    aug_likelihood=0.5,
                    st_angle_threshold=0.05,
                    neutral_drop_pct=0.25):
    ds_len = len(ds)
    t_y, t_x, t_z = target_dimensions
    images = np.zeros((batch_size, t_y, t_x, t_z))
    steering_angles = np.zeros(batch_size)

    k = 0
    while True:
        while k < batch_size:
            ds_index = np.random.randint(0, ds_len)

            (im_index, st_angle) = ds[ds_index]

            if abs(st_angle) < st_angle_threshold and np.random.random_sample() <= neutral_drop_pct:
                continue

            st_angle += 1
            st_angle /= 2
            im_path = os.path.realpath(os.path.join(directory, IMGS_PATH.format(im_index)))
            im_data = tf_load_png_data(im_path, sess)

            im_data, st_angle = \
                augment_image(im_data, st_angle, p=aug_likelihood) \
                if np.random.random_sample() <= data_aug_pct \
                else (im_data, st_angle)

            pos_x = 108
            pos_y = 128
            dim_y = 252
            dim_x = 126

            im_data = im_data[pos_y:pos_y + dim_y][pos_x:pos_x + dim_x]
            im_data = im_data / 255 - 0.5
            im_data = np.resize(im_data, (t_y, t_x, t_z))

            images[k] = im_data
            steering_angles[k] = st_angle

            k += 1

        yield images, np.clip(steering_angles, -1, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)
