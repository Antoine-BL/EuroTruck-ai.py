from __future__ import print_function

import math
import os

import cv2
import time

from tensorflow.python.keras.models import load_model

import pyvjoy

import numpy as np
from mss import mss
from tensorflow.python.keras import Sequential

from keras_tools import Model
from keras_tools.Model import create_model
from pyvjoy import VJoyDevice
from threading_ext.KeyboardTracker import KeyboardTracker

import tensorflow as tf

UINT8_MAXVALUE = 32768
MODEL_NAME = 'models/checkpoints/weights48-190.8600.hdf5'


def main():
    keyboard_tracker_thread = KeyboardTracker()
    keyboard_tracker_thread.pause()
    keyboard_tracker_thread.start()

    print('Loading model...')
    model = init_model()

    holding_down = False
    paused = True
    j = pyvjoy.VJoyDevice(1)
    with mss() as sct:
        while True:
            if keyboard_tracker_thread.check_for_kill():
                print('killing all threads')
                keyboard_tracker_thread.kill()
                break

            if not holding_down and keyboard_tracker_thread.check_for_pause():
                if paused:
                    msg = 'unpausing'
                else:
                    msg = 'pausing'

                paused = not paused
                print(msg + ' threads')
                holding_down = True
            elif holding_down \
                    and not keyboard_tracker_thread.check_for_rewind() \
                    and not keyboard_tracker_thread.check_for_pause():
                holding_down = False

            if not paused:
                monitor = {"top": 40, "left": 0, "width": 1024, "height": 728}
                dim_x = 280
                dim_y = 100
                pos_x = 100
                pos_y = 130

                screen = np.asarray(sct.grab(monitor))
                screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
                screen = cv2.resize(screen, (480, 270))
                screen = screen[pos_y:pos_y + dim_y, pos_x:pos_x + dim_x]
                cv2.imshow('window', screen)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

                screen = screen.reshape(1, dim_y, dim_x, 1)

                prediction = model.predict(screen)
                print(prediction)
                act_on_prediction(prediction, j)
            else:
                j.data.wAxisX = 16383
                j.update()

            time.sleep(0.1)


def init_model() -> Sequential:
    model_path = os.path.realpath(MODEL_NAME)
    dim_x = 280
    dim_y = 100
    model = create_model(dim_x, dim_y)
    model.load_weights(model_path)

    return model


def act_on_prediction(prediction, vjd: VJoyDevice):
    adjusted_val = 1 + ((prediction[0][0] + math.pi / 2) / math.pi *10) * (UINT8_MAXVALUE - 1)
    adjusted_val = int(round(adjusted_val, 0))
    print(adjusted_val)
    vjd.data.wAxisX = adjusted_val
    vjd.update()


if __name__ == "__main__":
    main()
