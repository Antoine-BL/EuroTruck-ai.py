from __future__ import print_function

import math
import os

import cv2
import time

from tensorflow.python.keras.models import load_model

import trainer
from trainer.model import model

import pyvjoy

import numpy as np
from mss import mss
from tensorflow.python.keras import Sequential

from pyvjoy import VJoyDevice
from threading_ext.KeyboardTracker import KeyboardTracker

UINT8_MAXVALUE = 32768
MODEL_NAME = 'models/last_try.h5'


def main():
    keyboard_tracker_thread = KeyboardTracker()
    keyboard_tracker_thread.pause()
    keyboard_tracker_thread.start()

    print('Loading model...')
    m = init_model()

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
                pos_x = 108
                pos_y = 128
                dim_y = 252
                dim_x = 126

                screen = np.asarray(sct.grab(monitor))
                screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2GRAY)
                screen = cv2.resize(screen, (480, 270))
                screen = screen[pos_y:pos_y + dim_y][pos_x:pos_x + dim_x]
                cv2.imshow('window', screen)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
                screen = screen / 255 - 0.5
                screen = np.array([np.resize(screen, (66, 200, 1))])

                prediction = m.predict(screen)
                print(prediction)
                act_on_prediction(prediction, j)
            else:
                j.data.wAxisX = 16383
                j.update()

            time.sleep(0.1)


def init_model() -> Sequential:
    model_path = os.path.realpath(MODEL_NAME)
    m = trainer.model.model()
    m.load_weights(model_path)

    return m


def act_on_prediction(prediction, vjd: VJoyDevice):
    adjusted_val = prediction[0][0] * UINT8_MAXVALUE
    adjusted_val = int(round(adjusted_val, 0))
    print(adjusted_val)

    vjd.data.wAxisX = UINT8_MAXVALUE // 2
    vjd.update()


def __std_image(img):
    mean = np.mean(img)
    std = np.std(img)
    return (img - mean)/std


if __name__ == "__main__":
    main()
