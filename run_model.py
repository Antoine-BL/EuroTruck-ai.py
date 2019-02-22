from __future__ import print_function

import os

import cv2
import time

import pyvjoy

import numpy as np
from mss import mss
from tensorflow.python.keras import Sequential

from keras_tools.Model import create_model
from pyvjoy import VJoyDevice
from threading_ext.KeyboardTracker import KeyboardTracker

UINT8_MAXVALUE = 32768


def main():
    keyboard_tracker_thread = KeyboardTracker()
    keyboard_tracker_thread.pause()
    keyboard_tracker_thread.start()

    print('Loading model...')
    model = load_model()

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
                dim_x = 200
                dim_y = 66
                pos_x = 147
                pos_y = 128

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

            time.sleep(0.1)


def load_model() -> Sequential:
    model_path = os.path.realpath('models/ROI-Full-NN-balanced-50e.h5')

    model = create_model()
    model.load_weights(model_path)

    return model


def act_on_prediction(prediction, vjd: VJoyDevice):
    adjusted_val = 1 + prediction[0][0] * (UINT8_MAXVALUE - 1)
    adjusted_val = int(round(adjusted_val, 0))
    vjd.set_axis(pyvjoy.HID_USAGE_X, 0x1)
    vjd.update()
    print(prediction[0][0])


if __name__ == "__main__":
    main()
