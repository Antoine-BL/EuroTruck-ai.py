from __future__ import print_function

import os
import time

import inputs

from threading_ext.GameRecorder import GameRecorder
from threading_ext.GamePadTracker import GamePadTracker
from threading_ext.KeyboardTracker import KeyboardTracker
from threading_ext.RecordingThread import RecordingThread

DATA_FILENAME = 'session-{}.npy'
DATA_BASEDIR = 'data/unsorted'


def main():
    xboxctrl = inputs.devices.gamepads[0]
    jstest = GameRecorder(xboxctrl)

    data_path = create_training_data_directory()
    session_number = find_session_num(data_path)

    threads = []

    gamepad_tracker_thread = GamePadTracker(jstest)
    screenshot_thread = RecordingThread(data_path, session_number, jstest)
    keyboard_tracker_thread = KeyboardTracker()

    threads.append(gamepad_tracker_thread)
    threads.append(screenshot_thread)
    threads.append(keyboard_tracker_thread)

    for thread in threads:
        thread.pause()

    for thread in threads:
        thread.start()

    holding_down = False
    paused = True
    while True:
        if keyboard_tracker_thread.check_for_kill():
            print('killing all threads')
            for thread in threads:
                thread.kill()
            break

        if not holding_down and keyboard_tracker_thread.check_for_pause():
            if paused:
                msg = 'unpausing'
            else:
                msg = 'pausing'
                
            paused = not paused
            print(msg + ' threads')            
            gamepad_tracker_thread.pause()
            screenshot_thread.pause()
            holding_down = True
        elif not holding_down and keyboard_tracker_thread.check_for_rewind():
            print('Oops, let\'s forget that last part')
            screenshot_thread.rewind()
            holding_down = True
        elif holding_down \
                and not keyboard_tracker_thread.check_for_rewind()\
                and not keyboard_tracker_thread.check_for_pause():
            holding_down = False

        time.sleep(0.1)


def find_session_num(data_path):
    session_number = 1
    while True:
        file_name = data_path.format(session_number)

        if os.path.isfile(file_name):
            session_number += 1
        else:
            print('Saving data in file: {}'.format(file_name))
            break
    return session_number


def create_training_data_directory():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_path, DATA_BASEDIR)

    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    return os.path.join(data_path, DATA_FILENAME)


if __name__ == "__main__":
    main()
