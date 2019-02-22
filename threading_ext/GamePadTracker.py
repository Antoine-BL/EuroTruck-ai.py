from threading_ext import GameRecorder
from threading_ext.PausableThread import PausableThread


class GamePadTracker(PausableThread):
    def __init__(self, recorder: GameRecorder):
        super(GamePadTracker, self).__init__()
        self.recorder = recorder

    def run(self):
        while not self.killed:
            self.recorder.process_events()
            self.sleep_if_paused()
