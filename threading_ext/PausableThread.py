import threading
import time


class PausableThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.killed = False
        self.paused = False

    def sleep_if_paused(self):
        while self.paused:
            time\
                .sleep(1)
            if not self.paused and not self.killed:
                for i in range(3, 1):
                    print('starting in{}'.format(i))

    def pause(self):
        self.paused = not self.paused

    def kill(self):
        self.killed = True
