import inputs

from threading_ext.PausableThread import PausableThread

KILL_KEY_COMBINATION = {'KEY_LEFTCTRL', 'KEY_LEFTSHIFT', 'KEY_Q'}
PAUSE_KEY_COMBINATION = {'KEY_LEFTCTRL', 'KEY_LEFTSHIFT', 'KEY_P'}
REWIND_KEY_COMBINATION = {'KEY_LEFTCTRL', 'KEY_LEFTSHIFT', 'KEY_D'}


class KeyboardTracker(PausableThread):
    def __init__(self):
        super(KeyboardTracker, self).__init__()
        self.pressed_keys = set()

    def run(self):
        while not self.killed:
            self.handle_events()

    def handle_events(self):
        events = inputs.get_key()
        if events:
            for event in events:
                self.handle_event(event)

    def handle_event(self, event):
        if event.ev_type != 'Key':
            return

        if event.code not in self.pressed_keys and event.state == 1:
            self.pressed_keys.add(event.code)
        elif event.code in self.pressed_keys and event.state == 0:
            self.pressed_keys.remove(event.code)

    def check_for_kill(self):
        return KILL_KEY_COMBINATION <= self.pressed_keys

    def check_for_pause(self):
        return PAUSE_KEY_COMBINATION <= self.pressed_keys

    def check_for_rewind(self):
        return REWIND_KEY_COMBINATION <= self.pressed_keys
