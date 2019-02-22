"""Input recorder for XBOX360 controller.
(Modified from https://raw.githubusercontent.com/zeth/inputs/master/examples/jstest.py)"""
import inputs


TRIGGER_MAX = 255
JOY_MAX = 32767
JOY_MIN = 32768

JOYSTICKS = [
 'LSY',
 'LSX',
 'RSY',
 'RSX'
]

TRIGGERS = [
 'LT',
 'RT'
]

EVENT_ABB = (
    # Absolute Events ======================
    #JoySticks
    ('Absolute-ABS_Y', 'LSY'),
    ('Absolute-ABS_X', 'LSX'),
    ('Absolute-ABS_RY', 'RSY'),
    ('Absolute-ABS_RX', 'RSX'),

    #Triggers
    ('Absolute-ABS_Z', 'LT'),
    ('Absolute-ABS_RZ', 'RT'),

    # D-PAD, aka HAT
    ('Absolute-ABS_HAT0X', 'HX'),
    ('Absolute-ABS_HAT0Y', 'HY'),

    # Button Events ========================
    # Face Buttons
    ('Key-BTN_NORTH', 'N'),
    ('Key-BTN_EAST', 'E'),
    ('Key-BTN_SOUTH', 'S'),
    ('Key-BTN_WEST', 'W'),

    # Other buttons
    ('Key-BTN_THUMBL', 'THL'),
    ('Key-BTN_THUMBR', 'THR'),
    ('Key-BTN_TL', 'TL'),
    ('Key-BTN_TR', 'TR'),
    ('Key-BTN_TL2', 'TL2'),
    ('Key-BTN_TR2', 'TR3'),
    ('Key-BTN_MODE', 'M'),
    ('Key-BTN_START', 'ST'),
)

# This is to reduce noise from the PlayStation controllers
# For the Xbox controller, you can set this to 0
MIN_ABS_DIFFERENCE = 0


class GameRecorder(object):
    def __init__(self, gamepad=None, abbrevs=EVENT_ABB):
        self.btn_state = {}
        self.old_btn_state = {}
        self.abs_state = {}
        self.old_abs_state = {}
        self.abbrevs = dict(abbrevs)
        self._other = 0

        self.init_state()

        self.gamepad = gamepad
        if not self.gamepad:
            self._get_gamepad()

    def init_state(self):
        for key, value in self.abbrevs.items():
            if key.startswith('Absolute'):
                self.abs_state[value] = 0
                self.old_abs_state[value] = 0
            elif key.startswith('Key'):
                self.btn_state[value] = 0
                self.old_btn_state[value] = 0

    def _get_gamepad(self):
        """Get a gamepad object."""
        try:
            self.gamepad = inputs.devices.gamepads[0]
        except IndexError:
            raise inputs.UnpluggedError("No gamepad found.")

    def handle_unknown_event(self, event, key):
        """Deal with unknown events."""
        if event.ev_type == 'Key':
            new_abbv = 'B' + str(self._other)
            self.btn_state[new_abbv] = 0
            self.old_btn_state[new_abbv] = 0
        elif event.ev_type == 'Absolute':
            new_abbv = 'A' + str(self._other)
            self.abs_state[new_abbv] = 0
            self.old_abs_state[new_abbv] = 0
        else:
            return None

        self.abbrevs[key] = new_abbv
        self._other += 1

        return self.abbrevs[key]

    def process_event(self, event):
        """Process the event into a state."""
        if event.ev_type == 'Sync':
            return
        if event.ev_type == 'Misc':
            return

        key = event.ev_type + '-' + event.code
        try:
            abbv = self.abbrevs[key]
        except KeyError:
            abbv = self.handle_unknown_event(event, key)
            if not abbv:
                return
        if event.ev_type == 'Key':
            self.old_btn_state[abbv] = self.btn_state[abbv]
            self.btn_state[abbv] = event.state
        if event.ev_type == 'Absolute':
            self.old_abs_state[abbv] = self.abs_state[abbv]
            self.abs_state[abbv] = event.state

    def format_state(self):
        """Format the state."""
        output_string = ""
        for key, value in self.abs_state.items():
            if key in TRIGGERS:
                value /= TRIGGER_MAX
            elif key in JOYSTICKS:
                value /= JOY_MAX
            output_string += key + ':' + '{:>4}'.format(str(value) + ' ')

        for key, value in self.btn_state.items():
            output_string += key + ':' + str(value) + ' '

        return output_string

    def output_state(self, ev_type, abbv):
        """Print out the output state."""
        if ev_type == 'Key':
            if self.btn_state[abbv] != self.old_btn_state[abbv]:
                print(self.format_state())
                return

        if abbv[0] == 'H':
            print(self.format_state())
            return

        difference = self.abs_state[abbv] - self.old_abs_state[abbv]
        if (abs(difference)) > MIN_ABS_DIFFERENCE:
            print(self.format_state())

    def flattened_state(self):
        abs_input_list = list(self.abs_state.values())
        btn_input_list = list(self.btn_state.values())

        for i in range(4):
            if abs_input_list[i] > 0:
                abs_input_list[i] = abs_input_list[i] / JOY_MAX
            elif abs_input_list[i] < 0:
                abs_input_list[i] = abs_input_list[i] / JOY_MIN

        abs_input_list[4] = abs_input_list[4] / TRIGGER_MAX
        abs_input_list[5] = abs_input_list[5] / TRIGGER_MAX

        return abs_input_list + btn_input_list

    def process_events(self):
        try:
            events = self.gamepad.read()
        except:
            events = []
        for event in events:
            self.process_event(event)
