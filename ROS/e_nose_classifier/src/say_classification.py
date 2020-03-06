import random
import time
from enum import Enum, auto

import pyroboy
import rospy
from std_msgs.msg import String


class ClassificationVoicer:
    class State(Enum):
        NULL = auto()
        START_SMELL = auto()
        SURE = auto()
        CONFUSED = auto()

    NEW_SMELL_QUOTES = [
        "Oh, do I smell something?",
        "There's a smelly smell in here that smells... smelly.",
        "Do I smell %CLASS%?",
        "Do you smell %CLASS% as well?",
        "Hmm... Does it smell like %CLASS%?"
    ]
    MAYBE_SMELL_QUOTES = [
        "Oh wait, let me smell a bit more",
        "Could it be %CLASS%?",
        "I'm not sure what it smells like",
        "This smell... I think I know it",
        "I'm sure I've smelled this before",
        "Do you think it smells like %CLASS%?"
    ]
    CONFUSED_SMELL_QUOTES = [
        "I'm too confused",
        "ERROR. Smell does not compute",
        "My smell sensors tell me it smells like whiskey tango foxtrot... I've given up on them",
        "I'm sorry, my computer brain cannot handle this weird smell"
    ]
    SURE_SMELL_QUOTES = [
        "Oh I'm sure this is %CLASS%!",
        "I'm sure I smell %CLASS%!",
        "My smell detector tells me to be very confident that it smells like %CLASS%. I trust it.",
        "It definitely smells like %CLASS%!",
        "I definitely smell %CLASS%!"
    ]

    TIME_RESET = 30.0

    CLASS_NULL = ['ref', 'null']

    def __init__(self):
        print('Initialiting ROS node...')
        rospy.init_node('roboy/e_nose/classification_voicer', anonymous=False)
        print('Initialiting Subcomponents...')

        self.current_class = None
        self.last_class_change_time = time.time()
        self.last_say_time = time.time()
        self.last_state_change_time = time.time()
        self.state = ClassificationVoicer.State.NULL

        rospy.Subscriber("roboy/e_nose/classification", String, self.callback)
        print('ros e_nose classification VOICER node started successfully')

    def run(self):
        rospy.spin()

    def say(self, text):
        if isinstance(text, list):
            text = random.choice(text)
        text = text.replace('%CLASS%', self.current_class)
        pyroboy.say(text)
        self.last_say_time = time.time()

    def change_state(self, state):
        print('State changed to ', state)
        self.state = state
        self.last_state_change_time = time.time()

    def callback(self, data):
        smell_class = data.data
        changed = False
        curr_time = time.time()

        if self.current_class != smell_class:
            self.last_class_change_time = curr_time
            self.current_class = smell_class
            changed = True

        if self.state == ClassificationVoicer.State.NULL:
            if changed:
                if smell_class not in self.CLASS_NULL:
                    # This is a new smell
                    self.change_state(ClassificationVoicer.State.START_SMELL)
                    self.say(self.NEW_SMELL_QUOTES)
        elif self.state == ClassificationVoicer.State.START_SMELL:
            if curr_time - self.last_class_change_time > 10:
                # No class change for 10 seconds => transition to sure
                self.change_state(ClassificationVoicer.State.SURE)
                self.say(self.SURE_SMELL_QUOTES)
            elif changed:
                # Smell changed within 10s
                self.change_state(ClassificationVoicer.State.CONFUSED)
                self.say(self.MAYBE_SMELL_QUOTES)
        elif self.state == ClassificationVoicer.State.CONFUSED:
            if curr_time - self.last_class_change_time > 10:
                if smell_class in self.CLASS_NULL:
                    # back to init
                    self.change_state(ClassificationVoicer.State.NONE)
                    self.say(self.CONFUSED_SMELL_QUOTES)
                else:
                    # No class change for 10 seconds => transition to sure
                    self.change_state(ClassificationVoicer.State.SURE)
                    self.say(self.SURE_SMELL_QUOTES)
            elif curr_time - self.last_state_change_time > self.TIME_RESET:
                # confusing stuff for too long => back to init
                self.change_state(ClassificationVoicer.State.NONE)
                self.say(self.CONFUSED_SMELL_QUOTES)
            elif changed:
                if smell_class not in self.CLASS_NULL and curr_time - self.last_say_time > 5:
                    # entertain while in confused state
                    self.say(self.MAYBE_SMELL_QUOTES)
        elif self.state == ClassificationVoicer.State.SURE:
            if curr_time - self.last_state_change_time > self.TIME_RESET:
                # back to init
                self.change_state(ClassificationVoicer.State.NONE)
            elif changed:
                if smell_class in self.CLASS_NULL and curr_time - self.last_state_change_time > 10:
                    # back to init
                    self.change_state(ClassificationVoicer.State.NONE)
                else:
                    # go to confused
                    self.change_state(ClassificationVoicer.State.CONFUSED)
                    self.say(self.MAYBE_SMELL_QUOTES)
        else:
            print('WTF, state is ', self.state)
            self.change_state(ClassificationVoicer.State.NULL)


if __name__ == '__main__':
    co = ClassificationVoicer()
    co.run()
