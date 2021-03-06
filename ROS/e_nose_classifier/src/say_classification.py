import random
import time
from enum import Enum, auto

ROS = False
try:
    import rospy
    from std_msgs.msg import String
    ROS = True
except ImportError:
    print('Failed to import ROS running in test mode')
    pass

try:
    import pyroboy
except ImportError:
    print('Failed to import pyroboy; just printing to stdout')
    pass


class ClassificationVoicer:
    """
    Sends classification to Roboy to say it in voice
    """
    class State(Enum):
        NULL = auto()
        START_SMELL = auto()
        SURE = auto()
        CONFUSED = auto()

    NEW_SMELL_QUOTES = [
        "Oh, do I smell something?",
        "There's a smelly smell in here that smells... smelly.",
        "I am not entirely sure, but do I smell %CLASS%?",
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

    CLASS_NAMES = {
        "none": "No Data",
        "red_wine": "Red Wine",
        "pinot_noir": "Pinot Noir",
        "orange_juice": "Orange Juice",
        "isopropanol": "Isopropanol",
        "acetone": "Acetone",
        "wodka": "Wodka",
        "raisin": "Raisin",
        "coffee": "Coffee"
    }

    TIME_RESET = 30.0
    TIME_UNTIL_SURE = 12.0
    TIME_ENTERTAIN_MIN = 5.0

    CLASS_NULL = ['ref', 'null']

    def __init__(self):
        self.current_class = None
        self.last_class_change_time = time.time()
        self.last_say_time = time.time()
        self.last_state_change_time = time.time()
        self.state = ClassificationVoicer.State.NULL

        if ROS:
            print('Initialiting ROS node...')
            rospy.init_node('e_nose_classification_voicer', anonymous=False)
            print('Initialiting Subcomponents...')

            rospy.Subscriber("/e_nose_classification", String, self.callback)
            print('ros e_nose classification VOICER node started successfully')

    def run(self):
        if ROS:
            rospy.spin()
        else:
            class Data:
                def __init__(self, data):
                    self.data = data
            try:
                while True:
                    var = input("Please enter something: ")
                    if var == '':
                        var = self.current_class
                    self.callback(Data(var))
                    if var.lower() == 'q':
                        break
            except KeyboardInterrupt:
                print('Interrupted...')

    def say(self, text):
        if isinstance(text, list):
            text = random.choice(text)
        cname = 'something'
        if self.current_class in self.CLASS_NAMES:
            cname = self.CLASS_NAMES[self.current_class]
        text = text.replace('%CLASS%', cname)
        self.last_say_time = time.time()
        print('Saying:', text)
        try:
            pyroboy.say(text)
        except:
            pass

    def change_state(self, state):
        print('State changed to:', state)
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
        print('Received class:', smell_class, '(has changed)' if changed else '(no change)')

        if self.state == ClassificationVoicer.State.NULL:
            if changed:
                if smell_class not in self.CLASS_NULL:
                    # This is a new smell
                    self.change_state(ClassificationVoicer.State.START_SMELL)
                    self.say(self.NEW_SMELL_QUOTES)
        elif self.state == ClassificationVoicer.State.START_SMELL:
            if curr_time - self.last_class_change_time > self.TIME_UNTIL_SURE:
                # No class change for 10 seconds => transition to sure
                self.change_state(ClassificationVoicer.State.SURE)
                self.say(self.SURE_SMELL_QUOTES)
            elif changed:
                # Smell changed within 10s
                self.change_state(ClassificationVoicer.State.CONFUSED)
                self.say(self.MAYBE_SMELL_QUOTES)
            elif curr_time - self.last_say_time > self.TIME_ENTERTAIN_MIN:
                # entertain while in start state
                self.say(self.MAYBE_SMELL_QUOTES)
        elif self.state == ClassificationVoicer.State.CONFUSED:
            if curr_time - self.last_class_change_time > 10:
                if smell_class in self.CLASS_NULL:
                    # back to init
                    self.change_state(ClassificationVoicer.State.NULL)
                    self.say(self.CONFUSED_SMELL_QUOTES)
                else:
                    # No class change for 10 seconds => transition to sure
                    self.change_state(ClassificationVoicer.State.SURE)
                    self.say(self.SURE_SMELL_QUOTES)
            elif curr_time - self.last_state_change_time > self.TIME_RESET:
                # confusing stuff for too long => back to init
                self.change_state(ClassificationVoicer.State.NULL)
                self.say(self.CONFUSED_SMELL_QUOTES)
            elif curr_time - self.last_say_time > self.TIME_ENTERTAIN_MIN:
                    # entertain while in confused state
                    self.say(self.MAYBE_SMELL_QUOTES)
        elif self.state == ClassificationVoicer.State.SURE:
            if curr_time - self.last_state_change_time > self.TIME_RESET:
                # back to init
                self.change_state(ClassificationVoicer.State.NULL)
            elif changed:
                if smell_class in self.CLASS_NULL and curr_time - self.last_state_change_time > self.TIME_UNTIL_SURE:
                    # back to init
                    self.change_state(ClassificationVoicer.State.NULL)
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
