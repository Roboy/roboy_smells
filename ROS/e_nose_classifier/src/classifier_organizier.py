from e_nose_classification_publisher import eNoseClassificationPublisher
from e_nose_classification_test import eNoseClassificationTestPublisher
from e_nose_subscriber import eNoseSubscriber
from online_reader import OnlineReader
import keyboard  # using module keyboard


class ClassifierOrganizer:
    def __init__(self):
        self.pub = eNoseClassificationPublisher()
        self.pub_test = eNoseClassificationTestPublisher()
        self.sub = eNoseSubscriber()
        self.online = OnlineReader()
        keyboard.on_press_key("space", startMeas)


    def startMeas(self):
        print('test')
        self.online.set_Breakpoint()


if __name__ == '__main__':
    co = ClassifierOrganizer()
