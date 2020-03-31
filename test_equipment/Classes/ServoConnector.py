import RPi.GPIO as GPIO
import time
from tube_router.servo_config import get_control_GPIO, get_channels


class ServoConnecor:
    """
    Connects to the servo, which moves the router to the specific position of an entrance.
    It has to be connected as shown in the wiring documentation
    """

    # load servo settings
    def __init__(self):
        """
        When connected as specified, the servo is controlled by a stetic pwm signal, which have to be send as long as
        the router is used.
        """

        servoPIN = get_control_GPIO()
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(servoPIN, GPIO.OUT)

        self.p = GPIO.PWM(servoPIN, 50)
        self.Channel = get_channels()
        self.p.start(self.Channel[0])

    def setSample(self, channel):
        """
        move router to new entrance, exact position is specified in the servo_config file
        :param channel: channel
        :return:
        """
        self.p.ChangeDutyCycle(self.Channel[channel])

    def __del__(self):
        self.p.stop()
        GPIO.cleanup()
