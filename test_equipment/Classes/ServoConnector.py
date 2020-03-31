import RPi.GPIO as GPIO
import time
from tube_router.servo_config import get_control_GPIO, get_channels


class ServoConnecor:
    # load servo settings
    def __init__(self):
        servoPIN = get_control_GPIO()

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(servoPIN, GPIO.OUT)

        self.p = GPIO.PWM(servoPIN, 50)
        self.Channel = get_channels()
        self.p.start(self.Channel[0])

    def setSample(self, channel):
        self.p.ChangeDutyCycle(self.Channel[channel])

    def __del__(self):
        self.p.stop()
        GPIO.cleanup()
