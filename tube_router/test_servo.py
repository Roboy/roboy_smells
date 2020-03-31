import RPi.GPIO as GPIO
import time
from servo_config import get_control_GPIO, get_channels

"""
This is a basic test file to see how the servo works
Connects to the servo, which moves the router to the specific position of an entrance.
It has to be connected as shown in the wiring documentation
When connected as specified, the servo is controlled by a stetic pwm signal, which have to be send as long as
the router is used.
"""

servoPIN = get_control_GPIO()
GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPIN, GPIO.OUT)

p = GPIO.PWM(servoPIN, 50)
C = get_channels()


def test_channels(channels, time_wait):
    """
    move to each channel and wait their for a specific time
    :param channels: list of the 7 entrances
    :param time_wait: how long to wait at each cannel
    :return:
    """
    for i in range(len(channels)):
        p.ChangeDutyCycle(channels[i])
        time.sleep(time_wait)


def test_channels_2(channels):
    """
    Same as above but with user interaction via keyboard
    :param channels: list of the 7 entrances
    :return:
    """
    num_channel_str = input("Choose channel: ")
    num_channel = int(num_channel_str)
    if num_channel < 1 or num_channel > 7:
        num_channel = 1
    p.ChangeDutyCycle(channels[int(num_channel) - 1])


p.start(C[0])  # initialise
try:
    while True:
        # test_channels(C, 5)
        test_channels_2(C)

except KeyboardInterrupt:
    p.stop()
    GPIO.cleanup()
