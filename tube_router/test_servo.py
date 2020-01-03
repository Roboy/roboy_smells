import RPi.GPIO as GPIO
import time
from servo_config import get_control_GPIO, get_channels

servoPIN = get_control_GPIO()
GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPIN, GPIO.OUT)

p = GPIO.PWM(servoPIN, 50)
C = get_channels()

def test_channels(channels, time_wait):
    for i in range(len(channels)):
        p.ChangeDutyCycle(channels[i])
        time.sleep(time_wait)

def test_channels_2(channels):
    num_channel_str = input("Choose channel: ")
    num_channel = int(num_channel_str)
    if num_channel < 1 or num_channel > 7:
        num_channel = 1
    p.ChangeDutyCycle(channels[int(num_channel)-1])

p.start(C[0]) # initialise
try:
    while True:
        #test_channels(C, 5)
        test_channels_2(C)
         
except KeyboardInterrupt:
    p.stop()
    GPIO.cleanup()
