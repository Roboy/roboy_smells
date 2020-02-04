import RPi.GPIO as GPIO
import time
from servo_config import get_control_GPIO, get_channels
import rospy
from std_msgs.msg import Int8

servoPIN = get_control_GPIO()
GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPIN, GPIO.OUT)

p = GPIO.PWM(servoPIN, 50)
C = get_channels()

current_channel = 1
p.start(C[current_channel - 1])  # initialise


def callback(data):
    current_channel = data.data
    print(current_channel)
    p.ChangeDutyCycle(C[current_channel - 1])


try:
    rospy.init_node('servo', anonymous=True)
    rospy.Subscriber('tube_channel', Int8, callback)
    rospy.spin()

except KeyboardInterrupt:
    p.stop()
    GPIO.cleanup()
