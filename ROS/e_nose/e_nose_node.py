#!/usr/bin/env python
# license removed for brevity
import json
import time
import rospy
from e_Nose.msg import e_nose
from e_nose_connector import eNoseConnector

e_nose_msg = e_nose()  # ros message


def enose_node():
    pub_enose = rospy.Publisher('env_sensor/enose_sensordata', e_nose, queue_size=10)
    rospy.init_node('e_nose_sensor', anonymous=True)
    rate = rospy.Rate(2)  # 10hz
    while not rospy.is_shutdown():
        e_nose.sensordata = eNose.detect()
        e_nose.timestamp = rospy.get_rostime()
        pub_enose.publish(e_nose)
        rate.sleep()


def init():
    global eNose
    eNose = eNoseConnector()


if __name__ == '__main__':
    try:
        # init()
        enose_node()
    except rospy.ROSInterruptException:
        pass
