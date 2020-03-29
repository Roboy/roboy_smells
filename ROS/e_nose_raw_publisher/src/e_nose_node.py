#!/usr/bin/env python

import time
import rospy
from ROS.e_nose_raw_publisher.msg import e_nose_raw
from test_equiment.Classes.eNoseConnector import eNoseConnector


class eNoseRawNode:

    """
    This module gets the sensordata from the connected eNose and publishes it raw data via ROS
    There are two possibilites:
    1)It Measures by it itself, so it connects to the eNose and publishes
    2) it receives the eNose data and the bme data from a different module and publishes it
    """

    def __init__(self):
        self.eNose = eNoseConnector()
        self.pub_e_nose = rospy.Publisher('enose_sensordata', e_nose_raw, queue_size=10)
        if rospy.is_shutdown:
            rospy.init_node('e_nose_sensor', anonymous=False)
        self.rate = rospy.Rate(1)  # 1hz
        print('e_nose_raw node started succesfully')

    def run_standalone(self):
        while not rospy.is_shutdown():
            msg = e_nose_raw()
            msg.sensordata = self.eNose.detect()
            msg.measurement_time = str(time.time())
            self.pub_e_nose.publish(msg)
            self.rate.sleep()

    def run_nonstandalone(self, e_nose_raw_data, bme_data, label=''):
        msg = e_nose_raw()
        msg.sensordata = e_nose_raw_data
        msg.label = label
        msg.environmentdata = bme_data
        msg.measurement_time = str(time.time())
        self.pub_e_nose.publish(msg)


if __name__ == '__main__':
    try:
        enose = eNoseRawNode()
        enose.run_standalone()
    except rospy.ROSInterruptException:
        pass
