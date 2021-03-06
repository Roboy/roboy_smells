#!/usr/bin/env python
# license removed for brevity
import rospy
from ROS.e_nose_classifier.msg import e_nose_raw
from ROS.e_nose_classifier.src.EventHook import EventHook


class eNoseSubscriber:
    """
    ROS node for receiving the raw sensor data from the data gathering maching
    Via the EventHook class its gets asynchronous to the classifier_organizer
    """

    def __init__(self):
        self.onUpdate: EventHook = EventHook()
        self.listener()
        self.sensor_values = [0.0] * 64
        self.bme_data = [0.0] * 5
        self.time = ''
        self.label = ''

    def callback(self, data):
        """
        callback of ROS listener to get data from message
        :param data: ros message data
        :return: values as global variables
        """
        print(rospy.get_caller_id() + "I heard %s", data.measurement_time)
        self.sensor_values = data.sensordata
        self.time = data.measurement_time
        self.label = data.label
        self.bme_data = data.environmentdata
        self.onUpdate()

    def listener(self):
        rospy.Subscriber("enose_sensordata", e_nose_raw, self.callback)
        print('started e_nose e_nose_sensor_raw_listener successfully')


if __name__ == '__main__':
    try:
        # init()
        ens = eNoseSubscriber()
    except rospy.ROSInterruptException:
        pass
