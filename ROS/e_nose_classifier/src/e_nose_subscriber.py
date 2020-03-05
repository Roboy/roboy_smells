#!/usr/bin/env python
# license removed for brevity
import rospy
from sklearn.externals import joblib
from e_nose_raw_publisher.msg import e_nose_raw
from ROS.e_nose_classifier.src.EventHook import EventHook


class eNoseSubscriber:
    def __init__(self):
        self.listener()
        self.onUpdate: EventHook = EventHook()
        self.sensorValues = [0.0] * 64

    def callback(self, data):
        print(rospy.get_caller_id() + "I heard %s", data.measurement_time)
        self.sensorValues = data.sensordata
        self.onUpdate()


    def listener(self):
        rospy.init_node('e_nose_sensor_raw_listener', anonymous=False)
        rospy.Subscriber("enose_sensordata", e_nose_raw, self.callback)
        print('started e_nose e_nose_sensor_raw_listener successfully')

        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()


if __name__ == '__main__':
    try:
        # init()
        ens = eNoseSubscriber()
    except rospy.ROSInterruptException:
        pass
