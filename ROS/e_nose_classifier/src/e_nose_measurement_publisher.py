#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32MultiArray


class eNoseMeasurementPublisher():
    def __init__(self):
        self.pub_meas = rospy.Publisher('e_nose_measurements', Float32MultiArray, queue_size=10)

    def send_classification(self, measurement):
        if not rospy.is_shutdown():
            msg = Float32MultiArray()
            msg.data = measurement
            self.pub_meas.publish(msg)
        else:
            print('no ROS connection')


if __name__ == '__main__':
    try:
        encp = eNoseMeasurementPublisher()
    except rospy.ROSInterruptException:
        pass
