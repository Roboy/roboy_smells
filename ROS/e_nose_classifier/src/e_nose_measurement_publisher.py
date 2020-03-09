#!/usr/bin/env python
import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

import matplotlib.pyplot as plt

class eNoseMeasurementPublisher():
    def __init__(self):
        self.pub_meas = rospy.Publisher('e_nose_measurements', numpy_msg(Floats), queue_size=10)

    def send_classification(self, measurement):
        print('send current measurement to e_nose_measurements')
        if not rospy.is_shutdown():
            print(measurement.shape)
            if measurement.shape[0] > 30:
                print(measurement)
                #plt.plot(measurement)
                #plt.show()
            self.pub_meas.publish(measurement.reshape(-1))
            print(measurement.reshape(-1).shape)
            print('published measurement')
        else:
            print('no ROS connection')


if __name__ == '__main__':
    try:
        encp = eNoseMeasurementPublisher()
    except rospy.ROSInterruptException:
        pass
