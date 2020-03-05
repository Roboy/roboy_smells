#!/usr/bin/env python
# license removed for brevity

import time
import rospy
from e_nose_raw_publisher.msg import e_nose_raw
from e_nose_connector import eNoseConnector


def enose_node():
    eNose = eNoseConnector()
    pub_e_nose = rospy.Publisher('enose_sensordata', e_nose_raw, queue_size=10)
    rospy.init_node('e_nose_sensor_publish', anonymous=False)
    rate = rospy.Rate(2)  # 10hz
    print('e_nose_sensor publisher node started succesfully')
    while not rospy.is_shutdown():
        msg = e_nose_raw()
        msg.sensordata = eNose.detect()
        msg.measurement_time = str(time.time())
        pub_e_nose.publish(msg)
        rate.sleep()


if __name__ == '__main__':
    try:
        enose_node()
    except rospy.ROSInterruptException:
        pass
