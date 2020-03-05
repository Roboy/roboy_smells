#!/usr/bin/env python
# license removed for brevity
import datetime
import rospy
from e_nose.msg import e_nose_raw
from online_reader import OnlineReader


class eNoseSubscriber:
    def __init__(self):
        self.reader = OnlineReader(5)
        self.listener()

    def callback(self, data):
        print(rospy.get_caller_id() + "I heard %s", data.measurement_time)
        self.reader.add_sample(data.sensordata)

    def listener(self):
        rospy.init_node('e_nose_sensor_raw_listener', anonymous=False)
        rospy.Subscriber("enose_sensordata", e_nose_raw, self.callback)
        print('started e_nose subscriber successfully')

        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()


if __name__ == '__main__':
    try:
        # init()
        ens = eNoseSubscriber()
    except rospy.ROSInterruptException:
        pass
