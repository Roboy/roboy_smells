#!/usr/bin/env python
import rospy
from std_msgs.msg import String


class eNoseClassificationTestPublisher():
    def __init__(self):
        self.pub_classifier = rospy.Publisher('e_nose_classification', String, queue_size=10)

    def send_classification(self, classified: String):
        if not rospy.is_shutdown():
            self.pub_classifier.publish(classified)
        else:
            print('no ROS connection')




if __name__ == '__main__':
    try:
        encp = eNoseClassificationTestPublisher()
    except rospy.ROSInterruptException:
        pass
