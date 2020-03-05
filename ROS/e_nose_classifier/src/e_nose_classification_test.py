#!/usr/bin/env python
import rospy
from std_msgs.msg import String


class e_nose_classification_test_publisher():
    def __init__(self):
        self.pub_classifier = rospy.Publisher('e_nose_classification', String, queue_size=10)
        rospy.init_node('e_nose_classifier_test', anonymous=False)
        print('ros e_nose classification test node started successfully')
        rospy.spin()

    def send_classification(self, classified: str):
        if not rospy.is_shutdown():
            self.pub_classifier.publish(str)
        else:
            print('no ROS connection')




if __name__ == '__main__':
    try:
        encp = e_nose_classification_test_publisher()
    except rospy.ROSInterruptException:
        pass
