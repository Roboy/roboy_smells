#!/usr/bin/env python
import rospy
from std_msgs.msg import String


class eNoseClassificationTestPublisher():
    """
        sends the classification done in the classifier_organizer to our demo view as a simple string
    """

    def __init__(self):
        self.pub_classifier = rospy.Publisher('e_nose_classification', String, queue_size=10)

    def send_classification(self, classified: String):
        """
        sends the classification as pure ROS string message
        :param classified: classification as string
        """
        if not rospy.is_shutdown():
            self.pub_classifier.publish(classified)
        else:
            print('no ROS connection')


if __name__ == '__main__':
    try:
        encp = eNoseClassificationTestPublisher()
    except rospy.ROSInterruptException:
        pass
