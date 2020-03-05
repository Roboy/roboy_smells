#!/usr/bin/env python
# license removed for brevity
import json
import time
import rospy
from std_msgs.msg import String


class e_nose_classification_publisher():
    def __init__(self):
        self.pub_classifier = rospy.Publisher('e_nose_classification', String, queue_size=10)
        rospy.init_node('e_nose_classifier_publish', anonymous=False)
        print('ros e_nose classification node started successfully')
        rospy.spin()

    def send_classification(self, classified: String):
        if not rospy.is_shutdown():
            text = str('Hey I smelled: ' + classified)
            self.pub_classifier.publish(self.get_json_text_message(31, text))
        else:
            print('no ROS connection')

    def get_json_text_message(id: int, text: String):
        sensordata_json = {
            "id": id,
            "textMessage": text,
        }
        return json.dumps(sensordata_json)


if __name__ == '__main__':
    try:
        encp = e_nose_classification_publisher()
    except rospy.ROSInterruptException:
        pass
