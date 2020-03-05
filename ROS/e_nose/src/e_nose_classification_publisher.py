#!/usr/bin/env python
# license removed for brevity
import json
import time
import rospy
from std_msgs.msg import String


class e_nose_classification_publisher():
    def __init__(self):
        self.pub_classifier = rospy.Publisher('env_sensor', String)
        rospy.init_node('e_nose_classifier', anonymous=False)

    def send_classification(self, classified: str):
        if not rospy.is_shutdown():
            text = str('Hey I smelled: ' + classified)
            self.pub_classifier.publish(self.get_json_text_message(31, text))
        else:
            print('no ROS connection')

    def get_json_text_message(id, text):
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
