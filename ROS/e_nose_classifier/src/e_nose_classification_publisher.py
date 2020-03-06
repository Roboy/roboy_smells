#!/usr/bin/env python
# license removed for brevity
import json
import time
import rospy
from std_msgs.msg import String


class eNoseClassificationPublisher():
    def __init__(self):
        self.pub_classifier = rospy.Publisher('e_nose_classification', String, queue_size=10)

    def send_classification(self, classified: String):
        if not rospy.is_shutdown():
            text = str('Hey I smelled: ' + classified)
            self.pub_classifier.publish(self.get_json_text_message(31, text))
            self.pub_classifier.publish(self.get_json_icon_message(27, classified))
        else:
            print('no ROS connection')

    def get_json_text_message(id: int, text: String):
        sensordata_json = {
            "id": id,
            "textMessage": text,
        }
        return json.dumps(sensordata_json)

    def get_json_icon_message(id: int, text: String):
        icon = ""
        if text == 'wodka':
            icon = 'Wodka'
        elif text == 'red_wine':
            icon = 'RedWine'
        elif text == 'orange_juice':
            icon = 'OrangeJuice'
        else:
            icon = 'Empty'

        sensordata_json = {
            "id": id,
            "currentIcon": icon,
        }
        return json.dumps(sensordata_json)


if __name__ == '__main__':
    try:
        encp = eNoseClassificationPublisher()
    except rospy.ROSInterruptException:
        pass
