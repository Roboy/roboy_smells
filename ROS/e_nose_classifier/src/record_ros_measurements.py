from ROS.e_nose_classifier.src.e_nose_subscriber import eNoseSubscriber
from test_equipment.Classes.CSVWriter import CSVWriter
import time
import numpy as np
import rospy


class ROSMessageRecorder:
    """
    Records the data received from the data gathering machine via ROS, uses the same script as the dgm
    and therefore works the similiar except it
    """

    def __init__(self, record_data=False):
        print('Initialiting ROS node...')
        rospy.init_node('e_nose_classifier_publish', anonymous=True)
        print('Initialiting Subc+omponents...')
        self.sub = eNoseSubscriber()
        self.sub.onUpdate += self.gotNewSample
        self.writer = CSVWriter('data_recording_' + str(time.time()))
        rospy.spin()

    def gotNewSample(self):
        """
        write sample to csv
        """
        self.writer.writeSample(self.sub.time, self.sub.sensor_values, self.sub.bme_data, self.sub.label)


if __name__ == '__main__':
    co = ROSMessageRecorder()
