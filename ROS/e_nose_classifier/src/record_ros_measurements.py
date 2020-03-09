
from ROS.e_nose_classifier.src.e_nose_subscriber import eNoseSubscriber
from test_equiment.Classes.CSVWriter import CSVWriter
import time
import numpy as np
import rospy


class ROSMessageRecorder:
    def __init__(self, record_data=False):
        print('Initialiting ROS node...')
        rospy.init_node('e_nose_classifier_publish', anonymous=True)
        print('Initialiting Subcomponents...')
        self.sub = eNoseSubscriber()
        self.sub.onUpdate += self.gotNewSample
        self.writer = CSVWriter('data_recording_'+str(time.time()))


    def gotNewSample(self):
        if self.record:
            self.writer.writeSample(self.sub.time, self.sub.sensor_values, self.sub.bme_data, self.sub.label)


if __name__ == '__main__':
    co = ROSMessageRecorder()
