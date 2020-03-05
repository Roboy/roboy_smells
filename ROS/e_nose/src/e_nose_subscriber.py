#!/usr/bin/env python
# license removed for brevity
import rospy
from sklearn.externals import joblib
from e_nose.msg import e_nose_raw
from online_reader import OnlineReader
from measurements import DataType
from sklearn.neighbors import KNeighborsClassifier


class eNoseSubscriber:
    def __init__(self):
        self.reader = OnlineReader(5)
        self.listener()
        self.model: KNeighborsClassifier = joblib.load('presets/knn_classifier.joblib.pkl')

    def callback(self, data):
        print(rospy.get_caller_id() + "I heard %s", data.measurement_time)
        self.reader.add_sample(data.sensordata)
        meas = self.reader.get_last_n_as_measurement(1)
        pred = self.model.predict(meas.get_data_as(DataType.HIGH_PASS))
        print(pred)

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
