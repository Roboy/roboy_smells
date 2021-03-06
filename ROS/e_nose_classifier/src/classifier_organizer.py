from ROS.e_nose_classifier.src.e_nose_classification_publisher import eNoseClassificationPublisher
from ROS.e_nose_classifier.src.e_nose_classification_test import eNoseClassificationTestPublisher
from ROS.e_nose_classifier.src.e_nose_subscriber import eNoseSubscriber
from classification.lstm_model import SmelLSTM
from classification.knn import KNN
from classification.naive_bayes import GNB
from e_nose.measurements import DataType, StandardizationType
from e_nose.online_reader import OnlineReader
import time
import numpy as np
import rospy


class ClassifierOrganizer:
    """
    This class is responsible for the organization of receiving the pure data from the data gathering machine (dgm)
     via ROS. Then it is modeled to Measurment objects, similiar to the computation of the sampling files generated by
     the dgm.
     It computes the classification either by knn or by a lstm.
     It shares the predicted classification via ROS to OUI or to our demo view
     Has to be executed with on a computer connected to a keyboard as it is required to set the start and endpoints
     of a measurement manualy via a shortcut
    """

    def __init__(self):
        print('Initialiting ROS node...')
        rospy.init_node('classifier_organizer', anonymous=True)
        print('Initialiting Subcomponents...')
        self.pub = eNoseClassificationPublisher()
        self.pub_test = eNoseClassificationTestPublisher()
        self.sub = eNoseSubscriber()
        failing_channels = [22, 31]
        working_channels = np.ones(64, bool)
        working_channels[failing_channels] = False
        num_working_channels = np.count_nonzero(working_channels)
        print(num_working_channels)
        self.online = OnlineReader(5, standardization=StandardizationType.LAST_REFERENCE,
                                   override_working_channels=working_channels)
        self.from_sample = 0
        self.sub.onUpdate += self.got_new_sample
        self.online.invoke_callback += self.gathered_data
        self.use_neural_network = True
        self.datatype = DataType.HIGH_PASS
        self.recording = False

        if self.use_neural_network:
            sequence_length = 45
            self.classifier_lstm = SmelLSTM(input_shape=(1, sequence_length, num_working_channels), num_classes=6, dim_hidden=12,
                                  stateful=True, data_type=self.datatype)
            model_name = 'RecurrentModelTrainable_4c22e426_6_batch_size=64,data_preprocessing=high_pass,dim_hidden=12,lr=0.031239,return_sequences=True,use__2020-03-30_23-30-574gm3qftn'
            self.classifier_lstm.summary()
            self.classifier_lstm.load_weights(model_name, checkpoint=240, path='classification/models/')

        else:
            self.classifier_knn = KNN(num_working_channels, data_dir='data', data_type=self.datatype)
            self.classifier_gnb = GNB(data_dir='data', data_type=self.datatype)
        print('ros e_nose classification node started successfully')

    def startMeas(self):
        """
        triggers start and endpoint of an endpoint via keyboard interaction

        """
        try:
            while True:
                self.pub_test.send_classification('ref')
                self.pub.send_classification('ref')
                var = input("Please enter something: ")
                if var.lower() == 'q':
                    break
                if not self.recording:
                    print('restarting classification', var)
                    self.from_sample = self.online.current_length
                    self.online.set_trigger_in(9)
                    self.recording = True
                else:
                    self.recording = False
                    self.online.invoke_at = 99999999999
                    self.pub_test.send_classification('ref')
                    self.pub.send_classification('ref')

        except KeyboardInterrupt:
            print('Interrupted...')

    def gathered_data(self):
        """
        if enough datapoints are received to do a preditction
        send data to the specified models and send the result to the ROS nodes
        :return: nothing
        """
        print('gathered data')
        data = self.online.get_since_n_as_measurement(self.from_sample)
        # print(data.correct_channels)
        # print(data.get_data().shape)
        data_for_classifier = data.get_data_as(self.datatype)
        self.pub_meas.send_classification(data_for_classifier)

        if self.use_neural_network:
            prediction = self.classifier_lstm.predict_live(data)
            self.pub_test.send_classification(prediction)
            self.pub.send_classification(prediction)
        else:
            if (data_for_classifier.shape[0] > 40):
                prediction_knn = self.classifier_knn.predict_from_batch(data_for_classifier)
                prediction_gnb = self.classifier_gnb.predict_from_batch(data_for_classifier)
                print('prediction_knn: ', prediction_knn)
                print('prediction_gnb: ', prediction_gnb)
                self.pub_test.send_classification(prediction_gnb)
                self.pub.send_classification(prediction_gnb)
            else:
                self.pub_test.send_classification('ref')
                self.pub.send_classification('ref')
        self.online.set_trigger_in(2)
        print('sequence length:', data_for_classifier.shape[0])

    def got_new_sample(self):
        """
        adds a sample to the onlinereader which is doing the same as the meas
        """
        self.online.add_sample(self.sub.sensor_values)


if __name__ == '__main__':
    co = ClassifierOrganizer()
    co.startMeas()
