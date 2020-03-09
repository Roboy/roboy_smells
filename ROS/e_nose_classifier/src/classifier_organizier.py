from ROS.e_nose_classifier.src.e_nose_classification_publisher import eNoseClassificationPublisher
from ROS.e_nose_classifier.src.e_nose_classification_test import eNoseClassificationTestPublisher
from ROS.e_nose_classifier.src.e_nose_subscriber import eNoseSubscriber
from ROS.e_nose_classifier.src.e_nose_measurement_publisher import eNoseMeasurementPublisher
from classification.lstm_model import SmelLSTM
from classification.knn import KNN
from classification.naive_bayes import GNB
from e_nose.measurements import DataType, StandardizationType
from e_nose.online_reader import OnlineReader
import time
import numpy as np
import rospy



class ClassifierOrganizer:
    def __init__(self):
        print('Initialiting ROS node...')
        rospy.init_node('classifier_organizer', anonymous=True)
        print('Initialiting Subcomponents...')
        self.pub = eNoseClassificationPublisher()
        self.pub_test = eNoseClassificationTestPublisher()
        self.pub_meas = eNoseMeasurementPublisher()
        self.sub = eNoseSubscriber()
        failing_channels = [22, 31]
        #failing_channels = [0, 1, 2, 3, 4, 5, 6, 7, 22, 24, 25, 26, 27, 28, 29, 30, 31, 35, 36, 37, 38, 39, 56, 58, 59, 60, 61, 62, 63]
        #failing_channels = [0, 1, 2, 3, 4, 5, 6, 7, 22, 24, 25, 26, 27, 28, 29, 30, 31, 35, 36, 37, 38, 39, 56, 57, 58, 59, 60, 61, 62, 63]
        #failing_channels = [2, 3, 4, 5, 15, 16, 22, 23, 25, 26, 27, 28, 29, 31, 35, 36, 38, 39, 56, 59, 60, 61]
        working_channels = np.ones(64, bool)
        working_channels[failing_channels] = False
        num_working_channels = np.count_nonzero(working_channels)
        print(num_working_channels)
        self.online = OnlineReader(5, standardization=StandardizationType.LAST_REFERENCE, override_working_channels=working_channels)
        self.from_sample = 0
        self.sub.onUpdate += self.got_new_sample
        self.online.invoke_callback += self.gathered_data
        self.use_neural_network = False

        if self.use_neural_network:
            self.lstm1 = SmelLSTM(input_shape=(1, 50, num_working_channels), num_classes=6, hidden_dim_simple=6, stateful=False)
            self.lstm2 = SmelLSTM(input_shape=(1, 1, num_working_channels), num_classes=6, hidden_dim_simple=12, stateful=True)
# self.model_name = "LSTMTrainable_b625122c_11_batch_size=64,dim_hidden=16,lr=0.073956,return_sequences=True_2020-03-04_19-04-41c78mu_or"
            # self.model_name = 'LSTMTrainable_15750966_1740_batch_size=128,dim_hidden=6,lr=0.004831,return_sequences=True_2020-03-05_08-08-45fs4p25pg' 
            # self.model_name = 'LSTMTrainable_9c63b3de_15_batch_size=128,data_preprocessing=high_pass,dim_hidden=6,lr=0.018191,return_sequences=True_2020-03-05_14-41-14slhxixe7'
            # self.model_name = 'LSTMTrainable_987a098a_3_batch_size=128,data_preprocessing=high_pass,dim_hidden=6,lr=0.076584,return_sequences=True_2020-03-05_14-39-01_0r6hi60'
            # self.model_name = 'LSTMTrainable_f9244f1c_1_batch_size=128,data_preprocessing=high_pass,dim_hidden=6,lr=0.070814,return_sequences=True_2020-03-05_23-30-57teprjxra'
            #self.model_name = 'LSTMTrainable_231f7674_15_batch_size=128,data_preprocessing=high_pass,dim_hidden=12,lr=0.050039,stateful=True,use_lstm=True_2020-03-06_21-24-4970osptsr'

            self.model_name = 'LSTMTrainable_b8effb2c_12_batch_size=64,data_preprocessing=full,dim_hidden=6,lr=0.031576,use_lstm=True_2020-03-09_15-44-15nhhqrc38'
            self.model_name_2 = 'LSTMTrainable_acc8d910_3_batch_size=128,data_preprocessing=full,dim_hidden=12,lr=0.07236,use_lstm=False_2020-03-09_16-00-30jtxol1uj'

            #self.classifier.summary()
            self.lstm1.load_weights(self.model_name, checkpoint=260, path='classification/models/lstm_stateless/')
            self.lstm2.load_weights(self.model_name_2, checkpoint=120, path='classification/models/lstm_stateful/')

        else:
            self.datatype = DataType.HIGH_PASS
            self.classifier = KNN(num_working_channels, data_dir='data', data_type=self.datatype)
            self.classifier2 = GNB(data_dir='data', data_type=self.datatype)
        print('ros e_nose classification node started successfully')

    def startMeas(self):
        try:
            while True:
                var = input("Please enter something: ")
                print('restarting classification', var)
                self.from_sample = self.online.current_length
                self.online.set_trigger_in(4)
                if var.lower() == 'q':
                    break
        except KeyboardInterrupt:
            print('Interrupted...')

    def gathered_data(self):
        print('gathered data')
        data = self.online.get_since_n_as_measurement(self.from_sample)
        #print(data.correct_channels)
        #print(data.get_data().shape)
        data_for_classifier = data.get_data_as(self.datatype)
        self.pub_meas.send_classification(data_for_classifier)

        if self.use_neural_network:
            prediction = self.lstm2.predict_live(data)
            self.pub_test.send_classification(prediction)
            self.pub.send_classification(prediction)
        else:
            if (data_for_classifier.shape[0] > 40):
                prediction = self.classifier.predict(data_for_classifier)
                prediction_gnb = self.classifier2.predict(data_for_classifier)
                print('prediction_knn: ', prediction)
                print('prediction_gnb: ', prediction_gnb)
                self.pub_test.send_classification(prediction_gnb)
                self.pub.send_classification(prediction_gnb)
            else:
                self.pub_test.send_classification('no_data')
                self.pub.send_classification('no_data')
        self.online.set_trigger_in(2)
        print('sequence length:',data_for_classifier.shape[0])

    def got_new_sample(self):
        self.online.add_sample(self.sub.sensor_values)


if __name__ == '__main__':
    co = ClassifierOrganizer()
    co.startMeas()
