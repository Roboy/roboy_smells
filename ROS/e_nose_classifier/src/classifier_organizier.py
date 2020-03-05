from ROS.e_nose_classifier.src.e_nose_classification_publisher import eNoseClassificationPublisher
from ROS.e_nose_classifier.src.e_nose_classification_test import eNoseClassificationTestPublisher
from ROS.e_nose_classifier.src.e_nose_subscriber import eNoseSubscriber
from ROS.e_nose_classifier.src.online_reader import OnlineReader
from classification.lstm_model import SmelLSTM
from ROS.e_nose_classifier.src.measurements import DataType
import rospy
import keyboard  # using module keyboard


class ClassifierOrganizer:
    def __init__(self):
        print('Initialiting ROS node...')
        rospy.init_node('e_nose_classifier_publish', anonymous=False)
        print('Initialiting Subcomponents...')
        self.pub = eNoseClassificationPublisher()
        self.pub_test = eNoseClassificationTestPublisher()
        self.sub = eNoseSubscriber()
        self.online = OnlineReader(5)
        self.sub.onUpdate += self.gotNewSample
        self.classifier = SmelLSTM(input_shape=(1,1,42), num_classes=6, hidden_dim_simple=6, data_type=DataType.FULL)
        self.model_name = 'LSTMTrainable_9bbda05c_13_batch_size=128,data_preprocessing=full,dim_hidden=6,lr=0.0070389,return_sequences=True_2020-03-05_14-41-03jgrcr7l1'
        self.classifier.load_weights(self.model_name, checkpoint=200, path='classification/models/rnn/')
        keyboard.on_press_key("space", self.startMeas)
        print('ros e_nose classification node started successfully')
        rospy.spin()

    def startMeas(self):
        print('test')
        self.online.set_trigger_in(self.gatheredData, 50)

    def gatheredData(self):
        print('gathered data')
        data = self.online.get_last_n_as_measurement(50)
        prediction = self.classifier.predict_live(data)
        print('prediction: ', prediction)
        self.pub_test.send_classification(prediction)

    def gotNewSample(self):
        self.online.add_sample(self.sub.sensorValues)


if __name__ == '__main__':
    co = ClassifierOrganizer()
