from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
import numpy as np

from e_nose.measurements import DataType
from classification import data_loading as dl

class GNB:
    def __init__(self, input_dim: int = 62, last_avg: int = 3, data_dir: str = '../data_train', sequence_length: int = 45,
                 data_type: DataType = DataType.HIGH_PASS,
                 classes_list: list = ['acetone', 'isopropanol', 'orange_juice', 'pinot_noir', 'raisin', 'wodka']):
        """
        Class for a classifier based on a Gaussian Naive Bayes approach defining training and prediction function.
        The saturated sensor values are assumed to be Gaussian distributed. During training mean and covariance matrix of
        a Gaussian are fit to the underlying data for each class. During inference a new datapoint the parameterized model is
        used to find the corresponding class with the highest probability.
        For debugging (checking distributions of the data) the class contains PCA that is fit to the data during training,
        which can be used to visualize the two strongest components of the saturated datapoints.

        :param input_dim:           Number of dimensions of input data.
        :param last_avg:            Number of last time steps used to compute mean of saturated channel.
        :param data_dir:            Path to data directory containing training csv files that are used to fit model.
        :param sequence_length:     Specifies time step of a measurement sequence at which data points are extracted.
                                    Sensor channels should be saturated at that point.
        :param data_type:           Type of data preprocessing.
        :param classes_list:        List of classes to be learnt by model.
        """
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.data_type = data_type
        self.last_avg = last_avg
        self.classes_list = classes_list
        self.classes_dict = {}
        for i, c in enumerate(classes_list):
            self.classes_dict[c] = i
        self.model = GaussianNB()
        self.data_dir = data_dir
        self.pca = PCA(2)
        self.fit()
        self.correct_channels


    def fit(self):
        """
        Fits the model parameters and an additional PCA to the training data.
        """
        measurements_tr, measurements_te, self.correct_channels = dl.get_measurements_train_test_from_dir(self.data_dir, self.data_dir)
        train_data, train_labels = dl.get_data_simple_models(measurements_tr, batch_size=1, sequence_length=self.sequence_length, dimension=self.input_dim,
                                                   data_type=self.data_type, classes_dict=self.classes_dict)

        nb_tr_data = np.mean(train_data[:, -self.last_avg:-1, :], axis=1)
        nb_tr_labels = train_labels[:, -1, :]
        self.pca.fit(nb_tr_data)
        #print(knn_tr_data.shape, knn_tr_labels.shape)
        self.model.fit(nb_tr_data, nb_tr_labels.flatten())

    def predict_from_batch(self, data: np.ndarray) -> str:
        """
        Classifies given data batch and returns prediction.

        :param data:                Data array of shape (dimensions), (sequence_length, dimensions)
                                    or (1, sequence_length, dimensions)
        :return:                    Prediction for the given data sequence or data point.
        """
        if len(data.shape) < 3:
            data = np.expand_dims(data, axis=0)
            if len(data.shape) < 3:
                data = np.expand_dims(data, axis=0)
        if data.shape[1] > self.last_avg:
            sample = np.mean(data[0, -self.last_avg:-1, :], axis=0)
        else:
            sample = data[0, -1]

        sample = np.expand_dims(sample, axis=0)
        p = self.model.predict(sample).flatten()[0]
        prediction = self.classes_list[p]
        return prediction