import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNNC

from e_nose.measurements import DataType
from classification import data_loading as dl

class KNN:
    def __init__(self, input_dim: int = 62, last_avg: int = 3, data_dir: str = '../data_train', sequence_length: int = 45,
                 data_type: DataType = DataType.HIGH_PASS,
                 classes_list: list = ['acetone', 'isopropanol', 'orange_juice', 'pinot_noir', 'raisin', 'wodka'],
                 weights: str = 'distance', metric: str = 'euclidean', num_neighbors: int = 5):
        """
        Class for a classifier based on k-nearest-neighbor approach defining training and prediction function.
        The saturated sensor values of the same class are assumed to have a small distance, whereas the distance between
        data points of different classes should be large. During inference the classes of the num_neghbors nearest
        neighbors are used to predict the class of the new datapoint by performing a (weighted) majority vote.
        Our best performing model uses 5 neighbors, the euclidean space and a distance weighting.

        :param input_dim:           Number of dimensions of input data.
        :param last_avg:            Number of last time steps used to compute mean of saturated channel.
        :param data_dir:            Path to data directory containing training csv files that are used to fit model.
        :param sequence_length:     Specifies time step of a measurement sequence at which data points are extracted.
                                    Sensor channels should be saturated at that point.
        :param data_type:           Type of data preprocessing.
        :param classes_list:        List of classes to be learnt by model.
        :param weights:             Kind of weighting of the neighbors.
        :param metric:              Metric space. For more options we refer to the sklearn library.
        :param num_neighbors:       Number of neighbors to consider.
        """
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.data_type = data_type
        self.last_avg = last_avg
        self.classes_list = classes_list
        self.model = KNNC(num_neighbors, weights=weights, metric=metric)
        self.data_dir = data_dir
        self.classes_dict = {}
        for i, c in enumerate(classes_list):
            self.classes_dict[c] = i
        self.fit()

    def fit(self):
        """
        Fits the model parameters and to the training data.
        """
        measurements_tr, measurements_te, correct_channels = dl.get_measurements_train_test_from_dir(self.data_dir, self.data_dir)
        train_data, train_labels = dl.get_data_simple_models(measurements_tr, batch_size=1, sequence_length=self.sequence_length, dimension=self.input_dim,
                                                   data_type=self.data_type, classes_dict=self.classes_dict)
        knn_tr_data = np.mean(train_data[:, -self.last_avg:-1, :], axis=1)
        knn_tr_labels = train_labels[:, -1, :]
        print(knn_tr_data.shape, knn_tr_labels.shape)
        self.model.fit(knn_tr_data, knn_tr_labels.flatten())

    def predict_from_batch(self, data):
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
            sample = data[-1]

        sample = np.expand_dims(sample, axis=0)
        print('sample shape', sample.shape)
        p = self.model.predict(sample).flatten()[0]
        print(p)
        prediction = self.classes_list[p]
        return prediction