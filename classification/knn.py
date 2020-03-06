from sklearn.neighbors import NearestNeighbors
import numpy as np

from e_nose import file_reader
from e_nose import data_processing as dp

from e_nose.measurements import DataType
from sklearn.neighbors import KNeighborsClassifier as KNNC
from classification import data_loading as dl

class KNN:
    def __init__(self, input_dim, last_avg=5, weights='distance', metric='euclidean', data_dir='../data_train', sequence_length=50, num_neighbors=5, data_type=DataType.HIGH_PASS, classes_list=['coffee_powder', 'isopropanol', 'orange_juice', 'raisin', 'red_wine', 'wodka']):
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.data_type = data_type
        self.last_avg = last_avg
        self.classes_list = classes_list
        self.model = KNNC(num_neighbors, weights=weights, metric=metric)
        self.data_dir = data_dir
        self.fit()

    def fit(self):
        measurements_tr, measurements_te, correct_channels = dl.get_measurements_train_test_from_dir(self.data_dir, self.data_dir)
        train_data, train_labels = dl.get_data_knn(measurements_tr, batch_size=1, sequence_length=self.sequence_length, dimension=self.input_dim,
                                                   data_type=self.data_type)
        knn_tr_data = np.mean(train_data[:, -self.last_avg:-1, :], axis=1)
        knn_tr_labels = train_labels[:, -1, :]
        print(knn_tr_data.shape, knn_tr_labels.shape)
        self.model.fit(knn_tr_data, knn_tr_labels.flatten())

    def predict(self, data):

        if data.shape[0] > self.last_avg:
            sample = np.mean(data[-self.last_avg:-1, :], axis=0)
        else:
            sample = data[-1]

        sample = np.expand_dims(sample, axis=0)
        print('sample shape', sample.shape)
        p = self.model.predict(sample).flatten()[0]
        print(p)
        prediction = self.classes_list[p]
        return prediction



#knn = KNN(34)

#s = np.random.randn(1, 34)
#print(s.shape)
#print('pred: ', knn.predict(s))
