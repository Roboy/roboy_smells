from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
import numpy as np
from e_nose.measurements import DataType
from classification import data_loading as dl

import matplotlib.pyplot as plt

class GNB:
    def __init__(self, input_dim=62, last_avg=3, data_dir='data', filter=None, sequence_length=50, data_type=DataType.HIGH_PASS, classes_list=['acetone', 'isopropanol', 'orange_juice', 'pinot_noir', 'raisin', 'wodka']):
        self.input_dim = input_dim
        self.filter = filter
        self.sequence_length = sequence_length
        self.data_type = data_type
        self.last_avg = last_avg
        self.classes_list = classes_list
        self.classes_dict = {}
        for i, c in enumerate(classes_list):
            self.classes_dict[c] = i
        self.model = GaussianNB()
        print('datadir', data_dir)
        self.data_dir = data_dir
        self.pca = PCA(2)
        self.fit()
        self.correct_channels

        plt.ion()
        plt.show()

    def fit(self):
        measurements_tr, measurements_te, self.correct_channels = dl.get_measurements_train_test_from_dir(self.data_dir, self.data_dir)

        print(len(measurements_tr))
        measurements = []
        if self.filter is not None:
            for m in measurements_tr:
                if m.label == self.filter:
                    continue
                measurements.append(m)
        else:
            measurements = measurements_tr

        train_data, train_labels = dl.get_data_knn(measurements, batch_size=1, sequence_length=self.sequence_length, dimension=self.input_dim,
                                                   data_type=self.data_type, classes_dict=self.classes_dict)

        self.nb_tr_data = np.mean(train_data[:, -self.last_avg:-1, :], axis=1)
        self.nb_tr_labels = train_labels[:, -1, :]
        self.pca.fit(self.nb_tr_data)
        #print(knn_tr_data.shape, knn_tr_labels.shape)

        print(self.nb_tr_data.shape)
        self.model.fit(self.nb_tr_data, self.nb_tr_labels.flatten())

    def predict(self, data):

        if data.shape[0] > self.last_avg:
            sample = np.mean(data[-self.last_avg:-1, :], axis=0)
        else:
            sample = data[-1]

        plt.scatter(self.nb_tr_data[:, 0], self.nb_tr_data[:, 1], c=self.nb_tr_labels[:, 0], marker='.')
        plt.scatter(data[0,0], data[0, 1], marker='d')
        plt.draw()
        plt.pause(0.001)

        sample = np.expand_dims(sample, axis=0)
        print('sample shape', sample.shape)
        p = self.model.predict(sample).flatten()[0]
        print(p)
        prediction = self.classes_list[p]
        return prediction

nb = GNB()


