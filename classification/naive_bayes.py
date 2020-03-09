from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
import numpy as np
from e_nose.measurements import DataType
from classification import data_loading as dl

class GNB:
    def __init__(self, input_dim=62, last_avg=3, data_dir='../data_train', sequence_length=50, data_type=DataType.HIGH_PASS, classes_list=['acetone', 'isopropanol', 'orange_juice', 'pinot_noir', 'raisin', 'wodka']):
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
        measurements_tr, measurements_te, self.correct_channels = dl.get_measurements_train_test_from_dir(self.data_dir, self.data_dir)
        train_data, train_labels = dl.get_data_knn(measurements_tr, batch_size=1, sequence_length=self.sequence_length, dimension=self.input_dim,
                                                   data_type=self.data_type, classes_dict=self.classes_dict)

        nb_tr_data = np.mean(train_data[:, -self.last_avg:-1, :], axis=1)
        nb_tr_labels = train_labels[:, -1, :]
        self.pca.fit(nb_tr_data)
        #print(knn_tr_data.shape, knn_tr_labels.shape)
        self.model.fit(nb_tr_data, nb_tr_labels.flatten())

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

nb = GNB()


