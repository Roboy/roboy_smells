import classification.data_loading as cdl
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import mahalanobis
import numpy as np
from e_nose.measurements import DataType
measurements = cdl.get_measurements_from_dir("data")
print("length measurements: ", len(measurements))

training, test = cdl.train_test_split(measurements)
training_X = []
training_Y = []
test_X = []
test_Y = []

for i, measurement in enumerate(test):
    for x, tmp in enumerate(measurement.get_data_as(DataType.HIGH_PASS)):
        test_X.append(tmp)
        test_Y.append(measurement.label)

for i, measurement in enumerate(training):
    for x, tmp in enumerate(measurement.get_data_as(DataType.HIGH_PASS)):
        training_X.append(tmp)
        training_Y.append(measurement.label)

cov_X = np.linalg.inv(np.cov(training_X))

#print(cov_X)

for i in range(1, 21):
    fitting = KNeighborsClassifier(i,'distance',metric='minkowski',p=2,algorithm='auto')
    fitting.fit(training_X, training_Y)
    predicted = fitting.predict(test_X)
    acc = accuracy_score(test_Y, predicted)
    print("accuracy for ", i, " neighbours: ", acc)


def mahalanobis(a, b):
    return mahalanobis(a, b, cov_X)

def eu