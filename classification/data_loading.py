import numpy as np

from e_nose import file_reader
from e_nose import data_processing as dp


def get_measurements_from_dir(directory_name='../data'):
    functionalisations, correct_channels, data = file_reader.read_all_files_in_folder(directory_name)
    measurements_per_file = {}
    for file in data:
        measurements_per_file[file] = dp.get_labeled_measurements(data[file], correct_channels, functionalisations)

    measurements = []
    for file in measurements_per_file:
        adding = dp.standardize_measurements(measurements_per_file[file])
        if adding is not None:
            measurements.extend(adding)

    return np.array(measurements)

def train_test_split(measurements, split=0.8):
    labels_measurements = [m.label for m in measurements]
    labels = np.unique(labels_measurements)

    for i, l in enumerate(labels):
        indices_label = np.argwhere(np.array(labels_measurements) == l).flatten()

        num_samples = indices_label.size
        if i == 0:
            print(int(split*num_samples))
            measurements_train = measurements[indices_label][:int(split*num_samples)]
            measurements_test = measurements[indices_label][int(split*num_samples):]
        else:
            measurements_train = np.append(measurements_train, measurements[indices_label][:int(split*num_samples)])
            measurements_test = np.append(measurements_test, measurements[indices_label][int(split*num_samples):])

    np.random.shuffle(measurements_train)
    np.random.shuffle(measurements_test)

    return measurements_train, measurements_test

def shuffle(dataset_one, dataset_two=None):
    np.random.shuffle(dataset_one)
    if dataset_two is not None:
        np.random.shuffle(dataset_two)
        return dataset_one, dataset_two
    else:
        return dataset_one

