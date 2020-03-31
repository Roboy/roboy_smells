import numpy as np
import tensorflow as tf

from e_nose import file_reader
from e_nose import data_processing as dp
from e_nose.measurements import StandardizationType, DataType, Measurement
from scipy import signal
from typing import List

"""
This file contains data loading utilities for training the different ML models.
"""

def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int) -> np.ndarray:
    """
    This method initialises the scipy butter lowpass filter with the given parameters and then
    filters the given data.

    :param data: input data to be filtered
    :param cutoff: cutoff frequency
    :param fs: sample rate of the data signal
    :param order: the order of the filter
    :return: filtered data
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

def low_pass_mean_std_measurement(measurements: List[Measurement],
                                  sample_rate: float=0.5,
                                  cutoff_freq: float=0.02,
                                  order: int=2) -> list:
    """
    This method filters the data for each measurement using a low pass filter and then normalizes the data with
    mean and standard deviation.

    :param measurements: measurements to be processed
    :param sample_rate: sample rate of the data points in the measurements
    :param cutoff_freq: cutoff frequency for the low pass filter
    :param order: order of the low pass filter
    :return: filtered and normalized signal for every measurement
    """
    for meas in measurements:
        data = meas.get_data()
        ys = np.zeros_like(data)
        for i in range(data.shape[1]):
            y = butter_lowpass_filter(data[:,i], cutoff_freq, sample_rate, order)
            ys[:,i] = y

        for i in range(data.shape[0]):
            mean = np.mean(ys[i, :])
            var = np.std(ys[i, :])
            ys[i, :] = (ys[i, :] - mean) / var

        meas.data = ys
    return measurements

def get_measurements_train_test_from_dir(train_dir: str = '../data_train', test_dir: str = '../data_val') -> (np.ndarray, np.ndarray, int):
    """
    This method gets train and test data from the respective directories. It uses the correct channels found in
    the test file to guarantee that the models are not trained on less data than there is available in the test
    files

    :param train_dir: path to directory containing the train data
    :param test_dir: path to directory containing the test data
    :return: array of train and test measurements and the number of broken channels.
    """
    functionalisations_train, correct_channels, data_train = file_reader.read_all_files_in_folder(train_dir)
    functionalisations_test, correct_channels, data_test = file_reader.read_all_files_in_folder(test_dir)

    combined = data_train.copy()
    combined.update(data_test)

    correct_channels = dp.detect_broken_channels_multi_files(functionalisations_test, combined)
    dp.detect_broken_channels_multi_files

    measurements_per_file_test = {}
    for file in data_test:
        measurements_per_file_test[file] = dp.get_labeled_measurements(data_test[file], correct_channels, functionalisations_test)#, start_offset=[-5,  -2, 0, 2, 5])

    measurements_test = []
    for file in measurements_per_file_test:
        adding = dp.standardize_measurements(measurements_per_file_test[file], StandardizationType.LAST_REFERENCE)
        if adding is not None:
            measurements_test.extend(adding)

    measurements_per_file_train = {}
    for file in data_train:

        measurements_per_file_train[file] = dp.get_labeled_measurements(data_train[file], correct_channels, functionalisations_train)#, start_offset=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])

    measurements_train = []
    for file in measurements_per_file_train:
        adding = dp.standardize_measurements(measurements_per_file_train[file], StandardizationType.LAST_REFERENCE)
        if adding is not None:
            measurements_train.extend(adding)

    return np.array(measurements_train), np.array(measurements_test), np.count_nonzero(correct_channels)

def get_measurements_from_dir(directory_name: str= '../data_train') -> np.ndarray:
    """
    This method returns the standarized measurements found in the data in the specified directory.

    :param directory_name: path of the directory containing the data
    :return: array of measurements
    """
    functionalisations, correct_channels, data = file_reader.read_all_files_in_folder(directory_name)
    print(np.count_nonzero(correct_channels))
    measurements_per_file = {}
    for file in data:
        measurements_per_file[file] = dp.get_labeled_measurements(data[file], correct_channels, functionalisations)

    measurements = []
    for file in measurements_per_file:
        adding = dp.standardize_measurements(measurements_per_file[file], StandardizationType.LAST_REFERENCE)
        if adding is not None:
            measurements.extend(adding)

    return np.array(measurements)


def train_test_split(measurements: np.ndarray, split: float = 0.8) -> (np.ndarray, np.ndarray):
    """
    This method splits an array of measurements into train and test data sets with the specified split.

    :param measurements: array of measurements
    :param split: percentage of data that should be in the train data set
    :return: train measurement array and test measurement array
    """
    labels_measurements = [m.label for m in measurements]
    labels = np.unique(labels_measurements)

    for i, l in enumerate(labels):
        indices_label = np.argwhere(np.array(labels_measurements) == l).flatten()

        num_samples = indices_label.size
        if i == 0:
            measurements_train = measurements[indices_label][:int(split*num_samples)]
            measurements_test = measurements[indices_label][int(split*num_samples):]
        else:
            measurements_train = np.append(measurements_train, measurements[indices_label][:int(split*num_samples)])
            measurements_test = np.append(measurements_test, measurements[indices_label][int(split*num_samples):])

    np.random.shuffle(measurements_train)
    np.random.shuffle(measurements_test)

    return measurements_train, measurements_test

def shuffle(dataset_one: np.ndarray, dataset_two: np.ndarray = None) -> np.ndarray:
    """
    Function to shuffle one or respectively two datasets.

    :param dataset_one:                 Dataset array to be shuffled.
    :param dataset_two:                 Second dataset to be shuffled if defined.
    :return:                            Shuffled datasets.
    """
    np.random.shuffle(dataset_one)
    if dataset_two is not None:
        np.random.shuffle(dataset_two)
        return dataset_one, dataset_two
    else:
        return dataset_one

def get_batched_data(measurements: list, classes_dict: dict = None, masking_value: float = 100., data_type: DataType = DataType.HIGH_PASS,
                     batch_size: int = 64,
                     sequence_length: int = 45,
                     dimension: int = 62,
                     return_sequences: bool = True) -> (np.ndarray, np.ndarray, list):
    """
    This function loads the data and labels as batches for training a stateful recurrent model. Therefore a certain ordering
    of the data is required. For further information we refer to the stateful training tutorial of the tf.keras library.

    :param measurements:                List of Measurement objects from which data will be obtained.
    :param classes_dict:                Classes dictionary.
    :param masking_value:               Masking value used to pad sequences. Data points with this value will ignored by network.
    :param data_type:                   Type of data preprocessing.
    :param batch_size:                  Batch size.
    :param sequence_length:             Length of data sequence.
    :param dimension:                   Number of dimensions of data. Should be equal to the least common number of
                                        correctly working channels for the given measurements.
    :param return_sequences:            If set to True, batched labels will be of shape (batch_size, sequence_length, 1),
                                        otherwise (batch_size, 1)
    :return:                            Batched data array of shape (Number of batches, batch_size, sequence_length, dimension),
                                        batched label array of shape (Number of batches, batch_size, (sequence_length,) 1),
                                        list of batch indices where sequences start and therefore state resets should be performed.
    """
    if classes_dict == None:
        classes_list = ['acetone', 'isopropanol', 'orange_juice', 'pinot_noir', 'raisin', 'wodka']
        classes_dict = {}
        for i, c in enumerate(classes_list):
            classes_dict[c] = i

    measurement_indices = np.arange(len(measurements))
    np.random.shuffle(measurement_indices)

    padding = batch_size-(measurement_indices.size % batch_size)
    measurement_indices = np.append(measurement_indices, np.ones(padding, dtype=int) * (-1))
    measurement_indices = np.reshape(measurement_indices, (-1, batch_size))

    batches_data = []
    batches_labels = []

    for i in range(measurement_indices.shape[0]):
        batch_indices = measurement_indices[i]

        batch_list = []
        batch_list_labels = []
        max_len = 0
        for b in range(batch_size):
            index = batch_indices[b]
            if index != -1:
                series_data = measurements[index].get_data_as(data_type)
                series_labels = np.ones(shape=(series_data.shape[0], 1), dtype=int) * classes_dict[measurements[index].label]
            else:
                series_data = np.ones(shape=(1, dimension), dtype=float) * masking_value
                series_labels = np.ones(shape=(1, 1), dtype=int) * 0

            if series_data.shape[0] > max_len:
                max_len = series_data.shape[0]
            batch_list.append(series_data)
            batch_list_labels.append(series_labels)

        batch = np.ones(shape=(batch_size, max_len, dimension), dtype=float) * masking_value
        batch_labels = np.ones(shape=(batch_size, max_len, 1), dtype=int) * 0

        for b in range(batch_size):
            batch[b, :batch_list[b].shape[0]] = batch_list[b]
            batch_labels[b, :batch_list_labels[b].shape[0]] = batch_list_labels[b]
        batches_data.append(batch)
        batches_labels.append(batch_labels)

    for i, ba in enumerate(batches_data):
        ba_labels = batches_labels[i]
        padding_length = sequence_length - (ba.shape[1] % sequence_length)
        if padding_length != sequence_length:
            ba = np.append(ba, np.ones(shape=(batch_size, padding_length, dimension), dtype=float) * masking_value, axis=1)
            ba_labels = np.append(batches_labels[i], np.ones(shape=(batch_size, padding_length, 1), dtype=int) * 0, axis=1)
        split = int(ba.shape[1] / sequence_length)

        ba = np.array(np.split(ba, split, axis=1))
        ba_labels = np.array(np.split(ba_labels, split, axis=1))

        if i == 0:
            batches_data_done = ba
            batches_labels_done = ba_labels
            starting_indices = np.array([0])
        else:
            starting_indices = np.append(starting_indices, batches_data_done.shape[0])
            batches_data_done = np.append(batches_data_done, ba, axis=0)
            batches_labels_done = np.append(batches_labels_done, ba_labels, axis=0)

    batches_labels_done = batches_labels_done.astype(int)

    if return_sequences == False:
        batches_labels_done_stateless = np.empty(shape=(batches_labels_done.shape[0],
                                                        batches_labels_done.shape[1],
                                                        batches_labels_done.shape[3]))
        for i, y in enumerate(batches_labels_done):
            batches_labels_done_stateless[i] = y[:, 0, :]
        batches_labels_done = batches_labels_done_stateless

    return batches_data_done, batches_labels_done, starting_indices

def get_data_stateless(measurements: list, dimension: int = 62, return_sequences: bool = True, sequence_length: int = 45,
                       masking_value: float = 100.,
                       batch_size: int = 64,
                       classes_dict: dict = None,
                       data_type: DataType = DataType.HIGH_PASS) -> tf.data.Dataset:
    """
    This function loads the data and labels as batches for training a stateless recurrent model.

    :param measurements:                List of Measurement objects from which data will be obtained.
    :param dimension:                   Number of dimensions of data. Should be equal to the least common number of
                                        correctly working channels for the given measurements.
    :param return_sequences:            If set to True, label batches will be of shape (batch_size, sequence_length, 1),
                                        otherwise (batch_size, 1).
    :param sequence_length:             Length of data sequence.
    :param masking_value:               Masking value used to pad sequences. Data points with this value will ignored by network.
    :param batch_size:                  Batch size.
    :param classes_dict:                Classes dictionary.
    :param data_type:                   Type of data preprocessing.
    :return:                            tf.data.Dataset containing data and label batches.
    """
    if classes_dict == None:
        classes_list = ['acetone', 'isopropanol', 'orange_juice', 'pinot_noir', 'raisin', 'wodka']
        classes_dict = {}
        for i, c in enumerate(classes_list):
            classes_dict[c] = i

    padding = batch_size - len(measurements)%batch_size

    full_length = len(measurements) + padding

    full_data = np.ones(shape=(full_length, sequence_length, dimension)) * masking_value

    if return_sequences:
        full_labels = np.zeros(shape=(full_length, sequence_length, 1), dtype=int)
    else:
        full_labels = np.zeros(shape=(full_length, 1), dtype=int)
    labels_shape = full_labels.shape[1:]

    for i, m in enumerate(measurements):
        d = m.get_data_as(data_type)
        if d.shape[1] != dimension:
            raise ValueError("Dimension mismatch")
        if d.shape[0] < sequence_length:
            raise ValueError("Measurement too short!")
        full_data[i] = m.get_data_as(data_type)[:sequence_length, :]
        full_labels[i] = np.ones(shape=labels_shape, dtype=int)*classes_dict[m.label]

    return tf.data.Dataset.from_tensor_slices((tf.constant(full_data), tf.constant(full_labels)))

def get_data_simple_models(measurements: list, dimension: int = 62, return_sequences: bool = True, sequence_length: int = 45,
                 masking_value: float = 100.,
                 batch_size: int = 1,
                 classes_dict: dict = None,
                 data_type: DataType = DataType.HIGH_PASS) -> (np.ndarray, np.ndarray):
    """
    This function loads data and labels from a list of measurements for kNN and naive Bayes classifier.
    For this the sub samples of the data sequence after a certain time specified by sequence length are taken.

    :param measurements:                List of Measurement objects from which data will be obtained
    :param dimension:                   Number of dimensions of data. Should be equal to the least common number of
                                        correctly working channels for the given measurements.
    :param return_sequences:            If set to True, label batches will be of shape (Number of samples, sequence_length, 1),
                                        otherwise (Number of samples, 1).
    :param sequence_length:             The data points at sequence_length will be taken.
    :param masking_value:               List of Measurement objects from which data will be obtained.
    :param batch_size:                  Batch size.
    :param classes_dict:                Classes dictionary.
    :param data_type:                   Type of data preprocessing.
    :return:                            Data array of shape (Number of samples, sequence_length, dimension)
                                        labels array of shape (Number of samples, (sequence_length,) 1)
    """
    if classes_dict == None:
        classes_list = ['acetone', 'isopropanol', 'orange_juice', 'pinot_noir', 'raisin', 'wodka']
        classes_dict = {}
        for i, c in enumerate(classes_list):
            classes_dict[c] = i

    padding = batch_size - len(measurements)%batch_size

    if padding == batch_size:
        padding = 0

    full_length = len(measurements) + padding

    full_data = np.ones(shape=(full_length, sequence_length, dimension)) * masking_value

    if return_sequences:
        full_labels = np.zeros(shape=(full_length, sequence_length, 1), dtype=int)
    else:
        full_labels = np.zeros(shape=(full_length, 1), dtype=int)
    labels_shape = full_labels.shape[1:]

    for i, m in enumerate(measurements):
        d = m.get_data_as(data_type)
        if d.shape[1] != dimension:
            raise ValueError("Dimension mismatch")
        if d.shape[0] < sequence_length:
            raise ValueError("Measurement too short!")
        full_data[i] = m.get_data_as(data_type)[:sequence_length, :]
        full_labels[i] = np.ones(shape=labels_shape, dtype=int)*classes_dict[m.label]

    return full_data, full_labels