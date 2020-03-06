import numpy as np
import tensorflow as tf

from e_nose import file_reader
from e_nose import data_processing as dp
from e_nose.measurements import StandardizationType, DataType
from scipy import signal

def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

def low_pass_mean_std_measurement(measurements, sample_rate=0.5, cutoff_freq=0.02, order=2):
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

def get_measurements_train_test_from_dir(train_dir='../data', test_dir='../data'):
    functionalisations_train, correct_channels, data_train = file_reader.read_all_files_in_folder(train_dir)
    functionalisations_test, correct_channels, data_test = file_reader.read_all_files_in_folder(test_dir)

    combined = data_train.copy()
    combined.update(data_test)

    correct_channels = dp.find_broken_channels_multi_files(functionalisations_test, combined)
    dp.find_broken_channels_multi_files

    measurements_per_file_test = {}
    for file in data_test:
        measurements_per_file_test[file] = dp.get_labeled_measurements(data_test[file], correct_channels, functionalisations_test, start_offset=[-5, -4, -3, -2, -1 0, 1, 2, 3, 4, 5])

    measurements_test = []
    for file in measurements_per_file_test:
        adding = dp.standardize_measurements(measurements_per_file_test[file], StandardizationType.LAST_REFERENCE)
        if adding is not None:
            measurements_test.extend(adding)

    measurements_per_file_train = {}
    for file in data_train:
        measurements_per_file_train[file] = dp.get_labeled_measurements(data_train[file], correct_channels, functionalisations_train)

    measurements_train = []
    for file in measurements_per_file_train:
        adding = dp.standardize_measurements(measurements_per_file_train[file], StandardizationType.LAST_REFERENCE)
        if adding is not None:
            measurements_train.extend(adding)

    return np.array(measurements_train), np.array(measurements_test), np.count_nonzero(correct_channels)

def get_measurements_from_dir(directory_name='../data'):
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


def train_test_split(measurements, split=0.8):
    labels_measurements = [m.label for m in measurements]
    labels = np.unique(labels_measurements)

    for i, l in enumerate(labels):
        indices_label = np.argwhere(np.array(labels_measurements) == l).flatten()

        num_samples = indices_label.size
        if i == 0:
            #print(int(split*num_samples))
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


def get_batched_data(measurements, classes_dict, masking_value, data_type=DataType.HIGH_PASS, batch_size=4, sequence_length=4, dimension=64, return_sequences=True):

    measurement_indices = np.arange(len(measurements))
    np.random.shuffle(measurement_indices)

    padding = batch_size-(measurement_indices.size % batch_size)
    measurement_indices = np.append(measurement_indices, np.ones(padding, dtype=int) * int(masking_value))
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
            #print(index)
            if index != masking_value:
                series_data = measurements[index].get_data_as(data_type)
                #print(classes_dict[measurements[index].label])
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
        #print("ba:", ba.shape)
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
    #print(type(batches_labels_done))

    #print(batches_data_done.shape)
    #print(batches_labels_done.shape)

    if return_sequences == False:
        batches_labels_done_stateless = np.empty(shape=(batches_labels_done.shape[0],
                                                        batches_labels_done.shape[1],
                                                        batches_labels_done.shape[3]))
        for i, y in enumerate(batches_labels_done):
            batches_labels_done_stateless[i] = y[:, 0, :]
        batches_labels_done = batches_labels_done_stateless
    #print('batches_labels_done.shape: ', batches_labels_done.shape)
    #print('batches_labels_done: ', batches_labels_done)


    return batches_data_done, batches_labels_done, starting_indices


def get_data_stateless(measurements, dimension=35, return_sequences=True, augment=False, sequence_length=50, masking_value=100., batch_size=64, classes_dict=None, data_type=DataType.HIGH_PASS):
    if classes_dict == None:
        classes_list = ['coffee_powder', 'isopropanol', 'orange_juice', 'raisin', 'red_wine', 'wodka']
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

    #indices = np.arange(full_labels.shape[0])
    #np.random.shuffle(indices)

    print(full_data.shape, full_labels.shape)

    return tf.data.Dataset.from_tensor_slices((tf.constant(full_data), tf.constant(full_labels)))

'''
measurements = get_measurements_from_dir('../data_test')[:6]
data, labels = get_data_stateless(measurements, return_sequences=False , dimension=42)
print(labels)

import tensorflow as tf
dataset = tf.data.Dataset.from_tensor_slices((tf.constant(data), tf.constant(labels)))

dataset = dataset.batch(2)

print(dataset)

for i in range(2):
    if i > 0:
        dataset = dataset.shuffle(len(measurements))
    for X, y in dataset:
        #print(X)
        print(y)
    print('#############################################################')

'''
