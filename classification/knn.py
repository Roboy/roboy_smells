from sklearn.neighbors import NearestNeighbors
import numpy as np

from e_nose import file_reader
from e_nose import data_processing as dp

data_dir = '../data'

def euk_dist(a, b):
    return np.sqrt(np.sum(np.square(a-b)))

def preprocess(data_dir_name):
    functionalisations, correct_channels, data = file_reader.read_all_files_in_folder(data_dir_name)

    measurements_per_file = {}
    for file in data:
        measurements_per_file[file] = dp.get_labeled_measurements(data[file], correct_channels, functionalisations)

    measurements = []
    for file in measurements_per_file:
        #print("file: ",file)
        adding = dp.standardize_measurements(measurements_per_file[file])
        if adding is not None:
            measurements.extend(adding)

    print(len(measurements))


    labels = list(set([m.label for m in measurements]))
    measurements_by_labels = {}
    for i, l in enumerate(labels):
        measurements_by_labels[l] = []
    for m in measurements:
        for i, l in enumerate(labels):
            if m.label == l:
                measurements_by_labels[l].append(m)




num_classes = len(labels)
num_active_channels = measurements[0].get_data().shape[1]
print(num_classes, num_active_channels)

centroids = np.zeros([num_classes, num_active_channels])
print(measurements_by_labels['wodka'][1].get_data().shape)

data = np.zeros([num_classes, 10, num_active_channels])
for c, label in enumerate(labels):
    for run in range(10):
        #print(np.mean(measurements_by_labels[label][run].get_data()[-5:, :], axis=0).shape)
        data[c, run, :] = np.mean(measurements_by_labels[label][run].get_data()[-5:, :], axis=0)
        #print(data.shape)
    #print('c: ', centroids[c,:].shape)
    #print('datameand: ', np.mean(data[c, :, :], axis=0).shape)
    centroids[c, :] = np.mean(data[c, :, :], axis=0)

print(centroids.shape)

data_flattened = data.reshape(-1, data.shape[-1])

nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(data_flattened)
distances, indices = nbrs.kneighbors(data_flattened)
distances = distances[:,1:]
indices = indices[:,1:]

def get_classes(classes):
    unique_classes, counts = np.unique(classes, return_counts=True)
    winner_index = np.argwhere(counts == np.amax(counts)).flatten()
    return unique_classes[winner_index]


neighbor_classes = (indices / 10).astype(int)

correct_counter = 0
false_counter = 0
unknown_counter = 0

for i in range(neighbor_classes.shape[0]):
    gt_class = int(i / 10)
    pred_classes = get_classes(neighbor_classes[i])
    if pred_classes.size == 1:
        if pred_classes[0] == gt_class:
            correct_counter += 1
        else:
            false_counter += 1
    else:
        unknown_counter += 1

accuracy = correct_counter / (correct_counter + false_counter + unknown_counter)
print("accuracy: ", accuracy)