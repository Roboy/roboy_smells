import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
from . import file_reader
from . import data_processing as dp
from .measurements import DataType


LOG_DIR = 'logs'
DATA_DIR = 'data'
datatype = DataType.TOTAL_AVG

#TODO maybe make this configurable so multiple metadata files are possible
metadata = 'metadata.tsv'

# Read in data
#TODO: make the data dir configurable
functionalisations, correct_channels, data = file_reader.read_all_files_in_folder(DATA_DIR)

# Get measurements out of data and standardize it
measurements_per_file = {}
for file in data:
    measurements_per_file[file] = dp.get_labeled_measurements(data[file], correct_channels, functionalisations)

measurements = []
print('Using the following files for analysis:')
for file in measurements_per_file:
    print("file: ", file)
    adding = dp.standardize_measurements_2(measurements_per_file[file])
    if adding is not None:
        measurements.extend(adding)

print('Total of', len(measurements), 'measurements')
assert(len(measurements) > 0)

# Save data to tf variable
ms = np.zeros((len(measurements), measurements[0].get_data().shape[1]))
ls = []
for i, measurement in enumerate(measurements):
    #TODO make type of measurement average configureable
    # IF YOU WANT TO CHANGE WHAT TYPE AVERAGE IS USED DO IT HERE!
    ms[i, :] = measurement.get_data_as(datatype)
    ls.append(measurement.label)

# Save metadata
with open(os.path.join(LOG_DIR, metadata), 'w') as metadata_file:
    for row in ls:
        metadata_file.write('%s\n' % row)

tf_ms = tf.Variable(ms, name="measurements")

with tf.Session() as sess:
    saver = tf.train.Saver([tf_ms])

    sess.run(tf_ms.initializer)
    print(tf_ms.shape)
    #TODO maybe make this configurable so multiple ckpt files are possible
    saver.save(sess, os.path.join(LOG_DIR, 'ms.ckpt'))

    config = projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = tf_ms.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = metadata
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)

print()
print('Successfully created tensorboard files. Use the following command the launch tensorboard:')
print('tensorboard --logdir=logs')