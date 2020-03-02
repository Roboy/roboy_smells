import numpy as np
import tensorflow as tf
import random

from classification.data_loading import get_measurements_from_dir, train_test_split, shuffle


def get_batched_data(measurements, masking_value, batch_size=3, sequence_length=3, dimension=64):

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
                series_data = measurements[index].get_data()
                #print(classes_dict[measurements[index].label])
                series_labels = np.ones(shape=(series_data.shape[0], 1), dtype=int) * classes_dict[measurements[index].label]
            else:
                series_data = np.ones(shape=(1, dimension), dtype=float) * masking_value
                series_labels = np.ones(shape=(1, 1), dtype=int) * masking_value

            if series_data.shape[0] > max_len:
                max_len = series_data.shape[0]
            batch_list.append(series_data)
            batch_list_labels.append(series_labels)

        batch = np.ones(shape=(batch_size, max_len, dimension), dtype=float) * masking_value
        batch_labels = np.ones(shape=(batch_size, max_len, 1), dtype=int) * masking_value

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
            ba_labels = np.append(batches_labels[i], np.ones(shape=(batch_size, padding_length, 1), dtype=int) * masking_value, axis=1)
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
    print(type(batches_labels_done))
    print(batches_labels_done)

    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(batches_data), tf.constant(batches_labels)))

    return dataset, starting_indices

def make_model(input_shape, dim_hidden, num_classes, masking_value):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Masking(mask_value=masking_value,
                                      batch_input_shape=input_shape))
    model.add(tf.keras.layers.LSTM(dim_hidden, return_sequences=True, stateful=True))
    model.add(tf.keras.layers.Dense(num_classes))

    return model

def make_classes_dict(classes):
    classes_dict = {}
    for i, c in enumerate(classes):
        classes_dict[c] = i
    return classes_dict

def get_class(c, class_dict):
    return list(class_dict.keys())[list(class_dict.values()).index(c)]

def loss(model, X, y, training=False):
    y_pred = model(X, training=training)
    return loss_object(y_true=y, y_pred=y_pred)

def grad(model, X, y):
    with tf.GradientTape() as tape:
        loss_value = loss(model, X, y, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


batch_size = 1
sequence_length = 1
dim = 64

masking_value = 10.
learning_rate = 0.0005
dim_hidden = 16

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

measurements = get_measurements_from_dir()
measurements_train, measurements_val = train_test_split(measurements)

classes_list = np.unique([m.label for m in measurements])
classes_dict = make_classes_dict(classes_list)


input_shape = (batch_size, sequence_length, dim)

print(classes_list)
print(classes_dict)
#print(classes_dict['cream_cheese'], get_class(5, classes_dict))

lstm = make_model(input_shape, dim_hidden, classes_list.size, masking_value)

num_epochs = 1


steps = 0

for e in range(num_epochs):

    measurements_train = shuffle(measurements_train)
    data_train, starting_indices_train = get_batched_data(measurements_train, masking_value, batch_size=batch_size,
                                                          sequence_length=sequence_length, dimension=dim)

    counter = 0

    for X, y, in data_train:
        if counter in starting_indices_train:
            lstm.reset_states()

        l, grads = grad(lstm, X, y)
        optimizer.apply_gradients(zip(grads, lstm.trainable_variables))
        counter += 1
        steps += 1
        print(counter)

    measurements_val = shuffle(measurements_val)
    data_val, starting_indices_val = get_batched_data(measurements_val, masking_value, batch_size=batch_size,
                                                      sequence_length=sequence_length, dimension=dim)
    counter_val = 0

    for X, y in data_val:
        if counter_val in starting_indices_val:
            lstm.reset_states()

        loss(lstm, X, y, training=False)
        counter_val +=1

