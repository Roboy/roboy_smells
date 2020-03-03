import tensorflow as tf
import numpy as np
import classification.data_loading as dl
import os

def make_classes_dict(classes):
    classes_dict = {}
    for i, c in enumerate(classes):
        classes_dict[c] = i
    return classes_dict

def get_class(c, class_dict):
    return list(class_dict.keys())[list(class_dict.values()).index(c)]

def make_model(input_shape, dim_hidden, num_classes, masking_value):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Masking(mask_value=masking_value,
                                      batch_input_shape=input_shape, dtype=tf.float64))
    model.add(tf.keras.layers.LSTM(dim_hidden, return_sequences=True, stateful=True))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes)))

    # take this out again
    # model.add(tf.keras.layers.Dense(num_classes))

    return model

path = '/Users/max/ray_results/lstm_roboy/LSTMTrainable_4fbbf998_2_batch_size=16,dim_hidden=8,lr=0.034882_2020-03-03_20-43-532ns_sdaq/checkpoint_220/model_weights'


batch_size = 1
sequence_length = 1
dim = 49
masking_value = 100.

input_shape = (batch_size, sequence_length, dim)


data_path = os.path.join(os.getcwd(), '../data')
measurements = dl.get_measurements_from_dir(data_path)


classes_list = np.unique([m.label for m in measurements])
classes_dict = make_classes_dict(classes_list)

lstm = make_model(input_shape=input_shape, dim_hidden=8, num_classes=6, masking_value=masking_value)
lstm.load_weights(path)

print(classes_dict)

'''

np.random.shuffle(measurements)
m = measurements[0]

label_gt = m.label
data = m.get_data()
'''



for m in measurements:
    label_gt = m.label
    print('label: ', label_gt)
    data = m.get_data()

    for i in range(data.shape[0]):
        sample = np.empty(shape=(1, sequence_length, dim))
        sample[0] = data[sequence_length*i:sequence_length*(i+1), :]

        #print('label: ', label_gt)
        #print('data shape: ', data.shape)
        #print('sample shape: ', sample.shape)

        prediction = lstm(sample)
        prediction = np.argmax(prediction.numpy(), axis=-1).flatten()

        print('prediction')
        for p in prediction:
            print(classes_list[p])
        input(' ')

    lstm.reset_states()