import tensorflow as tf
import numpy as np
import classification.data_loading as dl
import os

from classification.lstm_model import make_model

def make_classes_dict(classes):
    classes_dict = {}
    for i, c in enumerate(classes):
        classes_dict[c] = i
    return classes_dict

def get_class(c, class_dict):
    return list(class_dict.keys())[list(class_dict.values()).index(c)]


path = '/Users/max/ray_results/lstm_roboy/LSTMTrainable_2017d1d6_5_batch_size=32,dim_hidden=16,lr=0.041688_2020-03-04_11-11-35kny0zxpp/checkpoint_40/model_weights'


batch_size = 1
sequence_length = 8
dim = 49
masking_value = 100.

input_shape = (batch_size, sequence_length, dim)


data_path = os.path.join(os.getcwd(), '../data_test')
measurements = dl.get_measurements_from_dir(data_path)

classes_list_pred = ['apple_juice', 'coffee_powder', 'orange_juice', 'raisin', 'red_Wine', 'wodka']

classes_list_pred = ['apple_juice', 'coffee_powder', 'isopropanol', 'orange_juice', 'raisin', 'red_Wine', 'red_wine', 'wodka']
classes_list_gt = np.unique([m.label for m in measurements])
classes_dict_gt = make_classes_dict(classes_list_gt)

lstm = make_model(input_shape=input_shape, dim_hidden=16, num_classes=len(classes_list_pred), masking_value=masking_value)
lstm.load_weights(path)

#print(classes_dict)

'''

np.random.shuffle(measurements)
m = measurements[0]

label_gt = m.label
data = m.get_data()
'''

border = None
counter = 0
counter_correct = 0

lstm.reset_states()
for m in measurements:
    label_gt = m.label
    if label_gt not in classes_list_pred:
        continue
    print('label: ', label_gt)
    data = m.get_data()

    end = int(data.shape[0] / sequence_length)
    for i in range(end):

        sample = np.empty(shape=(1, sequence_length, dim))
        sample[0] = data[sequence_length*i:sequence_length*(i+1), :]

        #print('label: ', label_gt)
        #print('data shape: ', data.shape)
        #print('sample shape: ', sample.shape)

        prediction = lstm(sample)
        prediction = np.argmax(prediction.numpy(), axis=-1).flatten()

        '''
        print('prediction')
        for p in prediction:
            print(classes_list_pred[p])
        input(' ')
        '''
        #print('i', i)
        for j, p in enumerate(prediction):
            print(classes_list_pred[p])
            if border is not None and (i*sequence_length + j < border):
                continue
            if classes_list_pred[p] == label_gt:
                counter_correct += 1
            counter += 1

    lstm.reset_states()

accuracy = counter_correct/counter
print('accuracy:', accuracy)