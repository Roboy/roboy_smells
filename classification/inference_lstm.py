import tensorflow as tf
import numpy as np
import classification.data_loading as dl
import os

from classification.lstm_model import make_model
from classification.util import get_classes_list, get_class, get_classes_dict, hot_fix_label_issue

from e_nose.measurements import DataType

#model_name = "LSTMTrainable_b625122c_11_batch_size=64,dim_hidden=16,lr=0.073956,return_sequences=True_2020-03-04_19-04-41c78mu_or"
#checkpoint = 150

model_name = 'LSTMTrainable_ae3ce78a_36_batch_size=32,dim_hidden=16,lr=0.0018266,return_sequences=True_2020-03-04_21-40-27rsrdzrqd'
checkpoint = 90

#model_path = '/Users/max/ray_results/lstm_roboy/'

path = './models/rnn/' + model_name + '/checkpoint_' + str(checkpoint) + '/model_weights'
#path = model_path + model_name + '/checkpoint_' + str(checkpoint) + '/model_weights'

batch_size = 1
sequence_length = 10
#dim = 49
dim = 42
input_shape = (batch_size, sequence_length, dim)

masking_value = 100.
dim_hidden = 16
return_sequences = True

data_path = os.path.join(os.getcwd(), '../data_test')
measurements = dl.get_measurements_from_dir(data_path)
measurements = hot_fix_label_issue(measurements)

#classes_list_pred = ['apple_juice', 'coffee_powder', 'orange_juice', 'raisin', 'red_Wine', 'wodka']
#classes_list_pred = ['apple_juice', 'coffee_powder', 'isopropanol', 'orange_juice', 'raisin', 'red_wine', 'wodka']
classes_list_pred = ['coffee_powder', 'isopropanol', 'orange_juice', 'raisin', 'red_wine', 'wodka']


classes_list_gt = get_classes_list(measurements)
print(classes_list_gt)
classes_dict_gt = get_classes_dict(classes_list_gt)

lstm = make_model(input_shape=input_shape, dim_hidden=dim_hidden, num_classes=len(classes_list_pred), masking_value=masking_value, return_sequences=return_sequences)
lstm.summary()
lstm.load_weights(path)

#print(classes_dict)

'''

np.random.shuffle(measurements)
m = measurements[0]

label_gt = m.label
data = m.get_data()
'''

lower_threshold = 0
upper_threshold = 1000
counter = 0
counter_correct = 0

lstm.reset_states()
for m in measurements:
    label_gt = m.label
    if label_gt not in classes_list_pred or label_gt == 'coffee_powder' or label_gt == 'raisin':
        continue
    print('label: ', label_gt)
    data = m.get_data_as(DataType.FULL)

    end = int(data.shape[0] / sequence_length)
    for i in range(end):

        sample = np.empty(shape=(1, sequence_length, dim))
        sample[0] = data[sequence_length*i:sequence_length*(i+1), :]

        #print('label: ', label_gt)
        #print('data shape: ', data.shape)
        #print('sample shape: ', sample.shape)

        prediction = lstm(sample)
        prediction = np.argmax(prediction.numpy(), axis=-1).flatten()

        #print('i', i)
        for j, p in enumerate(prediction):
            #print(classes_list_pred[p])
            if (i*sequence_length + j < lower_threshold) or (i*sequence_length + j > upper_threshold):
                continue
            if classes_list_pred[p].lower() == label_gt.lower():
                counter_correct += 1
            counter += 1
        #input(' ')

    lstm.reset_states()

accuracy = counter_correct/counter
print('accuracy:', accuracy)