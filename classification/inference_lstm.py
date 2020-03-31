import os

import classification.data_loading as dl
from classification.lstm_model import SmelLSTM
from classification.util import get_classes_list
from e_nose.measurements import DataType

# specify model to test
model_name = 'LSTMTrainable_15750966_1740_batch_size=128,dim_hidden=6,lr=0.004831,return_sequences=True_2020-03-05_08-08-45fs4p25pg'
checkpoint = 200
path = './models/rnn/'

# model configuration
# get this from model_name
dim_hidden = 6

# configure input data
batch_size = 1 # Does not need to match batch_size in in model_name. Set this to 1 in order to analyse one sample at a time.
sequence_length = 45
# If return_sequences = True (which is usually the case), specifiying the exact sequence length is not necessary to load the model.
# However, to obtain good results the sequence to predict should be similiar to the training sequences (similiar starting point and sequence length).
dim = 34 # Dimensions of input data (the number of channels model was trained on). Must equal the number of working channels of data to test on.
input_shape = (batch_size, sequence_length, dim)
masking_value = 100. # Values to be ignored.

# data loading
data_path_train = os.path.join(os.getcwd(), '../data_train')
data_path_test = os.path.join(os.getcwd(), '../data_test')
measurements_tr, measurements_te, correct_func = dl.get_measurements_train_test_from_dir(data_path_train, data_path_test)
print('correct_func', correct_func)

classes_list = get_classes_list(measurements_te)
print(classes_list)

lstm = SmelLSTM(input_shape=input_shape, dim_hidden=dim_hidden, num_classes=len(classes_list), masking_value=masking_value)
lstm.summary()

lstm.load_weights(model_name=model_name, checkpoint=checkpoint, path=path)


counter = 0
counter_correct = 0

# loop over measurements to be classified
for m in measurements_te:
    if m.label == 'raisin' or m.label == 'coffee_powder':
        continue
    #max_length = m.data.shape[0]
    print('Ground Truth: ', m.label)
    data = m.get_data_as(DataType.HIGH_PASS)[:sequence_length]
    print(data.shape)

    prediction = lstm.predict_from_batch(data)
    lstm.reset_states()
    print('prediction: ', prediction)
    #input(' ')

    if prediction == m.label:
        counter_correct += 1
    counter += 1

print('accuracy: ', counter_correct/counter)