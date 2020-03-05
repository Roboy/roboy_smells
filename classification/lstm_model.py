from e_nose.measurements import DataType
import numpy as np

def make_model(input_shape, dim_hidden, num_classes, masking_value=100., return_sequences=True):
    from tensorflow import keras

    model = keras.models.Sequential()
    model.add(keras.layers.Masking(mask_value=masking_value,
                                      batch_input_shape=input_shape))
    model.add(keras.layers.LSTM(dim_hidden, return_sequences=return_sequences, stateful=True))

    if return_sequences:
        model.add(keras.layers.TimeDistributed(keras.layers.Dense(num_classes)))
    else:
        model.add(keras.layers.Dense(num_classes))

    return model


def make_model_deeper(input_shape, num_classes, hidden_dim_1=32, hidden_dim_2=16, dropout=0.5, masking_value=100., return_sequences=True):
    from tensorflow import keras

    model = keras.models.Sequential()
    model.add(keras.layers.Masking(mask_value=masking_value,
                                      batch_input_shape=input_shape))
    model.add(keras.layers.LSTM(hidden_dim_1, return_sequences=True, stateful=True))

    if return_sequences:
        if dropout > 0.:
            model.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.5)))
        model.add(keras.layers.TimeDistributed(keras.layers.Dense(hidden_dim_2)))
        model.add(keras.layers.TimeDistributed(keras.layers.Dense(num_classes)))
    else:
        if dropout > 0.:
            model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(hidden_dim_2))
        model.add(keras.layers.Dense(num_classes))

    return model


#def get_model_with_weights(model_name, checkpoint, path='./models/rnn/', input_dimension=42, sequence_length=1, hidden_dim=8, num_classes=6, simple_model=True):
#    path_to_model = path + model_name + '/checkpoint_' + str(checkpoint) + '/model_weights'

#    model.summary()
#    model.load_weights(path_to_model)
#    return model


class SmelLSTM:
    def __init__(self, input_shape, num_classes, masking_value=100., return_sequences=True, simple_model=True,
                 hidden_dim_simple=6, data_type=DataType.HIGH_PASS, classes_list = ['coffee_powder', 'isopropanol', 'orange_juice', 'raisin', 'red_wine', 'wodka']):

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.masking_value = masking_value
        self.return_sequences = return_sequences
        self.simple_model = simple_model
        self.hidden_dim_simple = hidden_dim_simple
        self.batch_size = input_shape[0]
        self.sequence_length = input_shape[1]
        self.dimension = input_shape[2]
        self.data_type = data_type
        self.classes_list = classes_list

        if self.simple_model:
            self.model = make_model(input_shape=self.input_shape, dim_hidden=self.hidden_dim_simple, num_classes=self.num_classes, return_sequences=self.return_sequences)
        else:
            self.model = make_model_deeper(input_shape=self.input_shape, num_classes=self.num_classes)

    def load_weights(self, model_name, checkpoint, path='./models/rnn/'):
        path_to_model = path + model_name + '/checkpoint_' + str(checkpoint) + '/model_weights'
        self.model.load_weights(path_to_model)

    def predict_live(self, measurement):
        data = measurement.get_data_as(self.data_type)
        print(data.shape)
        self.model.reset_states()
        sample = np.empty(shape=self.input_shape)
        for d in range(data.shape[0]):
            sample[0, 0, :] = data[d, :]
            y = self.model(sample, training=False)
        prediction = self.classes_list[np.argmax(y.numpy(), -1).flatten()[0]]
        return prediction

    def summary(self):
        self.model.summary()
