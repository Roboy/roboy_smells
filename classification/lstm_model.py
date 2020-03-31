import numpy as np
import tensorflow as tf

from e_nose.measurements import DataType
from e_nose.measurements import Measurement

class SmelLSTM:

    def __init__(self, input_shape: tuple, num_classes: int = 6, masking_value: float = 100., return_sequences: bool = True, simple_model: bool = True,
                 stateful: bool = True,
                 dim_hidden: int = 6, data_type: DataType = DataType.HIGH_PASS, LSTM: bool = True,
                 classes_list: list = ['acetone', 'isopropanol', 'orange_juice', 'pinot_noir', 'raisin', 'wodka']):
        """
        Class for a classifier based on recurrent neural networks defining the architecture, prediction functions and more.
        Our best performing model architecture consists in a stateful LSTM layer followed by a fully connected layer
        that returns logits for the different classes at each time step, which we call SmelLSTM.
        However, also other models are possible, for instance simple RNN, more FC layers, stateless model, sequence classifying models and more.

        :param input_shape:         Shape of input data.
        :param num_classes:         Number of different classes.
        :param masking_value:       Value of input data to be ignored.
        :param return_sequences:    If set to True, model will output one prediction per time step, otherwise one prediction for one sequence.
        :param simple_model:        Specifies whether to use simple model with one hidden FC layer after recurrent layer or deeper model.
        :param stateful:            Specifies whether to use a stateful recurrent model.
        :param dim_hidden:          Number of neurons in hidden layer for simple model.
                                    For the deeper model the number of neurons for all hidden layers will also be obtained from this value.
        :param data_type:           Type of data preprocessing.
        :param LSTM:                Specifies whether LSTM or simple RNN will be used.
        :param classes_list:        List of classes to be learnt by model.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.masking_value = masking_value
        self.return_sequences = return_sequences
        self.simple_model = simple_model
        self.stateful = stateful
        self.dim_hidden = dim_hidden
        self.batch_size = input_shape[0]
        self.sequence_length = input_shape[1]
        self.dimension = input_shape[2]
        self.data_type = data_type
        self.classes_list = classes_list
        self.LSTM = LSTM

        if self.simple_model:
            self.model = self.make_model()
        else:
            self.model = self.make_model_deeper()

    def load_weights(self, model_name: str, checkpoint: int, path: str = './models/'):
        """
        Loads the model weights for given model_name and checkpoint.

        :param model_name:          Name of weights set.
        :param checkpoint:          Training checkpoint equivalent to epoch at which weights are extracted.
        :param path:                Path to model directories.
        """
        path_to_model = path + model_name + '/checkpoint_' + str(checkpoint) + '/model_weights'
        self.model.load_weights(path_to_model)

    def make_model(self) -> tf.keras.Model:
        """
        Defines RNN or LSTM model with following simple architecture:
        Masking layer --> Recurrent layer [output: dim_hidden] --> FC layer [output: num_classes]

        :return:                    Model.
        """
        from tensorflow import keras

        model = keras.models.Sequential()
        model.add(keras.layers.Masking(mask_value=self.masking_value,
                                       batch_input_shape=self.input_shape))
        if self.LSTM:
            model.add(keras.layers.LSTM(self.dim_hidden, return_sequences=self.return_sequences, stateful=self.stateful))
        else:
            model.add(keras.layers.SimpleRNN(self.dim_hidden, return_sequences=self.return_sequences, stateful=self.stateful))

        if self.return_sequences:
            model.add(keras.layers.TimeDistributed(keras.layers.Dense(self.num_classes)))
        else:
            model.add(keras.layers.Dense(self.num_classes))

        return model

    def make_model_deeper(self) -> tf.keras.Model:
        """
        Defines RNN or LSTM model with following deeper architecture:
        Masking layer --> Recurrent layer [output: dim_hidden] --> FC layer [output: dim_hidden/2] --> FC layer [output: num_classes]

        :return:                    Model.
        """
        from tensorflow import keras
        
        hidden_dim_1 = self.dim_hidden
        hidden_dim_2 = int(hidden_dim_1 * 0.5)

        model = keras.models.Sequential()
        model.add(keras.layers.Masking(mask_value=self.masking_value,
                                       batch_input_shape=self.input_shape))

        if self.LSTM:
            model.add(keras.layers.LSTM(hidden_dim_1, return_sequences=self.return_sequences, stateful=self.stateful))
        else:
            model.add(keras.layers.SimpleRNN(hidden_dim_1, return_sequences=self.return_sequences, stateful=self.stateful))

        if self.return_sequences:
            model.add(keras.layers.TimeDistributed(keras.layers.Dense(hidden_dim_2)))
            model.add(keras.layers.TimeDistributed(keras.layers.Dense(self.num_classes)))
        else:
            model.add(keras.layers.Dense(hidden_dim_2))
            model.add(keras.layers.Dense(self.num_classes))

        return model

    def predict_from_batch(self, data_batch: np.ndarray, debugging: bool = False) -> str:
        """
        Classifies given data batch and returns prediction.

        :param data_batch:          Array of data sequence to be classified. Should be of shape (1, sequence length, dimensions) or (sequence length, dimensions).
                                    For shape (dimensions) data will be treated as a sequence of length = 1.
        :param debugging:           If set to True, additional classification information will be printed.
        :return:                    Predicted class for given sequence.
        """
        if len(data_batch.shape) < 3:
            data_batch = np.expand_dims(data_batch, axis=0)
            if len(data_batch.shape) < 3:
                data_batch = np.expand_dims(data_batch, axis=0)
        if self.stateful:
            self.reset_states()
        y = self.model(data_batch, training=False)
        class_indices = np.argmax(y.numpy(), -1).flatten()
        if debugging:
            print("class_indices: ", class_indices)
        classes = [self.classes_list[c] for c in class_indices]
        counts = np.bincount(class_indices)
        class_index_most = np.argmax(counts)
        if debugging:
            print("classes: ", classes)
        prediction = self.classes_list[class_index_most]
        return prediction

    def predict_live(self, measurement: Measurement) -> object:
        """
        Classifies given measurement and returns prediction.

        :param measurement:         Measurement object (e_nose.measurements.Measurement) to be classified
        :return:                    Predicted class for given measurement.
        """
        data = measurement.get_data_as(self.data_type)
        if self.stateful:
            self.model.reset_states()
        sample = np.empty(shape=self.input_shape)
        for d in range(data.shape[0]):
            sample[0, 0, :] = data[d, :]
            y = self.model(sample, training=False)
            prediction = self.classes_list[np.argmax(y.numpy(), -1).flatten()[0]]

        return prediction

    def reset_states(self):
        """
        Resets the internal states of recurrent layer. Only used for stateful models.
        """
        self.model.reset_states()

    def summary(self):
        """
        Prints model summary.
        """
        self.model.summary()
