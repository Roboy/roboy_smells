import os
from tensorflow.keras.layers import Conv1D, Flatten, Add, Dense, Layer, Multiply
from tensorflow.keras import Model
import numpy as np
import classification.triplet_util as tu
import classification.data_loading as dl

"""
This file describes the model used for the 1dCNN with the triplet loss and outputs predictions in a high dimensional 
latent space.
It is similar to the model defined in cnn1d_latent.py but is declared in a way that results in much fewer parameters.
"""
def load_data(path: str, num_triplets_train: int = 300, num_triplets_val: int = 300) -> (np.ndarray, np.ndarray):
    """
    Loads the data from the specified path in the correct format
    :param num_triplets_train: number of triplets in the train data set
    :param num_triplets_val: number of triplets in the val data set
    :param path: path to data
    :return: train_batch and validation_batch for the training
    """
    # Read in data
    measurements = dl.get_measurements_from_dir(path)
    ms_train, ms_val = dl.train_test_split(measurements, 0.7)

    train_triplets, train_labels = tu.create_triplets(ms_train, num_triplets_train)
    val_triplets, val_labels = tu.create_triplets(ms_val, num_triplets_val)

    train_batch, val_batch = tu.getInputBatchFromTriplets(train_triplets, val_triplets)
    return train_batch, val_batch

####################
# MODEL SETUP
####################
class RecurrentLayer(Layer):
    """
    The recurrent layer of WaveNet
    """
    def __init__(self, dilation_rate:int=1, filter_size:int=64):
        """
        :param dilation_rate: dilation_rate for the recurrent layer
        :param filter_size: the filter size of the CNN
        """
        super(RecurrentLayer, self).__init__()
        self.sigm_out = Conv1D(filter_size, 2, dilation_rate=2 ** dilation_rate, padding='causal', activation='sigmoid')
        self.tanh_out = Conv1D(filter_size, 2, dilation_rate=2 ** dilation_rate, padding='causal', activation='tanh')
        self.same_out = Conv1D(filter_size, 1, padding='same')

    def call(self, x):
        """
        This method is called during the forward pass of the recurrent layer.
        :param x: input to the recurrent layer
        :return: output of the recurrent layer
        """
        original_x = x

        x_t = self.tanh_out(x)
        x_s = self.sigm_out(x)

        x = Multiply()([x_t, x_s])
        x = self.same_out(x)
        x_skips = x
        x = Add()([original_x, x])

        return x_skips, x


class Model1DCNN(Model):
    """
    Defines the whole model
    """
    def __init__(self, dilations:int=3, filter_size:int=64, input_shape:tuple=(64, 49)):
        """
        :param dilations: number of dilations ("hidden" layers in the recurrent architecture)
        :param filter_size: filter size of the CNN
        :param input_shape: input shape of the network
        """
        super(Model1DCNN, self).__init__()

        self.residual = []
        self.dilations = dilations
        self.same_in = Conv1D(1, 1, padding='same', activation='relu')
        self.causal = Conv1D(filter_size, 2, padding='causal', input_shape=input_shape, activation='relu')

        for i in range(1, dilations + 1):
            self.residual.append(RecurrentLayer(dilation_rate=i, filter_size=filter_size))

        self.same_out_1 = Conv1D(filter_size, 1, padding='same', activation='relu')
        self.same_out_2 = Conv1D(8, 1, padding='same', activation='relu')

        self.d1 = Dense(400, activation='relu')
        self.d2 = Dense(200, activation='relu')
        self.d3 = Dense(20)

    def call(self, x):
        """
        This method is called during the forward pass of the network.
        :param x: input to the network
        :return: output of the network (latent space)
        """
        x_skips = []

        x = self.causal(x)
        for i in range(self.dilations):
            x_skip, x = self.residual[i](x)
            x_skips.append(x_skip)

        x = Add()(x_skips)
        x = self.same_out_1(x)
        x = self.same_out_2(x)
        x = Flatten()(x)
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)
