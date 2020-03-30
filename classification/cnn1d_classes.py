from tensorflow.keras.layers import Conv1D, Flatten, Add, Dense, Layer, Multiply
from tensorflow.keras import Model
import classification.data_loading as dl
import numpy as np

"""
This file describes the model used for the 1dCNN with the cross entropy loss and directly predicts classes
"""
def load_data(path: str):
    """
    Loads the data from the specified path in the correct format
    :param path: path to data
    :return:
    """
    # Read in data
    measurements = dl.get_measurements_from_dir(path)
    ms_train, ms_val = dl.train_test_split(measurements, 0.8)

    #    train_triplets, train_labels = tu.create_triplets(ms_train, num_triplets_train)
    #    val_triplets, val_labels = tu.create_triplets(ms_val, num_triplets_val)

    #    train_batch, val_batch = tu.getInputBatchFromTriplets(train_triplets, val_triplets)
    xt = np.zeros((0, 32, 49))
    yt = np.zeros((0))
    xv = np.zeros((0, 32, 49))
    yv = np.zeros((0))

    classes_list = np.unique([m.label for m in measurements])

    for m in ms_train:
        for i in range(m.get_data().shape[0] - 33):
            xt = np.vstack((xt, np.expand_dims(m.get_data()[i:i+32, :], axis=0)))
            yt = np.append(yt, np.argwhere(classes_list == m.label))

    for m in ms_val:
        for i in range(m.get_data().shape[0] - 33):
            xv = np.vstack((xv, np.expand_dims(m.get_data()[i:i + 32, :], axis=0)))
            yv = np.append(yv, np.argwhere(classes_list == m.label))

    return xt, np.expand_dims(yt, axis=1).astype(int), xv, np.expand_dims(yv, axis=1).astype(int), classes_list.shape[0]

####################
# MODEL SETUP
####################
class RecurrentLayer(Layer):
    """
    The recurrent layer of WaveNet
    """
    def __init__(self, dilation_rate=1, filter_size=64):
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
    def __init__(self, num_classes: int, dilations: int = 3, filter_size: int =64, input_shape: tuple=(32, 49)):
        """
        :param num_classes: number of classes
        :param dilations: number of dilations ("hidden" layers in the recurrent architecture)
        :param filter_size: filter size of the CNN
        :param input_shape: input shape of the network
        """
        super(Model1DCNN, self).__init__()

        self.residual = []
        self.dilations = dilations
        self.causal = Conv1D(filter_size, 2, padding='causal', input_shape=input_shape)

        for i in range(1, dilations + 1):
            self.residual.append(RecurrentLayer(dilation_rate=i, filter_size=filter_size))

        self.same_out_1 = Conv1D(filter_size, 1, padding='same', activation='relu')
        self.same_out_2 = Conv1D(8, 1, padding='same', activation='relu')

        self.d2 = Dense(100, activation='relu')
        self.d3 = Dense(num_classes, activation='softmax')

    def call(self, x):
        """
        This method is called during the forward pass of the network.
        :param x: input to the network
        :return: output of the network (classes)
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
        x = self.d2(x)
        return self.d3(x)
