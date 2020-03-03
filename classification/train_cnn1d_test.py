from __future__ import absolute_import, division, print_function, unicode_literals
import os

from pathlib import Path
from e_nose import file_reader
from e_nose import data_processing as dp
from e_nose.measurements import DataType

import classification.triplet_util as tu

import argparse
from ray.tune.integration.keras import TuneReporterCallback

import numpy as np
import classification.data_loading as dl
import ray
from ray import tune

parent_path = os.getcwd()

num_triplets_train = 300
num_triplets_val = 300


def load_data():
    # Read in data
    measurements = dl.get_measurements_from_dir(os.path.join(parent_path, '../data'))
    ms_train, ms_val = dl.train_test_split(measurements, 0.7)

    train_triplets, train_labels = tu.create_triplets(ms_train, num_triplets_train)
    val_triplets, val_labels = tu.create_triplets(ms_val, num_triplets_val)

    train_batch, val_batch = tu.getInputBatchFromTriplets(train_triplets, val_triplets)
    return train_batch, val_batch


parser = argparse.ArgumentParser()
parser.add_argument(
    "--smoke-test", action="store_true", help="Finish quickly for testing")
args, _ = parser.parse_known_args()


def train_cnn(config, reporter):
    import tensorflow as tf

    ## CONFIG
    batch_size = 900

    ## LOAD DATA
    train_batch, val_batch = load_data()

    ####################
    # MODEL SETUP
    ####################
    class RecurrentLayer(tf.keras.layers.Layer):
        def __init__(self, dilation_rate=1, filter_size=64):
            super(RecurrentLayer, self).__init__()
            self.sigm_out = tf.keras.layers.Conv1D(filter_size, 2, dilation_rate=2 ** dilation_rate, padding='causal', activation='sigmoid')
            self.tanh_out = tf.keras.layers.Conv1D(filter_size, 2, dilation_rate=2 ** dilation_rate, padding='causal', activation='tanh')
            self.same_out = tf.keras.layers.Conv1D(filter_size, 1, padding='same')
        def call(self, x):
            original_x = x

            x_t = self.tanh_out(x)
            x_s = self.sigm_out(x)

            x = tf.keras.layers.Multiply()([x_t, x_s])
            x = self.same_out(x)
            x_skips = x
            x = tf.keras.layers.Add()([original_x, x])

            return x_skips, x

    class Model1DCNN(tf.keras.Model):
        def __init__(self, dilations=3, filter_size=64, input_shape=(64, 49)):
            super(Model1DCNN, self).__init__()

            self.residual = []
            self.dilations = dilations
            self.causal = tf.keras.layers.Conv1D(filter_size, 2, padding='causal', input_shape=input_shape)

            for i in range(1, dilations + 1):
                self.residual.append(RecurrentLayer(dilation_rate=i, filter_size=filter_size))

            self.same_out_1 = tf.keras.layers.Conv1D(filter_size, 1, padding='same', activation='relu')
            self.same_out_2 = tf.keras.layers.Conv1D(filter_size, 1, padding='same', activation='relu')

            self.d1 = tf.keras.layers.Dense(400, activation='relu')
            self.d2 = tf.keras.layers.Dense(200, activation='relu')
            self.d3 = tf.keras.layers.Dense(20)

        def call(self, x):
            x_skips = []

            x = self.causal(x)
            for i in range(self.dilations):
                x_skip, x = self.residual[i](x)
                x_skips.append(x_skip)

            x = tf.keras.layers.Add()(x_skips)
            x = self.same_out_1(x)
            x = self.same_out_2(x)
            x = tf.keras.layers.Flatten()(x)
            x = self.d1(x)
            x = self.d2(x)
            return self.d3(x)

    ####################
    # LOSS DEFINITION
    ####################
    def triplet_loss(self, feats):
        m = 0.01

        diff_pos = feats[0:batch_size:3] - feats[1:batch_size:3]
        diff_pos_sq = tf.math.reduce_sum(diff_pos ** 2, axis=1)
        diff_neg = feats[0:batch_size:3] - feats[2:batch_size:3]
        diff_neg_sq = tf.math.reduce_sum(diff_neg ** 2, axis=1)

        L_triplet = tf.math.reduce_sum(tf.math.maximum(0.0, 1 - diff_neg_sq / (diff_pos_sq + m)))

        L_pair = tf.math.reduce_sum(diff_pos_sq)
        return L_triplet + L_pair

    model = Model1DCNN(dilations=config["num_dilations"], filter_size=config["filter_size"])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            lr=config["lr"]
        ),
        loss=triplet_loss)

    model.fit(
        x=train_batch, y=train_batch,
        batch_size=batch_size,
        validation_data=(val_batch, val_batch),
        epochs=20,
        shuffle=False,
        callbacks=[TuneReporterCallback(reporter)],
        verbose=0)


ray.init(num_cpus=2 if args.smoke_test else None)

tune.run(
    train_cnn,
    name="cnn_wavenet_roboy",
    num_samples=1,
    config={
        "lr": tune.sample_from(lambda spec: np.random.uniform(0.001, 0.1)),
        "num_dilations": tune.grid_search([3]),
        "filter_size": tune.grid_search([64, 128])
    })
