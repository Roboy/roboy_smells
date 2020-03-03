import argparse
import os
import numpy as np
from tensorflow.keras.layers import Conv1D, Flatten, Add, Dense, Layer, Multiply
from tensorflow.keras import Model
import classification.triplet_util as tu
import classification.data_loading as dl

from ray import tune

parent_path = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--smoke-test", action="store_true", help="Finish quickly for testing")
args, _ = parser.parse_known_args()

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

####################
# MODEL SETUP
####################
class RecurrentLayer(Layer):
    def __init__(self, dilation_rate=1, filter_size=64):
        super(RecurrentLayer, self).__init__()
        self.sigm_out = Conv1D(filter_size, 2, dilation_rate=2 ** dilation_rate, padding='causal', activation='sigmoid')
        self.tanh_out = Conv1D(filter_size, 2, dilation_rate=2 ** dilation_rate, padding='causal', activation='tanh')
        self.same_out = Conv1D(filter_size, 1, padding='same')
    def call(self, x):
        original_x = x

        x_t = self.tanh_out(x)
        x_s = self.sigm_out(x)

        x = Multiply()([x_t, x_s])
        x = self.same_out(x)
        x_skips = x
        x = Add()([original_x, x])

        return x_skips, x
class Model1DCNN(Model):
    def __init__(self, dilations=3, filter_size=64, input_shape=(64, 49)):
        super(Model1DCNN, self).__init__()

        self.residual = []
        self.dilations = dilations
        self.causal = Conv1D(filter_size, 2, padding='causal', input_shape=input_shape)

        for i in range(1, dilations + 1):
            self.residual.append(RecurrentLayer(dilation_rate=i, filter_size=filter_size))

        self.same_out_1 = Conv1D(filter_size, 1, padding='same', activation='relu')
        self.same_out_2 = Conv1D(8, 1, padding='same', activation='relu')

        self.d1 = Dense(400, activation='relu')
        self.d2 = Dense(200, activation='relu')
        self.d3 = Dense(20)

    def call(self, x):
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

class CNNTrainable(tune.Trainable):
    def _setup(self, config):
        import tensorflow as tf

        ## CONFIG
        batch_size = 900

        ## LOAD DATA
        train_batch, val_batch = load_data()

        self.train_ds = tf.data.Dataset.from_tensor_slices((train_batch)).batch(batch_size)
        self.val_ds = tf.data.Dataset.from_tensor_slices((val_batch)).batch(batch_size)

        self.model = Model1DCNN(dilations=config["num_dilations"], filter_size=config["filter_size"])
        self.optimizer = tf.keras.optimizers.Adam(lr=config["lr"])
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.val_loss = tf.keras.metrics.Mean(name="val_loss")

        @tf.function
        def triplet_loss(feats):
            m = 0.01
            diff_pos = feats[0:batch_size:3] - feats[1:batch_size:3]
            diff_pos_sq = tf.math.reduce_sum(diff_pos ** 2, axis=1)
            diff_neg = feats[0:batch_size:3] - feats[2:batch_size:3]
            diff_neg_sq = tf.math.reduce_sum(diff_neg ** 2, axis=1)

            L_triplet = tf.math.reduce_sum(tf.math.maximum(0.0, 1 - diff_neg_sq / (diff_pos_sq + m)))

            L_pair = tf.math.reduce_sum(diff_pos_sq)
            return L_triplet + L_pair

        self.loss_object = triplet_loss

        @tf.function
        def train_step(batch):
            with tf.GradientTape() as tape:
                predictions = self.model(batch)
                loss = self.loss_object(predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            self.train_loss(loss)

        @tf.function
        def val_step(batch):
            predictions = self.model(batch)
            loss = self.loss_object(predictions)

            self.val_loss(loss)


        self.tf_train_step = train_step
        self.tf_val_step = val_step

    def _save(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model_weights")
        self.model.save_weights(checkpoint_path, save_format="tf")
        return tmp_checkpoint_dir

    def _restore(self, checkpoint):
        checkpoint_path = os.path.join(checkpoint, "model_weights")
        self.model.load_weights(checkpoint_path)

    def _train(self):
        self.train_loss.reset_states()
        self.val_loss.reset_states()

        for idx, batch in enumerate(self.train_ds):
            self.tf_train_step(batch)

        for batch in self.val_ds:
            self.tf_val_step(batch)

        # It is important to return tf.Tensors as numpy objects.
        return {
            "epoch": self.iteration,
            "loss": self.train_loss.result().numpy(),
            "val_loss": self.val_loss.result().numpy(),
        }

tune.run(
    CNNTrainable,
    stop={"training_iteration": 5 if args.smoke_test else 20},
    verbose=1,
    name="cnn_wavenet_roboy",
    checkpoint_freq=10,
    checkpoint_at_end=True,
    config={
        "lr": tune.sample_from(lambda spec: np.random.uniform(0.0001, 0.001)),
        "num_dilations": tune.grid_search([3]),
        "filter_size": tune.grid_search([64, 128])
    })
