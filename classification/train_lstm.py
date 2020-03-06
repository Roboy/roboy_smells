import os
import argparse
import ray
from ray import tune

import numpy as np
from classification.data_loading import get_measurements_from_dir, train_test_split, shuffle, get_batched_data, get_measurements_train_test_from_dir, get_data_stateless
from classification.lstm_model import make_model, make_model_deeper
from classification.util import get_class, get_classes_list, get_classes_dict, hot_fix_label_issue

from e_nose.measurements import DataType

# STUFF
parent_path = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument(
    "--smoke-test", action="store_true", help="Finish quickly for testing")
args, _ = parser.parse_known_args()
gpus = []

dir_train_max = '../data_train'
dir_val_max = '../data_test'

# LOAD FROM FILES
#measurements = get_measurements_from_dir(os.path.join(parent_path, '../data'))
measurements_in_train, measurements_in_test, num_correct_channels = get_measurements_train_test_from_dir(os.path.join(parent_path, dir_train_max), os.path.join(parent_path, dir_val_max))
print('number of correct_channels:', num_correct_channels)

measurements_in_train = hot_fix_label_issue(measurements_in_train)
measurements_in_test = hot_fix_label_issue(measurements_in_test)

print("measurements_in_train", len(measurements_in_train))
print("measurements_in_test", len(measurements_in_test))

class LSTMTrainable(tune.Trainable):
    def _setup(self, config):
        import tensorflow as tf

        ####################
        # GENERAL CONFIG
        ####################
        # INPUT SHAPE
        self.batch_size = config["batch_size"]
        self.sequence_length = 50
        self.dim = num_correct_channels
        self.input_shape = (self.batch_size, self.sequence_length, self.dim)

        # OTHER STUFF
        self.masking_value = 100.
        #self.classes_list = get_classes_list(measurements)
        self.classes_list = get_classes_list(measurements_in_train)
        self.classes_dict = get_classes_dict(self.classes_list)
        self.num_classes = self.classes_list.size
        self.return_sequences = config["return_sequences"]
        self.stateful = config["stateful"]
        self.use_lstm = config["use_lstm"]


        ####################
        # LOAD DATA
        ####################
        #self.measurements_train, self.measurements_val = train_test_split(measurements)
        self.measurements_train = measurements_in_train
        self.measurements_val = measurements_in_test
        self.num_measurements_train = len(measurements_in_train)

        if config["data_preprocessing"] is "full":
            self.data_type = DataType.FULL
        else:
            self.data_type = DataType.HIGH_PASS

        if not self.stateful:
            self.tf_dataset_train = get_data_stateless(self.measurements_train, dimension=self.dim,
                                                       sequence_length=self.sequence_length,
                                                       return_sequences=self.return_sequences,
                                                       data_type=self.data_type)
            self.tf_dataset_val = get_data_stateless(self.measurements_val, dimension=self.dim,
                                                     sequence_length=self.sequence_length,
                                                     return_sequences=self.return_sequences,
                                                     data_type=self.data_type)

        ####################
        # MODEL SETUP
        ####################


        if config["dim_hidden"] == 1000:
            self.model = make_model_deeper(input_shape=self.input_shape, num_classes=self.num_classes, masking_value=self.masking_value, return_sequences=self.return_sequences, stateful=self.stateful, LSTM=self.use_lstm)
        else:
            self.model = make_model(input_shape=self.input_shape, dim_hidden=config["dim_hidden"], num_classes=self.num_classes, masking_value=self.masking_value, return_sequences=self.return_sequences, stateful=self.stateful)
        self.model.summary()

        ####################
        # OPTIMIZER
        ####################
        self.optimizer = tf.keras.optimizers.Adam(lr=config["lr"])

        ####################
        # LOSS DEFINITION
        ####################
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.val_loss = tf.keras.metrics.Mean(name="val_loss")
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="train_accuracy")
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="val_accuracy")

        @tf.function
        def compute_masked_loss(y_true, y_pred, mask):
            loss = self.loss_object(y_true=y_true, y_pred=y_pred)
            masked_loss = tf.reduce_mean(tf.boolean_mask(loss, mask))
            return masked_loss

        @tf.function
        def train_step(X, y, mask):
            with tf.GradientTape() as tape:
                y_pred = self.model(X, training=True)
                loss_value = compute_masked_loss(y_true=y, y_pred=y_pred, mask=mask)
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            self.train_loss(loss_value)
            self.train_accuracy(y, y_pred, mask)
            #print('class_dict: ', self.classes_dict)
            #print('y_gt: ')
            #print(y.numpy())
            #print('y_pred: ')
            #print(np.argmax(y_pred.numpy(), axis=-1))

        @tf.function
        def val_step(X, y, mask):
            y_pred = self.model(X, training=False)
            loss_value = compute_masked_loss(y_true=y, y_pred=y_pred, mask=mask)
            self.val_loss(loss_value)
            self.val_accuracy(y, y_pred, mask)

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
        import tensorflow as tf

        self.debugging_tool = 0

        self.train_loss.reset_states()
        self.val_loss.reset_states()
        self.train_accuracy.reset_states()
        self.val_accuracy.reset_states()

        if self.stateful:

            self.measurements_train, self.measurements_val = shuffle(self.measurements_train, self.measurements_val)

            data_train, labels_train, starting_indices_train = get_batched_data(self.measurements_train,
                                                                            classes_dict=self.classes_dict,
                                                                            masking_value=self.masking_value,
                                                                            data_type=self.data_type,
                                                                            batch_size=self.batch_size,
                                                                            sequence_length=self.sequence_length,
                                                                            dimension=self.dim,
                                                                            return_sequences=self.return_sequences)

            data_val, labels_val, starting_indices_val = get_batched_data(self.measurements_val,
                                                                      classes_dict=self.classes_dict,
                                                                      masking_value=self.masking_value,
                                                                      data_type=self.data_type,
                                                                      batch_size=self.batch_size,
                                                                      sequence_length=self.sequence_length,
                                                                      dimension=self.dim,
                                                                      return_sequences=self.return_sequences)

            self.train_ds = tf.data.Dataset.from_tensor_slices((tf.constant(data_train), tf.constant(labels_train)))
            self.val_ds = tf.data.Dataset.from_tensor_slices((tf.constant(data_val), tf.constant(labels_val)))

        else:

            self.train_ds = self.tf_dataset_train.shuffle(buffer_size=self.num_measurements_train).batch(self.batch_size)
            self.val_ds = self.tf_dataset_val.shuffle(buffer_size=self.num_measurements_train).batch(self.batch_size)

        for idx, (X, y) in enumerate(self.train_ds):

            if self.stateful:
                if idx in starting_indices_train:
                    self.model.reset_states()
                    #print('reset', idx)
            mask = self.model.layers[0](X)._keras_mask
            self.tf_train_step(X, y, mask)

        for idx, (X, y) in enumerate(self.val_ds):
            if self.stateful:
                if idx in starting_indices_val:
                    self.model.reset_states()
            mask = self.model.layers[0](X)._keras_mask
            self.tf_val_step(X, y, mask)

        # It is important to return tf.Tensors as numpy objects.
        return {
            "epoch": self.iteration,
            "loss": self.train_loss.result().numpy(),
            "test_loss": self.val_loss.result().numpy(),
            "train_acc": self.train_accuracy.result().numpy(),
            "test_acc": self.val_accuracy.result().numpy(),
            #"classes_dict": self.classes_dict
        }


ray.init(num_cpus=12 if args.smoke_test else None)
tune.run(
    LSTMTrainable,
    stop={"training_iteration": 5 if args.smoke_test else 300},
    verbose=1,
    name="lstm_roboy_friday_5",
    num_samples=8,
    checkpoint_freq=20,
    checkpoint_at_end=True,
    config={
        "lr": tune.sample_from(lambda spec: np.random.uniform(0.0001, 0.05)),
        "batch_size": tune.grid_search([64, 128]),
        "dim_hidden": tune.grid_search([6, 12, 50]),
        "return_sequences": tune.grid_search([True]),
        "data_preprocessing": tune.grid_search(["high_pass"]),
        "stateful": tune.grid_search([False]),
        "use_lstm": tune.grid_search([False, True])
    })
