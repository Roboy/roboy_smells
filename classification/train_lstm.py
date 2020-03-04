import os
import argparse
import ray
from ray import tune

import numpy as np
from classification.data_loading import get_measurements_from_dir, train_test_split, shuffle, get_batched_data
from classification.lstm_model import make_model, make_model_deeper
from classification.util import get_class, get_classes_list, get_classes_dict, hot_fix_label_issue

# STUFF
parent_path = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument(
    "--smoke-test", action="store_true", help="Finish quickly for testing")
args, _ = parser.parse_known_args()
gpus = []

# LOAD FROM FILES
#measurements = get_measurements_from_dir(os.path.join(parent_path, '../data'))
measurements_in_train = get_measurements_from_dir(os.path.join(parent_path, '../data'))
measurements_in_test = get_measurements_from_dir(os.path.join(parent_path, '../data_test'))

measurements_in_train = hot_fix_label_issue(measurements_in_train)
measurements_in_test = hot_fix_label_issue(measurements_in_test)

class LSTMTrainable(tune.Trainable):
    def _setup(self, config):
        import tensorflow as tf

        ####################
        # GENERAL CONFIG
        ####################
        # INPUT SHAPE
        self.batch_size = config["batch_size"]
        self.sequence_length = 10
        self.dim = 42
        self.input_shape = (self.batch_size, self.sequence_length, self.dim)

        # OTHER STUFF
        self.masking_value = 100.
        #self.classes_list = get_classes_list(measurements)
        self.classes_list = get_classes_list(measurements_in_train)
        self.classes_dict = get_classes_dict(self.classes_list)
        self.num_classes = self.classes_list.size
        self.return_sequences = config["return_sequences"]


        ####################
        # LOAD DATA
        ####################
        #self.measurements_train, self.measurements_val = train_test_split(measurements)
        self.measurements_train, _ = train_test_split(measurements_in_train, split=1.)
        _, self.measurements_val = train_test_split(measurements_in_test, split=0.)

        ####################
        # MODEL SETUP
        ####################
        if config["dim_hidden"] == 1000:
            self.model = make_model_deeper(input_shape=self.input_shape, num_classes=self.num_classes, masking_value=self.masking_value, return_sequences=self.return_sequences)
        else:
            self.model = make_model(input_shape=self.input_shape, dim_hidden=config["dim_hidden"], num_classes=self.num_classes, masking_value=self.masking_value, return_sequences=self.return_sequences)
        self.model.summary()

        ####################
        # OPTIMIZER
        ####################
        self.optimizer = tf.keras.optimizers.Adam(lr=config["lr"])

        ####################
        # LOSS DEFINITION
        ####################
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.val_loss = tf.keras.metrics.Mean(name="val_loss")
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="train_accuracy")
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="val_accuracy")

        @tf.function
        def train_step(X, y):
            with tf.GradientTape() as tape:
                y_pred = self.model(X, training=True)
                loss_value = self.loss_object(y_true=y, y_pred=y_pred)
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            self.train_loss(loss_value)
            self.train_accuracy(y, y_pred)
            #print('class_dict: ', self.classes_dict)
            #print('y_gt: ')
            #print(y.numpy())
            #print('y_pred: ')
            #print(np.argmax(y_pred.numpy(), axis=-1))

        @tf.function
        def val_step(X, y):
            y_pred = self.model(X, training=False)
            loss_value = self.loss_object(y_true=y, y_pred=y_pred)
            self.val_loss(loss_value)
            self.val_accuracy(y, y_pred)

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
        print('classes_list:', self.classes_list)
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
                    print(e)

        self.debugging_tool = 0

        self.train_loss.reset_states()
        self.val_loss.reset_states()
        self.train_accuracy.reset_states()
        self.val_accuracy.reset_states()

        self.measurements_train, self.measurements_val = shuffle(self.measurements_train, self.measurements_val)

        data_train, labels_train, starting_indices_train = get_batched_data(self.measurements_train,
                                                                            classes_dict=self.classes_dict,
                                                                            masking_value=self.masking_value,
                                                                            batch_size=self.batch_size,
                                                                            sequence_length=self.sequence_length,
                                                                            dimension=self.dim,
                                                                            return_sequences=self.return_sequences)

        data_val, labels_val, starting_indices_val = get_batched_data(self.measurements_val,
                                                                      classes_dict=self.classes_dict,
                                                                      masking_value=self.masking_value,
                                                                      batch_size=self.batch_size,
                                                                      sequence_length=self.sequence_length,
                                                                      dimension=self.dim,
                                                                      return_sequences=self.return_sequences)

        self.train_ds = tf.data.Dataset.from_tensor_slices((tf.constant(data_train), tf.constant(labels_train)))
        self.val_ds = tf.data.Dataset.from_tensor_slices((tf.constant(data_val), tf.constant(labels_val)))

        for idx, (X, y) in enumerate(self.train_ds):
            if idx in starting_indices_train:
                self.model.reset_states()
                #print('reset', idx)
            self.tf_train_step(X, y)

        for idx, (X, y) in enumerate(self.val_ds):
            if idx in starting_indices_val:
                self.model.reset_states()
            self.tf_val_step(X, y)

        # It is important to return tf.Tensors as numpy objects.
        return {
            "epoch": self.iteration,
            "loss": self.train_loss.result().numpy(),
            "test_loss": self.val_loss.result().numpy(),
            "train_acc": self.train_accuracy.result().numpy(),
            "test_acc": self.val_accuracy.result().numpy(),
            #"classes_dict": self.classes_dict
        }


ray.init(num_cpus=2 if args.smoke_test else None)
tune.run(
    LSTMTrainable,
    stop={"training_iteration": 5 if args.smoke_test else 150},
    verbose=1,
    name="lstm_roboy",
    num_samples=8,
    checkpoint_freq=10,
    checkpoint_at_end=True,
    config={
        "lr": tune.sample_from(lambda spec: np.random.uniform(0.001, 0.08)),
        "batch_size": tune.grid_search([32, 64]),
        "dim_hidden": tune.grid_search([8, 10, 16, 1000]),
        "return_sequences": tune.grid_search([True])
    })
