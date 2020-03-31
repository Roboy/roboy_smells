import os
import argparse
import ray
from ray import tune
import numpy as np

from classification.data_loading import shuffle, get_batched_data, get_measurements_train_test_from_dir, get_data_stateless
from classification.lstm_model import SmelLSTM
from classification.util import get_classes_list, get_classes_dict
from e_nose.measurements import DataType

####################
# TEST SETUP
####################
parser = argparse.ArgumentParser()
parser.add_argument(
    "--smoke-test", action="store_true", help="Finish quickly for testing")
args, _ = parser.parse_known_args()
gpus = []

####################
# LOAD MEASUREMENTS
####################
current_path = os.getcwd()
dir_train = '../data_train' # specify this
dir_val = '../data_test' # specify this
measurements_in_train, measurements_in_val, num_correct_channels = get_measurements_train_test_from_dir(os.path.join(current_path, dir_train), os.path.join(current_path, dir_val))

class RecurrentModelTrainable(tune.Trainable):
    """
    This class implements the automated training and testing of recurrent neural network models with hyperparameter search.
    For further information about the ray.tune.Trainable class and its hyperparameter search we refer to the documentation of the ray library.
    """
    def _setup(self, config: dict):
        """
        Sets up everything needed for training and hyperparameter search.

        :param config:              Configuration that specifies the hyperparameter search space.
        """
        import tensorflow as tf

        ####################
        # GENERAL CONFIG
        ####################
        # INPUT DATA
        self.batch_size = config["batch_size"]
        self.sequence_length = 45
        self.dim = num_correct_channels
        self.input_shape = (self.batch_size, self.sequence_length, self.dim)

        # OTHER STUFF
        self.masking_value = 100.
        self.classes_list = get_classes_list(measurements_in_train)
        self.classes_dict = get_classes_dict(self.classes_list)
        self.num_classes = self.classes_list.size

        self.return_sequences = config["return_sequences"]

        self.stateful = True

        self.use_lstm = config["use_lstm"]


        ####################
        # LOAD DATA
        ####################
        self.measurements_train = measurements_in_train
        self.measurements_val = measurements_in_val
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
        if config["dim_hidden"] == 100:
            model = SmelLSTM(input_shape=self.input_shape, dim_hidden=config["dim_hidden"], simple_model=False, num_classes=self.num_classes, masking_value=self.masking_value, return_sequences=self.return_sequences, stateful=self.stateful, LSTM=self.use_lstm)
        else:
            model = SmelLSTM(input_shape=self.input_shape, dim_hidden=config["dim_hidden"], simple_model=True, num_classes=self.num_classes, masking_value=self.masking_value, return_sequences=self.return_sequences, stateful=self.stateful, LSTM=self.use_lstm)
        self.model = model.model
        self.model.summary()

        ####################
        # OPTIMIZER
        ####################
        self.optimizer = tf.keras.optimizers.Adam(lr=config["lr"])

        ####################
        # LOSS AND ACCURACY
        ####################
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.val_loss = tf.keras.metrics.Mean(name="val_loss")
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="train_accuracy")
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="val_accuracy")

        @tf.function
        def compute_masked_loss(y_true: tf.Tensor, y_pred: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
            """
            Computes masked loss that ignores entries containing padding values.

            :param y_true:          Tensor contaning labels.
            :param y_pred:          Tensor containing predictions.
            :param mask:            Boolean tensor of shape (batch_size, sequence_length) that specifies values to be ignored.
            :return:                Masked loss.
            """
            loss = self.loss_object(y_true=y_true, y_pred=y_pred)
            masked_loss = tf.reduce_mean(tf.boolean_mask(loss, mask))
            return masked_loss

        @tf.function
        def train_step(X: tf.Tensor, y: tf.Tensor, mask: tf.Tensor):
            """
            Performs one train step on batch. Computes loss and accuracy and performs gradient update.

            :param X:               Data tensor of shape (batch_size, sequence_length, dimensions).
            :param y:               Label tensor of shape (batch_size, sequence_length, 1) if return_sequences = True or
                                    (batch_size, 1) for return_sequences = False.
            :param mask:            Boolean tensor of shape (batch_size, sequence_length) that specifies values to be ignored.
            """
            with tf.GradientTape() as tape:
                y_pred = self.model(X, training=True)
                loss_value = compute_masked_loss(y_true=y, y_pred=y_pred, mask=mask)
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            self.train_loss(loss_value)
            self.train_accuracy(y, y_pred, mask)

        @tf.function
        def val_step(X: tf.Tensor, y: tf.Tensor, mask: tf.Tensor):
            """
            Performs one validation step on batch. Computes loss and accuracy.

            :param X:               Data tensor of shape (batch_size, sequence_length, dimensions).
            :param y:               Label tensor of shape (batch_size, sequence_length, 1) if return_sequences = True or
                                    (batch_size, 1) for return_sequences = False.
            :param mask:            Boolean tensor of shape (batch_size, sequence_length) that specifies values to be ignored.
            """
            y_pred = self.model(X, training=False)
            loss_value = compute_masked_loss(y_true=y, y_pred=y_pred, mask=mask)
            self.val_loss(loss_value)
            self.val_accuracy(y, y_pred, mask)

        self.tf_train_step = train_step
        self.tf_val_step = val_step

    def _save(self, tmp_checkpoint_dir: str) -> str:
        """
        Internal tune function to save model at checkpoint .

        :param tmp_checkpoint_dir:  Checkpoint directory.
        :return:                    Checkpoint directory.
        """
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model_weights")
        self.model.save_weights(checkpoint_path, save_format="tf")
        return tmp_checkpoint_dir

    def _restore(self, checkpoint: str):
        """
        Internal tune function to restore model from checkpoint.

        :param checkpoint:          Checkpoint directory.
        """
        checkpoint_path = os.path.join(checkpoint, "model_weights")
        self.model.load_weights(checkpoint_path)

    def _train(self) -> dict:
        """
        Internal tune train function. Called every epoch.

        :return:                    Dictionary containing information about training progress.
        """
        import tensorflow as tf

        # Reset states of tracked losses and accuracies.
        self.train_loss.reset_states()
        self.val_loss.reset_states()
        self.train_accuracy.reset_states()
        self.val_accuracy.reset_states()

        # Reload and shuffle data.
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

        # Training loop.
        for idx, (X, y) in enumerate(self.train_ds):
            if self.stateful:
                # Perform state resets of recurrent model for stateful configuration.
                if idx in starting_indices_train:
                    self.model.reset_states()
            # Compute mask.
            mask = self.model.layers[0](X)._keras_mask
            self.tf_train_step(X, y, mask)

        # Validation loop.
        for idx, (X, y) in enumerate(self.val_ds):
            if self.stateful:
                # Perform state resets of recurrent model for stateful configuration.
                if idx in starting_indices_val:
                    self.model.reset_states()
            # Compute mask.
            mask = self.model.layers[0](X)._keras_mask
            self.tf_val_step(X, y, mask)

        # It is important to return tf.Tensors as numpy objects.
        return {
            "epoch": self.iteration,
            "loss": self.train_loss.result().numpy(),
            "test_loss": self.val_loss.result().numpy(),
            "train_acc": self.train_accuracy.result().numpy(),
            "test_acc": self.val_accuracy.result().numpy()
        }

####################
# RUN RAY TUNE
####################
ray.init(num_cpus=6 if args.smoke_test else None)
tune.run(
    RecurrentModelTrainable,
    stop={"training_iteration": 5 if args.smoke_test else 350},
    verbose=1,
    name="smellstm_roboy",
    num_samples=8,
    checkpoint_freq=20,
    config={
        "lr": tune.sample_from(lambda spec: np.random.uniform(0.0001, 0.1)),
        "batch_size": tune.grid_search([64, 128]),
        "dim_hidden": tune.grid_search([6, 12]), # If dim_hidden = 100, the deeper model architecture will be used.
        "return_sequences": tune.grid_search([True]),
        "data_preprocessing": tune.grid_search(["high_pass"]),
        "use_lstm": tune.grid_search([False, True])
    })