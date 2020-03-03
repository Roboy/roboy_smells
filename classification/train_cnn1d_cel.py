import argparse
import os
import numpy as np
from classification.models.cnn1d_classes import load_data, Model1DCNN
from ray import tune

parent_path = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--smoke-test", action="store_true", help="Finish quickly for testing")
args, _ = parser.parse_known_args()

x_train, y_train, x_val, y_val, num_classes = load_data(os.path.join(parent_path, 'data'))

class CNNTrainable(tune.Trainable):
    def _setup(self, config):
        import tensorflow as tf

        ## CONFIG
        batch_size = 30

        self.train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
        self.val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)

        self.model = Model1DCNN(num_classes=num_classes, dilations=config["num_dilations"], filter_size=config["filter_size"])
        self.optimizer = tf.keras.optimizers.Adam(lr=config["lr"])
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.val_loss = tf.keras.metrics.Mean(name="val_loss")
        self.train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        self.val_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                predictions = self.model(x)
                loss = self.loss_object(y, predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            self.train_loss(loss)
            self.train_acc(y, predictions)

        @tf.function
        def val_step(x, y):
            predictions = self.model(x)
            loss = self.loss_object(y, predictions)

            self.val_loss(loss)
            self.val_acc(y, predictions)

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
        self.train_acc.reset_states()
        self.val_acc.reset_states()
        self.train_loss.reset_states()
        self.val_loss.reset_states()

        self.train_ds.shuffle(buffer_size=1000)
        for idx, (x,y) in enumerate(self.train_ds):
            self.tf_train_step(x, y)

        for x,y in self.val_ds:
            self.tf_val_step(x, y)

        # It is important to return tf.Tensors as numpy objects.
        return {
            "epoch": self.iteration,
            "loss_train": self.train_loss.result().numpy(),
            "loss_val": self.val_loss.result().numpy(),
            "acc_train": self.train_acc.result().numpy(),
            "acc_val": self.val_acc.result().numpy(),
        }

tune.run(
    CNNTrainable,
    stop={"training_iteration": 5 if args.smoke_test else 30},
    verbose=1,
    name="cnn_wavenet_roboy",
    checkpoint_freq=10,
    checkpoint_at_end=True,
    num_samples=1,
    config={
        "lr": tune.sample_from(lambda spec: np.random.uniform(0.0001, 0.001)),
        "num_dilations": tune.grid_search([2]),
        "filter_size": tune.grid_search([64, 128])
    })
