import argparse
import os
import numpy as np

from ray import tune
from classification.models.cnn1d_latent import load_data, Model1DCNN

parent_path = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--smoke-test", action="store_true", help="Finish quickly for testing")
args, _ = parser.parse_known_args()

train_batch, val_batch = load_data(os.path.join(parent_path, 'data'))

class CNNTrainable(tune.Trainable):
    def _setup(self, config):
        import tensorflow as tf

        ## CONFIG
        batch_size = 900

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
