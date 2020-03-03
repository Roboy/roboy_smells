import numpy as np
#import tensorflow as tf
import datetime
from classification.data_loading import get_measurements_from_dir, train_test_split, shuffle


def get_batched_data(measurements, classes_dict, masking_value, batch_size=4, sequence_length=4, dimension=64):

    measurement_indices = np.arange(len(measurements))
    np.random.shuffle(measurement_indices)

    padding = batch_size-(measurement_indices.size % batch_size)
    measurement_indices = np.append(measurement_indices, np.ones(padding, dtype=int) * int(masking_value))
    measurement_indices = np.reshape(measurement_indices, (-1, batch_size))

    batches_data = []
    batches_labels = []

    for i in range(measurement_indices.shape[0]):
        batch_indices = measurement_indices[i]

        batch_list = []
        batch_list_labels = []
        max_len = 0
        for b in range(batch_size):
            index = batch_indices[b]
            #print(index)
            if index != masking_value:
                series_data = measurements[index].get_data()
                #print(classes_dict[measurements[index].label])
                series_labels = np.ones(shape=(series_data.shape[0], 1), dtype=int) * classes_dict[measurements[index].label]
            else:
                series_data = np.ones(shape=(1, dimension), dtype=float) * masking_value
                series_labels = np.ones(shape=(1, 1), dtype=int) * 0

            if series_data.shape[0] > max_len:
                max_len = series_data.shape[0]
            batch_list.append(series_data)
            batch_list_labels.append(series_labels)

        batch = np.ones(shape=(batch_size, max_len, dimension), dtype=float) * masking_value
        batch_labels = np.ones(shape=(batch_size, max_len, 1), dtype=int) * 0

        for b in range(batch_size):
            batch[b, :batch_list[b].shape[0]] = batch_list[b]
            batch_labels[b, :batch_list_labels[b].shape[0]] = batch_list_labels[b]
        batches_data.append(batch)
        batches_labels.append(batch_labels)

    for i, ba in enumerate(batches_data):
        #print("ba:", ba.shape)
        ba_labels = batches_labels[i]
        padding_length = sequence_length - (ba.shape[1] % sequence_length)
        if padding_length != sequence_length:
            ba = np.append(ba, np.ones(shape=(batch_size, padding_length, dimension), dtype=float) * masking_value, axis=1)
            ba_labels = np.append(batches_labels[i], np.ones(shape=(batch_size, padding_length, 1), dtype=int) * 0, axis=1)
        split = int(ba.shape[1] / sequence_length)

        ba = np.array(np.split(ba, split, axis=1))
        ba_labels = np.array(np.split(ba_labels, split, axis=1))

        if i == 0:
            batches_data_done = ba
            batches_labels_done = ba_labels
            starting_indices = np.array([0])
        else:
            starting_indices = np.append(starting_indices, batches_data_done.shape[0])
            batches_data_done = np.append(batches_data_done, ba, axis=0)
            batches_labels_done = np.append(batches_labels_done, ba_labels, axis=0)

    batches_labels_done = batches_labels_done.astype(int)
    print(type(batches_labels_done))

    print(batches_data_done.shape)
    print(batches_labels_done.shape)

    return batches_data_done, batches_labels_done, starting_indices

def make_classes_dict(classes):
    classes_dict = {}
    for i, c in enumerate(classes):
        classes_dict[c] = i
    return classes_dict

def get_class(c, class_dict):
    return list(class_dict.keys())[list(class_dict.values()).index(c)]

'''
def make_model(input_shape, dim_hidden, num_classes, masking_value):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Masking(mask_value=masking_value,
                                      batch_input_shape=input_shape, dtype=tf.float64))
    model.add(tf.keras.layers.LSTM(dim_hidden, return_sequences=True, stateful=True))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes)))

    # take this out again
    #model.add(tf.keras.layers.Dense(num_classes))

    return model

@tf.function
def loss(model, X, y, training=False):
    y_pred = model(X, training=training)
    return loss_object(y_true=y, y_pred=y_pred)

@tf.function
def train_step(model, X, y, optimizer):
    with tf.GradientTape() as tape:
        loss_value = loss(model, X, y, training=True)
    grads = tape.gradient(loss_value, model.trainable_variables)
    #tf.print(grads)
    optimizer.apply_gradients(zip(grads, lstm.trainable_variables))
    return loss_value


batch_size = 4
sequence_length = 4
dim = 49

masking_value = 100.
learning_rate = 0.0005
num_epochs = 100
dim_hidden = 16

optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

measurements = get_measurements_from_dir('../data_2')
measurements_train, measurements_val = train_test_split(measurements)

classes_list = np.unique([m.label for m in measurements])
classes_dict = make_classes_dict(classes_list)


input_shape = (batch_size, sequence_length, dim)

print(classes_list)
print(classes_dict)
#print(classes_dict['cream_cheese'], get_class(5, classes_dict))

lstm = make_model(input_shape, dim_hidden, classes_list.size, masking_value)



test_masking_data = np.random.randn(7, input_shape[0], input_shape[1], input_shape[2])
test_masking_data[2:5, :, :, :] = masking_value

test_masking_labels = np.ones((7, input_shape[0], input_shape[1], 1))
test_masking_labels[1] = 2
test_masking_labels[2] = 3
test_masking_labels[3] = 5

print(test_masking_data.shape)
print(test_masking_labels.shape)

dataset = tf.data.Dataset.from_tensor_slices((tf.constant(test_masking_data), tf.constant(test_masking_labels)))


for X, y in dataset:
    X_ = tf.reduce_sum(X, axis=-1)
    y_ = tf.reduce_sum(X, axis=-1)
    tf.print('X:', X_)
    tf.print('y:', y)

    weights_before = lstm.get_weights()
    l = train_step(lstm, X, y, optimizer)
    weigths_after = lstm.get_weights()
    print('train loss: ')
    tf.print(l)

    print('diff weights: ')
    tf.print(weights_before[0] - weigths_after[0])
    tf.print(weights_before[1] - weigths_after[1])
    tf.print(weights_before[2] - weigths_after[2])
    tf.print(weights_before[3] - weigths_after[3])



current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'

train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

file_path = './models/rnn/'
model_name = 'model_' + current_time + '.h5'

lstm.summary()
step = 0
min_val_loss = 999.

for e in range(num_epochs):

    measurements_train = shuffle(measurements_train)
    data_train, starting_indices_train = get_batched_data(measurements_train, masking_value, batch_size=batch_size,
                                                          sequence_length=sequence_length, dimension=dim)

    counter = 0

    for X, y, in data_train:
        
        if counter in starting_indices_train:
            lstm.reset_states()

        l = train_step(lstm, X, y, optimizer)

        with train_summary_writer.as_default():
            tf.summary.scalar('train_loss', l, step=step)

        #print('train loss epoch ', e, ' step ', step, ': ')
        #tf.print(l)
        #print(' ')
        counter += 1
        step += 1

    measurements_val = shuffle(measurements_val)
    data_val, starting_indices_val = get_batched_data(measurements_val, masking_value, batch_size=batch_size,
                                                      sequence_length=sequence_length, dimension=dim)
    counter_val = 0
    
    l_val = 0.
    for X, y in data_val:
        if counter_val in starting_indices_val:
            lstm.reset_states()

        l_val += loss(lstm, X, y, training=False)
        counter_val += 1

    l_val /= counter_val

    lstm.reset_states()
    with test_summary_writer.as_default():
        tf.summary.scalar('test_loss', l_val, step=step)

    if l_val < min_val_loss:
        min_val_loss = l_val
        lstm.save(file_path + model_name)
'''

import os
import argparse

#import tensorflow.constant
#import tensorflow.data.Dataset.from_tensor_slices

import ray
from ray import tune

parent_path = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--smoke-test", action="store_true", help="Finish quickly for testing")
args, _ = parser.parse_known_args()

class LSTMTrainable(tune.Trainable):
    def _setup(self, config):
        import tensorflow as tf

        ## CONFIG
        self.batch_size = config["batch_size"]
        self.sequence_length = 8
        self.dim = 49
        self.masking_value = 100.
        self.input_shape = (self.batch_size, self.sequence_length, self.dim)

        ## LOAD DATA
        measurements = get_measurements_from_dir(os.path.join(parent_path, '../data'))

        classes_list = np.unique([m.label for m in measurements])
        self.classes_dict = make_classes_dict(classes_list)
        self.num_classes = classes_list.size

        self.measurements_train, self.measurements_val = train_test_split(measurements)

        self.debugging_tool = False

        ####################
        # MODEL SETUP
        ####################
        def make_model(input_shape, dim_hidden, num_classes, masking_value):
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Masking(mask_value=masking_value,
                                              batch_input_shape=input_shape, dtype=tf.float64))
            model.add(tf.keras.layers.LSTM(dim_hidden, return_sequences=True, stateful=True))
            model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes)))

            # take this out again
            # model.add(tf.keras.layers.Dense(num_classes))

            return model

        self.model = make_model(input_shape=self.input_shape, dim_hidden=config["dim_hidden"], num_classes=self.num_classes, masking_value=self.masking_value)

        self.model.summary()
        print(self.classes_dict)

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

        #@tf.function
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
                                                                            dimension=self.dim)

        data_val, labels_val, starting_indices_val = get_batched_data(self.measurements_val,
                                                                      classes_dict=self.classes_dict,
                                                                      masking_value=self.masking_value,
                                                                      batch_size=self.batch_size,
                                                                      sequence_length=self.sequence_length,
                                                                      dimension=self.dim)

        self.train_ds = tf.data.Dataset.from_tensor_slices((tf.constant(data_train), tf.constant(labels_train)))
        self.val_ds = tf.data.Dataset.from_tensor_slices((tf.constant(data_val), tf.constant(labels_val)))

        for idx, (X, y) in enumerate(self.train_ds):
            if idx in starting_indices_train:
                self.model.reset_states()
                print('reset', idx)
            self.tf_train_step(X, y)

            '''
            if self.iteration == self.debugging_tool:
                print('e: ', self.iteration)
                inp = input('num to epoch')
                if inp is not None:
                    self.debugging_tool = inp
                else:
                    self.debugging_tool = self.iteration
            '''

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
            "test_acc": self.val_accuracy.result().numpy()
        }


ray.init(num_cpus=2 if args.smoke_test else None)
tune.run(
    LSTMTrainable,
    stop={"training_iteration": 5 if args.smoke_test else 500},
    verbose=1,
    name="lstm_roboy",
    num_samples=5,
    checkpoint_freq=10,
    checkpoint_at_end=True,
    config={
        "lr": tune.sample_from(lambda spec: np.random.uniform(0.001, 0.05)),
        "batch_size": tune.grid_search([1, 4, 16]),
        #"lr": tune.grid_search([0.001, 0.005, 0.01, 0.03]),
        "dim_hidden": tune.grid_search([8, 20])
        #"filter_size": tune.grid_search([64, 128])
    })

