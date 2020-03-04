def make_model(input_shape, dim_hidden, num_classes, masking_value=100.):
    from tensorflow import keras

    model = keras.model.Sequential()
    model.add(keras.layers.Masking(mask_value=masking_value,
                                      batch_input_shape=input_shape, dtype=tf.float64))
    model.add(keras.layers.LSTM(dim_hidden, return_sequences=True, stateful=True))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(num_classes)))

    return model