def make_model(input_shape, dim_hidden, num_classes, masking_value=100., return_sequences=True):
    from tensorflow import keras

    model = keras.models.Sequential()
    model.add(keras.layers.Masking(mask_value=masking_value,
                                      batch_input_shape=input_shape))
    model.add(keras.layers.LSTM(dim_hidden, return_sequences=return_sequences, stateful=True))

    if return_sequences:
        model.add(keras.layers.TimeDistributed(keras.layers.Dense(num_classes)))
    else:
        model.add(keras.layers.Dense(num_classes))

    return model


def make_model_deeper(input_shape, num_classes, masking_value=100.):
    from tensorflow import keras

    model = keras.models.Sequential()
    model.add(keras.layers.Masking(mask_value=masking_value,
                                      batch_input_shape=input_shape))
    model.add(keras.layers.LSTM(32, return_sequences=True, stateful=True))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(16)))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(num_classes)))

    return model