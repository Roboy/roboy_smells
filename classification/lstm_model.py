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


def make_model_deeper(input_shape, num_classes, hidden_dim_1=32, hidden_dim_2=16, dropout=0.5, masking_value=100., return_sequences=True):
    from tensorflow import keras

    model = keras.models.Sequential()
    model.add(keras.layers.Masking(mask_value=masking_value,
                                      batch_input_shape=input_shape))
    model.add(keras.layers.LSTM(hidden_dim_1, return_sequences=True, stateful=True))

    if return_sequences:
        if dropout > 0.:
            model.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.5)))
        model.add(keras.layers.TimeDistributed(keras.layers.Dense(hidden_dim_2)))
        model.add(keras.layers.TimeDistributed(keras.layers.Dense(num_classes)))
    else:
        if dropout > 0.:
            model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(hidden_dim_2))
        model.add(keras.layers.Dense(num_classes))

    return model