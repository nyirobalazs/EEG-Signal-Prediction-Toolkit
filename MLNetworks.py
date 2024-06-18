import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf
# Suppress TensorFlow logging - not showing infos
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Conv1D, Flatten, MultiHeadAttention, LayerNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


def load_model(model_name, input_shape, output_layer_dim=1, dropout_rate=0.2):
    """
    Load the model based on the model name.

    Parameters
    ----------
        model_name : str
            The name of the model to load.
        input_shape : tuple
            The input shape of the model.
        output_layer_dim : int, optional
            The output layer dimension.
        dropout_rate : float, optional
            The dropout rate.

    Returns
    -------
        model : tensorflow.keras.Model
            The loaded model.
    """
    if model_name == 'LSTM':
        return LSTMNetwork(input_shape, dropout_rate, output_layer_dim).build_model()
    elif model_name == 'BiLSTM':
        return BidirectionalLSTMNetwork(input_shape, dropout_rate, output_layer_dim).build_model()
    elif model_name == 'Hybrid':
        return HybridNetwork(input_shape, output_layer_dim).build_model()
    else:
        raise ValueError(f"[ERROR] Model {model_name} is not defined.")

class LSTMNetwork:
    def __init__(self, input_shape, dropout_rate=0.2, output_layer_dim=1):
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.output_layer_dim = output_layer_dim

        self.model = self.build_model()

    def build_model(self):

        model = Sequential()

        model.add(LSTM(100, return_sequences=True, input_shape=(self.input_shape[0], self.input_shape[1]), kernel_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(LSTM(100, return_sequences=True, kernel_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(LSTM(50, return_sequences=False, kernel_regularizer=l2(0.001)))
        model.add(Dense(self.output_layer_dim))

        return model

class BidirectionalLSTMNetwork:
    def __init__(self, input_shape, dropout_rate=0.2, output_layer_dim=1):
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.output_layer_dim = output_layer_dim

        self.model = self.build_model()

    def build_model(self):

        model = Sequential()

        # Input layer
        model.add(Input(shape=(self.input_shape[0], self.input_shape[1])))

        # First Bidirectional LSTM layer with ReLU activation
        model.add(Bidirectional(LSTM(100, return_sequences=True)))
        model.add(Activation('relu'))
        model.add(Dropout(self.dropout_rate))

        # Second Bidirectional LSTM layer with ReLU activation
        model.add(Bidirectional(LSTM(50, return_sequences=True)))
        model.add(Activation('relu'))
        model.add(Dropout(self.dropout_rate))

        # Third Bidirectional LSTM layer with ReLU activation
        model.add(Bidirectional(LSTM(25, return_sequences=False)))
        model.add(Activation('relu'))
        model.add(Dropout(self.dropout_rate))

        # Output layer
        model.add(Dense(self.output_layer_dim))

        # model = Sequential()
        # model.add(LSTM(50, activation='relu', input_shape=self.input_shape, return_sequences=True))
        # model.add(Dense(self.output_layer_dim))

        return model


class HybridNetwork:
    def __init__(self, input_shape, output_layer_dim=1):
        self.input_shape = input_shape
        self.output_layer_dim = output_layer_dim

        self.model = self.build_model()

    def build_model(self):
        # Input layer
        input_layer = Input(shape=(self.input_shape[0], self.input_shape[1]))

        # TCN block
        tcn_block = Conv1D(filters=64, kernel_size=3, padding='causal', activation='relu')(input_layer)
        tcn_block = Flatten()(tcn_block)

        # Transformer-like block
        transformer_block = MultiHeadAttention(num_heads=1, key_dim=32)(input_layer, input_layer, input_layer)
        transformer_block = LayerNormalization(epsilon=1e-6)(transformer_block)

        # Concatenate TCN and Transformer blocks
        transformer_block = tf.squeeze(transformer_block, axis=1)
        concatenated = tf.concat([tcn_block, transformer_block], axis=-1)

        # Dense layers
        dense_layer = Dense(units=64, activation='relu')(concatenated)
        output_layer = Dense(self.output_layer_dim)(dense_layer)

        model = Model(inputs=input_layer, outputs=output_layer)

        return model