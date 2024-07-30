"""
EEG Signal Prediction project.

Copyright (C) 2024 Balazs Nyiro, University of Bath

This program is free software: you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation, either version  3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
for more details.
"""

import tensorflow as tf
# Suppress TensorFlow logging - not showing infos
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Bidirectional
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Multiply, Permute, RepeatVector, Lambda
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization, Add, TimeDistributed, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LeakyReLU, PReLU, ELU, Activation
from tensorflow.keras.optimizers import Adam

# Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from Logger import Logger
logger = Logger(__name__, code_file_name="MLNetworks.py")


class ConcatenateLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.concat(inputs, axis=-1)


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
    elif model_name == 'CNNLSTMAttention':
        return CNNLSTMAttentionNetwork(input_shape, output_layer_dim, dropout_rate).build_model
    else:
        logger.error(f"[ValueError] Model {model_name} is not defined.")


# NETWORKS ---------------------------------------------------------------------
# 1. CNN LSTM Attention Network ------------------------------------------------
class CNNLSTMAttentionNetwork:
    """
    CNN LSTM Attention Network class. This class builds a model with the following architecture:
    - Convolutional block with 64 filters and kernel size 3 (twice)
    - LSTM block with 100 units (twice)
    - Multi-head attention block with 4 heads and key dimension 32
    - Dense block with 64 units
    """

    def __init__(self, input_shape, output_layer_dim=1, dropout_rate=0.2, activation='prelu', l2_reg=0.01):
        self.input_shape = input_shape
        self.l2_reg = l2_reg
        self.output_layer_dim = output_layer_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.model = self.build_model

    @property
    def build_model(self):
        input_layer = Input(shape=(self.input_shape[0], self.input_shape[1]))

        # CNN block with L2 regularization
        cnn_block = Conv1D(filters=128, kernel_size=3, padding='same', kernel_regularizer=l2(self.l2_reg))(input_layer)
        if self.activation == 'leaky_relu':
            cnn_block = LeakyReLU(alpha=0.01)(cnn_block)
        elif self.activation == 'prelu':
            cnn_block = PReLU()(cnn_block)
        elif self.activation == 'elu':
            cnn_block = ELU(alpha=1.0)(cnn_block)
        else:
            cnn_block = Activation(self.activation)(cnn_block)
        cnn_block = Dropout(self.dropout_rate)(cnn_block)

        # Flatten the CNN block for concatenation
        cnn_block_flat = Flatten()(cnn_block)

        # Bidirectional LSTM block
        lstm_block = Bidirectional(LSTM(100, return_sequences=True))(cnn_block)
        lstm_block = Dropout(self.dropout_rate)(lstm_block)

        # Attention block
        attention_block = MultiHeadAttention(num_heads=2, key_dim=64)(lstm_block, lstm_block, lstm_block)
        attention_block = LayerNormalization(epsilon=1e-6)(attention_block)

        # Flatten the attention block for concatenation
        attention_block_flat = Flatten()(attention_block)

        # Concatenate Flattened CNN and Flattened Attention blocks
        concatenated = ConcatenateLayer()([cnn_block_flat, attention_block_flat])

        # Dense layers with L2 regularization
        dense_layer = Dense(128, kernel_regularizer=l2(self.l2_reg))(concatenated)
        if self.activation == 'leaky_relu':
            dense_layer = LeakyReLU(alpha=0.01)(dense_layer)
        elif self.activation == 'prelu':
            dense_layer = PReLU()(dense_layer)
        elif self.activation == 'elu':
            dense_layer = ELU(alpha=1.0)(dense_layer)
        else:
            dense_layer = Activation(self.activation)(dense_layer)
        dense_layer = Dropout(self.dropout_rate)(dense_layer)

        # Output layer with explicit linear activation and L2 regularization
        output_layer = Dense(self.output_layer_dim, activation='linear', kernel_regularizer=l2(self.l2_reg))(dense_layer)

        model = Model(inputs=input_layer, outputs=output_layer)

        return model

    # def __init__(self, input_shape, output_layer_dim=1, dropout_rate=0.2, num_heads=4):
    #     self.input_shape = input_shape
    #     self.output_layer_dim = output_layer_dim
    #     self.dropout_rate = dropout_rate
    #     self.num_heads = num_heads
    #
    # def build_model(self):
    #     input_layer = Input(shape=(self.input_shape[0], self.input_shape[1]))
    #
    #     # Convolutional block
    #     conv_block = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu',
    #                         kernel_regularizer=l2(0.001))(input_layer)
    #     conv_block = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu',
    #                         kernel_regularizer=l2(0.001))(conv_block)
    #     conv_block = Dropout(self.dropout_rate)(conv_block)
    #
    #     # Bidirectional LSTM block
    #     lstm_block = Bidirectional(LSTM(200, return_sequences=True, kernel_regularizer=l2(0.001)))(conv_block)
    #     lstm_block = Bidirectional(LSTM(200, return_sequences=True, kernel_regularizer=l2(0.001)))(lstm_block)
    #     lstm_block = Dropout(self.dropout_rate)(lstm_block)
    #
    #     # Attention block
    #     attention_block = MultiHeadAttention(num_heads=self.num_heads, key_dim=32)(lstm_block, lstm_block)
    #     attention_block = Add()([attention_block, lstm_block])
    #     attention_block = LayerNormalization(epsilon=1e-6)(attention_block)
    #
    #     # Dense block
    #     dense_block = TimeDistributed(Dense(128, activation='relu'))(attention_block)
    #     dense_block = Flatten()(dense_block)
    #     output_layer = Dense(self.output_layer_dim)(dense_block)
    #
    #     model = Model(inputs=input_layer, outputs=output_layer)
    #     return model


# 2. LSTM Network --------------------------------------------------------------
class LSTMNetwork:
    """
    LSTM Network class. This class builds a model with the following architecture:
    - LSTM layer with 100 units (twice)
    - LSTM layer with 50 units
    """

    def __init__(self, input_shape, dropout_rate=0.2, output_layer_dim=1, batch_size=32):
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.output_layer_dim = output_layer_dim
        self.batch_size = batch_size

        self.model = self.build_model()

    def build_model(self):

        model = Sequential()

        model.add(LSTM(100, return_sequences=True, input_shape=(self.input_shape[0], self.input_shape[1]), kernel_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(LSTM(100, return_sequences=True, kernel_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(LSTM(25, kernel_regularizer=l2(0.001)))
        model.add(Dense(self.output_layer_dim, activation='linear'))

        return model


# 3. Bidirectional LSTM Network ------------------------------------------------
class BidirectionalLSTMNetwork:
    """
    Bidirectional LSTM Network class. This class builds a model with the following architecture:
    - Bidirectional LSTM layer with 100 units
    - Bidirectional LSTM layer with 50 units
    - Bidirectional LSTM layer with 25 units
    """
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

        return model


# 4. Hybrid Network ------------------------------------------------------------
class HybridNetwork:
    """
    Hybrid Network class. This class builds a model with the following architecture:
    - TCN block with 64 filters and kernel size 3
    - Transformer-like block with 1 head and key dimension 32
    - Dense block with 64 units
    """
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
