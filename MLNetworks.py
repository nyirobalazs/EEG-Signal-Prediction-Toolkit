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

    def __init__(self, input_shape, output_layer_dim=1, dropout_rate=0.2, activation='elu', l2_reg=0.01):
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
        elif self.activation == 'glu':
            cnn_block = Activation('tanh')(cnn_block)
            cnn_block = Conv1D(filters=128, kernel_size=3, padding='same', kernel_regularizer=l2(self.l2_reg))(cnn_block)
            cnn_block = Multiply()([cnn_block, Activation('sigmoid')(cnn_block)])
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
        elif self.activation == 'glu':
            dense_layer = Activation('tanh')(dense_layer)
            dense_layer = Dense(128, kernel_regularizer=l2(self.l2_reg))(dense_layer)
            dense_layer = Multiply()([dense_layer, Activation('sigmoid')(dense_layer)])
        else:
            dense_layer = Activation(self.activation)(dense_layer)
        dense_layer = Dropout(self.dropout_rate)(dense_layer)

        # Output layer with explicit linear activation and L2 regularization
        output_layer = Dense(self.output_layer_dim, activation='linear', kernel_regularizer=l2(self.l2_reg))(dense_layer)

        model = Model(inputs=input_layer, outputs=output_layer)

        return model


# 2. LSTM Network --------------------------------------------------------------
class LSTMNetwork:
    """
    LSTM Network class. This class builds a model with the following architecture:
    - LSTM layer with 100 units (twice)
    - LSTM layer with 50 units
    """

    def __init__(self, input_shape, dropout_rate, output_layer_dim,
                 lstm_units=64, dense_units=32,
                 activation='relu', l2_reg=0.01):
        self.input_shape = input_shape
        self.output_layer_dim = output_layer_dim
        self.dropout_rate = dropout_rate  # This should be a float, e.g., 0.2
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.model = self.build_model()

    def apply_activation(self, dense_layer):
        if self.activation == 'leaky_relu':
            return LeakyReLU(alpha=0.01)(dense_layer)
        elif self.activation == 'prelu':
            return PReLU()(dense_layer)
        elif self.activation == 'elu':
            return ELU(alpha=1.0)(dense_layer)
        elif self.activation == 'glu':
            dense_layer_tanh = Dense(self.dense_units, kernel_regularizer=l2(self.l2_reg))(dense_layer)
            dense_layer_tanh = Activation('tanh')(dense_layer_tanh)
            dense_layer_sigmoid = Dense(self.dense_units, kernel_regularizer=l2(self.l2_reg))(dense_layer)
            dense_layer_sigmoid = Activation('sigmoid')(dense_layer_sigmoid)
            return Multiply()([dense_layer_tanh, dense_layer_sigmoid])
        elif self.activation in ['relu', 'sigmoid', 'tanh', 'softmax']:  # List common activations
            return Activation(self.activation)(dense_layer)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

    def build_model(self):
        input_layer = Input(shape=self.input_shape)

        # First LSTM layer
        lstm_layer_1 = LSTM(self.lstm_units, return_sequences=True)(input_layer)
        lstm_layer_1 = BatchNormalization()(lstm_layer_1)
        lstm_layer_1 = Dropout(self.dropout_rate)(lstm_layer_1)

        # Second LSTM layer
        lstm_layer_2 = LSTM(self.lstm_units, return_sequences=True)(lstm_layer_1)
        lstm_layer_2 = BatchNormalization()(lstm_layer_2)
        lstm_layer_2 = Dropout(self.dropout_rate)(lstm_layer_2)

        # Third LSTM layer
        lstm_layer_3 = LSTM(self.lstm_units)(lstm_layer_2)
        lstm_layer_3 = BatchNormalization()(lstm_layer_3)
        lstm_layer_3 = Dropout(self.dropout_rate)(lstm_layer_3)

        # Dense layers
        dense_layer_1 = Dense(self.dense_units, kernel_regularizer=l2(self.l2_reg))(lstm_layer_3)
        dense_layer_1 = self.apply_activation(dense_layer_1)
        dense_layer_1 = Dropout(self.dropout_rate)(dense_layer_1)

        dense_layer_2 = Dense(self.dense_units, kernel_regularizer=l2(self.l2_reg))(dense_layer_1)
        dense_layer_2 = self.apply_activation(dense_layer_2)
        dense_layer_2 = Dropout(self.dropout_rate)(dense_layer_2)

        # Output layer
        output_layer = Dense(self.output_layer_dim, activation='linear')(dense_layer_2)

        model = Model(inputs=input_layer, outputs=output_layer)
        return model

# 3. Bidirectional LSTM Network ------------------------------------------------
class BidirectionalLSTMNetwork:
    """
    Bidirectional LSTM Network class. This class builds a model with the following architecture:
    - Bidirectional LSTM layer with 100 units (three times)
    """

    def __init__(self, input_shape, dropout_rate, output_layer_dim,
                 lstm_units=64, dense_units=32,
                 activation='relu', l2_reg=0.01):
        self.input_shape = input_shape
        self.output_layer_dim = output_layer_dim
        # Assign dropout_rate to self
        self.dropout_rate = dropout_rate
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.model = self.build_model()

    def apply_activation(self, dense_layer):
        if self.activation == 'leaky_relu':
            return LeakyReLU(alpha=0.01)(dense_layer)
        elif self.activation == 'prelu':
            return PReLU()(dense_layer)
        elif self.activation == 'elu':
            return ELU(alpha=1.0)(dense_layer)
        elif self.activation == 'glu':
            dense_layer_tanh = Dense(self.dense_units, kernel_regularizer=l2(self.l2_reg))(dense_layer)
            dense_layer_tanh = Activation('tanh')(dense_layer_tanh)
            dense_layer_sigmoid = Dense(self.dense_units, kernel_regularizer=l2(self.l2_reg))(dense_layer)
            dense_layer_sigmoid = Activation('sigmoid')(dense_layer_sigmoid)
            return Multiply()([dense_layer_tanh, dense_layer_sigmoid])
        else:
            return Activation(self.activation)(dense_layer)

    def build_model(self):
        input_layer = Input(shape=self.input_shape)

        # First Bidirectional LSTM layer
        lstm_layer_1 = Bidirectional(LSTM(self.lstm_units, return_sequences=True))(input_layer)
        lstm_layer_1 = BatchNormalization()(lstm_layer_1)
        lstm_layer_1 = Dropout(self.dropout_rate)(lstm_layer_1)

        # Second Bidirectional LSTM layer
        lstm_layer_2 = Bidirectional(LSTM(self.lstm_units, return_sequences=True))(lstm_layer_1)
        lstm_layer_2 = BatchNormalization()(lstm_layer_2)
        lstm_layer_2 = Dropout(self.dropout_rate)(lstm_layer_2)

        # Third Bidirectional LSTM layer
        lstm_layer_3 = Bidirectional(LSTM(self.lstm_units))(lstm_layer_2)
        lstm_layer_3 = BatchNormalization()(lstm_layer_3)
        lstm_layer_3 = Dropout(self.dropout_rate)(lstm_layer_3)

        # Dense layers
        dense_layer_1 = Dense(self.dense_units, kernel_regularizer=l2(self.l2_reg))(lstm_layer_3)
        dense_layer_1 = self.apply_activation(dense_layer_1)
        dense_layer_1 = Dropout(self.dropout_rate)(dense_layer_1)

        dense_layer_2 = Dense(self.dense_units, kernel_regularizer=l2(self.l2_reg))(dense_layer_1)
        dense_layer_2 = self.apply_activation(dense_layer_2)
        dense_layer_2 = Dropout(self.dropout_rate)(dense_layer_2)

        # Output layer
        output_layer = Dense(self.output_layer_dim, activation='linear')(dense_layer_2)

        model = Model(inputs=input_layer, outputs=output_layer)
        return model


# 4. Hybrid Network ------------------------------------------------------------
class HybridNetwork:
    """
        Redesigned Hybrid Network class for EEG signal forecasting.
        This class builds a model with the following architecture:
        - TCN block with 64 filters and kernel size 3
        - Transformer block with 2 heads and key dimension 32
        - LSTM block for better temporal understanding
        - Dense block with custom activation and units
        - Output layer for multi-step ahead predictions
    """
    def __init__(self, input_shape, output_layer_dim=10, dropout_rate=0.2, lstm_units=100, dense_units=64, activation='relu'):
        self.input_shape = input_shape
        self.output_layer_dim = output_layer_dim
        self.dropout_rate = dropout_rate
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.activation = activation
        self.model = self.build_model()

    def build_model(self):
        # Input layer
        input_layer = Input(shape=(self.input_shape[0], self.input_shape[1]))

        # TCN block
        tcn_block = Conv1D(filters=64, kernel_size=3, padding='causal', activation='relu')(input_layer)
        tcn_block = BatchNormalization()(tcn_block)
        tcn_block = Dropout(self.dropout_rate)(tcn_block)
        tcn_block = Flatten()(tcn_block)

        # Transformer block
        transformer_block = MultiHeadAttention(num_heads=2, key_dim=32)(input_layer, input_layer)
        transformer_block = LayerNormalization(epsilon=1e-6)(transformer_block)
        transformer_block = Dropout(self.dropout_rate)(transformer_block)
        transformer_block = Flatten()(transformer_block)

        # Concatenate TCN and Transformer blocks
        concatenated = tf.keras.layers.Concatenate()([tcn_block, transformer_block])

        # LSTM block
        lstm_block = LSTM(self.lstm_units, return_sequences=False)(input_layer)
        lstm_block = BatchNormalization()(lstm_block)
        lstm_block = Dropout(self.dropout_rate)(lstm_block)

        # Combine all blocks
        combined = tf.keras.layers.Concatenate(axis=-1)([concatenated, lstm_block])

        # Dense layers
        dense_layer = Dense(units=self.dense_units, activation=self.activation)(combined)
        dense_layer = Dropout(self.dropout_rate)(dense_layer)

        # Output layer for multi-step ahead forecasting
        output_layer = Dense(self.output_layer_dim, activation='linear')(dense_layer)

        model = Model(inputs=input_layer, outputs=output_layer)
        return model
