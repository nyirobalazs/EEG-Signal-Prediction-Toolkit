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

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# Set environment variables for debugging
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["NCCL_DEBUG"] = "INFO"

import tensorflow as tf

# Suppress TensorFlow logging - not showing infos
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

from Logger import Logger
logger = Logger(__name__, code_file_name="MLEngine.py")

# ============================== CONSTANTS ==============================

# Permanent variables
MODEL_CHECKPOINT_FOLDER_NAME = 'models'
MODEL_CHECKPOINT_FORMAT = 'h5'
LOGS_FOLDER_NAME = 'logs'
TENSORNOARD_LOGS_FOLDER_NAME = 'tensor_boards'

# ======================================================================


def mean_accuracy_within_tolerance(y_true, y_pred):
    # Calculate the difference between the true and predicted values and check if the difference is within a tolerance
    y_pred = tf.cast(y_pred, tf.float32)
    diff = tf.abs(y_true - y_pred)
    return tf.reduce_mean(tf.cast(tf.less(diff, 0.01), tf.float32))

def configure_memory_growth():
    """
    Configures TensorFlow to allow memory growth for the GPU.
    """
    try:
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        logger.info(f"Physical GPUs: {len(logical_gpus)}. Memory growth is configured.")
    except Exception as e:
        logger.error(f"Error in configuring memory growth: {e}")

# Configure the memory growth for the GPU
configure_memory_growth()

class MLEngine:
    """
    A class used to represent a Machine Learning Engine

    ...

    Attributes
    ----------
    model : tensorflow.keras.Model
        a compiled tensorflow model
    optimizer : str
        the optimizer to use during model compilation
    loss : str
        the loss function to use during model compilation
    metrics : list
        the list of metrics to use during model compilation

    Methods
    -------
    compile_model():
        Compiles the model with the specified optimizer, loss function, and metrics.
    train_model(x_train, y_train, batch_size=32, epochs=1, validation_data=None, save_path=None, model_name=None):
        Trains the model with the provided training data and parameters.
    calculate_errors(evalY, predicted_segments):
        Calculates the Mean Absolute Error, Root Mean Squared Error, and R2 Score between the actual and predicted values.
    evaluate_model(x_eval, y_eval, input_size, step_size):
        Evaluates the model with the provided evaluation data and parameters.
    save_model(filepath):
        Saves the model to the specified filepath.
    load_model(filepath):
        Loads a model from the specified filepath.
    """

    def __init__(self, model, strategy, optimizer='adam',  # Add the strategy parameter
                 loss='mean_absolute_error',
                 metrics=[mean_accuracy_within_tolerance],
                 initial_learning_rate=None,
                 decay_steps=None,
                 decay_rate=None,
                 staircase=False,
                 is_test_code=False,
                 early_stopping_patience=50,
                 loss_monitor='val_loss',
                 loss_mode='min',
                 reduce_lr_factor=0.1,
                 reduce_lr_patience=50,
                 min_lr=0.01):
        """
        Constructs all the necessary attributes for the MLEngine object.

        Parameters
        ----------
            model : tensorflow.keras.Model
                a compiled tensorflow model
            optimizer : str, optional
                the optimizer to use during model compilation (default is 'adam')
            loss : str, optional
                the loss function to use during model compilation (default is 'mean_squared_error')
            metrics : list, optional
                the list of metrics to use during model compilation (default is ['accuracy'])
        """
        # Set the policy to 'mixed_float16' for mixed precision training
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)

        # Distribute your training across multiple GPUs
        self.strategy = strategy
        with self.strategy.scope():
            self.model = model
            self.optimizer = optimizer
            self.loss = loss
            self.metrics = metrics
            self.initial_learning_rate = initial_learning_rate
            self.decay_steps = decay_steps
            self.decay_rate = decay_rate
            self.staircase = staircase
            self.compile_model()
            self.is_test_code = is_test_code
            self.early_stopping_patience = early_stopping_patience
            self.loss_monitor = loss_monitor
            self.loss_mode = loss_mode
            self.reduce_lr_factor = reduce_lr_factor
            self.reduce_lr_patience = reduce_lr_patience
            self.min_lr = min_lr


    def compile_model(self):
        """Compiles the model with the specified optimizer, loss function, and metrics."""
        if self.optimizer == 'adam' and (self.initial_learning_rate is None and self.decay_steps is None and self.decay_rate is None and self.staircase is None):

            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.initial_learning_rate,
                decay_steps=self.decay_steps,
                decay_rate=self.decay_rate,
                staircase=self.staircase)
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def train_model(self, X_train, y_train, batch_size=32, epochs=100, save_path=None,
                    model_name=None, split_ratio=0.2, model_type='many_to_one'):
        """
        Trains the model with the provided training data and parameters.

        Parameters
        ----------
            X_train : numpy.ndarray
                the training data
            y_train : numpy.ndarray
                the target values for the training data
            batch_size : int, optional
                the number of samples per gradient update (default is 32)
            epochs : int, optional
                the number of epochs to train the model (default is 1)
            validation_data : tuple, optional
                the data on which to evaluate the loss and any model metrics at the end of each epoch (default is None)
            save_path : str, optional
                the path to save the model and logs (default is None)
            model_name : str, optional
                the name of the model (default is None)
            split_ratio : float, optional
                the proportion of the dataset to include in the validation split (default is 0.2)
        """

        # Define the callbacks
        logger.info(f"Training the model with {model_type} model type.")

        early_stopping = EarlyStopping(monitor=self.loss_monitor, patience=self.early_stopping_patience,
                                       verbose=1 if self.is_test_code else 0, mode=self.loss_mode,
                                       restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(f'{save_path}/{MODEL_CHECKPOINT_FOLDER_NAME}/{model_name}.{MODEL_CHECKPOINT_FORMAT}', monitor=self.loss_monitor,
                                           save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor=self.loss_monitor, factor=self.reduce_lr_factor,
                                      patience=self.reduce_lr_patience, min_lr=self.min_lr)
        csv_logger = CSVLogger(f'{save_path}/{LOGS_FOLDER_NAME}/{model_name}_training.log')
        tensorboard = TensorBoard(log_dir=f'{save_path}/{LOGS_FOLDER_NAME}/{TENSORNOARD_LOGS_FOLDER_NAME}')

        # Add the callbacks to the model training
        with self.strategy.scope():
            self.history = self.model.fit(X_train, y_train, batch_size=batch_size,
                                          epochs=epochs, validation_split=split_ratio,
                                          callbacks=[early_stopping, model_checkpoint, reduce_lr, csv_logger,
                                                     tensorboard],
                                          verbose=1 if self.is_test_code else 0,
                                          shuffle=False)

    def calculate_errors(self, evalY, predicted_segments):
        """
        Calculates the Mean Absolute Error, Root Mean Squared Error, and R2 Score between the actual and predicted values.

        Parameters
        ----------
            evalY : numpy.ndarray
                the actual values
            predicted_segments : numpy.ndarray
                the predicted values

        Returns
        -------
            mae : float
                the Mean Absolute Error
            rmse : float
                the Root Mean Squared Error
            r2 : float
                the R2 Score
        """
        # Flatten the arrays for comparison
        evalY_flattened = evalY.flatten()
        predicted_segments_flattened = predicted_segments.flatten()

        # Calculate Mean Absolute Error
        mae = mean_absolute_error(evalY_flattened, predicted_segments_flattened)

        # Calculate Root Mean Squared Error
        rmse = np.sqrt(mean_squared_error(evalY_flattened, predicted_segments_flattened))

        # Calculate R2 Score
        r2 = r2_score(evalY_flattened, predicted_segments_flattened)

        return mae, rmse, r2

    def predict(self, x_eval, mode_type='many_to_one'):
        """
        Predicts the values of the provided data.
        """

        # Predict the values
        with self.strategy.scope():
            predicted_segments = self.model.predict(x_eval, verbose=1 if self.is_test_code else 0)

        return predicted_segments

    def evaluate_model(self, x_eval, y_eval, input_size, step_size, TrainDataPrepare_class, channel_index=0, model_type='many_to_one'):
        """
        Evaluates the model with the provided evaluation data and parameters.

        Parameters
        ----------
            x_eval : numpy.ndarray
                the evaluation data
            y_eval : numpy.ndarray
                the target values for the evaluation data
            input_size : int
                the size of the input data
            step_size : int
                the size of the step
            TrainDataPrepare_class : class
                the TrainDataPrepare class for denormalizing the data
            channel_index : int, optional
                the index of the channel to evaluate (default is 0)

        Returns
        -------
            evaluation : dict
                a dictionary containing the Mean Absolute Error, Root Mean Squared Error, R2 Score, and predicted values
        """
        # Predict the segments
        predicted_segments = self.predict(x_eval, model_type)
        predicted_fitted = TrainDataPrepare_class.scaler.inverse_transform_single_channel(predicted_segments, channel_index=channel_index)

        # Denormalize the data
        predicted_fitted = predicted_fitted.reshape(-1, 1)
        predicted_fitted = np.concatenate((np.zeros((input_size, 1)), predicted_fitted, np.zeros((step_size, 1))),
                                          axis=0).squeeze()
        # predicted_fitted = np.pad(predicted_fitted, ((input_size, step_size), (0, 0)), 'constant', constant_values=0).squeeze()

        # Calculate the errors
        mae, rmse, r2 = self.calculate_errors(y_eval, predicted_segments)

        # create a dictionary with the evaluation results and predicted_fitted
        evaluation = {'mae': mae, 'rmse': rmse, 'r2': r2, 'predicted_fitted': predicted_fitted}
        logger.info(f"Evaluation results: MAE={mae}, RMSE={rmse}, R2={r2}. Predicted shape: {predicted_fitted.shape}")
        return evaluation


    def save_model(self, filepath):
        """
        Saves the model to the specified filepath.

        Parameters
        ----------
            filepath : str
                the path to save the model
        """
        self.model.save(filepath)

    def load_model(self, filepath):
        """
        Loads a model from the specified filepath.

        Parameters
        ----------
            filepath : str
                the path to load the model from
        """
        self.model = tf.keras.models.load_model(filepath)
