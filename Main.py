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
import json
import argparse
import os

from Loader import SourceFileLoader
from Preprocess import Preprocess
from Training import Training
from Logger import Logger

# ============================== GLOBAL VARIABLES ==============================

# Required configuration keys and their expected types
REQUIRED_KEYS = {
    "IS_TEST_CODE": bool,  # Indicates if the code is being tested
    "LOAD_DIRECTORY": str,  # Directory from which to load data
    "SAVE_DIRECTORY": str,  # Directory to save outputs
    "FREQUENCY_RATE": int,  # Sampling frequency rate
    "TEST_SIZE": float,  # Proportion of data to be used for testing
    "IS_NORMALIZE": bool,  # Flag to indicate if normalization is to be applied
    "NORMALIZE_METHOD": str,  # Method to be used for normalization
    "NORMALIZE_RANGE": list,  # Range to be used for normalization
    "EVENT_ID_CHANNEL_INDEX": int,  # Index of the channel containing event IDs
    "TRIAL_START_IDS": list,  # List of trial start IDs
    "FAILED_SEGMENT_IDS": list,  # List of IDs for segments that failed processing
    "TRIGGER_IDS": list,  # List of trigger IDs
    "MODE": str,  # Mode of operation
    "EPOCHS": int,  # Number of training epochs
    "BATCH_SIZE": int,  # Training batch size
    "REQUIRED_STRUCTURE_NAMES": list,  # List of required structure names for processing
    "TRAIN_PROGRAM": dict, # Dictionary containing training program configurations
    "LOSS_FUNCTION": str,  # Loss function to be used
    "INITIAL_LEARNING_RATE": float,  # Initial learning rate
    "DECAY_STEPS": int,  # Number of steps for learning rate decay
    "DECAY_RATE": float,  # Rate of learning rate decay
    "STAIRCASE": bool,  # Flag to indicate if learning rate decay is to be applied in a staircase manner
    "EARLY_STOPPING_PATIENCE": int,  # Patience for early stopping
    "LOSS_MONITOR": str,  # Metric to monitor for loss
    "LOSS_MODE": str,  # Mode for loss monitoring
    "REDUCE_LR_FACTOR": float,  # Factor for reducing learning rate
    "REDUCE_LR_PATIENCE": int,  # Patience for reducing learning rate
    "MIN_LR": float,  # Minimum learning rate
    "DROPOUT_RATE": float  # Dropout rate
}

# =================================================================================


def validate_config(config):
    """
    Validates the configuration dictionary against the required keys and types.

    Parameters:
    - config (dict): The configuration dictionary to validate.

    Raises:
    - KeyError: If a required key is missing from the configuration.
    - TypeError: If the type of a configuration item does not match its expected type.
    - ValueError: If 'NORMALIZE_RANGE' does not contain exactly two numeric values.
    """
    for key, expected_type in REQUIRED_KEYS.items():
        if key not in config:
            raise KeyError(f"Missing required configuration key: {key}")
        if not isinstance(config[key], expected_type):
            raise TypeError(f"Incorrect type for key '{key}'. Expected {expected_type}, got {type(config[key])}.")
    if len(config["NORMALIZE_RANGE"]) != 2 or not all(isinstance(i, (int, float)) for i in config["NORMALIZE_RANGE"]):
        raise ValueError("NORMALIZE_RANGE must be a list of two numeric values.")


def load_config(json_path):
    """
    Loads the configuration from a JSON file and updates the global variables accordingly.

    Parameters:
    - json_path (str): The path to the JSON configuration file.
    """
    with open(json_path, 'r') as f:
        config = json.load(f)
    validate_config(config)
    config["NORMALIZE_RANGE"] = tuple(config["NORMALIZE_RANGE"])  # Convert list to tuple for consistency
    globals().update(config)


def select_config_file(settings_dir):
    """
    Selects the newest configuration file from a specified directory.

    Parameters:
    - settings_dir (str): The directory containing configuration files.

    Returns:
    - str: The path to the most recently modified configuration file.

    Raises:
    - FileNotFoundError: If no JSON configuration files are found in the specified directory.
    """
    config_files = [f for f in os.listdir(settings_dir) if f.endswith('.json')]
    if not config_files:
        raise FileNotFoundError("No JSON configuration files found in the settings directory.")
    config_files = sorted(config_files, key=lambda x: os.path.getmtime(os.path.join(settings_dir, x)), reverse=True)
    return os.path.join(settings_dir, config_files[0])


def main(config_path=None):
    """
    Main function to initialize and run the data processing and training pipeline.

    Parameters:
    - config_path (str, optional): The path to the JSON configuration file. If not provided, the newest configuration file from the 'settings' directory is used.
    """
    
    # Other initializations
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
    
    if config_path is None:
        settings_dir = os.path.join(os.path.dirname(__file__), 'settings')
        config_path = select_config_file(settings_dir)

    load_config(config_path)  # Load configurations

    logger = Logger(__name__, code_file_name="Main.py", folder_path=SAVE_DIRECTORY)  # Initialize logger
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())  # Define TensorFlow distribution strategy

    logger.log_system_info(strategy=strategy)  # Log system information

    # Initialize processing classes
    loader = SourceFileLoader()
    preprocessor = Preprocess(eventID_channel_ind=EVENT_ID_CHANNEL_INDEX)
    trainer = Training(eventID_channel_ind=EVENT_ID_CHANNEL_INDEX,
                       fs=FREQUENCY_RATE,
                       test_size=TEST_SIZE,
                       is_normalize=IS_NORMALIZE,
                       mode=MODE,
                       main_saving_path=SAVE_DIRECTORY,
                       train_program=TRAIN_PROGRAM,
                       is_test_code=IS_TEST_CODE,
                       epochs=EPOCHS,
                       batch_size=BATCH_SIZE,
                       normalizing_range=NORMALIZE_RANGE,
                       normalizing_method=NORMALIZE_METHOD,
                       strategy=strategy,
                       loss_function=LOSS_FUNCTION,
                       initial_learning_rate=INITIAL_LEARNING_RATE,
                       decay_steps=DECAY_STEPS,
                       decay_rate=DECAY_RATE,
                       staircase=STAIRCASE,
                       early_stopping_patience=EARLY_STOPPING_PATIENCE,
                       loss_monitor=LOSS_MONITOR,
                       loss_mode=LOSS_MODE,
                       reduce_lr_factor=REDUCE_LR_FACTOR,
                       reduce_lr_patience=REDUCE_LR_PATIENCE,
                       min_lr=MIN_LR,
                       dropout_rate=DROPOUT_RATE
                       )
 
    # Execute the pipeline
    source_file_directory = loader.loader_pipeline(LOAD_DIRECTORY)
    segments_dict = preprocessor.preprocessing_pipeline(source_file_directory,
                                                        trial_start_ids=TRIAL_START_IDS,
                                                        mode=MODE,
                                                        failed_segment_ids=FAILED_SEGMENT_IDS,
                                                        trigger_ids=TRIGGER_IDS,
                                                        structure_name_list=REQUIRED_STRUCTURE_NAMES)
    trainer.training_pipeline(segments_dict, config_path=config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load configurations from a JSON file.')
    parser.add_argument('config_path', type=str, nargs='?', default=None, help='Path to the configuration JSON file.')
    args = parser.parse_args()

    main(args.config_path)