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

import logging
import platform
import os
import psutil
import tensorflow as tf
from datetime import datetime

# ============================== CONSTANTS ==============================

# Permanent variables
LOG_FOLDER_ENV_VAR = 'log'
DEFAULT_LOG_LEVEL = logging.DEBUG
DEFAULT_LOG_FORMAT = '%(asctime)s - %(levelname)s - {code_file_name}:%(lineno)d - %(message)s'
DEFAULT_TIME_FORMAT = '%d-%m-%Y-%H-%M'
DEFAULT_DIVIDER_LENGTH = 100
DEFAULT_DIVIDER_SYMBOL = '-'
DEFAULT_NEW_LINE = True
DEFAULT_LOG_FOLDER = 'logs'

# ======================================================================


class Logger:
    def __init__(self, name, code_file_name, folder_path=None):

        self.code_file_name = code_file_name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(DEFAULT_LOG_LEVEL)
        self.folder_path = folder_path

        # Create file handlers
        self._set_handlers('info')
        self._set_handlers('error')

    @staticmethod
    def _create_log_file_name(log_file_type):
        return f"{log_file_type}_{datetime.now().strftime(DEFAULT_TIME_FORMAT)}.log"

    def _create_log_folder_path(self):
        if self.folder_path and not os.getenv(LOG_FOLDER_ENV_VAR):
            log_folder = os.path.join(self.folder_path, DEFAULT_LOG_FOLDER)
            if not os.path.exists(log_folder):
                os.makedirs(log_folder)
            os.environ[LOG_FOLDER_ENV_VAR] = log_folder
            return log_folder
        elif os.getenv(LOG_FOLDER_ENV_VAR):
            return os.getenv(LOG_FOLDER_ENV_VAR)
        else:
            return f"{os.getcwd()}/{DEFAULT_LOG_FOLDER}"

    def _create_log_file_path(self, log_file_type):
        if os.getenv(LOG_FOLDER_ENV_VAR):
            return os.path.join(os.getenv(LOG_FOLDER_ENV_VAR), self._create_log_file_name(log_file_type))
        else:
            return os.path.join(DEFAULT_LOG_FOLDER, self._create_log_file_name(log_file_type))

    @staticmethod
    def _get_log_file_path():
        return os.getenv(LOG_FOLDER_ENV_VAR)
    @staticmethod
    def _create_log_file_if_not_exists(filename):
        """Creates log file if it doesn't exist."""
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                f.write(f"{filename}.\n")

    def _create_file_handler(self, log_file_type):
        log_file_path = self._create_log_file_path(log_file_type)
        self._create_log_file_if_not_exists(log_file_path)
        handler = logging.FileHandler(log_file_path)
        if log_file_type == 'info':
            handler.setLevel(logging.INFO)
        elif log_file_type == 'error':
            handler.setLevel(logging.ERROR)
        else:
            handler.setLevel(DEFAULT_LOG_LEVEL)
        formatter = logging.Formatter(DEFAULT_LOG_FORMAT.format(code_file_name=self.code_file_name))
        handler.setFormatter(formatter)
        return handler

    def _add_file_handler(self, handler):
        self.logger.addHandler(handler)

    def _create_stream_handler(self, log_file_type):
        handler = logging.StreamHandler()
        if log_file_type == 'info':
            handler.setLevel(logging.INFO)
        elif log_file_type == 'error':
            handler.setLevel(logging.ERROR)
        formatter = logging.Formatter(DEFAULT_LOG_FORMAT.format(code_file_name=self.code_file_name))
        handler.setFormatter(formatter)
        return handler

    def _remove_stream_handler(self):
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                self.logger.removeHandler(handler)

    def _set_handlers(self, log_file_type):
        try:
            if not os.getenv(LOG_FOLDER_ENV_VAR):
                pass
                # handler = self._create_stream_handler(log_file_type)
                # self._add_file_handler(handler)
            elif os.getenv(LOG_FOLDER_ENV_VAR) and any(isinstance(handler, logging.StreamHandler) for handler in self.logger.handlers): # If stream handler exists, remove it and add file handler
                handler = self._create_file_handler(log_file_type)
                self._add_file_handler(handler)
                self._remove_stream_handler()
            elif os.getenv(LOG_FOLDER_ENV_VAR) and not any(isinstance(handler, logging.StreamHandler) for handler in self.logger.handlers): # If stream handler doesn't exist, add file handler
                handler = self._create_file_handler(log_file_type)
                self._add_file_handler(handler)
            else:
                handler = self._create_stream_handler()
                self._add_file_handler(handler)
        except Exception as e:
            self.logger.error(f"Error in setting handlers: {e}")

    def _check_log_folder(self):
        if os.getenv(LOG_FOLDER_ENV_VAR) and not any(isinstance(handler, logging.FileHandler) for handler in self.logger.handlers):
            info_handler = self._create_file_handler('info')
            error_handler = self._create_file_handler('error')
            self._add_file_handler(info_handler)
            self._add_file_handler(error_handler)
        elif not os.getenv(LOG_FOLDER_ENV_VAR):
            os.environ[LOG_FOLDER_ENV_VAR] = self._create_log_folder_path()
            info_handler = self._create_file_handler('info')
            error_handler = self._create_file_handler('error')
            self._add_file_handler(info_handler)
            self._add_file_handler(error_handler)

    def info(self, message):
        self._check_log_folder()
        self.logger.info(message)

    def error(self, message):
        self._check_log_folder()
        self.logger.error(message)

    def debug(self, message):
        self._check_log_folder()
        self.logger.debug(message)

    def warning(self, message):
        self._check_log_folder()
        self.logger.warning(message)

    def create_divider_line(self, length=DEFAULT_DIVIDER_LENGTH, symbol=DEFAULT_DIVIDER_SYMBOL, new_line=DEFAULT_NEW_LINE):
        """
        Creates a divider line.

        Parameters
        ----------
        length: int, optional
            The length of the divider line.
        symbol: str, optional
            The symbol to use for the divider line.
        new_line: bool, optional
            Whether to add a new line after the divider line.
        """
        divider_line = f"{symbol * length}"
        if new_line:
            divider_line += "\n"
        return divider_line

    def trial_settings(self, model_name, input_size, step_size):
        """
        Logs the trial settings.

        Parameters
        ----------
        model_name: str
            The name of the model.
        input_size: str
            The input size.
        step_size: int
            The step size.
        """
        trial_settings = (
            f"\n\n"
            f"{self.create_divider_line(length=DEFAULT_DIVIDER_LENGTH, symbol=DEFAULT_DIVIDER_SYMBOL, new_line=DEFAULT_NEW_LINE)}\n"
            f"Trial Settings:\n"
            f"===============\n"
            f"  - Model: {model_name}\n"
            f"  - Input Size: {input_size}\n"
            f"  - Step Size: {step_size}\n\n"
            f"{self.create_divider_line(length=DEFAULT_DIVIDER_LENGTH, symbol=DEFAULT_DIVIDER_SYMBOL, new_line=DEFAULT_NEW_LINE)}"
        )
        self.info(trial_settings)

    def log_system_info(self, strategy=None):
        """
        Logs system details.

        Parameters
        ----------
        strategy: tf.distribute.Strategy, optional
            TensorFlow distribution strategy.
        """
        try:
            if strategy is not None:

                self.info(f"Test")

                try:
                    self.info(f'TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}')
                except Exception as e:
                    self.error(f"Error in checking if TensorFlow is built with CUDA: {e}")

                gpus = tf.config.experimental.list_physical_devices('GPU')
                system_info = (
                               f"\n"
                               f"\nSystem Information:\n"
                               f"====================\n"
                               f"  - Platform: {platform.system()}\n"
                               f"  - Platform Release: {platform.release()}\n"
                               f"  - Platform Version: {platform.version()}\n"
                               f"  - Architecture: {platform.machine()}\n"
                               f"  - Hostname: {platform.node()}\n"
                               f"  - CPUs: {os.cpu_count()}\n"
                               f"  - CPU Model: {platform.processor()}\n"
                               f"  - Total RAM: {round(psutil.virtual_memory().total / (1024. ** 3))} GB\n")

                self.info(system_info)

                for gpu in gpus:
                    details = tf.config.experimental.get_device_details(gpu)
                    gpu_info = (
                        f"\n"
                        f"  \n- GPU {gpu.name}: {details['device_name']}\n"
                        f"    - Compute Capability: {details['compute_capability']}\n"
                        f"{self.create_divider_line(length=DEFAULT_DIVIDER_LENGTH, symbol='=', new_line=DEFAULT_NEW_LINE)}"
                    )
                    self.info(gpu_info)

        except Exception as e:
            self.error(f"Error in logging system information: {e}")
