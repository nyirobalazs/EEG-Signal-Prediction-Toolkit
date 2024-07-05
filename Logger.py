import logging
import platform
import os
import psutil
import tensorflow as tf
from datetime import datetime

class Logger:
    def __init__(self, name, code_file_name, info_log_file=None, error_log_file=None, folder_path=None):

        self.code_file_name = code_file_name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.folder_path = folder_path

        # Create log folder if it doesn't exist
        #log_folder_path = self._create_log_folder_path()

        # Create file handlers
        self._set_handlers('info')
        self._set_handlers('error')

    @staticmethod
    def _create_log_file_name(log_file_type):
        return f"{log_file_type}_{datetime.now().strftime('%d-%m-%Y-%H-%M')}.log"

    def _create_log_folder_path(self):
        if self.folder_path and not os.getenv('LOG_FOLDER'):
            log_folder = os.path.join(self.folder_path, 'logs')
            if not os.path.exists(log_folder):
                os.makedirs(log_folder)
            os.environ['LOG_FOLDER'] = log_folder
            return log_folder
        elif os.getenv('LOG_FOLDER'):
            return os.getenv('LOG_FOLDER')
        else:
            return f"{os.getcwd()}/logs"

    def _create_log_file_path(self, log_file_type):
        if os.getenv('LOG_FOLDER'):
            return os.path.join(os.getenv('LOG_FOLDER'), self._create_log_file_name(log_file_type))
        else:
            return os.path.join('logs', self._create_log_file_name(log_file_type))

    @staticmethod
    def _get_log_file_path():
        return os.getenv('LOG_FOLDER')
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
            handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(f'%(asctime)s - %(levelname)s - {self.code_file_name}:%(lineno)d - %(message)s')
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
        formatter = logging.Formatter(f'%(asctime)s - %(levelname)s - {self.code_file_name}:%(lineno)d - %(message)s')
        handler.setFormatter(formatter)
        return handler

    def _remove_stream_handler(self):
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                self.logger.removeHandler(handler)

    def _set_handlers(self, log_file_type):
        try:
            if not os.getenv('LOG_FOLDER'):
                pass
                # handler = self._create_stream_handler(log_file_type)
                # self._add_file_handler(handler)
            elif os.getenv('LOG_FOLDER') and any(isinstance(handler, logging.StreamHandler) for handler in self.logger.handlers): # If stream handler exists, remove it and add file handler
                handler = self._create_file_handler(log_file_type)
                self._add_file_handler(handler)
                self._remove_stream_handler()
            elif os.getenv('LOG_FOLDER') and not any(isinstance(handler, logging.StreamHandler) for handler in self.logger.handlers): # If stream handler doesn't exist, add file handler
                handler = self._create_file_handler(log_file_type)
                self._add_file_handler(handler)
            else:
                handler = self._create_stream_handler()
                self._add_file_handler(handler)
        except Exception as e:
            self.logger.error(f"Error in setting handlers: {e}")

    def _check_log_folder(self):
        if os.getenv('LOG_FOLDER') and not any(isinstance(handler, logging.FileHandler) for handler in self.logger.handlers):
            info_handler = self._create_file_handler('info')
            error_handler = self._create_file_handler('error')
            self._add_file_handler(info_handler)
            self._add_file_handler(error_handler)
        elif not os.getenv('LOG_FOLDER'):
            os.environ['LOG_FOLDER'] = self._create_log_folder_path()
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
            f"Trial Settings:\n"
            f"  - Model: {model_name}\n"
            f"  - Input Size: {input_size}\n"
            f"  - Step Size: {step_size}\n"
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
                system_info = ("\nSystem Information:\n"
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
                        f"  - GPU {gpu.name}: {details['device_name']}\n"
                        f"    - Compute Capability: {details['compute_capability']}\n"
                    )
                    self.info(gpu_info)

        except Exception as e:
            self.error(f"Error in logging system information: {e}")
