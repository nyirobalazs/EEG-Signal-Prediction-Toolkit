import tensorflow as tf
import os
import platform

from Loader import SourceFileLoader
from Preprocess import Preprocess
from Training import Training

# CONFIGURATIONS --------------------------------------------------------------
# TODO: Add the configurations to a separate configuration file

IS_TEST_CODE = False                                                                                                    # Test code flag (if True then only use for first 1000 time steps)
LOAD_DIRECTORY = r"C:\Users\ban28\OneDrive - University of Bath\Documents\PHD\PhD code\Data from Damien\EEG prediction tool\data"      # Directory containing the .mat files
SAVE_DIRECTORY = r"C:\Users\ban28\OneDrive - University of Bath\Documents\PHD\PhD code\Data from Damien\EEG prediction tool\save"      # Directory to save the data
FREQUENCY_RATE = 250                                                                                                    # Sampling frequency of the EEG data
TEST_SIZE = 0.2                                                                                                         # Test size ratio (base setting is 0.2)
IS_NORMALIZE = True                                                                                                     # Normalize the data (if True then normalize the data)
NORMALIZE_METHOD = 'minmax'                                                                                             # Normalization method (base setting is 'minmax')
NORMALIZE_RANGE = (0, 1)                                                                                                # Normalization range (base setting is (0, 1))
INPUT_SEQUENCE_LENGTH = 2000                                                                                            # Input sequence length for the model (base setting is 2000)
EVENT_ID_CHANNEL_INDEX = 3                                                                                              # Event ID channel index, which contains the trigger IDs (base setting is 3)
TRIAL_START_IDS = [768, 1023]                                                                                           # Trial start IDs which mark the beginning of a trials (base setting is 768 and 1023, format: [768, 1023]
FAILED_SEGMENT_IDS = [1023]                                                                                             # Failed segment IDs which are not used (base setting is 1023, format: [1023])
TRIGGER_IDS = [1, 2]                                                                                                    # Trigger IDs. These classes are used for classification (base setting is 1 and 2, format: [1, 2])
MODE = "many-to-one"                                                                                                    # Mode for the model (base setting is "many-to-one")
EPOCHS = 50                                                                                                             # Number of epochs for the training
BATCH_SIZE = 32                                                                                                         # Batch size for the training
REQUIRED_STRUCTURE_NAMES = ['xLeft', 'xRight', 'trainY']                                                                # Names of the required structures from the .mat files for the training
TRAIN_PROGRAM = {'LSTM': {'input_sizes': [100, 250, 500, 1000, 2000],
                            'step_sizes': [1, 100, 250, 500, 1000],
                            },
                 'BiLSTM': {'input_sizes': [100, 250, 500, 1000, 2000],
                            'step_sizes': [1, 100, 250, 500, 1000],
                            },
                 }

# MAIN FUNCTION ---------------------------------------------------------------
def main():

    # Initialize the classes
    strategy = tf.distribute.MirroredStrategy()
    loader = SourceFileLoader()
    preprocessor = Preprocess(eventID_channel_ind=EVENT_ID_CHANNEL_INDEX)
    trainer = Training(eventID_channel_ind=EVENT_ID_CHANNEL_INDEX,
                       fs=FREQUENCY_RATE,
                       test_size=TEST_SIZE,
                       is_normalize=IS_NORMALIZE,
                       input_sequence_length=INPUT_SEQUENCE_LENGTH,
                       mode=MODE,
                       main_saving_path=SAVE_DIRECTORY,
                       train_program=TRAIN_PROGRAM,
                       is_test_code=IS_TEST_CODE,
                       epochs=EPOCHS,
                       batch_size=BATCH_SIZE,
                       normalizing_range=NORMALIZE_RANGE,
                       normalizing_method=NORMALIZE_METHOD,
                       strategy=strategy)  # Add the strategy parameter

    # Print the system details
    print_system_info(strategy)

    # Run the pipeline
    source_file_directory = loader.loader_pipeline(LOAD_DIRECTORY)
    segments_dict = preprocessor.preprocessing_pipeline(source_file_directory,
                                                        trial_start_ids=TRIAL_START_IDS,
                                                        mode=MODE,
                                                        failed_segment_ids=FAILED_SEGMENT_IDS,
                                                        trigger_ids=TRIGGER_IDS,
                                                        structure_name_list=REQUIRED_STRUCTURE_NAMES)
    trainer.training_pipeline(segments_dict)


def print_system_info(strategy=None):
    """
    Prints the system details.

    Parameters
    ----------
        strategy: tf.distribute.Strategy
    """
    if strategy is not None:
        try:
            try:
                print(f'[INFO] TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}')
            except Exception as e:
                print(f"[ERROR] Error in checking if TensorFlow is built with CUDA: {e}")

            gpus = tf.config.experimental.list_physical_devices('GPU')
            print("System Information:"
                  f"  - Platform: {platform.system()}"
                  f"  - Platform Release: {platform.release()}"
                  f"  - Platform Version: {platform.version()}"
                  f"  - Architecture: {platform.machine()}"
                  f"  - Hostname: {platform.node()}"
                  f"  - CPUs: {os.cpu_count()}"
                  f"  - CPU Model: {platform.processor()}"
                  f"  - Total RAM: {round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024. ** 3))} GB")

            for gpu in gpus:
                details = tf.config.experimental.get_device_details(gpu)
                print(f"  - GPU {gpu.name}: {details['device_name']}")
                print(f"    - Compute Capability: {details['compute_capability']}")

        except Exception as e:
            print(f"[ERROR] Error in getting system details: {e}")


if __name__ == "__main__":
    main()