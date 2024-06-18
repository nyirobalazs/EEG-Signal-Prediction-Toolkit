import os
import csv
import numpy as np
import datetime
from scipy.io import savemat

class Saver:
    """
    A class used to handle saving of data in various formats such as .mat and .csv

    ...

    Methods
    -------
    create_directory(path, experiment_folder=False, folder_name=None):
        Creates a directory at the specified path.
    save_evaluations_to_mat(path, evaluations, eventID_channel, filename='EEG_rec.mat'):
        Saves the evaluation results to a .mat file.
    save_evaluations_to_csv(path, evaluations, filename='Overall Results.csv', model_name=None):
        Saves the evaluation results to a .csv file.
    create_new_csv(filepath, header_list, out_value_dict):
        Creates a new .csv file with the specified header and values.
    append_to_csv(filepath, out_value_dict):
        Appends values to an existing .csv file.
    get_parent_directory(path):
        Returns the parent directory of the specified path.
    """

    def __init__(self):
        pass

    # DIRECTORY FUNCTIONS -------------------------------------------------------
    def create_directory(self, path, experiment_folder=False, folder_name=None):
        """
        Creates a directory at the specified path.

        Parameters
        ----------
            path : str
                The path where the directory should be created.
            experiment_folder : bool, optional
                If True, an experiment folder with the current timestamp is created within the path.
            folder_name : str, optional
                The name of the folder to be created within the path.
        """
        if not path:
            raise ValueError("[ERROR] Path is not defined.")

        if experiment_folder:
            path = os.path.join(path, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        elif folder_name is not None:
            path = os.path.join(path, folder_name)
        else:
            path = path

        if not os.path.exists(path):
            os.makedirs(path)

        return path

    # SAVE FUNCTIONS ------------------------------------------------------------
    def save_evaluations_to_mat(self, path, evaluations, eventID_channel, filename='EEG_rec.mat'):
        """
        Saves the evaluation results to a .mat file.

        Parameters
        ----------
            path : str
                The path where the .mat file should be saved.
            evaluations : dict
                The evaluation results to be saved.
            eventID_channel : numpy.ndarray
                The event ID channel data.
            filename : str, optional
                The name of the .mat file.
        """
        all_channels = []
        try:
            filepath = os.path.join(path, filename)

            for key, value in evaluations.items():
                all_channels.append(value['predicted_fitted'])
            print(f"   [INFO] Evaluation results length: {len(all_channels)} channels. Shape of the data: {all_channels[0].shape}. Event ID channel shape: {eventID_channel[:len(all_channels[0])].shape}")
            all_channels.append(eventID_channel[:len(all_channels[0])])
            all_signal_channels = np.array(all_channels)
            savemat(file_name=filepath, mdict={'EEG_rec': all_signal_channels})
            print(f"   [INFO] Evaluation .mat file saved to {path}.")
        except Exception as e:
            raise ValueError(f"[ERROR] Error in saving evaluations to .mat file: {e}. Shape of the data: {all_channels.shape}")


    def save_evaluations_to_csv(self, path, evaluations, filename='Overall Results.csv', model_name=None):
        """
        Saves the evaluation results to a .csv file.

        Parameters
        ----------
            path : str
                The path where the .csv file should be saved.
            evaluations : dict
                The evaluation results to be saved.
            filename : str, optional
                The name of the .csv file.
            model_name : str, optional
                The name of the model.
        """
        try:
            if model_name is None:
                model_name = 'Unknown Model' # default model name
            else:
                filename = f'{model_name}_{filename}'

            filepath = os.path.join(path, filename)
            mean_absolute_errors = []
            root_mean_squared_errors = []
            r2_scores = []
            channel_IDs = []

            # Get the evaluation results
            for key, value in evaluations.items():
                mean_absolute_errors.append(value['mae'])
                root_mean_squared_errors.append(value['rmse'])
                r2_scores.append(value['r2'])
                channel_IDs.append(key)
            # -------------------------
            mean_MAEs = np.mean(mean_absolute_errors)
            mean_RMSEs = np.mean(root_mean_squared_errors)
            mean_R2s = np.mean(r2_scores)
            # -------------------------
            median_MAEs = np.median(mean_absolute_errors)
            median_RMSEs = np.median(root_mean_squared_errors)
            median_R2s = np.median(r2_scores)
            # -------------------------
            std_MAEs = np.std(mean_absolute_errors)
            std_RMSEs = np.std(root_mean_squared_errors)
            std_R2s = np.std(r2_scores)

            # Create header
            header_list = ['Model', 'Mean MAE', 'Mean RMSE', 'Mean R2', 'Median MAE', 'Median RMSE', 'Median R2', 'Std MAE', 'Std RMSE', 'Std R2']
            for channel in channel_IDs:
                header_list.append(f'MAE_{channel}')
                header_list.append(f'RMSE_{channel}')
                header_list.append(f'R2_{channel}')

            # Create values
            out_value_dict = {'Model': model_name,
                              'Mean MAE': mean_MAEs,
                              'Mean RMSE': mean_RMSEs,
                              'Mean R2': mean_R2s,
                              'Median MAE': median_MAEs,
                              'Median RMSE': median_RMSEs,
                              'Median R2': median_R2s,
                              'Std MAE': std_MAEs,
                              'Std RMSE': std_RMSEs,
                              'Std R2': std_R2s}

            for i, channel in enumerate(channel_IDs):
                out_value_dict[f'MAE_{channel}'] = mean_absolute_errors[i]
                out_value_dict[f'RMSE_{channel}'] = root_mean_squared_errors[i]
                out_value_dict[f'R2_{channel}'] = r2_scores[i]

            # if csv file does not exist, create it and write the header
            if not os.path.exists(filepath):
                self.create_new_csv(filepath, header_list, out_value_dict)
            else:
                # read header from existing file
                with open(filepath, 'r') as file:
                    reader = csv.reader(file)
                    current_header = next(reader)
                # if header is different, create a new file with the new header
                if current_header != header_list:
                    with open(filepath, 'r') as file:
                        reader = csv.reader(file)
                        next(reader, None)
                        for row in reader:
                            read_value_dict = dict(zip(header_list, row))
                    # create new csv file with new header and all the values from the old file and the new values
                    self.create_new_csv(filepath, header_list, read_value_dict)
                    self.append_to_csv(filepath,  out_value_dict)
                # if header is the same, append to the file
                else:
                    self.append_to_csv(filepath, out_value_dict)
        except Exception as e:
            raise ValueError(f"[ERROR] Error in saving evaluations to CSV: {e}")


    def create_new_csv(self, filepath, header_list, out_value_dict):
        """
        Creates a new .csv file with the specified header and values.

        Parameters
        ----------
            filepath : str
                The path where the .csv file should be created.
            header_list : list
                The list of headers for the .csv file.
            out_value_dict : dict
                The values to be written to the .csv file.
        """
        try:
            with open(filepath, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header_list)
                writer.writerow(out_value_dict.values())
        except Exception as e:
            raise ValueError(f"[ERROR] Error in creating new CSV file: {e}")

    def append_to_csv(self, filepath, out_value_dict):
        """
        Appends values to an existing .csv file.

        Parameters
        ----------
            filepath : str
                The path of the .csv file.
            out_value_dict : dict
                The values to be appended to the .csv file.
        """
        try:
            with open(filepath, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(out_value_dict.values())
        except Exception as e:
            raise ValueError(f"[ERROR] Error in appending to CSV file: {e}")

    # GETTERS -------------------------------------------------------------------

    def get_parent_directory(self, path):
        return os.path.dirname(path)
