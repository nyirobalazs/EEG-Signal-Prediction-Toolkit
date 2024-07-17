from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class Scaler:
    def __init__(self, normalizing_range=(0, 1), normalizing_method="minmax", eventID_channel_ind=3):
        self.scaler_method = normalizing_method
        self.scaler_range = normalizing_range
        self.scaler_used = False
        self.eventID_channel_ind = eventID_channel_ind
        self.scalers = []

    def multi_channel_scaler(self, data, is_training=True):

        normalised_data = []
        for channel in range(len(data)):
            channel_data = data[channel, :].reshape(-1, 1)
            if is_training:
                if self.scaler_method == "minmax":
                    scaler = MinMaxScaler(feature_range=self.scaler_range)
                elif self.scaler_method == "standard":
                    scaler = StandardScaler()
                else:
                    raise ValueError(f"[ERROR] Normalization method {self.scaler_method} is not defined.")
                normalised_channel = scaler.fit_transform(channel_data).flatten()
                self.scalers.append(scaler)
            else:
                normalised_channel = self.scalers[channel].transform(channel_data).flatten()
            normalised_data.append(normalised_channel)
        return np.array(normalised_data)

    def inverse_transform_single_channel(self, data, channel_index=0):
        return self.scalers[channel_index].inverse_transform(data)

    def get_scalers(self):
        return self.scalers


class TrainDataPrepare:

    def __init__(self, eventID_channel_ind=3, normalizing_range=(0, 1), normalizing_method="minmax"):

        self.eventID_channel_ind = eventID_channel_ind
        self.scaler_used = False
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.X = None
        self.Y = None
        self.eventID_channel_data = None

        self.scaler = Scaler(normalizing_range=normalizing_range, normalizing_method=normalizing_method,
                             eventID_channel_ind=eventID_channel_ind)


    # DATA PREPARATION -----------------------------------------------------------
    def create_dataset_many_to_one(self, data, network_inp_len=100, forecast_step_size=1):
        """
        Create dataset for LSTM model.

        Parameters:
        data (numpy array): Normalized EEG data.

        Returns:
        X (numpy array): Input data for LSTM.
        Y (numpy array): Output data for LSTM.
        """
        X, Y = [], []

        for i in range(len(data) - network_inp_len - forecast_step_size):
            X.append(data[i:(i + network_inp_len)])
            Y.append(data[i + network_inp_len + forecast_step_size - 1])
        return np.array(X), np.array(Y)

    def prepare_many_to_one_data(self, input_segments=None, network_inp_len=100, forecast_step_size=1,
                                 include_eventID=True, is_test_code=False, is_training=True):

        if input_segments is None:
            raise ValueError("[ERROR] Input segments are not defined.")

        # 1. RESHAPE THE DATA ----------------------------------------------------
        # 1.1 if multiple segments are included then concatenate them into one segment
        if isinstance(input_segments, list) and len(input_segments) > 1 and not is_test_code:
            segment = np.concatenate((input_segments[0], input_segments[1]), axis=1)
        else:
            segment = np.array(input_segments[0])
        # 1.1 if event ID is included then remove the event ID channel
        if segment.shape[0] > self.eventID_channel_ind:
            self.eventID_channel_data = segment[self.eventID_channel_ind, :]
            segment = np.delete(segment, self.eventID_channel_ind, axis=0)

        # 2. NORMALIZE THE DATA --------------------------------------------------
        if is_training:
            normalized_data = self.scaler.multi_channel_scaler(segment, is_training=True)
        else:
            normalized_data = self.scaler.multi_channel_scaler(segment, is_training=False)

        # 3. CREATE THE DATASET --------------------------------------------------

        X_channels, Y_channels = [], []
        for channel_data in normalized_data:
            X, Y = self.create_dataset_many_to_one(channel_data,
                                                   network_inp_len=network_inp_len,
                                                   forecast_step_size=forecast_step_size)
            X_channels.append(X)
            Y_channels.append(Y)

        return np.array(X_channels), np.array(Y_channels)


    # GETTERS -------------------------------------------------------------------

    def get_selected_segment(self, input_segments, required_channel_index):

        if isinstance(input_segments, list):
            np_segments = np.array(input_segments)
        else:
            np_segments = input_segments
        return np_segments[:, required_channel_index,:]

    def get_scaler(self):
        return self.scaler.get_scalers()

    def get_eventID_channel_data(self):
        return self.eventID_channel_data


    # PIPELINE ------------------------------------------------------------------

    def prepare_data_pipeline(self, segments, forecast_step_size=1, test_size=0.2, is_training=True,
                              required_channel_index=None, shuffle=False, random_seed=42, mode="many-to-one",
                              input_sequence_len=2000, is_test_code=False, split_data=True, include_eventID=True):

        # Check if the event ID channel index is set
        if include_eventID and self.eventID_channel_ind is None:
            raise ValueError("[ERROR] Event ID channel index is not set.")
        elif include_eventID and self.eventID_channel_ind == required_channel_index:
            raise ValueError("[ERROR] Event ID channel index and required channel index are the same.")

        # 1. Prepare the data

        if mode == "many-to-one":
            if required_channel_index is not None:
                raise ValueError("[ERROR] Required channel index mode is not finished yet")
            else:
                x, y = self.prepare_many_to_one_data(input_segments=segments, network_inp_len=input_sequence_len,
                                                     forecast_step_size=forecast_step_size, include_eventID=include_eventID,
                                                     is_test_code=is_test_code, is_training=is_training)

                print(f"   [INFO] Data preparation finished. X shape: {x.shape}, Y shape: {y.shape}")
                return x, y, self.get_eventID_channel_data(), self.get_scaler()
        else:
            raise ValueError(f"[ERROR] Mode {mode} is not defined.")
