import numpy as np
from scipy.signal import butter, lfilter, freqz
from scipy.fftpack import fft


class Preprocess:
    def __init__(self, eventID_channel_ind=3):
        self.segments = []
        self.eventID_channel_ind = eventID_channel_ind  # Event ID channel index
        self.segments_dict = {}

    # SOURCE FILE DIRECTORY TEST -----------------------------------------------
    def test_source_file_directory(self, source_file_directory):
        """
        Test source file directory. Must be a list of dictionaries containing 'trainX', 'trainY', 'testX', 'testY' keys.

        Parameters:
            source_file_directory (list): List of dictionaries containing 'trainX', 'trainY', 'testX', 'testY' keys.
        """
        # TODO: test for 'trainX', 'trainY', 'testX', 'testY' keys

        try:
            if not isinstance(source_file_directory, list):
                raise ValueError("[ERROR] Source file directory must be a list.")
            if not all([isinstance(file_data, dict) for file_data in source_file_directory]):
                raise ValueError("[ERROR] Source file directory must contain dictionaries.")

        except Exception as e:
            raise ValueError(f"[ERROR] Error in testing source file directory: {e}")

    # SEGMENT CUTTERS -----------------------------------------------------------
    def cut_segments(self, source_file_directory, trial_start_ids=None, mode='many-to-one', structure_name='trainY'):
        """
        Cut segments from source file directory based on the mode.
        mode: 'many-to-one' - Leave whole dataset together. All channels joined together from different source files.
              'event_segments' - Cut segments based on event segments.
        """

        if trial_start_ids is None:
            trial_start_ids = [768, 1023]
        if mode == 'many-to-one':
            for file_data in source_file_directory:
                source_structure = file_data[structure_name]
                self.segments.append(source_structure)
                # Break the loop if the structure name is 'trainY' to only test on one training set
                if structure_name == 'trainY':
                    break

        elif mode == 'event_segments':
            for file_data in source_file_directory:
                source_structure = file_data[structure_name]
                if self.eventID_channel_ind >= source_structure.shape[0]:
                    raise ValueError("[ERROR] Event ID channel index out of range.")
                else:
                    eventID_channel = source_structure[self.eventID_channel_ind, :]
                    indices = np.where(np.isin(eventID_channel, trial_start_ids))[0]
                    for num, index in enumerate(indices):
                        segment = source_structure[:, index:indices[num + 1] if num + 1 < len(indices) else None]
                        self.segments.append(segment)
        else:
            raise ValueError("[ERROR] Invalid mode. Choose either 'many-to-one' or 'event_segments'.")

    # SEGMENT FILTERS ----------------------------------------------------------
    def filter_failed_segments(self, failed_segment_ids):
        """
        Filter failed segments based on the failed segment IDs.
        Parameters:
            failed_segment_ids (list): List of failed segment IDs.
        """
        try:
            self.segments = [segment for segment in self.segments if
                             segment[self.eventID_channel_ind, 0] not in failed_segment_ids]
        except Exception as e:
            raise ValueError(f"[ERROR] Error in filtering failed segments: {e}")

    def filter_segments_by_trigger(self, trigger_ids):
        """
        Filter segments based on the trigger IDs.
        Parameters:
            trigger_ids (list): List of trigger IDs.
        """
        try:
            self.segments = [segment for segment in self.segments if
                             any(np.isin(segment[self.eventID_channel_ind, :], trigger_ids))]
        except Exception as e:
            raise ValueError(f"[ERROR] Error in filtering segments by trigger: {e}")

    # SIGNAL FILTERS -----------------------------------------------------------
    def butter_lowpass(self, cutoff, fs, order=5):
        try:
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            return b, a
        except Exception as e:
            raise ValueError(f"Error in butter_lowpass: {e}")

    def butter_highpass(self, cutoff, fs, order=5):
        try:
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='high', analog=False)
            return b, a
        except Exception as e:
            raise ValueError(f"[ERROR] Error in butter_highpass: {e}")

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        try:
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='band')
            return b, a
        except Exception as e:
            raise ValueError(f"[ERROR] Error in butter_bandpass: {e}")

    def butter_filter(self, data, b, a):
        try:
            y = lfilter(b, a, data)
            return y
        except Exception as e:
            raise ValueError(f"[ERROR] Error in butter_filter: {e}")

    def low_cut_filter(self, cutoff, fs, order=5):
        try:
            for segment in self.segments:
                b, a = self.butter_lowpass(cutoff, fs, order=order)
                for i in range(3):
                    segment[i, :] = self.butter_filter(segment[i, :], b, a)
        except Exception as e:
            raise ValueError(f"[ERROR] Error in low_cut_filter: {e}")

    def high_cut_filter(self, cutoff, fs, order=5):
        try:
            for segment in self.segments:
                b, a = self.butter_highpass(cutoff, fs, order=order)
                for i in range(3):
                    segment[i, :] = self.butter_filter(segment[i, :], b, a)
        except Exception as e:
            raise ValueError(f"[ERROR] Error in high_cut_filter: {e}")

    def bandpass_filter(self, lowcut, highcut, fs, order=5):
        try:
            for segment in self.segments:
                b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
                for i in range(3):
                    segment[i, :] = self.butter_filter(segment[i, :], b, a)
        except Exception as e:
            raise ValueError(f"[ERROR] Error in bandpass_filter: {e}")

    def linenoise_filter(self, linenoise, fs, order=5):
        try:
            self.bandpass_filter(linenoise - 1, linenoise + 1, fs, order)
        except Exception as e:
            raise ValueError(f"[ERROR] Error in linenoise_filter: {e}")

    def fast_fourier_transform(self):
        try:
            for segment in self.segments:
                for i in range(3):
                    segment[i, :] = fft(segment[i, :])
        except Exception as e:
            raise ValueError(f"[ERROR] Error in fast_fourier_transform: {e}")

    # GETTERS ------------------------------------------------------------------
    def get_segments_dict(self):
        return self.segments_dict

    def get_segments(self):
        return self.segments

    def clear_segments(self):
        self.segments = []

    # PREPROCESSING PIPELINE ----------------------------------------------------
    def preprocessing_pipeline(self, source_file_directory, trial_start_ids=None, failed_segment_ids=None,
                               trigger_ids=None, filters=None, structure_name_list=None, mode='event_segments'):
        """
        Executes the preprocessing pipeline on the provided source file directory.

        Parameters:
            source_file_directory (list): List of dictionaries containing 'trainX', 'trainY', 'testX', 'testY' keys.
            trial_start_ids (list, optional): List of trial start IDs. Defaults to [768, 1023].
            failed_segment_ids (list, optional): List of failed segment IDs. Defaults to an empty list.
            trigger_ids (list, optional): List of trigger IDs. Defaults to an empty list.
            filters (dict, optional): Dictionary of filters to apply. Defaults to an empty dictionary.
            structure_name_list (list, optional): List of structure names. Defaults to ['trainY'].
            mode (str, optional): Mode of operation. Can be 'many-to-one' or 'event_segments'. Defaults to 'event_segments'.

        Returns:
            dict: Dictionary of preprocessed segments.
        """
        if trial_start_ids is None:
            trial_start_ids = [768, 1023]
        if filters is None:
            filters = {}
        if trigger_ids is None:
            trigger_ids = []
        if failed_segment_ids is None:
            failed_segment_ids = []
        if structure_name_list is None:
            structure_name = ['trainY', ]

        # Test source file directory
        self.test_source_file_directory(source_file_directory)

        for structure_name in structure_name_list:
            # Cut segments from source file directory
            self.cut_segments(source_file_directory, trial_start_ids, mode, structure_name)

            if mode in ['event_segments']:
                # Apply filters to segments
                if failed_segment_ids:
                    self.filter_failed_segments(failed_segment_ids)
                if trigger_ids:
                    self.filter_segments_by_trigger(trigger_ids)

            # Apply signal filters defined in filters dictionary
            for filter_name, params in filters.items():
                if filter_name == 'low_cut':
                    self.low_cut_filter(*params)
                elif filter_name == 'high_cut':
                    self.high_cut_filter(*params)
                elif filter_name == 'bandpass':
                    self.bandpass_filter(*params)
                elif filter_name == 'linenoise':
                    self.linenoise_filter(*params)
                elif filter_name == 'fft':
                    self.fast_fourier_transform()

            self.segments_dict[structure_name] = self.get_segments()
            self.clear_segments()

        print(f"   [INFO] Preprocessing pipeline completed. Mode: {mode}. Number of segments: {len(self.segments_dict[structure_name_list[0]])}")

        return self.get_segments_dict()
