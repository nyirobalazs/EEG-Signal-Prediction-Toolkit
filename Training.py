import MLNetworks
import MLEngine
from TrainDataPrepare import TrainDataPrepare
from Saver import Saver
from tqdm import tqdm

from plot import Plot
from Logger import Logger
logger = Logger(__name__, code_file_name="Training.py")


class Training:
    """
    The Training class is responsible for setting up and executing the training pipeline for a given model.

    Attributes:
        eventID_channel_ind (int): The index of the event ID channel.
        test_size (float): The proportion of the dataset to include in the test split.
        is_normalize (bool): Whether to normalize the data.
        mode (str): The mode of operation, either "many-to-one" or another specified mode.
        train_program (dict): The training program configuration. Format: {'model_name': {'input_sizes': [], 'step_sizes': []}}
        main_saving_path (str): The main path where the training results will be saved.
        is_test_code (bool): Whether the code is being run in test mode.
        train_prep (TrainDataPrepare): An instance of the TrainDataPrepare class.
        saver (Saver): An instance of the Saver class.
    """

    def __init__(self,
                 eventID_channel_ind=3,
                 fs=250,
                 test_size=0.2,
                 is_normalize=True,
                 mode="many-to-one",
                 train_program=None,
                 main_saving_path=None,
                 is_test_code=False,
                 epochs=10,
                 batch_size=32,
                 normalizing_range=(0, 1),
                 normalizing_method="minmax",
                 strategy=None,
                 loss_function=None,
                 initial_learning_rate=0.6,
                 decay_steps=10000,
                 decay_rate=0.98,
                 staircase=True,
                 early_stopping_patience=50,
                 loss_monitor='val_loss',
                 loss_mode='min',
                 reduce_lr_factor=0.1,
                 reduce_lr_patience=50,
                 min_lr=0.01,
                 dropout_rate=0.2):

        self.strategy = strategy

        self.eventID_channel_ind = eventID_channel_ind
        self.fs = fs
        self.test_size = test_size
        self.is_normalize = is_normalize
        self.mode = mode
        self.train_program = self.training_program_formater(train_program)
        self.main_saving_path = main_saving_path
        self.is_test_code = is_test_code
        self.epochs = epochs
        self.batch_size = batch_size

        self.loss_function = loss_function
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase
        self.early_stopping_patience = early_stopping_patience
        self.loss_monitor = loss_monitor
        self.loss_mode = loss_mode
        self.reduce_lr_factor = reduce_lr_factor
        self.reduce_lr_patience = reduce_lr_patience
        self.min_lr = min_lr
        self.dropout_rate = dropout_rate

        self.check_train_program()
        self.train_prep = TrainDataPrepare(eventID_channel_ind=eventID_channel_ind,
                                           normalizing_range=normalizing_range,
                                           normalizing_method=normalizing_method)
        self.saver = Saver()
        self.plotter = Plot()

    def training_program_formater(self, train_program):
        try:
            for model_name, model_details in train_program.items():
                for key, value in model_details.items():
                    if key == 'input_sizes' or key == 'step_sizes':
                        for i, val in enumerate(value):
                            if str(val)[-1] == 's':
                                train_program[model_name][key][i] = self.convert_sec_to_samples(int(val[:-1]), self.fs)
            return train_program
        except Exception as e:
            logger.error(f"Error in training program formater: {e}. Training program should be in the format: {{'model_name': {{'input_sizes': [], 'step_sizes': []}}}}. The input sizes and step sizes should be in samples or seconds with 's' at the end.")
            return None

    def convert_sec_to_samples(self, sec, fs):
        """
        Convert seconds to samples.

        Args:
            sec (int): The number of seconds.
            fs (int): The sampling frequency.

        Returns:
            int: The number of samples.
        """
        return sec * fs

    def check_train_program(self):
        """
        Checks the training program configuration for any missing or incorrect values.
        Raises a ValueError if any issues are found.
        """

        if self.train_program is None:
            raise ValueError("[ERROR] The train_program is not defined. Please define the train_program.")
        else:
            for model_name, model_details in self.train_program.items():
                if 'step_sizes' not in model_details.keys():
                    logger.error(f"[ValueError] Step sizes are not defined for the model {model_name}. Please define the step sizes.")
                    raise
                if 'input_sizes' not in model_details.keys():
                    logger.error(f"[ValueError] Input sizes are not defined for the model {model_name}. Please define the input sizes.")
                    raise
                if len(model_details['input_sizes']) > 1 and len(model_details['input_sizes']) != len(
                        model_details['step_sizes']):
                    logger.error(f"[ValueError] The number of input sizes should be equal to the number of step sizes for the model {model_name}.")
                    raise

    @staticmethod
    def zip_input_step_sizes(model_details):
        """
        Zips the input sizes and step sizes for a given model.

        Args:
            model_details (dict): The details of the model. Format: {'input_sizes': [], 'step_sizes': []}

        Returns:
            list: A list of tuples containing the input sizes and step sizes.
        """
        if len(model_details['input_sizes']) == 1:
            return [(model_details['input_sizes'][0], step_size) for step_size in model_details['step_sizes']]
        elif len(model_details['input_sizes']) == len(model_details['step_sizes']):
            # print zip object elements
            return list(zip(model_details['input_sizes'], model_details['step_sizes']))
        else:
            logger.error(f"[ValueError] The number of input sizes should be equal to the number of step sizes.")
            raise

    def training_pipeline(self, segments_dict):
        """
        Executes the training pipeline for each model in the training program.

        Args:
            segments_dict (dict): A dictionary containing the data segments.
        """
        # Calculate total iterations
        number_of_models = len(self.train_program)
        sum_of_input_step_sizes = sum([len(self.zip_input_step_sizes(model_details)) for model_details in self.train_program.values()])
        number_of_channels = segments_dict[list(segments_dict.keys())[0]][0].shape[0]
        number_of_sides = 2
        total_iterations = number_of_models * sum_of_input_step_sizes * number_of_channels * number_of_sides

        # Create a progress bar
        # Create a progress bar
        pbar = tqdm(total=total_iterations, desc=f">>> Training pipeline for {len(self.train_program)} models.",
                    bar_format='{l_bar}%s{bar}%s{r_bar}' % ('\033[37m', '\033[0m'))

        # 1. Create the main directory
        main_directory = self.saver.create_directory(self.main_saving_path, experiment_folder=True)

        # 2. Loop through the models
        for model_name, model_details in self.train_program.items():
            # 2.1 Create the model directory
            model_directory = self.saver.create_directory(main_directory, experiment_folder=False, folder_name=model_name)

            # 2.2 Create the evaluation directory
            evaluation_directory = self.saver.create_directory(model_directory, experiment_folder=False, folder_name="Results")

            # 3. Loop through the input sizes and step sizes for each model
            for input_size, step_size in self.zip_input_step_sizes(model_details):

                logger.trial_settings(model_name=model_name, input_size=input_size, step_size=step_size)

                eval_result_dicts = {}

                # 4. Loop through the sides.
                for side in ['Left', 'Right']:

                    # 4.1 Get the segments for the side
                    train_segment = segments_dict[f'x{side}']

                    # 4.2 Prepare the data
                    X_channels, Y_channels, _, scalers = self.train_prep.prepare_data_pipeline(segments=train_segment,
                                                                                               forecast_step_size=step_size,
                                                                                               test_size=self.test_size,
                                                                                               is_training=True,
                                                                                               required_channel_index=None,
                                                                                               shuffle=True,
                                                                                               random_seed=42, mode=self.mode,
                                                                                               input_sequence_len=input_size,
                                                                                               is_test_code=self.is_test_code,
                                                                                               include_eventID=False)

                    eval_segment = segments_dict[f'trainY']
                    evalX_channels, evalY_channels, eventID_channel, _ = self.train_prep.prepare_data_pipeline(segments=eval_segment,
                                                                                                               forecast_step_size=step_size,
                                                                                                               test_size=self.test_size,
                                                                                                               is_training=False,
                                                                                                               required_channel_index=None,
                                                                                                               shuffle=False,
                                                                                                               random_seed=42,
                                                                                                               mode=self.mode,
                                                                                                               input_sequence_len=input_size,
                                                                                                               is_test_code=self.is_test_code,
                                                                                                               split_data=False)

                    # 5. Loop through the channels and train and evaluate the model
                    for channel_index in range(segments_dict[list(segments_dict.keys())[0]][0].shape[0]):
                        # Exclude the event ID channel
                        if channel_index != self.eventID_channel_ind:

                            # Log tqdm progress
                            logger.info(f"Training model for channel {channel_index} | side {side} | channel {channel_index}.")

                            # 5.1 Create the model name for saving
                            model_name_side = f"{model_name}_input_{input_size}_step_{step_size}_channel_{channel_index}_side_{side}"

                            # 5.2 Reshape required channel data
                            X = X_channels[channel_index].reshape(-1, input_size, 1)
                            y = Y_channels[channel_index]

                            # 5.3 Train the model
                            with self.strategy.scope():

                                logger.info(f"Training model for channel {channel_index} and side {side}.")

                                model = MLNetworks.load_model(model_name=model_name,
                                                              input_shape=(input_size, 1),
                                                              output_layer_dim=1,
                                                              dropout_rate=self.dropout_rate)

                                engine = MLEngine.MLEngine(model=model,
                                                           strategy=self.strategy,
                                                           initial_learning_rate=self.initial_learning_rate,
                                                           decay_steps=self.decay_steps,
                                                           decay_rate=self.decay_rate,
                                                           staircase=self.staircase,
                                                           is_test_code=self.is_test_code,
                                                           early_stopping_patience=self.early_stopping_patience,
                                                           loss_monitor=self.loss_monitor,
                                                           loss_mode=self.loss_function,
                                                           reduce_lr_factor=self.reduce_lr_factor,
                                                           reduce_lr_patience=self.reduce_lr_patience,
                                                           min_lr=self.min_lr)

                            engine.train_model(X, y,
                                               batch_size=self.batch_size,
                                               epochs=self.epochs,
                                               save_path=model_directory,
                                               model_name=model_name_side,
                                               split_ratio=self.test_size)

                            # 6. Evaluate the model
                            # 6.1 Prepare the evaluation data
                            evalX = evalX_channels[channel_index].reshape(-1, input_size, 1)
                            evalY = evalY_channels[channel_index]

                            # 6.2 Evaluate the model
                            evaluation = engine.evaluate_model(x_eval=evalX, y_eval=evalY,
                                                               input_size=input_size,
                                                               step_size=step_size,
                                                               TrainDataPrepare_class=self.train_prep,
                                                               channel_index=channel_index)

                            self.plotter.plot_pred_vs_true(y_true=eval_segment[0][channel_index, :], # Use the original data
                                                           y_pred=evaluation['predicted_fitted'],
                                                           save_path=evaluation_directory,
                                                           model_name=model_name,
                                                           channel_id=channel_index,
                                                           input_size=input_size,
                                                           step_size=step_size,
                                                           x_min=0,
                                                           x_max=500,
                                                           is_test_code=self.is_test_code)

                            eval_result_dicts[f'channel_{channel_index}_side_{side}'] = evaluation

                            logger.info(f"Model evaluated for channel {channel_index} and side {side}.")

                            # Update the progress bar
                            pbar.update(1)

                # 7. Save the evaluation results
                # 7.1 Create evaluation subdirectory for the step size
                step_size_directory = self.saver.create_directory(evaluation_directory, experiment_folder=False, folder_name=f"Input_{input_size}_Step_{step_size}")

                # 7.2 Save the evaluation results
                # change values from 0 to input_size and from end to -step_size with zeros in eventID_channel
                eventID_channel = eventID_channel.reshape(-1, 1)
                eventID_channel[:input_size] = 0
                eventID_channel[-step_size:] = 0
                eventID_channel = eventID_channel.squeeze()
                self.saver.save_evaluations_to_mat(path=step_size_directory, evaluations=eval_result_dicts, eventID_channel=eventID_channel)

                # 7.3 Save the evaluation results to a CSV file into the main directory
                csv_model_name = f"{model_name}_input_{input_size}_step_{step_size}"
                self.saver.save_evaluations_to_csv(path=main_directory, evaluations=eval_result_dicts, model_name=csv_model_name)

        # Close the progress bar
        pbar.close()
