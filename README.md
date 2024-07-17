# EEG-Signal-Prediction-Toolkit


This project is a machine learning pipeline for training and evaluating various models on EEG data. The models include Dense, LSTM, Bidirectional LSTM, and Hybrid networks. The pipeline includes data preprocessing, model training, model evaluation, and result saving.

## Table of Contents

- [Setup](#setup)
- [Usage](#usage)
- [Input and Output Files](#input-and-output-files)
- [Config.json File](#configjson-file)
- [Output Library Structure](#output-library-structure)
- [For Beginners](#for-beginners)

## Setup

To set up the project, you need to install the required Python packages. You can do this by running the following command in your terminal:

```bash
pip install -r requirements.txt
```

The required packages are listed in the `requirements.txt` file.

## Usage

To use the project, you need to run the `Training.py` script. This script sets up and executes the training pipeline for a given model. The script takes as input a dictionary of data segments and a training program configuration.

The training program configuration is a dictionary that specifies the models to train and their parameters. The format of the training program configuration is as follows:

```json
{
    "model_name": {
        "input_sizes": [],
        "step_sizes": []
    }
}
```

The `input_sizes` and `step_sizes` can be specified in samples or seconds with 's' at the end. For example, an input size of 1 second can be specified as "1s".

The `Training.py` script outputs the trained models and their evaluation results. The models are saved in the `models` directory and the evaluation results are saved in the `results` directory.

## Input and Output Files

The input to the project is a dictionary of data segments. Each data segment is a 2D numpy array where the rows represent channels and the columns represent time points. The data segments should be preprocessed and ready for training.

The output of the project includes the trained models and their evaluation results. The trained models are saved as `.h5` files in the `models` directory. The evaluation results include the mean absolute error (MAE), root mean squared error (RMSE), and R2 score for each model and are saved as `.csv` files in the `results` directory.

## Config.json File

The `config.json` file is used to configure the training pipeline. It includes parameters such as the event ID channel index, sampling frequency, test size, normalization settings, mode of operation, training program, main saving path, and test code flag.

Here is an example of a `config.json` file:

```json
{
    "eventID_channel_ind": 3,
    "fs": 250,
    "test_size": 0.2,
    "is_normalize": true,
    "mode": "many-to-one",
    "train_program": {
        "DenseNetwork": {
            "input_sizes": ["1s"],
            "step_sizes": ["0.5s"]
        }
    },
    "main_saving_path": "/path/to/save/directory",
    "is_test_code": false
}
```

You can modify the parameters in the `config.json` file to customize the training pipeline.

## Output Library Structure

The output library structure is organized as follows:

- `models`: This directory contains the trained models. Each model is saved as a `.h5` file.
- `results`: This directory contains the evaluation results. The results are saved as `.csv` files. Each file includes the MAE, RMSE, and R2 score for each model.

## For Beginners

If you are a beginner programmer, here are some steps to run the project:

1. Install Python on your computer. You can download Python from the official website: https://www.python.org/downloads/
2. Install the required Python packages. Open your terminal, navigate to the project directory, and run the command: `pip install -r requirements.txt`
3. Run the `Training.py` script. In your terminal, run the command: `python Training.py`
4. Check the `models` and `results` directories for the output.

Remember to modify the `config.json` file to customize the training pipeline. You can open the `config.json` file in a text editor, change the parameters, and save the file.

## Additional Steps

To run the provided code, you need to follow these steps:

1. **Install the necessary packages**: The required packages are listed in the `requirements.txt` file. You can install them using pip by running the command `pip install -r requirements.txt` in your terminal.

2. **Prepare your data**: The code expects your data to be in a specific format. The data should be a dictionary of segments, where each segment is a 2D numpy array. The dictionary keys should be 'xLeft', 'xRight', and 'trainY'. The 'xLeft' and 'xRight' keys correspond to the left and right sides of the data, respectively. The 'trainY' key corresponds to the training labels.

3. **Configure the settings**: The `config.json` file contains various settings for the training process. Here are some of the parameters you can change:

    - `eventID_channel_ind`: The index of the event ID channel.
    - `fs`: The sampling frequency.
    - `test_size`: The proportion of the dataset to include in the test split.
    - `is_normalize`: Whether to normalize the data.
    - `mode`: The mode of operation, either "many-to-one" or another specified mode.
    - `train_program`: The training program configuration. Format: {'model_name': {'input_sizes': [], 'step_sizes': []}}
    - `main_saving_path`: The main path where the training results will be saved.
    - `is_test_code`: Whether the code is being run in test mode.
    - `epochs`: The number of epochs for training.
    - `batch_size`: The batch size for training.
    - `normalizing_range`: The range for normalization.
    - `normalizing_method`: The method for normalization.
    - `strategy`: The strategy for training.

    Note: The `input_sizes` and `step_sizes` can be given in seconds by appending 's' to the end of the value. For example, '1s' would correspond to 1 second.

4. **Run the code**: Once you have installed the necessary packages, prepared your data, and configured the settings, you can run the code. The main function to call is `training_pipeline` from the `Training` class. This function takes the segments dictionary as input and runs the training process.

The output of the code includes:

- **Model files**: The trained models are saved in the directory specified by `main_saving_path` in the `config.json` file. The models are saved in a hierarchical structure, with a separate directory for each model, input size, and step size.

- **Evaluation results**: The evaluation results are saved in a .mat file and a .csv file. The .mat file contains the predicted values for each channel and side, along with the event ID channel. The .csv file contains various evaluation metrics for each channel and side, including the mean absolute error, root mean squared error, and R2 score.

- **Plots**: Plots of the true vs predicted values are also saved for each channel and side. These plots can help you visually assess the performance of the models.

Remember to always check the logs for any errors or warnings during the training process. The logs can provide valuable information about what might be going wrong.

## Project Files

The main files in this project include:

- `Loader.py`: Contains functions for loading and preprocessing the data.
- `Main.py`: The main entry point for the project.
- `Saver.py`: Contains functions for saving the models and evaluation results.
- `Training.py`: Contains the main training pipeline.

Ensure you have these files in your project directory to run the project successfully.

## Note

This project uses TensorFlow for building and training the models. It is recommended to run this project on a machine with a GPU for faster training.
```

This Markdown file provides a comprehensive guide to your project, covering setup, usage, configuration, input and output file formats, and additional details for beginners. Make sure to place this file as `README.md` in the root directory of your project on GitHub.
