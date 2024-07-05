import numpy as np
import plotly.graph_objects as go

from Logger import Logger
logger = Logger(__name__, code_file_name="Plot.py")

class Plot:
    def __init__(self):
        pass

    def save_plot(self, fig, save_path):
        try:
            fig.write_image(save_path)
        except Exception as e:
            logger.error(f"[ValueError] Error in saving plot: {e}")
            raise

    def save_plot_html(self, fig, save_path):
        try:
            fig.write_html(save_path)
        except Exception as e:
            logger.error(f"[ValueError] Error in saving plot as HTML: {e}")
            raise

    def plot_pred_vs_true(self, y_true, y_pred, save_path=None, model_name=None, eventID=None, channel_id=None,
                          input_size=None, step_size=None, x_label=None, y_label=None, x_min=None, x_max=None, is_test_code=False):
        """
        Plots the predicted values against the true values.

        :param y_true: (numpy array) True values.
        :param y_pred: (numpy array) Predicted values.
        :param save_path: (str) Path to save the plot. Optional.
        :param model_name: (str) Name of the model. Optional.
        :param eventID: (numpy array) Event ID channel data. Optional.
        :param channel_id: (int) Channel ID. Optional.
        :param input_size: (int) Input size. Optional.
        :param step_size: (int) Step size. Optional.
        :param x_label: (str) X-axis label. Optional.
        :param y_label: (str) Y-axis label. Optional.
        :param x_min: (int) Minimum x-axis value. Optional.
        :param x_max: (int) Maximum x-axis value. Optional.
        :param is_test_code: (bool) Whether to display the plot. Default is False.
        """
        # Input checks
        if not isinstance(y_true, np.ndarray):
            logger.error("[ValueError] y_true should be a numpy array.")
            raise
        if not isinstance(y_pred, np.ndarray):
            logger.error("[ValueError] y_pred should be a numpy array.")
            raise
        if save_path and not isinstance(save_path, str):
            logger.error("[ValueError] save_path should be a string.")
            raise
        if model_name and not isinstance(model_name, str):
            logger.error("[ValueError] model_name should be a string.")
            raise
        if eventID and not isinstance(eventID, np.ndarray):
            logger.error("[ValueError] eventID should be a numpy array.")
            raise
        if channel_id and not isinstance(channel_id, int):
            logger.error("[ValueError] channel_id should be an integer.")
            raise
        if input_size and not isinstance(input_size, int):
            logger.error("[ValueError] input_size should be an integer.")
            raise
        if step_size and not isinstance(step_size, int):
            logger.error("[ValueError] step_size should be an integer.")
            raise
        if x_label and not isinstance(x_label, str):
            logger.error("[ValueError] x_label should be a string.")
            raise
        if y_label and not isinstance(y_label, str):
            logger.error("[ValueError] y_label should be a string.")
            raise
        if x_min and not isinstance(x_min, int):
            logger.error("[ValueError] x_min should be an integer.")
            raise
        if x_max and not isinstance(x_max, int):
            logger.error("[ValueError] x_max should be an integer.")
            raise
        if not isinstance(is_test_code, bool):
            logger.error("[ValueError] is_test_code should be a boolean.")
            raise

        if x_min is None:
            x_min = 0
        if x_max is None:
            x_max = len(y_true)

        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        logger.info(f"y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")

        if model_name not in [None, 'Unknown Model'] and channel_id is not None and input_size is not None and step_size is not None:
            title = f"True vs Predicted Values \n {model_name} - Channel {channel_id} - Input Size: {input_size} - Step Size: {step_size}"
        else:
            title = "True vs Predicted Values"

        if x_label is None:
            x_label = "Samples"
        if y_label is None:
            y_label = "Value"

        fig = go.Figure()

        fig.add_trace(go.Scatter(y=y_true, mode='lines', name='True', line=dict(color='black', width=0.5)))
        fig.add_trace(go.Scatter(y=y_pred, mode='lines', name='Predicted', line=dict(color='red', width=0.5, dash='dash')))

        if input_size:
            fig.add_vrect(x0=0, x1=input_size-1, fillcolor='green', opacity=0.4, layer='below', line_width=0, annotation_text='Input Size')
        if step_size:
            fig.add_vrect(x0=input_size-1, x1=input_size + step_size-1, fillcolor='red', opacity=0.4, layer='below', line_width=0, annotation_text='Step Size')

        fig.update_layout(
            title=title,
            xaxis=dict(
                title=x_label,
                range=[x_min, x_max],
                showgrid=True,  # Add this line to show x-axis grid lines
                gridcolor='lightgrey',  # Color of the grid lines
                gridwidth=1  # Width of the grid lines
            ),
            yaxis=dict(
                title=y_label,
                showgrid=True,  # Add this line to show y-axis grid lines
                gridcolor='lightgrey',  # Color of the grid lines
                gridwidth=1  # Width of the grid lines
            ),
            legend=dict(orientation='h')
        )

        if save_path:
            file_name = f"{model_name}_Channel_{channel_id}_Input_{input_size}_Step_{step_size}_Pred_vs_True"
            file_path_png = f"{save_path}/{file_name}.png"
            file_path_html = f"{save_path}/{file_name}.html"
            self.save_plot(fig, file_path_png)
            self.save_plot_html(fig, file_path_html)

        if is_test_code:
            fig.show()
