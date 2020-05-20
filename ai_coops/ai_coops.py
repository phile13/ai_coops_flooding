# -*- coding: utf-8 -*-
import json
import numpy as np
from keras.models import model_from_json


def load_model(json_model_filename, h5_model_weights_filename):
    """
    load_model(json_model_filename, h5_model_weights_filename) -- loads keras model and weights

    Parameters
    ----------
    json_model_filename : STRING
        Name of the json model file.
    h5_model_weights_filename : STRING
        Name of the .h5 weights file.
    Returns
    -------
    Keras Model if successful or
    None if not successful
    """

    try:
        loaded_model = None
        with open(json_model_filename, 'r') as json_file:
            json_model = json.load(json_file)
            loaded_model = model_from_json(json_model)

        loaded_model.load_weights(h5_model_weights_filename)
        return loaded_model
    except Exception:
        return None


def save_model(model, json_model_filename, h5_model_weights_filename):
    """
    load_model(json_model_filename, h5_model_weights_filename) -- loads keras model and weights

    Parameters
    ----------
        model : Keras Model
            Keras Model object that you want to save
        json_model_filename : STRING
            Name of the json model file.
        h5_model_weights_filename : STRING
            Name of the .h5 weights file.
    Returns
    -------
        Boolean
            If the files were successfully saved
    """

    try:
        model_json = model.to_json()
        with open(json_model_filename, "w") as json_file:
            json_file.write(model_json)

        model.save_weights(h5_model_weights_filename)
        return True
    except Exception:
        return False


def create_sliding_window_dataset(data, window_size, x_start=0, x_stop=-2, y_start=-2, y_stop=None):
    """
    Parameters
    ----------
    data : Array of Data
        Array of Data.
    window_size : TYPE
        The size of window which you will slide across the dataset.
    x_start : TYPE, optional
        Start of Slice for the X data. The default is 0.
    x_stop : TYPE, optional
        End of Slice for the X data. The default is -2.
    y_start : TYPE, optional
        Start of Slice for the y data.  The default is -2.
    y_stop : TYPE, optional
        End of Slice for the y data. The default is None.

    Returns
    -------
    2D Tuple of X data and y data
        This is the windowed model X and y data.

    """
    X = []
    Y = []
    num_records = len(data)
    for outer_index in range(num_records-window_size):
        X.append(np.array(data[outer_index:outer_index+window_size, x_start:x_stop]))
        Y.append(np.array(data[outer_index:outer_index+window_size, y_start:y_stop]))
    return (np.array(X), np.array(Y))


def read_station_csv(csv_filename):
    """

    Parameters
    ----------
    csv_filename : STRING
        Name of csv file to read in.

    Returns
    -------
    Numpy Array
        Station Data.

    """
    use_cols = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    return np.loadtxt(csv_filename, delimiter=',', skiprows=1, usecols=use_cols, converters={2: convert_sensor_used, 14: convert_if_verified_null})


def read_part_station_csv(csv_filename, start=0, length=1000):
    """

    Parameters
    ----------
    csv_filename : STRING
        Name of csv file to read in.
    start : INT
        Where to start grabbing rows, not including skipped header row. The default is 0.
    length : INT
        How many rows to grab.  The default is 1000.

    Returns
    -------
    Numpy Array
        Station Data.

    """
    use_cols = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    return np.loadtxt(csv_filename, delimiter=',', skiprows=(start+1), max_rows=length, usecols=use_cols, converters={2: convert_sensor_used, 14: convert_if_verified_null})


def convert_sensor_used(sensor_used):
    mapping = {b"A1": 1.0, b"Y1": 2.0, b"N1": 3.0, b"NT": 4.0}

    if sensor_used in mapping:
        return mapping[sensor_used]
    else:
        return 0


def convert_if_verified_null(verified):
    return float(verified) if verified != b'' else 0.0
