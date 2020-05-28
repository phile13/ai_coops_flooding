# -*- coding: utf-8 -*-
import numpy as np
from os import path
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
        with open(json_model_filename, 'r') as json_file:
            loaded_model = model_from_json(json_file.read())

            loaded_model.load_weights(h5_model_weights_filename)
            return loaded_model
        return None
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

        if path.exists(json_model_filename) is False:
            return False

        model.save_weights(h5_model_weights_filename)
        return path.exists(h5_model_weights_filename)
    except Exception:
        return False


def create_sliding_window_dataset(data, window_size, x_cols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), y_cols=(12, 13)):
    """

    Parameters
    ----------
    data : Array of Data
        Array of Data.
    window_size : INT
        The size of window which you will slide across the dataset.
    x_cols : Array INT, optional
        Array of columns to use. The default is (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11).
    y_cols : Array INT, optional
        Array of columns to use. The default is (12, 13).

    Returns
    -------
    2D Tuple of X data and a single y data point
        This is the windowed model X and y data.

    """
    X = []
    Y = []
    num_records = len(data)
    for outer_index in range(num_records-window_size):
        X.append(np.array(data[outer_index:outer_index+window_size, x_cols]))
        Y.append(np.array(data[outer_index:outer_index+window_size, y_cols]))
    return (np.array(X), np.array(Y))


def missing_y_create_sliding_window_dataset(data, window_size, y_col=-1):
    """
    Parameters
    ----------
    data : Array of Data
        Array of Data.
    window_size : INT
        The size of window which you will slide across the dataset.
    y_col : INT
        The col index of the y value

    Returns
    -------
    2D Tuple of X data and a single y data point
        This is the windowed model X and y data. The y value that is to be predicted is set to -9999

    """
    X = []
    Y = []
    num_records = len(data)
    for outer_index in range(num_records-window_size):
        row = np.array(data[outer_index:outer_index+window_size, :])
        y = row[-1][y_col]
        row[-1][y_col] = -9999
        X.append(row)
        Y.append(y)
    return (np.array(X), np.array(Y))


def read_station_csv(csv_filename, use_cols=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)):
    """

    Parameters
    ----------
    csv_filename : STRING
        Name of csv file to read in.
    use_cols : Tuple of INTs. The default is (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15).
        list of cols to be pulled from csv

    Returns
    -------
    Numpy Array
        Station Data.

    """
    return np.loadtxt(csv_filename, delimiter=',', skiprows=1, usecols=use_cols, converters={2: convert_sensor_used, 14: convert_if_verified_null})


def read_part_station_csv(csv_filename, start=0, length=1000, use_cols=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)):
    """

    Parameters
    ----------
    csv_filename : STRING
        Name of csv file to read in.
    start : INT
        Where to start grabbing rows, not including skipped header row. The default is 0.
    length : INT
        How many rows to grab.  The default is 1000.
    use_cols : Tuple of INTs
        list of cols to be pulled from csv. The default is (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15).
    Returns
    -------
    Numpy Array
        Station Data.

    """
    return np.loadtxt(csv_filename, delimiter=',', skiprows=(start+1), max_rows=length, usecols=use_cols, converters={2: convert_sensor_used, 14: convert_if_verified_null})


def convert_sensor_used(sensor_used):
    mapping = {b"A1": 1.0, b"Y1": 2.0, b"N1": 3.0, b"NT": 4.0}

    if sensor_used in mapping:
        return mapping[sensor_used]
    else:
        return 0


def convert_if_verified_null(verified):
    return float(verified) if verified != b'' else 0.0


def shuffle_identically(x, y):
    """

    Parameters
    ----------
    x : Numpy Array
        Array 1.
    y : Numpy Array
        Array 2.

    Returns
    -------
    TYPE
        Shuffled Array 1.
    TYPE
        Shuffled Array 2.

    """
    shuffled_indicies = np.arange(x.shape[0])
    np.random.shuffle(shuffled_indicies)
    return (x[shuffled_indicies], y[shuffled_indicies])


def station_data_generator(station_csv_filenames, window_size, data_start, data_size, shuffle=True, x_cols=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], y_cols=[14, 15], reshape_x=None, reshape_y=None):
    """

    Parameters
    ----------
    station_csv_filenames : String
        Name of CSV file with Station Data.
    window_size : Int
        Size of sliding window.
    data_start : Int
        Starting row in data.
    data_size : Int
        Number of rows to collect from data.
    shuffle : TYPE, optional
        Whether to shuffle the windowed data that is yielded. The default is True.
    x_cols : List of INTs
        list of cols that are the x values
    y_cols : List of INTs
        list of cols that are the y values
    reshape_x : Tuple
        reshape tuple. Default None, means do not reshape x
    reshape_y : Tuple
        reshape tuple. Default None, means do not reshape y
    Yields
    ------
    x : Numpy Array
        X Data.
    y : Numpy Array
        Y Data.

    """
    use_cols = sorted(x_cols + y_cols)
    x_cols_adj, y_cols_adj = adjust_cols(use_cols, x_cols, y_cols)
    while True:
        for station_csv_filename in station_csv_filenames:
            data = read_part_station_csv(station_csv_filename, data_start, data_size, use_cols)
            x, y = create_sliding_window_dataset(data, window_size, x_cols_adj, y_cols_adj)
            if shuffle:
                x, y = shuffle_identically(x, y)
            if reshape_x:
                x = x.reshape(reshape_x)
            if reshape_y:
                y = y.reshape(reshape_y)
            yield x, y


def missing_y_station_data_generator(station_csv_filenames, window_size, data_start, data_size, shuffle=True, x_cols=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], y_col=14, true_pairs=None):
    """

    Parameters
    ----------
    station_csv_filenames : String
        Name of CSV file with Station Data.
    window_size : Int
        Size of sliding window.
    data_start : Int
        Starting row in data.
    data_size : Int
        Number of rows to collect from data.
    shuffle : TYPE, optional
        Whether to shuffle the windowed data that is yielded. The default is True.
    x_cols : List of INTs
        list of cols that are the x values
    y_col : List of INTs
        list of cols that are the y values
    true_pairs : 2D array of INTs
        Sets row[true_pair[1]] = 9999 if that row[true_pair[0]] == 0
    Yields
    ------
    x : Numpy Array
        X Data.
    y : Numpy Array
        Y Data.

    """
    use_cols = sorted(x_cols)
    while True:
        for station_csv_filename in station_csv_filenames:
            data = read_part_station_csv(station_csv_filename, data_start, data_size, use_cols)
            if true_pairs is not None:
                data = fix_missing_y_data(data, use_cols, true_pairs)
            x, y = missing_y_create_sliding_window_dataset(data, window_size, x_cols.index(y_col))
            if shuffle:
                x, y = shuffle_identically(x, y)
            yield x, y


def adjust_cols(use_cols, x_cols, y_cols):
    x_cols_adj = []
    y_cols_adj = []
    for index in range(len(use_cols)):
        use_col = use_cols[index]
        try:
            x_cols.index(use_col)
            x_cols_adj.append(index)
        except Exception:
            pass
        try:
            y_cols.index(use_col)
            y_cols_adj.append(index)
        except Exception:
            pass
    return x_cols_adj, y_cols_adj


def fix_missing_y_data(data, use_cols, true_pairs):
    fixed_data = []
    true_pairs_adj = adjust_true_pairs(use_cols, true_pairs)
    for outer_index in range(len(data)):
        row = data[outer_index]
        for inner_index in range(len(true_pairs_adj)):
            if row[true_pairs_adj[inner_index][0]] == 0:
                row[true_pairs_adj[inner_index][1]] = 9999
        fixed_data.append(row)
    return fixed_data


def adjust_true_pairs(use_cols, true_pairs):
    true_pairs_adj = []
    for index in range(len(true_pairs)):
        true_pair = true_pairs[index]
        true_pairs_adj.append((use_cols.index(true_pair[0]), use_cols.index(true_pair[1])))
    return true_pairs_adj


def prep_station_data_generator(station_csv_filenames, window_size, data_start, data_size, shuffle=True):
    """

    Parameters
    ----------
    station_csv_filenames : String
        Name of CSV file with Station Data.
    window_size : Int
        Size of sliding window.
    data_start : Int
        Starting row in data.
    data_size : Int
        Number of rows to collect from data.
    shuffle : TYPE, optional
        Whether to shuffle the windowed data that is yielded. The default is True.

    Yields
    ------
    x : Numpy Array
        X Data.
    y : Numpy Array
        Y Data.

    """
    while True:
        for station_csv_filename in station_csv_filenames:
            data = read_part_station_csv(station_csv_filename, data_start, data_size)[0:-2]

            x, y = create_sliding_window_dataset(data, window_size)
            if shuffle:
                x, y = shuffle_identically(x, y)
            yield x, y
