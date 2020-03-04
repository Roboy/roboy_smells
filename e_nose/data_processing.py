# data_processing_old.py
from typing import List, Optional, Tuple, Mapping, Dict

import numpy as np
from .measurements import Measurement, DataRowsSet_t, WorkingChannels_t, Functionalisations_t, StandardizationType


def standardize_measurements(measurements: List[Measurement],
                             type: StandardizationType = StandardizationType.LAST_REFERENCE,
                             remove_ref: bool = True) -> List[Measurement]:
    """ Sets the standardization of all measurement according to the given parameter """

    if type == StandardizationType.LAST_REFERENCE:
        return standardize_measurements_lastref(measurements, remove_ref)

    elif type == StandardizationType.LOWPASS_FILTER:
        return standardize_measurements_lowpass(measurements, remove_ref)

    elif type == StandardizationType.BEGINNING_AVG:
        clean_measurements = []
        for measurement in measurements:
            measurement.set_reference(None, StandardizationType.BEGINNING_AVG)
            if not measurement.is_reference() or not remove_ref:
                clean_measurements.append(measurement)
        return clean_measurements

    raise ValueError('Unknown standardization type')


def standardize_measurements_lastref(measurements: List[Measurement], remove_ref: bool = True) \
        -> List[Measurement]:
    """ Sets the standardization of all measurement to the Reference Measurement before """
    last_null_meas = None
    clean_measurements = []

    for measurement in measurements:
        isref = measurement.is_reference()
        if isref:
            last_null_meas = measurement

        measurement.set_reference(last_null_meas, StandardizationType.LAST_REFERENCE)

        if not isref or not remove_ref:
            if last_null_meas is None:
                raise ValueError("ERROR - NO NULL MEASUREMENT FOUND")
            clean_measurements.append(measurement)

    return clean_measurements


def standardize_measurements_lowpass(measurements: List[Measurement], remove_ref: bool = True) \
        -> List[Measurement]:
    """ Sets the standardization of all measurement to the lowpass of all LOG data before (needs continuous data!) """
    clean_measurements = []

    l1_filter = None
    l1_factor = 1e-3

    for measurement in measurements:
        isref = measurement.is_reference()

        data = measurement.get_data(False, force=True, log=True, only_working=False)

        # Init filter to avg of first five measurements
        if l1_filter is None:
            l1_filter = np.mean(data[:5], axis=0)

        measurement.set_reference(l1_filter, StandardizationType.LOWPASS_FILTER)

        for i in range(len(data)):
            l1_filter = (l1_filter + data[i] * l1_factor) / (1.0 + l1_factor)

        if not isref or not remove_ref:
            clean_measurements.append(measurement)

    return clean_measurements


def remove_ref(measurements: List[Measurement]) -> List[Measurement]:
    """ Removes Reference Measurements from the list """
    clean_measurements = []
    for measurement in measurements:
        if measurement.label == 'ref' or measurement.label == 'null':
            continue
        else:
            clean_measurements.append(measurement)
    return clean_measurements


def remove_broken_channels(functionalisations: Functionalisations_t, working_channels: WorkingChannels_t,
                           data: DataRowsSet_t) \
        -> Tuple[Functionalisations_t, WorkingChannels_t, DataRowsSet_t]:
    """
    Removes all channels that are tagged as not working from the given dataset
    :param functionalisations:
    :param working_channels:
    :param data:
    :return:
    """
    functionalisations = np.array(functionalisations)[working_channels]
    for measurement in data:
        data[measurement]['channels'] = np.array(data[measurement]['channels'])[working_channels]
    working_channels = np.array(working_channels)[working_channels]
    return functionalisations, working_channels, data


def remove_broken_channels_multi_files(
        data_tuple: Tuple[Functionalisations_t, WorkingChannels_t, Dict[str, DataRowsSet_t]]) \
        -> Tuple[Functionalisations_t, WorkingChannels_t, Mapping[str, DataRowsSet_t]]:
    """
    Removes all channels that are tagged as not working from the given MULTIPLE datasets
    :param data_tuple: Data Tuple as returned by read_all_files_in_folder
    :return:
    """
    functionalisations, working_channels, all_data = data_tuple
    for file in all_data:
        for measurement in all_data[file]:
            all_data[file][measurement]['channels'] = np.array(all_data[file][measurement]['channels'])[
                working_channels]
    functionalisations = np.array(functionalisations)[working_channels]
    working_channels = np.array(working_channels)[working_channels]
    return functionalisations, working_channels, all_data


def get_labeled_measurements(data: DataRowsSet_t, correct_channels: WorkingChannels_t,
                             functionalisations: Functionalisations_t, debug=False) \
        -> List[Measurement]:
    """
    Extracts the individual Measurements from a DataRowSet by splitting every time the label changes
    :param data:
    :param correct_channels:
    :param functionalisations:
    :param debug:
    :return:
    """
    current_label = ''
    current_measurement: Optional[np.ndarray] = None
    current_temperature = 0
    current_gas = 0
    current_humidity = 0
    current_pressure = 0
    current_altitude = 0
    time_stamp = ''
    measurements: List[Measurement] = []

    for ts in data:
        row_data = data[ts]
        if debug:
            # print("row data label:",row_data['label'])
            # print("current label:", current_label)
            same = current_label == row_data['label']
            issame = current_label is row_data['label']
            # print("same:",same,"; issame:",issame)

        if current_label != row_data['label']:
            # change of labels
            if debug:
                print("change in labels; cl:", current_label, " - rdl:", row_data['label'])

            if current_label != '':
                if debug:
                    print("new measurement; cl:", current_label, " - rdl:", row_data['label'])
                meas = Measurement(current_measurement, current_label, time_stamp, correct_channels, functionalisations,
                                   current_temperature, current_gas, current_humidity, current_pressure,
                                   current_altitude)
                measurements.append(meas)

            current_label = row_data['label']
            current_temperature = row_data['temperature']
            current_gas = row_data['gas']
            current_humidity = row_data['humidity']
            current_pressure = row_data['pressure']
            current_altitude = row_data['altitude']
            time_stamp = ts
            current_measurement = None

        if current_label != '' and current_label == row_data['label']:
            if current_measurement is None:
                current_measurement = row_data['channels']
            else:
                current_measurement = np.vstack((current_measurement, row_data['channels']))

    if current_label != '':
        meas = Measurement(current_measurement, current_label, time_stamp, correct_channels, functionalisations,
                           current_temperature, current_gas, current_humidity, current_pressure, current_altitude)
        measurements.append(meas)

    return measurements


def high_pass_logdata(data: np.ndarray, init: Optional[np.ndarray] = None) -> np.ndarray:
    """ Filters out slow trends from logarithmic data; zeroes the avg of the first 5 samples.
     WILL NOT modify the passed array in-place"""

    out_data = np.copy(data)
    l1_filter = np.mean(data[:3], axis=0)
    if init is not None:
        l1_filter = init
    l1_factor = 1e-3
    for i in range(len(data)):
        l1_filter = (l1_filter + data[i] * l1_factor) / (1.0 + l1_factor)
        out_data[i] -= l1_filter
    return out_data


def get_measurement_peak_average(data: np.ndarray, num_samples=10) \
        -> np.ndarray:
    max_index = np.argmax(np.abs(data), axis=0)
    # get adjecent samples
    all_peak_data = []

    for i, max_i in enumerate(max_index):
        add_right = max_i + int(num_samples / 2) + 1
        add_left = max_i - int(num_samples / 2)

        if max_i >= (len(data) - num_samples / 2):
            add_right = len(data) - 1

        if max_i < (num_samples / 2):
            add_left = 0

        peak_data = np.mean(np.array(data[add_left:add_right, i]))
        all_peak_data.append(peak_data)

    return np.array(all_peak_data)


def group_meas_data_by_functionalisation(data: np.ndarray, functionalisations: Functionalisations_t) \
        -> np.ndarray:
    averages = None
    for row in data:
        if averages is None:
            _, averages = group_row_data_by_functionalities(row, functionalisations)
        else:
            _, tmp = group_row_data_by_functionalities(row, functionalisations)
            averages = np.vstack((averages, tmp))

    return averages


def group_row_data_by_functionalities(row: np.ndarray, functionalities: Functionalisations_t) \
        -> Tuple[Mapping[int, Dict[str, List[float]]], List[float]]:
    """
    Groups a single Row of e_nose measurement data by their functionalities
    :param row:
    :param functionalities:
    :return: Tuple consisting of values grouped by functionalization, averages by functionalization
    """
    grouped_data: Mapping[int, Dict[str, List[float]]] = {}
    averaged_data: List[float] = [0] * (1 + np.max(list(map(int, functionalities))))
    for i in np.unique(functionalities):
        grouped_data[int(i)] = {'values': []}

    for value, function in np.vstack((row, functionalities)).T:
        grouped_data[int(function)]['values'].append(float(value))
        averaged_data[int(function)] = averaged_data[int(function)] + float(value)

    for i in np.unique(functionalities):
        averaged_data[int(i)] = averaged_data[int(i)] / len(grouped_data[i]['values'])

    return grouped_data, averaged_data
