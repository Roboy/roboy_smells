# data_processing_old.py
import numpy as np
from .measurements import Measurement


def standardize_measurements(measurements):
    last_null_meas = None
    clean_measurements = []
    last_meas = None

    for measurement in measurements:
        if last_meas is None or last_meas != measurement.label:
            last_meas = measurement.label

        if measurement.label == 'ref' or measurement.label == 'null':
            last_null_meas = measurement
        else:
            if last_null_meas is None:
                print("ERROR - NO NULL MEASUREMENT FOUND")
                return

            measurement.reference_measurement = last_null_meas
            clean_measurements.append(measurement)

    return clean_measurements


def get_labeled_measurements(data, correct_channels, functionalisations, debug=False):
    current_label = ''
    current_measurement = None
    current_temperature = 0
    current_gas = 0
    current_humidty = 0
    current_pressure = 0
    current_altitude = 0
    measurements = []

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
                                   current_temperature, current_gas, current_humidty, current_pressure,
                                   current_altitude)
                measurements.append(meas)

            current_label = row_data['label']
            current_temperature = row_data['temperature']
            current_gas = row_data['gas']
            current_humidty = row_data['humidity']
            current_pressure = row_data['pressure']
            current_altitude = row_data['altitude']
            time_stamp = ts
            current_measurement = None

        if current_label != '' and current_label == row_data['label']:
            if current_measurement is None:
                current_measurement = row_data['channels']
            else:
                current_measurement = np.vstack((current_measurement, row_data['channels']))

    if current_label is not '':
        meas = Measurement(current_measurement, current_label, time_stamp, correct_channels, functionalisations,
                           current_temperature, current_gas, current_humidty, current_pressure, current_altitude)
        measurements.append(meas)

    return measurements


def get_measurement_peak_average(data, num_samples=10):
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


def group_meas_data_by_functionalisation(data, functionalisations):
    averages = None
    for row in data:
        if averages is None:
            _, averages = group_row_data_by_functionalities(row, functionalisations)
        else:
            _, tmp = group_row_data_by_functionalities(row, functionalisations)
            averages = np.vstack((averages, tmp))

    return averages


def group_row_data_by_functionalities(row, functionalities):
    grouped_data = {}
    averaged_data = np.zeros(1 + np.max(list(map(int, functionalities))))
    for i in np.unique(functionalities):
        grouped_data[i] = {'values': []}

    for value, function in np.vstack((row, functionalities)).T:
        grouped_data[function]['values'].append(value)
        averaged_data[int(function)] = averaged_data[int(function)] + float(value)

    for i in np.unique(functionalities):
        averaged_data[int(i)] = averaged_data[int(i)] / len(grouped_data[i]['values'])

    return grouped_data, averaged_data
