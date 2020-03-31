from typing import List, Optional

import numpy as np
from .measurements import DataType, DataRowsSet_t, Measurement, Functionalisations_t
import matplotlib.pyplot as plt

plt.rcdefaults()

"""
This file contains various methods for easily showing (printing and plotting) relevant information about measurement
objects.
"""

def pretty_print(data: DataRowsSet_t):
    """
    Prints all the data available in a given measurement data
    :param data: e_nose sensor data
    """
    for ts in data:
        print(ts, ":", data[ts]['channels'], ";label:", data[ts]['label'])


def draw_bar_meas(measurements: List[Measurement],
                    datatypes: List[DataType],
                    standardize: bool=True,
                    force: bool=False,
                    num_last: int=1,
                    num_samples: int=1):
    """
    This method draws all measurements in the measurements array as a bar graph using all specified datatypes.


    :param measurements: List of measurements that should be drawn
    :param datatypes: List of datatypes that the measurements should be drawn for
    :param standardize: When standardize is True, the reference measurement will be used to standardise the data if available.
    :param force: When force is True, cached data will be ignored (i.e. recalculated)
    :param num_last: needed for DataType.LAST_AVG as is average the num_last measurements
    :param num_samples: total number of measurements
    """
    colors = ['xkcd:black', 'xkcd:blue', 'xkcd:brown', 'xkcd:golden yellow', 'xkcd:emerald green',
              'xkcd:baby blue', 'xkcd:magenta', 'xkcd:violet', 'xkcd:lightgreen']
    for measurement in measurements:
        plt.figure(figsize=(15, 6))
        groups = np.unique(measurement.correct_functionalisations)

        for i, datatype in enumerate(datatypes):
            plt.subplot(1, len(datatypes), i + 1)

            if datatype is DataType.GROUPED_PEAK_AVG or datatype is DataType.GROUPED_TOTAL_AVG:
                y_pos = np.arange(len(groups))
                plt.xticks(y_pos, groups)
            elif datatype is DataType.LAST_AVG or datatype is DataType.PEAK_AVG or datatype is DataType.TOTAL_AVG:
                y_pos = np.arange(len(measurement.correct_functionalisations))
            else:
                plt.text("not supported (yet?)")
                continue

            barlist = plt.bar(y_pos, measurement.get_data_as(datatype, standardize, force, num_last, num_samples),
                              align='center', color=colors, alpha=0.5)
            plt.title("{} Channels: {} at {}".format(datatype, measurement.label, measurement.ts))

            if datatype is DataType.LAST_AVG or datatype is DataType.PEAK_AVG or datatype is DataType.TOTAL_AVG:
                for i in range(len(measurement.correct_functionalisations)):
                    barlist[i].set_color(colors[int(measurement.correct_functionalisations[i])])

        plt.show()


def draw_bar_meas_direct_comp(measurements: List[Measurement],
                                functionalisations: Functionalisations_t,
                                datatype: DataType,
                                standardize: bool=True,
                                force: bool=False,
                                num_last: int=1,
                                num_samples: int=1):
    """
    This method uses the specified datatype method and groups all measurements of the same label together to be able
    to easily compare them. Only group datatypes are supported.

    :param measurements: List of measurements that should be drawn
    :param functionalisations: functionalisation for the measurements
    :param datatype: datatype that should be drawn
    :param standardize: When standardize is True, the reference measurement will be used to standardise the data if available.
    :param force: When force is True, cached data will be ignored (i.e. recalculated)
    :param num_last: needed for DataType.LAST_AVG as is average the num_last measurements
    :param num_samples: total number of measurements
    """
    if not datatype.is_grouped():
        print("Data type not supported (yet?)")
        return

    colors = ['xkcd:black', 'xkcd:blue', 'xkcd:brown', 'xkcd:golden yellow', 'xkcd:emerald green',
              'xkcd:baby blue', 'xkcd:magenta', 'xkcd:violet', 'xkcd:lightgreen']

    groups = np.unique(functionalisations)
    y_pos = np.arange(len(groups))

    sorted_measurements = sorted(measurements, key=lambda m: m.label)

    last_label = ''
    fig, ax = plt.subplots()
    count = 0

    label_counts = {}
    for measurement in sorted_measurements:
        if measurement.label in label_counts:
            label_counts[measurement.label] = label_counts[measurement.label] + 1
        else:
            label_counts[measurement.label] = 1

    for measurement in sorted_measurements:
        if last_label != measurement.label and last_label != '':
            ax.set_ylabel('R/R0')
            ax.set_title(last_label)
            ax.set_xticks([0, 1, 2, 3, 4])
            plt.show()

            fig, ax = plt.subplots()
            count = 0

        width = 1 / label_counts[measurement.label]
        bar = ax.bar(y_pos + width * count,
                     measurement.get_data_as(datatype, standardize, force, num_last, num_samples),
                     width, align='center', color=colors)

        for i in range(len(bar)):
            bar[i].set_edgecolor('xkcd:white')
        count += 1
        last_label = measurement.label

    ax.set_ylabel('R/R0')
    ax.set_title(last_label)
    ax.set_xticks([0, 1, 2, 3, 4])
    plt.show()


def draw_meas_channel_over_time(measurement: Measurement,
                                functionalisations: Functionalisations_t,
                                standardize: bool=True):
    """
    Draws each channels data over time for one measurement

    :param measurement: Measurement that should be drawn
    :param functionalisations: functionalisations of that measurement
    :param standardize: When standardize is True, the reference measurement will be used to standardise the data if available.
    """
    colors = ['xkcd:black', 'xkcd:blue', 'xkcd:brown', 'xkcd:golden yellow', 'xkcd:emerald green',
              'xkcd:baby blue', 'xkcd:magenta', 'xkcd:violet', 'xkcd:lightgreen']

    fig, ax = plt.subplots()
    data = measurement.get_data(standardize)
    # reconfigure data so that channels are in one array

    for i in range(data.shape[1]):
        ax.plot(range(len(data)), data[:, i], color=colors[functionalisations[i]])

    plt.show()


def draw_meas_grad_over_time(measurement: Measurement,
                             functionalisations: Functionalisations_t,
                             standardize: bool=True,
                             draw_ref: bool=True):
    """
    This method draws the gradient of the given measurement's data.

    :param measurement: Measurement that should be drawn
    :param functionalisations: functionalisations of that measurement
    :param draw_ref: True if the reference measurement shall be drawn as well
    :param standardize: When standardize is True, the reference measurement will be used to standardise the data if available.
    """
    colors = ['xkcd:black', 'xkcd:blue', 'xkcd:brown', 'xkcd:golden yellow', 'xkcd:emerald green',
              'xkcd:baby blue', 'xkcd:magenta', 'xkcd:violet', 'xkcd:lightgreen']

    fig, ax = plt.subplots()
    data = measurement.get_data_as(DataType.GRADIENTS)
    if draw_ref:
        ref_data = measurement.reference_measurement.get_data_as(DataType.GRADIENTS, standardize)
        data = np.vstack((ref_data, data))

    for i in range(data.shape[1]):
        ax.plot(range(len(data)), data[:, i], color=colors[functionalisations[i]])

    plt.show()


def draw_all_channel_data_as_line(all_data: np.ndarray,
                                  functionalisations: Functionalisations_t,
                                  num_from: int=0,
                                  num_to: int=-1,
                                  secondary: np.ndarray=None):
    """
    This method draws all the data of all channels from the measurement files (not a list of measurements!)
    as line graphs.

    It is possible to provide a set of secondary data (e.g. temperature/humidity) also as an array and it will be
    plotted over the e_nose data.

    :param all_data: the data as an array
    :param functionalisations: functionalisations of each channel -> the lines will be colored accordingly
    :param num_from: datapoint from which to draw
    :param num_to: datapoint to which should be drawn, -1 to show everything (other -x not supported)
    :param secondary: data array for the secondary axis.
    """
    draw_selected_channel_data_as_line(all_data, functionalisations, list(range(len(functionalisations))), num_from,
                                       num_to, secondary)


def draw_selected_channel_data_as_line(all_data: np.ndarray,
                                       functionalisations: Functionalisations_t,
                                       channels: List[int],
                                       num_from: int=0,
                                       num_to: int=-1,
                                       secondary: np.ndarray=None):
    """
    This method draws all the data of the selected channels from the measurement files (not a list of measurements!)
    as line graphs.

    It is possible to provide a set of secondary data (e.g. temperature/humidity) also as an array and it will be
    plotted over the e_nose data.

    :param all_data: the data as an array
    :param functionalisations: functionalisations of each channel -> the lines will be colored accordingly
    :param channels: channels to be plotted
    :param num_from: datapoint from which to draw
    :param num_to: datapoint to which should be drawn, -1 to show everything (other -x not supported)
    :param secondary: data array for the secondary axis.
    """
    colors = ['xkcd:black', 'xkcd:blue', 'xkcd:brown', 'xkcd:golden yellow', 'xkcd:emerald green',
              'xkcd:baby blue', 'xkcd:magenta', 'xkcd:violet', 'xkcd:lightgreen']

    for file in all_data:
        print(file)
        data = all_data[file]
        fig: Optional[plt.Figure] = None
        ax: Optional[plt.Axes] = None
        ax2: Optional[plt.Axes] = None
        fig, ax = plt.subplots()

        if secondary is not None:
            # Second Y axis
            ax2: plt.Axes = ax.twinx()
            ax2.set_ylabel(secondary, color='r')

        num_channels = len(channels)

        # reconfigure data so that channels are in one array
        data_matrix = np.zeros((len(data), num_channels))
        secondary_data = np.zeros(len(data))

        last_label = ""
        for i, time in enumerate(data):
            if data[time]['label'] != last_label:
                if i > num_from:
                    if num_to == -1 or i < num_to:
                        ax.axvline(x=i)
                        # This is a hacky way I found by digging in the code to make the text appear exactly above the figure
                        ax.text(i, 1.05, last_label, ha='right', rotation=90, va='bottom', transform=ax._xaxis_transform)
                last_label = data[time]['label']
            channel_data = data[time]['channels'][channels]
            channel_data[np.argwhere(channel_data > 30000)] = 0
            data_matrix[i, :] = channel_data
            if secondary is not None:
                secondary_data[i] = data[time][secondary]

        for i in range(num_channels):
            ax.plot(range(num_from, num_from+len(data_matrix[num_from:num_to, i])), data_matrix[num_from:num_to, i],
                    color=colors[functionalisations[i]])

        if secondary is not None:
            ax2.plot(range(num_from, num_from+len(secondary_data[num_from:num_to])), secondary_data[num_from:num_to],
                    color='r')

        plt.show()
