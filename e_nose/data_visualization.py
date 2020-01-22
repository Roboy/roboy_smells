import numpy as np
from .measurements import DataType
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt


def pretty_print(data):
    for ts in data:
        print(ts,":",data[ts]['channels'],";label:",data[ts]['label'])

def draw_bar_meas(measurements, datatypes, standardize=True, force=False, num_last=1, num_samples=1):
    colors = ['xkcd:green', 'xkcd:blue', 'xkcd:brown', 'xkcd:yellow', 'xkcd:black']
    for measurement in measurements:
        plt.figure(figsize=(15, 6))
        groups = np.unique(measurement.correct_functionalisations)

        for i, datatype in enumerate(datatypes):
            plt.subplot(1, len(datatypes), i+1)

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


def draw_bar_meas_direct_comp(measurements, functionalisations, datatype, standardize=True, force=False, num_last=1, num_samples=1):
    if not datatype.is_grouped():
        print("Data type not supported (yet?)")
        return

    colors = ['xkcd:green','xkcd:blue','xkcd:brown','xkcd:yellow','xkcd:black']

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
            ax.set_xticks([0,1,2,3,4])
            plt.show()

            fig, ax = plt.subplots()
            count = 0

        width = 1/label_counts[measurement.label]
        bar = ax.bar(y_pos + width*count, measurement.get_data_as(datatype, standardize, force, num_last, num_samples),
                     width, align='center', color=colors)

        for i in range(len(bar)):
            bar[i].set_edgecolor('xkcd:white')
        count += 1
        last_label = measurement.label

    ax.set_ylabel('R/R0')
    ax.set_title(last_label)
    ax.set_xticks([0, 1, 2, 3, 4])
    plt.show()


def draw_meas_channel_over_time(measurement, functionalisations,standardize=True, draw_ref=True):
    colors = ['xkcd:green','xkcd:blue','xkcd:brown','xkcd:yellow','xkcd:black']

    fig, ax = plt.subplots()
    data = measurement.get_data(standardize)
    # reconfigure data so that channels are in one array

    for i in range(data.shape[1]):
        ax.plot(range(len(data)), data[:,i], color=colors[functionalisations[i]])

    plt.show()


def draw_meas_grad_over_time(measurement, functionalisations,standardize=True, draw_ref=True):
    colors = ['xkcd:green','xkcd:blue','xkcd:brown','xkcd:yellow','xkcd:black']

    fig, ax = plt.subplots()
    data = measurement.get_data_as(DataType.GRADIENTS)
    if draw_ref:
        ref_data = measurement.reference_measurement.get_data_as(DataType.GRADIENTS, standardize)
        data = np.vstack((ref_data, data))

    for i in range(data.shape[1]):
        ax.plot(range(len(data)), data[:,i], color=colors[functionalisations[i]])

    plt.show()


def draw_all_channel_data_as_line(all_data, functionalisations, num_from = 0, num_to = -1):

    colors = ['xkcd:green','xkcd:blue','xkcd:brown','xkcd:yellow','xkcd:black']

    for file in all_data:
        print(file)
        data = all_data[file]
        fig, ax = plt.subplots()

        # reconfigure data so that channels are in one array
        data_matrix = np.zeros((len(data), 64))

        last_label = ""
        for i, time in enumerate(data):
            if data[time]['label'] != last_label:
                if i > num_from:
                    if num_to == -1 or i < num_to:
                        ax.axvline(x=(i - num_from))
                        ax.text((i - num_from - 50), 27000, last_label, ha='right', rotation=90, va='bottom')
                last_label = data[time]['label']
            channel_data = data[time]['channels']
            channel_data[np.argwhere(channel_data > 45000)] = 0
            data_matrix[i,:] = channel_data

        for i in range(64):
            ax.plot(range(len(data_matrix[num_from:num_to,i])), data_matrix[num_from:num_to,i], color=colors[functionalisations[i]])

        plt.show()


