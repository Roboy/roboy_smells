import numpy as np
from datetime import datetime, date, time, timezone
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def pretty_print(data):
    for ts in data:
        print(ts,":",data[ts]['channels'],";label:",data[ts]['label'])

def pretty_print_meas(measurements, p):
    for ts in measurements:
        print(ts,"; label:", measurements[ts]['label'])
        if 'avgs' in measurements[ts]:
            print("channel averages:", measurements[ts]['avgs'])   
            
        if 'func_avg' in measurements[ts]:
            print("channel group averages:", measurements[ts]['func_avg'])   
            
        #print(measurements[ts]['data'])


def pretty_draw_meas(measurements, functionalisations, failures, show_all_channels=False):
    colors = ['xkcd:green', 'xkcd:blue', 'xkcd:brown', 'xkcd:yellow', 'xkcd:black']
    for measurement in measurements:
        plt.figure(figsize=(15, 6))

        plt.subplot(1, 2, 1)
        groups = np.unique(functionalisations)
        y_pos = np.arange(len(groups))
        plt.bar(y_pos, measurement.get_total_average_by_group(functionalisations), align='center', color=colors,
                alpha=0.5)
        plt.xticks(y_pos, groups)
        plt.title("Grouped Channels: {} at {}".format(measurement.label, measurement.ts))

        if show_all_channels:
            plt.subplot(1, 2, 2)
            y_pos = np.arange(len(functionalisations))
            barlist = plt.bar(y_pos, measurement.get_total_average(), align='center', alpha=0.5)

            for i in range(len(functionalisations)):
                barlist[i].set_color(colors[int(functionalisations[i])])
                
                #if failures[i]:
                #    barlist[i].set_edgecolor('xkcd:red')

            plt.title("All Channels: {} at {}".format(measurement.label, measurement.ts))

        plt.show()


def pretty_draw_meas_peak(measurements, functionalisations, failures, show_all_channels=False):
    colors = ['xkcd:green', 'xkcd:blue', 'xkcd:brown', 'xkcd:yellow', 'xkcd:black']
    for measurement in measurements:
        plt.figure(figsize=(15, 6))

        plt.subplot(1, 2, 1)
        groups = np.unique(functionalisations)
        y_pos = np.arange(len(groups))
        plt.bar(y_pos, measurement.get_peak_average_by_group(functionalisations), align='center', color=colors,
                alpha=0.5)
        plt.xticks(y_pos, groups)
        plt.title("Grouped Channels: {} at {}".format(measurement.label, measurement.ts))

        if show_all_channels:
            plt.subplot(1, 2, 2)
            y_pos = np.arange(len(functionalisations))
            barlist = plt.bar(y_pos, measurement.get_peak_average(), align='center', alpha=0.5)

            for i in range(len(functionalisations)):
                barlist[i].set_color(colors[int(functionalisations[i])])
                if failures[i]:
                    barlist[i].set_edgecolor('xkcd:red')

            plt.title("All Channels: {} at {}".format(measurement.label, measurement.ts))

        plt.show()
            
def pretty_draw_direct_comp(measurements, functionalisations, failures=None, show_all_channels=False):
    measurements_grouped_by_label = {}
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
        bar = ax.bar(y_pos + width*count, measurement.get_total_average_by_group(functionalisations),
                     width, align='center', color=colors)

        for i in range(len(bar)):
            bar[i].set_edgecolor('xkcd:white')
        count += 1
        last_label = measurement.label

    ax.set_ylabel('R/R0')
    ax.set_title(last_label)
    ax.set_xticks([0, 1, 2, 3, 4])
    plt.show()

def pretty_draw_all(measurements, functionalisations, failures, show_all_channels=False):
    measurements_grouped_by_label = {}
    colors = ['xkcd:green','xkcd:blue','xkcd:brown','xkcd:yellow','xkcd:black']

    groups = np.unique(functionalisations)
    y_pos = np.arange(len(groups))

    last_label = ''
    fig, ax = plt.subplots()
    count = 0

    label_dict = {}
    for measurement in measurements:
        if measurement.label not in label_dict:
            label_dict[measurement.label] = []
        label_dict[measurement.label].append(measurement)

    for label in label_dict:
        pretty_draw_meas(label_dict[label], functionalisations, failures)

        fig, ax = plt.subplots()
        count = 0

        width = 1/len(label_dict[label])

        for measurement in label_dict[label]:
            bar = ax.bar(y_pos + width*count, measurement.get_total_average_by_group(functionalisations),
                         width, align='center', color=colors)
            count += 1

            for i in range(len(bar)):
                bar[i].set_edgecolor('xkcd:white')

        ax.set_ylabel('R/R0')
        ax.set_title(last_label)
        ax.set_xticks([0, 1, 2, 3, 4])
        plt.show()

def pretty_draw_all_peak(measurements, functionalisations, failures, show_all_channels=False):
    colors = ['xkcd:green','xkcd:blue','xkcd:brown','xkcd:yellow','xkcd:black']

    groups = np.unique(functionalisations)
    y_pos = np.arange(len(groups))

    label_dict = {}
    for measurement in measurements:
        if measurement.label not in label_dict:
            label_dict[measurement.label] = []
        label_dict[measurement.label].append(measurement)

    for label in label_dict:
        pretty_draw_meas_peak(label_dict[label], functionalisations, failures)

        fig, ax = plt.subplots()
        count = 0

        width = 1/len(label_dict[label])

        for measurement in label_dict[label]:
            bar = ax.bar(y_pos + width*count, measurement.get_peak_average_by_group(functionalisations),
                         width, align='center', color=colors)
            count += 1

            for i in range(len(bar)):
                bar[i].set_edgecolor('xkcd:white')

        ax.set_ylabel('R/R0')
        ax.set_title(label)
        ax.set_xticks([0, 1, 2, 3, 4])
        plt.show()

def pretty_draw_everything(measurements, functionalisations, failures, show_all_channels=False):
    colors = ['xkcd:green','xkcd:blue','xkcd:brown','xkcd:yellow','xkcd:black']

    groups = np.unique(functionalisations)
    y_pos = np.arange(len(groups))

    plt.figure()

    label_dict = {}
    for measurement in measurements:
        if measurement.label not in label_dict:
            label_dict[measurement.label] = []
        label_dict[measurement.label].append(measurement)

    i = 0
    for label in label_dict:

        ax = plt.subplot(len(label_dict), 1, i+1)
        count = 0

        width = 1/len(label_dict[label])

        for measurement in label_dict[label]:
            bar = ax.bar(y_pos + width*count, measurement.get_total_average_by_group(functionalisations),
                         width, align='center', color=colors)
            count += 1

            for i in range(len(bar)):
                bar[i].set_edgecolor('xkcd:white')

        ax.set_ylabel('R/R0')
        ax.set_title(label)
        ax.set_xticks([0, 1, 2, 3, 4])

        for c, num_samples in enumerate((3,5,10)):
            ax = plt.subplot(len(label_dict), c+2, i+1)
            count = 0

            width = 1/len(label_dict[label])

            for measurement in label_dict[label]:
                bar = ax.bar(y_pos + width*count, measurement.get_peak_average_by_group(functionalisations, num_samples),
                             width, align='center', color=colors)
                count += 1

                for i in range(len(bar)):
                    bar[i].set_edgecolor('xkcd:white')

            ax.set_ylabel('R/R0')
            ax.set_title(label)
            ax.set_xticks([0, 1, 2, 3, 4])

    plt.show()
