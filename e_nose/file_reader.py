# file_reader.py
import numpy as np
import csv
from datetime import datetime
import os
import glob
import time


def convert_to_datetime(possible_date):
    # Fri Jan  3 12:19:00 2020
    return datetime.strptime(possible_date, "%a %b %d %H:%M:%S %Y")


def read_data_csv(file_name, debug=False):
    functionalisations = [2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4,
                          4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 2,
                          2, 2]
    failures = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    correct_channels = np.invert(np.array(failures).astype(bool))
    # stored in the form of timestamp as key, as value you have another dictionary with 'channels',
    # 'temperature, gas, humidity, pressure, altitude, label',
    data = {}

    with open(file_name) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        # skip header
        next(reader)
        i = 0
        # parsing data
        for row in reader:
            row_data = {'channels': np.array(row[1:-6]).astype(np.float),
                        'temperature': float(row[-6]),
                        'gas': float(row[-5]),
                        'humidity': float(row[-4]),
                        'pressure': float(row[-3]),
                        'altitude': float(row[-2]),
                        'label': row[-1]}
            data[row[0]] = row_data

    # sorted(data, key=lambda x: datetime.strptime(x[1], '%a %b %d %Y'))
    return functionalisations, correct_channels, data


def read_all_files_in_folder(folder_name, extension="csv", debug=False):
    all_data = {}
    for file in glob.glob(os.path.join(folder_name, '*.{}'.format(extension))):
        functionalisations, correct_channels, data = read_data_csv(file, debug)
        all_data[file] = data

    return functionalisations, correct_channels, all_data
