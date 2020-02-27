# file_reader.py
import csv
import glob
import os
import re
from datetime import datetime
from typing import Tuple, Mapping, List

import numpy as np

from .measurements import Functionalisations_t, WorkingChannels_t, DataRowsSet_t


def convert_to_datetime(possible_date: str) -> datetime:
    # Fri Jan  3 12:19:00 2020
    return datetime.strptime(possible_date, "%a %b %d %H:%M:%S %Y")


def get_sensor_spec(sensor_id: int) -> Tuple[Functionalisations_t, WorkingChannels_t]:
    functionalisations: List[int] = []
    failures: List[int] = []

    if sensor_id == 4:
        functionalisations = [2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3,
                              3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2]
        failures            = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif sensor_id == 5:
        functionalisations = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
                              3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4,
                              5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0,
                              6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7]
        # Channel 15 & 23 disabled as it gives huge numbers (but it kinda works..?)
        # Channel 22, 27, 31, 35, 39 are always stuck to the same number
        failures            = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                               0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1,
                               0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
        print('Unknown Sensor ID %i! No functionalisation and channel failure data available' % sensor_id)
    correct_channels = np.invert(np.array(failures).astype(bool))
    print('using sensor %i specification' % sensor_id)

    return functionalisations, correct_channels

def read_data_csv(file_name: str, debug=False) -> Tuple[Functionalisations_t, WorkingChannels_t, DataRowsSet_t]:
    """
    Reads one CSV file of Data
    :param file_name:
    :param debug:
    :return:
    """

    sensor_id = 4
    m = re.search('2020-(\d+)-', file_name)
    if m is not None and int(m.group(1)) >= 2:
        # Measurements taken after february are using the new sensor
        sensor_id = 5
    functionalisations, correct_channels = get_sensor_spec(sensor_id)

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


def read_all_files_in_folder(folder_name: str, extension="csv", debug=False)\
        -> Tuple[Functionalisations_t, WorkingChannels_t, Mapping[str, DataRowsSet_t]]:
    """
    Reads all CSV files in a folder and returns functionalizations, working cannels, Dict[filename, Data]
    :param folder_name:
    :param extension:
    :param debug:
    :return:
    """
    all_data = {}
    functionalisations = []
    correct_channels = []

    if not os.path.isdir(folder_name):
        print('Directory "%s" does not exist!' % folder_name)

    for file in glob.glob(os.path.join(folder_name, '*.{}'.format(extension))):
        print('Reading file %s' % file)
        functionalisations, correct_channels, data = read_data_csv(file, debug)
        all_data[file] = data

    print('Read %i files' % len(all_data))
    return functionalisations, correct_channels, all_data
