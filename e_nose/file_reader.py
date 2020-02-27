# file_reader.py
import csv
import glob
import os
from datetime import datetime
from typing import Tuple, Mapping

import numpy as np

from .measurements import Functionalisations_t, WorkingChannels_t, DataRowsSet_t


def convert_to_datetime(possible_date: str) -> datetime:
    # Fri Jan  3 12:19:00 2020
    return datetime.strptime(possible_date, "%a %b %d %H:%M:%S %Y")


def read_data_csv(file_name: str, debug=False) -> Tuple[Functionalisations_t, WorkingChannels_t, DataRowsSet_t]:
    """
    Reads one CSV file of Data
    :param file_name:
    :param debug:
    :return:
    """
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
    for file in glob.glob(os.path.join(folder_name, '*.{}'.format(extension))):
        functionalisations, correct_channels, data = read_data_csv(file, debug)
        all_data[file] = data

    return functionalisations, correct_channels, all_data
