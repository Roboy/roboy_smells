# file_reader.py
import csv
import glob
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Tuple, Mapping, List

import numpy as np

from .measurements import Functionalisations_t, WorkingChannels_t, DataRowsSet_t


def convert_to_datetime(possible_date: str) -> datetime:
    # Fri Jan  3 12:19:00 2020
    return datetime.strptime(possible_date, "%a %b %d %H:%M:%S %Y")


def load_sensor_preset(preset_file: str) -> np.ndarray:
    """ Load a given sensor preset-file from the folder ./presets/ (relative to this file or to the working dir) """
    localpath = Path(__file__).absolute().parent.joinpath("presets/" + preset_file)
    if localpath.is_file():
        return np.loadtxt(localpath, int)
    else:
        return np.loadtxt(preset_file, int)


def get_sensor_spec(sensor_id: int) -> Tuple[Functionalisations_t, WorkingChannels_t]:
    """ Get the specification of a certain sensor aka.
        the Functionalisations and which channels are known to be broken
    """
    functionalisations: np.ndarray = np.array([])
    failures: np.ndarray = np.array([])

    if sensor_id == 4:
        functionalisations = np.array(
            [2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3,
             3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4,
             4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2]
        )
        failures = np.array(
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )
    elif sensor_id == 5:
        functionalisations = load_sensor_preset('LasVegas.preset')
        # Channel 15, 16 & 23 disabled as it gives huge numbers (but it kinda works..?)
        failures_huge = [15, 16, 23]
        # Channel 22, 31 are shorts and always stuck to the lower bound (347.9)
        failures_shorts = [22, 31]
        # Channels are IN SOME MEASUREMENTS stuck to the lower bound
        failures_mid_low = [3, 4, 22, 25, 26, 27, 28, 29, 31, 35, 36, 38, 39, 60]
        # More channels that are stuck somewhere
        failures_more = [2, 3, 4, 5, 22, 25, 26, 27, 28, 29, 31, 35, 36, 38, 39, 56, 59, 60, 61]
        failures_too_many = [0, 1, 2, 3, 4, 5, 6, 7,
                             22, 24, 25, 26, 27, 28, 29, 30, 31, 35, 36, 37, 38, 39, 56, 58, 59, 60, 61, 62]
        '''failures = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1,
             0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )'''
        failures = np.zeros(64, bool)
        #failures[failures_huge] = True
        failures[failures_shorts] = True
        #failures[failures_mid_low] = True
        #failures[failures_more] = True
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
                        'label': row[-1].lower()}
            data[row[0]] = row_data

    # sorted(data, key=lambda x: datetime.strptime(x[1], '%a %b %d %Y'))
    return functionalisations, correct_channels, data


def read_all_files_in_folder(folder_name: str, extension="csv", debug=False) \
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
