from typing import Dict, Any

import numpy as np
import data_processing as dp
from enum import Enum, auto


class DataType(Enum):
    TOTAL_AVG = auto()
    PEAK_AVG = auto()
    GROUPED_TOTAL_AVG = auto()
    GROUPED_PEAK_AVG = auto()
    GRADIENTS = auto()
    LAST_AVG = auto()
    STANDARDIZED = auto()
    GROUPED = auto()

    def is_grouped(self):
        return self is self.GROUPED_TOTAL_AVG or self is self.GROUPED_PEAK_AVG


class Measurement:
    cached_data: Dict[Any, Any]

    def __init__(self, data, label, time_stamp, correct_channels, functionalisations, temperature=0, gas=0, humidty=0,
                 pressure=0, altitude=0):
        self.label = label
        self.ts = time_stamp
        self.correct_channels = correct_channels
        self.data = data
        self.functionalisations = functionalisations
        self.correct_functionalisations = np.array(functionalisations)[correct_channels]
        self.reference_measurement = None
        self.cached_data = {}
        self.temperature = temperature
        self.gas = gas
        self.humidity = humidty
        self.pressure = pressure
        self.altitude = altitude

    def get_data(self, standardize=True, force=False):
        if standardize:
            if DataType.STANDARDIZED not in self.cached_data or force:
                if self.reference_measurement is None:
                    self.cached_data[DataType.STANDARDIZED] = \
                        100 * (self.data[:, self.correct_channels] / (
                                1e-15 + self.get_data_as(DataType.LAST_AVG, False, num_last=10)) - 1)
                else:
                    self.cached_data[DataType.STANDARDIZED] = \
                        100 * (self.data[:, self.correct_channels] / (
                                1e-15 + self.reference_measurement.get_data_as(DataType.LAST_AVG, False,
                                                                               num_last=10)) - 1)

            return self.cached_data[DataType.STANDARDIZED]

        return self.data[:, self.correct_channels]

    def get_data_as(self, datatype, standardize=True, force=False, num_last=10, num_samples=10):

        # having standardize is default, the non-standardized data will not be cached
        if datatype in self.cached_data and not force and standardize:
            return self.cached_data[datatype]

        data_as = None
        if datatype is DataType.LAST_AVG:
            data_as = np.mean(self.get_data(standardize, force)[-num_last:, :], axis=0)
        elif datatype is DataType.TOTAL_AVG:
            data_as = np.mean(self.get_data(standardize, force), axis=0)
        elif datatype is DataType.PEAK_AVG:
            data_as = dp.get_measurement_peak_average(self.get_data(standardize, force=force))
        elif datatype is DataType.GRADIENTS:
            data_as = np.gradient(self.get_data(standardize, force), axis=1)
        elif datatype is DataType.GROUPED_TOTAL_AVG:
            data_as = np.mean(self.get_data_as(DataType.GROUPED, standardize, force), axis=0)
        elif datatype is DataType.GROUPED_PEAK_AVG:
            data_as = \
                dp.get_measurement_peak_average(self.get_data_as(DataType.GROUPED, standardize, force), num_samples)
        elif datatype is DataType.GROUPED:
            data_as = \
                dp.group_meas_data_by_functionalisation(self.get_data(standardize, force),
                                                        self.correct_functionalisations)

        if standardize:
            self.cached_data[datatype] = data_as
        return data_as

    def get_data_extended(self, datatype, standardize=True, force=False, num_last=10, num_samples=10):
        pure_data = self.get_data_as(datatype, standardize, force, num_last, num_samples)
        pure_data = np.append(pure_data, [self.humidity, self.pressure, self.altitude])
        return pure_data
