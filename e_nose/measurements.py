import numpy as np
from enum import Enum, auto
from typing import Dict, Any, List, Mapping, Optional

# Useful Datatype definitions

Functionalisations_t = List[int]
WorkingChannels_t = List[bool]
DataRow_t = Dict[str, Any]
# Only available in Python 3.8 ff
"""TypedDict('DataRow_t', {
    'channels': np.ndarray,
    'temperature': float,
    'gas': float,
    'humidity': float,
    'pressure': float,
    'altitude': float,
    'label': str
})"""
DataRowsSet_t = Mapping[str, DataRow_t]


class DataType(Enum):
    """"""
    TOTAL_AVG = auto()
    """ Total average over """
    PEAK_AVG = auto()
    GROUPED_TOTAL_AVG = auto()
    GROUPED_PEAK_AVG = auto()
    GRADIENTS = auto()
    LAST_AVG = auto()
    STANDARDIZED = auto()
    GROUPED = auto()

    def is_grouped(self) -> bool:
        """
        Returns whether this DataType is grouped aka. TODO:..?
        :return:
        """
        return self is self.GROUPED_TOTAL_AVG or self is self.GROUPED_PEAK_AVG


class Measurement:
    """
    One Measurement
    """

    def __init__(self, data: np.ndarray, label: str, time_stamp: str,
                 correct_channels: WorkingChannels_t, functionalisations: Functionalisations_t,
                 temperature: float = 0, gas: float = 0, humidty: float = 0,
                 pressure: float = 0, altitude: float = 0):
        self.label = label
        self.ts: str = time_stamp
        self.correct_channels = correct_channels
        self.data: np.ndarray = data
        self.logdata: np.ndarray = np.log(data)
        self.functionalisations = functionalisations
        self.correct_functionalisations: Functionalisations_t = np.array(functionalisations)[correct_channels]
        """ Functionalisations of ONLY the working channels"""
        self.reference_measurement = None
        self.cached_data: Dict[DataType, np.ndarray] = {}
        """ Caches data for certain evaluation types """
        self.temperature = temperature
        self.gas = gas
        self.humidity = humidty
        self.pressure = pressure
        self.altitude = altitude

    def get_data(self, standardize: bool = True, force: bool = False) -> np.ndarray:
        """

        :param standardize:
        :param force:
        :return:
        """
        # Import here to avoid circular references...
        from . import data_processing as dp

        if standardize:
            if DataType.STANDARDIZED not in self.cached_data or force:
                if self.reference_measurement is None:
                    self.cached_data[DataType.STANDARDIZED] = \
                        dp.high_pass_logdata(self.logdata[:, self.correct_channels])
                else:
                    self.cached_data[DataType.STANDARDIZED] = \
                        self.logdata[:, self.correct_channels] - np.log(self.reference_measurement.get_data_as(DataType.LAST_AVG, False, um_last=10))

            return self.cached_data[DataType.STANDARDIZED]

        return self.data[:, self.correct_channels]

    def get_data_as(self, datatype: DataType, standardize: bool = True, force: bool = False,
                    num_last: int = 10, num_samples: int = 10) \
            -> np.ndarray:
        """

        :param datatype:
        :param standardize:
        :param force:
        :param num_last:
        :param num_samples:
        :return:
        """
        # Import here to avoid circular references...
        from . import data_processing as dp

        # having standardize is default, the non-standardized data will not be cached
        if datatype in self.cached_data and not force and standardize:
            return self.cached_data[datatype]

        data_as: Optional[np.ndarray] = None
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

    def get_data_extended(self, datatype: DataType, standardize: bool = True, force: bool = False,
                          num_last: int = 10, num_samples: int = 10, temperature: bool = False, gas: bool = False,
                          humidity: bool = False, pressure: bool = False, altitude: bool = False) \
            -> np.ndarray:
        """

        :param datatype:
        :param standardize:
        :param force:
        :param num_last:
        :param num_samples:
        :return:
        """
        pure_data = self.get_data_as(datatype, standardize, force, num_last, num_samples)
        if temperature:
            pure_data = np.append(pure_data, self.temperature)
        if gas:
            pure_data = np.append(pure_data, self.gas)
        if humidity:
            pure_data = np.append(pure_data, self.humidity)
        if pressure:
            pure_data = np.append(pure_data, self.pressure)
        if altitude:
            pure_data = np.append(pure_data, self.altitude)

        return pure_data
