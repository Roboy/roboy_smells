import numpy as np
from enum import Enum, auto
from typing import Dict, Any, List, Mapping, Optional, Union

"""
Additional documentation: https://devanthro.atlassian.net/wiki/spaces/WS1920/pages/631111682/RS+-+Data+Pipeline
"""

# Useful Datatype definitions

Functionalisations_t = Union[List[int], np.ndarray]
WorkingChannels_t = Union[List[bool], np.ndarray]
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
    FULL = auto()
    HIGH_PASS = auto()
    DIFFERENTIAL = auto()
    """ Differential Data: 1st element is AVG of all channels, then the differentials for each channel """

    def is_grouped(self) -> bool:
        """
        Returns whether this DataType is grouped aka. TODO:..?
        :return:
        """
        return self is self.GROUPED_TOTAL_AVG or self is self.GROUPED_PEAK_AVG


class StandardizationType(Enum):
    """"""
    LAST_REFERENCE = auto()
    """ Standardize by the avg of the last few samples of the last reference measurement before/with this measurement
    
        (Reference measurement themselves will be standardized by their own ending)
    """
    BEGINNING_AVG = auto()
    """ Standardize by the avg value of the first few (3) samples of the same measurement """
    LOWPASS_FILTER = auto()
    """ Standardize by a low-pass filter over the historic data before this measurement """


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
        self.reference_measurement: Union[Measurement, np.ndarray, None] = None
        self.standardization_type: StandardizationType = StandardizationType.BEGINNING_AVG
        self.cached_data: Dict[DataType, np.ndarray] = {}
        """ Caches data for certain evaluation types """
        self.cached_logdata: Dict[DataType, np.ndarray] = {}
        """ Caches logdata for certain evaluation types """
        self.temperature = temperature
        self.gas = gas
        self.humidity = humidty
        self.pressure = pressure
        self.altitude = altitude

    def set_reference(self, reference_measurement: Union['Measurement', np.ndarray, None],
                      standardization_type: StandardizationType = StandardizationType.BEGINNING_AVG) -> ():
        self.reference_measurement = reference_measurement
        self.standardization_type = standardization_type
        # Standardizaion type change invalidates the cached data
        self.clear_cache()

    def clear_cache(self):
        self.cached_data = {}
        self.cached_logdata = {}

    def is_reference(self) -> bool:
        return self.label in ['ref', 'null']

    def get_data(self, standardize: bool = True, force: bool = False, log: bool = True,
                 only_working: bool = True) -> np.ndarray:
        """
        :param standardize: When standardize is True, the reference measurement will be used to standardise the data if available.
        :param force: When force is True, cached data will be ignored (i.e. recalculated)
        :param log: Logarithmic data
        :param only_working: Only return working channels; False only guaranteed to work with standardize OFF
        :return: returns the data of the measurement as dictionary with the timestamps as keys
        """
        #         # Import here to avoid circular references...
        from . import data_processing as dp

        data = self.logdata if log else self.data
        cache = self.cached_logdata if log else self.cached_data
        mask = self.correct_channels if only_working else np.ones(data.shape[1], bool)

        if standardize:
            if DataType.STANDARDIZED not in cache or force:
                if self.reference_measurement is None:
                    if not log:
                        print("WARNING! USING LOG HIGH PASS FOR NORMAL DATA!")
                    cache[DataType.STANDARDIZED] = dp.high_pass_logdata(data[:, mask])
                elif isinstance(self.reference_measurement, Measurement):
                    reference = self.reference_measurement.get_data_as(DataType.LAST_AVG, False, num_last=10, log=log)
                    cache[DataType.STANDARDIZED] = dp.high_pass_logdata(data[:, mask], init=reference)
                elif isinstance(self.reference_measurement, np.ndarray):
                    if not log:
                        print("WARNING! USING LOG REFERENCE FOR NORMAL DATA!")
                    cache[DataType.STANDARDIZED] = dp.high_pass_logdata(data, init=self.reference_measurement)[:, mask]
                else:
                    raise TypeError(
                        "ERROR: Invalid state of reference-measurement: " + str(type(self.reference_measurement)) + str(
                            type(Measurement)))

            return cache[DataType.STANDARDIZED]

        return data[:, mask]

    def get_data_as(self, datatype: DataType, standardize: bool = True, force: bool = False, log: bool = True,
                    num_last: int = 10, num_samples: int = 10) \
            -> np.ndarray:
        """
        :param log: enable logging
        :param datatype: see description of datatype above, more detailed description can be found here: https://devanthro.atlassian.net/wiki/spaces/WS1920/pages/edit-v2/631111682
        :param standardize: When standardize is True, the reference measurement will be used to standardise the data if available.
        :param force: When force is True, cached data will be ignored (i.e. recalculated)
        :param num_last: needed for DataType.LAST_AVG as is average the num_last measurements
        :param num_samples: total number of measurements
        :return: list of computed channels as mentioned in datatype and extended by specified bme sensor data
        """
        # Import here to avoid circular references...
        from . import data_processing as dp

        cache = self.cached_logdata if log else self.cached_data

        # having standardize is default, the non-standardized data will not be cached
        if datatype in cache and not force and standardize:
            return cache[datatype]

        # print('requesting datatype', datatype)

        data_as: Optional[np.ndarray] = None
        if datatype is DataType.LAST_AVG:
            data_as = np.mean(self.get_data(standardize, force, log=log)[-num_last:, :], axis=0)
        elif datatype is DataType.HIGH_PASS:
            data_as = self.get_data(standardize, force, log=log)
        elif datatype is DataType.FULL:
            data_as = dp.full_pre_processing(self.get_data(standardize, force, log=log))
        elif datatype is DataType.DIFFERENTIAL:
            data_as = dp.differential_pre_processing(self.get_data(standardize, force, log=True))
        elif datatype is DataType.TOTAL_AVG:
            data_as = np.mean(self.get_data(standardize, force, log=log), axis=0)
        elif datatype is DataType.PEAK_AVG:
            data_as = dp.get_measurement_peak_average(self.get_data(standardize, force=force, log=log))
        elif datatype is DataType.GRADIENTS:
            data_as = np.gradient(self.get_data(standardize, force, log=log), axis=1)
        elif datatype is DataType.GROUPED_TOTAL_AVG:
            data_as = np.mean(self.get_data_as(DataType.GROUPED, standardize, force, log=log), axis=0)
        elif datatype is DataType.GROUPED_PEAK_AVG:
            data_as = \
                dp.get_measurement_peak_average(self.get_data_as(DataType.GROUPED, standardize, force, log=log),
                                                num_samples)
        elif datatype is DataType.GROUPED:
            data_as = \
                dp.group_meas_data_by_functionalisation(self.get_data(standardize, force, log=log),
                                                        self.correct_functionalisations)
        else:
            print("no type found")

        if standardize and data_as is not None:
            if log:
                self.cached_logdata[datatype] = data_as
            else:
                self.cached_data[datatype] = data_as
        return data_as

    def get_data_extended(self, datatype: DataType, standardize: bool = True, force: bool = False, log: bool = True,
                          num_last: int = 10, num_samples: int = 10, temperature: bool = False, gas: bool = False,
                          humidity: bool = False, pressure: bool = False, altitude: bool = False) \
            -> np.ndarray:
        """
        :param log: enable logging
        :param gas: extends eNose data by gas
        :param temperature: extends eNose data by temperature
        :param altitude: extends eNose data by altitude
        :param pressure: extends eNose data by pressure
        :param humidity: extends eNose data by humidty
        :param datatype: detailed description can be found here: https://devanthro.atlassian.net/wiki/spaces/WS1920/pages/edit-v2/631111682
        :param standardize: When standardize is True, the reference measurement will be used to standardise the data if available.
        :param force: When force is True, cached data will be ignored (i.e. recalculated)
        :param num_last: needed for DataType.LAST_AVG as is average the num_last measurements
        :param num_samples: total number of measurements
        :return: list of computed channels as mentioned in datatype and extended by specified bme sensor data
        """
        pure_data = self.get_data_as(datatype, standardize, force, log, num_last, num_samples)
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
