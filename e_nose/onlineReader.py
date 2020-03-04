import numpy as np

from e_nose.file_reader import get_sensor_spec
from e_nose.measurements import Measurement, StandardizationType


class OnlineReader:
    """
    Class that reads in subsequent samples of the sensor and allows to generate Measurement objects with reference data
    from the last n samples
    """
    l1_factor = 1e-3
    """ Factor for the low-pass filter """

    def __init__(self, sensor_id: int,
                 standardization: StandardizationType = StandardizationType.LOWPASS_FILTER,
                 max_history_length: int = 100_000):
        """

        :param sensor_id: ID of the sensor to use => to know the functionalisations and failure bits
        :param standardization: The kind of standardization to apply to the live data;
                they may not be entirely correct, though...
        :param max_history_length: Max length of the history data... to avoid using too much memory
        """
        self.data: np.ndarray = np.empty((0, 64))
        self.log_lowpass: np.ndarray = np.empty((0, 64))
        self.log_lowpass_current = None
        self.max_history_length: int = max_history_length
        self.standardization: StandardizationType = standardization

        self.functionalisations, self.working_channels = get_sensor_spec(sensor_id)

    def set_standardization_type(self, standardization: StandardizationType):
        self.standardization = standardization

    def add_sample(self, sample: np.ndarray) -> ():
        """ Adds a single sample to the internal state; updates the internal low-pass automatically """
        if self.log_lowpass_current is None:
            self.log_lowpass_current = np.log(sample)

        self.log_lowpass_current = (self.log_lowpass_current + np.log(sample) * self.l1_factor) / (1.0 + self.l1_factor)

        self.data = np.vstack((self.data, sample))
        self.log_lowpass = np.vstack((self.log_lowpass, self.log_lowpass_current))

        if len(self.data.shape) > self.max_history_length:
            remove = len(self.data.shape) - self.max_history_length
            self.data = np.delete(self.data, range(remove))
            self.log_lowpass = np.delete(self.log_lowpass, range(remove))

    def add_samples(self) -> ():
        raise NotImplemented

    def get_last_n_as_measurement(self, n: int = 300) -> Measurement:
        """
        Returns a measurement object for the last n data samples with the reference set to the lowpass-filter of the
        beginning of the series
        :param n:
        :return:
        """
        measurement = Measurement(self.data[-n:],
                                  '', '',  # label & timestamp unknown
                                  self.working_channels, self.functionalisations,
                                  0, 0, 0, 0, 0  # No BME-Data
                                  )

        if self.standardization == StandardizationType.BEGINNING_AVG:
            measurement.set_reference(None, StandardizationType.BEGINNING_AVG)

        elif self.standardization == StandardizationType.LOWPASS_FILTER:
            measurement.set_reference(self.log_lowpass[-n], StandardizationType.LOWPASS_FILTER)

        elif self.standardization == StandardizationType.LAST_REFERENCE:
            # We cannot actually perform this standardization as we hav got no idea what the last
            # reference measurement might have been...
            # To simulate the same behaviour though, we just set it to the average of the last 10 samples just before
            reference = np.mean(np.log(self.data[-(n+10):-n, :]), axis=0)
            measurement.set_reference(reference, StandardizationType.LAST_REFERENCE)

        return measurement
