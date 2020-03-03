import numpy as np

from e_nose.file_reader import get_sensor_spec
from e_nose.measurements import Measurement


class OnlineReader:
    """
    Class that reads in subsequent samples of the sensor and allows to generate Measurement objects with reference data
    from the last n samples
    """
    l1_factor = 1e-3
    """ Factor for the low-pass filter """

    def __init__(self, sensor_id: int, max_history_length: int = 100_000):
        """

        :param sensor_id: ID of the sensor to use => to know the functionalisations and failure bits
        :param max_history_length: Max length of the history data... to avoid using too much memory
        """
        self.data: np.ndarray = np.empty((0, 64))
        self.log_lowpass: np.ndarray = np.empty((0, 64))
        self.log_lowpass_current = None
        self.max_history_length: int = max_history_length

        self.functionalisations, self.working_channels = get_sensor_spec(sensor_id)

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
        measurement.reference_measurement = self.log_lowpass[-n]

        return measurement
