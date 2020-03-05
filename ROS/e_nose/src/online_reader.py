import numpy as np
from pathlib import Path
from measurements import Measurement, StandardizationType, DataRowsSet_t, DataRow_t


class OnlineReader:
    """
    Class that reads in subsequent samples of the sensor and allows to generate Measurement objects with reference data
    from the last n samples
    """
    l1_factor = 1e-3
    """ Factor for the low-pass filter """

    def __init__(self, sensor_id, standardization=StandardizationType.LOWPASS_FILTER, max_history_length=100000):
        """

        :param sensor_id: ID of the sensor to use => to know the functionalisations and failure bits
        :param standardization: The kind of standardization to apply to the live data;
                they may not be entirely correct, though...
        :param max_history_length: Max length of the history data... to avoid using too much memory
        """
        self.data_buffer = np.empty((max_history_length, 64))
        """ data buffer that contains empty rows for all data that might come in one day """
        self.log_lowpass_buffer = np.empty((max_history_length, 64))
        """ log_lowpass data buffer that contains empty rows for all data that might come in one day """
        self.current_length = 0
        """ current length of the data buffer """
        self.log_lowpass_current = None
        self.max_history_length= max_history_length
        self.standardization = standardization

        self.functionalisations, self.working_channels = self.get_sensor_spec(sensor_id)

    def set_standardization_type(self, standardization):
        self.standardization = standardization

    def add_sample(self, sample):
        """ Adds a single sample to the internal state; updates the internal low-pass automatically """
        if self.log_lowpass_current is None:
            self.log_lowpass_current = np.log(sample)

        self.log_lowpass_current = (self.log_lowpass_current + np.log(sample) * self.l1_factor) / (1.0 + self.l1_factor)

        self.data_buffer[self.current_length] = sample
        self.log_lowpass_buffer[self.current_length] = self.log_lowpass_current

        self.current_length += 1

    def get_last_n_as_measurement(self, n=300):
        """
        Returns a measurement object for the last n data samples with the reference set to the lowpass-filter of the
        beginning of the series
        :param n:
        :return:
        """
        measurement = Measurement(self.data_buffer[self.current_length - n:self.current_length],
                                  '', '',  # label & timestamp unknown
                                  self.working_channels, self.functionalisations,
                                  0, 0, 0, 0, 0  # No BME-Data
                                  )

        if self.standardization == StandardizationType.BEGINNING_AVG:
            measurement.set_reference(None, StandardizationType.BEGINNING_AVG)

        elif self.standardization == StandardizationType.LOWPASS_FILTER:
            measurement.set_reference(self.log_lowpass_buffer[self.current_length - n],
                                      StandardizationType.LOWPASS_FILTER)

        elif self.standardization == StandardizationType.LAST_REFERENCE:
            # We cannot actually perform this standardization as we hav got no idea what the last
            # reference measurement might have been...
            # To simulate the same behaviour though, we just set it to the average of the last 10 samples just before
            reference = np.mean(np.log(self.data_buffer[self.current_length - (n + 10):self.current_length - n, :]),
                                axis=0)
            measurement.set_reference(reference, StandardizationType.LAST_REFERENCE)

        return measurement

    def get_sensor_spec(self,sensor_id: int):
        functionalisations = np.array([])
        failures = np.array([])

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
            functionalisations = self.load_sensor_preset('LasVegas.preset')
            # Channel 15, 16 & 23 disabled as it gives huge numbers (but it kinda works..?)
            failures_huge = [15, 16, 23]
            # Channel 22, 31, 27, 35, 39 are always stuck to the lower bound (347.9)
            failures_too_low = [22, 31]
            # Channels are IN SOME MEASUREMENTS stuck to the lower bound
            failures_mid_low = [3, 4, 22, 25, 26, 27, 28, 29, 31, 35, 36, 38, 39, 60]
            # More channels that are stuck somewhere
            failures_more = [2, 3, 4, 5, 22, 25, 26, 27, 28, 29, 31, 35, 36, 38, 39, 56, 59, 60, 61]
            '''failures = np.array(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1,
                 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            )'''
            failures = np.zeros(64, bool)
            failures[failures_huge] = True
            failures[failures_too_low] = True
            failures[failures_mid_low] = True
            failures[failures_more] = True
        else:
            print('Unknown Sensor ID %i! No functionalisation and channel failure data available' % sensor_id)
        correct_channels = np.invert(np.array(failures).astype(bool))
        print('using sensor %i specification' % sensor_id)

        return functionalisations, correct_channels


    def load_sensor_preset(self,preset_file):
        localpath = Path(__file__).absolute().parent.joinpath("presets/" + preset_file)
        if localpath.is_file():
            return np.loadtxt(localpath, int)
        else:
            return np.loadtxt(preset_file, int)