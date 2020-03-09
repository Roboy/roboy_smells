from typing import List

import numpy as np

from e_nose.EventHook import EventHook
from e_nose.file_reader import get_sensor_spec
from e_nose.measurements import Measurement, StandardizationType, DataRowsSet_t, DataRow_t, Functionalisations_t, \
    WorkingChannels_t


class OnlineReader:
    """
    Class that reads in subsequent samples of the sensor and allows to generate Measurement objects with reference data
    from the last n samples
    """
    l1_factor = 1e-3
    """ Factor for the low-pass filter """

    def __init__(self, sensor_id: int,
                 standardization: StandardizationType = StandardizationType.LOWPASS_FILTER,
                 override_functionalisations: Functionalisations_t = None,
                 override_working_channels: WorkingChannels_t = None,
                 max_history_length: int = 100_000):
        """

        :param sensor_id: ID of the sensor to use => to know the functionalisations and failure bits
        :param standardization: The kind of standardization to apply to the live data;
                they may not be entirely correct, though...
        :param max_history_length: Max length of the history data... to avoid using too much memory
        """
        self.data_buffer: np.ndarray = np.empty((max_history_length, 64))
        """ data buffer that contains empty rows for all data that might come in one day """
        self.log_lowpass_buffer: np.ndarray = np.empty((max_history_length, 64))
        """ log_lowpass data buffer that contains empty rows for all data that might come in one day """
        self.current_length: int = 0
        self.invoke_callback: EventHook = EventHook()
        self.invoke_at = 99999999999
        """ current length of the data buffer """
        self.log_lowpass_current = None
        self.max_history_length: int = max_history_length
        self.standardization: StandardizationType = standardization

        self.functionalisations, self.working_channels = get_sensor_spec(sensor_id)
        if override_functionalisations is not None:
            self.functionalisations = override_functionalisations
        if override_working_channels is not None:
            self.working_channels = override_working_channels

    def set_standardization_type(self, standardization: StandardizationType):
        self.standardization = standardization

    def add_sample(self, sample: np.ndarray) -> ():
        """ Adds a single sample to the internal state; updates the internal low-pass automatically """
        if self.log_lowpass_current is None:
            self.log_lowpass_current = np.log(sample)

        self.log_lowpass_current = (self.log_lowpass_current + np.log(sample) * self.l1_factor) / (1.0 + self.l1_factor)

        self.data_buffer[self.current_length] = sample
        self.log_lowpass_buffer[self.current_length] = self.log_lowpass_current

        self.current_length += 1

        if self.current_length > self.invoke_at:
            self.invoke_at = 99999999999
            self.invoke_callback()

        # if len(self.data.shape) > self.max_history_length:
        #    remove = len(self.data.shape) - self.max_history_length
        #    self.data = np.delete(self.data, range(remove))
        #    self.log_lowpass = np.delete(self.log_lowpass, range(remove))

    def set_trigger_in(self, in_n: int = 50):
        """ Sets a trigger to call the given callback function in in_n steps """
        self.set_trigger_at(self.current_length + in_n)

    def set_trigger_at(self, at: int = 50):
        """ Sets a trigger to call the given callback function at the given sample-count """
        self.invoke_at = at

    def get_since_n_as_measurement(self, n):
        return self.get_last_n_as_measurement(self.current_length - n)

    def get_last_n_as_measurement(self, n: int = 300) -> Measurement:
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


class FileAsOnlineReader:
    """ Builds an OnlineReader and feeds the data from one file to it """

    def __init__(self, sensor_id: int, data: DataRowsSet_t,
                 standardization: StandardizationType = StandardizationType.LOWPASS_FILTER):
        self.sensor_id = sensor_id
        self.standardization = standardization
        self.reader: OnlineReader = OnlineReader(sensor_id, standardization, max_history_length=len(data))
        """ The reader that is fed the data from the file. """
        self.currpos = 0
        self.indices: List[str] = list(data.keys())
        self.data: DataRowsSet_t = data

    def reset(self):
        self.reader = OnlineReader(self.sensor_id, self.standardization, max_history_length=len(self.data))
        self.currpos = 0

    def set_standardization_type(self, standardization: StandardizationType):
        self.standardization = standardization
        self.reader.set_standardization_type(standardization)

    def feed_samples(self, n: int = 1):
        """ Feed the next n samples to the OnlineReader
        returns if there are no more samples to be fed """
        for i in range(n):
            if self.currpos >= len(self.indices):
                return
            d = self.data_at_index(self.currpos)
            self.reader.add_sample(d['channels'])
            self.currpos += 1

    def data_at_index(self, index: int) -> DataRow_t:
        return self.data[self.indices[index]]

    def get_current_label(self) -> str:
        """ Returns the current label of the ground truth data """
        pos = self.currpos
        if pos >= len(self.indices):
            pos = len(self.indices) - 1
        return self.data_at_index(pos)['label']

    def get_last_n_as_measurement(self, n: int = 300) -> Measurement:
        return self.reader.get_last_n_as_measurement(n)

    def get_all_measurements_every(self, n: int, m: int = -1, initial_offset: int = 10, add_labels: bool = True) -> List[Measurement]:
        """ Plays the whole file and creates a Measurement object
            over the length of the last m samples every n samples
            :param n: returns a new Measurement every n
            :param m: each Measurement object encompasses the previous m samples
            :param initial_offset: Skip the first few samples (required for StandardizationType.LAST_REFERENCE)
            :param add_labels: whether to add the ground truth labels to the data
        """

        if m < 0:
            m = n

        if m > n:
            # skip the first few samples so the first measurement object has enough to look back on
            self.feed_samples(m - n)
        self.feed_samples(initial_offset)

        measurements: List[Measurement] = []
        while self.currpos < len(self.indices):
            self.feed_samples(n)
            # print(self.currpos)
            meas = self.get_last_n_as_measurement(m)
            meas.label = self.get_current_label()
            measurements.append(meas)

        return measurements
