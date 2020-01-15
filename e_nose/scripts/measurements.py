import numpy as np
import data_visualization as dv
import data_processing as dp


class Measurement:
    def __init__(self, data, label, time_stamp, correct_channels, functionalisations):
        self.label = label
        self.ts = time_stamp
        self.correct_channels = correct_channels
        self.data = data
        self.functionalisations = functionalisations
        self.correct_functionalisations = np.array(functionalisations)[correct_channels]
        self.reference_measurement = None
        self.standardized_data = None
        self.total_avg = None
        self.last_average = None
        self.peak_avg = None
        self.grouped_data = None
        self.total_avg_by_group = None
        self.peak_avg_by_group = None
        self.gradients = None

    def get_data(self, standardize=True, force=False):
        if standardize:
            if self.standardized_data is None or force:
                if self.reference_measurement is None:
                    self.standardized_data = 100*(self.data[:, self.correct_channels]/(1e-15 + self.get_last_average(10, standardize=False, force=force))-1)
                else:
                    self.standardized_data = 100*(self.data[:, self.correct_channels]/(1e-15 + self.reference_measurement.get_last_average(10, standardize=False, force=force))-1)

            return self.standardized_data

        return self.data[:, self.correct_channels]

    def get_last_average(self, num_last, standardize=True, force=False):
        if self.last_average is None or force:
            self.last_average = np.mean(self.get_data(standardize, force)[-num_last:,:], axis=0)
        return self.last_average

    def get_total_average(self, standardize=True, force=False):
        if self.total_avg is None or force:
            self.total_avg = np.mean(self.get_data(standardize, force), axis=0)
        return self.total_avg

    def get_peak_average(self, force=False):
        if self.peak_avg is None or force:
            self.peak_avg = dp.get_measurement_peak_average(self.get_data(force=force))
        return self.peak_avg

    def get_gradients(self):
        if self.gradients is None:
            self.gradients = np.gradient(self.get_data(), axis=1)
        return self.gradients

    # METHODS FOR GROUPED BY FUNCTIONALISATION
    def get_grouped_measurements(self, force=False):
        if self.grouped_data is None:
            self.grouped_data = dp.group_meas_data_by_functionalisation(self.get_data(force=force), self.correct_functionalisations)
        return self.grouped_data

    def get_total_average_by_group(self, force=False):
        if self.total_avg_by_group is None:
            self.total_avg_by_group = np.mean(self.get_grouped_measurements(force=force), axis=0)
        return self.total_avg_by_group

    def get_peak_average_by_group(self, num_samples=10, force=False):
        if self.peak_avg_by_group is None:
            self.peak_avg_by_group = \
                dp.get_measurement_peak_average(self.get_grouped_measurements(force=force), num_samples)
        return self.peak_avg_by_group

    # METHODS FOR VISUALIZATION OF SINGLE MEASUREMENTS
    def pretty_print(self):
        dv.pretty_print(self)

    def pretty_draw(self):
        dv.pretty_draw_meas(self)
