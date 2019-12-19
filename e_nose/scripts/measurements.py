import numpy as np
import data_visualization as dv
import data_processing_2 as dp


class Measurement:
    def __init__(self, data, label, time_stamp):
        self.data = data
        self.label = label
        self.ts = time_stamp
        self.reference_measurement = None
        self.standardized_data = None
        self.total_avg = None
        self.peak_avg = None
        self.grouped_data = None
        self.total_avg_by_group = None
        self.peak_avg_by_group = None

    def get_data(self, standardize=True):
        if standardize:
            if self.standardized_data is None:
                if self.reference_measurement is None:
                    print("ERROR - NO REFERENCE MEASUREMENT")
                    return self.data
                self.standardized_data = 100*(self.data/self.reference_measurement.get_total_average(standardize=False)-1)

            return self.standardized_data

        return self.data

    def get_total_average(self, standardize=True):
        if self.total_avg is None:
            self.total_avg = np.mean(self.get_data(standardize), axis=0)
        return self.total_avg

    def get_peak_average(self):
        if self.peak_avg is None:
            self.peak_avg = dp.get_measurement_peak_average(self.get_data())
        return self.peak_avg

    # METHODS FOR GROUPED BY FUNCTIONALISATION
    def get_grouped_measurements(self, functionalisations):
        if self.grouped_data is None:
            self.grouped_data = dp.group_meas_data_by_functionalisation(self.get_data(), functionalisations)
        return self.grouped_data

    def get_total_average_by_group(self, functionalisations):
        if self.total_avg_by_group is None:
            self.total_avg_by_group = np.mean(self.get_grouped_measurements(functionalisations), axis=0)
        return self.total_avg_by_group

    def get_peak_average_by_group(self, functionalisations, num_samples = 10):
        if self.peak_avg_by_group is None:
            self.peak_avg_by_group = \
                dp.get_measurement_peak_average(self.get_grouped_measurements(functionalisations), num_samples)
        return self.peak_avg_by_group

    # METHODS FOR VISUALIZATION OF SINGLE MEASUREMENTS
    def pretty_print(self):
        dv.pretty_print(self)

    def pretty_draw(self):
        dv.pretty_draw_meas(self)
