#data_processing_old.py
import numpy as np
from measurements import Measurement

def standardize_measurements_2(measurements, num_channels=64, debug=False):
    in_label = False
    prev_meas = None

    last_null_meas = None
    clean_measurements = []

    for measurement in measurements:

        if measurement.label == 'Reset[NUL]':
            last_null_meas = measurement
        else:
            if last_null_meas is None:
                print("ERROR - NO NULL MEASUREMENT FOUND")
                return

            measurement.reference_measurement = last_null_meas
            clean_measurements.append(measurement)
            
    return clean_measurements


def standardize_measurements(data, num_channels=64, use_last=5, debug=False):
    in_label = False
    prev_meas = None

    for ts in data:
        row_data = data[ts]

        if prev_meas is None:
            prev_meas = row_data['channels']
        else:
            if debug:
                # print("row data shape:",np.array(row_data['channels']).shape)
                # print("prev_meas shape:", prev_meas.shape)
                # print("row_data label: ",row_data['label'])
                print(ts)
            prev_meas = np.vstack((prev_meas, row_data['channels']))
            prev_meas = prev_meas[-use_last:, :]

        # assuming that two different labels are not directly after one another
        if row_data['label'] != '' and not in_label:
            in_label = True
            current_means = np.mean(prev_meas, axis=0)

        if row_data['label'] == '':
            in_label = False

        if in_label:
            if debug:
                print("current_means:", current_means)
                print("channels:", row_data['channels'])
                print("standardized:", row_data['channels'] / current_means)
            row_data['channels'] = row_data['channels'] / current_means

    return data

def get_labeled_measurements(data, debug=False):
    current_label = ''
    current_measurement = None
    measurements = []
    
    for ts in data:
        row_data = data[ts]
        if debug:
            ##print("row data label:",row_data['label'])
            #print("current label:", current_label)
            same = current_label == row_data['label']
            issame = current_label is row_data['label']
            #print("same:",same,"; issame:",issame)
        
        if current_label != row_data['label']:
            #change of labels
            if debug:
                print("change in labels; cl:", current_label," - rdl:", row_data['label'])
            
            if current_label != '':
                if debug:
                    print("new measurement; cl:", current_label," - rdl:", row_data['label'])
                meas = Measurement(current_measurement, current_label, time_stamp)
                measurements.append(meas)
                
            current_label = row_data['label']
            time_stamp = ts
            meas_data = {}
            
        if current_label != '' and current_label == row_data['label']:
            if current_measurement is None:
                current_measurement = row_data['channels']
            else:
                current_measurement = np.vstack((current_measurement, row_data['channels']))    
         
    if current_label is not '':
        meas = Measurement(current_measurement, current_label, time_stamp)
        measurements.append(meas)
        
    return measurements


def get_measurement_peak_average(data, num_samples=10):
    max_index = np.argmax(np.abs(data), axis=0)
    # get adjecent samples
    all_peak_data = []
    print("data.len", len(data))
    print("data.shape", data.shape)

    for i, max_i in enumerate(max_index):
        add_right = max_i + int(num_samples/2)+1
        add_left = max_i-int(num_samples/2)

        if max_i >= (len(data)-num_samples/2):
            add_right = len(data)-1

        if max_i < (num_samples/2):
            add_left = 0

        peak_data = np.mean(np.array(data[add_left:add_right,i]))
        print(peak_data.shape)
        print("i: ",i,"; max_i:",max_i, "with data:",data[:,i])
        print("len data:",len(data[:,i]))
        all_peak_data.append(peak_data)

    return all_peak_data


def group_meas_data_by_functionalisation(data, functionalisations):
    averages = None
    for row in data:
        if averages is None:
            _, averages = group_row_data_by_functionalities(row, functionalisations)
        else:
            _, tmp = group_row_data_by_functionalities(row, functionalisations)
            averages = np.vstack((averages, tmp))

    return averages


def group_row_data_by_functionalities(row, functionalities):
    grouped_data = {}
    averaged_data = np.zeros(1+np.max(list(map(int, functionalities))))
    for i in np.unique(functionalities):
        grouped_data[i] = {'values': []}
        
    for value, function in np.vstack((row,functionalities)).T:
        grouped_data[function]['values'].append(value)
        averaged_data[int(function)] = averaged_data[int(function)] + float(value)
    
    for i in np.unique(functionalities):
        averaged_data[int(i)] = averaged_data[int(i)]/len(grouped_data[i]['values'])
    
    return grouped_data, averaged_data
