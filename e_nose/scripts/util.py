import numpy as np
import csv
from datetime import datetime, date, time, timezone

def convert_to_datetime(possible_date):
    return datetime.strptime(possible_date, "%d.%m.%Y - %H:%M:%S")

def read_data_csv(file_name, debug=False):
    sensorId = 0
    functionalisations = []
    failures = []
    #dict with timestamp as key and channels as values
    base_levels = {}
    classes = []

    #stored in the form of timestamp as key, as value you have another dictionary with 'channels', 'label', 'pred_label'
    data = {}


    with open('data.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        #skip the first line 
        next(reader)
        #get sensor ID, dont know if that's every useful but hey :D
        sensorId = [int(s) for s in next(reader)[0].split(':') if s.isdigit()][0]
        #get non-working channels
        failures = [int(c) for c in next(reader)[0].split(':')[1]]
        #get the functionalisation of the different channels
        functionalisations = next(reader)
        functionalisations[0] = int(functionalisations[0].split(':')[1])
        #get already calculated base-levels
        s = next(reader)
        while s[0].startswith('#baseLevel'):
            base_levels[convert_to_datetime(s[0][11:])] = s[1:]
            s = next(reader)

        #get classes
        classes = s
        classes[0] = classes[0].split(':')[1]

        #skip header
        next(reader)

        if debug:
            print(sensorId)
            print(failures)
            print(functionalisations)
            print(base_levels)
            print(classes)

        #parsing data
        for row in reader:
            row_data = {}
            row_data['channels'] = row[1:-2]
            row_data['label'] = row[-2]
            row_data['pred_label'] = row[-1]
            data[convert_to_datetime(row[0])] = row_data

        if debug:
            print(data)
    return sensorId, failures, functionalisations, base_levels, classes, data
