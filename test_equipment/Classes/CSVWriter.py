import csv
import time


class CSVWriter:
    """
    Writes the sensor data of the eNose and the  bme680 into a csv file.
    First it creates a filename containing the measured odors, the current time and how many loops are done.
    Second it creates a header with columns for time, the 64 channels of the eNose, the 5 channels of the bme sensor
    and the labeling.
    It receives for each measurement all these informations and creates a new entry as specified.
    """

    def __init__(self, filename):
        """
        create a file
        :param filename: filename of the csv document
        """
        self.csvfile = open(filename, 'w')
        self.filewriter = csv.writer(self.csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL,
                                     escapechar='\\')
        # create header
        channelstring = []
        for i in range(64):
            channelstring.append(str('Channel' + str(i)))
        headerChannel = str(channelstring).translate({ord('['): '', ord(']'): '', ord('\''): ''})
        # write header
        self.filewriter.writerow(
            ['Time', headerChannel, 'Temperature', 'Gas', 'Humidity', 'Pressure', 'Altitude', 'Label'])

    def writeSample(self, time, eNoseSample, bmeSample, label):
        """
        create new entry in csv file
        :param time: time of measurement
        :param eNoseSample: array of 64 entries from eNose
        :param bmeSample: array of 5 entrries from bme680 sensor
        :param label: actual label
        :return: write information to csv
        """
        eNoseCorrected = str(eNoseSample).translate({ord('['): '', ord(']'): '', ord('\''): ''})
        bmeCorrected = str(bmeSample).translate({ord('['): '', ord(']'): '', ord('\''): ''})
        self.filewriter.writerow([time, eNoseCorrected, bmeCorrected, label])

    @staticmethod
    def get_filename(labels, num_loops, time_loop, ref_time):
        """
        generate filename with given information
        :param labels: array of all samples
        :param num_loops: how many loops will be performed
        :param time_loop: how long is the time of a measurement of a sample
        :param ref_time: how long is the time of a reference measurement
        :return: generated filename
        """
        time_sec = time.time()
        time_now = time.gmtime(time_sec)
        filename = 'data_'
        for i in range(1, len(labels)):
            filename = filename + labels[i] + '_'
        time_string = str(time_now[0]) + '-' + str(time_now[1]) + '-' + str(time_now[2]) + '_' + str(
            time_now[3]) + '_' + str(time_now[4])
        filename = filename + str(num_loops) + '_loops_for_' + str(time_loop / 60) + '_min_referenceTime_' + str(
            ref_time / 60) + '_min_' + time_string + '.csv'
        print('filename: ', filename)
        return filename

    def __del__(self):
        print('Closing csv')
        try:
            self.csvfile.close()
        except Exception:
            pass
