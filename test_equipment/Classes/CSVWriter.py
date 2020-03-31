import csv
import time


class CSVWriter:
    def __init__(self, filename):
        self.csvfile = open(filename, 'w')
        self.filewriter = csv.writer(self.csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL,
                                     escapechar='\\')
        channelstring = []
        for i in range(64):
            channelstring.append(str('Channel' + str(i)))
        headerChannel = str(channelstring).translate({ord('['): '', ord(']'): '', ord('\''): ''})
        # write header
        self.filewriter.writerow(
            ['Time', headerChannel, 'Temperature', 'Gas', 'Humidity', 'Pressure', 'Altitude', 'Label'])

    def writeSample(self, time, eNoseSample, bmeSample, label):
        eNoseCorrected = str(eNoseSample).translate({ord('['): '', ord(']'): '', ord('\''): ''})
        bmeCorrected = str(bmeSample).translate({ord('['): '', ord(']'): '', ord('\''): ''})
        self.filewriter.writerow([time, eNoseCorrected, bmeCorrected, label])

    def writeGradient(self, time, gradient):
        gradientCorrected = str(gradient).translate({ord('['): '', ord(']'): '', ord('\''): ''})
        self.filewriter.writerow([time, gradientCorrected])

    @staticmethod
    def get_filename(labels, num_loops, time_loop, ref_time):
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
