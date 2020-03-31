import time
from datetime import datetime
from datetime import timedelta
from Classes.CSVWriter import CSVWriter
from Classes.BMEConnector import BMEConnector
from Classes.ServoConnector import ServoConnecor
from Classes.eNoseConnector import eNoseConnector
from Classes.TimeOverCalculator import TimeCalculator
from Uploader.Uploader import Uploader
from random import choices


class TestEquimentRunner:
    """
    Main class for gather data with the data gathering machine.
    It
    It also used a random sampling with a equally distribution so that each sample gets more or less measured the same
    amount
    """

    def get_next_sample(self):
        """
        Calculates the next random sample, the less a sample is measured the more likely it is that it is the next sample
        :return: the index of a sample
        """
        weights_samples = [1 / self.class_sampled[0], 1 / self.class_sampled[1],
                           1 / self.class_sampled[2], 1 / self.class_sampled[3],
                           1 / self.class_sampled[4], 1 / self.class_sampled[5],
                           1 / self.class_sampled[6]]
        rand_int = choices(range(1, 8), weights_samples)
        self.class_sampled[rand_int[0] - 1] += 1
        return rand_int[0]

    # labels[0-7] = labels of 0-7 samples, numLoops = number of iterations to be done, timeLoop = timelength of a
    # single sample in seconds
    def __init__(self, labels, numLoops, timeLoopSample, timeLoopRef, useGradientTime=False, filename=None):
        """
        Loops through the different sample which is given in the array, which must be in the correct order. Otherwise
        it will label a measurement with a differen odor. It switches the router automatically to the next entrance.
        The more often a odor is measured the more unlikley it becomes selected in the next round, so each sample should
        be measured equally.
        It can be chosen if a measurement of a sample ends after a specific time peroid or if the last ten measurements
        are below a certain threshold
        The data received from the measurements are saved in a CSV file and uploaded automatically to a gdrive folder.
        Before a sample gets measured, a reference measurement will be done. Also afterwards to give the eNose a break.
        How to use this scipt is explained in the README.md.
        :param labels: array of labels, in correct order of entrances
        :param numLoops: total number of loops which shall be done
        :param timeLoopSample: time of how long a sample shall be measured e.g. 5 min
        :param timeLoopRef: time of how long a reference measurement shall be done e.g. 30min
        """
        self.class_sampled = [1, 1, 1, 1, 1, 1, 1]
        # initialize eNose
        eNose = eNoseConnector()
        # initialize bme680
        bme = BMEConnector()
        # initialize csv file
        if filename is None:
            filename = CSVWriter.get_filename(labels, num_loops, timeLoopSample, timeLoopRef)
        sampleWriter = CSVWriter(filename)
        # initialize servo
        servo = ServoConnecor()

        # choose a fixed time or switch sample if measured gradient is below a threshold
        timer = TimeCalculator(labels, timeLoopSample, timeLoopRef, useGradientTime)
        t_started = datetime.now()
        print('starting measurement at: ', t_started)
        # start detection
        loopsDone = 0
        while loopsDone < numLoops:
            currentPos = 0
            nextPos = self.get_next_sample()
            print('round: ', loopsDone + 1)
            while nextPos != len(labels):
                if labels[currentPos]:
                    # goto position i, start fan,start with smelling for time
                    servo.setSample(currentPos)
                    timer.nextPos(currentPos)
                    while timer.nextSample():
                        eNoseSample = eNose.detect()
                        bmeSample = bme.detect()
                        timer.addSampleForGradient(eNoseSample)
                        sampleWriter.writeSample(time.time(), eNoseSample, bmeSample, labels[currentPos])
                        time.sleep(0.5)
                    time_now = datetime.now()
                    feedback = time_now.strftime("%H:%M:%S") + '  current sample ' + labels[
                        currentPos] + ' measured'
                    if currentPos == 0:
                        currentPos = nextPos
                    else:
                        currentPos = 0
                        nextPos = self.get_next_sample()
                    feedback += str(', next sample will be: ' + labels[currentPos])
                    print(feedback)
            loopsDone += 1

        servo.setSample(0)
        t_end = time.time() + timeLoopRef
        while time.time() < t_end:
            eNoseSample = eNose.detect()
            bmeSample = bme.detect()
            sampleWriter.writeSample(time.time(), eNoseSample, bmeSample, labels[0])
            time.sleep(0.5)
        time_now = datetime.now()
        feedback = time_now.strftime("%H:%M:%S") + '  current sample ' + labels[0] + ' measured'
        print(feedback)

        t_timeNeeded = datetime.now() - t_started
        time_end = datetime.now()
        print('Finished measurement at: ', time_end.strftime("%m %d %Y %H:%M:%S"), ' needed time: ', t_timeNeeded)

        # upload the generated csv file to gdrive roboy smells folder
        up = Uploader()
        up.uploadToGdrive(filename)


labelsList = ['ref', 'raisin', 'acetone', 'orange_juice', 'pinot_noir', 'isopropanol', 'wodka']
num_loops = 20
time_loop_min = 2  # in minutes
time_loop = 60. * time_loop_min
time_ref_min = 2  # in minutes
time_ref = time_ref_min * 60
expected_time = num_loops * (len(labelsList) * (time_loop_min + time_ref_min)) + time_ref_min
expected_time_end = datetime.now() + timedelta(minutes=expected_time)
print('expected time: ', timedelta(minutes=expected_time), ' hours stoppes at: ',
      expected_time_end.strftime("%H:%M:%S"))
TestEquimentRunner(labelsList, num_loops, time_loop, time_ref, False)
