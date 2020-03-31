from datetime import datetime
from datetime import timedelta
import numpy as np


class TimeCalculator:
    """
    Caluclates if it is time for the next sample, wich can be done by two different methods)
    1) by time: specifiy how long the reference measurement and the sample time will last, it will switch accordingly
    2) by gradient: no time needed, it switches if the last ten measurements are below a certain threshold
        -> eNose is regenerated
    """

    def __init__(self, sampleTime, refTime, useGradient=False):
        """
        :param sampleTime: how long a measurement of an odour will last
        :param refTime: how long a reference measurement will last
        :param useGradient: if timing shall be ignored and gradients shall be used
        """
        self.sampleTime = sampleTime
        self.refTime = refTime
        self.useGradient = useGradient
        self.t_end = datetime.now()
        self.measQueue = []

    def nextPos(self, nextPos):
        """
        reset timer for next position
        :param nextPos: number of next position as it reference and sample time is different
        :return: new end time
        """
        if nextPos == 0:
            self.t_end = datetime.now() + timedelta(seconds=self.refTime)
        else:
            self.t_end = datetime.now() + timedelta(seconds=self.sampleTime)

    def nextSample(self):
        """
        check if is time for the next sample by either checking gradients or if enough time passed
        :return:
        """
        if self.useGradient:
            if len(self.measQueue) > 9:
                count = sum(i > 1 for i in self.gradients)
                # check if all 10 gradients are below 2, the actual threshold have to be fitted to own needs
                # and to sensor
                print('count ', count)
                if count < 2:
                    print('Changing to next pos: Gradients are :', self.gradients)
                    return False
                else:
                    return True
            else:
                return True
        else:
            return datetime.now() < self.t_end

    def addSampleForGradient(self, measurement):
        """
        add a new sample for calculating gradients
        :param measurement: an array of 64 values from eNose
        :return: array of 10 gradients
        """
        self.measQueue.append(measurement)
        if len(self.measQueue) > 10:
            self.measQueue.pop(0)
            self.gradients = np.array(np.gradient(self.measQueue))
