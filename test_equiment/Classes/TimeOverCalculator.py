from datetime import datetime
from datetime import timedelta
import numpy as np
from CSVWriter import CSVWriter


class TimeCalculator:
    def __init__(self, labels, sampleTime, refTime, useGradient=False):
        self.labels = labels
        self.sampleTime = sampleTime
        self.refTime = refTime
        self.useGradient = useGradient
        self.t_end = datetime.now()
        self.measQueue = []
        self.sampleWriter = CSVWriter("gradients.csv")

    def nextPos(self, nextPos):
        if nextPos == 0:
            self.t_end = datetime.now() + timedelta(seconds=self.refTime)
        else:
            self.t_end = datetime.now() + timedelta(seconds=self.sampleTime)

    def nextSample(self):
        return datetime.now() < self.t_end

    def addSampleForGradient(self, measurement):
        self.measQueue.append(measurement)
        if len(self.measQueue) > 10:
            self.measQueue.pop(0)
            gradients = np.array(np.gradient(self.measQueue))
            self.sampleWriter.writeSample(datetime.now(), gradients)
            # print('Current Gradients: ', gradients)

    # else:
    #    print('Queue not long enough')
