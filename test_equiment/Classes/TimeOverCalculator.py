from datetime import datetime
from datetime import timedelta
import numpy as np


class TimeCalculator:
    def __init__(self, labels, sampleTime, refTime, useGradient=False):
        self.labels = labels
        self.sampleTime = sampleTime
        self.refTime = refTime
        self.useGradient = useGradient
        self.t_end = datetime.now()
        self.measQueue = []

    def nextPos(self, nextPos):
        if nextPos == 0:
            self.t_end = datetime.now() + timedelta(seconds=self.refTime)
        else:
            self.t_end = datetime.now() + timedelta(seconds=self.sampleTime)

    def nextSample(self):
        if self.useGradient:
            if len(self.measQueue) > 9:
                count = sum(i > 1 for i in self.gradients)
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
        self.measQueue.append(measurement  )
        if len(self.measQueue) > 10:
            self.measQueue.pop(0)
            self.gradients = np.array(np.gradient(self.measQueue))
