import time
from datetime import datetime
from datetime import timedelta
from test_equipment.Classes.ServoConnector import ServoConnecor
from test_equipment.Classes.TimeOverCalculator import TimeCalculator
from test_equipment.Classes.BMEConnector import BMEConnector
from test_equipment.Classes.eNoseConnector import eNoseConnector
from ROS.e_nose_raw_publisher.msg import e_nose_raw
from random import choices


class TestEquimentRunner:
    """
    Works the same as the TestRunner in the test_equipment package with the difference it does not save the samples
    to a csv file but sends it via ROS
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

    # labels[0-7] = labels of 0-7 samples, numLoops = number of iterations to be done, timeLoop = time length of a
    # single sample in seconds
    def __init__(self, labels, num_loop, num_Loops, timeLoopSample, timeLoopRef):
        """

        :param labels: array of labels, in correct order of entrances
        :param num_loop: total number of loops which shall be done
        :param timeLoopSample: time of how long a sample shall be measured e.g. 5 min
        :param timeLoopRef: time of how long a reference measurement shall be done e.g. 30min
        """
        # counter of how often a sammple is measured starts with 1 as with 0 it would give a diveded through 0 error
        self.class_sampled = [1, 1, 1, 1, 1, 1, 1]
        # initialize eNose
        eNose = eNoseConnector()
        # initialize servo
        servo = ServoConnecor()
        # initialize ROS node
        e_nose_ros = e_nose_raw()
        # initialize BMEConnector
        bme = BMEConnector()

        # choose a fixed time or switch sample if measured gradient is below a threshold
        timer = TimeCalculator(labels, timeLoopSample, timeLoopRef)
        t_started = datetime.now()
        print('starting measurement at: ', t_started)
        # start detection
        loopsDone = 0
        while loopsDone < num_loop:
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
                        bme_raw = bme.detect()
                        e_nose_ros.run_nonstandalone(eNoseSample, bme_raw, labels[currentPos])
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
        t_timeNeeded = datetime.now() - t_started
        time_end = datetime.now()
        print('Finished measurement at: ', time_end.strftime("%m %d %Y %H:%M:%S"), ' needed time: ', t_timeNeeded)


labelsList = ['ref', 'raisin', 'acetone', 'orange_juice', 'pinot_noir', 'isopropanol', 'wodka']
num_loops = 10
time_loop_min = 2  # in minutes
time_loop = 60. * time_loop_min
time_ref_min = 2  # in minutes
time_ref = time_ref_min * 60
expected_time = num_loops * (len(labelsList) * (time_loop_min + time_ref_min)) + time_ref_min
expected_time_end = datetime.now() + timedelta(minutes=expected_time)
print('expected time: ', timedelta(minutes=expected_time), ' hours stoppes at: ',
      expected_time_end.strftime("%H:%M:%S"))
TestEquimentRunner(labelsList, num_loops, time_loop, time_ref)
