import time
from datetime import datetime
from datetime import timedelta
from test_equiment.Classes.ServoConnector import ServoConnecor
from test_equiment.Classes.TimeOverCalculator import TimeCalculator
from test_equiment.Classes.BMEConnector import BMEConnector
from ROS.e_nose_raw_publisher.src.e_nose_connector import eNoseConnector
from ROS.e_nose_raw_publisher.src.e_nose_node import eNoseRawNode
from random import choices


class TestEquimentRunner:

    def get_next_sample(self):
        weights_samples = [1 / self.class_sampled[0], 1 / self.class_sampled[1],
                           1 / self.class_sampled[2], 1 / self.class_sampled[3],
                           1 / self.class_sampled[4], 1 / self.class_sampled[5],
                           1 / self.class_sampled[6]]
        rand_int = choices(range(1, 8), weights_samples)
        self.class_sampled[rand_int[0] - 1] += 1
        return rand_int[0]

    # labels[0-7] = labels of 0-7 samples, numLoops = number of iterations to be done, timeLoop = time length of a
    # single sample in seconds
    def __init__(self, labels, total_num_loop, sub_num_Loops, timeLoopSample, timeLoopRef):
        self.class_sampled = [1, 1, 1, 1, 1, 1, 1]
        # initialize eNose
        eNose = eNoseConnector()
        # initialize servo
        servo = ServoConnecor()
        # initialize ROS node
        e_nose_ros = eNoseRawNode()
        # initialize BMEConnector
        bme = BMEConnector()

        for y in range(total_num_loop):
            # choose a fixed time or switch sample if measured gradient is below a threshold
            timer = TimeCalculator(labels, timeLoopSample, timeLoopRef)
            t_started = datetime.now()
            print('starting measurement at: ', t_started)
            # start detection
            loopsDone = 0
            while loopsDone < sub_num_Loops:
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
                            bme = bme.detect()
                            e_nose_ros.run_nonstandalone(eNoseSample, bme, labels[currentPos])
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
total_num_loops = 10
sub_num_loops = 20
time_loop_min = 2  # in minutes
time_loop = 60. * time_loop_min
time_ref_min = 2  # in minutes
time_ref = time_ref_min * 60
expected_time = total_num_loops * sub_num_loops * (len(labelsList) * (time_loop_min + time_ref_min)) + time_ref_min
expected_time_end = datetime.now() + timedelta(minutes=expected_time)
print('expected time: ', timedelta(minutes=expected_time), ' hours stoppes at: ',
      expected_time_end.strftime("%H:%M:%S"))
TestEquimentRunner(labelsList, total_num_loops, sub_num_loops, time_loop, time_ref)
