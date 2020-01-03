import time
import sys
import csv

#load bme
import board
from busio import I2C
import adafruit_bme680 as adabme

#load eNose
import re
import serial
import serial.tools.list_ports

#load servo
import RPi.GPIO as GPIO
import time
from servo_config import get_control_GPIO, get_channels


class CSVWriter:
    def __init__(self,filename):
        self.csvfile = open(filename, 'w')
        self.filewriter = csv.writer(self.csvfile, delimiter=',',quotechar=' ', quoting=csv.QUOTE_MINIMAL,escapechar='\\')
        channelstring = []
        for i in range(64):
            channelstring.append(str('Channel'+str(i)))
        headerChannel = str(channelstring).translate({ord('['): '', ord(']'): '', ord('\''): ''})                         
        #write header                      
        self.filewriter.writerow(['Time',headerChannel,'Temperature','Gas','Humidity','Pressure','Altitude','Label'])
    
    def writeSample(self,time, eNoseSample,bmeSample,label):
        eNoseCorrected = str(eNoseSample).translate({ord('['): '', ord(']'): '', ord('\''): ''})
        bmeCorrected = str(bmeSample).translate({ord('['): '', ord(']'): '', ord('\''): ''})
        self.filewriter.writerow([time,eNoseCorrected,bmeCorrected,label])
        
    def __del__(self):
        print('Closing csv')
        try:
            self.csvfile.close()
        except Exception:
            pass

class ServoConnecor:
    #load servo settings
    def __init__(self):
        servoPIN = get_control_GPIO()

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(servoPIN, GPIO.OUT)
       
        self.p = GPIO.PWM(servoPIN, 50)
        self.Channel = get_channels()
        '''
        self.Channel = [3.8,   #channel 1 
             4.8,   #channel 2
             5.8,   #channel 3
             6.75,  #channel 4
             7.8,   #channel 5
             8.8,   #channel 6
             10.    #channel 7
            ]
        '''
        self.p.start(self.Channel[0])
        
    def setSample(self,channel):
        self.p.ChangeDutyCycle(self.Channel[channel])
        
    def __del__(self):
        self.p.stop()
        GPIO.cleanup()

class BMEConnector:
    #init bme680
    def __init__(self):
        #load bme settings
        self.i2c = I2C(board.SCL, board.SDA)
        self.bme680 = adabme.Adafruit_BME680_I2C(self.i2c, debug=False)
        self.bme680.sea_level_pressure = 978.33 

    #read data from bme
    def detect(self):
        sensordata = [None]*5
        sensordata[0] = self.bme680.temperature
        sensordata[1] = self.bme680.gas 
        sensordata[2] = self.bme680.humidity
        sensordata[3] = self.bme680.pressure
        sensordata[4] = self.bme680.altitude
        return sensordata
            
    def __del__(self):
        print('closing bme680 connection')
        #self.i2c.close()

class eNoseConnector:
    """ Connects to an eNose on the given port with the given baudrate.
        Parses the input in a new thread and updates its values accordingly.
        After each full received frame, the onUpdate is triggered.
        
        Use onUpdate like this: connector.onUpdate += <callbackMethod>"""
    

    def __init__(self, port: str = None, baudrate: int = 115200, channels: int = 64):
        self.sensorValues = [0.0] * channels
        self.channels = channels
        
        if port is None:
            port = self.find_port()

        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=10)

        self.first_measurement = True

    
    def __del__(self):
        print('Closing eNose connection')
        try:
            self.ser.close()
        except Exception:
            pass
        
    def detect(self):
        
        try:
            #print("detect")
            #print(self.ser.is_open)
            line = self.ser.readline()
            #print(line)
            #line looks like: count=___,var1=____._,var2=____._,....
            match = re.match(b'^count=([0-9]+),(var.+)$', line)
            if match is not None:
                if self.first_measurement:
                    print('Start measuring!')
                    self.first_measurement = False
                self.channels = int(match.group(1))
                sensors_data = match.group(2)
                self.sensorValues = [float(d.split(b'=')[1]) for d in sensors_data.split(b',')]
                #print("received data for %i sensors (actually %i)" % (self.channels, len(self.sensorValues)))
                #print(sensors_data)
                #print(self.sensorValues)
                return self.sensorValues
            else:
                print('No pattern matched!')
        except:
            print('Exception raised')


    @staticmethod
    def find_port():
        ports = list(serial.tools.list_ports.comports())
        port = None
        for p in ports:
            print('Checking port %s / %s' % (p[0], p[1]))
            if "CP2102" in p[1]:
                port = p
                break

        if port is None:
            print('Could not find a connected eNose')
            return None

        print('Using the eNose connected on:')
        print(port[0] + ' / ' + port[1])
        return port[0]

class TestEquimentRunner:
    #labels[0-7] = labels of 0-7 samples, numLoops = number of iterations to be done, timeLoop = timelength of a single sample in seconds
    def __init__(self,labels,numLoops,timeLoop,filename):
       
        #initialize eNose
        eNose = eNoseConnector()
        #initialize bme680
        bme = BMEConnector()
        #initialize csv file
        sampleWriter = CSVWriter(filename)
        #initialize servo
        servo = ServoConnecor()
        
        #start detection
        loopsDone = 0
        while(loopsDone<numLoops):
            currentPos = 0
            nextPos = 1
            print('round: ',loopsDone+1)
            while(nextPos != len(labels)):
                if(labels[currentPos]):
                    #goto position i, start fan,start with smelling for time 
                    servo.setSample(currentPos)
                    t_end = time.time() + timeLoop
                    while(time.time() < t_end):
                        eNoseSample = eNose.detect()
                        bmeSample = bme.detect()
                        sampleWriter.writeSample(time.time(),eNoseSample,bmeSample,labels[currentPos])
                        time.sleep(0.2)
                    time_now = time.gmtime(time.time())
                    time_str = str(time_now[3]) + ':' + str(time_now[4])
                    feedback = time_str + '  current sample '+labels[currentPos]+' measured'
                    if(currentPos == 0):
                        currentPos = nextPos
                    else:
                        currentPos = 0
                        nextPos +=1
                    feedback +=str(', next sample will be: ' + labels[currentPos])
                    print(feedback)
            loopsDone +=1
        
        servo.setSample(0)
        t_end = time.time() + timeLoop
        while(time.time() < t_end):
            eNoseSample = eNose.detect()
            bmeSample = bme.detect()
            sampleWriter.writeSample(time.time(),eNoseSample,bmeSample,labels[0])
            time.sleep(0.2)
        time_now = time.gmtime(time.time())
        time_str = str(time_now[3]) + ':' + str(time_now[4])
        feedback = time_str + '  current sample '+labels[0]+' measured'
        print(feedback)


def get_filename(labels, num_loops, time_loop):
    time_sec = time.time()
    time_now = time.gmtime(time_sec)
    #print(time_now)
    filename = 'data_'
    for i in range(1, len(labels)):
        filename = filename + labels[i] + '_' 
    #print(filename)
    #print(time_now[0])
    time_string = str(time_now[0]) + '-' + str(time_now[1]) + '-' + str(time_now[2]) + '_' + str(time_now[3]) + ':' + str(time_now[4])
    filename = filename + str(num_loops) + '_loops_for_' + str(time_loop) + '_min_' + time_string + '.csv'
    print('filename: ', filename)
    return filename

labels = ['null','wodka','orange_juice','red_wine','lemon_juice','coffee','garlic']
num_loops = 10
time_loop_min = 10. # in minutes
time_loop = 60. * time_loop_min
filename = get_filename(labels, num_loops, time_loop_min)
TestEquimentRunner(labels, num_loops, time_loop, filename)
#enose = eNoseConnector()
#enose.detect()
