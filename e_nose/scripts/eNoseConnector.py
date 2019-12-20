import re
from threading import Thread

from EventHook import EventHook
from threading import Lock
import serial
import serial.tools.list_ports
import sys

class eNoseConnector:
    """ Connects to an eNose on the given port with the given baudrate.
        Parses the input in a new thread and updates its values accordingly.
        After each full received frame, the onUpdate is triggered.
        
        Use onUpdate like this: connector.onUpdate += <callbackMethod>"""
    
    

    def __init__(self, port: str = None, baudrate: int = 115200, channels: int = 64):

        self.onUpdate: EventHook = EventHook()
        self.finished = False
        self.sensorValues = [0.0] * channels
        self.channels = channels

        #self._readerThread = Thread(target=self._readLoop, daemon=True)
        #self._readerThreadRun = False
        port = '/dev/ttyUSB0'
        if port is None:
            port = self.find_port()

        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=10)
        #self._readerThreadRun = True
        #self._readerThread.start()
        print("Reading data startet")
    
    def __del__(self):
        print('Closing...')
        self.finished = True
        #if self._readerThreadRun:
        #    self._readerThreadRun = False
        #    self._readerThread.join()
        try:
            self.ser.close()
        except Exception:
            pass
            
    def readLoop(self):
        print('Read Loop started')
        while self.finished==False:
            try:
                line = self.ser.readline()
                # line looks like: count=___,var1=____._,var2=____._,....
                match = re.match(b'^count=([0-9]+),(var.+)$', line)
                if match is not None:
                    self.channels = int(match.group(1))
                    sensors_data = match.group(2)
                    self.sensorValues = [float(d.split(b'=')[1]) for d in sensors_data.split(b',')]
                    print("received data for %i sensors (actually %i)" % (self.channels, len(self.sensorValues)))
                    self.onUpdate()
                else:
                    print('line: ',line)    
            except KeyboardInterrupt:
                print('Interrupted, closing...')
                self.finished=True
            except:
                print('Exception raised')

        print('Read Loop finished')

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


if __name__ == '__main__':
    connector = eNoseConnector()
    connector.readLoop()
    def onUpdate():
        print('sensor values: ',connector.sensorValues)
    connector.onUpdate += onUpdate
    
