#!/usr/bin/env python
import re
import serial
import serial.tools.list_ports


class eNoseConnector:
    """ Connects to an eNose on the given port with the given baudrate.
        Parses the input in a new thread and updates its values accordingly.
        After each full received frame, the onUpdate is triggered.

        Use onUpdate like this: connector.onUpdate += <callbackMethod>"""

    def __init__(self, port=None, baudrate=115200, channels=64):
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
            line = self.ser.readline()
            # line looks like: count=___,var1=____._,var2=____._,....
            match = re.match(b'^count=([0-9]+),(var.+)$', line)
            if match is not None:
                if self.first_measurement:
                    print('Start measuring!')
                    self.first_measurement = False
                self.channels = int(match.group(1))
                sensors_data = match.group(2)
                self.sensorValues = [float(d.split(b'=')[1]) for d in sensors_data.split(b',')]
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
