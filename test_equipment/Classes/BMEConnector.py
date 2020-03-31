import board
from busio import I2C
import adafruit_bme680 as adabme


class BMEConnector:
    # init bme680
    def __init__(self):
        """
        connects to bme680 sensor via I2C
        """
        # load bme settings
        self.i2c = I2C(board.SCL, board.SDA)
        self.bme680 = adabme.Adafruit_BME680_I2C(self.i2c, debug=False)
        self.bme680.sea_level_pressure = 978.33  #estimated pressure of germany
        self.ErrorMsg = False

        # read data from bme

    def detect(self):
        """
        get current sensor values of bme680 sensor
        :return: sensor values as arry
        """
        try:
            sensordata = [None] * 5
            sensordata[0] = self.bme680.temperature
            sensordata[1] = self.bme680.gas
            sensordata[2] = self.bme680.humidity
            sensordata[3] = self.bme680.pressure
            sensordata[4] = self.bme680.altitude
            return sensordata
        except:
            if not self.ErrorMsg:
                print("Sensor quited, no bme sensor avalaible any more!")
                self.ErrorMsg = True
            return [0] * 5

    def __del__(self):
        print('closing bme680 connection')
        # self.i2c.close()
