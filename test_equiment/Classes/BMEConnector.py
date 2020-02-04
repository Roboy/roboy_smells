import board
from busio import I2C
import adafruit_bme680 as adabme


class BMEConnector:
    # init bme680
    def __init__(self):
        # load bme settings
        self.i2c = I2C(board.SCL, board.SDA)
        self.bme680 = adabme.Adafruit_BME680_I2C(self.i2c, debug=False)
        self.bme680.sea_level_pressure = 978.33

        # read data from bme

    def detect(self):
        sensordata = [None] * 5
        sensordata[0] = self.bme680.temperature
        sensordata[1] = self.bme680.gas
        sensordata[2] = self.bme680.humidity
        sensordata[3] = self.bme680.pressure
        sensordata[4] = self.bme680.altitude
        return sensordata

    def __del__(self):
        print('closing bme680 connection')
        # self.i2c.close()
