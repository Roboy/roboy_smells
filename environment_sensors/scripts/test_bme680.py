import time
import board
from busio import I2C
import adafruit_bme680 as adabme

i2c = I2C(board.SCL, board.SDA)
bme680 = adabme.Adafruit_BME680_I2C(i2c, debug=False)
bme680.sea_level_pressure = 953.25

while True:
   print("\nTemperature: %0.1f C" % bme680.temperature)
   print(bme680.gas)
   print("Humidity: %0.1f %%" % bme680.humidity)
   print("Pressure: %0.3f hPa" % bme680.pressure)
   print("Altitude: %0.2f meters" % bme680.altitude)
   time.sleep(1)
