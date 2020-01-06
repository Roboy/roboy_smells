#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String, Float32 
import time
import board
import sys

from busio import I2C
import adafruit_bme680 as adabme

bme680 = None

def bme680_node():
   pub_temp = rospy.Publisher('env_sensor/current_temp', Float32, queue_size=10)
   pub_gas = rospy.Publisher('env_sensor/current_gas', Float32, queue_size=10)
   pub_humidity = rospy.Publisher('env_sensor/current_humidity', Float32, queue_size=10)

   pub_pressure = rospy.Publisher('env_sensor/current_pressure', Float32, queue_size=10)

   pub_altitude = rospy.Publisher('env_sensor/current_altitude', Float32, queue_size=10)
   rospy.init_node('bme680', anonymous=True)
   rate = rospy.Rate(1) # 10hz
   while not rospy.is_shutdown():
      current_temp, current_gas, current_humidity, current_pressure, current_altitude = detect()
      '''
      temp_str = ("Current temp in deg C is {0:.2f}").format(current_temp)
      gas_str = ("Current gas resistance in Ohm is {0:.2f}").format(current_gas)
      humid_str = ("Current humidity in % is {0:.2f}").format(current_humidity)
      pressure_str = ("Current air pressure in hPa is {0:.2f}").format(current_pressure)
      altitude_str = ("Current altitude in m is {0:.2f}").format(current_altitude)
      '''
      pub_temp.publish(current_temp)
      pub_gas.publish(current_gas)
      pub_humidity.publish(current_humidity)
      pub_pressure.publish(current_pressure)
      pub_altitude.publish(current_altitude)
      rate.sleep()


def detect():
   global bme680
   temp = bme680.temperature
   gas = bme680.gas 
   humidity = bme680.humidity
   pressure = bme680.pressure
   altitude = bme680.altitude
   return temp, gas, humidity, pressure, altitude
    
def init():
   global bme680
   i2c = I2C(board.SCL, board.SDA)
   bme680 = adabme.Adafruit_BME680_I2C(i2c, debug=False)
   bme680.sea_level_pressure = 953.25

if __name__ == '__main__':
   try:
      init()
      bme680_node()
   except rospy.ROSInterruptException:
      pass