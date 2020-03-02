#!/usr/bin/env python
# license removed for brevity
import json
import time
import rospy
from std_msgs.msg import String
import board
from busio import I2C
import adafruit_bme680 as adabme

bme680 = None



def bme680_node():
    pub_temp = rospy.Publisher('env_sensor/current_temp', String, queue_size=10)
    pub_gas = rospy.Publisher('env_sensor/current_gas', String, queue_size=10)
    pub_humidity = rospy.Publisher('env_sensor/current_humidity', String, queue_size=10)
    pub_pressure = rospy.Publisher('env_sensor/current_pressure', String, queue_size=10)
    pub_altitude = rospy.Publisher('env_sensor/current_altitude', String, queue_size=10)
    rospy.init_node('bme680', anonymous=True)
    rate = rospy.Rate(1)  # 10hz
    print('started ros node successfully!')
    while not rospy.is_shutdown():
        current_temp, current_gas, current_humidity, current_pressure, current_altitude = detect()
        pub_temp.publish(get_json_message(1, "Graph", "Temperature", 1, 0xFFFFFF, current_temp, time.time()))
        pub_gas.publish(get_json_message(2, "Graph", "Gas", 2, 0xFFFFFF, current_gas, time.time()))
        pub_humidity.publish(get_json_message(3, "Graph", "Humidity", 3, 0xFFFFFF, current_humidity, time.time()))
        pub_pressure.publish(get_json_message(4, "Graph", "Pressure", 4, 0xFFFFFF, current_pressure, time.time()))
        pub_altitude.publish(get_json_message(5, "Graph", "Altitude", 5, 0xFFFFFF, current_altitude, time.time()))
        rate.sleep()


def detect():
    try:
        global bme680
        temp = bme680.temperature
        gas = bme680.gas
        humidity = bme680.humidity
        pressure = bme680.pressure
        altitude = bme680.altitude
        return temp, gas, humidity, pressure, altitude
    except:
        print('error in ros node, lost connection to bme680 sensor')
        return 0,0,0,0,0


def init():
    global bme680
    i2c = I2C(board.SCL, board.SDA)
    bme680 = adabme.Adafruit_BME680_I2C(i2c, debug=False)
    bme680.sea_level_pressure = 953.25


def get_json_message(id, type, title, position, color, value, timestamp):
    sensordata_json = {
        "id": id,
        "type": type,
        "title": title,
        "position": position,
        "color": color,
        "datapoint": value,
        "timestamp": timestamp
    }
    return json.dumps(sensordata_json)


if __name__ == '__main__':
    try:
        init()
        bme680_node()
    except rospy.ROSInterruptException:
        pass
