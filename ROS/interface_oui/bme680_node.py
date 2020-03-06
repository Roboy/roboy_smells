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
    send_warning = False
    pub_temp = rospy.Publisher('env_sensor', String, queue_size=10)
    pub_temp_warn = rospy.Publisher('env_sensor', String, queue_size=10)
    pub_gas = rospy.Publisher('env_sensor', String, queue_size=10)
    pub_humidity = rospy.Publisher('env_sensor', String, queue_size=10)
    pub_pressure = rospy.Publisher('env_sensor', String, queue_size=10)
    pub_altitude = rospy.Publisher('env_sensor', String, queue_size=10)
    rospy.init_node('bme680', anonymous=False)
    rate = rospy.Rate(1)  # 1hz
    print('started ros node successfully!')
    while not rospy.is_shutdown():
        current_temp, current_gas, current_humidity, current_pressure, current_altitude = detect()
        pub_temp.publish(get_json_message(1, current_temp, time.time()))
        if current_temp > 40 and not send_warning:
            send_warning = True
            pub_temp_warn.publish(
                get_json_toastr_message(37, str("WARNING: TEMPERATURE is high: ", current_temp), [255, 0, 0, 255]))
        elif current_temp < 35:
            send_warning = False
        # pub_gas.publish(get_json_message(2, current_gas, time.time()))
        pub_humidity.publish(get_json_message(38, current_humidity, time.time()))
        # pub_pressure.publish(get_json_message(4, current_pressure, time.time()))
        # pub_altitude.publish(get_json_message(5, current_altitude, time.time()))
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
        return 0, 0, 0, 0, 0


def init():
    global bme680
    i2c = I2C(board.SCL, board.SDA)
    bme680 = adabme.Adafruit_BME680_I2C(i2c, debug=False)
    bme680.sea_level_pressure = 953.25


def get_json_message(id, value, timestamp):
    sensordata_json = {
        "id": id,
        "graphDatapoint": value,
        "graphDatapointTime": timestamp
    }
    return json.dumps(sensordata_json)


def get_json_toastr_message(id, value, color):
    sensordata_json = {
        "id": id,
        "toastrMessage": value,
        "toastrColor": color

    }
    return json.dumps(sensordata_json)


if __name__ == '__main__':
    try:
        init()
        bme680_node()
    except rospy.ROSInterruptException:
        pass
