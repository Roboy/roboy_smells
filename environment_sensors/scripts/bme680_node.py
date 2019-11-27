#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String

current_temp = 0
current_air = 0
current_humidity = 0

def bme680_node():
    pub_temp = rospy.Publisher('env_sensor/current_temp', String, queue_size=10)
    pub_air = rospy.Publisher('env_sensor/current_temp', String, queue_size=10)
    pub_humidity = rospy.Publisher('env_sensor/current_temp', String, queue_size=10)
    rospy.init_node('bme680', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        temp_str = "Current temp is {}".format(current_temp)
	air_str = "Current air quality is {}".format(current_temp)
	humid_str = "Current humidity is {}".format(current_temp)
        pub_temp.publish(temp_str)
        pub_air.publish(air_str)
        pub_humidity.publish(humid_str)
        rate.sleep()

if __name__ == '__main__':
    try:
        bme680_node()
    except rospy.ROSInterruptException:
        pass
