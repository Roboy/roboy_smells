This ROS directory contains everything needed to start smelling, classifiy the data and to send the result to an graphical user interface.
For this four packages exits:

    -eNose classification (e_nose_classifier)
    -eNose data sampling (e_nose_raw_publisher)
    -demo view of classification (demo)
    -conntection to OUI(interface_oui)
    
For each package the full reposity is needed as it depends on other packages.
To start a node, be on the root directory.

eNose classification:

must be used on a stronger computer as the lstm computation will need long
receives the raw data from the data gathering machine, computes the classificaton,
puplishes the result as string and in a specific json format for OUI

    -python3 -m ROS.e_nose_classifier.src.classifier_organizer.py

additional it can record the received data for this:
    
    -python3 -m ROS.e_nose_classifier.src.record_ros_measurement.py

eNose data sampling:

must be used on a rasperry pi as it needs the GPIO pins
which are connected to the eNose, fan, router, bme680 according to the wiring scheme
it sends the raw data to the classifier
Has two modes:
1) a simple measurment and send logic

        -python3 -m ROS.e_nose_raw_publisher.src.e_nose_node.py
        
2) Automatic random sampling with labiling similiar to the TestRunner

        -python3 -m ROS.e_nose_raw_publisher.src.e_nose_node.py
    
demo view:
simple tkinter app which displays the received classification with a background color

        -python3 -m ROS.demo.demo.py

OUI:
sends the bme680 sensor data in a specific format to OUI

    -python3 -m ROS.interface_oui.bme680_node.py



    
  