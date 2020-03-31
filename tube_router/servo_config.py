"""
configuration of the servo, at which position an entrance is, and which GPIO pin is used for control.
"""

def get_channels():
    C = [3.7,  # channel 1
         4.7,  # channel 2
         5.8,  # channel 3
         6.8,  # channel 4
         7.9,  # channel 5
         9,  # channel 6
         10.1  # channel 7
         ]
    return C


def get_control_GPIO():
    control_GPIO = 17  # pin 11
    return control_GPIO
