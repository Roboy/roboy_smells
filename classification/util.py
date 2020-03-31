import numpy as np

def get_classes_dict(classes: list) -> dict:
    """
    Creates classes dictionary of classes list.

    :param classes:                 Classes list.
    :return:                        Classes dictionary
    """
    classes_dict = {}
    for i, c in enumerate(classes):
        classes_dict[c] = i
    return classes_dict

def get_class(c: str, class_dict: dict) -> list:
    """
    Gets class index for given class from classes dictionary.

    :param c:                       Class string.
    :param class_dict:              Classes dictionary.
    :return:                        Class index.
    """
    return list(class_dict.keys())[list(class_dict.values()).index(c)]

def get_classes_list(measurements: list) -> list:
    """
    Get list of labels occuring in measurement list in alphabetical order.

    :param measurements:            List of measurements.
    :return:                        List of classes.
    """
    return np.unique([m.label for m in measurements])

def hot_fix_label_issue(measurements: list) -> list:
    """
    Aligns measurement labels with upper and lower case letters.

    :param measurements:            List of measurements that could contain labels with upper case letters.
    :return:                        List of measurements only containing labels with lower case letters.
    """
    for i in range(len(measurements)):
        measurements[i].label = measurements[i].label.lower()
    return measurements