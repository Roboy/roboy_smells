import numpy as np

def get_classes_dict(classes):
    classes_dict = {}
    for i, c in enumerate(classes):
        classes_dict[c] = i
    return classes_dict

def get_class(c, class_dict):
    return list(class_dict.keys())[list(class_dict.values()).index(c)]

def get_classes_list(measurements):
    return np.unique([m.label for m in measurements])