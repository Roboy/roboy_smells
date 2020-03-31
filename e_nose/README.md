# E-Nose

This package contains all the classes used for getting and processing data from the E-Nose.  
Check the Roboy Confluence project page for more details.

Most important is the class `Measurement` which can be found in `measurements.py`.
The `Measurement` objects contain a continuous time-series of samples of one smell.

`data_processing.py` contains all the algorithms used to pre-process data.

### How to get `Measurement` objects:

#### Data-Source
Either reading from CSV files using `file_reader.py`:
```python
from e_nose import file_reader
from e_nose import data_processing as dp
from e_nose.measurements import Measurement, StandardizationType

data_tuple = file_reader.read_all_files_in_folder('./data/')
functionalisations, working_channels, all_data = data_tuple

measurements = []
for file in all_data:
    meas = dp.get_labeled_measurements(all_data[file], working_channels, functionalisations)
    # Standardize the measurements using the standardization type
    # (This method also removes all reference measurements)
    adding = dp.standardize_measurements(meas, StandardizationType.LAST_REFERENCE)
    print("file:", file, "has", len(adding), "standardized measurements")
    if adding is not None:
        measurements.extend(adding)

print("Total number of measurements:", len(measurements))
```

or reading from a live data stream using `online_reader.py`:
```python
from e_nose.measurements import DataType, StandardizationType
from e_nose.online_reader import OnlineReader

online = OnlineReader(5, standardization=StandardizationType.LAST_REFERENCE)
[...]
# Dynamically add samples when they arrive
online.add_sample(sample)
[...]
measurement = online.get_since_n_as_measurement(50)
```


### Processing Methods & Standardization Types

Having a list of `Measurement` objects you usually want to standardize them and use the data in log space (default).

Standardization should be done using the given method in `data_processing.py`:
```python
from e_nose import data_processing as dp
from e_nose.measurements import Measurement, StandardizationType

# The order of measurement objects is important: they should be continuous.
standardized_measurements = dp.standardize_measurements(measurements, StandardizationType.LAST_REFERENCE)
```

We have implemented 3 ways of standardizing Measurements:

#### `StandardizationType.LAST_REFERENCE`
Uses the average of the last 10 samples of the last reference measurement before this measurement.

As we usually take a `ref` measurement between every normal one, this basically nulls the sensor perfectly.

#### `StandardizationType.BEGINNING_AVG`
Uses the average of the first 3 samples of the same measurement.

This is especially useful if no context is available because it does not depend on any other measurement objects.

#### `StandardizationType.LOWPASS_FILTER`
Uses a lowpass filter that has been running over all the samples as the standardization.
The lowpass filter basically gives the slow drifts of the sensor, therefore standardizing with it removes this drift,
but it cannot cope with sensor biases from the strong smell of the measurements just before.

This is especially useful if you cannot be sure that there was a reference measurement just before the new measurement,
because in theory this makes the sensor adapt to slow changes.
