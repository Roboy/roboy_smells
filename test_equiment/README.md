Data Gathering Machine

The dgm gatheres the data from the eNose and bme680 sensor.
It automatically connects to the eNose and the environment sensor and starts measuring.
Before start some settings have to be done. First the samples have to be labeled in the right order.
Also the time of taking a reference measurement as well as the time of taking a sample measurement
Also how many loops shall be done and if it selects the next sample by time or if the gradient of the last 10 measurements
is below a certain threshold (which have to be fitted to your needs, see TimeOverCalculater.py)

    labelsList = ['ref', 'raisin', 'acetone', 'orange_juice', 'pinot_noir', 'isopropanol', 'wodka']
    num_loops = 20
    time_loop_min = 2  # in minutes
    time_loop = 60. * time_loop_min
    time_ref_min = 2  # in minutes
    time_ref = time_ref_min * 60
    expected_time = num_loops * (len(labelsList) * (time_loop_min + time_ref_min)) + time_ref_min
    expected_time_end = datetime.now() + timedelta(minutes=expected_time)
    print('expected time: ', timedelta(minutes=expected_time), ' hours stoppes at: ',expected_time_end.strftime("%H:%M:%S"))
    TestEquimentRunner(labelsList, num_loops, time_loop, time_ref, False)
    
The order of taking samples is randomly, but at the end each sample should be choosen equally. 
As the less taken ones have a higher chance of beeing taken. 

The script also does the switching of the samples automatically via the router. It moves accordingly the selected entrance.
The data is saved in a csv file in this style:
       
        ['Time', Channel 1-64, 'Temperature', 'Gas', 'Humidity', 'Pressure', 'Altitude', 'Label'])
        
This file can be read by the file_reader in the e_nose package. After the script finishes it gets automatically uploaded
to an gdrive folder:
https://drive.google.com/drive/folders/1xjcx6ju0zyZLqKheKkeA-f_Aa2HLEJGM

As the raspi should be accessible via ssh and a single run take serveral hours, its advisable to start the script in
a tmux session.

This can be done via:

    tmux new -s data
    
Then start the script, after configuration:

    python3 TestRunner.py
