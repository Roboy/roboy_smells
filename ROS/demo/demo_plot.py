# !/usr/bin/env python
import rospy
from std_msgs.msg import String
from rospy_tutorials.msg import Floats
from e_nose_raw_publisher.msg import e_nose_raw
import tkinter as tk

import threading
import time

import e_nose.data_processing as dp
from matplotlib.backends.backend_tkagg import ( FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

class FullScreenApp(tk.Frame):
    def __init__(self, master, **kwargs):

        self.normalize = True

        tk.Frame.__init__(self, master)
        self.master = master
        pad = 3
        self._geom = '200x200+0+0'
        master.geometry("{0}x{1}+0+0".format(
            master.winfo_screenwidth() - pad, master.winfo_screenheight() - pad))
        master.bind('<Escape>', self.toggle_geom)

        self.sensor_data = np.zeros((0,64))
        self.meas_data = np.zeros((0,64))

        fig = Figure(figsize=(5, 4), dpi=100)
        t = np.arange(0, 3, .01)
        fig, (self.ax1, self.ax2) = plt.subplots(2)
        #(self.ax1, self.ax2) = fig.add_subplot(211)

        self.canvas = FigureCanvasTkAgg(fig, master=master)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(self.canvas, master)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        def on_key_press(event):
            key_press_handler(event, self.canvas, toolbar)
            self.gen_random_data()
            self.canvas.draw()

        self.canvas.mpl_connect("key_press_event", on_key_press)

        def _quit():
            root.quit()  # stops mainloop
            root.destroy()  # this is necessary on Windows to prevent
            # Fatal Python Error: PyEval_RestoreThread: NULL tstate

        button = tk.Button(master=master, text="Quit", command=_quit)
        button.pack(side=tk.BOTTOM)

        root.after(300, self.update_canvas)

    def update_canvas(self):
        self.canvas.draw()
        root.after(1000, self.update_canvas)


    def toggle_geom(self, event):
        geom = self.master.winfo_geometry()
        self.master.geometry(self._geom)
        self._geom = geom

    def update_meas_data(self, md):
        self.ax1.clear()
        self.meas_data = md
        self.ax1.plot(self.meas_data)

    def update_sensor_values(self, sd):
        self.ax2.clear()
        self.sensor_data = np.vstack((self.sensor_data, np.expand_dims(sd, axis=0)))
        self.sensor_data = self.sensor_data[-100:, :]
        if self.normalize and len(self.sensor_data) > 10:
            norm = dp.high_pass_logdata(self.sensor_data)
            self.ax2.plot(norm)
            print(norm.shape)
        else:
            self.ax2.plot(self.sensor_data)
            print(self.sensor_data.shape)

    def gen_random_data(self):
        print("2")
        #self.update_sensor_values(np.random.rand(64))

root = tk.Tk()
app = FullScreenApp(root)

def callback(data):
    app.update_sensor_values(list(data.sensordata))

def callback_meas(data):
    print('received enose measurement')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    d = np.array(data.data).reshape((-1,62))
    print(d.shape)
    app.update_meas_data(d)

def listener():
    print()
    rospy.Subscriber("enose_sensordata", e_nose_raw, callback)

def listener_meas():
    rospy.Subscriber("/e_nose_measurements", Floats, callback_meas)

if __name__ == '__main__':
    rospy.init_node('demo_plot_gui', anonymous=True)
    listener()
    listener_meas()
    root.mainloop()
