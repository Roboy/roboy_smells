# !/usr/bin/env python
import rospy
from std_msgs.msg import String
from e_nose_raw_publisher.msg import e_nose_raw
import tkinter as tk

from matplotlib.backends.backend_tkagg import ( FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
import numpy as np

"""
This is an application that takes input from the e_nose sensor and displays the data in matplotlib graphs.
"""

class FullScreenApp(tk.Frame):
    def __init__(self, master, **kwargs):
        """
        initialised the TKinter App and the matplotlib canvas
        :param master: root node of the tkinter app
        :param kwargs:
        """
        tk.Frame.__init__(self, master)
        self.master = master
        pad = 3
        self._geom = '200x200+0+0'
        master.geometry("{0}x{1}+0+0".format(
            master.winfo_screenwidth() - pad, master.winfo_screenheight() - pad))
        master.bind('<Escape>', self.toggle_geom)

        self.sensor_data = np.zeros((0,64))

        fig = Figure(figsize=(5, 4), dpi=100)
        t = np.arange(0, 3, .01)
        self.ax = fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(fig, master=master)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(self.canvas, master)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        def _quit():
            root.quit()  # stops mainloop
            root.destroy()  # this is necessary on Windows to prevent
            # Fatal Python Error: PyEval_RestoreThread: NULL tstate

        button = tk.Button(master=master, text="Quit", command=_quit)
        button.pack(side=tk.BOTTOM)

        root.after(300, self.update_canvas)

    def update_canvas(self):
        """
        This updates the canvas (plots new data) every 1 second (1000 ms). If this would be done in the update sensor
        values method, the app crashes because of a collision between ROS and tk.
        """
        self.canvas.draw()
        root.after(1000, self.update_canvas)


    def toggle_geom(self, event):
        """
        Toggles fullscreen on/off
        :param event: not used
        """
        geom = self.master.winfo_geometry()
        self.master.geometry(self._geom)
        self._geom = geom

    def update_sensor_values(self, sd):
        """
        Adds new sensor value data from the e_nose sensor.
        :param sd: sensor data
        """
        self.ax.clear()
        self.sensor_data = np.vstack((self.sensor_data, np.expand_dims(sd, axis=0)))
        self.sensor_data = self.sensor_data[:100, :]
        self.ax.plot(self.sensor_data)
        self.ax.set_yscale('log')

root = tk.Tk()
app = FullScreenApp(root)

def callback(data):
    app.update_sensor_values(list(data.sensordata))

def listener():
    rospy.Subscriber("enose_sensordata", e_nose_raw, callback)

if __name__ == '__main__':
    rospy.init_node('demo_plot_gui', anonymous=True)
    listener()
    root.mainloop()
