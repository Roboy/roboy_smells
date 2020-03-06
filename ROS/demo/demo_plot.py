# !/usr/bin/env python
import rospy
from std_msgs.msg import String
import tkinter as tk

from matplotlib.backends.backend_tkagg import ( FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import numpy as np

class FullScreenApp(tk.Frame):
    def __init__(self, master, **kwargs):
        tk.Frame.__init__(self, master)
        self.master = master
        pad = 3
        self._geom = '200x200+0+0'
        master.geometry("{0}x{1}+0+0".format(
            master.winfo_screenwidth() - pad, master.winfo_screenheight() - pad))
        master.bind('<Escape>', self.toggle_geom)

        self.sensor_data = []

        fig = Figure(figsize=(5, 4), dpi=100)
        t = np.arange(0, 3, .01)
        self.ax = fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(fig, master=master)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(self.canvas, master)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        def on_key_press(event):
            print("you pressed {}".format(event.key))
            key_press_handler(event, self.canvas, toolbar)
            self.gen_random_data()

        self.canvas.mpl_connect("key_press_event", on_key_press)

        def _quit():
            root.quit()  # stops mainloop
            root.destroy()  # this is necessary on Windows to prevent
            # Fatal Python Error: PyEval_RestoreThread: NULL tstate

        button = tk.Button(master=master, text="Quit", command=_quit)
        button.pack(side=tk.BOTTOM)

    def toggle_geom(self, event):
        geom = self.master.winfo_geometry()
        print(geom, self._geom)
        self.master.geometry(self._geom)
        self._geom = geom

    def update_sensor_data(self, sensor_data):
        self.ax.clear()
        self.sensor_data.append(sensor_data)
        self.ax.plot(self.sensor_data)
        self.canvas.draw()

    def gen_random_data(self):
        self.update_sensor_data(np.random.rand(64))

root = tk.Tk()
app = FullScreenApp(root)

def callback(data):
    app.update_sensor_values(data.sensordata)

def listener():
    print()
    rospy.Subscriber("enose_sensordata", e_nose_raw, callback)

if __name__ == '__main__':
    root.mainloop()
    listener()
